# A Comprehensive Guide to Compilers

Written for the Cudro project intern. Every concept is anchored to something
we build or have built — read it with `src/` open beside you.

---

## 1. What a compiler is

A **compiler** is a program that translates a program from one representation
into another while *preserving meaning*. Usually: human-friendly text →
machine-executable form.

| Term | Meaning | Example |
|---|---|---|
| Source language | what humans write | `.cudro` specs, C++ |
| Target language | what executes | our emitted C kernels, x86 machine code |
| Interpreter | executes directly, no translation step | Python REPL |
| AOT compilation | translate before running | `gcc hello.c` at build time |
| JIT compilation | translate *while* running, on demand | libtcc compiling our kernel; PyTorch 2.0 |

Key identity: **an AOT compiler is just a JIT whose inputs arrived early.**
Same pipeline, different scheduling. Cudro exposes both modes (`--emit-c` vs
`--jit-run`).

## 2. The pipeline — three parts, one invariant

Every serious compiler splits into:

```
FRONT-END            MIDDLE-END                BACK-END
characters→tokens    IR transforms             target code
→AST→checked AST     (folding, AD passes,      (codegen, JIT load)
                     optimizations)
```

- **Front-end**: understands the *source language* only.
- **Back-end**: understands the *target machine* only.
- **Middle-end**: understands neither — it works on an intermediate
  representation (IR), which is why optimizations can be shared across all
  source languages and all targets.

Cudro's instantiation:

| Stage | Module | Input → Output |
|---|---|---|
| Lexer | `lexer.*` | characters → tokens |
| Parser | `parser.*` | tokens → AST |
| Sema | `sema.*` | AST → annotated/validated AST |
| Lowering | `lower.*` | AST → expression DAG (+desugar) |
| Folding | inside lower | DAG → smaller DAG |
| Autodiff | `ad.*` | DAG → DAG + gradient subgraphs |
| Codegen | `codegen_c.*` | DAG → C source text |
| JIT | `jit_tcc.*` | C text → callable function pointer |

## 3. The front-end

### 3.1 Lexing

Groups raw characters into **tokens**: the words and punctuation of the
language. Responsibilities:

- classify each token (`Number`, `KwRobot`, `{`, …)
- record its **source location** (file:line:col) — every good error message
  ever printed starts here
- discard **trivia**: whitespace, comments
- never stall: on garbage, consume ≥1 character, report, continue
  (**always-advance invariant**)

Design lessons already paid for in `src/lexer.cpp`:

- **Dedicated keyword kinds** (`KwRobot`, not `Identifier("robot")`) — do
  classification once here so the parser matches enum values.
- **Lexical decisions encode grammar decisions**: `-` starts a number only
  when followed by a digit, because our grammar has negative literals but no
  subtraction operator. Add arithmetic later ⇒ revisit consciously, or
  `x - 3` silently lexes as `x`, `-3`.
- **Loops, not recursion**, on error paths: recursion burns a stack frame per
  skipped character; hostile input becomes a crash instead of diagnostics.
- **`string_view`s are loans, not ownership.** Tokens point into the caller's
  buffer; whoever owns the buffer must outlive every borrower. ASan caught a
  real use-after-free in our own tests over exactly this.

### 3.2 Parsing

Turns the flat token stream into a tree reflecting the grammar. Our method:
**recursive descent** — one function per grammar rule.

Grammar rules are written in EBNF:

```
joint := 'joint' IDENT '{' 'type' ... ';' 'axis' vec3 ';' ... '}'
vec3  := '[' number ',' number ',' number ']'
```

become functions `parseJoint()`, `parseVec3()` — rule structure maps directly
to call structure, which is why hand-written recursive descent is the best
*teaching* parser technology even though parser generators exist.

Concepts that matter:

- **Lookahead**: how many tokens you need to see before deciding which rule
  applies. Ours needs 1 (LL(1)) because the grammar was designed for it.
- **Left recursion kills recursive descent**: a rule like `expr := expr '+' term`
  recurses forever; handled via loops/restructuring (relevant if constraints
  ever grow infix operators).
- **Error recovery**: on unexpected input, report with location, then **skip
  tokens until a sync point** (`;`, `}`) so one typo doesn't cascade into 40
  fake errors. A compiler that reports one honest error beats one that
  reports thirty lies.
- **AST design**: plain structs mirroring the grammar, every node carrying
  its `Location`. Keep semantics OUT of the tree — that's the next stage.

### 3.3 Semantic analysis (sema)

Parsing proves *shape*; sema proves *meaning*: names resolve (does link `ee`
exist?), dimensions agree ([0,0] is not a vec3), declarations aren't
duplicated. Implemented with **symbol tables** — scoped dictionaries from
name → declaration.

Why separate from the parser? Same reason compilers have stages at all: each
stage gets small enough to be *obviously correct*, and errors surface at the
earliest possible stage with the best possible message. A dimension mismatch
should be rejected in M2 with "arm1.j2.axis has 2 components, expected 3" —
not crash M5's code generator.

## 4. The middle-end

### 4.1 Why an IR — and why ours is an expression DAG

Parse trees encode *syntax*; IRs should encode *computation*. Properties worth
wanting: simple to construct, trivial to analyze, easy to transform, cheap to
lower to any target.

Our **expression DAG** (directed acyclic graph): nodes = operations
(`const`, `input(q_i)`, `rot_z`, `matmul`, `dot`, …), edges = data flow.
FK chains get *inlined* into pure arithmetic; constraint sugar gets
*desugared* into core ops. One small op set ⇒ every backend consumes the same
thing ⇒ adding LLVM/CUDA targets is mechanical (this is idea.md R1 made
structural).

Bigger systems use **SSA form** (each variable assigned once; φ-nodes merge
control flow) — the industry-standard middle IR since LLVM popularized it.
Our DSL has no loops or branches in constraint math, so a DAG captures
everything SSA would, minus the machinery.

### 4.2 Passes

A **pass** = one traversal of the IR performing one kind of transformation.
Composable, testable in isolation:

- **Constant folding** — evaluate `2*3.14` at compile time. Our first real
  optimization, ~30 lines.
- **Desugaring/lowering** — replace high-level nodes with core-op
  equivalents (`plane{...}` → dot/sub chain).
- **Dead code elimination** — delete nodes no output depends on.
- **Common subexpression reuse** — hash-consing identical subtrees (DAGs make
  this natural).

### 4.3 Autodiff as a pass

The projection kernel needs Jacobians $\partial g/\partial q$. We compute them by writing an analytical pass over our own DAG using **forward-mode automatic differentiation**:

- every value carries a **dual number** $(v, \dot{v})$ — the derivative seeded per input $q_i$;
- chain rule applied mechanically per node type: $(u \cdot w)' = u' \cdot w + u \cdot w'$, etc.;
- **Activity Analysis**: Scans the DAG forward before differentiation. For serial kinematic chains and branching trees, joint $q_j$ appears in only one transform along each path. By identifying whether operand $A$ or $B$ is active in `MatMul(A, B)`, the product rule collapses from exponential $O(2^N)$ binary branching to a linear $O(N)$ sequence of matrix multiplies!
- **Multi-Constraint $M \times N$ Matrix Assembly**: For arbitrary constraint lists $\{g_0, \dots, g_{M-1}\}$, the pass generates a 2D DAG node matrix `jacobian_nodes[c][j] = ∂g_c/∂q_j`, appending analytical gradient subgraphs directly to the unified DAG.
- **Rodrigues Symbolic Differentiation**: For arbitrary spatial unit axes $\hat{k}$, revolute rotations are differentiated symbolically using the exact Rodrigues matrix derivative $\frac{\partial R}{\partial \theta} = -\sin\theta \mathbf{I} + \sin\theta \hat{k}\hat{k}^T + \cos\theta [\hat{k}]_\times$.

Contrast **reverse mode** (backpropagation): one sweep for all inputs, dominant in ML, needs a tape and more bookkeeping. Forward wins when outputs $\le$ inputs or when zero runtime allocations and structural sparsity matter most. Validation: analytic derivatives must match **central finite differences** to $\le 10^{-5}$ — you don't trust AD to check AD.

## 5. The back-end

### 5.1 Codegen strategies

| Strategy | Our verdict |
|---|---|
| Emit readable C, compile with libtcc | **chosen** — debuggable output (~ms compile); RobCoGen precedent |
| Emit LLVM IR, JIT via ORC | M6.5 comparison arm — better code quality, opaque dumps, heavier dependency |
| Direct machine-code emission | never — register allocation alone is a semester |

Emitting C means C is "our assembly": the generated kernel for a plane
constraint reads like unrolled textbook linear algebra — which is precisely
why generated code is auditable and why numerics match our Eigen references
bit-closely.

**Multi-Constraint Inlined Solver**: For simultaneous $M \ge 1$ constraints, the back-end emits an inlined Levenberg-Marquardt solver. When $M=1$, it executes a branchless closed-form rank-1 formula; when $M > 1$, it forms the symmetric damped normal equations $(J J^T + \lambda I) y = -g$ directly on the C stack (0 heap allocations) and solves for joint updates $\Delta q = J^T y$ via inlined Gaussian elimination with partial row pivoting and signed pivot clamping (`diag = (diag >= 0 ? 1e-12 : -1e-12)`). This achieves a 1 kHz hard real-time control loop with median latencies of $3.53\ \mu$s and 0 deadline misses.

Kernel shape (the Kernel ABI decision): signature takes obstacle/sphere
positions as **runtime array parameters**, never baked constants — moving
objects become argument updates; only structural spec changes recompile.

### 5.2 SIMD and batching

AVX2 = one thread, one instruction, 8 floats (data parallelism *in-register*).
Threads = multiple cores (task parallelism). Different axes; compose later.

Placement principle: parallelize the dimension that is *independent*. In
planning, candidate configurations are mutually independent — batch across
them. Inside one FK chain each transform depends on the previous — nothing to
overlap. So the emitter chooses batch-over-configurations layout itself
instead of hoping `-O3 -march=native` discovers it: auto-vectorizers only
pattern-match loops that already exist in source, and our source doesn't
exist until runtime.

Runtime dispatch: check `__builtin_cpu_supports("avx2")` at startup, fall
back to scalar otherwise.

### 5.3 JIT execution

libtcc compiles the emitted string in-memory (~5–20 ms) and hands back a
function pointer. Alternatives and trade-offs:

| Mechanism | Latency | Code quality | Weight |
|---|---|---|---|
| libtcc | ~ms | ~-O1 | tiny — right for on-robot |
| cc subprocess + dlopen | ~150 ms | -O3 | zero deps beyond cc |
| LLVM ORC | ~50–200 ms | best | hundreds of MB, steep API |
| NVRTC (CUDA) | ~100–500 ms incl. ctx init | GPU-class | needs GPU runtime |

This table IS the research content of runtime compilation: compile-time speed
vs steady-state quality. Hot-swap pattern (compile new kernel off-thread,
atomic pointer swap) keeps control loops from ever waiting on the compiler.

## 6. Industrial infrastructure — where LLVM/MLIR fit

- **LLVM**: libraries for IR construction, optimization passes, codegen for
  every CPU/GPU vendor. Use it when target breadth and peak code quality
  matter more than build weight. Our M6.5 uses only IRBuilder + ORC.
- **MLIR**: meta-compiler infrastructure on top of LLVM ideas — custom
  *dialects*, reusable passes, progressive lowering (e.g. our DAG ops →
  `arith`/`math` → `vector` → LLVM → PTX). Powers TensorFlow/PyTorch
  compilers. Adoption cost: dialects/TableGen/conversion frameworks have a
  famously steep curve — appropriate *after* you've felt the problems they
  solve, i.e., after M1–M6.

Mapping our DAG to either is mechanical because the DAG is small and closed:
`const` → ConstantFP/`arith.constant`, `fadd` → CreateFAdd/`arith.addf`, …

## 7. Testing and validating compilers

Compilers translate claims into code; validation proves the claims survive.
Ground truth always comes from something independent:

| Technique | Catches | In Cudro |
|---|---|---|
| Differential testing | wrong translations | kernel ↔ hand-written Eigen reference, 10k random configs, ≤1e-6 rel. |
| Finite differences | broken autodiff | numeric Jacobian vs analytic ≤1e-5 |
| Golden/snapshot tests | silent regressions | committed token/AST/C-output dumps |
| Property tests | subtle invariants | feasible points stay put; g(q*)≈0 post-projection |
| Fuzz-lite | crashes on hostile input | random byte strings must yield clean errors, never segfaults |
| Sanitizers (ASan/UBSan) | memory/UB bugs | on every test binary, day one — caught our first real bug |

Numerical tolerances are judgment calls set per-stage (operation order
differs between scalar/SIMD/reference paths).

## 8. Glossary

- **Token** — classified chunk of source text + location
- **AST** — Abstract Syntax Tree; parse result
- **IR** — Intermediate Representation; analysis/transformation substrate
- **SSA** — Static Single Assignment; each name assigned once
- **Pass** — one IR-in/IR-out transformation
- **Lowering/desugaring** — replacing higher-level constructs with core ones
- **Dual number** — value paired with its derivative; forward-mode AD atom
- **JIT/AOT** — just-in-time vs ahead-of-time compilation
- **SIMD** — single instruction, multiple data lanes
- **Differential testing** — comparing two independent implementations

## 9. Where to go next

- Read `src/lexer.cpp` end-to-end; trace `--dump-tokens` output against it
- Engineering a Compiler (Cooper & Torczon) — chapters on parsing & IR map
  almost 1:1 to our milestones
- Crafting Interpreters (Nystrom) — free online; do part I alongside M1/M2
- LLVM Kaleidoscope tutorial — do after M5 to feel Tier-1 infra
- MLIR documentation — after M6.5, if the hunger persists
