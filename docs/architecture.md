# Cudro — Compiler Architecture

A runtime-recompiling constraint compiler for real-time motion planning.
This document is the source of truth for the pipeline we are building.

---

## 1. The pipeline: data transformations

A compiler is a sequence of representation changes. Each stage consumes one
representation and emits another; each is owned by one module.

```
spec/panda7.cudro  (text)
   │
   │ M1 LEXER      lexer.*        → tokens (with file:line:col)
   │ M1 PARSER     parser.*       → AST (robot/joint/link/task decls + exprs)
   │ M2 SEMA       sema.*         → annotated AST; unknown refs, dim
   │                                mismatches rejected here
   │ M3 LOWER      lower.*        → expression DAG (FK chains inlined);
   │               fold             desugaring of constraint kinds +
   │                                constant folding happen here
   │ M4 AD PASS    ad.*           → DAG + ∂g/∂q gradient subgraphs grafted on
   │ M5 CODEGEN-C  codegen_c.*    → std::string of scalar C source
   │                                  (projection = Levenberg-Marquardt iterations)
   │ M6 CODEGEN-C  codegen_c.*    → second emitter: batched AVX2-friendly C
   │ M5 JIT        jit_tcc.*      → libtcc compiles string in-memory:
   │                                  project(q_in, n, q_out), evaluate_constraints(q, n, g)
   │ M7 PLANNER    planner.*      → Constrained RRT-Connect motion planner
   │                                  embedding JIT projection in inner loop
   │ M8 opt        codegen_cuda.* → same DAG → CUDA C → NVRTC module
   ▼
src/main.cpp — CLI driver; each stage exposed as a flag:
   --dump-tokens | --dump-ast | --check | --dump-dag | --dump-jacobian | --emit-c | --jit-run | --jit-bench | --plan
reference/   — independent hand-written Eigen FK + projection (trust anchor)
tests/       — 10 per-stage suites + validation harness
```

Front-end = stages ①–②, middle-end = ③–⑤, back-end = ⑥–⑦, application layer = ⑧. This three-part
split is the invariant shared by every production compiler.

Key design property: **the middle-end owns a small, self-contained expression
DAG with a fixed op set.** Every back-end (C text, LLVM IR, CUDA) consumes that
DAG and nothing else — this is what makes adding M6.5/M8 mostly mechanical and
what makes the "portable IR" claim (idea.md R1) structurally true rather than
rhetorical.

## 2. Tech stack (locked)

| Component | Choice | Notes |
|---|---|---|
| Host language | C++20 / GCC 13.3 | matches VAMP/McVAMP ecosystem |
| Build | CMake 3.28 | doctest via FetchContent for tests |
| Front-end | hand-written lexer + recursive-descent parser | no parser generators — writing it is the lesson |
| Middle-end IR | custom AST → expression DAG | ~500 lines we own entirely |
| Autodiff | forward-mode dual-number pass over the DAG | McVAMP gets this free from CppAD; we build it to demystify it |
| Primary backend | emit readable C source | debuggable output; RobCoGen precedent for source-to-source |
| JIT runtime | libtcc | ~5–15 ms compile; barely optimizes (~-O1 quality) — the latency/quality tension IS the research content |
| Reference math | Eigen | ground truth for differential validation |
| SIMD target | AVX2 (8×f32), runtime dispatch + scalar fallback | `__builtin_cpu_supports("avx2")` |
| Motion Planner | C-RRT-Connect | Constrained RRT using in-kernel LM projection |
| Stretch backends | CUDA via NVRTC | RTX 3060 + nvcc present |

## 3. Spec language (`.cudro`)

### Grammar (M1 implements exactly this)

```
spec        := (robot | task | clearance)*
robot       := 'robot' IDENT '{' joint* link* '}'
joint       := 'joint' IDENT '{' 'type' ('revolute'|'fixed') ';'
               'axis' vec3 ';' 'origin' vec3 ';'
               ('limits' '[' number ',' number ']';')? '}'
link        := 'link' IDENT '{' ('spheres' sphere+)?
               ('parent' IDENT ';')? ('joint_ref' IDENT ';')? '}'
task        := 'task' IDENT '{' 'link' IDENT ';'
               (plane_c | relative_pose_c | com_above_c)+ '}'
plane_c     := 'plane' '{' ('link' IDENT ';')? 'point_on_link' vec3 ';'
               'normal' vec3 ';' 'offset' number ';' '}'
clearance   := 'clearance' '{' 'min_distance' number ';' '}'
vec3        := '[' number ',' number ',' number ']'
sphere      := '[' number ',' number ',' number ',' number ']'   // cx,cy,cz,r
```

### Desugaring preview (M3)

A surface-level constraint kind expands into core DAG expressions:

```
plane { point_on_link p; normal n; offset d }
  ⇒   g(q) = dot(n_world, R(q)·p + t(q)) − d          == 0
```

The planner/harness then projects candidate configurations onto `{ q : g(q)=0 }`.

## 4. Repository layout

```
Cudro/
├── docs/              idea.md, architecture.md, plan.md, compiler-guide.md, robotics-knowledge.md, issues.md, learning-tracker.md
├── docs/codebook/     16 chapter-by-chapter educational compiler guides
├── include/cudro/     public headers, one per stage/module
├── src/               implementations mirroring headers + main.cpp
├── reference/         eigen_reference.hpp/cpp — hand-written FK/projection/Jacobian
├── spec/              panda7.cudro (7-DoF), planar2r.cudro (2-DoF), panda7_constrained.cudro (M=6 obstacles),
│                      bimanual14.cudro (14-DoF branching dual-arm), arbitrary_axes_6r.cudro (skew axes)
└── tests/             11 doctest suites per stage + differential testing + real-time benchmarks
```

## 5. Validation strategy

Ground truth always comes from something *independent*, never from the thing
under test:

| Technique | Validates | Used at |
|---|---|---|
| Differential testing vs Eigen reference | generated kernels | M5–M8 (≤1e-4 rel., 10k random configs × 2 robots) |
| Central finite differences | analytic Jacobian | M4 (≤1e-5 rel.) |
| Golden/snapshot dumps | lexer/parser/codegen regressions | M1+ |
| Property tests | projection sanity (feasible points stay put; g(q*)≈0; limits respected) | M5+ |
| Constrained trajectory validation | C-RRT-Connect planner waypoints | M7 (`test_planner`) |
| Real-time loop jitter benchmark | 1 kHz control loop deadline compliance (1,000 steps) | Real-Time (`test_complex_workloads`) |
| Adversarial singularity stress suite | 10,000 singular/near-singular configurations (100% finite outputs) | Robustness (`test_complex_workloads`) |
| Fuzz-lite + ASan/UBSan | crash-freedom on malformed specs | M1+ (`fuzz_spec`) |

Numerical tolerances are judgment calls set per-stage (operation order differs
between scalar/SIMD/Eigen/reference paths).

## 6. Backend Trade-Off & Compiler Optimization Passes

libtcc wins compile latency (headline metric: ~5–9 ms) but generates unoptimized machine code without an SSA optimizer or loop unroller; McVAMP's ahead-of-time gcc/clang -O3 builds win steady-state throughput.

To close this gap without sacrificing JIT speed, Cudro implements target-specific compiler optimizations across the middle and back ends:

### Phase 1 — Memory & Emission Optimizations
- **Whole-Program Constant Hash-Consing**: Deduplicates identical numeric constants (`0.0`, `1.0`, joint offsets) into single canonical nodes, pruning DAG size by up to 79.5%.
- **Back-End Explicit Loop Unrolling**: Directly unrolls $4 \times 4$ matrix operations (`MatMul`, `MatVecMul`) in generated C, bypassing TCC's loop indexing, branch overhead, and register spills.
- **Zero-Allocation Emitter Streaming**: Replaces dynamic `std::string` allocations with static string caching during JIT code generation.
- **Zero-Allocation Planner Scratchpads**: Pre-allocated scratch buffers eliminate inner-loop heap allocations during RRT tree expansion.

### Phase 2 — Analytical Jacobian Codegen & Activity Analysis
- **Activity-Guided Automatic Differentiation**: Scans DAG inputs to identify which intermediate nodes depend on each joint index. For serial kinematic chains, at most one factor in `MatMul(A, B)` is active, preventing exponential $O(2^N)$ product-rule branching and reducing derivatives to a linear chain of $O(N)$ operations.
- **Unified Single-Pass Kernel Evaluator**: `evaluate_dag(q, out_g, out_J)` computes both constraints and exact analytical Jacobians in a single forward pass.
- **Zero-Perturbation Levenberg-Marquardt**: Eliminates the $N$-iteration finite-difference perturbation loop inside `project_single`, slashing projection latency on 7-DoF manipulators to 11.6 $\mu$s/solve with exact machine precision.

### Phase 3 — Multi-Constraint Normal Equations & Branching Kinematics
- **Arbitrary Multi-Constraint Matrix Assembly ($M \ge 1$)**: Synthesizes a full $M \times N$ 2D analytical Jacobian matrix DAG, enabling simultaneous satisfaction of $M=2, 3, 6, \dots$ constraints on arbitrary body links.
- **Inlined Normal Equations Linear Solver**: Emits a stack-allocated $M \times M$ linear system $(J J^T + \lambda I) y = -g$ solved via inlined Gaussian elimination with partial row pivoting and signed pivot clamping (`diag = (diag >= 0 ? 1e-12 : -1e-12)`). Emits a branchless closed-form rank-1 update when $M=1$.
- **Branching Kinematic Trees**: Implements topological iterative transform resolution in `build_all_link_transforms`, seamlessly compiling dual-arm bimanual robots (e.g. 14-DoF bimanual) regardless of link specification order.
- **Symbolic Rodrigues Differentiation**: Differentiates revolute joints with arbitrary 3D spatial rotation axes (`RotAxis`) analytically using Rodrigues matrix derivatives.

### Phase 4 — 1 kHz Real-Time Controller Loop & Adversarial Singularity Resilience
- **Hard Real-Time 1 kHz Loop Compliance**: Benchmarked in a closed-loop controller tracking task over 1,000 steps with a 1,000 $\mu$s deadline: achieves **median latency of 3.53 $\mu$s**, **peak jitter of 53.95 $\mu$s**, and **zero missed deadlines**.
- **Adversarial Singularity Resilience**: Stress-tested across 10,000 adversarial, gimbal-locked, and near-singular configurations: achieves **100% finite outputs** with zero NaN or Inf escapes.

## 7. Deliberately out of scope

Per idea.md: beating McVAMP's published numbers, mesh collision, new planning
algorithms, general robotics framework. Narrow claim: closing the
ahead-of-time → runtime compilation gap the McVAMP authors named, at learning
scale.

