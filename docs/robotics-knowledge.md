# Cudro — Robotics Knowledge Primer

The minimum robotics needed to work on this compiler, in plain language.
Living document: expanded whenever new questions come up during development
(see §9 Lessons Log).

---

## 1. How robots are modeled

A robot arm is a chain/tree of rigid bodies (**links**) connected by
(**joints**). Each revolute joint contributes one degree of freedom (DoF): its
rotation angle. A 7-joint arm ⇒ 7 DoF ⇒ its **configuration** is a vector
q ∈ ℝ⁷ — one angle per joint.

Cudro's spec mirrors this directly: `robot { joint … link … }` declares the
kinematic tree. We store each joint as axis + origin (a translation), which is
equivalent to textbook Denavit–Hartenberg parameters but simpler to reason
about.

## 2. Forward kinematics (FK)

Given angles q, FK computes where every link ended up in the world:
compose rotations/translations joint by joint out from the base.

- Mathematically: 4×4 homogeneous transforms multiplied along the chain,
  T(q) = T₁(q₁)·T₂(q₂)···Tₙ(qₙ).
- In our compiler, FK is **not a runtime library call** — it is *inlined into
  the expression DAG* during lowering (M3). The generated kernel contains the
  fully unrolled arithmetic for this specific robot. That unrolling is why
  generated code is fast and why changing the robot requires recompiling
  (which is cheap — that's the whole point of Cudro).

Rotation about unit axis k by angle θ uses Rodrigues' formula; you will meet
it when writing the DAG ops in M3/M4.

## 3. Constraint taxonomy in Cudro

Three families, all declared declaratively:

1. **Joint limits** — box inequalities loᵢ ≤ qᵢ ≤ hiᵢ per joint. Handled by
   clamping, not calculus.
2. **Collision constraints** — links approximated by spheres/capsules
   (declared in the spec). Feasibility = pairwise distances stay above a
   clearance margin. Sphere–sphere distance is trivially computable, which is
   why we use them instead of mesh collision.
3. **Task/manifold constraints** — e.g. "end-effector stays on this plane":
   a smooth equation g(q) = 0. These are the interesting ones; they define a
   **manifold** (a curved surface inside configuration space) and require
   projection.

## 4. Manifolds and projection — the heart of the kernel

For task constraints, the feasible set is M = { q : g(q) = 0 }. A sampled or
interpolated configuration q generally lands slightly off M. **Projection**
finds the nearest feasible configuration:

```
minimize ‖q − q₀‖²   subject to   g(q) = 0
```

Solved iteratively with Gauss-Newton / Levenberg-Marquardt:

```
repeat until ‖g(q)‖ tiny:
    Δq  = −Jᵀ(JJᵀ + λI)⁻¹ g(q)        # J = ∂g/∂q, the Jacobian
    q  += Δq
```

Why this matters to the compiler:

- The **generated kernel's job is exactly this loop**, specialized to one
  robot's g and J — unrolled, branch-free, batched across many q vectors.
- **Zero/near-zero Jacobian ⇒ zero step ⇒ frozen wherever it started**
  (feasible or not). Near-singular J is why LM adds damping λ — and why
  McVAMP compiles a branchless Cholesky solve for `(JJᵀ+λI)⁻¹`.
- Equalities (task constraints) get Newton iterations; inequalities (limits,
  clearance) get clamping/penalty handling. Both appear in the emitted C.

## 5. Motion planning context (why anyone wants this kernel)

Sampling-based planners throw thousands of random configurations at the
problem, connect promising ones into a tree/path, then verify feasibility.
With task constraints, every sample needs projection first — so the
projection kernel **is the inner loop**, and kernel speed ≈ planner speed.
VAMP/McVAMP exploited exactly this with ahead-of-time compiled SIMD code;
Cudro produces equivalent specialization at *runtime* when the spec arrives.

Our M7 deliverable is deliberately minimal: a toy sampler demonstrating the
kernel in that role — not a competitive planner.

## 6. Why SIMD batches fit planning perfectly

Planners are embarrassingly parallel *across samples*: 512 candidate
configurations are mutually independent, so one AVX2 instruction can advance
8 of them simultaneously (one thread, 8-wide float registers). Inside a
single FK chain the opposite holds — each joint transform depends on the
previous — so there is nothing to overlap. Conclusion baked into the
architecture: **we choose batch-over-configurations as the parallel dimension
and write the emitter around it**, rather than hoping an optimizer discovers it.

Threads (multiple cores) are a different axis again; they compose with SIMD
later but solve different problems.

## 7. Units and conventions

SI throughout: meters, radians. World frame right-handed, z-up. Rotation
matrices act as R·p for world-position of a body-fixed point p.

## 8. Robots used in this project

| Robot | DoF | Role |
|---|---|---|
| `panda7` | 7 | primary; Panda-like dims; plane + collision constraints |
| `planar2r` | 2 | R5 generalization test; structurally different (planar), runs through untouched pipeline |
| `bimanual14` | 14 | Branching kinematic tree with dual arms sharing a torso; tests topological lowering and simultaneous dual-arm coordination |
| `panda7_constrained` | 7 | Dense $M=6$ simultaneous multi-link constraints on forearm, elbow, and end-effector |
| `arbitrary_axes_6r` | 6 | 6-DOF manipulator with compound spatial skew rotation axes (`RotAxis`) testing analytical Rodrigues differentiation |

## 9. Lessons log (updated as questions arise)

- **2026-08-25 — What breaks if ∇g(q) = 0?** Newton step Δq = −J⁺g becomes
  zero regardless of error g(q): the iterate freezes, feasible or not.
  Ill-conditioned (near-zero) J motivates LM damping. (Intern answer scored ½.)
- **2026-08-25 — Why doesn't `-O3 -march=native` give us SIMD for free?**
  Auto-vectorizers pattern-match loops that already exist in source with known
  trips/alignment; our kernel text doesn't exist until the constraint arrives
  at runtime. And FK chains are serially dependent anyway. Parallelism
  placement is a *codegen decision*, not an optimization afterthought.
- **2026-08-25 — Threads vs SIMD?** Not the same thing. AVX2 = one thread,
  one instruction, 8 floats (data parallelism in-register). Threads = multiple
  cores (task parallelism). Different axes; compose later.
- **2026-08-25 — "Isn't ahead-of-time safer than JIT?"** Right when the
  constraint set is truly frozen (certified product, closed task list): no
  compiler on-robot, fully optimized code, audited binary. The gap in the
  argument: constraint *kinds* are known at deploy, but *instances* arrive at
  runtime from sensing/tasking (grasped object → keep-out zone; human →
  exclusion sphere). AOT-vs-JIT is a spectrum: production hybrids AOT the
  stable core (kinematics) and specialize the volatile tail. Key identity:
  **an AOT compiler is a JIT whose inputs arrived early** — same pipeline,
  `--emit-c` at build time vs load-at-runtime. Hot-swap pattern means the
  control loop never waits on the compiler.
- **2026-08-25 — Why is time-to-first-kernel the thesis number?** Steady-state
  speedup was already proven by VAMP/McVAMP ahead-of-time; reproducing it adds
  nothing novel. The unpublished claim is sub-second spec→kernel turnaround.
  If #3 is small, the project's narrow contribution stands regardless of #2.
- **2026-08-25 — How would VLAs / world models use this?** Neural components
  are proposers/predictors with no guarantees; our kernel is the enforcer
  between proposal and motor. Three patterns: (1) safety filter projecting VLA
  action chunks before execution; (2) batched feasibility scoring inside
  world-model/MPC rollouts (batch × horizon = our M6 shape); (3) perception
  *writes the spec* — IR as the interface between neural and symbolic.
  Integration cost analysis: ~0% compiler-core change, ~a week of glue post-M5
  (facade API + async hot-swap); hence the Kernel ABI decision now recorded in
  docs/plan.md.
- **2026-08-26 — Parser: why panic-mode recovery with sync tokens?** A single
  typo shouldn't cascade into 40 fake errors. The parser reports the first
  error, then skips tokens until it hits a structural anchor (`;`, `}`, or a
  declaration keyword) so the rest of the file still parses. This is how
  production compilers (Clang, Rustc) achieve "one real error per typo."
- **2026-08-26 — Parser: why `consume` not `expect`?** Naming matters.
  `expect` sounds like "I hope this is here"; `consume` means "I'm taking this
  token and moving on." The verb matches the action: it consumes the token from
  the stream. Also clarifies that the function *always advances* on success.
- **2026-08-26 — Parser: why `parse_number_safe` returns NaN instead of throwing?**
  Exceptions in parsing are expensive and break the "all errors in one pass"
  guarantee. Returning NaN lets the parser continue building the AST (so you
  still get a tree) while the diagnostic bag collects the error. The NaN
  propagates through the AST and gets caught later by validation.
- **2026-08-26 — Parser: why `match` vs `consume`?** `match` = "if this token
  is here, eat it and return true" (optional). `consume` = "this token MUST be
  here, eat it or error" (mandatory). The distinction prevents bugs where
  optional fields are accidentally required or vice versa.
- **2026-08-26 — Parser: why the `plane` keyword is consumed by the caller?**
  In `parse_task()`, `match(TokenKind::KwPlane)` consumes the `plane` keyword
  before calling `parse_plane_constraint()`. This keeps the constraint parser
  focused on the constraint body (`{ ... }`), not the keyword that introduced it.
  Clean separation: the task loop decides *which* constraint, the parser builds
  *that* constraint.
- **2026-09-10 — Inlined Solver: why signed pivot clamping in Gaussian elimination?**
  Clamping small pivots with `diag = 1e-12` unconditionally flips negative pivots to positive, inverting the descent direction of $\Delta q$ and diverging on singular boundaries. Preserving signs `(diag >= 0 ? 1e-12 : -1e-12)` guarantees stable descent across 10,000 adversarial singular sweeps.
- **2026-09-10 — Real-Time Control: how does 1 kHz hard real-time control loop achieve 0 deadline misses?**
  Stack-allocated fixed arrays ($A[M \times M]$ and $y[M]$), branchless straight-line C, and zero heap allocations (`malloc`/`free`) ensure deterministic bounded execution ($3.53\ \mu$s median, $53.95\ \mu$s max jitter vs 1,000 $\mu$s deadline).
- **2026-09-10 — Kinematics Lowering: why must link lowering topologically sort parent dependencies?**
  Declarations in `.cudro` or branching dual-arm robots can list child links prior to parent links. Iteratively resolving links whose parents have already been constructed guarantees correct forward kinematic matrix products.

## 10. Why this matters now (modern robotics landscape)

| Shift | What changed | Why specialized kernels matter |
|---|---|---|
| Massive parallelism | Isaac/cuRobo-class simulation of thousands of robot instances | FK+collision+constraints dominate; generic calls are the wall |
| Learned policies everywhere | VLAs propose actions | proposals still need hard verification — learning is soft, safety must be exact |
| Humanoids/bimanual arrived commercially | 20–40 DoF whole-body control, dozens of simultaneous constraints | hand-written kernels stopped scaling; automation necessary |

Supporting angles: edge hardware (Jetson-class, offline) demands lightweight
runtime specialization (libtcc over LLVM monoliths on-robot); compiled kernels
are human-readable, diffable, auditable artifacts vs opaque weights; fleets get
one binary plus per-unit specs instead of N precompiled variants; research
iteration becomes file-edit rather than rebuild-cycle. Honest caveat: these
describe where the *technique* lands industrially — this project validates the
*mechanism* at learning scale only.
