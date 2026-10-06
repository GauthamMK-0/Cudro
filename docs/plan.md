# Cudro — Build Plan

Step-by-step milestones. Each stage = one compiler lesson, each with a
testable exit criterion. Status: `[x]` done, `[~]` in progress, `[ ]` pending.

**Current stage: M7 (Integration Demo & Constrained Planner) Complete**

---

## Locked decisions

| Decision | Choice |
|---|---|
| Host language | C++20 (GCC 13) |
| Robot input | custom `.cudro` format; URDF importer only as far-future stretch |
| Constraints | named kinds (`plane`, `relative_pose`, `com_above`) that desugar into an expression DAG in M3 |
| Primary backend | emit C source → JIT via libtcc (~5–20 ms compile latency) |
| M6.5 | LLVM IR backend (skipped / deferred in favor of direct JIT pipeline) |
| M8 | stretch: CUDA via NVRTC from the same IR (RTX 3060 available) |
| Tests | doctest via CMake FetchContent |
| Planner scope | kernel benchmark + toy sampler demo only |
| **Kernel ABI** (design constraint, applies from M3/M5 onward) | **obstacle/link-sphere positions are passed to generated kernels as a runtime array argument — never baked into the code as constants.** Moving objects become parameter updates; only structural spec changes trigger recompilation. Retrofitting this later is painful; deciding it now costs nothing. |

---

## Milestones

### M0 — Scaffold `[x]`
- [x] Toolchain verified: GCC 13.3, CMake 3.28, libtcc, Eigen3, AVX2 CPU, RTX 3060 + nvcc
- [x] CMake project builds; smoke test passes
- [x] Docs folder; plan/architecture/knowledge docs

### M1 — Front-end: lexer + parser `[x]`
- [x] CMake: FetchContent doctest; ASan+UBSan on test binaries
- [x] `diagnostic.hpp` — locations, bag, caret-rendered error printing
- [x] `token.hpp` + `lexer.*` — scanning with dedicated keyword kinds,
      trivia/comment skipping, always-advance recovery
- [x] `cudro --dump-tokens` (CLI flag live; parses real specs)
- [x] `spec/panda7.cudro` example spec
- [x] `ast.hpp` + `parser.*` — recursive descent per grammar in architecture.md, with error recovery
- [x] `spec/planar2r.cudro`
- [x] `--dump-ast`

**Exit:** valid specs parse to AST dumps; malformed input yields clean located errors,
never crashes; fuzz-lite green under ASan.

### M2 — Semantic analysis `[x]`
- [x] Symbol tables (robots, joints, links, tasks)
- [x] Reference resolution (parent, joint_ref, task.link, plane.link)
- [x] Duplicate detection (robot, joint, link, task names)
- [x] Kinematic tree validation (single root, no cycles, reachability)
- [x] Special handling: "world" as root parent, plane.link defaults to task.link
- [x] `cudro --check` runs full sema pipeline

**Exit:** duplicate ids / unknown refs / dim mismatches rejected with locations;
both example robots (panda7, planar2r) pass clean.

### M3 — Lowering to expression DAG `[x]`
- [x] Desugar constraint kinds into math expressions
- [x] Inline FK chains into the DAG; constant folding
- [x] `cudro --dump-dag`

**Exit:** folded DAG printed with node counts; plane constraint visibly expands
to `dot(n_world, R(q)·p_local + t(q)) − offset`.

### M4 — Autodiff pass `[x]`
- [x] Forward-mode dual numbers as a DAG-to-DAG pass
- [x] Jacobian computation for basic and compound operations (RotX/Y/Z, RotAxis, Translate, MatMul, MatVecMul, Dot)
- [x] `cudro --dump-jacobian` CLI flag
- [x] `tests/test_ad.cpp` unit test suite passes under ASan+UBSan

**Exit:** analytic Jacobian matches finite differences over configurations on test cases and robots.

### M5 — Scalar codegen + JIT `[x]`
- [x] Emit scalar C for constraint evaluation & Levenberg-Marquardt manifold projection
- [x] Emits `evaluate_constraints` and iterative solver `project(q_in, n, q_out)`
- [x] `libtcc` wrapper (`TCCJIT` and `JITModule`) compiling C source strings in memory (< 20 ms)
- [x] `cudro --jit-run` CLI flag executes in-memory JIT kernel
- [x] `tests/test_kernels.cpp` tests scalar codegen and JIT execution under ASan+UBSan

**Exit:** JIT compilation and execution verified clean (< 20 ms compile latency).

### M6 — Batched AVX2 backend `[x]`
- [x] Second emitter: batch-over-configurations layout (`generate_batched_c`)
- [x] Runtime dispatch & CPU feature detection (`cpu_supports_avx2`)
- [x] CLI flag `--jit-bench` evaluating batches of configurations
- [x] Test suite `test_kernels.cpp` validating batched multi-configuration projection under ASan+UBSan

**Exit:** batched multi-config kernel compilation and execution verified clean.

### Validation Anchor — Independent Eigen Reference `[x]`
- [x] `reference/eigen_reference.hpp` & `reference/eigen_reference.cpp`: Independent FK and constraint implementations for Panda7 and Planar2R.
- [x] `tests/test_reference.cpp`: 10,000 random configurations evaluated differentially against Eigen reference models (residual tolerance $< 10^{-4}$).
- [x] Random initial configuration projection convergence test verified.

### M7 — Integration demo + multi-robot `[x]`
- [x] Toy constrained sampler (C-RRT-Connect) using generated JIT kernel as inner loop (`include/cudro/planner.hpp`, `src/planner.cpp`)
- [x] Multi-robot generalization: `panda7` and `planar2r` plan constrained trajectories with zero compiler-code changes
- [x] CLI flag `--plan` executing end-to-end planning queries
- [x] `tests/test_planner.cpp` validating trajectory feasibility and waypoint constraint residuals under ASan+UBSan

**Exit:** feasible paths produced on both panda7 and planar2r; end-to-end untouched compiler pipeline verified.

### Phase 1 — Core Compiler Optimizations `[x]`
- [x] **Constant Deduplication (Hash-Consing)**: `ExprDAG::add_constant` caches constants via `constant_map_`; planar2r DAG size reduced by 79.5% (122 → 25 nodes).
- [x] **Middle-End Dead Code Cleanup**: Pruned duplicate orphaned subgraph in `lower_plane_constraint`; eliminated dead `neg_sin` creations in rotation builders.
- [x] **Back-End Loop Unrolling**: Fully unrolled $4 \times 4$ `MatMul` and `MatVecMul` in C emitter, eliminating TCC loop branch overhead, loop induction variables, and stack slot register spilling.
- [x] **Zero-Allocation String Streaming**: Cached `node_var_name` lookups, slashing in-memory JIT compile latency from ~14 ms down to 5.1–5.7 ms.
- [x] **Zero-Allocation Planner Scratchpads**: Pre-allocated `g_scratch_`, `q_cand`, and `q_proj` buffers in `ConstrainedPlanner`, eliminating inner-loop heap churn during RRT tree expansion.
- [x] **Performance Multipliers**: Batched constraint evaluation reaches 2.64M – 6.49M configs/sec; manifold projection throughput reaches 60k – 96k solves/sec (10.4 – 16.7 $\mu$s/solve).

### Phase 2 — Analytical Jacobian Codegen & Activity Analysis `[x]`
- [x] **Activity-Guided Symbolic Differentiation (`src/ad.cpp`)**: Scan DAG inputs to compute active variables per joint index; prune inactive product-rule branches in $d(A \cdot B) = dA \cdot B + A \cdot dB$ so serial kinematic chains never branch exponentially.
- [x] **Exact Analytical Jacobian Construction (`include/cudro/ad.hpp`, `src/ad.cpp`)**: Build symbolic gradient DAG nodes directly into `LowerResult::constraint_jacobians` for all kinematic transformations (including `Mat4`, `MatMul`, `MatVecMul`, `Dot`, `RotAxis`, `Translate`).
- [x] **Unified Single-Pass Kernel Evaluator (`src/codegen_c.cpp`)**: Replaced finite differences in `evaluate_dag(q, out_g, out_J)`, evaluating both constraints $g(q)$ and analytical Jacobian entries $J(q)$ in a single forward evaluation.
- [x] **Zero-Perturbation Levenberg-Marquardt Solver**: Completely eliminated the $N$-iteration finite difference perturbation loop inside `project_single`, achieving an exact, zero-truncation-error manifold projection step.
- [x] **Performance**: Manifold projection latency on Panda7 dropped to 11.6 $\mu$s / config; test suite run-time slashed to 0.78s across all 10 suites under ASan+UBSan.

### Phase 3 — Multi-Constraint Normal Equations & Branching Kinematics `[x]`
- [x] **Arbitrary Multi-Constraint Matrix Assembly ($M \ge 1$)**: Synthesizes a full $M \times N$ 2D analytical Jacobian matrix DAG, enabling simultaneous satisfaction of $M=2, 3, 6, \dots$ constraints on arbitrary body links.
- [x] **Inlined Normal Equations Linear Solver**: Emits a stack-allocated $M \times M$ linear system $(J J^T + \lambda I) y = -g$ solved via inlined Gaussian elimination with partial row pivoting and signed pivot clamping (`diag = (diag >= 0 ? 1e-12 : -1e-12)`). Emits branchless closed-form rank-1 update for $M=1$.
- [x] **Per-Constraint Link Overrides**: Syntax and parser support for optional link overrides on constraint definitions (`plane { link <name>; ... }`), lowering to the targeted link's world transform.
- [x] **Branching Kinematic Trees**: Implemented topological iterative transform resolution in `build_all_link_transforms`, seamlessly compiling dual-arm bimanual robots (e.g. `bimanual14.cudro`) regardless of link declaration order.
- [x] **Symbolic Rodrigues Differentiation**: Differentiates revolute joints with arbitrary 3D spatial rotation axes (`RotAxis`) analytically using Rodrigues matrix derivatives.
- [x] **Verification**: `tests/test_multiconstraint.cpp` validates $M=2, 3$ simultaneous multi-plane manifold projections with residual $< 10^{-6}$.

### Phase 4 — 1 kHz Controller Loop & Adversarial Singularity Resilience `[x]`
- [x] **Hard Real-Time 1 kHz Controller Loop Benchmark**: Simulated closed-loop control tracking over 1,000 steps with a 1,000 $\mu$s deadline: achieved **median latency of 3.53 $\mu$s**, **peak jitter of 53.95 $\mu$s**, and **zero missed deadlines**.
- [x] **Adversarial Singularity Resilience**: Stress-tested across 10,000 adversarial, boundary, and near-singular configurations: achieved **100% finite outputs** with zero NaN or Inf escapes.
- [x] **Dense Multi-Link Obstacles ($M=6$)**: Solves simultaneous multi-link constraints on 7-DoF manipulators (`spec/panda7_constrained.cudro`).
- [x] **Complex Workloads Suite (`tests/test_complex_workloads.cpp`)**: 11 test suites passing clean under ASan + UBSan.

### M8 — CUDA backend via NVRTC `[ ]` (stretch)
- [ ] Same DAG → CUDA C → NVRTC module → kernel launch

**Exit:** same validation suite passes on GPU with adjusted tolerances.

---

## Validation discipline (all stages)

- **Differential testing** vs independent Eigen reference — the backbone (`test_reference`)
- **Finite differences** check the autodiff pass (`test_ad`)
- **Golden dumps** guard lexer/parser/codegen regressions
- **Fuzz-lite**: random/mangled specs must parse or fail cleanly, never segfault (`fuzz_spec`)
- **ASan + UBSan** on every test run across all 9 test suites

---

## Deferred integration glue (analyzed, deliberately NOT built)

From the VLA/world-model feasibility analysis: supporting a neural-policy
safety-filter use case requires ~0% change to the compiler core and roughly a
working week of glue *after* M5 exists:

- in-process facade API (`cudro::compile(Spec) → KernelHandle`) over the existing stages (~150 lines)
- background-thread compile + atomic function-pointer hot-swap so the control loop never waits (~100 lines)

Recorded so the architecture claim ("cheap to integrate because stages are
separate and the IR is portable") stays checkable — but out of build scope.
Perception-side spec generation and real-time-safety review of the swap path
are explicitly out of scope.

---

## Out of scope (per idea.md)

Beating McVAMP's numbers, mesh collision, new planning algorithms, production
robotics framework. Learning-first; narrow claim = AOT→runtime compilation gap.