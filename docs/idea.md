# Runtime-Recompiling Constraint Compiler for Real-Time Motion Planning

## Core Idea

Build a small compiler that takes a robot's kinematic constraints (joint
limits, link geometry, manifold/task constraints) as **declarative input**
and generates a specialized SIMD/GPU kernel for constraint-projection during
real-time motion planning — with the compilation step itself happening
**at runtime**, so a robot or planner can accept a newly specified
constraint (or a new robot description) and get a specialized kernel
without an offline build/compile cycle.

This targets a **named, current, open problem** — but a narrower one than
"no automatic compiler exists." A 2026 vectorized manifold-constrained
motion planner (McVAMP, Purdue, to appear IROS 2026, building on the VAMP
library) already contains an automatic tracing compiler: given a
constraint specification, it uses Pinocchio + CppAD to trace and generate
branch-free, loop-unrolled SIMD code for the constraint function, its
Jacobian, and even a full Levenberg-Marquardt projection step (including a
custom branchless Cholesky solve) — validated across three structurally
different robots (7-DoF arm, 14-DoF bimanual system, 28-DoF humanoid) and
several constraint types (plane/line, TSR pose, closed-linkage, center of
mass). VAMP, its predecessor, already does the same thing for forward
kinematics and collision checking straight from URDF files.

What the authors explicitly flag as unsolved is narrower: their compiler
runs **ahead-of-time** — you must know the constraint set before tracing —
and they name "dynamic constraint and robot compilation" as the specific
future direction. That is the actual open slot this project targets: make
the compile step happen at runtime/JIT, rather than as an offline build.

**What this project is:** a compiler with a small declarative IR for robot
constraints (joint limits, collision primitives, task-space manifold
constraints), a code-generation pass that specializes constraint-projection
code per-robot **without requiring an offline recompilation cycle when the
constraint set changes**, and a validation harness comparing against both
a hand-written reference and, where feasible, against the ahead-of-time
compiled baseline's numbers.

**What this project deliberately is not:** a general-purpose robotics
planning framework, a new planning algorithm, a claim to match or beat
McVAMP's published speedup numbers, or a claim that automatic
constraint-to-SIMD compilation itself is new — it isn't. The contribution
is narrowly about closing the ahead-of-time → runtime gap the McVAMP
authors themselves identified as open, validated at smaller scale than
their 28-DoF humanoid.

---

## Prior Art This Project Builds On (Read First)

Before writing any code, the compiler pass design should be checked
against these systems, since duplicating their approach without
acknowledging it would not be a compiler contribution, just a
reimplementation:

- **VAMP** (Thomason, Kingston, Kavraki — "Motions in Microseconds via
  Vectorized Sampling-Based Planning," ICRA 2024). Tracing compiler,
  URDF-in, SIMD FK/collision-out. This is the direct precedent for R1–R2
  below; read the tracing-compiler section closely before reinventing it.
- **McVAMP** (Iyer, Chang, Liu, Gu, Kingston — IROS 2026, to appear).
  Extends VAMP's tracing compiler to constraint functions, Jacobians, and
  a vectorized LM projection step. This is the direct precedent for R3.
  The stated future-work gap ("requires knowledge of constraints a priori")
  is the actual target of this project.
- **RobCoGen** and related kinematics/dynamics DSLs (Frigerio et al.).
  Established pattern of declarative robot description → generated
  optimized code, predating both of the above by a decade. Useful prior
  art for IR design, not a competitor on the compilation-target front.
- **cuRobo**. GPU-based, but enforces constraints via soft penalties in
  trajectory optimization rather than hard manifold projection — different
  enough in approach that it's not direct prior art for R3/R4, but worth
  citing as the GPU alternative if this project ends up choosing a CUDA
  target (see R3).

---

## Requirements

### R1 — Declarative constraint IR
- Define a small language/schema for describing: joint limits (per-DoF
  min/max), link geometry (spheres/capsules for collision — no need for
  full mesh collision), and manifold/task constraints (e.g. "end-effector
  stays on this plane," "two arms maintain fixed relative pose").
- Design this to be genuinely portable — not a thin wrapper around
  Pinocchio/CppAD-specific tracing — since a toolchain-independent IR is
  one of the few places this project can claim an actual delta over
  VAMP/McVAMP rather than reproducing them.

### R2 — Constraint-projection compilation (scalar/CPU reference)
- Implement the compiler pass that lowers the IR into constraint-projection
  code: given a candidate configuration, project it onto the feasible
  manifold (joint limits + collision + task constraints).
- Establish correctness of this scalar path independently, before touching
  vectorization — this step is standard practice and not itself a novelty
  claim, but it's required scaffolding for R3/R4.

### R3 — Runtime/JIT SIMD specialization pass (the actual novel piece)
- Add a lowering pass that vectorizes the generated projection code across
  many candidate configurations simultaneously, **compiled at runtime**
  when a new constraint or robot description is supplied — no
  offline/ahead-of-time build step required to get a specialized kernel.
- This is the part that isn't already published: McVAMP's tracing compiler
  is ahead-of-time. Demonstrate the actual delta by timing "time from new
  constraint description to first specialized-kernel execution" and
  showing it's compatible with runtime operation (e.g. under a second,
  not a recompile-and-relink cycle).
- Pick CPU SIMD intrinsics or a CUDA kernel as the primary target and
  document the choice. Note: if CUDA is chosen, this is also a genuine
  differentiator from McVAMP (CPU-SIMD-only) and worth stating as such —
  but don't claim it as the JIT contribution; keep those two claims
  separate since a reviewer familiar with cuRobo will ask.

### R4 — Validation against hand-written and ahead-of-time-compiled baselines
- Reproduce a simplified version of McVAMP's benchmark: a robot arm
  smaller than their 28-DoF humanoid (6–7 DoF is reasonable), with joint
  limits and simple collision constraints, sampling-based constrained
  motion planning.
- Compare three things, honestly: (a) correctness — generated projection
  matches a hand-written reference for the same robot; (b) steady-state
  performance — runtime-compiled kernel vs. scalar baseline, reporting
  actual speedup even if far short of McVAMP's numbers at this smaller
  scale; **(c) the actual point of this project — compilation latency and
  overhead of runtime specialization vs. an ahead-of-time build, since
  that's the axis where this work can claim something new.**
- **Measure single-config (batch=1) call latency separately from
  batch throughput** — this is the number that matters for a
  streaming/per-tick consumer (VLA action validation, world-model
  rollout projection), as distinct from the batch throughput number
  that matters for sampling-based planning. Both numbers reported.

### R5 — Multi-robot generalization test
- Feed the compiler a second, structurally different robot description
  (different DoF count, different constraint set) without changing
  compiler code, and confirm correct, specialized code is produced for
  that robot too — via the runtime path, not a rebuild.

### R6 — Correctness before performance, always
- No speedup or latency number is reported until the generated projection
  code has been validated against a hand-written/reference implementation
  for numerical correctness on both robots in R5.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Robot Constraint Description (declarative IR)                     │
│  - joint limits per DoF                                              │
│  - link geometry (spheres/capsules for collision)                    │
│  - task/manifold constraints (e.g. end-effector plane constraint)    │
│  Supplied at RUNTIME — no offline build step assumed                 │
└───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Parser → IR             │
                   └──────────┬──────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │ Constraint-Projection Codegen   │  scalar/CPU reference
                   │ (Phase: correctness first)       │  path, validated first
                   └──────────┬──────────────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │ Runtime SIMD/GPU Specialization  │  vectorize across
                   │ (compiled just-in-time, per-robot)│  candidate configs,
                   └──────────┬──────────────────┘  no rebuild required
                               │
                               ▼
         ┌────────────────────────────────────────────┐
         │ Sampling-Based Motion Planner (uses           │
         │ generated projection kernel as its inner loop) │
         └──────────────────┬─────────────────────────┘
                             │
                             ▼
         ┌────────────────────────────────────────────┐
         │ Validation Harness                              │
         │ - correctness vs. hand-written reference          │
         │ - performance vs. scalar baseline                  │
         │ - compilation latency vs. ahead-of-time baseline   │
         │ - **single-config latency** vs. batch throughput  │
         │ - multi-robot generalization test (R5)             │
         └────────────────────────────────────────────┘
```

### Locked Design Decisions

| Decision | Choice |
|---|---|
| Host language | C++20 (GCC 13) |
| Robot input | custom `.cudro` format; URDF importer only as far-future stretch |
| Constraints | named kinds (`plane`, `relative_pose`, `com_above`) that desugar into an expression DAG in M3 |
| Primary backend | emit C source → JIT via libtcc (~5–20 ms compile latency) |
| **M6.5** | **official** (not stretch): LLVM IR back-end + tcc-vs-LLVM comparison — this is the experiment that validates the libtcc design choice |
| M8 | stretch: CUDA via NVRTC + Jitify from the same DAG (RTX 3060 available) |
| Tests | doctest via CMake FetchContent |
| Planner scope | kernel benchmark + toy sampler demo only |
| **Kernel ABI** (design constraint, applies from M3/M5 onward) | **obstacle/link-sphere positions are passed to generated kernels as a runtime array argument — never baked into the code as constants.** Moving objects become parameter updates; only structural spec changes trigger recompilation. Retrofitting this later is painful; deciding it now costs nothing. |

---

## Validation discipline (all stages)

- **Differential testing** vs independent Eigen reference — the backbone
- **Finite differences** check the autodiff pass (M4)
- **Golden dumps** guard lexer/parser/DAG/codegen regressions
- **Fuzz-lite**: random/mangled specs must parse or fail cleanly, never segfault
- **ASan + UBSan** on every test run
- **Measure single-config (batch=1) call latency separately from
  batch throughput** — this is the number that matters for a
  streaming/per-tick consumer (VLA action validation, world-model
  rollout projection), as distinct from the batch throughput number
  that matters for sampling-based planning. Both numbers reported.

---

## Honest scope boundary

State clearly in any write-up: automatic compilation of
constraint-projection kernels from a declarative robot/constraint
description is **not new** — VAMP and McVAMP already do this, ahead of
time, validated on robots up to 28 DoF. What is not yet published, per the
McVAMP authors' own stated future work, is doing this compilation at
**runtime** rather than ahead of time. That is the specific, narrow claim
this project makes, validated at smaller scale than McVAMP's demonstrations
and explicitly positioned as closing a gap those authors named themselves —
not as a from-scratch solution to an unaddressed problem.