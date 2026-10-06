# Cudro ⚡

**A Domain-Specific Compiler for Robot Kinematics and Constraint Manifold Projection**

---

## 📌 Overview

**Cudro** is a lightweight, dependency-free domain-specific compiler written from scratch in C++. It parses declarative robot kinematic descriptions and task manifold constraints (`.cudro` specifications), lowers them into an inlined, hash-consed **Expression DAG**, computes analytical constraint Jacobians via **forward-mode dual numbers / symbolic AD**, and emits specialized, allocation-free C kernels compiled **just-in-time (JIT) in memory** in `< 10 ms` via `libtcc`.

### What Cudro Does
State-of-the-art motion planning algorithms (such as Constrained RRT) require repeatedly projecting robot configurations onto task constraint manifolds ($g(q) = 0$, e.g. maintaining an end-effector orientation or keeping a tool tip on a plane). 

Generic libraries often evaluate forward kinematics and Jacobians through deep matrix multiplications, virtual dispatch, or dynamic heap allocations. Cudro takes a compiler approach:
- **Kinematic Inlining**: Lowers forward kinematics along kinematic trees directly into straight-line scalar operations with common subexpression elimination.
- **Analytical Derivatives via DAG AD**: Generates exact symbolic Jacobian expressions directly inside the DAG, eliminating numerical finite-difference perturbations.
- **In-Memory JIT Compilation**: Compiles specialized C kernels containing an inlined Levenberg-Marquardt (LM) solver directly into machine code via `libtcc` in milliseconds.
- **Constrained Motion Planning**: Integrates directly with a C-RRT planner to generate continuous constraint-satisfying paths without external optimization solver dependencies.

---

## 🏗️ Compiler Architecture

Cudro is structured as a classical, clean three-part compiler pipeline:

```
                      [ .cudro Spec File ]
                               │
            ┌──────────────────▼──────────────────┐
            │  1. Front-End: Lexer & Parser       │
            │     - Tokenizer with caret error UI │
            │     - Recursive descent parser      │
            └──────────────────┬──────────────────┘
                               │ AST
            ┌──────────────────▼──────────────────┐
            │  2. Semantic Analysis (Sema)        │
            │     - Kinematic tree validation     │
            │     - Symbol & reference resolution │
            └──────────────────┬──────────────────┘
                               │ Validated AST
            ┌──────────────────▼──────────────────┐
            │  3. Lowering & Desugaring           │
            │     - FK inlining (Rodrigues rot)   │
            │     - Constant folding              │
            └──────────────────┬──────────────────┘
                               │ Expression DAG
            ┌──────────────────▼──────────────────┐
            │  4. Automatic Differentiation (AD)  │
            │     - Forward-mode dual numbers     │
            │     - Jacobian ∂g/∂q calculation    │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │  5. Code Generation & JIT Runtime   │
            │     - Scalar & Batched Emitters     │
            │     - Levenberg-Marquardt solver    │
            │     - In-memory libtcc compilation  │
            └──────────────────┬──────────────────┘
                               │
               Executable Function Pointers:
     project(), evaluate_constraints(), project_batch()
```

---

## 🚀 Getting Started

### Prerequisites

- **C++ Compiler**: GCC 13+ or Clang (C++20 support required)
- **Build System**: CMake 3.16+
- **JIT Library**: `tcc` / `libtcc-dev`
- **Linear Algebra**: `Eigen3` (`libeigen3-dev` for reference testing)

On Ubuntu / Debian:
```bash
sudo apt-get update
sudo apt-get install -y cmake g++ libtcc-dev tcc libeigen3-dev
```

---

### Building Cudro

```bash
git clone <repo-url>
cd Cudro

# Configure & build
cmake -B build -S .
cmake --build build
```

---

### Running Tests

Cudro comes with a complete suite of 11 test suites guarded by **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)**:

```bash
ctest --test-dir build --output-on-failure
```

#### Test Suites
1. `smoke`: Toolchain and scaffold verification
2. `lexer`: Tokenization and syntax error recovery
3. `parser`: AST node creation, grammar validation, and panic-mode recovery
4. `sema`: Duplicate detection, reference resolution, kinematic tree cycle checks
5. `dag`: DAG IR builder, constant folding, inlined FK lowering
6. `ad`: Forward-mode dual numbers and analytical Jacobians
7. `kernels`: Scalar and batched multi-configuration JIT execution and LM projection
8. `reference`: **Differential validation** against independent Eigen reference models across 10,000 configurations
9. `planner`: **Constrained motion planning** (C-RRT-Connect) validating continuous manifold trajectory generation on multi-robot models
10. `fuzz`: Fuzz-lite crash-freedom test under random byte streams
11. `complex_workloads`: 1 kHz control loop benchmark, 14-DoF bimanual branching dual-arms, $M=6$ multi-link constraints, arbitrary spatial skew axes (`RotAxis`), and 10,000 adversarial singularity stress tests

---

## 🛠️ CLI Usage & Flags

The `cudro` binary exposes every stage of the compiler pipeline:

```bash
# 1. Check syntax and semantic validity
./build/cudro --check spec/panda7.cudro

# 2. Inspect token stream
./build/cudro --dump-tokens spec/panda7.cudro

# 3. View Abstract Syntax Tree (AST)
./build/cudro --dump-ast spec/panda7.cudro

# 4. Inspect lowered Expression DAG
./build/cudro --dump-dag spec/panda7.cudro

# 5. Inspect computed analytical Jacobians
./build/cudro --dump-jacobian spec/panda7.cudro

# 6. Emit generated C source
./build/cudro --emit-c spec/panda7.cudro

# 7. Run in-memory JIT compile and evaluate manifold projection
./build/cudro --jit-run spec/panda7.cudro

# 8. Benchmark batched multi-configuration projection throughput
./build/cudro --jit-bench spec/planar2r.cudro

# 9. Plan a constraint-satisfying trajectory via in-kernel JIT solver
./build/cudro --plan spec/panda7.cudro
```

---


## 📝 Example Specification (`.cudro`)

```cudro
robot panda7 {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0.333]; limits [-2.8973, 2.8973]; }
  joint j2 { type revolute; axis [0,1,0]; origin [0,0,0]; limits [-1.7628, 1.7628]; }
  joint j3 { type revolute; axis [0,0,1]; origin [0,-0.316,0]; limits [-2.8973, 2.8973]; }
  joint j4 { type revolute; axis [0,1,0]; origin [0,0,0.0825]; }
  joint j5 { type revolute; axis [0,0,1]; origin [0,0.384,0]; limits [-2.8973, 2.8973]; }
  joint j6 { type revolute; axis [0,1,0]; origin [0,0,0]; }
  joint j7 { type revolute; axis [0,0,1]; origin [0,0.107,0]; limits [-2.8973, 2.8973]; }

  link base   { spheres [[0,0,0.06, 0.07]]; parent world; joint_ref j1; }
  link arm1   { spheres [[0,0,0.15, 0.06]]; parent base; joint_ref j2; }
  link arm2   { spheres [[0,0,0.12, 0.05]]; parent arm1; joint_ref j3; }
  link forearm { spheres [[0,0,0.18, 0.048]]; parent arm2; joint_ref j5; }
  link ee     { spheres [[0,0,0.02, 0.03]]; parent forearm; joint_ref j7; }
}

task cup_on_table {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.02; }
}

clearance { min_distance 0.03; }
```

---

## 💻 C++ API Usage

You can use Cudro directly as an in-process library (`cudro_core`):

```cpp
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>
#include <cudro/codegen_c.hpp>
#include <cudro/jit_tcc.hpp>

// 1. Parse & Check
cudro::DiagnosticBag diags;
cudro::Lexer lexer("spec.cudro", spec_source_string, diags);
auto tokens = lexer.tokenize();
cudro::Parser parser(tokens, diags);
auto spec = parser.parse();

cudro::Sema sema(spec, diags);
if (!sema.analyze()) {
    cudro::print_diagnostics(diags, spec_source_string);
    return;
}

// 2. Lower to Expression DAG
cudro::ExprDAG dag;
auto constraint_outputs = cudro::lower(spec, dag);

cudro::LowerResult lr;
lr.dag = std::move(dag);
lr.constraint_outputs = std::move(constraint_outputs);
lr.num_inputs = lr.dag.num_inputs();

// 3. Generate C code
std::string c_code = cudro::generate_scalar_c(lr);

// 4. JIT Compile in Memory (< 10 ms)
auto mod = cudro::TCCJIT::compile(c_code);
auto project_fn = mod.get_symbol<int(*)(const float*, int, float*)>("project");
auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");

// 5. Execute projection
std::vector<float> q_init = {0.2f, 0.3f};
std::vector<float> q_proj(lr.num_inputs);
int status = project_fn(q_init.data(), lr.num_inputs, q_proj.data());
if (status == 0) {
    // Successfully converged onto the constraint manifold
}
```

---

## 📊 Performance Characteristics

| Metric | Measured Value (x86_64 Linux Benchmark) | Capability / Notes |
|---|---|---|
| **1 kHz Control Step Latency** | **3.53 $\mu$s median latency** (max 53.95 $\mu$s jitter) | Zero allocations in inner loop; well within 1,000 $\mu$s cycle budget |
| **Adversarial Singularity Resilience** | **100% finite outputs** across 10,000 stress steps | **Zero NaN / Inf escapes** under near-singular rank conditions |
| **Simultaneous Multi-Constraint Solving** | **Arbitrary $M \ge 1$ ($M=2, 3, 6, \dots$)** | Inlined damped normal equations with signed partial pivoting |
| **Branching Kinematic Trees** | **14-DOF Bimanual Dual-Arm** (`bimanual14.cudro`) | Topological iterative resolution of arbitrary link hierarchies |
| **Spatial Skew Axis Geometry** | **Arbitrary unit axes $\hat{k}$** (`arbitrary_axes_6r.cudro`) | Exact symbolic Rodrigues differentiation (zero truncation error) |
| **In-Memory JIT Compilation Latency** | **5.1 – 9.5 ms** | **~2.5× faster** (from ~14 ms) |
| **Batched Constraint Evaluation Throughput** | **2,640,000 – 6,490,000 configs / sec** | **~8× – 20× faster** |
| **Batched Manifold Projection Throughput** | **85,920 – 120,000 full LM solves / sec** | **~10× – 14× faster** |
| **Panda 7-DOF Projection Latency** | **11.6 $\mu$s / solve** (down from 16.7 $\mu$s) | **Exact analytical gradient (zero truncation error)** |
| **Planar2R Lowered DAG Size** | **25 nodes** (down from 122) | **79.5% node reduction** via whole-program hash-consing |
| **Jacobian Codegen Strategy** | **Single-pass unified evaluator `evaluate_dag(q, g, J)`** | Eliminates $N$ finite-diff DAG passes per LM step |
| **Differential Error vs Eigen Reference** | $< 10^{-4}$ across 10,000 random configurations | 100% verified agreement (30,022 assertions) |
