# Cudro ⚡

**A Runtime-Recompiling Constraint Compiler for Real-Time Robotic Motion Planning & Safety Validation**

---

## 📌 Overview

**Cudro** is a lightweight, domain-specific compiler designed for robotics motion planning and real-time safety filtering. It takes declarative robot kinematic descriptions and task-space manifold constraints (`.cudro` specifications), lowers them to an inlined **Expression DAG**, differentiates them via **forward-mode dual numbers**, and generates specialized C kernels compiled **just-in-time (JIT) at runtime** in `< 15 ms` via `libtcc`.

### The Core Problem Cudro Solves
State-of-the-art vectorized motion planners (e.g. *McVAMP*, IROS 2026) use ahead-of-time (AOT) tracing compilers to generate loop-unrolled SIMD kernels for constraint projection. However, AOT compilers require offline recompilation and relinking whenever a constraint or robot geometry changes.

**Cudro closes this AOT → Runtime gap**:
- **Dynamic Task Adaptation**: Accept newly perceived constraints (e.g., table height changes, new tool lengths, dynamic keep-out zones) on the fly.
- **Instant Specialization**: Compile specialized machine code directly in memory in milliseconds without restarting or rebuilding the host process.
- **Manifold Projection**: Damped Levenberg-Marquardt (LM) iterative solver inside the generated kernel projects unconstrained configurations onto safe task manifolds at high frequencies.
- **Safety Layer for Vision-Language-Action (VLA) Models**: Project noisy neural policy action proposals onto certified constraint manifolds at 100–1000 Hz.

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

Cudro comes with a complete suite of 10 test suites guarded by **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)**:

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

// 4. JIT Compile in Memory (< 15 ms)
auto mod = cudro::TCCJIT::compile(c_code);
auto project_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");

// 5. Execute in real-time control loop
std::vector<float> q_init = {0.2f, 0.3f};
std::vector<float> q_proj(lr.num_inputs);
project_fn(q_init.data(), lr.num_inputs, q_proj.data());
```

---

## 📊 Performance Characteristics

| Metric | Measured Value |
|---|---|
| **JIT Compilation Latency** | ~9 – 14 ms (in-memory) |
| **Batched Constraint Evaluation Throughput** | ~330,000 configs / sec |
| **Batched Manifold Projection Throughput** | ~8,500 full LM solves / sec |
| **Differential Error vs Eigen Reference** | $< 10^{-4}$ across 10,000 random configurations |

---

## 📖 Codebook & Developer Documentation

For a detailed file-by-file walkthrough of compiler concepts (lexer, recursive descent parsing, symbol tables, DAG lowering, dual-number AD, C emission, JIT compilation, and CUDA integration), see the [Cudro Codebook](docs/codebook/README.md).
