# Cudro ⚡

**A Runtime-Recompiling Constraint Compiler for Real-Time Robotic Motion Planning & Safety Validation**

---

## 📌 Overview

**Cudro** is a lightweight, domain-specific compiler designed for robotics motion planning and real-time safety filtering. It takes declarative robot kinematic descriptions and task-space manifold constraints (`.cudro` specifications), lowers them to an inlined **Expression DAG**, differentiates them via **forward-mode dual numbers**, and generates specialized C kernels compiled **just-in-time (JIT) at runtime** in `< 20 ms` via `libtcc`.

### The Core Problem Cudro Solves
State-of-the-art vectorized motion planners (e.g. *McVAMP*, IROS 2026) use ahead-of-time (AOT) tracing compilers to generate loop-unrolled SIMD kernels for constraint projection. However, AOT compilers require offline recompilation and relinking whenever a constraint or robot geometry changes.

**Cudro closes this AOT → Runtime gap**:
- **Dynamic Task Adaptation**: Accept newly perceived constraints (e.g., table height changes, new tool lengths, dynamic keep-out zones) on the fly.
- **Instant Specialization**: Compile specialized machine code directly in memory in milliseconds without restarting or rebuilding the host process.
- **Safety Layer for Vision-Language-Action (VLA) Models**: Project unconstrained/noisy neural policy action proposals onto certified constraint manifolds at 100–1000 Hz.

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
            │     - Scalar C & Batched Emitters   │
            │     - In-memory libtcc compilation  │
            └──────────────────┬──────────────────┘
                               │
               Executable Function Pointer:
        project() / project_batch() in < 20 ms
```

---

## 🚀 Getting Started

### Prerequisites

- **C++ Compiler**: GCC 13+ or Clang (C++20 support required)
- **Build System**: CMake 3.16+
- **JIT Library**: `tcc` / `libtcc-dev`
- **Linear Algebra**: `Eigen3` (optional, for reference testing)

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

Cudro comes with a complete suite of unit and regression tests guarded by **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)**:

```bash
ctest --test-dir build --output-on-failure
```

Test Suites:
1. `smoke`: Scaffolding verification
2. `lexer`: Tokenization and syntax recovery
3. `parser`: AST node creation and grammar rules
4. `sema`: Duplicate detection, reference resolution, cycle checks
5. `dag`: DAG IR builder, constant folding, FK lowering
6. `ad`: Forward-mode automatic differentiation and numerical Jacobians
7. `kernels`: Scalar and batched multi-configuration JIT execution
8. `fuzz`: Fuzz-lite crash-freedom test under random byte streams

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

# 6. Run end-to-end in-memory JIT compile and evaluate at q=0
./build/cudro --jit-run spec/panda7.cudro

# 7. Benchmark batched multi-configuration projection
./build/cudro --jit-bench spec/panda7.cudro
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

You can use Cudro directly as a static library (`cudro_core`) inside your C++ application:

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

// 2. Lower to DAG
cudro::ExprDAG dag;
auto constraint_outputs = cudro::lower(spec, dag);

cudro::LowerResult lr;
lr.dag = std::move(dag);
lr.constraint_outputs = std::move(constraint_outputs);
lr.num_inputs = lr.dag.num_inputs();

// 3. Generate C code
std::string c_code = cudro::generate_scalar_c(lr);

// 4. JIT Compile in Memory (< 20 ms)
auto mod = cudro::TCCJIT::compile(c_code);
auto project_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");

// 5. Execute in real-time control loop
std::vector<float> q = {0.0f, 0.1f, -0.2f, 0.0f, 0.5f, 0.0f, 0.0f};
std::vector<float> g(lr.constraint_outputs.size());
project_fn(q.data(), lr.num_inputs, g.data());
```

---

## 📖 Codebook & Developer Documentation

For a detailed file-by-file walkthrough of compiler concepts (lexer, recursive descent parsing, symbol tables, DAG lowering, dual-number AD, C emission, JIT compilation, and CUDA integration), see the [Cudro Codebook](docs/codebook/README.md).
