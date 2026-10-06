# Chapter 15: Compiler CLI Driver & Tools

## Files Covered
- `src/main.cpp`

---

## 1. Architectural Purpose
`main.cpp` is the **command-line interface** to the entire Cudro compiler pipeline. It exposes every stage as a flag, enabling inspection and testing at each level:

```bash
cudro --dump-tokens   spec/panda7.cudro
cudro --dump-ast      spec/panda7.cudro
cudro --check         spec/panda7.cudro
cudro --dump-dag      spec/panda7.cudro
cudro --dump-jacobian spec/panda7.cudro
cudro --emit-c        spec/panda7.cudro
cudro --jit-run       spec/panda7.cudro
cudro --jit-bench     spec/panda7.cudro
```

---

## 2. CLI Commands Reference

| Flag | Description |
|---|---|
| `--dump-tokens` | Tokenizes spec and prints token stream with file:line:col locations |
| `--dump-ast` | Parses spec and prints formatted AST hierarchy |
| `--check` | Runs semantic analysis (symbol resolution, kinematic tree validation, cycle checks) |
| `--dump-dag` | Lowers AST to Expression DAG and prints nodes after constant folding |
| `--dump-jacobian` | Computes and prints analytical Jacobian matrix $\frac{\partial g}{\partial q}$ |
| `--emit-c` | Emits scalar C source code containing DAG evaluation and LM projection solver |
| `--jit-run` | Compiles kernel in memory via `libtcc` (< 15 ms) and executes manifold projection on $q=0$ |
| `--jit-bench` | Compiles batched kernel in memory and benchmarks throughput on 10,000 configurations |
| `--version` | Displays compiler version string |
| `--help`, `-h` | Prints usage help message |

---

## 3. Implementation Highlights

### High-Resolution Timing and Diagnostics
`main.cpp` captures both in-memory compilation latency and steady-state execution throughput:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
auto mod = cudro::TCCJIT::compile(batched_src);
auto t1 = std::chrono::high_resolution_clock::now();
double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
```