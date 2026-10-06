# Cudro Codebook: Compilers for Robotics

Welcome to the **Cudro Codebook**. This guide walks through every single file in `include/` and `src/` as a textbook case study on building a domain-specific compiler for robotics.

## Chapter Directory

1. **[Chapter 01: Core Versioning & Metadata](01_version.md)** (`include/cudro/version.hpp`)
2. **[Chapter 02: Source Locations & Diagnostics](02_diagnostics.md)** (`include/cudro/diagnostic.hpp`, `src/diagnostic.cpp`)
3. **[Chapter 03: Lexical Tokens & Keywords](03_tokens.md)** (`include/cudro/token.hpp`, `src/token.cpp`)
4. **[Chapter 04: The Lexer Scanner Engine](04_lexer.md)** (`include/cudro/lexer.hpp`, `src/lexer.cpp`)
5. **[Chapter 05: Abstract Syntax Trees (AST)](05_ast.md)** (`include/cudro/ast.hpp`, `src/ast.cpp`)
6. **[Chapter 06: Recursive Descent Parser](06_parser.md)** (`include/cudro/parser.hpp`, `src/parser.cpp`)
7. **[Chapter 07: Semantic Analysis & Symbol Resolution](07_sema.md)** (`include/cudro/sema.hpp`, `src/sema.cpp`)
8. **[Chapter 08: Expression Directed Acyclic Graph (DAG) IR](08_dag.md)** (`include/cudro/dag.hpp`, `src/dag.cpp`)
9. **[Chapter 09: Lowering & Kinematics Inlining](09_lower.md)** (`include/cudro/lower.hpp`, `src/lower.cpp`)
10. **[Chapter 10: Automatic Differentiation Pass](10_ad.md)** (`include/cudro/ad.hpp`, `src/ad.cpp`)
11. **[Chapter 11: C Source Code Generation](11_codegen_c.md)** (`include/cudro/codegen_c.hpp`, `src/codegen_c.cpp`)
12. **[Chapter 12: In-Memory JIT Compilation via TinyCC](12_jit_tcc.md)** (`include/cudro/jit_tcc.hpp`, `src/jit_tcc.cpp`)
13. **[Chapter 13: Industrial LLVM IR & ORC JIT Backend](13_codegen_llvm.md)** (`include/cudro/codegen_llvm.hpp`, `src/codegen_llvm.cpp`)
14. **[Chapter 14: GPU Massive Parallelism via CUDA NVRTC](14_codegen_cuda.md)** (`include/cudro/codegen_cuda.hpp`, `src/codegen_cuda.cpp`)
15. **[Chapter 15: Compiler CLI Driver & Tools](15_main.md)** (`src/main.cpp`)
16. **[Chapter 16: Constrained Motion Planner Integration](16_planner.md)** (`include/cudro/planner.hpp`, `src/planner.cpp`)

---

## ⚡ Compiler Optimizations & Evolutionary Case Studies

The codebook documents both the **naive baseline practices** and the **production-grade optimized compiler passes**, including the architectural rationale, before/after code snippets, and benchmarked improvements:

| Chapter | Optimization Topic | Old Baseline Practice | Updated Optimized Practice | Observed Benchmark Change |
|---|---|---|---|---|
| **[Chapter 08](08_dag.md)** | Constant Deduplication | Unconditional node allocation for identical constants | Whole-program hash-consing with `-0.0` normalization | **79.5% DAG node reduction** (122 $\to$ 25 nodes) |
| **[Chapter 09](09_lower.md)** | Branching Kinematic Sorting | Parent-existence check failed on child-first definitions | Topological iterative resolver (`build_all_link_transforms`) | **Arbitrary link ordering & dual-arm branching trees supported** |
| **[Chapter 10](10_ad.md)** | Multi-Constraint Matrix AD | Single-scalar $M=1$ or finite-difference perturbations | Exact $M \times N$ symbolic matrix AD + Rodrigues skew derivatives | **Simultaneous multi-plane manifolds**; 0 truncation error |
| **[Chapter 11](11_codegen_c.md)** | Inlined Normal Equations & Status Codes | $M=1$ rank-1 closed form only; silent failure on divergence or iteration cap | Inlined Gauss elimination with signed clamp and explicit status return codes (`SUCCESS`, `MAX_ITERS`, `NUMERICAL_ERROR`) | **1 kHz control step benchmark (< 4 $\mu$s median)**; 100% finite outputs; explicit convergence detection |
| **[Chapter 12](12_jit_tcc.md)** | In-Memory JIT | Dynamic string allocations & bloated C sources | Static lookup string cache + canonical constants | **JIT compile latency slashed to 5.1–9.5 ms** (~2.5× faster) |
| **[Chapter 16](16_planner.md)** | Planner Scratchpads | Dynamic `std::vector` allocations in RRT inner loop | Pre-allocated `g_scratch_` and hoisted candidate buffers | **0 heap allocations**; test runtime dropped 0.50s $\to$ 0.05s |


