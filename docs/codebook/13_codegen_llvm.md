# Chapter 13: Industrial LLVM IR & ORC JIT Backend

## Files Covered
- `include/cudro/codegen_llvm.hpp`
- `src/codegen_llvm.cpp`

---

## 1. Architectural Purpose
This is the **Tier-1 industrial backend** (M6.5 milestone). It proves our portable IR claim by lowering the **same Expression DAG** to **LLVM IR** and JIT-compiling via **ORC (On-Request Compilation)**.

### Why LLVM?
| Aspect | libtcc (C) | LLVM IR |
|---|---|---|
| Code Quality | ~-O1 | **-O3** (full optimization) |
| Compile Latency | **~5–20 ms** | ~50–200 ms |
| Register Alloc | None (C compiler) | **Full greedy/SSA** |
| Vectorization | Manual AVX2 | **Auto-vectorizer + loop vectorizer** |
| Targets | C compiler only | **x86, ARM, RISC-V, GPU (NVPTX)** |
| Dependency | 600 KB | ~300 MB |

### The Trade-off
We measure **both** backends on the **same DAG**:
- libtcc: wins on compile latency (our headline metric)
- LLVM: wins on steady-state throughput (traditional metric)

This **latency vs. quality tension** IS the research content of runtime compilation.

---

## 2. LLVM Interface (`include/cudro/codegen_llvm.hpp`)

```cpp
struct LLVMCodegenOptions {
    bool optimize = true;       // -O3
    bool avx2 = true;           // Target features
};

class LLVMCodegen {
public:
    LLVMCodegen(const ADResult& ad, int num_inputs, int num_constraints,
                const LLVMCodegenOptions& opts = {});

    // Emit LLVM IR to string (for debugging/dumping)
    std::string emit_ir() const;

    // JIT compile and return function pointer
    template<typename Fn>
    Fn jit_compile() const;

private:
    const ADResult& ad_;
    int num_inputs_, num_constraints_;
    LLVMCodegenOptions opts_;
    std::unique_ptr<llvm::LLVMContext> ctx_;
    std::unique_ptr<llvm::Module> module_;
    std::unique_ptr<llvm::IRBuilder<>> builder_;
    llvm::Function* kernel_fn_ = nullptr;
};
```

---

## 3. IR Construction (`src/codegen_llvm.cpp`)

### A. Module Setup
```cpp
LLVMCodegen::LLVMCodegen(...) {
    ctx_ = std::make_unique<llvm::LLVMContext>();
    module_ = std::make_unique<llvm::Module>("cudro_kernel", *ctx_);
    builder_ = std::make_unique<llvm::IRBuilder<>>(*ctx_);
    
    // Target machine for optimization
    auto target = llvm::TargetRegistry::lookupTarget("x86_64", ...);
    target_machine_ = target->createTargetMachine(...);
    
    // Function signature: void project(float* q, int n, float* g, float* J)
    // Batched version: void project_batch(float* q, int n, int batch, float* g, float* J)
    llvm::FunctionType* fn_type = ...;
    kernel_fn_ = llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage,
                                         "project", module_.get());
    
    // Entry block
    auto* entry = llvm::BasicBlock::Create(*ctx_, "entry", kernel_fn_);
    builder_->SetInsertPoint(entry);
    
    // Function arguments
    llvm::Value* q_ptr = kernel_fn_->arg_begin();
    // ... g_ptr, J_ptr
}
```

### B. DAG → LLVM IR Lowering
```cpp
llvm::Value* LLVMCodegen::lower_node(int node_idx) {
    const Node& n = dag_[node_idx];
    if (cached_values_.count(node_idx)) return cached_values_[node_idx];
    
    llvm::Value* result = nullptr;
    switch (n.kind) {
        case NodeKind::Input: {
            // q[input_index] from function argument
            llvm::Value* idx = llvm::ConstantInt::get(int32_t, n.input_index);
            llvm::Value* ptr = builder_->CreateGEP(builder_->getFloatTy(), q_ptr, idx);
            result = builder_->CreateLoad(builder_->getFloatTy(), ptr);
            break;
        }
        case NodeKind::Constant: {
            result = llvm::ConstantFP::get(*ctx_, llvm::APFloat(n.constant_value));
            break;
        }
        case NodeKind::Add: {
            auto lhs = lower_node(n.operands[0]);
            auto rhs = lower_node(n.operands[1]);
            result = builder_->CreateFAdd(lhs, rhs);
            break;
        }
        case NodeKind::Mul: {
            auto lhs = lower_node(n.operands[0]);
            auto rhs = lower_node(n.operands[1]);
            result = builder_->CreateFMul(lhs, rhs);
            break;
        }
        case NodeKind::MatMul: {
            // Call helper or inline 4x4 multiply
            result = lower_matmul(n.operands[0], n.operands[1]);
            break;
        }
        // ... Rot, Translate, Dot, Norm, SphereDist ...
    }
    
    cached_values_[node_idx] = result;
    return result;
}
```

### C. 4×4 Matrix Multiply in LLVM IR
```cpp
llvm::Value* LLVMCodegen::lower_matmul(int lhs, int rhs) {
    // Allocate 16-element result array on stack
    auto* result_arr = builder_->CreateAlloca(builder_->getFloatTy(), 
                                               llvm::ConstantInt::get(16));
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            llvm::Value* sum = llvm::ConstantFP::get(*ctx_, 0.0);
            for (int k = 0; k < 4; ++k) {
                auto* a = builder_->CreateLoad(builder_->getFloatTy(),
                    builder_->CreateGEP(builder_->getFloatTy(), lhs_val,
                        llvm::ConstantInt::get(i*4 + k)));
                auto* b = builder_->CreateLoad(builder_->getFloatTy(),
                    builder_->CreateGEP(builder_->getFloatTy(), rhs_val,
                        llvm::ConstantInt::get(k*4 + j)));
                auto* prod = builder_->CreateFMul(a, b);
                sum = builder_->CreateFAdd(sum, prod);
            }
            builder_->CreateStore(sum,
                builder_->CreateGEP(builder_->getFloatTy(), result_arr,
                    llvm::ConstantInt::get(i*4 + j)));
        }
    }
    return result_arr;
}
```

---

## 4. ORC JIT Execution
```cpp
template<typename Fn>
Fn LLVMCodegen::jit_compile() const {
    // Create ORC JIT
    auto jit = llvm::orc::LLJITBuilder().create();
    
    // Add module
    auto rt = jit->getMainJITDylib();
    auto tsm = llvm::orc::ThreadSafeModule(std::move(module_), std::move(ctx_));
    jit->addIRModule(rt, std::move(tsm));
    
    // Lookup symbol
    auto sym = jit->lookup("project");
    return reinterpret_cast<Fn>(sym.getAddress());
}
```

### ORC V2 Advantages
- **Lazy compilation**: Functions compiled on first call
- **Concurrent compilation**: Multiple threads can compile simultaneously
- **Symbol management**: Proper symbol resolution, no global state

---

## 5. Comparison Experiment (M6.5 Milestone)
| Metric | libtcc (C) | LLVM ORC |
|---|---|---|
| Compile Latency | **~10 ms** | ~100 ms |
| Kernel Throughput | 1.0× (baseline) | **2–5× faster** |
| Binary Size | N/A (source) | Larger |
| Register Alloc | C compiler | **LLVM's greedy** |
| Vectorization | Manual AVX2 | **Auto-vectorizer** |

This comparison **is the research contribution**: proving that runtime compilation can approach AOT quality, and quantifying the latency cost.

---

## 6. Robotics Context
In a production robot system, you might:
- Use **libtcc** for interactive re-tasking (sub-second kernel updates)
- Use **LLVM** for the final deployed kernel (maximum throughput)

Our compiler produces **both from the same IR** — proving the portable IR claim (R1).