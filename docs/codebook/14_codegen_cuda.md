# Chapter 14: GPU Massive Parallelism via CUDA NVRTC

## Files Covered
- `include/cudro/codegen_cuda.hpp`
- `src/codegen_cuda.cpp`

---

## 1. Architectural Purpose
This is the **M8 stretch milestone** — a second backend from the **same Expression DAG** targeting **NVIDIA GPUs** via **NVRTC (Runtime CUDA Compilation)**.

### Why CUDA?
| Factor | CPU (AVX2) | GPU (CUDA) |
|---|---|---|
| Parallelism | 8-wide (AVX2) | **Thousands of threads** |
| Batch Throughput | ~8 configs/cycle | **512–4096 configs/cycle** |
| Latency | ~10 μs/batch | ~100 μs/kernel launch |
| Use Case | Interactive, low-latency | **Massive rollout scoring** (world models, MPC) |

### Why NVRTC (not offline NVCC)?
- **Runtime compilation**: Spec arrives → CUDA C emitted → NVRTC compiles → kernel runs
- Same JIT workflow as libtcc/LLVM, just targeting PTX instead of x86
- `nvcc` already installed on the robot (Jetson, desktop GPU)

---

## 2. CUDA Interface (`include/cudro/codegen_cuda.hpp`)

```cpp
struct CUDACodegenOptions {
    int block_size = 256;       // Threads per block
    bool use_fast_math = true;  // --use_fast_math for NVRTC
};

class CUDACodegen {
public:
    CUDACodegen(const ADResult& ad, int num_inputs, int num_constraints,
                const CUDACodegenOptions& opts = {});

    // Emit CUDA C source
    std::string emit_cuda() const;

    // JIT compile via NVRTC and return kernel launcher
    struct KernelHandle {
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        int num_inputs;
        int num_constraints;
        
        // Launch kernel
        void launch(const float* q_batch, int batch_size, 
                    float* g_out, float* J_out, CUstream stream = 0);
        
        ~KernelHandle(); // Cleanup module
    };
    
    KernelHandle jit_compile() const;
};
```

---

## 2. CUDA Kernel Structure (`src/codegen_cuda.cpp`)

### A. Kernel Signature
```cuda
__global__ void project_kernel(
    const float* __restrict__ q_batch,   // [batch * num_inputs]
    int batch_size,
    float* __restrict__ g_out_batch,     // [batch * num_constraints]
    float* __restrict__ J_out_batch      // [batch * num_constraints * num_inputs]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread processes ONE configuration
    for (int idx = tid; idx < batch_size; idx += stride) {
        // Load this config's q
        float q[28]; // max DoFs
        for (int d = 0; d < num_inputs; ++d) {
            q[d] = q_batch[idx * num_inputs + d];
        }
        
        // Evaluate DAG (same arithmetic as CPU, but scalar per-thread)
        float tmp[256]; // DAG node values
        // ... unrolled arithmetic ...
        
        // Write constraint outputs
        for (int c = 0; c < num_constraints; ++c) {
            g_out_batch[idx * num_constraints + c] = tmp[constraint_outputs[c]];
        }
        // Write Jacobian
        for (int c = 0; c < num_constraints; ++c) {
            for (int d = 0; d < num_inputs; ++d) {
                J_out_batch[idx * num_constraints * num_inputs + c * num_inputs + d] 
                    = tmp[jacobians[c][d]];
            }
        }
    }
}
```

### B. One Thread = One Configuration
Unlike CPU AVX2 where 8 configs share a thread via SIMD, **each GPU thread handles one full configuration**. This maps perfectly to the DAG:
- No lane divergence within a thread
- Natural mapping: DAG evaluation is sequential per config
- Massive parallelism: 10,000 configs → 10,000 threads

### C. NVRTC Compilation
```cpp
CUDACodegen::KernelHandle CUDACodegen::jit_compile() const {
    std::string cuda_source = emit_cuda();
    
    // NVRTC program
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, cuda_source.c_str(), "cudro_kernel.cu", 0, nullptr, nullptr);
    
    // Compile options
    const char* opts[] = {
        "--gpu-architecture=compute_70",  // Volta+ (adjust for GPU)
        "--use_fast_math",
        "-I/usr/include"
    };
    nvrtcCompileProgram(prog, 3, opts);
    
    // Get PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::string ptx(ptx_size, '\0');
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    
    // Load PTX into CUDA driver API
    CUmodule module;
    cuModuleLoadData(&module, ptx.c_str());
    
    CUfunction function;
    cuModuleGetFunction(&function, module, "project_kernel");
    
    return {module, function, num_inputs_, num_constraints_};
}
```

### D. Kernel Launch
```cpp
void KernelHandle::launch(const float* q_batch, int batch_size,
                          float* g_out, float* J_out, CUstream stream) {
    int block = 256;
    int grid = (batch_size + block - 1) / block;
    
    void* args[] = {&q_batch, &batch_size, &g_out, &J_out};
    cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, stream, args, nullptr);
    cuStreamSynchronize(stream);
}
```

---

## 3. Same DAG, Different Backend
The **exact same ADResult** (DAG + Jacobians) feeds:
1. `codegen_c.cpp` → CPU scalar / AVX2
2. `codegen_llvm.cpp` → LLVM IR / ORC JIT
3. `codegen_cuda.cpp` → CUDA C / NVRTC

**This is the portable IR proof** (R1): one IR, three backends, zero IR changes.

---

## 4. Robotics Context: World Models & MPC
GPU kernels shine in **world-model-based planning**:
- Sample 10,000 candidate trajectories
- Roll out in learned world model (latent space)
- Score each with constraint projection kernel
- **10,000 configs in one kernel launch** = massive throughput

This is the workload cuRobo targets — our kernel plugs into that pipeline.

---

## 4. Testing
- Emit CUDA source → compile with NVRTC → verify numerics match CPU
- Benchmark: 10,000 configs, report throughput (configs/second)
- Validate against CPU batched kernel (≤1e-5 rel., adjusted for fast-math)