#pragma once

#include <cudro/dag.hpp>
#include <cudro/lower.hpp>

#include <vector>
#include <string>
#include <string_view>

namespace cudro {

struct CodegenOptions {
    bool batched = false;
    bool avx2 = false;
    bool debug_comments = false;
};

// Generate scalar C kernel: int project(const float* q_in, int num_inputs, float* q_out)
// Returns: 0 = SUCCESS (converged), 1 = MAX_ITERS exceeded, 2 = NUMERICAL_ERROR
std::string generate_scalar_c(const LowerResult& lower_result, const CodegenOptions& opts = {});

// Generate batched C kernel: void project_batch(const float* q_batch, int batch_size, int num_inputs, float* q_out_batch)
std::string generate_batched_c(const LowerResult& lower_result, const CodegenOptions& opts = {});

// Dynamic CPU feature detection helper
bool cpu_supports_avx2();

} // namespace cudro