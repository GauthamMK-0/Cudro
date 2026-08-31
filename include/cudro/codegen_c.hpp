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
};

std::string generate_scalar_c(const LowerResult& lower_result, const CodegenOptions& opts = {});

} // namespace cudro