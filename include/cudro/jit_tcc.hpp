#pragma once

#include <cudro/dag.hpp>

namespace cudro {

struct JITModule {
    void* handle = nullptr;
    
    ~JITModule();
    
    template<typename Fn>
    Fn get_symbol(const char* name);
};

class TCCJIT {
public:
    // Compile C source string and return JITModule with the kernel
    static JITModule compile(const std::string& c_source);
    
    // Convenience: compile and get function pointer in one call
    template<typename Fn>
    static Fn compile_and_get(const std::string& c_source, const char* symbol) {
        JITModule mod = compile(c_source);
        if (mod.handle) {
            return mod.get_symbol<Fn>(symbol);
        }
        return nullptr;
    }
};

} // namespace cudro