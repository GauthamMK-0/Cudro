#include <cudro/jit_tcc.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef CUDRO_HAS_TCC
#include <libtcc.h>
#else
// Mock TCC types when libtcc is not available
typedef struct TCCState TCCState;
#endif

namespace cudro {

JITModule TCCJIT::compile(const std::string& c_source) {
    JITModule mod;
    
#ifdef CUDRO_HAS_TCC
    TCCState* state = tcc_new();
    if (!state) {
        fprintf(stderr, "Failed to create TCC state\n");
        return mod;
    }
    
    tcc_set_output_type(state, TCC_OUTPUT_MEMORY);
    
    // Add standard includes
    tcc_add_include_path(state, "/usr/include");
    tcc_add_include_path(state, "/usr/local/include");
    tcc_add_include_path(state, "/usr/include/x86_64-linux-gnu");
    tcc_add_include_path(state, "/usr/lib/gcc/x86_64-linux-gnu/13/include");
    
    // Compile the source
    if (tcc_compile_string(state, c_source.c_str()) != 0) {
        fprintf(stderr, "TCC compilation failed\n");
        tcc_delete(state);
        return mod;
    }
    
    // Relocate and get symbol
    if (tcc_relocate(state, TCC_RELOCATE_AUTO) != 0) {
        fprintf(stderr, "TCC relocation failed\n");
        tcc_delete(state);
        return mod;
    }
    
    mod.handle = state;
    return mod;
#else
    (void)c_source;
    fprintf(stderr, "TCC not available - compile with CUDRO_HAS_TCC\n");
    return mod;
#endif
}

JITModule::~JITModule() {
#ifdef CUDRO_HAS_TCC
    if (handle) {
        tcc_delete(static_cast<TCCState*>(handle));
        handle = nullptr;
    }
#endif
}

template<typename Fn>
Fn JITModule::get_symbol(const char* name) {
#ifdef CUDRO_HAS_TCC
    if (handle) {
        void* ptr = tcc_get_symbol(static_cast<TCCState*>(handle), name);
        if (ptr) {
            return reinterpret_cast<Fn>(ptr);
        }
    }
#endif
    (void)name;
    return nullptr;
}

// Explicit instantiations
template void(*JITModule::get_symbol<void(*)(const float*, int, float*)>(const char*))(const float*, int, float*);
template int(*JITModule::get_symbol<int(*)(const float*, int, float*)>(const char*))(const float*, int, float*);
template void(*JITModule::get_symbol<void(*)(const float*, float*, float*)>(const char*))(const float*, float*, float*);
template void(*JITModule::get_symbol<void(*)(const float*, int, int, float*)>(const char*))(const float*, int, int, float*);

} // namespace cudro