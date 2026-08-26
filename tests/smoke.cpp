#include <cudro/version.hpp>

#include <cassert>
#include <cstdio>

int main() {
    assert(cudro::version_major >= 0);

#ifdef __x86_64__
    assert(__builtin_cpu_supports("avx2"));
    std::printf("avx2: supported\n");
#else
    std::printf("avx2: unknown arch\n");
#endif

    std::printf("smoke: ok\n");
    return 0;
}
