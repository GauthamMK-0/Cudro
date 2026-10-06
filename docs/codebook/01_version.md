# Chapter 01: Core Versioning & Metadata

## Files Covered
- `include/cudro/version.hpp`

---

## 1. Architectural Purpose
In any production compiler (such as GCC, Clang, or Rustc), tracking compiler revisions, semantic versions, and stage tags is necessary for reproducible builds, telemetry, and diagnosing bug reports from users. In a runtime compiler, versioning also prevents loading cached binary artifacts compiled by older revisions of the compiler.

`version.hpp` provides compile-time constants exposed in the root `cudro` namespace.

---

## 2. Complete Header Code

```cpp
#pragma once

namespace cudro {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 0;
inline constexpr const char* version_string = "cudro 0.0 (stage M0: scaffold)";

}
```

---

## 3. Detailed Component Breakdown

### A. `#pragma once` Header Guards
```cpp
#pragma once
```
- **What it does:** Tells the preprocessor to include this file only once per translation unit (.cpp file).
- **Why it matters in compilers:** Compilers have large dependency graphs. Without guards, cyclic or repeated includes cause duplicate definition errors (`redefinition of '...'`).

### B. Inline Constexpr Constants
```cpp
inline constexpr int version_major = 0;
inline constexpr int version_minor = 0;
inline constexpr const char* version_string = "cudro 0.0 (stage M0: scaffold)";
```
- **`constexpr`:** Evaluated strictly at compile time. Zero runtime initialization overhead.
- **`inline` (C++17):** Allows the global variable to be defined in a header without violating the One Definition Rule (ODR) when included across multiple `.cpp` files.
- **`version_string`:** Human-readable string rendered by `cudro --version` or banner outputs.

---

## 4. Interaction with Other Compiler Stages
- **CLI Driver (`src/main.cpp`):** Used when the user calls `./build/cudro` without arguments or with `--version`.
- **Diagnostics:** Can stamp generated C or CUDA kernel files with the exact compiler version used to generate them for auditing and reproducibility.
