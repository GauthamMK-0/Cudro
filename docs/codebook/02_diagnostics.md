# Chapter 02: Source Locations & Diagnostics

## Files Covered
- `include/cudro/diagnostic.hpp`
- `src/diagnostic.cpp`

---

## 1. Architectural Purpose
A compiler is only as good as its error messages. If a user writes an invalid robot specification (e.g., `origin [0, 0]` missing the Z-coordinate), the compiler must pinpoint the exact file, line number, column, and print a caret (`^`) pointing to the offending character.

The diagnostics module provides:
1. **Source Location Tracking (`Location`):** Tracking `(file, line, col)` coordinates.
2. **Diagnostic Container (`DiagnosticBag`):** Collecting multiple errors rather than aborting at the first failure.
3. **Caret Rendering Engine (`print_diagnostics`):** Formatting visual terminal feedback with exact line slices.

---

## 2. Detailed Code Breakdown

### A. Location Data Structure (`include/cudro/diagnostic.hpp`)

```cpp
struct Location {
    std::string_view file;
    int line = 1;
    int col = 1;
};
```
- **`std::string_view file`:** A lightweight non-owning view of the filename string.
- **`line` and `col`:** 1-indexed coordinates tracking the cursor in the source stream.

### B. Diagnostic Bag (`include/cudro/diagnostic.hpp`)

```cpp
enum class Severity { Error, Warning };

struct Diagnostic {
    Severity severity;
    Location where;
    std::string message;
};

class DiagnosticBag {
public:
    void error(Location where, std::string message);
    void warning(Location where, std::string message);
    bool has_errors() const { return !entries_.empty(); }
    const std::vector<Diagnostic>& all() const { return entries_; }

private:
    std::vector<Diagnostic> entries_;
};
```
- **Error Accumulation:** Instead of calling `exit(1)` on the first syntax error, the compiler records the diagnostic in `entries_` and recovers. This allows reporting all syntax errors in a `.cudro` file at once.

### C. Visual Caret Rendering (`src/diagnostic.cpp`)

```cpp
std::string_view find_line(std::string_view source, int line) {
    int current = 1;
    size_t start = 0;
    for (size_t i = 0; i <= source.size(); ++i) {
        if (i == source.size() || source[i] == '\n') {
            if (current == line)
                return source.substr(start, i - start);
            ++current;
            start = i + 1;
        }
    }
    return {};
}

void print_diagnostics(const DiagnosticBag& diags, std::string_view source) {
    for (const auto& d : diags.all()) {
        std::printf("%.*s:%d:%d: %s: %.*s\n",
                    static_cast<int>(d.where.file.size()), d.where.file.data(),
                    d.where.line, d.where.col, severity_name(d.severity),
                    static_cast<int>(d.message.size()), d.message.data());
        std::string_view line = find_line(source, d.where.line);
        std::printf("%.*s\n", static_cast<int>(line.size()), line.data());
        for (int i = 1; i < d.where.col; ++i)
            std::fputc(' ', stdout);
        std::printf("^\n");
    }
}
```

#### How Caret Printing Works:
1. `find_line` scans the original source text counting `\n` characters until reaching target `d.where.line`.
2. It prints the standard GCC/Clang style header: `file.cudro:14:5: error: unexpected character '@'`.
3. It prints the full source code line: `  @ robot panda7 {`.
4. It loops `d.where.col - 1` times emitting spaces `' '` followed by `^`, directly aligning under the offending token.

---

## 3. Robotics & Compiler Context
When a roboticist is authoring constraints or when a perception system (like a VLA model) is programmatically generating `.cudro` files, syntax errors must be clearly pinpointed so downstream motion planners do not crash silently.
