#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cudro {

struct Location {
    std::string_view file;
    int line = 1;
    int col = 1;
};

enum class Severity {
    Error,
    Warning,
};

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

void print_diagnostics(const DiagnosticBag& diags, std::string_view source);

}
