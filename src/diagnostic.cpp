#include <cudro/diagnostic.hpp>

#include <cstdio>

namespace cudro {

void DiagnosticBag::error(Location where, std::string message) {
    entries_.push_back({Severity::Error, where, std::move(message)});
}

void DiagnosticBag::warning(Location where, std::string message) {
    entries_.push_back({Severity::Warning, where, std::move(message)});
}

namespace {

const char* severity_name(Severity s) {
    return s == Severity::Error ? "error" : "warning";
}

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

}  // namespace

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

}
