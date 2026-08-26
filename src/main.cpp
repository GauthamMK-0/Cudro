#include <cudro/diagnostic.hpp>
#include <cudro/lexer.hpp>
#include <cudro/version.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--dump-tokens") {
        if (argc != 3) {
            std::fprintf(stderr, "usage: cudro --dump-tokens <file>\n");
            return 2;
        }
        std::ifstream in(argv[2]);
        if (!in) {
            std::fprintf(stderr, "error: cannot open %s\n", argv[2]);
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::string src = buf.str();

        cudro::DiagnosticBag diags;
        cudro::Lexer lexer(argv[2], src, diags);
        for (const auto& t : lexer.tokenize()) {
            std::printf("%3d:%-3d %-16s %.*s\n", t.where.line, t.where.col,
                        cudro::token_kind_name(t.kind),
                        static_cast<int>(t.text.size()), t.text.data());
        }
        print_diagnostics(diags, src);
        return diags.has_errors() ? 1 : 0;
    }

    std::printf("%s\n", cudro::version_string);
    return 0;
}
