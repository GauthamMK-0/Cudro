#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>

#include <random>
#include <string>

TEST_CASE("Fuzz-lite: parser & sema must not crash on random ASCII sequences") {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> len_dist(10, 200);
    std::uniform_int_distribution<int> char_dist(32, 126); // printable ASCII

    for (int iter = 0; iter < 100; ++iter) {
        int len = len_dist(rng);
        std::string junk;
        junk.reserve(len);
        for (int i = 0; i < len; ++i) {
            junk.push_back(static_cast<char>(char_dist(rng)));
        }

        cudro::DiagnosticBag diags;
        cudro::Lexer lexer("<fuzz>", junk, diags);
        auto tokens = lexer.tokenize();

        cudro::Parser parser(tokens, diags);
        auto spec = parser.parse();
        if (!diags.has_errors()) {
            cudro::Sema sema(spec, diags);
            sema.analyze();
        }
        // Verification: test must complete cleanly without crash / ASan error
    }
}
