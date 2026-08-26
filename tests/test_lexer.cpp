#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/lexer.hpp>

#include <string>
#include <vector>

namespace {

struct LexResult {
    std::vector<cudro::Token> tokens;
    cudro::DiagnosticBag diags;
};

LexResult lex(const std::string& source) {
    static const std::string file = "<test>";
    cudro::DiagnosticBag bag;
    cudro::Lexer lexer(file, source, bag);
    LexResult r;
    r.tokens = lexer.tokenize();
    r.diags = std::move(bag);
    return r;
}

std::vector<cudro::TokenKind> kinds(const LexResult& r) {
    std::vector<cudro::TokenKind> out;
    for (const auto& t : r.tokens)
        if (t.kind != cudro::TokenKind::EndOfFile)
            out.push_back(t.kind);
    return out;
}

}

TEST_CASE("keywords, identifiers, punctuation") {
    std::string src = "robot panda7 { }";
    auto r = lex(src);
    auto k = kinds(r);
    REQUIRE(k == std::vector<cudro::TokenKind>{
                     cudro::TokenKind::KwRobot, cudro::TokenKind::Identifier,
                     cudro::TokenKind::LBrace, cudro::TokenKind::RBrace});
    CHECK(r.tokens[1].text == "panda7");
    CHECK_FALSE(r.diags.has_errors());
}

TEST_CASE("keyword prefix stays an identifier") {
    std::string src = "plane planeX";
    auto r = lex(src);
    CHECK(r.tokens[0].kind == cudro::TokenKind::KwPlane);
    CHECK(r.tokens[1].kind == cudro::TokenKind::Identifier);
    CHECK(r.tokens[1].text == "planeX");
}

TEST_CASE("numbers with sign and decimals") {
    std::string src = "[ -2.8973, 0.333 ]";
    auto r = lex(src);
    auto k = kinds(r);
    REQUIRE(k.size() == 5);
    CHECK(k[0] == cudro::TokenKind::LBracket);
    CHECK(k[1] == cudro::TokenKind::Number);
    CHECK(r.tokens[1].text == "-2.8973");
    CHECK(k[2] == cudro::TokenKind::Comma);
    CHECK(k[3] == cudro::TokenKind::Number);
    CHECK(r.tokens[3].text == "0.333");
    CHECK(k[4] == cudro::TokenKind::RBracket);
}

TEST_CASE("comments skipped and locations track lines") {
    std::string src = "// header comment\nrobot {\n}\n";
    auto r = lex(src);
    REQUIRE(r.tokens.size() == 4);
    CHECK(r.tokens[0].kind == cudro::TokenKind::KwRobot);
    CHECK(r.tokens[0].where.line == 2);
    CHECK(r.tokens[0].where.col == 1);
    CHECK(r.tokens[2].where.line == 3);
}

TEST_CASE("unexpected character reports and recovers") {
    std::string src = "@ robot";
    auto r = lex(src);
    REQUIRE(r.diags.has_errors());
    CHECK(r.diags.all().size() == 1);
    CHECK(r.diags.all()[0].where.col == 1);
    bool saw_robot = false;
    for (const auto& t : r.tokens)
        saw_robot |= (t.kind == cudro::TokenKind::KwRobot);
    CHECK(saw_robot);
}
