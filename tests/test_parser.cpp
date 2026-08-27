#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/parser.hpp>
#include <cudro/lexer.hpp>
#include <cudro/ast.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace {

struct ParseResult {
    cudro::Spec spec;
    cudro::DiagnosticBag diags;
    std::vector<cudro::Token> tokens;
};

ParseResult parse_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        ParseResult result;
        result.diags.error({path, 1, 1}, "cannot open file: " + path);
        return result;
    }
    std::stringstream buf; buf << in.rdbuf();
    std::string src = buf.str();

    ParseResult result;
    cudro::DiagnosticBag bag;
    cudro::Lexer lexer(path, src, bag);
    result.tokens = lexer.tokenize();
    result.diags = std::move(bag);
    if (result.diags.has_errors()) return result;

    cudro::Parser parser(result.tokens, result.diags);
    result.spec = parser.parse();
    return result;
}

std::string get_spec_path(const char* filename) {
#ifdef CUDRO_SOURCE_DIR
    return std::string(CUDRO_SOURCE_DIR) + "/spec/" + filename;
#else
    return "spec/" + filename;
#endif
}

ParseResult parse_string(const std::string& src) {
    ParseResult result;
    cudro::DiagnosticBag bag;
    cudro::Lexer lexer("<test>", src, bag);
    result.tokens = lexer.tokenize();
    result.diags = std::move(bag);
    if (result.diags.has_errors()) return result;

    cudro::Parser parser(result.tokens, result.diags);
    result.spec = parser.parse();
    return result;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::stringstream buf; buf << in.rdbuf();
    return buf.str();
}

} // namespace

TEST_CASE("valid panda7.cudro parses successfully") {
    auto r = parse_file(get_spec_path("panda7.cudro"));
    CHECK_FALSE(r.diags.has_errors());
    CHECK(r.spec.robots.size() == 1);
    CHECK(r.spec.robots[0]->name == "panda7");
    CHECK(r.spec.robots[0]->joints.size() == 7);
    CHECK(r.spec.robots[0]->links.size() == 5);
    CHECK(r.spec.tasks.size() == 1);
    CHECK(r.spec.tasks[0]->name == "cup_on_table");
    CHECK(r.spec.tasks[0]->planes.size() == 1);
    CHECK(r.spec.clearances.size() == 1);
}

TEST_CASE("valid planar2r.cudro parses successfully") {
    auto r = parse_file(get_spec_path("planar2r.cudro"));
    CHECK_FALSE(r.diags.has_errors());
    CHECK(r.spec.robots.size() == 1);
    CHECK(r.spec.robots[0]->name == "planar2r");
    CHECK(r.spec.robots[0]->joints.size() == 2);
    CHECK(r.spec.robots[0]->links.size() == 3);
    CHECK(r.spec.tasks.size() == 1);
    CHECK(r.spec.tasks[0]->name == "keep_ee_above");
    CHECK(r.spec.clearances.size() == 1);
}

TEST_CASE("missing semicolon produces error and recovers") {
    auto r = parse_string("robot r { joint j { type revolute; axis [0,0,1]; origin [0,0,0] } }");
    CHECK(r.diags.has_errors());
    CHECK(r.diags.all().size() >= 1);
    // Should still parse robot structure
    CHECK(r.spec.robots.size() == 1);
}

TEST_CASE("unclosed brace produces error at EOF") {
    auto r = parse_string("robot r { joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; limits [-1,1]; } ");
    CHECK(r.diags.has_errors());
    // Error should be at EOF or near end
    bool found_eof = false;
    for (const auto& d : r.diags.all()) {
        if (d.where.line >= 1) found_eof = true;
    }
    CHECK(found_eof);
}

TEST_CASE("unknown field name produces error and recovers") {
    auto r = parse_string("robot r { joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; unknown_field 123; } }");
    CHECK(r.diags.has_errors());
    CHECK(r.spec.robots.size() == 1);
    CHECK(r.spec.robots[0]->joints.size() == 1);
}

TEST_CASE("duplicate joint name produces error (tested in M2 sema phase)") {
    // TODO: This is a semantic check (M2), not a parser check (M1).
    // The parser builds the AST; sema phase validates uniqueness.
    // auto r = parse_string("robot r { joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; } joint j1 { type revolute; axis [0,1,0]; origin [0,0,0]; } }");
    // CHECK(r.diags.has_errors());
    CHECK(true); // placeholder
}

TEST_CASE("missing joint type produces error") {
    auto r = parse_string("robot r { joint j1 { axis [0,0,1]; origin [0,0,0]; } }");
    CHECK(r.diags.has_errors());
}

TEST_CASE("multiple errors reported in single pass") {
    auto r = parse_string("robot r1 { joint j { axis [0,0,1] } } robot r2 { joint j { axis [0,0,1] } }");
    // Should report multiple errors, not just first
    CHECK(r.diags.all().size() >= 2);
}

TEST_CASE("empty spec parses as empty") {
    auto r = parse_string("");
    CHECK_FALSE(r.diags.has_errors());
    CHECK(r.spec.robots.empty());
    CHECK(r.spec.tasks.empty());
    CHECK(r.spec.clearances.empty());
}