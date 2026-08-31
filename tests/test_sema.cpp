#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/sema.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>

namespace {

cudro::DiagnosticBag analyze_spec_string(const std::string& src) {
    cudro::DiagnosticBag diags;
    cudro::Lexer lexer("<test>", src, diags);
    auto tokens = lexer.tokenize();
    cudro::Parser parser(tokens, diags);
    auto spec = parser.parse();
    if (!diags.has_errors()) {
        cudro::Sema sema(spec, diags);
        sema.analyze();
    }
    return diags;
}

} // namespace

TEST_CASE("Sema passes valid robotic specs") {
    std::string src = R"(
robot arm {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0.1]; limits [-3.14, 3.14]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link ee { spheres [[0,0,0.2, 0.03]]; parent base; }
}
task keep_level {
  link ee;
  plane { point_on_link [0,0,0]; normal [0,0,1]; offset 0.5; }
}
clearance { min_distance 0.01; }
)";

    auto diags = analyze_spec_string(src);
    CHECK_FALSE(diags.has_errors());
}

TEST_CASE("Sema rejects duplicate robot/joint/link/task names") {
    std::string src = R"(
robot arm {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  joint j1 { type revolute; axis [0,1,0]; origin [0,0,0]; }
  link base { parent world; joint_ref j1; }
}
)";

    auto diags = analyze_spec_string(src);
    CHECK(diags.has_errors());
}

TEST_CASE("Sema rejects unknown parent link reference") {
    std::string src = R"(
robot arm {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  link base { parent nonexistent_link; joint_ref j1; }
}
)";

    auto diags = analyze_spec_string(src);
    CHECK(diags.has_errors());
}

TEST_CASE("Sema rejects unknown joint_ref") {
    std::string src = R"(
robot arm {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  link base { parent world; joint_ref nonexistent_joint; }
}
)";

    auto diags = analyze_spec_string(src);
    CHECK(diags.has_errors());
}

TEST_CASE("Sema rejects cyclic kinematic chains") {
    std::string src = R"(
robot loop {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  joint j2 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  link link1 { parent link2; joint_ref j1; }
  link link2 { parent link1; joint_ref j2; }
}
)";

    auto diags = analyze_spec_string(src);
    CHECK(diags.has_errors());
}
