#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/ad.hpp>
#include <cudro/dag.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>

#include <cmath>
#include <vector>

namespace {

cudro::Spec parse_and_check(const std::string& src) {
    cudro::DiagnosticBag diags;
    cudro::Lexer lexer("<test>", src, diags);
    auto tokens = lexer.tokenize();
    REQUIRE_FALSE(diags.has_errors());
    cudro::Parser parser(tokens, diags);
    auto spec = parser.parse();
    REQUIRE_FALSE(diags.has_errors());
    cudro::Sema sema(spec, diags);
    REQUIRE(sema.analyze());
    return spec;
}

} // namespace

TEST_CASE("AD basic scalar arithmetic chain") {
    // f(q0, q1) = (q0 * q1) + sin(q0)
    cudro::ExprDAG dag;
    int q0 = dag.add_input(0, "q0");
    int q1 = dag.add_input(1, "q1");
    int mul = dag.add_binary(cudro::NodeKind::Mul, q0, q1);
    int sin_q0 = dag.add_unary(cudro::NodeKind::Sin, q0);
    int out = dag.add_binary(cudro::NodeKind::Add, mul, sin_q0);

    // Differentiate w.r.t inputs
    auto ad = cudro::differentiate(dag, {out}, 2);
    REQUIRE(ad.num_constraints == 1);
    REQUIRE(ad.num_inputs == 2);

    // Check at q = [0, 0]:
    // df/dq0 = q1 + cos(q0) = 0 + 1 = 1
    // df/dq1 = q0 = 0
    CHECK(std::abs(ad.jacobians[0][0] - 1.0) < 1e-4);
    CHECK(std::abs(ad.jacobians[0][1] - 0.0) < 1e-4);
}

TEST_CASE("AD on planar2r robot kinematics and plane constraint") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; limits [-3.14, 3.14]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; limits [-3.14, 3.14]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task keep_ee_above {
  link ee;
  plane { point_on_link [0,0,0]; normal [0,0,1]; offset 0.1; }
}
clearance { min_distance 0.02; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    auto ad = cudro::differentiate(dag, constraint_outputs, dag.num_inputs());
    REQUIRE(ad.num_constraints == 1);
    REQUIRE(ad.num_inputs == 2);
    // Plane constraint normal is z [0, 0, 1], planar robot rotates around z axis -> z derivative is 0
    CHECK(std::abs(ad.jacobians[0][0]) < 1e-4);
    CHECK(std::abs(ad.jacobians[0][1]) < 1e-4);
}
