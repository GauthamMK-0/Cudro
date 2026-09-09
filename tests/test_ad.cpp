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

TEST_CASE("AD multi-constraint validation on planar2r (M=2, N=2)") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; limits [-3.14159, 3.14159]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; limits [-3.14159, 3.14159]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee   { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task dual_link_task {
  link ee;
  plane { link base; point_on_link [0.5,0,0]; normal [1,0,0]; offset 0.3; }
  plane { link ee;   point_on_link [0.0,0,0]; normal [0,1,0]; offset 0.5; }
}
clearance { min_distance 0.02; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 2);

    int num_inputs = dag.num_inputs();
    auto jacobian_nodes = cudro::build_analytical_jacobians(dag, constraint_outputs, num_inputs);
    REQUIRE(jacobian_nodes.size() == 2);
    REQUIRE(jacobian_nodes[0].size() == 2);
    REQUIRE(jacobian_nodes[1].size() == 2);

    // Test across several configuration points
    std::vector<std::vector<double>> test_configs = {
        {0.3, 0.4},
        {-0.5, 0.8},
        {1.2, -0.7}
    };

    const double h = 1e-5;
    for (const auto& q : test_configs) {
        auto vals = cudro::evaluate_dag(dag, num_inputs, q.data());

        for (int c = 0; c < 2; ++c) {
            for (int j = 0; j < num_inputs; ++j) {
                double ad_grad = vals[jacobian_nodes[c][j]];

                // Central finite difference
                std::vector<double> q_plus = q;
                std::vector<double> q_minus = q;
                q_plus[j] += h;
                q_minus[j] -= h;

                auto vals_plus = cudro::evaluate_dag(dag, num_inputs, q_plus.data());
                auto vals_minus = cudro::evaluate_dag(dag, num_inputs, q_minus.data());

                double g_plus = vals_plus[constraint_outputs[c]];
                double g_minus = vals_minus[constraint_outputs[c]];
                double fd_grad = (g_plus - g_minus) / (2.0 * h);

                CHECK(std::abs(ad_grad - fd_grad) < 1e-4);
            }
        }
    }
}

TEST_CASE("AD triple-constraint validation on panda7 (M=3, N=7)") {
    std::string src = R"(
robot panda7 {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0.333]; limits [-2.8973, 2.8973]; }
  joint j2 { type revolute; axis [0,1,0]; origin [0,0,0]; limits [-1.7628, 1.7628]; }
  joint j3 { type revolute; axis [0,0,1]; origin [0,-0.316,0]; limits [-2.8973, 2.8973]; }
  joint j4 { type revolute; axis [0,1,0]; origin [0,0,0.0825]; }
  joint j5 { type revolute; axis [0,0,1]; origin [0,0.384,0]; limits [-2.8973, 2.8973]; }
  joint j6 { type revolute; axis [0,1,0]; origin [0,0,0]; }
  joint j7 { type revolute; axis [0,0,1]; origin [0,0.107,0]; limits [-2.8973, 2.8973]; }

  link base   { spheres [[0,0,0.06, 0.07]]; parent world; joint_ref j1; }
  link arm1   { spheres [[0,0,0.15, 0.06], [0,0,0.30, 0.055]]; parent base; joint_ref j2; }
  link arm2   { spheres [[0,0,0.12, 0.05]]; parent arm1; joint_ref j3; }
  link forearm { spheres [[0,0,0.18, 0.048]]; parent arm2; joint_ref j5; }
  link ee     { spheres [[0,0,0.02, 0.03]]; parent forearm; joint_ref j7; }
}

task point_contact {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [1,0,0]; offset 0.4; }
  plane { point_on_link [0,0,0.02]; normal [0,1,0]; offset 0.0; }
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.3; }
}

clearance { min_distance 0.03; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 3);

    int num_inputs = dag.num_inputs();
    REQUIRE(num_inputs == 7);

    auto jacobian_nodes = cudro::build_analytical_jacobians(dag, constraint_outputs, num_inputs);
    REQUIRE(jacobian_nodes.size() == 3);
    for (int c = 0; c < 3; ++c) {
        REQUIRE(jacobian_nodes[c].size() == 7);
    }

    // Test across several configuration points
    std::vector<std::vector<double>> test_configs = {
        {0.1, -0.2, 0.3, -1.0, 0.2, 1.2, -0.5},
        {-0.4, 0.5, -0.6, -0.8, 0.3, 0.9, 0.4},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };

    const double h = 1e-5;
    for (const auto& q : test_configs) {
        auto vals = cudro::evaluate_dag(dag, num_inputs, q.data());

        for (int c = 0; c < 3; ++c) {
            for (int j = 0; j < num_inputs; ++j) {
                double ad_grad = vals[jacobian_nodes[c][j]];

                // Central finite difference
                std::vector<double> q_plus = q;
                std::vector<double> q_minus = q;
                q_plus[j] += h;
                q_minus[j] -= h;

                auto vals_plus = cudro::evaluate_dag(dag, num_inputs, q_plus.data());
                auto vals_minus = cudro::evaluate_dag(dag, num_inputs, q_minus.data());

                double g_plus = vals_plus[constraint_outputs[c]];
                double g_minus = vals_minus[constraint_outputs[c]];
                double fd_grad = (g_plus - g_minus) / (2.0 * h);

                CHECK(std::abs(ad_grad - fd_grad) < 1e-4);
            }
        }
    }
}

