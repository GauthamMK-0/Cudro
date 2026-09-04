#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "eigen_reference.hpp"

#include <cudro/ad.hpp>
#include <cudro/codegen_c.hpp>
#include <cudro/jit_tcc.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>

#include <random>
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

TEST_CASE("Differential validation on Planar 2R vs Eigen reference (10,000 configs)") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; limits [-3.1415, 3.1415]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; limits [-3.1415, 3.1415]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task keep_ee_x {
  link ee;
  plane { point_on_link [0,0,0]; normal [1,0,0]; offset 0.5; }
}
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    cudro::LowerResult lr;
    lr.dag = dag;
    lr.constraint_outputs = constraint_outputs;
    lr.num_inputs = 2;

    std::string c_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(c_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);

    Eigen::Vector3d normal(1.0, 0.0, 0.0);
    double offset = 0.5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-3.1415, 3.1415);

    const int num_samples = 10000;
    double max_err_g = 0.0;
    double max_err_J = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        Eigen::Vector2d q(dist(rng), dist(rng));

        // 1. Evaluate via Eigen reference
        double g_ref = cudro::reference::Planar2RReference::evaluate_plane_constraint(q, normal, offset);
        auto J_ref = cudro::reference::Planar2RReference::jacobian(q, normal, offset);

        // 2. Evaluate via JIT compiled kernel
        std::vector<float> q_f = {static_cast<float>(q(0)), static_cast<float>(q(1))};
        std::vector<float> g_jit(1, 0.0f);
        eval_fn(q_f.data(), 2, g_jit.data());

        // 3. Evaluate via AD pass
        double q_d[2] = {q(0), q(1)};
        std::vector<double> dag_vals = cudro::evaluate_dag(dag, 2, q_d);
        double g_dag = dag_vals[constraint_outputs[0]];

        double err_g = std::abs(g_ref - g_jit[0]);
        if (err_g > max_err_g) max_err_g = err_g;

        CHECK(std::abs(g_ref - g_dag) < 1e-4);
        CHECK(err_g < 1e-4);
    }

    CHECK(max_err_g < 1e-4);
}

TEST_CASE("Differential validation on Panda 7 vs Eigen reference (10,000 configs)") {
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
  link arm1   { spheres [[0,0,0.15, 0.06]]; parent base; joint_ref j2; }
  link arm2   { spheres [[0,0,0.12, 0.05]]; parent arm1; joint_ref j3; }
  link forearm { spheres [[0,0,0.18, 0.048]]; parent arm2; joint_ref j5; }
  link ee     { spheres [[0,0,0.02, 0.03]]; parent forearm; joint_ref j7; }
}

task cup_on_table {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [0,1,0]; offset 0.1; }
}
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    cudro::LowerResult lr;
    lr.dag = dag;
    lr.constraint_outputs = constraint_outputs;
    lr.num_inputs = 7;

    std::string c_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(c_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);

    Eigen::Vector3d normal(0.0, 1.0, 0.0);
    double offset = 0.1;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist(-1.5, 1.5);

    const int num_samples = 10000;
    double max_err_g = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        Eigen::Matrix<double, 7, 1> q;
        std::vector<float> q_f(7);
        for (int j = 0; j < 7; ++j) {
            q(j) = dist(rng);
            q_f[j] = static_cast<float>(q(j));
        }

        // 1. Evaluate via Eigen reference
        double g_ref = cudro::reference::Panda7Reference::evaluate_plane_constraint(q, normal, offset);

        // 2. Evaluate via JIT compiled kernel
        std::vector<float> g_jit(1, 0.0f);
        eval_fn(q_f.data(), 7, g_jit.data());

        double err_g = std::abs(g_ref - g_jit[0]);
        if (err_g > max_err_g) max_err_g = err_g;

        CHECK(err_g < 1e-4);
    }

    CHECK(max_err_g < 1e-4);
}

TEST_CASE("Projection accuracy over random initial configurations (1,000 samples)") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; limits [-3.1415, 3.1415]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; limits [-3.1415, 3.1415]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task keep_ee_x {
  link ee;
  plane { point_on_link [0,0,0]; normal [1,0,0]; offset 0.8; }
}
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    cudro::LowerResult lr;
    lr.dag = dag;
    lr.constraint_outputs = constraint_outputs;
    lr.num_inputs = 2;

    std::string c_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(c_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(eval_fn != nullptr);
    REQUIRE(proj_fn != nullptr);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(0.1f, 1.2f);

    const int num_samples = 1000;
    int converged_count = 0;

    for (int i = 0; i < num_samples; ++i) {
        std::vector<float> q_init = {dist(rng), dist(rng)};
        std::vector<float> q_proj(2, 0.0f);
        proj_fn(q_init.data(), 2, q_proj.data());

        std::vector<float> g_proj(1, 0.0f);
        eval_fn(q_proj.data(), 2, g_proj.data());

        if (std::abs(g_proj[0]) < 1e-3f) {
            converged_count++;
        }
    }

    CHECK(converged_count >= 990);
}
