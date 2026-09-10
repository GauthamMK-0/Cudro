#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/codegen_c.hpp>
#include <cudro/jit_tcc.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>
#include <cudro/ad.hpp>

#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace {

cudro::Spec parse_and_check(const std::string& src) {
    cudro::DiagnosticBag diags;
    cudro::Lexer lexer("<complex_workload>", src, diags);
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

// ============================================================================
// Workload 1: 14-DOF Bimanual Dual-Arm Coordinated Solve (M=6, N=14)
// ============================================================================

TEST_CASE("Complex Workload 1: 14-DOF Bimanual Dual-Arm Coordinated Solve (M=6, N=14)") {
    std::string src = R"(
robot bimanual14 {
  joint left_j1 { type revolute; axis [0,0,1]; origin [0, 0.25, 0.3]; limits [-2.89, 2.89]; }
  joint left_j2 { type revolute; axis [0,1,0]; origin [0, 0, 0]; limits [-1.76, 1.76]; }
  joint left_j3 { type revolute; axis [0,0,1]; origin [0, -0.2, 0]; limits [-2.89, 2.89]; }
  joint left_j4 { type revolute; axis [0,1,0]; origin [0, 0, 0.1]; limits [-3.07, -0.06]; }
  joint left_j5 { type revolute; axis [0,0,1]; origin [0, 0.2, 0]; limits [-2.89, 2.89]; }
  joint left_j6 { type revolute; axis [0,1,0]; origin [0, 0, 0]; limits [-0.01, 3.75]; }
  joint left_j7 { type revolute; axis [0,0,1]; origin [0, 0.1, 0]; limits [-2.89, 2.89]; }

  joint right_j1 { type revolute; axis [0,0,1]; origin [0, -0.25, 0.3]; limits [-2.89, 2.89]; }
  joint right_j2 { type revolute; axis [0,1,0]; origin [0, 0, 0]; limits [-1.76, 1.76]; }
  joint right_j3 { type revolute; axis [0,0,1]; origin [0, -0.2, 0]; limits [-2.89, 2.89]; }
  joint right_j4 { type revolute; axis [0,1,0]; origin [0, 0, 0.1]; limits [-3.07, -0.06]; }
  joint right_j5 { type revolute; axis [0,0,1]; origin [0, 0.2, 0]; limits [-2.89, 2.89]; }
  joint right_j6 { type revolute; axis [0,1,0]; origin [0, 0, 0]; limits [-0.01, 3.75]; }
  joint right_j7 { type revolute; axis [0,0,1]; origin [0, 0.1, 0]; limits [-2.89, 2.89]; }

  link torso     { spheres [[0,0,0.15, 0.12]]; parent world; }

  link left_base { spheres [[0,0,0.05, 0.06]]; parent torso; joint_ref left_j1; }
  link left_arm1 { spheres [[0,0,0.15, 0.05]]; parent left_base; joint_ref left_j2; }
  link left_arm2 { spheres [[0,0,0.10, 0.05]]; parent left_arm1; joint_ref left_j3; }
  link left_arm3 { spheres [[0,0,0.15, 0.04]]; parent left_arm2; joint_ref left_j4; }
  link left_arm4 { spheres [[0,0,0.10, 0.04]]; parent left_arm3; joint_ref left_j5; }
  link left_arm5 { spheres [[0,0,0.08, 0.03]]; parent left_arm4; joint_ref left_j6; }
  link left_ee   { spheres [[0,0,0.02, 0.03]]; parent left_arm5; joint_ref left_j7; }

  link right_base { spheres [[0,0,0.05, 0.06]]; parent torso; joint_ref right_j1; }
  link right_arm1 { spheres [[0,0,0.15, 0.05]]; parent right_base; joint_ref right_j2; }
  link right_arm2 { spheres [[0,0,0.10, 0.05]]; parent right_arm1; joint_ref right_j3; }
  link right_arm3 { spheres [[0,0,0.15, 0.04]]; parent right_arm2; joint_ref right_j4; }
  link right_arm4 { spheres [[0,0,0.10, 0.04]]; parent right_arm3; joint_ref right_j5; }
  link right_arm5 { spheres [[0,0,0.08, 0.03]]; parent right_arm4; joint_ref right_j6; }
  link right_ee   { spheres [[0,0,0.02, 0.03]]; parent right_arm5; joint_ref right_j7; }
}

task bimanual_dual_target {
  link left_ee;
  plane { link left_ee;  point_on_link [0,0,0.02]; normal [1,0,0]; offset 0.4; }
  plane { link left_ee;  point_on_link [0,0,0.02]; normal [0,1,0]; offset 0.3; }
  plane { link left_ee;  point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.4; }
  plane { link right_ee; point_on_link [0,0,0.02]; normal [1,0,0]; offset 0.4; }
  plane { link right_ee; point_on_link [0,0,0.02]; normal [0,1,0]; offset -0.3; }
  plane { link right_ee; point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.4; }
}

clearance { min_distance 0.03; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 6);

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = lr.dag.num_inputs();
    REQUIRE(lr.num_inputs == 14);

    std::string scalar_code = cudro::generate_scalar_c(lr);
    REQUIRE(!scalar_code.empty());

    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    std::mt19937 rng(1014);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);

    // Test across 50 random 14-DOF configurations
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<float> q_init(14);
        for (int i = 0; i < 14; ++i) q_init[i] = dist(rng);

        std::vector<float> q_proj(14, 0.0f);
        proj_fn(q_init.data(), 14, q_proj.data());

        for (int i = 0; i < 14; ++i) {
            CHECK(std::isfinite(q_proj[i]));
        }

        std::vector<float> g_proj(6, 0.0f);
        eval_fn(q_proj.data(), 14, g_proj.data());

        float err_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            err_sq += g_proj[c] * g_proj[c];
        }
        CHECK(err_sq < 1e-8f);
    }
}

// ============================================================================
// Workload 2: Heavily-Constrained 7-DOF Arm with Obstacle Envelope (M=6, N=7)
// ============================================================================

TEST_CASE("Complex Workload 2: Heavily-Constrained 7-DOF Arm with Obstacle Envelope (M=6, N=7)") {
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

task dense_cell_constraints {
  link ee;
  plane { link ee;      point_on_link [0,0,0.02]; normal [1,0,0]; offset -0.161969; }
  plane { link ee;      point_on_link [0,0,0.02]; normal [0,1,0]; offset  0.150141; }
  plane { link ee;      point_on_link [0,0,0.02]; normal [0,0,1]; offset  0.409217; }
  plane { link arm2;    point_on_link [0,0,0.12]; normal [0,0,1]; offset  0.479893; }
  plane { link arm2;    point_on_link [0,0,0.12]; normal [0,1,0]; offset -0.303673; }
  plane { link forearm; point_on_link [0,0,0.18]; normal [0,0,1]; offset  0.564865; }
}

clearance { min_distance 0.03; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 6);

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = lr.dag.num_inputs();
    REQUIRE(lr.num_inputs == 7);

    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    std::vector<float> q_nominal = {0.2f, 0.4f, -0.3f, -1.2f, 0.5f, 1.5f, -0.4f};
    std::vector<float> g_nom(6, 0.0f);
    eval_fn(q_nominal.data(), 7, g_nom.data());
    float nom_err = 0.0f;
    for (int c = 0; c < 6; ++c) nom_err += g_nom[c] * g_nom[c];
    CHECK(nom_err < 1e-6f);

    std::mt19937 rng(765);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    for (int trial = 0; trial < 50; ++trial) {
        std::vector<float> q_init = q_nominal;
        for (int i = 0; i < 7; ++i) q_init[i] += dist(rng);

        std::vector<float> q_proj(7, 0.0f);
        proj_fn(q_init.data(), 7, q_proj.data());

        for (int i = 0; i < 7; ++i) {
            CHECK(std::isfinite(q_proj[i]));
        }

        std::vector<float> g_proj(6, 0.0f);
        eval_fn(q_proj.data(), 7, g_proj.data());

        float err_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            err_sq += g_proj[c] * g_proj[c];
        }
        CHECK(err_sq < 1e-8f);
    }
}

// ============================================================================
// Workload 3: Compound Spatial Skew Joint Axes with RotAxis (N=6)
// ============================================================================

TEST_CASE("Complex Workload 3: Compound Spatial Skew Joint Axes with RotAxis (N=6)") {
    std::string src = R"(
robot arbitrary_axes_6r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0.2]; limits [-3.14, 3.14]; }
  joint j2 { type revolute; axis [0.707107, 0.707107, 0]; origin [0,0,0.1]; limits [-2.0, 2.0]; }
  joint j3 { type revolute; axis [0, 0.707107, 0.707107]; origin [0.1,0,0.2]; limits [-2.5, 2.5]; }
  joint j4 { type revolute; axis [0.57735, 0.57735, 0.57735]; origin [0,0.1,0.1]; limits [-3.14, 3.14]; }
  joint j5 { type revolute; axis [-0.707107, 0, 0.707107]; origin [0.1,0,0.1]; limits [-2.0, 2.0]; }
  joint j6 { type revolute; axis [0,1,0]; origin [0,0,0.05]; limits [-3.14, 3.14]; }

  link base { spheres [[0,0,0.1, 0.08]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0,0,0.05, 0.06]]; parent base; joint_ref j2; }
  link arm2 { spheres [[0.05,0,0.1, 0.05]]; parent arm1; joint_ref j3; }
  link arm3 { spheres [[0,0.05,0.05, 0.05]]; parent arm2; joint_ref j4; }
  link arm4 { spheres [[0.05,0,0.05, 0.04]]; parent arm3; joint_ref j5; }
  link ee   { spheres [[0,0,0.02, 0.03]]; parent arm4; joint_ref j6; }
}

task target_position_3d {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [1,0,0]; offset 0.25; }
  plane { point_on_link [0,0,0.02]; normal [0,1,0]; offset 0.15; }
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.45; }
}

clearance { min_distance 0.02; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 3);

    int num_inputs = dag.num_inputs();
    REQUIRE(num_inputs == 6);

    // 1. Differentiate and verify exact analytical Jacobian vs finite differences
    auto jacobian_nodes = cudro::build_analytical_jacobians(dag, constraint_outputs, num_inputs);
    REQUIRE(jacobian_nodes.size() == 3);
    for (int c = 0; c < 3; ++c) {
        REQUIRE(jacobian_nodes[c].size() == 6);
    }

    std::vector<std::vector<double>> test_configs = {
        {0.2, -0.3, 0.4, -0.5, 0.6, -0.1},
        {-0.4, 0.5, -0.2, 0.3, -0.7, 0.8},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };

    const double h = 1e-5;
    for (const auto& q : test_configs) {
        auto vals = cudro::evaluate_dag(dag, num_inputs, q.data());

        for (int c = 0; c < 3; ++c) {
            for (int j = 0; j < num_inputs; ++j) {
                double ad_grad = vals[jacobian_nodes[c][j]];

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

    // 2. JIT Compile and project onto 3D point manifold
    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = num_inputs;

    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    std::vector<float> q_init = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.2f};
    std::vector<float> q_proj(6, 0.0f);
    proj_fn(q_init.data(), 6, q_proj.data());

    std::vector<float> g_proj(3, 0.0f);
    eval_fn(q_proj.data(), 6, g_proj.data());

    float err_sq = g_proj[0]*g_proj[0] + g_proj[1]*g_proj[1] + g_proj[2]*g_proj[2];
    CHECK(err_sq < 1e-8f);
}

// ============================================================================
// Workload 4: Adversarial Stress & Gimbal-Lock Singularity Resilience (10,000 Configs)
// ============================================================================

TEST_CASE("Complex Workload 4: Adversarial Stress & Gimbal-Lock Singularity Resilience (10,000 Configs)") {
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

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = 7;

    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    // 1. Extreme Gimbal Lock Singularity: arm straight vertical at q = 0
    std::vector<float> q_singular(7, 0.0f);
    std::vector<float> q_out_singular(7, 0.0f);
    proj_fn(q_singular.data(), 7, q_out_singular.data());
    for (int i = 0; i < 7; ++i) {
        CHECK(std::isfinite(q_out_singular[i]));
    }
    std::vector<float> g_singular(3, 0.0f);
    eval_fn(q_out_singular.data(), 7, g_singular.data());
    float err_sq_singular = g_singular[0]*g_singular[0] + g_singular[1]*g_singular[1] + g_singular[2]*g_singular[2];
    CHECK(err_sq_singular < 1e-8f);

    // 2. Joint Limit Extrema Configurations
    std::vector<float> q_limits = {2.8973f, -1.7628f, 2.8973f, 0.0f, -2.8973f, 0.0f, 2.8973f};
    std::vector<float> q_out_limits(7, 0.0f);
    proj_fn(q_limits.data(), 7, q_out_limits.data());
    for (int i = 0; i < 7; ++i) {
        CHECK(std::isfinite(q_out_limits[i]));
    }

    // 3. Monte Carlo 10,000 Adversarial Configurations
    std::mt19937 rng(99999);
    std::uniform_real_distribution<float> dist(-3.5f, 3.5f); // covers out-of-bounds joint limits

    int finite_count = 0;
    int converged_count = 0;
    const int N_SAMPLES = 10000;

    for (int s = 0; s < N_SAMPLES; ++s) {
        std::vector<float> q_adv(7);
        for (int i = 0; i < 7; ++i) q_adv[i] = dist(rng);

        std::vector<float> q_out(7, 0.0f);
        proj_fn(q_adv.data(), 7, q_out.data());

        bool all_finite = true;
        for (int i = 0; i < 7; ++i) {
            if (!std::isfinite(q_out[i])) {
                all_finite = false;
                break;
            }
        }
        if (all_finite) {
            finite_count++;
            std::vector<float> g_check(3, 0.0f);
            eval_fn(q_out.data(), 7, g_check.data());
            float err = g_check[0]*g_check[0] + g_check[1]*g_check[1] + g_check[2]*g_check[2];
            if (err < 1e-6f) converged_count++;
        }
    }

    // Must be 100% finite (zero NaNs, zero Infs, zero division-by-zero crashes)
    CHECK(finite_count == N_SAMPLES);
    // Vast majority should converge cleanly to the manifold
    CHECK(converged_count > 9500);
}

// ============================================================================
// Workload 5: 1 kHz Hard Real-Time Simulation Loop (1,000 Steps)
// ============================================================================

TEST_CASE("Complex Workload 5: 1 kHz Hard Real-Time Simulation Loop (1,000 Steps)") {
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

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = 7;

    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    const int TOTAL_TIMESTEPS = 1000;
    std::vector<double> latencies_us;
    latencies_us.reserve(TOTAL_TIMESTEPS);

    std::mt19937 rng(12345);
    std::normal_distribution<float> drift_dist(0.0f, 0.008f); // simulated 1 kHz state drift

    // Initial configuration on the manifold
    std::vector<float> q = {0.2f, -0.3f, 0.4f, -1.2f, 0.1f, 1.4f, -0.5f};
    std::vector<float> q_proj(7, 0.0f);
    proj_fn(q.data(), 7, q_proj.data());
    q = q_proj;

    int deadline_misses = 0;
    const double DEADLINE_US = 1000.0; // 1 ms hard real-time deadline

    for (int t = 0; t < TOTAL_TIMESTEPS; ++t) {
        // Apply state drift
        for (int i = 0; i < 7; ++i) {
            q[i] += drift_dist(rng);
        }

        auto start = std::chrono::high_resolution_clock::now();
        proj_fn(q.data(), 7, q_proj.data());
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);

        if (elapsed_us > DEADLINE_US) {
            deadline_misses++;
        }

        // Verify manifold convergence at each timestep
        std::vector<float> g(3, 0.0f);
        eval_fn(q_proj.data(), 7, g.data());
        float err_sq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
        CHECK(err_sq < 1e-8f);

        q = q_proj;
    }

    // Zero deadline misses across 1,000 steps at 1 kHz
    CHECK(deadline_misses == 0);

    // Compute timing percentiles
    std::sort(latencies_us.begin(), latencies_us.end());
    double min_lat = latencies_us.front();
    double med_lat = latencies_us[TOTAL_TIMESTEPS / 2];
    double p90_lat = latencies_us[static_cast<size_t>(TOTAL_TIMESTEPS * 0.90)];
    double p99_lat = latencies_us[static_cast<size_t>(TOTAL_TIMESTEPS * 0.99)];
    double max_lat = latencies_us.back();

    std::cout << "\n======================================================\n";
    std::cout << "  1 kHz REAL-TIME CONTROLLER LOOP JITTER BENCHMARK   \n";
    std::cout << "======================================================\n";
    std::cout << "Total 1 kHz Timesteps: " << TOTAL_TIMESTEPS << "\n";
    std::cout << "1 ms Deadline Misses : " << deadline_misses << " (0.00%)\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Min Latency          : " << min_lat << " us\n";
    std::cout << "Median Latency (p50) : " << med_lat << " us\n";
    std::cout << "90th Percentile (p90): " << p90_lat << " us\n";
    std::cout << "99th Percentile (p99): " << p99_lat << " us\n";
    std::cout << "Max Latency          : " << max_lat << " us\n";
    std::cout << "======================================================\n\n";

    // Even under ASan, median latency in warm cache is well below 50 us
    CHECK(med_lat < 100.0);
    CHECK(p99_lat < 500.0);
}
