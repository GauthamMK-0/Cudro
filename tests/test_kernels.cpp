#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/codegen_c.hpp>
#include <cudro/jit_tcc.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>

#include <cmath>
#include <vector>
#include <random>

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

TEST_CASE("Scalar Codegen and JIT execution on panda7") {
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
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.02; }
}

clearance { min_distance 0.03; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = lr.dag.num_inputs();

    std::string c_code = cudro::generate_scalar_c(lr, cudro::CodegenOptions());
    REQUIRE(!c_code.empty());

    auto mod = cudro::TCCJIT::compile(c_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);

    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    std::vector<float> q(7, 0.0f);
    std::vector<float> g(1, 0.0f);
    eval_fn(q.data(), 7, g.data());

    // At q=0, constraint g evaluates to 0.333
    CHECK(std::abs(g[0] - 0.333f) < 1e-4f);
}

TEST_CASE("Manifold projection on planar2r") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task keep_ee_x {
  link ee;
  plane { point_on_link [0,0,0]; normal [1,0,0]; offset 1.0; }
}
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = lr.dag.num_inputs();

    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);

    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    // Initial non-zero angle where Jacobian is non-zero
    std::vector<float> q_init = {0.2f, 0.3f};
    std::vector<float> g_init(1, 0.0f);
    eval_fn(q_init.data(), 2, g_init.data());
    CHECK(std::abs(g_init[0]) > 0.01f);

    std::vector<float> q_proj(2, 0.0f);
    proj_fn(q_init.data(), 2, q_proj.data());

    std::vector<float> g_proj(1, 0.0f);
    eval_fn(q_proj.data(), 2, g_proj.data());
    CHECK(std::abs(g_proj[0]) < 1e-4f);
}

TEST_CASE("Batched Kernel Compilation and Multi-Config Execution") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; }
  link base { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1 { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task keep_ee_above {
  link ee;
  plane { point_on_link [0,0,0]; normal [0,0,1]; offset 0.1; }
}
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = lr.dag.num_inputs();

    // Check CPU feature detection
    bool has_avx2 = cudro::cpu_supports_avx2();
    (void)has_avx2;

    std::string batched_code = cudro::generate_batched_c(lr);
    REQUIRE(!batched_code.empty());

    auto mod = cudro::TCCJIT::compile(batched_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_batch_fn = mod.get_symbol<void(*)(const float*, int, int, float*)>("evaluate_batch");
    REQUIRE(eval_batch_fn != nullptr);

    const int batch_size = 16;
    std::vector<float> q_batch(batch_size * 2, 0.0f);
    std::vector<float> g_batch(batch_size * 1, 0.0f);

    eval_batch_fn(q_batch.data(), batch_size, 2, g_batch.data());

    for (int b = 0; b < batch_size; ++b) {
        CHECK(std::abs(g_batch[b] - (-0.1f)) < 1e-4f);
    }
}

TEST_CASE("Multi-constraint projection on planar2r (M=2, N=2)") {
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

    cudro::LowerResult lr;
    lr.dag = std::move(dag);
    lr.constraint_outputs = std::move(constraint_outputs);
    lr.num_inputs = lr.dag.num_inputs();
    REQUIRE(lr.num_inputs == 2);

    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    std::vector<std::vector<float>> test_inits = {
        {0.2f, 0.3f},
        {-0.4f, 0.6f},
        {0.5f, -0.3f}
    };

    for (const auto& q_init : test_inits) {
        std::vector<float> g_init(2, 0.0f);
        eval_fn(q_init.data(), 2, g_init.data());

        std::vector<float> q_proj(2, 0.0f);
        proj_fn(q_init.data(), 2, q_proj.data());

        std::vector<float> g_proj(2, 0.0f);
        eval_fn(q_proj.data(), 2, g_proj.data());

        // Check both constraints converged
        CHECK(std::abs(g_proj[0]) < 1e-4f);
        CHECK(std::abs(g_proj[1]) < 1e-4f);
        float err_sq = g_proj[0] * g_proj[0] + g_proj[1] * g_proj[1];
        CHECK(err_sq < 1e-8f);
    }
}

TEST_CASE("Dual-constraint line tracking on panda7 (M=2, N=7)") {
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

task line_tracking {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.4; }
  plane { point_on_link [0,0,0.02]; normal [0,1,0]; offset 0.0; }
}

clearance { min_distance 0.03; }
)";

    auto spec = parse_and_check(src);
    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 2);

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

    // Test across 50 pseudo-random configurations
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 50; ++trial) {
        std::vector<float> q_init(7);
        for (int i = 0; i < 7; ++i) {
            q_init[i] = dist(rng);
        }

        std::vector<float> q_proj(7, 0.0f);
        proj_fn(q_init.data(), 7, q_proj.data());

        // Check for finite outputs (no NaN / Inf)
        for (int i = 0; i < 7; ++i) {
            CHECK(std::isfinite(q_proj[i]));
        }

        std::vector<float> g_proj(2, 0.0f);
        eval_fn(q_proj.data(), 7, g_proj.data());

        float err_sq = g_proj[0] * g_proj[0] + g_proj[1] * g_proj[1];
        CHECK(err_sq < 1e-8f);
    }
}

TEST_CASE("Triple-constraint Cartesian point lock on panda7 (M=3, N=7)") {
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
    lr.num_inputs = lr.dag.num_inputs();
    REQUIRE(lr.num_inputs == 7);

    // Compile scalar kernel - exercises 3x3 inlined linear system solver with pivoting
    std::string scalar_code = cudro::generate_scalar_c(lr);
    auto mod = cudro::TCCJIT::compile(scalar_code);
    REQUIRE(mod.handle != nullptr);

    auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
    REQUIRE(eval_fn != nullptr);
    auto proj_fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(proj_fn != nullptr);

    // Test across 50 pseudo-random configurations
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.2f, 1.2f);

    for (int trial = 0; trial < 50; ++trial) {
        std::vector<float> q_init(7);
        for (int i = 0; i < 7; ++i) {
            q_init[i] = dist(rng);
        }

        std::vector<float> q_proj(7, 0.0f);
        proj_fn(q_init.data(), 7, q_proj.data());

        for (int i = 0; i < 7; ++i) {
            CHECK(std::isfinite(q_proj[i]));
        }

        std::vector<float> g_proj(3, 0.0f);
        eval_fn(q_proj.data(), 7, g_proj.data());

        float err_sq = g_proj[0] * g_proj[0] + g_proj[1] * g_proj[1] + g_proj[2] * g_proj[2];
        CHECK(err_sq < 1e-8f);
    }
}

TEST_CASE("Batched projection with multi-constraints (M=2 and M=3)") {
    // 1. Batched M=2 on panda7_line
    std::string src2 = R"(
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
task line_tracking {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.4; }
  plane { point_on_link [0,0,0.02]; normal [0,1,0]; offset 0.0; }
}
clearance { min_distance 0.03; }
)";

    auto spec2 = parse_and_check(src2);
    cudro::ExprDAG dag2;
    auto constraint_outputs2 = cudro::lower(spec2, dag2);
    REQUIRE(constraint_outputs2.size() == 2);

    cudro::LowerResult lr2;
    lr2.dag = std::move(dag2);
    lr2.constraint_outputs = std::move(constraint_outputs2);
    lr2.num_inputs = lr2.dag.num_inputs();

    std::string batched_code2 = cudro::generate_batched_c(lr2);
    auto mod2 = cudro::TCCJIT::compile(batched_code2);
    REQUIRE(mod2.handle != nullptr);

    auto proj_batch_fn2 = mod2.get_symbol<void(*)(const float*, int, int, float*)>("project_batch");
    REQUIRE(proj_batch_fn2 != nullptr);
    auto eval_batch_fn2 = mod2.get_symbol<void(*)(const float*, int, int, float*)>("evaluate_batch");
    REQUIRE(eval_batch_fn2 != nullptr);

    const int B = 32;
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q_batch2(B * 7);
    for (int i = 0; i < B * 7; ++i) q_batch2[i] = dist(rng);

    std::vector<float> q_proj_batch2(B * 7, 0.0f);
    proj_batch_fn2(q_batch2.data(), B, 7, q_proj_batch2.data());

    std::vector<float> g_batch2(B * 2, 0.0f);
    eval_batch_fn2(q_proj_batch2.data(), B, 7, g_batch2.data());

    for (int b = 0; b < B; ++b) {
        float err_sq = g_batch2[b * 2 + 0] * g_batch2[b * 2 + 0] +
                       g_batch2[b * 2 + 1] * g_batch2[b * 2 + 1];
        CHECK(err_sq < 1e-8f);
    }

    // 2. Batched M=3 on panda7_point
    std::string src3 = R"(
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

    auto spec3 = parse_and_check(src3);
    cudro::ExprDAG dag3;
    auto constraint_outputs3 = cudro::lower(spec3, dag3);
    REQUIRE(constraint_outputs3.size() == 3);

    cudro::LowerResult lr3;
    lr3.dag = std::move(dag3);
    lr3.constraint_outputs = std::move(constraint_outputs3);
    lr3.num_inputs = lr3.dag.num_inputs();

    std::string batched_code3 = cudro::generate_batched_c(lr3);
    auto mod3 = cudro::TCCJIT::compile(batched_code3);
    REQUIRE(mod3.handle != nullptr);

    auto proj_batch_fn3 = mod3.get_symbol<void(*)(const float*, int, int, float*)>("project_batch");
    REQUIRE(proj_batch_fn3 != nullptr);
    auto eval_batch_fn3 = mod3.get_symbol<void(*)(const float*, int, int, float*)>("evaluate_batch");
    REQUIRE(eval_batch_fn3 != nullptr);

    std::vector<float> q_batch3(B * 7);
    for (int i = 0; i < B * 7; ++i) q_batch3[i] = dist(rng);

    std::vector<float> q_proj_batch3(B * 7, 0.0f);
    proj_batch_fn3(q_batch3.data(), B, 7, q_proj_batch3.data());

    std::vector<float> g_batch3(B * 3, 0.0f);
    eval_batch_fn3(q_proj_batch3.data(), B, 7, g_batch3.data());

    for (int b = 0; b < B; ++b) {
        float err_sq = g_batch3[b * 3 + 0] * g_batch3[b * 3 + 0] +
                       g_batch3[b * 3 + 1] * g_batch3[b * 3 + 1] +
                       g_batch3[b * 3 + 2] * g_batch3[b * 3 + 2];
        CHECK(err_sq < 1e-8f);
    }
}

