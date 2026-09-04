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
