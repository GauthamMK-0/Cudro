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

    auto fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
    REQUIRE(fn != nullptr);

    std::vector<float> q(7, 0.0f);
    std::vector<float> g(1, 0.0f);
    fn(q.data(), 7, g.data());

    // At q=0, constraint g evaluates to 0.333
    CHECK(std::abs(g[0] - 0.333f) < 1e-4f);
}
