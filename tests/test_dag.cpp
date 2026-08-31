#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/dag.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>

#include <sstream>

TEST_CASE("ExprDAG construction and constant folding") {
    cudro::ExprDAG dag;
    int c1 = dag.add_constant(10.0);
    int c2 = dag.add_constant(2.5);
    int add = dag.add_binary(cudro::NodeKind::Add, c1, c2);
    int mul = dag.add_binary(cudro::NodeKind::Mul, add, c2);

    int folded = dag.fold_constants();
    CHECK(folded > 0);
    CHECK(dag[mul].kind == cudro::NodeKind::Constant);
    CHECK(dag[mul].constant_value == doctest::Approx(31.25));
}

TEST_CASE("DAG lowering produces connected kinematic tree") {
    std::string src = R"(
robot planar2r {
  joint j1 { type revolute; axis [0,0,1]; origin [0,0,0]; }
  joint j2 { type revolute; axis [0,0,1]; origin [1.0,0,0]; }
  link base   { spheres [[0,0,0, 0.05]]; parent world; joint_ref j1; }
  link arm1   { spheres [[0.5,0,0, 0.04]]; parent base; joint_ref j2; }
  link ee     { spheres [[0.5,0,0, 0.03]]; parent arm1; }
}
task keep_ee_above {
  link ee;
  plane { point_on_link [0,0,0]; normal [0,0,1]; offset 0.1; }
}
)";

    cudro::DiagnosticBag diags;
    cudro::Lexer lexer("<test>", src, diags);
    auto tokens = lexer.tokenize();
    cudro::Parser parser(tokens, diags);
    auto spec = parser.parse();
    REQUIRE_FALSE(diags.has_errors());
    cudro::Sema sema(spec, diags);
    REQUIRE(sema.analyze());

    cudro::ExprDAG dag;
    auto constraint_outputs = cudro::lower(spec, dag);
    REQUIRE(constraint_outputs.size() == 1);
    CHECK(dag.size() > 50);
    CHECK(dag.num_inputs() == 2);

    std::ostringstream oss;
    dag.dump(oss);
    std::string dump_str = oss.str();
    CHECK(dump_str.find("Expression DAG") != std::string::npos);
}
