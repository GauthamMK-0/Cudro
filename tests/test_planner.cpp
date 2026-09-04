#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cudro/planner.hpp>
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

TEST_CASE("Constrained Motion Planning on Planar 2R") {
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

    auto eval_fn = mod.get_symbol<cudro::ConstrainedPlanner::EvaluateFn>("evaluate_constraints");
    auto proj_fn = mod.get_symbol<cudro::ConstrainedPlanner::ProjectFn>("project");
    REQUIRE(eval_fn != nullptr);
    REQUIRE(proj_fn != nullptr);

    cudro::JointLimits limits;
    limits.lower = {-3.1415f, -3.1415f};
    limits.upper = {3.1415f, 3.1415f};

    cudro::PlannerOptions opts;
    opts.max_iterations = 3000;
    opts.step_size = 0.15f;
    opts.goal_bias = 0.2f;
    opts.constraint_tolerance = 1e-3f;

    cudro::ConstrainedPlanner planner(2, 1, proj_fn, eval_fn, limits, opts);

    // Initial configurations that project onto manifold
    std::vector<float> start_init = {0.3f, 0.5f};
    std::vector<float> goal_init = {0.8f, 1.0f};

    auto result = planner.plan(start_init, goal_init);
    std::printf("Planar2R Result: success=%d, msg=%s, iters=%d, projections=%d, time=%.2f ms, path_size=%zu\n",
                result.success, result.message.c_str(), result.iterations, result.projection_count, result.planning_time_ms, result.path.size());
    REQUIRE(result.success);


    CHECK(result.path.size() >= 2);
    CHECK(result.planning_time_ms > 0.0);

    // Validate that every waypoint on the trajectory satisfies constraints
    bool path_valid = planner.validate_path(result.path, 1e-3f);
    CHECK(path_valid);
}

TEST_CASE("Constrained Motion Planning on Panda 7-DOF Manipulator") {
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

task table_constraint {
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

    auto eval_fn = mod.get_symbol<cudro::ConstrainedPlanner::EvaluateFn>("evaluate_constraints");
    auto proj_fn = mod.get_symbol<cudro::ConstrainedPlanner::ProjectFn>("project");
    REQUIRE(eval_fn != nullptr);
    REQUIRE(proj_fn != nullptr);

    cudro::JointLimits limits;
    limits.lower = {-2.8f, -1.7f, -2.8f, -1.7f, -2.8f, -1.7f, -2.8f};
    limits.upper = {2.8f, 1.7f, 2.8f, 1.7f, 2.8f, 1.7f, 2.8f};

    cudro::PlannerOptions opts;
    opts.max_iterations = 4000;
    opts.step_size = 0.2f;
    opts.goal_bias = 0.2f;
    opts.constraint_tolerance = 1e-3f;

    cudro::ConstrainedPlanner planner(7, 1, proj_fn, eval_fn, limits, opts);

    std::vector<float> start_init = {0.2f, 0.1f, 0.3f, 0.0f, 0.1f, 0.0f, 0.0f};
    std::vector<float> goal_init = {-0.3f, -0.2f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f};

    auto result = planner.plan(start_init, goal_init);
    REQUIRE(result.success);
    CHECK(result.path.size() >= 2);
    CHECK(result.planning_time_ms > 0.0);

    bool path_valid = planner.validate_path(result.path, 1e-3f);
    CHECK(path_valid);
}
