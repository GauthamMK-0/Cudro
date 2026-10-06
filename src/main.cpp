#include <cudro/ast.hpp>
#include <cudro/diagnostic.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>
#include <cudro/ad.hpp>
#include <cudro/codegen_c.hpp>
#include <cudro/jit_tcc.hpp>
#include <cudro/planner.hpp>
#include <cudro/version.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <chrono>

namespace {

struct ASTDumper : cudro::ASTVisitor {
    int indent = 0;

    void print_indent() {
        for (int i = 0; i < indent; ++i) std::printf("  ");
    }

    void visit(const cudro::Spec& n) override {
        std::printf("Spec\n");
        indent++;
        for (auto& r : n.robots) r->accept(*this);
        for (auto& t : n.tasks) t->accept(*this);
        for (auto& c : n.clearances) c->accept(*this);
        indent--;
    }

    void visit(const cudro::RobotDecl& n) override {
        print_indent(); std::printf("RobotDecl: %s\n", n.name.c_str());
        indent++;
        for (auto& j : n.joints) j->accept(*this);
        for (auto& l : n.links) l->accept(*this);
        indent--;
    }

    void visit(const cudro::JointDecl& n) override {
        print_indent(); std::printf("JointDecl: %s\n", n.name.c_str());
        indent++;
        print_indent(); std::printf("type: %s\n", n.type.c_str());
        print_indent(); std::printf("axis: [%.3f, %.3f, %.3f]\n", n.axis.x, n.axis.y, n.axis.z);
        print_indent(); std::printf("origin: [%.3f, %.3f, %.3f]\n", n.origin.x, n.origin.y, n.origin.z);
        if (n.limits) {
            print_indent(); std::printf("limits: [%.4f, %.4f]\n", n.limits->first, n.limits->second);
        }
        indent--;
    }

    void visit(const cudro::LinkDecl& n) override {
        print_indent(); std::printf("LinkDecl: %s\n", n.name.c_str());
        indent++;
        if (n.parent) { print_indent(); std::printf("parent: %s\n", n.parent->c_str()); }
        if (n.joint_ref) { print_indent(); std::printf("joint_ref: %s\n", n.joint_ref->c_str()); }
        for (auto& s : n.spheres) {
            print_indent(); std::printf("sphere: [%.3f, %.3f, %.3f, %.3f]\n", s->center.x, s->center.y, s->center.z, s->radius);
        }
        indent--;
    }

    void visit(const cudro::TaskDecl& n) override {
        print_indent(); std::printf("TaskDecl: %s\n", n.name.c_str());
        indent++;
        print_indent(); std::printf("link: %s\n", n.link.c_str());
        for (auto& p : n.planes) p->accept(*this);
        indent--;
    }

    void visit(const cudro::PlaneConstraint& n) override {
        print_indent(); std::printf("PlaneConstraint\n");
        indent++;
        print_indent(); std::printf("link: %s\n", n.link.c_str());
        print_indent(); std::printf("point_on_link: [%.3f, %.3f, %.3f]\n", n.point_on_link.x, n.point_on_link.y, n.point_on_link.z);
        print_indent(); std::printf("normal: [%.3f, %.3f, %.3f]\n", n.normal.x, n.normal.y, n.normal.z);
        print_indent(); std::printf("offset: %.3f\n", n.offset);
        indent--;
    }

    void visit(const cudro::ClearanceDecl& n) override {
        print_indent(); std::printf("ClearanceDecl: min_distance = %.3f\n", n.min_distance);
    }
};

void print_usage() {
    std::fprintf(stderr, "usage: cudro <command> [spec-file]\n"
                         "commands:\n"
                         "  --dump-tokens     Tokenize spec and print token stream\n"
                         "  --dump-ast        Parse spec and print AST structure\n"
                         "  --check           Run semantic analysis and kinematic checks\n"
                         "  --dump-dag        Lower spec to Expression DAG and print nodes\n"
                         "  --dump-jacobian   Compute and print analytical Jacobian matrix\n"
                         "  --emit-c          Emit scalar C source code\n"
                         "  --jit-run         JIT-compile in memory and run projection on q=0\n"
                         "  --jit-bench       Benchmark batched in-memory JIT execution\n"
                         "  --plan            Plan a constraint-satisfying trajectory via in-kernel JIT\n"
                         "  --version         Print compiler version\n"
                         "  --help, -h        Print this help message\n");
}


} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "--version") {
        std::printf("%s\n", cudro::version_string);
        return 0;
    }

    if (cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    if (argc < 3) {
        std::fprintf(stderr, "error: missing spec file\n\n");
        print_usage();
        return 1;
    }

    std::string spec_file = argv[2];
    std::ifstream in(spec_file);
    if (!in) {
        std::fprintf(stderr, "error: cannot open %s\n", spec_file.c_str());
        return 1;
    }
    std::stringstream buf; buf << in.rdbuf();
    std::string src = buf.str();

    cudro::DiagnosticBag diags;

    // Lexer
    cudro::Lexer lexer(spec_file, src, diags);
    auto tokens = lexer.tokenize();
    if (diags.has_errors()) { cudro::print_diagnostics(diags, src); return 1; }

    if (cmd == "--dump-tokens") {
        for (const auto& t : tokens) {
            std::printf("%3d:%-3d %-16s %.*s\n", t.where.line, t.where.col,
                        cudro::token_kind_name(t.kind),
                        static_cast<int>(t.text.size()), t.text.data());
        }
        return 0;
    }

    // Parser
    cudro::Parser parser(tokens, diags);
    auto spec = parser.parse();
    if (diags.has_errors()) { cudro::print_diagnostics(diags, src); return 1; }

    if (cmd == "--dump-ast") {
        ASTDumper dumper;
        spec.accept(dumper);
        return 0;
    }

    if (cmd == "--check") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        std::printf("OK: semantic checks and kinematic validation passed.\n");
        return 0;
    }

    if (cmd == "--dump-dag") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        cudro::lower(spec, dag);
        dag.dump(std::cout);
        return 0;
    }

    if (cmd == "--dump-jacobian") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        auto constraint_outputs = cudro::lower(spec, dag);
        auto ad_result = cudro::differentiate(dag, constraint_outputs, dag.num_inputs());
        std::cout << "Jacobian (" << ad_result.num_constraints << "x" << ad_result.num_inputs << "):\n";
        for (int c = 0; c < ad_result.num_constraints; ++c) {
            std::cout << "  constraint " << c << ": ";
            for (int i = 0; i < ad_result.num_inputs; ++i) {
                std::printf("%.6e ", ad_result.jacobians[c][i]);
            }
            std::cout << "\n";
        }
        return 0;
    }

    if (cmd == "--emit-c") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        auto constraint_outputs = cudro::lower(spec, dag);
        cudro::LowerResult lr;
        lr.dag = std::move(dag);
        lr.constraint_outputs = std::move(constraint_outputs);
        lr.num_inputs = lr.dag.num_inputs();

        std::string c_source = cudro::generate_scalar_c(lr);
        std::cout << c_source << "\n";
        return 0;
    }

    if (cmd == "--jit-run") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        auto constraint_outputs = cudro::lower(spec, dag);
        
        cudro::LowerResult lr;
        lr.dag = std::move(dag);
        lr.constraint_outputs = std::move(constraint_outputs);
        lr.num_inputs = lr.dag.num_inputs();
        
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string c_source = cudro::generate_scalar_c(lr);
        auto mod = cudro::TCCJIT::compile(c_source);
        auto t1 = std::chrono::high_resolution_clock::now();
        double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!mod.handle) {
            std::fprintf(stderr, "JIT compilation failed\n");
            return 1;
        }
        
        auto eval_fn = mod.get_symbol<void(*)(const float*, int, float*)>("evaluate_constraints");
        auto proj_fn = mod.get_symbol<int(*)(const float*, int, float*)>("project");
        if (!eval_fn || !proj_fn) {
            std::fprintf(stderr, "Failed to resolve kernel entry points\n");
            return 1;
        }
        
        int num_inputs = lr.dag.num_inputs();
        int num_constraints = lr.constraint_outputs.size();
        std::vector<float> q(num_inputs, 0.0f);
        std::vector<float> g_init(num_constraints, 0.0f);
        std::vector<float> q_proj(num_inputs, 0.0f);
        std::vector<float> g_proj(num_constraints, 0.0f);
        
        eval_fn(q.data(), num_inputs, g_init.data());
        int status = proj_fn(q.data(), num_inputs, q_proj.data());
        eval_fn(q_proj.data(), num_inputs, g_proj.data());

        const char* status_str = (status == 0) ? "SUCCESS (converged)" 
                               : (status == 1) ? "MAX_ITERS exceeded" 
                               : "NUMERICAL_ERROR";

        std::printf("JIT Compilation Time: %.2f ms\n", compile_ms);
        std::printf("Projection Status: %s (code %d)\n", status_str, status);
        std::printf("Initial q = [");
        for (int i = 0; i < num_inputs; ++i) std::printf("%.3f%s", q[i], i + 1 < num_inputs ? ", " : "");
        std::printf("] -> g(q) = [");
        for (int i = 0; i < num_constraints; ++i) std::printf("%.4f%s", g_init[i], i + 1 < num_constraints ? ", " : "");
        std::printf("]\n");

        std::printf("Projected q* = [");
        for (int i = 0; i < num_inputs; ++i) std::printf("%.3f%s", q_proj[i], i + 1 < num_inputs ? ", " : "");
        std::printf("] -> g(q*) = [");
        for (int i = 0; i < num_constraints; ++i) std::printf("%.4e%s", g_proj[i], i + 1 < num_constraints ? ", " : "");
        std::printf("]\n");

        return 0;
    }

    if (cmd == "--jit-bench") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        auto constraint_outputs = cudro::lower(spec, dag);
        
        cudro::LowerResult lr;
        lr.dag = std::move(dag);
        lr.constraint_outputs = std::move(constraint_outputs);
        lr.num_inputs = lr.dag.num_inputs();
        
        std::string batched_src = cudro::generate_batched_c(lr);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        auto mod = cudro::TCCJIT::compile(batched_src);
        auto t1 = std::chrono::high_resolution_clock::now();
        double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!mod.handle) {
            std::fprintf(stderr, "JIT compilation failed\n");
            return 1;
        }
        
        auto batch_proj_fn = mod.get_symbol<void(*)(const float*, int, int, float*)>("project_batch");
        auto batch_eval_fn = mod.get_symbol<void(*)(const float*, int, int, float*)>("evaluate_batch");
        if (!batch_proj_fn || !batch_eval_fn) {
            std::fprintf(stderr, "Failed to resolve batched symbols\n");
            return 1;
        }
        
        const int batch_size = 10000;
        int num_inputs = lr.dag.num_inputs();
        int num_constraints = lr.constraint_outputs.size();
        
        std::vector<float> q_batch(batch_size * num_inputs, 0.1f);
        std::vector<float> q_proj_batch(batch_size * num_inputs, 0.0f);
        std::vector<float> g_batch(batch_size * num_constraints, 0.0f);
        
        std::printf("JIT Compile Latency: %.2f ms\n", compile_ms);

        auto exec_t0 = std::chrono::high_resolution_clock::now();
        batch_eval_fn(q_batch.data(), batch_size, num_inputs, g_batch.data());
        auto exec_t1 = std::chrono::high_resolution_clock::now();
        double eval_us = std::chrono::duration<double, std::micro>(exec_t1 - exec_t0).count();

        auto proj_t0 = std::chrono::high_resolution_clock::now();
        batch_proj_fn(q_batch.data(), batch_size, num_inputs, q_proj_batch.data());
        auto proj_t1 = std::chrono::high_resolution_clock::now();
        double proj_us = std::chrono::duration<double, std::micro>(proj_t1 - proj_t0).count();

        std::printf("Batched Constraint Eval: %d configs in %.2f us (%.2f ns/config, %.2f M configs/sec)\n",
                    batch_size, eval_us, (eval_us * 1000.0) / batch_size, (batch_size / eval_us));
        std::printf("Batched Manifold Project: %d configs in %.2f us (%.2f us/config, %.2f k projections/sec)\n",
                    batch_size, proj_us, (proj_us) / batch_size, (batch_size * 1000.0 / proj_us));
        return 0;
    }

    if (cmd == "--plan") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        auto constraint_outputs = cudro::lower(spec, dag);
        cudro::LowerResult lr;
        lr.dag = std::move(dag);
        lr.constraint_outputs = std::move(constraint_outputs);
        lr.num_inputs = lr.dag.num_inputs();

        std::string c_source = cudro::generate_scalar_c(lr);
        auto mod = cudro::TCCJIT::compile(c_source);
        if (!mod.handle) {
            std::fprintf(stderr, "JIT compilation failed\n");
            return 1;
        }

        auto eval_fn = mod.get_symbol<cudro::ConstrainedPlanner::EvaluateFn>("evaluate_constraints");
        auto proj_fn = mod.get_symbol<cudro::ConstrainedPlanner::ProjectFn>("project");
        if (!eval_fn || !proj_fn) {
            std::fprintf(stderr, "Failed to resolve kernel entry points\n");
            return 1;
        }

        int dof = lr.num_inputs;
        int num_constraints = lr.constraint_outputs.size();

        cudro::JointLimits limits;
        limits.lower.assign(dof, -3.14159f);
        limits.upper.assign(dof, 3.14159f);

        // Read robot limits if available
        if (!spec.robots.empty()) {
            const auto& robot = spec.robots[0];
            for (size_t i = 0; i < robot->joints.size() && static_cast<int>(i) < dof; ++i) {
                if (robot->joints[i]->limits) {
                    limits.lower[i] = static_cast<float>(robot->joints[i]->limits->first);
                    limits.upper[i] = static_cast<float>(robot->joints[i]->limits->second);
                }
            }
        }

        cudro::PlannerOptions opts;
        opts.max_iterations = 4000;
        opts.step_size = 0.15f;
        opts.goal_bias = 0.2f;
        opts.constraint_tolerance = 1e-3f;

        cudro::ConstrainedPlanner planner(dof, num_constraints, proj_fn, eval_fn, limits, opts);

        // Find valid start and goal configurations on the manifold
        std::vector<float> start(dof, 0.0f);
        std::vector<float> goal(dof, 0.0f);
        bool found_start = false, found_goal = false;
        std::mt19937 seed_rng(12345);

        for (int attempt = 0; attempt < 500; ++attempt) {
            std::vector<float> q_rand(dof);
            for (int i = 0; i < dof; ++i) {
                std::uniform_real_distribution<float> dist(limits.lower[i], limits.upper[i]);
                q_rand[i] = dist(seed_rng);
            }
            std::vector<float> q_proj;
            if (planner.project_configuration(q_rand, q_proj)) {
                if (!found_start) {
                    start = q_proj;
                    found_start = true;
                } else if (!found_goal) {
                    float d = 0.0f;
                    for (int i = 0; i < dof; ++i) d += (q_proj[i] - start[i]) * (q_proj[i] - start[i]);
                    if (d > 0.1f) {
                        goal = q_proj;
                        found_goal = true;
                        break;
                    }
                }
            }
        }

        if (!found_start || !found_goal) {
            std::printf("FAILED: No reachable configurations found for the specified constraint on this robot.\n");
            return 1;
        }

        std::printf("Planning constrained trajectory (%d DoF, %d constraints)...\n", dof, num_constraints);
        auto result = planner.plan(start, goal);

        if (result.success) {
            std::printf("SUCCESS: Found feasible path in %.2f ms (%d iterations, %d JIT projection calls)\n",
                        result.planning_time_ms, result.iterations, result.projection_count);
            std::printf("Trajectory Waypoints: %zu\n", result.path.size());
            for (size_t k = 0; k < result.path.size(); ++k) {
                std::vector<float> g(num_constraints, 0.0f);
                eval_fn(result.path[k].data(), dof, g.data());
                std::printf("  [%2zu] q = [", k);
                for (int i = 0; i < dof; ++i) std::printf("%.3f%s", result.path[k][i], i + 1 < dof ? ", " : "");
                std::printf("] | g = [");
                for (int c = 0; c < num_constraints; ++c) std::printf("%.2e%s", g[c], c + 1 < num_constraints ? ", " : "");
                std::printf("]\n");
            }
        } else {
            std::printf("FAILED: %s (in %.2f ms)\n", result.message.c_str(), result.planning_time_ms);
            return 1;
        }


        return 0;
    }

    std::fprintf(stderr, "error: unknown command %s\n\n", cmd.c_str());
    print_usage();
    return 1;
}