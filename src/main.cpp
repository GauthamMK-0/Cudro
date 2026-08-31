#include <cudro/ast.hpp>
#include <cudro/diagnostic.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/sema.hpp>
#include <cudro/lower.hpp>
#include <cudro/ad.hpp>
#include <cudro/codegen_c.hpp>
#include <cudro/jit_tcc.hpp>
#include <cudro/version.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: cudro <command> <spec-file>\n"
                        "commands: --dump-tokens, --dump-ast, --check, --version\n");
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "--version") {
        std::printf("%s\n", cudro::version_string);
        return 0;
    }

    if (argc < 3) {
        std::fprintf(stderr, "error: missing spec file\n");
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
        std::printf("OK\n");
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

    if (cmd == "--jit-run") {
        cudro::Sema sema(spec, diags);
        if (!sema.analyze()) {
            cudro::print_diagnostics(diags, src);
            return 1;
        }
        cudro::ExprDAG dag;
        auto constraint_outputs = cudro::lower(spec, dag);
        
        // Create LowerResult for codegen
        cudro::LowerResult lr;
        lr.dag = std::move(dag);
        lr.constraint_outputs = std::move(constraint_outputs);
        lr.num_inputs = lr.dag.num_inputs();
        
        // Generate C code
        cudro::CodegenOptions opts;
        std::string c_source = cudro::generate_scalar_c(lr, cudro::CodegenOptions());
        
        // Debug: print generated C code
        std::cerr << "=== Generated C Code ===\n" << c_source << "\n=== End C Code ===\n";
        
        // JIT compile
        
        // JIT compile
        auto mod = cudro::TCCJIT::compile(c_source);
        if (!mod.handle) {
            std::fprintf(stderr, "JIT compilation failed\n");
            return 1;
        }
        
        // Get function pointer
        auto fn = mod.get_symbol<void(*)(const float*, int, float*)>("project");
        if (!fn) {
            std::fprintf(stderr, "Failed to get symbol 'project'\n");
            return 1;
        }
        
        // Run with zero config
        int num_inputs = lr.dag.num_inputs();
        int num_constraints = lr.constraint_outputs.size();
        std::vector<float> q(lr.dag.num_inputs(), 0.0f);
        std::vector<float> out_g(lr.constraint_outputs.size(), 0.0f);
        
        std::printf("Running JIT kernel with q=0...\n");
        fn(q.data(), num_inputs, out_g.data());
        
        std::printf("g = [");
        for (int i = 0; i < num_constraints; ++i) {
            std::printf("%g", out_g[i]);
            if (i < num_constraints - 1) std::printf(", ");
        }
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
        
        cudro::CodegenOptions opts;
        std::string batched_src = cudro::generate_batched_c(lr, opts);
        
        auto mod = cudro::TCCJIT::compile(batched_src);
        if (!mod.handle) {
            std::fprintf(stderr, "JIT compilation failed\n");
            return 1;
        }
        
        auto batch_fn = mod.get_symbol<void(*)(const float*, int, int, float*)>("project_batch");
        if (!batch_fn) {
            std::fprintf(stderr, "Failed to get symbol 'project_batch'\n");
            return 1;
        }
        
        const int batch_size = 1024;
        int num_inputs = lr.dag.num_inputs();
        int num_constraints = lr.constraint_outputs.size();
        
        std::vector<float> q_batch(batch_size * num_inputs, 0.1f);
        std::vector<float> g_batch(batch_size * num_constraints, 0.0f);
        
        std::printf("Benchmarking batched kernel on %d configurations...\n", batch_size);
        batch_fn(q_batch.data(), batch_size, num_inputs, g_batch.data());
        std::printf("Done. Evaluated %d configurations successfully.\n", batch_size);
        return 0;
    }

    std::fprintf(stderr, "error: unknown command %s\n", cmd.c_str());
    return 1;
}