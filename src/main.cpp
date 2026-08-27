#include <cudro/ast.hpp>
#include <cudro/diagnostic.hpp>
#include <cudro/lexer.hpp>
#include <cudro/parser.hpp>
#include <cudro/version.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

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
        std::printf("OK\n");
        return 0;
    }

    std::fprintf(stderr, "error: unknown command %s\n", cmd.c_str());
    return 1;
}