#pragma once

#include <cudro/ast.hpp>
#include <cudro/diagnostic.hpp>
#include <cudro/token.hpp>

#include <vector>
#include <memory>
#include <string>
#include <optional>

namespace cudro {

class Parser {
public:
    Parser(const std::vector<Token>& tokens, DiagnosticBag& diags)
        : tokens_(tokens), diags_(diags), pos_(0) {}

    Spec parse();

private:
    // Navigation
    const Token& peek(size_t offset = 0) const;
    const Token& current() const;
    bool check(TokenKind kind) const;
    Token advance();
    bool match(TokenKind kind);
    Token consume(TokenKind kind, const char* message);

    // Error recovery
    void synchronize();
    void report_error(const Token& token, const char* message);

    // Grammar entry points
    Spec parse_spec();
    std::unique_ptr<RobotDecl> parse_robot();
    std::unique_ptr<JointDecl> parse_joint();
    std::unique_ptr<LinkDecl> parse_link();
    std::unique_ptr<TaskDecl> parse_task();
    std::unique_ptr<ClearanceDecl> parse_clearance();
    std::unique_ptr<PlaneConstraint> parse_plane_constraint();
    Vec3 parse_vec3();
    Sphere parse_sphere();
    double parse_number_safe();  // Returns NaN on error instead of throwing
    std::string parse_ident();

    const std::vector<Token>& tokens_;
    DiagnosticBag& diags_;
    size_t pos_ = 0;
};

} // namespace cudro