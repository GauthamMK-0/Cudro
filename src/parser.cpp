#include <cudro/parser.hpp>
#include <cudro/ast.hpp>

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>

namespace cudro {

// ============================================================================
// Navigation
// ============================================================================

const Token& Parser::peek(size_t offset) const {
    size_t idx = std::min(pos_ + offset, tokens_.size() - 1);
    return tokens_[idx];
}

const Token& Parser::current() const {
    return peek(0);
}

bool Parser::check(TokenKind kind) const {
    return current().kind == kind;
}

Token Parser::advance() {
    if (!check(TokenKind::EndOfFile))
        ++pos_;
    return tokens_[pos_ - 1];
}

bool Parser::match(TokenKind kind) {
    if (check(kind)) {
        advance();
        return true;
    }
    return false;
}

Token Parser::consume(TokenKind kind, const char* message) {
    if (check(kind))
        return advance();

    // Report error at current token
    const Token& t = current();
    diags_.error(t.where, std::string(message) + ", got " + token_kind_name(t.kind));

    // Panic-mode recovery: skip until sync point
    synchronize();

    // Return dummy error token
    return Token{TokenKind::EndOfFile, "", current().where};
}

// ============================================================================
// Error recovery
// ============================================================================

void Parser::report_error(const Token& token, const char* message) {
    diags_.error(token.where, message);
}

void Parser::synchronize() {
    advance(); // consume the bad token

    while (!check(TokenKind::EndOfFile)) {
        if (check(TokenKind::Semicolon)) {
            advance();
            return;
        }
        if (check(TokenKind::RBrace)) {
            return;
        }
        switch (peek().kind) {
            case TokenKind::KwRobot:
            case TokenKind::KwJoint:
            case TokenKind::KwLink:
            case TokenKind::KwTask:
            case TokenKind::KwClearance:
                return;
            default:
                break;
        }
        advance();
    }
}

// Safe number parsing - returns NaN on error instead of throwing
double Parser::parse_number_safe() {
    if (!check(TokenKind::Number)) {
        diags_.error(current().where, "expected number, got " + std::string(token_kind_name(current().kind)));
        synchronize();
        return std::numeric_limits<double>::quiet_NaN();
    }
    Token t = advance();
    // Copy to string for std::stod (requires null-terminated string)
    std::string num_str(t.text);
    size_t pos = 0;
    double val = std::stod(num_str, &pos);
    if (pos != num_str.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return val;
}

// ============================================================================
// Grammar: Spec
// ============================================================================

Spec Parser::parse_spec() {
    Spec spec;
    while (!check(TokenKind::EndOfFile)) {
        if (check(TokenKind::KwRobot)) {
            spec.robots.push_back(parse_robot());
        } else if (check(TokenKind::KwTask)) {
            spec.tasks.push_back(parse_task());
        } else if (check(TokenKind::KwClearance)) {
            spec.clearances.push_back(parse_clearance());
        } else {
            diags_.error(current().where, "expected 'robot', 'task', or 'clearance'");
            advance();
        }
    }
    return spec;
}

Spec Parser::parse() {
    return parse_spec();
}

// ============================================================================
// Grammar: Robot
// ============================================================================

std::unique_ptr<RobotDecl> Parser::parse_robot() {
    auto robot = std::make_unique<RobotDecl>();
    robot->loc = consume(TokenKind::KwRobot, "expected 'robot'").where;
    robot->name = parse_ident();
    consume(TokenKind::LBrace, "expected '{' after robot name");

    while (!check(TokenKind::RBrace) && !check(TokenKind::EndOfFile)) {
        if (check(TokenKind::KwJoint)) {
            robot->joints.push_back(parse_joint());
        } else if (check(TokenKind::KwLink)) {
            robot->links.push_back(parse_link());
        } else {
            diags_.error(current().where, "expected 'joint' or 'link' inside robot");
            advance();
        }
    }
    consume(TokenKind::RBrace, "expected '}' after robot body");
    return robot;
}

// ============================================================================
// Grammar: Joint
// ============================================================================

std::unique_ptr<JointDecl> Parser::parse_joint() {
    auto joint = std::make_unique<JointDecl>();
    joint->loc = consume(TokenKind::KwJoint, "expected 'joint'").where;
    joint->name = parse_ident();
    consume(TokenKind::LBrace, "expected '{' after joint name");

    // Required: type
    consume(TokenKind::KwType, "expected 'type'");
    Token type_tok = consume(TokenKind::KwRevolute, "expected 'revolute' or 'fixed'");
    if (type_tok.kind == TokenKind::KwFixed) {
        joint->type = "fixed";
    } else {
        joint->type = "revolute";
    }
    consume(TokenKind::Semicolon, "expected ';' after type");

    // Required: axis
    consume(TokenKind::KwAxis, "expected 'axis'");
    joint->axis = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after axis");

    // Required: origin
    consume(TokenKind::KwOrigin, "expected 'origin'");
    joint->origin = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after origin");

    // Optional: limits
    if (match(TokenKind::KwLimits)) {
        consume(TokenKind::LBracket, "expected '[' for limits");
        double lo = parse_number_safe();
        consume(TokenKind::Comma, "expected ',' in limits");
        double hi = parse_number_safe();
        consume(TokenKind::RBracket, "expected ']' for limits");
        if (!std::isnan(lo) && !std::isnan(hi)) {
            joint->limits = std::make_pair(lo, hi);
        }
        consume(TokenKind::Semicolon, "expected ';' after limits");
    }

    consume(TokenKind::RBrace, "expected '}' after joint");
    return joint;
}

// ============================================================================
// Grammar: Link
// ============================================================================

std::unique_ptr<LinkDecl> Parser::parse_link() {
    auto link = std::make_unique<LinkDecl>();
    link->loc = consume(TokenKind::KwLink, "expected 'link'").where;
    link->name = parse_ident();
    consume(TokenKind::LBrace, "expected '{' after link name");

    while (!check(TokenKind::RBrace) && !check(TokenKind::EndOfFile)) {
        if (match(TokenKind::KwSpheres)) {
            consume(TokenKind::LBracket, "expected '[' after 'spheres'");
            // First sphere
            link->spheres.push_back(std::make_unique<Sphere>(parse_sphere()));
            while (match(TokenKind::Comma)) {
                link->spheres.push_back(std::make_unique<Sphere>(parse_sphere()));
            }
            consume(TokenKind::RBracket, "expected ']' after spheres list");
            consume(TokenKind::Semicolon, "expected ';' after spheres");
        } else if (match(TokenKind::KwParent)) {
            link->parent = parse_ident();
            consume(TokenKind::Semicolon, "expected ';' after parent");
        } else if (match(TokenKind::KwJointRef)) {
            link->joint_ref = parse_ident();
            consume(TokenKind::Semicolon, "expected ';' after joint_ref");
        } else {
            diags_.error(current().where, "expected 'spheres', 'parent', or 'joint_ref' inside link");
            advance();
        }
    }
    consume(TokenKind::RBrace, "expected '}' after link");
    return link;
}

// ============================================================================
// Grammar: Task
// ============================================================================

std::unique_ptr<TaskDecl> Parser::parse_task() {
    auto task = std::make_unique<TaskDecl>();
    task->loc = consume(TokenKind::KwTask, "expected 'task'").where;
    task->name = parse_ident();
    consume(TokenKind::LBrace, "expected '{' after task name");

    consume(TokenKind::KwLink, "expected 'link'");
    task->link = parse_ident();
    consume(TokenKind::Semicolon, "expected ';' after link");

    while (!check(TokenKind::RBrace) && !check(TokenKind::EndOfFile)) {
        if (match(TokenKind::KwPlane)) {
            task->planes.push_back(parse_plane_constraint());
        } else {
            diags_.error(current().where, "expected 'plane' constraint");
            advance();
        }
    }
    consume(TokenKind::RBrace, "expected '}' after task");
    return task;
}

// ============================================================================
// Grammar: Plane Constraint
// ============================================================================

std::unique_ptr<PlaneConstraint> Parser::parse_plane_constraint() {
    auto plane = std::make_unique<PlaneConstraint>();
    // KwPlane already consumed by match() in parse_task()
    plane->loc = consume(TokenKind::LBrace, "expected '{' after 'plane'").where;

    if (match(TokenKind::KwLink)) {
        plane->link = parse_ident();
        consume(TokenKind::Semicolon, "expected ';' after link");
    }

    consume(TokenKind::KwPointOnLink, "expected 'point_on_link'");
    plane->point_on_link = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after point_on_link");

    consume(TokenKind::KwNormal, "expected 'normal'");
    plane->normal = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after normal");

    consume(TokenKind::KwOffset, "expected 'offset'");
    plane->offset = parse_number_safe();
    consume(TokenKind::Semicolon, "expected ';' after offset");

    consume(TokenKind::RBrace, "expected '}' after plane");
    return plane;
}

// ============================================================================
// Grammar: Clearance
// ============================================================================

std::unique_ptr<ClearanceDecl> Parser::parse_clearance() {
    auto clearance = std::make_unique<ClearanceDecl>();
    clearance->loc = consume(TokenKind::KwClearance, "expected 'clearance'").where;
    consume(TokenKind::LBrace, "expected '{' after 'clearance'");

    consume(TokenKind::KwMinDistance, "expected 'min_distance'");
    clearance->min_distance = parse_number_safe();
    consume(TokenKind::Semicolon, "expected ';' after min_distance");

    consume(TokenKind::RBrace, "expected '}' after clearance");
    return clearance;
}

// ============================================================================
// Grammar: Primitives
// ============================================================================

Vec3 Parser::parse_vec3() {
    Vec3 v;
    v.loc = consume(TokenKind::LBracket, "expected '['").where;
    v.x = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    v.y = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    v.z = parse_number_safe();
    consume(TokenKind::RBracket, "expected ']'");
    return v;
}

Sphere Parser::parse_sphere() {
    Sphere s;
    s.loc = consume(TokenKind::LBracket, "expected '['").where;
    s.center.x = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    s.center.y = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    s.center.z = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    s.radius = parse_number_safe();
    consume(TokenKind::RBracket, "expected ']'");
    return s;
}

std::string Parser::parse_ident() {
    Token t = consume(TokenKind::Identifier, "expected identifier");
    return std::string(t.text);
}

} // namespace cudro