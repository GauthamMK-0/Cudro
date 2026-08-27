#pragma once

#include <cudro/diagnostic.hpp>
#include <cudro/token.hpp>

#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace cudro {

// Forward declarations
struct Spec;
struct RobotDecl;
struct JointDecl;
struct LinkDecl;
struct TaskDecl;
struct PlaneConstraint;
struct ClearanceDecl;
struct Vec3;
struct Sphere;

// Base AST node with source location
struct ASTNode {
    Location loc;
    virtual ~ASTNode() = default;
    virtual void accept(class ASTVisitor& v) = 0;
};

// Visitor interface for AST passes (dump, sema, lower, serialize)
struct ASTVisitor {
    virtual ~ASTVisitor() = default;
    virtual void visit(const Spec& n) = 0;
    virtual void visit(const RobotDecl& n) = 0;
    virtual void visit(const JointDecl& n) = 0;
    virtual void visit(const LinkDecl& n) = 0;
    virtual void visit(const TaskDecl& n) = 0;
    virtual void visit(const PlaneConstraint& n) = 0;
    virtual void visit(const ClearanceDecl& n) = 0;
};

// ============================================================================
// Value types (plain structs, no inheritance, used by value)
// ============================================================================

struct Vec3 {
    Location loc;
    double x = 0, y = 0, z = 0;
};

struct Sphere {
    Location loc;
    Vec3 center;
    double radius = 0;
};

// ============================================================================
// Declaration nodes (heap-allocated, inherit from ASTNode)
// ============================================================================

struct JointDecl : ASTNode {
    std::string name;
    std::string type;  // "revolute" | "fixed"
    Vec3 axis;
    Vec3 origin;
    std::optional<std::pair<double, double>> limits;
    void accept(class ASTVisitor& v) override;
};

struct LinkDecl : ASTNode {
    std::string name;
    std::vector<std::unique_ptr<Sphere>> spheres;
    std::optional<std::string> parent;
    std::optional<std::string> joint_ref;
    void accept(class ASTVisitor& v) override;
};

struct RobotDecl : ASTNode {
    std::string name;
    std::vector<std::unique_ptr<JointDecl>> joints;
    std::vector<std::unique_ptr<LinkDecl>> links;
    void accept(class ASTVisitor& v) override;
};

struct PlaneConstraint : ASTNode {
    std::string link;
    Vec3 point_on_link;
    Vec3 normal;
    double offset = 0;
    void accept(class ASTVisitor& v) override;
};

struct TaskDecl : ASTNode {
    std::string name;
    std::string link;
    std::vector<std::unique_ptr<PlaneConstraint>> planes;
    void accept(class ASTVisitor& v) override;
};

struct ClearanceDecl : ASTNode {
    double min_distance = 0;
    void accept(class ASTVisitor& v) override;
};

// ============================================================================
// Top-level spec
// ============================================================================

struct Spec {
    std::vector<std::unique_ptr<RobotDecl>> robots;
    std::vector<std::unique_ptr<TaskDecl>> tasks;
    std::vector<std::unique_ptr<ClearanceDecl>> clearances;

    void accept(class ASTVisitor& v);
};

// ============================================================================
// Factory helpers
// ============================================================================

std::unique_ptr<JointDecl> make_joint(std::string name, std::string type,
                                       Vec3 axis, Vec3 origin,
                                       std::optional<std::pair<double,double>> limits, Location loc);
std::unique_ptr<LinkDecl> make_link(std::string name,
                                     std::vector<std::unique_ptr<Sphere>> spheres,
                                     std::optional<std::string> parent,
                                     std::optional<std::string> joint_ref, Location loc);
std::unique_ptr<RobotDecl> make_robot(std::string name,
                                       std::vector<std::unique_ptr<JointDecl>> joints,
                                       std::vector<std::unique_ptr<LinkDecl>> links, Location loc);
std::unique_ptr<PlaneConstraint> make_plane(std::string link, Vec3 point, Vec3 normal, double offset, Location loc);
std::unique_ptr<TaskDecl> make_task(std::string name, std::string link,
                                     std::vector<std::unique_ptr<PlaneConstraint>> planes, Location loc);
std::unique_ptr<ClearanceDecl> make_clearance(double min_distance, Location loc);

} // namespace cudro