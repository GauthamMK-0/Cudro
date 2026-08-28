#pragma once

#include <cudro/ast.hpp>
#include <cudro/diagnostic.hpp>

#include <unordered_map>
#include <string>
#include <vector>

namespace cudro {

class Sema {
public:
    Sema(const Spec& spec, DiagnosticBag& diags)
        : spec_(spec), diags_(diags) {}

    bool analyze(); // returns true if no errors

private:
    const Spec& spec_;
    DiagnosticBag& diags_;

    // Symbol tables
    std::unordered_map<std::string, const RobotDecl*> robot_table_;
    std::unordered_map<std::string, const JointDecl*> joint_table_;
    std::unordered_map<std::string, const LinkDecl*> link_table_;
    std::unordered_map<std::string, const TaskDecl*> task_table_;

    void build_tables();
    void check_robot(const RobotDecl& robot);
    void check_task(const TaskDecl& task);
    void check_clearance(const ClearanceDecl& clearance);
    void validate_parent_tree(const RobotDecl& robot);
};

} // namespace cudro