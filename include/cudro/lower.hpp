#pragma once

#include <cudro/ast.hpp>
#include <cudro/dag.hpp>

#include <unordered_map>
#include <vector>
#include <string>
#include <optional>

namespace cudro {

struct LowerResult {
    ExprDAG dag;
    std::vector<int> constraint_outputs; // DAG indices for each constraint g(q)
    std::vector<std::vector<int>> constraint_jacobians; // Will be filled by AD pass (M4)
    int num_inputs; // number of joint DoFs
};

struct LowerContext {
    ExprDAG& dag;
    const Spec& spec;
    const RobotDecl* robot; // pointer since spec stores unique_ptr
    
    std::unordered_map<std::string, int> joint_index_; // joint name -> q index
    std::unordered_map<std::string, int> link_transforms_; // link name -> DAG node index (4x4 matrix)
    std::unordered_map<std::string, int> joint_index_in_robot_; // joint name -> index in robot
    std::unordered_map<std::string, int> link_idx_; // link name -> index in robot
    
    LowerContext(ExprDAG& dag, const Spec& spec) : dag(dag), spec(spec), robot(spec.robots[0].get()) {
        // Build joint index mapping
        for (size_t i = 0; i < robot->joints.size(); ++i) {
            joint_index_[robot->joints[i]->name] = (int)i;
            joint_index_in_robot_[robot->joints[i]->name] = (int)i;
        }
        // Build link index
        for (size_t i = 0; i < robot->links.size(); ++i) {
            link_idx_[robot->links[i]->name] = (int)i;
        }
        // Pre-create input nodes for all q
        for (size_t i = 0; i < robot->joints.size(); ++i) {
            dag.add_input((int)i, "q" + std::to_string(i));
        }
    }
    
    int total_dofs() const { return (int)robot->joints.size(); }
    
    int build_vec3(const Vec3& v);
    
    void build_all_link_transforms();
    int build_link_transform(const std::string& link_name);
    int build_joint_transform(const std::string& joint_name, int q_input);
    int lower_plane_constraint(const std::string& task_link, const PlaneConstraint& plane);
};

std::vector<int> lower(const Spec& spec, ExprDAG& dag);

} // namespace cudro