#include <cudro/lower.hpp>

#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

using namespace cudro::dag_builders;

namespace cudro {

// ============================================================================
// Helper: Build Vec3 from AST node
// ============================================================================

int LowerContext::build_vec3(const Vec3& v) {
    int x = dag.add_constant(v.x);
    int y = dag.add_constant(v.y);
    int z = dag.add_constant(v.z);
    return dag.add_ternary(NodeKind::Vec3, x, y, z);
}

// ============================================================================
// FK Chain Building
// ============================================================================

void LowerContext::build_all_link_transforms() {
    std::unordered_set<std::string> processed;
    
    while (processed.size() < robot->links.size()) {
        bool progress = false;
        for (const auto& link : robot->links) {
            if (processed.count(link->name)) continue;
            
            bool ready = true;
            if (link->parent) {
                const std::string& parent_name = *link->parent;
                if (parent_name != "world" && !processed.count(parent_name)) {
                    ready = false;
                }
            }
            
            if (ready) {
                int transform = build_link_transform(link->name);
                link_transforms_[link->name] = transform;
                processed.insert(link->name);
                progress = true;
            }
        }
        if (!progress) break;
    }
}

int LowerContext::build_link_transform(const std::string& link_name) {
    auto it = link_idx_.find(link_name);
    if (it == link_idx_.end()) return -1;
    
    const LinkDecl* link = robot->links[it->second].get();
    
    int parent_transform = -1;
    if (link->parent) {
        const std::string& parent_name = *link->parent;
        if (parent_name == "world") {
            parent_transform = mat4_identity(dag);
        } else {
            auto it = link_transforms_.find(parent_name);
            if (it != link_transforms_.end()) {
                parent_transform = it->second;
            }
        }
    }
    
    if (link->joint_ref) {
        int joint_idx = joint_index_in_robot_[*link->joint_ref];
        int q_input = joint_idx;
        int joint_transform = build_joint_transform(*link->joint_ref, q_input);
        
        if (parent_transform >= 0) {
            return mat4_mul(dag, parent_transform, joint_transform);
        }
        return joint_transform;
    }
    
    return parent_transform;
}

// ============================================================================
// Joint Transform Construction
// ============================================================================

int LowerContext::build_joint_transform(const std::string& joint_name, int q_input) {
    auto it = joint_index_in_robot_.find(joint_name);
    if (it == joint_index_in_robot_.end()) return -1;
    
    const JointDecl* joint = robot->joints[it->second].get();
    
    // Build rotation matrix based on axis
    int rot_matrix;
    if (joint->axis.x == 1.0 && joint->axis.y == 0.0 && joint->axis.z == 0.0) {
        rot_matrix = mat4_rot_x(dag, q_input);
    } else if (joint->axis.x == 0.0 && joint->axis.y == 1.0 && joint->axis.z == 0.0) {
        rot_matrix = mat4_rot_y(dag, q_input);
    } else if (joint->axis.x == 0.0 && joint->axis.y == 0.0 && joint->axis.z == 1.0) {
        rot_matrix = mat4_rot_z(dag, q_input);
    } else {
        // Arbitrary axis - use RotAxis
        int axis_x = dag.add_constant(joint->axis.x);
        int axis_y = dag.add_constant(joint->axis.y);
        int axis_z = dag.add_constant(joint->axis.z);
        rot_matrix = dag.add_quaternary(NodeKind::RotAxis, axis_x, axis_y, axis_z, q_input);
    }
    
    // Translation matrix from origin
    int trans_matrix = mat4_translate(dag, 
        dag.add_constant(joint->origin.x),
        dag.add_constant(joint->origin.y),
        dag.add_constant(joint->origin.z));
    
    // Joint transform = Rot * Translate
    return mat4_mul(dag, rot_matrix, trans_matrix);
}

// ============================================================================
// Constraint Lowering
// ============================================================================

int LowerContext::lower_plane_constraint(const std::string& task_link, const PlaneConstraint& plane) {
    // g(q) = dot(normal_world, p_world) - offset
    // p_world = link_transform * point_on_link
    int link_transform = link_transforms_.at(task_link);
    
    // point_on_link in local frame -> homogeneous vec4 (w=1)
    int point_local = dag.add_quaternary(NodeKind::Vec4, 
        dag.add_constant(plane.point_on_link.x),
        dag.add_constant(plane.point_on_link.y),
        dag.add_constant(plane.point_on_link.z),
        dag.add_constant(1.0));
    
    // Transform point: p_world = link_transform * point_local
    int p_world = dag.add_binary(NodeKind::MatVecMul, link_transform, point_local);
    
    // Normal vector
    int normal = dag.add_ternary(NodeKind::Vec3,
        dag.add_constant(plane.normal.x),
        dag.add_constant(plane.normal.y),
        dag.add_constant(plane.normal.z));
    
    // dot_prod = dot(normal, p_world)
    int dot_prod = dag.add_binary(NodeKind::Dot, normal, p_world);
    
    // g = dot_prod - offset
    int offset = dag.add_constant(plane.offset);
    return dag.add_binary(NodeKind::Sub, dot_prod, offset);
}

std::vector<int> lower(const Spec& spec, ExprDAG& dag) {
    LowerContext ctx{dag, spec};
    std::vector<int> constraint_outputs;
    
    // 1. Build FK transforms for all links
    ctx.build_all_link_transforms();
    
    // 2. Desugar each task constraint
    for (const auto& task : spec.tasks) {
        for (const auto& plane : task->planes) {
            const std::string& target_link = plane->link.empty() ? task->link : plane->link;
            int constraint_node = ctx.lower_plane_constraint(target_link, *plane);
            constraint_outputs.push_back(constraint_node);
        }
    }
    
    // 3. Constant folding
    dag.fold_constants();
    
    return constraint_outputs;
}

} // namespace cudro