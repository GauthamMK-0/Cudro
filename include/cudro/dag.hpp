#pragma once

#include <cudro/token.hpp>

#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <string_view>
#include <iostream>

namespace cudro {

enum class NodeKind {
    // Inputs & Constants
    Input,          // q[i] — joint configuration input
    Constant,       // 0.333, 1.0, etc.

    // Unary
    Neg,            // -x
    Cos,            // cos(x)
    Sin,            // sin(x)

    // Binary Arithmetic
    Add, Sub, Mul, Div,

    // Vector constructors
    Vec3,           // Construct vec3 from 3 scalars
    Vec4,           // Construct vec4 from 4 scalars

    // Matrix operations
    Mat4,           // 4x4 matrix from 16 scalars (row-major)
    MatMul,         // 4x4 * 4x4
    MatVecMul,      // 4x4 * vec4 -> vec4
    Mat4Identity,   // Identity matrix

    // Robotics-specific
    RotX,           // Rotation around X axis (angle)
    RotY,           // Rotation around Y axis (angle)
    RotZ,           // Rotation around Z axis (angle)
    Translate,      // Translation matrix from vec3
    RotAxis,        // Rotation around arbitrary axis (axis_vec3, angle)

    // Vector ops
    Dot,            // Dot product of two vec3
    Norm,           // Euclidean norm of vec3
    SubVec3,        // Extract vec3 from vec4 (drop w)
    SubMat3,        // Extract 3x3 upper-left from 4x4

    // Collision / Distance
    SphereDist,     // Distance between two spheres

    // Control Flow (for projection iteration)
    // (future: Phi nodes for SSA form if needed)
};

struct Node {
    NodeKind kind;
    std::vector<int> operands;  // Indices of child nodes in the DAG
    double constant_value = 0;  // For Constant nodes
    int input_index = -1;       // For Input nodes: which q[i]
    std::string debug_name;     // For dumping (e.g., "q3", "Rz_j2")
};

class ExprDAG {
public:
    int add_node(NodeKind kind, std::vector<int> operands = {});
    int add_constant(double v);
    int add_input(int index, std::string_view name = "");
    int add_unary(NodeKind kind, int operand);
    int add_binary(NodeKind kind, int lhs, int rhs);
    int add_ternary(NodeKind kind, int a, int b, int c);
    int add_quaternary(NodeKind kind, int a, int b, int c, int d);
    int add_variadic(NodeKind kind, std::vector<int> operands);

    // Access
    const std::vector<Node>& nodes() const { return nodes_; }
    Node& operator[](int idx) { return nodes_[idx]; }
    const Node& operator[](int idx) const { return nodes_[idx]; }
    int size() const { return (int)nodes_.size(); }

    int num_inputs() const {
        int count = 0;
        for (const auto& n : nodes_) {
            if (n.kind == NodeKind::Input) count++;
        }
        return count;
    }

    void dump(std::ostream&) const; // --dump-dag output

    // Constant folding
    int fold_constants();

private:
    std::vector<Node> nodes_;
};

// Convenience builders
namespace dag_builders {
    int mat4_identity(ExprDAG& dag);
    int mat4_translate(ExprDAG& dag, int tx, int ty, int tz);
    int mat4_rot_x(ExprDAG& dag, int angle);
    int mat4_rot_y(ExprDAG& dag, int angle);
    int mat4_rot_z(ExprDAG& dag, int angle);
    int mat4_rot_axis(ExprDAG& dag, int axis_x, int axis_y, int axis_z, int angle);
    int mat4_translate_vec3(ExprDAG& dag, int vec3);
    int mat4_mul(ExprDAG& dag, int lhs, int rhs);
    int mat4_vec4_mul(ExprDAG& dag, int mat, int vec4);
    int vec4_from_vec3(ExprDAG& dag, int vec3, int w);
    int vec4_from_scalars(ExprDAG& dag, int x, int y, int z, int w);
    int vec3_dot(ExprDAG& dag, int a, int b);
    int vec3_sub(ExprDAG& dag, int a, int b);
} // namespace dag_builders

} // namespace cudro