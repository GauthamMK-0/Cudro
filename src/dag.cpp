#include <cudro/dag.hpp>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

namespace cudro {

int ExprDAG::add_node(NodeKind kind, std::vector<int> operands) {
    Node n;
    n.kind = kind;
    n.operands = std::move(operands);
    nodes_.push_back(n);
    return (int)nodes_.size() - 1;
}

int ExprDAG::add_constant(double v) {
    Node n;
    n.kind = NodeKind::Constant;
    n.constant_value = v;
    nodes_.push_back(n);
    return (int)nodes_.size() - 1;
}

int ExprDAG::add_input(int index, std::string_view name) {
    Node n;
    n.kind = NodeKind::Input;
    n.input_index = index;
    if (!name.empty()) n.debug_name = std::string(name);
    nodes_.push_back(n);
    return (int)nodes_.size() - 1;
}

int ExprDAG::add_unary(NodeKind kind, int operand) {
    return add_node(kind, std::vector<int>{operand});
}

int ExprDAG::add_binary(NodeKind kind, int lhs, int rhs) {
    return add_node(kind, std::vector<int>{lhs, rhs});
}

int ExprDAG::add_ternary(NodeKind kind, int a, int b, int c) {
    return add_node(kind, std::vector<int>{a, b, c});
}

int ExprDAG::add_quaternary(NodeKind kind, int a, int b, int c, int d) {
    return add_node(kind, std::vector<int>{a, b, c, d});
}

int ExprDAG::add_variadic(NodeKind kind, std::vector<int> operands) {
    return add_node(kind, std::move(operands));
}

int ExprDAG::fold_constants() {
    int folded = 0;
    for (int i = 0; i < size(); ++i) {
        Node& n = nodes_[i];
        // Only fold binary ops with constant children
        if ((n.kind == NodeKind::Add || n.kind == NodeKind::Sub ||
             n.kind == NodeKind::Mul || n.kind == NodeKind::Div) &&
            n.operands.size() == 2) {
            int lhs = n.operands[0];
            int rhs = n.operands[1];
            if (nodes_[lhs].kind == NodeKind::Constant &&
                nodes_[rhs].kind == NodeKind::Constant) {
                double a = nodes_[lhs].constant_value;
                double b = nodes_[rhs].constant_value;
                double result = 0;
                switch (n.kind) {
                    case NodeKind::Add: result = a + b; break;
                    case NodeKind::Sub: result = a - b; break;
                    case NodeKind::Mul: result = a * b; break;
                    case NodeKind::Div: result = a / b; break;
                }
                n.kind = NodeKind::Constant;
                n.constant_value = result;
                n.operands.clear();
                folded++;
            }
        }
    }
    return folded;
}

void ExprDAG::dump(std::ostream& os) const {
    static const char* kind_names[] = {
        "Input", "Constant", "Neg", "Cos", "Sin", "Add", "Sub", "Mul", "Div",
        "Vec3", "Vec4", "Mat4", "MatMul", "MatVecMul", "Mat4Identity",
        "RotX", "RotY", "RotZ", "Translate", "RotAxis",
        "Dot", "SubVec3", "SubMat3", "SphereDist"
    };

    std::cerr << "DEBUG dump: DAG size = " << size() << std::endl;
    os << "Expression DAG (" << size() << " nodes):\n";
    for (int i = 0; i < size(); ++i) {
        const Node& n = nodes_[i];
        os << std::setw(4) << i << ": " << kind_names[(int)n.kind];
        if (!n.debug_name.empty()) os << "  // " << n.debug_name;
        if (n.kind == NodeKind::Constant) {
            os << " = " << n.constant_value;
        } else if (n.kind == NodeKind::Input) {
            os << " q[" << n.input_index << "]";
        }
        if (!n.operands.empty()) {
            os << "  operands: [";
            for (size_t j = 0; j < n.operands.size(); ++j) {
                if (j) os << ", ";
                os << n.operands[j];
            }
            os << "]";
        }
        os << "\n";
    }
}

} // namespace cudro

// ============================================================================
// Builder functions
// ============================================================================

namespace cudro::dag_builders {

int mat4_identity(ExprDAG& dag) {
    return dag.add_variadic(NodeKind::Mat4, {
        dag.add_constant(1.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(1.0), dag.add_constant(0.0), dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0), dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0)
    });
}

int mat4_translate(ExprDAG& dag, int tx, int ty, int tz) {
    return dag.add_variadic(NodeKind::Mat4, {
        dag.add_constant(1.0), dag.add_constant(0.0), dag.add_constant(0.0), tx,
        dag.add_constant(0.0), dag.add_constant(1.0), dag.add_constant(0.0), ty,
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0), tz,
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0)
    });
}

int mat4_rot_x(ExprDAG& dag, int angle) {
    int cos_t = dag.add_unary(NodeKind::Cos, angle);
    int sin_t = dag.add_unary(NodeKind::Sin, angle);
    int neg_sin = dag.add_unary(NodeKind::Neg, dag.add_unary(NodeKind::Sin, angle));
    
    return dag.add_variadic(NodeKind::Mat4, {
        dag.add_constant(1.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0),
        dag.add_constant(0.0), cos_t, dag.add_unary(NodeKind::Neg, sin_t), dag.add_constant(0.0),
        dag.add_constant(0.0), sin_t, cos_t, dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0)
    });
}

int mat4_rot_y(ExprDAG& dag, int angle) {
    int cos_t = dag.add_unary(NodeKind::Cos, angle);
    int sin_t = dag.add_unary(NodeKind::Sin, angle);
    int neg_sin = dag.add_unary(NodeKind::Neg, dag.add_unary(NodeKind::Sin, angle));
    
    return dag.add_variadic(NodeKind::Mat4, {
        cos_t, dag.add_constant(0.0), sin_t, dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(1.0), dag.add_constant(0.0), dag.add_constant(0.0),
        dag.add_unary(NodeKind::Neg, sin_t), dag.add_constant(0.0), cos_t, dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0)
    });
}

int mat4_rot_z(ExprDAG& dag, int angle) {
    int cos_t = dag.add_unary(NodeKind::Cos, angle);
    int sin_t = dag.add_unary(NodeKind::Sin, angle);
    int neg_sin = dag.add_unary(NodeKind::Neg, dag.add_unary(NodeKind::Sin, angle));
    
    return dag.add_variadic(NodeKind::Mat4, {
        cos_t, dag.add_unary(NodeKind::Neg, sin_t), dag.add_constant(0.0), dag.add_constant(0.0),
        sin_t, cos_t, dag.add_constant(0.0), dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0), dag.add_constant(0.0),
        dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(0.0), dag.add_constant(1.0)
    });
}

int mat4_rot_axis(ExprDAG& dag, int axis_x, int axis_y, int axis_z, int angle) {
    // Rodrigues' rotation formula for arbitrary axis
    // R = I + sin*K + (1-cos)*K^2
    // where K is cross-product matrix of axis
    // For simplicity, delegate to codegen which can use Rodrigues formula
    // Here we just create a special RotAxis node
    return dag.add_quaternary(NodeKind::RotAxis, axis_x, axis_y, axis_z, angle);
}

int mat4_translate_vec3(ExprDAG& dag, int vec3) {
    // Translation matrix from vec3
    // [1 0 0 x]
    // [0 1 0 y]
    // [0 0 1 z]
    // [0 0 0 1]
    // Extract x,y,z from vec3 using SubVec3? For now, assume we have x,y,z separately
    // We'll need a SubVec3 operation to extract components
    // For now, placeholder
    return mat4_identity(dag);
}

int mat4_mul(ExprDAG& dag, int lhs, int rhs) {
    return dag.add_binary(NodeKind::MatMul, lhs, rhs);
}

int mat4_vec4_mul(ExprDAG& dag, int mat, int vec4) {
    return dag.add_binary(NodeKind::MatVecMul, mat, vec4);
}

int vec4_from_vec3(ExprDAG& dag, int vec3, int w) {
    // Extract x,y,z from vec3 and append w
    // Need SubVec3 to extract x,y,z
    return dag.add_constant(0.0); // placeholder
}

int vec4_from_scalars(ExprDAG& dag, int x, int y, int z, int w) {
    return dag.add_quaternary(NodeKind::Vec4, x, y, z, w);
}

int vec3_dot(ExprDAG& dag, int a, int b) {
    return dag.add_binary(NodeKind::Dot, a, b);
}

int vec3_sub(ExprDAG& dag, int a, int b) {
    return dag.add_binary(NodeKind::Sub, a, b);
}

} // namespace cudro::dag_builders