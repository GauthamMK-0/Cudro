#include <cudro/ad.hpp>

#include <vector>
#include <cmath>
#include <iostream>

namespace cudro {

// Evaluate the DAG with concrete input values to get node values
// Uses a structured representation for matrices/vectors
struct DAGValue {
    enum class Type { Scalar, Vec3, Vec4, Mat4 } type;
    double scalar = 0.0;
    double vec3[3] = {0, 0, 0};
    double vec4[4] = {0, 0, 0, 0};
    double mat4[16] = {0};  // row-major
};

std::vector<double> evaluate_dag(const ExprDAG& dag, int num_inputs, const double* q) {
    std::vector<DAGValue> values(dag.size());
    for (size_t i = 0; i < dag.size(); ++i) {
        const Node& n = dag[i];
        DAGValue result;
        
        switch (n.kind) {
            case NodeKind::Input: {
                result.type = DAGValue::Type::Scalar;
                result.scalar = q[n.input_index];
                break;
            }
            case NodeKind::Constant: {
                result.type = DAGValue::Type::Scalar;
                result.scalar = n.constant_value;
                break;
            }
            case NodeKind::Add: {
                auto lhs = values[n.operands[0]];
                auto rhs = values[n.operands[1]];
                result.type = lhs.type;
                if (lhs.type == DAGValue::Type::Scalar) {
                    result.scalar = lhs.scalar + rhs.scalar;
                }
                break;
            }
            case NodeKind::Sub: {
                auto lhs = values[n.operands[0]];
                auto rhs = values[n.operands[1]];
                result.type = lhs.type;
                if (lhs.type == DAGValue::Type::Scalar) {
                    result.scalar = lhs.scalar - rhs.scalar;
                }
                break;
            }
            case NodeKind::Mul: {
                auto lhs = values[n.operands[0]];
                auto rhs = values[n.operands[1]];
                result.type = lhs.type;
                if (lhs.type == DAGValue::Type::Scalar) {
                    result.scalar = lhs.scalar * rhs.scalar;
                }
                break;
            }
            case NodeKind::Div: {
                auto lhs = values[n.operands[0]];
                auto rhs = values[n.operands[1]];
                result.type = lhs.type;
                if (lhs.type == DAGValue::Type::Scalar) {
                    result.scalar = lhs.scalar / rhs.scalar;
                }
                break;
            }
            case NodeKind::Neg: {
                auto v = values[n.operands[0]];
                result.type = v.type;
                if (v.type == DAGValue::Type::Scalar) {
                    result.scalar = -v.scalar;
                }
                break;
            }
            case NodeKind::Cos: {
                auto v = values[n.operands[0]];
                result.type = DAGValue::Type::Scalar;
                result.scalar = std::cos(v.scalar);
                break;
            }
            case NodeKind::Sin: {
                auto v = values[n.operands[0]];
                result.type = DAGValue::Type::Scalar;
                result.scalar = std::sin(v.scalar);
                break;
            }
            case NodeKind::Vec3: {
                auto x = values[n.operands[0]];
                auto y = values[n.operands[1]];
                auto z = values[n.operands[2]];
                result.type = DAGValue::Type::Vec3;
                result.vec3[0] = x.scalar;
                result.vec3[1] = y.scalar;
                result.vec3[2] = z.scalar;
                break;
            }
            case NodeKind::Vec4: {
                auto x = values[n.operands[0]];
                auto y = values[n.operands[1]];
                auto z = values[n.operands[2]];
                auto w = values[n.operands[3]];
                result.type = DAGValue::Type::Vec4;
                result.vec4[0] = x.scalar;
                result.vec4[1] = y.scalar;
                result.vec4[2] = z.scalar;
                result.vec4[3] = w.scalar;
                break;
            }
            case NodeKind::Mat4: {
                result.type = DAGValue::Type::Mat4;
                for (int i = 0; i < 16; ++i) {
                    result.mat4[i] = values[n.operands[i]].scalar;
                }
                break;
            }
            case NodeKind::MatMul: {
                auto lhs = values[n.operands[0]];
                auto rhs = values[n.operands[1]];
                result.type = DAGValue::Type::Mat4;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        double sum = 0;
                        for (int k = 0; k < 4; ++k) {
                            sum += lhs.mat4[i*4 + k] * rhs.mat4[k*4 + j];
                        }
                        result.mat4[i*4 + j] = sum;
                    }
                }
                break;
            }
            case NodeKind::MatVecMul: {
                auto mat = values[n.operands[0]];
                auto vec = values[n.operands[1]];
                result.type = DAGValue::Type::Vec4;
                for (int i = 0; i < 4; ++i) {
                    double sum = 0;
                    for (int j = 0; j < 4; ++j) {
                        sum += mat.mat4[i*4 + j] * vec.vec4[j];
                    }
                    result.vec4[i] = sum;
                }
                break;
            }
            case NodeKind::Mat4Identity: {
                result.type = DAGValue::Type::Mat4;
                for (int i = 0; i < 16; ++i) {
                    result.mat4[i] = (i % 5 == 0) ? 1.0 : 0.0;
                }
                break;
            }
            case NodeKind::RotX: {
                auto lhs = values[n.operands[0]];
                double angle = lhs.scalar;
                double c = std::cos(angle);
                double s = std::sin(angle);
                result.type = DAGValue::Type::Mat4;
                result.mat4[0] = 1; result.mat4[1] = 0; result.mat4[2] = 0; result.mat4[3] = 0;
                result.mat4[4] = 0; result.mat4[5] = c; result.mat4[6] = -s; result.mat4[7] = 0;
                result.mat4[8] = 0; result.mat4[9] = s; result.mat4[10] = c; result.mat4[11] = 0;
                result.mat4[12] = 0; result.mat4[13] = 0; result.mat4[14] = 0; result.mat4[15] = 1;
                break;
            }
            case NodeKind::RotY: {
                auto lhs = values[n.operands[0]];
                double angle = lhs.scalar;
                double c = std::cos(angle);
                double s = std::sin(angle);
                result.type = DAGValue::Type::Mat4;
                result.mat4[0] = c; result.mat4[1] = 0; result.mat4[2] = s; result.mat4[3] = 0;
                result.mat4[4] = 0; result.mat4[5] = 1; result.mat4[6] = 0; result.mat4[7] = 0;
                result.mat4[8] = -s; result.mat4[9] = 0; result.mat4[10] = c; result.mat4[11] = 0;
                result.mat4[12] = 0; result.mat4[13] = 0; result.mat4[14] = 0; result.mat4[15] = 1;
                break;
            }
            case NodeKind::RotZ: {
                auto lhs = values[n.operands[0]];
                double angle = lhs.scalar;
                double c = std::cos(angle);
                double s = std::sin(angle);
                result.type = DAGValue::Type::Mat4;
                result.mat4[0] = c; result.mat4[1] = -s; result.mat4[2] = 0; result.mat4[3] = 0;
                result.mat4[4] = s; result.mat4[5] = c; result.mat4[6] = 0; result.mat4[7] = 0;
                result.mat4[8] = 0; result.mat4[9] = 0; result.mat4[10] = 1; result.mat4[11] = 0;
                result.mat4[12] = 0; result.mat4[13] = 0; result.mat4[14] = 0; result.mat4[15] = 1;
                break;
            }
            case NodeKind::RotAxis: {
                // RotAxis operands: axis_x, axis_y, axis_z, angle
                auto ax_val = values[n.operands[0]].scalar;
                auto ay_val = values[n.operands[1]].scalar;
                auto az_val = values[n.operands[2]].scalar;
                auto angle = values[n.operands[3]].scalar;
                double len = std::sqrt(ax_val * ax_val + ay_val * ay_val + az_val * az_val);
                if (len > 1e-12) { ax_val /= len; ay_val /= len; az_val /= len; }
                double c = std::cos(angle);
                double s = std::sin(angle);
                double c1 = 1.0 - c;
                result.type = DAGValue::Type::Mat4;
                result.mat4[0] = c + ax_val * ax_val * c1;
                result.mat4[1] = ax_val * ay_val * c1 - az_val * s;
                result.mat4[2] = ax_val * az_val * c1 + ay_val * s;
                result.mat4[3] = 0;

                result.mat4[4] = ay_val * ax_val * c1 + az_val * s;
                result.mat4[5] = c + ay_val * ay_val * c1;
                result.mat4[6] = ay_val * az_val * c1 - ax_val * s;
                result.mat4[7] = 0;

                result.mat4[8] = az_val * ax_val * c1 - ay_val * s;
                result.mat4[9] = az_val * ay_val * c1 + ax_val * s;
                result.mat4[10] = c + az_val * az_val * c1;
                result.mat4[11] = 0;

                result.mat4[12] = 0; result.mat4[13] = 0; result.mat4[14] = 0; result.mat4[15] = 1;
                break;
            }
            case NodeKind::Translate: {
                double tx = (n.operands.size() == 3) ? values[n.operands[0]].scalar : values[n.operands[0]].vec3[0];
                double ty = (n.operands.size() == 3) ? values[n.operands[1]].scalar : values[n.operands[0]].vec3[1];
                double tz = (n.operands.size() == 3) ? values[n.operands[2]].scalar : values[n.operands[0]].vec3[2];
                result.type = DAGValue::Type::Mat4;
                result.mat4[0] = 1; result.mat4[1] = 0; result.mat4[2] = 0; result.mat4[3] = tx;
                result.mat4[4] = 0; result.mat4[5] = 1; result.mat4[6] = 0; result.mat4[7] = ty;
                result.mat4[8] = 0; result.mat4[9] = 0; result.mat4[10] = 1; result.mat4[11] = tz;
                result.mat4[12] = 0; result.mat4[13] = 0; result.mat4[14] = 0; result.mat4[15] = 1;
                break;
            }
            case NodeKind::Dot: {
                auto lhs = values[n.operands[0]];
                auto rhs = values[n.operands[1]];
                const double* a = (lhs.type == DAGValue::Type::Vec4) ? lhs.vec4 : lhs.vec3;
                const double* b = (rhs.type == DAGValue::Type::Vec4) ? rhs.vec4 : rhs.vec3;
                result.type = DAGValue::Type::Scalar;
                result.scalar = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
                break;
            }
            case NodeKind::SubVec3: {
                auto v = values[n.operands[0]];
                result.type = DAGValue::Type::Vec3;
                result.vec3[0] = v.vec4[0];
                result.vec3[1] = v.vec4[1];
                result.vec3[2] = v.vec4[2];
                break;
            }
            case NodeKind::SubMat3: {
                break;
            }
            default:
                break;
        }
        values[i] = result;
    }
    // Extract constraint values as scalars (first component for vectors, etc.)
    std::vector<double> scalar_values(dag.size());
    for (size_t i = 0; i < dag.size(); ++i) {
        const auto& v = values[i];
        if (v.type == DAGValue::Type::Scalar) {
            scalar_values[i] = v.scalar;
        } else if (v.type == DAGValue::Type::Vec3) {
            scalar_values[i] = v.vec3[0];  // x component
        } else if (v.type == DAGValue::Type::Vec4) {
            scalar_values[i] = v.vec4[0];
        } else if (v.type == DAGValue::Type::Mat4) {
            scalar_values[i] = v.mat4[0];  // top-left element
        }
    }
    return scalar_values;
}

static int make_zero_mat4(ExprDAG& dag) {
    int zero = dag.add_constant(0.0);
    return dag.add_variadic(NodeKind::Mat4, std::vector<int>(16, zero));
}

static int make_zero_vec4(ExprDAG& dag) {
    int zero = dag.add_constant(0.0);
    return dag.add_quaternary(NodeKind::Vec4, zero, zero, zero, zero);
}

static int make_zero_vec3(ExprDAG& dag) {
    int zero = dag.add_constant(0.0);
    return dag.add_ternary(NodeKind::Vec3, zero, zero, zero);
}

static bool is_constant_zero(const ExprDAG& dag, int node_idx) {
    if (node_idx < 0 || node_idx >= dag.size()) return false;
    const Node& n = dag[node_idx];
    return (n.kind == NodeKind::Constant && n.constant_value == 0.0);
}

static bool is_constant_one(const ExprDAG& dag, int node_idx) {
    if (node_idx < 0 || node_idx >= dag.size()) return false;
    const Node& n = dag[node_idx];
    return (n.kind == NodeKind::Constant && n.constant_value == 1.0);
}

static std::vector<bool> compute_activity(const ExprDAG& dag, int input_idx) {
    std::vector<bool> active(dag.size(), false);
    for (int i = 0; i < dag.size(); ++i) {
        const Node& n = dag[i];
        if (n.kind == NodeKind::Input && n.input_index == input_idx) {
            active[i] = true;
        } else {
            for (int op : n.operands) {
                if (op >= 0 && op < static_cast<int>(active.size()) && active[op]) {
                    active[i] = true;
                    break;
                }
            }
        }
    }
    return active;
}

static int diff_dag_node(ExprDAG& dag, int node_idx, int input_idx,
                         const std::vector<bool>& active,
                         std::unordered_map<int, int>& cache) {
    if (node_idx < 0 || node_idx >= static_cast<int>(active.size()) || !active[node_idx]) {
        const Node& n = dag[node_idx];
        if (n.kind == NodeKind::Mat4 || n.kind == NodeKind::MatMul || n.kind == NodeKind::Mat4Identity) {
            return make_zero_mat4(dag);
        }
        if (n.kind == NodeKind::Vec4 || n.kind == NodeKind::MatVecMul) {
            return make_zero_vec4(dag);
        }
        if (n.kind == NodeKind::Vec3 || n.kind == NodeKind::SubVec3) {
            return make_zero_vec3(dag);
        }
        return dag.add_constant(0.0);
    }

    auto it = cache.find(node_idx);
    if (it != cache.end()) {
        return it->second;
    }

    const Node n = dag[node_idx]; // copy by value because dag may reallocate
    int result = -1;

    switch (n.kind) {
        case NodeKind::Input: {
            result = (n.input_index == input_idx) ? dag.add_constant(1.0) : dag.add_constant(0.0);
            break;
        }
        case NodeKind::Constant: {
            result = dag.add_constant(0.0);
            break;
        }
        case NodeKind::Neg: {
            int d_u = diff_dag_node(dag, n.operands[0], input_idx, active, cache);
            result = is_constant_zero(dag, d_u) ? d_u : dag.add_unary(NodeKind::Neg, d_u);
            break;
        }
        case NodeKind::Cos: {
            int u = n.operands[0];
            int d_u = diff_dag_node(dag, u, input_idx, active, cache);
            int sin_u = dag.add_unary(NodeKind::Sin, u);
            int neg_sin_u = dag.add_unary(NodeKind::Neg, sin_u);
            result = is_constant_one(dag, d_u) ? neg_sin_u : dag.add_binary(NodeKind::Mul, neg_sin_u, d_u);
            break;
        }
        case NodeKind::Sin: {
            int u = n.operands[0];
            int d_u = diff_dag_node(dag, u, input_idx, active, cache);
            int cos_u = dag.add_unary(NodeKind::Cos, u);
            result = is_constant_one(dag, d_u) ? cos_u : dag.add_binary(NodeKind::Mul, cos_u, d_u);
            break;
        }
        case NodeKind::Add: {
            bool a0 = active[n.operands[0]];
            bool a1 = active[n.operands[1]];
            if (a0 && a1) {
                int d0 = diff_dag_node(dag, n.operands[0], input_idx, active, cache);
                int d1 = diff_dag_node(dag, n.operands[1], input_idx, active, cache);
                result = dag.add_binary(NodeKind::Add, d0, d1);
            } else if (a0) {
                result = diff_dag_node(dag, n.operands[0], input_idx, active, cache);
            } else if (a1) {
                result = diff_dag_node(dag, n.operands[1], input_idx, active, cache);
            } else {
                result = dag.add_constant(0.0);
            }
            break;
        }
        case NodeKind::Sub: {
            bool a0 = active[n.operands[0]];
            bool a1 = active[n.operands[1]];
            if (a0 && a1) {
                int d0 = diff_dag_node(dag, n.operands[0], input_idx, active, cache);
                int d1 = diff_dag_node(dag, n.operands[1], input_idx, active, cache);
                result = dag.add_binary(NodeKind::Sub, d0, d1);
            } else if (a0) {
                result = diff_dag_node(dag, n.operands[0], input_idx, active, cache);
            } else if (a1) {
                int d1 = diff_dag_node(dag, n.operands[1], input_idx, active, cache);
                result = dag.add_unary(NodeKind::Neg, d1);
            } else {
                result = dag.add_constant(0.0);
            }
            break;
        }
        case NodeKind::Mul: {
            bool a0 = active[n.operands[0]];
            bool a1 = active[n.operands[1]];
            int u = n.operands[0], v = n.operands[1];
            if (a0 && a1) {
                int du = diff_dag_node(dag, u, input_idx, active, cache);
                int dv = diff_dag_node(dag, v, input_idx, active, cache);
                int t1 = is_constant_one(dag, du) ? v : dag.add_binary(NodeKind::Mul, du, v);
                int t2 = is_constant_one(dag, dv) ? u : dag.add_binary(NodeKind::Mul, u, dv);
                result = dag.add_binary(NodeKind::Add, t1, t2);
            } else if (a0) {
                int du = diff_dag_node(dag, u, input_idx, active, cache);
                result = is_constant_one(dag, du) ? v : dag.add_binary(NodeKind::Mul, du, v);
            } else if (a1) {
                int dv = diff_dag_node(dag, v, input_idx, active, cache);
                result = is_constant_one(dag, dv) ? u : dag.add_binary(NodeKind::Mul, u, dv);
            } else {
                result = dag.add_constant(0.0);
            }
            break;
        }
        case NodeKind::Mat4: {
            std::vector<int> d_ops;
            d_ops.reserve(16);
            for (int k = 0; k < 16; ++k) {
                d_ops.push_back(diff_dag_node(dag, n.operands[k], input_idx, active, cache));
            }
            result = dag.add_variadic(NodeKind::Mat4, std::move(d_ops));
            break;
        }
        case NodeKind::Translate: {
            int zero = dag.add_constant(0.0);
            int dtx = diff_dag_node(dag, n.operands[0], input_idx, active, cache);
            int dty = diff_dag_node(dag, n.operands[1], input_idx, active, cache);
            int dtz = diff_dag_node(dag, n.operands[2], input_idx, active, cache);
            result = dag.add_variadic(NodeKind::Mat4, {
                zero, zero, zero, dtx,
                zero, zero, zero, dty,
                zero, zero, zero, dtz,
                zero, zero, zero, zero
            });
            break;
        }
        case NodeKind::RotAxis: {
            // Operands: ax, ay, az, angle
            int ax = n.operands[0];
            int ay = n.operands[1];
            int az = n.operands[2];
            int ang = n.operands[3];
            int d_ang = diff_dag_node(dag, ang, input_idx, active, cache);
            if (is_constant_zero(dag, d_ang)) {
                result = make_zero_mat4(dag);
            } else {
                int s = dag.add_unary(NodeKind::Sin, ang);
                int c = dag.add_unary(NodeKind::Cos, ang);
                int neg_s = dag.add_unary(NodeKind::Neg, s);
                // d(c) = -s * d_ang, d(s) = c * d_ang, d(c1) = s * d_ang
                int dc = is_constant_one(dag, d_ang) ? neg_s : dag.add_binary(NodeKind::Mul, neg_s, d_ang);
                int ds = is_constant_one(dag, d_ang) ? c : dag.add_binary(NodeKind::Mul, c, d_ang);
                int dc1 = is_constant_one(dag, d_ang) ? s : dag.add_binary(NodeKind::Mul, s, d_ang);

                auto mul = [&](int x, int y) { return dag.add_binary(NodeKind::Mul, x, y); };
                auto add = [&](int x, int y) { return dag.add_binary(NodeKind::Add, x, y); };
                auto sub = [&](int x, int y) { return dag.add_binary(NodeKind::Sub, x, y); };
                int zero = dag.add_constant(0.0);

                // m00 = dc + ax*ax*dc1
                int m00 = add(dc, mul(mul(ax, ax), dc1));
                // m01 = ax*ay*dc1 - az*ds
                int m01 = sub(mul(mul(ax, ay), dc1), mul(az, ds));
                // m02 = ax*az*dc1 + ay*ds
                int m02 = add(mul(mul(ax, az), dc1), mul(ay, ds));

                // m10 = ay*ax*dc1 + az*ds
                int m10 = add(mul(mul(ay, ax), dc1), mul(az, ds));
                // m11 = dc + ay*ay*dc1
                int m11 = add(dc, mul(mul(ay, ay), dc1));
                // m12 = ay*az*dc1 - ax*ds
                int m12 = sub(mul(mul(ay, az), dc1), mul(ax, ds));

                // m20 = az*ax*dc1 - ay*ds
                int m20 = sub(mul(mul(az, ax), dc1), mul(ay, ds));
                // m21 = az*ay*dc1 + ax*ds
                int m21 = add(mul(mul(az, ay), dc1), mul(ax, ds));
                // m22 = dc + az*az*dc1
                int m22 = add(dc, mul(mul(az, az), dc1));

                result = dag.add_variadic(NodeKind::Mat4, {
                    m00, m01, m02, zero,
                    m10, m11, m12, zero,
                    m20, m21, m22, zero,
                    zero, zero, zero, zero
                });
            }
            break;
        }
        case NodeKind::MatMul: {
            bool a0 = active[n.operands[0]];
            bool a1 = active[n.operands[1]];
            int a = n.operands[0], b = n.operands[1];
            if (a0 && a1) {
                int da = diff_dag_node(dag, a, input_idx, active, cache);
                int db = diff_dag_node(dag, b, input_idx, active, cache);
                int t1 = dag.add_binary(NodeKind::MatMul, da, b);
                int t2 = dag.add_binary(NodeKind::MatMul, a, db);
                result = dag.add_binary(NodeKind::Add, t1, t2);
            } else if (a0) {
                int da = diff_dag_node(dag, a, input_idx, active, cache);
                result = dag.add_binary(NodeKind::MatMul, da, b);
            } else if (a1) {
                int db = diff_dag_node(dag, b, input_idx, active, cache);
                result = dag.add_binary(NodeKind::MatMul, a, db);
            } else {
                result = make_zero_mat4(dag);
            }
            break;
        }
        case NodeKind::MatVecMul: {
            bool a0 = active[n.operands[0]];
            bool a1 = active[n.operands[1]];
            int m = n.operands[0], v = n.operands[1];
            if (a0 && !a1) {
                int dm = diff_dag_node(dag, m, input_idx, active, cache);
                result = dag.add_binary(NodeKind::MatVecMul, dm, v);
            } else if (!a0 && a1) {
                int dv = diff_dag_node(dag, v, input_idx, active, cache);
                result = dag.add_binary(NodeKind::MatVecMul, m, dv);
            } else if (a0 && a1) {
                int dm = diff_dag_node(dag, m, input_idx, active, cache);
                int dv = diff_dag_node(dag, v, input_idx, active, cache);
                int t1 = dag.add_binary(NodeKind::MatVecMul, dm, v);
                int t2 = dag.add_binary(NodeKind::MatVecMul, m, dv);
                result = dag.add_binary(NodeKind::Add, t1, t2);
            } else {
                result = make_zero_vec4(dag);
            }
            break;
        }
        case NodeKind::Dot: {
            bool a0 = active[n.operands[0]];
            bool a1 = active[n.operands[1]];
            int a = n.operands[0], b = n.operands[1];
            if (a0 && !a1) {
                int da = diff_dag_node(dag, a, input_idx, active, cache);
                result = dag.add_binary(NodeKind::Dot, da, b);
            } else if (!a0 && a1) {
                int db = diff_dag_node(dag, b, input_idx, active, cache);
                result = dag.add_binary(NodeKind::Dot, a, db);
            } else if (a0 && a1) {
                int da = diff_dag_node(dag, a, input_idx, active, cache);
                int db = diff_dag_node(dag, b, input_idx, active, cache);
                int t1 = dag.add_binary(NodeKind::Dot, da, b);
                int t2 = dag.add_binary(NodeKind::Dot, a, db);
                result = dag.add_binary(NodeKind::Add, t1, t2);
            } else {
                result = dag.add_constant(0.0);
            }
            break;
        }
        default: {
            result = dag.add_constant(0.0);
            break;
        }
    }

    cache[node_idx] = result;
    return result;
}

std::vector<std::vector<int>> build_analytical_jacobians(ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs) {
    int num_constraints = static_cast<int>(constraint_outputs.size());
    std::vector<std::vector<int>> jacobian_nodes(num_constraints);

    for (int j = 0; j < num_inputs; ++j) {
        auto active = compute_activity(dag, j);
        std::unordered_map<int, int> cache;
        for (int c = 0; c < num_constraints; ++c) {
            if (jacobian_nodes[c].empty()) {
                jacobian_nodes[c].resize(num_inputs);
            }
            int grad_node = diff_dag_node(dag, constraint_outputs[c], j, active, cache);
            jacobian_nodes[c][j] = grad_node;
        }
    }

    return jacobian_nodes;
}

void build_analytical_jacobians(LowerResult& lr) {
    lr.constraint_jacobians = build_analytical_jacobians(lr.dag, lr.constraint_outputs, lr.num_inputs);
}

ADResult differentiate(const ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs) {
    ADResult result;
    result.num_inputs = num_inputs;
    result.num_constraints = static_cast<int>(constraint_outputs.size());
    result.dag = dag;  // copy the original DAG

    // Build exact analytical Jacobian nodes into result.dag
    result.jacobian_nodes = build_analytical_jacobians(result.dag, constraint_outputs, num_inputs);

    // Evaluate analytical Jacobian values at nominal q = 0
    std::vector<double> q_nominal(num_inputs, 0.0);
    auto nominal_values = evaluate_dag(result.dag, num_inputs, q_nominal.data());

    result.jacobians.resize(result.num_constraints);
    for (int c = 0; c < result.num_constraints; ++c) {
        result.jacobians[c].resize(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            int node_idx = result.jacobian_nodes[c][i];
            result.jacobians[c][i] = nominal_values[node_idx];
        }
    }

    return result;
}

} // namespace cudro