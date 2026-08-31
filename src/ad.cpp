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

ADResult differentiate(const ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs) {
    ADResult result;
    result.num_inputs = num_inputs;
    result.num_constraints = static_cast<int>(constraint_outputs.size());
    result.dag = dag;  // copy the original DAG

    int num_constraints = static_cast<int>(constraint_outputs.size());
    
    // Jacobian: [constraint][input] = derivative value
    result.jacobians.resize(num_constraints);
    for (auto& row : result.jacobians) {
        row.assign(num_inputs, 0.0);
    }

    // Evaluate at nominal q = 0
    std::vector<double> q_nominal(num_inputs, 0.0);
    std::vector<double> constraint_vals_nominal(constraint_outputs.size());
    
    auto nominal_values = evaluate_dag(dag, num_inputs, q_nominal.data());
    for (int c = 0; c < static_cast<int>(constraint_outputs.size()); ++c) {
        constraint_vals_nominal[c] = nominal_values[constraint_outputs[c]];
    }
    
    // Finite differences for each input
    const double h = 1e-6;
    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
        std::vector<double> q_perturbed = q_nominal;
        q_perturbed[input_idx] += 1e-6;
        
        auto perturbed_values = evaluate_dag(dag, num_inputs, q_perturbed.data());
        
        for (int c = 0; c < static_cast<int>(constraint_outputs.size()); ++c) {
            double val_perturbed = perturbed_values[constraint_outputs[c]];
            double deriv = (val_perturbed - constraint_vals_nominal[c]) / 1e-6;
            result.jacobians[c][input_idx] = deriv;
        }
    }
    
    result.num_inputs = num_inputs;
    result.num_constraints = static_cast<int>(constraint_outputs.size());
    
    return result;
}

} // namespace cudro