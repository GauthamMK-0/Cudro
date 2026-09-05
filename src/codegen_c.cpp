#include <cudro/codegen_c.hpp>

#include <cudro/dag.hpp>
#include <cudro/ad.hpp>
#include <cudro/lower.hpp>

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>

namespace cudro {

// Helper to get a unique variable name for a DAG node (cached to avoid heap churn)
static const std::string& node_var_name(int idx) {
    static std::vector<std::string> cache;
    if (idx >= static_cast<int>(cache.size())) {
        cache.resize(idx + 128);
    }
    if (cache[idx].empty()) {
        cache[idx] = "n" + std::to_string(idx);
    }
    return cache[idx];
}

// Emits the DAG node evaluation logic into a function body
static void emit_dag_body(std::ostringstream& out, const ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs) {
    out << "    // Input copy\n";
    for (int i = 0; i < num_inputs; ++i) {
        out << "    float q" << i << " = q[" << i << "];\n";
    }
    out << "\n";
    
    // Declare node variables
    out << "    // Node declarations\n";
    for (int i = 0; i < dag.size(); ++i) {
        const Node& n = dag[i];
        std::string var = node_var_name(i);
        if (n.kind == NodeKind::Input) {
            out << "    float " << var << " = q" << n.input_index << ";\n";
        } else if (n.kind == NodeKind::Constant) {
            out << "    float " << var << " = " << n.constant_value << "f;\n";
        } else if (n.kind == NodeKind::Mat4 || n.kind == NodeKind::Mat4Identity ||
                   n.kind == NodeKind::RotX || n.kind == NodeKind::RotY || n.kind == NodeKind::RotZ ||
                   n.kind == NodeKind::RotAxis || n.kind == NodeKind::Translate || n.kind == NodeKind::MatMul) {
            out << "    float " << var << "[4][4];\n";
        } else if (n.kind == NodeKind::Vec4 || n.kind == NodeKind::MatVecMul) {
            out << "    float " << var << "[4];\n";
        } else if (n.kind == NodeKind::Vec3 || n.kind == NodeKind::SubVec3) {
            out << "    float " << var << "[3];\n";
        } else {
            out << "    float " << var << ";\n";
        }
    }
    out << "\n";
    
    // Evaluate DAG in topological order
    out << "    // Evaluate DAG\n";
    for (int i = 0; i < dag.size(); ++i) {
        const Node& n = dag[i];
        switch (n.kind) {
            case NodeKind::Input:
            case NodeKind::Constant:
                break;
                
            case NodeKind::Add:
                out << "    " << node_var_name(i) << " = " 
                    << node_var_name(n.operands[0]) << " + " 
                    << node_var_name(n.operands[1]) << ";\n";
                break;
            case NodeKind::Sub:
                out << "    " << node_var_name(i) << " = " 
                    << node_var_name(n.operands[0]) << " - " 
                    << node_var_name(n.operands[1]) << ";\n";
                break;
            case NodeKind::Mul:
                out << "    " << node_var_name(i) << " = " 
                    << node_var_name(n.operands[0]) << " * " 
                    << node_var_name(n.operands[1]) << ";\n";
                break;
            case NodeKind::Div:
                out << "    " << node_var_name(i) << " = " 
                    << node_var_name(n.operands[0]) << " / " 
                    << node_var_name(n.operands[1]) << ";\n";
                break;
            case NodeKind::Neg:
                out << "    " << node_var_name(i) << " = -" 
                    << node_var_name(n.operands[0]) << ";\n";
                break;
            case NodeKind::Cos:
                out << "    " << node_var_name(i) << " = cosf(" 
                    << node_var_name(n.operands[0]) << ");\n";
                break;
            case NodeKind::Sin:
                out << "    " << node_var_name(i) << " = sinf(" 
                    << node_var_name(n.operands[0]) << ");\n";
                break;
            case NodeKind::Vec3:
                out << "    " << node_var_name(i) << "[0] = " << node_var_name(n.operands[0]) << "; "
                    << node_var_name(i) << "[1] = " << node_var_name(n.operands[1]) << "; "
                    << node_var_name(i) << "[2] = " << node_var_name(n.operands[2]) << ";\n";
                break;
            case NodeKind::Vec4:
                out << "    " << node_var_name(i) << "[0] = " << node_var_name(n.operands[0]) << "; "
                    << node_var_name(i) << "[1] = " << node_var_name(n.operands[1]) << "; "
                    << node_var_name(i) << "[2] = " << node_var_name(n.operands[2]) << "; "
                    << node_var_name(i) << "[3] = " << node_var_name(n.operands[3]) << ";\n";
                break;
            case NodeKind::Mat4:
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        out << "    " << node_var_name(i) << "[" << r << "][" << c << "] = " 
                            << node_var_name(n.operands[r*4 + c]) << ";\n";
                    }
                }
                break;
            case NodeKind::Mat4Identity:
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        out << "    " << node_var_name(i) << "[" << r << "][" << c << "] = " 
                            << (r == c ? "1.0f" : "0.0f") << ";\n";
                    }
                }
                break;
            case NodeKind::RotX:
                out << "    {\n"
                    << "        float cx = cosf(" << node_var_name(n.operands[0]) << "); float sx = sinf(" << node_var_name(n.operands[0]) << ");\n"
                    << "        " << node_var_name(i) << "[0][0] = 1.0f; " << node_var_name(i) << "[0][1] = 0.0f; " << node_var_name(i) << "[0][2] = 0.0f; " << node_var_name(i) << "[0][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[1][0] = 0.0f; " << node_var_name(i) << "[1][1] = cx; " << node_var_name(i) << "[1][2] = -sx; " << node_var_name(i) << "[1][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[2][0] = 0.0f; " << node_var_name(i) << "[2][1] = sx; " << node_var_name(i) << "[2][2] = cx; " << node_var_name(i) << "[2][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[3][0] = 0.0f; " << node_var_name(i) << "[3][1] = 0.0f; " << node_var_name(i) << "[3][2] = 0.0f; " << node_var_name(i) << "[3][3] = 1.0f;\n"
                    << "    }\n";
                break;
            case NodeKind::RotY:
                out << "    {\n"
                    << "        float cy = cosf(" << node_var_name(n.operands[0]) << "); float sy = sinf(" << node_var_name(n.operands[0]) << ");\n"
                    << "        " << node_var_name(i) << "[0][0] = cy; " << node_var_name(i) << "[0][1] = 0.0f; " << node_var_name(i) << "[0][2] = sy; " << node_var_name(i) << "[0][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[1][0] = 0.0f; " << node_var_name(i) << "[1][1] = 1.0f; " << node_var_name(i) << "[1][2] = 0.0f; " << node_var_name(i) << "[1][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[2][0] = -sy; " << node_var_name(i) << "[2][1] = 0.0f; " << node_var_name(i) << "[2][2] = cy; " << node_var_name(i) << "[2][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[3][0] = 0.0f; " << node_var_name(i) << "[3][1] = 0.0f; " << node_var_name(i) << "[3][2] = 0.0f; " << node_var_name(i) << "[3][3] = 1.0f;\n"
                    << "    }\n";
                break;
            case NodeKind::RotZ:
                out << "    {\n"
                    << "        float cz = cosf(" << node_var_name(n.operands[0]) << "); float sz = sinf(" << node_var_name(n.operands[0]) << ");\n"
                    << "        " << node_var_name(i) << "[0][0] = cz; " << node_var_name(i) << "[0][1] = -sz; " << node_var_name(i) << "[0][2] = 0.0f; " << node_var_name(i) << "[0][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[1][0] = sz; " << node_var_name(i) << "[1][1] = cz; " << node_var_name(i) << "[1][2] = 0.0f; " << node_var_name(i) << "[1][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[2][0] = 0.0f; " << node_var_name(i) << "[2][1] = 0.0f; " << node_var_name(i) << "[2][2] = 1.0f; " << node_var_name(i) << "[2][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[3][0] = 0.0f; " << node_var_name(i) << "[3][1] = 0.0f; " << node_var_name(i) << "[3][2] = 0.0f; " << node_var_name(i) << "[3][3] = 1.0f;\n"
                    << "    }\n";
                break;
            case NodeKind::RotAxis:
                out << "    {\n"
                    << "        float ax = " << node_var_name(n.operands[0]) << ";\n"
                    << "        float ay = " << node_var_name(n.operands[1]) << ";\n"
                    << "        float az = " << node_var_name(n.operands[2]) << ";\n"
                    << "        float ang = " << node_var_name(n.operands[3]) << ";\n"
                    << "        float len = sqrtf(ax*ax + ay*ay + az*az);\n"
                    << "        if (len > 1e-6f) { ax /= len; ay /= len; az /= len; }\n"
                    << "        float c = cosf(ang), s = sinf(ang), c1 = 1.0f - c;\n"
                    << "        " << node_var_name(i) << "[0][0] = c + ax*ax*c1; " << node_var_name(i) << "[0][1] = ax*ay*c1 - az*s; " << node_var_name(i) << "[0][2] = ax*az*c1 + ay*s; " << node_var_name(i) << "[0][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[1][0] = ay*ax*c1 + az*s; " << node_var_name(i) << "[1][1] = c + ay*ay*c1; " << node_var_name(i) << "[1][2] = ay*az*c1 - ax*s; " << node_var_name(i) << "[1][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[2][0] = az*ax*c1 - ay*s; " << node_var_name(i) << "[2][1] = az*ay*c1 + ax*s; " << node_var_name(i) << "[2][2] = c + az*az*c1; " << node_var_name(i) << "[2][3] = 0.0f;\n"
                    << "        " << node_var_name(i) << "[3][0] = 0.0f; " << node_var_name(i) << "[3][1] = 0.0f; " << node_var_name(i) << "[3][2] = 0.0f; " << node_var_name(i) << "[3][3] = 1.0f;\n"
                    << "    }\n";
                break;
            case NodeKind::Translate:
                out << "    " << node_var_name(i) << "[0][0] = 1.0f; " << node_var_name(i) << "[0][1] = 0.0f; " << node_var_name(i) << "[0][2] = 0.0f; " << node_var_name(i) << "[0][3] = " << node_var_name(n.operands[0]) << ";\n"
                    << "    " << node_var_name(i) << "[1][0] = 0.0f; " << node_var_name(i) << "[1][1] = 1.0f; " << node_var_name(i) << "[1][2] = 0.0f; " << node_var_name(i) << "[1][3] = " << node_var_name(n.operands[1]) << ";\n"
                    << "    " << node_var_name(i) << "[2][0] = 0.0f; " << node_var_name(i) << "[2][1] = 0.0f; " << node_var_name(i) << "[2][2] = 1.0f; " << node_var_name(i) << "[2][3] = " << node_var_name(n.operands[2]) << ";\n"
                    << "    " << node_var_name(i) << "[3][0] = 0.0f; " << node_var_name(i) << "[3][1] = 0.0f; " << node_var_name(i) << "[3][2] = 0.0f; " << node_var_name(i) << "[3][3] = 1.0f;\n";
                break;
            case NodeKind::MatMul: {
                const auto& a = node_var_name(n.operands[0]);
                const auto& b = node_var_name(n.operands[1]);
                const auto& dst = node_var_name(i);
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        out << "    " << dst << "[" << r << "][" << c << "] = "
                            << a << "[" << r << "][0] * " << b << "[0][" << c << "] + "
                            << a << "[" << r << "][1] * " << b << "[1][" << c << "] + "
                            << a << "[" << r << "][2] * " << b << "[2][" << c << "] + "
                            << a << "[" << r << "][3] * " << b << "[3][" << c << "];\n";
                    }
                }
                break;
            }
            case NodeKind::MatVecMul: {
                const auto& m = node_var_name(n.operands[0]);
                const auto& v = node_var_name(n.operands[1]);
                const auto& dst = node_var_name(i);
                for (int r = 0; r < 4; ++r) {
                    out << "    " << dst << "[" << r << "] = "
                        << m << "[" << r << "][0] * " << v << "[0] + "
                        << m << "[" << r << "][1] * " << v << "[1] + "
                        << m << "[" << r << "][2] * " << v << "[2] + "
                        << m << "[" << r << "][3] * " << v << "[3];\n";
                }
                break;
            }
            case NodeKind::SubVec3:
                out << "    " << node_var_name(i) << "[0] = " << node_var_name(n.operands[0]) << "[0];\n"
                    << "    " << node_var_name(i) << "[1] = " << node_var_name(n.operands[0]) << "[1];\n"
                    << "    " << node_var_name(i) << "[2] = " << node_var_name(n.operands[0]) << "[2];\n";
                break;
            case NodeKind::Dot:
                out << "    " << node_var_name(i) << " = " << node_var_name(n.operands[0]) << "[0] * " << node_var_name(n.operands[1]) << "[0] + "
                    << node_var_name(n.operands[0]) << "[1] * " << node_var_name(n.operands[1]) << "[1] + "
                    << node_var_name(n.operands[0]) << "[2] * " << node_var_name(n.operands[1]) << "[2];\n";
                break;
            default:
                break;
        }
    }
    
    // Copy constraint values to out_g
    out << "\n    // Copy constraint values\n";
    for (size_t c = 0; c < constraint_outputs.size(); ++c) {
        out << "    out_g[" << c << "] = " << node_var_name(constraint_outputs[c]) << ";\n";
    }
}

// Emits common helper routines (evaluate_dag, evaluate_jacobian, project_single)
static void emit_shared_routines(std::ostringstream& out, const ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs) {
    int num_constraints = static_cast<int>(constraint_outputs.size());
    
    // Static inline DAG evaluator
    out << "static inline void evaluate_dag(const float* q, float* out_g) {\n";
    emit_dag_body(out, dag, constraint_outputs, num_inputs);
    out << "}\n\n";

    // Static inline numerical Jacobian evaluator
    out << "static inline void evaluate_jacobian(const float* q, const float* g_curr, float* out_J) {\n"
        << "    float q_pert[" << num_inputs << "];\n"
        << "    float g_pert[" << num_constraints << "];\n"
        << "    const float h = 1e-4f;\n"
        << "    const float inv_h = 1.0f / h;\n"
        << "    for (int j = 0; j < " << num_inputs << "; ++j) {\n"
        << "        for (int i = 0; i < " << num_inputs << "; ++i) {\n"
        << "            q_pert[i] = q[i] + (i == j ? h : 0.0f);\n"
        << "        }\n"
        << "        evaluate_dag(q_pert, g_pert);\n"
        << "        for (int c = 0; c < " << num_constraints << "; ++c) {\n"
        << "            out_J[c * " << num_inputs << " + j] = (g_pert[c] - g_curr[c]) * inv_h;\n"
        << "        }\n"
        << "    }\n"
        << "}\n\n";

    // Static inline Levenberg-Marquardt projection for single configuration
    out << "static inline void project_single(const float* q_in, int num_inputs, float* q_out) {\n"
        << "    float q[" << num_inputs << "];\n"
        << "    for (int i = 0; i < " << num_inputs << "; ++i) q[i] = q_in[i];\n"
        << "    float g[" << num_constraints << "];\n"
        << "    float J[" << (num_constraints * num_inputs) << "];\n"
        << "    const float lambda = 1e-3f;\n"
        << "    const int max_iters = 20;\n"
        << "    const float tol_sq = 1e-8f;\n\n"
        << "    for (int iter = 0; iter < max_iters; ++iter) {\n"
        << "        evaluate_dag(q, g);\n"
        << "        float err_sq = 0.0f;\n"
        << "        for (int c = 0; c < " << num_constraints << "; ++c) {\n"
        << "            err_sq += g[c] * g[c];\n"
        << "        }\n"
        << "        if (err_sq < tol_sq) break;\n\n"
        << "        evaluate_jacobian(q, g, J);\n\n";

    if (num_constraints == 1) {
        out << "        // Fast rank-1 LM step\n"
            << "        float denom = lambda;\n"
            << "        for (int j = 0; j < " << num_inputs << "; ++j) {\n"
            << "            denom += J[j] * J[j];\n"
            << "        }\n"
            << "        for (int j = 0; j < " << num_inputs << "; ++j) {\n"
            << "            q[j] -= (g[0] * J[j]) / denom;\n"
            << "        }\n";
    } else {
        out << "        // Multi-constraint damped normal equations solve (J*J^T + lambda*I) y = -g\n"
            << "        float A[" << num_constraints << "][" << num_constraints << "];\n"
            << "        float rhs[" << num_constraints << "];\n"
            << "        float y[" << num_constraints << "];\n"
            << "        for (int r = 0; r < " << num_constraints << "; ++r) {\n"
            << "            rhs[r] = -g[r];\n"
            << "            for (int c = 0; c < " << num_constraints << "; ++c) {\n"
            << "                float sum = (r == c) ? lambda : 0.0f;\n"
            << "                for (int k = 0; k < " << num_inputs << "; ++k) {\n"
            << "                    sum += J[r * " << num_inputs << " + k] * J[c * " << num_inputs << " + k];\n"
            << "                }\n"
            << "                A[r][c] = sum;\n"
            << "            }\n"
            << "        }\n"
            << "        // Forward elimination with partial pivoting\n"
            << "        for (int k = 0; k < " << num_constraints << "; ++k) {\n"
            << "            int max_row = k;\n"
            << "            float max_val = fabsf(A[k][k]);\n"
            << "            for (int r = k + 1; r < " << num_constraints << "; ++r) {\n"
            << "                if (fabsf(A[r][k]) > max_val) { max_val = fabsf(A[r][k]); max_row = r; }\n"
            << "            }\n"
            << "            if (max_row != k) {\n"
            << "                for (int c = 0; c < " << num_constraints << "; ++c) {\n"
            << "                    float tmp = A[k][c]; A[k][c] = A[max_row][c]; A[max_row][c] = tmp;\n"
            << "                }\n"
            << "                float tmp_rhs = rhs[k]; rhs[k] = rhs[max_row]; rhs[max_row] = tmp_rhs;\n"
            << "            }\n"
            << "            float diag = A[k][k];\n"
            << "            if (fabsf(diag) < 1e-12f) diag = 1e-12f;\n"
            << "            for (int r = k + 1; r < " << num_constraints << "; ++r) {\n"
            << "                float factor = A[r][k] / diag;\n"
            << "                for (int c = k; c < " << num_constraints << "; ++c) {\n"
            << "                    A[r][c] -= factor * A[k][c];\n"
            << "                }\n"
            << "                rhs[r] -= factor * rhs[k];\n"
            << "            }\n"
            << "        }\n"
            << "        // Back-substitution\n"
            << "        for (int k = " << num_constraints << " - 1; k >= 0; --k) {\n"
            << "            float sum = rhs[k];\n"
            << "            for (int c = k + 1; c < " << num_constraints << "; ++c) {\n"
            << "                sum -= A[k][c] * y[c];\n"
            << "            }\n"
            << "            float diag = A[k][k];\n"
            << "            if (fabsf(diag) < 1e-12f) diag = 1e-12f;\n"
            << "            y[k] = sum / diag;\n"
            << "        }\n"
            << "        // Update q = q + J^T * y\n"
            << "        for (int j = 0; j < " << num_inputs << "; ++j) {\n"
            << "            float delta = 0.0f;\n"
            << "            for (int c = 0; c < " << num_constraints << "; ++c) {\n"
            << "                delta += J[c * " << num_inputs << " + j] * y[c];\n"
            << "            }\n"
            << "            q[j] += delta;\n"
            << "        }\n";
    }

    out << "    }\n\n"
        << "    for (int i = 0; i < " << num_inputs << "; ++i) {\n"
        << "        q_out[i] = q[i];\n"
        << "    }\n"
        << "}\n\n";
}

std::string generate_scalar_c(const LowerResult& lower_result, const CodegenOptions& opts) {
    (void)opts;
    const ExprDAG& dag = lower_result.dag;
    int num_inputs = lower_result.num_inputs;
    const auto& constraint_outputs = lower_result.constraint_outputs;
    
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    
    // Header
    out << "// Generated by Cudro Scalar Emitter - do not edit\n";
    out << "#include <math.h>\n";
    out << "#include <stddef.h>\n\n";
    
    emit_shared_routines(out, dag, constraint_outputs, num_inputs);
    
    // Public C entry points
    out << "// Evaluates constraint vector g(q)\n";
    out << "void evaluate_constraints(const float* q, int num_inputs, float* out_g) {\n";
    out << "    (void)num_inputs;\n";
    out << "    evaluate_dag(q, out_g);\n";
    out << "}\n\n";

    out << "// Manifold projection: projects q_in onto { q : g(q) = 0 } yielding q_out\n";
    out << "void project(const float* q_in, int num_inputs, float* q_out) {\n";
    out << "    project_single(q_in, num_inputs, q_out);\n";
    out << "}\n";
    
    return out.str();
}

std::string generate_batched_c(const LowerResult& lower_result, const CodegenOptions& opts) {
    (void)opts;
    const ExprDAG& dag = lower_result.dag;
    int num_inputs = lower_result.num_inputs;
    int num_constraints = lower_result.constraint_outputs.size();
    const auto& constraint_outputs = lower_result.constraint_outputs;
    
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    
    // Header
    out << "// Generated by Cudro Batched Emitter - do not edit\n";
    out << "#include <math.h>\n";
    out << "#include <stddef.h>\n\n";
    
    emit_shared_routines(out, dag, constraint_outputs, num_inputs);

    // Batched evaluation entry point
    out << "// Evaluates constraint vector for a batch of configurations\n";
    out << "void evaluate_batch(const float* q_batch, int batch_size, int num_inputs, float* out_g_batch) {\n";
    out << "    (void)num_inputs;\n";
    out << "    for (int b = 0; b < batch_size; ++b) {\n";
    out << "        evaluate_dag(q_batch + b * " << num_inputs << ", out_g_batch + b * " << num_constraints << ");\n";
    out << "    }\n";
    out << "}\n\n";

    // Batched projection entry point
    out << "// Projects a batch of configurations onto the constraint manifold\n";
    out << "void project_batch(const float* q_batch, int batch_size, int num_inputs, float* q_out_batch) {\n";
    out << "    for (int b = 0; b < batch_size; ++b) {\n";
    out << "        project_single(q_batch + b * num_inputs, num_inputs, q_out_batch + b * num_inputs);\n";
    out << "    }\n";
    out << "}\n";
    
    return out.str();
}

bool cpu_supports_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

} // namespace cudro