#pragma once

#include <cudro/dag.hpp>
#include <cudro/lower.hpp>

#include <vector>
#include <string>

namespace cudro {

struct ADResult {
    ExprDAG dag;                          // Original DAG + gradient nodes appended
    std::vector<std::vector<double>> jacobians;  // [constraint][input] = derivative value
    int num_inputs;
    int num_constraints;
    std::vector<std::vector<int>> jacobian_nodes; // [constraint][input] = node index in dag
};

// Evaluates all DAG nodes with concrete inputs
std::vector<double> evaluate_dag(const ExprDAG& dag, int num_inputs, const double* q);

// Forward-mode autodiff pass: computes Jacobians for all constraint outputs
// w.r.t. all inputs (q[0]...q[n-1]) using forward-mode dual numbers.
ADResult differentiate(const ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs);

// Appends analytical gradient nodes to dag for all constraint outputs w.r.t all inputs.
// Returns a 2D matrix of node indices: [constraint][input] -> node index in dag.
std::vector<std::vector<int>> build_analytical_jacobians(ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs);

// Convenience wrapper: builds analytical Jacobians and populates lr.constraint_jacobians
void build_analytical_jacobians(LowerResult& lr);

} // namespace cudro