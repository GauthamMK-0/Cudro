#pragma once

#include <cudro/dag.hpp>

#include <vector>
#include <string>

namespace cudro {

struct ADResult {
    ExprDAG dag;                          // Original DAG + gradient nodes appended
    std::vector<std::vector<double>> jacobians;  // [constraint][input] = derivative value
    int num_inputs;
    int num_constraints;
};

// Evaluates all DAG nodes with concrete inputs
std::vector<double> evaluate_dag(const ExprDAG& dag, int num_inputs, const double* q);

// Forward-mode autodiff pass: computes Jacobians for all constraint outputs
// w.r.t. all inputs (q[0]...q[n-1]) using forward-mode dual numbers.
ADResult differentiate(const ExprDAG& dag, const std::vector<int>& constraint_outputs, int num_inputs);

} // namespace cudro