#pragma once

#include <vector>
#include <functional>
#include <string>
#include <random>

namespace cudro {

// Configuration of the Constrained RRT Planner
struct PlannerOptions {
    int max_iterations = 2000;
    float step_size = 0.1f;            // Maximum step distance in joint space
    float goal_bias = 0.15f;           // Probability of sampling the goal directly
    float constraint_tolerance = 1e-3f; // Acceptable constraint residual
    float max_projection_distance = 0.3f; // Max allowed drift during projection
    uint32_t seed = 42;
};

// Result of a planning query
struct PlannerResult {
    bool success = false;
    std::vector<std::vector<float>> path; // Waypoints from start to goal
    int iterations = 0;
    int projection_count = 0;
    double planning_time_ms = 0.0;
    std::string message;
};

// Joint limits for bounds checking & random sampling
struct JointLimits {
    std::vector<float> lower;
    std::vector<float> upper;
};

class ConstrainedPlanner {
public:
    using ProjectFn = void(*)(const float* q_in, int num_inputs, float* q_out);
    using EvaluateFn = void(*)(const float* q, int num_inputs, float* out_g);

    ConstrainedPlanner(int dof, int num_constraints,
                       ProjectFn project_fn,
                       EvaluateFn eval_fn,
                       JointLimits limits,
                       PlannerOptions options = {});

    // Plans a constraint-satisfying trajectory between start and goal
    PlannerResult plan(const std::vector<float>& start, const std::vector<float>& goal);

    // Projects a single configuration onto the manifold
    bool project_configuration(const std::vector<float>& q_in, std::vector<float>& q_out);

    // Validates that all configurations in a path satisfy constraints and joint limits
    bool validate_path(const std::vector<std::vector<float>>& path, float tolerance = 1e-3f) const;

private:
    int dof_;
    int num_constraints_;
    ProjectFn project_fn_;
    EvaluateFn eval_fn_;
    JointLimits limits_;
    PlannerOptions options_;

    // Tree node for Constrained RRT
    struct Node {
        std::vector<float> q;
        int parent_idx = -1;
    };

    std::vector<float> sample_random_config(const std::vector<float>& goal, std::mt19937& rng);
    int find_nearest_node(const std::vector<Node>& tree, const std::vector<float>& target) const;
    float distance_sq(const std::vector<float>& a, const std::vector<float>& b) const;
};

} // namespace cudro
