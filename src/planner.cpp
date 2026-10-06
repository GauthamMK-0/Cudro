#include <cudro/planner.hpp>

#include <cmath>
#include <chrono>
#include <algorithm>

namespace cudro {

ConstrainedPlanner::ConstrainedPlanner(int dof, int num_constraints,
                                       ProjectFn project_fn,
                                       EvaluateFn eval_fn,
                                       JointLimits limits,
                                       PlannerOptions options)
    : dof_(dof), num_constraints_(num_constraints),
      project_fn_(project_fn), eval_fn_(eval_fn),
      limits_(std::move(limits)), options_(options) {
    if (limits_.lower.empty()) {
        limits_.lower.assign(dof_, -3.14159f);
    }
    if (limits_.upper.empty()) {
        limits_.upper.assign(dof_, 3.14159f);
    }
    g_scratch_.assign(num_constraints_, 0.0f);
}

float ConstrainedPlanner::distance_sq(const std::vector<float>& a, const std::vector<float>& b) const {
    float sum = 0.0f;
    for (int i = 0; i < dof_; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int ConstrainedPlanner::find_nearest_node(const std::vector<Node>& tree, const std::vector<float>& target) const {
    int best_idx = 0;
    float best_dist = distance_sq(tree[0].q, target);
    for (size_t i = 1; i < tree.size(); ++i) {
        float d = distance_sq(tree[i].q, target);
        if (d < best_dist) {
            best_dist = d;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

bool ConstrainedPlanner::project_configuration(const std::vector<float>& q_in, std::vector<float>& q_out) {
    q_out.resize(dof_);
    int status = project_fn_(q_in.data(), dof_, q_out.data());
    if (status != 0) {
        return false;
    }

    eval_fn_(q_out.data(), dof_, g_scratch_.data());

    float err_sq = 0.0f;
    for (int c = 0; c < num_constraints_; ++c) {
        err_sq += g_scratch_[c] * g_scratch_[c];
    }

    return (std::sqrt(err_sq) <= options_.constraint_tolerance);
}

std::vector<float> ConstrainedPlanner::sample_random_config(const std::vector<float>& goal, std::mt19937& rng) {
    std::uniform_real_distribution<float> goal_dist(0.0f, 1.0f);
    if (goal_dist(rng) < options_.goal_bias) {
        return goal;
    }

    std::vector<float> q(dof_);
    for (int i = 0; i < dof_; ++i) {
        std::uniform_real_distribution<float> joint_dist(limits_.lower[i], limits_.upper[i]);
        q[i] = joint_dist(rng);
    }
    return q;
}

bool ConstrainedPlanner::validate_path(const std::vector<std::vector<float>>& path, float tolerance) const {
    if (path.empty()) return false;
    for (const auto& waypoint : path) {
        eval_fn_(waypoint.data(), dof_, g_scratch_.data());
        float err_sq = 0.0f;
        for (int c = 0; c < num_constraints_; ++c) {
            err_sq += g_scratch_[c] * g_scratch_[c];
        }
        if (std::sqrt(err_sq) > tolerance) {
            return false;
        }
    }
    return true;
}

PlannerResult ConstrainedPlanner::plan(const std::vector<float>& start, const std::vector<float>& goal) {
    auto t0 = std::chrono::high_resolution_clock::now();
    PlannerResult result;
    result.success = false;

    // 1. Verify start and goal
    std::vector<float> start_proj;
    std::vector<float> goal_proj;
    result.projection_count += 2;

    if (!project_configuration(start, start_proj)) {
        result.message = "Start configuration cannot be projected onto constraint manifold";
        return result;
    }
    if (!project_configuration(goal, goal_proj)) {
        result.message = "Goal configuration cannot be projected onto constraint manifold";
        return result;
    }

    // 2. Initialize bidirectional trees
    std::vector<Node> tree_start;
    std::vector<Node> tree_goal;
    tree_start.push_back(Node{start_proj, -1});
    tree_goal.push_back(Node{goal_proj, -1});

    std::vector<Node>* tree_a = &tree_start;
    std::vector<Node>* tree_b = &tree_goal;
    bool swapped = false;

    std::mt19937 rng(options_.seed);
    std::vector<float> q_cand(dof_);
    std::vector<float> q_proj(dof_);

    auto extend_tree = [&](std::vector<Node>& tree, const std::vector<float>& target, int& new_node_idx) -> bool {
        int near_idx = find_nearest_node(tree, target);
        const auto& q_near = tree[near_idx].q;

        float dist = std::sqrt(distance_sq(q_near, target));
        if (dist < 1e-4f) return false;

        float step = std::min(options_.step_size, dist);
        float scale = step / dist;
        for (int i = 0; i < dof_; ++i) {
            q_cand[i] = q_near[i] + (target[i] - q_near[i]) * scale;
        }

        result.projection_count++;
        if (!project_configuration(q_cand, q_proj)) {
            return false;
        }

        float step_len = std::sqrt(distance_sq(q_near, q_proj));
        if (step_len < 1e-4f) return false;

        // Check joint limits
        for (int i = 0; i < dof_; ++i) {
            if (q_proj[i] < limits_.lower[i] || q_proj[i] > limits_.upper[i]) {
                return false;
            }
        }

        new_node_idx = static_cast<int>(tree.size());
        tree.push_back(Node{q_proj, near_idx});
        return true;
    };

    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        result.iterations = iter + 1;

        // Sample random target
        std::vector<float> q_rand = sample_random_config(goal_proj, rng);

        // Extend tree A towards q_rand
        int new_idx_a = -1;
        if (!extend_tree(*tree_a, q_rand, new_idx_a)) {
            continue;
        }

        const auto& q_new_a = (*tree_a)[new_idx_a].q;

        // Try to connect tree B towards q_new_a
        int curr_b_idx = -1;
        bool advanced_b = extend_tree(*tree_b, q_new_a, curr_b_idx);

        if (advanced_b) {
            const auto& q_new_b = (*tree_b)[curr_b_idx].q;
            float connect_dist = std::sqrt(distance_sq(q_new_a, q_new_b));

            if (connect_dist <= options_.step_size * 1.5f) {
                // Trees connected!
                // Path from tree_start root to connection point
                std::vector<std::vector<float>> path_a;
                int curr = swapped ? curr_b_idx : new_idx_a;
                const auto& actual_start_tree = swapped ? *tree_b : *tree_a;
                while (curr != -1) {
                    path_a.push_back(actual_start_tree[curr].q);
                    curr = actual_start_tree[curr].parent_idx;
                }
                std::reverse(path_a.begin(), path_a.end());

                // Path from connection point to tree_goal root
                std::vector<std::vector<float>> path_b;
                curr = swapped ? new_idx_a : curr_b_idx;
                const auto& actual_goal_tree = swapped ? *tree_a : *tree_b;
                while (curr != -1) {
                    path_b.push_back(actual_goal_tree[curr].q);
                    curr = actual_goal_tree[curr].parent_idx;
                }

                // Combine paths
                path_a.insert(path_a.end(), path_b.begin(), path_b.end());

                result.success = true;
                result.path = std::move(path_a);
                result.message = "Feasible constraint-satisfying trajectory found via C-RRT-Connect";
                break;
            }
        }

        // Swap trees
        std::swap(tree_a, tree_b);
        swapped = !swapped;
    }

    if (!result.success) {
        result.message = "Max iterations reached without connecting trees";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.planning_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

} // namespace cudro
