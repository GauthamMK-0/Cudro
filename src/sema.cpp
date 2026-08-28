#include <cudro/sema.hpp>

#include <cstdio>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <functional>

namespace cudro {

bool Sema::analyze() {
    build_tables();
    for (const auto& robot : spec_.robots) {
        check_robot(*robot);
    }
    for (const auto& task : spec_.tasks) {
        check_task(*task);
    }
    for (const auto& clearance : spec_.clearances) {
        check_clearance(*clearance);
    }
    return !diags_.has_errors();
}

void Sema::build_tables() {
    // Build robot table
    for (const auto& robot : spec_.robots) {
        if (!robot_table_.emplace(robot->name, robot.get()).second) {
            diags_.error(robot->loc, "duplicate robot name: " + robot->name);
        }
    }

    // Build joint and link tables (per-robot scope)
    for (const auto& robot : spec_.robots) {
        for (const auto& joint : robot->joints) {
            if (!joint_table_.emplace(joint->name, joint.get()).second) {
                diags_.error(joint->loc, "duplicate joint name: " + joint->name);
            }
        }
        for (const auto& link : robot->links) {
            if (!link_table_.emplace(link->name, link.get()).second) {
                diags_.error(link->loc, "duplicate link name: " + link->name);
            }
        }
    }

    // Build task table
    for (const auto& task : spec_.tasks) {
        if (!task_table_.emplace(task->name, task.get()).second) {
            diags_.error(task->loc, "duplicate task name: " + task->name);
        }
    }
}

void Sema::check_robot(const RobotDecl& robot) {
    // Validate link parent references
    for (const auto& link : robot.links) {
        if (link->parent) {
            const std::string& parent = *link->parent;
            if (parent != "world" && !link_table_.count(parent)) {
                diags_.error(link->loc, "unknown parent link: " + parent);
            }
        }
        if (link->joint_ref) {
            if (!joint_table_.count(*link->joint_ref)) {
                diags_.error(link->loc, "unknown joint_ref: " + *link->joint_ref);
            }
        }
    }

    // Validate parent graph is a tree (no cycles, single root)
    validate_parent_tree(robot);
}

void Sema::check_task(const TaskDecl& task) {
    // Task's link must exist
    if (!link_table_.count(task.link)) {
        diags_.error(task.loc, "task references unknown link: " + task.link);
    }
    // Each plane's link must exist (defaults to task.link if empty)
    for (const auto& plane : task.planes) {
        const std::string& plane_link = plane->link.empty() ? task.link : plane->link;
        if (!link_table_.count(plane_link)) {
            diags_.error(plane->loc, "plane constraint references unknown link: " + plane_link);
        }
    }
}

void Sema::check_clearance(const ClearanceDecl& clearance) {
    // min_distance should be non-negative
    if (clearance.min_distance < 0) {
        diags_.error(clearance.loc, "min_distance must be non-negative");
    }
}

void Sema::validate_parent_tree(const RobotDecl& robot) {
    // Build adjacency list
    std::unordered_map<std::string, std::vector<std::string>> children;
    std::unordered_set<std::string> has_parent;
    std::string root;

    for (const auto& link : robot.links) {
        if (link->parent) {
            const std::string& parent = *link->parent;
            if (parent != "world") {
                children[parent].push_back(link->name);
                has_parent.insert(link->name);
            } else {
                // "world" parent means this link is a root
                if (!root.empty()) {
                    diags_.error(link->loc, "multiple root links (multiple links with parent 'world' or no parent): " + link->name + " and " + root);
                }
                root = link->name;
            }
        } else {
            if (!root.empty()) {
                diags_.error(link->loc, "multiple root links (missing parent): " + link->name + " and " + root);
            }
            root = link->name;
        }
    }

    if (root.empty()) {
        diags_.error(robot.loc, "robot has no root link (all links have parent)");
        return;
    }

    // DFS for cycle detection
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> rec_stack;

    std::function<void(const std::string&)> dfs = [&](const std::string& name) {
        visited.insert(name);
        rec_stack.insert(name);
        auto it = children.find(name);
        if (it != children.end()) {
            for (const auto& child : it->second) {
                if (rec_stack.count(child)) {
                    diags_.error(robot.loc, "cycle detected in parent chain: " + child);
                } else if (!visited.count(child)) {
                    dfs(child);
                }
            }
        }
        rec_stack.erase(name);
    };

    dfs(root);

    // Check all links reachable from root
    for (const auto& link : robot.links) {
        if (!visited.count(link->name)) {
            diags_.error(link->loc, "link not reachable from root: " + link->name);
        }
    }
}

} // namespace cudro