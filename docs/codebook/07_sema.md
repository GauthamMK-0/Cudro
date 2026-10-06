# Chapter 07: Semantic Analysis & Symbol Resolution

## Files Covered
- `include/cudro/sema.hpp`
- `src/sema.cpp`

---

## 1. Architectural Purpose
Semantic analysis bridges the **syntax tree** (what the parser built) and the **meaning** of the program. The parser only validates *structure*; sema validates *meaning*:

1. **Name Resolution**: Every `link ee` in a task must refer to a declared link. Every `joint_ref j1` must refer to a declared joint.
2. **Uniqueness**: Robot names, joint names, link names, task names must all be unique within their scope.
3. **Parent/Child Consistency**: Link `parent` references must form a valid tree rooted at `world` (special keyword for root frame).

---

## 2. Sema Interface (`include/cudro/sema.hpp`)

```cpp
class Sema {
public:
    Sema(const Spec& spec, DiagnosticBag& diags)
        : spec_(spec), diags_(diags) {}

    bool analyze(); // returns true if no errors

private:
    const Spec& spec_;
    DiagnosticBag& diags_;

    // Symbol tables
    std::unordered_map<std::string, const RobotDecl*> robot_table_;
    std::unordered_map<std::string, const JointDecl*> joint_table_;
    std::unordered_map<std::string, const LinkDecl*> link_table_;
    std::unordered_map<std::string, const TaskDecl*> task_table_;

    void build_tables();
    void check_robot(const RobotDecl&);
    void check_task(const TaskDecl&);
    void check_clearance(const ClearanceDecl&);
    void validate_parent_tree(const RobotDecl&);
};
```

### Symbol Tables
Four `std::unordered_map<string, const Decl*>` tables provide **O(1) lookup** for:
- Robot names → `RobotDecl*`
- Joint names → `JointDecl*`
- Link names → `LinkDecl*`
- Task names → `TaskDecl*`

---

## 3. Analysis Passes (`src/sema.cpp`)

### A. Build Tables (Phase 1)
```cpp
void Sema::build_tables() {
    // Build robot table
    for (const auto& robot : spec_.robots) {
        if (!robot_table_.emplace(robot->name, robot.get()).second) {
            diags_.error(robot->loc, "duplicate robot name: " + robot->name);
        }
        // Index joints & links
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
    for (const auto& task : spec_.tasks) {
        if (!task_table_.emplace(task->name, task.get()).second) {
            diags_.error(task->loc, "duplicate task name: " + task->name);
        }
    }
}
```
Uses `emplace` return value (`.second` is `bool`) to detect duplicates atomically.

---

### B. Check Robot (Phase 2)
```cpp
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
```

**Key point:** `"world"` is a special keyword meaning "root frame" — it's treated as a valid root parent without requiring a link table entry.

---

### C. Check Task (Phase 3)
```cpp
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
```

**Key point:** Plane constraint's `link` field is optional — if empty, defaults to the task's link.

---

### C. Kinematic Tree Validation
```cpp
void Sema::validate_parent_tree(const RobotDecl& robot) {
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
```

**Key features:**
- `"world"` parent = root indicator (same as no parent)
- Single root enforced (error if >1 root)
- Cycle detection via DFS with recursion stack
- Reachability check: all links must be reachable from root

---

### D. Main Entry Point
```cpp
bool Sema::analyze() {
    build_tables();
    for (const auto& robot : spec_.robots) check_robot(*robot);
    for (const auto& task : spec_.tasks) check_task(*task);
    for (const auto& clearance : spec_.clearances) check_clearance(*clearance);
    return !diags_.has_errors();
}
```

---

## 3. Robotics Context
In a robotics compiler, semantic errors correspond to **physical impossibilities**:
- `joint_ref j5` on a link that doesn't connect to joint `j5` → kinematic chain broken
- `parent arm2` for a link whose parent is `arm1` → kinematic tree malformed
- `task cup_on_table` referencing link `ee` that doesn't exist → constraint unattached

Catching these at **compile time** (before JIT) prevents the robot from attempting infeasible motions at runtime. The compiler becomes a **static verifier of kinematic correctness**.

---

## 4. Testing
`tests/test_sema.cpp` (to be added) covers:
- Valid panda7 spec → passes
- Duplicate joint name → error
- Unknown parent link → error
- Unknown joint_ref → error
- Task referencing non-existent link → error
- Cyclic parent graph → error
- Multiple root links → error
- Cyclic parent chain → error
- Unreachable link → error
- Plane constraint defaults to task link
- "world" parent treated as root