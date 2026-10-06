# Chapter 09: Lowering & Kinematics Inlining

## Files Covered
- `include/cudro/lower.hpp`
- `src/lower.cpp`

---

## 1. Architectural Purpose
**Lowering** transforms the validated AST into the **Expression DAG IR**. It performs two critical tasks:

1. **Desugaring**: Constraint kinds (`plane`, `relative_pose`) → core math expressions
2. **FK Inlining**: Kinematic chains → unrolled arithmetic DAG (no function calls, no loops)

This is where robotics domain knowledge becomes pure arithmetic.

---

## 2. Lowering Interface (`include/cudro/lower.hpp`)

```cpp
struct LowerResult {
    ExprDAG dag;
    std::vector<int> constraint_outputs; // DAG indices for each constraint g(q)
    std::vector<std::vector<int>> constraint_jacobians; // 2D matrix [constraint][input] filled by AD pass (Chapter 10)
    int num_inputs; // number of joint DoFs
};

LowerResult lower(const Spec& spec);
```

---

## 2. Lowering Algorithm (`src/lower.cpp`)

### A. Entry Point
```cpp
LowerResult lower(const Spec& spec) {
    ExprDAG dag;
    LowerContext ctx{dag, spec};
    LowerResult result;
    result.num_inputs = ctx.total_dofs();

    // 1. Build FK transform DAG for each link (topologically sorted)
    ctx.build_all_link_transforms();

    // 2. Desugar each task constraint (supports multi-link targets)
    for (const auto& task : spec.tasks) {
        for (const auto& plane : task->planes) {
            const std::string& target_link = plane->link.empty() ? task->link : plane->link;
            int constraint_node = ctx.lower_plane_constraint(target_link, *plane);
            result.constraint_outputs.push_back(constraint_node);
        }
    }

    // 3. Clearance (collision) constraints
    // Sphere-sphere distances added here

    // 4. Constant folding
    dag.fold_constants();

    return result;
}
```

---

## 3. Kinematics Inlining (`src/lower.cpp`)

### A. Building Link Transforms (FK)
```cpp
void LowerContext::build_all_link_transforms() {
    // Topological order: base → tip
    for (const auto& link : robot_.links) {
        if (!link.parent) continue; // base link
        
        int parent_transform = link_transforms_[*link.parent];
        int joint_idx = joint_index_[*link.joint_ref];
        int q_input = dag_.add_input(joint_idx, "q" + std::to_string(joint_idx));
        
        // Joint transform: Rot(axis, q) * Translate(origin)
        int joint_transform = build_joint_transform(link.joint_ref, q_input);
        
        // Link transform = parent * joint
        int link_transform = dag_.add_binary(NodeKind::MatMul, parent_transform, joint_transform);
        link_transforms_[link.name] = link_transform;
    }
}
```

### B. Joint Transform Construction
```cpp
int LowerContext::build_joint_transform(const std::string& joint_name, int q_input) {
    const JointDecl& joint = joint_map_[joint_name];
    
    int rot_node;
    if (joint.axis == Vec3{0,0,1}) rot_node = dag_.add_unary(NodeKind::RotZ, q_input);
    else if (joint.axis == Vec3{0,1,0}) rot_node = dag_.add_unary(NodeKind::RotY, q_input);
    else if (joint.axis == Vec3{1,0,0}) rot_node = dag_.add_unary(NodeKind::RotX, q_input);
    else rot_node = dag_.add_binary(NodeKind::RotAxis, q_input, dag_.add_vec3(joint.axis));
    
    // Translation matrix from origin
    int trans = dag_.add_unary(NodeKind::Translate, dag_.add_vec3(joint.origin));
    
    // Joint transform = Rot * Translate
    return dag_.add_binary(NodeKind::MatMul, rot_node, trans);
}
```

### C. Plane Constraint Desugaring
```cpp
int LowerContext::lower_plane_constraint(const std::string& link_name, const PlaneConstraint& plane) {
    // Get link's world transform (4x4 matrix)
    int link_transform = link_transforms_.at(link_name);
    
    // Extract world position of point_on_link
    // p_world = link_transform * [px, py, pz, 1]^T  (translation component)
    int point_local = dag_.add_vec3(plane.point_on_link);
    int point_world = extract_translation(dag_.add_binary(NodeKind::MatMul, link_transform, point_local));
    
    // g(q) = dot(normal_world, p_world) - offset
    int normal = dag_.add_vec3(plane.normal);
    int dot = dag_.add_binary(NodeKind::Dot, normal, point_world);
    int offset = dag_.add_constant(plane.offset);
    
    return dag_.add_binary(NodeKind::Sub, dot, offset);
}
```

### Result: Unrolled Arithmetic DAG
For a 7-DoF arm with a plane constraint, the DAG contains ~200 nodes representing:
```
q1 → RotZ → Translate → MatMul → q2 → RotX → ... → p_world → dot(n, p) - d
```
**No function calls. No loops. Pure arithmetic.** This is what makes the generated kernel fast and SIMD-friendly.

---

## 3. Constant Folding (Revisited)
After all desugaring, `dag.fold_constants()` collapses compile-time arithmetic:
- `dot([0,0,1], [0,0,0.02]) - 0.02` → `Constant(0.0)`
- `MatMul(Identity, X)` → `X`

This removes dead arithmetic before codegen.

---

## 4. Robotics Context
Lowering is where **robotics becomes arithmetic**:
- URDF-style kinematic tree → flat arithmetic DAG
- Constraint kinds (`plane`) → `dot(n, p) - d` 
- Joint limits → handled separately in projection (clamping, not DAG)
- Collision spheres → `SphereDist` nodes in DAG (added later)

The DAG is now **pure math** — ready for autodiff (Chapter 10) and codegen (Chapter 11).

---

## 5. Testing
`tests/test_dag.cpp` and `tests/test_kernels.cpp` verify:
- FK chain node count matches expected
- Constraint values evaluate accurately across multi-link configurations

---

## 6. Evolution & Optimization: Dead Subgraph Pruning & Direct Transform Reuse

### A. Old Practice: Orphaned Subgraph Construction
In early versions of the constraint lowering pass:
1. `lower_plane_constraint` generated an orphaned temporary matrix-vector calculation that duplicated link frame operations before computing the dot product.
2. The rotation builders in `dag.cpp` (`mat4_rot_x`, `mat4_rot_y`, `mat4_rot_z`) constructed redundant intermediate negation nodes that were left disconnected after constant folding.

```cpp
// OLD PRACTICE: Re-lowering parts of the kinematic chain or creating dead nodes
int point_local = dag.add_quaternary(NodeKind::Vec4, px, py, pz, one);
int temp_subgraph = dag.add_binary(NodeKind::MatVecMul, link_transform, point_local);
// Redundant conversions and unused negation nodes left in DAG
```

**The Problem:**
Even if an expression is never consumed by the constraint output, a flat DAG compiler without full DCE (dead code elimination) will still emit C stack variables for every node in the DAG vector. This generated dozens of dead C variables in the emitted kernel.

---

### B. Updated Optimized Implementation: Clean Direct Lowering
We streamlined `lower_plane_constraint` to directly construct the homogeneous homogeneous vector product and wire it straight to `NodeKind::Dot`:

```cpp
// UPDATED OPTIMIZED CODE: Clean, minimal DAG wiring
int LowerContext::lower_plane_constraint(const std::string& task_link, const PlaneConstraint& plane) {
    int link_transform = link_transforms_.at(task_link);
    
    // Homogeneous point in link frame: [px, py, pz, 1.0]
    int point_local = dag.add_quaternary(NodeKind::Vec4, 
        dag.add_constant(plane.point_on_link.x),
        dag.add_constant(plane.point_on_link.y),
        dag.add_constant(plane.point_on_link.z),
        dag.add_constant(1.0));
    
    // Direct homogeneous transform: p_world = link_transform * point_local
    int p_world = mat4_vec4_mul(dag, link_transform, point_local);
    
    // Homogeneous normal vector: [nx, ny, nz, 0.0]
    int normal_world = dag.add_quaternary(NodeKind::Vec4,
        dag.add_constant(plane.normal.x),
        dag.add_constant(plane.normal.y),
        dag.add_constant(plane.normal.z),
        dag.add_constant(0.0));
    
    // Constraint equation: g(q) = dot(normal_world, p_world) - offset
    int dot_node = dag.add_binary(NodeKind::Dot, normal_world, p_world);
    int offset_node = dag.add_constant(plane.offset);
    return dag.add_binary(NodeKind::Sub, dot_node, offset_node);
}
```

### C. Observed Change & Concrete Metrics
- **Zero Dead Variables**: Every variable emitted in `evaluate_dag` participates directly in the forward kinematic or constraint evaluation chain.
- **Topologically Pure DAG**: Eliminates disconnected nodes and prevents dead subgraphs from being propagated into the Automatic Differentiation pass (Chapter 10).

---

## 4. Evolution Case Study: Topological Sorting in Branching Kinematic Trees

### A. The Old Hazard: Declaration-Order Dependency in FK Inlining
In the baseline lowering implementation:
```cpp
// OLD PRACTICE: Checked if parent link existed in robot, NOT if its transform was built!
bool ready = true;
if (link->parent) {
    if (!link_idx_.count(*link->parent)) {
        if (*link->parent != "world") ready = false;
    }
}
```
**The Failure Mode:**
In a serial robot like `panda7`, links were declared in strict base-to-tip order (`base`, `arm1`, `arm2`, ...), so parent transforms were coincidentally built first. However, in **branching kinematic trees** (e.g. `bimanual14` with a torso branching into left and right arms) or when links were declared out of topological order, a child link would evaluate `ready == true` in the first pass while its parent transform was still `-1`. This caused child links to be emitted without multiplying their parent transform!

### B. Updated Robust Implementation: True Topological Progression
We updated `build_all_link_transforms` to check against the set of *already-processed* links:
```cpp
// UPDATED ROBUST CODE: Guarantees parent transform is built before child
bool ready = true;
if (link->parent) {
    const std::string& parent_name = *link->parent;
    if (parent_name != "world" && !processed.count(parent_name)) {
        ready = false;
    }
}
```
If a parent link hasn't been lowered yet, the child yields until subsequent iterations resolve the parent, guaranteeing correct $T_{\text{world} \to \text{child}} = T_{\text{world} \to \text{parent}} \cdot T_{\text{joint}}$ composition across arbitrary branching robots (bimanual manipulators, humanoids, multi-limbed systems).

### C. Multi-Link Task Constraints
Previously, task lowering passed `task.link` unconditionally:
```cpp
// UPDATED: Allow individual constraints to target intermediate links
const std::string& target_link = plane->link.empty() ? task->link : plane->link;
int constraint_node = ctx.lower_plane_constraint(target_link, *plane);
```
This enables simultaneous multi-link constraints (e.g., constraining end-effector 3D position while simultaneously enforcing elbow and forearm obstacle planes).