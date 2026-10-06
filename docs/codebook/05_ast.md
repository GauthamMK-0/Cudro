# Chapter 05: Abstract Syntax Tree (AST)

## Files Covered
- `include/cudro/ast.hpp`
- `src/ast.cpp`

---

## 1. Architectural Purpose
The AST is the **parse-time representation** of the `.cudro` spec. It mirrors the grammar structure directly:
- `Spec` (top-level) → list of `RobotDecl`, `TaskDecl`, `ClearanceDecl`
- `RobotDecl` → joints + links
- `JointDecl`, `LinkDecl` → kinematic parameters
- `TaskDecl` → constraint blocks (`PlaneConstraint`)
- `ClearanceDecl` → global collision margin

**Key principle**: The AST carries **only syntax and source locations** — no semantic validation (that's M2/Sema), no lowering (M3), no codegen (M5+).

---

## 2. Location-Aware Base (`include/cudro/ast.hpp`)

```cpp
struct Node {
    Location loc;
    virtual ~Node() = default;
};
```
Every AST node inherits `Location` so error messages point to the exact declaration.

---

## 2. Primitive Value Nodes

### Vec3 (3-component vector literal)
```cpp
struct Vec3 : Node {
    double x, y, z;
};
```
Represents `[x, y, z]` in the source.

### Sphere (collision sphere)
```cpp
struct Sphere : Node {
    Vec3 center;
    double radius;
};
```
Represents `[cx, cy, cz, r]`.

---

## 3. Declaration Nodes

### Joint Declaration
```cpp
struct JointDecl : Node {
    std::string name;           // e.g., "j1"
    std::string type;           // "revolute" | "fixed"
    Vec3 axis;                  // rotation axis (world or parent frame)
    Vec3 origin;                // translation from parent
    std::optional<std::pair<double, double>> limits; // joint limits [min, max]
};
```
Maps to:
```
joint j1 { type revolute; axis [0,0,1]; origin [0,0,0.333];
           limits [-2.8973, 2.8973]; }
```

### Link Declaration
```cpp
struct LinkDecl : Node {
    std::string name;           // e.g., "arm1"
    std::vector<Sphere> spheres; // collision spheres
    std::optional<std::string> parent;    // parent link name
    std::optional<std::string> joint_ref; // connecting joint name
};
```

### Robot Declaration
```cpp
struct RobotDecl : Node {
    std::string name;                    // e.g., "panda7"
    std::vector<JointDecl> joints;
    std::vector<LinkDecl> links;
};
```

---

## 4. Constraint Nodes

### Plane Constraint
```cpp
struct PlaneConstraint : Node {
    std::string link;        // which link the constraint applies to
    Vec3 point_on_link;      // point on the link (local frame)
    Vec3 normal;             // plane normal in world frame
    double offset;           // plane equation: dot(normal, p) - offset = 0
};
```
Maps to:
```
task cup_on_table {
  link ee;
  plane { point_on_link [0,0,0.02]; normal [0,0,1]; offset 0.02; }
}
```

### Task Declaration
```cpp
struct TaskDecl : Node {
    std::string name;              // e.g., "cup_on_table"
    std::string link;              // end-effector / constrained link
    std::vector<PlaneConstraint> planes;
};
```

### Clearance Declaration
```cpp
struct ClearanceDecl : Node {
    double min_distance = 0.0;
};
```
Maps to:
```
clearance { min_distance 0.03; }
```

---

## 5. Top-Level Spec
```cpp
struct Spec {
    std::vector<RobotDecl> robots;
    std::vector<TaskDecl> tasks;
    std::vector<ClearanceDecl> clearances;
};
```

---

## 4. Robotics DSL Mapping

| AST Node | Robotics Concept |
|---|---|
| `RobotDecl` | Full kinematic chain (URDF equivalent) |
| `JointDecl` | 1 DoF with axis, limits, transform |
| `LinkDecl` | Rigid body with collision spheres |
| `PlaneConstraint` | Manifold constraint g(q) = 0 (e.g., end-effector on plane) |
| `ClearanceDecl` | Global collision margin for sphere-sphere checks |

The AST is intentionally **flat and serializable** — no pointers between nodes (names are `std::string` references resolved later in Sema). This makes the AST trivially copyable, debuggable, and amenable to future JSON/YAML interchange formats.