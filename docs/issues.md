# Cudro Compiler Issues Log

Tracks compiler-related issues encountered during development.

---

## Issue 1: Missing `<optional>` Include in token.hpp

**Date:** 2026-08-26  
**Component:** Front-end (lexer/token)  
**Severity:** Build failure  

**Issue:** `token.hpp` used `std::optional<TokenKind>` but did not include `<optional>`. Compiled by luck in some translation units due to transitive includes.

**Cause:** Header self-sufficiency not enforced — `token.hpp` used `std::optional` but relied on transitive inclusion from other headers.

**Solution:** Added `#include <optional>` to `token.hpp`. Enforced rule: every header must include what it directly uses.

---

## Issue 2: Dangling `string_view` in Test Harness

**Date:** 2026-08-26  
**Component:** Lexer tests  
**Severity:** Heap-use-after-free (caught by ASan)  

**Issue:** Test helper `lex(const std::string&)` took a temporary `std::string` (from string literal), returned tokens holding `string_view` into it. The temporary died at end of full-expression, leaving dangling views.

**Cause:** `Token` holds `std::string_view` into caller's buffer (zero-copy design). Test passed literals directly to `lex()`, creating temporaries that died before `CHECK()` assertions ran.

**Solution:** Changed tests to store source in a named `std::string` variable before calling `lex()`, ensuring buffer lifetime exceeds token lifetime. Mirrors production CLI where `src` outlives tokens.

**Lesson:** `string_view` is a loan, not ownership. Caller must keep buffer alive.

---

## Issue 3: Lexer Recursion on Error Path

**Date:** 2026-08-26  
**Component:** Lexer  
**Severity:** Stack overflow on malicious input  

**Issue:** Error path in `Lexer::next()` used `return next();` recursively. A 10 MB file of `@@@@...` would consume one stack frame per character → stack overflow.

**Cause:** Recursive error recovery instead of iterative loop.

**Solution:** Refactored `next()` to use `while (true)` loop with internal `advance()` on error. Maintains "always-advance" invariant with O(1) stack usage.

**Lesson:** Recursion on error paths is dangerous for compilers processing untrusted input.

---

## Issue 4: Number Parsing Throws on Valid Input

**Date:** 2026-08-27  
**Component:** Parser (`parse_number`)  
**Severity:** Runtime crash (`std::invalid_argument` from `std::stod`)  

**Issue:** `parse_number()` used `std::stod(t.text)` where `t.text` is a `string_view` (not null-terminated). `std::stod` requires null-terminated string → read past buffer → crash.

**Cause:** `Token::text` is `string_view` into source buffer; no null terminator guaranteed. `std::stod` requires C-string.

**Solution:** Copy `string_view` to `std::string` before calling `std::stod`, or use `std::from_chars` (C++17). Implemented safe wrapper `parse_number_safe()` that returns NaN on error instead of throwing.

**Lesson:** `string_view` is not C-string compatible. Always copy or use `from_chars` for numeric parsing.

---

## Issue 5: Missing Semicolon After Joint `type` Field

**Date:** 2026-08-27  
**Component:** Parser (joint declaration)  
**Severity:** Parse error cascading  

**Issue:** Grammar requires semicolon after `type revolute;`, but parser didn't consume it. Parser then tried to parse `axis` when current token was `;` → error recovery kicked in, skipping tokens until sync point.

**Cause:** Missing `consume(TokenKind::Semicolon)` after type field in `parse_joint()`.

**Solution:** Added `consume(TokenKind::Semicolon, "expected ';' after type");` after type field in `parse_joint()`.

---

## Issue 6: Duplicate Keyword Consumption in Plane Constraint

**Date:** 2026-08-27  
**Component:** Parser (task/plane constraint)  
**Severity:** Parse failure on valid input  

**Issue:** `parse_task()` called `match(TokenKind::KwPlane)` which consumed `plane`, then `parse_plane_constraint()` tried to `consume(TokenKind::KwPlane)` again → unexpected `{` → error recovery → parse failure.

**Cause:** Keyword consumed twice — once by caller (`match` in `parse_task`), once by callee (`consume` in `parse_plane_constraint`).

**Solution:** Removed `consume(TokenKind::KwPlane)` from `parse_plane_constraint()`. Keyword already consumed by caller's `match()`. Added comment documenting ownership.

**Lesson:** Keyword consumption ownership must be clear — either caller or callee consumes, never both.

---

## Issue 7: Parser `match` vs `consume` Semantics Confusion

**Date:** 2026-08-27  
**Component:** Parser design  
**Severity:** Design confusion leading to bugs  

**Issue:** Inconsistent use of `match` (optional, returns bool) vs `consume` (mandatory, errors on missing). Led to bugs where optional fields were required or vice versa.

**Cause:** Naming didn't clearly communicate intent. `expect` (previous name) sounded like "hope this is here"; `consume` clearly means "take this token or error."

**Solution:** Renamed `expect` → `consume` for mandatory tokens. Kept `match` for optional (returns bool). Added `match` debug logging during debugging.

**Lesson:** Verb choice matters: `consume` = mandatory + advances; `match` = optional + conditional advance.

---

## Issue 8: Number Parsing Throws on Error Recovery

**Date:** 2026-08-27  
**Component:** Parser error recovery  
**Severity:** Crash during error recovery  

**Issue:** During panic-mode recovery, parser encountered invalid tokens and tried to parse numbers for error recovery paths. `std::stod` threw on invalid input (e.g., `}` where number expected).

**Cause:** `parse_number()` used `std::stod` which throws on invalid input. No safe fallback during error recovery.

**Solution:** Created `parse_number_safe()` returning `NaN` on error instead of throwing. Uses `std::stod` on copied string with position check. Returns `NaN` if parsing fails. Error recovery continues with NaN placeholder; validation catches it later.

**Lesson:** Parsers must never throw during error recovery. Use safe fallbacks (NaN, dummy nodes) to keep parsing.

---

## Issue 9: Test File Path Resolution in CTest

**Date:** 2026-08-27  
**Component:** Parser tests (`test_parser.cpp`)  
**Severity:** Test failure under CTest (passed manually)  

**Issue:** Tests used relative paths (`spec/panda7.cudro`) which worked from project root but failed under CTest (runs from `build/` directory).

**Cause:** Relative paths resolved relative to CTest working directory (`build/`), not source directory.

**Solution:** Added `CUDRO_SOURCE_DIR` compile definition via CMake (`target_compile_definitions(test_parser PRIVATE CUDRO_SOURCE_DIR="${CMAKE_SOURCE_DIR}")`). Test helper `get_spec_path()` prepends `CUDRO_SOURCE_DIR` when defined.

---

## Issue 10: Duplicate Function Definition in Parser

**Date:** 2026-08-27  
**Component:** Parser (`synchronize`)  
**Severity:** Build failure (redefinition)  

**Issue:** Added debug version of `synchronize()` but forgot to remove original → two definitions.

**Cause:** Edit added new implementation without removing old one.

**Solution:** Removed duplicate, kept version with debug logging. Cleaned up.

---

## Issue 11: Duplicate Joint Name Detection Not in Parser

**Date:** 2026-08-27  
**Component:** Parser vs Sema boundary  
**Severity:** Test expectation mismatch  

**Issue:** Test expected parser to detect duplicate joint names. Parser only builds AST; semantic validation (duplicate detection) belongs in Sema phase (M2).

**Cause:** Test assumed parser does semantic validation.

**Solution:** Updated test to mark as M2 semantic check (placeholder). Parser only builds AST; Sema will validate uniqueness.

**Lesson:** Clear phase boundaries — parser = syntax only; sema = semantics.

---

## Issue 12: Link Parser Sphere List Trailing Comma

**Date:** 2026-08-27  
**Component:** Parser (link spheres)  
**Severity:** Parse error on valid input  

**Issue:** Sphere list parsing expected comma after every sphere, but last sphere has no trailing comma. Parser's `match(TokenKind::Comma)` failed on `]`, causing error recovery.

**Cause:** Loop `while (match(TokenKind::Comma))` expected comma before next sphere, but last element has no comma.

**Solution:** This is correct behavior — `match` returns false on `]`, loop exits, then `consume(RBracket)` succeeds. Debug logging showed this works correctly. False alarm.

---

## Issue 13: AST Visitor Missing for Value Types

**Date:** 2026-08-27  
**Component:** AST visitor / dumper  
**Severity:** Build failure  

**Issue:** `ASTDumper` implemented `visit(const Vec3&)` and `visit(const Sphere&)` but these are value types (not `ASTNode` subclasses), so `override` failed.

**Cause:** `Vec3` and `Sphere` are plain structs (value types), not `ASTNode` subclasses. Visitor interface only has pure virtual for declaration nodes.

**Solution:** Removed `visit` overrides for `Vec3` and `Sphere` from `ASTDumper`. Dumper prints them inline when visiting parent nodes.

**Lesson:** Value types don't participate in visitor pattern; only polymorphic AST nodes do.

---

## Issue 14: Linker Error: Missing `Parser::parse()`

**Date:** 2026-08-27  
**Component:** Parser interface  
**Severity:** Link error  

**Issue:** `main.cpp` called `parser.parse()` but only `parse_spec()` was implemented as public method.

**Cause:** Forgot to add public `parse()` method that calls `parse_spec()`.

**Solution:** Added `Spec Parser::parse() { return parse_spec(); }` to `parser.cpp`.

---

## Issue 15: Sema Duplicate Detection Not Implemented Yet

**Date:** 2026-08-27  
**Component:** Sema (M2)  
**Severity:** Test expectation  

**Issue:** Test for duplicate joint name expected parser error, but duplicate detection is semantic (M2), not syntactic (M1).

**Cause:** Test written before M2 implemented.

**Solution:** Marked test as M2 placeholder. Will implement in `Sema::build_tables()` using `emplace` return value to detect duplicates.

---

## Issue 16: "world" Parent Not Recognized as Special Root

**Date:** 2026-08-28  
**Component:** Sema (kinematic tree validation)  
**Severity:** False positive error  

**Issue:** Spec uses `parent world` for base link (standard robotics convention), but sema treated "world" as unknown link.

**Cause:** `validate_parent_tree` treated all parents equally, requiring them to exist in link table.

**Solution:** Treat "world" as special keyword meaning "root frame" — links with `parent world` are roots, no link table lookup needed.

**Lesson:** Domain-specific conventions (world frame) must be encoded in validation logic, not treated as generic identifiers.

---

## Issue 17: Plane Constraint Missing Link Field

**Date:** 2026-08-28  
**Component:** Sema (task/plane validation)  
**Severity:** False positive error  

**Issue:** Spec has `task { link ee; plane { ... } }` but plane constraint lacks `link` field. Sema checked `plane->link` which was empty → "unknown link: " error.

**Cause:** Plane constraint's `link` field optional (defaults to task's link), but sema required it explicitly.

**Solution:** In `check_task`, use `plane->link.empty() ? task.link : plane->link` for validation. Default to task's link when empty.

**Lesson:** Default values / inheritance must be handled in semantic analysis, not just syntax.

---

## Issue 18: Multiple Root Links Error on Valid Spec

**Date:** 2026-08-28  
**Component:** Sema (kinematic tree validation)  
**Severity:** False positive error  

**Issue:** Valid panda7 spec has base link with `parent world` and no other root links, but sema reported "multiple root links" and "no root link".

**Cause:** `validate_parent_tree` treated "world" parent as regular parent (requiring link table entry), so base link appeared to have parent but parent not in table → not counted as root. Meanwhile other links had valid parents, so no root found.

**Solution:** In `validate_parent_tree`, treat `parent == "world"` as root indicator (same as no parent). Count links with `parent == "world"` or no parent as roots. Error if >1 root.

**Lesson:** Root detection must account for domain-specific root indicators (world frame).

---

## Issue 19: Autodiff Pass Matrix Operations Support

**Date:** 2026-08-28  
**Component:** M4 Autodiff pass  
**Severity:** Resolved  

**Issue:** Autodiff required full forward kinematics matrix and vector evaluation (MatMul, MatVecMul, RotX/Y/Z, RotAxis, Translate, Dot, SubVec3) across all DAG expressions.

**Solution:** Implemented structured `DAGValue` supporting `Scalar`, `Vec3`, `Vec4`, and `Mat4` types in `evaluate_dag` alongside forward-mode dual differentiation in `ad.cpp`.

**Lesson:** DAG evaluation must natively support multi-dimensional matrix/vector types when kinematic operations are lowered into the IR.

---

## Issue 20: TCC Compatibility with GCC Intrinsics Headers

**Date:** 2026-09-04  
**Component:** M5/M6 Codegen & JIT Runtime  
**Severity:** In-memory JIT compile error  

**Issue:** Including GCC-specific `<immintrin.h>` inside C source strings passed to `libtcc` failed because TCC does not support GCC-specific ADX and BMI built-in intrinsics.

**Solution:** Emitted standard C loops and arithmetic in JIT kernels; used GCC `__builtin_cpu_supports("avx2")` on the host side to manage hardware feature dispatch.

**Lesson:** Source-to-source compilers targeting lightweight JIT runtimes (TinyCC) must emit clean standard C99/C11 rather than host-compiler-proprietary header extensions.

---

## Issue 21: Manifold Projection on Unreachable Task Constraints

**Date:** 2026-09-04  
**Component:** M5 Codegen / M7 Planner  
**Severity:** Non-convergence  

**Issue:** When a constraint plane is geometrically unreachable for a given kinematic structure (e.g. asking a planar 2R robot in the XY plane to reach $z = 0.1$, or asking Panda7 to reach below its base), the Jacobian column rank becomes 0 and LM projection cannot reduce the residual.

**Solution:** Added validation checks and diagnostic reporting in `--plan` and `--jit-run` to detect unreachable manifolds and notify users when no valid starting configuration can satisfy the constraint.

**Lesson:** Manifold projection algorithms require constraints that intersect the robot's reachable workspace.

---

## Issue 22: Constrained RRT-Connect Manifold Step & Tree Bridging

**Date:** 2026-09-04  
**Component:** M7 Constrained Motion Planner  
**Severity:** Search non-convergence  

**Issue:** Unidirectional RRT exploring lower-dimensional constraint manifolds in high-dimensional joint spaces often suffered from tangent-space drift when unconstrained random samples were projected.

**Solution:** Implemented Bidirectional Constrained RRT-Connect (`C-RRT-Connect`) with step distance rescaling and dual-tree connection bridging.

**Lesson:** Bidirectional tree expansion is essential for constrained motion planning on non-linear manifolds.

---

## Issue 23: Exponential DAG Explosion in Naive Product-Rule Matrix Derivatives

**Date:** 2026-09-05  
**Component:** M4 / Phase 2 Automatic Differentiation (`ad.cpp`)  
**Severity:** Combinatorial DAG node explosion ($O(2^N)$)  

**Issue:** Differentiating matrix multiplications $(A \cdot B)' = A' \cdot B + A \cdot B'$ naively branches on every link transform in the kinematic chain. For a 7-DoF robot like Panda, this produced exponential node explosion ($2^7 = 128$ branches per output) and thousands of dead operations in the DAG.

**Cause:** Lack of activity analysis to determine which operand in $A \cdot B$ actually depends on the seed joint input $q_j$.

**Solution:** Implemented linear-time forward activity analysis (`compute_activity`) scanning the DAG before differentiation. Because each joint angle $q_j$ appears in only one link transform in a serial chain, at most one operand is active ($a_0$ or $a_1$), collapsing the product rule to a single matrix multiplication ($O(N)$ operations). Inactive terms return typed zero constants without DAG expansion.

**Lesson:** Automatic differentiation over structured kinematic chains requires domain-aware activity analysis to prune zero-derivative subtrees before AST/DAG node construction.

---

## Issue 24: Topological Ordering Hazard in Branching Kinematic Trees

**Date:** 2026-09-10  
**Component:** M3 Lowering (`lower.cpp` / `build_all_link_transforms`)  
**Severity:** Silent kinematic disconnect / build failure on branching trees  

**Issue:** Lowering branching kinematic trees (such as 14-DoF bimanual dual-arm robots) failed or attached child links to the world root when link declarations in the `.cudro` file were not listed in strict root-to-tip preorder.

**Cause:** The link resolution loop checked `if (!link_idx_.count(*link->parent))`, which merely verified that the parent link was defined somewhere in the robot AST, rather than verifying whether the parent's world transform had *already been computed* in the current lowering pass. When child links appeared prior to parent links or across branches, children attempted to multiply against uncomputed parent transforms.

**Solution:** Introduced a topological iterative resolution loop in `build_all_link_transforms`:
```cpp
std::unordered_set<std::string> processed;
while (processed.size() < robot.links.size()) {
    bool made_progress = false;
    for (const auto& link : robot.links) {
        if (processed.count(link.name)) continue;
        std::string parent_name = link.parent.value_or("world");
        if (parent_name != "world" && !processed.count(parent_name)) {
            continue; // Parent not ready yet
        }
        // Build child transform = T_parent * T_joint_origin * R_joint(q)
        processed.insert(link.name);
        made_progress = true;
    }
    if (!made_progress) break; // Cycle or disconnected tree detected
}
```

**Lesson:** Kinematic compilation cannot assume sequential root-to-tip declaration order. Lowering passes must topologically sort links by dependency order.

---

## Issue 25: Unsigned Pivot Clamping Inverting Definiteness in Singular Projections

**Date:** 2026-09-10  
**Component:** M5 / Phase 3 Codegen (`codegen_c.cpp` / `project_single`)  
**Severity:** Step reversal & divergence under near-singular rank conditions  

**Issue:** During Gaussian elimination of the multi-constraint normal equations $(J J^T + \lambda I) y = -g$, near-singular configurations yielded near-zero diagonal pivots (`fabsf(diag) < 1e-12f`). Clamping with `diag = 1e-12f` caused steps to diverge and wander away from the manifold.

**Cause:** Naive unsigned clamping `if (fabsf(diag) < 1e-12f) diag = 1e-12f;` unconditionally flipped negative near-zero values to positive. In floating-point arithmetic with numerical round-off, a pivot that was $-10^{-13}$ was forced to $+10^{-12}$, inverting the sign of that component of $y$ and reversing the correction step $\Delta q = J^T y$.

**Solution:** Preserved the sign during pivot safeguarding:
```c
float diag = A[k][k];
if (fabsf(diag) < 1e-12f) diag = (diag >= 0.0f) ? 1e-12f : -1e-12f;
```
Applied in both forward elimination and back-substitution loops. Stress tested across 10,000 adversarial singularity configurations with 100% finite outputs.

**Lesson:** In linear solvers for optimization and manifold projection, pivot safeguards must preserve the algebraic sign of pivots to avoid flipping descent directions.

---

## Updated Summary Statistics

| Category | Count |
|----------|-------|
| Build failures | 4 |
| Runtime crashes (ASan) | 3 |
| Parse/logic errors | 4 |
| Test infrastructure | 2 |
| Design/API confusion | 2 |
| Semantic validation (M2) | 3 |
| JIT & Solver integration | 5 |
| Automatic differentiation | 1 |
| Kinematics lowering (M3) | 1 |
| **Total** | **25** |

---

## Patterns Learned

1. **Headers must be self-sufficient** — include what you use
2. **`string_view` is a loan** — caller owns buffer lifetime
3. **Never recurse on error paths** — use loops for O(1) stack
4. **`string_view` ≠ C-string** — use `from_chars` or copy for parsing
5. **Clear ownership for token consumption** — caller or callee, not both
6. **`match` (optional) vs `consume` (mandatory)** — naming prevents bugs
7. **Parsers never throw** — safe fallbacks (NaN, dummy nodes) for error recovery
8. **Phase boundaries matter** — parser = syntax, sema = semantics
9. **CTest runs from build dir** — use `CMAKE_SOURCE_DIR` for test resources
10. **Visitor pattern only for polymorphic nodes** — value types handled inline
11. **Standard C for JIT runtimes** — avoid host compiler header extensions in emitted C
12. **Bidirectional search on manifolds** — C-RRT-Connect overcomes tangent drift
13. **Activity analysis collapses product rules** — prune inactive matrix factors before DAG expansion
14. **Topological sorting on kinematic trees** — resolve arbitrary link order and branching structures
15. **Signed pivot preservation** — never invert algebraic signs when clamping near-zero pivots