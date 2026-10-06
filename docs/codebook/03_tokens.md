# Chapter 03: Lexical Tokens & Keywords

## Files Covered
- `include/cudro/token.hpp`
- `src/token.cpp`

---

## 1. Architectural Purpose
Tokens are the atomic vocabulary of the compiler. The lexer emits a stream of `Token` objects; the parser consumes them. This chapter covers the token taxonomy, the dedicated keyword kinds strategy, and the lookup table that maps lexemes to enum values.

---

## 2. Token Kind Enumeration (`include/cudro/token.hpp`)

```cpp
enum class TokenKind {
    EndOfFile,
    Identifier,
    Number,
    KwRobot, KwJoint, KwLink, KwTask, KwPlane, KwRelativePose,
    KwComAbove, KwClearance, KwType, KwRevolute, KwFixed, KwAxis,
    KwOrigin, KwLimits, KwSpheres, KwParent, KwJointRef,
    KwMinDistance, KwPointOnLink, KwNormal, KwOffset,
    LBrace, RBrace,
    LBracket, RBracket,
    Semicolon, Comma,
};
```

### Design Decision: Dedicated Keyword Kinds
Many simple lexers return a single `Identifier` kind for all words and force the parser to `strcmp(token.text, "robot")`. We chose **dedicated `Kw*` enum values** for every keyword.

| Approach | Parser Code | Trade-off |
|---|---|---|
| Generic `Identifier` | `if (token.text == "robot")` | Simple lexer, parser full of string compares |
| **Dedicated `KwRobot`** | `expect(TokenKind::KwRobot)` | **Lexer does classification once; parser matches enums** |

This moves work from the parser (hot path, complex logic) to the lexer (single pass, O(1) hash lookup per identifier). The parser stays clean and type-safe.

### Why No `LParen` / `RParen`?
Our `.cudro` grammar uses only `[ ]` for vectors and `{ }` for blocks. No parentheses appear in the grammar, so we omit them — keeping the enum minimal.

---

## 3. Token Structure (`include/cudro/token.hpp`)

```cpp
struct Token {
    TokenKind kind;
    std::string_view text;
    Location where;
};
```
- **`kind`**: The classified token type (enum above).
- **`text`**: Zero-copy view into the original source buffer. This is why the caller must keep the source string alive.
- **`where`**: Source location for error reporting.

---

## 4. Keyword Lookup Table (`src/token.cpp`)

```cpp
namespace {

const std::unordered_map<std::string_view, TokenKind> kKeywords = {
    {"robot", TokenKind::KwRobot},
    {"joint", TokenKind::KwJoint},
    {"link", TokenKind::KwLink},
    {"task", TokenKind::KwTask},
    {"plane", TokenKind::KwPlane},
    {"relative_pose", TokenKind::KwRelativePose},
    {"com_above", TokenKind::KwComAbove},
    {"clearance", TokenKind::KwClearance},
    {"type", TokenKind::KwType},
    {"revolute", TokenKind::KwRevolute},
    {"fixed", TokenKind::KwFixed},
    {"axis", TokenKind::KwAxis},
    {"origin", TokenKind::KwOrigin},
    {"limits", TokenKind::KwLimits},
    {"spheres", TokenKind::KwSpheres},
    {"parent", TokenKind::KwParent},
    {"joint_ref", TokenKind::KwJointRef},
    {"min_distance", TokenKind::KwMinDistance},
    {"point_on_link", TokenKind::KwPointOnLink},
    {"normal", TokenKind::KwNormal},
    {"offset", TokenKind::KwOffset},
};

}

std::optional<TokenKind> lookup_keyword(std::string_view word) {
    auto it = kKeywords.find(word);
    if (it == kKeywords.end())
        return std::nullopt;
    return it->second;
}
```

### Why `std::unordered_map<std::string_view, TokenKind>`?
- **O(1) average lookup** — critical because every identifier in the source triggers one hash lookup.
- **`std::string_view` keys** — zero-copy keys into the same source buffer.
- **Compile-time constant map** — initialized once at program startup; zero runtime overhead per lookup.

### `std::optional<TokenKind>` Return
- Returns `TokenKind` if `word` is a keyword.
- Returns `std::nullopt` if `word` is a user-defined identifier (robot name, link name, etc.).
- The lexer calls `lookup_keyword` on every scanned identifier; if `nullopt`, it emits `Identifier`.

---

## 5. Token Kind to Human-Readable String (`src/token.cpp`)

```cpp
const char* token_kind_name(TokenKind kind) {
    switch (kind) {
        case TokenKind::EndOfFile: return "end of file";
        case TokenKind::Identifier: return "identifier";
        case TokenKind::Number: return "number";
        case TokenKind::KwRobot: return "'robot'";
        // ... all other cases ...
        case TokenKind::Comma: return "','";
    }
    return "?";
}
```
Used by `--dump-tokens` CLI output and diagnostic formatting. Each case returns a human-readable string (e.g., `"'robot'"` with quotes for keywords).

## 6. Complete Token Reference Table

| TokenKind | Lexeme(s) | Category | Purpose in Grammar |
|---|---|---|---|
| **EndOfFile** | (none) | Sentinel | Marks end of token stream; parser termination |
| **Identifier** | `[a-zA-Z_][a-zA-Z0-9_]*` | User name | Robot names, link names, joint names, task names |
| **Number** | `-?\d+(\.\d+)?` | Literal | Floating-point values: coordinates, angles, limits, offsets |
| **KwRobot** | `robot` | Declaration | Starts a robot definition block |
| **KwJoint** | `joint` | Declaration | Starts a joint definition inside robot |
| **KwLink** | `link` | Declaration | Starts a link definition inside robot |
| **KwTask** | `task` | Declaration | Starts a task constraint block |
| **KwPlane** | `plane` | Constraint | Plane manifold constraint: `dot(normal, p) - offset = 0` |
| **KwRelativePose** | `relative_pose` | Constraint | Fixed relative pose between two links (future) |
| **KwComAbove** | `com_above` | Constraint | Center-of-mass height constraint (future) |
| **KwClearance** | `clearance` | Global | Global collision margin setting |
| **KwType** | `type` | Field | Joint type: `revolute` or `fixed` |
| **KwRevolute** | `revolute` | Value | Rotational joint (1 DoF) |
| **KwFixed** | `fixed` | Value | Fixed joint (0 DoF, constant transform) |
| **KwAxis** | `axis` | Field | Joint rotation axis vector `[x,y,z]` |
| **KwOrigin** | `origin` | Field | Joint origin translation `[x,y,z]` |
| **KwLimits** | `limits` | Field | Joint position limits `[min, max]` |
| **KwSpheres** | `spheres` | Field | List of collision spheres for a link |
| **KwParent** | `parent` | Field | Parent link name in kinematic tree |
| **KwJointRef** | `joint_ref` | Field | Joint connecting this link to parent |
| **KwMinDistance** | `min_distance` | Field | Global minimum clearance distance |
| **KwPointOnLink** | `point_on_link` | Field | Point on link for plane constraint (local frame) |
| **KwNormal** | `normal` | Field | Plane normal vector in world frame `[x,y,z]` |
| **KwOffset** | `offset` | Field | Plane offset scalar `d` in `dot(n, p) - d = 0` |
| **LBrace** | `{` | Punctuation | Opens block (robot, joint, link, task, plane, clearance) |
| **RBrace** | `}` | Punctuation | Closes block |
| **LBracket** | `[` | Punctuation | Opens vector literal `[x, y, z]` or `[min, max]` |
| **RBracket** | `]` | Punctuation | Closes vector literal |
| **Semicolon** | `;` | Punctuation | Terminates field declarations |
| **Comma** | `,` | Punctuation | Separates vector components |

---

## 7. Robotics DSL Context
The token kinds mirror the declarative robotics vocabulary:
- **Robot Structure**: `robot`, `joint`, `link`, `type`, `revolute`, `fixed`
- **Kinematics**: `axis`, `origin`, `limits`
- **Collision Geometry**: `spheres`, `parent`, `joint_ref`
- **Task Constraints**: `task`, `plane`, `relative_pose`, `com_above`, `point_on_link`, `normal`, `offset`
- **Global Settings**: `clearance`, `min_distance`

This token taxonomy directly maps to the robotics domain concepts (URDF-like kinematic trees, collision spheres, manifold constraints).