# Chapter 06: Recursive Descent Parser

## Files Covered
- `include/cudro/parser.hpp`
- `src/parser.cpp`

---

## 1. Architectural Purpose
The parser consumes the token stream from the lexer and builds the **Abstract Syntax Tree (AST)**. It implements the formal grammar of `.cudro` via **recursive descent**: one C++ function per grammar rule.

### Grammar Coverage
```
spec         := (robot | task | clearance)*
robot        := 'robot' IDENT '{' joint* link* '}'
joint        := 'joint' IDENT '{' 'type' ('revolute'|'fixed') ';'
               'axis' vec3 ';' 'origin' vec3 ';'
               ('limits' '[' number ',' number ']';')? '}'
link         := 'link' IDENT '{' ('spheres' sphere+)? ('parent' IDENT ';')?
               ('joint_ref' IDENT ';')? '}'
task         := 'task' IDENT '{' 'link' IDENT ';' plane_c+ '}'
plane_c      := 'plane' '{' 'point_on_link' vec3 ';' 'normal' vec3 ';'
               'offset' number ';' '}'
clearance    := 'clearance' '{' 'min_distance' number ';' '}'
vec3         := '[' number ',' number ',' number ']'
sphere       := '[' number ',' number ',' number ',' number ']'
```

---

## 2. Parser Class Interface (`include/cudro/parser.hpp`)

```cpp
class Parser {
public:
    Parser(const std::vector<Token>& tokens, DiagnosticBag& diags)
        : tokens_(tokens), diags_(diags), pos_(0) {}

    Spec parse();

private:
    // Navigation
    const Token& peek(size_t offset = 0) const;
    const Token& current() const;
    bool check(TokenKind kind) const;
    Token advance();
    bool match(TokenKind kind);
    Token consume(TokenKind kind, const char* message);

    // Error recovery
    void synchronize();
    void report_error(const Token& token, const char* message);

    // Grammar entry points
    Spec parse_spec();
    std::unique_ptr<RobotDecl> parse_robot();
    std::unique_ptr<JointDecl> parse_joint();
    std::unique_ptr<LinkDecl> parse_link();
    std::unique_ptr<TaskDecl> parse_task();
    std::unique_ptr<ClearanceDecl> parse_clearance();
    std::unique_ptr<PlaneConstraint> parse_plane_constraint();
    Vec3 parse_vec3();
    Sphere parse_sphere();
    double parse_number_safe();  // Returns NaN on error instead of throwing
    std::string parse_ident();

    const std::vector<Token>& tokens_;
    DiagnosticBag& diags_;
    size_t pos_ = 0;
};
```

### Design: `std::unique_ptr` Ownership
AST nodes are heap-allocated via `std::make_unique`. The parser owns the entire tree and moves it into the `Spec` result. This avoids manual memory management and ensures exception safety.

---

## 2. Core Parsing Helpers (`src/parser.cpp`)

### A. Token Navigation
```cpp
const Token& Parser::peek(size_t offset = 0) const {
    return tokens_[std::min(pos_ + offset, tokens_.size() - 1)];
}

const Token& Parser::current() const {
    return peek(0);
}

bool Parser::check(TokenKind kind) const {
    return current().kind == kind;
}

Token Parser::advance() {
    if (!check(TokenKind::EndOfFile))
        ++pos_;
    return tokens_[pos_ - 1];
}

bool Parser::match(TokenKind kind) {
    if (check(kind)) {
        advance();
        return true;
    }
    return false;
}
```

### B. Consume with Error Recovery (`consume`)
```cpp
Token Parser::consume(TokenKind kind, const char* message) {
    if (check(kind))
        return advance();

    // Report error at current token
    const Token& t = current();
    diags_.error(t.where, std::string(message) + ", got " + token_kind_name(t.kind));

    // Panic-mode recovery: skip until synchronizing token (; or })
    synchronize();

    // Return dummy error token so parsing can continue
    return Token{TokenKind::EndOfFile, "", current().where};
}
```

### C. Safe Number Parsing (`parse_number_safe`)
```cpp
double Parser::parse_number_safe() {
    if (!check(TokenKind::Number)) {
        diags_.error(current().where, "expected number, got " + std::string(token_kind_name(current().kind)));
        synchronize();
        return std::numeric_limits<double>::quiet_NaN();
    }
    Token t = advance();
    // Copy to string for std::stod (requires null-terminated string)
    std::string num_str(t.text);
    size_t pos = 0;
    double val = std::stod(num_str, &pos);
    if (pos != num_str.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return val;
}
```

### Error Recovery Strategy: Panic Mode + Sync Points
- On unexpected token: report diagnostic at exact location.
- **Skip tokens** until a synchronization point: `;` (end of field) or `}` (end of block) or declaration keyword.
- Return dummy `EndOfFile` token so the caller can continue building the AST (with missing fields).
- **Result**: Multiple syntax errors reported in one run, not just the first.

---

## 3. Grammar Rule Implementations

### A. Top-Level Spec (`parse_spec`)
```cpp
Spec Parser::parse_spec() {
    Spec spec;
    while (!check(TokenKind::EndOfFile)) {
        if (check(TokenKind::KwRobot)) {
            spec.robots.push_back(parse_robot());
        } else if (check(TokenKind::KwTask)) {
            spec.tasks.push_back(parse_task());
        } else if (check(TokenKind::KwClearance)) {
            spec.clearances.push_back(parse_clearance());
        } else {
            diags_.error(current().where, "expected 'robot', 'task', or 'clearance'");
            advance();
        }
    }
    return spec;
}
```

### B. Robot Declaration (`parse_robot`)
```cpp
std::unique_ptr<RobotDecl> Parser::parse_robot() {
    auto robot = std::make_unique<RobotDecl>();
    robot->loc = consume(TokenKind::KwRobot, "expected 'robot'").where;
    robot->name = parse_ident();
    consume(TokenKind::LBrace, "expected '{' after robot name");

    while (!check(TokenKind::RBrace) && !check(TokenKind::EndOfFile)) {
        if (check(TokenKind::KwJoint)) {
            robot->joints.push_back(parse_joint());
        } else if (check(TokenKind::KwLink)) {
            robot->links.push_back(parse_link());
        } else {
            diags_.error(current().where, "expected 'joint' or 'link' inside robot");
            advance();
        }
    }
    consume(TokenKind::RBrace, "expected '}' after robot body");
    return robot;
}
```

### C. Joint Declaration (`parse_joint`)
```cpp
std::unique_ptr<JointDecl> Parser::parse_joint() {
    auto joint = std::make_unique<JointDecl>();
    joint->loc = consume(TokenKind::KwJoint, "expected 'joint'").where;
    joint->name = parse_ident();
    consume(TokenKind::LBrace, "expected '{' after joint name");

    // Required: type
    consume(TokenKind::KwType, "expected 'type'");
    Token type_tok = consume(TokenKind::KwRevolute, "expected 'revolute' or 'fixed'");
    if (type_tok.kind == TokenKind::KwFixed) {
        joint->type = "fixed";
    } else {
        joint->type = "revolute";
    }
    consume(TokenKind::Semicolon, "expected ';' after type");

    // Required: axis
    consume(TokenKind::KwAxis, "expected 'axis'");
    joint->axis = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after axis");

    // Required: origin
    consume(TokenKind::KwOrigin, "expected 'origin'");
    joint->origin = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after origin");

    // Optional: limits
    if (match(TokenKind::KwLimits)) {
        consume(TokenKind::LBracket, "expected '[' for limits");
        double lo = parse_number_safe();
        consume(TokenKind::Comma, "expected ',' in limits");
        double hi = parse_number_safe();
        consume(TokenKind::RBracket, "expected ']' for limits");
        if (!std::isnan(lo) && !std::isnan(hi)) {
            joint->limits = std::make_pair(lo, hi);
        }
        consume(TokenKind::Semicolon, "expected ';' after limits");
    }

    consume(TokenKind::RBrace, "expected '}' after joint");
    return joint;
}
```

### D. Plane Constraint (`parse_plane_constraint`)
```cpp
std::unique_ptr<PlaneConstraint> Parser::parse_plane_constraint() {
    auto plane = std::make_unique<PlaneConstraint>();
    // KwPlane already consumed by match() in parse_task()
    plane->loc = consume(TokenKind::LBrace, "expected '{' after 'plane'").where;

    // Optional link override (e.g. for multi-link tasks)
    if (match(TokenKind::KwLink)) {
        plane->link = parse_ident();
        consume(TokenKind::Semicolon, "expected ';' after link");
    }

    consume(TokenKind::KwPointOnLink, "expected 'point_on_link'");
    plane->point_on_link = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after point_on_link");

    consume(TokenKind::KwNormal, "expected 'normal'");
    plane->normal = parse_vec3();
    consume(TokenKind::Semicolon, "expected ';' after normal");

    consume(TokenKind::KwOffset, "expected 'offset'");
    plane->offset = parse_number_safe();
    consume(TokenKind::Semicolon, "expected ';' after offset");

    consume(TokenKind::RBrace, "expected '}' after plane");
    return plane;
}
```

### E. Vec3 Parser (`parse_vec3`)
```cpp
Vec3 Parser::parse_vec3() {
    Vec3 v;
    v.loc = consume(TokenKind::LBracket, "expected '['").where;
    v.x = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    v.y = parse_number_safe();
    consume(TokenKind::Comma, "expected ','");
    v.z = parse_number_safe();
    consume(TokenKind::RBracket, "expected ']'");
    return v;
}
```

---

## 3. Error Recovery Design

The parser uses **panic-mode recovery** with sync tokens (`;`, `}`):
1. `consume` reports error when token doesn't match.
2. `consume` advances through tokens until it hits `;`, `}`, or EOF.
3. Returns dummy `EndOfFile` token so caller's AST node construction proceeds (with missing fields).
4. Parsing continues — subsequent declarations still get parsed.

This means a single typo like `type revolute` (missing semicolon) produces **one error** and allows the rest of the file to parse successfully.

---

## 4. Robotics Context
The parser enforces the declarative grammar that robotics engineers will write (or VLAs will generate). Key properties:
- **Mandatory fields** (`type`, `axis`, `origin`) are strictly required — missing one produces a clear error.
- **Optional fields** (`limits`) are gracefully handled via `std::optional`.
- **Locations** on every AST node enable precise error messages: `spec/panda7.cudro:14:5: error: expected ';' after axis`.

---

## 5. Testing
`tests/test_parser.cpp` covers:
- Valid `panda7.cudro` → full AST with correct structure
- Missing semicolon → single error, rest of file parsed
- Unclosed brace → error at EOF with correct location
- Unknown field name → error with sync recovery
- Multiple errors reported in single pass
- Duplicate joint name detection (deferred to M2 sema phase)