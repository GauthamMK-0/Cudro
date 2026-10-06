# Chapter 04: The Lexer Scanner Engine

## Files Covered
- `include/cudro/lexer.hpp`
- `src/lexer.cpp`

---

## 1. Architectural Purpose
The lexer (scanner) transforms a raw character stream into a classified token stream. It is the only component that touches raw characters; everything downstream operates purely on `Token` objects.

Our lexer implements:
1. **Character-by-character scanning** with lookahead (`peek`, `advance`)
2. **Trivia skipping** (whitespace, `//` comments)
3. **Token classification** (punctuation, numbers, identifiers/keywords)
4. **Error recovery** that never stalls (always advances at least one character)
5. **Both streaming (`next()`) and batch (`tokenize()`) interfaces**

---

## 2. Class Interface (`include/cudro/lexer.hpp`)

```cpp
class Lexer {
public:
    Lexer(std::string_view file, std::string_view source, DiagnosticBag& diags)
        : file_(file), src_(source), diags_(diags) {}

    Token next();
    std::vector<Token> tokenize();

private:
    char peek(size_t offset = 0) const;
    char advance();
    bool at_end() const;
    Location here() const;
    Token make(TokenKind kind, size_t start, Location start_loc);
    void skip_trivia();

    std::string_view file_;
    std::string_view src_;
    DiagnosticBag& diags_;
    size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;
};
```

### Member Variables
| Member | Purpose |
|---|---|
| `file_` | Filename string_view for diagnostic location |
| `src_` | **Original source text** — tokens hold `string_view` into this buffer |
| `diags_` | Reference to the diagnostic bag for error reporting |
| `pos_` | Current byte offset into `src_` |
| `line_`, `col_` | 1-indexed line/column for diagnostics |

---

## 3. Core Implementation (`src/lexer.cpp`)

### A. Trivia Skipping (`skip_trivia`)

```cpp
void Lexer::skip_trivia() {
    while (!at_end()) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
        } else if (c == '/' && peek(1) == '/') {
            while (!at_end() && peek() != '\n')
                advance();
        } else {
            break;
        }
    }
}
```
- Skips spaces, tabs, carriage returns, newlines
- **`//` line comments**: Recognizes `//` and skips until newline
- Does **NOT** skip `/* */` block comments (not in our grammar)

### B. The Main Scan Loop (`next`)

```cpp
Token Lexer::next() {
    while (true) {
        skip_trivia();
        if (at_end())
            return {TokenKind::EndOfFile, {}, here()};

        Location start_loc = here();
        size_t start = pos_;
        char c = peek();

        // 1. Punctuation (single-char tokens)
        if (auto kind = punctuation_kind(c)) {
            advance();
            return make(*kind, start, start_loc);
        }

        // 2. Numbers (with optional leading '-' and optional decimal)
        if (is_digit(c) || (c == '-' && is_digit(peek(1)))) {
            advance();
            while (is_digit(peek()))
                advance();
            if (peek() == '.' ) {
                advance();
                while (is_digit(peek()))
                    advance();
            }
            return make(TokenKind::Number, start, start_loc);
        }

        // 3. Identifiers / Keywords
        if (is_ident_start(c)) {
            while (is_ident_char(peek()))
                advance();
            std::string_view word = src_.substr(start, pos_ - start);
            if (auto kw = lookup_keyword(word))
                return make(*kw, start, start_loc);
            return make(TokenKind::Identifier, start, start_loc);
        }

        // 4. Error Recovery — ALWAYS ADVANCE
        diags_.error(start_loc, "unexpected character '" + std::string(1, c) + "'");
        advance();
    }
}
```

### Key Design Decisions Explained

#### 1. `while (true)` Loop Instead of Recursion
The original prototype used `return next();` in the error case. This caused **unbounded recursion** — one stack frame per garbage character. A 10 MB file of `@@@@...` would overflow the stack. The loop guarantees **O(1) stack usage** regardless of input.

#### 2. Number Scanning: `-` Only When Followed by Digit
```cpp
if (is_digit(c) || (c == '-' && is_digit(peek(1))))
```
This encodes the grammar decision: **`-` is part of a number literal ONLY when followed by a digit**. If we unconditionally treated `-` as number-start:
- `origin [0 - 1, 0, 0]` would tokenize as `Number("-1")` instead of `Number("0")`, `Number("-1")`
- Future arithmetic expressions (`x - 3`) would silently mis-tokenize as `Identifier("x")`, `Number(-3)`

This is **lexical design coupling**: the lexer encodes grammar decisions. If we add arithmetic to constraint expressions, this rule must be revisited consciously.

#### 3. `lookup_keyword` on Every Identifier
```cpp
if (auto kw = lookup_keyword(word))
    return make(*kw, start, start_loc);
return make(TokenKind::Identifier, start, start_loc);
```
Every scanned identifier is checked against the keyword table. Keywords (`robot`, `joint`, `plane`) become dedicated enum values; user names (`panda7`, `ee`) become `Identifier`.

#### 4. Always-Advance Error Recovery
```cpp
diags_.error(start_loc, "unexpected character '" + std::string(1, c) + "'");
advance();
```
Even on invalid input, the lexer **consumes at least one character**. This guarantees:
- The lexer never infinite-loops on garbage input
- Multiple errors in one file are all reported (the parser can continue)
- The "always-advance" invariant is the hallmark of production lexers (Clang, Rustc, GCC)

### C. Batch Tokenization (`tokenize`)

```cpp
std::vector<Token> Lexer::tokenize() {
    std::vector<Token> out;
    while (true) {
        Token t = next();
        bool eof = t.kind == TokenKind::EndOfFile;
        out.push_back(std::move(t));
        if (eof)
            break;
    }
    return out;
}
```
Simple loop collecting tokens until `EndOfFile`. Used by the CLI `--dump-tokens` flag.

---

## 4. Robotics Context
When a perception module (e.g., a vision-language model) generates a `.cudro` spec at runtime, it may produce syntax errors. The lexer's error recovery ensures **every** syntax error is reported with precise location, not just the first one. This is critical for automated spec generation pipelines.

---

## 5. Testing
`tests/test_lexer.cpp` validates:
- Keyword vs identifier distinction (`plane` vs `planeX`)
- Number parsing with signs and decimals (`-2.8973`, `0.333`)
- Comment skipping with correct line tracking
- Error recovery produces diagnostics but continues scanning