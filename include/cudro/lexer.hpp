#pragma once

#include <cudro/diagnostic.hpp>
#include <cudro/token.hpp>

#include <string_view>
#include <vector>

namespace cudro {

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

}
