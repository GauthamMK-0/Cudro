#include <cudro/lexer.hpp>

#include <cctype>
#include <optional>

namespace cudro {

namespace {

bool is_ident_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool is_ident_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

bool is_digit(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

std::optional<TokenKind> punctuation_kind(char c) {
    switch (c) {
        case '{': return TokenKind::LBrace;
        case '}': return TokenKind::RBrace;
        case '[': return TokenKind::LBracket;
        case ']': return TokenKind::RBracket;
        case ';': return TokenKind::Semicolon;
        case ',': return TokenKind::Comma;
        default: return std::nullopt;
    }
}

}

char Lexer::peek(size_t offset) const {
    return pos_ + offset < src_.size() ? src_[pos_ + offset] : '\0';
}

char Lexer::advance() {
    char c = src_[pos_++];
    if (c == '\n') {
        ++line_;
        col_ = 1;
    } else {
        ++col_;
    }
    return c;
}

bool Lexer::at_end() const {
    return pos_ >= src_.size();
}

Location Lexer::here() const {
    return {file_, line_, col_};
}

Token Lexer::make(TokenKind kind, size_t start, Location start_loc) {
    return {kind, src_.substr(start, pos_ - start), start_loc};
}

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

Token Lexer::next() {
    while (true) {
        skip_trivia();
        if (at_end())
            return {TokenKind::EndOfFile, {}, here()};

        Location start_loc = here();
        size_t start = pos_;
        char c = peek();

        if (auto kind = punctuation_kind(c)) {
            advance();
            return make(*kind, start, start_loc);
        }

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

        if (is_ident_start(c)) {
            while (is_ident_char(peek()))
                advance();
            std::string_view word = src_.substr(start, pos_ - start);
            if (auto kw = lookup_keyword(word))
                return make(*kw, start, start_loc);
            return make(TokenKind::Identifier, start, start_loc);
        }

        diags_.error(start_loc, "unexpected character '" + std::string(1, c) + "'");
        advance();
    }
}

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

}
