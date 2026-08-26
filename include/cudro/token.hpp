#pragma once

#include <cudro/diagnostic.hpp>
#include <optional>
#include <string_view>

namespace cudro {

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

struct Token {
    TokenKind kind;
    std::string_view text;
    Location where;
};

std::optional<TokenKind> lookup_keyword(std::string_view word);
const char* token_kind_name(TokenKind kind);

}
