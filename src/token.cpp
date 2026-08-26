#include <cudro/token.hpp>

#include <unordered_map>

namespace cudro {

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

const char* token_kind_name(TokenKind kind) {
    switch (kind) {
        case TokenKind::EndOfFile: return "end of file";
        case TokenKind::Identifier: return "identifier";
        case TokenKind::Number: return "number";
        case TokenKind::KwRobot: return "'robot'";
        case TokenKind::KwJoint: return "'joint'";
        case TokenKind::KwLink: return "'link'";
        case TokenKind::KwTask: return "'task'";
        case TokenKind::KwPlane: return "'plane'";
        case TokenKind::KwRelativePose: return "'relative_pose'";
        case TokenKind::KwComAbove: return "'com_above'";
        case TokenKind::KwClearance: return "'clearance'";
        case TokenKind::KwType: return "'type'";
        case TokenKind::KwRevolute: return "'revolute'";
        case TokenKind::KwFixed: return "'fixed'";
        case TokenKind::KwAxis: return "'axis'";
        case TokenKind::KwOrigin: return "'origin'";
        case TokenKind::KwLimits: return "'limits'";
        case TokenKind::KwSpheres: return "'spheres'";
        case TokenKind::KwParent: return "'parent'";
        case TokenKind::KwJointRef: return "'joint_ref'";
        case TokenKind::KwMinDistance: return "'min_distance'";
        case TokenKind::KwPointOnLink: return "'point_on_link'";
        case TokenKind::KwNormal: return "'normal'";
        case TokenKind::KwOffset: return "'offset'";
        case TokenKind::LBrace: return "'{'";
        case TokenKind::RBrace: return "'}'";
        case TokenKind::LBracket: return "'['";
        case TokenKind::RBracket: return "']'";
        case TokenKind::Semicolon: return "';'";
        case TokenKind::Comma: return "','";
    }
    return "?";
}

}
