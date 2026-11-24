#include "clay_pnr/ir/clay_ir.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace clay_pnr::ir {

// ---------------------------------------------------------------------------
// 1. ClayDevice / ClayIntentEntry / ClayFile helpers
// ---------------------------------------------------------------------------

double ClayDevice::param_or(const std::string& key, double fallback) const noexcept {
    for (const auto& kv : params) {
        if (kv.first == key) return kv.second;
    }
    return fallback;
}

bool ClayDevice::has_param(const std::string& key) const noexcept {
    for (const auto& kv : params) {
        if (kv.first == key) return true;
    }
    return false;
}

std::string ClayDevice::pin_or(const std::string& key, const std::string& fallback) const {
    for (const auto& kv : pins) {
        if (kv.first == key) return kv.second;
    }
    return fallback;
}

std::string ClayDevice::str_param_or(const std::string& key, const std::string& fallback) const {
    for (const auto& kv : str_params) {
        if (kv.first == key) return kv.second;
    }
    return fallback;
}

std::string ClayIntentEntry::kwarg_or(const std::string& key, const std::string& fallback) const {
    for (const auto& kv : kwargs) {
        if (kv.first == key) return kv.second;
    }
    return fallback;
}

const std::vector<std::string>* ClayIntentEntry::list_kwarg(const std::string& key) const noexcept {
    for (const auto& kv : list_kwargs) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

const ClayLayoutIntent* ClayFile::intent_for_cell(const std::string& cname) const noexcept {
    for (const auto& li : layout_intents) {
        if (li.cell_name == cname) return &li;
    }
    return nullptr;
}

const ClayAnnotateBlock* ClayFile::annotations_for_cell(const std::string& cname) const noexcept {
    for (const auto& ab : annotations) {
        if (ab.cell_name == cname) return &ab;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// 2. Exact SI unit table (parity with clay/parser_modules/_constants.py)
// ---------------------------------------------------------------------------

namespace {

bool lookup_simple_unit(const std::string& u, double& out) noexcept {
    static const std::unordered_map<std::string, double> kUnits = {
        {"Hz", 1.0}, {"kHz", 1e3}, {"MHz", 1e6}, {"GHz", 1e9},
        {"V", 1.0}, {"mV", 1e-3}, {"uV", 1e-6},
        {"A", 1.0}, {"mA", 1e-3}, {"uA", 1e-6}, {"nA", 1e-9}, {"pA", 1e-12},
        {"F", 1.0}, {"pF", 1e-12}, {"fF", 1e-15}, {"nF", 1e-9}, {"uF", 1e-6},
        {"Ohm", 1.0}, {"kOhm", 1e3}, {"MOhm", 1e6},
        {"H", 1.0}, {"pH", 1e-12}, {"nH", 1e-9}, {"uH", 1e-6}, {"mH", 1e-3},
        {"s", 1.0}, {"ms", 1e-3}, {"us", 1e-6}, {"ns", 1e-9}, {"ps", 1e-12},
        {"m", 1.0}, {"mm", 1e-3}, {"um", 1e-6}, {"nm", 1e-9},
        {"deg", 1.0}, {"rad", 57.2957795},
        {"dB", 1.0}, {"dBm", 1.0},
        {"k", 1e3}, {"u", 1e-6}, {"n", 1e-9}, {"p", 1e-12}, {"f", 1e-15},
    };
    auto it = kUnits.find(u);
    if (it == kUnits.end()) return false;
    out = it->second;
    return true;
}

std::string canonicalize_pin(const std::string& dev_kind, const std::string& dev_type, const std::string& raw) {
    if (dev_kind == "device") {
        if (dev_type == "pnp" || dev_type == "npn") {
            if (raw == "collector") return "c";
            if (raw == "base") return "b";
            if (raw == "emitter") return "e";
            return raw;
        }
        if (raw == "drain") return "d";
        if (raw == "gate") return "g";
        if (raw == "source") return "s";
        if (raw == "body") return "b";
    }
    return raw;
}

bool is_functional_role_kw(const std::string& kw) {
    static const std::unordered_set<std::string> kFuns = {
        "functional_group", "input_transconductor", "output_buffer", "feedforward_path",
        "dominant_pole_cap", "compensation_zero", "tail_current_source", "cascode_helper",
        "pole_tracking_cap", "dual_feedforward_pair", "loop_filter_pole", "current_mirror",
        "diode_load", "common_mode_feedback", "sample_hold_switch", "regulated_cascode",
        "gain_boost_amp", "output_class_AB", "dynamic_bias", "bandgap_core",
        "dropout_pass_device", "ldo_feedback_divider", "ptat_branch", "pll_loop_filter",
    };
    return kFuns.find(kw) != kFuns.end();
}

bool is_role_kw(const std::string& kw) {
    return kw == "pmos_pair" || kw == "nmos_pair" ||
           kw == "pmos_mirror" || kw == "nmos_mirror" ||
           kw == "pmos_cs" || kw == "nmos_cs";
}

bool is_digital_kw(const std::string& kw) {
    return kw == "inv" || kw == "nand2" || kw == "dff" || kw == "dff_async_clear";
}

// ---------------------------------------------------------------------------
// 3. Lexer
// ---------------------------------------------------------------------------

enum class TokKind : uint8_t {
    Eof, Ident, Number, String, RowRange,
    At, Colon, ColonColon, ColonEq, Semi, Comma, Dot,
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Arrow, Cascade, Eq, EqEq, BangEq, Lt, Le, Gt, Ge,
    Plus, Minus, Star, StarStar, Slash, AndAnd, OrOr, Bang, Percent,
};

struct Token {
    TokKind kind = TokKind::Eof;
    std::string text;
    double num_val = 0.0;
    int32_t line = 1;
    int32_t col = 1;
};

bool lex_clay(const std::string& src, std::vector<Token>& out, std::string& err, int32_t& err_line, int32_t& err_col) {
    out.clear();
    out.reserve(src.size() / 4 + 16);
    size_t i = 0;
    const size_t n = src.size();
    int32_t line = 1, col = 1;

    auto advance = [&]() -> char {
        const char c = src[i++];
        if (c == '\n') { ++line; col = 1; } else { ++col; }
        return c;
    };

    // Skip UTF-8 BOM if present
    if (n >= 3 && static_cast<unsigned char>(src[0]) == 0xEF &&
        static_cast<unsigned char>(src[1]) == 0xBB &&
        static_cast<unsigned char>(src[2]) == 0xBF) {
        i = 3;
    }

    while (i < n) {
        const char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            advance();
            continue;
        }
        // Comments: // ... or /* ... */
        if (c == '/' && i + 1 < n && src[i + 1] == '/') {
            while (i < n && src[i] != '\n') advance();
            continue;
        }
        if (c == '/' && i + 1 < n && src[i + 1] == '*') {
            advance(); advance();
            while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) advance();
            if (i + 1 < n) { advance(); advance(); }
            continue;
        }

        const int32_t tline = line, tcol = col;

        // Strings ('...' or "...")
        if (c == '"' || c == '\'') {
            const char quote = advance();
            std::string val;
            bool closed = false;
            while (i < n) {
                const char ch = advance();
                if (ch == quote) { closed = true; break; }
                if (ch == '\\' && i < n) {
                    const char esc = advance();
                    if (esc == 'n') val.push_back('\n');
                    else if (esc == 't') val.push_back('\t');
                    else if (esc == 'r') val.push_back('\r');
                    else val.push_back(esc);
                } else {
                    val.push_back(ch);
                }
            }
            if (!closed) {
                err = "unterminated string literal";
                err_line = tline; err_col = tcol;
                return false;
            }
            out.push_back(Token{TokKind::String, std::move(val), 0.0, tline, tcol});
            continue;
        }

        // Identifiers
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const size_t start = i;
            while (i < n && (std::isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_')) {
                advance();
            }
            out.push_back(Token{TokKind::Ident, src.substr(start, i - start), 0.0, tline, tcol});
            continue;
        }

        // Numbers or RowRange (e.g. 0..3)
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
            const size_t start = i;
            bool is_int_digits = std::isdigit(static_cast<unsigned char>(c));
            while (i < n && std::isdigit(static_cast<unsigned char>(src[i]))) advance();
            if (is_int_digits && i + 1 < n && src[i] == '.' && src[i + 1] == '.') {
                advance(); advance();
                while (i < n && std::isdigit(static_cast<unsigned char>(src[i]))) advance();
                out.push_back(Token{TokKind::RowRange, src.substr(start, i - start), 0.0, tline, tcol});
                continue;
            }
            if (i < n && src[i] == '.') {
                advance();
                while (i < n && std::isdigit(static_cast<unsigned char>(src[i]))) advance();
            }
            if (i < n && (src[i] == 'e' || src[i] == 'E')) {
                size_t save_i = i;
                int32_t save_l = line, save_c = col;
                advance();
                if (i < n && (src[i] == '+' || src[i] == '-')) advance();
                if (i < n && std::isdigit(static_cast<unsigned char>(src[i]))) {
                    while (i < n && std::isdigit(static_cast<unsigned char>(src[i]))) advance();
                } else {
                    i = save_i; line = save_l; col = save_c;
                }
            }
            const std::string txt = src.substr(start, i - start);
            const double val = std::strtod(txt.c_str(), nullptr);
            out.push_back(Token{TokKind::Number, txt, val, tline, tcol});
            continue;
        }

        // Two-char punctuation/operators
        if (i + 1 < n) {
            const char c2 = src[i + 1];
            if (c == ':' && c2 == ':') { advance(); advance(); out.push_back(Token{TokKind::ColonColon, "::", 0.0, tline, tcol}); continue; }
            if (c == ':' && c2 == '=') { advance(); advance(); out.push_back(Token{TokKind::ColonEq, ":=", 0.0, tline, tcol}); continue; }
            if (c == '-' && c2 == '>') { advance(); advance(); out.push_back(Token{TokKind::Arrow, "->", 0.0, tline, tcol}); continue; }
            if (c == '>' && c2 == '>') { advance(); advance(); out.push_back(Token{TokKind::Cascade, ">>", 0.0, tline, tcol}); continue; }
            if (c == '=' && c2 == '=') { advance(); advance(); out.push_back(Token{TokKind::EqEq, "==", 0.0, tline, tcol}); continue; }
            if (c == '!' && c2 == '=') { advance(); advance(); out.push_back(Token{TokKind::BangEq, "!=", 0.0, tline, tcol}); continue; }
            if (c == '<' && c2 == '=') { advance(); advance(); out.push_back(Token{TokKind::Le, "<=", 0.0, tline, tcol}); continue; }
            if (c == '>' && c2 == '=') { advance(); advance(); out.push_back(Token{TokKind::Ge, ">=", 0.0, tline, tcol}); continue; }
            if (c == '*' && c2 == '*') { advance(); advance(); out.push_back(Token{TokKind::StarStar, "**", 0.0, tline, tcol}); continue; }
            if (c == '&' && c2 == '&') { advance(); advance(); out.push_back(Token{TokKind::AndAnd, "&&", 0.0, tline, tcol}); continue; }
            if (c == '|' && c2 == '|') { advance(); advance(); out.push_back(Token{TokKind::OrOr, "||", 0.0, tline, tcol}); continue; }
        }

        // Single-char tokens
        advance();
        TokKind tk = TokKind::Eof;
        switch (c) {
            case '@': tk = TokKind::At; break;
            case ':': tk = TokKind::Colon; break;
            case ';': tk = TokKind::Semi; break;
            case ',': tk = TokKind::Comma; break;
            case '.': tk = TokKind::Dot; break;
            case '(': tk = TokKind::LParen; break;
            case ')': tk = TokKind::RParen; break;
            case '{': tk = TokKind::LBrace; break;
            case '}': tk = TokKind::RBrace; break;
            case '[': tk = TokKind::LBracket; break;
            case ']': tk = TokKind::RBracket; break;
            case '=': tk = TokKind::Eq; break;
            case '<': tk = TokKind::Lt; break;
            case '>': tk = TokKind::Gt; break;
            case '+': tk = TokKind::Plus; break;
            case '-': tk = TokKind::Minus; break;
            case '*': tk = TokKind::Star; break;
            case '/': tk = TokKind::Slash; break;
            case '!': tk = TokKind::Bang; break;
            case '%': tk = TokKind::Percent; break;
            default:
                err = std::string("unexpected character '") + c + "'";
                err_line = tline; err_col = tcol;
                return false;
        }
        out.push_back(Token{tk, std::string(1, c), 0.0, tline, tcol});
    }

    out.push_back(Token{TokKind::Eof, "", 0.0, line, col});
    return true;
}

// ---------------------------------------------------------------------------
// 4. Recursive-Descent Parser
// ---------------------------------------------------------------------------

struct ClayParser {
    const std::vector<Token>& toks;
    size_t pos = 0;
    std::string err;
    int32_t err_line = 0;
    int32_t err_col = 0;

    const Token& peek(size_t off = 0) const noexcept {
        const size_t idx = pos + off;
        return (idx < toks.size()) ? toks[idx] : toks.back();
    }
    const Token& bump() noexcept {
        const Token& t = peek();
        if (pos + 1 < toks.size()) ++pos;
        return t;
    }
    bool at(TokKind k) const noexcept { return peek().kind == k; }
    bool at_kw(const char* kw) const noexcept { return peek().kind == TokKind::Ident && peek().text == kw; }

    bool fail(const std::string& msg) {
        if (err.empty()) {
            err = msg;
            err_line = peek().line;
            err_col = peek().col;
        }
        return false;
    }
    bool expect(TokKind k, const char* what) {
        if (at(k)) { bump(); return true; }
        return fail(std::string("expected ") + what + ", got '" + peek().text + "'");
    }
    bool expect_ident(std::string& out, const char* what) {
        if (at(TokKind::Ident)) { out = bump().text; return true; }
        return fail(std::string("expected ") + what + ", got '" + peek().text + "'");
    }

    bool skip_balanced(TokKind open_k, TokKind close_k) {
        if (!expect(open_k, "opening delimiter")) return false;
        int depth = 1;
        while (depth > 0 && !at(TokKind::Eof)) {
            if (at(open_k)) ++depth;
            else if (at(close_k)) --depth;
            bump();
        }
        if (depth != 0) return fail("unclosed block delimiter");
        return true;
    }

    bool parse_dotted_name(std::string& out) {
        if (!expect_ident(out, "identifier")) return false;
        while (at(TokKind::Dot)) {
            bump();
            std::string seg;
            if (!expect_ident(seg, "dotted segment")) return false;
            out.push_back('.');
            out += seg;
        }
        return true;
    }

    bool parse_number(double& out_val, std::string* raw_out = nullptr) {
        double sign = 1.0;
        std::string raw;
        if (at(TokKind::Minus)) { sign = -1.0; raw += bump().text; }
        else if (at(TokKind::Plus)) { raw += bump().text; }
        if (!at(TokKind::Number)) return fail("expected number");
        const Token num = bump();
        raw += num.text;
        double val = sign * num.num_val;

        if (at(TokKind::Ident)) {
            // Try compound unit X/Y first, then simple unit X
            if (peek(1).kind == TokKind::Slash && peek(2).kind == TokKind::Ident) {
                const std::string comp = peek(0).text + "/" + peek(2).text;
                double mult = 1.0;
                if (resolve_si_unit(comp, mult)) {
                    bump(); bump(); bump();
                    raw += comp;
                    val *= mult;
                    if (raw_out) *raw_out = raw;
                    out_val = val;
                    return true;
                }
            }
            double mult = 1.0;
            if (resolve_si_unit(peek().text, mult)) {
                raw += bump().text;
                val *= mult;
            }
        }
        if (raw_out) *raw_out = raw;
        out_val = val;
        return true;
    }

    // Constant arithmetic expression parser (for sizing blocks)
    bool parse_expr(double& out) { return parse_expr_add(out); }

    bool parse_expr_add(double& out) {
        if (!parse_expr_mul(out)) return false;
        while (at(TokKind::Plus) || at(TokKind::Minus)) {
            const TokKind op = bump().kind;
            double rhs = 0.0;
            if (!parse_expr_mul(rhs)) return false;
            out = (op == TokKind::Plus) ? (out + rhs) : (out - rhs);
        }
        return true;
    }

    bool parse_expr_mul(double& out) {
        if (!parse_expr_unary(out)) return false;
        while (at(TokKind::Star) || at(TokKind::Slash)) {
            const TokKind op = bump().kind;
            double rhs = 0.0;
            if (!parse_expr_unary(rhs)) return false;
            out = (op == TokKind::Star) ? (out * rhs) : ((rhs != 0.0) ? (out / rhs) : 0.0);
        }
        return true;
    }

    bool parse_expr_unary(double& out) {
        if (at(TokKind::Minus)) { bump(); if (!parse_expr_unary(out)) return false; out = -out; return true; }
        if (at(TokKind::Plus)) { bump(); return parse_expr_unary(out); }
        return parse_expr_pow(out);
    }

    bool parse_expr_pow(double& out) {
        if (!parse_expr_atom(out)) return false;
        if (at(TokKind::StarStar)) {
            bump();
            double rhs = 0.0;
            if (!parse_expr_pow(rhs)) return false;
            out = std::pow(out, rhs);
        }
        return true;
    }

    bool parse_expr_atom(double& out) {
        if (at(TokKind::Number)) return parse_number(out);
        if (at(TokKind::LParen)) {
            bump();
            if (!parse_expr(out)) return false;
            return expect(TokKind::RParen, "')'");
        }
        if (at(TokKind::Ident)) {
            bump();
            if (at(TokKind::LParen)) {
                if (!skip_balanced(TokKind::LParen, TokKind::RParen)) return false;
            }
            out = 0.0;
            return true;
        }
        return fail("expected expression");
    }

    bool parse_id_list(std::vector<std::string>& out) {
        if (!expect(TokKind::LParen, "'('")) return false;
        while (!at(TokKind::RParen) && !at(TokKind::Eof)) {
            std::string id;
            if (!expect_ident(id, "identifier in list")) return false;
            out.push_back(std::move(id));
            if (at(TokKind::Comma)) bump();
            else break;
        }
        return expect(TokKind::RParen, "')'");
    }

    bool parse_device_body(ClayDevice& dev) {
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            if (at_kw("native_param")) {
                bump();
                std::string k;
                if (!expect_ident(k, "native_param key")) return false;
                if (!expect(TokKind::Eq, "'='")) return false;
                if (!at(TokKind::String)) return fail("expected string for native_param");
                dev.str_params.emplace_back("native_param:" + k, bump().text);
                if (!expect(TokKind::Semi, "';'")) return false;
                continue;
            }
            std::string k;
            if (!expect_ident(k, "device field name")) return false;
            if (!expect(TokKind::Eq, "'='")) return false;
            if (at(TokKind::String)) {
                dev.str_params.emplace_back(k, bump().text);
            } else if (at(TokKind::LParen)) {
                std::vector<std::string> ids;
                if (!parse_id_list(ids)) return false;
                for (size_t idx = 0; idx < ids.size(); ++idx) {
                    dev.pins.emplace_back(k + "_" + std::to_string(idx), ids[idx]);
                }
            } else if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                double v = 0.0;
                if (!parse_number(v)) return false;
                dev.params.emplace_back(k, v);
            } else if (at(TokKind::Ident)) {
                const std::string val = bump().text;
                if (k == "model") {
                    dev.str_params.emplace_back(k, val);
                } else {
                    dev.pins.emplace_back(canonicalize_pin(dev.decl_kind, dev.device_type, k), val);
                }
            } else {
                return fail("unexpected device field value");
            }
            if (!expect(TokKind::Semi, "';'")) return false;
        }
        return expect(TokKind::RBrace, "'}'");
    }

    bool parse_functional_group(const std::string& kind, ClayCell& cell) {
        ClayFunctionalGroup fg;
        fg.kind = kind;
        if (!expect_ident(fg.name, "functional group name")) return false;
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            std::string k;
            if (!expect_ident(k, "functional attribute name")) return false;
            if (!expect(TokKind::Eq, "'='")) return false;
            if (at(TokKind::LParen)) {
                std::vector<std::string> ids;
                if (!parse_id_list(ids)) return false;
                if (k == "devices") {
                    fg.devices = ids;
                }
                std::string joined;
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (i) joined.push_back(',');
                    joined += ids[i];
                }
                fg.attrs.emplace_back(k, std::move(joined));
            } else if (at(TokKind::String)) {
                fg.attrs.emplace_back(k, bump().text);
            } else if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                double v = 0.0;
                std::string raw;
                if (!parse_number(v, &raw)) return false;
                fg.attrs.emplace_back(k, std::move(raw));
            } else if (at(TokKind::Ident)) {
                fg.attrs.emplace_back(k, bump().text);
            } else {
                return fail("unexpected functional_group attribute value");
            }
            if (at(TokKind::Comma) || at(TokKind::Semi)) bump();
        }
        if (!expect(TokKind::RBrace, "'}'")) return false;
        cell.functional_groups.push_back(std::move(fg));
        return true;
    }

    bool parse_topology_block(ClayCell& cell) {
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            if (!at(TokKind::Ident)) return fail("expected topology item");
            const std::string kw = peek().text;

            if (kw == "device") {
                const Token kw_tok = bump();
                ClayDevice dev;
                dev.decl_kind = "device";
                dev.span = SourceSpan{kw_tok.line, kw_tok.col};
                if (!expect_ident(dev.name, "device name")) return false;
                if (!expect(TokKind::Colon, "':'")) return false;
                if (!expect_ident(dev.device_type, "device type")) return false;
                if (!parse_device_body(dev)) return false;
                if (!dev.has_param("M")) {
                    dev.params.emplace_back("M", 1.0);
                }
                cell.devices.push_back(std::move(dev));
            } else if (kw == "res" || kw == "cap" || kw == "ind" || kw == "varactor") {
                const Token kw_tok = bump();
                ClayDevice dev;
                dev.decl_kind = kw;
                dev.device_type = kw;
                dev.span = SourceSpan{kw_tok.line, kw_tok.col};
                if (!expect_ident(dev.name, "passive device name")) return false;
                if (!parse_device_body(dev)) return false;
                cell.devices.push_back(std::move(dev));
            } else if (is_role_kw(kw) || is_digital_kw(kw)) {
                const Token kw_tok = bump();
                ClayDevice dev;
                dev.decl_kind = kw;
                dev.device_type = kw;
                dev.span = SourceSpan{kw_tok.line, kw_tok.col};
                if (!expect_ident(dev.name, "role/digital name")) return false;
                if (!parse_device_body(dev)) return false;
                cell.devices.push_back(std::move(dev));
            } else if (kw == "instance") {
                bump();
                ClayInstance inst;
                if (!expect_ident(inst.name, "instance name")) return false;
                if (!expect(TokKind::Eq, "'='")) return false;
                if (!parse_dotted_name(inst.target)) return false;
                if (at(TokKind::ColonColon)) {
                    bump();
                    std::string cname;
                    if (!expect_ident(cname, "cell name after '::'")) return false;
                    inst.target += "::" + cname;
                }
                if (!expect(TokKind::LBrace, "'{'")) return false;
                while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
                    std::string p, net;
                    if (!expect_ident(p, "port name")) return false;
                    if (!expect(TokKind::Eq, "'='")) return false;
                    if (!expect_ident(net, "net name")) return false;
                    if (!expect(TokKind::Semi, "';'")) return false;
                    inst.port_map.emplace_back(std::move(p), std::move(net));
                }
                if (!expect(TokKind::RBrace, "'}'")) return false;
                if (at(TokKind::Semi)) bump();
                cell.instances.push_back(std::move(inst));
            } else if (is_functional_role_kw(kw)) {
                bump();
                if (!parse_functional_group(kw, cell)) return false;
            } else if (peek(1).kind == TokKind::Eq) {
                // topology_field: key = string | number | ident | object_literal ;
                bump(); bump();
                if (at(TokKind::LBrace)) {
                    if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
                } else if (at(TokKind::String) || at(TokKind::Ident)) {
                    bump();
                } else if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                    double v = 0.0;
                    if (!parse_number(v)) return false;
                } else {
                    return fail("unexpected topology_field value");
                }
                if (!expect(TokKind::Semi, "';'")) return false;
            } else {
                return fail("unexpected token in topology block: '" + kw + "'");
            }
        }
        return expect(TokKind::RBrace, "'}'");
    }

    bool parse_cell_def(ClayFile& file, SourceSpan cell_span) {
        std::string cname;
        if (!expect_ident(cname, "cell name")) return false;
        // cell_alias_def: cell Foo = pkg::Bar { ... };
        if (at(TokKind::Eq)) {
            bump();
            std::string target;
            while (!at(TokKind::LBrace) && !at(TokKind::Semi) && !at(TokKind::Eof)) {
                target += bump().text;
            }
            if (at(TokKind::LBrace)) {
                if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
            }
            if (at(TokKind::Semi)) bump();
            file.cell_aliases.push_back(ClayCellAlias{std::move(cname), std::move(target), cell_span});
            return true;
        }
        ClayCell cell;
        cell.name = std::move(cname);
        cell.span = cell_span;
        if (!expect(TokKind::LParen, "'('")) return false;
        while (!at(TokKind::RParen) && !at(TokKind::Eof)) {
            ClayPort p;
            if (!expect_ident(p.name, "port name")) return false;
            if (at(TokKind::Colon)) {
                bump();
                if (!expect_ident(p.role, "port role")) return false;
            }
            cell.ports.push_back(std::move(p));
            if (at(TokKind::Comma)) bump();
            else break;
        }
        if (!expect(TokKind::RParen, "')'")) return false;
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            if (!at(TokKind::Ident)) return fail("expected cell body block");
            const std::string bkw = bump().text;
            if (bkw == "topology") {
                if (!parse_topology_block(cell)) return false;
            } else if (bkw == "sizing") {
                if (!expect(TokKind::LBrace, "'{'")) return false;
                while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
                    std::string k;
                    if (!expect_ident(k, "sizing key")) return false;
                    if (!expect(TokKind::Eq, "'='")) return false;
                    double v = 0.0;
                    if (!parse_expr(v)) return false;
                    if (!expect(TokKind::Semi, "';'")) return false;
                    cell.sizing.emplace_back(std::move(k), v);
                }
                if (!expect(TokKind::RBrace, "'}'")) return false;
            } else if (bkw == "spec") {
                if (!expect(TokKind::LBrace, "'{'")) return false;
                while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
                    ClaySpecField sf;
                    sf.span = SourceSpan{peek().line, peek().col};
                    if (!expect_ident(sf.name, "spec metric")) return false;
                    if (!expect(TokKind::Colon, "':'")) return false;
                    if (!expect_ident(sf.type_id, "spec type")) return false;
                    if (at(TokKind::Ge) || at(TokKind::Le) || at(TokKind::Eq) ||
                        at(TokKind::Gt) || at(TokKind::Lt)) {
                        sf.op = bump().text;
                    } else {
                        return fail("expected comparison operator in spec");
                    }
                    if (!parse_number(sf.value)) return false;
                    while (at_kw("across_corners") || at_kw("with_margin")) {
                        bump();
                        if (!skip_balanced(TokKind::LParen, TokKind::RParen)) return false;
                    }
                    if (!expect(TokKind::Semi, "';'")) return false;
                    cell.specs.push_back(std::move(sf));
                }
                if (!expect(TokKind::RBrace, "'}'")) return false;
            } else if (bkw == "corners") {
                if (!expect(TokKind::LBrace, "'{'")) return false;
                while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
                    std::string c;
                    if (!expect_ident(c, "corner name")) return false;
                    cell.corners.push_back(std::move(c));
                    if (at(TokKind::Semi)) bump();
                }
                if (!expect(TokKind::RBrace, "'}'")) return false;
            } else if (bkw == "temps") {
                if (!expect(TokKind::LBrace, "'{'")) return false;
                while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
                    double sign = 1.0;
                    if (at(TokKind::Minus)) { sign = -1.0; bump(); }
                    else if (at(TokKind::Plus)) { bump(); }
                    if (!at(TokKind::Number)) return fail("expected temperature number");
                    const double tv = sign * bump().num_val;
                    if (at_kw("C")) bump();
                    cell.temps.push_back(tv);
                    if (at(TokKind::Semi)) bump();
                }
                if (!expect(TokKind::RBrace, "'}'")) return false;
            } else if (bkw == "bias_tree") {
                if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
            } else if (bkw == "combinator") {
                std::string dummy;
                if (!expect_ident(dummy, "combinator name")) return false;
                if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
            } else {
                return fail("unexpected cell body block: '" + bkw + "'");
            }
        }
        if (!expect(TokKind::RBrace, "'}'")) return false;
        file.cells.push_back(std::move(cell));
        return true;
    }

    bool parse_layout_intent(ClayFile& file, SourceSpan li_span) {
        ClayLayoutIntent li;
        li.span = li_span;
        if (!expect_ident(li.cell_name, "layout_intent cell name")) return false;
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            const Token at_tok = peek();
            if (!expect(TokKind::At, "'@'")) return false;
            ClayIntentEntry ent;
            ent.span = SourceSpan{at_tok.line, at_tok.col};
            if (!expect_ident(ent.kind, "layout_intent kind")) return false;
            if (at(TokKind::LParen)) {
                bump();
                while (!at(TokKind::RParen) && !at(TokKind::Eof)) {
                    if (at(TokKind::Ident) && peek(1).kind == TokKind::Eq) {
                        const std::string k = bump().text;
                        bump();  // '='
                        if (at(TokKind::LParen)) {
                            std::vector<std::string> ids;
                            if (!parse_id_list(ids)) return false;
                            ent.list_kwargs.emplace_back(k, std::move(ids));
                        } else if (at(TokKind::String) || at(TokKind::Ident) || at(TokKind::RowRange)) {
                            ent.kwargs.emplace_back(k, bump().text);
                        } else if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                            double v = 0.0;
                            std::string raw;
                            if (!parse_number(v, &raw)) return false;
                            ent.kwargs.emplace_back(k, std::move(raw));
                        } else {
                            return fail("unexpected layout_intent kwarg value");
                        }
                    } else if (at(TokKind::Ident) && peek(1).kind == TokKind::Arrow) {
                        // @shield(aggressor -> victim, shield_net)
                        ent.positional.push_back(bump().text);
                        bump();  // '->'
                        std::string victim;
                        if (!expect_ident(victim, "shield victim")) return false;
                        ent.positional.push_back(std::move(victim));
                    } else if (at(TokKind::String) || at(TokKind::Ident) || at(TokKind::RowRange)) {
                        ent.positional.push_back(bump().text);
                    } else if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                        double v = 0.0;
                        std::string raw;
                        if (!parse_number(v, &raw)) return false;
                        ent.positional.push_back(std::move(raw));
                    } else {
                        return fail("unexpected layout_intent argument");
                    }
                    if (at(TokKind::Comma)) bump();
                    else break;
                }
                if (!expect(TokKind::RParen, "')'")) return false;
            }
            if (!expect(TokKind::Semi, "';'")) return false;
            li.entries.push_back(std::move(ent));
        }
        if (!expect(TokKind::RBrace, "'}'")) return false;
        file.layout_intents.push_back(std::move(li));
        return true;
    }

    bool parse_annotate(ClayFile& file) {
        ClayAnnotateBlock ab;
        if (!expect_ident(ab.cell_name, "annotate cell name")) return false;
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            ClayAnnotateEntry ent;
            if (at(TokKind::At)) {
                bump();
                std::string tgt;
                if (!expect_ident(tgt, "target after '@'")) return false;
                ent.target = (tgt == "cell") ? "__cell__" : ("@" + tgt);
            } else {
                if (!expect_ident(ent.target, "annotation target")) return false;
            }
            if (!expect(TokKind::Colon, "':'")) return false;
            if (!expect(TokKind::At, "'@'")) return false;
            if (!expect_ident(ent.kind, "annotation kind")) return false;
            if (at(TokKind::LParen)) {
                bump();
                while (!at(TokKind::RParen) && !at(TokKind::Eof)) {
                    if (at(TokKind::Ident) && peek(1).kind == TokKind::Eq) {
                        const std::string k = bump().text;
                        bump();
                        if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                            double v = 0.0;
                            std::string raw;
                            if (!parse_number(v, &raw)) return false;
                            ent.kwargs.emplace_back(k, std::move(raw));
                        } else if (at(TokKind::Ident) || at(TokKind::String)) {
                            ent.kwargs.emplace_back(k, bump().text);
                        } else {
                            return fail("unexpected annotation kwarg value");
                        }
                    } else if (at(TokKind::Number) || at(TokKind::Minus) || at(TokKind::Plus)) {
                        double v = 0.0;
                        std::string raw;
                        if (!parse_number(v, &raw)) return false;
                        ent.args.push_back(std::move(raw));
                    } else if (at(TokKind::Ident) || at(TokKind::String)) {
                        ent.args.push_back(bump().text);
                    } else {
                        return fail("unexpected annotation argument");
                    }
                    if (at(TokKind::Comma)) bump();
                    else break;
                }
                if (!expect(TokKind::RParen, "')'")) return false;
            }
            if (!expect(TokKind::Semi, "';'")) return false;
            ab.entries.push_back(std::move(ent));
        }
        if (!expect(TokKind::RBrace, "'}'")) return false;
        file.annotations.push_back(std::move(ab));
        return true;
    }

    bool parse_flow(ClayFile& file) {
        ClayFlowBlock fb;
        if (!expect_ident(fb.cell_name, "flow cell name")) return false;
        if (!expect(TokKind::LBrace, "'{'")) return false;
        while (!at(TokKind::RBrace) && !at(TokKind::Eof)) {
            std::string kw;
            if (!expect_ident(kw, "flow entry")) return false;
            if (kw == "pdk") {
                if (!at(TokKind::String)) return fail("expected pdk string in flow");
                fb.pdk = bump().text;
            } else if (kw == "corners") {
                if (!parse_id_list(fb.corners)) return false;
            } else if (kw == "steps") {
                if (!parse_id_list(fb.steps)) return false;
            } else if (kw == "layout") {
                if (!expect(TokKind::Eq, "'='")) return false;
                if (!parse_dotted_name(fb.layout_module)) return false;
            } else if (kw == "policy") {
                if (!expect(TokKind::Eq, "'='")) return false;
                if (!expect_ident(fb.policy, "policy name")) return false;
            } else {
                return fail("unexpected flow entry: '" + kw + "'");
            }
            if (!expect(TokKind::Semi, "';'")) return false;
        }
        if (!expect(TokKind::RBrace, "'}'")) return false;
        file.flows.push_back(std::move(fb));
        return true;
    }

    bool parse_file(ClayFile& file) {
        while (!at(TokKind::Eof)) {
            if (!at(TokKind::Ident)) return fail("expected top-level declaration");
            const Token kw_tok = bump();
            const std::string& kw = kw_tok.text;
            const SourceSpan sp{kw_tok.line, kw_tok.col};
            if (kw == "package") {
                if (!parse_dotted_name(file.package_name)) return false;
                if (!expect(TokKind::Semi, "';'")) return false;
            } else if (kw == "pdk") {
                if (!at(TokKind::String)) return fail("expected pdk string");
                file.pdk = bump().text;
                if (!expect(TokKind::Semi, "';'")) return false;
            } else if (kw == "import") {
                std::string imp;
                std::string alias;
                if (at(TokKind::String)) imp = bump().text;
                else if (!parse_dotted_name(imp)) return false;
                if (at_kw("as")) {
                    bump();
                    if (!expect_ident(alias, "import alias")) return false;
                }
                if (!expect(TokKind::Semi, "';'")) return false;
                file.imports.push_back(imp);
                file.import_decls.push_back(ClayImport{std::move(imp), std::move(alias), sp});
            } else if (kw == "cell") {
                if (!parse_cell_def(file, sp)) return false;
            } else if (kw == "layout_intent") {
                if (!parse_layout_intent(file, sp)) return false;
            } else if (kw == "annotate") {
                if (!parse_annotate(file)) return false;
                if (!file.annotations.empty()) file.annotations.back().span = sp;
            } else if (kw == "flow") {
                if (!parse_flow(file)) return false;
                if (!file.flows.empty()) file.flows.back().span = sp;
            } else if (kw == "policy") {
                std::string pname;
                if (!expect_ident(pname, "policy name")) return false;
                if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
                file.policy_names.push_back(std::move(pname));
                file.policy_count += 1;
            } else if (kw == "quality") {
                std::string qname;
                if (!expect_ident(qname, "quality name")) return false;
                if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
                file.quality_names.push_back(std::move(qname));
                file.quality_count += 1;
            } else if (kw == "verify") {
                std::string vname;
                if (at(TokKind::Ident)) vname = peek().text;
                while (!at(TokKind::LBrace) && !at(TokKind::Semi) && !at(TokKind::Eof)) bump();
                if (at(TokKind::LBrace)) {
                    if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
                } else if (at(TokKind::Semi)) {
                    bump();
                }
                if (!vname.empty()) file.verify_names.push_back(std::move(vname));
                file.verify_count += 1;
            } else if (kw == "fn") {
                std::string fname;
                if (!expect_ident(fname, "fn name")) return false;
                if (!skip_balanced(TokKind::LParen, TokKind::RParen)) return false;
                if (!skip_balanced(TokKind::LBrace, TokKind::RBrace)) return false;
                file.fn_names.push_back(std::move(fname));
                file.fn_count += 1;
            } else {
                return fail("unknown top-level declaration '" + kw + "'");
            }
        }
        return true;
    }
};

}  // namespace

bool resolve_si_unit(const std::string& unit, double& out_multiplier) noexcept {
    const size_t slash = unit.find('/');
    if (slash == std::string::npos) {
        return lookup_simple_unit(unit, out_multiplier);
    }
    if (unit.find('/', slash + 1) != std::string::npos) return false;
    double num = 1.0, den = 1.0;
    if (!lookup_simple_unit(unit.substr(0, slash), num)) return false;
    if (!lookup_simple_unit(unit.substr(slash + 1), den) || den == 0.0) return false;
    out_multiplier = num / den;
    return true;
}

ParseResult parse_clay_source(const std::string& source) {
    ParseResult res{};
    std::vector<Token> toks;
    if (!lex_clay(source, toks, res.error, res.line, res.col)) {
        res.ok = false;
        return res;
    }
    ClayParser parser{toks, 0, "", 0, 0};
    if (!parser.parse_file(res.file)) {
        res.ok = false;
        res.error = parser.err;
        res.line = parser.err_line;
        res.col = parser.err_col;
        return res;
    }
    res.ok = true;
    return res;
}

ParseResult parse_clay_file(const std::string& path) {
    ParseResult res{};
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        res.ok = false;
        res.error = "failed to open .clay file: " + path;
        return res;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_clay_source(ss.str());
}

}  // namespace clay_pnr::ir
