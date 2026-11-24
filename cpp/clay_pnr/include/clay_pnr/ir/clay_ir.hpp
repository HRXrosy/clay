#pragma once

// Pure C++20 Clay (.clay) DSL IR & Recursive-Descent Parser (Phase 4A).
// Strictly free of <Python.h>, Lark, and third-party dependencies.
// Supports the complete .clay surface grammar:
//   package, pdk, import, cell (ports, topology { device/res/cap/ind/varactor,
//   roles, digital primitives, instances, functional_group }, sizing, spec,
//   corners, temps, bias_tree, combinator), layout_intent, annotate,
//   flow, policy, quality, verify, fn.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace clay_pnr::ir {

struct SourceSpan {
    int32_t line = 0;
    int32_t col = 0;
};

struct ClayImport {
    std::string path;
    std::string alias;
    SourceSpan span{};
};

struct ClayCellAlias {
    std::string name;
    std::string target;
    SourceSpan span{};
};

struct ClayPort {
    std::string name;
    std::string role;  // empty if omitted
};

struct ClayDevice {
    std::string decl_kind;    // "device", "res", "cap", "ind", "varactor", role/digital kw
    std::string name;         // e.g. "M1", "R2", "C0"
    std::string device_type;  // e.g. "pmos", "nmos", "pmos3v", "pnp", "res", "cap"
    std::vector<std::pair<std::string, std::string>> pins;        // canonicalized (d,g,s,b / c,b,e / p,n)
    std::vector<std::pair<std::string, double>> params;           // SI-scaled numeric parameters (W, L, M, nf, value, area)
    std::vector<std::pair<std::string, std::string>> str_params;  // model, native_param, etc.
    SourceSpan span{};

    double param_or(const std::string& key, double fallback = 0.0) const noexcept;
    bool has_param(const std::string& key) const noexcept;
    std::string pin_or(const std::string& key, const std::string& fallback = "") const;
    std::string str_param_or(const std::string& key, const std::string& fallback = "") const;
};

struct ClayInstance {
    std::string name;
    std::string target;
    std::vector<std::pair<std::string, std::string>> port_map;
};

struct ClayFunctionalGroup {
    std::string kind;  // "functional_group" or behavioral noun keyword
    std::string name;
    std::vector<std::string> devices;
    std::vector<std::pair<std::string, std::string>> attrs;
    SourceSpan span{};
};

struct ClaySpecField {
    std::string name;
    std::string type_id;
    std::string op;
    double value = 0.0;
    SourceSpan span{};
};

struct ClayCell {
    std::string name;
    SourceSpan span{};
    std::vector<ClayPort> ports;
    std::vector<ClayDevice> devices;
    std::vector<ClayInstance> instances;
    std::vector<ClayFunctionalGroup> functional_groups;
    std::vector<std::pair<std::string, double>> sizing;
    std::vector<ClaySpecField> specs;
    std::vector<std::string> corners;
    std::vector<double> temps;
};

struct ClayIntentEntry {
    std::string kind;  // "match", "symmetric", "order", "band", "place_group", "objective", ...
    std::vector<std::string> positional;
    std::vector<std::pair<std::string, std::string>> kwargs;
    std::vector<std::pair<std::string, std::vector<std::string>>> list_kwargs;
    SourceSpan span{};

    std::string kwarg_or(const std::string& key, const std::string& fallback = "") const;
    const std::vector<std::string>* list_kwarg(const std::string& key) const noexcept;
};

struct ClayLayoutIntent {
    std::string cell_name;
    SourceSpan span{};
    std::vector<ClayIntentEntry> entries;
};

struct ClayAnnotateEntry {
    std::string target;  // "__cell__" for @cell or bare device/group identifier
    std::string kind;    // "matching", "guard_ring", "orient", "fold", ...
    std::vector<std::string> args;
    std::vector<std::pair<std::string, std::string>> kwargs;
};

struct ClayAnnotateBlock {
    std::string cell_name;
    SourceSpan span{};
    std::vector<ClayAnnotateEntry> entries;
};

struct ClayFlowBlock {
    std::string cell_name;
    SourceSpan span{};
    std::string pdk;
    std::vector<std::string> corners;
    std::vector<std::string> steps;
    std::string layout_module;
    std::string policy;
};

struct ClayFile {
    std::string package_name;
    std::string pdk;
    std::vector<std::string> imports;
    std::vector<ClayImport> import_decls;
    std::vector<ClayCellAlias> cell_aliases;
    std::vector<ClayCell> cells;
    std::vector<ClayLayoutIntent> layout_intents;
    std::vector<ClayAnnotateBlock> annotations;
    std::vector<ClayFlowBlock> flows;
    std::vector<std::string> policy_names;
    std::vector<std::string> quality_names;
    std::vector<std::string> verify_names;
    std::vector<std::string> fn_names;
    size_t policy_count = 0;
    size_t quality_count = 0;
    size_t verify_count = 0;
    size_t fn_count = 0;

    const ClayLayoutIntent* intent_for_cell(const std::string& cname) const noexcept;
    const ClayAnnotateBlock* annotations_for_cell(const std::string& cname) const noexcept;
};

struct ParseResult {
    bool ok = false;
    ClayFile file;
    std::string error;
    int32_t line = 0;
    int32_t col = 0;
};

// Resolve Clay SI unit suffix (exact parity with clay.parser_modules._constants._resolve_unit_multiplier).
// Returns false if unit is unknown/malformed.
bool resolve_si_unit(const std::string& unit, double& out_multiplier) noexcept;

// Parse .clay source text directly into typed C++20 IR (zero Python / zero Lark).
ParseResult parse_clay_source(const std::string& source);

// Read and parse a .clay file from disk.
ParseResult parse_clay_file(const std::string& path);

}  // namespace clay_pnr::ir

