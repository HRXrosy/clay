// Standalone pure C++20 Clay Engine CLI (ZERO Python / ZERO third-party dependencies).
// Commands:
//   clay_engine_cli info | --version
//   clay_engine_cli selftest
//   clay_engine_cli parse <design.clay>
//   clay_engine_cli synth <design.clay> -o <out.gds> [--pdk <gf180mcu|sky130>] [--no-route]
//   clay_engine_cli gds-inspect <file.gds> [--top <cell>]
//   clay_engine_cli gds-emit-json <spec.json | -> -o <out.gds>
//   clay_engine_cli pcell-mos <nmos|pmos> <w_nm> <l_nm> <fingers> <gf180mcu|sky130> -o <out.gds>

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/polygon_set.hpp"
#include "clay_pnr/core/symbol_table.hpp"
#include "clay_pnr/engine/standalone_synth.hpp"
#include "clay_pnr/gds/gds_stream.hpp"
#include "clay_pnr/ir/clay_ir.hpp"
#include "clay_pnr/layout/layout_db.hpp"
#include "clay_pnr/pcell/mos_pcell.hpp"
#include "clay_pnr/place/banded_placer.hpp"
#include "clay_pnr/place/compactor.hpp"
#include "clay_pnr/place/envelope_placer.hpp"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace clay_pnr::core;
using namespace clay_pnr::gds;

void print_usage() {
    std::printf(
        "clay_engine_cli (Pure C++20 Main Engine - zero Python / zero third-party deps)\n"
        "Usage:\n"
        "  clay_engine_cli --version | info\n"
        "  clay_engine_cli selftest\n"
        "  clay_engine_cli parse <design.clay>\n"
        "  clay_engine_cli synth <design.clay> -o <out.gds> [--pdk <gf180mcu|sky130>] [--no-route]\n"
        "  clay_engine_cli gds-inspect <file.gds> [--top <cell>]\n"
        "  clay_engine_cli gds-emit-json <spec.json | -> -o <out.gds>\n"
        "  clay_engine_cli net-connectivity  (stdin JSON -> stdout JSON)\n"
        "  clay_engine_cli net-wirelength    (stdin JSON -> stdout JSON)\n"
        "  clay_engine_cli pcell-mos <nmos|pmos> <w_nm> <l_nm> <fingers> <gf180mcu|sky130> -o <out.gds>\n");
}

int cmd_info() {
    std::printf("{\n");
    std::printf("  \"engine\": \"clay_engine_cpp20\",\n");
    std::printf("  \"phase\": \"phase5-cpp20-main-engine\",\n");
    std::printf("  \"python_dependency\": false,\n");
    std::printf("  \"modules\": [\"core.geometry\", \"core.polygon_set\", \"core.symbol_table\", "
                "\"core.session_budget\", \"ir.clay_parser\", \"gds.stream\", \"layout.layout_db\", "
                "\"layout.connectivity\", \"layout.net_wirelength\", \"pcell.mos_pcell\", "
                "\"place.sweep_union\", \"place.bvh\", \"place.seqpair\", \"place.compactor\", "
                "\"place.envelope_placer\", \"place.banded_placer\", \"route.pathfinder\", "
                "\"route.gridless\", \"engine.standalone_synth\"]\n");
    std::printf("}\n");
    return 0;
}

int cmd_selftest() {
    const auto t0 = std::chrono::steady_clock::now();
    // 1. Exact PolygonSet boolean + non-convex erosion round-trip
    const PolygonSet a = PolygonSet::from_boxes({Box64{0, 0, 200, 100}, Box64{0, 0, 100, 200}});
    const PolygonSet b = PolygonSet::from_box(Box64{50, 50, 150, 150});
    const PolygonSet diff = PolygonSet::op_not(a, b);
    const PolygonSet eroded = a.sized(-10, -10);
    if (diff.empty() || eroded.empty()) {
        std::fprintf(stderr, "selftest failed: unexpected empty PolygonSet\n");
        return 1;
    }
    // 2. Native GDSII write + read round-trip
    GdsLibrary lib;
    lib.name = "SELFTEST";
    auto& top = lib.add_cell("TOP");
    for (const auto& r : diff.rects()) {
        top.add_box(34, 0, r);
    }
    const auto bytes = write_gds_bytes(lib);
    GdsLibrary reloaded;
    std::string err;
    if (!read_gds_bytes(bytes.data(), bytes.size(), reloaded, &err)) {
        std::fprintf(stderr, "selftest failed: GDS reload error: %s\n", err.c_str());
        return 1;
    }
    const auto summary = summarize_gds_layers(reloaded, "TOP");
    if (summary.size() != 1 || summary[0].union_area_dbu2 != diff.area_u128()) {
        std::fprintf(stderr, "selftest failed: GDS area mismatch\n");
        return 1;
    }
    // 3. Pure C++20 DAG Compactor + 2D Envelope Placer + MOS PCell smoke
    const auto cmp = clay_pnr::place::compact_axis_dag({10, 7, 4}, {{0, 1, 12}, {0, 2, 13}});
    if (cmp.status != clay_pnr::place::CompactStatus::Ok || cmp.extent != 19) {
        std::fprintf(stderr, "selftest failed: compactor mismatch\n");
        return 1;
    }
    clay_pnr::place::EnvelopeProblem ep{};
    ep.gap = 1.0;
    ep.groups = {{"A", 8.0, 6.0, 9, 7}, {"B", 8.0, 6.0, 9, 7}, {"C", 12.0, 5.0, 13, 6}};
    ep.symmetry_pairs = {{0, 1}};
    const auto esol = clay_pnr::place::solve_envelope_min(ep);
    if ((esol.status != clay_pnr::place::EnvelopeStatus::Optimal &&
         esol.status != clay_pnr::place::EnvelopeStatus::Feasible) ||
        !clay_pnr::place::verify_envelope_solution(ep, esol.xs, esol.ys)) {
        std::fprintf(stderr, "selftest failed: envelope placer failed\n");
        return 1;
    }
    const auto pcell = clay_pnr::pcell::generate_mos_pcell(
        clay_pnr::pcell::MosSpec{"M1", true, 2000, 280, 4, 1, 1},
        clay_pnr::pcell::gf180mcu_mos_rules());
    if (pcell.terminals.gate_rects.size() != 4) {
        std::fprintf(stderr, "selftest failed: mos pcell gate count\n");
        return 1;
    }

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("{\"status\":\"PASS\",\"gds_bytes\":%zu,\"area_dbu2\":%llu,\"envelope_obj\":%lld,\"elapsed_ms\":%.3f}\n",
                bytes.size(), static_cast<unsigned long long>(summary[0].union_area_dbu2.lo),
                static_cast<long long>(esol.objective), ms);
    return 0;
}

int cmd_gds_inspect(const std::string& path, const std::string& top_cell) {
    const auto t0 = std::chrono::steady_clock::now();
    GdsLibrary lib;
    std::string err;
    if (!read_gds_file(path, lib, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const auto tops = lib.top_cell_names();
    const std::string chosen = top_cell.empty() ? (tops.empty() ? "" : tops.front()) : top_cell;
    const auto layers = summarize_gds_layers(lib, chosen);
    Box64 total_bb{};
    bool has_bb = false;
    for (const auto& ls : layers) {
        if (!has_bb) {
            total_bb = ls.bbox;
            has_bb = true;
        } else {
            if (ls.bbox.x0 < total_bb.x0) total_bb.x0 = ls.bbox.x0;
            if (ls.bbox.y0 < total_bb.y0) total_bb.y0 = ls.bbox.y0;
            if (ls.bbox.x1 > total_bb.x1) total_bb.x1 = ls.bbox.x1;
            if (ls.bbox.y1 > total_bb.y1) total_bb.y1 = ls.bbox.y1;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double um_scale = lib.dbu_meters * 1e6;
    std::printf("{\n");
    std::printf("  \"library\": \"%s\",\n", lib.name.c_str());
    std::printf("  \"top_cell\": \"%s\",\n", chosen.c_str());
    std::printf("  \"cell_count\": %zu,\n", lib.cells.size());
    std::printf("  \"dbu_um\": %.9g,\n", um_scale);
    std::printf("  \"layer_count\": %zu,\n", layers.size());
    if (has_bb) {
        std::printf("  \"bbox_dbu\": [%lld, %lld, %lld, %lld],\n",
                    static_cast<long long>(total_bb.x0), static_cast<long long>(total_bb.y0),
                    static_cast<long long>(total_bb.x1), static_cast<long long>(total_bb.y1));
        std::printf("  \"size_um\": [%.4f, %.4f],\n",
                    static_cast<double>(total_bb.x1 - total_bb.x0) * um_scale,
                    static_cast<double>(total_bb.y1 - total_bb.y0) * um_scale);
    }
    std::printf("  \"layers\": [\n");
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& ls = layers[i];
        std::printf("    {\"layer\": %d, \"datatype\": %d, \"raw\": %zu, \"canonical\": %zu, \"area_dbu2\": %llu}%s\n",
                    static_cast<int>(ls.layer), static_cast<int>(ls.datatype),
                    ls.raw_polygon_count, ls.canonical_rect_count,
                    static_cast<unsigned long long>(ls.union_area_dbu2.lo),
                    (i + 1 < layers.size()) ? "," : "");
    }
    std::printf("  ],\n");
    std::printf("  \"elapsed_ms\": %.3f\n", ms);
    std::printf("}\n");
    return 0;
}

// Minimal self-contained zero-dependency JSON reader for ParameterizedLibrary GDS emission.
struct JsonReader {
    const std::string& s;
    size_t pos = 0;

    void skip_ws() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }
    bool consume(char c) {
        skip_ws();
        if (pos < s.size() && s[pos] == c) { ++pos; return true; }
        return false;
    }
    bool parse_string(std::string& out) {
        skip_ws();
        if (pos >= s.size() || s[pos] != '"') return false;
        ++pos;
        out.clear();
        while (pos < s.size()) {
            char ch = s[pos++];
            if (ch == '"') return true;
            if (ch == '\\' && pos < s.size()) {
                char esc = s[pos++];
                if (esc == 'n') out.push_back('\n');
                else if (esc == 't') out.push_back('\t');
                else if (esc == 'r') out.push_back('\r');
                else out.push_back(esc);
            } else {
                out.push_back(ch);
            }
        }
        return false;
    }
    bool parse_number(double& out) {
        skip_ws();
        const char* start = s.c_str() + pos;
        char* end = nullptr;
        out = std::strtod(start, &end);
        if (end == start) return false;
        pos = static_cast<size_t>(end - s.c_str());
        return true;
    }
    bool parse_i64(int64_t& out) {
        double v = 0.0;
        if (!parse_number(v)) return false;
        out = static_cast<int64_t>(std::llround(v));
        return true;
    }
    bool skip_value() {
        skip_ws();
        if (pos >= s.size()) return false;
        const char c = s[pos];
        if (c == '"') { std::string tmp; return parse_string(tmp); }
        if (c == '{') {
            ++pos;
            if (consume('}')) return true;
            while (true) {
                std::string k;
                if (!parse_string(k) || !consume(':') || !skip_value()) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == '[') {
            ++pos;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value()) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == 't' && s.compare(pos, 4, "true") == 0) { pos += 4; return true; }
        if (c == 'f' && s.compare(pos, 5, "false") == 0) { pos += 5; return true; }
        if (c == 'n' && s.compare(pos, 4, "null") == 0) { pos += 4; return true; }
        double num = 0.0;
        return parse_number(num);
    }
};

bool parse_parameterized_library_json(const std::string& txt, GdsLibrary& lib, std::string& err) {
    JsonReader jr{txt, 0};
    if (!jr.consume('{')) { err = "expected JSON object"; return false; }
    if (jr.consume('}')) return true;
    while (true) {
        std::string key;
        if (!jr.parse_string(key) || !jr.consume(':')) { err = "malformed top-level key"; return false; }
        if (key == "name") {
            if (!jr.parse_string(lib.name)) { err = "invalid library name"; return false; }
        } else if (key == "unit_user") {
            if (!jr.parse_number(lib.user_unit)) { err = "invalid unit_user"; return false; }
        } else if (key == "unit_db_meters") {
            if (!jr.parse_number(lib.dbu_meters)) { err = "invalid unit_db_meters"; return false; }
        } else if (key == "cells") {
            if (!jr.consume('[')) { err = "cells must be array"; return false; }
            if (!jr.consume(']')) {
                while (true) {
                    if (!jr.consume('{')) { err = "cell must be object"; return false; }
                    std::string cname = "CELL";
                    struct RawPoly {
                        int16_t layer = 0;
                        int16_t datatype = 0;
                        std::vector<Point64> pts;
                    };
                    std::vector<RawPoly> polys;
                    if (!jr.consume('}')) {
                        while (true) {
                            std::string ckey;
                            if (!jr.parse_string(ckey) || !jr.consume(':')) { err = "bad cell key"; return false; }
                            if (ckey == "name") {
                                if (!jr.parse_string(cname)) { err = "bad cell name"; return false; }
                            } else if (ckey == "polygons") {
                                if (!jr.consume('[')) { err = "polygons must be array"; return false; }
                                if (!jr.consume(']')) {
                                    while (true) {
                                        if (!jr.consume('{')) { err = "polygon entry must be object"; return false; }
                                        int64_t layer = 0, dt = 0;
                                        std::vector<Point64> verts;
                                        if (!jr.consume('}')) {
                                            while (true) {
                                                std::string pkey;
                                                if (!jr.parse_string(pkey) || !jr.consume(':')) { err = "bad poly key"; return false; }
                                                if (pkey == "layer") {
                                                    if (!jr.parse_i64(layer)) return false;
                                                } else if (pkey == "datatype") {
                                                    if (!jr.parse_i64(dt)) return false;
                                                } else if (pkey == "polygon") {
                                                    if (!jr.consume('{')) return false;
                                                    if (!jr.consume('}')) {
                                                        while (true) {
                                                            std::string gkey;
                                                            if (!jr.parse_string(gkey) || !jr.consume(':')) return false;
                                                            if (gkey == "vertices") {
                                                                if (!jr.consume('[')) return false;
                                                                if (!jr.consume(']')) {
                                                                    while (true) {
                                                                        if (!jr.consume('{')) return false;
                                                                        int64_t vx = 0, vy = 0;
                                                                        if (!jr.consume('}')) {
                                                                            while (true) {
                                                                                std::string vkey;
                                                                                if (!jr.parse_string(vkey) || !jr.consume(':')) return false;
                                                                                if (vkey == "x") { if (!jr.parse_i64(vx)) return false; }
                                                                                else if (vkey == "y") { if (!jr.parse_i64(vy)) return false; }
                                                                                else { if (!jr.skip_value()) return false; }
                                                                                if (jr.consume('}')) break;
                                                                                if (!jr.consume(',')) return false;
                                                                            }
                                                                        }
                                                                        verts.push_back(Point64{vx, vy});
                                                                        if (jr.consume(']')) break;
                                                                        if (!jr.consume(',')) return false;
                                                                    }
                                                                }
                                                            } else {
                                                                if (!jr.skip_value()) return false;
                                                            }
                                                            if (jr.consume('}')) break;
                                                            if (!jr.consume(',')) return false;
                                                        }
                                                    }
                                                } else {
                                                    if (!jr.skip_value()) return false;
                                                }
                                                if (jr.consume('}')) break;
                                                if (!jr.consume(',')) return false;
                                            }
                                        }
                                        if (verts.size() >= 3) {
                                            polys.push_back(RawPoly{static_cast<int16_t>(layer), static_cast<int16_t>(dt), std::move(verts)});
                                        }
                                        if (jr.consume(']')) break;
                                        if (!jr.consume(',')) return false;
                                    }
                                }
                            } else {
                                if (!jr.skip_value()) return false;
                            }
                            if (jr.consume('}')) break;
                            if (!jr.consume(',')) return false;
                        }
                    }
                    auto& cell = lib.add_cell(cname);
                    for (auto& b : polys) {
                        cell.add_polygon(b.layer, b.datatype, b.pts);
                    }
                    if (jr.consume(']')) break;
                    if (!jr.consume(',')) return false;
                }
            }
        } else {
            if (!jr.skip_value()) return false;
        }
        if (jr.consume('}')) break;
        if (!jr.consume(',')) { err = "expected comma in top object"; return false; }
    }
    return true;
}

int cmd_gds_emit_json(const std::string& in_path, const std::string& out_path) {
    std::ostringstream ss;
    if (in_path == "-") {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream ifs(in_path, std::ios::binary);
        if (!ifs) {
            std::fprintf(stderr, "error: cannot open input JSON: %s\n", in_path.c_str());
            return 1;
        }
        ss << ifs.rdbuf();
    }
    GdsLibrary lib;
    std::string err;
    if (!parse_parameterized_library_json(ss.str(), lib, err)) {
        std::fprintf(stderr, "error: JSON parse failed: %s\n", err.c_str());
        return 1;
    }
    if (!write_gds_file(lib, out_path, &err)) {
        std::fprintf(stderr, "error: GDS write failed: %s\n", err.c_str());
        return 1;
    }
    return 0;
}

int cmd_pcell_mos(const std::string& kind, int64_t w_nm, int64_t l_nm, int32_t fingers,
                  const std::string& pdk, const std::string& out_path) {
    const auto rules = (pdk == "sky130") ? clay_pnr::pcell::sky130_mos_rules()
                                         : clay_pnr::pcell::gf180mcu_mos_rules();
    clay_pnr::pcell::MosSpec spec{};
    spec.name = (kind == "pmos") ? "PMOS_PCELL" : "NMOS_PCELL";
    spec.is_pmos = (kind == "pmos");
    spec.w_nm = w_nm;
    spec.l_nm = l_nm;
    spec.fingers = fingers;
    const auto res = clay_pnr::pcell::generate_mos_pcell(spec, rules);
    clay_pnr::layout::LayoutDB db;
    db.name = "CLAY_PCELL";
    db.cells.push_back(res.cell);
    std::string err;
    if (!db.write_gds(out_path, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::printf("{\"cell\":\"%s\",\"pdk\":\"%s\",\"polygons\":%zu,\"bbox_nm\":[%lld,%lld,%lld,%lld]}\n",
                spec.name.c_str(), rules.pdk_id.c_str(), res.cell.polygons.size(),
                static_cast<long long>(res.terminals.bbox.x0), static_cast<long long>(res.terminals.bbox.y0),
                static_cast<long long>(res.terminals.bbox.x1), static_cast<long long>(res.terminals.bbox.y1));
    return 0;
}

int cmd_parse(const std::string& clay_path) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto pr = clay_pnr::ir::parse_clay_file(clay_path);
    if (!pr.ok) {
        std::fprintf(stderr, "parse error (%d:%d): %s\n", pr.line, pr.col, pr.error.c_str());
        return 1;
    }
    size_t total_devs = 0, total_fgs = 0, total_intents = 0, total_annots = 0;
    for (const auto& c : pr.file.cells) {
        total_devs += c.devices.size();
        total_fgs += c.functional_groups.size();
    }
    for (const auto& li : pr.file.layout_intents) total_intents += li.entries.size();
    for (const auto& ab : pr.file.annotations) total_annots += ab.entries.size();
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("{\"ok\":true,\"package\":\"%s\",\"pdk\":\"%s\",\"cells\":%zu,"
                "\"cell_aliases\":%zu,\"imports\":%zu,"
                "\"devices\":%zu,\"functional_groups\":%zu,\"intents\":%zu,"
                "\"annotations\":%zu,\"flows\":%zu,"
                "\"policies\":%zu,\"qualities\":%zu,\"verifies\":%zu,\"fns\":%zu,"
                "\"elapsed_ms\":%.3f}\n",
                pr.file.package_name.c_str(), pr.file.pdk.c_str(),
                pr.file.cells.size(), pr.file.cell_aliases.size(), pr.file.import_decls.size(),
                total_devs, total_fgs,
                total_intents, total_annots, pr.file.flows.size(),
                pr.file.policy_count, pr.file.quality_count, pr.file.verify_count, pr.file.fn_count,
                ms);
    return 0;
}

int cmd_synth(const std::string& clay_path, const std::string& out_gds,
              const std::string& pdk_override, bool emit_routes) {
    const auto t0 = std::chrono::steady_clock::now();
    clay_pnr::engine::SynthOptions opts{};
    opts.pdk_override = pdk_override;
    opts.emit_routes = emit_routes;
    const auto res = clay_pnr::engine::synthesize_clay_file_to_gds(clay_path, out_gds, opts);
    if (!res.ok) {
        std::fprintf(stderr, "synth error: %s\n", res.error.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("{\"status\":\"PASS\",\"cell\":\"%s\",\"pdk\":\"%s\","
                "\"devices\":%zu,\"nets\":%zu,\"routed_nets\":%zu,\"polygons\":%zu,"
                "\"bbox_nm\":[%lld,%lld,%lld,%lld],\"size_um\":[%.3f,%.3f],\"elapsed_ms\":%.3f}\n",
                res.cell_name.c_str(), res.pdk_id.c_str(),
                res.device_count, res.net_count, res.routed_net_count, res.polygon_count,
                static_cast<long long>(res.die_bbox_nm.x0), static_cast<long long>(res.die_bbox_nm.y0),
                static_cast<long long>(res.die_bbox_nm.x1), static_cast<long long>(res.die_bbox_nm.y1),
                static_cast<double>(res.die_bbox_nm.width()) * 1e-3,
                static_cast<double>(res.die_bbox_nm.height()) * 1e-3,
                ms);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "--version" || cmd == "-v" || cmd == "info") {
        return cmd_info();
    }
    if (cmd == "selftest") {
        return cmd_selftest();
    }
    if (cmd == "parse") {
        if (argc < 3) {
            std::fprintf(stderr, "error: parse requires <design.clay>\n");
            return 1;
        }
        return cmd_parse(argv[2]);
    }
    if (cmd == "synth") {
        if (argc < 3) {
            std::fprintf(stderr, "error: synth requires <design.clay> -o <out.gds>\n");
            return 1;
        }
        std::string clay_path = argv[2];
        std::string out_path;
        std::string pdk_override;
        bool emit_routes = true;
        for (int i = 3; i < argc; ++i) {
            if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                out_path = argv[++i];
            } else if (std::strcmp(argv[i], "--pdk") == 0 && i + 1 < argc) {
                pdk_override = argv[++i];
            } else if (std::strcmp(argv[i], "--no-route") == 0) {
                emit_routes = false;
            }
        }
        if (out_path.empty()) {
            std::fprintf(stderr, "error: synth requires -o <out.gds>\n");
            return 1;
        }
        return cmd_synth(clay_path, out_path, pdk_override, emit_routes);
    }
    if (cmd == "gds-inspect") {
        if (argc < 3) {
            std::fprintf(stderr, "error: gds-inspect requires <file.gds>\n");
            return 1;
        }
        std::string gds_path = argv[2];
        std::string top_cell;
        for (int i = 3; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--top") == 0) {
                top_cell = argv[i + 1];
                ++i;
            }
        }
        return cmd_gds_inspect(gds_path, top_cell);
    }
    if (cmd == "gds-emit-json") {
        std::string in_path = "-";
        std::string out_path;
        for (int i = 2; i < argc; ++i) {
            if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                out_path = argv[++i];
            } else {
                in_path = argv[i];
            }
        }
        if (out_path.empty()) {
            std::fprintf(stderr, "error: gds-emit-json requires -o <out.gds>\n");
            return 1;
        }
        return cmd_gds_emit_json(in_path, out_path);
    }
    if (cmd == "net-connectivity") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        const std::string txt = ss.str();
        JsonReader jr{txt, 0};
        std::vector<clay_pnr::layout::NetShapeBox> shapes;
        std::vector<clay_pnr::layout::LayerKey> via_layers;
        if (!jr.consume('{')) { std::fprintf(stderr, "error: expected JSON object\n"); return 1; }
        if (!jr.consume('}')) {
            while (true) {
                std::string k;
                if (!jr.parse_string(k) || !jr.consume(':')) { std::fprintf(stderr, "error: malformed JSON\n"); return 1; }
                if (k == "shapes") {
                    if (!jr.consume('[')) return 1;
                    if (!jr.consume(']')) {
                        while (true) {
                            if (!jr.consume('{')) return 1;
                            clay_pnr::layout::NetShapeBox sb{};
                            if (!jr.consume('}')) {
                                while (true) {
                                    std::string sk;
                                    if (!jr.parse_string(sk) || !jr.consume(':')) return 1;
                                    if (sk == "net_id") {
                                        int64_t nid = 0;
                                        if (!jr.parse_i64(nid)) return 1;
                                        sb.net_id = static_cast<uint32_t>(nid);
                                    } else if (sk == "layer") {
                                        int64_t l = 0, d = 0;
                                        if (!jr.consume('[') || !jr.parse_i64(l) || !jr.consume(',') ||
                                            !jr.parse_i64(d) || !jr.consume(']')) return 1;
                                        sb.key = clay_pnr::layout::LayerKey{static_cast<int16_t>(l), static_cast<int16_t>(d)};
                                    } else if (sk == "bbox") {
                                        if (!jr.consume('[') || !jr.parse_i64(sb.box.x0) || !jr.consume(',') ||
                                            !jr.parse_i64(sb.box.y0) || !jr.consume(',') ||
                                            !jr.parse_i64(sb.box.x1) || !jr.consume(',') ||
                                            !jr.parse_i64(sb.box.y1) || !jr.consume(']')) return 1;
                                    } else {
                                        if (!jr.skip_value()) return 1;
                                    }
                                    if (jr.consume('}')) break;
                                    if (!jr.consume(',')) return 1;
                                }
                            }
                            shapes.push_back(sb);
                            if (jr.consume(']')) break;
                            if (!jr.consume(',')) return 1;
                        }
                    }
                } else if (k == "via_layers") {
                    if (!jr.consume('[')) return 1;
                    if (!jr.consume(']')) {
                        while (true) {
                            int64_t l = 0, d = 0;
                            if (!jr.consume('[') || !jr.parse_i64(l) || !jr.consume(',') ||
                                !jr.parse_i64(d) || !jr.consume(']')) return 1;
                            via_layers.push_back(clay_pnr::layout::LayerKey{static_cast<int16_t>(l), static_cast<int16_t>(d)});
                            if (jr.consume(']')) break;
                            if (!jr.consume(',')) return 1;
                        }
                    }
                } else {
                    if (!jr.skip_value()) return 1;
                }
                if (jr.consume('}')) break;
                if (!jr.consume(',')) return 1;
            }
        }
        const auto res = clay_pnr::layout::verify_net_connectivity_exact(shapes, via_layers);
        std::printf("{\"component_of\":[");
        for (size_t i = 0; i < res.component_of.size(); ++i) {
            std::printf("%s[%u,%u]", i ? "," : "", res.component_of[i].first, res.component_of[i].second);
        }
        std::printf("],\"merged_pairs\":[");
        for (size_t i = 0; i < res.merged_pairs.size(); ++i) {
            std::printf("%s[%u,%u]", i ? "," : "", res.merged_pairs[i].first, res.merged_pairs[i].second);
        }
        std::printf("],\"distinct\":%s,\"shape_root\":[", res.distinct ? "true" : "false");
        for (size_t i = 0; i < res.shape_root.size(); ++i) {
            std::printf("%s%u", i ? "," : "", res.shape_root[i]);
        }
        std::printf("],\"short_count\":%zu}\n", res.short_count);
        return 0;
    }
    if (cmd == "net-wirelength") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        const std::string txt = ss.str();
        JsonReader jr{txt, 0};
        std::vector<std::vector<Point64>> nets;
        std::vector<Box64> bboxes;
        if (!jr.consume('{')) { std::fprintf(stderr, "error: expected JSON object\n"); return 1; }
        if (!jr.consume('}')) {
            while (true) {
                std::string k;
                if (!jr.parse_string(k) || !jr.consume(':')) return 1;
                if (k == "nets") {
                    if (!jr.consume('[')) return 1;
                    if (!jr.consume(']')) {
                        while (true) {
                            std::vector<Point64> pts;
                            if (!jr.consume('[')) return 1;
                            if (!jr.consume(']')) {
                                while (true) {
                                    int64_t x = 0, y = 0;
                                    if (!jr.consume('[') || !jr.parse_i64(x) || !jr.consume(',') ||
                                        !jr.parse_i64(y) || !jr.consume(']')) return 1;
                                    pts.push_back(Point64{x, y});
                                    if (jr.consume(']')) break;
                                    if (!jr.consume(',')) return 1;
                                }
                            }
                            nets.push_back(std::move(pts));
                            if (jr.consume(']')) break;
                            if (!jr.consume(',')) return 1;
                        }
                    }
                } else if (k == "bboxes") {
                    if (!jr.consume('[')) return 1;
                    if (!jr.consume(']')) {
                        while (true) {
                            Box64 b{};
                            if (!jr.consume('[') || !jr.parse_i64(b.x0) || !jr.consume(',') ||
                                !jr.parse_i64(b.y0) || !jr.consume(',') ||
                                !jr.parse_i64(b.x1) || !jr.consume(',') ||
                                !jr.parse_i64(b.y1) || !jr.consume(']')) return 1;
                            bboxes.push_back(b);
                            if (jr.consume(']')) break;
                            if (!jr.consume(',')) return 1;
                        }
                    }
                } else {
                    if (!jr.skip_value()) return 1;
                }
                if (jr.consume('}')) break;
                if (!jr.consume(',')) return 1;
            }
        }
        const auto res = clay_pnr::layout::compute_net_wirelength_exact(nets, bboxes);
        std::printf("{\"net_total_span\":[");
        for (size_t i = 0; i < res.net_total_span.size(); ++i) {
            std::printf("%s%lld", i ? "," : "", static_cast<long long>(res.net_total_span[i]));
        }
        std::printf("],\"net_max_pin_span\":[");
        for (size_t i = 0; i < res.net_max_pin_span.size(); ++i) {
            std::printf("%s%lld", i ? "," : "", static_cast<long long>(res.net_max_pin_span[i]));
        }
        std::printf("],\"netlist_total_hpwl\":%lld,\"overlapping_pairs\":%lld}\n",
                    static_cast<long long>(res.netlist_total_hpwl),
                    static_cast<long long>(res.overlapping_pairs));
        return 0;
    }
    if (cmd == "pcell-mos") {
        if (argc < 8) {
            std::fprintf(stderr, "error: pcell-mos <nmos|pmos> <w_nm> <l_nm> <fingers> <gf180mcu|sky130> -o <out.gds>\n");
            return 1;
        }
        const std::string kind = argv[2];
        const int64_t w_nm = std::atoll(argv[3]);
        const int64_t l_nm = std::atoll(argv[4]);
        const int32_t fingers = static_cast<int32_t>(std::atoi(argv[5]));
        const std::string pdk = argv[6];
        std::string out_path;
        for (int i = 7; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) {
                out_path = argv[++i];
            }
        }
        if (out_path.empty()) {
            std::fprintf(stderr, "error: pcell-mos requires -o <out.gds>\n");
            return 1;
        }
        return cmd_pcell_mos(kind, w_nm, l_nm, fingers, pdk, out_path);
    }
    // Compatibility flag form: `clay_engine_cli -o <out.gds>` reads JSON from stdin (drop-in for clay_layout_gds_bin).
    if (cmd == "-o" && argc >= 3) {
        return cmd_gds_emit_json("-", argv[2]);
    }
    print_usage();
    return 1;
}
