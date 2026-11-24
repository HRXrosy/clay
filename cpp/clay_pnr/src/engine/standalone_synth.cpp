#include "clay_pnr/engine/standalone_synth.hpp"
#include "clay_pnr/pcell/mos_pcell.hpp"
#include "clay_pnr/place/banded_placer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>

namespace clay_pnr::engine {

namespace {

using clay_pnr::core::Box64;
using clay_pnr::core::Point64;
using clay_pnr::ir::ClayCell;
using clay_pnr::ir::ClayDevice;
using clay_pnr::ir::ClayFile;
using clay_pnr::ir::ClayLayoutIntent;
using clay_pnr::ir::ParseResult;
using clay_pnr::ir::parse_clay_file;
using clay_pnr::ir::parse_clay_source;
using clay_pnr::layout::LayerKey;
using clay_pnr::layout::LayerPoly;
using clay_pnr::layout::LayoutDB;
using clay_pnr::layout::NetLayerBoxes;
using clay_pnr::layout::merge_net_boxes_exact;
using clay_pnr::pcell::MosPCellResult;
using clay_pnr::pcell::MosSpec;
using clay_pnr::pcell::MosTechRules;
using clay_pnr::pcell::generate_mos_pcell;
using clay_pnr::pcell::gf180mcu_mos_rules;
using clay_pnr::pcell::sky130_mos_rules;
using clay_pnr::place::BandedBlock;
using clay_pnr::place::BandedProblem;
using clay_pnr::place::BandedSolution;
using clay_pnr::place::BandedStatus;
using clay_pnr::place::solve_banded_area;

struct PdkRouteLayers {
    std::string pdk_id;
    int64_t grid_nm = 5;
    MosTechRules mos_rules{};
    LayerKey m1{34, 0};
    LayerKey via1{35, 0};
    LayerKey m2{36, 0};
    LayerKey via2{38, 0};
    LayerKey m3{42, 0};
    LayerKey m2_label{36, 10};
    LayerKey poly{30, 0};
    LayerKey res_mk{62, 0};
    LayerKey mim_cap{117, 0};
    int64_t via_size_nm = 260;
    int64_t via_enc_nm = 60;
};

PdkRouteLayers resolve_pdk_layers(const std::string& pdk_raw) {
    PdkRouteLayers p{};
    if (pdk_raw.find("sky130") != std::string::npos) {
        p.pdk_id = "sky130";
        p.grid_nm = 5;
        p.mos_rules = sky130_mos_rules();
        p.m1 = LayerKey{68, 20};
        p.via1 = LayerKey{68, 44};
        p.m2 = LayerKey{69, 20};
        p.via2 = LayerKey{69, 44};
        p.m3 = LayerKey{70, 20};
        p.m2_label = LayerKey{69, 5};
        p.poly = LayerKey{66, 20};
        p.res_mk = LayerKey{66, 13};
        p.mim_cap = LayerKey{89, 44};
        p.via_size_nm = 150;
        p.via_enc_nm = 55;
        return p;
    }
    // Default to GF180MCU (open-source 180nm benchmark PDK)
    p.pdk_id = pdk_raw.empty() ? "gf180mcu" : pdk_raw;
    p.grid_nm = 5;
    p.mos_rules = gf180mcu_mos_rules();
    p.m1 = LayerKey{34, 0};
    p.via1 = LayerKey{35, 0};
    p.m2 = LayerKey{36, 0};
    p.via2 = LayerKey{38, 0};
    p.m3 = LayerKey{42, 0};
    p.m2_label = LayerKey{36, 10};
    p.poly = LayerKey{30, 0};
    p.res_mk = LayerKey{62, 0};
    p.mim_cap = LayerKey{117, 0};
    p.via_size_nm = 260;
    p.via_enc_nm = 60;
    return p;
}

inline int64_t snap_up(int64_t v, int64_t g) noexcept {
    if (g <= 1) return v;
    const int64_t r = v % g;
    return (r == 0) ? v : ((v >= 0) ? (v + (g - r)) : (v - r));
}

struct MacroPin {
    std::string pin_name;
    std::string net_name;
    Box64 local_box{};
};

struct GeneratedMacro {
    std::string name;
    std::string kind;
    int32_t band = 1;
    Box64 local_bbox{};
    std::vector<LayerPoly> polygons;
    std::vector<MacroPin> pins;
};

bool is_pmos_type(const std::string& t) {
    return t.rfind("pmos", 0) == 0 || t.rfind("pch", 0) == 0 || t.rfind("pfet", 0) == 0;
}

bool is_nmos_type(const std::string& t) {
    return t.rfind("nmos", 0) == 0 || t.rfind("nch", 0) == 0 || t.rfind("nfet", 0) == 0;
}

GeneratedMacro build_device_macro(const ClayDevice& dev, const PdkRouteLayers& pdk) {
    GeneratedMacro m{};
    m.name = dev.name;
    m.kind = dev.device_type;
    const int64_t g = pdk.grid_nm;

    if (dev.decl_kind == "device" && (is_pmos_type(dev.device_type) || is_nmos_type(dev.device_type))) {
        const bool pmos = is_pmos_type(dev.device_type);
        m.band = pmos ? 2 : 0;
        const double w_m = dev.param_or("W", 2e-6);
        const double l_m = dev.param_or("L", 0.5e-6);
        const int64_t mult = std::max<int64_t>(1, static_cast<int64_t>(std::llround(dev.param_or("M", 1.0))));
        const int64_t nf_decl = std::max<int64_t>(1, static_cast<int64_t>(std::llround(dev.param_or("nf", 1.0))));
        int64_t w_nm = snap_up(std::max<int64_t>(420, static_cast<int64_t>(std::llround(w_m * 1e9))), g);
        const int64_t l_nm = snap_up(std::max<int64_t>(180, static_cast<int64_t>(std::llround(l_m * 1e9))), g);
        int64_t fingers = mult * nf_decl;
        // Fold wide single-finger devices cleanly into compact multi-finger layouts
        while (w_nm > 10000 && fingers < 32) {
            w_nm = snap_up(w_nm / 2, g);
            fingers *= 2;
        }
        fingers = std::clamp<int64_t>(fingers, 1, 64);

        MosSpec spec{dev.name, pmos, w_nm, l_nm, static_cast<int32_t>(fingers), 0, 0};
        const MosPCellResult res = generate_mos_pcell(spec, pdk.mos_rules);
        // Normalize macro origin to (0, 0)
        const Box64 bb = res.terminals.bbox;
        const int64_t ox = bb.x0;
        const int64_t oy = bb.y0;
        m.local_bbox = Box64{0, 0, bb.x1 - ox, bb.y1 - oy};
        m.polygons.reserve(res.cell.polygons.size());
        for (const auto& lp : res.cell.polygons) {
            LayerPoly shifted{lp.key, {}};
            shifted.vertices.reserve(lp.vertices.size());
            for (const auto& pt : lp.vertices) {
                shifted.vertices.push_back(Point64{pt.x - ox, pt.y - oy});
            }
            m.polygons.push_back(std::move(shifted));
        }

        auto add_pin_rects = [&](const std::string& pkey, const std::vector<Box64>& rects) {
            const std::string net = dev.pin_or(pkey);
            if (net.empty() || rects.empty()) return;
            // Pick representative M1 contact strip (first rectangle, shifted to local origin)
            const Box64& r = rects.front();
            m.pins.push_back(MacroPin{pkey, net, Box64{r.x0 - ox, r.y0 - oy, r.x1 - ox, r.y1 - oy}});
        };
        add_pin_rects("s", res.terminals.source_rects);
        add_pin_rects("d", res.terminals.drain_rects);
        add_pin_rects("g", res.terminals.gate_rects);
        return m;
    }

    // Center-band passive / BJT macros
    m.band = 1;
    int64_t mw = 3000;
    int64_t mh = 3000;
    const int64_t mult = std::max<int64_t>(1, static_cast<int64_t>(std::llround(dev.param_or("M", 1.0))));

    if (dev.decl_kind == "cap") {
        // MIM cap ~1 fF/um^2 -> area_um2 ~ C_fF
        const double c_ff = std::max(50.0, dev.param_or("value", 500e-15) * 1e15 * static_cast<double>(mult));
        const double side_um = std::clamp(std::sqrt(c_ff), 4.0, 28.0);
        mw = snap_up(static_cast<int64_t>(std::llround(side_um * 1000.0)), g);
        mh = mw;
        m.local_bbox = Box64{0, 0, mw, mh};
        auto add_rect = [&](LayerKey lk, Box64 b) {
            m.polygons.push_back(LayerPoly{lk, {
                Point64{b.x0, b.y0}, Point64{b.x1, b.y0}, Point64{b.x1, b.y1}, Point64{b.x0, b.y1}
            }});
        };
        add_rect(pdk.m2, Box64{0, 0, mw, mh});
        add_rect(pdk.mim_cap, Box64{400, 400, mw - 400, mh - 400});
        const Box64 p_box{100, 100, 600, 600};
        const Box64 n_box{mw - 600, mh - 600, mw - 100, mh - 100};
        add_rect(pdk.m1, p_box);
        add_rect(pdk.m1, n_box);
        if (const std::string pnet = dev.pin_or("p"); !pnet.empty()) m.pins.push_back({"p", pnet, p_box});
        if (const std::string nnet = dev.pin_or("n"); !nnet.empty()) m.pins.push_back({"n", nnet, n_box});
        return m;
    }

    if (dev.decl_kind == "res") {
        const double r_ohm = std::max(100.0, dev.param_or("value", 2000.0));
        const double len_um = std::clamp(std::sqrt(r_ohm / 200.0) * 2.0, 3.0, 22.0);
        mw = snap_up(static_cast<int64_t>(std::llround(len_um * 1000.0)), g);
        mh = snap_up(2000, g);
        m.local_bbox = Box64{0, 0, mw, mh};
        auto add_rect = [&](LayerKey lk, Box64 b) {
            m.polygons.push_back(LayerPoly{lk, {
                Point64{b.x0, b.y0}, Point64{b.x1, b.y0}, Point64{b.x1, b.y1}, Point64{b.x0, b.y1}
            }});
        };
        add_rect(pdk.poly, Box64{0, 400, mw, mh - 400});
        add_rect(pdk.res_mk, Box64{400, 300, mw - 400, mh - 300});
        const Box64 p_box{0, 500, 400, mh - 500};
        const Box64 n_box{mw - 400, 500, mw, mh - 500};
        add_rect(pdk.m1, p_box);
        add_rect(pdk.m1, n_box);
        if (const std::string pnet = dev.pin_or("p"); !pnet.empty()) m.pins.push_back({"p", pnet, p_box});
        if (const std::string nnet = dev.pin_or("n"); !nnet.empty()) m.pins.push_back({"n", nnet, n_box});
        return m;
    }

    // BJT / varactor / generic macro
    const int64_t side = snap_up(static_cast<int64_t>(std::clamp<int64_t>(3000 + 600 * mult, 3000, 12000)), g);
    m.local_bbox = Box64{0, 0, side, side};
    auto add_rect = [&](LayerKey lk, Box64 b) {
        m.polygons.push_back(LayerPoly{lk, {
            Point64{b.x0, b.y0}, Point64{b.x1, b.y0}, Point64{b.x1, b.y1}, Point64{b.x0, b.y1}
        }});
    };
    add_rect(pdk.mos_rules.active, Box64{200, 200, side - 200, side - 200});
    int64_t px_off = 300;
    for (const auto& [pk, net] : dev.pins) {
        const Box64 pbox{px_off, 300, std::min(side - 300, px_off + 500), 800};
        add_rect(pdk.m1, pbox);
        m.pins.push_back(MacroPin{pk, net, pbox});
        px_off += 700;
        if (px_off + 500 >= side) px_off = 300;
    }
    return m;
}

}  // namespace

StandaloneSynthResult synthesize_clay_ir(const ClayFile& file, const SynthOptions& opts) {
    StandaloneSynthResult res{};
    if (file.cells.empty()) {
        res.error = "ClayFile contains no cell definitions";
        return res;
    }
    const ClayCell& cell = file.cells.front();
    const std::string pdk_choice = opts.pdk_override.empty() ? file.pdk : opts.pdk_override;
    const PdkRouteLayers pdk = resolve_pdk_layers(pdk_choice);
    const int64_t g = pdk.grid_nm;

    res.cell_name = cell.name;
    res.pdk_id = pdk.pdk_id;
    res.grid_nm = g;
    res.device_count = cell.devices.size();

    if (cell.devices.empty()) {
        res.error = "cell '" + cell.name + "' contains no devices";
        return res;
    }

    // 1. Materialize all devices into pure C++20 PCell/passive macros
    std::vector<GeneratedMacro> macros;
    macros.reserve(cell.devices.size());
    std::unordered_map<std::string, int32_t> dev_to_idx;
    for (size_t i = 0; i < cell.devices.size(); ++i) {
        dev_to_idx[cell.devices[i].name] = static_cast<int32_t>(i);
        macros.push_back(build_device_macro(cell.devices[i], pdk));
    }

    // 2. Build BandedProblem from device macros + layout_intent
    BandedProblem bp{};
    bp.gap_nm = snap_up(std::max<int64_t>(500, opts.block_gap_nm), g);
    bp.grid_nm = g;
    bp.enforce_band_order = true;
    bp.objective_area = true;
    bp.dense_2d = true;
    bp.max_evals = opts.max_place_evals;

    bp.blocks.reserve(macros.size());
    for (const auto& m : macros) {
        BandedBlock bb{};
        bb.name = m.name;
        bb.w = m.local_bbox.width();
        bb.h = m.local_bbox.height();
        bb.band = m.band;
        bp.blocks.push_back(std::move(bb));
    }

    // Consume @order(A, B, left_of|above) & @anchor(A, edge=left|right) from layout_intent
    if (const ClayLayoutIntent* li = file.intent_for_cell(cell.name)) {
        for (const auto& ent : li->entries) {
            if (ent.kind == "order" && ent.positional.size() >= 3) {
                auto ia = dev_to_idx.find(ent.positional[0]);
                auto ib = dev_to_idx.find(ent.positional[1]);
                if (ia != dev_to_idx.end() && ib != dev_to_idx.end() && ia->second != ib->second) {
                    if (ent.positional[2] == "left_of") {
                        bp.order_pairs.emplace_back(ia->second, ib->second);
                    } else if (ent.positional[2] == "above") {
                        bp.vertical_order_pairs.emplace_back(ib->second, ia->second);
                    }
                }
            } else if (ent.kind == "anchor" && !ent.positional.empty()) {
                auto ia = dev_to_idx.find(ent.positional[0]);
                if (ia != dev_to_idx.end()) {
                    const std::string edge = ent.kwarg_or("edge");
                    if (edge == "left") bp.blocks[static_cast<size_t>(ia->second)].left_pull = true;
                    else if (edge == "right") bp.blocks[static_cast<size_t>(ia->second)].right_pull = true;
                }
            }
        }
    }

    // 3. Pure C++20 Banded 2D Area Placement
    const BandedSolution sol = solve_banded_area(bp);
    if (sol.status != BandedStatus::Optimal && sol.status != BandedStatus::Feasible) {
        res.error = "banded placement failed: " + sol.detail;
        return res;
    }

    // 4. Assemble LayoutDB + collect global pin positions per net
    LayoutDB db;
    db.name = cell.name;
    auto& top_cell = db.add_cell(cell.name);

    struct NetPinSite {
        int64_t cx = 0;
        int64_t cy = 0;
        Box64 box{};
    };
    std::map<std::string, std::vector<NetPinSite>> net_pins;

    res.macros.reserve(macros.size());
    for (size_t i = 0; i < macros.size(); ++i) {
        const int64_t ox = sol.xs[i];
        const int64_t oy = sol.ys[i];
        const auto& m = macros[i];
        res.macros.push_back(PlacedMacroInfo{
            m.name, m.kind, m.band,
            Box64{ox, oy, ox + m.local_bbox.width(), oy + m.local_bbox.height()}
        });
        for (const auto& lp : m.polygons) {
            std::vector<Point64> pts;
            pts.reserve(lp.vertices.size());
            for (const auto& pt : lp.vertices) {
                pts.push_back(Point64{pt.x + ox, pt.y + oy});
            }
            top_cell.add_polygon(lp.key, std::move(pts));
        }
        for (const auto& mp : m.pins) {
            const Box64 gb{
                mp.local_box.x0 + ox, mp.local_box.y0 + oy,
                mp.local_box.x1 + ox, mp.local_box.y1 + oy
            };
            const int64_t cx = snap_up((gb.x0 + gb.x1) / 2, g);
            const int64_t cy = snap_up((gb.y0 + gb.y1) / 2, g);
            net_pins[mp.net_name].push_back(NetPinSite{cx, cy, gb});
        }
    }

    res.net_count = net_pins.size();

    // 5. Pure C++20 Multi-Layer Manhattan Trunk-and-Branch Router + Per-Net Merge
    if (opts.emit_routes && !net_pins.empty()) {
        const int64_t w_half = snap_up(std::max<int64_t>(100, opts.wire_width_nm / 2), g);
        const int64_t pitch = snap_up(std::max<int64_t>(opts.wire_width_nm * 2, opts.track_pitch_nm), g);
        const int64_t v_half = snap_up(pdk.via_size_nm / 2, g);
        const int64_t pad_half = snap_up(v_half + pdk.via_enc_nm, g);

        std::vector<NetLayerBoxes> route_boxes;
        int64_t track_y = snap_up(sol.H + pitch, g);
        size_t routed = 0;

        for (const auto& [net, sites] : net_pins) {
            if (sites.empty()) continue;
            // Emit text label on first pin / trunk
            if (sites.size() == 1) {
                top_cell.add_text(pdk.m2_label, net, Point64{sites[0].cx, sites[0].cy});
                continue;
            }
            int64_t min_x = sites[0].cx;
            int64_t max_x = sites[0].cx;
            for (const auto& s : sites) {
                min_x = std::min(min_x, s.cx);
                max_x = std::max(max_x, s.cx);
            }
            std::vector<Box64> m2_boxes;
            std::vector<Box64> m3_boxes;
            std::vector<Box64> v1_boxes;
            std::vector<Box64> v2_boxes;

            // M2 horizontal trunk along dedicated track_y
            m2_boxes.push_back(Box64{
                min_x - pad_half, track_y - w_half,
                max_x + pad_half, track_y + w_half
            });
            top_cell.add_text(pdk.m2_label, net, Point64{min_x, track_y});

            // M3 vertical risers + VIA1/VIA2 stacks from each pin site to trunk
            for (const auto& s : sites) {
                const int64_t ylo = std::min(s.cy, track_y) - pad_half;
                const int64_t yhi = std::max(s.cy, track_y) + pad_half;
                m3_boxes.push_back(Box64{s.cx - w_half, ylo, s.cx + w_half, yhi});
                // M2 landing pad at pin site + VIA1 (M1-M2) + VIA2 (M2-M3)
                m2_boxes.push_back(Box64{s.cx - pad_half, s.cy - pad_half, s.cx + pad_half, s.cy + pad_half});
                v1_boxes.push_back(Box64{s.cx - v_half, s.cy - v_half, s.cx + v_half, s.cy + v_half});
                v2_boxes.push_back(Box64{s.cx - v_half, s.cy - v_half, s.cx + v_half, s.cy + v_half});
                // VIA2 at trunk junction
                v2_boxes.push_back(Box64{s.cx - v_half, track_y - v_half, s.cx + v_half, track_y + v_half});
            }

            route_boxes.push_back(NetLayerBoxes{net, pdk.m2, std::move(m2_boxes)});
            route_boxes.push_back(NetLayerBoxes{net, pdk.m3, std::move(m3_boxes)});
            route_boxes.push_back(NetLayerBoxes{net, pdk.via1, std::move(v1_boxes)});
            route_boxes.push_back(NetLayerBoxes{net, pdk.via2, std::move(v2_boxes)});

            track_y += pitch;
            ++routed;
        }
        res.routed_net_count = routed;

        // Exact per-net x per-layer polygon merge via pure C++20 PolygonSet
        const auto merged = merge_net_boxes_exact(route_boxes);
        for (const auto& lp : merged) {
            top_cell.polygons.push_back(lp);
        }
    }

    // 6. Export LayoutDB -> GdsLibrary (zero Python, zero KLayout)
    db.snap_all_to_grid(g);
    res.gds_lib = db.to_gds_library();
    res.polygon_count = top_cell.polygons.size();
    (void)top_cell.compute_bbox(res.die_bbox_nm);
    res.ok = true;
    return res;
}

StandaloneSynthResult synthesize_clay_source(const std::string& source, const SynthOptions& opts) {
    const ParseResult pr = parse_clay_source(source);
    if (!pr.ok) {
        StandaloneSynthResult res{};
        res.error = "parse error at line " + std::to_string(pr.line) + ":" + std::to_string(pr.col) + ": " + pr.error;
        return res;
    }
    return synthesize_clay_ir(pr.file, opts);
}

StandaloneSynthResult synthesize_clay_file_to_gds(
    const std::string& clay_path,
    const std::string& out_gds_path,
    const SynthOptions& opts) {
    const ParseResult pr = parse_clay_file(clay_path);
    if (!pr.ok) {
        StandaloneSynthResult res{};
        res.error = "parse error at line " + std::to_string(pr.line) + ":" + std::to_string(pr.col) + ": " + pr.error;
        return res;
    }
    StandaloneSynthResult res = synthesize_clay_ir(pr.file, opts);
    if (!res.ok) return res;
    std::string werr;
    if (!clay_pnr::gds::write_gds_file(res.gds_lib, out_gds_path, &werr)) {
        res.ok = false;
        res.error = "GDS write failed: " + werr;
    }
    return res;
}

}  // namespace clay_pnr::engine
