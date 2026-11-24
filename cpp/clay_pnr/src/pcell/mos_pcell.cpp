// Pure C++20 Rule-Driven Parameterized MOS & Shared-Diffusion Row PCell Engine.
// Strictly free of <Python.h> and KLayout.

#include "clay_pnr/pcell/mos_pcell.hpp"

#include <algorithm>
#include <stdexcept>

namespace clay_pnr::pcell {

namespace {

inline int64_t snap_up_grid(int64_t v, int64_t g) noexcept {
    if (g <= 1) return v;
    const int64_t r = v % g;
    return (r == 0) ? v : (v + (g - r));
}

inline void add_sd_contacts(LayoutCell& cell,
                            int64_t cx,
                            int64_t act_y0,
                            int64_t act_y1,
                            const MosTechRules& rules,
                            Box64& out_m1_strap) {
    const int64_t cs = rules.cont_size_nm;
    const int64_t csp = rules.cont_space_nm;
    const int64_t aenc = rules.active_enc_cont_nm;
    const int64_t avail_h = (act_y1 - act_y0) - 2 * aenc;
    int64_t n_cont = 1;
    if (avail_h >= cs) {
        n_cont = 1 + (avail_h - cs) / (cs + csp);
    }
    const int64_t stack_h = n_cont * cs + (n_cont - 1) * csp;
    const int64_t y_start = clay_pnr::layout::snap_coord_nearest_up(
        (act_y0 + act_y1 - stack_h) / 2, rules.grid_nm);
    const int64_t cx0 = clay_pnr::layout::snap_coord_nearest_up(cx - cs / 2, rules.grid_nm);
    const int64_t cx1 = cx0 + cs;

    for (int64_t k = 0; k < n_cont; ++k) {
        const int64_t cy0 = y_start + k * (cs + csp);
        cell.add_box(rules.contact, Box64{cx0, cy0, cx1, cy0 + cs});
    }

    const int64_t m1_w = std::max(rules.m1_min_w_nm, cs + 2 * rules.m1_enc_cont_nm);
    const int64_t m1_x0 = clay_pnr::layout::snap_coord_nearest_up(cx - m1_w / 2, rules.grid_nm);
    const int64_t m1_x1 = m1_x0 + m1_w;
    const int64_t m1_y0 = std::min(act_y0, y_start - rules.m1_enc_cont_nm);
    const int64_t m1_y1 = std::max(act_y1, y_start + stack_h + rules.m1_enc_cont_nm);
    out_m1_strap = Box64{m1_x0, m1_y0, m1_x1, m1_y1};
    cell.add_box(rules.metal1, out_m1_strap);
}

}  // namespace

MosTechRules gf180mcu_mos_rules() noexcept {
    MosTechRules r{};
    r.pdk_id = "gf180mcu";
    r.grid_nm = 5;
    r.active = LayerKey{22, 0};
    r.poly = LayerKey{30, 0};
    r.contact = LayerKey{33, 0};
    r.metal1 = LayerKey{34, 0};
    r.nplus = LayerKey{32, 0};
    r.pplus = LayerKey{31, 0};
    r.nwell = LayerKey{21, 0};
    r.poly_ext_nm = 220;
    r.poly_to_cont_nm = 180;
    r.cont_size_nm = 220;
    r.cont_space_nm = 280;
    r.active_enc_cont_nm = 100;
    r.m1_enc_cont_nm = 60;
    r.m1_min_w_nm = 230;
    r.implant_enc_active_nm = 230;
    r.nwell_enc_active_nm = 430;
    r.gate_strap_space_nm = 280;
    return r;
}

MosTechRules sky130_mos_rules() noexcept {
    MosTechRules r{};
    r.pdk_id = "sky130";
    r.grid_nm = 5;
    r.active = LayerKey{65, 20};
    r.poly = LayerKey{66, 20};
    r.contact = LayerKey{66, 44};
    r.metal1 = LayerKey{67, 20};
    r.nplus = LayerKey{93, 44};
    r.pplus = LayerKey{94, 20};
    r.nwell = LayerKey{64, 20};
    r.poly_ext_nm = 130;
    r.poly_to_cont_nm = 110;
    r.cont_size_nm = 170;
    r.cont_space_nm = 170;
    r.active_enc_cont_nm = 60;
    r.m1_enc_cont_nm = 80;
    r.m1_min_w_nm = 170;
    r.implant_enc_active_nm = 130;
    r.nwell_enc_active_nm = 180;
    r.gate_strap_space_nm = 200;
    return r;
}

SharedRowResult generate_shared_diffusion_row(
    const std::string& row_name,
    bool is_pmos,
    int64_t w_nm,
    int64_t l_nm,
    const std::vector<std::string>& finger_owners,
    int32_t dummies_left,
    int32_t dummies_right,
    const MosTechRules& rules) {
    SharedRowResult out{};
    out.cell.name = row_name;
    const int64_t g = std::max<int64_t>(1, rules.grid_nm);
    const int64_t W = snap_up_grid(std::max<int64_t>(w_nm, rules.cont_size_nm + 2 * rules.active_enc_cont_nm), g);
    const int64_t L = snap_up_grid(std::max<int64_t>(l_nm, 100), g);
    const int32_t dl = std::max<int32_t>(0, dummies_left);
    const int32_t dr = std::max<int32_t>(0, dummies_right);
    const int32_t n_act = static_cast<int32_t>(finger_owners.size());
    const int32_t total_f = dl + n_act + dr;
    if (total_f <= 0) return out;

    // Gate-to-gate spacing holds one S/D contact + 2 * poly_to_cont_nm.
    const int64_t sd_inner_w = snap_up_grid(2 * rules.poly_to_cont_nm + rules.cont_size_nm, g);
    const int64_t pitch = L + sd_inner_w;
    const int64_t sd_end_w = snap_up_grid(rules.poly_to_cont_nm + rules.cont_size_nm + rules.active_enc_cont_nm, g);
    out.gate_pitch_nm = pitch;

    const int64_t act_x0 = 0;
    const int64_t act_x1 = 2 * sd_end_w + total_f * L + (total_f - 1) * sd_inner_w;
    const int64_t act_y0 = 0;
    const int64_t act_y1 = W;
    out.active_bbox = Box64{act_x0, act_y0, act_x1, act_y1};
    out.cell.add_box(rules.active, out.active_bbox);

    // Implant (N+/P+) and NWELL (for PMOS)
    const LayerKey imp = is_pmos ? rules.pplus : rules.nplus;
    out.cell.add_box(imp, Box64{
        act_x0 - rules.implant_enc_active_nm,
        act_y0 - rules.implant_enc_active_nm,
        act_x1 + rules.implant_enc_active_nm,
        act_y1 + rules.implant_enc_active_nm,
    });
    if (is_pmos) {
        out.cell.add_box(rules.nwell, Box64{
            act_x0 - rules.nwell_enc_active_nm,
            act_y0 - rules.nwell_enc_active_nm,
            act_x1 + rules.nwell_enc_active_nm,
            act_y1 + rules.nwell_enc_active_nm,
        });
    }

    // Draw poly gates and record per-owner gate boxes.
    const int64_t poly_y0 = act_y0 - rules.poly_ext_nm;
    const int64_t poly_y1 = act_y1 + rules.poly_ext_nm;
    std::vector<Box64> sd_straps(static_cast<size_t>(total_f + 1));

    // Draw S/D contact columns: 0 .. total_f
    for (int32_t sdi = 0; sdi <= total_f; ++sdi) {
        int64_t cx = 0;
        if (sdi == 0) {
            cx = act_x0 + rules.active_enc_cont_nm + rules.cont_size_nm / 2;
        } else if (sdi == total_f) {
            cx = act_x1 - rules.active_enc_cont_nm - rules.cont_size_nm / 2;
        } else {
            const int64_t left_gate_x1 = sd_end_w + (sdi - 1) * pitch + L;
            cx = left_gate_x1 + sd_inner_w / 2;
        }
        add_sd_contacts(out.cell, cx, act_y0, act_y1, rules, sd_straps[static_cast<size_t>(sdi)]);
    }

    // Draw all gate fingers (left dummies, active fingers, right dummies).
    for (int32_t fi = 0; fi < total_f; ++fi) {
        const int64_t gx0 = sd_end_w + fi * pitch;
        const int64_t gx1 = gx0 + L;
        const Box64 gate_box{gx0, poly_y0, gx1, poly_y1};
        out.cell.add_box(rules.poly, gate_box);

        if (fi >= dl && fi < dl + n_act) {
            const int32_t act_idx = fi - dl;
            const std::string& owner = finger_owners[static_cast<size_t>(act_idx)];
            auto& term = out.member_terminals[owner];
            term.gate_rects.push_back(gate_box);
            // Alternate source/drain per finger so adjacent fingers share S/D naturally!
            const Box64& left_sd = sd_straps[static_cast<size_t>(fi)];
            const Box64& right_sd = sd_straps[static_cast<size_t>(fi + 1)];
            if ((act_idx & 1) == 0) {
                term.source_rects.push_back(left_sd);
                term.drain_rects.push_back(right_sd);
            } else {
                term.drain_rects.push_back(left_sd);
                term.source_rects.push_back(right_sd);
            }
        }
    }

    out.cell.snap_to_grid(g);

    // Center the whole row cleanly around (0, 0) on the manufacturing grid.
    Box64 raw_bb{};
    if (out.cell.compute_bbox(raw_bb)) {
        const int64_t dx = -clay_pnr::layout::snap_coord_nearest_up((raw_bb.x0 + raw_bb.x1) / 2, g);
        const int64_t dy = -clay_pnr::layout::snap_coord_nearest_up((raw_bb.y0 + raw_bb.y1) / 2, g);
        out.cell.translate(dx, dy);
        out.active_bbox = Box64{out.active_bbox.x0 + dx, out.active_bbox.y0 + dy,
                                out.active_bbox.x1 + dx, out.active_bbox.y1 + dy};
        for (auto& [owner, term] : out.member_terminals) {
            (void)owner;
            auto shift_vec = [&](std::vector<Box64>& v) {
                for (auto& b : v) b = Box64{b.x0 + dx, b.y0 + dy, b.x1 + dx, b.y1 + dy};
            };
            shift_vec(term.source_rects);
            shift_vec(term.drain_rects);
            shift_vec(term.gate_rects);
            // Compute per-member terminal bbox.
            bool init = false;
            Box64 mbb{};
            auto acc = [&](const std::vector<Box64>& v) {
                for (const auto& b : v) {
                    if (!init) { mbb = b; init = true; }
                    else {
                        mbb.x0 = std::min(mbb.x0, b.x0);
                        mbb.y0 = std::min(mbb.y0, b.y0);
                        mbb.x1 = std::max(mbb.x1, b.x1);
                        mbb.y1 = std::max(mbb.y1, b.y1);
                    }
                }
            };
            acc(term.source_rects);
            acc(term.drain_rects);
            acc(term.gate_rects);
            term.bbox = mbb;
        }
        out.cell.compute_bbox(out.bbox);
    }

    return out;
}

MosPCellResult generate_mos_pcell(const MosSpec& spec, const MosTechRules& rules) {
    const int32_t f = std::max<int32_t>(1, spec.fingers);
    std::vector<std::string> owners(static_cast<size_t>(f), spec.name);
    SharedRowResult row = generate_shared_diffusion_row(
        spec.name, spec.is_pmos, spec.w_nm, spec.l_nm, owners,
        spec.dummies_left, spec.dummies_right, rules);
    MosPCellResult res{};
    res.cell = std::move(row.cell);
    res.active_bbox = row.active_bbox;
    res.gate_pitch_nm = row.gate_pitch_nm;
    auto it = row.member_terminals.find(spec.name);
    if (it != row.member_terminals.end()) {
        res.terminals = std::move(it->second);
    }
    res.terminals.bbox = row.bbox;
    return res;
}

}  // namespace clay_pnr::pcell

