// Pure C++20 implementation of PathFinderGridCore and RectClipperCore.
// Strictly standalone C++20 with zero CPython runtime dependencies.

#include "clay_pnr/route/pathfinder_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace clay_pnr::route {

using core::Box64;
using core::CheckpointHook;
using core::ceil_div_i64;
using core::floor_div_i64;
using core::py_round_i64;

bool PathFinderGridCore::init(
    int32_t nx_, int32_t ny_, int32_t n_layers_,
    int64_t x0_, int64_t y0_, int64_t pitch_,
    const std::vector<uint8_t>& pref_h,
    const std::vector<int32_t>& min_w,
    const std::vector<int32_t>& min_s,
    const std::vector<int32_t>& v_cut,
    const std::vector<int32_t>& v_eb,
    const std::vector<int32_t>& v_ea,
    const std::vector<int32_t>& v_sp
) {
    if (nx_ <= 0 || ny_ <= 0 || n_layers_ <= 0 || n_layers_ > 30 || pitch_ <= 0) {
        return false;
    }
    const size_t nl = static_cast<size_t>(n_layers_);
    const size_t nv = nl - 1;
    if (pref_h.size() != nl || min_w.size() != nl || min_s.size() != nl ||
        v_cut.size() != nv || v_eb.size() != nv || v_ea.size() != nv || v_sp.size() != nv) {
        return false;
    }
    const int64_t nxy64 = static_cast<int64_t>(nx_) * static_cast<int64_t>(ny_);
    const int64_t ntot64 = nxy64 * static_cast<int64_t>(n_layers_);
    if (ntot64 <= 0 || ntot64 * kDirRunStates > std::numeric_limits<int32_t>::max()) {
        return false;
    }

    nx = nx_;
    ny = ny_;
    n_layers = n_layers_;
    nxy = static_cast<int32_t>(nxy64);
    ntotal = static_cast<int32_t>(ntot64);
    nvia_total = (n_layers > 1) ? (n_layers - 1) * nxy : 0;
    x0_nm = x0_;
    y0_nm = y0_;
    pitch_nm = pitch_;

    pref_horiz = pref_h;
    min_width_nm = min_w;
    min_space_nm = min_s;
    via_cut_size_nm = v_cut;
    via_enc_below_nm = v_eb;
    via_enc_above_nm = v_ea;
    via_cut_space_nm = v_sp;

    obs_by_gi.assign(nl, {});
    via_obs_by_gl.assign(nv, {});
    mim_clear_by_gi.assign(nl, {});
    mim_forbidden_by_gl.assign(nv, {});
    has_envelope = false;
    env_x0 = 0;
    env_y0 = 0;
    env_x1 = 0;
    env_y1 = 0;

    over_active_bits.clear();
    over_active_gi_floor = n_layers;
    over_active_penalty = 0.0;

    nets.clear();

    const size_t ntot_sz = static_cast<size_t>(ntotal);
    occ.assign(ntot_sz, 0);
    history.assign(ntot_sz, 0.0);
    pad_keepout_count.assign(ntot_sz, 0);
    via_pad_history.assign(static_cast<size_t>(std::max<int32_t>(1, nvia_total)), 0.0);
    reserved_owner.assign(ntot_sz, -1);
    unrouted_pin_count.assign(ntot_sz, 0);
    metal_owner.assign(ntot_sz, -1);
    committed_owner.assign(ntot_sz, -1);

    touched_occ.clear();
    touched_pad_keepout.clear();
    touched_reserved.clear();
    touched_unrouted_pins.clear();
    touched_metal_owner.clear();
    touched_committed.clear();
    round_exact_pk.clear();

    astar_stamp.assign(ntot_sz, 0U);
    astar_g.assign(ntot_sz, 0.0);
    astar_parent.assign(ntot_sz, -1);
    astar_gen = 2;

    dir_stamp.clear();
    dir_g.clear();
    dir_parent.clear();
    dir_gen = 2;
    dir_runs_arr.clear();
    dir_runs_stamp.clear();
    dir_runs_gen = 1;

    tree_stamp.assign(ntot_sz, 0U);
    tree_gen = 1;
    wire_stamp.assign(ntot_sz, 0U);
    wire_gen = 1;
    mirror_stamp.assign(ntot_sz, 0U);
    mirror_gen = 1;
    dedup_stamp.assign(ntot_sz, 0U);
    dedup_gen = 1;
    self_pk_stamp.assign(ntot_sz, 0U);
    self_pk_gen = 1;
    bfs_stamp.assign(ntot_sz, 0U);
    bfs_gen = 1;

    heap.reserve(4096);
    return true;
}

uint32_t PathFinderGridCore::next_astar_gen() {
    astar_gen += 2;
    if (astar_gen >= 0xFFFFFFFCU) {
        std::fill(astar_stamp.begin(), astar_stamp.end(), 0U);
        astar_gen = 2;
    }
    return astar_gen;
}

uint32_t PathFinderGridCore::next_dir_gen() {
    if (dir_stamp.empty()) {
        const size_t nstates = static_cast<size_t>(ntotal) * static_cast<size_t>(kDirRunStates);
        dir_stamp.assign(nstates, 0U);
        dir_g.assign(nstates, 0.0);
        dir_parent.assign(nstates, -1);
        dir_runs_arr.assign(static_cast<size_t>(ntotal), 0);
        dir_runs_stamp.assign(static_cast<size_t>(ntotal), 0U);
        dir_gen = 2;
        dir_runs_gen = 1;
    }
    dir_gen += 2;
    if (dir_gen >= 0xFFFFFFFCU) {
        std::fill(dir_stamp.begin(), dir_stamp.end(), 0U);
        dir_gen = 2;
    }
    return dir_gen;
}

uint32_t PathFinderGridCore::next_gen(std::vector<uint32_t>& arr, uint32_t& gen) {
    ++gen;
    if (gen == 0U || gen >= 0xFFFFFFFEU) {
        std::fill(arr.begin(), arr.end(), 0U);
        gen = 1;
    }
    return gen;
}

std::vector<Off2D> PathFinderGridCore::compute_via_keepout_offsets(
    int64_t pitch,
    int64_t pad_half,
    int64_t wire_half,
    int64_t ms,
    bool exact_diagonal
) {
    std::vector<Off2D> out;
    if (pad_half <= wire_half) return out;
    const int32_t r = static_cast<int32_t>(std::max<int64_t>(
        0, static_cast<int64_t>(std::ceil(
               (static_cast<double>(pad_half) + static_cast<double>(ms) * 0.5) / static_cast<double>(pitch)))
    ));
    const double f_half = std::max(static_cast<double>(pad_half), static_cast<double>(ms) * 0.5);
    for (int32_t dx = -r; dx <= r; ++dx) {
        for (int32_t dy = -r; dy <= r; ++dy) {
            if (exact_diagonal && dx != 0 && dy != 0) {
                const double gx = std::max(
                    0.0, static_cast<double>(std::abs(dx) * pitch) - static_cast<double>(pad_half) - f_half);
                const double gy = std::max(
                    0.0, static_cast<double>(std::abs(dy) * pitch) - static_cast<double>(pad_half) - f_half);
                if (std::hypot(gx, gy) >= static_cast<double>(ms)) {
                    continue;
                }
            }
            out.push_back(Off2D{dx, dy});
        }
    }
    return out;
}

int64_t PathFinderGridCore::via_pad_extra_halo_nm(int32_t gi, int32_t wire_width_nm) const noexcept {
    const int64_t wire_half = std::max<int64_t>(0, wire_width_nm / 2);
    int64_t extra = 0;
    for (int32_t gl : {gi - 1, gi}) {
        if (gl < 0 || gl + 1 >= n_layers) continue;
        const int32_t vsz = via_cut_size_nm[static_cast<size_t>(gl)];
        if (vsz <= 0) continue;
        const int32_t enc = (gl == gi - 1) ? via_enc_above_nm[static_cast<size_t>(gl)]
                                           : via_enc_below_nm[static_cast<size_t>(gl)];
        const int64_t pad_half = (static_cast<int64_t>(vsz) + 1) / 2 + std::max<int64_t>(0, enc);
        if (pad_half - wire_half > extra) {
            extra = pad_half - wire_half;
        }
    }
    return std::max<int64_t>(0, extra);
}

void PathFinderGridCore::raster_rect_into_bits(
    std::vector<uint64_t>& bits,
    int32_t layer_offset,
    double x0, double y0, double x1, double y1,
    int64_t halo_nm
) const noexcept {
    const double lo_x = std::min(x0, x1) * kNmPerUm - static_cast<double>(halo_nm);
    const double hi_x = std::max(x0, x1) * kNmPerUm + static_cast<double>(halo_nm);
    const double lo_y = std::min(y0, y1) * kNmPerUm - static_cast<double>(halo_nm);
    const double hi_y = std::max(y0, y1) * kNmPerUm + static_cast<double>(halo_nm);
    const double inv_p = 1.0 / static_cast<double>(pitch_nm);
    int32_t ix_lo = std::max<int32_t>(0, static_cast<int32_t>(std::floor((lo_x - static_cast<double>(x0_nm)) * inv_p)));
    int32_t ix_hi = std::min<int32_t>(nx - 1, static_cast<int32_t>(std::ceil((hi_x - static_cast<double>(x0_nm)) * inv_p)));
    int32_t iy_lo = std::max<int32_t>(0, static_cast<int32_t>(std::floor((lo_y - static_cast<double>(y0_nm)) * inv_p)));
    int32_t iy_hi = std::min<int32_t>(ny - 1, static_cast<int32_t>(std::ceil((hi_y - static_cast<double>(y0_nm)) * inv_p)));
    // Trim boundaries so inner loop needs no per-cell comparisons
    while (ix_lo <= ix_hi && static_cast<double>(x0_nm + static_cast<int64_t>(ix_lo) * pitch_nm) < lo_x) ++ix_lo;
    while (ix_hi >= ix_lo && static_cast<double>(x0_nm + static_cast<int64_t>(ix_hi) * pitch_nm) > hi_x) --ix_hi;
    while (iy_lo <= iy_hi && static_cast<double>(y0_nm + static_cast<int64_t>(iy_lo) * pitch_nm) < lo_y) ++iy_lo;
    while (iy_hi >= iy_lo && static_cast<double>(y0_nm + static_cast<int64_t>(iy_hi) * pitch_nm) > hi_y) --iy_hi;
    if (ix_lo > ix_hi || iy_lo > iy_hi) return;
    for (int32_t iy = iy_lo; iy <= iy_hi; ++iy) {
        const int32_t row_base = layer_offset + iy * nx;
        for (int32_t ix = ix_lo; ix <= ix_hi; ++ix) {
            bitset_set(bits, row_base + ix);
        }
    }
}

void PathFinderGridCore::mark_envelope_blocked_bits(
    std::vector<uint64_t>& bits,
    int32_t layer_offset,
    int64_t edge_half_nm
) const noexcept {
    if (!has_envelope) return;
    if ((env_x1 - env_x0) < 2 * edge_half_nm || (env_y1 - env_y0) < 2 * edge_half_nm) {
        return;
    }
    const double inv_p = 1.0 / static_cast<double>(pitch_nm);
    const int32_t vx_lo = std::max<int32_t>(
        0, static_cast<int32_t>(std::ceil(static_cast<double>(env_x0 + edge_half_nm - x0_nm) * inv_p)));
    const int32_t vx_hi = std::min<int32_t>(
        nx - 1, static_cast<int32_t>(std::floor(static_cast<double>(env_x1 - edge_half_nm - x0_nm) * inv_p)));
    const int32_t vy_lo = std::max<int32_t>(
        0, static_cast<int32_t>(std::ceil(static_cast<double>(env_y0 + edge_half_nm - y0_nm) * inv_p)));
    const int32_t vy_hi = std::min<int32_t>(
        ny - 1, static_cast<int32_t>(std::floor(static_cast<double>(env_y1 - edge_half_nm - y0_nm) * inv_p)));
    if (vx_lo > vx_hi || vy_lo > vy_hi) return;
    for (int32_t iy = 0; iy < ny; ++iy) {
        const int32_t row = layer_offset + iy * nx;
        for (int32_t ix = 0; ix < vx_lo; ++ix) bitset_set(bits, row + ix);
        for (int32_t ix = vx_hi + 1; ix < nx; ++ix) bitset_set(bits, row + ix);
    }
    for (int32_t iy = 0; iy < vy_lo; ++iy) {
        const int32_t row = layer_offset + iy * nx;
        for (int32_t ix = vx_lo; ix <= vx_hi; ++ix) bitset_set(bits, row + ix);
    }
    for (int32_t iy = vy_hi + 1; iy < ny; ++iy) {
        const int32_t row = layer_offset + iy * nx;
        for (int32_t ix = vx_lo; ix <= vx_hi; ++ix) bitset_set(bits, row + ix);
    }
}

void PathFinderGridCore::set_over_active(
    const std::vector<RectUm>& rects_um,
    int32_t gi_floor,
    double penalty
) {
    over_active_gi_floor = gi_floor;
    over_active_penalty = penalty;
    over_active_bits.clear();
    if (penalty <= 0.0 || gi_floor >= n_layers || rects_um.empty()) {
        return;
    }
    over_active_bits.assign((static_cast<size_t>(nxy) + 63U) >> 6, 0ULL);
    for (const auto& r : rects_um) {
        raster_rect_into_bits(over_active_bits, 0, r.x0, r.y0, r.x1, r.y1, 0);
    }
}

void PathFinderGridCore::clear_static_obstacles() noexcept {
    for (auto& v : obs_by_gi) v.clear();
    for (auto& v : via_obs_by_gl) v.clear();
    for (auto& v : mim_clear_by_gi) v.clear();
    for (auto& v : mim_forbidden_by_gl) v.clear();
    has_envelope = false;
    env_x0 = 0;
    env_y0 = 0;
    env_x1 = 0;
    env_y1 = 0;
}

void PathFinderGridCore::set_envelope(int64_t ex0, int64_t ey0, int64_t ex1, int64_t ey1) noexcept {
    has_envelope = true;
    env_x0 = ex0;
    env_y0 = ey0;
    env_x1 = ex1;
    env_y1 = ey1;
}

void PathFinderGridCore::add_obs_rect(
    int32_t gi, double x0, double y0, double x1, double y1, int32_t owner_id
) {
    if (gi < 0 || gi >= n_layers) return;
    const double lo_x = std::min(x0, x1);
    const double hi_x = std::max(x0, x1);
    const double lo_y = std::min(y0, y1);
    const double hi_y = std::max(y0, y1);
    obs_by_gi[static_cast<size_t>(gi)].push_back(ObsRect{
        gi, x0, y0, x1, y1,
        py_round_i64(lo_x * kNmPerUm),
        py_round_i64(lo_y * kNmPerUm),
        py_round_i64(hi_x * kNmPerUm),
        py_round_i64(hi_y * kNmPerUm),
        owner_id
    });
}

void PathFinderGridCore::add_via_obs_rect(
    int32_t gl, double x0, double y0, double x1, double y1,
    int32_t owner_id, bool is_cap_plate_via
) {
    if (gl < 0 || gl >= n_layers - 1) return;
    const double lo_x = std::min(x0, x1);
    const double hi_x = std::max(x0, x1);
    const double lo_y = std::min(y0, y1);
    const double hi_y = std::max(y0, y1);
    via_obs_by_gl[static_cast<size_t>(gl)].push_back(ViaObsRect{
        gl, x0, y0, x1, y1,
        py_round_i64(lo_x * kNmPerUm),
        py_round_i64(lo_y * kNmPerUm),
        py_round_i64(hi_x * kNmPerUm),
        py_round_i64(hi_y * kNmPerUm),
        owner_id, is_cap_plate_via
    });
}

void PathFinderGridCore::add_mim_clear_rect(
    int32_t gi, int64_t bx0, int64_t by0, int64_t bx1, int64_t by1,
    int64_t clr_nm, int32_t owner_id
) {
    if (gi < 0 || gi >= n_layers) return;
    mim_clear_by_gi[static_cast<size_t>(gi)].push_back(MimClearRect{
        gi, bx0, by0, bx1, by1, clr_nm, owner_id
    });
}

void PathFinderGridCore::add_mim_forbidden_cut_rect(
    int32_t gl, int64_t bx0, int64_t by0, int64_t bx1, int64_t by1
) {
    if (gl < 0 || gl >= n_layers - 1) return;
    mim_forbidden_by_gl[static_cast<size_t>(gl)].push_back(MimForbiddenCutRect{
        gl, bx0, by0, bx1, by1
    });
}

void PathFinderGridCore::prepare_net(
    int32_t net_id,
    int32_t route_w_nm,
    int32_t own_width_floor,
    uint32_t allowed_mask,
    const std::vector<int32_t>& pin_cell_idxs,
    const std::vector<std::vector<std::pair<int64_t, int64_t>>>& pin_xy_by_gi,
    int32_t trunk_gi,
    bool add_unowned_via_pad_extra,
    bool build_via_blocked
) {
    if (net_id < 0) return;
    if (static_cast<size_t>(net_id) >= nets.size()) {
        nets.resize(static_cast<size_t>(net_id) + 1);
    }
    NetData& nd = nets[static_cast<size_t>(net_id)];
    nd.route_w_nm = route_w_nm;
    nd.own_width_floor = own_width_floor;
    nd.trunk_gi = trunk_gi;
    nd.allowed_mask = allowed_mask;

    const size_t nwords_total = (static_cast<size_t>(ntotal) + 63U) >> 6;
    const size_t nwords_xy = (static_cast<size_t>(nxy) + 63U) >> 6;
    nd.blocked_bits.assign(nwords_total, 0ULL);
    nd.neck_bits.assign(nwords_xy, 0ULL);

    // Unique pin cells + neck bits
    nd.unique_pin_cells.clear();
    const uint32_t dgen = next_gen(dedup_stamp, dedup_gen);
    const int32_t neck_r = std::max<int32_t>(
        1, static_cast<int32_t>(std::ceil(static_cast<double>(route_w_nm) / static_cast<double>(pitch_nm)))
    );
    for (int32_t pidx : pin_cell_idxs) {
        if (pidx < 0 || pidx >= ntotal) continue;
        if (dedup_stamp[static_cast<size_t>(pidx)] != dgen) {
            dedup_stamp[static_cast<size_t>(pidx)] = dgen;
            nd.unique_pin_cells.push_back(pidx);
        }
        int32_t px = 0, py = 0;
        unpack_xy(pidx, px, py);
        const int32_t ylo = std::max<int32_t>(0, py - neck_r);
        const int32_t yhi = std::min<int32_t>(ny - 1, py + neck_r);
        const int32_t xlo = std::max<int32_t>(0, px - neck_r);
        const int32_t xhi = std::min<int32_t>(nx - 1, px + neck_r);
        for (int32_t iy = ylo; iy <= yhi; ++iy) {
            const int32_t row = iy * nx;
            for (int32_t ix = xlo; ix <= xhi; ++ix) {
                bitset_set(nd.neck_bits, row + ix);
            }
        }
    }

    // Active MIM clearance entries for this net
    static const std::vector<std::pair<int64_t, int64_t>> kEmptyPinXy;
    std::vector<std::vector<MimClearRect>> mim_active(static_cast<size_t>(n_layers));
    for (int32_t gi = 0; gi < n_layers; ++gi) {
        const auto& entries = mim_clear_by_gi[static_cast<size_t>(gi)];
        if (entries.empty()) continue;
        const auto& pxy = (static_cast<size_t>(gi) < pin_xy_by_gi.size())
                              ? pin_xy_by_gi[static_cast<size_t>(gi)]
                              : kEmptyPinXy;
        for (const auto& e : entries) {
            if (e.owner_id >= 0 && e.owner_id == net_id) {
                bool covered = false;
                for (const auto& pt : pxy) {
                    if (e.bx0_nm <= pt.first && pt.first <= e.bx1_nm &&
                        e.by0_nm <= pt.second && pt.second <= e.by1_nm) {
                        covered = true;
                        break;
                    }
                }
                if (covered) continue;
            }
            mim_active[static_cast<size_t>(gi)].push_back(e);
        }
    }

    // 1. Rasterize per-layer blocked_bits
    const double inv_p = 1.0 / static_cast<double>(pitch_nm);
    for (int32_t gi = 0; gi < n_layers; ++gi) {
        const int32_t layer_off = gi * nxy;
        const int64_t halo = static_cast<int64_t>(min_space_nm[static_cast<size_t>(gi)])
                             + static_cast<int64_t>(route_w_nm / 2);
        const int64_t pad_extra = add_unowned_via_pad_extra ? via_pad_extra_halo_nm(gi, route_w_nm) : 0;
        for (const auto& ob : obs_by_gi[static_cast<size_t>(gi)]) {
            if (ob.owner_id >= 0 && ob.owner_id == net_id) continue;
            const int64_t h = halo + (ob.owner_id == -1 ? pad_extra : 0);
            raster_rect_into_bits(nd.blocked_bits, layer_off, ob.x0_um, ob.y0_um, ob.x1_um, ob.y1_um, h);
        }
        // MIM clearance (strict L-inf: dx < clr && dy < clr)
        const auto& mrects = mim_active[static_cast<size_t>(gi)];
        if (!mrects.empty()) {
            const int64_t w_half = std::max<int64_t>(route_w_nm, min_width_nm[static_cast<size_t>(gi)]) / 2;
            for (const auto& mr : mrects) {
                const double lo_x = static_cast<double>(mr.bx0_nm) - static_cast<double>(mr.clr_nm + w_half);
                const double hi_x = static_cast<double>(mr.bx1_nm) + static_cast<double>(mr.clr_nm + w_half);
                const double lo_y = static_cast<double>(mr.by0_nm) - static_cast<double>(mr.clr_nm + w_half);
                const double hi_y = static_cast<double>(mr.by1_nm) + static_cast<double>(mr.clr_nm + w_half);
                const int32_t ix_lo = std::max<int32_t>(
                    0, static_cast<int32_t>(std::floor((lo_x - static_cast<double>(x0_nm)) * inv_p)));
                const int32_t ix_hi = std::min<int32_t>(
                    nx - 1, static_cast<int32_t>(std::ceil((hi_x - static_cast<double>(x0_nm)) * inv_p)));
                const int32_t iy_lo = std::max<int32_t>(
                    0, static_cast<int32_t>(std::floor((lo_y - static_cast<double>(y0_nm)) * inv_p)));
                const int32_t iy_hi = std::min<int32_t>(
                    ny - 1, static_cast<int32_t>(std::ceil((hi_y - static_cast<double>(y0_nm)) * inv_p)));
                for (int32_t iy = iy_lo; iy <= iy_hi; ++iy) {
                    const int64_t cy = y0_nm + static_cast<int64_t>(iy) * pitch_nm;
                    if (static_cast<double>(cy) < lo_y || static_cast<double>(cy) > hi_y) continue;
                    const int64_t dy = std::max<int64_t>({0, mr.by0_nm - (cy + w_half), (cy - w_half) - mr.by1_nm});
                    if (dy >= mr.clr_nm) continue;
                    const int32_t row = layer_off + iy * nx;
                    for (int32_t ix = ix_lo; ix <= ix_hi; ++ix) {
                        const int64_t cx = x0_nm + static_cast<int64_t>(ix) * pitch_nm;
                        if (static_cast<double>(cx) < lo_x || static_cast<double>(cx) > hi_x) continue;
                        const int64_t dx = std::max<int64_t>({0, mr.bx0_nm - (cx + w_half), (cx - w_half) - mr.bx1_nm});
                        if (dx < mr.clr_nm) {
                            bitset_set(nd.blocked_bits, row + ix);
                        }
                    }
                }
            }
        }
        if (has_envelope) {
            const int64_t edge_half = std::max<int64_t>(route_w_nm, min_width_nm[static_cast<size_t>(gi)]) / 2;
            mark_envelope_blocked_bits(nd.blocked_bits, layer_off, edge_half);
        }
    }

    // 2. Compute via_blocked_bits (exact match to _via_infeasible_cells)
    nd.via_blocked_bits.clear();
    const int32_t n_vias = std::max<int32_t>(0, n_layers - 1);
    const size_t nwords_via = (static_cast<size_t>(nvia_total) + 63U) >> 6;
    bool any_via_blk = false;
    auto ensure_via_bits = [&]() {
        if (!any_via_blk) {
            nd.via_blocked_bits.assign(nwords_via, 0ULL);
            any_via_blk = true;
        }
    };

    for (int32_t gl = 0; build_via_blocked && gl < n_vias; ++gl) {
        if (((allowed_mask >> gl) & 1U) == 0U || ((allowed_mask >> (gl + 1)) & 1U) == 0U) {
            continue;
        }
        const int32_t vsz = via_cut_size_nm[static_cast<size_t>(gl)];
        if (vsz <= 0) continue;
        const int64_t cut_h = std::max<int64_t>(1, (static_cast<int64_t>(vsz) + 1) / 2);
        const int32_t rb = via_enc_below_nm[static_cast<size_t>(gl)];
        const int32_t ra = via_enc_above_nm[static_cast<size_t>(gl)];
        const int32_t layer_off = gl * nxy;

        // 2a. Physical via cut obstacles (cap_plate_via / device_cut)
        const auto& vobs = via_obs_by_gl[static_cast<size_t>(gl)];
        if (!vobs.empty()) {
            const int64_t vsp = std::max<int64_t>(0, via_cut_space_nm[static_cast<size_t>(gl)]);
            const int64_t reach_nm = cut_h + vsp;
            for (const auto& vo : vobs) {
                const double lo_x = std::min(vo.x0_um, vo.x1_um) * kNmPerUm - static_cast<double>(reach_nm);
                const double hi_x = std::max(vo.x0_um, vo.x1_um) * kNmPerUm + static_cast<double>(reach_nm);
                const double lo_y = std::min(vo.y0_um, vo.y1_um) * kNmPerUm - static_cast<double>(reach_nm);
                const double hi_y = std::max(vo.y0_um, vo.y1_um) * kNmPerUm + static_cast<double>(reach_nm);
                const int32_t ix_lo = std::max<int32_t>(
                    0, static_cast<int32_t>(std::floor((lo_x - static_cast<double>(x0_nm)) * inv_p)));
                const int32_t ix_hi = std::min<int32_t>(
                    nx - 1, static_cast<int32_t>(std::ceil((hi_x - static_cast<double>(x0_nm)) * inv_p)));
                const int32_t iy_lo = std::max<int32_t>(
                    0, static_cast<int32_t>(std::floor((lo_y - static_cast<double>(y0_nm)) * inv_p)));
                const int32_t iy_hi = std::min<int32_t>(
                    ny - 1, static_cast<int32_t>(std::ceil((hi_y - static_cast<double>(y0_nm)) * inv_p)));
                for (int32_t iy = iy_lo; iy <= iy_hi; ++iy) {
                    const int64_t cy = y0_nm + static_cast<int64_t>(iy) * pitch_nm;
                    if (static_cast<double>(cy) < lo_y || static_cast<double>(cy) > hi_y) continue;
                    const int64_t dy = std::max<int64_t>({0, vo.by0_nm - (cy + cut_h), (cy - cut_h) - vo.by1_nm});
                    for (int32_t ix = ix_lo; ix <= ix_hi; ++ix) {
                        const int64_t cx = x0_nm + static_cast<int64_t>(ix) * pitch_nm;
                        if (static_cast<double>(cx) < lo_x || static_cast<double>(cx) > hi_x) continue;
                        if (!vo.is_cap_plate_via && vo.owner_id >= 0 && vo.owner_id == net_id &&
                            cx - cut_h == vo.bx0_nm && cy - cut_h == vo.by0_nm &&
                            cx + cut_h == vo.bx1_nm && cy + cut_h == vo.by1_nm) {
                            continue;
                        }
                        const int64_t dx = std::max<int64_t>({0, vo.bx0_nm - (cx + cut_h), (cx - cut_h) - vo.bx1_nm});
                        if ((dx == 0 && dy == 0) ||
                            std::hypot(static_cast<double>(dx), static_cast<double>(dy)) < static_cast<double>(vsp)) {
                            ensure_via_bits();
                            bitset_set(nd.via_blocked_bits, layer_off + iy * nx + ix);
                        }
                    }
                }
            }
        }

        // 2b. MIM forbidden cut touch
        for (const auto& mf : mim_forbidden_by_gl[static_cast<size_t>(gl)]) {
            const double lo_x = static_cast<double>(mf.bx0_nm) - static_cast<double>(cut_h);
            const double hi_x = static_cast<double>(mf.bx1_nm) + static_cast<double>(cut_h);
            const double lo_y = static_cast<double>(mf.by0_nm) - static_cast<double>(cut_h);
            const double hi_y = static_cast<double>(mf.by1_nm) + static_cast<double>(cut_h);
            const int32_t ix_lo = std::max<int32_t>(
                0, static_cast<int32_t>(std::floor((lo_x - static_cast<double>(x0_nm)) * inv_p)));
            const int32_t ix_hi = std::min<int32_t>(
                nx - 1, static_cast<int32_t>(std::ceil((hi_x - static_cast<double>(x0_nm)) * inv_p)));
            const int32_t iy_lo = std::max<int32_t>(
                0, static_cast<int32_t>(std::floor((lo_y - static_cast<double>(y0_nm)) * inv_p)));
            const int32_t iy_hi = std::min<int32_t>(
                ny - 1, static_cast<int32_t>(std::ceil((hi_y - static_cast<double>(y0_nm)) * inv_p)));
            for (int32_t iy = iy_lo; iy <= iy_hi; ++iy) {
                const int64_t cy = y0_nm + static_cast<int64_t>(iy) * pitch_nm;
                if (static_cast<double>(cy) < lo_y || static_cast<double>(cy) > hi_y) continue;
                if (cy + cut_h < mf.by0_nm || mf.by1_nm < cy - cut_h) continue;
                for (int32_t ix = ix_lo; ix <= ix_hi; ++ix) {
                    const int64_t cx = x0_nm + static_cast<int64_t>(ix) * pitch_nm;
                    if (static_cast<double>(cx) < lo_x || static_cast<double>(cx) > hi_x) continue;
                    if (!(cx + cut_h < mf.bx0_nm || mf.bx1_nm < cx - cut_h)) {
                        ensure_via_bits();
                        bitset_set(nd.via_blocked_bits, layer_off + iy * nx + ix);
                    }
                }
            }
        }

        // 2c. MIM bottom-plate routing clearance for via landing pads
        for (int side = 0; side < 2; ++side) {
            const int32_t gs = (side == 0) ? gl : (gl + 1);
            const int32_t enc = (side == 0) ? rb : ra;
            const auto& mrects = mim_active[static_cast<size_t>(gs)];
            if (mrects.empty()) continue;
            const int64_t pad_enc = cut_h + std::max<int64_t>(0, enc);
            const int64_t reach_extra = std::max<int64_t>(pad_enc, static_cast<int64_t>(own_width_floor / 2));
            const int32_t mw_gs = min_width_nm[static_cast<size_t>(gs)];
            for (const auto& mr : mrects) {
                const double lo_x = static_cast<double>(mr.bx0_nm) - static_cast<double>(mr.clr_nm + reach_extra);
                const double hi_x = static_cast<double>(mr.bx1_nm) + static_cast<double>(mr.clr_nm + reach_extra);
                const double lo_y = static_cast<double>(mr.by0_nm) - static_cast<double>(mr.clr_nm + reach_extra);
                const double hi_y = static_cast<double>(mr.by1_nm) + static_cast<double>(mr.clr_nm + reach_extra);
                const int32_t ix_lo = std::max<int32_t>(
                    0, static_cast<int32_t>(std::floor((lo_x - static_cast<double>(x0_nm)) * inv_p)));
                const int32_t ix_hi = std::min<int32_t>(
                    nx - 1, static_cast<int32_t>(std::ceil((hi_x - static_cast<double>(x0_nm)) * inv_p)));
                const int32_t iy_lo = std::max<int32_t>(
                    0, static_cast<int32_t>(std::floor((lo_y - static_cast<double>(y0_nm)) * inv_p)));
                const int32_t iy_hi = std::min<int32_t>(
                    ny - 1, static_cast<int32_t>(std::ceil((hi_y - static_cast<double>(y0_nm)) * inv_p)));
                for (int32_t iy = iy_lo; iy <= iy_hi; ++iy) {
                    const int64_t cy = y0_nm + static_cast<int64_t>(iy) * pitch_nm;
                    if (static_cast<double>(cy) < lo_y || static_cast<double>(cy) > hi_y) continue;
                    const int32_t row_xy = iy * nx;
                    for (int32_t ix = ix_lo; ix <= ix_hi; ++ix) {
                        const int64_t cx = x0_nm + static_cast<int64_t>(ix) * pitch_nm;
                        if (static_cast<double>(cx) < lo_x || static_cast<double>(cx) > hi_x) continue;
                        const int32_t xy = row_xy + ix;
                        if (any_via_blk && bitset_test(nd.via_blocked_bits, layer_off + xy)) continue;
                        const int32_t w_at = bitset_test(nd.neck_bits, xy) ? mw_gs : route_w_nm;
                        const int64_t pad_g = std::max<int64_t>(
                            pad_enc, static_cast<int64_t>(std::max(w_at, own_width_floor) / 2)
                        );
                        const int64_t dx = std::max<int64_t>({0, mr.bx0_nm - (cx + pad_g), (cx - pad_g) - mr.bx1_nm});
                        const int64_t dy = std::max<int64_t>({0, mr.by0_nm - (cy + pad_g), (cy - pad_g) - mr.by1_nm});
                        if (dx < mr.clr_nm && dy < mr.clr_nm) {
                            ensure_via_bits();
                            bitset_set(nd.via_blocked_bits, layer_off + xy);
                        }
                    }
                }
            }
        }

        // 2d. Envelope & exact device-pad-intrusion (strict open-interior intersection
        // with min_space-grown foreign device rect; algebraically 100% equivalent to
        // via_landing_feasible device-pad-intrusion).
        if (rb > 0 || ra > 0) {
            if (has_envelope) {
                const int64_t via_reach = std::max<int64_t>(
                    cut_h + std::max<int64_t>(0, rb),
                    cut_h + std::max<int64_t>(0, ra)
                );
                ensure_via_bits();
                mark_envelope_blocked_bits(nd.via_blocked_bits, layer_off, via_reach);
            }
            for (int side = 0; side < 2; ++side) {
                const int32_t gs = (side == 0) ? gl : (gl + 1);
                const int32_t enc = (side == 0) ? rb : ra;
                if (enc <= 0) continue;
                const int64_t pad_h = cut_h + static_cast<int64_t>(enc);
                const int64_t ms = static_cast<int64_t>(min_space_nm[static_cast<size_t>(gs)]);
                const int64_t reach = pad_h + ms;
                for (const auto& ob : obs_by_gi[static_cast<size_t>(gs)]) {
                    if (ob.owner_id >= 0 && ob.owner_id == net_id) continue;
                    // Strict inequalities: ob.bx0_nm - reach < cx < ob.bx1_nm + reach
                    // and ob.by0_nm - reach < cy < ob.by1_nm + reach
                    const int64_t lo_x = ob.bx0_nm - reach;
                    const int64_t hi_x = ob.bx1_nm + reach;
                    const int64_t lo_y = ob.by0_nm - reach;
                    const int64_t hi_y = ob.by1_nm + reach;
                    // Smallest ix with x0 + ix*pitch > lo_x  <=> ix >= floor_div(lo_x - x0, pitch) + 1
                    const int32_t ix_lo = static_cast<int32_t>(std::max<int64_t>(0, floor_div_i64(lo_x - x0_nm, pitch_nm) + 1));
                    // Largest ix with x0 + ix*pitch < hi_x   <=> ix <= ceil_div(hi_x - x0, pitch) - 1
                    const int32_t ix_hi = static_cast<int32_t>(std::min<int64_t>(nx - 1, ceil_div_i64(hi_x - x0_nm, pitch_nm) - 1));
                    const int32_t iy_lo = static_cast<int32_t>(std::max<int64_t>(0, floor_div_i64(lo_y - y0_nm, pitch_nm) + 1));
                    const int32_t iy_hi = static_cast<int32_t>(std::min<int64_t>(ny - 1, ceil_div_i64(hi_y - y0_nm, pitch_nm) - 1));
                    if (ix_lo > ix_hi || iy_lo > iy_hi) continue;
                    ensure_via_bits();
                    for (int32_t iy = iy_lo; iy <= iy_hi; ++iy) {
                        const int32_t row = layer_off + iy * nx;
                        for (int32_t ix = ix_lo; ix <= ix_hi; ++ix) {
                            bitset_set(nd.via_blocked_bits, row + ix);
                        }
                    }
                }
            }
        }
    }

    // 3. Precompute via-cost stencils & via-pad keepout offsets for trunk vs neck
    nd.stencil_trunk.assign(static_cast<size_t>(n_vias), {});
    nd.stencil_neck.assign(static_cast<size_t>(n_vias), {});
    nd.pk_soft_trunk.assign(static_cast<size_t>(n_vias * 2), {});
    nd.pk_soft_neck.assign(static_cast<size_t>(n_vias * 2), {});
    nd.pk_exact_trunk.assign(static_cast<size_t>(n_vias * 2), {});
    nd.pk_exact_neck.assign(static_cast<size_t>(n_vias * 2), {});

    for (int32_t gl = 0; gl < n_vias; ++gl) {
        const int32_t vsz = via_cut_size_nm[static_cast<size_t>(gl)];
        const int32_t rb = via_enc_below_nm[static_cast<size_t>(gl)];
        const int32_t ra = via_enc_above_nm[static_cast<size_t>(gl)];
        for (int side = 0; side < 2; ++side) {
            const int32_t g = (side == 0) ? gl : (gl + 1);
            const int32_t enc = (side == 0) ? rb : ra;
            if (enc <= 0) continue;
            const int64_t ph = static_cast<int64_t>(vsz / 2) + static_cast<int64_t>(enc);
            const int64_t ms = static_cast<int64_t>(min_space_nm[static_cast<size_t>(g)]);
            const int64_t wh_trunk = static_cast<int64_t>(route_w_nm / 2);
            const int64_t wh_neck = static_cast<int64_t>(min_width_nm[static_cast<size_t>(g)] / 2);
            const size_t slot = static_cast<size_t>(gl * 2 + side);
            nd.pk_soft_trunk[slot] = compute_via_keepout_offsets(pitch_nm, ph, wh_trunk, ms, false);
            nd.pk_soft_neck[slot] = compute_via_keepout_offsets(pitch_nm, ph, wh_neck, ms, false);
            nd.pk_exact_trunk[slot] = compute_via_keepout_offsets(pitch_nm, ph, wh_trunk, ms, true);
            nd.pk_exact_neck[slot] = compute_via_keepout_offsets(pitch_nm, ph, wh_neck, ms, true);
            for (const auto& o : nd.pk_exact_trunk[slot]) {
                nd.stencil_trunk[static_cast<size_t>(gl)].push_back(StencilOff{o.dx, o.dy, g});
            }
            for (const auto& o : nd.pk_exact_neck[slot]) {
                nd.stencil_neck[static_cast<size_t>(gl)].push_back(StencilOff{o.dx, o.dy, g});
            }
        }
    }
}

void PathFinderGridCore::set_net_trunk(int32_t net_id, int32_t trunk_gi) noexcept {
    if (net_id >= 0 && static_cast<size_t>(net_id) < nets.size()) {
        nets[static_cast<size_t>(net_id)].trunk_gi = trunk_gi;
    }
}

void PathFinderGridCore::set_net_blocked_cells(int32_t net_id, const std::vector<int32_t>& idxs) {
    if (net_id < 0) return;
    if (static_cast<size_t>(net_id) >= nets.size()) {
        nets.resize(static_cast<size_t>(net_id) + 1);
    }
    NetData& nd = nets[static_cast<size_t>(net_id)];
    const size_t nwords = (static_cast<size_t>(ntotal) + 63U) >> 6;
    nd.blocked_bits.assign(nwords, 0ULL);
    for (int32_t idx : idxs) {
        if (idx >= 0 && idx < ntotal) {
            bitset_set(nd.blocked_bits, idx);
        }
    }
}

void PathFinderGridCore::set_net_via_blocked_cells(int32_t net_id, const std::vector<int32_t>& vidxs) {
    if (net_id < 0) return;
    if (static_cast<size_t>(net_id) >= nets.size()) {
        nets.resize(static_cast<size_t>(net_id) + 1);
    }
    NetData& nd = nets[static_cast<size_t>(net_id)];
    nd.via_blocked_bits.clear();
    if (n_layers <= 1 || vidxs.empty()) return;
    const size_t nwords = (static_cast<size_t>(nvia_total) + 63U) >> 6;
    for (int32_t vidx : vidxs) {
        if (vidx >= 0 && vidx < nvia_total) {
            if (nd.via_blocked_bits.empty()) {
                nd.via_blocked_bits.assign(nwords, 0ULL);
            }
            bitset_set(nd.via_blocked_bits, vidx);
        }
    }
}

bool PathFinderGridCore::is_blocked_cell(
    int32_t net_id, int32_t ix, int32_t iy, int32_t gi, bool check_committed
) const noexcept {
    if (!in_bounds(ix, iy) || gi < 0 || gi >= n_layers) return false;
    const int32_t idx = cell_idx(ix, iy, gi);
    if (net_id >= 0 && static_cast<size_t>(net_id) < nets.size()) {
        if (bitset_test(nets[static_cast<size_t>(net_id)].blocked_bits, idx)) return true;
    }
    const int32_t ro = reserved_owner[static_cast<size_t>(idx)];
    if (ro >= 0 && ro != net_id) return true;
    if (check_committed) {
        const int32_t co = committed_owner[static_cast<size_t>(idx)];
        if (co >= 0 && co != net_id) return true;
    }
    return false;
}

bool PathFinderGridCore::is_committed_foreign(
    int32_t net_id, int32_t ix, int32_t iy, int32_t gi
) const noexcept {
    if (!in_bounds(ix, iy) || gi < 0 || gi >= n_layers) return false;
    const int32_t co = committed_owner[static_cast<size_t>(cell_idx(ix, iy, gi))];
    return (co >= 0 && co != net_id);
}

std::vector<std::vector<Box64>> PathFinderGridCore::foreign_committed_rects(int32_t net_id) const {
    std::vector<std::vector<Box64>> out(static_cast<size_t>(n_layers));
    const int64_t half_pitch = pitch_nm / 2;
    for (int32_t idx : touched_committed) {
        const int32_t own = committed_owner[static_cast<size_t>(idx)];
        if (own < 0 || own == net_id) continue;
        int32_t ix = 0, iy = 0, gi = 0;
        unpack_idx(idx, ix, iy, gi);
        const int64_t reach = half_pitch + static_cast<int64_t>(min_space_nm[static_cast<size_t>(gi)]);
        const int64_t cx = x0_nm + static_cast<int64_t>(ix) * pitch_nm;
        const int64_t cy = y0_nm + static_cast<int64_t>(iy) * pitch_nm;
        out[static_cast<size_t>(gi)].push_back(Box64{cx - reach, cy - reach, cx + reach, cy + reach});
    }
    return out;
}

std::vector<std::pair<int32_t, int32_t>> PathFinderGridCore::get_blocked_cells(
    int32_t net_id, int32_t gi
) const {
    std::vector<std::pair<int32_t, int32_t>> out;
    if (net_id < 0 || static_cast<size_t>(net_id) >= nets.size() || gi < 0 || gi >= n_layers) {
        return out;
    }
    const auto& bits = nets[static_cast<size_t>(net_id)].blocked_bits;
    if (bits.empty()) return out;
    const int32_t layer_off = gi * nxy;
    for (int32_t iy = 0; iy < ny; ++iy) {
        const int32_t row = layer_off + iy * nx;
        for (int32_t ix = 0; ix < nx; ++ix) {
            if (bitset_test(bits, row + ix)) {
                out.emplace_back(ix, iy);
            }
        }
    }
    return out;
}

std::vector<std::pair<int32_t, int32_t>> PathFinderGridCore::get_via_infeasible_cells(
    int32_t net_id, int32_t gl
) const {
    std::vector<std::pair<int32_t, int32_t>> out;
    if (net_id < 0 || static_cast<size_t>(net_id) >= nets.size() || gl < 0 || gl >= n_layers - 1) {
        return out;
    }
    const auto& bits = nets[static_cast<size_t>(net_id)].via_blocked_bits;
    if (bits.empty()) return out;
    const int32_t layer_off = gl * nxy;
    for (int32_t iy = 0; iy < ny; ++iy) {
        const int32_t row = layer_off + iy * nx;
        for (int32_t ix = 0; ix < nx; ++ix) {
            if (bitset_test(bits, row + ix)) {
                out.emplace_back(ix, iy);
            }
        }
    }
    return out;
}

bool PathFinderGridCore::pin_is_trapped(
    int32_t net_id, int32_t ix, int32_t iy, int32_t gi, uint32_t allowed_mask
) {
    if (net_id < 0 || static_cast<size_t>(net_id) >= nets.size()) return false;
    const NetData& nd = nets[static_cast<size_t>(net_id)];
    const int32_t start_idx = cell_idx(ix, iy, gi);
    const bool self_blk = bitset_test(nd.blocked_bits, start_idx);
    const bool via_up_blk = (gi + 1 < n_layers) && bitset_test(nd.via_blocked_bits, gi * nxy + iy * nx + ix);
    if (!self_blk && !via_up_blk) return false;

    constexpr size_t kMaxPocket = 64;
    const uint32_t bgen = next_gen(bfs_stamp, bfs_gen);
    std::vector<int32_t> q;
    q.reserve(128);
    size_t head = 0;
    bfs_stamp[static_cast<size_t>(start_idx)] = bgen;
    q.push_back(start_idx);
    size_t seen_count = 1;

    static constexpr int32_t kDx[4] = {1, -1, 0, 0};
    static constexpr int32_t kDy[4] = {0, 0, 1, -1};
    while (head < q.size()) {
        const int32_t cur = q[head++];
        int32_t cx = 0, cy = 0, cg = 0;
        unpack_idx(cur, cx, cy, cg);
        if (cg != gi) return false;
        if (seen_count > kMaxPocket) return false;
        if ((cx == 0 || cy == 0 || cx == nx - 1 || cy == ny - 1) && !bitset_test(nd.blocked_bits, cur)) {
            return false;
        }
        for (int d = 0; d < 4; ++d) {
            const int32_t nix = cx + kDx[d];
            const int32_t niy = cy + kDy[d];
            if (!in_bounds(nix, niy)) continue;
            const int32_t nidx = cg * nxy + niy * nx + nix;
            if (!bitset_test(nd.blocked_bits, nidx) && bfs_stamp[static_cast<size_t>(nidx)] != bgen) {
                bfs_stamp[static_cast<size_t>(nidx)] = bgen;
                q.push_back(nidx);
                ++seen_count;
            }
        }
        for (int dg = 0; dg < 2; ++dg) {
            const int32_t gl = (dg == 0) ? cg : (cg - 1);
            const int32_t ng = (dg == 0) ? (cg + 1) : (cg - 1);
            if (ng < 0 || ng >= n_layers || ((allowed_mask >> ng) & 1U) == 0U) continue;
            if (bitset_test(nd.via_blocked_bits, gl * nxy + cy * nx + cx)) continue;
            const int32_t nidx = ng * nxy + cy * nx + cx;
            if (bitset_test(nd.blocked_bits, nidx)) continue;
            if (bfs_stamp[static_cast<size_t>(nidx)] != bgen) {
                bfs_stamp[static_cast<size_t>(nidx)] = bgen;
                q.push_back(nidx);
                ++seen_count;
            }
        }
    }
    return true;
}

void PathFinderGridCore::reset_history() noexcept {
    std::fill(history.begin(), history.end(), 0.0);
    std::fill(via_pad_history.begin(), via_pad_history.end(), 0.0);
}

void PathFinderGridCore::reset_committed() noexcept {
    for (int32_t idx : touched_committed) {
        committed_owner[static_cast<size_t>(idx)] = -1;
    }
    touched_committed.clear();
}

void PathFinderGridCore::begin_round(const std::vector<int32_t>& active_net_ids) {
    for (int32_t idx : touched_occ) occ[static_cast<size_t>(idx)] = 0;
    touched_occ.clear();
    for (int32_t idx : touched_pad_keepout) pad_keepout_count[static_cast<size_t>(idx)] = 0;
    touched_pad_keepout.clear();
    for (int32_t idx : touched_reserved) reserved_owner[static_cast<size_t>(idx)] = -1;
    touched_reserved.clear();
    for (int32_t idx : touched_unrouted_pins) unrouted_pin_count[static_cast<size_t>(idx)] = 0;
    touched_unrouted_pins.clear();
    for (int32_t idx : touched_metal_owner) metal_owner[static_cast<size_t>(idx)] = -1;
    touched_metal_owner.clear();
    round_exact_pk.clear();

    for (int32_t nid : active_net_ids) {
        if (nid < 0 || static_cast<size_t>(nid) >= nets.size()) continue;
        for (int32_t pidx : nets[static_cast<size_t>(nid)].unique_pin_cells) {
            if (unrouted_pin_count[static_cast<size_t>(pidx)] == 0) {
                touched_unrouted_pins.push_back(pidx);
            }
            unrouted_pin_count[static_cast<size_t>(pidx)] += 1;
        }
    }
}

void PathFinderGridCore::mark_net_starting(int32_t net_id) noexcept {
    if (net_id < 0 || static_cast<size_t>(net_id) >= nets.size()) return;
    for (int32_t pidx : nets[static_cast<size_t>(net_id)].unique_pin_cells) {
        if (unrouted_pin_count[static_cast<size_t>(pidx)] > 0) {
            unrouted_pin_count[static_cast<size_t>(pidx)] -= 1;
        }
    }
}

void PathFinderGridCore::commit_net_route(
    int32_t net_id,
    const std::vector<int32_t>& wire_cells,
    const std::vector<int32_t>& extra_seed_cells,
    bool ok
) {
    if (ok) {
        const uint32_t wgen = next_gen(wire_stamp, wire_gen);
        for (int32_t idx : wire_cells) {
            if (idx < 0 || idx >= ntotal) continue;
            if (wire_stamp[static_cast<size_t>(idx)] == wgen) continue;
            wire_stamp[static_cast<size_t>(idx)] = wgen;
            if (occ[static_cast<size_t>(idx)] == 0) {
                touched_occ.push_back(idx);
            }
            occ[static_cast<size_t>(idx)] += 1;
            if (metal_owner[static_cast<size_t>(idx)] == -1) {
                metal_owner[static_cast<size_t>(idx)] = net_id;
                touched_metal_owner.push_back(idx);
            }
        }
        for (int32_t idx : extra_seed_cells) {
            if (idx < 0 || idx >= ntotal) continue;
            if (metal_owner[static_cast<size_t>(idx)] == -1) {
                metal_owner[static_cast<size_t>(idx)] = net_id;
                touched_metal_owner.push_back(idx);
            }
        }
        // Emergent vias: cell idx on gl where idx + nxy is also in wire_cells
        if (net_id >= 0 && static_cast<size_t>(net_id) < nets.size() && n_layers > 1) {
            const NetData& nd = nets[static_cast<size_t>(net_id)];
            const uint32_t dgen = next_gen(dedup_stamp, dedup_gen);
            for (int32_t idx : wire_cells) {
                if (idx < 0 || idx >= nvia_total) continue;
                // Check if (ix, iy, gl+1) is also in wire_cells
                if (wire_stamp[static_cast<size_t>(idx + nxy)] != wgen) continue;
                int32_t ix = 0, iy = 0, gl = 0;
                unpack_idx(idx, ix, iy, gl);
                const int32_t xy = iy * nx + ix;
                const bool in_neck = bitset_test(nd.neck_bits, xy);
                const int32_t vcell_idx = idx;  // gl * nxy + xy
                for (int side = 0; side < 2; ++side) {
                    const int32_t g = (side == 0) ? gl : (gl + 1);
                    const size_t slot = static_cast<size_t>(gl * 2 + side);
                    const auto& soft_offs = in_neck ? nd.pk_soft_neck[slot] : nd.pk_soft_trunk[slot];
                    const auto& exact_offs = in_neck ? nd.pk_exact_neck[slot] : nd.pk_exact_trunk[slot];
                    const int32_t goff = g * nxy;
                    for (const auto& o : soft_offs) {
                        const int32_t jx = ix + o.dx;
                        const int32_t jy = iy + o.dy;
                        if (!in_bounds(jx, jy)) continue;
                        const int32_t kidx = goff + jy * nx + jx;
                        if (dedup_stamp[static_cast<size_t>(kidx)] != dgen) {
                            dedup_stamp[static_cast<size_t>(kidx)] = dgen;
                            if (pad_keepout_count[static_cast<size_t>(kidx)] == 0) {
                                touched_pad_keepout.push_back(kidx);
                            }
                            pad_keepout_count[static_cast<size_t>(kidx)] += 1;
                        }
                    }
                    for (const auto& o : exact_offs) {
                        const int32_t jx = ix + o.dx;
                        const int32_t jy = iy + o.dy;
                        if (!in_bounds(jx, jy)) continue;
                        const int32_t kidx = goff + jy * nx + jx;
                        round_exact_pk.push_back(ExactPadKeepoutEntry{kidx, net_id, vcell_idx});
                    }
                }
            }
        }
    } else {
        for (int32_t idx : extra_seed_cells) {
            if (idx < 0 || idx >= ntotal) continue;
            if (metal_owner[static_cast<size_t>(idx)] == -1) {
                metal_owner[static_cast<size_t>(idx)] = net_id;
                touched_metal_owner.push_back(idx);
            }
        }
    }
}

bool PathFinderGridCore::try_reserve_mirror_cell(int32_t net_b_id, int32_t idx) {
    if (idx < 0 || idx >= ntotal) return false;
    if (reserved_owner[static_cast<size_t>(idx)] == -1 && occ[static_cast<size_t>(idx)] == 0) {
        reserved_owner[static_cast<size_t>(idx)] = net_b_id;
        touched_reserved.push_back(idx);
        return true;
    }
    return false;
}

FinishRoundResult PathFinderGridCore::finish_round(double hist_fac) {
    FinishRoundResult out;

    // 1. Overused cells (occ >= 2)
    for (int32_t idx : touched_occ) {
        if (occ[static_cast<size_t>(idx)] >= 2) {
            out.overused.push_back(idx);
        }
    }

    // 2. Pad-contended cells from round_exact_pk
    // Sort by (cell_idx, owner_net_id, vcell_idx) and deduplicate
    auto& epk = round_exact_pk;
    std::sort(epk.begin(), epk.end(), [](const ExactPadKeepoutEntry& a, const ExactPadKeepoutEntry& b) {
        if (a.cell_idx != b.cell_idx) return a.cell_idx < b.cell_idx;
        if (a.owner_net_id != b.owner_net_id) return a.owner_net_id < b.owner_net_id;
        return a.vcell_idx < b.vcell_idx;
    });
    epk.erase(std::unique(epk.begin(), epk.end(), [](const ExactPadKeepoutEntry& a, const ExactPadKeepoutEntry& b) {
        return a.cell_idx == b.cell_idx && a.owner_net_id == b.owner_net_id && a.vcell_idx == b.vcell_idx;
    }), epk.end());

    for (size_t i = 0; i < epk.size(); ) {
        const int32_t cidx = epk[i].cell_idx;
        size_t j = i;
        while (j < epk.size() && epk[j].cell_idx == cidx) ++j;
        const int32_t mo = metal_owner[static_cast<size_t>(cidx)];
        if (mo >= 0) {
            PadContendedRow row;
            row.cell_idx = cidx;
            row.metal_owner_id = mo;
            int32_t last_own = -2;
            for (size_t k = i; k < j; ++k) {
                const int32_t own = epk[k].owner_net_id;
                if (own != mo) {
                    if (own != last_own) {
                        row.foreign_owners.push_back(own);
                        last_own = own;
                    }
                    row.foreign_vcells.push_back(epk[k].vcell_idx);
                }
            }
            if (!row.foreign_owners.empty()) {
                out.pad_contended.push_back(std::move(row));
            }
        }
        i = j;
    }

    // 3. Bump history ONLY if unconverged
    out.converged = out.overused.empty() && out.pad_contended.empty();
    if (!out.converged) {
        for (int32_t idx : out.overused) {
            history[static_cast<size_t>(idx)] += hist_fac;
        }
        for (const auto& row : out.pad_contended) {
            history[static_cast<size_t>(row.cell_idx)] += hist_fac;
            for (int32_t vcell : row.foreign_vcells) {
                via_pad_history[static_cast<size_t>(vcell)] += hist_fac;
            }
        }
    }

    // Sort results deterministically by (ix, iy, gi)
    auto cell_coord_less = [this](int32_t a, int32_t b) {
        int32_t ax = 0, ay = 0, ag = 0, bx = 0, by = 0, bg = 0;
        unpack_idx(a, ax, ay, ag);
        unpack_idx(b, bx, by, bg);
        if (ax != bx) return ax < bx;
        if (ay != by) return ay < by;
        return ag < bg;
    };
    std::sort(out.overused.begin(), out.overused.end(), cell_coord_less);
    std::sort(out.pad_contended.begin(), out.pad_contended.end(),
              [&](const PadContendedRow& a, const PadContendedRow& b) {
                  return cell_coord_less(a.cell_idx, b.cell_idx);
              });

    return out;
}

void PathFinderGridCore::commit_maze_cells(
    int32_t net_id, const std::vector<int32_t>& idxs, bool overwrite
) {
    for (int32_t idx : idxs) {
        if (idx < 0 || idx >= ntotal) continue;
        if (overwrite) {
            if (committed_owner[static_cast<size_t>(idx)] == -1) {
                touched_committed.push_back(idx);
            }
            committed_owner[static_cast<size_t>(idx)] = net_id;
        } else if (committed_owner[static_cast<size_t>(idx)] == -1) {
            committed_owner[static_cast<size_t>(idx)] = net_id;
            touched_committed.push_back(idx);
        }
    }
}

void PathFinderGridCore::commit_maze_halo(
    int32_t net_id,
    const std::vector<int32_t>& wires,
    int32_t wire_w_nm,
    int32_t extra_sp_nm,
    int32_t max_other_w_nm,
    bool use_taper,
    int32_t via_partner_half_nm,
    bool overwrite_wires,
    bool include_via_halo
) {
    const NetData* nd = (net_id >= 0 && static_cast<size_t>(net_id) < nets.size())
                            ? &nets[static_cast<size_t>(net_id)]
                            : nullptr;
    const uint32_t wgen = next_gen(wire_stamp, wire_gen);
    for (int32_t idx : wires) {
        if (idx < 0 || idx >= ntotal) continue;
        wire_stamp[static_cast<size_t>(idx)] = wgen;
        if (overwrite_wires) {
            if (committed_owner[static_cast<size_t>(idx)] == -1) {
                touched_committed.push_back(idx);
            }
            committed_owner[static_cast<size_t>(idx)] = net_id;
        } else if (committed_owner[static_cast<size_t>(idx)] == -1) {
            committed_owner[static_cast<size_t>(idx)] = net_id;
            touched_committed.push_back(idx);
        }
    }
    for (int32_t idx : wires) {
        if (idx < 0 || idx >= ntotal) continue;
        int32_t ix = 0, iy = 0, gi = 0;
        unpack_idx(idx, ix, iy, gi);
        const int32_t xy = iy * nx + ix;
        const int32_t w = (use_taper && nd && bitset_test(nd->neck_bits, xy))
                              ? min_width_nm[static_cast<size_t>(gi)]
                              : wire_w_nm;
        const int64_t rad_nm = static_cast<int64_t>(w / 2) + min_space_nm[static_cast<size_t>(gi)]
                               + extra_sp_nm + (max_other_w_nm / 2);
        const int32_t r = static_cast<int32_t>(std::max<int64_t>(
            0, static_cast<int64_t>(std::ceil(static_cast<double>(rad_nm) / static_cast<double>(pitch_nm)))
        ));
        if (r > 0) {
            const int32_t goff = gi * nxy;
            const int32_t xlo = std::max<int32_t>(0, ix - r);
            const int32_t xhi = std::min<int32_t>(nx - 1, ix + r);
            const int32_t ylo = std::max<int32_t>(0, iy - r);
            const int32_t yhi = std::min<int32_t>(ny - 1, iy + r);
            for (int32_t jy = ylo; jy <= yhi; ++jy) {
                const int32_t row = goff + jy * nx;
                for (int32_t jx = xlo; jx <= xhi; ++jx) {
                    const int32_t kidx = row + jx;
                    if (committed_owner[static_cast<size_t>(kidx)] == -1) {
                        committed_owner[static_cast<size_t>(kidx)] = net_id;
                        touched_committed.push_back(kidx);
                    }
                }
            }
        }
    }
    // Via-pad halo
    if (include_via_halo && n_layers > 1) {
        for (int32_t idx : wires) {
            if (idx < 0 || idx >= nvia_total) continue;
            if (wire_stamp[static_cast<size_t>(idx + nxy)] != wgen) continue;
            int32_t ix = 0, iy = 0, gl = 0;
            unpack_idx(idx, ix, iy, gl);
            const int32_t vsz = via_cut_size_nm[static_cast<size_t>(gl)];
            const int32_t rb = via_enc_below_nm[static_cast<size_t>(gl)];
            const int32_t ra = via_enc_above_nm[static_cast<size_t>(gl)];
            for (int side = 0; side < 2; ++side) {
                const int32_t g = (side == 0) ? gl : (gl + 1);
                const int32_t enc = (side == 0) ? rb : ra;
                if (enc <= 0) continue;
                const int64_t pad_half = static_cast<int64_t>(vsz / 2) + enc;
                const int64_t rad_nm = pad_half + min_space_nm[static_cast<size_t>(g)]
                                       + extra_sp_nm + via_partner_half_nm;
                const int32_t r = static_cast<int32_t>(std::max<int64_t>(
                    0, static_cast<int64_t>(std::ceil(static_cast<double>(rad_nm) / static_cast<double>(pitch_nm)))
                ));
                const int32_t goff = g * nxy;
                const int32_t xlo = std::max<int32_t>(0, ix - r);
                const int32_t xhi = std::min<int32_t>(nx - 1, ix + r);
                const int32_t ylo = std::max<int32_t>(0, iy - r);
                const int32_t yhi = std::min<int32_t>(ny - 1, iy + r);
                for (int32_t jy = ylo; jy <= yhi; ++jy) {
                    const int32_t row = goff + jy * nx;
                    for (int32_t jx = xlo; jx <= xhi; ++jx) {
                        const int32_t kidx = row + jx;
                        if (committed_owner[static_cast<size_t>(kidx)] == -1) {
                            committed_owner[static_cast<size_t>(kidx)] = net_id;
                            touched_committed.push_back(kidx);
                        }
                    }
                }
            }
        }
    }
}

void PathFinderGridCore::clear_occupancy() noexcept {
    for (int32_t idx : touched_occ) occ[static_cast<size_t>(idx)] = 0;
    touched_occ.clear();
}

void PathFinderGridCore::set_occupancy_cell(int32_t idx, int32_t cnt) {
    if (idx >= 0 && idx < ntotal && cnt != 0) {
        if (occ[static_cast<size_t>(idx)] == 0) {
            touched_occ.push_back(idx);
        }
        occ[static_cast<size_t>(idx)] = cnt;
    }
}

void PathFinderGridCore::clear_history_values() noexcept {
    std::fill(history.begin(), history.end(), 0.0);
}

void PathFinderGridCore::clear_via_pad_history_values() noexcept {
    std::fill(via_pad_history.begin(), via_pad_history.end(), 0.0);
}

void PathFinderGridCore::clear_pad_keepout_counts() noexcept {
    for (int32_t idx : touched_pad_keepout) pad_keepout_count[static_cast<size_t>(idx)] = 0;
    touched_pad_keepout.clear();
}

void PathFinderGridCore::set_pad_keepout_count_cell(int32_t idx, int32_t cnt) {
    if (idx >= 0 && idx < ntotal && cnt != 0) {
        if (pad_keepout_count[static_cast<size_t>(idx)] == 0) {
            touched_pad_keepout.push_back(idx);
        }
        pad_keepout_count[static_cast<size_t>(idx)] = cnt;
    }
}

std::vector<int32_t> PathFinderGridCore::net_footprint(
    int32_t net_id,
    const std::vector<int32_t>& wires,
    int32_t route_w_nm,
    bool use_neck
) {
    std::vector<int32_t> out;
    const NetData* nd = (net_id >= 0 && static_cast<size_t>(net_id) < nets.size())
                            ? &nets[static_cast<size_t>(net_id)]
                            : nullptr;
    const uint32_t wgen = next_gen(wire_stamp, wire_gen);
    for (int32_t idx : wires) {
        if (idx >= 0 && idx < ntotal) {
            wire_stamp[static_cast<size_t>(idx)] = wgen;
        }
    }

    // Match PathFinderEngine._net_footprint: via_low_at[(ix, iy)] = last gi with (gi+1) present
    std::vector<int32_t> via_low_at(static_cast<size_t>(nxy), -1);
    for (int32_t idx : wires) {
        if (idx >= 0 && idx < nvia_total && wire_stamp[static_cast<size_t>(idx + nxy)] == wgen) {
            int32_t ix = 0, iy = 0, gi = 0;
            unpack_idx(idx, ix, iy, gi);
            via_low_at[static_cast<size_t>(iy * nx + ix)] = gi;
        }
    }

    const uint32_t dgen = next_gen(dedup_stamp, dedup_gen);
    for (int32_t idx : wires) {
        if (idx < 0 || idx >= ntotal) continue;
        int32_t ix = 0, iy = 0, gi = 0;
        unpack_idx(idx, ix, iy, gi);
        const int32_t xy = iy * nx + ix;
        const int32_t w = (use_neck && nd && bitset_test(nd->neck_bits, xy))
                              ? min_width_nm[static_cast<size_t>(gi)]
                              : route_w_nm;
        int64_t pad_h = 0;
        const int32_t gl = via_low_at[static_cast<size_t>(xy)];
        if (gl >= 0) {
            int32_t enc = 0;
            if (gi == gl) enc = via_enc_below_nm[static_cast<size_t>(gl)];
            else if (gi == gl + 1) enc = via_enc_above_nm[static_cast<size_t>(gl)];
            if (enc > 0) {
                pad_h = static_cast<int64_t>(via_cut_size_nm[static_cast<size_t>(gl)] / 2) + enc;
            }
        }
        const int64_t half = std::max<int64_t>(w / 2, pad_h);
        const int64_t ms = min_space_nm[static_cast<size_t>(gi)];
        const int32_t r = static_cast<int32_t>(std::max<int64_t>(
            0, static_cast<int64_t>(std::ceil(
                   (static_cast<double>(half) + static_cast<double>(ms) * 0.5) / static_cast<double>(pitch_nm)))
        ));
        const int32_t goff = gi * nxy;
        const int32_t xlo = std::max<int32_t>(0, ix - r);
        const int32_t xhi = std::min<int32_t>(nx - 1, ix + r);
        const int32_t ylo = std::max<int32_t>(0, iy - r);
        const int32_t yhi = std::min<int32_t>(ny - 1, iy + r);
        for (int32_t jy = ylo; jy <= yhi; ++jy) {
            const int32_t row = goff + jy * nx;
            for (int32_t jx = xlo; jx <= xhi; ++jx) {
                const int32_t kidx = row + jx;
                if (dedup_stamp[static_cast<size_t>(kidx)] != dgen) {
                    dedup_stamp[static_cast<size_t>(kidx)] = dgen;
                    out.push_back(kidx);
                }
            }
        }
    }
    return out;
}

double PathFinderGridCore::compute_via_cost_fast(
    const NetData& nd,
    int32_t ix, int32_t iy, int32_t gl,
    double pres_fac, double hist_fac
) const noexcept {
    const int32_t xy = iy * nx + ix;
    double cost = 0.0;
    const double vh = via_pad_history[static_cast<size_t>(gl * nxy + xy)];
    if (vh > 0.0) {
        cost += vh * hist_fac * (1.0 + pres_fac);
    }
    const auto& stencil = bitset_test(nd.neck_bits, xy)
                              ? nd.stencil_neck[static_cast<size_t>(gl)]
                              : nd.stencil_trunk[static_cast<size_t>(gl)];
    for (const auto& s : stencil) {
        const int32_t jx = ix + s.dx;
        const int32_t jy = iy + s.dy;
        if (!in_bounds(jx, jy)) continue;
        const int32_t cidx = s.g * nxy + jy * nx + jx;
        int32_t f_occ = occ[static_cast<size_t>(cidx)];
        if (f_occ == 0 && unrouted_pin_count[static_cast<size_t>(cidx)] > 0) {
            f_occ = 1;
        }
        if (f_occ > 0) {
            const double base = 1.0 + history[static_cast<size_t>(cidx)] * hist_fac;
            cost += base * (static_cast<double>(f_occ) * pres_fac);
        }
    }
    return cost;
}

double PathFinderGridCore::compute_cell_cost_fast(
    int32_t idx, int32_t xy, int32_t gi,
    double pres_fac, double hist_fac,
    int32_t trunk_gi,
    bool has_self_pk, uint32_t spk_gen,
    bool has_mirror, uint32_t mgen, double mirror_pen,
    bool use_via_cost, const NetData* nd,
    int32_t ix, int32_t iy,
    uint32_t wgen
) const noexcept {
    const int32_t c_occ = occ[static_cast<size_t>(idx)];
    int32_t pad_occ = pad_keepout_count[static_cast<size_t>(idx)];
    if (has_self_pk && pad_occ > 0 && self_pk_stamp[static_cast<size_t>(idx)] == spk_gen) {
        pad_occ -= 1;
    }
    const double base = 1.0 + history[static_cast<size_t>(idx)] * hist_fac;
    const double trunk_pen = (trunk_gi > gi) ? (kTrunkBelowPenalty * static_cast<double>(trunk_gi - gi)) : 0.0;
    const double over_pen = (over_active_penalty > 0.0 && gi >= over_active_gi_floor && bitset_test(over_active_bits, xy))
                                ? over_active_penalty
                                : 0.0;
    double val = base * (1.0 + static_cast<double>(c_occ + pad_occ) * pres_fac) - 1.0 + trunk_pen + over_pen;
    if (has_mirror && mirror_stamp[static_cast<size_t>(idx)] != mgen) {
        val += mirror_pen;
    }
    if (use_via_cost && nd != nullptr && wire_stamp[static_cast<size_t>(idx)] != wgen) {
        double extra = 0.0;
        if (gi > 0 && wire_stamp[static_cast<size_t>(idx - nxy)] == wgen) {
            const double vc = compute_via_cost_fast(*nd, ix, iy, gi - 1, pres_fac, hist_fac);
            if (vc > 0.0) extra += vc;
        }
        if (gi + 1 < n_layers && wire_stamp[static_cast<size_t>(idx + nxy)] == wgen) {
            const double vc = compute_via_cost_fast(*nd, ix, iy, gi, pres_fac, hist_fac);
            if (vc > 0.0) extra += vc;
        }
        val += extra;
    }
    return val;
}

int PathFinderGridCore::astar_undirected(
    int32_t net_id,
    bool check_committed,
    const std::vector<int32_t>& sources,
    int32_t target_idx,
    uint32_t allowed_mask,
    bool use_pf_cost,
    double pres_fac,
    double hist_fac,
    int32_t trunk_gi,
    bool use_via_cost,
    bool has_self_pk,
    uint32_t spk_gen,
    bool has_mirror,
    uint32_t mgen,
    double mirror_pen,
    uint32_t wgen,
    bool has_window,
    int32_t win_x0, int32_t win_y0, int32_t win_x1, int32_t win_y1,
    int64_t max_expansions,
    CheckpointHook checkpoint,
    int64_t checkpoint_interval,
    std::vector<int32_t>& out_path
) {
    out_path.clear();
    const NetData* nd = (net_id >= 0 && static_cast<size_t>(net_id) < nets.size())
                            ? &nets[static_cast<size_t>(net_id)]
                            : nullptr;
    const uint64_t* blk_ptr = (nd && !nd->blocked_bits.empty()) ? nd->blocked_bits.data() : nullptr;
    const uint64_t* vblk_ptr = (nd && !nd->via_blocked_bits.empty()) ? nd->via_blocked_bits.data() : nullptr;
    const int32_t* res_ptr = reserved_owner.data();
    const int32_t* com_ptr = check_committed ? committed_owner.data() : nullptr;

    int32_t tgt_ix = 0, tgt_iy = 0, tgt_gi = 0;
    unpack_idx(target_idx, tgt_ix, tgt_iy, tgt_gi);

    const uint32_t gen_open = next_astar_gen();
    const uint32_t gen_closed = gen_open + 1;
    heap.clear();

    for (int32_t sidx : sources) {
        int32_t sx = 0, sy = 0, sgi = 0;
        unpack_idx(sidx, sx, sy, sgi);
        if (((allowed_mask >> sgi) & 1U) == 0U) continue;
        astar_stamp[static_cast<size_t>(sidx)] = gen_open;
        astar_g[static_cast<size_t>(sidx)] = 0.0;
        astar_parent[static_cast<size_t>(sidx)] = -1;
        const double h = static_cast<double>(std::abs(sx - tgt_ix) + std::abs(sy - tgt_iy))
                         + static_cast<double>(std::abs(sgi - tgt_gi)) * kViaPenalty;
        heap.push(HeapNode{h, -0.0, 0.0, sx, sy, sgi, sidx});
    }
    if (heap.empty()) return 0;

    int64_t expansions = 0;
    const uint64_t chk_mask = (checkpoint_interval > 0 && (checkpoint_interval & (checkpoint_interval - 1)) == 0)
                                  ? static_cast<uint64_t>(checkpoint_interval - 1)
                                  : 0ULL;

    static constexpr int32_t kDx[4] = {1, -1, 0, 0};
    static constexpr int32_t kDy[4] = {0, 0, 1, -1};

    while (!heap.empty()) {
        const HeapNode cur = heap.pop();
        const int32_t cidx = cur.idx;
        if (astar_stamp[static_cast<size_t>(cidx)] == gen_closed) continue;
        astar_stamp[static_cast<size_t>(cidx)] = gen_closed;

        if (cidx == target_idx) {
            int32_t p = cidx;
            while (p != -1) {
                out_path.push_back(p);
                p = astar_parent[static_cast<size_t>(p)];
            }
            std::reverse(out_path.begin(), out_path.end());
            return 1;
        }

        ++expansions;
        if (expansions > max_expansions) return 0;
        if (checkpoint.enabled()) {
            const bool fire = (chk_mask != 0ULL)
                                  ? ((static_cast<uint64_t>(expansions) & chk_mask) == 0ULL)
                                  : (checkpoint_interval <= 1 || (expansions % checkpoint_interval) == 0);
            if (fire && !checkpoint.call_now()) {
                return -1;
            }
        }

        const int32_t cix = cur.ix;
        const int32_t ciy = cur.iy;
        const int32_t cgi = cur.gi;
        const double cg = cur.g;
        const bool hpref = (pref_horiz[static_cast<size_t>(cgi)] != 0);
        const double step_h = hpref ? 1.0 : (1.0 + kWrongDirPenalty);
        const double step_v = hpref ? (1.0 + kWrongDirPenalty) : 1.0;
        const int32_t layer_off = cgi * nxy;

        // 4 in-plane neighbors
        for (int d = 0; d < 4; ++d) {
            const int32_t nix = cix + kDx[d];
            const int32_t niy = ciy + kDy[d];
            if (nix < 0 || nix >= nx || niy < 0 || niy >= ny) continue;
            if (has_window && (nix < win_x0 || nix > win_x1 || niy < win_y0 || niy > win_y1)) continue;
            const int32_t nxy_i = niy * nx + nix;
            const int32_t nidx = layer_off + nxy_i;
            const uint32_t nst = astar_stamp[static_cast<size_t>(nidx)];
            if (nst == gen_closed) continue;

            const bool is_target = (nidx == target_idx);
            if (!is_target) {
                if (blk_ptr != nullptr &&
                    ((blk_ptr[static_cast<size_t>(nidx) >> 6] >> (static_cast<uint32_t>(nidx) & 63U)) & 1ULL)) {
                    continue;
                }
                const int32_t ro = res_ptr[static_cast<size_t>(nidx)];
                if (ro >= 0 && ro != net_id) continue;
                if (com_ptr != nullptr) {
                    const int32_t co = com_ptr[static_cast<size_t>(nidx)];
                    if (co >= 0 && co != net_id) continue;
                }
            }

            double tentative = cg + ((d < 2) ? step_h : step_v);
            if (!is_target) {
                if (use_pf_cost) {
                    tentative += compute_cell_cost_fast(
                        nidx, nxy_i, cgi, pres_fac, hist_fac, trunk_gi,
                        has_self_pk, spk_gen, has_mirror, mgen, mirror_pen,
                        use_via_cost, nd, nix, niy, wgen
                    );
                } else if (has_mirror && mirror_stamp[static_cast<size_t>(nidx)] != mgen) {
                    tentative += mirror_pen;
                }
            }

            if (nst != gen_open || tentative < astar_g[static_cast<size_t>(nidx)]) {
                astar_stamp[static_cast<size_t>(nidx)] = gen_open;
                astar_g[static_cast<size_t>(nidx)] = tentative;
                astar_parent[static_cast<size_t>(nidx)] = cidx;
                const double h = static_cast<double>(std::abs(nix - tgt_ix) + std::abs(niy - tgt_iy))
                                 + static_cast<double>(std::abs(cgi - tgt_gi)) * kViaPenalty;
                heap.push(HeapNode{tentative + h, -tentative, tentative, nix, niy, cgi, nidx});
            }
        }

        // Up/down via neighbors (gi - 1, gi + 1)
        const int32_t cxy = ciy * nx + cix;
        for (int dg = 0; dg < 2; ++dg) {
            const int32_t ngi = (dg == 0) ? (cgi - 1) : (cgi + 1);
            if (ngi < 0 || ngi >= n_layers || ((allowed_mask >> ngi) & 1U) == 0U) continue;
            const int32_t gl = (dg == 0) ? ngi : cgi;
            if (vblk_ptr != nullptr) {
                const int32_t vidx = gl * nxy + cxy;
                if (((vblk_ptr[static_cast<size_t>(vidx) >> 6] >> (static_cast<uint32_t>(vidx) & 63U)) & 1ULL)) {
                    continue;
                }
            }
            const int32_t nidx = ngi * nxy + cxy;
            const uint32_t nst = astar_stamp[static_cast<size_t>(nidx)];
            if (nst == gen_closed) continue;
            if (has_window && (cix < win_x0 || cix > win_x1 || ciy < win_y0 || ciy > win_y1)) continue;

            const bool is_target = (nidx == target_idx);
            if (!is_target) {
                if (blk_ptr != nullptr &&
                    ((blk_ptr[static_cast<size_t>(nidx) >> 6] >> (static_cast<uint32_t>(nidx) & 63U)) & 1ULL)) {
                    continue;
                }
                const int32_t ro = res_ptr[static_cast<size_t>(nidx)];
                if (ro >= 0 && ro != net_id) continue;
                if (com_ptr != nullptr) {
                    const int32_t co = com_ptr[static_cast<size_t>(nidx)];
                    if (co >= 0 && co != net_id) continue;
                }
            }

            double v_cost = kViaPenalty;
            if (use_via_cost && nd != nullptr) {
                const double vc = compute_via_cost_fast(*nd, cix, ciy, gl, pres_fac, hist_fac);
                if (vc > 0.0) v_cost += vc;
            }
            double tentative = cg + v_cost;
            if (!is_target) {
                if (use_pf_cost) {
                    tentative += compute_cell_cost_fast(
                        nidx, cxy, ngi, pres_fac, hist_fac, trunk_gi,
                        has_self_pk, spk_gen, has_mirror, mgen, mirror_pen,
                        use_via_cost, nd, cix, ciy, wgen
                    );
                } else if (has_mirror && mirror_stamp[static_cast<size_t>(nidx)] != mgen) {
                    tentative += mirror_pen;
                }
            }

            if (nst != gen_open || tentative < astar_g[static_cast<size_t>(nidx)]) {
                astar_stamp[static_cast<size_t>(nidx)] = gen_open;
                astar_g[static_cast<size_t>(nidx)] = tentative;
                astar_parent[static_cast<size_t>(nidx)] = cidx;
                const double h = static_cast<double>(std::abs(cix - tgt_ix) + std::abs(ciy - tgt_iy))
                                 + static_cast<double>(std::abs(ngi - tgt_gi)) * kViaPenalty;
                heap.push(HeapNode{tentative + h, -tentative, tentative, cix, ciy, ngi, nidx});
            }
        }
    }
    return 0;
}

int PathFinderGridCore::astar_directed(
    int32_t net_id,
    bool check_committed,
    const std::vector<int32_t>& sources,
    int32_t target_idx,
    uint32_t allowed_mask,
    const std::vector<DirPrefEntry>& dir_pref,
    bool has_dir_runs,
    uint32_t dr_gen,
    bool use_pf_cost,
    double pres_fac,
    double hist_fac,
    int32_t trunk_gi,
    bool use_via_cost,
    bool has_self_pk,
    uint32_t spk_gen,
    bool has_mirror,
    uint32_t mgen,
    double mirror_pen,
    uint32_t wgen,
    bool has_window,
    int32_t win_x0, int32_t win_y0, int32_t win_x1, int32_t win_y1,
    int64_t max_expansions,
    CheckpointHook checkpoint,
    int64_t checkpoint_interval,
    std::vector<int32_t>& out_path
) {
    out_path.clear();
    const NetData* nd = (net_id >= 0 && static_cast<size_t>(net_id) < nets.size())
                            ? &nets[static_cast<size_t>(net_id)]
                            : nullptr;
    const uint64_t* blk_ptr = (nd && !nd->blocked_bits.empty()) ? nd->blocked_bits.data() : nullptr;
    const uint64_t* vblk_ptr = (nd && !nd->via_blocked_bits.empty()) ? nd->via_blocked_bits.data() : nullptr;
    const int32_t* res_ptr = reserved_owner.data();
    const int32_t* com_ptr = check_committed ? committed_owner.data() : nullptr;

    int32_t tgt_ix = 0, tgt_iy = 0, tgt_gi = 0;
    unpack_idx(target_idx, tgt_ix, tgt_iy, tgt_gi);

    const uint32_t gen_open = next_dir_gen();
    const uint32_t gen_closed = gen_open + 1;
    dir_heap.clear();

    for (int32_t sidx : sources) {
        int32_t sx = 0, sy = 0, sgi = 0;
        unpack_idx(sidx, sx, sy, sgi);
        if (((allowed_mask >> sgi) & 1U) == 0U) continue;
        int32_t r0 = 0;
        if (has_dir_runs && static_cast<size_t>(sgi) < dir_pref.size() && dir_pref[static_cast<size_t>(sgi)].active) {
            if (dir_runs_stamp[static_cast<size_t>(sidx)] == dr_gen) {
                r0 = std::min<int32_t>(dir_runs_arr[static_cast<size_t>(sidx)], kDirRunCap);
                if (r0 < 0) r0 = 0;
            }
        }
        const int32_t st_idx = sidx * kDirRunStates + r0;
        dir_stamp[static_cast<size_t>(st_idx)] = gen_open;
        dir_g[static_cast<size_t>(st_idx)] = 0.0;
        dir_parent[static_cast<size_t>(st_idx)] = -1;
        const double h = static_cast<double>(std::abs(sx - tgt_ix) + std::abs(sy - tgt_iy))
                         + static_cast<double>(std::abs(sgi - tgt_gi)) * kViaPenalty;
        dir_heap.push(DirHeapNode{h, -0.0, 0.0, sx, sy, sgi, r0, st_idx});
    }
    if (dir_heap.empty()) return 0;

    int64_t expansions = 0;
    const uint64_t chk_mask = (checkpoint_interval > 0 && (checkpoint_interval & (checkpoint_interval - 1)) == 0)
                                  ? static_cast<uint64_t>(checkpoint_interval - 1)
                                  : 0ULL;

    static constexpr int32_t kDx[4] = {1, -1, 0, 0};
    static constexpr int32_t kDy[4] = {0, 0, 1, -1};

    while (!dir_heap.empty()) {
        const DirHeapNode cur = dir_heap.pop();
        const int32_t cst = cur.state_idx;
        if (dir_stamp[static_cast<size_t>(cst)] == gen_closed) continue;
        dir_stamp[static_cast<size_t>(cst)] = gen_closed;

        const int32_t cidx = cst / kDirRunStates;
        if (cidx == target_idx) {
            int32_t p = cst;
            while (p != -1) {
                out_path.push_back(p / kDirRunStates);
                p = dir_parent[static_cast<size_t>(p)];
            }
            std::reverse(out_path.begin(), out_path.end());
            return 1;
        }

        ++expansions;
        if (expansions > max_expansions) return 0;
        if (checkpoint.enabled()) {
            const bool fire = (chk_mask != 0ULL)
                                  ? ((static_cast<uint64_t>(expansions) & chk_mask) == 0ULL)
                                  : (checkpoint_interval <= 1 || (expansions % checkpoint_interval) == 0);
            if (fire && !checkpoint.call_now()) {
                return -1;
            }
        }

        const int32_t cix = cur.ix;
        const int32_t ciy = cur.iy;
        const int32_t cgi = cur.gi;
        const int32_t crun = cur.run;
        const double cg = cur.g;
        const bool hpref = (pref_horiz[static_cast<size_t>(cgi)] != 0);
        const DirPrefEntry dpe = (static_cast<size_t>(cgi) < dir_pref.size())
                                     ? dir_pref[static_cast<size_t>(cgi)]
                                     : DirPrefEntry{};
        const int32_t layer_off = cgi * nxy;

        // 4 in-plane moves
        for (int d = 0; d < 4; ++d) {
            const int32_t nix = cix + kDx[d];
            const int32_t niy = ciy + kDy[d];
            if (nix < 0 || nix >= nx || niy < 0 || niy >= ny) continue;
            if (has_window && (nix < win_x0 || nix > win_x1 || niy < win_y0 || niy > win_y1)) continue;
            const bool is_horiz = (d < 2);
            int32_t nrun = 0;
            double penalty = 0.0;
            if (dpe.active) {
                if (is_horiz != dpe.horiz) {
                    nrun = std::min<int32_t>(crun + 1, kDirRunCap);
                    penalty = dpe.weight * 1.0;
                    if (nrun > kDirRunFree) {
                        penalty += dpe.weight * kDirRunPenalty;
                    }
                } else {
                    nrun = crun;
                    penalty = 0.0;
                }
            } else {
                nrun = 0;
                penalty = (is_horiz == hpref) ? 0.0 : kWrongDirPenalty;
            }

            const int32_t nxy_i = niy * nx + nix;
            const int32_t nidx = layer_off + nxy_i;
            const int32_t nst_idx = nidx * kDirRunStates + nrun;
            const uint32_t nst = dir_stamp[static_cast<size_t>(nst_idx)];
            if (nst == gen_closed) continue;

            const bool is_target = (nidx == target_idx);
            if (!is_target) {
                if (blk_ptr != nullptr &&
                    ((blk_ptr[static_cast<size_t>(nidx) >> 6] >> (static_cast<uint32_t>(nidx) & 63U)) & 1ULL)) {
                    continue;
                }
                const int32_t ro = res_ptr[static_cast<size_t>(nidx)];
                if (ro >= 0 && ro != net_id) continue;
                if (com_ptr != nullptr) {
                    const int32_t co = com_ptr[static_cast<size_t>(nidx)];
                    if (co >= 0 && co != net_id) continue;
                }
            }

            double tentative = cg + (1.0 + penalty);
            if (!is_target) {
                if (use_pf_cost) {
                    tentative += compute_cell_cost_fast(
                        nidx, nxy_i, cgi, pres_fac, hist_fac, trunk_gi,
                        has_self_pk, spk_gen, has_mirror, mgen, mirror_pen,
                        use_via_cost, nd, nix, niy, wgen
                    );
                } else if (has_mirror && mirror_stamp[static_cast<size_t>(nidx)] != mgen) {
                    tentative += mirror_pen;
                }
            }

            if (nst != gen_open || tentative < dir_g[static_cast<size_t>(nst_idx)]) {
                dir_stamp[static_cast<size_t>(nst_idx)] = gen_open;
                dir_g[static_cast<size_t>(nst_idx)] = tentative;
                dir_parent[static_cast<size_t>(nst_idx)] = cst;
                const double h = static_cast<double>(std::abs(nix - tgt_ix) + std::abs(niy - tgt_iy))
                                 + static_cast<double>(std::abs(cgi - tgt_gi)) * kViaPenalty;
                dir_heap.push(DirHeapNode{tentative + h, -tentative, tentative, nix, niy, cgi, nrun, nst_idx});
            }
        }

        // Via moves (run resets to 0)
        const int32_t cxy = ciy * nx + cix;
        for (int dg = 0; dg < 2; ++dg) {
            const int32_t ngi = (dg == 0) ? (cgi - 1) : (cgi + 1);
            if (ngi < 0 || ngi >= n_layers || ((allowed_mask >> ngi) & 1U) == 0U) continue;
            const int32_t gl = (dg == 0) ? ngi : cgi;
            if (vblk_ptr != nullptr) {
                const int32_t vidx = gl * nxy + cxy;
                if (((vblk_ptr[static_cast<size_t>(vidx) >> 6] >> (static_cast<uint32_t>(vidx) & 63U)) & 1ULL)) {
                    continue;
                }
            }
            const int32_t nidx = ngi * nxy + cxy;
            const int32_t nst_idx = nidx * kDirRunStates;  // run == 0
            const uint32_t nst = dir_stamp[static_cast<size_t>(nst_idx)];
            if (nst == gen_closed) continue;

            const bool is_target = (nidx == target_idx);
            if (!is_target) {
                if (blk_ptr != nullptr &&
                    ((blk_ptr[static_cast<size_t>(nidx) >> 6] >> (static_cast<uint32_t>(nidx) & 63U)) & 1ULL)) {
                    continue;
                }
                const int32_t ro = res_ptr[static_cast<size_t>(nidx)];
                if (ro >= 0 && ro != net_id) continue;
                if (com_ptr != nullptr) {
                    const int32_t co = com_ptr[static_cast<size_t>(nidx)];
                    if (co >= 0 && co != net_id) continue;
                }
            }

            double v_cost = kViaPenalty;
            if (use_via_cost && nd != nullptr) {
                const double vc = compute_via_cost_fast(*nd, cix, ciy, gl, pres_fac, hist_fac);
                if (vc > 0.0) v_cost += vc;
            }
            double tentative = cg + v_cost;
            if (!is_target) {
                if (use_pf_cost) {
                    tentative += compute_cell_cost_fast(
                        nidx, cxy, ngi, pres_fac, hist_fac, trunk_gi,
                        has_self_pk, spk_gen, has_mirror, mgen, mirror_pen,
                        use_via_cost, nd, cix, ciy, wgen
                    );
                } else if (has_mirror && mirror_stamp[static_cast<size_t>(nidx)] != mgen) {
                    tentative += mirror_pen;
                }
            }

            if (nst != gen_open || tentative < dir_g[static_cast<size_t>(nst_idx)]) {
                dir_stamp[static_cast<size_t>(nst_idx)] = gen_open;
                dir_g[static_cast<size_t>(nst_idx)] = tentative;
                dir_parent[static_cast<size_t>(nst_idx)] = cst;
                const double h = static_cast<double>(std::abs(cix - tgt_ix) + std::abs(ciy - tgt_iy))
                                 + static_cast<double>(std::abs(ngi - tgt_gi)) * kViaPenalty;
                dir_heap.push(DirHeapNode{tentative + h, -tentative, tentative, cix, ciy, ngi, 0, nst_idx});
            }
        }
    }
    return 0;
}

void PathFinderGridCore::update_dir_runs(
    uint32_t tgen,
    uint32_t dr_gen,
    const std::vector<int32_t>& new_cells,
    const std::vector<DirPrefEntry>& dir_pref
) {
    if (dir_runs_arr.empty()) {
        next_dir_gen();
    }
    const uint32_t bgen = next_gen(bfs_stamp, bfs_gen);
    std::vector<int32_t> comp;
    std::vector<int32_t> stack;
    static constexpr int32_t kDx[4] = {1, -1, 0, 0};
    static constexpr int32_t kDy[4] = {0, 0, 1, -1};

    for (int32_t cidx : new_cells) {
        int32_t cx = 0, cy = 0, cg = 0;
        unpack_idx(cidx, cx, cy, cg);
        if (static_cast<size_t>(cg) >= dir_pref.size() || !dir_pref[static_cast<size_t>(cg)].active) continue;
        if (bfs_stamp[static_cast<size_t>(cidx)] == bgen) continue;

        comp.clear();
        stack.clear();
        bfs_stamp[static_cast<size_t>(cidx)] = bgen;
        comp.push_back(cidx);
        stack.push_back(cidx);
        int32_t min_x = cx, max_x = cx, min_y = cy, max_y = cy;
        const int32_t layer_off = cg * nxy;

        while (!stack.empty()) {
            const int32_t u = stack.back();
            stack.pop_back();
            int32_t ux = 0, uy = 0;
            unpack_xy(u, ux, uy);
            if (ux < min_x) min_x = ux;
            if (ux > max_x) max_x = ux;
            if (uy < min_y) min_y = uy;
            if (uy > max_y) max_y = uy;
            for (int d = 0; d < 4; ++d) {
                const int32_t vx = ux + kDx[d];
                const int32_t vy = uy + kDy[d];
                if (vx < 0 || vx >= nx || vy < 0 || vy >= ny) continue;
                const int32_t v = layer_off + vy * nx + vx;
                if (tree_stamp[static_cast<size_t>(v)] == tgen && bfs_stamp[static_cast<size_t>(v)] != bgen) {
                    bfs_stamp[static_cast<size_t>(v)] = bgen;
                    comp.push_back(v);
                    stack.push_back(v);
                }
            }
        }
        const int32_t ext = dir_pref[static_cast<size_t>(cg)].horiz ? (max_y - min_y) : (max_x - min_x);
        for (int32_t u : comp) {
            dir_runs_stamp[static_cast<size_t>(u)] = dr_gen;
            dir_runs_arr[static_cast<size_t>(u)] = ext;
        }
    }
}

int PathFinderGridCore::route_one_net(
    int32_t net_id,
    bool check_committed,
    bool use_pf_cost,
    const std::vector<int32_t>& pin_cells,
    uint32_t allowed_mask,
    const std::vector<int32_t>& seed_cells,
    double pres_fac,
    double hist_fac,
    int32_t trunk_gi,
    bool use_via_cost,
    bool has_dir_pref,
    const std::vector<DirPrefEntry>& dir_pref,
    bool has_mirror,
    const std::vector<int32_t>& mirror_cells,
    double mirror_pen,
    bool use_window_ladder,
    int64_t max_expansions,
    CheckpointHook checkpoint,
    int64_t checkpoint_interval,
    std::vector<int32_t>& out_wire_cells
) {
    out_wire_cells.clear();
    if (pin_cells.empty()) return 1;

    const uint32_t tgen = next_gen(tree_stamp, tree_gen);
    const uint32_t wgen = next_gen(wire_stamp, wire_gen);
    uint32_t mgen = 0;
    if (has_mirror) {
        mgen = next_gen(mirror_stamp, mirror_gen);
        for (int32_t mc : mirror_cells) {
            if (mc >= 0 && mc < ntotal) {
                mirror_stamp[static_cast<size_t>(mc)] = mgen;
            }
        }
    }

    std::vector<int32_t> tree_vec;
    std::vector<int32_t> targets;
    const bool has_seeds = !seed_cells.empty();
    if (has_seeds) {
        tree_vec.reserve(seed_cells.size() + 64);
        for (int32_t s : seed_cells) {
            if (s >= 0 && s < ntotal && tree_stamp[static_cast<size_t>(s)] != tgen) {
                tree_stamp[static_cast<size_t>(s)] = tgen;
                tree_vec.push_back(s);
            }
        }
        targets = pin_cells;
    } else {
        const int32_t s0 = pin_cells[0];
        tree_stamp[static_cast<size_t>(s0)] = tgen;
        tree_vec.push_back(s0);
        wire_stamp[static_cast<size_t>(s0)] = wgen;
        out_wire_cells.push_back(s0);
        targets.assign(pin_cells.begin() + 1, pin_cells.end());
    }

    uint32_t dr_gen = 0;
    if (has_dir_pref) {
        if (dir_runs_stamp.empty()) next_dir_gen();
        dr_gen = next_gen(dir_runs_stamp, dir_runs_gen);
        update_dir_runs(tgen, dr_gen, tree_vec, dir_pref);
    }

    std::vector<int32_t> path;
    path.reserve(256);

    for (int32_t target : targets) {
        if (tree_stamp[static_cast<size_t>(target)] == tgen) {
            if (has_seeds && wire_stamp[static_cast<size_t>(target)] != wgen) {
                wire_stamp[static_cast<size_t>(target)] = wgen;
                out_wire_cells.push_back(target);
            }
            continue;
        }

        if (checkpoint.enabled() && !checkpoint.call_now()) {
            return -1;
        }

        // Compute bbox of tree_vec + target for window ladder
        int32_t tx = 0, ty = 0;
        unpack_xy(target, tx, ty);
        int32_t min_x = tx, max_x = tx, min_y = ty, max_y = ty;
        for (int32_t s : tree_vec) {
            int32_t sx = 0, sy = 0;
            unpack_xy(s, sx, sy);
            if (sx < min_x) min_x = sx;
            else if (sx > max_x) max_x = sx;
            if (sy < min_y) min_y = sy;
            else if (sy > max_y) max_y = sy;
        }
        const int32_t span = std::max(max_x - min_x, max_y - min_y);
        const int32_t slack0 = std::max<int32_t>(8, span / 4);

        int rc = 0;
        if (use_window_ladder) {
            const int32_t slacks[4] = {slack0, slack0 * 2, slack0 * 4, -1};
            for (int k = 0; k < 4; ++k) {
                const bool hw = (slacks[k] >= 0);
                const int32_t sl = slacks[k];
                if (has_dir_pref) {
                    rc = astar_directed(
                        net_id, check_committed, tree_vec, target, allowed_mask,
                        dir_pref, true, dr_gen,
                        use_pf_cost, pres_fac, hist_fac, trunk_gi, use_via_cost,
                        false, 0, has_mirror, mgen, mirror_pen, wgen,
                        hw, min_x - sl, min_y - sl, max_x + sl, max_y + sl,
                        max_expansions, checkpoint, checkpoint_interval, path
                    );
                } else {
                    rc = astar_undirected(
                        net_id, check_committed, tree_vec, target, allowed_mask,
                        use_pf_cost, pres_fac, hist_fac, trunk_gi, use_via_cost,
                        false, 0, has_mirror, mgen, mirror_pen, wgen,
                        hw, min_x - sl, min_y - sl, max_x + sl, max_y + sl,
                        max_expansions, checkpoint, checkpoint_interval, path
                    );
                }
                if (rc != 0) break;  // 1 = found, -1 = aborted
            }
        } else {
            if (has_dir_pref) {
                rc = astar_directed(
                    net_id, check_committed, tree_vec, target, allowed_mask,
                    dir_pref, true, dr_gen,
                    use_pf_cost, pres_fac, hist_fac, trunk_gi, use_via_cost,
                    false, 0, has_mirror, mgen, mirror_pen, wgen,
                    false, 0, 0, 0, 0,
                    max_expansions, checkpoint, checkpoint_interval, path
                );
            } else {
                rc = astar_undirected(
                    net_id, check_committed, tree_vec, target, allowed_mask,
                    use_pf_cost, pres_fac, hist_fac, trunk_gi, use_via_cost,
                    false, 0, has_mirror, mgen, mirror_pen, wgen,
                    false, 0, 0, 0, 0,
                    max_expansions, checkpoint, checkpoint_interval, path
                );
            }
        }

        if (rc < 0) return -1;
        if (rc == 0) {
            out_wire_cells.clear();
            return 0;
        }
        for (int32_t c : path) {
            if (tree_stamp[static_cast<size_t>(c)] != tgen) {
                tree_stamp[static_cast<size_t>(c)] = tgen;
                tree_vec.push_back(c);
            }
            if (wire_stamp[static_cast<size_t>(c)] != wgen) {
                wire_stamp[static_cast<size_t>(c)] = wgen;
                out_wire_cells.push_back(c);
            }
        }
        if (has_dir_pref) {
            update_dir_runs(tgen, dr_gen, path, dir_pref);
        }
    }
    return 1;
}

// ===================== RectClipperCore =====================

void subtract_rects_rec(
    int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
    const std::vector<Box64>& holes,
    size_t start_idx,
    std::vector<Box64>& out
) {
    if (rx0 >= rx1 || ry0 >= ry1) return;
    const size_t n = holes.size();
    for (size_t i = start_idx; i < n; ++i) {
        const Box64& h = holes[i];
        const int64_t ix0 = (rx0 > h.x0) ? rx0 : h.x0;
        const int64_t iy0 = (ry0 > h.y0) ? ry0 : h.y0;
        const int64_t ix1 = (rx1 < h.x1) ? rx1 : h.x1;
        const int64_t iy1 = (ry1 < h.y1) ? ry1 : h.y1;
        if (ix0 >= ix1 || iy0 >= iy1) continue;
        const size_t next_idx = i + 1;
        if (rx0 < ix0) subtract_rects_rec(rx0, ry0, ix0, ry1, holes, next_idx, out);
        if (ix1 < rx1) subtract_rects_rec(ix1, ry0, rx1, ry1, holes, next_idx, out);
        if (ry0 < iy0) subtract_rects_rec(ix0, ry0, ix1, iy0, holes, next_idx, out);
        if (iy1 < ry1) subtract_rects_rec(ix0, iy1, ix1, ry1, holes, next_idx, out);
        return;
    }
    out.push_back(Box64{rx0, ry0, rx1, ry1});
}

void RectClipperCore::clear() noexcept {
    layers.clear();
}

void RectClipperCore::add_layer_holes(int64_t layer, int64_t datatype, std::vector<Box64> holes) {
    if (holes.empty()) return;
    LayerHoles lh;
    lh.layer = layer;
    lh.datatype = datatype;
    lh.min_x = std::numeric_limits<int64_t>::max();
    lh.min_y = std::numeric_limits<int64_t>::max();
    lh.max_x = std::numeric_limits<int64_t>::min();
    lh.max_y = std::numeric_limits<int64_t>::min();
    for (const auto& h : holes) {
        if (h.x0 < lh.min_x) lh.min_x = h.x0;
        if (h.y0 < lh.min_y) lh.min_y = h.y0;
        if (h.x1 > lh.max_x) lh.max_x = h.x1;
        if (h.y1 > lh.max_y) lh.max_y = h.y1;
    }
    lh.holes = std::move(holes);
    layers.push_back(std::move(lh));
}

const LayerHoles* RectClipperCore::find_layer(int64_t layer, int64_t datatype) const noexcept {
    for (const auto& lh : layers) {
        if (lh.layer == layer && lh.datatype == datatype) return &lh;
    }
    return nullptr;
}

bool RectClipperCore::run_fragments(
    int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
    const LayerHoles& lh, int64_t floor_w,
    std::vector<Box64>& keep_out
) {
    keep_out.clear();
    if (rx0 >= rx1 || ry0 >= ry1) return false;
    if (rx1 <= lh.min_x || rx0 >= lh.max_x || ry1 <= lh.min_y || ry0 >= lh.max_y) {
        const int64_t sw = std::min(rx1 - rx0, ry1 - ry0);
        if (sw < floor_w) return true;
        keep_out.push_back(Box64{rx0, ry0, rx1, ry1});
        return false;
    }
    scratch_raw.clear();
    subtract_rects_rec(rx0, ry0, rx1, ry1, lh.holes, 0, scratch_raw);
    bool dropped = false;
    for (const auto& r : scratch_raw) {
        const int64_t sw = std::min(r.x1 - r.x0, r.y1 - r.y0);
        if (sw < floor_w) {
            dropped = true;
            continue;
        }
        keep_out.push_back(r);
    }
    return dropped;
}

const std::vector<Box64>& RectClipperCore::clip_emit(
    int64_t layer, int64_t datatype,
    int64_t x0, int64_t y0, int64_t x1, int64_t y1,
    int64_t min_w
) {
    const LayerHoles* lh = find_layer(layer, datatype);
    if (lh == nullptr || lh->holes.empty()) {
        scratch_keep.clear();
        scratch_keep.push_back(Box64{x0, y0, x1, y1});
        return scratch_keep;
    }
    const int64_t full_w = std::min<int64_t>(x1 - x0, y1 - y0);
    const int64_t floor_w = std::min<int64_t>(min_w, std::max<int64_t>(1, full_w));
    const bool dropped = run_fragments(x0, y0, x1, y1, *lh, floor_w, scratch_keep);
    if (dropped && full_w > min_w) {
        Box64 spines[3];
        if ((x1 - x0) >= (y1 - y0)) {
            const int64_t mid = floor_div_i64(y0 + y1, 2);
            const int64_t cy0 = mid - floor_div_i64(min_w, 2);
            spines[0] = Box64{x0, cy0, x1, cy0 + min_w};
            spines[1] = Box64{x0, y0, x1, y0 + min_w};
            spines[2] = Box64{x0, y1 - min_w, x1, y1};
        } else {
            const int64_t mid = floor_div_i64(x0 + x1, 2);
            const int64_t cx0 = mid - floor_div_i64(min_w, 2);
            spines[0] = Box64{cx0, y0, cx0 + min_w, y1};
            spines[1] = Box64{x0, y0, x0 + min_w, y1};
            spines[2] = Box64{x1 - min_w, y0, x1, y1};
        }
        const bool d2 = run_fragments(spines[0].x0, spines[0].y0, spines[0].x1, spines[0].y1, *lh, floor_w, scratch_keep2);
        if (d2) {
            for (int s = 1; s < 3; ++s) {
                const bool alt_drop = run_fragments(
                    spines[s].x0, spines[s].y0, spines[s].x1, spines[s].y1, *lh, floor_w, scratch_alt);
                if (!scratch_alt.empty() && !alt_drop) {
                    scratch_keep2.insert(scratch_keep2.end(), scratch_alt.begin(), scratch_alt.end());
                    break;
                }
            }
        }
        scratch_keep.insert(scratch_keep.end(), scratch_keep2.begin(), scratch_keep2.end());
    }
    return scratch_keep;
}

const std::vector<Box64>& RectClipperCore::subtract_rects(
    int64_t layer, int64_t datatype,
    int64_t x0, int64_t y0, int64_t x1, int64_t y1
) {
    scratch_raw.clear();
    if (x0 >= x1 || y0 >= y1) {
        return scratch_raw;
    }
    const LayerHoles* lh = find_layer(layer, datatype);
    if (lh == nullptr || lh->holes.empty() ||
        x1 <= lh->min_x || x0 >= lh->max_x || y1 <= lh->min_y || y0 >= lh->max_y) {
        scratch_raw.push_back(Box64{x0, y0, x1, y1});
        return scratch_raw;
    }
    subtract_rects_rec(x0, y0, x1, y1, lh->holes, 0, scratch_raw);
    return scratch_raw;
}

}  // namespace clay_pnr::route
