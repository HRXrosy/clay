// Pure C++20 Gridless EdgeContext legality/congestion, Hard Connectivity Preflight,
// Cartesian A*, and Axis Price Cache implementation.
// Strictly free of <Python.h>.

#include "clay_pnr/route/gridless_search.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clay_pnr::route {

// ---------------------------------------------------------------------------
// 1. PureEdgeContext legality, congestion, mirror & static pricing
// ---------------------------------------------------------------------------

bool PureEdgeContext::reject_active_mirror_pair_box(
    int32_t gi, const Box64& primary_box, const Box64& partner_box, int64_t partner_half) {
    const PureEdgeLayerData& L = layers[static_cast<size_t>(gi)];
    const int64_t required = std::max(L.space, mirror_intent_space);
    bool rejected = false;
    auto check_blk = [&](const Box64& blk) {
        if (!rects_clear_linf(partner_box, blk, required)) {
            record_rej(
                EdgeRejKey{REJ_MIRROR_PAIR_METAL, gi, partner_half, required, 1, 6, 0, 0, blk},
                partner_box, mirror_ctx->net_tag);
            rejected = true;
        }
    };
    check_blk(primary_box);
    for (const auto& blk : L.attempt_metal) {
        check_blk(blk);
    }
    if (rejected) {
        mirror_pair_conflict_seen = true;
    }
    return rejected;
}

int PureEdgeContext::check_wire_legal_single(
    int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t& out_present) {
    if (gi < 0 || gi >= n_layers) return -1;
    const PureEdgeLayerData& L = layers[static_cast<size_t>(gi)];
    const int64_t hw = L.hw;
    const int64_t x0 = (ax <= bx ? ax : bx) - hw;
    const int64_t x1 = (ax <= bx ? bx : ax) + hw;
    const int64_t y0 = (ay <= by ? ay : by) - hw;
    const int64_t y1 = (ay <= by ? by : ay) + hw;
    const Box64 wb{x0, y0, x1, y1};

    if (has_env && (wb.x0 < env_box.x0 || wb.y0 < env_box.y0 ||
                    wb.x1 > env_box.x1 || wb.y1 > env_box.y1)) {
        record_rej(EdgeRejKey{REJ_ROUTING_ENVELOPE, gi, hw, -1, 1, 0, 0, 0, env_box},
                   wb, net_tag);
        return 0;
    }
    if (!L.mim_bottom.empty()) {
        bool mim_hit = false;
        for (size_t i = 0; i < L.mim_bottom.size(); ++i) {
            const auto& m = L.mim_bottom[i];
            if (!rects_clear_linf(wb, m.blocker, m.clearance)) {
                record_rej(EdgeRejKey{REJ_MIM_BOTTOM_ROUTING, gi, hw, m.clearance, 1, 4,
                                      static_cast<uint32_t>(i), 0, m.blocker},
                           wb, m.owner_tag);
                mim_hit = true;
            }
        }
        if (mim_hit) return 0;
    }
    if (!L.mim_derived.empty()) {
        bool mim_hit = false;
        for (size_t i = 0; i < L.mim_derived.size(); ++i) {
            const auto& m = L.mim_derived[i];
            const bool clear = m.euclidean
                ? rects_clear_euclid_exact(wb, m.blocker, m.clearance)
                : rects_clear_linf(wb, m.blocker, m.clearance);
            if (!clear) {
                record_rej(EdgeRejKey{REJ_MIM_DERIVED_UNRELATED, gi, hw, m.clearance, 1, 4,
                                      static_cast<uint32_t>(i), m.kind_id, m.blocker},
                           wb, m.owner_tag, m.rule_tag);
                mim_hit = true;
            }
        }
        if (mim_hit) return 0;
    }
    if (!L.mim_carve.empty()) {
        bool mim_hit = false;
        for (size_t i = 0; i < L.mim_carve.size(); ++i) {
            const auto& m = L.mim_carve[i];
            if (closed_rects_touch(wb, m.blocker)) {
                record_rej(EdgeRejKey{REJ_MIM_CARVE_BOTTOM, gi, hw, 0, 1, 4,
                                      static_cast<uint32_t>(i), m.kind_id, m.blocker},
                           wb, m.owner_tag, m.kind_tag);
                mim_hit = true;
            }
        }
        if (mim_hit) return 0;
    }
    if (needs_wide_fallback_legal(gi, wb, hw)) {
        return -1;
    }
    const int64_t sp = L.space;
    if (L.obs_idx) {
        const Box64 q{wb.x0 - sp, wb.y0 - sp, wb.x1 + sp, wb.y1 + sp};
        bool obs_hit = false;
        L.obs_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id == L.net_obs_id) return;
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            if (!rects_clear_linf(wb, b, sp)) {
                record_rej(EdgeRejKey{REJ_FIXED_METAL, gi, hw, sp, 1, 1, item.owner_id, 0, b},
                           wb, L.obs_idx->entries[item.entry_idx].owner_tag);
                obs_hit = true;
            }
        });
        if (obs_hit) return 0;
    }
    int64_t present_total = 0;
    if (L.metal_idx) {
        const int64_t half = floor_div_pos(sp, 2);
        const Box64 ebx{wb.x0 - half, wb.y0 - half, wb.x1 + half, wb.y1 + half};
        const Box64 q{wb.x0 - sp, wb.y0 - sp, wb.x1 + sp, wb.y1 + sp};
        metal_contacts_buf.clear();
        metal_spacings_buf.clear();
        bool need_fallback = false;
        L.metal_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (need_fallback || item.owner_id == L.net_metal_id) return;
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            int64_t sx = 0, sy = 0;
            box_sep(wb, b, sx, sy);
            if (sx == 0 && sy == 0) {
                metal_contacts_buf.push_back(MetalHit{b, item.owner_id, L.metal_idx->entries[item.entry_idx].owner_tag});
            } else if (sp > 0 && sx < sp && sy < sp &&
                       euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                                        static_cast<uint64_t>(sp))) {
                const bool is_vp = !L.via_pad_set.empty() &&
                                   L.via_pad_set.find(OwnerBoxKey{item.owner_id, b}) != L.via_pad_set.end();
                const uint8_t pers = (item.owner_id < L.metal_owner_persistent.size())
                    ? L.metal_owner_persistent[item.owner_id] : 0;
                if (!is_vp && pers == 2) {
                    need_fallback = true;
                    return;
                }
                const bool owner_supra = (item.owner_id < L.metal_owner_supra_min.size()) &&
                                         (L.metal_owner_supra_min[item.owner_id] != 0);
                if (is_vp || pers == 1 || L.net_is_supra_min || owner_supra) {
                    metal_spacings_buf.push_back(MetalHit{b, item.owner_id, L.metal_idx->entries[item.entry_idx].owner_tag});
                }
            }
            const int64_t ox0 = std::max(ebx.x0, b.x0 - half);
            const int64_t oy0 = std::max(ebx.y0, b.y0 - half);
            const int64_t ox1 = std::min(ebx.x1, b.x1 + half);
            const int64_t oy1 = std::min(ebx.y1, b.y1 + half);
            if (ox0 >= ox1 || oy0 >= oy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
            present_total += pres * ncx * ncy;
        });
        if (need_fallback) return -1;
        if (!metal_spacings_buf.empty()) {
            for (const auto& h : metal_spacings_buf) {
                record_rej(EdgeRejKey{REJ_COMMITTED_METAL_SPACING, gi, hw, sp, 1, 2, h.owner_id, 0, h.blocker},
                           wb, h.owner_tag);
            }
            return 0;
        }
        if (!metal_contacts_buf.empty()) {
            for (const auto& h : metal_contacts_buf) {
                record_rej(EdgeRejKey{REJ_COMMITTED_METAL_CONTACT, gi, hw, 0, 1, 2, h.owner_id, 0, h.blocker},
                           wb, h.owner_tag);
            }
            return 0;
        }
    }
    out_present = present_total;
    return 1;
}

int PureEdgeContext::check_wire_legal(
    int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t& out_present) {
    const int st = check_wire_legal_single(gi, ax, ay, bx, by, out_present);
    if (st <= 0 || !mirror_ctx) return st;
    const int64_t m_ax = mirror_axis2 - ax;
    const int64_t m_bx = mirror_axis2 - bx;
    const int64_t p_hw = layers[static_cast<size_t>(gi)].hw;
    const int64_t b_hw = mirror_ctx->layers[static_cast<size_t>(gi)].hw;
    const Box64 primary_box{
        (ax <= bx ? ax : bx) - p_hw, (ay <= by ? ay : by) - p_hw,
        (ax <= bx ? bx : ax) + p_hw, (ay <= by ? by : ay) + p_hw
    };
    const Box64 partner_box{
        (m_ax <= m_bx ? m_ax : m_bx) - b_hw, (ay <= by ? ay : by) - b_hw,
        (m_ax <= m_bx ? m_bx : m_ax) + b_hw, (ay <= by ? by : ay) + b_hw
    };
    if (reject_active_mirror_pair_box(gi, primary_box, partner_box, b_hw)) {
        return 0;
    }
    int64_t partner_pres = 0;
    const int mst = mirror_ctx->check_wire_legal_single(gi, m_ax, ay, m_bx, by, partner_pres);
    return (mst <= 0) ? mst : 1;
}

int PureEdgeContext::check_via_legal_single(
    int64_t x, int64_t y, int32_t lo, int32_t n_flag,
    int64_t& out_pres_lo, int64_t& out_pres_hi, int64_t& out_pres_cut) {
    if (lo < 0 || lo >= n_layers - 1) return -1;
    const PureEdgeViaData& V = vias[static_cast<size_t>(lo)];
    const int64_t bh = (n_flag == 1) ? V.bh1 : V.bh0;
    const int64_t ah = (n_flag == 1) ? V.ah1 : V.ah0;
    const int64_t ext = (n_flag == 1) ? V.ext1 : V.ext0;
    const Box64 pad_lo{x - bh, y - bh, x + bh, y + bh};
    const Box64 pad_hi{x - ah, y - ah, x + ah, y + ah};
    const Box64 cut{x - ext, y - ext, x + ext, y + ext};

    if (needs_wide_fallback_legal(lo, pad_lo, bh) ||
        needs_wide_fallback_legal(lo + 1, pad_hi, ah)) {
        return -1;
    }

    const auto& forb = (n_flag == 1) ? V.forb_xy_1 : V.forb_xy_0;
    if (!forb.empty() && forb.find({x, y}) != forb.end()) {
        record_rej(EdgeRejKey{REJ_CONFLICT_BRANCH_VIA, lo, ext, -1, 0, 0, 0, 0, Box64{}},
                   cut, net_tag);
        return 0;
    }

    if ((x % grid) != 0 || (y % grid) != 0) {
        record_rej(EdgeRejKey{REJ_MANUFACTURING_GRID, lo, bh, grid, 0, 0, 0, 0, Box64{}},
                   pad_lo, net_tag);
        return 0;
    }

    site_rejs_buf.clear();
    bool static_legal = true;

    auto check_static_pad = [&](int32_t p_gi, const Box64& p_pad, int64_t p_half) {
        const PureEdgeLayerData& L = layers[static_cast<size_t>(p_gi)];
        if (has_env && (p_pad.x0 < env_box.x0 || p_pad.y0 < env_box.y0 ||
                        p_pad.x1 > env_box.x1 || p_pad.y1 > env_box.y1)) {
            site_rejs_buf.push_back(PureEdgeSiteRej{
                EdgeRejKey{REJ_ROUTING_ENVELOPE, p_gi, p_half, -1, 1, 0, 0, 0, env_box},
                p_pad, net_tag, 0
            });
            static_legal = false;
        }
        for (size_t i = 0; i < L.mim_bottom.size(); ++i) {
            const auto& m = L.mim_bottom[i];
            if (!rects_clear_linf(p_pad, m.blocker, m.clearance)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_MIM_BOTTOM_ROUTING, p_gi, p_half, m.clearance, 1, 4,
                               static_cast<uint32_t>(i), 0, m.blocker},
                    p_pad, m.owner_tag, 0
                });
                static_legal = false;
            }
        }
        for (size_t i = 0; i < L.mim_derived.size(); ++i) {
            const auto& m = L.mim_derived[i];
            const bool clear = m.euclidean
                ? rects_clear_euclid_exact(p_pad, m.blocker, m.clearance)
                : rects_clear_linf(p_pad, m.blocker, m.clearance);
            if (!clear) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_MIM_DERIVED_UNRELATED, p_gi, p_half, m.clearance, 1, 4,
                               static_cast<uint32_t>(i), m.kind_id, m.blocker},
                    p_pad, m.owner_tag, m.rule_tag
                });
                static_legal = false;
            }
        }
        for (size_t i = 0; i < L.mim_carve.size(); ++i) {
            const auto& m = L.mim_carve[i];
            if (closed_rects_touch(p_pad, m.blocker)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_MIM_CARVE_BOTTOM, p_gi, p_half, 0, 1, 4,
                               static_cast<uint32_t>(i), m.kind_id, m.blocker},
                    p_pad, m.owner_tag, m.kind_tag
                });
                static_legal = false;
            }
        }
        if (!L.obs_idx) return;
        const int64_t lsp = L.space;
        const Box64 q{p_pad.x0 - lsp, p_pad.y0 - lsp, p_pad.x1 + lsp, p_pad.y1 + lsp};
        sn_hits_buf.clear();
        L.obs_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            if (item.owner_id != L.net_obs_id) {
                if (!rects_clear_linf(p_pad, b, lsp)) {
                    site_rejs_buf.push_back(PureEdgeSiteRej{
                        EdgeRejKey{REJ_FIXED_METAL, p_gi, p_half, lsp, 1, 1, item.owner_id, 0, b},
                        p_pad, L.obs_idx->entries[item.entry_idx].owner_tag, 0
                    });
                    static_legal = false;
                }
            } else {
                int64_t sx = 0, sy = 0;
                if (euclid_in_open_band(p_pad, b, lsp, sx, sy)) {
                    if (!L.abstract_set.empty() && L.abstract_set.find(b) != L.abstract_set.end()) {
                        return;
                    }
                    sn_hits_buf.push_back(SnHit{b, item.owner_id, L.obs_idx->entries[item.entry_idx].owner_tag, sx, sy});
                }
            }
        });
        for (const auto& h : sn_hits_buf) {
            if (raw_gap_metallised_pure(p_pad, h.blocker, h.sx, h.sy,
                                        L.attempt_metal, L.obs_idx, L.abstract_set, L.metal_idx, cover_buf)) {
                continue;
            }
            site_rejs_buf.push_back(PureEdgeSiteRej{
                EdgeRejKey{REJ_FIXED_SAME_NET_VIA_PAD, p_gi, p_half, lsp, 1, 1, h.owner_id, 0, h.blocker},
                p_pad, h.owner_tag, 0
            });
            static_legal = false;
        }
    };

    check_static_pad(lo, pad_lo, bh);
    check_static_pad(lo + 1, pad_hi, ah);

    if (has_env && (cut.x0 < env_box.x0 || cut.y0 < env_box.y0 ||
                    cut.x1 > env_box.x1 || cut.y1 > env_box.y1)) {
        site_rejs_buf.push_back(PureEdgeSiteRej{
            EdgeRejKey{REJ_ROUTING_ENVELOPE, lo, ext, -1, 1, 0, 0, 0, env_box},
            cut, net_tag, 0
        });
        static_legal = false;
    }
    for (size_t i = 0; i < V.mim_forb_cut.size(); ++i) {
        const auto& m = V.mim_forb_cut[i];
        if (closed_rects_touch(cut, m.blocker)) {
            site_rejs_buf.push_back(PureEdgeSiteRej{
                EdgeRejKey{REJ_MIM_FORBIDDEN_CUT_TOUCH, lo, ext, 0, 1, 4,
                           static_cast<uint32_t>(i), 0, m.blocker},
                cut, m.owner_tag, 0
            });
            static_legal = false;
        }
    }
    for (size_t i = 0; i < V.mim_top_marker.size(); ++i) {
        const auto& m = V.mim_top_marker[i];
        if (!rects_clear_linf(cut, m.blocker, m.clearance)) {
            site_rejs_buf.push_back(PureEdgeSiteRej{
                EdgeRejKey{REJ_MIM_TOP_VIA_MARKER, lo, ext, m.clearance, 1, 4,
                           static_cast<uint32_t>(i), 0, m.blocker},
                cut, m.owner_tag, 0
            });
            static_legal = false;
        }
    }
    const int64_t vsp = V.vsp;
    if (V.obs_cut_idx) {
        const Box64 fq{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
        V.obs_cut_idx->for_each_intersecting(fq, [&](const BucketItem& item) {
            const Box64 vb{item.x0, item.y0, item.x1, item.y1};
            if (!(vb == cut) && !rects_clear_linf(cut, vb, vsp)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_FIXED_CUT, lo, ext, vsp, 1, 5, 0, 0, vb},
                    cut, 0, 0
                });
                static_legal = false;
            }
        });
    } else if (!V.fixed_cuts_fallback.empty()) {
        for (const auto& vb : V.fixed_cuts_fallback) {
            if (!(vb == cut) && !rects_clear_linf(cut, vb, vsp)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_FIXED_CUT, lo, ext, vsp, 1, 5, 0, 0, vb},
                    cut, 0, 0
                });
                static_legal = false;
            }
        }
    }

    if (!static_legal) {
        flush_site_rejs();
        return 0;
    }

    bool dyn_legal = true;
    out_pres_lo = 0;
    out_pres_hi = 0;
    out_pres_cut = 0;

    auto check_dyn_pad = [&](int32_t p_gi, const Box64& p_pad, int64_t p_half, int64_t& p_pres_out) {
        const PureEdgeLayerData& L = layers[static_cast<size_t>(p_gi)];
        if (!L.metal_idx) return;
        const int64_t lsp = L.space;
        const int64_t lsh = floor_div_pos(lsp, 2);
        const Box64 ebx{p_pad.x0 - lsh, p_pad.y0 - lsh, p_pad.x1 + lsh, p_pad.y1 + lsh};
        const Box64 q{p_pad.x0 - lsp, p_pad.y0 - lsp, p_pad.x1 + lsp, p_pad.y1 + lsp};
        L.metal_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id == L.net_metal_id) return;
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            int64_t sx = 0, sy = 0;
            box_sep(p_pad, b, sx, sy);
            if (sx == 0 && sy == 0) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_COMMITTED_METAL_CONTACT, p_gi, p_half, 0, 1, 2, item.owner_id, 0, b},
                    p_pad, L.metal_idx->entries[item.entry_idx].owner_tag, 0
                });
                dyn_legal = false;
            } else if (lsp > 0 && sx < lsp && sy < lsp &&
                       euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                                        static_cast<uint64_t>(lsp))) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_COMMITTED_METAL_SPACING, p_gi, p_half, lsp, 1, 2, item.owner_id, 0, b},
                    p_pad, L.metal_idx->entries[item.entry_idx].owner_tag, 0
                });
                dyn_legal = false;
            }
            if (dyn_legal) {
                const int64_t ox0 = std::max(ebx.x0, b.x0 - lsh);
                const int64_t oy0 = std::max(ebx.y0, b.y0 - lsh);
                const int64_t ox1 = std::min(ebx.x1, b.x1 + lsh);
                const int64_t oy1 = std::min(ebx.y1, b.y1 + lsh);
                if (ox0 < ox1 && oy0 < oy1) {
                    const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
                    const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
                    p_pres_out += pres * ncx * ncy;
                }
            }
        });
    };

    check_dyn_pad(lo, pad_lo, bh, out_pres_lo);
    check_dyn_pad(lo + 1, pad_hi, ah, out_pres_hi);

    if (V.cut_idx) {
        const int64_t ch = floor_div_pos(vsp, 2);
        const Box64 ebx{cut.x0 - ch, cut.y0 - ch, cut.x1 + ch, cut.y1 + ch};
        const Box64 cq{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
        V.cut_idx->for_each_intersecting(cq, [&](const BucketItem& item) {
            const Box64 vb{item.x0, item.y0, item.x1, item.y1};
            if (item.owner_id == V.net_cut_id) {
                if (!(vb == cut) && !rects_clear_linf(cut, vb, vsp)) {
                    site_rejs_buf.push_back(PureEdgeSiteRej{
                        EdgeRejKey{REJ_COMMITTED_SELF_CUT, lo, ext, vsp, 1, 3, item.owner_id, 0, vb},
                        cut, V.cut_idx->entries[item.entry_idx].owner_tag, 0
                    });
                    dyn_legal = false;
                }
                return;
            }
            if (closed_rects_touch(cut, vb)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_COMMITTED_CUT_CONTACT, lo, ext, 0, 1, 3, item.owner_id, 0, vb},
                    cut, V.cut_idx->entries[item.entry_idx].owner_tag, 0
                });
                dyn_legal = false;
            } else if (!rects_clear_linf(cut, vb, vsp)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_COMMITTED_CUT_SPACING, lo, ext, vsp, 1, 3, item.owner_id, 0, vb},
                    cut, V.cut_idx->entries[item.entry_idx].owner_tag, 0
                });
                dyn_legal = false;
            }
            if (dyn_legal) {
                const int64_t ix0 = std::max(ebx.x0, vb.x0 - ch);
                const int64_t iy0 = std::max(ebx.y0, vb.y0 - ch);
                const int64_t ix1 = std::min(ebx.x1, vb.x1 + ch);
                const int64_t iy1 = std::min(ebx.y1, vb.y1 + ch);
                if (ix0 < ix1 && iy0 < iy1) {
                    const int64_t ncx = std::max<int64_t>(1, (ix1 - ix0) / rc);
                    const int64_t ncy = std::max<int64_t>(1, (iy1 - iy0) / rc);
                    out_pres_cut += pres * ncx * ncy;
                }
            }
        });
    }

    if (!V.attempt_cuts.empty()) {
        for (const auto& vb : V.attempt_cuts) {
            if (!(vb == cut) && !rects_clear_linf(cut, vb, vsp)) {
                site_rejs_buf.push_back(PureEdgeSiteRej{
                    EdgeRejKey{REJ_ATTEMPT_CUT, lo, ext, vsp, 1, 0, 0, 0, vb},
                    cut, net_tag, 0
                });
                dyn_legal = false;
            }
        }
    }

    if (!dyn_legal) {
        flush_site_rejs();
        return 0;
    }
    return 1;
}

int PureEdgeContext::check_via_legal(
    int64_t x, int64_t y, int32_t lo, int32_t n_flag,
    int64_t& out_pres_lo, int64_t& out_pres_hi, int64_t& out_pres_cut) {
    const int st = check_via_legal_single(x, y, lo, n_flag, out_pres_lo, out_pres_hi, out_pres_cut);
    if (st <= 0 || !mirror_ctx) return st;
    const int64_t rx = mirror_axis2 - x;
    int64_t mp_lo = 0, mp_hi = 0, mp_cut = 0;
    const int mst = mirror_ctx->check_via_legal_single(rx, y, lo, n_flag, mp_lo, mp_hi, mp_cut);
    if (mst <= 0) return mst;
    const PureEdgeViaData& V = vias[static_cast<size_t>(lo)];
    const int64_t bh = (n_flag == 1) ? V.bh1 : V.bh0;
    const int64_t ah = (n_flag == 1) ? V.ah1 : V.ah0;
    const int64_t ext = (n_flag == 1) ? V.ext1 : V.ext0;
    const Box64 p_pad_lo{x - bh, y - bh, x + bh, y + bh};
    const Box64 m_pad_lo{rx - bh, y - bh, rx + bh, y + bh};
    if (reject_active_mirror_pair_box(lo, p_pad_lo, m_pad_lo, bh)) return 0;
    const Box64 p_pad_hi{x - ah, y - ah, x + ah, y + ah};
    const Box64 m_pad_hi{rx - ah, y - ah, rx + ah, y + ah};
    if (reject_active_mirror_pair_box(lo + 1, p_pad_hi, m_pad_hi, ah)) return 0;
    const Box64 p_cut{x - ext, y - ext, x + ext, y + ext};
    const Box64 m_cut{rx - ext, y - ext, rx + ext, y + ext};
    if (!rects_clear_linf(p_cut, m_cut, V.vsp)) {
        record_rej(
            EdgeRejKey{REJ_MIRROR_PAIR_CUT, lo, ext, V.vsp, 1, 6, 0, 0, p_cut},
            m_cut, mirror_ctx->net_tag);
        mirror_pair_conflict_seen = true;
        return 0;
    }
    return 1;
}

int PureEdgeContext::eval_box_congestion(
    int32_t gi, const Box64& box, int64_t half, int64_t present_hint, int64_t& out_cost) const noexcept {
    if (gi < 0 || gi >= n_layers) return -1;
    if (needs_wide_fallback_cong(gi, box, half)) return -1;
    const PureEdgeLayerData& L = layers[static_cast<size_t>(gi)];
    const int64_t sp = L.space;
    const int64_t sh = floor_div_pos(sp, 2);
    const Box64 ebx{box.x0 - sh, box.y0 - sh, box.x1 + sh, box.y1 + sh};
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    int64_t total = 0;

    if (present_hint >= 0) {
        total = present_hint;
    } else if (L.metal_idx) {
        L.metal_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id == L.net_metal_id) return;
            const int64_t ox0 = std::max(ebx.x0, item.x0 - sh);
            const int64_t oy0 = std::max(ebx.y0, item.y0 - sh);
            const int64_t ox1 = std::min(ebx.x1, item.x1 + sh);
            const int64_t oy1 = std::min(ebx.y1, item.y1 + sh);
            if (ox0 >= ox1 || oy0 >= oy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
            total += pres * ncx * ncy;
        });
    }
    if (L.obs_idx && L.net_obs_id != UINT32_MAX) {
        L.obs_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id != L.net_obs_id) return;
            int64_t sx = 0, sy = 0;
            if (euclid_in_open_band(box, Box64{item.x0, item.y0, item.x1, item.y1}, sp, sx, sy)) {
                total += pres;
            }
        });
    }
    if (L.att_idx) {
        L.att_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            int64_t sx = 0, sy = 0;
            if (euclid_in_open_band(box, Box64{item.x0, item.y0, item.x1, item.y1}, sp, sx, sy)) {
                total += pres;
            }
        });
    }
    if (hist_grid) {
        total += hist_grid->eval_units(gi, ebx, rc);
    }
    bool ovf = false;
    out_cost = scale_congestion_units(total, mul_scale, cost_factor_scale, ovf);
    return ovf ? -1 : 1;
}

int PureEdgeContext::eval_via_cut_congestion(
    int32_t lo, const Box64& cut, int64_t present_cut_hint, int64_t& out_cost) const noexcept {
    if (lo < 0 || lo >= n_layers - 1) return -1;
    const PureEdgeViaData& V = vias[static_cast<size_t>(lo)];
    const int64_t vsp = V.vsp;
    const int64_t ch = floor_div_pos(vsp, 2);
    const Box64 ebx{cut.x0 - ch, cut.y0 - ch, cut.x1 + ch, cut.y1 + ch};
    int64_t total = 0;
    if (present_cut_hint >= 0) {
        total = present_cut_hint;
    } else if (V.cut_idx) {
        const Box64 cq{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
        V.cut_idx->for_each_intersecting(cq, [&](const BucketItem& item) {
            if (item.owner_id == V.net_cut_id) return;
            const int64_t ix0 = std::max(ebx.x0, item.x0 - ch);
            const int64_t iy0 = std::max(ebx.y0, item.y0 - ch);
            const int64_t ix1 = std::min(ebx.x1, item.x1 + ch);
            const int64_t iy1 = std::min(ebx.y1, item.y1 + ch);
            if (ix0 >= ix1 || iy0 >= iy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ix1 - ix0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (iy1 - iy0) / rc);
            total += pres * ncx * ncy;
        });
    }
    if (hist_grid) {
        total += hist_grid->eval_units(static_cast<int64_t>(n_layers) + lo, ebx, rc);
    }
    bool ovf = false;
    out_cost = scale_congestion_units(total, mul_scale, cost_factor_scale, ovf);
    return ovf ? -1 : 1;
}

int PureEdgeContext::eval_via_congestion(
    int64_t x, int64_t y, int32_t lo, int32_t n_flag,
    int64_t pres_lo_hint, int64_t pres_hi_hint, int64_t pres_cut_hint,
    int64_t& out_cost) const noexcept {
    if (lo < 0 || lo >= n_layers - 1) return -1;
    const PureEdgeViaData& V = vias[static_cast<size_t>(lo)];
    const int64_t bh = (n_flag == 1) ? V.bh1 : V.bh0;
    const int64_t ah = (n_flag == 1) ? V.ah1 : V.ah0;
    const int64_t ext = (n_flag == 1) ? V.ext1 : V.ext0;
    int64_t c0 = 0, c1 = 0, cc = 0;
    if (eval_box_congestion(lo, Box64{x - bh, y - bh, x + bh, y + bh}, bh, pres_lo_hint, c0) < 0) return -1;
    if (eval_box_congestion(lo + 1, Box64{x - ah, y - ah, x + ah, y + ah}, ah, pres_hi_hint, c1) < 0) return -1;
    if (eval_via_cut_congestion(lo, Box64{x - ext, y - ext, x + ext, y + ext}, pres_cut_hint, cc) < 0) return -1;
    out_cost = c0 + c1 + cc;
    return 1;
}

int PureEdgeContext::eval_wire_price(
    int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
    int64_t present_hint, int64_t& out_cost) const noexcept {
    if (gi < 0 || gi >= n_layers) return -1;
    const int64_t hw = layers[static_cast<size_t>(gi)].hw;
    const Box64 wb{
        (ax <= bx ? ax : bx) - hw, (ay <= by ? ay : by) - hw,
        (ax <= bx ? bx : ax) + hw, (ay <= by ? by : ay) + hw
    };
    int64_t cong = 0;
    if (eval_box_congestion(gi, wb, hw, present_hint, cong) < 0) return -1;
    out_cost = eval_edge_objective(gi, ax, ay, bx, by) + cong;
    return 1;
}

int PureEdgeContext::eval_via_static_cost_fresh(
    int64_t x, int64_t y, int32_t lo, int64_t& out_cost, int32_t& out_model_n) {
    int64_t p_lo = 0, p_hi = 0, p_cut = 0;
    const int st0 = check_via_legal(x, y, lo, 0, p_lo, p_hi, p_cut);
    if (st0 < 0) return -1;
    if (st0 == 0) {
        if (!single_escape_ok) return 0;
        int64_t sp_lo = 0, sp_hi = 0, sp_cut = 0;
        const int st1 = check_via_legal(x, y, lo, 1, sp_lo, sp_hi, sp_cut);
        if (st1 < 0) return -1;
        if (st1 == 0) return 0;
        int64_t cong1 = 0;
        if (eval_via_congestion(x, y, lo, 1, sp_lo, sp_hi, sp_cut, cong1) < 0) return -1;
        out_cost = via_cost + single_via_penalty + cong1;
        out_model_n = 1;
        return 1;
    }
    int64_t cong0 = 0;
    if (eval_via_congestion(x, y, lo, 0, p_lo, p_hi, p_cut, cong0) < 0) return -1;
    out_cost = via_cost + cong0;
    out_model_n = 0;
    if (single_escape_ok && cong0 > single_via_penalty) {
        int64_t sp_lo = 0, sp_hi = 0, sp_cut = 0;
        const int st1 = check_via_legal(x, y, lo, 1, sp_lo, sp_hi, sp_cut);
        if (st1 < 0) return -1;
        if (st1 == 1) {
            int64_t cong1 = 0;
            if (eval_via_congestion(x, y, lo, 1, sp_lo, sp_hi, sp_cut, cong1) < 0) return -1;
            const int64_t scost = via_cost + single_via_penalty + cong1;
            if (scost < out_cost) {
                out_cost = scost;
                out_model_n = 1;
            }
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// 2. Pure C++20 Hard Connectivity Preflight
// ---------------------------------------------------------------------------

PreflightResult run_hard_connectivity_preflight_pure(
    PureEdgeContext& ctx,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int32_t>& layers,
    std::vector<int64_t> start_ids,
    std::vector<int64_t> goal_ids,
    int64_t max_exp,
    bool within_cap,
    const GridlessSearchHooks& hooks) {
    PreflightResult res{};
    auto& cart = ctx.cart;
    cart.reset_for_graph(xs, ys, layers, std::max<int32_t>(0, ctx.n_layers - 1), max_exp, within_cap);

    std::vector<uint8_t> reached;
    std::vector<uint8_t> tested_edges;
    std::vector<uint8_t> is_target;
    try {
        reached.assign(static_cast<size_t>(cart.potential_node_count), 0);
        tested_edges.assign(static_cast<size_t>(cart.packed_edge_count), 0);
        is_target.assign(static_cast<size_t>(cart.potential_node_count), 0);
    } catch (const std::bad_alloc&) {
        res.status = -1;
        return res;
    }

    std::sort(start_ids.begin(), start_ids.end());
    start_ids.erase(std::unique(start_ids.begin(), start_ids.end()), start_ids.end());
    std::sort(goal_ids.begin(), goal_ids.end());
    goal_ids.erase(std::unique(goal_ids.begin(), goal_ids.end()), goal_ids.end());

    const auto& seed_ids = (start_ids.size() <= goal_ids.size()) ? start_ids : goal_ids;
    const auto& target_ids = (start_ids.size() <= goal_ids.size()) ? goal_ids : start_ids;
    for (int64_t tid : target_ids) {
        if (tid >= 0 && tid < cart.potential_node_count) {
            is_target[static_cast<size_t>(tid)] = 1;
        }
    }

    std::vector<int64_t> frontier;
    frontier.reserve(std::min<size_t>(static_cast<size_t>(cart.potential_node_count), seed_ids.size() + 1024));
    bool hard_connected = false;
    for (int64_t sid : seed_ids) {
        if (sid >= 0 && sid < cart.potential_node_count) {
            reached[static_cast<size_t>(sid)] = 1;
            frontier.push_back(sid);
            if (is_target[static_cast<size_t>(sid)]) hard_connected = true;
        }
    }

    auto call_wd = [&](int32_t kind, int64_t nodes, int64_t edges, bool packed) -> bool {
        return !hooks.preflight_watchdog || hooks.preflight_watchdog(hooks.ctx, kind, nodes, edges, packed);
    };

    int64_t connectivity_nodes = 0;
    int64_t connectivity_edges = 0;
    if (!call_wd(0, 0, 0, cart.packed_enabled)) {
        res.status = -2;
        return res;
    }

    if (!hard_connected) {
        if (!call_wd(0, 0, 0, cart.packed_enabled)) {
            res.status = -2;
            return res;
        }
        cart.enable_packed(true);
    }

    const int64_t nx = cart.nx;
    const int64_t ny = cart.ny;
    const int64_t nl = cart.nl;
    const int64_t plane = cart.plane;
    const int64_t slot_cnt = cart.packed_slot_count;

    auto eval_planar_hard = [&](int32_t e_ul, int64_t e_ax, int64_t e_ay, int64_t e_bx, int64_t e_by,
                                int64_t eid, bool& out_legal) -> bool {
        ++cart.static_geometry_work;
        if ((cart.static_geometry_work & 4095) == 0) {
            if (!call_wd(0, connectivity_nodes, connectivity_edges, cart.packed_enabled)) {
                return false;
            }
        }
        int64_t slot = -1;
        if (cart.packed_enabled) {
            slot = eid % slot_cnt;
            if (cart.packed_keys[static_cast<size_t>(slot)] == eid) {
                out_legal = (cart.packed_costs[static_cast<size_t>(slot)] != CartesianGraphCache::PACKED_ILLEGAL);
                return true;
            }
        }
        int64_t pres_val = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
        bool rej_inc = false;
        const int st = ctx.check_wire_legal(e_ul, e_ax, e_ay, e_bx, e_by, pres_val);
        if (st >= 0) {
            out_legal = (st == 1);
            rej_inc = (st == 0);
        } else {
            if (!hooks.wide_wire) {
                out_legal = false;
                rej_inc = false;
            } else if (!hooks.wide_wire(hooks.ctx, e_ul, e_ax, e_ay, e_bx, e_by, out_legal, pres_val, rej_inc)) {
                return false;
            }
        }
        if (cart.packed_enabled && (out_legal || rej_inc)) {
            const size_t su = static_cast<size_t>(slot);
            cart.packed_keys[su] = eid;
            cart.packed_costs[su] = out_legal
                ? CartesianGraphCache::PACKED_LEGAL_UNPRICED
                : CartesianGraphCache::PACKED_ILLEGAL;
            cart.packed_present[su * 3 + 0] = out_legal ? pres_val : CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
            cart.packed_present[su * 3 + 1] = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
            cart.packed_present[su * 3 + 2] = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
        }
        return true;
    };

    auto eval_via_hard = [&](int64_t e_ux, int64_t e_uy, int32_t e_lo, int64_t eid, bool& out_legal) -> bool {
        ++cart.static_geometry_work;
        if ((cart.static_geometry_work & 4095) == 0) {
            if (!call_wd(0, connectivity_nodes, connectivity_edges, cart.packed_enabled)) {
                return false;
            }
        }
        int64_t slot = -1;
        if (cart.packed_enabled) {
            slot = eid % slot_cnt;
            if (cart.packed_keys[static_cast<size_t>(slot)] == eid) {
                out_legal = (cart.packed_costs[static_cast<size_t>(slot)] != CartesianGraphCache::PACKED_ILLEGAL);
                return true;
            }
        }
        int64_t p_lo = -1, p_hi = -1, p_cut = -1;
        bool rej_inc = false;
        uint8_t model_n = 0;
        const int st0 = ctx.check_via_legal(e_ux, e_uy, e_lo, 0, p_lo, p_hi, p_cut);
        if (st0 >= 0) {
            out_legal = (st0 == 1);
            rej_inc = (st0 == 0);
        } else {
            if (!hooks.wide_via) {
                out_legal = false;
            } else if (!hooks.wide_via(hooks.ctx, e_ux, e_uy, e_lo, 0, out_legal, p_lo, p_hi, p_cut, rej_inc)) {
                return false;
            }
        }
        if (!out_legal && ctx.single_escape_ok) {
            int64_t sp_lo = -1, sp_hi = -1, sp_cut = -1;
            bool s_rej = false;
            const int st1 = ctx.check_via_legal(e_ux, e_uy, e_lo, 1, sp_lo, sp_hi, sp_cut);
            if (st1 >= 0) {
                out_legal = (st1 == 1);
                s_rej = (st1 == 0);
            } else {
                if (!hooks.wide_via) {
                    out_legal = false;
                } else if (!hooks.wide_via(hooks.ctx, e_ux, e_uy, e_lo, 1, out_legal, sp_lo, sp_hi, sp_cut, s_rej)) {
                    return false;
                }
            }
            if (out_legal) {
                model_n = 1;
                p_lo = sp_lo; p_hi = sp_hi; p_cut = sp_cut;
            } else {
                rej_inc = rej_inc || s_rej;
            }
        }
        if (out_legal) {
            cart.set_via_model(eid, model_n);
        }
        if (cart.packed_enabled && (out_legal || rej_inc)) {
            const size_t su = static_cast<size_t>(slot);
            cart.packed_keys[su] = eid;
            cart.packed_costs[su] = out_legal
                ? CartesianGraphCache::PACKED_LEGAL_UNPRICED
                : CartesianGraphCache::PACKED_ILLEGAL;
            if (out_legal) {
                cart.packed_present[su * 3 + 0] = p_lo;
                cart.packed_present[su * 3 + 1] = p_hi;
                cart.packed_present[su * 3 + 2] = p_cut;
            } else {
                cart.packed_present[su * 3 + 0] = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
                cart.packed_present[su * 3 + 1] = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
                cart.packed_present[su * 3 + 2] = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
            }
        }
        return true;
    };

    size_t f_idx = 0;
    while (!hard_connected && f_idx < frontier.size()) {
        if (connectivity_nodes > 0 && (connectivity_nodes & 1023) == 0) {
            if (!call_wd(0, connectivity_nodes, connectivity_edges, cart.packed_enabled)) {
                res.status = -2;
                return res;
            }
        }
        const int64_t node_id = frontier[f_idx++];
        const int32_t lp = static_cast<int32_t>(node_id / plane);
        const int64_t rem = node_id - static_cast<int64_t>(lp) * plane;
        const int32_t yp = static_cast<int32_t>(rem / nx);
        const int32_t xp = static_cast<int32_t>(rem - static_cast<int64_t>(yp) * nx);
        const int64_t ux = cart.xs[static_cast<size_t>(xp)];
        const int64_t uy = cart.ys[static_cast<size_t>(yp)];
        const int32_t ul = cart.ordered_layers[static_cast<size_t>(lp)];
        ++connectivity_nodes;

        auto try_planar = [&](int32_t nxp, int32_t nyp, int64_t eid, int64_t target_id) -> bool {
            if (tested_edges[static_cast<size_t>(eid)]) return true;
            tested_edges[static_cast<size_t>(eid)] = 1;
            ++connectivity_edges;
            bool legal = false;
            if (!eval_planar_hard(ul, ux, uy, cart.xs[static_cast<size_t>(nxp)], cart.ys[static_cast<size_t>(nyp)], eid, legal)) {
                return false;
            }
            if (legal && !reached[static_cast<size_t>(target_id)]) {
                reached[static_cast<size_t>(target_id)] = 1;
                frontier.push_back(target_id);
                if (is_target[static_cast<size_t>(target_id)]) hard_connected = true;
            }
            return true;
        };

        if (!hard_connected && xp + 1 < nx) {
            if (!try_planar(xp + 1, yp, cart.horiz_edge_id(lp, yp, xp), node_id + 1)) { res.status = -2; return res; }
        }
        if (!hard_connected && xp - 1 >= 0) {
            if (!try_planar(xp - 1, yp, cart.horiz_edge_id(lp, yp, xp - 1), node_id - 1)) { res.status = -2; return res; }
        }
        if (!hard_connected && yp + 1 < ny) {
            if (!try_planar(xp, yp + 1, cart.vert_edge_id(lp, yp, xp), node_id + nx)) { res.status = -2; return res; }
        }
        if (!hard_connected && yp - 1 >= 0) {
            if (!try_planar(xp, yp - 1, cart.vert_edge_id(lp, yp - 1, xp), node_id - nx)) { res.status = -2; return res; }
        }
        if (hard_connected) break;

        for (int dir = -1; dir <= 1 && !hard_connected; dir += 2) {
            const int32_t nlp = lp + dir;
            if (nlp < 0 || nlp >= nl) continue;
            const int32_t other = cart.ordered_layers[static_cast<size_t>(nlp)];
            if (other != ul + dir) continue;
            const int32_t lo = (ul < other) ? ul : other;
            const int64_t eid = cart.via_edge_id_pos(lo, yp, xp);
            if (tested_edges[static_cast<size_t>(eid)]) continue;
            tested_edges[static_cast<size_t>(eid)] = 1;
            ++connectivity_edges;
            bool legal = false;
            if (!eval_via_hard(ux, uy, lo, eid, legal)) { res.status = -2; return res; }
            if (legal) {
                const int64_t target_id = node_id + static_cast<int64_t>(dir) * plane;
                if (!reached[static_cast<size_t>(target_id)]) {
                    reached[static_cast<size_t>(target_id)] = 1;
                    frontier.push_back(target_id);
                    if (is_target[static_cast<size_t>(target_id)]) hard_connected = true;
                }
            }
        }
    }

    if (!call_wd(0, connectivity_nodes, connectivity_edges, cart.packed_enabled)) {
        res.status = -2;
        return res;
    }
    if (hard_connected && !cart.packed_enabled && !cart.packed_disabled) {
        cart.preflight_tested_edges = std::move(tested_edges);
    }

    res.status = hard_connected ? 1 : 0;
    res.connectivity_nodes = connectivity_nodes;
    res.connectivity_edges = connectivity_edges;
    res.packed_enabled = cart.packed_enabled;
    return res;
}

// ---------------------------------------------------------------------------
// 3. Pure C++20 Cartesian A*
// ---------------------------------------------------------------------------

AStarResult run_cartesian_astar_pure(
    PureEdgeContext& ctx,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int32_t>& layers,
    const std::vector<AStarEndpointSpec>& starts,
    const std::vector<AStarEndpointSpec>& goals,
    int64_t budget,
    int64_t solve_exp_total,
    int64_t solve_exp_budget,
    int64_t max_exp,
    bool within_cap,
    bool indexed_goal_domain,
    bool has_incumbent,
    int64_t incumbent_score,
    bool force_packed,
    const GridlessSearchHooks& hooks) {
    AStarResult res{};
    auto& cart = ctx.cart;
    cart.reset_for_graph(xs, ys, layers, std::max<int32_t>(0, ctx.n_layers - 1), max_exp, within_cap);

    if (force_packed && !cart.packed_enabled) {
        cart.enable_packed(true);
    }

    const int64_t nx = cart.nx;
    const int64_t ny = cart.ny;
    const int64_t nl = cart.nl;
    const int64_t plane = cart.plane;
    const int64_t pnodes = cart.potential_node_count;
    const int64_t slot_cnt = cart.packed_slot_count;
    const int64_t step_cost = ctx.step_cost;
    const int64_t via_cost = ctx.via_cost;
    const int64_t turn_cost = ctx.turn_cost;

    auto call_wd = [&](int32_t kind, int64_t exps, int64_t tap, bool packed) -> bool {
        return !hooks.astar_watchdog || hooks.astar_watchdog(hooks.ctx, kind, exps, tap, packed);
    };

    const bool use_bbox_hmin = (!indexed_goal_domain && !within_cap);
    int64_t gx0 = 0, gx1 = 0, gy0 = 0, gy1 = 0;
    int32_t gl0 = 0, gl1 = 0;
    int64_t min_goal_lb = 0;
    struct GoalCoord { int64_t x, y; int32_t l; int64_t lb; };
    std::vector<GoalCoord> gcoords;
    gcoords.reserve(goals.size());
    for (size_t i = 0; i < goals.size(); ++i) {
        const auto& g = goals[i];
        const int64_t x = cart.xs[static_cast<size_t>(g.xp)];
        const int64_t y = cart.ys[static_cast<size_t>(g.yp)];
        const int32_t l = cart.ordered_layers[static_cast<size_t>(g.lp)];
        gcoords.push_back(GoalCoord{x, y, l, g.lb});
        if (i == 0) {
            gx0 = gx1 = x; gy0 = gy1 = y; gl0 = gl1 = l; min_goal_lb = g.lb;
        } else {
            if (x < gx0) gx0 = x; else if (x > gx1) gx1 = x;
            if (y < gy0) gy0 = y; else if (y > gy1) gy1 = y;
            if (l < gl0) gl0 = l; else if (l > gl1) gl1 = l;
            if (g.lb < min_goal_lb) min_goal_lb = g.lb;
        }
    }

    const bool use_pot2d = (!use_bbox_hmin && goals.size() > 16);
    std::vector<Potential2DCore> layer_pots;
    std::vector<int64_t> hmin_cache;
    bool use_hmin_cache = false;
    if (use_pot2d) {
        layer_pots.resize(static_cast<size_t>(nl));
        for (int32_t lp = 0; lp < static_cast<int32_t>(nl); ++lp) {
            const int32_t ul = cart.ordered_layers[static_cast<size_t>(lp)];
            std::unordered_map<std::pair<int64_t, int64_t>, int64_t, PairHash> proj;
            proj.reserve(gcoords.size() * 2 + 1);
            for (const auto& gc : gcoords) {
                int32_t dl = ul - gc.l;
                if (dl < 0) dl = -dl;
                const int64_t c = gc.lb + static_cast<int64_t>(dl) * via_cost;
                auto it = proj.find({gc.x, gc.y});
                if (it == proj.end()) proj.emplace(std::make_pair(gc.x, gc.y), c);
                else if (c < it->second) it->second = c;
            }
            std::vector<Pt2D> pts;
            pts.reserve(proj.size());
            for (const auto& kv : proj) pts.push_back(Pt2D{kv.first.first, kv.first.second, kv.second});
            layer_pots[static_cast<size_t>(lp)].init(std::move(pts), step_cost);
        }
        if (pnodes > 0 && pnodes <= 4000000) {
            try {
                hmin_cache.assign(static_cast<size_t>(pnodes), -1);
                use_hmin_cache = true;
            } catch (const std::bad_alloc&) {
                hmin_cache.clear();
                use_hmin_cache = false;
            }
        }
    }

    auto eval_hmin = [&](int32_t h_xp, int32_t h_yp, int32_t h_lp) noexcept -> int64_t {
        const int64_t x = cart.xs[static_cast<size_t>(h_xp)];
        const int64_t y = cart.ys[static_cast<size_t>(h_yp)];
        const int32_t ul = cart.ordered_layers[static_cast<size_t>(h_lp)];
        if (use_bbox_hmin) {
            const int64_t dx = (x < gx0) ? (gx0 - x) : ((x > gx1) ? (x - gx1) : 0);
            const int64_t dy = (y < gy0) ? (gy0 - y) : ((y > gy1) ? (y - gy1) : 0);
            const int64_t dl = (ul < gl0) ? (gl0 - ul) : ((ul > gl1) ? (ul - gl1) : 0);
            return (dx + dy) * step_cost + dl * via_cost + min_goal_lb;
        }
        if (use_pot2d) {
            const int64_t pid = static_cast<int64_t>(h_lp) * plane + static_cast<int64_t>(h_yp) * nx + h_xp;
            if (use_hmin_cache) {
                const int64_t cached = hmin_cache[static_cast<size_t>(pid)];
                if (cached >= 0) return cached;
                const int64_t val = layer_pots[static_cast<size_t>(h_lp)].eval(x, y);
                hmin_cache[static_cast<size_t>(pid)] = val;
                return val;
            }
            return layer_pots[static_cast<size_t>(h_lp)].eval(x, y);
        }
        int64_t best = std::numeric_limits<int64_t>::max();
        for (const auto& gc : gcoords) {
            int64_t dx = x - gc.x; if (dx < 0) dx = -dx;
            int64_t dy = y - gc.y; if (dy < 0) dy = -dy;
            int32_t dl = ul - gc.l; if (dl < 0) dl = -dl;
            const int64_t cand = (dx + dy) * step_cost + static_cast<int64_t>(dl) * via_cost + gc.lb;
            if (cand < best) best = cand;
        }
        return best;
    };

    const bool use_dense_goals = (pnodes > 0 && pnodes <= 8000000);
    std::vector<int32_t> goal_dense;
    std::unordered_map<int64_t, int32_t> goal_sparse;
    if (use_dense_goals) {
        goal_dense.assign(static_cast<size_t>(pnodes), -1);
        for (size_t i = 0; i < goals.size(); ++i) {
            const int64_t pid = static_cast<int64_t>(goals[i].lp) * plane +
                                static_cast<int64_t>(goals[i].yp) * nx + goals[i].xp;
            goal_dense[static_cast<size_t>(pid)] = static_cast<int32_t>(i);
        }
    } else {
        goal_sparse.reserve(goals.size() * 2 + 1);
        for (size_t i = 0; i < goals.size(); ++i) {
            const int64_t pid = static_cast<int64_t>(goals[i].lp) * plane +
                                static_cast<int64_t>(goals[i].yp) * nx + goals[i].xp;
            goal_sparse[pid] = static_cast<int32_t>(i);
        }
    }
    auto lookup_goal = [&](int64_t pid) noexcept -> int32_t {
        if (use_dense_goals) return goal_dense[static_cast<size_t>(pid)];
        auto it = goal_sparse.find(pid);
        return (it == goal_sparse.end()) ? -1 : it->second;
    };

    constexpr int64_t INF_DIST = std::numeric_limits<int64_t>::max();
    const int64_t total_states = pnodes * 3;
    bool use_dense_states = (total_states > 0 && total_states <= 8000000);
    std::vector<int64_t> dist_dense;
    std::vector<int64_t> prev_dense;
    std::vector<int32_t> evia_dense;
    struct SparseState { int64_t dist; int64_t prev; int32_t evia; };
    std::unordered_map<int64_t, SparseState> state_sparse;
    if (use_dense_states) {
        try {
            dist_dense.assign(static_cast<size_t>(total_states), INF_DIST);
            prev_dense.assign(static_cast<size_t>(total_states), -1);
            evia_dense.assign(static_cast<size_t>(total_states), -1);
        } catch (const std::bad_alloc&) {
            dist_dense.clear(); prev_dense.clear(); evia_dense.clear();
            use_dense_states = false;
        }
    }
    auto make_sid = [&](int32_t s_lp, int32_t s_yp, int32_t s_xp, int8_t s_d) noexcept -> int64_t {
        return (static_cast<int64_t>(s_lp) * plane + static_cast<int64_t>(s_yp) * nx + s_xp) * 3 + s_d;
    };
    auto get_dist = [&](int64_t sid) noexcept -> int64_t {
        if (use_dense_states) return dist_dense[static_cast<size_t>(sid)];
        auto it = state_sparse.find(sid);
        return (it == state_sparse.end()) ? INF_DIST : it->second.dist;
    };
    auto set_state = [&](int64_t sid, int64_t d, int64_t pr, int32_t ev) {
        if (use_dense_states) {
            const size_t idx_s = static_cast<size_t>(sid);
            dist_dense[idx_s] = d;
            prev_dense[idx_s] = pr;
            evia_dense[idx_s] = ev;
        } else {
            state_sparse[sid] = SparseState{d, pr, ev};
        }
    };

    int64_t expansions = 0;
    int64_t terminal_access_priced = 0;
    std::unordered_map<TermViaKey, int64_t, TermViaKeyHash> exact_term_cache;

    auto eval_exact_terminal = [&](const AStarEndpointSpec& ep, int64_t& out_cost) -> bool {
        if (!ep.has_via) { out_cost = 0; return true; }
        const TermViaKey k{ep.vx, ep.vy, ep.vlo, ep.vn};
        auto it = exact_term_cache.find(k);
        if (it != exact_term_cache.end()) { out_cost = it->second; return true; }
        if (!call_wd(3, expansions, terminal_access_priced, cart.packed_enabled)) {
            return false;
        }
        const int32_t nf = (ep.vn == 1) ? 1 : 0;
        int64_t cong = 0;
        if (ctx.eval_via_congestion(ep.vx, ep.vy, ep.vlo, nf, -1, -1, -1, cong) >= 0) {
            out_cost = via_cost + (nf == 1 ? ctx.single_via_penalty : 0) + cong;
        } else {
            if (!hooks.wide_cong || !hooks.wide_cong(hooks.ctx, 2, ep.vx, ep.vy, ep.vlo, ep.vn, 0, out_cost)) {
                return false;
            }
        }
        exact_term_cache.emplace(k, out_cost);
        ++terminal_access_priced;
        if (!call_wd(3, expansions, terminal_access_priced, cart.packed_enabled)) {
            return false;
        }
        return true;
    };

    auto eval_planar_static_cost = [&](int32_t p_ul, int64_t p_ax, int64_t p_ay, int64_t p_bx, int64_t p_by,
                                       int64_t eid, bool& out_legal, int64_t& out_cost) -> bool {
        ++cart.static_geometry_work;
        if ((cart.static_geometry_work & 4095) == 0) {
            if (!call_wd(1, expansions, terminal_access_priced, cart.packed_enabled)) {
                return false;
            }
        }
        if (!cart.packed_enabled && !cart.packed_disabled && cart.graph_within_potential_cap && slot_cnt > 0) {
            const bool force_hit = (!cart.preflight_tested_edges.empty() &&
                                    cart.preflight_tested_edges[static_cast<size_t>(eid)] != 0);
            if (force_hit || cart.static_geometry_work > slot_cnt) {
                cart.enable_packed(true);
            }
        }
        int64_t slot = -1;
        bool hard_legal = false;
        int64_t pres_hint = CartesianGraphCache::PACKED_PRESENT_UNKNOWN;
        if (cart.packed_enabled) {
            slot = eid % slot_cnt;
            const size_t su = static_cast<size_t>(slot);
            if (cart.packed_keys[su] == eid) {
                const int64_t cached = cart.packed_costs[su];
                if (cached == CartesianGraphCache::PACKED_ILLEGAL) {
                    out_legal = false;
                    return true;
                }
                if (cached == CartesianGraphCache::PACKED_LEGAL_UNPRICED) {
                    hard_legal = true;
                    pres_hint = cart.packed_present[su * 3];
                } else {
                    out_legal = true;
                    out_cost = cached;
                    return true;
                }
            }
        }
        if (!hard_legal) {
            bool rej_inc = false;
            const int st = ctx.check_wire_legal(p_ul, p_ax, p_ay, p_bx, p_by, pres_hint);
            if (st >= 0) {
                hard_legal = (st == 1);
                rej_inc = (st == 0);
            } else {
                if (!hooks.wide_wire) {
                    hard_legal = false;
                } else if (!hooks.wide_wire(hooks.ctx, p_ul, p_ax, p_ay, p_bx, p_by, hard_legal, pres_hint, rej_inc)) {
                    return false;
                }
            }
            if (!hard_legal) {
                if (cart.packed_enabled && rej_inc) {
                    const size_t su = static_cast<size_t>(slot);
                    cart.packed_keys[su] = eid;
                    cart.packed_costs[su] = CartesianGraphCache::PACKED_ILLEGAL;
                }
                out_legal = false;
                return true;
            }
        }
        if (ctx.eval_wire_price(p_ul, p_ax, p_ay, p_bx, p_by, pres_hint, out_cost) < 0) {
            int64_t cong = 0;
            if (!hooks.wide_cong || !hooks.wide_cong(hooks.ctx, 0, p_ul, p_ax, p_ay, p_bx, p_by, cong)) {
                return false;
            }
            out_cost = ctx.eval_edge_objective(p_ul, p_ax, p_ay, p_bx, p_by) + cong;
        }
        out_legal = true;
        if (cart.packed_enabled && out_cost >= 0) {
            const size_t su = static_cast<size_t>(slot);
            cart.packed_keys[su] = eid;
            cart.packed_costs[su] = out_cost;
        }
        return true;
    };

    auto eval_via_static_cost = [&](int64_t v_ux, int64_t v_uy, int32_t v_lo, int64_t eid,
                                    bool& out_legal, int64_t& out_cost, uint8_t& out_model_n) -> bool {
        ++cart.static_geometry_work;
        if ((cart.static_geometry_work & 4095) == 0) {
            if (!call_wd(1, expansions, terminal_access_priced, cart.packed_enabled)) {
                return false;
            }
        }
        if (!cart.packed_enabled && !cart.packed_disabled && cart.graph_within_potential_cap && slot_cnt > 0) {
            const bool force_hit = (!cart.preflight_tested_edges.empty() &&
                                    cart.preflight_tested_edges[static_cast<size_t>(eid)] != 0);
            if (force_hit || cart.static_geometry_work > slot_cnt) {
                cart.enable_packed(true);
            }
        }
        int64_t slot = -1;
        bool hard_legal = false;
        uint8_t sel_n = 0;
        int64_t p_lo = -1, p_hi = -1, p_cut = -1;
        if (cart.packed_enabled) {
            slot = eid % slot_cnt;
            const size_t su = static_cast<size_t>(slot);
            if (cart.packed_keys[su] == eid) {
                const int64_t cached = cart.packed_costs[su];
                if (cached == CartesianGraphCache::PACKED_ILLEGAL) {
                    out_legal = false;
                    return true;
                }
                if (cached == CartesianGraphCache::PACKED_LEGAL_UNPRICED) {
                    hard_legal = true;
                    sel_n = cart.get_via_model(eid);
                    p_lo = cart.packed_present[su * 3 + 0];
                    p_hi = cart.packed_present[su * 3 + 1];
                    p_cut = cart.packed_present[su * 3 + 2];
                } else {
                    out_legal = true;
                    out_cost = cached;
                    out_model_n = cart.get_via_model(eid);
                    return true;
                }
            }
        }
        if (!hard_legal) {
            bool rej_inc = false;
            const int st0 = ctx.check_via_legal(v_ux, v_uy, v_lo, 0, p_lo, p_hi, p_cut);
            bool def_legal = false;
            if (st0 >= 0) {
                def_legal = (st0 == 1);
                rej_inc = (st0 == 0);
            } else {
                if (!hooks.wide_via) {
                    def_legal = false;
                } else if (!hooks.wide_via(hooks.ctx, v_ux, v_uy, v_lo, 0, def_legal, p_lo, p_hi, p_cut, rej_inc)) {
                    return false;
                }
            }
            if (def_legal) {
                sel_n = 0;
            } else {
                bool sing_legal = false;
                if (ctx.single_escape_ok) {
                    bool s_rej = false;
                    const int st1 = ctx.check_via_legal(v_ux, v_uy, v_lo, 1, p_lo, p_hi, p_cut);
                    if (st1 >= 0) {
                        sing_legal = (st1 == 1);
                        s_rej = (st1 == 0);
                    } else {
                        if (!hooks.wide_via) {
                            sing_legal = false;
                        } else if (!hooks.wide_via(hooks.ctx, v_ux, v_uy, v_lo, 1, sing_legal, p_lo, p_hi, p_cut, s_rej)) {
                            return false;
                        }
                    }
                    rej_inc = rej_inc || s_rej;
                }
                if (!sing_legal) {
                    if (cart.packed_enabled && rej_inc) {
                        const size_t su = static_cast<size_t>(slot);
                        cart.packed_keys[su] = eid;
                        cart.packed_costs[su] = CartesianGraphCache::PACKED_ILLEGAL;
                    }
                    out_legal = false;
                    return true;
                }
                sel_n = 1;
            }
        }
        uint8_t model_n = sel_n;
        if (model_n == 1) {
            int64_t cong1 = 0;
            if (ctx.eval_via_congestion(v_ux, v_uy, v_lo, 1, p_lo, p_hi, p_cut, cong1) < 0) {
                if (!hooks.wide_cong || !hooks.wide_cong(hooks.ctx, 1, v_ux, v_uy, v_lo, 1, 0, cong1)) {
                    return false;
                }
            }
            out_cost = via_cost + ctx.single_via_penalty + cong1;
        } else {
            int64_t cong0 = 0;
            if (ctx.eval_via_congestion(v_ux, v_uy, v_lo, 0, p_lo, p_hi, p_cut, cong0) < 0) {
                if (!hooks.wide_cong || !hooks.wide_cong(hooks.ctx, 1, v_ux, v_uy, v_lo, 0, 0, cong0)) {
                    return false;
                }
            }
            out_cost = via_cost + cong0;
            if (ctx.single_escape_ok && cong0 > ctx.single_via_penalty) {
                int64_t sp_lo = -1, sp_hi = -1, sp_cut = -1;
                bool s_leg = false, s_rej = false;
                const int st1 = ctx.check_via_legal(v_ux, v_uy, v_lo, 1, sp_lo, sp_hi, sp_cut);
                if (st1 >= 0) {
                    s_leg = (st1 == 1);
                } else {
                    if (!hooks.wide_via) {
                        s_leg = false;
                    } else if (!hooks.wide_via(hooks.ctx, v_ux, v_uy, v_lo, 1, s_leg, sp_lo, sp_hi, sp_cut, s_rej)) {
                        return false;
                    }
                }
                if (s_leg) {
                    int64_t cong1 = 0;
                    if (ctx.eval_via_congestion(v_ux, v_uy, v_lo, 1, sp_lo, sp_hi, sp_cut, cong1) < 0) {
                        if (!hooks.wide_cong || !hooks.wide_cong(hooks.ctx, 1, v_ux, v_uy, v_lo, 1, 0, cong1)) {
                            return false;
                        }
                    }
                    const int64_t scost = via_cost + ctx.single_via_penalty + cong1;
                    if (scost < out_cost) {
                        out_cost = scost;
                        model_n = 1;
                    }
                }
            }
        }
        cart.set_via_model(eid, model_n);
        out_model_n = model_n;
        out_legal = true;
        if (cart.packed_enabled && out_cost >= 0) {
            const size_t su = static_cast<size_t>(slot);
            cart.packed_keys[su] = eid;
            cart.packed_costs[su] = out_cost;
        }
        return true;
    };

    std::priority_queue<AStarStartItem, std::vector<AStarStartItem>, AStarStartItemGreater> start_frontier;
    std::priority_queue<AStarPQItem, std::vector<AStarPQItem>, AStarPQItemGreater> pq;

    for (size_t i = 0; i < starts.size(); ++i) {
        const auto& s = starts[i];
        start_frontier.push(AStarStartItem{s.lb + eval_hmin(s.xp, s.yp, s.lp), static_cast<int32_t>(i)});
    }

    bool has_best_goal = false;
    int64_t best_goal_cost = INF_DIST;
    AStarPQItem best_goal_item{};
    int64_t best_goal_sid = -1;

    auto is_better_goal = [](int64_t c_cost, const AStarPQItem& cand,
                             int64_t b_cost, const AStarPQItem& best) noexcept -> bool {
        if (c_cost != b_cost) return c_cost < b_cost;
        if (cand.xp != best.xp) return cand.xp < best.xp;
        if (cand.yp != best.yp) return cand.yp < best.yp;
        if (cand.lp != best.lp) return cand.lp < best.lp;
        return cand.last_dir < best.last_dir;
    };

    int32_t status = -1;

    while (!pq.empty() || !start_frontier.empty()) {
        const int64_t pq_f = pq.empty() ? INF_DIST : pq.top().f;
        const int64_t sf_f = start_frontier.empty() ? INF_DIST : start_frontier.top().f_lb;
        const int64_t queue_lb = (sf_f < pq_f) ? sf_f : pq_f;

        if (has_incumbent && queue_lb >= incumbent_score &&
            (!has_best_goal || best_goal_cost >= incumbent_score)) {
            status = 1;
            break;
        }
        if (has_best_goal && queue_lb > best_goal_cost) {
            status = 0;
            break;
        }
        if (!start_frontier.empty() && sf_f == queue_lb && (pq.empty() || sf_f <= pq_f)) {
            const AStarStartItem si = start_frontier.top();
            start_frontier.pop();
            const auto& st = starts[static_cast<size_t>(si.start_idx)];
            int64_t base = 0;
            if (!eval_exact_terminal(st, base)) { res.status = -2; return res; }
            const int64_t s0_id = make_sid(st.lp, st.yp, st.xp, 0);
            if (base < get_dist(s0_id)) {
                set_state(s0_id, base, -1, si.start_idx);
                pq.push(AStarPQItem{base + eval_hmin(st.xp, st.yp, st.lp), st.xp, st.yp, st.lp, 0, base});
            }
            continue;
        }
        if ((expansions & 1023) == 0) {
            if (!call_wd(2, expansions, terminal_access_priced, cart.packed_enabled)) {
                res.status = -2;
                return res;
            }
        }
        const AStarPQItem u = pq.top();
        pq.pop();
        const int64_t u_id = make_sid(u.lp, u.yp, u.xp, u.last_dir);
        const int64_t du = get_dist(u_id);
        if (u.queued_g != du) continue;

        if (has_incumbent && u.f >= incumbent_score &&
            (!has_best_goal || best_goal_cost >= incumbent_score)) {
            status = 1;
            break;
        }
        if (has_best_goal && u.f > best_goal_cost) {
            status = 0;
            break;
        }

        const int64_t phys_id = static_cast<int64_t>(u.lp) * plane +
                                static_cast<int64_t>(u.yp) * nx + u.xp;
        const int32_t g_idx = lookup_goal(phys_id);
        if (g_idx >= 0) {
            int64_t tcost = 0;
            if (!eval_exact_terminal(goals[static_cast<size_t>(g_idx)], tcost)) { res.status = -2; return res; }
            const int64_t cand_cost = du + tcost;
            if (!has_best_goal || is_better_goal(cand_cost, u, best_goal_cost, best_goal_item)) {
                has_best_goal = true;
                best_goal_cost = cand_cost;
                best_goal_item = u;
                best_goal_sid = u_id;
            }
        }

        if (expansions >= budget || solve_exp_total + expansions >= solve_exp_budget) {
            if (has_best_goal && (!has_incumbent || best_goal_cost < incumbent_score)) {
                status = 0;
            } else if (has_incumbent) {
                status = 1;
            } else {
                status = (solve_exp_total + expansions >= solve_exp_budget) ? 4 : 3;
            }
            break;
        }

        ++expansions;
        const int64_t ux = cart.xs[static_cast<size_t>(u.xp)];
        const int64_t uy = cart.ys[static_cast<size_t>(u.yp)];
        const int32_t ul = cart.ordered_layers[static_cast<size_t>(u.lp)];
        const int8_t ulast = u.last_dir;

        auto relax_planar = [&](int32_t vxp, int32_t vyp, int8_t vdir, int64_t eid) -> bool {
            bool legal = false;
            int64_t scost = 0;
            if (!eval_planar_static_cost(ul, ux, uy, cart.xs[static_cast<size_t>(vxp)],
                                         cart.ys[static_cast<size_t>(vyp)], eid, legal, scost)) {
                return false;
            }
            if (!legal) return true;
            const int64_t turn = (ulast != 0 && vdir != ulast) ? turn_cost : 0;
            const int64_t nd = du + scost + turn;
            const int64_t v_id = make_sid(u.lp, vyp, vxp, vdir);
            if (nd < get_dist(v_id)) {
                set_state(v_id, nd, u_id, -1);
                pq.push(AStarPQItem{nd + eval_hmin(vxp, vyp, u.lp), vxp, vyp, u.lp, vdir, nd});
            }
            return true;
        };

        if (u.xp + 1 < nx) {
            if (!relax_planar(u.xp + 1, u.yp, 1, cart.horiz_edge_id(u.lp, u.yp, u.xp))) { res.status = -2; return res; }
        }
        if (u.xp - 1 >= 0) {
            if (!relax_planar(u.xp - 1, u.yp, 1, cart.horiz_edge_id(u.lp, u.yp, u.xp - 1))) { res.status = -2; return res; }
        }
        if (u.yp + 1 < ny) {
            if (!relax_planar(u.xp, u.yp + 1, 2, cart.vert_edge_id(u.lp, u.yp, u.xp))) { res.status = -2; return res; }
        }
        if (u.yp - 1 >= 0) {
            if (!relax_planar(u.xp, u.yp - 1, 2, cart.vert_edge_id(u.lp, u.yp - 1, u.xp))) { res.status = -2; return res; }
        }

        const int8_t next_dir = static_cast<int8_t>(ctx.via_succ_dirs[static_cast<size_t>(ulast)]);
        for (int dir = -1; dir <= 1; dir += 2) {
            const int32_t vlp = u.lp + dir;
            if (vlp < 0 || vlp >= nl) continue;
            const int32_t other = cart.ordered_layers[static_cast<size_t>(vlp)];
            if (other != ul + dir) continue;
            const int32_t lo = (ul < other) ? ul : other;
            const int64_t eid = cart.via_edge_id_pos(lo, u.yp, u.xp);
            bool legal = false;
            int64_t scost = 0;
            uint8_t model_n = 0;
            if (!eval_via_static_cost(ux, uy, lo, eid, legal, scost, model_n)) { res.status = -2; return res; }
            if (!legal) continue;
            const int64_t nd = du + scost;
            const int64_t v_id = make_sid(vlp, u.yp, u.xp, next_dir);
            if (nd < get_dist(v_id)) {
                const int32_t ev_tag = (lo << 1) | static_cast<int32_t>(model_n);
                set_state(v_id, nd, u_id, ev_tag);
                pq.push(AStarPQItem{nd + eval_hmin(u.xp, u.yp, vlp), u.xp, u.yp, vlp, next_dir, nd});
            }
        }
    }

    if (status < 0) {
        if (has_best_goal) {
            status = (has_incumbent && best_goal_cost >= incumbent_score) ? 1 : 0;
        } else if (has_incumbent) {
            status = 1;
        } else {
            status = 2;
        }
    }

    int32_t root_start_idx = -1;
    if (status == 0) {
        int64_t cur = best_goal_sid;
        while (cur >= 0) {
            const int8_t d = static_cast<int8_t>(cur % 3);
            const int64_t pid = cur / 3;
            const int32_t lp = static_cast<int32_t>(pid / plane);
            const int64_t rem = pid - static_cast<int64_t>(lp) * plane;
            const int32_t yp = static_cast<int32_t>(rem / nx);
            const int32_t xp = static_cast<int32_t>(rem - static_cast<int64_t>(yp) * nx);
            int64_t pr = -1;
            int32_t ev = -1;
            if (use_dense_states) {
                pr = prev_dense[static_cast<size_t>(cur)];
                ev = evia_dense[static_cast<size_t>(cur)];
            } else {
                const auto& st = state_sparse[cur];
                pr = st.prev;
                ev = st.evia;
            }
            int32_t vlo = -1, vn = 0;
            if (pr < 0) {
                root_start_idx = ev;
            } else if (ev >= 0) {
                vlo = ev >> 1;
                vn = ev & 1;
            }
            res.path.push_back(AStarPathNode{
                cart.xs[static_cast<size_t>(xp)], cart.ys[static_cast<size_t>(yp)],
                cart.ordered_layers[static_cast<size_t>(lp)], d, vlo, vn
            });
            cur = pr;
        }
        std::reverse(res.path.begin(), res.path.end());
    }

    res.status = status;
    res.expansions = expansions;
    res.terminal_access_priced = terminal_access_priced;
    res.root_start_idx = root_start_idx;
    res.packed_enabled = cart.packed_enabled;
    return res;
}

// ---------------------------------------------------------------------------
// 4. Pure C++20 Axis Price Cache Context Runner
// ---------------------------------------------------------------------------

int run_axis_price_ctx_pure(
    PureAxisPriceCache& cache,
    const PureEdgeContext& ctx,
    int64_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
    int64_t remaining_budget,
    int64_t& steps_charged,
    int64_t& static_price_steps,
    int64_t& price_cache_hits,
    size_t max_run_cache,
    int64_t& out_cost,
    bool& budget_stopped,
    const AxisPriceHooks& hooks) {
    if (ax == bx && ay == by) {
        out_cost = 0;
        return 1;
    }
    if (ax != bx && ay != by) {
        return -2;
    }

    auto call_wd = [&](int32_t kind_id) -> bool {
        return !hooks.watchdog || hooks.watchdog(hooks.ctx, kind_id);
    };

    const AxisRunCacheKey rck{gi, ax, ay, bx, by};
    auto rcit = cache.run_cache.find(rck);
    if (rcit != cache.run_cache.end()) {
        ++price_cache_hits;
        if ((price_cache_hits & 1023) == 0) {
            if (!call_wd(1)) return -1;
        }
        out_cost = rcit->second;
        return 1;
    }

    auto charge_one = [&]() -> int {
        if (steps_charged >= remaining_budget) {
            budget_stopped = true;
            return 0;
        }
        ++steps_charged;
        ++static_price_steps;
        if ((static_price_steps & 255) == 0) {
            if (!call_wd(0)) return -1;
        }
        return 1;
    };

    auto eval_seg = [&](int64_t p_ax, int64_t p_ay, int64_t p_bx, int64_t p_by, int64_t& val) -> bool {
        if (ctx.eval_wire_price(static_cast<int32_t>(gi), p_ax, p_ay, p_bx, p_by, -1, val) >= 0) {
            return true;
        }
        return hooks.fallback_price && hooks.fallback_price(hooks.ctx, gi, p_ax, p_ay, p_bx, p_by, val);
    };

    const bool horizontal = (ay == by);
    const int64_t first_coord = horizontal ? ax : ay;
    const int64_t last_coord = horizontal ? bx : by;
    const int64_t fixed = horizontal ? ay : ax;
    const int8_t sign = (last_coord > first_coord) ? 1 : -1;
    const auto& axes = horizontal ? cache.xs : cache.ys;
    const int64_t lo = (first_coord < last_coord) ? first_coord : last_coord;
    const int64_t hi = (first_coord < last_coord) ? last_coord : first_coord;

    const auto it_first = std::upper_bound(axes.begin(), axes.end(), lo);
    const auto it_stop = std::lower_bound(axes.begin(), axes.end(), hi);
    const int32_t first = static_cast<int32_t>(it_first - axes.begin());
    const int32_t stop = static_cast<int32_t>(it_stop - axes.begin());

    auto eval_partial = [&](int64_t part_a, int64_t part_b, int64_t& part_val) -> int {
        const int ch = charge_one();
        if (ch <= 0) return ch;
        const bool ok = horizontal
            ? eval_seg(part_a, fixed, part_b, fixed, part_val)
            : eval_seg(fixed, part_a, fixed, part_b, part_val);
        return ok ? 1 : -1;
    };

    auto eval_range = [&](int32_t begin_i, int32_t end_i, int64_t& rng_val) -> int {
        if (begin_i >= end_i) {
            rng_val = 0;
            return 1;
        }
        const AxisLineKey lk{gi, fixed, static_cast<int8_t>(horizontal ? 1 : 0), sign};
        const AxisRangeKey rk{lk, begin_i, end_i};
        auto rit = cache.range_totals.find(rk);
        if (rit != cache.range_totals.end()) {
            const int ch = charge_one();
            if (ch <= 0) return ch;
            rng_val = rit->second;
            return 1;
        }
        auto& line = cache.lines[lk];
        if (line.valid.empty() && axes.size() > 1) {
            line.vals.assign(axes.size() - 1, 0);
            line.valid.assign(axes.size() - 1, 0);
        }
        int64_t total = 0;
        for (int32_t i = begin_i; i < end_i; ++i) {
            const size_t ui = static_cast<size_t>(i);
            if (!line.valid[ui]) {
                const int ch = charge_one();
                if (ch <= 0) return ch;
                int64_t seg_a = axes[ui];
                int64_t seg_b = axes[ui + 1];
                if (sign < 0) std::swap(seg_a, seg_b);
                int64_t sval = 0;
                const bool ok = horizontal
                    ? eval_seg(seg_a, fixed, seg_b, fixed, sval)
                    : eval_seg(fixed, seg_a, fixed, seg_b, sval);
                if (!ok) return -1;
                line.vals[ui] = sval;
                line.valid[ui] = 1;
            }
            total += line.vals[ui];
        }
        cache.range_totals[rk] = total;
        const int ch = charge_one();
        if (ch <= 0) return ch;
        rng_val = total;
        return 1;
    };

    int64_t total_cost = 0;
    if (first == stop) {
        const int st = eval_partial(first_coord, last_coord, total_cost);
        if (st <= 0) return st;
    } else {
        const int64_t leading = (sign > 0) ? axes[static_cast<size_t>(first)]
                                           : axes[static_cast<size_t>(stop - 1)];
        const int64_t trailing = (sign > 0) ? axes[static_cast<size_t>(stop - 1)]
                                            : axes[static_cast<size_t>(first)];
        int64_t a_val = 0, mid_val = 0, b_val = 0;
        int st = eval_partial(first_coord, leading, a_val);
        if (st <= 0) return st;
        st = eval_range(first, stop - 1, mid_val);
        if (st <= 0) return st;
        st = eval_partial(trailing, last_coord, b_val);
        if (st <= 0) return st;
        total_cost = a_val + mid_val + b_val;
    }

    if (cache.run_cache.size() < max_run_cache) {
        cache.run_cache.emplace(rck, total_cost);
    }
    out_cost = total_cost;
    return 1;
}

// ---------------------------------------------------------------------------
// 4. Pure C++20 Region-A* Via-Target Projector & Frontier Heap (Phase 1B)
// ---------------------------------------------------------------------------

PureViaTargetProjector::~PureViaTargetProjector() {
    clear_frontier_pure();
    for (NativeViaTargetCacheEntry* p : owned_cache_entries) delete p;
    owned_cache_entries.clear();
    native_target_cache.clear();
}

void PureViaTargetProjector::clear_frontier_pure() {
    frontier_heap.clear();
    for (NativeViaTargetCacheEntry* p : uncached_frontier_entries) delete p;
    uncached_frontier_entries.clear();
    rect_entry_counts.clear();
    frontier_seq = 0;
}

bool PureViaTargetProjector::is_goal_state(int64_t x_nm, int64_t y_nm, int32_t layer) const noexcept {
    auto it = goal_by_layer.find(layer);
    if (it != goal_by_layer.end() && it->second.find({x_nm, y_nm}) != it->second.end()) {
        return true;
    }
    if (target_singleton_none && layer == target_node_gi) {
        if (x_nm == target_node_x && y_nm == target_node_y) return false;
        if (!target_excluded_xy.empty() &&
            target_excluded_xy.find({x_nm, y_nm}) != target_excluded_xy.end()) {
            return false;
        }
        for (const auto& r : target_regions) {
            if (r.x0 <= x_nm && x_nm <= r.x1 && r.y0 <= y_nm && y_nm <= r.y1) {
                return true;
            }
        }
    }
    return false;
}

bool PureViaTargetProjector::has_cached_target(
    int32_t ul, int32_t other, int32_t rect,
    int64_t px, int64_t py, bool with_owners) const noexcept {
    const ViaTargetCacheKey key{ul, other, rect, px, py, static_cast<uint8_t>(with_owners ? 1 : 0)};
    return native_target_cache.find(key) != native_target_cache.end();
}

void PureViaTargetProjector::inc_rect_entry(
    int32_t layer, int32_t rect_id, int32_t ulast, int64_t bx, int64_t by) {
    const RectEntryKey rk{layer, rect_id, ulast, bx, by};
    rect_entry_counts[rk] += 1;
}

bool PureViaTargetProjector::frontier_top_f(int64_t& out_f) const noexcept {
    if (frontier_heap.empty()) return false;
    out_f = frontier_heap.front().lb_f;
    return true;
}

bool PureViaTargetProjector::pop_frontier_raw(RegionFrontierEntry& out_entry) noexcept {
    if (frontier_heap.empty()) return false;
    std::pop_heap(frontier_heap.begin(), frontier_heap.end(), RegionFrontierMinCmp{});
    out_entry = frontier_heap.back();
    frontier_heap.pop_back();
    return true;
}

const std::vector<int32_t>* PureViaTargetProjector::lookup_owner_ids_spatial(
    const PureSpatialIndex* other_idx, int64_t epx, int64_t epy) {
    const OwnerCacheKey okey{reinterpret_cast<uintptr_t>(other_idx), epx, epy};
    auto it = owner_ids_cache.find(okey);
    if (it != owner_ids_cache.end()) return &it->second;

    std::vector<int32_t> ids;
    if (other_idx) {
        const Box64 q{epx, epy, epx, epy};
        other_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            ids.push_back(static_cast<int32_t>(other_idx->entries[item.entry_idx].owner_id));
        });
    }
    auto ins = owner_ids_cache.emplace(okey, std::move(ids));
    return &ins.first->second;
}

int PureViaTargetProjector::project_clipped_pure(
    int32_t ul, int32_t other, int64_t px, int64_t py,
    int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
    const std::vector<Box64>& clipped,
    bool with_owners,
    OwnerIdsLookupFn owner_fn,
    void* owner_ctx,
    NativeViaTargetCacheEntry& out_entry) const {
    out_entry.has_cross = false;
    out_entry.has_owners = with_owners;
    out_entry.recs.clear();
    out_entry.recs.reserve(clipped.size() * 2);

    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> ends_seen;
    ends_seen.reserve(clipped.size() * 2 + 4);
    ends_seen.emplace(px, py);

    std::unordered_set<ViaPathKey, ViaPathKeyHash> seen_paths;
    seen_paths.reserve(clipped.size() * 3 + 4);

    for (const Box64& box : clipped) {
        const int64_t lx = box.x0;
        const int64_t ly = box.y0;
        const int64_t hx = box.x1;
        const int64_t hy = box.y1;

        if (lx <= px && px < hx && ly <= py && py < hy) {
            out_entry.has_cross = true;
            continue;
        }
        const int64_t ex_g = (px < lx) ? lx : ((px >= hx) ? (hx - 1) : px);
        const int64_t ey_g = (py < ly) ? ly : ((py >= hy) ? (hy - 1) : py);
        if (!ends_seen.emplace(ex_g, ey_g).second) continue;

        if (!(rx0 <= px && px < rx1 && ry0 <= py && py < ry1)) {
            return -1;
        }
        const int64_t lo_x = std::max(rx0, lx);
        const int64_t lo_y = std::max(ry0, ly);
        const int64_t hi_x = std::min(rx1, hx);
        const int64_t hi_y = std::min(ry1, hy);
        if (lo_x >= hi_x || lo_y >= hi_y) continue;

        const int64_t end_x = std::min(std::max(px, lo_x), hi_x - 1);
        const int64_t end_y = std::min(std::max(py, lo_y), hi_y - 1);
        if (end_x == px && end_y == py) {
            out_entry.has_cross = true;
            continue;
        }

        ViaPathKey cand_paths[2];
        int n_paths = 0;
        if (end_x == px || end_y == py) {
            cand_paths[0] = ViaPathKey{2, px, py, end_x, end_y, 0, 0};
            n_paths = 1;
        } else {
            const ViaPathKey pA{3, px, py, end_x, py, end_x, end_y};
            const ViaPathKey pB{3, px, py, px, end_y, end_x, end_y};
            if (end_x < px) {
                cand_paths[0] = pA;
                cand_paths[1] = pB;
            } else {
                cand_paths[0] = pB;
                cand_paths[1] = pA;
            }
            n_paths = 2;
        }

        const int64_t ex = end_x * grid;
        const int64_t ey = end_y * grid;
        const int64_t ebx = floor_div_pos(ex, rect_bucket_nm);
        const int64_t eby = floor_div_pos(ey, rect_bucket_nm);
        const bool end_is_goal = is_goal_state(ex, ey, ul);

        for (int pi = 0; pi < n_paths; ++pi) {
            const ViaPathKey& pk = cand_paths[pi];
            if (!seen_paths.insert(pk).second) continue;

            int32_t goal_waypoint = -1;
            if (neg_bucket_dedup) {
                if (pk.len == 2) {
                    if (end_is_goal) goal_waypoint = 1;
                } else {
                    if (is_goal_state(pk.x1 * grid, pk.y1 * grid, ul)) goal_waypoint = 1;
                    else if (end_is_goal) goal_waypoint = 2;
                }
            } else {
                if (end_is_goal) goal_waypoint = static_cast<int32_t>(pk.len) - 1;
            }

            const bool is_end_goal = (goal_waypoint == static_cast<int32_t>(pk.len) - 1);
            const int64_t prev_y = (pk.len == 2) ? pk.y0 : pk.y1;
            const int last_vdir = (end_y == prev_y) ? 1 : 2;
            const int32_t other_ulast = (last_vdir == 1) ? via_succ_dir_1 : via_succ_dir_2;

            NativeViaTargetRec rec;
            rec.pk = pk;
            rec.ex = ex;
            rec.ey = ey;
            rec.p1_y_nm = pk.y1 * grid;
            rec.goal_waypoint = goal_waypoint;
            rec.is_end_goal = is_end_goal;
            rec.other_ulast = other_ulast;
            rec.ebx = ebx;
            rec.eby = eby;
            rec.extra_turn = (pk.len > 2) ? turn_cost : 0;

            if (goal_waypoint < 0 && with_owners && owner_fn != nullptr) {
                const auto* oids = owner_fn(owner_ctx, end_x, end_y);
                if (!oids) return -2;
                rec.owners = *oids;
            }
            if (goal_waypoint >= 0 && !is_end_goal) {
                rec.has_tail_lb = false;
                rec.tail_lb = 0;
            } else if (is_end_goal) {
                rec.has_tail_lb = true;
                rec.tail_lb = hmin_core ? hmin_core->eval(ex, ey, ul) : 0;
            } else {
                rec.has_tail_lb = true;
                rec.tail_lb = via_cost + (hmin_core ? hmin_core->eval(ex, ey, other) : 0);
            }
            out_entry.recs.push_back(std::move(rec));
        }
    }
    return 1;
}

int PureViaTargetProjector::enqueue_via_targets_pure(
    int32_t ul, int32_t other, int32_t rect_id, int64_t px, int64_t py,
    int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
    const std::vector<Box64>* clipped_or_null,
    bool with_owners,
    OwnerIdsLookupFn owner_fn,
    void* owner_ctx,
    uint32_t candidate_u_idx,
    int64_t u_du, int32_t ulast, int64_t u_rate, int64_t base_lb,
    bool has_cutoff, int64_t cutoff,
    bool has_pattern_cutoff, int64_t pattern_cutoff,
    int64_t other_cap,
    bool event_budget_stopped,
    bool& out_u_used) {
    out_u_used = false;
    const ViaTargetCacheKey ckey{ul, other, rect_id, px, py, static_cast<uint8_t>(with_owners ? 1 : 0)};
    const NativeViaTargetCacheEntry* entry = nullptr;
    auto it_c = native_target_cache.find(ckey);
    if (it_c != native_target_cache.end()) {
        entry = it_c->second;
    } else {
        auto* fresh = new NativeViaTargetCacheEntry();
        const std::vector<Box64> empty_clipped;
        const auto& clipped = clipped_or_null ? *clipped_or_null : empty_clipped;
        const int pst = project_clipped_pure(
            ul, other, px, py, rx0, ry0, rx1, ry1, clipped,
            with_owners, owner_fn, owner_ctx, *fresh);
        if (pst < 0) {
            delete fresh;
            return pst;
        }
        if (!event_budget_stopped) {
            owned_cache_entries.push_back(fresh);
            native_target_cache.emplace(ckey, fresh);
        } else {
            uncached_frontier_entries.push_back(fresh);
        }
        entry = fresh;
    }

    const int64_t ux = px * grid;
    const int64_t uy = py * grid;
    const int32_t r_dest = -other - 2;
    bool sat_pruned = false;

    const size_t n_recs = entry->recs.size();
    for (size_t ri = 0; ri < n_recs; ++ri) {
        const NativeViaTargetRec& rec = entry->recs[ri];
        if (rec.goal_waypoint < 0 && other_cap >= 0) {
            bool any_open = false;
            for (int32_t owner_id : rec.owners) {
                const RectEntryKey rk{other, owner_id, rec.other_ulast, rec.ebx, rec.eby};
                auto it_r = rect_entry_counts.find(rk);
                const int32_t cnt = (it_r != rect_entry_counts.end()) ? it_r->second : 0;
                if (cnt < other_cap) {
                    any_open = true;
                    break;
                }
            }
            if (!any_open) {
                if (!rec.owners.empty()) sat_pruned = true;
                continue;
            }
        }
        int64_t lb_f = 0;
        if (!rec.has_tail_lb) {
            const int64_t gx_nm = rec.pk.x1 * grid;
            const int64_t gy_nm = rec.pk.y1 * grid;
            const int first_vdir = (rec.p1_y_nm == uy) ? 1 : 2;
            const int64_t goal_turn = (ulast != 0 && first_vdir != ulast) ? turn_cost : 0;
            int64_t dx = gx_nm - ux; if (dx < 0) dx = -dx;
            int64_t dy = gy_nm - uy; if (dy < 0) dy = -dy;
            lb_f = base_lb + (dx + dy) * u_rate + goal_turn;
        } else {
            const int first_vdir = (rec.p1_y_nm == uy) ? 1 : 2;
            const int64_t turn1 = (ulast != 0 && first_vdir != ulast) ? turn_cost : 0;
            int64_t dx = rec.ex - ux; if (dx < 0) dx = -dx;
            int64_t dy = rec.ey - uy; if (dy < 0) dy = -dy;
            lb_f = base_lb + (dx + dy) * u_rate + turn1 + rec.extra_turn + rec.tail_lb;
        }
        if (has_cutoff && lb_f > cutoff) continue;
        if (has_pattern_cutoff && lb_f >= pattern_cutoff) continue;
        out_u_used = true;
        frontier_seq += 1;
        frontier_heap.push_back(RegionFrontierEntry{
            lb_f, frontier_seq, candidate_u_idx, u_du, entry,
            static_cast<uint32_t>(ri), r_dest
        });
        std::push_heap(frontier_heap.begin(), frontier_heap.end(), RegionFrontierMinCmp{});
    }

    return (entry->has_cross ? 1 : 0) | (sat_pruned ? 2 : 0);
}

bool PureViaTargetProjector::check_frontier_entry_open(
    const RegionFrontierEntry& e, int32_t r_other, int64_t other_cap,
    bool& out_sat_pruned) const noexcept {
    out_sat_pruned = false;
    const NativeViaTargetRec& rec = e.cache_entry->recs[e.rec_idx];
    if (other_cap < 0 || rec.goal_waypoint >= 0) return true;
    bool any_open = false;
    for (int32_t owner_id : rec.owners) {
        const RectEntryKey rk{r_other, owner_id, rec.other_ulast, rec.ebx, rec.eby};
        auto it_r = rect_entry_counts.find(rk);
        const int32_t cnt = (it_r != rect_entry_counts.end()) ? it_r->second : 0;
        if (cnt >= other_cap) {
            out_sat_pruned = true;
        } else {
            any_open = true;
            break;
        }
    }
    return any_open;
}

// ---------------------------------------------------------------------------
// 5. Pure C++20 Region-Event Step Advancement (Phase 1B)
// ---------------------------------------------------------------------------

RegionStepResult advance_region_step_pure(
    PureEdgeContext& ctx,
    PureAxisPriceCache& acache,
    const RegionStepState& u,
    int64_t p1x, int64_t p1y,
    int64_t rest_len,
    int64_t next_rect,
    int64_t grid,
    int64_t expansions,
    int64_t budget,
    int64_t event_work,
    int64_t event_budget,
    int64_t rem_solve,
    int64_t& sp_steps,
    int64_t& pc_hits,
    size_t max_rc,
    const RegionStepHooks& hooks) {
    RegionStepResult res{};
    if (expansions >= budget || expansions + event_work >= event_budget || rem_solve <= 0) {
        res.status = 0;
        res.budget_stopped = true;
        return res;
    }
    res.ev_charged = 1;
    rem_solve -= 1;

    if (((event_work + 1) & 255) == 1 && hooks.watchdog) {
        if (!hooks.watchdog(hooks.ctx, 2)) {
            res.status = -1;
            return res;
        }
    }

    const int64_t vx = p1x * grid;
    const int64_t vy = p1y * grid;
    const int64_t vdir = (u.uy == vy) ? 1 : 2;

    int64_t dummy_pres = 0;
    const int wst = ctx.check_wire_legal(static_cast<int32_t>(u.ul), u.ux, u.uy, vx, vy, dummy_pres);
    if (wst == 0) {
        res.status = 0;
        return res;
    }
    if (wst < 0) {
        bool is_leg = false;
        if (!hooks.fallback_wire) {
            res.status = 0;
            return res;
        }
        if (!hooks.fallback_wire(hooks.ctx, u.ul, u.ux, u.uy, vx, vy, is_leg)) {
            res.status = -1;
            return res;
        }
        if (!is_leg) {
            res.status = 0;
            return res;
        }
    }

    const int64_t signed_dir = (vdir == 1) ? ((vx > u.ux) ? 1 : -1) : ((vy > u.uy) ? 2 : -2);
    int64_t static_cost = 0;
    int64_t start_x = u.run_x;
    int64_t start_y = u.run_y;

    AxisPriceHooks ahooks{hooks.ctx, hooks.watchdog, hooks.fallback_price};

    if (u.run_dir == 0) {
        start_x = u.ux;
        start_y = u.uy;
    } else if (signed_dir != u.run_dir) {
        int64_t c = 0;
        const int pst = run_axis_price_ctx_pure(
            acache, ctx, u.ul, u.run_x, u.run_y, u.ux, u.uy,
            rem_solve, res.price_charged, sp_steps, pc_hits, max_rc,
            c, res.budget_stopped, ahooks);
        if (pst < 0) {
            res.status = pst;
            return res;
        }
        if (pst == 0) {
            res.status = 0;
            return res;
        }
        static_cost = c;
        start_x = u.ux;
        start_y = u.uy;
    }

    int64_t bnd_x = 0, bnd_y = 0;
    if (start_x != vx && start_y != vy) {
        res.status = -2;
        return res;
    }
    if (settled_axis_prefix_core(start_x, start_y, vx, vy, acache.xs, acache.ys, bnd_x, bnd_y)) {
        int64_t pc = 0;
        const int pst = run_axis_price_ctx_pure(
            acache, ctx, u.ul, start_x, start_y, bnd_x, bnd_y,
            rem_solve, res.price_charged, sp_steps, pc_hits, max_rc,
            pc, res.budget_stopped, ahooks);
        if (pst < 0) {
            res.status = pst;
            return res;
        }
        if (pst == 0) {
            res.status = 0;
            return res;
        }
        static_cost += pc;
        start_x = bnd_x;
        start_y = bnd_y;
    }

    const int64_t turn = (u.ulast != 0 && vdir != u.ulast) ? ctx.turn_cost : 0;
    res.status = 1;
    res.vx = vx;
    res.vy = vy;
    res.vdir = vdir;
    res.out_rect = (rest_len == 0 && next_rect >= 0) ? next_rect : u.rect;
    res.start_x = start_x;
    res.start_y = start_y;
    res.signed_dir = signed_dir;
    res.step_cost = static_cost + turn;
    return res;
}

}  // namespace clay_pnr::route

