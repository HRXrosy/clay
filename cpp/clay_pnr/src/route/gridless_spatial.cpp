// Pure C++20 Gridless spatial index & exact gap metallisation implementation.
// Strictly free of <Python.h>.

#include "clay_pnr/route/gridless_spatial.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace clay_pnr::route {

uint32_t PureSpatialIndex::add_entry(const Box64& norm_box, uint32_t owner_id,
                                     uint64_t rect_tag, uint64_t owner_tag) {
    const ClaimKey ck{owner_id, static_cast<uintptr_t>(rect_tag)};
    uint32_t claim_idx = 0;
    const auto cit = claim_map.find(ck);
    if (cit != claim_map.end()) {
        claim_idx = cit->second;
        has_dup_claims = true;
    } else {
        claim_idx = next_claim_idx++;
        claim_map.emplace(ck, claim_idx);
    }

    const uint32_t entry_idx = static_cast<uint32_t>(entries.size());
    entries.push_back(PureSpatialEntry{norm_box, owner_id, claim_idx, rect_tag, owner_tag});

    const int64_t bx0 = floor_div_pos(norm_box.x0, cell);
    const int64_t bx1 = floor_div_pos(norm_box.x1, cell);
    const int64_t by0 = floor_div_pos(norm_box.y0, cell);
    const int64_t by1 = floor_div_pos(norm_box.y1, cell);

    if (!has_buckets) {
        has_buckets = true;
        min_bx = bx0;
        max_bx = bx1;
        min_by = by0;
        max_by = by1;
    } else {
        if (bx0 < min_bx) min_bx = bx0;
        if (bx1 > max_bx) max_bx = bx1;
        if (by0 < min_by) min_by = by0;
        if (by1 > max_by) max_by = by1;
    }

    const BucketItem bi{
        norm_box.x0, norm_box.y0, norm_box.x1, norm_box.y1,
        owner_id, claim_idx, entry_idx, 0
    };
    for (int64_t bx = bx0; bx <= bx1; ++bx) {
        for (int64_t by = by0; by <= by1; ++by) {
            const BucketKey bk{bx, by};
            const auto it = bucket_map.find(bk);
            if (it == bucket_map.end()) {
                const uint32_t idx = static_cast<uint32_t>(buckets.size());
                bucket_map.emplace(bk, idx);
                buckets.push_back(BucketSlot{bx, by, {}});
                buckets.back().items.push_back(bi);
            } else {
                buckets[it->second].items.push_back(bi);
            }
        }
    }

    compiled = false;
    return entry_idx;
}

void PureSpatialIndex::ensure_compiled() const {
    if (compiled) return;
    const size_t nb = buckets.size();
    flat_spans.resize(nb);
    size_t total = 0;
    for (size_t i = 0; i < nb; ++i) {
        total += buckets[i].items.size();
    }
    flat_items.clear();
    flat_items.reserve(total);
    for (size_t i = 0; i < nb; ++i) {
        const uint32_t off = static_cast<uint32_t>(flat_items.size());
        const auto& src = buckets[i].items;
        flat_items.insert(flat_items.end(), src.begin(), src.end());
        flat_spans[i] = FlatSpan{off, static_cast<uint32_t>(src.size())};
    }

    use_dense = false;
    dense_grid.clear();
    if (has_buckets) {
        const int64_t nx = max_bx - min_bx + 1;
        const int64_t ny = max_by - min_by + 1;
        if (nx > 0 && ny > 0 && nx <= 65536 && ny <= 65536 && (nx * ny) <= 65536) {
            use_dense = true;
            dense_nx = nx;
            dense_ny = ny;
            dense_grid.assign(static_cast<size_t>(nx * ny), -1);
            for (size_t i = 0; i < nb; ++i) {
                const int64_t gx = buckets[i].bx - min_bx;
                const int64_t gy = buckets[i].by - min_by;
                dense_grid[static_cast<size_t>(gx * ny + gy)] = static_cast<int32_t>(i);
            }
        }
    }

    if (seen_gen.size() < next_claim_idx) {
        seen_gen.resize(next_claim_idx, 0);
    }
    compiled = true;
}

PureSpatialIndex PureSpatialIndex::clone_geometry() const {
    PureSpatialIndex dst(cell);
    dst.entries = entries;
    dst.buckets = buckets;
    dst.bucket_map = bucket_map;
    dst.claim_map = claim_map;
    dst.next_claim_idx = next_claim_idx;
    dst.has_dup_claims = has_dup_claims;
    dst.has_buckets = has_buckets;
    dst.min_bx = min_bx;
    dst.max_bx = max_bx;
    dst.min_by = min_by;
    dst.max_by = max_by;
    dst.compiled = false;
    return dst;
}

bool PureSpatialIndex::has_foreign_intersecting(const Box64& q, uint32_t own_id) const noexcept {
    if (!has_buckets) return false;
    ensure_compiled();
    int64_t bx0 = floor_div_pos(q.x0, cell);
    int64_t bx1 = floor_div_pos(q.x1, cell);
    int64_t by0 = floor_div_pos(q.y0, cell);
    int64_t by1 = floor_div_pos(q.y1, cell);
    if (bx1 < min_bx || bx0 > max_bx || by1 < min_by || by0 > max_by) {
        return false;
    }
    if (bx0 < min_bx) bx0 = min_bx;
    if (bx1 > max_bx) bx1 = max_bx;
    if (by0 < min_by) by0 = min_by;
    if (by1 > max_by) by1 = max_by;

    const BucketItem* items_ptr = flat_items.data();
    for (int64_t bx = bx0; bx <= bx1; ++bx) {
        for (int64_t by = by0; by <= by1; ++by) {
            const int32_t bidx = find_bucket(bx, by);
            if (bidx < 0) continue;
            const FlatSpan span = flat_spans[static_cast<size_t>(bidx)];
            const BucketItem* p = items_ptr + span.offset;
            const BucketItem* end = p + span.count;
            for (; p < end; ++p) {
                if (p->owner_id == own_id) continue;
                if (p->x1 < q.x0 || q.x1 < p->x0 || p->y1 < q.y0 || q.y1 < p->y0) continue;
                return true;
            }
        }
    }
    return false;
}

bool PureSpatialIndex::has_foreign_linf_conflict(const Box64& box, int64_t sp, uint32_t own_id) const noexcept {
    if (!has_buckets) return false;
    ensure_compiled();
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    int64_t bx0 = floor_div_pos(q.x0, cell);
    int64_t bx1 = floor_div_pos(q.x1, cell);
    int64_t by0 = floor_div_pos(q.y0, cell);
    int64_t by1 = floor_div_pos(q.y1, cell);
    if (bx1 < min_bx || bx0 > max_bx || by1 < min_by || by0 > max_by) {
        return false;
    }
    if (bx0 < min_bx) bx0 = min_bx;
    if (bx1 > max_bx) bx1 = max_bx;
    if (by0 < min_by) by0 = min_by;
    if (by1 > max_by) by1 = max_by;

    const BucketItem* items_ptr = flat_items.data();
    for (int64_t bx = bx0; bx <= bx1; ++bx) {
        for (int64_t by = by0; by <= by1; ++by) {
            const int32_t bidx = find_bucket(bx, by);
            if (bidx < 0) continue;
            const FlatSpan span = flat_spans[static_cast<size_t>(bidx)];
            const BucketItem* p = items_ptr + span.offset;
            const BucketItem* end = p + span.count;
            for (; p < end; ++p) {
                if (p->owner_id == own_id) continue;
                if (p->x1 < q.x0 || q.x1 < p->x0 || p->y1 < q.y0 || q.y1 < p->y0) continue;
                if (!rects_clear_linf(box, Box64{p->x0, p->y0, p->x1, p->y1}, sp)) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool PureSpatialIndex::has_any_intersecting(const Box64& q) const noexcept {
    if (!has_buckets) return false;
    ensure_compiled();
    int64_t bx0 = floor_div_pos(q.x0, cell);
    int64_t bx1 = floor_div_pos(q.x1, cell);
    int64_t by0 = floor_div_pos(q.y0, cell);
    int64_t by1 = floor_div_pos(q.y1, cell);
    if (bx1 < min_bx || bx0 > max_bx || by1 < min_by || by0 > max_by) {
        return false;
    }
    if (bx0 < min_bx) bx0 = min_bx;
    if (bx1 > max_bx) bx1 = max_bx;
    if (by0 < min_by) by0 = min_by;
    if (by1 > max_by) by1 = max_by;

    const BucketItem* items_ptr = flat_items.data();
    for (int64_t bx = bx0; bx <= bx1; ++bx) {
        for (int64_t by = by0; by <= by1; ++by) {
            const int32_t bidx = find_bucket(bx, by);
            if (bidx < 0) continue;
            const FlatSpan span = flat_spans[static_cast<size_t>(bidx)];
            const BucketItem* p = items_ptr + span.offset;
            const BucketItem* end = p + span.count;
            for (; p < end; ++p) {
                if (p->x1 < q.x0 || q.x1 < p->x0 || p->y1 < q.y0 || q.y1 < p->y0) continue;
                return true;
            }
        }
    }
    return false;
}

bool raw_gap_metallised_pure(
    const Box64& candidate,
    const Box64& blocker,
    int64_t sx,
    int64_t sy,
    const std::vector<Box64>& attempt_metal,
    const PureSpatialIndex* obs_idx,
    const std::unordered_set<Box64, Box64Hash>& abstract_set,
    const PureSpatialIndex* metal_idx,
    std::vector<Box64>& cover_buf) {
    const Box64 gap = compute_gap_box(candidate, blocker, sx, sy);
    if (gap.x1 <= gap.x0 || gap.y1 <= gap.y0) return false;

    cover_buf.clear();
    if (candidate.x0 < gap.x1 && gap.x0 < candidate.x1 &&
        candidate.y0 < gap.y1 && gap.y0 < candidate.y1) {
        cover_buf.push_back(candidate);
    }
    for (const auto& rr : attempt_metal) {
        if (rr.x0 < gap.x1 && gap.x0 < rr.x1 && rr.y0 < gap.y1 && gap.y0 < rr.y1) {
            if (rr.x0 <= gap.x0 && rr.y0 <= gap.y0 && rr.x1 >= gap.x1 && rr.y1 >= gap.y1) {
                return true;
            }
            cover_buf.push_back(rr);
        }
    }
    bool quick_covered = false;
    if (obs_idx) {
        obs_idx->for_each_intersecting(gap, [&](const BucketItem& item) {
            if (quick_covered) return;
            const Box64 rr{item.x0, item.y0, item.x1, item.y1};
            if (!abstract_set.empty() && abstract_set.find(rr) != abstract_set.end()) {
                return;
            }
            if (rr.x0 < gap.x1 && gap.x0 < rr.x1 && rr.y0 < gap.y1 && gap.y0 < rr.y1) {
                if (rr.x0 <= gap.x0 && rr.y0 <= gap.y0 && rr.x1 >= gap.x1 && rr.y1 >= gap.y1) {
                    quick_covered = true;
                    return;
                }
                cover_buf.push_back(rr);
            }
        });
        if (quick_covered) return true;
    }
    if (metal_idx) {
        metal_idx->for_each_intersecting(gap, [&](const BucketItem& item) {
            if (quick_covered) return;
            const Box64 rr{item.x0, item.y0, item.x1, item.y1};
            if (rr.x0 < gap.x1 && gap.x0 < rr.x1 && rr.y0 < gap.y1 && gap.y0 < rr.y1) {
                if (rr.x0 <= gap.x0 && rr.y0 <= gap.y0 && rr.x1 >= gap.x1 && rr.y1 >= gap.y1) {
                    quick_covered = true;
                    return;
                }
                cover_buf.push_back(rr);
            }
        });
        if (quick_covered) return true;
    }
    if (cover_buf.empty()) return false;
    return box_fully_covered_by_rects(gap, cover_buf);
}

// ---------------------------------------------------------------------------
// Pure C++20 Specialized Spatial Scans
// ---------------------------------------------------------------------------

std::vector<PureSpatialIndex::MetalConflictHit> PureSpatialIndex::scan_metal_conflicts_pure(
    const Box64& box, int64_t space_nm, uint32_t own_id, uint32_t excl_id) const {
    const int64_t grow = (space_nm > 0) ? space_nm : 0;
    const Box64 q{box.x0 - grow, box.y0 - grow, box.x1 + grow, box.y1 + grow};
    std::vector<MetalConflictHit> hits;
    for_each_intersecting(q, [&](const BucketItem& item) {
        if (item.owner_id == own_id || item.owner_id == excl_id) return;
        const Box64 b{item.x0, item.y0, item.x1, item.y1};
        const bool touch = closed_rects_touch(box, b);
        if (touch || !rects_clear_euclid_exact(box, b, space_nm)) {
            hits.push_back(MetalConflictHit{item.entry_idx, touch});
        }
    });
    return hits;
}

PureSpatialIndex::CommittedMetalScanResult PureSpatialIndex::scan_committed_metal_pure(
    const Box64& box, int64_t sp, int64_t rc, int64_t pres, uint32_t own_id) const {
    CommittedMetalScanResult out;
    const int64_t half = floor_div_pos(sp, 2);
    const Box64 ebx{box.x0 - half, box.y0 - half, box.x1 + half, box.y1 + half};
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    for_each_intersecting(q, [&](const BucketItem& item) {
        if (item.owner_id == own_id) return;
        const Box64 blocker{item.x0, item.y0, item.x1, item.y1};
        int64_t sx = 0, sy = 0;
        box_sep(box, blocker, sx, sy);
        if (sx == 0 && sy == 0) {
            out.hits.push_back(MetalConflictHit{item.entry_idx, true});
        } else if (sp > 0 && sx < sp && sy < sp &&
                   euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                                    static_cast<uint64_t>(sp))) {
            out.hits.push_back(MetalConflictHit{item.entry_idx, false});
        }
        const int64_t ox0 = std::max(ebx.x0, blocker.x0 - half);
        const int64_t oy0 = std::max(ebx.y0, blocker.y0 - half);
        const int64_t ox1 = std::min(ebx.x1, blocker.x1 + half);
        const int64_t oy1 = std::min(ebx.y1, blocker.y1 + half);
        if (ox0 >= ox1 || oy0 >= oy1) return;
        const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
        const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
        out.present_total += pres * ncx * ncy;
    });
    return out;
}

std::vector<uint32_t> PureSpatialIndex::scan_obstacle_conflicts_pure(
    const Box64& box, int64_t sp, uint32_t own_id) const {
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    std::vector<uint32_t> hits;
    for_each_intersecting(q, [&](const BucketItem& item) {
        if (item.owner_id == own_id) return;
        const Box64 b{item.x0, item.y0, item.x1, item.y1};
        if (!rects_clear_linf(box, b, sp)) {
            hits.push_back(item.entry_idx);
        }
    });
    return hits;
}

PureSpatialIndex::CutConflictScanResult PureSpatialIndex::scan_cut_conflicts_pure(
    const Box64& cut, int64_t vsp, uint32_t own_id) const {
    CutConflictScanResult out;
    const Box64 q{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
    for_each_intersecting(q, [&](const BucketItem& item) {
        const Box64 other{item.x0, item.y0, item.x1, item.y1};
        if (item.owner_id == own_id) {
            if (other == cut) {
                ++out.present_same;
            } else if (!rects_clear_linf(cut, other, vsp)) {
                out.spacing_violations.push_back(CutSortItem{item.entry_idx, item.owner_id, other});
            }
            return;
        }
        if (closed_rects_touch(cut, other)) {
            out.foreign_contacts.push_back(CutSortItem{item.entry_idx, item.owner_id, other});
        } else if (!rects_clear_linf(cut, other, vsp)) {
            out.spacing_violations.push_back(CutSortItem{item.entry_idx, item.owner_id, other});
        }
    });
    return out;
}

PureSpatialIndex::CommittedCutScanResult PureSpatialIndex::scan_committed_cuts_pure(
    const Box64& cut, int64_t vsp, int64_t rc, int64_t pres, uint32_t own_id) const {
    CommittedCutScanResult out;
    const int64_t half = floor_div_pos(vsp, 2);
    const Box64 ebx{cut.x0 - half, cut.y0 - half, cut.x1 + half, cut.y1 + half};
    const Box64 q{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
    for_each_intersecting(q, [&](const BucketItem& item) {
        const Box64 other{item.x0, item.y0, item.x1, item.y1};
        if (item.owner_id == own_id) {
            if (!(other == cut) && !rects_clear_linf(cut, other, vsp)) {
                out.spacing_violations.push_back(CutSortItem{item.entry_idx, item.owner_id, other});
            }
            return;
        }
        if (closed_rects_touch(cut, other)) {
            out.foreign_contacts.push_back(CutSortItem{item.entry_idx, item.owner_id, other});
        } else if (!rects_clear_linf(cut, other, vsp)) {
            out.spacing_violations.push_back(CutSortItem{item.entry_idx, item.owner_id, other});
        }
        const int64_t ix0 = std::max(ebx.x0, other.x0 - half);
        const int64_t iy0 = std::max(ebx.y0, other.y0 - half);
        const int64_t ix1 = std::min(ebx.x1, other.x1 + half);
        const int64_t iy1 = std::min(ebx.y1, other.y1 + half);
        if (ix0 >= ix1 || iy0 >= iy1) return;
        out.present_total += pres * std::max<int64_t>(1, (ix1 - ix0) / rc)
                                  * std::max<int64_t>(1, (iy1 - iy0) / rc);
    });
    return out;
}

std::vector<PureSpatialIndex::FixedCutHit> PureSpatialIndex::scan_fixed_cut_conflicts_pure(
    const Box64& cut, int64_t vsp) const {
    const Box64 q{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
    std::vector<FixedCutHit> hits;
    for_each_intersecting(q, [&](const BucketItem& item) {
        const Box64 vb{item.x0, item.y0, item.x1, item.y1};
        if (vb == cut) return;
        if (!rects_clear_linf(cut, vb, vsp)) {
            hits.push_back(FixedCutHit{vb, item.entry_idx});
        }
    });
    if (hits.size() > 1) {
        std::sort(hits.begin(), hits.end(), [](const FixedCutHit& a, const FixedCutHit& b) {
            return a.box < b.box;
        });
    }
    return hits;
}

int64_t PureSpatialIndex::scan_same_net_band_count_pure(
    const Box64& box, int64_t sp, uint32_t target_id, bool filter_by_owner) const {
    if (filter_by_owner && target_id == UINT32_MAX) return 0;
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    int64_t count = 0;
    for_each_intersecting(q, [&](const BucketItem& item) {
        if (filter_by_owner && item.owner_id != target_id) return;
        const Box64 b{item.x0, item.y0, item.x1, item.y1};
        int64_t sx = 0, sy = 0;
        if (euclid_in_open_band(box, b, sp, sx, sy)) {
            ++count;
        }
    });
    return count;
}

std::vector<PureSpatialIndex::SameNetBandHit> PureSpatialIndex::scan_same_net_band_rects_pure(
    const Box64& box, int64_t sp, uint32_t target_id) const {
    std::vector<SameNetBandHit> hits;
    if (target_id == UINT32_MAX) return hits;
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    for_each_intersecting(q, [&](const BucketItem& item) {
        if (item.owner_id != target_id) return;
        const Box64 b{item.x0, item.y0, item.x1, item.y1};
        int64_t sx = 0, sy = 0;
        if (euclid_in_open_band(box, b, sp, sx, sy)) {
            hits.push_back(SameNetBandHit{item.entry_idx, sx, sy});
        }
    });
    return hits;
}

// ---------------------------------------------------------------------------
// Pure C++20 Batch Via-Site Filter
// ---------------------------------------------------------------------------

ViaSiteFilterResult batch_filter_via_sites_pure(const ViaSiteFilterRequest& req) {
    ViaSiteFilterResult out;
    const int64_t grid = std::max<int64_t>(1, req.grid);
    const int64_t local_extent = std::max({req.bh, req.ah, req.ext});

    static const std::vector<Box64> kEmptyBoxes;
    static const std::unordered_set<Box64, Box64Hash> kEmptyBoxSet;
    const auto& fixed_cuts = req.fixed_cuts ? *req.fixed_cuts : kEmptyBoxes;
    const auto& attempt_cuts = req.attempt_cuts ? *req.attempt_cuts : kEmptyBoxes;
    const auto& attempt_metal_lo = req.attempt_metal_lo ? *req.attempt_metal_lo : kEmptyBoxes;
    const auto& attempt_metal_hi = req.attempt_metal_hi ? *req.attempt_metal_hi : kEmptyBoxes;
    const auto& abstract_lo_set = req.abstract_lo_set ? *req.abstract_lo_set : kEmptyBoxSet;
    const auto& abstract_hi_set = req.abstract_hi_set ? *req.abstract_hi_set : kEmptyBoxSet;
    const auto& fh_lo = req.fh_lo ? *req.fh_lo : kEmptyBoxes;
    const auto& fh_hi = req.fh_hi ? *req.fh_hi : kEmptyBoxes;
    const auto& fh_cut = req.fh_cut ? *req.fh_cut : kEmptyBoxes;

    struct SiteRejRec {
        PureAggRejKey key;
        Box64 candidate_rect;
        uint64_t owner_tag;
    };
    struct SameNetHit {
        Box64 blocker;
        uint32_t owner_id;
        uint64_t owner_tag;
        int64_t sx;
        int64_t sy;
    };
    std::vector<SiteRejRec> site_rejs;
    site_rejs.reserve(16);
    std::vector<Box64> cover_buf;
    cover_buf.reserve(16);
    std::vector<SameNetHit> same_net_hits;
    same_net_hits.reserve(8);

    auto flush_site_rejs = [&]() {
        for (const auto& rec : site_rejs) {
            auto it = out.agg_rejs.find(rec.key);
            if (it == out.agg_rejs.end()) {
                out.agg_rejs.emplace(rec.key, PureAggRejVal{1, rec.candidate_rect, rec.owner_tag});
            } else {
                auto& val = it->second;
                val.count += 1;
                if (rec.candidate_rect.x0 < val.extent.x0) val.extent.x0 = rec.candidate_rect.x0;
                if (rec.candidate_rect.y0 < val.extent.y0) val.extent.y0 = rec.candidate_rect.y0;
                if (rec.candidate_rect.x1 > val.extent.x1) val.extent.x1 = rec.candidate_rect.x1;
                if (rec.candidate_rect.y1 > val.extent.y1) val.extent.y1 = rec.candidate_rect.y1;
            }
        }
    };

    auto eval_site = [&](int64_t sx_pt, int64_t sy_pt) {
        ++out.candidate_count;
        const Box64 pad_lo{sx_pt - req.bh, sy_pt - req.bh, sx_pt + req.bh, sy_pt + req.bh};
        if (req.has_atom) {
            if (!closed_rects_touch(pad_lo, req.atom_rect)) return;
            if (!owned_landing_has_min_throat(req.atom_rect, pad_lo, req.atom_min_width)) {
                return;
            }
        }

        const Box64 pad_hi{sx_pt - req.ah, sy_pt - req.ah, sx_pt + req.ah, sy_pt + req.ah};
        const Box64 cut{sx_pt - req.ext, sy_pt - req.ext, sx_pt + req.ext, sy_pt + req.ext};

        if ((!fh_lo.empty() && box_hits_any(pad_lo, fh_lo)) ||
            (!fh_hi.empty() && box_hits_any(pad_hi, fh_hi)) ||
            (!fh_cut.empty() && box_hits_any(cut, fh_cut))) {
            out.fallback_sites.emplace_back(sx_pt, sy_pt);
            return;
        }

        site_rejs.clear();

        if (req.forbidden_set && !req.forbidden_set->empty() &&
            req.forbidden_set->find({sx_pt, sy_pt}) != req.forbidden_set->end()) {
            site_rejs.push_back(SiteRejRec{
                PureAggRejKey{REJ_CONFLICT_BRANCH_VIA, req.lo, req.ext, -1, 0, 0, 0, Box64{0, 0, 0, 0}},
                cut, req.net_owner_tag
            });
            flush_site_rejs();
            return;
        }

        if ((sx_pt % grid) != 0 || (sy_pt % grid) != 0) {
            site_rejs.push_back(SiteRejRec{
                PureAggRejKey{REJ_MANUFACTURING_GRID, req.lo, req.bh, grid, 0, 0, 0, Box64{0, 0, 0, 0}},
                pad_lo, req.net_owner_tag
            });
            flush_site_rejs();
            return;
        }

        if (req.has_local_env) {
            const Box64 lcand{sx_pt - local_extent, sy_pt - local_extent,
                              sx_pt + local_extent, sy_pt + local_extent};
            if (lcand.x0 < req.local_env_box.x0 || lcand.y0 < req.local_env_box.y0 ||
                lcand.x1 > req.local_env_box.x1 || lcand.y1 > req.local_env_box.y1) {
                site_rejs.push_back(SiteRejRec{
                    PureAggRejKey{REJ_GROUP_ROUTE_LOCAL_ENVELOPE, req.lo, local_extent, -1, 1, 0, 0, req.local_env_box},
                    lcand, req.net_owner_tag
                });
                flush_site_rejs();
                return;
            }
        }

        bool static_legal = true;

        auto check_static_layer = [&](
            int32_t l_gi, const Box64& l_pad, int64_t l_half, int64_t l_sp,
            const PureSpatialIndex* l_oidx, uint32_t l_net_oid, uint8_t l_owner_src,
            const std::unordered_set<Box64, Box64Hash>& l_abs_set,
            const std::vector<Box64>& l_att_metal,
            const PureSpatialIndex* l_midx
        ) {
            if (req.has_env && (l_pad.x0 < req.env_box.x0 || l_pad.y0 < req.env_box.y0 ||
                                l_pad.x1 > req.env_box.x1 || l_pad.y1 > req.env_box.y1)) {
                site_rejs.push_back(SiteRejRec{
                    PureAggRejKey{REJ_ROUTING_ENVELOPE, l_gi, l_half, -1, 1, 0, 0, req.env_box},
                    l_pad, req.net_owner_tag
                });
                static_legal = false;
            }
            if (!l_oidx) return;
            const Box64 q{l_pad.x0 - l_sp, l_pad.y0 - l_sp, l_pad.x1 + l_sp, l_pad.y1 + l_sp};
            same_net_hits.clear();
            l_oidx->for_each_intersecting(q, [&](const BucketItem& item) {
                const Box64 b{item.x0, item.y0, item.x1, item.y1};
                if (item.owner_id != l_net_oid) {
                    if (!rects_clear_linf(l_pad, b, l_sp)) {
                        site_rejs.push_back(SiteRejRec{
                            PureAggRejKey{REJ_FIXED_METAL, l_gi, l_half, l_sp, 1, l_owner_src, item.owner_id, b},
                            l_pad, l_oidx->entries[item.entry_idx].owner_tag
                        });
                        static_legal = false;
                    }
                } else {
                    int64_t gap_sx = 0, gap_sy = 0;
                    if (euclid_in_open_band(l_pad, b, l_sp, gap_sx, gap_sy)) {
                        if (!l_abs_set.empty() && l_abs_set.find(b) != l_abs_set.end()) {
                            return;
                        }
                        same_net_hits.push_back(SameNetHit{
                            b, item.owner_id, l_oidx->entries[item.entry_idx].owner_tag, gap_sx, gap_sy
                        });
                    }
                }
            });
            for (const auto& h : same_net_hits) {
                if (raw_gap_metallised_pure(l_pad, h.blocker, h.sx, h.sy,
                                            l_att_metal, l_oidx, l_abs_set, l_midx, cover_buf)) {
                    continue;
                }
                site_rejs.push_back(SiteRejRec{
                    PureAggRejKey{REJ_FIXED_SAME_NET_VIA_PAD, l_gi, l_half, l_sp, 1, l_owner_src, h.owner_id, h.blocker},
                    l_pad, h.owner_tag
                });
                static_legal = false;
            }
        };

        check_static_layer(req.lo, pad_lo, req.bh, req.sp_lo, req.obs_lo, req.net_obs_lo, 1,
                           abstract_lo_set, attempt_metal_lo, req.metal_lo);
        check_static_layer(req.lo + 1, pad_hi, req.ah, req.sp_hi, req.obs_hi, req.net_obs_hi, 2,
                           abstract_hi_set, attempt_metal_hi, req.metal_hi);

        if (req.has_env && (cut.x0 < req.env_box.x0 || cut.y0 < req.env_box.y0 ||
                            cut.x1 > req.env_box.x1 || cut.y1 > req.env_box.y1)) {
            site_rejs.push_back(SiteRejRec{
                PureAggRejKey{REJ_ROUTING_ENVELOPE, req.lo, req.ext, -1, 1, 0, 0, req.env_box},
                cut, req.net_owner_tag
            });
            static_legal = false;
        }
        if (req.obs_cut_lo) {
            const Box64 fq{cut.x0 - req.vsp, cut.y0 - req.vsp, cut.x1 + req.vsp, cut.y1 + req.vsp};
            req.obs_cut_lo->for_each_intersecting(fq, [&](const BucketItem& item) {
                const Box64 vb{item.x0, item.y0, item.x1, item.y1};
                if (vb == cut) return;
                if (!rects_clear_linf(cut, vb, req.vsp)) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_FIXED_CUT, req.lo, req.ext, req.vsp, 1, 6, 0, vb},
                        cut, 0ULL
                    });
                    static_legal = false;
                }
            });
        } else if (!fixed_cuts.empty()) {
            for (const auto& vb : fixed_cuts) {
                if (vb == cut) continue;
                if (!rects_clear_linf(cut, vb, req.vsp)) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_FIXED_CUT, req.lo, req.ext, req.vsp, 1, 6, 0, vb},
                        cut, 0ULL
                    });
                    static_legal = false;
                }
            }
        }

        if (!static_legal) {
            flush_site_rejs();
            return;
        }

        bool dyn_legal = true;

        auto check_committed_layer = [&](
            int32_t l_gi, const Box64& l_pad, int64_t l_half, int64_t l_sp,
            const PureSpatialIndex* l_midx, uint32_t l_net_mid, uint8_t l_owner_src
        ) {
            if (!l_midx) return;
            const Box64 q{l_pad.x0 - l_sp, l_pad.y0 - l_sp, l_pad.x1 + l_sp, l_pad.y1 + l_sp};
            l_midx->for_each_intersecting(q, [&](const BucketItem& item) {
                if (item.owner_id == l_net_mid) return;
                const Box64 b{item.x0, item.y0, item.x1, item.y1};
                int64_t sep_x = 0, sep_y = 0;
                box_sep(l_pad, b, sep_x, sep_y);
                if (sep_x == 0 && sep_y == 0) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_COMMITTED_METAL_CONTACT, l_gi, l_half, 0, 1, l_owner_src, item.owner_id, b},
                        l_pad, l_midx->entries[item.entry_idx].owner_tag
                    });
                    dyn_legal = false;
                } else if (l_sp > 0 && sep_x < l_sp && sep_y < l_sp &&
                           euclid_sq_lt_128(static_cast<uint64_t>(sep_x), static_cast<uint64_t>(sep_y),
                                            static_cast<uint64_t>(l_sp))) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_COMMITTED_METAL_SPACING, l_gi, l_half, l_sp, 1, l_owner_src, item.owner_id, b},
                        l_pad, l_midx->entries[item.entry_idx].owner_tag
                    });
                    dyn_legal = false;
                }
            });
        };

        check_committed_layer(req.lo, pad_lo, req.bh, req.sp_lo, req.metal_lo, req.net_metal_lo, 3);
        check_committed_layer(req.lo + 1, pad_hi, req.ah, req.sp_hi, req.metal_hi, req.net_metal_hi, 4);

        if (req.cut_lo) {
            const Box64 cq{cut.x0 - req.vsp, cut.y0 - req.vsp, cut.x1 + req.vsp, cut.y1 + req.vsp};
            req.cut_lo->for_each_intersecting(cq, [&](const BucketItem& item) {
                const Box64 vb{item.x0, item.y0, item.x1, item.y1};
                if (item.owner_id == req.net_cut_lo) {
                    if (!(vb == cut) && !rects_clear_linf(cut, vb, req.vsp)) {
                        site_rejs.push_back(SiteRejRec{
                            PureAggRejKey{REJ_COMMITTED_SELF_CUT, req.lo, req.ext, req.vsp, 1, 5, item.owner_id, vb},
                            cut, req.cut_lo->entries[item.entry_idx].owner_tag
                        });
                        dyn_legal = false;
                    }
                    return;
                }
                if (closed_rects_touch(cut, vb)) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_COMMITTED_CUT_CONTACT, req.lo, req.ext, 0, 1, 5, item.owner_id, vb},
                        cut, req.cut_lo->entries[item.entry_idx].owner_tag
                    });
                    dyn_legal = false;
                } else if (!rects_clear_linf(cut, vb, req.vsp)) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_COMMITTED_CUT_SPACING, req.lo, req.ext, req.vsp, 1, 5, item.owner_id, vb},
                        cut, req.cut_lo->entries[item.entry_idx].owner_tag
                    });
                    dyn_legal = false;
                }
            });
        }

        if (!attempt_cuts.empty()) {
            for (const auto& vb : attempt_cuts) {
                if (vb == cut) continue;
                if (!rects_clear_linf(cut, vb, req.vsp)) {
                    site_rejs.push_back(SiteRejRec{
                        PureAggRejKey{REJ_ATTEMPT_CUT, req.lo, req.ext, req.vsp, 1, 0, 0, vb},
                        cut, req.net_owner_tag
                    });
                    dyn_legal = false;
                }
            }
        }

        if (!dyn_legal) {
            flush_site_rejs();
            return;
        }

        if (req.mirror_active) {
            out.fallback_sites.emplace_back(sx_pt, sy_pt);
        } else {
            out.legal_sites.emplace_back(sx_pt, sy_pt);
        }
    };

    if (req.use_explicit_sites) {
        if (req.explicit_sites) {
            for (const auto& p : *req.explicit_sites) eval_site(p.first, p.second);
        }
    } else {
        if (req.xs && req.ys) {
            for (int64_t x : *req.xs) {
                for (int64_t y : *req.ys) eval_site(x, y);
            }
            if (req.extra_pts && !req.extra_pts->empty()) {
                std::unordered_set<int64_t> xs_set(req.xs->begin(), req.xs->end());
                std::unordered_set<int64_t> ys_set(req.ys->begin(), req.ys->end());
                for (const auto& p : *req.extra_pts) {
                    if (xs_set.find(p.first) != xs_set.end() &&
                        ys_set.find(p.second) != ys_set.end()) {
                        continue;
                    }
                    eval_site(p.first, p.second);
                }
            }
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Pure C++20 Congestion & Fast Wire/Via Legality
// ---------------------------------------------------------------------------

int64_t eval_box_congestion_pure(
    const Box64& box, int64_t sp, int64_t rc, int64_t pres,
    const PureSpatialIndex* metal_idx, uint32_t net_metal_id,
    const PureSpatialIndex* obs_idx, uint32_t net_obs_id,
    const PureSpatialIndex* att_idx,
    const HistMap& hmap,
    int64_t mul_scale, int64_t cost_factor_scale,
    bool& overflow) {
    const int64_t half = floor_div_pos(sp, 2);
    const Box64 ebx{box.x0 - half, box.y0 - half, box.x1 + half, box.y1 + half};
    const Box64 q{box.x0 - sp, box.y0 - sp, box.x1 + sp, box.y1 + sp};
    int64_t total = 0;

    if (metal_idx) {
        metal_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id == net_metal_id) return;
            const int64_t ox0 = std::max(ebx.x0, item.x0 - half);
            const int64_t oy0 = std::max(ebx.y0, item.y0 - half);
            const int64_t ox1 = std::min(ebx.x1, item.x1 + half);
            const int64_t oy1 = std::min(ebx.y1, item.y1 + half);
            if (ox0 >= ox1 || oy0 >= oy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
            total += pres * ncx * ncy;
        });
    }
    if (obs_idx && net_obs_id != UINT32_MAX) {
        obs_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id != net_obs_id) return;
            int64_t sx = 0, sy = 0;
            if (euclid_in_open_band(box, Box64{item.x0, item.y0, item.x1, item.y1}, sp, sx, sy)) {
                total += pres;
            }
        });
    }
    if (att_idx) {
        att_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            int64_t sx = 0, sy = 0;
            if (euclid_in_open_band(box, Box64{item.x0, item.y0, item.x1, item.y1}, sp, sx, sy)) {
                total += pres;
            }
        });
    }
    total += eval_history_units_cpp(hmap, ebx, rc);
    return scale_congestion_units(total, mul_scale, cost_factor_scale, overflow);
}

int64_t eval_cut_congestion_pure(
    const Box64& cut, int64_t vsp, int64_t rc, int64_t pres,
    const PureSpatialIndex* cut_idx, uint32_t net_cut_id,
    const HistMap& hmap,
    int64_t mul_scale, int64_t cost_factor_scale,
    bool& overflow) {
    const int64_t half = floor_div_pos(vsp, 2);
    const Box64 ebx{cut.x0 - half, cut.y0 - half, cut.x1 + half, cut.y1 + half};
    const Box64 q{cut.x0 - vsp, cut.y0 - vsp, cut.x1 + vsp, cut.y1 + vsp};
    int64_t total = 0;

    if (cut_idx) {
        cut_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id == net_cut_id) return;
            const int64_t ix0 = std::max(ebx.x0, item.x0 - half);
            const int64_t iy0 = std::max(ebx.y0, item.y0 - half);
            const int64_t ix1 = std::min(ebx.x1, item.x1 + half);
            const int64_t iy1 = std::min(ebx.y1, item.y1 + half);
            if (ix0 >= ix1 || iy0 >= iy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ix1 - ix0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (iy1 - iy0) / rc);
            total += pres * ncx * ncy;
        });
    }
    total += eval_history_units_cpp(hmap, ebx, rc);
    return scale_congestion_units(total, mul_scale, cost_factor_scale, overflow);
}

std::vector<int64_t> batch_via_congestion_pure(const BatchViaCongestionRequest& req) {
    static const HistMap kEmptyHist;
    static const std::vector<Box64> kEmptyBoxes;
    const auto& hist_lo = req.hist_lo ? *req.hist_lo : kEmptyHist;
    const auto& hist_hi = req.hist_hi ? *req.hist_hi : kEmptyHist;
    const auto& hist_cut = req.hist_cut ? *req.hist_cut : kEmptyHist;
    const auto& wh_lo = req.wh_lo ? *req.wh_lo : kEmptyBoxes;
    const auto& wh_hi = req.wh_hi ? *req.wh_hi : kEmptyBoxes;
    const int64_t rc = std::max<int64_t>(1, req.rc);

    std::vector<int64_t> out;
    if (!req.sites) return out;
    out.resize(req.sites->size(), -1);
    for (size_t i = 0; i < req.sites->size(); ++i) {
        const int64_t x = (*req.sites)[i].first;
        const int64_t y = (*req.sites)[i].second;
        const Box64 pad_lo{x - req.bh, y - req.bh, x + req.bh, y + req.bh};
        const Box64 pad_hi{x - req.ah, y - req.ah, x + req.ah, y + req.ah};
        if ((wh_lo.empty() || !box_hits_any(pad_lo, wh_lo)) &&
            (wh_hi.empty() || !box_hits_any(pad_hi, wh_hi))) {
            const Box64 cut{x - req.ext, y - req.ext, x + req.ext, y + req.ext};
            bool ovf = false;
            const int64_t c_lo = eval_box_congestion_pure(
                pad_lo, req.sp_lo, rc, req.pres, req.metal_lo, req.net_metal_lo,
                req.obs_lo, req.net_obs_lo, req.att_lo, hist_lo, req.mul_scale, req.cost_scale, ovf);
            const int64_t c_hi = ovf ? -1 : eval_box_congestion_pure(
                pad_hi, req.sp_hi, rc, req.pres, req.metal_hi, req.net_metal_hi,
                req.obs_hi, req.net_obs_hi, req.att_hi, hist_hi, req.mul_scale, req.cost_scale, ovf);
            const int64_t c_cut = ovf ? -1 : eval_cut_congestion_pure(
                cut, req.vsp, rc, req.pres, req.cut_lo, req.net_cut_lo,
                hist_cut, req.mul_scale, req.cost_scale, ovf);
            if (!ovf) out[i] = c_lo + c_hi + c_cut;
        }
    }
    return out;
}

bool check_wire_legal_fast_pure(const WireLegalFastRequest& req, int64_t& present_total_out) {
    present_total_out = 0;
    const int64_t rc = std::max<int64_t>(1, req.rc);
    if (req.has_env) {
        if (req.wb.x0 < req.env.x0 || req.wb.y0 < req.env.y0 ||
            req.wb.x1 > req.env.x1 || req.wb.y1 > req.env.y1) {
            return false;
        }
    }
    if (req.obs_idx && req.obs_idx->has_foreign_linf_conflict(req.wb, req.sp, req.own_obs)) {
        return false;
    }
    if (req.wide_entries) {
        for (const auto& w : *req.wide_entries) {
            const Box64 wq{req.wb.x0 - w.w_sp, req.wb.y0 - w.w_sp,
                           req.wb.x1 + w.w_sp, req.wb.y1 + w.w_sp};
            if (w.core_idx && w.core_idx->has_foreign_intersecting(wq, w.own_core)) return false;
            if (w.att_idx && w.att_idx->has_foreign_intersecting(wq, w.own_att)) return false;
        }
    }
    if (req.metal_idx) {
        const int64_t half = floor_div_pos(req.sp, 2);
        const Box64 ebx{req.wb.x0 - half, req.wb.y0 - half, req.wb.x1 + half, req.wb.y1 + half};
        const Box64 q{req.wb.x0 - req.sp, req.wb.y0 - req.sp, req.wb.x1 + req.sp, req.wb.y1 + req.sp};
        bool conflict = false;
        req.metal_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (conflict || item.owner_id == req.own_metal) return;
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            int64_t sx = 0, sy = 0;
            box_sep(req.wb, b, sx, sy);
            if (sx == 0 && sy == 0) {
                conflict = true;
                return;
            }
            if (req.sp > 0 && sx < req.sp && sy < req.sp &&
                euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                                 static_cast<uint64_t>(req.sp))) {
                conflict = true;
                return;
            }
            const int64_t ox0 = std::max(ebx.x0, b.x0 - half);
            const int64_t oy0 = std::max(ebx.y0, b.y0 - half);
            const int64_t ox1 = std::min(ebx.x1, b.x1 + half);
            const int64_t oy1 = std::min(ebx.y1, b.y1 + half);
            if (ox0 >= ox1 || oy0 >= oy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
            present_total_out += req.pres * ncx * ncy;
        });
        if (conflict) return false;
    }
    return true;
}

int check_via_legal_fast_pure(const ViaLegalFastRequest& req,
                              int64_t& pres_lo, int64_t& pres_hi, int64_t& pres_cut) {
    pres_lo = 0;
    pres_hi = 0;
    pres_cut = 0;
    const int64_t grid = std::max<int64_t>(1, req.grid);
    const int64_t rc = std::max<int64_t>(1, req.rc);
    if ((req.x % grid) != 0 || (req.y % grid) != 0) return 0;

    const Box64 pad_lo{req.x - req.bh, req.y - req.bh, req.x + req.bh, req.y + req.bh};
    const Box64 pad_hi{req.x - req.ah, req.y - req.ah, req.x + req.ah, req.y + req.ah};
    const Box64 cut{req.x - req.ext, req.y - req.ext, req.x + req.ext, req.y + req.ext};

    if (req.has_env) {
        if (pad_lo.x0 < req.env.x0 || pad_lo.y0 < req.env.y0 || pad_lo.x1 > req.env.x1 || pad_lo.y1 > req.env.y1 ||
            pad_hi.x0 < req.env.x0 || pad_hi.y0 < req.env.y0 || pad_hi.x1 > req.env.x1 || pad_hi.y1 > req.env.y1 ||
            cut.x0 < req.env.x0 || cut.y0 < req.env.y0 || cut.x1 > req.env.x1 || cut.y1 > req.env.y1) {
            return 0;
        }
    }

    auto check_wide = [](const std::vector<FastWideCheck>* wv, const Box64& wbox) -> bool {
        if (!wv) return true;
        for (const auto& w : *wv) {
            const Box64 wq{wbox.x0 - w.w_sp, wbox.y0 - w.w_sp, wbox.x1 + w.w_sp, wbox.y1 + w.w_sp};
            if (w.core_idx && w.core_idx->has_foreign_intersecting(wq, w.own_core)) return false;
            if (w.att_idx && w.att_idx->has_foreign_intersecting(wq, w.own_att)) return false;
        }
        return true;
    };
    if (!check_wide(req.wide_lo, pad_lo)) return 0;
    if (!check_wide(req.wide_hi, pad_hi)) return 0;

    std::vector<Box64> cover_buf;
    auto check_obs_pad = [&](const PureSpatialIndex* oidx, uint32_t own, const Box64& pad, int64_t osp,
                             SameNetGapLoaderFn loader, void* lctx, const PureSpatialIndex* midx) -> int {
        if (!oidx) return 1;
        const Box64 q{pad.x0 - osp, pad.y0 - osp, pad.x1 + osp, pad.y1 + osp};
        bool conflict = false;
        struct SnHit { Box64 b; int64_t sx; int64_t sy; };
        std::vector<SnHit> sn_hits;
        oidx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (conflict) return;
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            if (item.owner_id != own) {
                if (!rects_clear_linf(pad, b, osp)) conflict = true;
            } else {
                int64_t sx = 0, sy = 0;
                if (euclid_in_open_band(pad, b, osp, sx, sy)) {
                    sn_hits.push_back(SnHit{b, sx, sy});
                }
            }
        });
        if (conflict) return 0;
        if (!sn_hits.empty()) {
            std::vector<Box64> att_m;
            std::unordered_set<Box64, Box64Hash> abs_set;
            if (loader && !loader(lctx, att_m, abs_set)) return -1;
            for (const auto& h : sn_hits) {
                if (!abs_set.empty() && abs_set.find(h.b) != abs_set.end()) continue;
                if (!raw_gap_metallised_pure(pad, h.b, h.sx, h.sy, att_m, oidx, abs_set, midx, cover_buf)) {
                    return 0;
                }
            }
        }
        return 1;
    };

    const int ol = check_obs_pad(req.obs_lo, req.own_obs_lo, pad_lo, req.sp_lo,
                                 req.load_gap_lo, req.gap_lo_ctx, req.metal_lo);
    if (ol <= 0) return ol;
    const int oh = check_obs_pad(req.obs_hi, req.own_obs_hi, pad_hi, req.sp_hi,
                                 req.load_gap_hi, req.gap_hi_ctx, req.metal_hi);
    if (oh <= 0) return oh;

    if (req.obs_cut_lo) {
        const Box64 fq{cut.x0 - req.vsp, cut.y0 - req.vsp, cut.x1 + req.vsp, cut.y1 + req.vsp};
        bool fc_conflict = false;
        req.obs_cut_lo->for_each_intersecting(fq, [&](const BucketItem& item) {
            if (fc_conflict) return;
            const Box64 vb{item.x0, item.y0, item.x1, item.y1};
            if (!(vb == cut) && !rects_clear_linf(cut, vb, req.vsp)) fc_conflict = true;
        });
        if (fc_conflict) return 0;
    }

    auto check_metal_pad = [&](const PureSpatialIndex* midx, uint32_t own,
                               const Box64& pad, int64_t msp, int64_t& pres_out) -> bool {
        pres_out = 0;
        if (!midx) return true;
        const int64_t half = floor_div_pos(msp, 2);
        const Box64 ebx{pad.x0 - half, pad.y0 - half, pad.x1 + half, pad.y1 + half};
        const Box64 q{pad.x0 - msp, pad.y0 - msp, pad.x1 + msp, pad.y1 + msp};
        bool conflict = false;
        midx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (conflict || item.owner_id == own) return;
            const Box64 b{item.x0, item.y0, item.x1, item.y1};
            int64_t sx = 0, sy = 0;
            box_sep(pad, b, sx, sy);
            if ((sx == 0 && sy == 0) ||
                (msp > 0 && sx < msp && sy < msp &&
                 euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                                  static_cast<uint64_t>(msp)))) {
                conflict = true;
                return;
            }
            const int64_t ox0 = std::max(ebx.x0, b.x0 - half);
            const int64_t oy0 = std::max(ebx.y0, b.y0 - half);
            const int64_t ox1 = std::min(ebx.x1, b.x1 + half);
            const int64_t oy1 = std::min(ebx.y1, b.y1 + half);
            if (ox0 >= ox1 || oy0 >= oy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
            pres_out += req.pres * ncx * ncy;
        });
        return !conflict;
    };
    if (!check_metal_pad(req.metal_lo, req.own_metal_lo, pad_lo, req.sp_lo, pres_lo)) return 0;
    if (!check_metal_pad(req.metal_hi, req.own_metal_hi, pad_hi, req.sp_hi, pres_hi)) return 0;

    if (req.cut_lo) {
        const int64_t half = floor_div_pos(req.vsp, 2);
        const Box64 ebx{cut.x0 - half, cut.y0 - half, cut.x1 + half, cut.y1 + half};
        const Box64 cq{cut.x0 - req.vsp, cut.y0 - req.vsp, cut.x1 + req.vsp, cut.y1 + req.vsp};
        bool conflict = false;
        req.cut_lo->for_each_intersecting(cq, [&](const BucketItem& item) {
            if (conflict) return;
            const Box64 vb{item.x0, item.y0, item.x1, item.y1};
            if (item.owner_id == req.own_cut_lo) {
                if (!(vb == cut) && !rects_clear_linf(cut, vb, req.vsp)) conflict = true;
                return;
            }
            if (closed_rects_touch(cut, vb) || !rects_clear_linf(cut, vb, req.vsp)) {
                conflict = true;
                return;
            }
            const int64_t ix0 = std::max(ebx.x0, vb.x0 - half);
            const int64_t iy0 = std::max(ebx.y0, vb.y0 - half);
            const int64_t ix1 = std::min(ebx.x1, vb.x1 + half);
            const int64_t iy1 = std::min(ebx.y1, vb.y1 + half);
            if (ix0 >= ix1 || iy0 >= iy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ix1 - ix0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (iy1 - iy0) / rc);
            pres_cut += req.pres * ncx * ncy;
        });
        if (conflict) return 0;
    }

    if (req.att_cuts) {
        for (const auto& vb : *req.att_cuts) {
            if (!(vb == cut) && !rects_clear_linf(cut, vb, req.vsp)) return 0;
        }
    }
    return 1;
}

int64_t box_congestion_fast_pure(const BoxCongestionFastRequest& req, bool& overflow) {
    overflow = false;
    const int64_t rc = std::max<int64_t>(1, req.rc);
    const int64_t half = floor_div_pos(req.sp, 2);
    const Box64 ebx{req.box.x0 - half, req.box.y0 - half, req.box.x1 + half, req.box.y1 + half};
    const Box64 q{req.box.x0 - req.sp, req.box.y0 - req.sp, req.box.x1 + req.sp, req.box.y1 + req.sp};
    int64_t total = 0;

    if (req.present_hint >= 0) {
        total = req.present_hint;
    } else if (req.metal_idx) {
        req.metal_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id == req.own_metal) return;
            const int64_t ox0 = std::max(ebx.x0, item.x0 - half);
            const int64_t oy0 = std::max(ebx.y0, item.y0 - half);
            const int64_t ox1 = std::min(ebx.x1, item.x1 + half);
            const int64_t oy1 = std::min(ebx.y1, item.y1 + half);
            if (ox0 >= ox1 || oy0 >= oy1) return;
            const int64_t ncx = std::max<int64_t>(1, (ox1 - ox0) / rc);
            const int64_t ncy = std::max<int64_t>(1, (oy1 - oy0) / rc);
            total += req.pres * ncx * ncy;
        });
    }
    if (req.obs_idx && req.own_obs != UINT32_MAX) {
        req.obs_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            if (item.owner_id != req.own_obs) return;
            int64_t sx = 0, sy = 0;
            if (euclid_in_open_band(req.box, Box64{item.x0, item.y0, item.x1, item.y1}, req.sp, sx, sy)) {
                total += req.pres;
            }
        });
    }
    if (req.att_idx) {
        req.att_idx->for_each_intersecting(q, [&](const BucketItem& item) {
            int64_t sx = 0, sy = 0;
            if (euclid_in_open_band(req.box, Box64{item.x0, item.y0, item.x1, item.y1}, req.sp, sx, sy)) {
                total += req.pres;
            }
        });
    }
    if (req.hist_grid) {
        total += req.hist_grid->eval_units(req.gi, ebx, rc);
    }
    return scale_congestion_units(total, req.mul_scale, req.cfs, overflow);
}

}  // namespace clay_pnr::route

