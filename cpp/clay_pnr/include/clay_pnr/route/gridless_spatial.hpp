#pragma once

// Pure C++20 Gridless spatial index, CSR dense lookup grid, and exact gap metallisation.
// Strictly free of <Python.h>; owner/object metadata is carried as opaque uint64_t tags.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/hash.hpp"
#include "clay_pnr/route/gridless_core.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clay_pnr::route {

struct PureSpatialEntry {
    Box64 norm_box{};
    uint32_t owner_id = 0;
    uint32_t claim_idx = 0;
    uint64_t rect_tag = 0;
    uint64_t owner_tag = 0;  // opaque tag (0 == None; symbol id in C++20 or borrowed PyObject* cast in Python wrapper)
};

struct BucketItem {
    int64_t x0 = 0;
    int64_t y0 = 0;
    int64_t x1 = 0;
    int64_t y1 = 0;
    uint32_t owner_id = 0;
    uint32_t claim_idx = 0;
    uint32_t entry_idx = 0;
    uint32_t _pad = 0;
};

struct BucketSlot {
    int64_t bx = 0;
    int64_t by = 0;
    std::vector<BucketItem> items;
};

struct FlatSpan {
    uint32_t offset = 0;
    uint32_t count = 0;
};

struct PureSpatialIndex {
    int64_t cell = 6000;
    std::vector<PureSpatialEntry> entries;
    std::vector<BucketSlot> buckets;
    std::unordered_map<BucketKey, uint32_t, BucketKeyHash> bucket_map;

    // Claim deduplication by (owner_id, rect_tag)
    std::unordered_map<ClaimKey, uint32_t, ClaimKeyHash> claim_map;
    uint32_t next_claim_idx = 0;
    bool has_dup_claims = false;
    mutable std::vector<uint32_t> seen_gen;
    mutable uint32_t cur_gen = 0;

    // Bounding box in bucket coordinates
    bool has_buckets = false;
    int64_t min_bx = 0, max_bx = 0, min_by = 0, max_by = 0;

    // Lazily compiled flat CSR + dense 2D lookup grid
    mutable bool compiled = false;
    mutable bool use_dense = false;
    mutable int64_t dense_nx = 0, dense_ny = 0;
    mutable std::vector<int32_t> dense_grid;  // bucket_idx or -1
    mutable std::vector<FlatSpan> flat_spans;
    mutable std::vector<BucketItem> flat_items;

    explicit PureSpatialIndex(int64_t cell_nm = 6000) noexcept : cell(cell_nm > 0 ? cell_nm : 6000) {}

    // Insert normalized box and update spatial buckets; returns entry index.
    uint32_t add_entry(const Box64& norm_box, uint32_t owner_id,
                       uint64_t rect_tag, uint64_t owner_tag = 0);

    // Build flat CSR & optional dense 2D grid on demand.
    void ensure_compiled() const;

    // Deep-copy spatial geometry and bucket state (compiled cache recomputed lazily).
    PureSpatialIndex clone_geometry() const;

    inline uint32_t next_gen() const noexcept {
        if (++cur_gen == 0) {
            std::fill(seen_gen.begin(), seen_gen.end(), 0);
            cur_gen = 1;
        }
        return cur_gen;
    }

    inline int32_t find_bucket(int64_t bx, int64_t by) const noexcept {
        if (use_dense) {
            return dense_grid[static_cast<size_t>((bx - min_bx) * dense_ny + (by - min_by))];
        }
        const auto it = bucket_map.find(BucketKey{bx, by});
        return (it == bucket_map.end()) ? -1 : static_cast<int32_t>(it->second);
    }

    // Exact bbox query callback: calls fn(const BucketItem&) once per unique claim
    // intersecting query box `q`, in exact deterministic bucket order.
    template <typename Fn>
    inline void for_each_intersecting(const Box64& q, Fn&& fn) const {
        if (!has_buckets) return;
        ensure_compiled();

        int64_t bx0 = floor_div_pos(q.x0, cell);
        int64_t bx1 = floor_div_pos(q.x1, cell);
        int64_t by0 = floor_div_pos(q.y0, cell);
        int64_t by1 = floor_div_pos(q.y1, cell);

        if (bx1 < min_bx || bx0 > max_bx || by1 < min_by || by0 > max_by) {
            return;
        }
        if (bx0 < min_bx) bx0 = min_bx;
        if (bx1 > max_bx) bx1 = max_bx;
        if (by0 < min_by) by0 = min_by;
        if (by1 > max_by) by1 = max_by;

        const bool need_dedup = has_dup_claims || (bx0 != bx1) || (by0 != by1);
        const BucketItem* items_ptr = flat_items.data();

        if (!need_dedup) {
            const int32_t bidx = find_bucket(bx0, by0);
            if (bidx < 0) return;
            const FlatSpan fspan = flat_spans[static_cast<size_t>(bidx)];
            const BucketItem* p = items_ptr + fspan.offset;
            const BucketItem* end = p + fspan.count;
            for (; p < end; ++p) {
                if (p->x1 < q.x0 || q.x1 < p->x0 || p->y1 < q.y0 || q.y1 < p->y0) {
                    continue;
                }
                fn(*p);
            }
            return;
        }

        const uint32_t gen = next_gen();
        uint32_t* seen = seen_gen.data();
        for (int64_t bx = bx0; bx <= bx1; ++bx) {
            for (int64_t by = by0; by <= by1; ++by) {
                const int32_t bidx = find_bucket(bx, by);
                if (bidx < 0) continue;
                const FlatSpan fspan = flat_spans[static_cast<size_t>(bidx)];
                const BucketItem* p = items_ptr + fspan.offset;
                const BucketItem* end = p + fspan.count;
                for (; p < end; ++p) {
                    if (p->x1 < q.x0 || q.x1 < p->x0 || p->y1 < q.y0 || q.y1 < p->y0) {
                        continue;
                    }
                    if (seen[p->claim_idx] == gen) continue;
                    seen[p->claim_idx] = gen;
                    fn(*p);
                }
            }
        }
    }

    bool has_foreign_intersecting(const Box64& q, uint32_t own_id) const noexcept;
    bool has_foreign_linf_conflict(const Box64& box, int64_t sp, uint32_t own_id) const noexcept;
    bool has_any_intersecting(const Box64& q) const noexcept;

    // Pure zero-Python specialized conflict & congestion scans.
    struct MetalConflictHit {
        uint32_t entry_idx = 0;
        bool is_touch = false;
    };
    struct CommittedMetalScanResult {
        int64_t present_total = 0;
        std::vector<MetalConflictHit> hits;
    };
    struct CutSortItem {
        uint32_t entry_idx = 0;
        uint32_t owner_id = 0;
        Box64 box{};
    };
    struct CutConflictScanResult {
        int64_t present_same = 0;
        std::vector<CutSortItem> foreign_contacts;
        std::vector<CutSortItem> spacing_violations;
    };
    struct CommittedCutScanResult {
        int64_t present_total = 0;
        std::vector<CutSortItem> foreign_contacts;
        std::vector<CutSortItem> spacing_violations;
    };
    struct FixedCutHit {
        Box64 box{};
        uint32_t entry_idx = 0;
    };
    struct SameNetBandHit {
        uint32_t entry_idx = 0;
        int64_t sx = 0;
        int64_t sy = 0;
    };

    std::vector<MetalConflictHit> scan_metal_conflicts_pure(
        const Box64& box, int64_t space_nm, uint32_t own_id, uint32_t excl_id) const;
    CommittedMetalScanResult scan_committed_metal_pure(
        const Box64& box, int64_t sp, int64_t rc, int64_t pres, uint32_t own_id) const;
    std::vector<uint32_t> scan_obstacle_conflicts_pure(
        const Box64& box, int64_t sp, uint32_t own_id) const;
    CutConflictScanResult scan_cut_conflicts_pure(
        const Box64& cut, int64_t vsp, uint32_t own_id) const;
    CommittedCutScanResult scan_committed_cuts_pure(
        const Box64& cut, int64_t vsp, int64_t rc, int64_t pres, uint32_t own_id) const;
    std::vector<FixedCutHit> scan_fixed_cut_conflicts_pure(
        const Box64& cut, int64_t vsp) const;
    int64_t scan_same_net_band_count_pure(
        const Box64& box, int64_t sp, uint32_t target_id, bool filter_by_owner) const;
    std::vector<SameNetBandHit> scan_same_net_band_rects_pure(
        const Box64& box, int64_t sp, uint32_t target_id) const;
};

// Exact gap metallisation check for same-net open-band Euclidean clearance.
bool raw_gap_metallised_pure(
    const Box64& candidate,
    const Box64& blocker,
    int64_t sx,
    int64_t sy,
    const std::vector<Box64>& attempt_metal,
    const PureSpatialIndex* obs_idx,
    const std::unordered_set<Box64, Box64Hash>& abstract_set,
    const PureSpatialIndex* metal_idx,
    std::vector<Box64>& cover_buf);

// ---------------------------------------------------------------------------
// Pure C++20 Batch Via-Site Filter (zero <Python.h>)
// ---------------------------------------------------------------------------

struct PureAggRejKey {
    int32_t cat = 0;
    int32_t gi = 0;
    int64_t candidate_half = 0;
    int64_t required_space = -1;  // -1 means None
    uint8_t has_blocker = 0;
    uint8_t owner_src = 0;        // 0=net, 1=obs_lo, 2=obs_hi, 3=metal_lo, 4=metal_hi, 5=cut_lo, 6=None
    uint32_t owner_id = 0;
    Box64 blocker{};

    bool operator==(const PureAggRejKey& o) const noexcept {
        return cat == o.cat && gi == o.gi && candidate_half == o.candidate_half &&
               required_space == o.required_space && has_blocker == o.has_blocker &&
               owner_src == o.owner_src && owner_id == o.owner_id && blocker == o.blocker;
    }
};

struct PureAggRejKeyHash {
    size_t operator()(const PureAggRejKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(static_cast<uint32_t>(k.cat)) |
                           (static_cast<uint64_t>(static_cast<uint32_t>(k.gi)) << 16) |
                           (static_cast<uint64_t>(k.owner_src) << 32) |
                           (static_cast<uint64_t>(k.owner_id) << 40));
        h ^= mix64(static_cast<uint64_t>(k.candidate_half) ^
                   (static_cast<uint64_t>(k.required_space) * 0x9e3779b97f4a7c15ULL));
        if (k.has_blocker) {
            h ^= mix64(static_cast<uint64_t>(k.blocker.x0) ^
                       (mix64(static_cast<uint64_t>(k.blocker.y0)) << 1));
            h ^= mix64(static_cast<uint64_t>(k.blocker.x1) ^
                       (mix64(static_cast<uint64_t>(k.blocker.y1)) << 1));
        }
        return static_cast<size_t>(h);
    }
};

struct PureAggRejVal {
    int64_t count = 0;
    Box64 extent{};
    uint64_t owner_tag = 0;
};

struct ViaSiteFilterRequest {
    bool use_explicit_sites = false;
    const std::vector<int64_t>* xs = nullptr;
    const std::vector<int64_t>* ys = nullptr;
    const std::vector<std::pair<int64_t, int64_t>>* extra_pts = nullptr;
    const std::vector<std::pair<int64_t, int64_t>>* explicit_sites = nullptr;

    bool has_atom = false;
    Box64 atom_rect{};
    int64_t atom_min_width = 0;

    int32_t lo = 0;
    int64_t bh = 0;
    int64_t ah = 0;
    int64_t ext = 0;
    int64_t sp_lo = 0;
    int64_t sp_hi = 0;
    int64_t vsp = 0;
    int64_t grid = 1;
    uint64_t net_owner_tag = 0;

    const PureSpatialIndex* obs_lo = nullptr;
    const PureSpatialIndex* obs_hi = nullptr;
    const PureSpatialIndex* obs_cut_lo = nullptr;
    const PureSpatialIndex* metal_lo = nullptr;
    const PureSpatialIndex* metal_hi = nullptr;
    const PureSpatialIndex* cut_lo = nullptr;
    uint32_t net_obs_lo = UINT32_MAX;
    uint32_t net_obs_hi = UINT32_MAX;
    uint32_t net_metal_lo = UINT32_MAX;
    uint32_t net_metal_hi = UINT32_MAX;
    uint32_t net_cut_lo = UINT32_MAX;

    const std::vector<Box64>* fixed_cuts = nullptr;
    const std::vector<Box64>* attempt_cuts = nullptr;
    const std::vector<Box64>* attempt_metal_lo = nullptr;
    const std::vector<Box64>* attempt_metal_hi = nullptr;
    const std::unordered_set<Box64, Box64Hash>* abstract_lo_set = nullptr;
    const std::unordered_set<Box64, Box64Hash>* abstract_hi_set = nullptr;

    bool has_env = false;
    Box64 env_box{};
    bool has_local_env = false;
    Box64 local_env_box{};

    const std::unordered_set<std::pair<int64_t, int64_t>, PairHash>* forbidden_set = nullptr;
    const std::vector<Box64>* fh_lo = nullptr;
    const std::vector<Box64>* fh_hi = nullptr;
    const std::vector<Box64>* fh_cut = nullptr;
    bool mirror_active = false;
};

struct ViaSiteFilterResult {
    std::vector<std::pair<int64_t, int64_t>> legal_sites;
    std::vector<std::pair<int64_t, int64_t>> fallback_sites;
    int64_t candidate_count = 0;
    std::unordered_map<PureAggRejKey, PureAggRejVal, PureAggRejKeyHash> agg_rejs;
};

ViaSiteFilterResult batch_filter_via_sites_pure(const ViaSiteFilterRequest& req);

// ---------------------------------------------------------------------------
// Pure C++20 Congestion & Fast Wire/Via Legality Kernels (zero <Python.h>)
// ---------------------------------------------------------------------------

int64_t eval_box_congestion_pure(
    const Box64& box, int64_t sp, int64_t rc, int64_t pres,
    const PureSpatialIndex* metal_idx, uint32_t net_metal_id,
    const PureSpatialIndex* obs_idx, uint32_t net_obs_id,
    const PureSpatialIndex* att_idx,
    const HistMap& hmap,
    int64_t mul_scale, int64_t cost_factor_scale,
    bool& overflow);

int64_t eval_cut_congestion_pure(
    const Box64& cut, int64_t vsp, int64_t rc, int64_t pres,
    const PureSpatialIndex* cut_idx, uint32_t net_cut_id,
    const HistMap& hmap,
    int64_t mul_scale, int64_t cost_factor_scale,
    bool& overflow);

struct BatchViaCongestionRequest {
    const std::vector<std::pair<int64_t, int64_t>>* sites = nullptr;
    int64_t bh = 0, ah = 0, ext = 0;
    int64_t sp_lo = 0, sp_hi = 0, vsp = 0;
    int64_t rc = 1, pres = 0, mul_scale = 0, cost_scale = 1000;
    const PureSpatialIndex* metal_lo = nullptr;
    const PureSpatialIndex* metal_hi = nullptr;
    const PureSpatialIndex* obs_lo = nullptr;
    const PureSpatialIndex* obs_hi = nullptr;
    const PureSpatialIndex* att_lo = nullptr;
    const PureSpatialIndex* att_hi = nullptr;
    const PureSpatialIndex* cut_lo = nullptr;
    uint32_t net_metal_lo = UINT32_MAX;
    uint32_t net_metal_hi = UINT32_MAX;
    uint32_t net_obs_lo = UINT32_MAX;
    uint32_t net_obs_hi = UINT32_MAX;
    uint32_t net_cut_lo = UINT32_MAX;
    const HistMap* hist_lo = nullptr;
    const HistMap* hist_hi = nullptr;
    const HistMap* hist_cut = nullptr;
    const std::vector<Box64>* wh_lo = nullptr;
    const std::vector<Box64>* wh_hi = nullptr;
};

std::vector<int64_t> batch_via_congestion_pure(const BatchViaCongestionRequest& req);

struct FastWideCheck {
    int64_t w_sp = 0;
    const PureSpatialIndex* core_idx = nullptr;
    uint32_t own_core = UINT32_MAX;
    const PureSpatialIndex* att_idx = nullptr;
    uint32_t own_att = UINT32_MAX;
};

struct WireLegalFastRequest {
    Box64 wb{};
    int64_t sp = 0;
    int64_t rc = 1;
    int64_t pres = 0;
    bool has_env = false;
    Box64 env{};
    const PureSpatialIndex* obs_idx = nullptr;
    uint32_t own_obs = UINT32_MAX;
    const PureSpatialIndex* metal_idx = nullptr;
    uint32_t own_metal = UINT32_MAX;
    const std::vector<FastWideCheck>* wide_entries = nullptr;
};

bool check_wire_legal_fast_pure(const WireLegalFastRequest& req, int64_t& present_total_out);

using SameNetGapLoaderFn = bool (*)(void* ctx,
                                    std::vector<Box64>& attempt_metal,
                                    std::unordered_set<Box64, Box64Hash>& abstract_set);

struct ViaLegalFastRequest {
    int64_t x = 0, y = 0;
    int64_t bh = 0, ah = 0, ext = 0;
    int64_t sp_lo = 0, sp_hi = 0, vsp = 0;
    int64_t grid = 1, rc = 1, pres = 0;
    bool has_env = false;
    Box64 env{};
    const PureSpatialIndex* obs_lo = nullptr;
    const PureSpatialIndex* obs_hi = nullptr;
    const PureSpatialIndex* obs_cut_lo = nullptr;
    const PureSpatialIndex* metal_lo = nullptr;
    const PureSpatialIndex* metal_hi = nullptr;
    const PureSpatialIndex* cut_lo = nullptr;
    uint32_t own_obs_lo = UINT32_MAX;
    uint32_t own_obs_hi = UINT32_MAX;
    uint32_t own_metal_lo = UINT32_MAX;
    uint32_t own_metal_hi = UINT32_MAX;
    uint32_t own_cut_lo = UINT32_MAX;
    const std::vector<FastWideCheck>* wide_lo = nullptr;
    const std::vector<FastWideCheck>* wide_hi = nullptr;
    SameNetGapLoaderFn load_gap_lo = nullptr;
    void* gap_lo_ctx = nullptr;
    SameNetGapLoaderFn load_gap_hi = nullptr;
    void* gap_hi_ctx = nullptr;
    const std::vector<Box64>* att_cuts = nullptr;
};

// Returns: 1 = legal, 0 = illegal, -1 = loader callback error.
int check_via_legal_fast_pure(const ViaLegalFastRequest& req,
                              int64_t& pres_lo, int64_t& pres_hi, int64_t& pres_cut);

struct BoxCongestionFastRequest {
    int64_t gi = 0;
    Box64 box{};
    int64_t sp = 0;
    int64_t rc = 1;
    int64_t pres = 0;
    int64_t mul_scale = 0;
    int64_t cfs = 1000;
    int64_t present_hint = -1;
    const PureSpatialIndex* metal_idx = nullptr;
    uint32_t own_metal = UINT32_MAX;
    const PureSpatialIndex* obs_idx = nullptr;
    uint32_t own_obs = UINT32_MAX;
    const PureSpatialIndex* att_idx = nullptr;
    const HistoryGridCore* hist_grid = nullptr;
};

int64_t box_congestion_fast_pure(const BoxCongestionFastRequest& req, bool& overflow);

}  // namespace clay_pnr::route

