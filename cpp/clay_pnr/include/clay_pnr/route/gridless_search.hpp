#pragma once

// Pure C++20 Gridless legality, congestion, MIM/mirror checking, hard connectivity
// preflight, Cartesian A*, and axis price cache engine.
// Strictly free of <Python.h>; optional callbacks bridge in via type-erased C hooks.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/hash.hpp"
#include "clay_pnr/route/gridless_core.hpp"
#include "clay_pnr/route/gridless_spatial.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clay_pnr::route {

// ---------------------------------------------------------------------------
// 1. Pure C++20 EdgeContext structures (opaque uint64_t owner/kind tags)
// ---------------------------------------------------------------------------

struct EdgeRejKey {
    int32_t cat = 0;
    int32_t gi = 0;
    int64_t candidate_half = 0;
    int64_t required_space = -1;  // -1 means None
    uint8_t has_blocker = 0;
    uint8_t owner_src = 0;        // 0=net, 1=obs[gi], 2=metal[gi], 3=cut[gi], 4=mim_entry, 5=None, 6=mirror
    uint32_t owner_id = 0;
    uint32_t kind_id = 0;         // 0=default, >0 opaque kind slot
    Box64 blocker{};

    bool operator==(const EdgeRejKey& o) const noexcept {
        return cat == o.cat && gi == o.gi && candidate_half == o.candidate_half &&
               required_space == o.required_space && has_blocker == o.has_blocker &&
               owner_src == o.owner_src && owner_id == o.owner_id &&
               kind_id == o.kind_id && blocker == o.blocker;
    }
};

struct EdgeRejKeyHash {
    size_t operator()(const EdgeRejKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(static_cast<uint32_t>(k.cat)) |
                           (static_cast<uint64_t>(static_cast<uint32_t>(k.gi)) << 16) |
                           (static_cast<uint64_t>(k.owner_src) << 32) |
                           (static_cast<uint64_t>(k.owner_id) << 40));
        h ^= mix64(static_cast<uint64_t>(k.kind_id) * 0x517cc1b727220a95ULL ^
                   static_cast<uint64_t>(k.candidate_half) ^
                   (static_cast<uint64_t>(k.required_space) * 0x9e3779b97f4a7c15ULL));
        if (k.has_blocker) {
            h ^= static_cast<uint64_t>(Box64Hash{}(k.blocker));
        }
        return static_cast<size_t>(h);
    }
};

struct PureEdgeRejVal {
    int64_t count = 0;
    Box64 extent{};
    uint64_t owner_tag = 0;       // 0 == None
    uint64_t extra_kind_tag = 0;  // 0 == None
};

struct PureEdgeSiteRej {
    EdgeRejKey key{};
    Box64 candidate_rect{};
    uint64_t owner_tag = 0;
    uint64_t extra_kind_tag = 0;
};

struct PureWideCheckEntry {
    int64_t w_sp = 0;
    const PureSpatialIndex* core_idx = nullptr;
    uint32_t own_core = UINT32_MAX;
    const PureSpatialIndex* att_idx = nullptr;
    uint32_t own_att = UINT32_MAX;
};

struct PureMimClearanceItem {
    uint64_t owner_tag = 0;
    Box64 blocker{};
    int64_t clearance = 0;
};

struct PureMimDerivedItem {
    uint64_t owner_tag = 0;
    Box64 blocker{};
    int64_t clearance = 0;
    bool euclidean = false;
    uint64_t rule_tag = 0;
    uint32_t kind_id = 0;
};

struct PureMimCarveItem {
    uint64_t owner_tag = 0;
    Box64 blocker{};
    uint64_t kind_tag = 0;
    uint32_t kind_id = 0;
};

struct PureMimCutTouchItem {
    uint64_t owner_tag = 0;
    Box64 blocker{};
};

struct PureEdgeLayerData {
    int64_t hw = 0;
    int64_t min_hw = 0;
    int64_t space = 0;
    int64_t base_rate = 100;
    bool pref_h = true;
    bool off_pref_weight_is_one = true;
    double off_pref_weight = 1.0;
    bool net_is_supra_min = false;

    const PureSpatialIndex* obs_idx = nullptr;
    uint32_t net_obs_id = UINT32_MAX;

    const PureSpatialIndex* metal_idx = nullptr;
    uint32_t net_metal_id = UINT32_MAX;

    const PureSpatialIndex* att_idx = nullptr;

    std::vector<Box64> attempt_metal;
    std::unordered_set<Box64, Box64Hash> abstract_set;
    std::unordered_set<OwnerBoxKey, OwnerBoxKeyHash> via_pad_set;
    std::vector<uint8_t> metal_owner_supra_min;
    std::vector<uint8_t> metal_owner_persistent;  // 0=no, 1=always_hard, 2=fallback

    bool has_wide_rule = false;
    int64_t min_wide_ge = 0;
    std::vector<PureWideCheckEntry> wide_entries;

    std::vector<PureMimClearanceItem> mim_bottom;
    std::vector<PureMimDerivedItem> mim_derived;
    std::vector<PureMimCarveItem> mim_carve;
};

struct PureEdgeViaData {
    int64_t bh0 = 0, ah0 = 0, ext0 = 0;
    int64_t bh1 = 0, ah1 = 0, ext1 = 0;
    int64_t vsp = 0;

    const PureSpatialIndex* obs_cut_idx = nullptr;
    std::vector<Box64> fixed_cuts_fallback;

    const PureSpatialIndex* cut_idx = nullptr;
    uint32_t net_cut_id = UINT32_MAX;

    std::vector<Box64> attempt_cuts;
    std::vector<PureMimCutTouchItem> mim_forb_cut;
    std::vector<PureMimClearanceItem> mim_top_marker;
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> forb_xy_0;
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> forb_xy_1;
};

struct PureEdgeContext {
    uint64_t net_tag = 0;
    int32_t n_layers = 0;
    int64_t grid = 1;
    int64_t rc = 1;
    int64_t pres = 0;
    int64_t mul_scale = 0;
    int64_t cost_factor_scale = 1000;
    int64_t turn_cost = 0;
    int64_t via_cost = 40000;
    int64_t step_cost = 100;
    int64_t off_pref_cost = 100;
    bool single_escape_ok = false;
    int64_t single_via_penalty = 0;
    int32_t via_succ_dirs[3] = {0, 1, 2};
    bool has_env = false;
    Box64 env_box{};
    bool neg_pending_hist = false;
    int64_t hist_scale_num = 0;

    const HistoryGridCore* hist_grid = nullptr;

    std::vector<PureEdgeLayerData> layers;
    std::vector<PureEdgeViaData> vias;

    // Aggregated rejection buffer
    std::unordered_map<EdgeRejKey, PureEdgeRejVal, EdgeRejKeyHash> agg_rejs;
    int64_t pending_rej_count = 0;
    int64_t total_rej_gen = 0;

    // Reusable scratch buffers
    std::vector<PureEdgeSiteRej> site_rejs_buf;
    std::vector<Box64> cover_buf;
    struct SnHit { Box64 blocker; uint32_t owner_id; uint64_t owner_tag; int64_t sx; int64_t sy; };
    std::vector<SnHit> sn_hits_buf;
    struct MetalHit { Box64 blocker; uint32_t owner_id; uint64_t owner_tag; };
    std::vector<MetalHit> metal_contacts_buf;
    std::vector<MetalHit> metal_spacings_buf;

    // Shared Cartesian preflight + A* packed edge cache
    CartesianGraphCache cart;

    // Optional active mirror legality partner context
    PureEdgeContext* mirror_ctx = nullptr;
    int64_t mirror_axis2 = 0;
    int64_t mirror_intent_space = 0;
    bool mirror_pair_conflict_seen = false;

    inline void record_rej(const EdgeRejKey& key, const Box64& cand_rect,
                           uint64_t owner_tag, uint64_t extra_kind_tag = 0) {
        auto it = agg_rejs.find(key);
        if (it == agg_rejs.end()) {
            agg_rejs.emplace(key, PureEdgeRejVal{1, cand_rect, owner_tag, extra_kind_tag});
        } else {
            auto& val = it->second;
            val.count += 1;
            if (cand_rect.x0 < val.extent.x0) val.extent.x0 = cand_rect.x0;
            if (cand_rect.y0 < val.extent.y0) val.extent.y0 = cand_rect.y0;
            if (cand_rect.x1 > val.extent.x1) val.extent.x1 = cand_rect.x1;
            if (cand_rect.y1 > val.extent.y1) val.extent.y1 = cand_rect.y1;
        }
        ++pending_rej_count;
        ++total_rej_gen;
    }

    inline void flush_site_rejs() {
        for (const auto& rec : site_rejs_buf) {
            record_rej(rec.key, rec.candidate_rect, rec.owner_tag, rec.extra_kind_tag);
        }
        site_rejs_buf.clear();
    }

    inline bool needs_wide_fallback_legal(int32_t gi, const Box64& box, int64_t half) const noexcept {
        const PureEdgeLayerData& L = layers[static_cast<size_t>(gi)];
        if (!L.has_wide_rule) return false;
        if (2 * half >= L.min_wide_ge) return true;
        for (const auto& we : L.wide_entries) {
            const Box64 wq{box.x0 - we.w_sp, box.y0 - we.w_sp, box.x1 + we.w_sp, box.y1 + we.w_sp};
            if (we.core_idx && we.core_idx->has_foreign_intersecting(wq, we.own_core)) {
                return true;
            }
            if (we.att_idx && we.att_idx->has_foreign_intersecting(wq, we.own_att)) {
                return true;
            }
        }
        return false;
    }

    inline bool needs_wide_fallback_cong(int32_t gi, const Box64& box, int64_t half) const noexcept {
        const PureEdgeLayerData& L = layers[static_cast<size_t>(gi)];
        if (!L.has_wide_rule) return false;
        if (2 * half >= L.min_wide_ge) return true;
        for (const auto& we : L.wide_entries) {
            const Box64 wq{box.x0 - we.w_sp, box.y0 - we.w_sp, box.x1 + we.w_sp, box.y1 + we.w_sp};
            if (we.core_idx && we.core_idx->has_any_intersecting(wq)) {
                return true;
            }
            if (we.att_idx && we.att_idx->has_any_intersecting(wq)) {
                return true;
            }
        }
        return false;
    }

    inline int64_t eval_edge_objective(
        int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by) const noexcept {
        const PureEdgeLayerData& L = layers[static_cast<size_t>(gi)];
        int64_t dx = bx - ax;
        if (dx < 0) dx = -dx;
        int64_t dy = by - ay;
        if (dy < 0) dy = -dy;
        const int64_t len = dx + dy;
        int64_t cost = len * L.base_rate;
        const bool moving_h = (ay == by);
        if (L.pref_h != moving_h) {
            if (L.off_pref_weight_is_one) {
                cost += len * off_pref_cost;
            } else {
                cost += round_half_even_i64(static_cast<double>(len * off_pref_cost) * L.off_pref_weight);
            }
        }
        return cost;
    }

    inline int64_t total_pending_rej() const noexcept {
        return pending_rej_count + (mirror_ctx ? mirror_ctx->pending_rej_count : 0);
    }

    inline int64_t total_rej_generation() const noexcept {
        return total_rej_gen + (mirror_ctx ? mirror_ctx->total_rej_gen : 0);
    }

    bool reject_active_mirror_pair_box(
        int32_t gi, const Box64& primary_box, const Box64& partner_box, int64_t partner_half);
    int check_wire_legal_single(
        int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t& out_present);
    int check_wire_legal(
        int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t& out_present);
    int check_via_legal_single(
        int64_t x, int64_t y, int32_t lo, int32_t n_flag,
        int64_t& out_pres_lo, int64_t& out_pres_hi, int64_t& out_pres_cut);
    int check_via_legal(
        int64_t x, int64_t y, int32_t lo, int32_t n_flag,
        int64_t& out_pres_lo, int64_t& out_pres_hi, int64_t& out_pres_cut);
    int eval_box_congestion(
        int32_t gi, const Box64& box, int64_t half, int64_t present_hint, int64_t& out_cost) const noexcept;
    int eval_via_cut_congestion(
        int32_t lo, const Box64& cut, int64_t present_cut_hint, int64_t& out_cost) const noexcept;
    int eval_via_congestion(
        int64_t x, int64_t y, int32_t lo, int32_t n_flag,
        int64_t pres_lo_hint, int64_t pres_hi_hint, int64_t pres_cut_hint,
        int64_t& out_cost) const noexcept;
    int eval_wire_price(
        int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
        int64_t present_hint, int64_t& out_cost) const noexcept;
    int eval_via_static_cost_fresh(
        int64_t x, int64_t y, int32_t lo, int64_t& out_cost, int32_t& out_model_n);
};

// ---------------------------------------------------------------------------
// 2. Type-erased pure C++20 search hooks (all optional / nullptr in native CLI)
// ---------------------------------------------------------------------------

struct GridlessSearchHooks {
    using PreflightWatchdogFn = bool (*)(void* ctx, int32_t kind, int64_t nodes, int64_t edges, bool packed);
    using AStarWatchdogFn = bool (*)(void* ctx, int32_t kind, int64_t exps, int64_t tap, bool packed);
    using WideWireFn = bool (*)(void* ctx, int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
                                bool& out_legal, int64_t& out_pres, bool& out_rej_inc);
    using WideViaFn = bool (*)(void* ctx, int64_t x, int64_t y, int32_t lo, int32_t nf,
                               bool& out_legal, int64_t& out_plo, int64_t& out_phi,
                               int64_t& out_pcut, bool& out_rej_inc);
    using WideCongFn = bool (*)(void* ctx, int32_t kind, int64_t a0, int64_t a1, int64_t a2,
                                int64_t a3, int64_t a4, int64_t& out_cost);

    void* ctx = nullptr;
    PreflightWatchdogFn preflight_watchdog = nullptr;
    AStarWatchdogFn astar_watchdog = nullptr;
    WideWireFn wide_wire = nullptr;
    WideViaFn wide_via = nullptr;
    WideCongFn wide_cong = nullptr;
};

struct PreflightResult {
    int32_t status = 0;  // 1=connected, 0=disconnected, -1=oom, -2=hook_abort
    int64_t connectivity_nodes = 0;
    int64_t connectivity_edges = 0;
    bool packed_enabled = false;
};

PreflightResult run_hard_connectivity_preflight_pure(
    PureEdgeContext& ctx,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int32_t>& layers,
    std::vector<int64_t> start_ids,
    std::vector<int64_t> goal_ids,
    int64_t max_exp,
    bool within_cap,
    const GridlessSearchHooks& hooks = {});

struct AStarPathNode {
    int64_t x = 0;
    int64_t y = 0;
    int32_t gi = 0;
    int8_t last_dir = 0;
    int32_t via_lo = -1;
    int32_t via_n = 0;
};

struct AStarResult {
    int32_t status = -1;  // 0=found, 1=pruned_by_incumbent, 2=unroutable, 3=budget, 4=solve_budget, -2=hook_abort
    int64_t expansions = 0;
    int64_t terminal_access_priced = 0;
    int32_t root_start_idx = -1;
    bool packed_enabled = false;
    std::vector<AStarPathNode> path;
};

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
    const GridlessSearchHooks& hooks = {});

// ---------------------------------------------------------------------------
// 3. Pure C++20 Axis Price Cache
// ---------------------------------------------------------------------------

struct PureAxisPriceCache {
    std::vector<int64_t> xs;
    std::vector<int64_t> ys;
    std::unordered_map<AxisLineKey, AxisLineSegs, AxisLineKeyHash> lines;
    std::unordered_map<AxisRangeKey, int64_t, AxisRangeKeyHash> range_totals;
    std::unordered_map<AxisRunCacheKey, int64_t, AxisRunCacheKeyHash> run_cache;
};

struct AxisPriceHooks {
    using WatchdogFn = bool (*)(void* ctx, int32_t kind_id);
    using FallbackPriceFn = bool (*)(void* ctx, int64_t gi, int64_t ax, int64_t ay,
                                     int64_t bx, int64_t by, int64_t& out_val);
    void* ctx = nullptr;
    WatchdogFn watchdog = nullptr;
    FallbackPriceFn fallback_price = nullptr;
};

// Returns: 1=ok, 0=budget_stopped, -1=hook_error, -2=non_orthogonal_error.
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
    const AxisPriceHooks& hooks = {});

// ---------------------------------------------------------------------------
// 4. Pure C++20 Region-A* Via-Target Projector & Frontier Heap (Phase 1B)
// ---------------------------------------------------------------------------

struct NativeViaTargetRec {
    ViaPathKey pk{};
    int64_t ex = 0;
    int64_t ey = 0;
    int64_t p1_y_nm = 0;
    int32_t goal_waypoint = -1;
    bool is_end_goal = false;
    int32_t other_ulast = 0;
    std::vector<int32_t> owners;
    int64_t ebx = 0;
    int64_t eby = 0;
    int64_t extra_turn = 0;
    bool has_tail_lb = true;
    int64_t tail_lb = 0;
};

struct NativeViaTargetCacheEntry {
    bool has_cross = false;
    bool has_owners = false;
    std::vector<NativeViaTargetRec> recs;
};

struct RegionFrontierEntry {
    int64_t lb_f = 0;
    int64_t seq = 0;
    uint32_t u_idx = 0;
    int64_t u_du = 0;
    const NativeViaTargetCacheEntry* cache_entry = nullptr;
    uint32_t rec_idx = 0;
    int32_t r_dest = 0;
};

struct RegionFrontierMinCmp {
    bool operator()(const RegionFrontierEntry& a, const RegionFrontierEntry& b) const noexcept {
        if (a.lb_f != b.lb_f) return a.lb_f > b.lb_f;
        return a.seq > b.seq;
    }
};

using OwnerIdsLookupFn = const std::vector<int32_t>* (*)(void* ctx, int64_t end_x, int64_t end_y);

struct PureViaTargetProjector {
    std::unordered_map<int32_t, std::unordered_set<std::pair<int64_t, int64_t>, PairHash>> goal_by_layer;
    int64_t target_node_x = 0;
    int64_t target_node_y = 0;
    int32_t target_node_gi = -1;
    bool target_singleton_none = false;
    std::vector<Box64> target_regions;
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> target_excluded_xy;
    LayerHminCore* hmin_core = nullptr;
    int64_t grid = 1;
    int64_t turn_cost = 0;
    int64_t via_cost = 40000;
    int64_t rect_bucket_nm = 140;
    bool neg_bucket_dedup = false;
    int32_t via_succ_dir_1 = 0;
    int32_t via_succ_dir_2 = 0;

    std::unordered_map<OwnerCacheKey, std::vector<int32_t>, OwnerCacheKeyHash> owner_ids_cache;
    std::unordered_map<ViaTargetCacheKey, NativeViaTargetCacheEntry*, ViaTargetCacheKeyHash> native_target_cache;
    std::vector<NativeViaTargetCacheEntry*> owned_cache_entries;
    std::vector<NativeViaTargetCacheEntry*> uncached_frontier_entries;
    std::unordered_map<RectEntryKey, int32_t, RectEntryKeyHash> rect_entry_counts;
    std::vector<RegionFrontierEntry> frontier_heap;
    int64_t frontier_seq = 0;

    PureViaTargetProjector() = default;
    ~PureViaTargetProjector();
    PureViaTargetProjector(const PureViaTargetProjector&) = delete;
    PureViaTargetProjector& operator=(const PureViaTargetProjector&) = delete;

    void clear_frontier_pure();
    bool is_goal_state(int64_t x_nm, int64_t y_nm, int32_t layer) const noexcept;
    bool has_cached_target(int32_t ul, int32_t other, int32_t rect,
                           int64_t px, int64_t py, bool with_owners) const noexcept;
    void inc_rect_entry(int32_t layer, int32_t rect_id, int32_t ulast, int64_t bx, int64_t by);
    bool frontier_top_f(int64_t& out_f) const noexcept;
    bool pop_frontier_raw(RegionFrontierEntry& out_entry) noexcept;

    // Pure C++20 owner lookup directly from PureSpatialIndex::owner_id.
    const std::vector<int32_t>* lookup_owner_ids_spatial(
        const PureSpatialIndex* other_idx, int64_t epx, int64_t epy);

    // Project clipped target boxes into sorted/deduped NativeViaTargetCacheEntry.
    // Returns: 1=ok, -1=start outside rect, -2=owner_lookup_error.
    int project_clipped_pure(
        int32_t ul, int32_t other, int64_t px, int64_t py,
        int64_t rx0, int64_t ry0, int64_t rx1, int64_t ry1,
        const std::vector<Box64>& clipped,
        bool with_owners,
        OwnerIdsLookupFn owner_fn,
        void* owner_ctx,
        NativeViaTargetCacheEntry& out_entry) const;

    // Enqueue projected/cached via targets into frontier_heap.
    // Returns bitmask (>=0): bit0=has_cross, bit1=sat_pruned; or -1=outside_rect, -2=owner_error.
    int enqueue_via_targets_pure(
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
        bool& out_u_used);

    // Check rect-entry saturation for a popped frontier entry.
    // Returns true if open (keep), false if saturated/empty (pruned).
    bool check_frontier_entry_open(
        const RegionFrontierEntry& e, int32_t r_other, int64_t other_cap,
        bool& out_sat_pruned) const noexcept;
};

// ---------------------------------------------------------------------------
// 5. Pure C++20 Region-Event Step Advancement (Phase 1B)
// ---------------------------------------------------------------------------

struct RegionStepState {
    int64_t ux = 0;
    int64_t uy = 0;
    int64_t ul = 0;
    int64_t ulast = 0;
    int64_t rect = 0;
    int64_t run_x = 0;
    int64_t run_y = 0;
    int64_t run_dir = 0;
};

struct RegionStepHooks {
    using WatchdogFn = bool (*)(void* ctx, int32_t kind_id);
    using FallbackWireFn = bool (*)(void* ctx, int64_t ul, int64_t ux, int64_t uy,
                                    int64_t vx, int64_t vy, bool& out_legal);
    using FallbackPriceFn = bool (*)(void* ctx, int64_t gi, int64_t ax, int64_t ay,
                                     int64_t bx, int64_t by, int64_t& out_val);
    void* ctx = nullptr;
    WatchdogFn watchdog = nullptr;
    FallbackWireFn fallback_wire = nullptr;
    FallbackPriceFn fallback_price = nullptr;
};

struct RegionStepResult {
    int32_t status = 0;  // 1=valid step, 0=none/illegal/budget, -1=hook_error, -2=non_orthogonal
    int64_t ev_charged = 0;
    int64_t price_charged = 0;
    bool budget_stopped = false;
    int64_t vx = 0;
    int64_t vy = 0;
    int64_t vdir = 0;
    int64_t out_rect = 0;
    int64_t start_x = 0;
    int64_t start_y = 0;
    int64_t signed_dir = 0;
    int64_t step_cost = 0;
};

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
    const RegionStepHooks& hooks = {});

}  // namespace clay_pnr::route

