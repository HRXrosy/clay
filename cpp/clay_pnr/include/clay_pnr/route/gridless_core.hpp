#pragma once

// Pure C++20 algorithmic kernels and data structures for Clay's GridlessRouteEngine.
// Strictly free of CPython headers so core routing algorithms compile and test natively.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/hash.hpp"
#include "clay_pnr/core/status.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace clay_pnr::route {

using core::Box64;
using core::BucketKey;
using core::BucketKeyHash;
using core::ClaimKey;
using core::ClaimKeyHash;
using core::box_fully_covered_by_rects;
using core::box_sep;
using core::closed_rects_touch;
using core::compute_gap_box;
using core::euclid_in_open_band;
using core::euclid_sq_lt_128;
using core::floor_div_pos;
using core::mix64;
using core::normalize_box;
using core::owned_landing_has_min_throat;
using core::rects_clear_euclid_exact;
using core::rects_clear_linf;

// ---------------------------------------------------------------------------
// 1. Rejection categories & pure hash/key primitives
// ---------------------------------------------------------------------------

enum RejCategory : int32_t {
    REJ_CONFLICT_BRANCH_VIA = 0,
    REJ_MANUFACTURING_GRID = 1,
    REJ_GROUP_ROUTE_LOCAL_ENVELOPE = 2,
    REJ_ROUTING_ENVELOPE = 3,
    REJ_FIXED_METAL = 4,
    REJ_FIXED_SAME_NET_VIA_PAD = 5,
    REJ_FIXED_CUT = 6,
    REJ_COMMITTED_METAL_CONTACT = 7,
    REJ_COMMITTED_METAL_SPACING = 8,
    REJ_COMMITTED_CUT_CONTACT = 9,
    REJ_COMMITTED_CUT_SPACING = 10,
    REJ_COMMITTED_SELF_CUT = 11,
    REJ_ATTEMPT_CUT = 12,
    REJ_MIM_BOTTOM_ROUTING = 13,
    REJ_MIM_DERIVED_UNRELATED = 14,
    REJ_MIM_CARVE_BOTTOM = 15,
    REJ_MIM_FORBIDDEN_CUT_TOUCH = 16,
    REJ_MIM_TOP_VIA_MARKER = 17,
    REJ_MIRROR_PAIR_METAL = 18,
    REJ_MIRROR_PAIR_CUT = 19,
};

struct PairHash {
    size_t operator()(const std::pair<int64_t, int64_t>& p) const noexcept {
        return static_cast<size_t>(mix64(
            static_cast<uint64_t>(p.first) ^
            (mix64(static_cast<uint64_t>(p.second)) + 0x9e3779b97f4a7c15ULL)));
    }
};

struct Box64Hash {
    size_t operator()(const Box64& b) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(b.x0));
        h ^= mix64(static_cast<uint64_t>(b.y0) + 0x9e3779b97f4a7c15ULL);
        h ^= mix64(static_cast<uint64_t>(b.x1) + 0xbf58476d1ce4e5b9ULL);
        h ^= mix64(static_cast<uint64_t>(b.y1) + 0x94d049bb133111ebULL);
        return static_cast<size_t>(h);
    }
};

struct OwnerBoxKey {
    uint32_t owner_id = 0;
    Box64 box{};

    constexpr bool operator==(const OwnerBoxKey& o) const noexcept {
        return owner_id == o.owner_id && box == o.box;
    }
};

struct OwnerBoxKeyHash {
    size_t operator()(const OwnerBoxKey& k) const noexcept {
        return static_cast<size_t>(mix64(
            static_cast<uint64_t>(k.owner_id) * 0x9e3779b97f4a7c15ULL ^
            static_cast<uint64_t>(Box64Hash{}(k.box))));
    }
};

inline bool box_hits_any(const Box64& b, const std::vector<Box64>& halos) noexcept {
    for (const auto& h : halos) {
        if (closed_rects_touch(b, h)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// 2. Native2DPotential core (4-quadrant / 1D-Pareto / 2D KD-tree exact L1)
// ---------------------------------------------------------------------------

struct Pt1D {
    int64_t p = 0;
    int64_t v = 0;
};

struct Pt2D {
    int64_t x = 0;
    int64_t y = 0;
    int64_t c = 0;
};

struct KDNode2D {
    int64_t px = 0;
    int64_t py = 0;
    int64_t cost = 0;
    int64_t x0 = 0;
    int64_t y0 = 0;
    int64_t x1 = 0;
    int64_t y1 = 0;
    int64_t min_cost = 0;
    int32_t left = -1;
    int32_t right = -1;
};

std::vector<Pt1D> reduce_1d_pareto_cpp(std::vector<Pt1D>& pts, int64_t rate);

inline int64_t eval_1d_pareto_cpp(const std::vector<Pt1D>& pareto,
                                  int64_t coord, int64_t rate) noexcept {
    const size_t n = pareto.size();
    const int64_t p0 = pareto[0].p;
    int64_t best = pareto[0].v + ((coord < p0) ? (p0 - coord) : (coord - p0)) * rate;
    for (size_t i = 1; i < n; ++i) {
        const int64_t pi = pareto[i].p;
        const int64_t cand = pareto[i].v + ((coord < pi) ? (pi - coord) : (coord - pi)) * rate;
        if (cand < best) best = cand;
    }
    return best;
}

struct Potential2DCore {
    int64_t rate = 1;
    int64_t gx0 = 0, gx1 = 0, gy0 = 0, gy1 = 0;
    int64_t sw_k = 0, se_k = 0, nw_k = 0, ne_k = 0;
    std::vector<Pt1D> south_p, north_p, west_p, east_p;
    std::vector<Pt2D> raw_pts;
    std::vector<KDNode2D> kd_nodes;
    int32_t kd_root = -1;

    int32_t build_kd(std::vector<Pt2D>& items, size_t lo, size_t hi);
    void init(std::vector<Pt2D> pts, int64_t r);

    inline int64_t eval_interior(int64_t x, int64_t y) const noexcept {
        if (raw_pts.empty()) {
            return std::numeric_limits<int64_t>::max();
        }
        if (kd_root < 0) {
            const Pt2D* p = raw_pts.data();
            const size_t n = raw_pts.size();
            int64_t dx = x - p[0].x;
            if (dx < 0) dx = -dx;
            int64_t dy = y - p[0].y;
            if (dy < 0) dy = -dy;
            int64_t best = p[0].c + (dx + dy) * rate;
            for (size_t i = 1; i < n; ++i) {
                int64_t dxi = x - p[i].x;
                if (dxi < 0) dxi = -dxi;
                int64_t dyi = y - p[i].y;
                if (dyi < 0) dyi = -dyi;
                const int64_t cand = p[i].c + (dxi + dyi) * rate;
                if (cand < best) best = cand;
            }
            return best;
        }
        int64_t best = std::numeric_limits<int64_t>::max();
        int32_t stack[128];
        int top = 0;
        stack[top++] = kd_root;
        const KDNode2D* nodes = kd_nodes.data();
        auto box_lb = [&](const KDNode2D& nd) noexcept -> int64_t {
            const int64_t dx = (x < nd.x0) ? (nd.x0 - x) : ((x > nd.x1) ? (x - nd.x1) : 0);
            const int64_t dy = (y < nd.y0) ? (nd.y0 - y) : ((y > nd.y1) ? (y - nd.y1) : 0);
            return (dx + dy) * rate + nd.min_cost;
        };
        while (top > 0) {
            const int32_t idx = stack[--top];
            const KDNode2D& nd = nodes[idx];
            if (box_lb(nd) >= best) continue;
            int64_t dx = x - nd.px;
            if (dx < 0) dx = -dx;
            int64_t dy = y - nd.py;
            if (dy < 0) dy = -dy;
            const int64_t cand = (dx + dy) * rate + nd.cost;
            if (cand < best) best = cand;
            const int32_t l = nd.left;
            const int32_t r = nd.right;
            const int64_t lb_l = (l >= 0) ? box_lb(nodes[l]) : best;
            const int64_t lb_r = (r >= 0) ? box_lb(nodes[r]) : best;
            if (lb_l < best && lb_r < best) {
                if (lb_l <= lb_r) {
                    if (top < 126) {
                        stack[top++] = r;
                        stack[top++] = l;
                    }
                } else {
                    if (top < 126) {
                        stack[top++] = l;
                        stack[top++] = r;
                    }
                }
            } else if (lb_l < best) {
                if (top < 127) stack[top++] = l;
            } else if (lb_r < best) {
                if (top < 127) stack[top++] = r;
            }
        }
        return best;
    }

    inline int64_t eval(int64_t x, int64_t y) const noexcept {
        if (raw_pts.empty()) {
            return std::numeric_limits<int64_t>::max();
        }
        if (x <= gx0) {
            if (y <= gy0) return sw_k - (x + y) * rate;
            if (y >= gy1) return nw_k + (-x + y) * rate;
            return eval_1d_pareto_cpp(west_p, y, rate) - x * rate;
        }
        if (x >= gx1) {
            if (y <= gy0) return se_k + (x - y) * rate;
            if (y >= gy1) return ne_k + (x + y) * rate;
            return eval_1d_pareto_cpp(east_p, y, rate) + x * rate;
        }
        if (y <= gy0) return eval_1d_pareto_cpp(south_p, x, rate) - y * rate;
        if (y >= gy1) return eval_1d_pareto_cpp(north_p, x, rate) + y * rate;
        return eval_interior(x, y);
    }
};

// ---------------------------------------------------------------------------
// 3. NativeLayerHmin core (multi-layer exact heuristic lower bound)
// ---------------------------------------------------------------------------

struct LayerHminSpec {
    int32_t layer = 0;
    int64_t rate = 1;
    int64_t via_tail = 0;
    const Potential2DCore* pot = nullptr;
};

struct LayerHminCore {
    int32_t n_gi = 0;
    int64_t via_cost = 40000;
    size_t max_cache = 0;
    std::vector<LayerHminSpec> specs;
    std::vector<Box64> target_boxes;
    std::unordered_map<std::pair<int64_t, int64_t>, std::vector<int64_t>, PairHash> cache;

    int64_t eval(int64_t x, int64_t y, int32_t gi);
};

// ---------------------------------------------------------------------------
// 4. Exact 3D separable L1 min-plus distance transform
// ---------------------------------------------------------------------------

struct DistanceTransformGoal {
    int64_t x = 0;
    int64_t y = 0;
    int64_t layer = 0;
    int64_t cost = 0;
};

struct DistanceTransformSeed {
    size_t flat_idx = 0;
    int64_t cost = 0;
};

enum class DistanceTransformStatus : uint8_t {
    Ok = 0,
    Empty = 1,
    NonMonotonicAxis = 2,
    GoalNotFound = 3,
    Overflow = 4,
    BadAlloc = 5,
};

DistanceTransformStatus exact_distance_transform_from_seeds(
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int64_t>& layers,
    const std::vector<DistanceTransformSeed>& seeds,
    int64_t max_term,
    int64_t step_cost,
    int64_t via_cost,
    std::vector<int64_t>& out_pot);

DistanceTransformStatus exact_distance_transform_core(
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    const std::vector<int64_t>& layers,
    const std::vector<DistanceTransformGoal>& goals,
    int64_t step_cost,
    int64_t via_cost,
    std::vector<int64_t>& out_pot);

// ---------------------------------------------------------------------------
// 5. Throat events & grid projection kernels
// ---------------------------------------------------------------------------

inline int64_t ceil_sqrt_py_exact(int64_t value) noexcept {
    // Exact match to Python:
    //   root = math.isqrt(max(0, int(value)))
    //   return root if root * root == value else root + 1
    if (value < 0) return 1;
    if (value == 0) return 0;
    const uint64_t uv = static_cast<uint64_t>(value);
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(uv)));
    if (r > 4294967295ULL) r = 4294967295ULL;
    while (r < 4294967295ULL && (r + 1ULL) * (r + 1ULL) <= uv) ++r;
    while (r * r > uv) --r;
    return (r * r == uv) ? static_cast<int64_t>(r) : static_cast<int64_t>(r + 1ULL);
}

inline void grid_proj_preserve_sides(
    int64_t v, int64_t lo, int64_t hi, int64_t grid,
    int64_t out[2], int& count) noexcept {
    count = 0;
    if (grid <= 1) {
        out[count++] = v;
        return;
    }
    const int64_t fl = floor_div_pos(v, grid) * grid;
    const int64_t cl = (fl == v) ? fl : (fl + grid);
    if (lo <= fl && fl <= hi) out[count++] = fl;
    if (cl != fl && lo <= cl && cl <= hi) out[count++] = cl;
}

void compute_axis_events(
    const std::vector<int64_t>& base_vals,
    int64_t dom_lo, int64_t dom_hi,
    int64_t atom_lo, int64_t atom_hi,
    int64_t bh, int64_t grid,
    std::vector<int64_t>& out);

struct ThroatEventsResult {
    std::vector<int64_t> xs;
    std::vector<int64_t> ys;
    std::vector<std::pair<int64_t, int64_t>> thresh;
};

ThroatEventsResult compute_throat_events_core(
    const std::vector<int64_t>& base_xs,
    const std::vector<int64_t>& base_ys,
    int64_t x0, int64_t y0, int64_t x1, int64_t y1,
    int64_t bh, int64_t min_width, int64_t grid);

// ---------------------------------------------------------------------------
// 6. Congestion scaling & HistoryGridCore (O(1) 2D SAT + sparse fallback)
// ---------------------------------------------------------------------------

using HistMap = std::unordered_map<std::pair<int64_t, int64_t>, int64_t, PairHash>;

inline int64_t eval_history_units_cpp(const HistMap& hmap, const Box64& ebx, int64_t rc) noexcept {
    if (hmap.empty()) return 0;
    const int64_t cx0 = floor_div_pos(ebx.x0 - rc, rc);
    const int64_t cx1 = floor_div_pos(ebx.x1, rc) + 1;
    const int64_t cy0 = floor_div_pos(ebx.y0 - rc, rc);
    const int64_t cy1 = floor_div_pos(ebx.y1, rc) + 1;
    int64_t total = 0;
    for (int64_t cx = cx0; cx <= cx1; ++cx) {
        const int64_t cxl = cx * rc;
        if (!(cxl < ebx.x1 && cxl + rc > ebx.x0)) continue;
        for (int64_t cy = cy0; cy <= cy1; ++cy) {
            const int64_t cyl = cy * rc;
            if (!(cyl < ebx.y1 && cyl + rc > ebx.y0)) continue;
            auto it = hmap.find({cx, cy});
            if (it != hmap.end()) total += it->second;
        }
    }
    return total;
}

inline int64_t scale_congestion_units(
    int64_t total, int64_t mul_scale, int64_t cost_factor_scale, bool& overflow) noexcept {
    if (total == 0) return 0;
    if (total < 0 || mul_scale < 0 || cost_factor_scale <= 0) {
        overflow = true;
        return -1;
    }
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t hi = 0;
    const uint64_t lo = _umul128(static_cast<uint64_t>(total), static_cast<uint64_t>(mul_scale), &hi);
    const uint64_t half = static_cast<uint64_t>(cost_factor_scale / 2);
    if (hi == 0 && lo <= static_cast<uint64_t>(INT64_MAX) - half) {
        return static_cast<int64_t>((lo + half) / static_cast<uint64_t>(cost_factor_scale));
    }
    overflow = true;
    return -1;
#else
    if (mul_scale == 0) return 0;
    if (total > (INT64_MAX - cost_factor_scale) / mul_scale) {
        overflow = true;
        return -1;
    }
    return (total * mul_scale + cost_factor_scale / 2) / cost_factor_scale;
#endif
}

inline int64_t round_half_even_i64(double x) noexcept {
    const double f = std::floor(x);
    const double diff = x - f;
    if (diff < 0.5) return static_cast<int64_t>(f);
    if (diff > 0.5) return static_cast<int64_t>(f) + 1;
    const int64_t fi = static_cast<int64_t>(f);
    return (fi & 1LL) ? (fi + 1) : fi;
}

inline int64_t scale_floor_units(
    int64_t units, int64_t mul_scale, int64_t cost_factor_scale, bool& overflow) noexcept {
    if (units == 0) return 0;
    if (units < 0 || mul_scale < 0 || cost_factor_scale <= 0) {
        overflow = true;
        return -1;
    }
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t hi = 0;
    const uint64_t lo = _umul128(static_cast<uint64_t>(units), static_cast<uint64_t>(mul_scale), &hi);
    if (hi == 0) {
        return static_cast<int64_t>(lo / static_cast<uint64_t>(cost_factor_scale));
    }
    overflow = true;
    return -1;
#else
    if (mul_scale == 0) return 0;
    if (units > INT64_MAX / mul_scale) {
        overflow = true;
        return -1;
    }
    return (units * mul_scale) / cost_factor_scale;
#endif
}

struct HistoryRawCell {
    int64_t cx = 0;
    int64_t cy = 0;
    int64_t val = 0;
};

struct HistoryLayer {
    static constexpr uint64_t MAX_DENSE_DIM = 4096ULL;
    static constexpr uint64_t MAX_DENSE_CELLS = 262144ULL;

    bool dense = false;
    int64_t min_cx = 0, max_cx = 0, min_cy = 0, max_cy = 0;
    int64_t width = 0, height = 0, stride = 0;
    std::vector<int64_t> sat;
    std::unordered_map<std::pair<int64_t, int64_t>, int64_t, PairHash> sparse;

    bool empty() const noexcept { return !dense && sparse.empty(); }

    void clear() noexcept;
    void build_from_cells(const std::vector<HistoryRawCell>& cells);
    void unpack_dense_to_sparse();
    void increment(int64_t cx, int64_t cy, int64_t delta,
                   int64_t saturation_cap = std::numeric_limits<int64_t>::max());
    void decay(double factor);
    void compile();

    int64_t query_box(const Box64& ebx, int64_t rc) const noexcept {
        if (empty()) return 0;
        int64_t cx0 = floor_div_pos(ebx.x0, rc);
        int64_t cx1 = floor_div_pos(ebx.x1 - 1, rc);
        int64_t cy0 = floor_div_pos(ebx.y0, rc);
        int64_t cy1 = floor_div_pos(ebx.y1 - 1, rc);
        if (cx0 > cx1 || cy0 > cy1) return 0;
        if (cx1 < min_cx || cx0 > max_cx || cy1 < min_cy || cy0 > max_cy) return 0;
        if (cx0 < min_cx) cx0 = min_cx;
        if (cx1 > max_cx) cx1 = max_cx;
        if (cy0 < min_cy) cy0 = min_cy;
        if (cy1 > max_cy) cy1 = max_cy;
        if (dense) {
            const size_t x0 = static_cast<size_t>(cx0 - min_cx);
            const size_t x1 = static_cast<size_t>(cx1 - min_cx + 1);
            const size_t y0 = static_cast<size_t>(cy0 - min_cy);
            const size_t y1 = static_cast<size_t>(cy1 - min_cy + 1);
            const size_t st = static_cast<size_t>(stride);
            return sat[y1 * st + x1] - sat[y0 * st + x1] - sat[y1 * st + x0] + sat[y0 * st + x0];
        }
        const uint64_t area =
            static_cast<uint64_t>(cx1 - cx0 + 1) * static_cast<uint64_t>(cy1 - cy0 + 1);
        int64_t total = 0;
        if (area <= sparse.size()) {
            for (int64_t cx = cx0; cx <= cx1; ++cx) {
                for (int64_t cy = cy0; cy <= cy1; ++cy) {
                    auto it = sparse.find({cx, cy});
                    if (it != sparse.end()) total += it->second;
                }
            }
        } else {
            for (const auto& kv : sparse) {
                const int64_t cx = kv.first.first;
                const int64_t cy = kv.first.second;
                if (cx >= cx0 && cx <= cx1 && cy >= cy0 && cy <= cy1) {
                    total += kv.second;
                }
            }
        }
        return total;
    }
};

struct HistoryGridCore {
    static constexpr int64_t FAST_LAYER_LIMIT = 64;
    static constexpr uint64_t MAX_DENSE_CELLS = HistoryLayer::MAX_DENSE_CELLS;

    std::vector<HistoryLayer> fast_layers;
    std::unordered_map<int64_t, HistoryLayer> extra_layers;
    bool has_any = false;

    void clear() noexcept;
    void build(const std::unordered_map<int64_t, std::vector<HistoryRawCell>>& by_layer);
    void increment(int64_t rgi, int64_t cx, int64_t cy, int64_t delta,
                   int64_t saturation_cap = std::numeric_limits<int64_t>::max());
    void decay(double factor);
    void compile();

    const HistoryLayer* get_layer(int64_t rgi) const noexcept {
        if (!has_any) return nullptr;
        if (rgi >= 0 && rgi < static_cast<int64_t>(fast_layers.size())) {
            const HistoryLayer& hl = fast_layers[static_cast<size_t>(rgi)];
            return hl.empty() ? nullptr : &hl;
        }
        auto it = extra_layers.find(rgi);
        return (it == extra_layers.end() || it->second.empty()) ? nullptr : &it->second;
    }

    int64_t eval_units(int64_t rgi, const Box64& ebx, int64_t rc) const noexcept {
        const HistoryLayer* hl = get_layer(rgi);
        if (!hl) return 0;
        return hl->query_box(ebx, rc);
    }

    int64_t eval_pending_lb(
        int64_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
        int64_t pad, int64_t rc, int64_t hist_scale_num, int64_t cost_factor_scale,
        bool& overflow) const noexcept {
        if (!has_any || (ax == bx && ay == by)) return 0;
        const HistoryLayer* hl = get_layer(gi);
        if (!hl) return 0;
        const int64_t x0 = (ax <= bx ? ax : bx) - pad;
        const int64_t x1 = (ax <= bx ? bx : ax) + pad;
        const int64_t y0 = (ay <= by ? ay : by) - pad;
        const int64_t y1 = (ay <= by ? by : ay) + pad;
        const int64_t units = hl->query_box(Box64{x0, y0, x1, y1}, rc);
        if (units == 0) return 0;
        return scale_floor_units(units, hist_scale_num, cost_factor_scale, overflow);
    }
};

// ---------------------------------------------------------------------------
// 7. AxisPriceCache, ViaTargetProjector, Cartesian A*, Region-Event pure structs
// ---------------------------------------------------------------------------

struct AxisLineKey {
    int64_t gi = 0;
    int64_t fixed = 0;
    int8_t horizontal = 0;
    int8_t sign = 0;

    constexpr bool operator==(const AxisLineKey& o) const noexcept {
        return gi == o.gi && fixed == o.fixed && horizontal == o.horizontal && sign == o.sign;
    }
};

struct AxisLineKeyHash {
    size_t operator()(const AxisLineKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(k.fixed) ^
                           (static_cast<uint64_t>(k.gi) * 0x9e3779b97f4a7c15ULL));
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(k.horizontal)) << 1;
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(k.sign + 2)) << 3;
        return static_cast<size_t>(mix64(h));
    }
};

struct AxisRangeKey {
    AxisLineKey line{};
    int32_t begin = 0;
    int32_t end = 0;

    constexpr bool operator==(const AxisRangeKey& o) const noexcept {
        return line == o.line && begin == o.begin && end == o.end;
    }
};

struct AxisRangeKeyHash {
    size_t operator()(const AxisRangeKey& k) const noexcept {
        const uint64_t h = static_cast<uint64_t>(AxisLineKeyHash{}(k.line));
        const uint64_t be = (static_cast<uint64_t>(static_cast<uint32_t>(k.begin)) << 32) |
                            static_cast<uint64_t>(static_cast<uint32_t>(k.end));
        return static_cast<size_t>(mix64(h ^ mix64(be)));
    }
};

struct AxisLineSegs {
    std::vector<int64_t> vals;
    std::vector<uint8_t> valid;
};

struct AxisRunCacheKey {
    int64_t gi = 0;
    int64_t ax = 0, ay = 0, bx = 0, by = 0;

    constexpr bool operator==(const AxisRunCacheKey& o) const noexcept {
        return gi == o.gi && ax == o.ax && ay == o.ay && bx == o.bx && by == o.by;
    }
};

struct AxisRunCacheKeyHash {
    size_t operator()(const AxisRunCacheKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(k.gi) * 0x9e3779b97f4a7c15ULL ^
                           static_cast<uint64_t>(k.ax));
        h ^= mix64(static_cast<uint64_t>(k.ay) + 0xbf58476d1ce4e5b9ULL);
        h ^= mix64(static_cast<uint64_t>(k.bx) + 0x94d049bb133111ebULL);
        h ^= mix64(static_cast<uint64_t>(k.by) + 0x517cc1b727220a95ULL);
        return static_cast<size_t>(mix64(h));
    }
};

struct ViaPathKey {
    uint8_t len = 0;
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    constexpr bool operator==(const ViaPathKey& o) const noexcept {
        return len == o.len && x0 == o.x0 && y0 == o.y0 && x1 == o.x1 && y1 == o.y1 &&
               (len < 3 || (x2 == o.x2 && y2 == o.y2));
    }
};

struct ViaPathKeyHash {
    size_t operator()(const ViaPathKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(k.x0) ^ (static_cast<uint64_t>(k.len) << 56));
        h ^= mix64(static_cast<uint64_t>(k.y0) + 0x9e3779b97f4a7c15ULL);
        h ^= mix64(static_cast<uint64_t>(k.x1) + 0xbf58476d1ce4e5b9ULL);
        h ^= mix64(static_cast<uint64_t>(k.y1) + 0x94d049bb133111ebULL);
        if (k.len == 3) {
            h ^= mix64(static_cast<uint64_t>(k.x2) + 0x517cc1b727220a95ULL);
            h ^= mix64(static_cast<uint64_t>(k.y2) + 0x2545f4914f6cdd1dULL);
        }
        return static_cast<size_t>(h);
    }
};

struct OwnerCacheKey {
    uintptr_t idx_ptr = 0;
    int64_t px = 0;
    int64_t py = 0;

    constexpr bool operator==(const OwnerCacheKey& o) const noexcept {
        return idx_ptr == o.idx_ptr && px == o.px && py == o.py;
    }
};

struct OwnerCacheKeyHash {
    size_t operator()(const OwnerCacheKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(k.idx_ptr));
        h ^= mix64(static_cast<uint64_t>(k.px) + 0x9e3779b97f4a7c15ULL);
        h ^= mix64(static_cast<uint64_t>(k.py) + 0xbf58476d1ce4e5b9ULL);
        return static_cast<size_t>(h);
    }
};

struct ViaTargetCacheKey {
    int32_t ul = 0;
    int32_t other = 0;
    int32_t rectangle = 0;
    int64_t px = 0;
    int64_t py = 0;
    uint8_t with_owners = 0;

    constexpr bool operator==(const ViaTargetCacheKey& o) const noexcept {
        return ul == o.ul && other == o.other && rectangle == o.rectangle &&
               px == o.px && py == o.py && with_owners == o.with_owners;
    }
};

struct ViaTargetCacheKeyHash {
    size_t operator()(const ViaTargetCacheKey& k) const noexcept {
        uint64_t h = mix64(
            (static_cast<uint64_t>(static_cast<uint32_t>(k.ul)) << 32) ^
            static_cast<uint64_t>(static_cast<uint32_t>(k.other)) ^
            (static_cast<uint64_t>(k.with_owners) << 56));
        h ^= mix64(static_cast<uint64_t>(static_cast<uint32_t>(k.rectangle)) + 0x9e3779b97f4a7c15ULL);
        h ^= mix64(static_cast<uint64_t>(k.px) + 0xbf58476d1ce4e5b9ULL);
        h ^= mix64(static_cast<uint64_t>(k.py) + 0x94d049bb133111ebULL);
        return static_cast<size_t>(h);
    }
};

struct RectEntryKey {
    int32_t layer = 0;
    int32_t rect_id = 0;
    int32_t ulast = 0;
    int64_t bx = 0;
    int64_t by = 0;

    constexpr bool operator==(const RectEntryKey& o) const noexcept {
        return layer == o.layer && rect_id == o.rect_id && ulast == o.ulast &&
               bx == o.bx && by == o.by;
    }
};

struct RectEntryKeyHash {
    size_t operator()(const RectEntryKey& k) const noexcept {
        uint64_t h = mix64(
            (static_cast<uint64_t>(static_cast<uint32_t>(k.layer)) << 40) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(k.ulast)) << 32) ^
            static_cast<uint64_t>(static_cast<uint32_t>(k.rect_id)));
        h ^= mix64(static_cast<uint64_t>(k.bx) + 0x9e3779b97f4a7c15ULL);
        h ^= mix64(static_cast<uint64_t>(k.by) + 0xbf58476d1ce4e5b9ULL);
        return static_cast<size_t>(h);
    }
};

struct CartesianGraphCache {
    bool initialized = false;
    std::vector<int64_t> xs;
    std::vector<int64_t> ys;
    std::vector<int32_t> ordered_layers;
    std::vector<int32_t> layer_index;     // gi -> layer_pos or -1
    std::vector<int32_t> via_pair_index;  // lo -> pair_pos or -1

    int64_t nx = 0, ny = 0, nl = 0;
    int64_t plane = 0, potential_node_count = 0;
    int64_t horizontal_edges = 0, vertical_edges = 0, via_edges = 0;
    int64_t packed_edge_count = 0, packed_slot_count = 0;
    int64_t static_geometry_work = 0;
    bool graph_within_potential_cap = false;
    bool packed_enabled = false;
    bool packed_disabled = false;

    std::vector<int64_t> packed_keys;
    std::vector<int64_t> packed_costs;    // >=0 priced, -1 illegal, -2 legal_unpriced
    std::vector<int64_t> packed_present;  // 3 * packed_slot_count, -1 unknown
    bool via_model_use_flat = true;
    std::vector<uint8_t> via_model_flat;
    std::unordered_map<int64_t, uint8_t> via_model_map;
    std::vector<uint8_t> preflight_tested_edges;

    static constexpr int64_t PACKED_UNKNOWN = -1;
    static constexpr int64_t PACKED_ILLEGAL = -1;
    static constexpr int64_t PACKED_LEGAL_UNPRICED = -2;
    static constexpr int64_t PACKED_PRESENT_UNKNOWN = -1;

    void reset_for_graph(
        const std::vector<int64_t>& in_xs,
        const std::vector<int64_t>& in_ys,
        const std::vector<int32_t>& in_layers,
        int32_t max_gi,
        int64_t max_expansions,
        bool within_cap);

    inline int64_t horiz_edge_id(int32_t lp, int32_t yp, int32_t xp_min) const noexcept {
        return (static_cast<int64_t>(lp) * ny + yp) * (nx - 1) + xp_min;
    }

    inline int64_t vert_edge_id(int32_t lp, int32_t yp_min, int32_t xp) const noexcept {
        return horizontal_edges +
               (static_cast<int64_t>(lp) * (ny - 1) + yp_min) * nx + xp;
    }

    inline int64_t via_edge_id_pos(int32_t lo, int32_t yp, int32_t xp) const noexcept {
        return horizontal_edges + vertical_edges +
               static_cast<int64_t>(via_pair_index[static_cast<size_t>(lo)]) * plane +
               static_cast<int64_t>(yp) * nx + xp;
    }

    inline uint8_t get_via_model(int64_t via_eid) const noexcept {
        const int64_t voff = via_eid - (horizontal_edges + vertical_edges);
        if (via_model_use_flat) {
            return via_model_flat[static_cast<size_t>(voff)];
        }
        auto it = via_model_map.find(via_eid);
        return (it == via_model_map.end()) ? 0 : it->second;
    }

    inline void set_via_model(int64_t via_eid, uint8_t model_n) {
        const int64_t voff = via_eid - (horizontal_edges + vertical_edges);
        if (via_model_use_flat) {
            via_model_flat[static_cast<size_t>(voff)] = model_n;
        } else {
            via_model_map[via_eid] = model_n;
        }
    }

    bool enable_packed(bool force);
};

struct AStarEndpointSpec {
    int32_t xp = 0;
    int32_t yp = 0;
    int32_t lp = 0;
    bool has_via = false;
    int64_t vx = 0;
    int64_t vy = 0;
    int32_t vlo = 0;
    int32_t vn = 0;
    int64_t lb = 0;
};

struct TermViaKey {
    int64_t x = 0, y = 0;
    int32_t lo = 0, n = 0;

    constexpr bool operator==(const TermViaKey& o) const noexcept {
        return x == o.x && y == o.y && lo == o.lo && n == o.n;
    }
};

struct TermViaKeyHash {
    size_t operator()(const TermViaKey& k) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(k.x));
        h ^= mix64(static_cast<uint64_t>(k.y) + 0x9e3779b97f4a7c15ULL);
        h ^= mix64((static_cast<uint64_t>(static_cast<uint32_t>(k.lo)) << 32) |
                   static_cast<uint64_t>(static_cast<uint32_t>(k.n)));
        return static_cast<size_t>(h);
    }
};

struct AStarStartItem {
    int64_t f_lb = 0;
    int32_t start_idx = 0;
};

struct AStarStartItemGreater {
    bool operator()(const AStarStartItem& a, const AStarStartItem& b) const noexcept {
        if (a.f_lb != b.f_lb) return a.f_lb > b.f_lb;
        return a.start_idx > b.start_idx;
    }
};

struct AStarPQItem {
    int64_t f = 0;
    int32_t xp = 0;
    int32_t yp = 0;
    int32_t lp = 0;
    int8_t last_dir = 0;
    int64_t queued_g = 0;
};

struct AStarPQItemGreater {
    bool operator()(const AStarPQItem& a, const AStarPQItem& b) const noexcept {
        if (a.f != b.f) return a.f > b.f;
        if (a.xp != b.xp) return a.xp > b.xp;
        if (a.yp != b.yp) return a.yp > b.yp;
        if (a.lp != b.lp) return a.lp > b.lp;
        if (a.last_dir != b.last_dir) return a.last_dir > b.last_dir;
        return a.queued_g > b.queued_g;
    }
};

// Region-Event geometry helpers (exact match to region_event_geometry.py)
struct SmallPath {
    int len = 0;
    std::pair<int64_t, int64_t> pts[4]{};

    void push_dedup(int64_t x, int64_t y) noexcept {
        if (len == 0 || pts[len - 1].first != x || pts[len - 1].second != y) {
            pts[len++] = {x, y};
        }
    }

    bool operator==(const SmallPath& o) const noexcept {
        if (len != o.len) return false;
        for (int i = 0; i < len; ++i) {
            if (pts[i] != o.pts[i]) return false;
        }
        return true;
    }

    bool operator<(const SmallPath& o) const noexcept {
        const int m = (len < o.len) ? len : o.len;
        for (int i = 0; i < m; ++i) {
            if (pts[i] != o.pts[i]) return pts[i] < o.pts[i];
        }
        return len < o.len;
    }
};

struct SmallPathHash {
    size_t operator()(const SmallPath& p) const noexcept {
        uint64_t h = mix64(static_cast<uint64_t>(p.len));
        for (int i = 0; i < p.len; ++i) {
            h ^= mix64(static_cast<uint64_t>(p.pts[i].first) +
                       0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(i + 1));
            h ^= mix64(static_cast<uint64_t>(p.pts[i].second) +
                       0xbf58476d1ce4e5b9ULL * static_cast<uint64_t>(i + 1));
        }
        return static_cast<size_t>(h);
    }
};

enum class RegionGeomError : uint8_t {
    Ok = 0,
    EntryOutsideSource = 1,
    PortalMismatch = 2,
};

RegionGeomError compute_portal_paths_core(
    const Box64& src_rect, const Box64& tgt_rect,
    bool axis_is_x, int64_t direction, int64_t boundary, int64_t lo, int64_t hi,
    int64_t px, int64_t py,
    SmallPath out_paths[2], int& out_n) noexcept;

RegionGeomError compute_target_paths_core(
    const Box64& src_rect, const Box64& tgt_rect,
    int64_t px, int64_t py,
    SmallPath out_paths[2], int& out_n) noexcept;

inline bool settled_axis_prefix_core(
    int64_t ax, int64_t ay, int64_t bx, int64_t by,
    const std::vector<int64_t>& xs, const std::vector<int64_t>& ys,
    int64_t& out_x, int64_t& out_y) noexcept {
    if (ax == bx && ay == by) return false;
    const bool horizontal = (ay == by);
    const int64_t first = horizontal ? ax : ay;
    const int64_t last = horizontal ? bx : by;
    const auto& axes = horizontal ? xs : ys;
    if (axes.empty()) return false;
    if (last > first) {
        auto it = std::upper_bound(axes.begin(), axes.end(), last);
        if (it == axes.begin()) return false;
        --it;
        if (*it <= first) return false;
        out_x = horizontal ? *it : ax;
        out_y = horizontal ? ay : *it;
        return true;
    }
    auto it = std::lower_bound(axes.begin(), axes.end(), last);
    if (it == axes.end() || *it >= first) return false;
    out_x = horizontal ? *it : ax;
    out_y = horizontal ? ay : *it;
    return true;
}

}  // namespace clay_pnr::route
