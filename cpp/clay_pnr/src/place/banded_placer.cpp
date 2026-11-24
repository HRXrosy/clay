#include "clay_pnr/place/banded_placer.hpp"
#include "clay_pnr/place/compactor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>

namespace clay_pnr::place {

namespace {

inline int64_t ceil_grid(int64_t v, int64_t g) noexcept {
    if (g <= 1) return v;
    const int64_t r = v % g;
    if (r == 0) return v;
    return (v >= 0) ? (v + (g - r)) : (v - r);
}

inline int64_t floor_grid(int64_t v, int64_t g) noexcept {
    if (g <= 1) return v;
    const int64_t r = v % g;
    if (r == 0) return v;
    return (v >= 0) ? (v - r) : (v - (g + r));
}

inline int64_t ceil_div_i64(int64_t num, int64_t den) noexcept {
    if (den <= 0 || num <= 0) return 0;
    return (num + den - 1) / den;
}

inline uint64_t splitmix64(uint64_t& s) noexcept {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct GapTable {
    size_t n = 0;
    int64_t base_gap = 0;
    std::vector<int64_t> mat;

    void init(const BandedProblem& prob) {
        n = prob.blocks.size();
        base_gap = std::max<int64_t>(0, prob.gap_nm);
        mat.assign(n * n, base_gap);
        for (size_t i = 0; i < n; ++i) mat[i * n + i] = 0;
        for (const auto& pc : prob.pair_clearances) {
            if (pc.a >= 0 && pc.b >= 0 &&
                static_cast<size_t>(pc.a) < n && static_cast<size_t>(pc.b) < n &&
                pc.a != pc.b) {
                const int64_t g = std::max(base_gap, pc.gap_nm);
                const size_t ia = static_cast<size_t>(pc.a);
                const size_t ib = static_cast<size_t>(pc.b);
                mat[ia * n + ib] = std::max(mat[ia * n + ib], g);
                mat[ib * n + ia] = std::max(mat[ib * n + ia], g);
            }
        }
    }

    inline int64_t get(size_t i, size_t j) const noexcept {
        return mat[i * n + j];
    }
};

struct EvalMetrics {
    int64_t W = 0;
    int64_t H = 0;
    int64_t primary = 0;
    int64_t soft_cost = 0;
    bool legal = false;
};

void compute_bbox(
    const BandedProblem& prob,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    int64_t& out_w,
    int64_t& out_h) noexcept {
    const size_t n = prob.blocks.size();
    int64_t max_x = 0;
    int64_t max_y = 0;
    for (size_t i = 0; i < n; ++i) {
        max_x = std::max(max_x, xs[i] + prob.blocks[i].w);
        max_y = std::max(max_y, ys[i] + prob.blocks[i].h);
    }
    if (prob.exact_outline_width && prob.w_cap_nm > 0) {
        max_x = std::max(max_x, prob.w_cap_nm);
    }
    out_w = max_x;
    out_h = max_y;
}

int64_t compute_soft_cost(
    const BandedProblem& prob,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    int64_t W) noexcept {
    int64_t cost = 0;
    const size_t n = prob.blocks.size();
    for (size_t i = 0; i < n; ++i) {
        const auto& b = prob.blocks[i];
        if (b.left_pull) cost += xs[i] * 2;
        if (b.right_pull) cost += std::max<int64_t>(0, W - (xs[i] + b.w)) * 2;
    }
    for (const auto& pp : prob.pull_pairs) {
        if (pp.first >= 0 && pp.second >= 0 &&
            static_cast<size_t>(pp.first) < n && static_cast<size_t>(pp.second) < n) {
            const size_t a = static_cast<size_t>(pp.first);
            const size_t b = static_cast<size_t>(pp.second);
            const int64_t ax = xs[a] * 2 + prob.blocks[a].w;
            const int64_t bx = xs[b] * 2 + prob.blocks[b].w;
            const int64_t ay = ys[a] * 2 + prob.blocks[a].h;
            const int64_t by = ys[b] * 2 + prob.blocks[b].h;
            cost += (std::abs(ax - bx) + std::abs(ay - by)) / 2;
        }
    }
    return cost;
}

bool verify_with_gaps(
    const BandedProblem& prob,
    const GapTable& gaps,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    std::string* err) {
    const size_t n = prob.blocks.size();
    if (xs.size() != n || ys.size() != n) {
        if (err) *err = "coordinate vector size mismatch";
        return false;
    }
    const int64_t g = std::max<int64_t>(1, prob.grid_nm);
    int64_t W = 0, H = 0;
    compute_bbox(prob, xs, ys, W, H);

    if (prob.w_cap_nm > 0 && W > prob.w_cap_nm) {
        if (err) *err = "width exceeds w_cap_nm";
        return false;
    }
    if (prob.h_bound_nm > 0 && H > prob.h_bound_nm) {
        if (err) *err = "height exceeds h_bound_nm";
        return false;
    }
    if (prob.exact_outline_width && prob.w_cap_nm > 0 && W != prob.w_cap_nm) {
        if (err) *err = "width violates exact_outline_width";
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const auto& b = prob.blocks[i];
        if (xs[i] < 0 || ys[i] < 0) {
            if (err) *err = "negative coordinate";
            return false;
        }
        if (g > 1 && ((xs[i] % g) != 0 || (ys[i] % g) != 0)) {
            if (err) *err = "off manufacturing grid";
            return false;
        }
        if (ys[i] < b.y_min_keepout) {
            if (err) *err = "violates y_min_keepout";
            return false;
        }
        if (b.top_keepout_clear && prob.top_edge_keepout_nm > 0 &&
            ys[i] + b.h > H - prob.top_edge_keepout_nm) {
            if (err) *err = "violates top_edge_keepout";
            return false;
        }
        if (b.left_edge && xs[i] != 0) {
            if (err) *err = "violates left_edge anchor";
            return false;
        }
        if (b.right_edge && xs[i] + b.w != W) {
            if (err) *err = "violates right_edge anchor";
            return false;
        }
    }
    // Pairwise non-overlap & band order
    for (size_t i = 0; i < n; ++i) {
        const auto& bi = prob.blocks[i];
        for (size_t j = i + 1; j < n; ++j) {
            const auto& bj = prob.blocks[j];
            const int64_t gij = gaps.get(i, j);
            const bool sep_x = (xs[i] + bi.w + gij <= xs[j]) || (xs[j] + bj.w + gij <= xs[i]);
            const bool sep_y = (ys[i] + bi.h + gij <= ys[j]) || (ys[j] + bj.h + gij <= ys[i]);
            if (!sep_x && !sep_y) {
                if (err) *err = "pairwise overlap or clearance violation";
                return false;
            }
            if (prob.enforce_band_order && !bi.lateral && !bj.lateral && bi.band != bj.band) {
                if (bi.band < bj.band) {
                    if (ys[i] + bi.h + gij > ys[j]) {
                        if (err) *err = "band vertical ordering violated";
                        return false;
                    }
                } else {
                    if (ys[j] + bj.h + gij > ys[i]) {
                        if (err) *err = "band vertical ordering violated";
                        return false;
                    }
                }
            }
        }
    }
    // Horizontal order pairs
    for (const auto& op : prob.order_pairs) {
        const size_t a = static_cast<size_t>(op.first);
        const size_t b = static_cast<size_t>(op.second);
        if (a < n && b < n) {
            const int64_t gab = gaps.get(a, b);
            if (xs[a] + prob.blocks[a].w + gab > xs[b]) {
                if (err) *err = "horizontal order_pair violated";
                return false;
            }
        }
    }
    // Vertical order pairs
    for (const auto& vp : prob.vertical_order_pairs) {
        const size_t a = static_cast<size_t>(vp.first);
        const size_t b = static_cast<size_t>(vp.second);
        if (a < n && b < n) {
            const int64_t gab = gaps.get(a, b);
            if (ys[a] + prob.blocks[a].h + gab > ys[b]) {
                if (err) *err = "vertical order_pair violated";
                return false;
            }
        }
    }
    // Exact scaled aspect ratio check (using 128-bit safe comparison)
    if (prob.aspect_obligation && prob.aspect_scale > 0 && H > 0) {
        const auto lhs = core::mul_u64(static_cast<uint64_t>(W), static_cast<uint64_t>(prob.aspect_scale));
        const auto lo_rhs = core::mul_u64(static_cast<uint64_t>(H), static_cast<uint64_t>(prob.aspect_lower));
        const auto hi_rhs = core::mul_u64(static_cast<uint64_t>(H), static_cast<uint64_t>(prob.aspect_upper));
        if (lhs < lo_rhs || hi_rhs < lhs) {
            if (err) *err = "aspect_obligation violated";
            return false;
        }
    }
    return true;
}

// Topological sort of subset `nodes` using directed edges + priority tie-breaker.
std::vector<int32_t> topo_sort_subset(
    const std::vector<int32_t>& nodes,
    const std::vector<std::vector<int32_t>>& adj,
    const std::vector<int64_t>& priority) {
    const size_t m = nodes.size();
    if (m <= 1) return nodes;
    std::vector<int32_t> in_subset(adj.size(), -1);
    for (size_t i = 0; i < m; ++i) in_subset[static_cast<size_t>(nodes[i])] = static_cast<int32_t>(i);
    std::vector<int32_t> indeg(m, 0);
    for (size_t i = 0; i < m; ++i) {
        const int32_t u = nodes[i];
        for (int32_t v : adj[static_cast<size_t>(u)]) {
            const int32_t li = in_subset[static_cast<size_t>(v)];
            if (li >= 0) ++indeg[static_cast<size_t>(li)];
        }
    }
    auto cmp = [&](int32_t a, int32_t b) {
        if (priority[static_cast<size_t>(a)] != priority[static_cast<size_t>(b)]) {
            return priority[static_cast<size_t>(a)] > priority[static_cast<size_t>(b)];
        }
        return a > b;
    };
    std::priority_queue<int32_t, std::vector<int32_t>, decltype(cmp)> pq(cmp);
    for (size_t i = 0; i < m; ++i) {
        if (indeg[i] == 0) pq.push(nodes[i]);
    }
    std::vector<int32_t> order;
    order.reserve(m);
    while (!pq.empty()) {
        const int32_t u = pq.top();
        pq.pop();
        order.push_back(u);
        for (int32_t v : adj[static_cast<size_t>(u)]) {
            const int32_t li = in_subset[static_cast<size_t>(v)];
            if (li >= 0 && --indeg[static_cast<size_t>(li)] == 0) {
                pq.push(v);
            }
        }
    }
    if (order.size() < m) {
        // Fallback: append any remaining nodes (in case of cycle in order_pairs)
        std::vector<uint8_t> seen(adj.size(), 0);
        for (int32_t u : order) seen[static_cast<size_t>(u)] = 1;
        for (int32_t u : nodes) {
            if (!seen[static_cast<size_t>(u)]) order.push_back(u);
        }
    }
    return order;
}

// 2D DAG compaction pass preserving pairwise separation and all hard constraints.
bool compact_xy_pass(
    const BandedProblem& prob,
    const GapTable& gaps,
    std::vector<int64_t>& xs,
    std::vector<int64_t>& ys,
    bool x_first) {
    const size_t n = prob.blocks.size();
    const int64_t g = std::max<int64_t>(1, prob.grid_nm);

    auto compact_x = [&]() -> bool {
        std::vector<int64_t> exts(n);
        for (size_t i = 0; i < n; ++i) exts[i] = prob.blocks[i].w;
        std::vector<IntDiffEdge> edges;
        edges.reserve(n * 4);
        for (const auto& op : prob.order_pairs) {
            const size_t a = static_cast<size_t>(op.first);
            const size_t b = static_cast<size_t>(op.second);
            if (a < n && b < n) {
                edges.push_back(IntDiffEdge{
                    static_cast<uint32_t>(op.first), static_cast<uint32_t>(op.second),
                    ceil_grid(prob.blocks[a].w + gaps.get(a, b), g)});
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const auto& bi = prob.blocks[i];
            for (size_t j = i + 1; j < n; ++j) {
                const auto& bj = prob.blocks[j];
                const int64_t gij = gaps.get(i, j);
                const bool y_sep = (ys[i] + bi.h + gij <= ys[j]) || (ys[j] + bj.h + gij <= ys[i]);
                if (!y_sep) {
                    if (xs[i] + bi.w + gij <= xs[j]) {
                        edges.push_back(IntDiffEdge{
                            static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                            ceil_grid(bi.w + gij, g)});
                    } else if (xs[j] + bj.w + gij <= xs[i]) {
                        edges.push_back(IntDiffEdge{
                            static_cast<uint32_t>(j), static_cast<uint32_t>(i),
                            ceil_grid(bj.w + gij, g)});
                    }
                }
            }
        }
        const auto res = compact_axis_dag(exts, edges);
        if (res.status != CompactStatus::Ok) return false;
        std::vector<int64_t> nx(n);
        for (size_t i = 0; i < n; ++i) {
            nx[i] = ceil_grid(res.positions[i], g);
            if (prob.blocks[i].left_edge) nx[i] = 0;
        }
        // Align right_edge blocks to target W
        int64_t W = prob.exact_outline_width ? prob.w_cap_nm : 0;
        for (size_t i = 0; i < n; ++i) W = std::max(W, nx[i] + prob.blocks[i].w);
        if (g > 1) W = ceil_grid(W, g);
        for (size_t i = 0; i < n; ++i) {
            if (prob.blocks[i].right_edge) {
                const int64_t target = W - prob.blocks[i].w;
                if (target >= nx[i] && (target % g) == 0) {
                    nx[i] = target;
                }
            }
        }
        xs = std::move(nx);
        return true;
    };

    auto compact_y = [&]() -> bool {
        // Use longest path with per-node lower bounds y_min_keepout.
        std::vector<int64_t> ny(n, 0);
        for (size_t i = 0; i < n; ++i) {
            ny[i] = ceil_grid(std::max<int64_t>(0, prob.blocks[i].y_min_keepout), g);
        }
        std::vector<std::vector<std::pair<int32_t, int64_t>>> adj(n);
        std::vector<int32_t> indeg(n, 0);
        auto add_edge = [&](size_t u, size_t v, int64_t d) {
            adj[u].emplace_back(static_cast<int32_t>(v), ceil_grid(d, g));
            ++indeg[v];
        };
        for (const auto& vp : prob.vertical_order_pairs) {
            const size_t a = static_cast<size_t>(vp.first);
            const size_t b = static_cast<size_t>(vp.second);
            if (a < n && b < n) add_edge(a, b, prob.blocks[a].h + gaps.get(a, b));
        }
        for (size_t i = 0; i < n; ++i) {
            const auto& bi = prob.blocks[i];
            for (size_t j = i + 1; j < n; ++j) {
                const auto& bj = prob.blocks[j];
                const int64_t gij = gaps.get(i, j);
                if (prob.enforce_band_order && !bi.lateral && !bj.lateral && bi.band != bj.band) {
                    if (bi.band < bj.band) add_edge(i, j, bi.h + gij);
                    else add_edge(j, i, bj.h + gij);
                    continue;
                }
                const bool x_sep = (xs[i] + bi.w + gij <= xs[j]) || (xs[j] + bj.w + gij <= xs[i]);
                if (!x_sep) {
                    if (ys[i] + bi.h + gij <= ys[j]) add_edge(i, j, bi.h + gij);
                    else if (ys[j] + bj.h + gij <= ys[i]) add_edge(j, i, bj.h + gij);
                }
            }
        }
        std::queue<size_t> q;
        for (size_t i = 0; i < n; ++i) {
            if (indeg[i] == 0) q.push(i);
        }
        size_t visited = 0;
        while (!q.empty()) {
            const size_t u = q.front();
            q.pop();
            ++visited;
            for (const auto& [v, d] : adj[u]) {
                const size_t sv = static_cast<size_t>(v);
                ny[sv] = std::max(ny[sv], ceil_grid(ny[u] + d, g));
                if (--indeg[sv] == 0) q.push(sv);
            }
        }
        if (visited != n) return false;
        ys = std::move(ny);
        return true;
    };

    if (x_first) {
        return compact_x() && compact_y();
    }
    return compact_y() && compact_x();
}

// Construct one candidate banded placement for given intra-band orders and target width.
bool pack_banded_candidate(
    const BandedProblem& prob,
    const GapTable& gaps,
    const std::vector<int32_t>& band_ids,
    const std::vector<std::vector<int32_t>>& band_orders,
    const std::vector<int32_t>& lateral_order,
    int64_t target_width,
    bool allow_wrap,
    std::vector<int64_t>& out_x,
    std::vector<int64_t>& out_y) {
    const size_t n = prob.blocks.size();
    const int64_t g = std::max<int64_t>(1, prob.grid_nm);
    out_x.assign(n, 0);
    out_y.assign(n, 0);

    // Precompute incoming horizontal order constraints to preserve x_a + w_a + g <= x_b.
    std::vector<std::vector<int32_t>> h_preds(n);
    for (const auto& op : prob.order_pairs) {
        if (op.first >= 0 && op.second >= 0 &&
            static_cast<size_t>(op.first) < n && static_cast<size_t>(op.second) < n) {
            h_preds[static_cast<size_t>(op.second)].push_back(op.first);
        }
    }
    std::vector<std::vector<int32_t>> v_preds(n);
    for (const auto& vp : prob.vertical_order_pairs) {
        if (vp.first >= 0 && vp.second >= 0 &&
            static_cast<size_t>(vp.first) < n && static_cast<size_t>(vp.second) < n) {
            v_preds[static_cast<size_t>(vp.second)].push_back(vp.first);
        }
    }

    std::vector<uint8_t> placed(n, 0);
    std::vector<int32_t> placed_list;
    placed_list.reserve(n);

    int64_t band_y_cursor = 0;

    for (size_t bi = 0; bi < band_ids.size(); ++bi) {
        const auto& order = band_orders[bi];
        if (order.empty()) continue;

        // Track shelf rows inside this band: (row_y, row_top, row_x_cursor, members)
        struct ShelfRow {
            int64_t y0 = 0;
            int64_t y1 = 0;
            int64_t x_cur = 0;
            std::vector<int32_t> members;
        };
        std::vector<ShelfRow> shelves;
        shelves.push_back(ShelfRow{band_y_cursor, band_y_cursor, 0, {}});

        for (int32_t idx : order) {
            const size_t u = static_cast<size_t>(idx);
            const auto& blk = prob.blocks[u];

            // Compute minimum x from placed horizontal predecessors
            int64_t min_x_ord = 0;
            for (int32_t p : h_preds[u]) {
                if (placed[static_cast<size_t>(p)]) {
                    min_x_ord = std::max(
                        min_x_ord,
                        out_x[static_cast<size_t>(p)] + prob.blocks[static_cast<size_t>(p)].w +
                            gaps.get(static_cast<size_t>(p), u));
                }
            }
            // Compute minimum y from keepout, band floor, and placed vertical predecessors
            int64_t min_y_hard = std::max(band_y_cursor, blk.y_min_keepout);
            for (int32_t p : v_preds[u]) {
                if (placed[static_cast<size_t>(p)]) {
                    min_y_hard = std::max(
                        min_y_hard,
                        out_y[static_cast<size_t>(p)] + prob.blocks[static_cast<size_t>(p)].h +
                            gaps.get(static_cast<size_t>(p), u));
                }
            }

            // Pick best shelf (first shelf where block fits within target_width, or open new shelf if allow_wrap)
            int chosen_shelf = -1;
            int64_t chosen_x = 0;
            int64_t chosen_y = 0;

            for (size_t si = 0; si < shelves.size(); ++si) {
                const auto& sh = shelves[si];
                int64_t cx = std::max(sh.x_cur, min_x_ord);
                if (!sh.members.empty()) {
                    const size_t prev = static_cast<size_t>(sh.members.back());
                    cx = std::max(cx, out_x[prev] + prob.blocks[prev].w + gaps.get(prev, u));
                }
                cx = ceil_grid(cx, g);
                int64_t cy = ceil_grid(std::max(sh.y0, min_y_hard), g);

                // Ensure clearance against all already-placed blocks in earlier shelves of this band
                for (size_t sj = 0; sj < si; ++sj) {
                    for (int32_t prev_id : shelves[sj].members) {
                        const size_t pv = static_cast<size_t>(prev_id);
                        const int64_t gpv = gaps.get(pv, u);
                        const bool x_sep = (cx + blk.w + gpv <= out_x[pv]) ||
                                           (out_x[pv] + prob.blocks[pv].w + gpv <= cx);
                        if (!x_sep) {
                            cy = std::max(cy, ceil_grid(out_y[pv] + prob.blocks[pv].h + gpv, g));
                        }
                    }
                }
                // Check against members on the same shelf too (in case of pair_clearance > base)
                for (int32_t prev_id : sh.members) {
                    const size_t pv = static_cast<size_t>(prev_id);
                    const int64_t gpv = gaps.get(pv, u);
                    const bool y_sep = (cy + blk.h + gpv <= out_y[pv]) ||
                                       (out_y[pv] + prob.blocks[pv].h + gpv <= cy);
                    if (!y_sep) {
                        cx = std::max(cx, ceil_grid(out_x[pv] + prob.blocks[pv].w + gpv, g));
                    }
                }

                if (!allow_wrap || cx + blk.w <= target_width || sh.members.empty()) {
                    chosen_shelf = static_cast<int>(si);
                    chosen_x = cx;
                    chosen_y = cy;
                    break;
                }
            }

            if (chosen_shelf < 0) {
                // Start a new shelf inside this band
                int64_t next_y = min_y_hard;
                for (const auto& sh : shelves) {
                    for (int32_t prev_id : sh.members) {
                        const size_t pv = static_cast<size_t>(prev_id);
                        next_y = std::max(next_y, out_y[pv] + prob.blocks[pv].h + gaps.get(pv, u));
                    }
                }
                next_y = ceil_grid(next_y, g);
                const int64_t cx = ceil_grid(min_x_ord, g);
                shelves.push_back(ShelfRow{next_y, next_y + blk.h, cx + blk.w, {idx}});
                chosen_x = cx;
                chosen_y = next_y;
            } else {
                auto& sh = shelves[static_cast<size_t>(chosen_shelf)];
                sh.x_cur = std::max(sh.x_cur, chosen_x + blk.w);
                sh.y1 = std::max(sh.y1, chosen_y + blk.h);
                sh.members.push_back(idx);
            }

            out_x[u] = chosen_x;
            out_y[u] = chosen_y;
            placed[u] = 1;
            placed_list.push_back(idx);
        }

        // Advance band_y_cursor above all blocks in this band respecting cross-band gaps
        int64_t next_band_y = band_y_cursor;
        for (int32_t idx : order) {
            const size_t u = static_cast<size_t>(idx);
            int64_t max_g = gaps.base_gap;
            if (bi + 1 < band_ids.size()) {
                for (int32_t nxt : band_orders[bi + 1]) {
                    max_g = std::max(max_g, gaps.get(u, static_cast<size_t>(nxt)));
                }
            }
            next_band_y = std::max(next_band_y, ceil_grid(out_y[u] + prob.blocks[u].h + max_g, g));
        }
        band_y_cursor = next_band_y;
    }

    // Place lateral blocks to the right of core (or at min_x_ord) respecting clearances.
    int64_t core_right = 0;
    for (int32_t idx : placed_list) {
        const size_t u = static_cast<size_t>(idx);
        core_right = std::max(core_right, out_x[u] + prob.blocks[u].w);
    }
    int64_t lat_x = placed_list.empty() ? 0 : ceil_grid(core_right + gaps.base_gap, g);
    for (int32_t idx : lateral_order) {
        const size_t u = static_cast<size_t>(idx);
        const auto& blk = prob.blocks[u];
        int64_t cx = lat_x;
        int64_t cy = ceil_grid(std::max<int64_t>(0, blk.y_min_keepout), g);
        for (int32_t prev_id : placed_list) {
            const size_t pv = static_cast<size_t>(prev_id);
            const int64_t gpv = gaps.get(pv, u);
            const bool y_sep = (cy + blk.h + gpv <= out_y[pv]) ||
                               (out_y[pv] + prob.blocks[pv].h + gpv <= cy);
            if (!y_sep) {
                cx = std::max(cx, ceil_grid(out_x[pv] + prob.blocks[pv].w + gpv, g));
            }
        }
        for (int32_t p : h_preds[u]) {
            if (placed[static_cast<size_t>(p)]) {
                cx = std::max(
                    cx,
                    ceil_grid(out_x[static_cast<size_t>(p)] + prob.blocks[static_cast<size_t>(p)].w +
                                  gaps.get(static_cast<size_t>(p), u),
                              g));
            }
        }
        for (int32_t p : v_preds[u]) {
            if (placed[static_cast<size_t>(p)]) {
                cy = std::max(
                    cy,
                    ceil_grid(out_y[static_cast<size_t>(p)] + prob.blocks[static_cast<size_t>(p)].h +
                                  gaps.get(static_cast<size_t>(p), u),
                              g));
            }
        }
        out_x[u] = cx;
        out_y[u] = cy;
        placed[u] = 1;
        placed_list.push_back(idx);
        lat_x = ceil_grid(cx + blk.w + gaps.base_gap, g);
    }

    // Apply right_edge anchoring if requested
    int64_t W = prob.exact_outline_width ? prob.w_cap_nm : 0;
    for (size_t i = 0; i < n; ++i) W = std::max(W, out_x[i] + prob.blocks[i].w);
    if (g > 1) W = ceil_grid(W, g);
    for (size_t i = 0; i < n; ++i) {
        if (prob.blocks[i].right_edge) {
            const int64_t tx = W - prob.blocks[i].w;
            if (tx >= out_x[i] && (tx % g) == 0) out_x[i] = tx;
        }
    }

    // If aspect_obligation requires slightly more height, lift the highest band cleanly.
    if (prob.aspect_obligation && prob.aspect_scale > 0 && prob.aspect_upper > 0 && !band_ids.empty()) {
        int64_t curW = 0, curH = 0;
        compute_bbox(prob, out_x, out_y, curW, curH);
        const int64_t req_h = ceil_div_i64(curW * prob.aspect_scale, prob.aspect_upper);
        const int64_t max_h = (prob.aspect_lower > 0) ? ((curW * prob.aspect_scale) / prob.aspect_lower) : curH;
        if (curH < req_h) {
            const int64_t pad = ceil_grid(req_h - curH, g);
            if (curH + pad <= max_h) {
                const int32_t top_b = band_ids.back();
                for (size_t i = 0; i < n; ++i) {
                    if (!prob.blocks[i].lateral && prob.blocks[i].band == top_b) {
                        out_y[i] += pad;
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace

bool verify_banded_solution(
    const BandedProblem& prob,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    std::string* err) {
    GapTable gaps;
    gaps.init(prob);
    return verify_with_gaps(prob, gaps, xs, ys, err);
}

BandedSolution solve_banded_area(const BandedProblem& prob) {
    BandedSolution best{};
    const size_t n = prob.blocks.size();
    if (n == 0) {
        best.status = BandedStatus::Optimal;
        best.detail = "empty";
        return best;
    }
    for (const auto& b : prob.blocks) {
        if (b.w <= 0 || b.h <= 0) {
            best.status = BandedStatus::InvalidInput;
            best.detail = "non-positive block dimension";
            return best;
        }
    }

    GapTable gaps;
    gaps.init(prob);
    const int64_t g = std::max<int64_t>(1, prob.grid_nm);

    // Build horizontal DAG adjacency
    std::vector<std::vector<int32_t>> h_adj(n);
    for (const auto& op : prob.order_pairs) {
        if (op.first >= 0 && op.second >= 0 &&
            static_cast<size_t>(op.first) < n && static_cast<size_t>(op.second) < n) {
            h_adj[static_cast<size_t>(op.first)].push_back(op.second);
        }
    }

    // Partition blocks into sorted non-lateral bands + lateral list
    std::map<int32_t, std::vector<int32_t>> band_map;
    std::vector<int32_t> lateral_nodes;
    int64_t sum_area = 0;
    int64_t max_w = 0;
    int64_t sum_w = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_area += prob.blocks[i].w * prob.blocks[i].h;
        max_w = std::max(max_w, prob.blocks[i].w);
        sum_w += prob.blocks[i].w + gaps.base_gap;
        if (prob.blocks[i].lateral) lateral_nodes.push_back(static_cast<int32_t>(i));
        else band_map[prob.blocks[i].band].push_back(static_cast<int32_t>(i));
    }
    std::vector<int32_t> band_ids;
    std::vector<std::vector<int32_t>> band_nodes;
    int64_t min_band_h_sum = 0;
    int64_t max_band_sum_w = max_w;
    for (auto& [bid, nodes] : band_map) {
        band_ids.push_back(bid);
        band_nodes.push_back(nodes);
        int64_t bh = 0;
        int64_t bw = 0;
        for (int32_t u : nodes) {
            bh = std::max(bh, prob.blocks[static_cast<size_t>(u)].h);
            bw += prob.blocks[static_cast<size_t>(u)].w + gaps.base_gap;
        }
        min_band_h_sum += bh;
        max_band_sum_w = std::max(max_band_sum_w, bw);
    }
    if (band_ids.size() > 1 && prob.enforce_band_order) {
        min_band_h_sum += static_cast<int64_t>(band_ids.size() - 1) * gaps.base_gap;
    }

    best.lower_bound = prob.objective_area ? sum_area : min_band_h_sum;

    // Candidate width targets
    const int64_t w_lo = ceil_grid(max_w, g);
    const int64_t w_hi = (prob.w_cap_nm > 0)
                             ? prob.w_cap_nm
                             : ceil_grid(std::max(max_band_sum_w, sum_w), g);
    std::vector<int64_t> width_targets;
    width_targets.push_back(w_hi);
    if (prob.exact_outline_width && prob.w_cap_nm > 0) {
        width_targets.clear();
        width_targets.push_back(prob.w_cap_nm);
    } else {
        const int steps = 24;
        for (int s = 0; s <= steps; ++s) {
            const double frac = static_cast<double>(s) / static_cast<double>(steps);
            const int64_t wt = ceil_grid(
                w_lo + static_cast<int64_t>(static_cast<double>(std::max<int64_t>(0, w_hi - w_lo)) * frac), g);
            if (wt >= w_lo && (prob.w_cap_nm <= 0 || wt <= prob.w_cap_nm)) {
                width_targets.push_back(wt);
            }
        }
        // Add sqrt(area) aspect-guided widths
        const double sq = std::sqrt(static_cast<double>(std::max<int64_t>(1, sum_area)));
        for (double mult : {0.65, 0.8, 1.0, 1.25, 1.5, 1.8, 2.2}) {
            const int64_t wt = ceil_grid(static_cast<int64_t>(sq * mult), g);
            if (wt >= w_lo && (prob.w_cap_nm <= 0 || wt <= prob.w_cap_nm)) {
                width_targets.push_back(wt);
            }
        }
        std::sort(width_targets.begin(), width_targets.end());
        width_targets.erase(std::unique(width_targets.begin(), width_targets.end()), width_targets.end());
    }

    // Build deterministic heuristic priority strategies
    std::vector<std::vector<int64_t>> strategies;
    // 0: original order (edge-anchor aware)
    {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = static_cast<int64_t>(i);
            if (prob.blocks[i].left_edge || prob.blocks[i].left_pull) v -= 1000000;
            if (prob.blocks[i].right_edge || prob.blocks[i].right_pull) v += 1000000;
            p[i] = v;
        }
        strategies.push_back(std::move(p));
    }
    // 1: height descending (best for shelf/banded packing)
    {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = -prob.blocks[i].h;
            if (prob.blocks[i].left_edge) v -= 1000000000LL;
            if (prob.blocks[i].right_edge) v += 1000000000LL;
            p[i] = v;
        }
        strategies.push_back(std::move(p));
    }
    // 2: width descending
    {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = -prob.blocks[i].w;
            if (prob.blocks[i].left_edge) v -= 1000000000LL;
            if (prob.blocks[i].right_edge) v += 1000000000LL;
            p[i] = v;
        }
        strategies.push_back(std::move(p));
    }
    // 3: area descending
    {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = -(prob.blocks[i].w * prob.blocks[i].h);
            if (prob.blocks[i].left_edge) v -= 1000000000000LL;
            if (prob.blocks[i].right_edge) v += 1000000000000LL;
            p[i] = v;
        }
        strategies.push_back(std::move(p));
    }
    // 4: height ascending
    {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = prob.blocks[i].h;
            if (prob.blocks[i].left_edge) v -= 1000000000LL;
            if (prob.blocks[i].right_edge) v += 1000000000LL;
            p[i] = v;
        }
        strategies.push_back(std::move(p));
    }

    bool have_best = false;
    uint64_t evals = 0;
    const uint64_t max_evals = static_cast<uint64_t>(std::max<int32_t>(32, prob.max_evals));

    auto consider = [&](const std::vector<int64_t>& cx, const std::vector<int64_t>& cy) {
        ++evals;
        if (!verify_with_gaps(prob, gaps, cx, cy, nullptr)) return;
        int64_t W = 0, H = 0;
        compute_bbox(prob, cx, cy, W, H);
        const int64_t prim = prob.objective_area ? (W * H) : H;
        const int64_t sc = compute_soft_cost(prob, cx, cy, W);
        if (!have_best || prim < best.primary ||
            (prim == best.primary && (sc < best.soft_cost ||
                                      (sc == best.soft_cost && (H < best.H || (H == best.H && W < best.W)))))) {
            have_best = true;
            best.status = (prim <= best.lower_bound) ? BandedStatus::Optimal : BandedStatus::Feasible;
            best.xs = cx;
            best.ys = cy;
            best.W = W;
            best.H = H;
            best.primary = prim;
            best.soft_cost = sc;
        }
    };

    auto eval_config = [&](const std::vector<std::vector<int32_t>>& b_orders,
                           const std::vector<int32_t>& l_order,
                           int64_t wt,
                           bool allow_wrap) {
        std::vector<int64_t> cx, cy;
        if (!pack_banded_candidate(prob, gaps, band_ids, b_orders, l_order, wt, allow_wrap, cx, cy)) return;
        consider(cx, cy);
        // Try both X-then-Y and Y-then-X DAG compaction passes
        std::vector<int64_t> cx1 = cx, cy1 = cy;
        if (compact_xy_pass(prob, gaps, cx1, cy1, true)) consider(cx1, cy1);
        std::vector<int64_t> cx2 = cx, cy2 = cy;
        if (compact_xy_pass(prob, gaps, cx2, cy2, false)) consider(cx2, cy2);
    };

    std::vector<std::vector<int32_t>> best_band_orders;
    std::vector<int32_t> best_lat_order;
    int64_t best_wt = w_hi;
    bool best_wrap = prob.dense_2d;

    // Phase 1: Structured deterministic sweep across strategies x widths x wrap modes
    for (const auto& prio : strategies) {
        std::vector<std::vector<int32_t>> b_orders(band_ids.size());
        for (size_t bi = 0; bi < band_ids.size(); ++bi) {
            b_orders[bi] = topo_sort_subset(band_nodes[bi], h_adj, prio);
        }
        const auto l_order = topo_sort_subset(lateral_nodes, h_adj, prio);

        // Single-row per band pass
        eval_config(b_orders, l_order, w_hi, false);
        if (have_best && best_band_orders.empty()) {
            best_band_orders = b_orders;
            best_lat_order = l_order;
            best_wt = w_hi;
            best_wrap = false;
        }

        // Multi-row dense 2D width sweep
        if (prob.dense_2d || prob.objective_area) {
            for (int64_t wt : width_targets) {
                if (evals >= max_evals) break;
                const int64_t prev_prim = have_best ? best.primary : std::numeric_limits<int64_t>::max();
                eval_config(b_orders, l_order, wt, true);
                if (have_best && best.primary < prev_prim) {
                    best_band_orders = b_orders;
                    best_lat_order = l_order;
                    best_wt = wt;
                    best_wrap = true;
                }
            }
        }
        if (evals >= max_evals) break;
    }

    // Phase 2: Deterministic intra-band swap & width perturbation local search
    uint64_t rng = prob.seed;
    if (!best_band_orders.empty()) {
        std::vector<std::vector<int32_t>> cur_orders = best_band_orders;
        int64_t cur_wt = best_wt;
        while (evals + 3 <= max_evals && best.status != BandedStatus::Optimal) {
            auto cand_orders = cur_orders;
            int64_t cand_wt = cur_wt;
            const uint64_t action = splitmix64(rng) % 5;
            if (action == 0 && !width_targets.empty()) {
                cand_wt = width_targets[static_cast<size_t>(splitmix64(rng) % width_targets.size())];
            } else {
                // Swap two elements inside a band with >= 2 blocks (re-toposorted if order_pairs exist)
                const size_t bi = static_cast<size_t>(splitmix64(rng) % band_ids.size());
                if (cand_orders[bi].size() >= 2) {
                    const size_t sz = cand_orders[bi].size();
                    const size_t a = static_cast<size_t>(splitmix64(rng) % sz);
                    const size_t b = static_cast<size_t>(splitmix64(rng) % sz);
                    if (a != b) {
                        std::swap(cand_orders[bi][a], cand_orders[bi][b]);
                        if (!prob.order_pairs.empty()) {
                            std::vector<int64_t> prio(n, 0);
                            for (size_t k = 0; k < sz; ++k) {
                                prio[static_cast<size_t>(cand_orders[bi][k])] = static_cast<int64_t>(k);
                            }
                            cand_orders[bi] = topo_sort_subset(band_nodes[bi], h_adj, prio);
                        }
                    }
                }
            }
            const int64_t prev_prim = best.primary;
            eval_config(cand_orders, best_lat_order, cand_wt, best_wrap || prob.dense_2d);
            if (best.primary < prev_prim) {
                cur_orders = std::move(cand_orders);
                cur_wt = cand_wt;
            }
        }
    }

    best.evaluations = evals;
    if (!have_best) {
        best.status = BandedStatus::Infeasible;
        best.detail = "no legal banded placement found within constraints";
    } else {
        best.detail = (best.status == BandedStatus::Optimal) ? "optimal" : "feasible";
    }
    return best;
}

}  // namespace clay_pnr::place
