// Pure C++20 exact integer Difference-Constraint 1D & 2D Compactors.
// Strictly free of <Python.h>.

#include "clay_pnr/place/compactor.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <queue>
#include <utility>

namespace clay_pnr::place {

namespace {

inline bool add_overflows_i64(int64_t a, int64_t b, int64_t& out) noexcept {
    if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<int64_t>::min() - b)) {
        return true;
    }
    out = a + b;
    return false;
}

inline int64_t ceil_to_grid(int64_t v, int64_t grid) noexcept {
    if (grid <= 1) return v;
    const int64_t rem = v % grid;
    if (rem == 0) return v;
    if (v > 0) return v + (grid - rem);
    return v - rem;
}

}  // namespace

CompactAxisResult compact_axis_dag(const std::vector<int64_t>& extents,
                                   const std::vector<IntDiffEdge>& edges) {
    CompactAxisResult res{};
    const size_t n = extents.size();
    if (n == 0) {
        res.status = CompactStatus::InvalidInput;
        res.message = "compaction requires at least one block";
        return res;
    }
    for (size_t i = 0; i < n; ++i) {
        if (extents[i] <= 0) {
            res.status = CompactStatus::InvalidInput;
            res.message = "block extents must be positive integers";
            return res;
        }
    }

    std::vector<uint32_t> out_deg(n, 0);
    std::vector<uint32_t> indeg(n, 0);
    for (const auto& e : edges) {
        if (e.before >= n || e.after >= n) {
            res.status = CompactStatus::InvalidInput;
            res.message = "difference constraint names an out-of-range block";
            return res;
        }
        if (e.before == e.after) {
            res.status = CompactStatus::CycleDetected;
            res.message = "difference-constraint topology must be acyclic (self-loop)";
            return res;
        }
        if (e.min_delta < 0) {
            res.status = CompactStatus::InvalidInput;
            res.message = "DAG compaction requires non-negative min_delta";
            return res;
        }
        ++out_deg[e.before];
        ++indeg[e.after];
    }

    // CSR adjacency list for cache-friendly O(V + E) Kahn pass.
    std::vector<uint32_t> head(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        head[i + 1] = head[i] + out_deg[i];
    }
    struct EdgeSlot {
        uint32_t to;
        int64_t delta;
    };
    std::vector<uint32_t> cur = head;
    std::vector<EdgeSlot> csr(edges.size());
    for (const auto& e : edges) {
        csr[cur[e.before]++] = EdgeSlot{e.after, e.min_delta};
    }

    std::vector<int64_t> pos(n, 0);
    std::vector<uint32_t> queue;
    queue.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (indeg[i] == 0) {
            queue.push_back(static_cast<uint32_t>(i));
        }
    }

    size_t q_head = 0;
    while (q_head < queue.size()) {
        const uint32_t u = queue[q_head++];
        const int64_t pu = pos[u];
        const uint32_t e0 = head[u];
        const uint32_t e1 = head[u + 1];
        for (uint32_t ei = e0; ei < e1; ++ei) {
            const EdgeSlot& s = csr[ei];
            int64_t cand = 0;
            if (add_overflows_i64(pu, s.delta, cand)) {
                res.status = CompactStatus::Overflow;
                res.message = "integer overflow during longest-path compaction";
                return res;
            }
            if (cand > pos[s.to]) {
                pos[s.to] = cand;
            }
            if (--indeg[s.to] == 0) {
                queue.push_back(s.to);
            }
        }
    }

    if (queue.size() != n) {
        res.status = CompactStatus::CycleDetected;
        res.message = "difference-constraint topology must be acyclic";
        return res;
    }

    int64_t extent = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t hi = 0;
        if (add_overflows_i64(pos[i], extents[i], hi)) {
            res.status = CompactStatus::Overflow;
            res.message = "integer overflow computing compacted extent";
            return res;
        }
        if (hi > extent) extent = hi;
    }

    res.status = CompactStatus::Ok;
    res.positions = std::move(pos);
    res.extent = extent;
    return res;
}

CompactAxisResult compact_axis_general(const std::vector<int64_t>& extents,
                                       const std::vector<IntDiffEdge>& edges,
                                       const std::vector<int64_t>& lower_bounds) {
    CompactAxisResult res{};
    const size_t n = extents.size();
    if (n == 0) {
        res.status = CompactStatus::InvalidInput;
        res.message = "compaction requires at least one block";
        return res;
    }
    if (!lower_bounds.empty() && lower_bounds.size() != n) {
        res.status = CompactStatus::InvalidInput;
        res.message = "lower_bounds size must match extents size";
        return res;
    }
    for (size_t i = 0; i < n; ++i) {
        if (extents[i] <= 0) {
            res.status = CompactStatus::InvalidInput;
            res.message = "block extents must be positive integers";
            return res;
        }
    }

    std::vector<uint32_t> out_deg(n, 0);
    for (const auto& e : edges) {
        if (e.before >= n || e.after >= n) {
            res.status = CompactStatus::InvalidInput;
            res.message = "difference constraint names an out-of-range block";
            return res;
        }
        if (e.before == e.after) {
            if (e.min_delta > 0) {
                res.status = CompactStatus::Infeasible;
                res.message = "positive self-loop makes difference constraints infeasible";
                return res;
            }
            continue;
        }
        ++out_deg[e.before];
    }

    std::vector<uint32_t> head(n + 1, 0);
    for (size_t i = 0; i < n; ++i) head[i + 1] = head[i] + out_deg[i];
    struct EdgeSlot {
        uint32_t to;
        int64_t delta;
    };
    std::vector<uint32_t> cur = head;
    std::vector<EdgeSlot> csr(head[n]);
    for (const auto& e : edges) {
        if (e.before == e.after) continue;
        csr[cur[e.before]++] = EdgeSlot{e.after, e.min_delta};
    }

    std::vector<int64_t> pos(n, 0);
    if (!lower_bounds.empty()) {
        for (size_t i = 0; i < n; ++i) pos[i] = lower_bounds[i];
    }

    std::deque<uint32_t> q;
    std::vector<uint8_t> in_q(n, 1);
    std::vector<uint32_t> relax_count(n, 0);
    for (size_t i = 0; i < n; ++i) q.push_back(static_cast<uint32_t>(i));

    while (!q.empty()) {
        const uint32_t u = q.front();
        q.pop_front();
        in_q[u] = 0;
        const int64_t pu = pos[u];
        for (uint32_t ei = head[u]; ei < head[u + 1]; ++ei) {
            const EdgeSlot& s = csr[ei];
            int64_t cand = 0;
            if (add_overflows_i64(pu, s.delta, cand)) {
                res.status = CompactStatus::Overflow;
                res.message = "integer overflow in general difference-constraint solver";
                return res;
            }
            if (cand > pos[s.to]) {
                pos[s.to] = cand;
                if (++relax_count[s.to] > n) {
                    res.status = CompactStatus::Infeasible;
                    res.message = "positive cycle makes difference constraints infeasible";
                    return res;
                }
                if (!in_q[s.to]) {
                    q.push_back(s.to);
                    in_q[s.to] = 1;
                }
            }
        }
    }

    int64_t extent = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t hi = 0;
        if (add_overflows_i64(pos[i], extents[i], hi)) {
            res.status = CompactStatus::Overflow;
            res.message = "integer overflow computing compacted extent";
            return res;
        }
        if (hi > extent) extent = hi;
    }

    res.status = CompactStatus::Ok;
    res.positions = std::move(pos);
    res.extent = extent;
    return res;
}

Compact2DResult compact_2d_topology(const std::vector<int64_t>& widths,
                                    const std::vector<int64_t>& heights,
                                    const std::vector<IntDiffEdge>& h_edges,
                                    const std::vector<IntDiffEdge>& v_edges,
                                    const std::vector<SymmetryPairSpec>& sym_pairs,
                                    int64_t grid_step) {
    Compact2DResult out{};
    const size_t n = widths.size();
    if (n == 0 || heights.size() != n || grid_step <= 0) {
        out.status = CompactStatus::InvalidInput;
        out.message = "invalid 2D compaction dimensions or grid";
        return out;
    }
    for (const auto& sp : sym_pairs) {
        if (sp.a >= n || sp.b >= n || sp.a == sp.b) {
            out.status = CompactStatus::InvalidInput;
            out.message = "invalid symmetry pair index";
            return out;
        }
    }

    // 1. Vertical compaction: enforce y[a] == y[b] via 0-weight bidirectional arcs.
    std::vector<IntDiffEdge> v_all = v_edges;
    if (grid_step > 1) {
        for (auto& e : v_all) {
            if (e.min_delta > 0) e.min_delta = ceil_to_grid(e.min_delta, grid_step);
        }
    }
    for (const auto& sp : sym_pairs) {
        v_all.push_back(IntDiffEdge{sp.a, sp.b, 0});
        v_all.push_back(IntDiffEdge{sp.b, sp.a, 0});
    }
    const CompactAxisResult v_res = compact_axis_general(heights, v_all);
    if (v_res.status != CompactStatus::Ok) {
        out.status = v_res.status;
        out.message = "vertical compaction failed: " + v_res.message;
        return out;
    }

    // 2. Horizontal compaction:
    //    First solve base horizontal longest path. If symmetry pairs are present,
    //    compute the minimum legal shared axis2 = max_{(a,b)} (2*x[a] + 2*x[b] + w[a] + w[b])
    //    and iteratively tighten mirrored counterparts (plus re-solve H constraints) until
    //    fixed point! Because every update is monotone increasing and bounded, convergence
    //    terminates cleanly or detects infeasibility if a cycle appears.
    std::vector<IntDiffEdge> h_all = h_edges;
    if (grid_step > 1) {
        for (auto& e : h_all) {
            if (e.min_delta > 0) e.min_delta = ceil_to_grid(e.min_delta, grid_step);
        }
    }
    CompactAxisResult h_res = compact_axis_general(widths, h_all);
    if (h_res.status != CompactStatus::Ok) {
        out.status = h_res.status;
        out.message = "horizontal compaction failed: " + h_res.message;
        return out;
    }

    int64_t axis2 = 0;
    if (!sym_pairs.empty()) {
        // Compute required shared parity of axis2: 2*x_a + 2*x_b + w_a + w_b == axis2
        // => (w_a + w_b) mod 2 must agree across all symmetry pairs.
        const int64_t parity = (widths[sym_pairs[0].a] + widths[sym_pairs[0].b]) & 1LL;
        for (size_t i = 1; i < sym_pairs.size(); ++i) {
            const int64_t pi = (widths[sym_pairs[i].a] + widths[sym_pairs[i].b]) & 1LL;
            if (pi != parity) {
                out.status = CompactStatus::Infeasible;
                out.message = "symmetry pairs have incompatible width parity for exact integer axis";
                return out;
            }
        }
        std::vector<int64_t> lb = h_res.positions;
        for (int iter = 0; iter < 64; ++iter) {
            axis2 = 0;
            for (const auto& sp : sym_pairs) {
                const int64_t s = 2 * h_res.positions[sp.a] + 2 * h_res.positions[sp.b] +
                                  widths[sp.a] + widths[sp.b];
                if (s > axis2) axis2 = s;
            }
            if ((axis2 & 1LL) != parity) ++axis2;

            bool changed = false;
            for (const auto& sp : sym_pairs) {
                const int64_t need_sum = (axis2 - widths[sp.a] - widths[sp.b]) / 2;
                const int64_t cur_sum = h_res.positions[sp.a] + h_res.positions[sp.b];
                if (cur_sum < need_sum) {
                    const int64_t diff = need_sum - cur_sum;
                    // Push the right-hand member (or b on tie) to satisfy exact mirror equality
                    // without violating x[a] <= x[b] if a is left of b.
                    const uint32_t right = (h_res.positions[sp.b] >= h_res.positions[sp.a]) ? sp.b : sp.a;
                    const int64_t next_x = ceil_to_grid(h_res.positions[right] + diff, grid_step);
                    if (next_x > lb[right]) {
                        lb[right] = next_x;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
            h_res = compact_axis_general(widths, h_all, lb);
            if (h_res.status != CompactStatus::Ok) {
                out.status = h_res.status;
                out.message = "symmetric horizontal compaction failed: " + h_res.message;
                return out;
            }
        }
        // Final strict verification of symmetry equalities.
        const int64_t ref_axis2 = 2 * h_res.positions[sym_pairs[0].a] + 2 * h_res.positions[sym_pairs[0].b] +
                                  widths[sym_pairs[0].a] + widths[sym_pairs[0].b];
        for (const auto& sp : sym_pairs) {
            const int64_t cur_axis2 = 2 * h_res.positions[sp.a] + 2 * h_res.positions[sp.b] +
                                      widths[sp.a] + widths[sp.b];
            if (cur_axis2 != ref_axis2 || v_res.positions[sp.a] != v_res.positions[sp.b]) {
                out.status = CompactStatus::Infeasible;
                out.message = "symmetric horizontal compaction could not equalize shared mirror axis";
                return out;
            }
        }
        axis2 = ref_axis2;
    }

    out.status = CompactStatus::Ok;
    out.x = std::move(h_res.positions);
    out.y = std::move(v_res.positions);
    out.width = h_res.extent;
    out.height = v_res.extent;
    out.shared_axis2 = axis2;
    return out;
}

}  // namespace clay_pnr::place

