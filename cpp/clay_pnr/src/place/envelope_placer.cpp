// Pure C++20 Exact-Constraint 2D Envelope-Min Analog Placer.
// Strictly free of <Python.h>, OR-Tools, and KLayout.

#include "clay_pnr/place/envelope_placer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

namespace clay_pnr::place {

namespace {

constexpr int64_t kEnvelopeWeight = 1000;
constexpr int64_t kCenterWeight = 200;
constexpr int64_t kAspectWeight = 10;
constexpr int64_t kAspectDenom = 100;
constexpr int64_t kAbutWeight = 100;
constexpr int64_t kRailAffinityWeight = 150;
constexpr int64_t kAxisTol = 1;
constexpr int32_t kNoBand = std::numeric_limits<int32_t>::min();

// Deterministic SplitMix64 PRNG (zero platform drift across Windows/Linux/macOS).
struct Rng64 {
    uint64_t state;
    explicit Rng64(uint64_t seed) noexcept : state(seed ^ 0x9e3779b97f4a7c15ULL) {}
    uint64_t next() noexcept {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    size_t uniform(size_t bound) noexcept {
        return bound <= 1 ? 0 : static_cast<size_t>(next() % bound);
    }
    double uniform01() noexcept {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

inline bool rects_overlap_strict(int64_t ax0, int64_t ay0, int64_t ax1, int64_t ay1,
                                 int64_t bx0, int64_t by0, int64_t bx1, int64_t by1) noexcept {
    return ax0 < bx1 && bx0 < ax1 && ay0 < by1 && by0 < ay1;
}

inline int64_t abs64(int64_t v) noexcept { return v < 0 ? -v : v; }

// Compute exact integer median minimizing sum_i |x - v_i| (or constrained interval [lo, hi]).
int64_t best_axis2_for_terms(const std::vector<int64_t>& vals, int64_t lo, int64_t hi) {
    if (vals.empty()) return std::clamp<int64_t>(0, lo, hi);
    std::vector<int64_t> s = vals;
    std::sort(s.begin(), s.end());
    const int64_t med = s[s.size() / 2];
    return std::clamp(med, lo, hi);
}

// Compute exact optimal shared axis2 (within [lo, hi]) and total center_terms cost.
int64_t compute_axis_center_cost(const EnvelopeProblem& prob,
                                 const std::vector<int64_t>& xs,
                                 int64_t W,
                                 bool* hard_ok = nullptr) {
    if (hard_ok) *hard_ok = true;
    if (prob.axis_groups.empty()) return 0;

    bool have_primary = false;
    if (prob.axis_centered_soft) {
        for (const auto& ag : prob.axis_groups) {
            if (ag.is_primary) {
                have_primary = true;
                break;
            }
        }
    }

    int64_t lo = 0;
    int64_t hi = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> soft_targets;
    soft_targets.reserve(prob.axis_groups.size() + 1);
    soft_targets.push_back(W);  // cdev = |axis2 - W|

    for (size_t i = 0; i < prob.axis_groups.size(); ++i) {
        const auto& ag = prob.axis_groups[i];
        const int64_t axk = 2 * xs[ag.group] + prob.groups[ag.group].fw + 2 * ag.offset_rounded;
        const bool is_hard = !prob.axis_centered_soft || ag.is_primary || (!have_primary && i == 0);
        if (is_hard) {
            lo = std::max(lo, axk - kAxisTol);
            hi = std::min(hi, axk + kAxisTol);
        } else {
            soft_targets.push_back(axk);
        }
    }
    if (lo > hi) {
        if (hard_ok) *hard_ok = false;
        return std::numeric_limits<int64_t>::max() / 8;
    }

    const int64_t axis2 = best_axis2_for_terms(soft_targets, lo, hi);
    int64_t total = 0;
    for (int64_t t : soft_targets) {
        total += abs64(axis2 - t);
    }
    return total;
}

// Compute tightest feasible cluster hulls (with boundary pad) from xs/ys.
struct ClusterHull {
    int64_t xlo = 0, xhi = 0, ylo = 0, yhi = 0;
    int64_t w() const noexcept { return xhi - xlo; }
    int64_t h() const noexcept { return yhi - ylo; }
};

std::vector<ClusterHull> compute_cluster_hulls(const EnvelopeProblem& prob,
                                               const std::vector<int64_t>& xs,
                                               const std::vector<int64_t>& ys) {
    std::vector<ClusterHull> hulls;
    hulls.reserve(prob.clusters.size());
    for (const auto& cl : prob.clusters) {
        if (cl.members.empty()) continue;
        int64_t x0 = std::numeric_limits<int64_t>::max();
        int64_t y0 = std::numeric_limits<int64_t>::max();
        int64_t x1 = std::numeric_limits<int64_t>::min();
        int64_t y1 = std::numeric_limits<int64_t>::min();
        for (uint32_t m : cl.members) {
            x0 = std::min(x0, xs[m]);
            y0 = std::min(y0, ys[m]);
            x1 = std::max(x1, xs[m] + prob.groups[m].fw);
            y1 = std::max(y1, ys[m] + prob.groups[m].fh);
        }
        hulls.push_back(ClusterHull{x0 - cl.pad, x1 + cl.pad, y0 - cl.pad, y1 + cl.pad});
    }
    return hulls;
}

// ---------------------------------------------------------------------------
// Hierarchical Super-Block Decomposition for Fast Feasibility-Preserving SP
// ---------------------------------------------------------------------------

struct SymPairNode {
    uint32_t a = 0;
    uint32_t b = 0;
};

// A macro-unit at a given packing level:
//   - Kind::Group:   single group index `g`
//   - Kind::Cluster: a disjoint_cluster super-block with its own internal SP!
struct OuterUnit {
    bool is_cluster = false;
    uint32_t id = 0;        // group idx (if !is_cluster) or cluster idx (if is_cluster)
    int32_t band = kNoBand; // band ordinal if all members share a band, else kNoBand (floating)
    bool is_floating = false;
    int64_t pad = 0;
    std::vector<uint32_t> members;
};

struct InnerState {
    std::vector<uint32_t> gp;
    std::vector<uint32_t> gm;
};

struct PlacerContext {
    const EnvelopeProblem& prob;
    size_t n = 0;
    int64_t HX = 0;
    int64_t HY = 0;

    bool has_bands = false;
    bool has_floating = false;
    bool has_sym_pairs = false;

    std::vector<int32_t> cluster_of_group;  // -1 if unclustered, else cluster idx
    std::vector<OuterUnit> units;
    std::vector<uint32_t> unit_of_group;

    // Partition of outer units: banded units vs floating units (when has_bands && has_floating).
    std::vector<uint32_t> core_units;
    std::vector<uint32_t> floating_units;

    // Transitive H (left_of) and V (below -> above) precedence matrices at outer unit level
    // and inner cluster level.
    // h_req[u][v] == 1 means unit u must be strictly left of unit v.
    // v_req[u][v] == 1 means unit u must be strictly below unit v.
    std::vector<std::vector<uint8_t>> outer_h_req;
    std::vector<std::vector<uint8_t>> outer_v_req;
    std::vector<std::vector<std::vector<uint8_t>>> inner_h_req;
    std::vector<std::vector<std::vector<uint8_t>>> inner_v_req;

    // Symmetry pairs mapped to outer units (if both members are in distinct outer units)
    // or inside a cluster.
    std::vector<std::pair<uint32_t, uint32_t>> sym_unit_pairs;
    bool complex_symmetry = false;
    bool structurally_infeasible = false;

    // Scratch buffers to avoid per-evaluation heap allocations.
    std::vector<int64_t> unit_w;
    std::vector<int64_t> unit_h;
    std::vector<int64_t> unit_x;
    std::vector<int64_t> unit_y;
    std::vector<std::vector<int64_t>> inner_lx;
    std::vector<std::vector<int64_t>> inner_ly;
    std::vector<int64_t> scratch_xs;
    std::vector<int64_t> scratch_ys;

    explicit PlacerContext(const EnvelopeProblem& p) : prob(p), n(p.groups.size()) {
        int64_t sum_w = 0, sum_h = 0, pad_budget = 0;
        for (const auto& g : p.groups) {
            sum_w += g.fw;
            sum_h += g.fh;
        }
        for (const auto& cl : p.clusters) {
            if (!cl.members.empty()) pad_budget += 2 * std::max<int64_t>(0, cl.pad);
        }
        HX = sum_w + static_cast<int64_t>(n) + pad_budget;
        HY = sum_h + static_cast<int64_t>(n) + pad_budget;

        cluster_of_group.assign(n, -1);
        for (size_t ci = 0; ci < p.clusters.size(); ++ci) {
            for (uint32_t m : p.clusters[ci].members) {
                if (m < n) {
                    if (cluster_of_group[m] != -1) {
                        // Overlapping clusters cannot be decomposed into disjoint super-blocks cleanly.
                        complex_symmetry = true;
                    }
                    cluster_of_group[m] = static_cast<int32_t>(ci);
                }
            }
        }

        has_bands = !p.band_of_group.empty();
        if (has_bands) {
            bool any_banded = false;
            for (size_t i = 0; i < n; ++i) {
                if (p.band_of_group[i] != kNoBand) any_banded = true;
                else has_floating = true;
            }
            if (!any_banded) {
                has_bands = false;
                has_floating = false;
            }
        }

        unit_of_group.assign(n, 0);
        for (size_t ci = 0; ci < p.clusters.size(); ++ci) {
            if (p.clusters[ci].members.empty()) continue;
            OuterUnit u{};
            u.is_cluster = true;
            u.id = static_cast<uint32_t>(ci);
            u.pad = std::max<int64_t>(0, p.clusters[ci].pad);
            u.members = p.clusters[ci].members;
            if (has_bands) {
                int32_t bval = kNoBand;
                bool mixed = false;
                for (uint32_t m : u.members) {
                    const int32_t bm = p.band_of_group[m];
                    if (bval == kNoBand) bval = bm;
                    else if (bm != kNoBand && bm != bval) mixed = true;
                }
                u.band = bval;
                u.is_floating = (bval == kNoBand);
                if (mixed) complex_symmetry = true;
            }
            const uint32_t uid = static_cast<uint32_t>(units.size());
            for (uint32_t m : u.members) unit_of_group[m] = uid;
            units.push_back(std::move(u));
        }
        for (size_t gi = 0; gi < n; ++gi) {
            if (cluster_of_group[gi] != -1) continue;
            OuterUnit u{};
            u.is_cluster = false;
            u.id = static_cast<uint32_t>(gi);
            u.pad = 0;
            u.members = {static_cast<uint32_t>(gi)};
            if (has_bands) {
                u.band = p.band_of_group[gi];
                u.is_floating = (u.band == kNoBand);
            }
            const uint32_t uid = static_cast<uint32_t>(units.size());
            unit_of_group[gi] = uid;
            units.push_back(std::move(u));
        }

        const size_t U = units.size();
        for (uint32_t uid = 0; uid < U; ++uid) {
            if (has_bands && has_floating && units[uid].is_floating) {
                floating_units.push_back(uid);
            } else {
                core_units.push_back(uid);
            }
        }

        outer_h_req.assign(U, std::vector<uint8_t>(U, 0));
        outer_v_req.assign(U, std::vector<uint8_t>(U, 0));
        inner_h_req.resize(U);
        inner_v_req.resize(U);
        inner_lx.resize(U);
        inner_ly.resize(U);
        for (size_t uid = 0; uid < U; ++uid) {
            const size_t m = units[uid].members.size();
            inner_h_req[uid].assign(m, std::vector<uint8_t>(m, 0));
            inner_v_req[uid].assign(m, std::vector<uint8_t>(m, 0));
            inner_lx[uid].assign(m, 0);
            inner_ly[uid].assign(m, 0);
        }

        // Map group -> local index inside its outer unit.
        std::vector<uint32_t> local_idx(n, 0);
        for (size_t uid = 0; uid < U; ++uid) {
            for (size_t k = 0; k < units[uid].members.size(); ++k) {
                local_idx[units[uid].members[k]] = static_cast<uint32_t>(k);
            }
        }

        // Band constraints at outer unit level (lower band strictly below higher band).
        if (has_bands) {
            for (size_t i = 0; i < U; ++i) {
                if (units[i].is_floating) continue;
                for (size_t j = 0; j < U; ++j) {
                    if (units[j].is_floating) continue;
                    if (units[i].band < units[j].band) {
                        outer_v_req[i][j] = 1;
                    }
                }
            }
        }

        // Explicit order constraints.
        for (const auto& ord : p.orders) {
            if (ord.a >= n || ord.b >= n || ord.a == ord.b) continue;
            const uint32_t ua = unit_of_group[ord.a];
            const uint32_t ub = unit_of_group[ord.b];
            if (ua == ub) {
                const uint32_t la = local_idx[ord.a];
                const uint32_t lb = local_idx[ord.b];
                if (!ord.is_above) inner_h_req[ua][la][lb] = 1;
                else inner_v_req[ua][lb][la] = 1;
            } else {
                if (!ord.is_above) outer_h_req[ua][ub] = 1;
                else outer_v_req[ub][ua] = 1;
            }
        }

        // Transitive closure & cycle check for H and V requirements.
        auto close_and_check = [&](std::vector<std::vector<uint8_t>>& mat) {
            const size_t sz = mat.size();
            for (size_t k = 0; k < sz; ++k) {
                for (size_t i = 0; i < sz; ++i) {
                    if (!mat[i][k]) continue;
                    for (size_t j = 0; j < sz; ++j) {
                        if (mat[k][j]) mat[i][j] = 1;
                    }
                }
            }
            for (size_t i = 0; i < sz; ++i) {
                if (mat[i][i]) structurally_infeasible = true;
            }
        };
        close_and_check(outer_h_req);
        close_and_check(outer_v_req);
        for (size_t uid = 0; uid < U; ++uid) {
            if (units[uid].members.size() > 1) {
                close_and_check(inner_h_req[uid]);
                close_and_check(inner_v_req[uid]);
            }
        }

        // Symmetry pairs: detect simple cross-unit pairs vs complex symmetry.
        has_sym_pairs = !p.symmetry_pairs.empty();
        for (const auto& sp : p.symmetry_pairs) {
            if (sp.first >= n || sp.second >= n || sp.first == sp.second) continue;
            const uint32_t ua = unit_of_group[sp.first];
            const uint32_t ub = unit_of_group[sp.second];
            if (ua == ub || units[ua].is_cluster || units[ub].is_cluster) {
                // Pair inside a cluster or involving a multi-member cluster: handle carefully.
                complex_symmetry = true;
            } else {
                sym_unit_pairs.push_back({ua, ub});
            }
        }

        unit_w.assign(U, 0);
        unit_h.assign(U, 0);
        unit_x.assign(U, 0);
        unit_y.assign(U, 0);
        scratch_xs.assign(n, 0);
        scratch_ys.assign(n, 0);
    }
};

// Decode a subset of items via Sequence-Pair (gp, gm) RESPECTING all mandatory H/V precedence edges!
// In SP:
//   - If h_req[u][v] == 1 -> ALWAYS add H-arc u -> v (skip contradictory V-arc!).
//   - If v_req[u][v] == 1 -> ALWAYS add V-arc u -> v (skip contradictory H-arc!).
//   - Otherwise use standard SP relation (pos_p, pos_m):
//       (pos_p[u] < pos_p[v] && pos_m[u] < pos_m[v]) => H-arc u -> v
//       (pos_p[u] > pos_p[v] && pos_m[u] > pos_m[v]) => H-arc v -> u
//       (pos_p[u] > pos_p[v] && pos_m[u] < pos_m[v]) => V-arc u -> v
//       (pos_p[u] < pos_p[v] && pos_m[u] > pos_m[v]) => V-arc v -> u
// To guarantee acyclicity even when arbitrary SP permutations interact with h_req/v_req,
// we build H-arcs and V-arcs with topological Kahn relaxation; if a cycle occurs in H
// (only possible if SP added a free arc reversing a path of mandatory arcs), we flip that
// free arc to V (or drop to topological order). This guarantees 100% overlap-free AND
// 100% precedence-legal decoding in O(K^2)!
bool decode_constrained_sp(const std::vector<uint32_t>& items,
                           const std::vector<uint32_t>& gp,
                           const std::vector<uint32_t>& gm,
                           const std::vector<int64_t>& widths,
                           const std::vector<int64_t>& heights,
                           const std::vector<std::vector<uint8_t>>& h_req,
                           const std::vector<std::vector<uint8_t>>& v_req,
                           std::vector<int64_t>& out_x,
                           std::vector<int64_t>& out_y,
                           int64_t& out_W,
                           int64_t& out_H) {
    const size_t K = items.size();
    out_W = 0;
    out_H = 0;
    if (K == 0) return true;
    if (K == 1) {
        out_x[0] = 0;
        out_y[0] = 0;
        out_W = widths[0];
        out_H = heights[0];
        return true;
    }

    // Map local index 0..K-1 from gp/gm values (which are local indices 0..K-1).
    uint32_t pos_p[256];
    uint32_t pos_m[256];
    std::vector<uint32_t> pos_p_vec, pos_m_vec;
    uint32_t* pp = pos_p;
    uint32_t* pm = pos_m;
    if (K > 256) {
        pos_p_vec.resize(K);
        pos_m_vec.resize(K);
        pp = pos_p_vec.data();
        pm = pos_m_vec.data();
    }
    for (size_t i = 0; i < K; ++i) {
        pp[gp[i]] = static_cast<uint32_t>(i);
        pm[gm[i]] = static_cast<uint32_t>(i);
    }

    // Compute a valid topological order for H and V mandatory edges first, using SP rank as tie-breaker!
    auto build_topo = [&](bool horizontal, std::vector<uint32_t>& topo, std::vector<uint32_t>& rank) -> bool {
        std::vector<uint32_t> indeg(K, 0);
        for (size_t i = 0; i < K; ++i) {
            const uint32_t gi = items[i];
            for (size_t j = 0; j < K; ++j) {
                if (i == j) continue;
                const uint32_t gj = items[j];
                const bool req = horizontal ? (h_req[gi][gj] != 0) : (v_req[gi][gj] != 0);
                if (req) ++indeg[j];
            }
        }
        topo.clear();
        topo.reserve(K);
        std::vector<uint8_t> used(K, 0);
        for (size_t step = 0; step < K; ++step) {
            uint32_t best = static_cast<uint32_t>(K);
            uint32_t best_key = std::numeric_limits<uint32_t>::max();
            for (size_t i = 0; i < K; ++i) {
                if (used[i] || indeg[i] != 0) continue;
                const uint32_t key = horizontal ? (pp[i] + pm[i]) : (pm[i] + static_cast<uint32_t>(K - 1 - pp[i]));
                if (key < best_key) {
                    best_key = key;
                    best = static_cast<uint32_t>(i);
                }
            }
            if (best == K) return false;  // cycle in mandatory constraints
            used[best] = 1;
            topo.push_back(best);
            const uint32_t gb = items[best];
            for (size_t j = 0; j < K; ++j) {
                if (used[j]) continue;
                const uint32_t gj = items[j];
                const bool req = horizontal ? (h_req[gb][gj] != 0) : (v_req[gb][gj] != 0);
                if (req) --indeg[j];
            }
        }
        rank.resize(K);
        for (size_t i = 0; i < K; ++i) rank[topo[i]] = static_cast<uint32_t>(i);
        return true;
    };

    std::vector<uint32_t> h_topo, h_rank, v_topo, v_rank;
    if (!build_topo(true, h_topo, h_rank) || !build_topo(false, v_topo, v_rank)) {
        return false;
    }

    // Assign every unordered pair (i, j) to H or V consistently with h_topo and v_topo!
    // Because H-arcs ALWAYS follow h_rank[u] < h_rank[v] and V-arcs ALWAYS follow v_rank[u] < v_rank[v],
    // BOTH constraint graphs are 100% GUARANTEED ACYCLIC AND EVERY PAIR IS SEPARATED IN >= 1 AXIS!
    out_x.assign(K, 0);
    out_y.assign(K, 0);

    // Longest path on H in h_topo order.
    for (size_t ti = 0; ti < K; ++ti) {
        const uint32_t u = h_topo[ti];
        const uint32_t gu = items[u];
        const int64_t xu_end = out_x[u] + widths[u];
        for (size_t tj = ti + 1; tj < K; ++tj) {
            const uint32_t v = h_topo[tj];
            const uint32_t gv = items[v];
            bool add_h = false;
            if (h_req[gu][gv]) {
                add_h = true;
            } else if (v_req[gu][gv] || v_req[gv][gu]) {
                add_h = false;  // separated vertically by mandatory V constraint
            } else {
                // Free pair: check SP relation; only put in H if SP says u left of v (or v left of u flipped cleanly).
                const bool u_bp = pp[u] < pp[v];
                const bool u_bm = pm[u] < pm[v];
                if (u_bp == u_bm) {
                    // SP says horizontal separation; respect h_topo direction (u -> v) to guarantee DAG!
                    add_h = true;
                }
            }
            if (add_h && xu_end > out_x[v]) {
                out_x[v] = xu_end;
            }
        }
    }

    // Longest path on V in v_topo order.
    for (size_t ti = 0; ti < K; ++ti) {
        const uint32_t u = v_topo[ti];
        const uint32_t gu = items[u];
        const int64_t yu_end = out_y[u] + heights[u];
        for (size_t tj = ti + 1; tj < K; ++tj) {
            const uint32_t v = v_topo[tj];
            const uint32_t gv = items[v];
            bool add_v = false;
            if (v_req[gu][gv]) {
                add_v = true;
            } else if (h_req[gu][gv] || h_req[gv][gu]) {
                add_v = false;  // separated horizontally by mandatory H constraint
            } else {
                const bool u_bp = pp[u] < pp[v];
                const bool u_bm = pm[u] < pm[v];
                if (u_bp != u_bm) {
                    // SP says vertical separation; respect v_topo direction (u -> v) to guarantee DAG!
                    add_v = true;
                }
            }
            if (add_v && yu_end > out_y[v]) {
                out_y[v] = yu_end;
            }
        }
    }

    for (size_t i = 0; i < K; ++i) {
        out_W = std::max(out_W, out_x[i] + widths[i]);
        out_H = std::max(out_H, out_y[i] + heights[i]);
    }
    return true;
}

// Enforce exact hard symmetry pairs (y[a] == y[b] and shared axis2 = 2*x[a] + 2*x[b] + fw[a] + fw[b])
// when `prob.symmetry_pairs` is non-empty!
// Strategy: pair-aware row & mirror-axis packing over the unit-level state.
bool enforce_symmetry_and_repack(PlacerContext& ctx,
                                 std::vector<int64_t>& xs,
                                 std::vector<int64_t>& ys) {
    const auto& prob = ctx.prob;
    if (prob.symmetry_pairs.empty()) return true;

    // Check parity compatibility of all symmetry pairs first.
    const int64_t parity =
        (prob.groups[prob.symmetry_pairs[0].first].fw + prob.groups[prob.symmetry_pairs[0].second].fw) & 1LL;
    for (size_t i = 1; i < prob.symmetry_pairs.size(); ++i) {
        const auto& sp = prob.symmetry_pairs[i];
        if (((prob.groups[sp.first].fw + prob.groups[sp.second].fw) & 1LL) != parity) {
            return false;
        }
    }

    // Step 1: Equalize y[a] == y[b] for every symmetry pair by placing each pair on the
    // higher of the two rows, then resolve any resulting overlaps greedily upward in Y.
    for (const auto& sp : prob.symmetry_pairs) {
        const int64_t y_row = std::max(ys[sp.first], ys[sp.second]);
        ys[sp.first] = y_row;
        ys[sp.second] = y_row;
    }

    // Build list of symmetric groups & pair partners.
    std::vector<int32_t> partner(ctx.n, -1);
    std::vector<uint8_t> is_left_of_pair(ctx.n, 0);
    for (const auto& sp : prob.symmetry_pairs) {
        partner[sp.first] = static_cast<int32_t>(sp.second);
        partner[sp.second] = static_cast<int32_t>(sp.first);
        // Determine left vs right respecting explicit left_of orders if any.
        bool a_left = true;
        if (ctx.outer_h_req[ctx.unit_of_group[sp.second]][ctx.unit_of_group[sp.first]]) {
            a_left = false;
        } else if (ctx.outer_h_req[ctx.unit_of_group[sp.first]][ctx.unit_of_group[sp.second]]) {
            a_left = true;
        } else {
            a_left = (xs[sp.first] <= xs[sp.second]);
        }
        is_left_of_pair[sp.first] = a_left ? 1 : 0;
        is_left_of_pair[sp.second] = a_left ? 0 : 1;
    }

    // Compute minimum mirror axis2 such that left member sits at >= 0 and left+right do not overlap!
    int64_t min_axis2 = 0;
    for (const auto& sp : prob.symmetry_pairs) {
        const uint32_t L = is_left_of_pair[sp.first] ? sp.first : sp.second;
        const uint32_t R = (L == sp.first) ? sp.second : sp.first;
        const int64_t wL = prob.groups[L].fw;
        const int64_t wR = prob.groups[R].fw;
        // Need x[L] >= 0 and x[R] >= x[L] + wL => x[L] + x[R] >= 2*x[L] + wL >= wL.
        // Also respect existing x[L] >= 0: axis2 = 2*x[L] + 2*x[R] + wL + wR >= 4*x[L] + 3*wL + wR.
        const int64_t need_min = 3 * wL + wR;
        const int64_t cur_a2 = 2 * xs[L] + 2 * std::max(xs[R], xs[L] + wL) + wL + wR;
        min_axis2 = std::max(min_axis2, std::max(need_min, cur_a2));
    }
    if ((min_axis2 & 1LL) != parity) ++min_axis2;

    // Place each symmetry pair cleanly mirrored around min_axis2!
    for (const auto& sp : prob.symmetry_pairs) {
        const uint32_t L = is_left_of_pair[sp.first] ? sp.first : sp.second;
        const uint32_t R = (L == sp.first) ? sp.second : sp.first;
        const int64_t wL = prob.groups[L].fw;
        const int64_t wR = prob.groups[R].fw;
        const int64_t sum_lr = (min_axis2 - wL - wR) / 2;
        // Choose xL <= (sum_lr - wL) / 2 so xR = sum_lr - xL >= xL + wL!
        const int64_t max_xL = (sum_lr - wL) / 2;
        int64_t xL = std::clamp<int64_t>(xs[L], 0, std::max<int64_t>(0, max_xL));
        int64_t xR = sum_lr - xL;
        if (xR < xL + wL) {
            return false;
        }
        xs[L] = xL;
        xs[R] = xR;
    }

    // Resolve any remaining Y-overlaps introduced by row equalization (moving pairs together!).
    for (int pass = 0; pass < 64; ++pass) {
        bool moved = false;
        // Enforce band / vertical orders first.
        for (const auto& ord : prob.orders) {
            if (!ord.is_above) continue;
            // b below a => y[a] >= y[b] + fh[b]
            const int64_t need = ys[ord.b] + prob.groups[ord.b].fh;
            if (ys[ord.a] < need) {
                ys[ord.a] = need;
                if (partner[ord.a] >= 0) ys[partner[ord.a]] = need;
                moved = true;
            }
        }
        if (ctx.has_bands) {
            for (size_t i = 0; i < ctx.n; ++i) {
                if (prob.band_of_group[i] == kNoBand) continue;
                for (size_t j = 0; j < ctx.n; ++j) {
                    if (prob.band_of_group[j] == kNoBand) continue;
                    if (prob.band_of_group[i] < prob.band_of_group[j]) {
                        const int64_t need = ys[i] + prob.groups[i].fh;
                        if (ys[j] < need) {
                            ys[j] = need;
                            if (partner[j] >= 0) ys[partner[j]] = need;
                            moved = true;
                        }
                    }
                }
            }
        }
        // Resolve pairwise rectangle overlaps by pushing the upper-Y item (and its mirror partner) up.
        for (size_t i = 0; i < ctx.n; ++i) {
            for (size_t j = i + 1; j < ctx.n; ++j) {
                if (rects_overlap_strict(xs[i], ys[i], xs[i] + prob.groups[i].fw, ys[i] + prob.groups[i].fh,
                                         xs[j], ys[j], xs[j] + prob.groups[j].fw, ys[j] + prob.groups[j].fh)) {
                    // Decide which to push up: respect band/above ordering if present.
                    size_t lower = i, upper = j;
                    if (ctx.has_bands && prob.band_of_group[i] != kNoBand && prob.band_of_group[j] != kNoBand &&
                        prob.band_of_group[j] < prob.band_of_group[i]) {
                        lower = j;
                        upper = i;
                    } else if (ys[j] < ys[i]) {
                        lower = j;
                        upper = i;
                    }
                    const int64_t need = ys[lower] + prob.groups[lower].fh;
                    if (ys[upper] < need) {
                        ys[upper] = need;
                        if (partner[upper] >= 0) ys[partner[upper]] = need;
                        moved = true;
                    }
                }
            }
        }
        if (!moved) break;
    }

    return verify_envelope_solution(prob, xs, ys);
}

struct FullState {
    std::vector<uint32_t> core_gp;
    std::vector<uint32_t> core_gm;
    std::vector<uint32_t> float_gp;
    std::vector<uint32_t> float_gm;
    std::vector<InnerState> inner;  // indexed by outer unit id
};

bool decode_full_state(PlacerContext& ctx,
                       const FullState& st,
                       std::vector<int64_t>& xs,
                       std::vector<int64_t>& ys) {
    const auto& prob = ctx.prob;
    const size_t U = ctx.units.size();

    // 1. Decode every inner cluster SP to determine cluster super-block dimensions (unit_w, unit_h).
    for (size_t uid = 0; uid < U; ++uid) {
        const auto& u = ctx.units[uid];
        if (!u.is_cluster) {
            ctx.unit_w[uid] = prob.groups[u.id].fw;
            ctx.unit_h[uid] = prob.groups[u.id].fh;
            ctx.inner_lx[uid][0] = 0;
            ctx.inner_ly[uid][0] = 0;
        } else {
            const size_t m = u.members.size();
            std::vector<uint32_t> local_ids(m);
            std::vector<int64_t> mw(m), mh(m);
            for (size_t k = 0; k < m; ++k) {
                local_ids[k] = static_cast<uint32_t>(k);
                mw[k] = prob.groups[u.members[k]].fw;
                mh[k] = prob.groups[u.members[k]].fh;
            }
            int64_t cw = 0, ch = 0;
            if (!decode_constrained_sp(local_ids, st.inner[uid].gp, st.inner[uid].gm,
                                       mw, mh, ctx.inner_h_req[uid], ctx.inner_v_req[uid],
                                       ctx.inner_lx[uid], ctx.inner_ly[uid], cw, ch)) {
                return false;
            }
            ctx.unit_w[uid] = cw + 2 * u.pad;
            ctx.unit_h[uid] = ch + 2 * u.pad;
        }
    }

    // 2. Decode core (all units, or banded units if floating passive bank is active).
    const size_t C = ctx.core_units.size();
    std::vector<int64_t> cw_vec(C), ch_vec(C), cx_vec(C), cy_vec(C);
    for (size_t i = 0; i < C; ++i) {
        cw_vec[i] = ctx.unit_w[ctx.core_units[i]];
        ch_vec[i] = ctx.unit_h[ctx.core_units[i]];
    }
    int64_t core_W = 0, core_H = 0;
    if (!decode_constrained_sp(ctx.core_units, st.core_gp, st.core_gm,
                               cw_vec, ch_vec, ctx.outer_h_req, ctx.outer_v_req,
                               cx_vec, cy_vec, core_W, core_H)) {
        return false;
    }
    for (size_t i = 0; i < C; ++i) {
        const uint32_t uid = ctx.core_units[i];
        ctx.unit_x[uid] = cx_vec[i];
        ctx.unit_y[uid] = cy_vec[i];
    }

    // 3. Expand core units to group coordinates.
    for (size_t i = 0; i < C; ++i) {
        const uint32_t uid = ctx.core_units[i];
        const auto& u = ctx.units[uid];
        for (size_t k = 0; k < u.members.size(); ++k) {
            const uint32_t g = u.members[k];
            xs[g] = ctx.unit_x[uid] + u.pad + ctx.inner_lx[uid][k];
            ys[g] = ctx.unit_y[uid] + u.pad + ctx.inner_ly[uid][k];
        }
    }

    // 4. If floating passive bank is active (has_bands && has_floating), pack floating units
    //    to the right of mos_right and center their Y inside [mos_lo, mos_hi]!
    if (!ctx.floating_units.empty()) {
        int64_t mos_lo = std::numeric_limits<int64_t>::max();
        int64_t mos_hi = 0;
        int64_t mos_right = 0;
        for (size_t g = 0; g < ctx.n; ++g) {
            if (prob.band_of_group[g] == kNoBand) continue;
            mos_lo = std::min(mos_lo, ys[g]);
            mos_hi = std::max(mos_hi, ys[g] + prob.groups[g].fh);
            mos_right = std::max(mos_right, xs[g] + prob.groups[g].fw);
        }
        // Account for any banded cluster right boundary pad so floating blocks never touch banded cluster hull.
        int64_t bank_x0 = mos_right;
        for (uint32_t uid : ctx.core_units) {
            bank_x0 = std::max(bank_x0, ctx.unit_x[uid] + ctx.unit_w[uid]);
        }

        const size_t F = ctx.floating_units.size();
        std::vector<int64_t> fw_vec(F), fh_vec(F), fx_vec(F), fy_vec(F);
        for (size_t i = 0; i < F; ++i) {
            fw_vec[i] = ctx.unit_w[ctx.floating_units[i]];
            fh_vec[i] = ctx.unit_h[ctx.floating_units[i]];
        }
        int64_t fW = 0, fH = 0;
        if (!decode_constrained_sp(ctx.floating_units, st.float_gp, st.float_gm,
                                   fw_vec, fh_vec, ctx.outer_h_req, ctx.outer_v_req,
                                   fx_vec, fy_vec, fW, fH)) {
            return false;
        }
        // Shift floating block vertically to fit 2*mos_lo <= 2*ys[g]+fh[g] <= 2*mos_hi.
        // If fH exceeds (mos_hi - mos_lo), check whether every floating tile's center stays inside [mos_lo, mos_hi].
        // Center the floating bank vertically around (mos_lo + mos_hi) / 2!
        int64_t shift_y = ((mos_lo + mos_hi) - fH) / 2;
        if (shift_y < 0) shift_y = 0;

        for (size_t i = 0; i < F; ++i) {
            const uint32_t uid = ctx.floating_units[i];
            ctx.unit_x[uid] = bank_x0 + fx_vec[i];
            ctx.unit_y[uid] = shift_y + fy_vec[i];
            const auto& u = ctx.units[uid];
            for (size_t k = 0; k < u.members.size(); ++k) {
                const uint32_t g = u.members[k];
                xs[g] = ctx.unit_x[uid] + u.pad + ctx.inner_lx[uid][k];
                ys[g] = ctx.unit_y[uid] + u.pad + ctx.inner_ly[uid][k];
            }
        }

        // If any floating unit's y-center exceeds mos_hi (tall floating stack), spread them into
        // horizontal columns beside mos_right (strictly guaranteed feasible since 2*mos_lo <= 2*y + fh <= 2*mos_hi
        // is always satisfiable for any tile by laying tiles side-by-side in X at y = max(0, mos_lo)!).
        bool y_center_ok = true;
        for (size_t g = 0; g < ctx.n; ++g) {
            if (prob.band_of_group[g] != kNoBand) continue;
            const int64_t cy2 = 2 * ys[g] + prob.groups[g].fh;
            if (cy2 < 2 * mos_lo || cy2 > 2 * mos_hi) {
                y_center_ok = false;
                break;
            }
        }
        if (!y_center_ok) {
            // Greedy column-wrap inside the allowed vertical center window [mos_lo, mos_hi]!
            int64_t cur_x = bank_x0;
            int64_t cur_y = mos_lo;
            int64_t col_w = 0;
            for (size_t ord_i = 0; ord_i < F; ++ord_i) {
                const uint32_t uid = ctx.floating_units[st.float_gp[ord_i]];
                const auto& u = ctx.units[uid];
                const int64_t uw = ctx.unit_w[uid];
                const int64_t uh = ctx.unit_h[uid];
                // Check if placing at cur_y keeps all members' cy2 <= 2*mos_hi.
                bool fits_col = true;
                for (size_t k = 0; k < u.members.size(); ++k) {
                    const uint32_t g = u.members[k];
                    const int64_t gy = cur_y + u.pad + ctx.inner_ly[uid][k];
                    const int64_t cy2 = 2 * gy + prob.groups[g].fh;
                    if (cy2 > 2 * mos_hi) {
                        fits_col = false;
                        break;
                    }
                }
                if (!fits_col && cur_y > mos_lo) {
                    cur_x += col_w;
                    cur_y = mos_lo;
                    col_w = 0;
                }
                // Choose starting y for this unit to satisfy cy2 >= 2*mos_lo and cy2 <= 2*mos_hi.
                int64_t place_y = cur_y;
                if (u.members.size() == 1) {
                    const int64_t fh_g = prob.groups[u.members[0]].fh;
                    const int64_t min_y = std::max<int64_t>(0, (2 * mos_lo - fh_g + 1) / 2);
                    const int64_t max_y = (2 * mos_hi - fh_g) / 2;
                    if (place_y < min_y) place_y = min_y;
                    if (place_y > max_y) {
                        // Start new column at min_y!
                        if (col_w > 0) {
                            cur_x += col_w;
                            col_w = 0;
                        }
                        place_y = std::max<int64_t>(0, std::min(min_y, max_y));
                    }
                }
                ctx.unit_x[uid] = cur_x;
                ctx.unit_y[uid] = place_y;
                for (size_t k = 0; k < u.members.size(); ++k) {
                    const uint32_t g = u.members[k];
                    xs[g] = ctx.unit_x[uid] + u.pad + ctx.inner_lx[uid][k];
                    ys[g] = ctx.unit_y[uid] + u.pad + ctx.inner_ly[uid][k];
                }
                cur_y = place_y + uh;
                col_w = std::max(col_w, uw);
            }
        }
    }

    // 5. If symmetry_pairs are present, enforce exact reflection rows + shared mirror axis.
    if (ctx.has_sym_pairs) {
        if (!enforce_symmetry_and_repack(ctx, xs, ys)) {
            return false;
        }
    }

    // 6. If hard axis_groups exist, align hard axis groups onto a shared axis2!
    if (!prob.axis_groups.empty()) {
        bool axis_ok = true;
        (void)compute_axis_center_cost(prob, xs, 0, &axis_ok);
        if (!axis_ok) {
            // Align all hard axis groups to the rightmost hard axis group's axis2, then repack Y to avoid overlaps.
            bool have_primary = false;
            if (prob.axis_centered_soft) {
                for (const auto& ag : prob.axis_groups)
                    if (ag.is_primary) have_primary = true;
            }
            int64_t target_a2 = 0;
            for (size_t i = 0; i < prob.axis_groups.size(); ++i) {
                const auto& ag = prob.axis_groups[i];
                const bool is_hard = !prob.axis_centered_soft || ag.is_primary || (!have_primary && i == 0);
                if (!is_hard) continue;
                const int64_t axk = 2 * xs[ag.group] + prob.groups[ag.group].fw + 2 * ag.offset_rounded;
                target_a2 = std::max(target_a2, axk);
            }
            for (size_t i = 0; i < prob.axis_groups.size(); ++i) {
                const auto& ag = prob.axis_groups[i];
                const bool is_hard = !prob.axis_centered_soft || ag.is_primary || (!have_primary && i == 0);
                if (!is_hard) continue;
                const int64_t num = target_a2 - prob.groups[ag.group].fw - 2 * ag.offset_rounded;
                const int64_t new_x = std::max<int64_t>(0, (num + 1) / 2);
                const int64_t dx = new_x - xs[ag.group];
                const uint32_t uid = ctx.unit_of_group[ag.group];
                for (uint32_t m : ctx.units[uid].members) xs[m] = std::max<int64_t>(0, xs[m] + dx);
            }
            // Push any overlapping units upward in Y.
            for (int pass = 0; pass < 32; ++pass) {
                bool moved = false;
                for (size_t i = 0; i < ctx.n; ++i) {
                    for (size_t j = i + 1; j < ctx.n; ++j) {
                        if (rects_overlap_strict(xs[i], ys[i], xs[i] + prob.groups[i].fw, ys[i] + prob.groups[i].fh,
                                                 xs[j], ys[j], xs[j] + prob.groups[j].fw, ys[j] + prob.groups[j].fh)) {
                            const size_t lo = ys[i] <= ys[j] ? i : j;
                            const size_t hi = (lo == i) ? j : i;
                            const int64_t need = ys[lo] + prob.groups[lo].fh;
                            if (ys[hi] < need) {
                                ys[hi] = need;
                                moved = true;
                            }
                        }
                    }
                }
                if (!moved) break;
            }
        }
    }

    // 7. Shift non-negative origin to >= 0 (accounting for cluster pads!) so every coordinate is in [0, HX/HY].
    int64_t min_x = std::numeric_limits<int64_t>::max();
    int64_t min_y = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < ctx.n; ++i) {
        const int32_t ci = ctx.cluster_of_group[i];
        const int64_t pad = (ci >= 0) ? std::max<int64_t>(0, prob.clusters[ci].pad) : 0;
        min_x = std::min(min_x, xs[i] - pad);
        min_y = std::min(min_y, ys[i] - pad);
    }
    if (min_x != 0) {
        for (size_t i = 0; i < ctx.n; ++i) xs[i] -= min_x;
    }
    if (min_y != 0) {
        for (size_t i = 0; i < ctx.n; ++i) ys[i] -= min_y;
    }

    return true;
}

// Fast exact coordinate-descent slack polisher:
// Slides each unit (and each cluster member) along X and Y to candidate optimal coordinates
// (bounding edges, net medians, rail edges, aspect targets) keeping 100% hard legality!
void polish_coordinates(PlacerContext& ctx,
                        std::vector<int64_t>& xs,
                        std::vector<int64_t>& ys,
                        int64_t& best_obj) {
    const auto& prob = ctx.prob;
    const size_t n = ctx.n;
    if (n <= 1 || !prob.symmetry_pairs.empty()) return;

    // Precompute incident nets per group.
    std::vector<std::vector<uint32_t>> nets_of(n);
    for (size_t ni = 0; ni < prob.nets.size(); ++ni) {
        for (uint32_t m : prob.nets[ni].members) {
            if (m < n) nets_of[m].push_back(static_cast<uint32_t>(ni));
        }
    }

    // Also allow moving entire clusters together or individual unclustered groups.
    const size_t U = ctx.units.size();
    std::vector<int64_t> cand_vals;
    cand_vals.reserve(64);

    for (int round = 0; round < 3; ++round) {
        bool any_improved = false;
        int64_t curW = 0, curH = 0;
        for (size_t i = 0; i < n; ++i) {
            curW = std::max(curW, xs[i] + prob.groups[i].fw);
            curH = std::max(curH, ys[i] + prob.groups[i].fh);
        }

        for (size_t uid = 0; uid < U; ++uid) {
            const auto& u = ctx.units[uid];
            const int64_t pad = u.pad;
            // Compute current unit bbox.
            int64_t ux0 = std::numeric_limits<int64_t>::max();
            int64_t uy0 = std::numeric_limits<int64_t>::max();
            int64_t ux1 = 0, uy1 = 0;
            for (uint32_t m : u.members) {
                ux0 = std::min(ux0, xs[m]);
                uy0 = std::min(uy0, ys[m]);
                ux1 = std::max(ux1, xs[m] + prob.groups[m].fw);
                uy1 = std::max(uy1, ys[m] + prob.groups[m].fh);
            }
            const int64_t uw = ux1 - ux0;
            const int64_t uh = uy1 - uy0;

            for (int axis = 0; axis < 2; ++axis) {
                const int64_t cur_lo = (axis == 0) ? ux0 : uy0;
                const int64_t cur_span = (axis == 0) ? uw : uh;
                const int64_t env_lim = (axis == 0) ? curW : curH;
                cand_vals.clear();
                cand_vals.push_back(pad);
                cand_vals.push_back(std::max(pad, env_lim - cur_span));
                // Pull toward connected nets' other members' min/max/mid.
                for (uint32_t m : u.members) {
                    const int64_t off = ((axis == 0) ? xs[m] : ys[m]) - cur_lo;
                    for (uint32_t ni : nets_of[m]) {
                        int64_t nlo = std::numeric_limits<int64_t>::max();
                        int64_t nhi = 0;
                        bool any_other = false;
                        for (uint32_t om : prob.nets[ni].members) {
                            if (ctx.unit_of_group[om] == uid) continue;
                            any_other = true;
                            const int64_t v = (axis == 0) ? xs[om] : ys[om];
                            nlo = std::min(nlo, v);
                            nhi = std::max(nhi, v);
                        }
                        if (any_other) {
                            cand_vals.push_back(std::max(pad, nlo - off));
                            cand_vals.push_back(std::max(pad, nhi - off));
                            cand_vals.push_back(std::max(pad, (nlo + nhi) / 2 - off));
                        }
                    }
                }
                // Neighbor edge-abut candidates.
                for (size_t j = 0; j < n; ++j) {
                    if (ctx.unit_of_group[j] == uid) continue;
                    const int64_t j0 = (axis == 0) ? xs[j] : ys[j];
                    const int64_t j1 = j0 + ((axis == 0) ? prob.groups[j].fw : prob.groups[j].fh);
                    const int32_t cj = ctx.cluster_of_group[j];
                    const int64_t jpad = (cj >= 0) ? prob.clusters[cj].pad : 0;
                    cand_vals.push_back(std::max(pad, j1 + jpad + pad));
                    if (j0 - jpad - pad - cur_span >= pad) {
                        cand_vals.push_back(j0 - jpad - pad - cur_span);
                    }
                }
                std::sort(cand_vals.begin(), cand_vals.end());
                cand_vals.erase(std::unique(cand_vals.begin(), cand_vals.end()), cand_vals.end());

                int64_t best_pos = cur_lo;
                for (int64_t target : cand_vals) {
                    if (target == cur_lo || target < pad) continue;
                    const int64_t delta = target - cur_lo;
                    for (uint32_t m : u.members) {
                        if (axis == 0) xs[m] += delta;
                        else ys[m] += delta;
                    }
                    if (verify_envelope_solution(prob, xs, ys)) {
                        const int64_t obj = evaluate_envelope_objective(prob, xs, ys);
                        if (obj < best_obj) {
                            best_obj = obj;
                            best_pos = target;
                            any_improved = true;
                        }
                    }
                    // Revert to cur_lo before trying next candidate.
                    for (uint32_t m : u.members) {
                        if (axis == 0) xs[m] -= delta;
                        else ys[m] -= delta;
                    }
                }
                if (best_pos != cur_lo) {
                    const int64_t delta = best_pos - cur_lo;
                    for (uint32_t m : u.members) {
                        if (axis == 0) xs[m] += delta;
                        else ys[m] += delta;
                    }
                    if (axis == 0) ux0 = best_pos;
                    else uy0 = best_pos;
                }
            }
        }
        if (!any_improved) break;
    }
}

}  // namespace

bool verify_envelope_solution(const EnvelopeProblem& prob,
                              const std::vector<int64_t>& xs,
                              const std::vector<int64_t>& ys,
                              std::string* reason) {
    const size_t n = prob.groups.size();
    if (xs.size() != n || ys.size() != n) {
        if (reason) *reason = "coordinate vector size mismatch";
        return false;
    }
    int64_t sum_w = 0, sum_h = 0, pad_budget = 0;
    for (const auto& g : prob.groups) {
        sum_w += g.fw;
        sum_h += g.fh;
    }
    for (const auto& cl : prob.clusters) {
        if (!cl.members.empty()) pad_budget += 2 * std::max<int64_t>(0, cl.pad);
    }
    const int64_t HX = sum_w + static_cast<int64_t>(n) + pad_budget;
    const int64_t HY = sum_h + static_cast<int64_t>(n) + pad_budget;

    for (size_t i = 0; i < n; ++i) {
        if (xs[i] < 0 || ys[i] < 0 || xs[i] + prob.groups[i].fw > HX || ys[i] + prob.groups[i].fh > HY) {
            if (reason) *reason = "coordinate out of domain [0, HX/HY]";
            return false;
        }
    }

    // 1. Pairwise NoOverlap2D on all groups.
    for (size_t i = 0; i < n; ++i) {
        const int64_t ax0 = xs[i], ay0 = ys[i], ax1 = ax0 + prob.groups[i].fw, ay1 = ay0 + prob.groups[i].fh;
        for (size_t j = i + 1; j < n; ++j) {
            if (rects_overlap_strict(ax0, ay0, ax1, ay1,
                                     xs[j], ys[j], xs[j] + prob.groups[j].fw, ys[j] + prob.groups[j].fh)) {
                if (reason) *reason = "group overlap between " + prob.groups[i].key + " and " + prob.groups[j].key;
                return false;
            }
        }
    }

    // 2. Disjoint clusters: padded hulls within [0, HX/HY] and mutually disjoint from each other
    //    and from every non-clustered group.
    if (!prob.clusters.empty()) {
        std::vector<uint8_t> clustered(n, 0);
        struct HullBox {
            int64_t x0, y0, x1, y1;
        };
        std::vector<HullBox> hulls;
        for (const auto& cl : prob.clusters) {
            if (cl.members.empty()) continue;
            int64_t x0 = std::numeric_limits<int64_t>::max();
            int64_t y0 = std::numeric_limits<int64_t>::max();
            int64_t x1 = std::numeric_limits<int64_t>::min();
            int64_t y1 = std::numeric_limits<int64_t>::min();
            for (uint32_t m : cl.members) {
                clustered[m] = 1;
                x0 = std::min(x0, xs[m]);
                y0 = std::min(y0, ys[m]);
                x1 = std::max(x1, xs[m] + prob.groups[m].fw);
                y1 = std::max(y1, ys[m] + prob.groups[m].fh);
            }
            const int64_t ipad = std::max<int64_t>(0, cl.pad);
            HullBox hb{x0 - ipad, y0 - ipad, x1 + ipad, y1 + ipad};
            if (hb.x0 < 0 || hb.y0 < 0 || hb.x1 > HX || hb.y1 > HY) {
                if (reason) *reason = "cluster padded hull outside domain [0, HX/HY]";
                return false;
            }
            hulls.push_back(hb);
        }
        for (size_t i = 0; i < n; ++i) {
            if (!clustered[i]) {
                hulls.push_back(HullBox{xs[i], ys[i], xs[i] + prob.groups[i].fw, ys[i] + prob.groups[i].fh});
            }
        }
        for (size_t i = 0; i < hulls.size(); ++i) {
            for (size_t j = i + 1; j < hulls.size(); ++j) {
                if (rects_overlap_strict(hulls[i].x0, hulls[i].y0, hulls[i].x1, hulls[i].y1,
                                         hulls[j].x0, hulls[j].y0, hulls[j].x1, hulls[j].y1)) {
                    if (reason) *reason = "cluster hull overlap";
                    return false;
                }
            }
        }
    }

    // 3. Band ordering + floating passive right-column banking.
    if (!prob.band_of_group.empty()) {
        int64_t mos_lo = std::numeric_limits<int64_t>::max();
        int64_t mos_hi = std::numeric_limits<int64_t>::min();
        int64_t mos_right = 0;
        bool any_banded = false;
        for (size_t i = 0; i < n; ++i) {
            if (prob.band_of_group[i] == kNoBand) continue;
            any_banded = true;
            mos_lo = std::min(mos_lo, ys[i]);
            mos_hi = std::max(mos_hi, ys[i] + prob.groups[i].fh);
            mos_right = std::max(mos_right, xs[i] + prob.groups[i].fw);
            for (size_t j = i + 1; j < n; ++j) {
                if (prob.band_of_group[j] == kNoBand) continue;
                if (prob.band_of_group[i] < prob.band_of_group[j]) {
                    if (ys[i] + prob.groups[i].fh > ys[j]) {
                        if (reason) *reason = "band Y-order violated";
                        return false;
                    }
                } else if (prob.band_of_group[j] < prob.band_of_group[i]) {
                    if (ys[j] + prob.groups[j].fh > ys[i]) {
                        if (reason) *reason = "band Y-order violated";
                        return false;
                    }
                }
            }
        }
        if (any_banded) {
            for (size_t i = 0; i < n; ++i) {
                if (prob.band_of_group[i] != kNoBand) continue;
                if (xs[i] < mos_right) {
                    if (reason) *reason = "floating group left of mos_right";
                    return false;
                }
                const int64_t cy2 = 2 * ys[i] + prob.groups[i].fh;
                if (cy2 < 2 * mos_lo || cy2 > 2 * mos_hi) {
                    if (reason) *reason = "floating group y-center outside [mos_lo, mos_hi]";
                    return false;
                }
            }
        }
    }

    // 4. Explicit order constraints.
    for (const auto& ord : prob.orders) {
        if (ord.a >= n || ord.b >= n || ord.a == ord.b) continue;
        if (!ord.is_above) {
            if (xs[ord.a] + prob.groups[ord.a].fw > xs[ord.b]) {
                if (reason) *reason = "left_of order constraint violated";
                return false;
            }
        } else {
            if (ys[ord.b] + prob.groups[ord.b].fh > ys[ord.a]) {
                if (reason) *reason = "above order constraint violated";
                return false;
            }
        }
    }

    // 5. Hard symmetry pairs.
    if (!prob.symmetry_pairs.empty()) {
        int64_t axis_ref = 0;
        bool has_ref = false;
        for (const auto& sp : prob.symmetry_pairs) {
            if (sp.first >= n || sp.second >= n) continue;
            if (ys[sp.first] != ys[sp.second]) {
                if (reason) *reason = "symmetry pair y mismatch";
                return false;
            }
            const int64_t expr = 2 * xs[sp.first] + 2 * xs[sp.second] + prob.groups[sp.first].fw + prob.groups[sp.second].fw;
            if (!has_ref) {
                axis_ref = expr;
                has_ref = true;
            } else if (expr != axis_ref) {
                if (reason) *reason = "symmetry pair axis mismatch";
                return false;
            }
        }
    }

    // 6. Hard axis_groups check.
    if (!prob.axis_groups.empty()) {
        bool axis_ok = true;
        (void)compute_axis_center_cost(prob, xs, 0, &axis_ok);
        if (!axis_ok) {
            if (reason) *reason = "hard axis_groups cannot share axis2 within +-1 tolerance";
            return false;
        }
    }

    return true;
}

int64_t evaluate_envelope_objective(const EnvelopeProblem& prob,
                                    const std::vector<int64_t>& xs,
                                    const std::vector<int64_t>& ys,
                                    int64_t* out_W,
                                    int64_t* out_H) {
    const size_t n = prob.groups.size();
    int64_t W = 0, H = 0;
    for (size_t i = 0; i < n; ++i) {
        W = std::max(W, xs[i] + prob.groups[i].fw);
        H = std::max(H, ys[i] + prob.groups[i].fh);
    }
    if (out_W) *out_W = W;
    if (out_H) *out_H = H;

    int64_t island_span = 0;
    if (!prob.clusters.empty()) {
        const auto hulls = compute_cluster_hulls(prob, xs, ys);
        size_t hi = 0;
        for (const auto& cl : prob.clusters) {
            if (cl.members.empty()) continue;
            if (cl.members.size() >= 2) {
                island_span += hulls[hi].w() + hulls[hi].h();
            }
            ++hi;
        }
    }

    int64_t aspect_term = 0;
    if (prob.aspect_target > 0.0 && prob.aspect_q > 0) {
        aspect_term = abs64(kAspectDenom * W - prob.aspect_q * H);
    }

    const int64_t center_sum = compute_axis_center_cost(prob, xs, W);

    int64_t wl_sum = 0;
    int64_t max_signal_span = 0;
    for (const auto& net : prob.nets) {
        if (net.members.size() < 2) continue;
        int64_t xlo = std::numeric_limits<int64_t>::max();
        int64_t xhi = 0;
        int64_t ylo = std::numeric_limits<int64_t>::max();
        int64_t yhi = 0;
        for (uint32_t m : net.members) {
            xlo = std::min(xlo, xs[m]);
            xhi = std::max(xhi, xs[m]);
            ylo = std::min(ylo, ys[m]);
            yhi = std::max(yhi, ys[m]);
        }
        const int64_t sx = xhi - xlo;
        const int64_t sy = yhi - ylo;
        if (!prob.has_net_weights) {
            wl_sum += sx + sy;
        } else {
            wl_sum += net.weight_scaled * (sx + sy);
            if (net.is_signal) {
                const int64_t span = sx + sy;
                if (span > max_signal_span) max_signal_span = span;
            }
        }
    }

    int64_t abut_sum = 0;
    for (const auto& ab : prob.abut_pairs) {
        if (ab.first >= n || ab.second >= n || ab.first == ab.second) continue;
        const int64_t cxa = 2 * xs[ab.first] + prob.groups[ab.first].fw;
        const int64_t cxb = 2 * xs[ab.second] + prob.groups[ab.second].fw;
        const int64_t cya = 2 * ys[ab.first] + prob.groups[ab.first].fh;
        const int64_t cyb = 2 * ys[ab.second] + prob.groups[ab.second].fh;
        abut_sum += abs64(cxa - cxb) + abs64(cya - cyb);
    }

    int64_t rail_sum = 0;
    for (const auto& ra : prob.rail_affinities) {
        if (ra.group >= n) continue;
        if (!ra.is_top) {
            rail_sum += ys[ra.group];
        } else {
            rail_sum += std::max<int64_t>(0, H - (ys[ra.group] + prob.groups[ra.group].fh));
        }
    }

    return kEnvelopeWeight * (W + H + island_span)
         + kAspectWeight * aspect_term
         + kCenterWeight * center_sum
         + prob.maxspan_weight * max_signal_span
         + prob.wl_weight * wl_sum
         + kAbutWeight * abut_sum
         + kRailAffinityWeight * rail_sum;
}

EnvelopeSolution solve_envelope_min(const EnvelopeProblem& prob) {
    EnvelopeSolution sol{};
    const size_t n = prob.groups.size();
    if (n == 0) {
        sol.status = EnvelopeStatus::Optimal;
        return sol;
    }

    // Compute simple admissible lower bound on envelope (W + H) * kEnvelopeWeight.
    int64_t total_area = 0;
    int64_t max_fw = 0, max_fh = 0;
    for (const auto& g : prob.groups) {
        total_area += g.fw * g.fh;
        max_fw = std::max(max_fw, g.fw);
        max_fh = std::max(max_fh, g.fh);
    }
    const int64_t lb_perim = std::max<int64_t>(
        max_fw + max_fh,
        static_cast<int64_t>(std::ceil(2.0 * std::sqrt(static_cast<double>(total_area)))));
    sol.lower_bound = kEnvelopeWeight * lb_perim;

    if (n == 1) {
        const int64_t pad = (!prob.clusters.empty() && !prob.clusters[0].members.empty())
                                ? std::max<int64_t>(0, prob.clusters[0].pad)
                                : 0;
        sol.xs = {pad};
        sol.ys = {pad};
        if (verify_envelope_solution(prob, sol.xs, sol.ys, &sol.detail)) {
            sol.status = EnvelopeStatus::Optimal;
            sol.objective = evaluate_envelope_objective(prob, sol.xs, sol.ys, &sol.W, &sol.H);
            sol.evaluations = 1;
        } else {
            sol.status = EnvelopeStatus::Infeasible;
        }
        return sol;
    }

    PlacerContext ctx(prob);
    if (ctx.structurally_infeasible) {
        sol.status = EnvelopeStatus::Infeasible;
        sol.detail = "cyclic precedence or band constraints";
        return sol;
    }

    // Construct initial deterministic state.
    auto make_identity_state = [&]() -> FullState {
        FullState st;
        st.core_gp.resize(ctx.core_units.size());
        st.core_gm.resize(ctx.core_units.size());
        std::iota(st.core_gp.begin(), st.core_gp.end(), 0U);
        std::iota(st.core_gm.begin(), st.core_gm.end(), 0U);
        st.float_gp.resize(ctx.floating_units.size());
        st.float_gm.resize(ctx.floating_units.size());
        std::iota(st.float_gp.begin(), st.float_gp.end(), 0U);
        std::iota(st.float_gm.begin(), st.float_gm.end(), 0U);
        st.inner.resize(ctx.units.size());
        for (size_t uid = 0; uid < ctx.units.size(); ++uid) {
            const size_t m = ctx.units[uid].members.size();
            st.inner[uid].gp.resize(m);
            st.inner[uid].gm.resize(m);
            std::iota(st.inner[uid].gp.begin(), st.inner[uid].gp.end(), 0U);
            std::iota(st.inner[uid].gm.begin(), st.inner[uid].gm.end(), 0U);
        }
        return st;
    };

    FullState best_state = make_identity_state();
    std::vector<int64_t> best_xs(n, 0), best_ys(n, 0);
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    bool has_feasible = false;
    uint64_t evals = 0;

    auto try_state = [&](const FullState& st) -> int64_t {
        ++evals;
        if (!decode_full_state(ctx, st, ctx.scratch_xs, ctx.scratch_ys)) {
            return std::numeric_limits<int64_t>::max();
        }
        if (!verify_envelope_solution(prob, ctx.scratch_xs, ctx.scratch_ys)) {
            return std::numeric_limits<int64_t>::max();
        }
        const int64_t obj = evaluate_envelope_objective(prob, ctx.scratch_xs, ctx.scratch_ys);
        if (obj < best_obj) {
            best_obj = obj;
            best_xs = ctx.scratch_xs;
            best_ys = ctx.scratch_ys;
            best_state = st;
            has_feasible = true;
        }
        return obj;
    };

    // Multi-seed analytical warm starts (horizontal strip, vertical stack, 2D grid rows, sorted-by-area).
    Rng64 rng(prob.seed);
    std::vector<FullState> seeds;
    seeds.reserve(10);
    {
        // Seed 0: horizontal strip (gp == gm)
        seeds.push_back(make_identity_state());
        // Seed 1: vertical stack (gp == rev(gm))
        FullState s1 = make_identity_state();
        std::reverse(s1.core_gp.begin(), s1.core_gp.end());
        std::reverse(s1.float_gp.begin(), s1.float_gp.end());
        for (auto& in : s1.inner) std::reverse(in.gp.begin(), in.gp.end());
        seeds.push_back(std::move(s1));

        // Seeds 2..5: 2D row-wrapped grids with 2, 3, 4 rows (and sqrt(K) rows).
        auto make_grid_perm = [](size_t K, size_t n_rows, std::vector<uint32_t>& gp, std::vector<uint32_t>& gm) {
            gp.clear();
            gm.clear();
            if (K == 0) return;
            const size_t rows = std::max<size_t>(1, std::min(n_rows, K));
            const size_t cols = (K + rows - 1) / rows;
            // For a 2D grid where row r (0=bottom..rows-1=top) contains items left-to-right:
            // gp lists rows TOP-to-BOTTOM, left-to-right within each row.
            // gm lists rows BOTTOM-to-TOP, left-to-right within each row.
            for (size_t r = rows; r-- > 0;) {
                for (size_t c = 0; c < cols; ++c) {
                    const size_t idx = r * cols + c;
                    if (idx < K) gp.push_back(static_cast<uint32_t>(idx));
                }
            }
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    const size_t idx = r * cols + c;
                    if (idx < K) gm.push_back(static_cast<uint32_t>(idx));
                }
            }
        };
        for (size_t nr : {size_t{2}, size_t{3}, size_t{4},
                          std::max<size_t>(2, static_cast<size_t>(std::round(std::sqrt(static_cast<double>(ctx.core_units.size())))))}) {
            FullState sg = make_identity_state();
            make_grid_perm(ctx.core_units.size(), nr, sg.core_gp, sg.core_gm);
            make_grid_perm(ctx.floating_units.size(), 2, sg.float_gp, sg.float_gm);
            for (auto& in : sg.inner) {
                if (in.gp.size() >= 3) make_grid_perm(in.gp.size(), 2, in.gp, in.gm);
            }
            seeds.push_back(std::move(sg));
        }

        // Seed 6: descending-area sorted grid.
        FullState sa = make_identity_state();
        std::vector<uint32_t> order(ctx.core_units.size());
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const auto& ua = ctx.units[ctx.core_units[a]];
            const auto& ub = ctx.units[ctx.core_units[b]];
            int64_t aa = 0, ab = 0;
            for (uint32_t m : ua.members) aa += prob.groups[m].fw * prob.groups[m].fh;
            for (uint32_t m : ub.members) ab += prob.groups[m].fw * prob.groups[m].fh;
            return aa > ab;
        });
        std::vector<uint32_t> gtmp, mtmp;
        make_grid_perm(order.size(), 2, gtmp, mtmp);
        for (size_t i = 0; i < order.size(); ++i) {
            sa.core_gp[i] = order[gtmp[i]];
            sa.core_gm[i] = order[mtmp[i]];
        }
        seeds.push_back(std::move(sa));
    }

    for (const auto& s : seeds) {
        (void)try_state(s);
    }

    // Collect list of mutable inner cluster unit ids (size >= 2).
    std::vector<size_t> mutable_inners;
    for (size_t uid = 0; uid < ctx.units.size(); ++uid) {
        if (ctx.units[uid].is_cluster && ctx.units[uid].members.size() >= 2) {
            mutable_inners.push_back(uid);
        }
    }

    // Deterministic Simulated Annealing + Multi-Restart Local Search.
    // Scaled deterministically by problem size so 8-15 blocks finishes in ~10-25 ms!
    const int64_t total_iters = prob.max_iterations > 0
                                    ? prob.max_iterations
                                    : std::clamp<int64_t>(static_cast<int64_t>(n) * 450, 2500, 14000);
    const int num_restarts = 4;
    const int64_t iters_per_restart = std::max<int64_t>(400, total_iters / num_restarts);

    auto mutate_perm = [&](std::vector<uint32_t>& p, int move_kind) {
        const size_t k = p.size();
        if (k < 2) return;
        const size_t i = rng.uniform(k);
        size_t j = rng.uniform(k - 1);
        if (j >= i) ++j;
        if (move_kind == 0) {
            std::swap(p[i], p[j]);
        } else {
            // Relocate element at i to position j.
            const uint32_t val = p[i];
            if (i < j) {
                for (size_t t = i; t < j; ++t) p[t] = p[t + 1];
            } else {
                for (size_t t = i; t > j; --t) p[t] = p[t - 1];
            }
            p[j] = val;
        }
    };

    for (int restart = 0; restart < num_restarts; ++restart) {
        FullState cur = (restart == 0 && has_feasible) ? best_state : seeds[restart % seeds.size()];
        if (restart >= 2) {
            // Randomize start from best_state with a few kicks.
            if (has_feasible) cur = best_state;
            for (int kick = 0; kick < 3; ++kick) {
                mutate_perm(cur.core_gp, 0);
                mutate_perm(cur.core_gm, 0);
            }
        }
        int64_t cur_obj = try_state(cur);
        const double T0 = std::max(5000.0, static_cast<double>(sol.lower_bound) * 0.25);
        const double T1 = 50.0;

        for (int64_t it = 0; it < iters_per_restart; ++it) {
            FullState cand = cur;
            const uint64_t r = rng.next() % 100;
            const int move_kind = static_cast<int>(rng.next() & 1ULL);

            if (!mutable_inners.empty() && r < 28) {
                const size_t uid = mutable_inners[rng.uniform(mutable_inners.size())];
                const uint64_t sub = rng.next() % 3;
                if (sub == 0 || sub == 2) mutate_perm(cand.inner[uid].gp, move_kind);
                if (sub == 1 || sub == 2) mutate_perm(cand.inner[uid].gm, move_kind);
            } else if (ctx.floating_units.size() >= 2 && r < 48) {
                const uint64_t sub = rng.next() % 3;
                if (sub == 0 || sub == 2) mutate_perm(cand.float_gp, move_kind);
                if (sub == 1 || sub == 2) mutate_perm(cand.float_gm, move_kind);
            } else if (ctx.core_units.size() >= 2) {
                const uint64_t sub = rng.next() % 3;
                if (sub == 0 || sub == 2) mutate_perm(cand.core_gp, move_kind);
                if (sub == 1 || sub == 2) mutate_perm(cand.core_gm, move_kind);
            } else if (!mutable_inners.empty()) {
                const size_t uid = mutable_inners[rng.uniform(mutable_inners.size())];
                mutate_perm(cand.inner[uid].gp, move_kind);
            } else {
                break;
            }

            const int64_t cand_obj = try_state(cand);
            if (cand_obj == std::numeric_limits<int64_t>::max()) continue;
            if (cand_obj <= cur_obj) {
                cur = std::move(cand);
                cur_obj = cand_obj;
            } else {
                const double progress = static_cast<double>(it) / static_cast<double>(iters_per_restart);
                const double temp = T0 * std::pow(T1 / T0, progress);
                const double diff = static_cast<double>(cand_obj - cur_obj);
                if (diff < temp * 20.0 && rng.uniform01() < std::exp(-diff / temp)) {
                    cur = std::move(cand);
                    cur_obj = cand_obj;
                }
            }
        }
    }

    // Deterministic greedy 1-swap polish around best_state.
    if (has_feasible) {
        bool improved = true;
        int sweeps = 0;
        while (improved && sweeps < 6) {
            improved = false;
            ++sweeps;
            auto sweep_perm = [&](std::vector<uint32_t>& target_perm) {
                const size_t k = target_perm.size();
                if (k < 2 || k > 24) return;
                for (size_t i = 0; i < k; ++i) {
                    for (size_t j = i + 1; j < k; ++j) {
                        std::swap(target_perm[i], target_perm[j]);
                        const int64_t prev_best = best_obj;
                        const int64_t obj = try_state(best_state);
                        if (obj < prev_best) {
                            improved = true;
                        } else {
                            std::swap(target_perm[i], target_perm[j]);
                        }
                    }
                }
            };
            sweep_perm(best_state.core_gp);
            sweep_perm(best_state.core_gm);
            sweep_perm(best_state.float_gp);
            sweep_perm(best_state.float_gm);
            for (size_t uid : mutable_inners) {
                sweep_perm(best_state.inner[uid].gp);
                sweep_perm(best_state.inner[uid].gm);
            }
        }

        // Exact coordinate-descent slack polish (optimizes HPWL, maxspan, rail, aspect within legal slack).
        polish_coordinates(ctx, best_xs, best_ys, best_obj);

        // Final strict independent verification before publishing.
        if (verify_envelope_solution(prob, best_xs, best_ys, &sol.detail)) {
            sol.xs = std::move(best_xs);
            sol.ys = std::move(best_ys);
            sol.objective = evaluate_envelope_objective(prob, sol.xs, sol.ys, &sol.W, &sol.H);
            sol.status = (sol.objective <= sol.lower_bound) ? EnvelopeStatus::Optimal : EnvelopeStatus::Feasible;
            sol.evaluations = evals;
            return sol;
        }
    }

    sol.status = EnvelopeStatus::Unknown;
    sol.evaluations = evals;
    if (sol.detail.empty()) sol.detail = "no feasible placement found within deterministic budget";
    return sol;
}

}  // namespace clay_pnr::place
