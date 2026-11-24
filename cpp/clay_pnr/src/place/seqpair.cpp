// Pure C++20 implementation of sequence-pair longest path & layout search kernels.
// Strictly free of CPython headers.

#include "clay_pnr/place/seqpair.hpp"

#include <algorithm>
#include <utility>

namespace clay_pnr::place {

namespace {

struct CsrEdge {
    uint32_t to = 0;
    double w = 0.0;
    uint8_t has_w = 0;
    uint32_t raw_index = 0;
};

}  // namespace

LongestPathResult longest_path_positions(size_t num_nodes,
                                         const std::vector<DagEdge>& edges) {
    LongestPathResult result{};
    std::vector<uint32_t> out_deg(num_nodes, 0);
    std::vector<uint32_t> indeg(num_nodes, 0);

    for (const DagEdge& e : edges) {
        ++out_deg[e.u];
        ++indeg[e.v];
    }

    // Build CSR adjacency list preserving exact edge insertion order.
    std::vector<uint32_t> head(num_nodes + 1, 0);
    for (size_t i = 0; i < num_nodes; ++i) {
        head[i + 1] = head[i] + out_deg[i];
    }
    std::vector<uint32_t> cur = head;
    std::vector<CsrEdge> csr(edges.size());
    for (const DagEdge& e : edges) {
        const uint32_t pos_idx = cur[e.u]++;
        csr[pos_idx] = CsrEdge{e.v, e.weight, static_cast<uint8_t>(e.has_weight ? 1 : 0), e.raw_index};
    }

    // Kahn topological longest path.
    std::vector<double> pos(num_nodes, 0.0);
    std::vector<uint32_t> queue;
    queue.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        if (indeg[i] == 0) {
            queue.push_back(static_cast<uint32_t>(i));
        }
    }

    size_t q_head = 0;
    size_t seen = 0;
    while (q_head < queue.size()) {
        const uint32_t u = queue[q_head++];
        ++seen;
        const double pu = pos[u];
        const uint32_t e_begin = head[u];
        const uint32_t e_end = head[u + 1];
        for (uint32_t ei = e_begin; ei < e_end; ++ei) {
            const CsrEdge& ew = csr[ei];
            if (!ew.has_w) {
                result.status = LongestPathStatus::MissingWeight;
                result.missing_edge_index = ew.raw_index;
                result.resolved_count = seen;
                return result;
            }
            const double cand = pu + ew.w;
            if (cand > pos[ew.to]) {
                pos[ew.to] = cand;
            }
            if (--indeg[ew.to] == 0) {
                queue.push_back(ew.to);
            }
        }
    }

    result.resolved_count = seen;
    if (seen != num_nodes) {
        result.status = LongestPathStatus::CycleDetected;
        return result;
    }

    result.status = LongestPathStatus::Ok;
    result.positions = std::move(pos);
    return result;
}

void eval_band_sp_dims(const int* gp, const int* gm, int n,
                       const double* bw, const double* bh, double gap,
                       int* pos_plus, int* pos_minus,
                       double* xc, double* yc,
                       double& out_W, double& out_H) noexcept {
    for (int k = 0; k < n; ++k) {
        pos_plus[gp[k]] = k;
        pos_minus[gm[k]] = k;
        xc[k] = 0.0;
        yc[k] = 0.0;
    }
    // H longest path in gamma_plus topological order.
    for (int a = 0; a < n; ++a) {
        const int u = gp[a];
        const double pu = xc[u];
        const double wu = bw[u];
        const int mu = pos_minus[u];
        for (int b = a + 1; b < n; ++b) {
            const int v = gp[b];
            if (mu < pos_minus[v]) {
                const double cand = pu + (wu + bw[v]) * 0.5 + gap;
                if (cand > xc[v]) xc[v] = cand;
            }
        }
    }
    // V longest path in gamma_minus topological order.
    for (int a = 0; a < n; ++a) {
        const int u = gm[a];
        const double pu = yc[u];
        const double hu = bh[u];
        const int pu_idx = pos_plus[u];
        for (int b = a + 1; b < n; ++b) {
            const int v = gm[b];
            if (pu_idx > pos_plus[v]) {
                const double cand = pu + (hu + bh[v]) * 0.5 + gap;
                if (cand > yc[v]) yc[v] = cand;
            }
        }
    }
    double x_lo = xc[0] - bw[0] * 0.5;
    double x_hi = xc[0] + bw[0] * 0.5;
    double y_lo = yc[0] - bh[0] * 0.5;
    double y_hi = yc[0] + bh[0] * 0.5;
    for (int k = 1; k < n; ++k) {
        const double xl = xc[k] - bw[k] * 0.5;
        const double xh = xc[k] + bw[k] * 0.5;
        const double yl = yc[k] - bh[k] * 0.5;
        const double yh = yc[k] + bh[k] * 0.5;
        if (xl < x_lo) x_lo = xl;
        if (xh > x_hi) x_hi = xh;
        if (yl < y_lo) y_lo = yl;
        if (yh > y_hi) y_hi = yh;
    }
    out_W = x_hi - x_lo;
    out_H = y_hi - y_lo;
}

PackBandSpResult pack_band_sp(const std::vector<int>& gp,
                              const std::vector<int>& gm,
                              const std::vector<double>& bw,
                              const std::vector<double>& bh,
                              double gap) {
    PackBandSpResult res{};
    const int n = static_cast<int>(bw.size());
    if (n <= 0) {
        return res;
    }
    res.pos_plus.resize(static_cast<size_t>(n));
    res.pos_minus.resize(static_cast<size_t>(n));
    std::vector<double> xc(static_cast<size_t>(n));
    std::vector<double> yc(static_cast<size_t>(n));
    eval_band_sp_dims(gp.data(), gm.data(), n, bw.data(), bh.data(), gap,
                      res.pos_plus.data(), res.pos_minus.data(),
                      xc.data(), yc.data(), res.width, res.height);
    return res;
}

std::vector<PairRelEntry> extract_pair_relations(const std::vector<int>& pos_plus,
                                                 const std::vector<int>& pos_minus) {
    const int n = static_cast<int>(pos_plus.size());
    std::vector<PairRelEntry> out;
    if (n < 2) return out;
    out.reserve(static_cast<size_t>(n) * static_cast<size_t>(n - 1) / 2);
    for (int i = 0; i < n; ++i) {
        const int pi = pos_plus[static_cast<size_t>(i)];
        const int mi = pos_minus[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            const bool i_bp = pi < pos_plus[static_cast<size_t>(j)];
            const bool i_bm = mi < pos_minus[static_cast<size_t>(j)];
            PairRel rel = PairRel::IBelowJ;
            if (i_bp && i_bm) rel = PairRel::ILeftJ;
            else if (!i_bp && !i_bm) rel = PairRel::JLeftI;
            else if (i_bp && !i_bm) rel = PairRel::JBelowI;
            out.push_back(PairRelEntry{i, j, rel});
        }
    }
    return out;
}

SearchLayoutResult search_block_layout(const std::vector<double>& bw,
                                       const std::vector<double>& bh,
                                       double dev_gap) {
    SearchLayoutResult res{};
    const int n = static_cast<int>(bw.size());
    if (n < 2) {
        res.n = n;
        return res;
    }

    std::vector<int> pos_plus(static_cast<size_t>(n));
    std::vector<int> pos_minus(static_cast<size_t>(n));
    std::vector<double> xc(static_cast<size_t>(n));
    std::vector<double> yc(static_cast<size_t>(n));

    auto eval_cost = [&](const std::vector<int>& gp_cur,
                         const std::vector<int>& gm_cur) noexcept -> SpCost {
        double W = 0.0;
        double H = 0.0;
        eval_band_sp_dims(gp_cur.data(), gm_cur.data(), n, bw.data(), bh.data(), dev_gap,
                          pos_plus.data(), pos_minus.data(), xc.data(), yc.data(), W, H);
        return cost_from_dims(W, H);
    };

    // Seed: vertical stack (band 0 bottom -> band n-1 top).
    std::vector<int> gp(static_cast<size_t>(n));
    std::vector<int> gm(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        gp[static_cast<size_t>(i)] = n - 1 - i;
        gm[static_cast<size_t>(i)] = i;
    }
    SpCost best_cost = eval_cost(gp, gm);

    if (n <= 6) {
        // Exhaustive over both permutations in exact lexicographic order (n! <= 720).
        std::vector<int> best_gp = gp;
        std::vector<int> best_gm = gm;
        std::vector<int> cgp(static_cast<size_t>(n));
        std::vector<int> cgm(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) cgp[static_cast<size_t>(i)] = i;
        do {
            for (int i = 0; i < n; ++i) cgm[static_cast<size_t>(i)] = i;
            do {
                const SpCost c = eval_cost(cgp, cgm);
                if (sp_cost_better(c, best_cost)) {
                    best_cost = c;
                    best_gp = cgp;
                    best_gm = cgm;
                }
            } while (std::next_permutation(cgm.begin(), cgm.end()));
        } while (std::next_permutation(cgp.begin(), cgp.end()));
        gp.swap(best_gp);
        gm.swap(best_gm);
    } else {
        // Deterministic monotone pairwise-swap hill climb matching Python's exact snapshot semantics.
        bool improved = true;
        int sweeps = 0;
        std::vector<int> cand(static_cast<size_t>(n));
        while (improved && sweeps < 200) {
            improved = false;
            ++sweeps;
            for (int which = 0; which < 2; ++which) {
                const std::vector<int> perm = (which == 0) ? gp : gm;
                for (int a = 0; a < n; ++a) {
                    for (int b = a + 1; b < n; ++b) {
                        cand = perm;
                        std::swap(cand[static_cast<size_t>(a)], cand[static_cast<size_t>(b)]);
                        const SpCost c = (which == 0) ? eval_cost(cand, gm) : eval_cost(gp, cand);
                        if (sp_cost_better(c, best_cost)) {
                            best_cost = c;
                            if (which == 0) {
                                gp = cand;
                            } else {
                                gm = cand;
                            }
                            improved = true;
                        }
                    }
                }
            }
        }
    }

    for (int k = 0; k < n; ++k) {
        pos_plus[static_cast<size_t>(gp[static_cast<size_t>(k)])] = k;
        pos_minus[static_cast<size_t>(gm[static_cast<size_t>(k)])] = k;
    }
    res.n = n;
    res.pos_plus = std::move(pos_plus);
    res.pos_minus = std::move(pos_minus);
    return res;
}

SearchLayoutResult search_band_dev_layout(const std::vector<uint32_t>& band_off,
                                          const std::vector<double>& half_w,
                                          const std::vector<double>& half_h,
                                          double dev_gap) {
    SearchLayoutResult res{};
    if (band_off.size() < 3 || half_w.empty()) {
        return res;
    }
    const int n = static_cast<int>(band_off.size() - 1);
    const size_t total_devs = half_w.size();

    std::vector<int> pos_plus(static_cast<size_t>(n));
    std::vector<int> pos_minus(static_cast<size_t>(n));
    std::vector<double> xc(total_devs);
    std::vector<double> yc(total_devs);

    auto eval_dev_cost = [&](const std::vector<int>& gp_cur,
                             const std::vector<int>& gm_cur) noexcept -> SpCost {
        for (int k = 0; k < n; ++k) {
            pos_plus[static_cast<size_t>(gp_cur[static_cast<size_t>(k)])] = k;
            pos_minus[static_cast<size_t>(gm_cur[static_cast<size_t>(k)])] = k;
        }
        std::fill(xc.begin(), xc.end(), 0.0);
        std::fill(yc.begin(), yc.end(), 0.0);

        // H longest-path: iterate bands in gamma_plus topological order.
        for (int a = 0; a < n; ++a) {
            const int br = gp_cur[static_cast<size_t>(a)];
            const uint32_t b_lo = band_off[static_cast<size_t>(br)];
            const uint32_t b_hi = band_off[static_cast<size_t>(br) + 1];
            // Within-band chain (left-to-right).
            for (uint32_t k = b_lo; k + 1 < b_hi; ++k) {
                const double cand = xc[k] + (half_w[k] + half_w[k + 1] + dev_gap);
                if (cand > xc[k + 1]) xc[k + 1] = cand;
            }
            // Cross-band H edges: every u in br -> every v in bs (when br left_of bs).
            const int m_br = pos_minus[static_cast<size_t>(br)];
            for (int b = a + 1; b < n; ++b) {
                const int bs = gp_cur[static_cast<size_t>(b)];
                if (m_br < pos_minus[static_cast<size_t>(bs)]) {
                    const uint32_t s_lo = band_off[static_cast<size_t>(bs)];
                    const uint32_t s_hi = band_off[static_cast<size_t>(bs) + 1];
                    for (uint32_t u = b_lo; u < b_hi; ++u) {
                        const double xu = xc[u];
                        const double hwu = half_w[u];
                        for (uint32_t v = s_lo; v < s_hi; ++v) {
                            const double cand = xu + (hwu + half_w[v] + dev_gap);
                            if (cand > xc[v]) xc[v] = cand;
                        }
                    }
                }
            }
        }

        // V longest-path: iterate bands in gamma_minus topological order.
        for (int a = 0; a < n; ++a) {
            const int br = gm_cur[static_cast<size_t>(a)];
            const uint32_t b_lo = band_off[static_cast<size_t>(br)];
            const uint32_t b_hi = band_off[static_cast<size_t>(br) + 1];
            const int p_br = pos_plus[static_cast<size_t>(br)];
            for (int b = a + 1; b < n; ++b) {
                const int bs = gm_cur[static_cast<size_t>(b)];
                if (p_br > pos_plus[static_cast<size_t>(bs)]) {
                    const uint32_t s_lo = band_off[static_cast<size_t>(bs)];
                    const uint32_t s_hi = band_off[static_cast<size_t>(bs) + 1];
                    for (uint32_t u = b_lo; u < b_hi; ++u) {
                        const double yu = yc[u];
                        const double hhu = half_h[u];
                        for (uint32_t v = s_lo; v < s_hi; ++v) {
                            const double cand = yu + (hhu + half_h[v] + dev_gap);
                            if (cand > yc[v]) yc[v] = cand;
                        }
                    }
                }
            }
        }

        double x_lo = xc[0] - half_w[0];
        double x_hi = xc[0] + half_w[0];
        double y_lo = yc[0] - half_h[0];
        double y_hi = yc[0] + half_h[0];
        for (size_t d = 1; d < total_devs; ++d) {
            const double xl = xc[d] - half_w[d];
            const double xh = xc[d] + half_w[d];
            const double yl = yc[d] - half_h[d];
            const double yh = yc[d] + half_h[d];
            if (xl < x_lo) x_lo = xl;
            if (xh > x_hi) x_hi = xh;
            if (yl < y_lo) y_lo = yl;
            if (yh > y_hi) y_hi = yh;
        }
        return cost_from_dims(x_hi - x_lo, y_hi - y_lo);
    };

    std::vector<int> gp(static_cast<size_t>(n));
    std::vector<int> gm(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        gp[static_cast<size_t>(i)] = n - 1 - i;
        gm[static_cast<size_t>(i)] = i;
    }
    SpCost best_cost = eval_dev_cost(gp, gm);

    if (n <= 6) {
        std::vector<int> best_gp = gp;
        std::vector<int> best_gm = gm;
        std::vector<int> cgp(static_cast<size_t>(n));
        std::vector<int> cgm(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) cgp[static_cast<size_t>(i)] = i;
        do {
            for (int i = 0; i < n; ++i) cgm[static_cast<size_t>(i)] = i;
            do {
                const SpCost c = eval_dev_cost(cgp, cgm);
                if (sp_cost_better(c, best_cost)) {
                    best_cost = c;
                    best_gp = cgp;
                    best_gm = cgm;
                }
            } while (std::next_permutation(cgm.begin(), cgm.end()));
        } while (std::next_permutation(cgp.begin(), cgp.end()));
        gp.swap(best_gp);
        gm.swap(best_gm);
    } else {
        bool improved = true;
        int sweeps = 0;
        std::vector<int> cand(static_cast<size_t>(n));
        while (improved && sweeps < 200) {
            improved = false;
            ++sweeps;
            for (int which = 0; which < 2; ++which) {
                const std::vector<int> perm = (which == 0) ? gp : gm;
                for (int a = 0; a < n; ++a) {
                    for (int b = a + 1; b < n; ++b) {
                        cand = perm;
                        std::swap(cand[static_cast<size_t>(a)], cand[static_cast<size_t>(b)]);
                        const SpCost c = (which == 0) ? eval_dev_cost(cand, gm)
                                                      : eval_dev_cost(gp, cand);
                        if (sp_cost_better(c, best_cost)) {
                            best_cost = c;
                            if (which == 0) {
                                gp = cand;
                            } else {
                                gm = cand;
                            }
                            improved = true;
                        }
                    }
                }
            }
        }
    }

    for (int k = 0; k < n; ++k) {
        pos_plus[static_cast<size_t>(gp[static_cast<size_t>(k)])] = k;
        pos_minus[static_cast<size_t>(gm[static_cast<size_t>(k)])] = k;
    }
    res.n = n;
    res.pos_plus = std::move(pos_plus);
    res.pos_minus = std::move(pos_minus);
    return res;
}

}  // namespace clay_pnr::place
