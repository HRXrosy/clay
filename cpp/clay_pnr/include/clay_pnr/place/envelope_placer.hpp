#pragma once

// Pure C++20 Exact-Constraint 2D Envelope-Min Analog Placer.
// Strictly free of <Python.h>, OR-Tools, and KLayout.
//
// Solves the exact mathematical model of `clay.layout.synth.placement.place_envelope_min`:
//   - Hard AddNoOverlap2D over integer-grid padded footprints (fw[i], fh[i])
//   - Hard disjoint_clusters (guard-ring islands) with per-island boundary pad `island_pad`
//     on all 4 sides and hull-vs-hull / hull-vs-unclustered NoOverlap2D, plus intra-island
//     hull tightness (iw + ih) in the objective
//   - Hard band_of_group tier separation (lower band strictly below higher band) +
//     floating-tile right-column banking (xs[f] >= mos_right, 2*mos_lo <= 2*ys[f]+fh[f] <= 2*mos_hi)
//   - Hard order_constraints (left_of, above)
//   - Hard symmetry_pairs (ys[a] == ys[b] and shared mirror axis 2*xs[a]+2*xs[b]+fw[a]+fw[b] == const)
//   - Shared axis_groups (`shared_hard` and `centered_soft`, +-1 parity quantum + cdev=|axis2-W|)
//   - Exact integer objective:
//       1000 * (W + H + sum_{len>=2}(iw + ih))
//     +   10 * |100*W - q*H|            (if aspect_target > 0)
//     +  200 * sum(center_terms)
//     + maxspan_weight * max_signal_span
//     + wl_weight * sum(weighted_net_spans)
//     +  100 * sum(abut_l1_terms)
//     +  150 * sum(rail_affinity_terms)
//
// Deterministic (fixed SplitMix64 seed, single-threaded), verified by an independent
// hard-constraint checker before returning any incumbent.

#include <cstdint>
#include <string>
#include <vector>

namespace clay_pnr::place {

struct EnvelopeGroup {
    std::string key;
    double raw_w = 0.0;
    double raw_h = 0.0;
    int64_t fw = 1;  // ceil(raw_w + gap)
    int64_t fh = 1;  // ceil(raw_h + gap)
};

struct EnvelopeNet {
    std::vector<uint32_t> members;
    int64_t weight_scaled = 100;  // 1 if legacy unweighted, else max(1, round(w_net * 100))
    bool is_signal = false;
};

struct EnvelopeOrder {
    uint32_t a = 0;
    uint32_t b = 0;
    bool is_above = false;  // false => a left_of b; true => a above b (y[b]+fh[b] <= y[a])
};

struct EnvelopeCluster {
    std::vector<uint32_t> members;
    int64_t pad = 0;  // ceil(pad_um)
};

struct EnvelopeAxisGroup {
    uint32_t group = 0;
    int64_t offset_rounded = 0;  // round(off)
    bool is_primary = false;
};

struct EnvelopeRailAffinity {
    uint32_t group = 0;
    bool is_top = false;  // false => bottom (minimize y), true => top (minimize H - (y+fh))
};

struct EnvelopeProblem {
    std::vector<EnvelopeGroup> groups;
    std::vector<EnvelopeNet> nets;
    bool has_net_weights = false;
    std::vector<std::pair<uint32_t, uint32_t>> symmetry_pairs;
    std::vector<EnvelopeAxisGroup> axis_groups;
    bool axis_centered_soft = false;  // false => shared_hard, true => centered_soft
    std::vector<EnvelopeCluster> clusters;
    std::vector<int32_t> band_of_group;  // empty => unbanded; else size N, INT32_MIN => floating
    std::vector<EnvelopeOrder> orders;
    std::vector<std::pair<uint32_t, uint32_t>> abut_pairs;
    std::vector<EnvelopeRailAffinity> rail_affinities;

    double gap = 0.5;
    int64_t wl_weight = 1;
    int64_t maxspan_weight = 3000;
    double aspect_target = 0.0;  // <= 0 => disabled
    int64_t aspect_q = 0;        // round(aspect_target * 100)
    double max_time_s = 5.0;
    int64_t max_iterations = 0;  // 0 => deterministic default scaled by problem size
    uint64_t seed = 0;
};

enum class EnvelopeStatus : uint8_t {
    Optimal = 0,
    Feasible = 1,
    Infeasible = 2,
    Unknown = 3,
};

struct EnvelopeSolution {
    EnvelopeStatus status = EnvelopeStatus::Unknown;
    std::vector<int64_t> xs;
    std::vector<int64_t> ys;
    int64_t W = 0;
    int64_t H = 0;
    int64_t objective = 0;
    int64_t lower_bound = 0;
    uint64_t evaluations = 0;
    std::string detail;
};

// Independent exact hard-constraint verifier for any candidate integer placement (xs, ys).
// Returns true iff EVERY hard constraint in `prob` is satisfied exactly.
bool verify_envelope_solution(const EnvelopeProblem& prob,
                              const std::vector<int64_t>& xs,
                              const std::vector<int64_t>& ys,
                              std::string* reason = nullptr);

// Evaluates exact integer objective matching `clay/layout/synth/placement.py` verbatim.
int64_t evaluate_envelope_objective(const EnvelopeProblem& prob,
                                    const std::vector<int64_t>& xs,
                                    const std::vector<int64_t>& ys,
                                    int64_t* out_W = nullptr,
                                    int64_t* out_H = nullptr);

// Pure C++20 deterministic 2D Envelope-Min solver.
EnvelopeSolution solve_envelope_min(const EnvelopeProblem& prob);

}  // namespace clay_pnr::place

