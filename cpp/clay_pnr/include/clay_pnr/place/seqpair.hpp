#pragma once

// Pure C++20 sequence-pair constraint graph longest-path and band/block layout search kernels.
// Strictly free of CPython headers.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clay_pnr::place {

struct DagEdge {
    uint32_t u = 0;
    uint32_t v = 0;
    double weight = 0.0;
    bool has_weight = true;
    uint32_t raw_index = 0;  // original edge index for caller diagnostics
};

enum class LongestPathStatus : uint8_t {
    Ok = 0,
    MissingWeight = 1,
    CycleDetected = 2,
};

struct LongestPathResult {
    LongestPathStatus status = LongestPathStatus::Ok;
    std::vector<double> positions;
    size_t resolved_count = 0;
    uint32_t missing_edge_index = 0;
};

// Kahn topological-sort longest path on a weighted constraint DAG.
// Preserves exact edge insertion order and reports missing weights only on visited edges.
LongestPathResult longest_path_positions(size_t num_nodes,
                                         const std::vector<DagEdge>& edges);

struct SpCost {
    int out_of_band = 0;
    double area = 0.0;
};

inline constexpr SpCost cost_from_dims(double W, double H) noexcept {
    const double mn = W < H ? W : H;
    const double mx = W > H ? W : H;
    if (mn <= 0.0) {
        return SpCost{1, 1e18};
    }
    const double area = W * H;
    const double asp = mx / mn;
    const int oob = (1.4 <= asp && asp <= 1.9) ? 0 : 1;
    return SpCost{oob, area};
}

inline constexpr bool sp_cost_better(const SpCost& c, const SpCost& ref) noexcept {
    if (c.out_of_band != ref.out_of_band) {
        return c.out_of_band < ref.out_of_band;
    }
    return c.area < ref.area - 1e-3;
}

enum class PairRel : uint8_t {
    ILeftJ = 0,   // "i_left_j"
    JLeftI = 1,   // "j_left_i"
    IBelowJ = 2,  // "i_below_j"
    JBelowI = 3,  // "j_below_i"
};

struct PairRelEntry {
    int i = 0;
    int j = 0;
    PairRel rel = PairRel::ILeftJ;
};

struct PackBandSpResult {
    double width = 0.0;
    double height = 0.0;
    std::vector<int> pos_plus;
    std::vector<int> pos_minus;
};

struct SearchLayoutResult {
    int n = 0;
    std::vector<int> pos_plus;
    std::vector<int> pos_minus;
};

// Evaluates exact block-level sequence-pair (W, H) dimensions into caller-owned scratch buffers.
void eval_band_sp_dims(const int* gp, const int* gm, int n,
                       const double* bw, const double* bh, double gap,
                       int* pos_plus, int* pos_minus,
                       double* xc, double* yc,
                       double& out_W, double& out_H) noexcept;

// Packs a band/block-level sequence-pair (gp, gm) to (W, H, pos_plus, pos_minus).
PackBandSpResult pack_band_sp(const std::vector<int>& gp,
                              const std::vector<int>& gm,
                              const std::vector<double>& bw,
                              const std::vector<double>& bh,
                              double gap);

// Extracts canonical pairwise relative relations for all 0 <= i < j < n.
std::vector<PairRelEntry> extract_pair_relations(const std::vector<int>& pos_plus,
                                                 const std::vector<int>& pos_minus);

// Deterministic exhaustive (n <= 6) / pairwise-swap hill-climb (n > 6) block layout search.
SearchLayoutResult search_block_layout(const std::vector<double>& bw,
                                       const std::vector<double>& bh,
                                       double dev_gap);

// Deterministic device-level band area search given CSR-packed band device half-widths/heights.
// `band_off` has size (n_bands + 1); `half_w` and `half_h` have size total_devs.
SearchLayoutResult search_band_dev_layout(const std::vector<uint32_t>& band_off,
                                          const std::vector<double>& half_w,
                                          const std::vector<double>& half_h,
                                          double dev_gap);

}  // namespace clay_pnr::place
