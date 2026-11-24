#pragma once

// Pure C++20 Banded 2D Area / Height Macro Placer (Phase 3B).
// Strictly free of <Python.h>, OR-Tools, and SciPy.
// Solves deterministic integer-nm banded macro placement with:
//   - bottom/center/top band stacking (optional strict band order)
//   - lateral full-height blocks
//   - hard horizontal/vertical order pairs
//   - pair-specific clearance overrides
//   - bottom-edge / bottom-center / top-edge keepouts
//   - left/right hard edge anchors & soft pulls + pull pairs
//   - manufacturing grid snapping
//   - exact scaled aspect-ratio bounds (W * scale in [H * lower, H * upper])
//   - multi-width skyline + 2D DAG compaction + deterministic local search

#include "clay_pnr/core/geometry.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace clay_pnr::place {

struct BandedBlock {
    std::string name;
    int64_t w = 0;
    int64_t h = 0;
    int32_t band = 0;
    bool lateral = false;
    bool left_edge = false;
    bool right_edge = false;
    bool left_pull = false;
    bool right_pull = false;
    int64_t y_min_keepout = 0;
    bool top_keepout_clear = false;
};

struct BandedPairClearance {
    int32_t a = 0;
    int32_t b = 0;
    int64_t gap_nm = 0;
};

struct BandedProblem {
    std::vector<BandedBlock> blocks;
    int64_t gap_nm = 0;
    int64_t grid_nm = 1;
    int64_t w_cap_nm = 0;
    int64_t h_bound_nm = 0;
    int64_t bottom_edge_keepout_nm = 0;
    int64_t top_edge_keepout_nm = 0;
    bool strict_width_cap = false;
    bool exact_outline_width = false;
    bool enforce_band_order = true;
    bool objective_area = true;  // true=area (W*H), false=height (H)
    bool dense_2d = false;       // allow multi-row wrapping inside bands

    bool aspect_obligation = false;
    int64_t aspect_lower = 0;
    int64_t aspect_upper = 0;
    int64_t aspect_scale = 1000000;

    std::vector<std::pair<int32_t, int32_t>> order_pairs;           // x_a + w_a + gap <= x_b
    std::vector<std::pair<int32_t, int32_t>> vertical_order_pairs;  // y_a + h_a + gap <= y_b
    std::vector<BandedPairClearance> pair_clearances;
    std::vector<std::pair<int32_t, int32_t>> pull_pairs;

    int32_t max_evals = 2500;
    uint64_t seed = 0x434C41593342ULL;
};

enum class BandedStatus : int32_t {
    Optimal = 0,
    Feasible = 1,
    Infeasible = 2,
    InvalidInput = 3,
};

struct BandedSolution {
    BandedStatus status = BandedStatus::Infeasible;
    std::vector<int64_t> xs;
    std::vector<int64_t> ys;
    int64_t W = 0;
    int64_t H = 0;
    int64_t primary = 0;     // W * H if objective_area else H
    int64_t soft_cost = 0;
    int64_t lower_bound = 0;
    uint64_t evaluations = 0;
    std::string detail;
};

bool verify_banded_solution(
    const BandedProblem& prob,
    const std::vector<int64_t>& xs,
    const std::vector<int64_t>& ys,
    std::string* err = nullptr);

BandedSolution solve_banded_area(const BandedProblem& prob);

}  // namespace clay_pnr::place

