#pragma once

// Pure C++20 exact integer Difference-Constraint 1D & 2D Compactors.
// Strictly free of <Python.h>, OR-Tools, and SciPy.
//
// Mathematical guarantee for 1D acyclic difference-constraint compaction:
//   Given extents w_i > 0 and difference constraints x_after - x_before >= delta >= 0
//   with x_i >= 0:
//   Let x_v^* = max(0, max_{(u -> v)} (x_u^* + delta(u, v))) computed in topological order.
//   By induction on topological depth, x_v^* <= x_v for EVERY feasible x, so x^*
//   SIMULTANEOUSLY minimizes:
//     (1) enclosing extent E = max_v (x_v + w_v), AND
//     (2) tie-break sum sum_v x_v (uniquely!).
//   Hence O(V + E) exact integer topological longest path reproduces the exact unique
//   optimum of two-stage LP compaction (min E, then min sum x_v) with zero floating-point
//   error and zero external solver overhead.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace clay_pnr::place {

struct IntDiffEdge {
    uint32_t before = 0;
    uint32_t after = 0;
    int64_t min_delta = 0;  // enforces pos[after] - pos[before] >= min_delta
};

enum class CompactStatus : uint8_t {
    Ok = 0,
    InvalidInput = 1,
    CycleDetected = 2,
    Infeasible = 3,
    Overflow = 4,
};

struct CompactAxisResult {
    CompactStatus status = CompactStatus::Ok;
    std::vector<int64_t> positions;
    int64_t extent = 0;
    std::string message;
};

// Exact O(V + E) integer longest-path 1D compactor for acyclic difference constraints.
// All extents must be > 0 and all edge min_delta >= 0.
CompactAxisResult compact_axis_dag(const std::vector<int64_t>& extents,
                                   const std::vector<IntDiffEdge>& edges);

// General signed difference-constraint solver (handles equality pairs u - v == c
// encoded as u - v >= c and v - u >= -c, plus lower bounds pos[i] >= lb[i]).
// Uses deterministic SPFA / Bellman-Ford longest-path with positive-cycle detection.
// Finds componentwise minimum integer positions >= lb satisfying all edges.
CompactAxisResult compact_axis_general(const std::vector<int64_t>& extents,
                                       const std::vector<IntDiffEdge>& edges,
                                       const std::vector<int64_t>& lower_bounds = {});

struct SymmetryPairSpec {
    uint32_t a = 0;
    uint32_t b = 0;
};

struct Compact2DResult {
    CompactStatus status = CompactStatus::Ok;
    std::vector<int64_t> x;
    std::vector<int64_t> y;
    int64_t width = 0;
    int64_t height = 0;
    int64_t shared_axis2 = 0;  // 2 * shared symmetry axis X (if symmetry pairs present)
    std::string message;
};

// Pure C++20 2D Difference-Constraint Compactor with:
//   - horizontal difference constraints (x[after] - x[before] >= delta)
//   - vertical difference constraints (y[after] - y[before] >= delta)
//   - optional hard symmetry pairs: y[a] == y[b] and
//     2*x[a] + 2*x[b] + widths[a] + widths[b] == shared_axis2 (constant across pairs)
//   - optional manufacturing grid snap (grid > 1 rounds lower bounds & deltas safely upward)
Compact2DResult compact_2d_topology(const std::vector<int64_t>& widths,
                                    const std::vector<int64_t>& heights,
                                    const std::vector<IntDiffEdge>& h_edges,
                                    const std::vector<IntDiffEdge>& v_edges,
                                    const std::vector<SymmetryPairSpec>& sym_pairs = {},
                                    int64_t grid_step = 1);

}  // namespace clay_pnr::place

