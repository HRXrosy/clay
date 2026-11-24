#pragma once

// Pure C++20 segment-tree sweep-line rectangle union and layered Minkowski-difference
// forbidden box cover kernels for Clay placement.
// Strictly free of CPython headers.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/status.hpp"

#include <cstdint>
#include <vector>

namespace clay_pnr::place {

struct SegNode {
    int32_t count = 0;
    uint8_t occupied = 0;
    uint8_t full = 0;
};

struct SweepEvent {
    int64_t x = 0;
    int32_t lo = 0;
    int32_t hi = 0;
    int32_t delta = 0;
};

struct YSpan {
    int64_t y0 = 0;
    int64_t y1 = 0;
};

struct OpenRect {
    int64_t y0 = 0;
    int64_t y1 = 0;
    int64_t start_x = 0;
    int64_t end_x = 0;
};

// Reusable scratch buffers for rectangle_union_cover to avoid repeated allocations across batches.
struct SweepWorkspace {
    std::vector<int64_t> ys;
    std::vector<SweepEvent> events;
    std::vector<SegNode> tree;
    std::vector<YSpan> spans;
    std::vector<OpenRect> open_boxes;
    std::vector<OpenRect> current_boxes;
};

// Computes exact sorted disjoint rectangle union of positive boxes.
// Output matches clay.layout.spatial.rectangle_union.rectangle_union_cover box-for-box.
// Returns false iff cancelled by `poller`.
bool rectangle_union_cover(const std::vector<core::Box64>& boxes,
                           core::CheckpointPoller& poller,
                           std::vector<core::Box64>& out,
                           SweepWorkspace& ws);

// Convenience overload allocating a temporary workspace.
bool rectangle_union_cover(const std::vector<core::Box64>& boxes,
                           std::vector<core::Box64>& out,
                           core::CheckpointHook hook = {});

struct AtomRec {
    uint32_t layer_id = 0;  // 0 == wildcard (None)
    core::Box64 box{};
};

enum class ForbiddenStatus : uint8_t {
    Ok = 0,
    Cancelled = 1,
    NonPositiveBox = 2,
    LostAllGeometry = 3,
};

struct ForbiddenCoverResult {
    ForbiddenStatus status = ForbiddenStatus::Ok;
    std::vector<core::Box64> cover;
    uint64_t raw_count = 0;
    uint64_t unique_count = 0;
};

// Exact layered Minkowski-difference streaming forbidden box cover with batched binary-tree
// union and 8-witness containment pruning.
// Preserves wide-coordinate (outside int32) and non-shrinking fallback contracts.
ForbiddenCoverResult forbidden_box_cover(const std::vector<AtomRec>& left_atoms,
                                         const std::vector<AtomRec>& right_atoms,
                                         int64_t required_nm,
                                         core::CheckpointPoller& poller);

// Convenience overload accepting a CheckpointHook directly.
ForbiddenCoverResult forbidden_box_cover(const std::vector<AtomRec>& left_atoms,
                                         const std::vector<AtomRec>& right_atoms,
                                         int64_t required_nm,
                                         core::CheckpointHook hook = {});

enum class ExactUnionStatus : uint8_t {
    Ok = 0,
    LostAllGeometry = 1,
};

// Exact forbidden union for full2d_pair_cover._exact_forbidden_union.
// Preserves signed-32-bit coordinate and span compatibility and non-shrinking fallback.
ExactUnionStatus exact_forbidden_union(std::vector<core::Box64> boxes,
                                       std::vector<core::Box64>& out);

}  // namespace clay_pnr::place
