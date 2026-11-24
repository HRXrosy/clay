#pragma once

// Pure C++20 flat contiguous-array 2D BVH for exact integer rectangle clearance,
// conflict witness selection, and intersection queries.
// Strictly free of CPython headers.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clay_pnr::place {

struct BvhNode {
    core::Box64 bbox{};
    uint32_t first = 0;   // internal: left child idx; leaf: start idx in boxes
    uint32_t second = 0;  // internal: right child idx; leaf: count of boxes
    uint8_t is_leaf = 0;
};

enum class ConflictResult : int8_t {
    Cancelled = -1,
    NoConflict = 0,
    Conflict = 1,
};

struct ConflictWitness {
    core::Box64 box{};
    int64_t clearance_nm = 0;
    core::U128 area{};
};

class BvhIndex {
public:
    BvhIndex() = default;

    // Builds the flat BVH over `input_boxes` (taking ownership).
    // Returns false iff cancelled via `poller`.
    bool build(std::vector<core::Box64> input_boxes, core::CheckpointPoller& poller);
    bool build(std::vector<core::Box64> input_boxes, core::CheckpointHook hook = {});

    void clear() noexcept {
        boxes_.clear();
        nodes_.clear();
    }

    bool empty() const noexcept { return nodes_.empty(); }
    size_t size() const noexcept { return boxes_.size(); }
    const std::vector<core::Box64>& boxes() const noexcept { return boxes_; }
    const std::vector<BvhNode>& nodes() const noexcept { return nodes_; }

    // Exact L-infinity clearance conflict check for one query box `q`.
    ConflictResult query_conflict_one(const core::Box64& q, int64_t gap,
                                      core::CheckpointPoller& visit_poller) const;

    ConflictResult query_conflict_one(const core::Box64& q, int64_t gap) const;

    // Batch clearance conflict check over `queries`, short-circuiting on first conflict.
    ConflictResult query_conflict_any(const std::vector<core::Box64>& queries, int64_t gap,
                                      core::CheckpointPoller& visit_poller) const;

    // Deterministic conflict witness selection: among all indexed boxes with
    // box_clearance_linf(b, q) < gap, chooses minimum clearance, breaking ties by
    // maximum exact 128-bit area and then canonical Box64 lexicographic order.
    bool find_conflict_witness(const core::Box64& q, int64_t gap,
                               ConflictWitness& out_witness) const noexcept;

    // Collects all indexed boxes intersecting `q` (including edge/corner contacts iff
    // `include_touching` is true). Repeated boxes remain repeated.
    void query_intersecting(const core::Box64& q, bool include_touching,
                            std::vector<core::Box64>& out) const;

private:
    int32_t build_range(uint32_t lo, uint32_t hi, core::CheckpointPoller& poller);

    std::vector<core::Box64> boxes_;
    std::vector<BvhNode> nodes_;
};

}  // namespace clay_pnr::place
