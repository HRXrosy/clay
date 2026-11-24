// Pure C++20 implementation of flat 2D BVH spatial index. Strictly free of CPython headers.

#include "clay_pnr/place/bvh_index.hpp"

#include <algorithm>
#include <utility>

namespace clay_pnr::place {

using core::Box64;
using core::CheckpointHook;
using core::CheckpointPoller;
using core::U128;
using core::box_area_u128;
using core::box_clearance_linf;

bool BvhIndex::build(std::vector<Box64> input_boxes, CheckpointPoller& poller) {
    boxes_ = std::move(input_boxes);
    nodes_.clear();
    if (boxes_.empty()) {
        return true;
    }
    nodes_.reserve(boxes_.size() / 3 + 4);
    const int32_t root = build_range(0, static_cast<uint32_t>(boxes_.size()), poller);
    if (root < 0) {
        nodes_.clear();
        return false;
    }
    return true;
}

bool BvhIndex::build(std::vector<Box64> input_boxes, CheckpointHook hook) {
    CheckpointPoller poller(hook);
    return build(std::move(input_boxes), poller);
}

int32_t BvhIndex::build_range(uint32_t lo, uint32_t hi, CheckpointPoller& poller) {
    if (!poller.call_now()) return -1;
    Box64 bb = boxes_[lo];
    for (uint32_t i = lo; i < hi; ++i) {
        if (!poller.poll_index(static_cast<size_t>(i - lo))) return -1;
        const Box64& b = boxes_[i];
        if (b.x0 < bb.x0) bb.x0 = b.x0;
        if (b.y0 < bb.y0) bb.y0 = b.y0;
        if (b.x1 > bb.x1) bb.x1 = b.x1;
        if (b.y1 > bb.y1) bb.y1 = b.y1;
    }
    const uint32_t count = hi - lo;
    const uint32_t node_idx = static_cast<uint32_t>(nodes_.size());
    if (count <= 8) {
        nodes_.push_back(BvhNode{bb, lo, count, 1});
        return static_cast<int32_t>(node_idx);
    }
    nodes_.push_back(BvhNode{bb, 0, 0, 0});

    const bool split_x = (bb.x1 - bb.x0) >= (bb.y1 - bb.y0);
    const uint32_t mid = lo + (count >> 1);

    if (split_x) {
        std::nth_element(
            boxes_.begin() + lo, boxes_.begin() + mid, boxes_.begin() + hi,
            [](const Box64& a, const Box64& b) noexcept {
                const int64_t ka = a.x0 + a.x1;
                const int64_t kb = b.x0 + b.x1;
                return ka != kb ? ka < kb : a < b;
            });
    } else {
        std::nth_element(
            boxes_.begin() + lo, boxes_.begin() + mid, boxes_.begin() + hi,
            [](const Box64& a, const Box64& b) noexcept {
                const int64_t ka = a.y0 + a.y1;
                const int64_t kb = b.y0 + b.y1;
                return ka != kb ? ka < kb : a < b;
            });
    }
    if (!poller.call_now()) return -1;

    const int32_t left_idx = build_range(lo, mid, poller);
    if (left_idx < 0) return -1;
    const int32_t right_idx = build_range(mid, hi, poller);
    if (right_idx < 0) return -1;

    nodes_[node_idx].first = static_cast<uint32_t>(left_idx);
    nodes_[node_idx].second = static_cast<uint32_t>(right_idx);
    return static_cast<int32_t>(node_idx);
}

ConflictResult BvhIndex::query_conflict_one(const Box64& q, int64_t gap,
                                            CheckpointPoller& visit_poller) const {
    if (nodes_.empty()) return ConflictResult::NoConflict;
    const BvhNode* nodes_ptr = nodes_.data();
    const Box64* boxes_ptr = boxes_.data();
    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;
    if (!visit_poller.enabled()) {
        while (top > 0) {
            const BvhNode& node = nodes_ptr[stack[--top]];
            if (box_clearance_linf(node.bbox, q) >= gap) {
                continue;
            }
            if (node.is_leaf) {
                const uint32_t end = node.first + node.second;
                for (uint32_t i = node.first; i < end; ++i) {
                    if (box_clearance_linf(boxes_ptr[i], q) < gap) {
                        return ConflictResult::Conflict;
                    }
                }
            } else {
                stack[top++] = node.second;
                stack[top++] = node.first;
            }
        }
        return ConflictResult::NoConflict;
    }
    while (top > 0) {
        const uint32_t idx = stack[--top];
        if (!visit_poller.poll_step()) return ConflictResult::Cancelled;
        const BvhNode& node = nodes_ptr[idx];
        if (box_clearance_linf(node.bbox, q) >= gap) {
            continue;
        }
        if (node.is_leaf) {
            const uint32_t end = node.first + node.second;
            for (uint32_t i = node.first; i < end; ++i) {
                if (box_clearance_linf(boxes_ptr[i], q) < gap) {
                    return ConflictResult::Conflict;
                }
            }
        } else {
            stack[top++] = node.second;
            stack[top++] = node.first;
        }
    }
    return ConflictResult::NoConflict;
}

ConflictResult BvhIndex::query_conflict_one(const Box64& q, int64_t gap) const {
    CheckpointPoller no_poll;
    return query_conflict_one(q, gap, no_poll);
}

ConflictResult BvhIndex::query_conflict_any(const std::vector<Box64>& queries, int64_t gap,
                                            CheckpointPoller& visit_poller) const {
    if (nodes_.empty()) return ConflictResult::NoConflict;
    for (const Box64& q : queries) {
        const ConflictResult rc = query_conflict_one(q, gap, visit_poller);
        if (rc != ConflictResult::NoConflict) {
            return rc;
        }
    }
    return ConflictResult::NoConflict;
}

bool BvhIndex::find_conflict_witness(const Box64& q, int64_t gap,
                                     ConflictWitness& out_witness) const noexcept {
    if (nodes_.empty()) return false;
    const BvhNode* nodes_ptr = nodes_.data();
    const Box64* boxes_ptr = boxes_.data();
    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    bool found = false;
    ConflictWitness best{};

    while (top > 0) {
        const BvhNode& node = nodes_ptr[stack[--top]];
        const int64_t node_dist = box_clearance_linf(node.bbox, q);
        if (node_dist >= gap) {
            continue;
        }
        if (found && node_dist > best.clearance_nm) {
            continue;
        }
        if (node.is_leaf) {
            const uint32_t end = node.first + node.second;
            for (uint32_t i = node.first; i < end; ++i) {
                const Box64& b = boxes_ptr[i];
                const int64_t d = box_clearance_linf(b, q);
                if (d >= gap) continue;
                const U128 area = box_area_u128(b);
                bool better = false;
                if (!found || d < best.clearance_nm) {
                    better = true;
                } else if (d == best.clearance_nm) {
                    if (area > best.area) {
                        better = true;
                    } else if (area == best.area && b < best.box) {
                        better = true;
                    }
                }
                if (better) {
                    best = ConflictWitness{b, d, area};
                    found = true;
                }
            }
        } else {
            stack[top++] = node.second;
            stack[top++] = node.first;
        }
    }

    if (found) {
        out_witness = best;
    }
    return found;
}

void BvhIndex::query_intersecting(const Box64& q, bool include_touching,
                                  std::vector<Box64>& out) const {
    out.clear();
    if (nodes_.empty()) return;
    const BvhNode* nodes_ptr = nodes_.data();
    const Box64* boxes_ptr = boxes_.data();
    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const uint32_t idx = stack[--top];
        const BvhNode& node = nodes_ptr[idx];
        const int64_t dist = box_clearance_linf(node.bbox, q);
        if (dist > 0 || (dist == 0 && !include_touching)) {
            continue;
        }
        if (node.is_leaf) {
            const uint32_t end = node.first + node.second;
            for (uint32_t i = node.first; i < end; ++i) {
                const int64_t d = box_clearance_linf(boxes_ptr[i], q);
                if (d < 0 || (include_touching && d == 0)) {
                    out.push_back(boxes_ptr[i]);
                }
            }
        } else {
            stack[top++] = node.second;
            stack[top++] = node.first;
        }
    }
}

}  // namespace clay_pnr::place
