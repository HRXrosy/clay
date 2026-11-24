// Pure C++20 implementation of sweep-line rectangle union and layered Minkowski-difference
// forbidden box cover kernels. Strictly free of CPython headers.

#include "clay_pnr/place/sweep_union.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace clay_pnr::place {

namespace {

using core::Box64;
using core::CheckpointPoller;
using core::U128;
using core::box_area_u128;
using core::kI32Max;
using core::kI32Min;

struct PairStream {
    Box64 left_adj;  // pre-biased: {lx0 - req + 1, ly0 - req + 1, lx1 + req, ly1 + req}
    const Box64* right_ptr;
    size_t right_len;
    size_t idx;
};

struct HeapNode {
    Box64 box;
    uint32_t stream_idx;
};

inline bool right_order_less(const Box64& a, const Box64& b) noexcept {
    if (a.x1 != b.x1) return a.x1 > b.x1;
    if (a.y1 != b.y1) return a.y1 > b.y1;
    if (a.x0 != b.x0) return a.x0 > b.x0;
    return a.y0 > b.y0;
}

inline Box64 make_left_adj(const Box64& l, int64_t req) noexcept {
    return Box64{l.x0 - req + 1, l.y0 - req + 1, l.x1 + req, l.y1 + req};
}

inline Box64 minkowski_diff_adj(const Box64& la, const Box64& r) noexcept {
    return Box64{
        la.x0 - r.x1,
        la.y0 - r.y1,
        la.x1 - r.x0,
        la.y1 - r.y0,
    };
}

inline void heap_sift_down(std::vector<HeapNode>& heap, size_t i) noexcept {
    const size_t n = heap.size();
    HeapNode* data = heap.data();
    HeapNode moving = data[i];
    while (true) {
        const size_t left = (i << 1) + 1;
        if (left >= n) break;
        const size_t right = left + 1;
        size_t smallest = left;
        if (right < n && data[right].box < data[left].box) {
            smallest = right;
        }
        if (!(data[smallest].box < moving.box)) {
            break;
        }
        data[i] = data[smallest];
        i = smallest;
    }
    data[i] = moving;
}

void init_stream_heap(std::vector<PairStream>& streams, std::vector<HeapNode>& heap) {
    heap.clear();
    heap.reserve(streams.size());
    for (uint32_t si = 0; si < static_cast<uint32_t>(streams.size()); ++si) {
        streams[si].idx = 0;
        if (streams[si].right_len > 0) {
            const Box64 first = minkowski_diff_adj(streams[si].left_adj, streams[si].right_ptr[0]);
            heap.push_back(HeapNode{first, si});
        }
    }
    for (size_t i = heap.size() / 2; i > 0; --i) {
        heap_sift_down(heap, i - 1);
    }
}

inline bool pop_stream_heap(std::vector<PairStream>& streams,
                            std::vector<HeapNode>& heap,
                            Box64& out) noexcept {
    if (heap.empty()) return false;
    out = heap[0].box;
    const uint32_t si = heap[0].stream_idx;
    PairStream& st = streams[si];
    ++st.idx;
    if (st.idx < st.right_len) {
        heap[0].box = minkowski_diff_adj(st.left_adj, st.right_ptr[st.idx]);
        heap_sift_down(heap, 0);
    } else {
        if (heap.size() > 1) {
            heap[0] = heap.back();
            heap.pop_back();
            heap_sift_down(heap, 0);
        } else {
            heap.pop_back();
        }
    }
    return true;
}

}  // namespace

bool rectangle_union_cover(const std::vector<Box64>& boxes,
                           CheckpointPoller& poller,
                           std::vector<Box64>& out,
                           SweepWorkspace& ws) {
    out.clear();
    if (boxes.empty()) {
        return true;
    }

    // Fast path for a single box.
    if (boxes.size() == 1) {
        out.push_back(boxes[0]);
        return true;
    }

    ws.ys.clear();
    ws.ys.reserve(boxes.size() * 2);
    for (const Box64& b : boxes) {
        ws.ys.push_back(b.y0);
        ws.ys.push_back(b.y1);
    }
    std::sort(ws.ys.begin(), ws.ys.end());
    ws.ys.erase(std::unique(ws.ys.begin(), ws.ys.end()), ws.ys.end());

    const size_t ny = ws.ys.size();
    if (ny < 2) {
        return true;
    }
    if (poller.enabled()) {
        for (size_t i = 0; i < ny; i += 256) {
            if (!poller.call_now()) return false;
        }
    }
    const int32_t num_intervals = static_cast<int32_t>(ny - 1);

    ws.events.clear();
    ws.events.reserve(boxes.size() * 2);
    const int64_t* ys_begin = ws.ys.data();
    const int64_t* ys_end = ys_begin + ny;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!poller.poll_index(i)) return false;
        const Box64& b = boxes[i];
        const int32_t lo = static_cast<int32_t>(std::lower_bound(ys_begin, ys_end, b.y0) - ys_begin);
        const int32_t hi = static_cast<int32_t>(std::lower_bound(ys_begin, ys_end, b.y1) - ys_begin);
        ws.events.push_back(SweepEvent{b.x0, lo, hi, 1});
        ws.events.push_back(SweepEvent{b.x1, lo, hi, -1});
    }

    std::sort(ws.events.begin(), ws.events.end(),
              [](const SweepEvent& a, const SweepEvent& b) noexcept {
                  return a.x < b.x;
              });

    const size_t tree_size = static_cast<size_t>(num_intervals) * 4 + 4;
    ws.tree.assign(tree_size, SegNode{0, 0, 0});
    SegNode* tree_ptr = ws.tree.data();
    bool spans_dirty = true;

    auto seg_update = [tree_ptr, &spans_dirty](auto&& self, int32_t node, int32_t lo, int32_t hi,
                                               int32_t qlo, int32_t qhi, int32_t delta) noexcept -> void {
        SegNode& cur = tree_ptr[node];
        if (qlo <= lo && hi <= qhi) {
            cur.count += delta;
        } else {
            const int32_t mid = lo + ((hi - lo) >> 1);
            if (qlo < mid) {
                self(self, node << 1, lo, mid, qlo, qhi, delta);
            }
            if (qhi > mid) {
                self(self, (node << 1) | 1, mid, hi, qlo, qhi, delta);
            }
        }
        uint8_t new_occ = 0;
        uint8_t new_full = 0;
        if (cur.count > 0) {
            new_occ = 1;
            new_full = 1;
        } else if (hi - lo > 1) {
            const SegNode& lc = tree_ptr[node << 1];
            const SegNode& rc = tree_ptr[(node << 1) | 1];
            new_occ = static_cast<uint8_t>(lc.occupied | rc.occupied);
            new_full = static_cast<uint8_t>(lc.full & rc.full);
        }
        if (new_occ != cur.occupied || new_full != cur.full) {
            cur.occupied = new_occ;
            cur.full = new_full;
            spans_dirty = true;
        }
    };

    CheckpointPoller visit_poller(poller.hook, poller.mask);
    auto seg_intervals = [tree_ptr, ys_begin, &ws, &visit_poller](auto&& self, int32_t node,
                                                                  int32_t lo, int32_t hi) -> bool {
        if (visit_poller.enabled() && !visit_poller.poll_step()) return false;
        const SegNode& cur = tree_ptr[node];
        if (!cur.occupied) return true;
        if (cur.full) {
            const int64_t ylo = ys_begin[lo];
            const int64_t yhi = ys_begin[hi];
            if (!ws.spans.empty() && ws.spans.back().y1 == ylo) {
                ws.spans.back().y1 = yhi;
            } else {
                ws.spans.push_back(YSpan{ylo, yhi});
            }
            return true;
        }
        const int32_t mid = lo + ((hi - lo) >> 1);
        if (!self(self, node << 1, lo, mid)) return false;
        return self(self, (node << 1) | 1, mid, hi);
    };

    ws.open_boxes.clear();
    ws.current_boxes.clear();
    bool has_prev = false;
    int64_t prev_x = 0;
    CheckpointPoller event_poller(poller.hook, poller.mask);

    const size_t n_events = ws.events.size();
    size_t ei = 0;
    while (ei < n_events) {
        if (poller.enabled() && !poller.call_now()) return false;
        const int64_t x = ws.events[ei].x;
        if (has_prev && x > prev_x) {
            if (!spans_dirty) {
                // Coverage is identical to previous strip: extend all open boxes to x in O(S).
                for (OpenRect& ob : ws.open_boxes) {
                    ob.end_x = x;
                }
            } else {
                ws.spans.clear();
                if (!seg_intervals(seg_intervals, 1, 0, num_intervals)) {
                    return false;
                }
                spans_dirty = false;
                ws.current_boxes.clear();
                size_t oi = 0;
                size_t si = 0;
                const size_t no = ws.open_boxes.size();
                const size_t ns = ws.spans.size();
                while (oi < no && si < ns) {
                    if (poller.enabled() && !poller.poll_index(si)) return false;
                    const OpenRect& ob = ws.open_boxes[oi];
                    const YSpan& sp = ws.spans[si];
                    if (ob.y0 == sp.y0 && ob.y1 == sp.y1) {
                        ws.current_boxes.push_back(OpenRect{sp.y0, sp.y1, ob.start_x, x});
                        ++oi;
                        ++si;
                    } else if (ob.y0 < sp.y0 || (ob.y0 == sp.y0 && ob.y1 < sp.y1)) {
                        out.push_back(Box64{ob.start_x, ob.y0, ob.end_x, ob.y1});
                        ++oi;
                    } else {
                        ws.current_boxes.push_back(OpenRect{sp.y0, sp.y1, prev_x, x});
                        ++si;
                    }
                }
                while (oi < no) {
                    const OpenRect& ob = ws.open_boxes[oi++];
                    out.push_back(Box64{ob.start_x, ob.y0, ob.end_x, ob.y1});
                }
                while (si < ns) {
                    if (poller.enabled() && !poller.poll_index(si)) return false;
                    const YSpan& sp = ws.spans[si++];
                    ws.current_boxes.push_back(OpenRect{sp.y0, sp.y1, prev_x, x});
                }
                ws.open_boxes.swap(ws.current_boxes);
            }
        }

        while (ei < n_events && ws.events[ei].x == x) {
            if (event_poller.enabled() && !event_poller.poll_step()) return false;
            const SweepEvent& ev = ws.events[ei++];
            seg_update(seg_update, 1, 0, num_intervals, ev.lo, ev.hi, ev.delta);
        }
        prev_x = x;
        has_prev = true;
    }

    for (const OpenRect& ob : ws.open_boxes) {
        out.push_back(Box64{ob.start_x, ob.y0, ob.end_x, ob.y1});
    }
    std::sort(out.begin(), out.end());
    return true;
}

bool rectangle_union_cover(const std::vector<Box64>& boxes,
                           std::vector<Box64>& out,
                           core::CheckpointHook hook) {
    CheckpointPoller poller(hook);
    SweepWorkspace ws;
    return rectangle_union_cover(boxes, poller, out, ws);
}

ForbiddenCoverResult forbidden_box_cover(const std::vector<AtomRec>& left_atoms,
                                         const std::vector<AtomRec>& right_atoms,
                                         int64_t required_nm,
                                         CheckpointPoller& poller) {
    ForbiddenCoverResult result{};
    if (left_atoms.empty() || right_atoms.empty()) {
        if (!poller.call_now()) {
            result.status = ForbiddenStatus::Cancelled;
        }
        return result;
    }

    uint32_t max_right_layer = 0;
    for (const AtomRec& rec : right_atoms) {
        if (rec.layer_id > max_right_layer) {
            max_right_layer = rec.layer_id;
        }
    }

    // Group right atoms by layer_id (0 == None wildcard) and build all_right.
    std::vector<std::vector<Box64>> right_by_layer(static_cast<size_t>(max_right_layer) + 1);
    std::vector<Box64> all_right;
    all_right.reserve(right_atoms.size());
    for (const AtomRec& rec : right_atoms) {
        right_by_layer[rec.layer_id].push_back(rec.box);
        all_right.push_back(rec.box);
    }
    for (auto& grp : right_by_layer) {
        if (grp.size() > 1) {
            std::sort(grp.begin(), grp.end(), right_order_less);
        }
    }
    if (all_right.size() > 1) {
        std::sort(all_right.begin(), all_right.end(), right_order_less);
    }

    // Build sorted translated streams (O(L + R) memory).
    std::vector<PairStream> streams;
    streams.reserve(left_atoms.size() * 2);
    for (size_t i = 0; i < left_atoms.size(); ++i) {
        if (!poller.poll_index(i)) {
            result.status = ForbiddenStatus::Cancelled;
            return result;
        }
        const AtomRec& la = left_atoms[i];
        const Box64 ladj = make_left_adj(la.box, required_nm);
        if (la.layer_id == 0) {
            if (!all_right.empty()) {
                streams.push_back(PairStream{ladj, all_right.data(), all_right.size(), 0});
            }
        } else {
            if (!right_by_layer[0].empty()) {
                streams.push_back(PairStream{ladj, right_by_layer[0].data(),
                                             right_by_layer[0].size(), 0});
            }
            if (la.layer_id < right_by_layer.size() && !right_by_layer[la.layer_id].empty()) {
                streams.push_back(PairStream{ladj, right_by_layer[la.layer_id].data(),
                                             right_by_layer[la.layer_id].size(), 0});
            }
        }
    }

    if (streams.empty()) {
        if (!poller.call_now()) {
            result.status = ForbiddenStatus::Cancelled;
        }
        return result;
    }

    std::vector<HeapNode> heap;
    init_stream_heap(streams, heap);

    // Batched binary-tree union with 8-witness containment pruning.
    SweepWorkspace ws;
    std::vector<std::vector<Box64>> levels;
    std::vector<Box64> batch;
    batch.reserve(2048);
    std::vector<Box64> carry;
    std::vector<Box64> merged_in;
    std::array<Box64, 8> witnesses{};
    std::array<U128, 8> witness_areas{};
    size_t num_witnesses = 0;

    auto update_witnesses = [&](const std::vector<Box64>& run) {
        for (const Box64& b : run) {
            const U128 area = box_area_u128(b);
            bool dup = false;
            for (size_t k = 0; k < num_witnesses; ++k) {
                if (witnesses[k] == b) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            if (num_witnesses < 8) {
                witnesses[num_witnesses] = b;
                witness_areas[num_witnesses] = area;
                ++num_witnesses;
                size_t k = num_witnesses - 1;
                while (k > 0 && witness_areas[k] > witness_areas[k - 1]) {
                    std::swap(witnesses[k], witnesses[k - 1]);
                    std::swap(witness_areas[k], witness_areas[k - 1]);
                    --k;
                }
            } else if (area > witness_areas[7]) {
                witnesses[7] = b;
                witness_areas[7] = area;
                size_t k = 7;
                while (k > 0 && witness_areas[k] > witness_areas[k - 1]) {
                    std::swap(witnesses[k], witnesses[k - 1]);
                    std::swap(witness_areas[k], witness_areas[k - 1]);
                    --k;
                }
            }
        }
    };

    auto add_run = [&](const std::vector<Box64>& rows) -> bool {
        if (!rectangle_union_cover(rows, poller, carry, ws)) {
            return false;
        }
        size_t level = 0;
        while (level < levels.size() && !levels[level].empty()) {
            merged_in.clear();
            merged_in.reserve(levels[level].size() + carry.size());
            merged_in.insert(merged_in.end(), levels[level].begin(), levels[level].end());
            merged_in.insert(merged_in.end(), carry.begin(), carry.end());
            levels[level].clear();
            std::vector<Box64> next_carry;
            if (!rectangle_union_cover(merged_in, poller, next_carry, ws)) {
                return false;
            }
            carry.swap(next_carry);
            ++level;
        }
        if (level == levels.size()) {
            levels.push_back(std::move(carry));
            carry.clear();
        } else {
            levels[level].swap(carry);
            carry.clear();
        }
        update_witnesses(levels[level]);
        return true;
    };

    uint64_t raw_count = 0;
    uint64_t unique_count = 0;
    bool wide = false;
    bool has_prev = false;
    Box64 prev{};
    Box64 box{};

    while (pop_stream_heap(streams, heap, box)) {
        if (poller.enabled() && !poller.poll_index(static_cast<size_t>(raw_count))) {
            result.status = ForbiddenStatus::Cancelled;
            return result;
        }
        ++raw_count;
        if (has_prev && box == prev) {
            continue;
        }
        prev = box;
        has_prev = true;
        ++unique_count;

        if (box.x0 >= box.x1 || box.y0 >= box.y1) {
            result.status = ForbiddenStatus::NonPositiveBox;
            return result;
        }
        if (!wide && (box.x0 < kI32Min || box.x0 > kI32Max ||
                      box.y0 < kI32Min || box.y0 > kI32Max ||
                      box.x1 < kI32Min || box.x1 > kI32Max ||
                      box.y1 < kI32Min || box.y1 > kI32Max)) {
            wide = true;
        }
        if (wide) {
            continue;
        }

        // 8-witness exact containment pruning (each witness is an actual union rectangle).
        bool contained = false;
        for (size_t wi = 0; wi < num_witnesses; ++wi) {
            const Box64& w = witnesses[wi];
            if (w.x0 <= box.x0 && w.y0 <= box.y0 && box.x1 <= w.x1 && box.y1 <= w.y1) {
                contained = true;
                break;
            }
        }
        if (contained) {
            continue;
        }
        batch.push_back(box);
        if (batch.size() == 2048) {
            if (!add_run(batch)) {
                result.status = ForbiddenStatus::Cancelled;
                return result;
            }
            batch.clear();
        }
    }

    std::vector<Box64> exact;
    if (!wide) {
        if (!batch.empty()) {
            if (!add_run(batch)) {
                result.status = ForbiddenStatus::Cancelled;
                return result;
            }
            batch.clear();
        }
        for (auto& lvl : levels) {
            if (lvl.empty()) continue;
            if (exact.empty()) {
                exact.swap(lvl);
            } else {
                merged_in.clear();
                merged_in.reserve(exact.size() + lvl.size());
                merged_in.insert(merged_in.end(), exact.begin(), exact.end());
                merged_in.insert(merged_in.end(), lvl.begin(), lvl.end());
                lvl.clear();
                if (!rectangle_union_cover(merged_in, poller, exact, ws)) {
                    result.status = ForbiddenStatus::Cancelled;
                    return result;
                }
            }
        }
        if (unique_count > 0 && exact.empty()) {
            result.status = ForbiddenStatus::LostAllGeometry;
            return result;
        }
    }

    // Preserve wide-coordinate & non-shrinking fallback to canonical deduplicated boxes.
    if (wide || exact.size() >= unique_count) {
        exact.clear();
        exact.reserve(static_cast<size_t>(unique_count));
        init_stream_heap(streams, heap);
        has_prev = false;
        uint64_t idx = 0;
        while (pop_stream_heap(streams, heap, box)) {
            if (poller.enabled() && !poller.poll_index(static_cast<size_t>(idx++))) {
                result.status = ForbiddenStatus::Cancelled;
                return result;
            }
            if (has_prev && box == prev) continue;
            prev = box;
            has_prev = true;
            exact.push_back(box);
        }
    }

    if (!poller.call_now()) {
        result.status = ForbiddenStatus::Cancelled;
        return result;
    }

    result.status = ForbiddenStatus::Ok;
    result.cover = std::move(exact);
    result.raw_count = raw_count;
    result.unique_count = unique_count;
    return result;
}

ForbiddenCoverResult forbidden_box_cover(const std::vector<AtomRec>& left_atoms,
                                         const std::vector<AtomRec>& right_atoms,
                                         int64_t required_nm,
                                         core::CheckpointHook hook) {
    CheckpointPoller poller(hook);
    return forbidden_box_cover(left_atoms, right_atoms, required_nm, poller);
}

ExactUnionStatus exact_forbidden_union(std::vector<Box64> boxes,
                                       std::vector<Box64>& out) {
    std::sort(boxes.begin(), boxes.end());
    boxes.erase(std::unique(boxes.begin(), boxes.end()), boxes.end());

    if (boxes.size() <= 1) {
        out = std::move(boxes);
        return ExactUnionStatus::Ok;
    }

    // Preserve signed-32-bit coordinate and span compatibility contract.
    int64_t min_x = boxes[0].x0;
    int64_t max_x = boxes[0].x1;
    int64_t min_y = boxes[0].y0;
    int64_t max_y = boxes[0].y1;
    bool out_of_32bit = false;
    for (const Box64& b : boxes) {
        if (b.x0 < kI32Min || b.x0 > kI32Max || b.x1 < kI32Min || b.x1 > kI32Max ||
            b.y0 < kI32Min || b.y0 > kI32Max || b.y1 < kI32Min || b.y1 > kI32Max) {
            out_of_32bit = true;
            break;
        }
        if (b.x0 < min_x) min_x = b.x0;
        if (b.x1 > max_x) max_x = b.x1;
        if (b.y0 < min_y) min_y = b.y0;
        if (b.y1 > max_y) max_y = b.y1;
    }
    if (out_of_32bit || (max_x - min_x) > kI32Max || (max_y - min_y) > kI32Max) {
        out = std::move(boxes);
        return ExactUnionStatus::Ok;
    }

    CheckpointPoller no_poll;
    SweepWorkspace ws;
    std::vector<Box64> covered;
    rectangle_union_cover(boxes, no_poll, covered, ws);
    if (covered.empty()) {
        out.clear();
        return ExactUnionStatus::LostAllGeometry;
    }
    if (covered.size() < boxes.size()) {
        out = std::move(covered);
    } else {
        out = std::move(boxes);
    }
    return ExactUnionStatus::Ok;
}

}  // namespace clay_pnr::place
