// Pure C++20 exact Manhattan / orthogonal polygon boolean & morphology engine.
// Strictly free of <Python.h>.

#include "clay_pnr/core/polygon_set.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clay_pnr::core {

namespace {

// 4-bit state mask encoding which (in_a, in_b) states exist inside a y-interval subtree:
//   bit 0 (0b0001): (0, 0)
//   bit 1 (0b0010): (1, 0)
//   bit 2 (0b0100): (0, 1)
//   bit 3 (0b1000): (1, 1)
constexpr uint8_t kState00 = 0x1u;
constexpr uint8_t kState10 = 0x2u;
constexpr uint8_t kState01 = 0x4u;
constexpr uint8_t kState11 = 0x8u;

inline constexpr uint8_t apply_cover_mask(uint8_t m, bool cov_a, bool cov_b) noexcept {
    if (cov_a) {
        m = static_cast<uint8_t>(((m & (kState00 | kState01)) << 1) | (m & (kState10 | kState11)));
    }
    if (cov_b) {
        m = static_cast<uint8_t>(((m & (kState00 | kState10)) << 2) | (m & (kState01 | kState11)));
    }
    return m;
}

inline constexpr uint8_t target_mask_for(BoolOp op) noexcept {
    switch (op) {
        case BoolOp::Or:  return static_cast<uint8_t>(kState10 | kState01 | kState11);
        case BoolOp::And: return kState11;
        case BoolOp::Not: return kState10;
        case BoolOp::Xor: return static_cast<uint8_t>(kState10 | kState01);
    }
    return static_cast<uint8_t>(kState10 | kState01 | kState11);
}

struct BoolSegNode {
    int32_t cnt_a = 0;
    int32_t cnt_b = 0;
    uint8_t states = kState00;
};

struct BoolSweepEvent {
    int64_t x = 0;
    int32_t y_lo = 0;
    int32_t y_hi = 0;
    int8_t delta_a = 0;
    int8_t delta_b = 0;
};

struct ActiveStrip {
    int64_t y0 = 0;
    int64_t y1 = 0;
    int64_t x0 = 0;
};

struct YSpan {
    int64_t y0 = 0;
    int64_t y1 = 0;
};

void seg_init(std::vector<BoolSegNode>& tree, int32_t node, int32_t lo, int32_t hi) {
    tree[static_cast<size_t>(node)] = BoolSegNode{0, 0, kState00};
    if (hi - lo > 1) {
        const int32_t mid = lo + ((hi - lo) >> 1);
        seg_init(tree, node << 1, lo, mid);
        seg_init(tree, (node << 1) | 1, mid, hi);
    }
}

void seg_update(BoolSegNode* tree, int32_t node, int32_t lo, int32_t hi,
                int32_t qlo, int32_t qhi, int8_t da, int8_t db) noexcept {
    BoolSegNode& cur = tree[node];
    if (qlo <= lo && hi <= qhi) {
        cur.cnt_a += da;
        cur.cnt_b += db;
    } else {
        const int32_t mid = lo + ((hi - lo) >> 1);
        if (qlo < mid) {
            seg_update(tree, node << 1, lo, mid, qlo, qhi, da, db);
        }
        if (qhi > mid) {
            seg_update(tree, (node << 1) | 1, mid, hi, qlo, qhi, da, db);
        }
    }
    uint8_t child_mask = kState00;
    if (hi - lo > 1) {
        child_mask = static_cast<uint8_t>(tree[node << 1].states | tree[(node << 1) | 1].states);
    }
    cur.states = apply_cover_mask(child_mask, cur.cnt_a > 0, cur.cnt_b > 0);
}

void seg_collect_active(const BoolSegNode* tree, const int64_t* ys, int32_t node,
                        int32_t lo, int32_t hi, bool anc_a, bool anc_b,
                        uint8_t target_mask, std::vector<YSpan>& out) {
    const BoolSegNode& cur = tree[node];
    const uint8_t eff = apply_cover_mask(cur.states, anc_a, anc_b);
    if ((eff & target_mask) == 0u) {
        return;
    }
    if ((eff & static_cast<uint8_t>(~target_mask & 0x0Fu)) == 0u) {
        const int64_t y0 = ys[lo];
        const int64_t y1 = ys[hi];
        if (!out.empty() && out.back().y1 == y0) {
            out.back().y1 = y1;
        } else {
            out.push_back(YSpan{y0, y1});
        }
        return;
    }
    const bool next_a = anc_a || (cur.cnt_a > 0);
    const bool next_b = anc_b || (cur.cnt_b > 0);
    const int32_t mid = lo + ((hi - lo) >> 1);
    seg_collect_active(tree, ys, node << 1, lo, mid, next_a, next_b, target_mask, out);
    seg_collect_active(tree, ys, (node << 1) | 1, mid, hi, next_a, next_b, target_mask, out);
}

// General sweep-line boolean operation over two arbitrary rectangle lists.
std::vector<Box64> sweep_boolean_raw(const std::vector<Box64>& boxes_a,
                                     const std::vector<Box64>& boxes_b,
                                     BoolOp op) {
    std::vector<Box64> out;
    if (op == BoolOp::And && (boxes_a.empty() || boxes_b.empty())) {
        return out;
    }
    if (op == BoolOp::Not && boxes_a.empty()) {
        return out;
    }
    if ((op == BoolOp::Or || op == BoolOp::Xor) && boxes_a.empty() && boxes_b.empty()) {
        return out;
    }

    std::vector<int64_t> ys;
    ys.reserve((boxes_a.size() + boxes_b.size()) * 2);
    for (const auto& b : boxes_a) {
        if (b.x0 < b.x1 && b.y0 < b.y1) {
            ys.push_back(b.y0);
            ys.push_back(b.y1);
        }
    }
    for (const auto& b : boxes_b) {
        if (b.x0 < b.x1 && b.y0 < b.y1) {
            ys.push_back(b.y0);
            ys.push_back(b.y1);
        }
    }
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    if (ys.size() < 2) {
        return out;
    }

    const int32_t num_bands = static_cast<int32_t>(ys.size() - 1);
    const int64_t* ys_begin = ys.data();
    const int64_t* ys_end = ys_begin + ys.size();

    std::vector<BoolSweepEvent> events;
    events.reserve((boxes_a.size() + boxes_b.size()) * 2);
    for (const auto& b : boxes_a) {
        if (b.x0 >= b.x1 || b.y0 >= b.y1) continue;
        const int32_t lo = static_cast<int32_t>(std::lower_bound(ys_begin, ys_end, b.y0) - ys_begin);
        const int32_t hi = static_cast<int32_t>(std::lower_bound(ys_begin, ys_end, b.y1) - ys_begin);
        events.push_back(BoolSweepEvent{b.x0, lo, hi, 1, 0});
        events.push_back(BoolSweepEvent{b.x1, lo, hi, -1, 0});
    }
    for (const auto& b : boxes_b) {
        if (b.x0 >= b.x1 || b.y0 >= b.y1) continue;
        const int32_t lo = static_cast<int32_t>(std::lower_bound(ys_begin, ys_end, b.y0) - ys_begin);
        const int32_t hi = static_cast<int32_t>(std::lower_bound(ys_begin, ys_end, b.y1) - ys_begin);
        events.push_back(BoolSweepEvent{b.x0, lo, hi, 0, 1});
        events.push_back(BoolSweepEvent{b.x1, lo, hi, 0, -1});
    }
    if (events.empty()) {
        return out;
    }

    std::sort(events.begin(), events.end(),
              [](const BoolSweepEvent& lhs, const BoolSweepEvent& rhs) noexcept {
                  return lhs.x < rhs.x;
              });

    std::vector<BoolSegNode> tree(static_cast<size_t>(num_bands) * 4 + 4);
    seg_init(tree, 1, 0, num_bands);
    BoolSegNode* tree_ptr = tree.data();
    const uint8_t target_mask = target_mask_for(op);

    std::vector<ActiveStrip> active;
    std::vector<ActiveStrip> next_active;
    std::vector<YSpan> cur_spans;

    size_t idx = 0;
    const size_t n_ev = events.size();
    while (idx < n_ev) {
        const int64_t x_cur = events[idx].x;
        while (idx < n_ev && events[idx].x == x_cur) {
            const BoolSweepEvent& ev = events[idx];
            seg_update(tree_ptr, 1, 0, num_bands, ev.y_lo, ev.y_hi, ev.delta_a, ev.delta_b);
            ++idx;
        }
        cur_spans.clear();
        if (idx < n_ev) {
            seg_collect_active(tree_ptr, ys_begin, 1, 0, num_bands, false, false, target_mask, cur_spans);
        }

        // Coalesce horizontally across adjacent x-slabs when (y0, y1) matches identically.
        next_active.clear();
        size_t ai = 0;
        size_t si = 0;
        while (ai < active.size() && si < cur_spans.size()) {
            const ActiveStrip& a = active[ai];
            const YSpan& s = cur_spans[si];
            if (a.y0 == s.y0 && a.y1 == s.y1) {
                next_active.push_back(a);
                ++ai;
                ++si;
            } else if (a.y0 < s.y0 || (a.y0 == s.y0 && a.y1 < s.y1)) {
                if (x_cur > a.x0) {
                    out.push_back(Box64{a.x0, a.y0, x_cur, a.y1});
                }
                ++ai;
            } else {
                next_active.push_back(ActiveStrip{s.y0, s.y1, x_cur});
                ++si;
            }
        }
        while (ai < active.size()) {
            const ActiveStrip& a = active[ai++];
            if (x_cur > a.x0) {
                out.push_back(Box64{a.x0, a.y0, x_cur, a.y1});
            }
        }
        while (si < cur_spans.size()) {
            const YSpan& s = cur_spans[si++];
            next_active.push_back(ActiveStrip{s.y0, s.y1, x_cur});
        }
        active.swap(next_active);
    }

    std::sort(out.begin(), out.end());
    return out;
}

struct DSU {
    std::vector<uint32_t> parent;

    explicit DSU(size_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0u);
    }

    uint32_t find(uint32_t x) noexcept {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(uint32_t a, uint32_t b) noexcept {
        a = find(a);
        b = find(b);
        if (a != b) {
            if (a < b) parent[b] = a;
            else parent[a] = b;
        }
    }
};

struct DirectedSeg {
    int64_t a = 0;
    int64_t b = 0;
    int8_t dir = 1;  // +1: a -> b, -1: b -> a
};

// Cancel opposite overlapping intervals on the same scanline (fixed y or fixed x)
// and emit minimal oriented disjoint elementary segments.
struct OrientedEdge {
    Point64 from{};
    Point64 to{};
    uint8_t dir = 0;  // 0:Right(+x), 1:Up(+y), 2:Left(-x), 3:Down(-y)
};

void emit_cancelled_line(int64_t fixed_coord, bool is_horizontal,
                         const std::vector<DirectedSeg>& raw,
                         std::vector<OrientedEdge>& out) {
    if (raw.empty()) return;
    std::vector<int64_t> coords;
    coords.reserve(raw.size() * 2);
    for (const auto& s : raw) {
        coords.push_back(s.a);
        coords.push_back(s.b);
    }
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    if (coords.size() < 2) return;

    const size_t nb = coords.size() - 1;
    std::vector<int32_t> diff(nb + 1, 0);
    for (const auto& s : raw) {
        const size_t i0 = static_cast<size_t>(std::lower_bound(coords.begin(), coords.end(), s.a) - coords.begin());
        const size_t i1 = static_cast<size_t>(std::lower_bound(coords.begin(), coords.end(), s.b) - coords.begin());
        diff[i0] += s.dir;
        diff[i1] -= s.dir;
    }

    int32_t cur = 0;
    int32_t run_dir = 0;
    int64_t run_start = 0;
    for (size_t i = 0; i < nb; ++i) {
        cur += diff[i];
        const int32_t d = (cur > 0) ? 1 : ((cur < 0) ? -1 : 0);
        if (d != run_dir) {
            if (run_dir != 0) {
                const int64_t run_end = coords[i];
                if (is_horizontal) {
                    if (run_dir > 0) {
                        out.push_back(OrientedEdge{{run_start, fixed_coord}, {run_end, fixed_coord}, 0});
                    } else {
                        out.push_back(OrientedEdge{{run_end, fixed_coord}, {run_start, fixed_coord}, 2});
                    }
                } else {
                    if (run_dir > 0) {
                        out.push_back(OrientedEdge{{fixed_coord, run_start}, {fixed_coord, run_end}, 1});
                    } else {
                        out.push_back(OrientedEdge{{fixed_coord, run_end}, {fixed_coord, run_start}, 3});
                    }
                }
            }
            run_dir = d;
            run_start = coords[i];
        }
    }
    if (run_dir != 0) {
        const int64_t run_end = coords[nb];
        if (is_horizontal) {
            if (run_dir > 0) {
                out.push_back(OrientedEdge{{run_start, fixed_coord}, {run_end, fixed_coord}, 0});
            } else {
                out.push_back(OrientedEdge{{run_end, fixed_coord}, {run_start, fixed_coord}, 2});
            }
        } else {
            if (run_dir > 0) {
                out.push_back(OrientedEdge{{fixed_coord, run_start}, {fixed_coord, run_end}, 1});
            } else {
                out.push_back(OrientedEdge{{fixed_coord, run_end}, {fixed_coord, run_start}, 3});
            }
        }
    }
}

struct Point64Hash {
    size_t operator()(const Point64& p) const noexcept {
        uint64_t h = static_cast<uint64_t>(p.x) * 0x9e3779b97f4a7c15ULL;
        h ^= static_cast<uint64_t>(p.y) + 0xbf58476d1ce4e5b9ULL + (h << 6) + (h >> 2);
        return static_cast<size_t>(h);
    }
};

}  // namespace

PolygonSet PolygonSet::from_boxes(const std::vector<Box64>& boxes) {
    if (boxes.empty()) return PolygonSet{};
    if (boxes.size() == 1) {
        const Box64 b = normalize_box(boxes[0].x0, boxes[0].y0, boxes[0].x1, boxes[0].y1);
        return from_box(b);
    }
    std::vector<Box64> norm;
    norm.reserve(boxes.size());
    for (const auto& raw : boxes) {
        const Box64 b = normalize_box(raw.x0, raw.y0, raw.x1, raw.y1);
        if (b.x0 < b.x1 && b.y0 < b.y1) {
            norm.push_back(b);
        }
    }
    static const std::vector<Box64> kEmpty;
    return from_canonical_unchecked(sweep_boolean_raw(norm, kEmpty, BoolOp::Or));
}

PolygonSet PolygonSet::from_box(const Box64& box) {
    const Box64 b = normalize_box(box.x0, box.y0, box.x1, box.y1);
    if (b.x0 >= b.x1 || b.y0 >= b.y1) return PolygonSet{};
    PolygonSet ps;
    ps.rects_.push_back(b);
    return ps;
}

PolygonSet PolygonSet::from_canonical_unchecked(std::vector<Box64> canonical_rects) {
    PolygonSet ps;
    ps.rects_ = std::move(canonical_rects);
    return ps;
}

U128 PolygonSet::area_u128() const noexcept {
    U128 total{0, 0};
    for (const auto& b : rects_) {
        total = add_u128(total, box_area_u128(b));
    }
    return total;
}

std::optional<Box64> PolygonSet::bbox() const noexcept {
    if (rects_.empty()) return std::nullopt;
    Box64 bb = rects_[0];
    for (size_t i = 1; i < rects_.size(); ++i) {
        if (rects_[i].x0 < bb.x0) bb.x0 = rects_[i].x0;
        if (rects_[i].y0 < bb.y0) bb.y0 = rects_[i].y0;
        if (rects_[i].x1 > bb.x1) bb.x1 = rects_[i].x1;
        if (rects_[i].y1 > bb.y1) bb.y1 = rects_[i].y1;
    }
    return bb;
}

PolygonSet PolygonSet::boolean_op(const PolygonSet& a, const PolygonSet& b, BoolOp op) {
    if (op == BoolOp::Or) {
        if (a.empty()) return b;
        if (b.empty()) return a;
    } else if (op == BoolOp::And) {
        if (a.empty() || b.empty()) return PolygonSet{};
        const auto bba = a.bbox();
        const auto bbb = b.bbox();
        if (bba->x1 <= bbb->x0 || bbb->x1 <= bba->x0 || bba->y1 <= bbb->y0 || bbb->y1 <= bba->y0) {
            return PolygonSet{};
        }
    } else if (op == BoolOp::Not) {
        if (a.empty()) return PolygonSet{};
        if (b.empty()) return a;
        const auto bba = a.bbox();
        const auto bbb = b.bbox();
        if (bba->x1 <= bbb->x0 || bbb->x1 <= bba->x0 || bba->y1 <= bbb->y0 || bbb->y1 <= bba->y0) {
            return a;
        }
    } else if (op == BoolOp::Xor) {
        if (a.empty()) return b;
        if (b.empty()) return a;
    }
    return from_canonical_unchecked(sweep_boolean_raw(a.rects_, b.rects_, op));
}

PolygonSet PolygonSet::sized(int64_t dx, int64_t dy) const {
    if (rects_.empty() || (dx == 0 && dy == 0)) {
        return *this;
    }
    // Pure dilation (dx >= 0, dy >= 0): (U_i R_i) (+) K = U_i (R_i (+) K).
    if (dx >= 0 && dy >= 0) {
        std::vector<Box64> grown;
        grown.reserve(rects_.size());
        for (const auto& b : rects_) {
            grown.push_back(Box64{b.x0 - dx, b.y0 - dy, b.x1 + dx, b.y1 + dy});
        }
        return from_boxes(grown);
    }
    // Pure erosion (dx <= 0, dy <= 0): exact complement dual inside inflated frame.
    if (dx <= 0 && dy <= 0) {
        const int64_t sx = -dx;
        const int64_t sy = -dy;
        const auto bb = bbox();
        if (!bb.has_value()) return PolygonSet{};
        if ((bb->x1 - bb->x0) <= 2 * sx || (bb->y1 - bb->y0) <= 2 * sy) {
            return PolygonSet{};
        }
        const int64_t pad_x = sx + 1;
        const int64_t pad_y = sy + 1;
        const Box64 frame{bb->x0 - pad_x, bb->y0 - pad_y, bb->x1 + pad_x, bb->y1 + pad_y};
        const PolygonSet complement = op_not(from_box(frame), *this);
        const PolygonSet grown_comp = complement.sized(sx, sy);
        return op_not(*this, grown_comp);
    }
    // Mixed signs (one axis grows, the other shrinks): apply grow first, then shrink.
    if (dx > 0) {
        return sized(dx, 0).sized(0, dy);
    }
    return sized(0, dy).sized(dx, 0);
}

bool PolygonSet::intersects(const PolygonSet& other) const {
    if (rects_.empty() || other.rects_.empty()) return false;
    const auto bba = bbox();
    const auto bbb = other.bbox();
    if (bba->x1 <= bbb->x0 || bbb->x1 <= bba->x0 || bba->y1 <= bbb->y0 || bbb->y1 <= bba->y0) {
        return false;
    }
    // Fast two-pointer sweep along sorted x0.
    size_t j_start = 0;
    for (const auto& a : rects_) {
        while (j_start < other.rects_.size() && other.rects_[j_start].x1 <= a.x0) {
            ++j_start;
        }
        for (size_t j = j_start; j < other.rects_.size(); ++j) {
            const auto& b = other.rects_[j];
            if (b.x0 >= a.x1) break;
            if (b.x1 > a.x0 && b.y0 < a.y1 && a.y0 < b.y1) {
                return true;
            }
        }
    }
    return false;
}

bool PolygonSet::touches(const PolygonSet& other) const {
    if (rects_.empty() || other.rects_.empty()) return false;
    const auto bba = bbox();
    const auto bbb = other.bbox();
    if (!closed_rects_touch(*bba, *bbb)) {
        return false;
    }
    size_t j_start = 0;
    for (const auto& a : rects_) {
        while (j_start < other.rects_.size() && other.rects_[j_start].x1 < a.x0) {
            ++j_start;
        }
        for (size_t j = j_start; j < other.rects_.size(); ++j) {
            const auto& b = other.rects_[j];
            if (b.x0 > a.x1) break;
            if (closed_rects_touch(a, b)) {
                return true;
            }
        }
    }
    return false;
}

std::vector<PolygonSet> PolygonSet::connected_components(bool include_corner_touch) const {
    const size_t n = rects_.size();
    std::vector<PolygonSet> comps;
    if (n == 0) return comps;
    if (n == 1) {
        comps.push_back(*this);
        return comps;
    }

    // Sweep-line by x0 over active set ordered by x1 (sorted rects_ already ordered by x0).
    DSU dsu(n);
    std::vector<uint32_t> active;
    active.reserve(64);
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i) {
        const Box64& cur = rects_[i];
        // Prune boxes whose x1 < cur.x0 (cannot touch cur or any future box).
        size_t w = 0;
        for (size_t k = 0; k < active.size(); ++k) {
            const uint32_t j = active[k];
            const Box64& prev = rects_[j];
            if (prev.x1 < cur.x0) {
                continue;
            }
            active[w++] = j;
            if (include_corner_touch) {
                if (prev.y0 <= cur.y1 && cur.y0 <= prev.y1) {
                    dsu.unite(i, j);
                }
            } else {
                const bool share_v = (prev.x1 == cur.x0) &&
                                     (std::min(prev.y1, cur.y1) > std::max(prev.y0, cur.y0));
                const bool share_h = (prev.x1 > cur.x0) &&
                                     (prev.y1 == cur.y0 || cur.y1 == prev.y0);
                if (share_v || share_h) {
                    dsu.unite(i, j);
                }
            }
        }
        active.resize(w);
        active.push_back(i);
    }

    // Group canonical boxes by component root (deterministic order since rects_ is sorted).
    std::vector<int32_t> root_to_comp(n, -1);
    std::vector<std::vector<Box64>> groups;
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i) {
        const uint32_t r = dsu.find(i);
        if (root_to_comp[r] < 0) {
            root_to_comp[r] = static_cast<int32_t>(groups.size());
            groups.emplace_back();
        }
        groups[static_cast<size_t>(root_to_comp[r])].push_back(rects_[i]);
    }
    comps.reserve(groups.size());
    for (auto& g : groups) {
        comps.push_back(from_canonical_unchecked(std::move(g)));
    }
    return comps;
}

int64_t PolygonSet::contour_signed_area2(const std::vector<Point64>& pts) noexcept {
    const size_t n = pts.size();
    if (n < 3) return 0;
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const Point64& p = pts[i];
        const Point64& q = pts[(i + 1) % n];
        sum += p.x * q.y - q.x * p.y;
    }
    return sum;
}

std::vector<OrthoContour> PolygonSet::to_contours() const {
    std::vector<OrthoContour> result;
    if (rects_.empty()) return result;

    // Collect directed horizontal and vertical boundary segments per coordinate line.
    std::map<int64_t, std::vector<DirectedSeg>> horiz_lines;
    std::map<int64_t, std::vector<DirectedSeg>> vert_lines;
    for (const auto& b : rects_) {
        horiz_lines[b.y0].push_back(DirectedSeg{b.x0, b.x1, +1});  // bottom: +x
        horiz_lines[b.y1].push_back(DirectedSeg{b.x0, b.x1, -1});  // top:    -x
        vert_lines[b.x1].push_back(DirectedSeg{b.y0, b.y1, +1});   // right:  +y
        vert_lines[b.x0].push_back(DirectedSeg{b.y0, b.y1, -1});   // left:   -y
    }

    std::vector<OrientedEdge> edges;
    edges.reserve(rects_.size() * 4);
    for (const auto& [y, segs] : horiz_lines) {
        emit_cancelled_line(y, true, segs, edges);
    }
    for (const auto& [x, segs] : vert_lines) {
        emit_cancelled_line(x, false, segs, edges);
    }
    if (edges.empty()) return result;

    // Build outgoing adjacency map: Point64 -> list of edge indices.
    std::unordered_map<Point64, std::vector<size_t>, Point64Hash> outgoing;
    outgoing.reserve(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        outgoing[edges[i].from].push_back(i);
    }

    std::vector<uint8_t> used(edges.size(), 0);
    // Direction transition preference at 4-way kissing corners:
    // turning right (dir_out = (dir_in + 3) % 4) keeps interior on left without self-crossing.
    for (size_t start_idx = 0; start_idx < edges.size(); ++start_idx) {
        if (used[start_idx]) continue;
        std::vector<Point64> loop;
        size_t cur_edge = start_idx;
        const Point64 start_pt = edges[start_idx].from;
        while (true) {
            used[cur_edge] = 1;
            const OrientedEdge& e = edges[cur_edge];
            // Drop redundant collinear mid-vertex if previous edge had same direction.
            if (loop.size() >= 2) {
                const Point64& p0 = loop[loop.size() - 2];
                const Point64& p1 = loop[loop.size() - 1];
                if ((p0.x == p1.x && p1.x == e.from.x) || (p0.y == p1.y && p1.y == e.from.y)) {
                    loop.back() = e.from;
                } else {
                    loop.push_back(e.from);
                }
            } else {
                loop.push_back(e.from);
            }
            if (e.to == start_pt) {
                break;
            }
            const auto it = outgoing.find(e.to);
            if (it == outgoing.end()) break;
            const uint8_t pref_right = static_cast<uint8_t>((e.dir + 3u) & 3u);
            const uint8_t pref_left = static_cast<uint8_t>((e.dir + 1u) & 3u);
            size_t next_edge = edges.size();
            for (const uint8_t want_dir : {pref_right, pref_left, e.dir}) {
                for (size_t cand : it->second) {
                    if (!used[cand] && edges[cand].dir == want_dir) {
                        next_edge = cand;
                        break;
                    }
                }
                if (next_edge < edges.size()) break;
            }
            if (next_edge >= edges.size()) {
                for (size_t cand : it->second) {
                    if (!used[cand]) {
                        next_edge = cand;
                        break;
                    }
                }
            }
            if (next_edge >= edges.size()) break;
            cur_edge = next_edge;
        }
        // Final wrap-around collinear cleanup.
        while (loop.size() >= 3) {
            const Point64& prev = loop[loop.size() - 1];
            const Point64& curr = loop[0];
            const Point64& next = loop[1];
            if ((prev.x == curr.x && curr.x == next.x) || (prev.y == curr.y && curr.y == next.y)) {
                loop.erase(loop.begin());
            } else {
                break;
            }
        }
        if (loop.size() >= 4) {
            const int64_t area2 = contour_signed_area2(loop);
            result.push_back(OrthoContour{std::move(loop), area2 < 0});
        }
    }

    // Canonical deterministic ordering of contours.
    std::sort(result.begin(), result.end(),
              [](const OrthoContour& a, const OrthoContour& b) noexcept {
                  if (a.is_hole != b.is_hole) return !a.is_hole && b.is_hole;
                  return a.points < b.points;
              });
    return result;
}

}  // namespace clay_pnr::core

