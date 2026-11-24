#pragma once

// Pure C++20 exact Manhattan / orthogonal polygon boolean & morphology engine.
// Strictly free of <Python.h> and third-party dependencies.
//
// Canonical invariant:
//   `rects_` stores strictly disjoint, positive-area axis-aligned boxes (x0 < x1, y0 < y1),
//   vertically merged within each x-slab, horizontally merged across adjacent identical-y slabs,
//   and sorted lexicographically by (x0, y0, x1, y1).

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/status.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace clay_pnr::core {

enum class BoolOp : uint8_t {
    Or = 0,
    And = 1,
    Not = 2,
    Xor = 3,
};

struct Point64 {
    int64_t x = 0;
    int64_t y = 0;

    constexpr bool operator==(const Point64& o) const noexcept {
        return x == o.x && y == o.y;
    }
    constexpr bool operator!=(const Point64& o) const noexcept {
        return !(*this == o);
    }
    constexpr bool operator<(const Point64& o) const noexcept {
        return x != o.x ? x < o.x : y < o.y;
    }
};

// Closed orthogonal contour (vertices in cyclic order, no duplicate closing vertex).
// Outer boundaries are CCW (signed_area2 > 0); interior holes are CW (signed_area2 < 0).
struct OrthoContour {
    std::vector<Point64> points;
    bool is_hole = false;
};

class PolygonSet {
public:
    PolygonSet() = default;

    // Construct a canonical disjoint PolygonSet from an arbitrary (possibly overlapping) box list.
    static PolygonSet from_boxes(const std::vector<Box64>& boxes);
    static PolygonSet from_box(const Box64& box);

    // Construct directly from already-canonical disjoint rectangles (internal fast path).
    static PolygonSet from_canonical_unchecked(std::vector<Box64> canonical_rects);

    bool empty() const noexcept { return rects_.empty(); }
    size_t rect_count() const noexcept { return rects_.size(); }
    const std::vector<Box64>& rects() const noexcept { return rects_; }

    // Exact 128-bit area (overflow-free, exact since rects_ are strictly disjoint).
    U128 area_u128() const noexcept;

    // Exact bounding box (std::nullopt if empty).
    std::optional<Box64> bbox() const noexcept;

    // Canonical structural equality (unique canonical form -> vector equality is exact set equality).
    bool operator==(const PolygonSet& other) const noexcept { return rects_ == other.rects_; }
    bool operator!=(const PolygonSet& other) const noexcept { return rects_ != other.rects_; }

    // Exact sweep-line boolean operations.
    static PolygonSet boolean_op(const PolygonSet& a, const PolygonSet& b, BoolOp op);
    static PolygonSet op_or(const PolygonSet& a, const PolygonSet& b) {
        return boolean_op(a, b, BoolOp::Or);
    }
    static PolygonSet op_and(const PolygonSet& a, const PolygonSet& b) {
        return boolean_op(a, b, BoolOp::And);
    }
    static PolygonSet op_not(const PolygonSet& a, const PolygonSet& b) {
        return boolean_op(a, b, BoolOp::Not);
    }
    static PolygonSet op_xor(const PolygonSet& a, const PolygonSet& b) {
        return boolean_op(a, b, BoolOp::Xor);
    }

    // Exact L-infinity (square / Manhattan) morphological sizing:
    //   dx > 0, dy > 0: dilation (grow)
    //   dx < 0, dy < 0: erosion (shrink, exact complement dual around concavities & holes)
    //   mixed signs: axis-separated grow then shrink (matching Manhattan box morphology)
    PolygonSet sized(int64_t dx, int64_t dy) const;
    PolygonSet sized(int64_t d) const { return sized(d, d); }

    // Exact predicates:
    //   intersects(): strict positive-area overlap (equivalent to !op_and(*this, other).empty())
    //   touches():    closed-boundary contact including edges/corners (or overlap)
    bool intersects(const PolygonSet& other) const;
    bool touches(const PolygonSet& other) const;

    // Connected components over canonical disjoint boxes.
    //   include_corner_touch = true : closed-box touch (edges or corners) merges components (KLayout default).
    //   include_corner_touch = false: strict positive-length shared edge required to merge.
    std::vector<PolygonSet> connected_components(bool include_corner_touch = true) const;

    // Extract minimal collinear-simplified orthogonal boundary loops (CCW outers + CW holes).
    std::vector<OrthoContour> to_contours() const;

    // Signed doubled area of a closed contour (>0 CCW outer, <0 CW hole).
    static int64_t contour_signed_area2(const std::vector<Point64>& pts) noexcept;

private:
    std::vector<Box64> rects_;
};

}  // namespace clay_pnr::core

