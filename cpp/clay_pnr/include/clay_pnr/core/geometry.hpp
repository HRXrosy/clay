#pragma once

// Pure C++20 exact integer geometry primitives for Clay PnR.
// Strictly free of <Python.h> so core algorithms build and test natively.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace clay_pnr::core {

constexpr int64_t kI32Min = -(1LL << 31);
constexpr int64_t kI32Max = (1LL << 31) - 1;
constexpr int64_t kCoordLimit60 = (1LL << 60);
constexpr int64_t kCoordLimit61 = (1LL << 61);

struct Box64 {
    int64_t x0 = 0;
    int64_t y0 = 0;
    int64_t x1 = 0;
    int64_t y1 = 0;

    constexpr bool operator==(const Box64& o) const noexcept {
        return x0 == o.x0 && y0 == o.y0 && x1 == o.x1 && y1 == o.y1;
    }
    constexpr bool operator!=(const Box64& o) const noexcept {
        return !(*this == o);
    }
    constexpr bool operator<(const Box64& o) const noexcept {
        if (x0 != o.x0) return x0 < o.x0;
        if (y0 != o.y0) return y0 < o.y0;
        if (x1 != o.x1) return x1 < o.x1;
        return y1 < o.y1;
    }
    constexpr int64_t width() const noexcept { return x1 - x0; }
    constexpr int64_t height() const noexcept { return y1 - y0; }
};

// Exact 128-bit unsigned integer for overflow-free area & squared-distance arithmetic.
struct U128 {
    uint64_t hi = 0;
    uint64_t lo = 0;

    constexpr bool operator==(const U128& o) const noexcept {
        return hi == o.hi && lo == o.lo;
    }
    constexpr bool operator!=(const U128& o) const noexcept {
        return !(*this == o);
    }
    constexpr bool operator<(const U128& o) const noexcept {
        return hi != o.hi ? hi < o.hi : lo < o.lo;
    }
    constexpr bool operator>(const U128& o) const noexcept {
        return hi != o.hi ? hi > o.hi : lo > o.lo;
    }
    constexpr bool operator<=(const U128& o) const noexcept {
        return !(o < *this);
    }
    constexpr bool operator>=(const U128& o) const noexcept {
        return !(*this < o);
    }
};

inline constexpr U128 add_u128(U128 a, U128 b) noexcept {
    const uint64_t lo = a.lo + b.lo;
    const uint64_t carry = (lo < a.lo) ? 1ULL : 0ULL;
    const uint64_t hi = a.hi + b.hi + carry;
    return U128{hi, lo};
}

inline constexpr U128 mul_u64(uint64_t a, uint64_t b) noexcept {
    const uint64_t a_lo = a & 0xFFFFFFFFULL;
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = b & 0xFFFFFFFFULL;
    const uint64_t b_hi = b >> 32;
    const uint64_t p0 = a_lo * b_lo;
    const uint64_t p1 = a_lo * b_hi;
    const uint64_t p2 = a_hi * b_lo;
    const uint64_t p3 = a_hi * b_hi;
    const uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    const uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    const uint64_t lo = (p0 & 0xFFFFFFFFULL) | (mid << 32);
    return U128{hi, lo};
}

inline constexpr U128 box_area_u128(const Box64& b) noexcept {
    // Valid boxes satisfy x0 <= x1 and y0 <= y1; unsigned subtraction never overflows.
    const uint64_t w = static_cast<uint64_t>(b.x1) - static_cast<uint64_t>(b.x0);
    const uint64_t h = static_cast<uint64_t>(b.y1) - static_cast<uint64_t>(b.y0);
    return mul_u64(w, h);
}

inline constexpr Box64 normalize_box(int64_t x0, int64_t y0, int64_t x1, int64_t y1) noexcept {
    return Box64{
        x0 <= x1 ? x0 : x1,
        y0 <= y1 ? y0 : y1,
        x0 <= x1 ? x1 : x0,
        y0 <= y1 ? y1 : y0,
    };
}

inline constexpr bool closed_rects_touch(const Box64& a, const Box64& b) noexcept {
    return !(a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0);
}

// Exact L-infinity separation between two axis-aligned boxes (negative if overlapping).
inline constexpr int64_t box_clearance_linf(const Box64& a, const Box64& b) noexcept {
    const int64_t d0 = b.x0 - a.x1;
    const int64_t d1 = a.x0 - b.x1;
    const int64_t d2 = b.y0 - a.y1;
    const int64_t d3 = a.y0 - b.y1;
    const int64_t dx = d0 > d1 ? d0 : d1;
    const int64_t dy = d2 > d3 ? d2 : d3;
    return dx > dy ? dx : dy;
}

// Non-negative axis-wise gap between two closed boxes (0 when overlapping or touching).
inline constexpr void box_sep(const Box64& a, const Box64& b, int64_t& sx, int64_t& sy) noexcept {
    if (a.x1 < b.x0) {
        sx = b.x0 - a.x1;
    } else if (b.x1 < a.x0) {
        sx = a.x0 - b.x1;
    } else {
        sx = 0;
    }

    if (a.y1 < b.y0) {
        sy = b.y0 - a.y1;
    } else if (b.y1 < a.y0) {
        sy = a.y0 - b.y1;
    } else {
        sy = 0;
    }
}

// Exact match for gridless_types._rects_clear(a, b, min_gap):
// L-infinity separation >= min_gap (when min_gap <= 0, always true).
inline constexpr bool rects_clear_linf(const Box64& a, const Box64& b, int64_t min_gap) noexcept {
    if (min_gap <= 0) return true;
    int64_t sx = 0;
    if (a.x1 < b.x0) sx = b.x0 - a.x1;
    else if (b.x1 < a.x0) sx = a.x0 - b.x1;
    if (sx >= min_gap) return true;
    int64_t sy = 0;
    if (a.y1 < b.y0) sy = b.y0 - a.y1;
    else if (b.y1 < a.y0) sy = a.y0 - b.y1;
    return sy >= min_gap;
}

// Exact 128-bit check: dx*dx + dy*dy < sp*sp (given 0 <= dx < sp and 0 <= dy < sp).
inline bool euclid_sq_lt_128(uint64_t dx, uint64_t dy, uint64_t sp) noexcept {
    if (sp < (1ULL << 31)) {
        return (dx * dx + dy * dy) < (sp * sp);
    }
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t dx_hi = 0, dy_hi = 0, sp_hi = 0;
    const uint64_t dx_lo = _umul128(dx, dx, &dx_hi);
    const uint64_t dy_lo = _umul128(dy, dy, &dy_hi);
    const uint64_t sp_lo = _umul128(sp, sp, &sp_hi);
    const uint64_t sum_lo = dx_lo + dy_lo;
    const uint64_t carry = (sum_lo < dx_lo) ? 1ULL : 0ULL;
    const uint64_t sum_hi = dx_hi + dy_hi + carry;
    if (sum_hi != sp_hi) return sum_hi < sp_hi;
    return sum_lo < sp_lo;
#elif defined(__SIZEOF_INT128__)
    const unsigned __int128 lhs = static_cast<unsigned __int128>(dx) * dx +
                                  static_cast<unsigned __int128>(dy) * dy;
    const unsigned __int128 rhs = static_cast<unsigned __int128>(sp) * sp;
    return lhs < rhs;
#else
    const U128 lhs = add_u128(mul_u64(dx, dx), mul_u64(dy, dy));
    const U128 rhs = mul_u64(sp, sp);
    return lhs < rhs;
#endif
}

// Exact integer Euclidean clearance check:
// Returns true iff Euclidean distance between closed boxes a and b is >= space_nm
// (and non-overlapping when space_nm <= 0). Exact 128-bit integer arithmetic at all scales.
inline bool rects_clear_euclid_exact(const Box64& a, const Box64& b, int64_t space_nm) noexcept {
    if (space_nm <= 0) {
        return a.x1 <= b.x0 || b.x1 <= a.x0 || a.y1 <= b.y0 || b.y1 <= a.y0;
    }
    int64_t sx = 0, sy = 0;
    box_sep(a, b, sx, sy);
    if (sx >= space_nm || sy >= space_nm) return true;
    if (sx == 0 && sy == 0) return false;
    return !euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                             static_cast<uint64_t>(space_nm));
}

inline bool rects_clear_euclidean(const Box64& a, const Box64& b, int64_t space_nm) noexcept {
    return rects_clear_euclid_exact(a, b, space_nm);
}

inline bool euclid_in_open_band(const Box64& a, const Box64& b, int64_t sp,
                                int64_t& sx, int64_t& sy) noexcept {
    box_sep(a, b, sx, sy);
    if ((sx == 0 && sy == 0) || sp <= 0) return false;
    if (sx >= sp || sy >= sp) return false;
    return euclid_sq_lt_128(static_cast<uint64_t>(sx), static_cast<uint64_t>(sy),
                            static_cast<uint64_t>(sp));
}

inline bool owned_landing_has_min_throat(const Box64& atom, const Box64& pad,
                                         int64_t min_width) noexcept {
    if ((atom.x0 <= pad.x0 && atom.y0 <= pad.y0 && atom.x1 >= pad.x1 && atom.y1 >= pad.y1) ||
        (pad.x0 <= atom.x0 && pad.y0 <= atom.y0 && pad.x1 >= atom.x1 && pad.y1 >= atom.y1)) {
        return true;
    }
    const int64_t ox = std::min(atom.x1, pad.x1) - std::max(atom.x0, pad.x0);
    const int64_t oy = std::min(atom.y1, pad.y1) - std::max(atom.y0, pad.y0);
    if (ox < 0 || oy < 0) return false;
    if (min_width <= 0) return true;
    if (ox >= min_width || oy >= min_width) return true;
    return !euclid_sq_lt_128(static_cast<uint64_t>(ox), static_cast<uint64_t>(oy),
                             static_cast<uint64_t>(min_width));
}

inline constexpr Box64 compute_gap_box(const Box64& a, const Box64& b,
                                       int64_t sx, int64_t sy) noexcept {
    const int64_t gx0 = (sx == 0) ? (a.x0 > b.x0 ? a.x0 : b.x0) : (a.x1 < b.x1 ? a.x1 : b.x1);
    const int64_t gx1 = (sx == 0) ? (a.x1 < b.x1 ? a.x1 : b.x1) : (a.x0 > b.x0 ? a.x0 : b.x0);
    const int64_t gy0 = (sy == 0) ? (a.y0 > b.y0 ? a.y0 : b.y0) : (a.y1 < b.y1 ? a.y1 : b.y1);
    const int64_t gy1 = (sy == 0) ? (a.y1 < b.y1 ? a.y1 : b.y1) : (a.y0 > b.y0 ? a.y0 : b.y0);
    return Box64{gx0, gy0, gx1, gy1};
}

// Exact rectangle subtraction: returns true iff `gap` is completely covered by `cover`.
inline bool box_fully_covered_by_rects(const Box64& gap, const std::vector<Box64>& cover) {
    if (gap.x1 <= gap.x0 || gap.y1 <= gap.y0) return false;
    for (const auto& rr : cover) {
        if (rr.x0 <= gap.x0 && rr.y0 <= gap.y0 && rr.x1 >= gap.x1 && rr.y1 >= gap.y1) {
            return true;
        }
    }
    std::vector<Box64> pieces;
    pieces.reserve(16);
    pieces.push_back(gap);
    std::vector<Box64> next;
    next.reserve(16);
    for (const auto& c : cover) {
        next.clear();
        for (const auto& p : pieces) {
            if (c.x1 <= p.x0 || p.x1 <= c.x0 || c.y1 <= p.y0 || p.y1 <= c.y0) {
                next.push_back(p);
                continue;
            }
            const int64_t ix0 = std::max(p.x0, c.x0);
            const int64_t ix1 = std::min(p.x1, c.x1);
            const int64_t iy0 = std::max(p.y0, c.y0);
            const int64_t iy1 = std::min(p.y1, c.y1);
            if (p.x0 < ix0) next.push_back(Box64{p.x0, p.y0, ix0, p.y1});
            if (ix1 < p.x1) next.push_back(Box64{ix1, p.y0, p.x1, p.y1});
            if (p.y0 < iy0) next.push_back(Box64{ix0, p.y0, ix1, iy0});
            if (iy1 < p.y1) next.push_back(Box64{ix0, iy1, ix1, p.y1});
        }
        pieces.swap(next);
        if (pieces.empty()) return true;
    }
    return pieces.empty();
}

// Python-matching arithmetic helpers (floor division for arbitrary signs & banker's rounding).
inline constexpr int64_t floor_div_i64(int64_t a, int64_t b) noexcept {
    int64_t q = a / b;
    const int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        --q;
    }
    return q;
}

inline constexpr int64_t ceil_div_i64(int64_t a, int64_t b) noexcept {
    return -floor_div_i64(-a, b);
}

// Fast floor division when divisor c > 0.
inline constexpr int64_t floor_div_pos(int64_t a, int64_t c) noexcept {
    const int64_t q = a / c;
    const int64_t r = a % c;
    return (r < 0) ? (q - 1) : q;
}

// IEEE-754 FE_TONEAREST matches Python's round() (round-half-to-even).
inline int64_t py_round_i64(double v) noexcept {
    return static_cast<int64_t>(std::nearbyint(v));
}

// Exact IEEE-754 parity fast path for Clay's rigid placed coordinate quantization:
//   composed_nm = (local_um + placement_um) * 1000.0
//   exact half-nm ties round UP (translation-invariant).
// Returns true iff the value falls outside the tiny tie-dust window so no decimal
// round(..., 9) fallback is required.
inline bool placed_coord_nm_try_fast(double local_um, double placement_um, int64_t& out) noexcept {
    const double c = (local_um + placement_um) * 1000.0;
    if (c > -1e6 && c < 1e6) {
        const double m = std::floor(c);
        const double eps = c - (m + 0.5);
        if (eps > -3e-10) {
            out = static_cast<int64_t>(m) + 1;
            return true;
        }
        if (eps < -7e-10) {
            out = static_cast<int64_t>(m);
            return true;
        }
        return false;
    }
    if (c > -1e8 && c < 1e8) {
        const double m = std::floor(c);
        const double eps = c - (m + 0.5);
        if (eps >= 0.0) {
            out = static_cast<int64_t>(m) + 1;
            return true;
        }
        if (eps < -1e-7) {
            out = static_cast<int64_t>(m);
            return true;
        }
        return false;
    }
    return false;
}

}  // namespace clay_pnr::core

namespace clay_pnr {

using core::Box64;
using core::U128;
using core::add_u128;
using core::box_area_u128;
using core::box_clearance_linf;
using core::box_fully_covered_by_rects;
using core::box_sep;
using core::ceil_div_i64;
using core::closed_rects_touch;
using core::compute_gap_box;
using core::euclid_in_open_band;
using core::euclid_sq_lt_128;
using core::floor_div_i64;
using core::floor_div_pos;
using core::mul_u64;
using core::normalize_box;
using core::owned_landing_has_min_throat;
using core::placed_coord_nm_try_fast;
using core::py_round_i64;
using core::rects_clear_euclid_exact;
using core::rects_clear_euclidean;
using core::rects_clear_linf;

}  // namespace clay_pnr

