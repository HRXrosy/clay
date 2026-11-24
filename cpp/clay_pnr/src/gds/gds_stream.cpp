// Pure C++20 zero-dependency binary GDSII Stream Format v6.0 reader & writer.
// Strictly free of <Python.h>.

#include "clay_pnr/gds/gds_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace clay_pnr::gds {

namespace {

// GDSII record types (high byte) and data types (low byte).
constexpr uint16_t kRecHeader   = 0x0002;
constexpr uint16_t kRecBgnLib   = 0x0102;
constexpr uint16_t kRecLibName  = 0x0206;
constexpr uint16_t kRecUnits    = 0x0305;
constexpr uint16_t kRecEndLib   = 0x0400;
constexpr uint16_t kRecBgnStr   = 0x0502;
constexpr uint16_t kRecStrName  = 0x0606;
constexpr uint16_t kRecEndStr   = 0x0700;
constexpr uint16_t kRecBoundary = 0x0800;
constexpr uint16_t kRecPath     = 0x0900;
constexpr uint16_t kRecSref     = 0x0A00;
constexpr uint16_t kRecAref     = 0x0B00;
constexpr uint16_t kRecText     = 0x0C00;
constexpr uint16_t kRecLayer    = 0x0D02;
constexpr uint16_t kRecDataType = 0x0E02;
constexpr uint16_t kRecWidth    = 0x0F03;
constexpr uint16_t kRecXy       = 0x1003;
constexpr uint16_t kRecEndEl    = 0x1100;
constexpr uint16_t kRecSName    = 0x1206;
constexpr uint16_t kRecColRow   = 0x1302;
constexpr uint16_t kRecTextType = 0x1602;
constexpr uint16_t kRecString   = 0x1906;
constexpr uint16_t kRecStrans   = 0x1A01;
constexpr uint16_t kRecMag      = 0x1B05;
constexpr uint16_t kRecAngle    = 0x1C05;
constexpr uint16_t kRecPathType = 0x2102;
constexpr uint16_t kRecBox      = 0x2D00;
constexpr uint16_t kRecBoxType  = 0x2E02;
constexpr uint16_t kRecBgnExtn  = 0x3003;
constexpr uint16_t kRecEndExtn  = 0x3103;

inline void push_u16_be(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    buf.push_back(static_cast<uint8_t>(v & 0xFFu));
}

inline void push_i16_be(std::vector<uint8_t>& buf, int16_t v) {
    push_u16_be(buf, static_cast<uint16_t>(v));
}

inline void push_i32_be(std::vector<uint8_t>& buf, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    buf.push_back(static_cast<uint8_t>((u >> 24) & 0xFFu));
    buf.push_back(static_cast<uint8_t>((u >> 16) & 0xFFu));
    buf.push_back(static_cast<uint8_t>((u >> 8) & 0xFFu));
    buf.push_back(static_cast<uint8_t>(u & 0xFFu));
}

inline uint16_t read_u16_be(const uint8_t* p) noexcept {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

inline int16_t read_i16_be(const uint8_t* p) noexcept {
    return static_cast<int16_t>(read_u16_be(p));
}

inline int32_t read_i32_be(const uint8_t* p) noexcept {
    const uint32_t u = (static_cast<uint32_t>(p[0]) << 24) |
                       (static_cast<uint32_t>(p[1]) << 16) |
                       (static_cast<uint32_t>(p[2]) << 8) |
                       static_cast<uint32_t>(p[3]);
    return static_cast<int32_t>(u);
}

inline void write_empty_rec(std::vector<uint8_t>& buf, uint16_t rec) {
    push_u16_be(buf, 4u);
    push_u16_be(buf, rec);
}

inline void write_i16_rec(std::vector<uint8_t>& buf, uint16_t rec, int16_t val) {
    push_u16_be(buf, 6u);
    push_u16_be(buf, rec);
    push_i16_be(buf, val);
}

inline void write_i32_rec(std::vector<uint8_t>& buf, uint16_t rec, int32_t val) {
    push_u16_be(buf, 8u);
    push_u16_be(buf, rec);
    push_i32_be(buf, val);
}

inline void write_u16_rec(std::vector<uint8_t>& buf, uint16_t rec, uint16_t val) {
    push_u16_be(buf, 6u);
    push_u16_be(buf, rec);
    push_u16_be(buf, val);
}

inline void write_real_rec(std::vector<uint8_t>& buf, uint16_t rec, double val) {
    push_u16_be(buf, 12u);
    push_u16_be(buf, rec);
    const auto r = double_to_gds_real(val);
    buf.insert(buf.end(), r.begin(), r.end());
}

inline void write_str_rec(std::vector<uint8_t>& buf, uint16_t rec, const std::string& s) {
    const size_t pad = (s.size() & 1u) ? 1u : 0u;
    const size_t total = 4u + s.size() + pad;
    push_u16_be(buf, static_cast<uint16_t>(total));
    push_u16_be(buf, rec);
    buf.insert(buf.end(), s.begin(), s.end());
    if (pad) buf.push_back(0u);
}

inline void write_timestamp_rec(std::vector<uint8_t>& buf, uint16_t rec) {
    // Fixed deterministic timestamp (2000-01-01 00:00:00 mod & access) for reproducible SHA-256.
    push_u16_be(buf, 28u);
    push_u16_be(buf, rec);
    constexpr int16_t kStamp[12] = {2000, 1, 1, 0, 0, 0, 2000, 1, 1, 0, 0, 0};
    for (int16_t v : kStamp) {
        push_i16_be(buf, v);
    }
}

inline void write_xy_rec(std::vector<uint8_t>& buf, const std::vector<GdsPoint>& pts, bool close_ring) {
    const bool need_close = close_ring && !pts.empty() && !(pts.front() == pts.back());
    const size_t count = pts.size() + (need_close ? 1u : 0u);
    const size_t total = 4u + count * 8u;
    push_u16_be(buf, static_cast<uint16_t>(total));
    push_u16_be(buf, kRecXy);
    for (const auto& p : pts) {
        push_i32_be(buf, p.x);
        push_i32_be(buf, p.y);
    }
    if (need_close) {
        push_i32_be(buf, pts.front().x);
        push_i32_be(buf, pts.front().y);
    }
}

std::string read_str_payload(const uint8_t* p, size_t len) {
    while (len > 0 && p[len - 1] == 0u) {
        --len;
    }
    return std::string(reinterpret_cast<const char*>(p), len);
}

struct Affine2D {
    // [a00 a01] [x]   [tx]
    // [a10 a11] [y] + [ty]
    double a00 = 1.0, a01 = 0.0;
    double a10 = 0.0, a11 = 1.0;
    double tx = 0.0, ty = 0.0;

    static Affine2D from_ref(int32_t x, int32_t y, uint16_t strans, double mag, double angle_deg) noexcept {
        Affine2D t;
        const bool mirror_x = (strans & 0x8000u) != 0u;
        const double sy = mirror_x ? -mag : mag;
        const double sx = mag;
        // Exact trig for right angles (0, 90, 180, 270) to preserve integer coordinates exactly.
        double norm_ang = std::fmod(angle_deg, 360.0);
        if (norm_ang < 0.0) norm_ang += 360.0;
        double c = 1.0;
        double s = 0.0;
        if (std::fabs(norm_ang) < 1e-9 || std::fabs(norm_ang - 360.0) < 1e-9) {
            c = 1.0; s = 0.0;
        } else if (std::fabs(norm_ang - 90.0) < 1e-9) {
            c = 0.0; s = 1.0;
        } else if (std::fabs(norm_ang - 180.0) < 1e-9) {
            c = -1.0; s = 0.0;
        } else if (std::fabs(norm_ang - 270.0) < 1e-9) {
            c = 0.0; s = -1.0;
        } else {
            const double rad = norm_ang * 3.14159265358979323846 / 180.0;
            c = std::cos(rad);
            s = std::sin(rad);
        }
        t.a00 = c * sx;
        t.a01 = -s * sy;
        t.a10 = s * sx;
        t.a11 = c * sy;
        t.tx = static_cast<double>(x);
        t.ty = static_cast<double>(y);
        return t;
    }

    Affine2D compose(const Affine2D& child) const noexcept {
        Affine2D r;
        r.a00 = a00 * child.a00 + a01 * child.a10;
        r.a01 = a00 * child.a01 + a01 * child.a11;
        r.a10 = a10 * child.a00 + a11 * child.a10;
        r.a11 = a10 * child.a01 + a11 * child.a11;
        r.tx = a00 * child.tx + a01 * child.ty + tx;
        r.ty = a10 * child.tx + a11 * child.ty + ty;
        return r;
    }

    Point64 apply(int64_t x, int64_t y) const noexcept {
        const double nx = a00 * static_cast<double>(x) + a01 * static_cast<double>(y) + tx;
        const double ny = a10 * static_cast<double>(x) + a11 * static_cast<double>(y) + ty;
        return Point64{core::py_round_i64(nx), core::py_round_i64(ny)};
    }
};

// Decompose a simple orthogonal polygon (or general boundary fallback to bbox) into exact boxes.
void emit_boundary_boxes(const std::vector<GdsPoint>& xy,
                         const Affine2D& tf,
                         std::vector<Box64>& out) {
    if (xy.size() < 4) return;
    std::vector<Point64> pts;
    pts.reserve(xy.size());
    for (const auto& p : xy) {
        pts.push_back(tf.apply(p.x, p.y));
    }
    if (pts.size() >= 2 && pts.front() == pts.back()) {
        pts.pop_back();
    }
    if (pts.size() < 3) return;

    // Fast path: 4-vertex axis-aligned rectangle (95%+ of foundry shapes).
    if (pts.size() == 4) {
        int64_t min_x = pts[0].x, max_x = pts[0].x;
        int64_t min_y = pts[0].y, max_y = pts[0].y;
        bool ortho = true;
        for (size_t i = 0; i < 4; ++i) {
            const auto& p = pts[i];
            const auto& q = pts[(i + 1) & 3u];
            if (p.x != q.x && p.y != q.y) {
                ortho = false;
                break;
            }
            if (p.x < min_x) min_x = p.x;
            if (p.x > max_x) max_x = p.x;
            if (p.y < min_y) min_y = p.y;
            if (p.y > max_y) max_y = p.y;
        }
        if (ortho && min_x < max_x && min_y < max_y) {
            out.push_back(Box64{min_x, min_y, max_x, max_y});
            return;
        }
    }

    // Check if strictly Manhattan polygon (all edges H or V): exact scanline trapezoid/rect split!
    bool all_ortho = true;
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        const auto& q = pts[(i + 1) % pts.size()];
        if (p.x != q.x && p.y != q.y) {
            all_ortho = false;
            break;
        }
    }
    if (all_ortho) {
        // Vertical edge scanline parity decomposition (exact for any simple orthogonal polygon).
        struct VEdge {
            int64_t x = 0;
            int64_t y0 = 0;
            int64_t y1 = 0;
        };
        std::vector<VEdge> vedges;
        std::vector<int64_t> ys;
        vedges.reserve(pts.size() / 2 + 1);
        ys.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            const auto& p = pts[i];
            const auto& q = pts[(i + 1) % pts.size()];
            if (p.x == q.x && p.y != q.y) {
                const int64_t y0 = std::min(p.y, q.y);
                const int64_t y1 = std::max(p.y, q.y);
                vedges.push_back(VEdge{p.x, y0, y1});
                ys.push_back(y0);
                ys.push_back(y1);
            }
        }
        std::sort(ys.begin(), ys.end());
        ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
        if (ys.size() >= 2) {
            std::vector<int64_t> xs;
            for (size_t yi = 0; yi + 1 < ys.size(); ++yi) {
                const int64_t y0 = ys[yi];
                const int64_t y1 = ys[yi + 1];
                xs.clear();
                for (const auto& e : vedges) {
                    if (e.y0 <= y0 && e.y1 >= y1) {
                        xs.push_back(e.x);
                    }
                }
                std::sort(xs.begin(), xs.end());
                for (size_t xi = 0; xi + 1 < xs.size(); xi += 2) {
                    if (xs[xi] < xs[xi + 1]) {
                        out.push_back(Box64{xs[xi], y0, xs[xi + 1], y1});
                    }
                }
            }
            return;
        }
    }

    // Non-Manhattan polygon fallback: conservative bounding box.
    int64_t min_x = pts[0].x, max_x = pts[0].x;
    int64_t min_y = pts[0].y, max_y = pts[0].y;
    for (const auto& p : pts) {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }
    if (min_x < max_x && min_y < max_y) {
        out.push_back(Box64{min_x, min_y, max_x, max_y});
    }
}

void emit_path_boxes(const GdsPath& path, const Affine2D& tf, std::vector<Box64>& out) {
    if (path.xy.size() < 2 || path.width <= 0) return;
    const int64_t half_w = static_cast<int64_t>(path.width) / 2;
    const int64_t bgn_ext = (path.pathtype == 2 || path.pathtype == 1)
                                ? half_w
                                : ((path.pathtype == 4) ? static_cast<int64_t>(path.bgnextn) : 0);
    const int64_t end_ext = (path.pathtype == 2 || path.pathtype == 1)
                                ? half_w
                                : ((path.pathtype == 4) ? static_cast<int64_t>(path.endextn) : 0);
    const size_t n_seg = path.xy.size() - 1;
    for (size_t i = 0; i < n_seg; ++i) {
        const Point64 p0 = tf.apply(path.xy[i].x, path.xy[i].y);
        const Point64 p1 = tf.apply(path.xy[i + 1].x, path.xy[i + 1].y);
        const int64_t ext0 = (i == 0) ? bgn_ext : half_w;
        const int64_t ext1 = (i + 1 == n_seg) ? end_ext : half_w;
        if (p0.x == p1.x) {
            const int64_t ylo = std::min(p0.y, p1.y) - (p0.y <= p1.y ? ext0 : ext1);
            const int64_t yhi = std::max(p0.y, p1.y) + (p0.y <= p1.y ? ext1 : ext0);
            if (half_w > 0 && ylo < yhi) {
                out.push_back(Box64{p0.x - half_w, ylo, p0.x + half_w, yhi});
            }
        } else if (p0.y == p1.y) {
            const int64_t xlo = std::min(p0.x, p1.x) - (p0.x <= p1.x ? ext0 : ext1);
            const int64_t xhi = std::max(p0.x, p1.x) + (p0.x <= p1.x ? ext1 : ext0);
            if (half_w > 0 && xlo < xhi) {
                out.push_back(Box64{xlo, p0.y - half_w, xhi, p0.y + half_w});
            }
        } else {
            const int64_t xlo = std::min(p0.x, p1.x) - half_w;
            const int64_t xhi = std::max(p0.x, p1.x) + half_w;
            const int64_t ylo = std::min(p0.y, p1.y) - half_w;
            const int64_t yhi = std::max(p0.y, p1.y) + half_w;
            if (xlo < xhi && ylo < yhi) {
                out.push_back(Box64{xlo, ylo, xhi, yhi});
            }
        }
    }
}

void flatten_cell_rec(const std::unordered_map<std::string, const GdsCell*>& cell_map,
                      const GdsCell* cell,
                      const Affine2D& tf,
                      int depth,
                      std::map<LayerKey, std::vector<Box64>>& out) {
    if (!cell || depth > 64) return;
    for (const auto& b : cell->boundaries) {
        emit_boundary_boxes(b.xy, tf, out[LayerKey{b.layer, b.datatype}]);
    }
    for (const auto& p : cell->paths) {
        emit_path_boxes(p, tf, out[LayerKey{p.layer, p.datatype}]);
    }
    for (const auto& sr : cell->srefs) {
        const auto it = cell_map.find(sr.cell_name);
        if (it == cell_map.end()) continue;
        const Affine2D child_tf = tf.compose(Affine2D::from_ref(sr.x, sr.y, sr.strans, sr.mag, sr.angle));
        flatten_cell_rec(cell_map, it->second, child_tf, depth + 1, out);
    }
    for (const auto& ar : cell->arefs) {
        const auto it = cell_map.find(ar.cell_name);
        if (it == cell_map.end() || ar.cols <= 0 || ar.rows <= 0) continue;
        const int64_t col_dx = (static_cast<int64_t>(ar.col_end.x) - static_cast<int64_t>(ar.origin.x)) / ar.cols;
        const int64_t col_dy = (static_cast<int64_t>(ar.col_end.y) - static_cast<int64_t>(ar.origin.y)) / ar.cols;
        const int64_t row_dx = (static_cast<int64_t>(ar.row_end.x) - static_cast<int64_t>(ar.origin.x)) / ar.rows;
        const int64_t row_dy = (static_cast<int64_t>(ar.row_end.y) - static_cast<int64_t>(ar.origin.y)) / ar.rows;
        for (int16_t r = 0; r < ar.rows; ++r) {
            for (int16_t c = 0; c < ar.cols; ++c) {
                const int32_t ox = static_cast<int32_t>(static_cast<int64_t>(ar.origin.x) + c * col_dx + r * row_dx);
                const int32_t oy = static_cast<int32_t>(static_cast<int64_t>(ar.origin.y) + c * col_dy + r * row_dy);
                const Affine2D child_tf = tf.compose(Affine2D::from_ref(ox, oy, ar.strans, ar.mag, ar.angle));
                flatten_cell_rec(cell_map, it->second, child_tf, depth + 1, out);
            }
        }
    }
}

}  // namespace

void GdsCell::add_box(int16_t layer, int16_t datatype, const Box64& box) {
    const Box64 b = core::normalize_box(box.x0, box.y0, box.x1, box.y1);
    if (b.x0 >= b.x1 || b.y0 >= b.y1) return;
    GdsBoundary bd;
    bd.layer = layer;
    bd.datatype = datatype;
    bd.xy = {
        GdsPoint{static_cast<int32_t>(b.x0), static_cast<int32_t>(b.y0)},
        GdsPoint{static_cast<int32_t>(b.x1), static_cast<int32_t>(b.y0)},
        GdsPoint{static_cast<int32_t>(b.x1), static_cast<int32_t>(b.y1)},
        GdsPoint{static_cast<int32_t>(b.x0), static_cast<int32_t>(b.y1)},
        GdsPoint{static_cast<int32_t>(b.x0), static_cast<int32_t>(b.y0)},
    };
    boundaries.push_back(std::move(bd));
}

GdsCell& GdsLibrary::add_cell(std::string cell_name) {
    cells.push_back(GdsCell{std::move(cell_name), {}, {}, {}, {}, {}});
    return cells.back();
}

const GdsCell* GdsLibrary::find_cell(const std::string& cell_name) const noexcept {
    for (const auto& c : cells) {
        if (c.name == cell_name) return &c;
    }
    return nullptr;
}

std::vector<std::string> GdsLibrary::top_cell_names() const {
    std::unordered_set<std::string> referenced;
    for (const auto& c : cells) {
        for (const auto& sr : c.srefs) referenced.insert(sr.cell_name);
        for (const auto& ar : c.arefs) referenced.insert(ar.cell_name);
    }
    std::vector<std::string> tops;
    for (const auto& c : cells) {
        if (referenced.find(c.name) == referenced.end()) {
            tops.push_back(c.name);
        }
    }
    if (tops.empty() && !cells.empty()) {
        tops.push_back(cells.front().name);
    }
    return tops;
}

std::array<uint8_t, 8> double_to_gds_real(double value) noexcept {
    std::array<uint8_t, 8> out{};
    if (value == 0.0 || !std::isfinite(value)) {
        return out;
    }
    uint8_t sign = 0u;
    if (value < 0.0) {
        sign = 0x80u;
        value = -value;
    }
    int exp2 = 0;
    const double frac = std::frexp(value, &exp2);  // value = frac * 2^exp2, 0.5 <= frac < 1.0
    int64_t exp16 = core::ceil_div_i64(static_cast<int64_t>(exp2), 4);
    const int shift = static_cast<int>(4 * exp16 - exp2);  // in [0, 3]
    const double m16 = std::ldexp(frac, -shift);           // in [1/16, 1.0)
    uint64_t mant = static_cast<uint64_t>(std::llround(std::ldexp(m16, 56)));
    if (mant >= (1ULL << 56)) {
        mant >>= 4;
        ++exp16;
    }
    const int64_t biased = exp16 + 64;
    if (biased <= 0) {
        return out;
    }
    const uint8_t exp_byte = static_cast<uint8_t>(std::min<int64_t>(biased, 127));
    out[0] = static_cast<uint8_t>(sign | exp_byte);
    for (int i = 0; i < 7; ++i) {
        out[static_cast<size_t>(7 - i)] = static_cast<uint8_t>((mant >> (i * 8)) & 0xFFu);
    }
    return out;
}

double gds_real_to_double(const uint8_t bytes[8]) noexcept {
    uint64_t mant = 0;
    for (int i = 1; i < 8; ++i) {
        mant = (mant << 8) | static_cast<uint64_t>(bytes[i]);
    }
    if (mant == 0) {
        return 0.0;
    }
    const bool neg = (bytes[0] & 0x80u) != 0u;
    const int exp16 = static_cast<int>(bytes[0] & 0x7Fu) - 64;
    const double val = std::ldexp(static_cast<double>(mant), 4 * exp16 - 56);
    return neg ? -val : val;
}

std::vector<uint8_t> write_gds_bytes(const GdsLibrary& lib) {
    std::vector<uint8_t> buf;
    buf.reserve(4096);

    write_i16_rec(buf, kRecHeader, 600);  // GDSII v6.0
    write_timestamp_rec(buf, kRecBgnLib);
    write_str_rec(buf, kRecLibName, lib.name.empty() ? "CLAY_LIB" : lib.name);

    // UNITS: two 8-byte GDSII reals (user_unit, dbu_meters)
    push_u16_be(buf, 20u);
    push_u16_be(buf, kRecUnits);
    const auto r_user = double_to_gds_real(lib.user_unit);
    const auto r_dbu = double_to_gds_real(lib.dbu_meters);
    buf.insert(buf.end(), r_user.begin(), r_user.end());
    buf.insert(buf.end(), r_dbu.begin(), r_dbu.end());

    for (const auto& cell : lib.cells) {
        write_timestamp_rec(buf, kRecBgnStr);
        write_str_rec(buf, kRecStrName, cell.name);

        for (const auto& bd : cell.boundaries) {
            if (bd.xy.size() < 3) continue;
            write_empty_rec(buf, kRecBoundary);
            write_i16_rec(buf, kRecLayer, bd.layer);
            write_i16_rec(buf, kRecDataType, bd.datatype);
            write_xy_rec(buf, bd.xy, true);
            write_empty_rec(buf, kRecEndEl);
        }

        for (const auto& pa : cell.paths) {
            if (pa.xy.size() < 2) continue;
            write_empty_rec(buf, kRecPath);
            write_i16_rec(buf, kRecLayer, pa.layer);
            write_i16_rec(buf, kRecDataType, pa.datatype);
            if (pa.pathtype != 0) write_i16_rec(buf, kRecPathType, pa.pathtype);
            write_i32_rec(buf, kRecWidth, pa.width);
            if (pa.pathtype == 4) {
                write_i32_rec(buf, kRecBgnExtn, pa.bgnextn);
                write_i32_rec(buf, kRecEndExtn, pa.endextn);
            }
            write_xy_rec(buf, pa.xy, false);
            write_empty_rec(buf, kRecEndEl);
        }

        for (const auto& tx : cell.texts) {
            write_empty_rec(buf, kRecText);
            write_i16_rec(buf, kRecLayer, tx.layer);
            write_i16_rec(buf, kRecTextType, tx.texttype);
            write_xy_rec(buf, {GdsPoint{tx.x, tx.y}}, false);
            write_str_rec(buf, kRecString, tx.text);
            write_empty_rec(buf, kRecEndEl);
        }

        for (const auto& sr : cell.srefs) {
            write_empty_rec(buf, kRecSref);
            write_str_rec(buf, kRecSName, sr.cell_name);
            if (sr.strans != 0 || sr.mag != 1.0 || sr.angle != 0.0) {
                write_u16_rec(buf, kRecStrans, sr.strans);
                if (sr.mag != 1.0) write_real_rec(buf, kRecMag, sr.mag);
                if (sr.angle != 0.0) write_real_rec(buf, kRecAngle, sr.angle);
            }
            write_xy_rec(buf, {GdsPoint{sr.x, sr.y}}, false);
            write_empty_rec(buf, kRecEndEl);
        }

        for (const auto& ar : cell.arefs) {
            write_empty_rec(buf, kRecAref);
            write_str_rec(buf, kRecSName, ar.cell_name);
            if (ar.strans != 0 || ar.mag != 1.0 || ar.angle != 0.0) {
                write_u16_rec(buf, kRecStrans, ar.strans);
                if (ar.mag != 1.0) write_real_rec(buf, kRecMag, ar.mag);
                if (ar.angle != 0.0) write_real_rec(buf, kRecAngle, ar.angle);
            }
            push_u16_be(buf, 8u);
            push_u16_be(buf, kRecColRow);
            push_i16_be(buf, ar.cols);
            push_i16_be(buf, ar.rows);
            write_xy_rec(buf, {ar.origin, ar.col_end, ar.row_end}, false);
            write_empty_rec(buf, kRecEndEl);
        }

        write_empty_rec(buf, kRecEndStr);
    }

    write_empty_rec(buf, kRecEndLib);
    return buf;
}

bool write_gds_file(const GdsLibrary& lib, const std::string& path, std::string* err_out) {
    const auto bytes = write_gds_bytes(lib);
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        if (err_out) *err_out = "failed to open output GDS file for writing: " + path;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!ofs.good()) {
        if (err_out) *err_out = "failed to write all GDS bytes to: " + path;
        return false;
    }
    return true;
}

bool read_gds_bytes(const uint8_t* data, size_t size, GdsLibrary& out_lib, std::string* err_out) {
    out_lib = GdsLibrary{};
    enum class ElemKind : uint8_t { None, Boundary, Box, Path, Text, Sref, Aref };

    GdsCell* cur_cell = nullptr;
    ElemKind elem = ElemKind::None;
    GdsBoundary cur_bd{};
    GdsPath cur_path{};
    GdsText cur_text{};
    GdsSref cur_sref{};
    GdsAref cur_aref{};

    size_t pos = 0;
    while (pos + 4 <= size) {
        const uint16_t rec_len = read_u16_be(data + pos);
        if (rec_len < 4u || pos + rec_len > size) {
            if (err_out) *err_out = "truncated or malformed GDSII record at byte offset " + std::to_string(pos);
            return false;
        }
        const uint16_t rec_type = read_u16_be(data + pos + 2);
        const uint8_t* payload = data + pos + 4;
        const size_t plen = static_cast<size_t>(rec_len - 4u);
        pos += rec_len;

        switch (rec_type) {
            case kRecLibName:
                out_lib.name = read_str_payload(payload, plen);
                break;
            case kRecUnits:
                if (plen >= 16u) {
                    out_lib.user_unit = gds_real_to_double(payload);
                    out_lib.dbu_meters = gds_real_to_double(payload + 8);
                }
                break;
            case kRecBgnStr:
                out_lib.cells.emplace_back();
                cur_cell = &out_lib.cells.back();
                elem = ElemKind::None;
                break;
            case kRecStrName:
                if (cur_cell) cur_cell->name = read_str_payload(payload, plen);
                break;
            case kRecEndStr:
                cur_cell = nullptr;
                elem = ElemKind::None;
                break;
            case kRecEndLib:
                return true;
            case kRecBoundary:
                elem = ElemKind::Boundary;
                cur_bd = GdsBoundary{};
                break;
            case kRecBox:
                elem = ElemKind::Box;
                cur_bd = GdsBoundary{};
                break;
            case kRecPath:
                elem = ElemKind::Path;
                cur_path = GdsPath{};
                break;
            case kRecText:
                elem = ElemKind::Text;
                cur_text = GdsText{};
                break;
            case kRecSref:
                elem = ElemKind::Sref;
                cur_sref = GdsSref{};
                break;
            case kRecAref:
                elem = ElemKind::Aref;
                cur_aref = GdsAref{};
                break;
            case kRecLayer:
                if (plen >= 2u) {
                    const int16_t lyr = read_i16_be(payload);
                    if (elem == ElemKind::Boundary || elem == ElemKind::Box) cur_bd.layer = lyr;
                    else if (elem == ElemKind::Path) cur_path.layer = lyr;
                    else if (elem == ElemKind::Text) cur_text.layer = lyr;
                }
                break;
            case kRecDataType:
            case kRecBoxType:
                if (plen >= 2u) {
                    const int16_t dt = read_i16_be(payload);
                    if (elem == ElemKind::Boundary || elem == ElemKind::Box) cur_bd.datatype = dt;
                    else if (elem == ElemKind::Path) cur_path.datatype = dt;
                }
                break;
            case kRecTextType:
                if (plen >= 2u && elem == ElemKind::Text) {
                    cur_text.texttype = read_i16_be(payload);
                }
                break;
            case kRecPathType:
                if (plen >= 2u && elem == ElemKind::Path) {
                    cur_path.pathtype = read_i16_be(payload);
                }
                break;
            case kRecWidth:
                if (plen >= 4u && elem == ElemKind::Path) {
                    cur_path.width = read_i32_be(payload);
                }
                break;
            case kRecBgnExtn:
                if (plen >= 4u && elem == ElemKind::Path) cur_path.bgnextn = read_i32_be(payload);
                break;
            case kRecEndExtn:
                if (plen >= 4u && elem == ElemKind::Path) cur_path.endextn = read_i32_be(payload);
                break;
            case kRecSName:
                if (elem == ElemKind::Sref) cur_sref.cell_name = read_str_payload(payload, plen);
                else if (elem == ElemKind::Aref) cur_aref.cell_name = read_str_payload(payload, plen);
                break;
            case kRecString:
                if (elem == ElemKind::Text) cur_text.text = read_str_payload(payload, plen);
                break;
            case kRecStrans:
                if (plen >= 2u) {
                    const uint16_t st = read_u16_be(payload);
                    if (elem == ElemKind::Sref) cur_sref.strans = st;
                    else if (elem == ElemKind::Aref) cur_aref.strans = st;
                }
                break;
            case kRecMag:
                if (plen >= 8u) {
                    const double m = gds_real_to_double(payload);
                    if (elem == ElemKind::Sref) cur_sref.mag = m;
                    else if (elem == ElemKind::Aref) cur_aref.mag = m;
                }
                break;
            case kRecAngle:
                if (plen >= 8u) {
                    const double a = gds_real_to_double(payload);
                    if (elem == ElemKind::Sref) cur_sref.angle = a;
                    else if (elem == ElemKind::Aref) cur_aref.angle = a;
                }
                break;
            case kRecColRow:
                if (plen >= 4u && elem == ElemKind::Aref) {
                    cur_aref.cols = read_i16_be(payload);
                    cur_aref.rows = read_i16_be(payload + 2);
                }
                break;
            case kRecXy: {
                const size_t npts = plen / 8u;
                if (elem == ElemKind::Boundary || elem == ElemKind::Box) {
                    cur_bd.xy.reserve(cur_bd.xy.size() + npts);
                    for (size_t i = 0; i < npts; ++i) {
                        cur_bd.xy.push_back(GdsPoint{read_i32_be(payload + i * 8u), read_i32_be(payload + i * 8u + 4u)});
                    }
                } else if (elem == ElemKind::Path) {
                    cur_path.xy.reserve(cur_path.xy.size() + npts);
                    for (size_t i = 0; i < npts; ++i) {
                        cur_path.xy.push_back(GdsPoint{read_i32_be(payload + i * 8u), read_i32_be(payload + i * 8u + 4u)});
                    }
                } else if (elem == ElemKind::Text && npts >= 1u) {
                    cur_text.x = read_i32_be(payload);
                    cur_text.y = read_i32_be(payload + 4u);
                } else if (elem == ElemKind::Sref && npts >= 1u) {
                    cur_sref.x = read_i32_be(payload);
                    cur_sref.y = read_i32_be(payload + 4u);
                } else if (elem == ElemKind::Aref && npts >= 3u) {
                    cur_aref.origin = GdsPoint{read_i32_be(payload), read_i32_be(payload + 4u)};
                    cur_aref.col_end = GdsPoint{read_i32_be(payload + 8u), read_i32_be(payload + 12u)};
                    cur_aref.row_end = GdsPoint{read_i32_be(payload + 16u), read_i32_be(payload + 20u)};
                }
                break;
            }
            case kRecEndEl:
                if (cur_cell) {
                    if (elem == ElemKind::Boundary || elem == ElemKind::Box) {
                        cur_cell->boundaries.push_back(std::move(cur_bd));
                    } else if (elem == ElemKind::Path) {
                        cur_cell->paths.push_back(std::move(cur_path));
                    } else if (elem == ElemKind::Text) {
                        cur_cell->texts.push_back(std::move(cur_text));
                    } else if (elem == ElemKind::Sref) {
                        cur_cell->srefs.push_back(std::move(cur_sref));
                    } else if (elem == ElemKind::Aref) {
                        cur_cell->arefs.push_back(std::move(cur_aref));
                    }
                }
                elem = ElemKind::None;
                break;
            default:
                break;
        }
    }
    return true;
}

bool read_gds_file(const std::string& path, GdsLibrary& out_lib, std::string* err_out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        if (err_out) *err_out = "failed to open GDS file: " + path;
        return false;
    }
    const std::streamsize sz = ifs.tellg();
    if (sz < 0) {
        if (err_out) *err_out = "failed to stat GDS file: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (sz > 0 && !ifs.read(reinterpret_cast<char*>(buf.data()), sz)) {
        if (err_out) *err_out = "failed to read GDS file: " + path;
        return false;
    }
    return read_gds_bytes(buf.data(), buf.size(), out_lib, err_out);
}

std::map<LayerKey, std::vector<Box64>> flatten_boxes_by_layer(
    const GdsLibrary& lib,
    const std::string& top_cell) {
    std::map<LayerKey, std::vector<Box64>> out;
    if (lib.cells.empty()) return out;

    std::unordered_map<std::string, const GdsCell*> cell_map;
    cell_map.reserve(lib.cells.size());
    for (const auto& c : lib.cells) {
        cell_map.emplace(c.name, &c);
    }

    const GdsCell* root = nullptr;
    if (!top_cell.empty()) {
        root = lib.find_cell(top_cell);
    } else {
        const auto tops = lib.top_cell_names();
        if (!tops.empty()) root = lib.find_cell(tops.front());
    }
    if (!root) return out;

    flatten_cell_rec(cell_map, root, Affine2D{}, 0, out);
    return out;
}

std::vector<FlattenedLayerSummary> summarize_gds_layers(
    const GdsLibrary& lib,
    const std::string& top_cell) {
    const auto by_layer = flatten_boxes_by_layer(lib, top_cell);
    std::vector<FlattenedLayerSummary> out;
    out.reserve(by_layer.size());
    for (const auto& [lk, raw_boxes] : by_layer) {
        const PolygonSet ps = PolygonSet::from_boxes(raw_boxes);
        const auto bb = ps.bbox();
        if (!bb.has_value()) continue;
        FlattenedLayerSummary sum;
        sum.layer = lk.layer;
        sum.datatype = lk.datatype;
        sum.raw_polygon_count = raw_boxes.size();
        sum.canonical_rect_count = ps.rect_count();
        sum.union_area_dbu2 = ps.area_u128();
        sum.bbox = *bb;
        out.push_back(sum);
    }
    return out;
}

}  // namespace clay_pnr::gds

