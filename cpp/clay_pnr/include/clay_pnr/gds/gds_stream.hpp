#pragma once

// Pure C++20 zero-dependency binary GDSII Stream Format v6.0 reader & writer.
// Strictly free of <Python.h> and third-party libraries.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/polygon_set.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace clay_pnr::gds {

using core::Box64;
using core::Point64;
using core::PolygonSet;

struct GdsPoint {
    int32_t x = 0;
    int32_t y = 0;

    constexpr bool operator==(const GdsPoint& o) const noexcept {
        return x == o.x && y == o.y;
    }
};

struct GdsBoundary {
    int16_t layer = 0;
    int16_t datatype = 0;
    std::vector<GdsPoint> xy;  // closed polygon vertices (writer auto-closes if open)
};

struct GdsPath {
    int16_t layer = 0;
    int16_t datatype = 0;
    int16_t pathtype = 0;  // 0: flush, 1: round, 2: half-width ext, 4: custom ext
    int32_t width = 0;
    int32_t bgnextn = 0;
    int32_t endextn = 0;
    std::vector<GdsPoint> xy;
};

struct GdsText {
    int16_t layer = 0;
    int16_t texttype = 0;
    int32_t x = 0;
    int32_t y = 0;
    std::string text;
};

struct GdsSref {
    std::string cell_name;
    int32_t x = 0;
    int32_t y = 0;
    uint16_t strans = 0;   // bit 15 (0x8000): mirror X (y -> -y)
    double mag = 1.0;
    double angle = 0.0;    // CCW degrees
};

struct GdsAref {
    std::string cell_name;
    int16_t cols = 1;
    int16_t rows = 1;
    GdsPoint origin{};     // xy[0]
    GdsPoint col_end{};    // xy[1]: origin + cols * col_step
    GdsPoint row_end{};    // xy[2]: origin + rows * row_step
    uint16_t strans = 0;
    double mag = 1.0;
    double angle = 0.0;
};

struct GdsCell {
    std::string name;
    std::vector<GdsBoundary> boundaries;
    std::vector<GdsPath> paths;
    std::vector<GdsText> texts;
    std::vector<GdsSref> srefs;
    std::vector<GdsAref> arefs;

    void add_box(int16_t layer, int16_t datatype, const Box64& box);

    void add_polygon(int16_t layer, int16_t datatype, const std::vector<Point64>& pts) {
        if (pts.size() < 3) return;
        GdsBoundary b{};
        b.layer = layer;
        b.datatype = datatype;
        b.xy.reserve(pts.size() + 1);
        for (const auto& pt : pts) {
            b.xy.push_back(GdsPoint{static_cast<int32_t>(pt.x), static_cast<int32_t>(pt.y)});
        }
        if (!(b.xy.front() == b.xy.back())) {
            b.xy.push_back(b.xy.front());
        }
        boundaries.push_back(std::move(b));
    }

    void add_text(int16_t layer, int16_t texttype, std::string text, Point64 pos) {
        GdsText t{};
        t.layer = layer;
        t.texttype = texttype;
        t.x = static_cast<int32_t>(pos.x);
        t.y = static_cast<int32_t>(pos.y);
        t.text = std::move(text);
        texts.push_back(std::move(t));
    }
};

struct GdsLibrary {
    std::string name = "CLAY_LIB";
    double user_unit = 1e-3;       // dbu in user units (default 1 nm / 1 um = 1e-3)
    double dbu_meters = 1e-9;      // dbu in meters (default 1 nm = 1e-9)
    std::vector<GdsCell> cells;

    GdsCell& add_cell(std::string cell_name);
    const GdsCell* find_cell(const std::string& cell_name) const noexcept;
    std::vector<std::string> top_cell_names() const;
};

struct LayerKey {
    int16_t layer = 0;
    int16_t datatype = 0;

    constexpr bool operator==(const LayerKey& o) const noexcept {
        return layer == o.layer && datatype == o.datatype;
    }
    constexpr bool operator<(const LayerKey& o) const noexcept {
        return layer != o.layer ? layer < o.layer : datatype < o.datatype;
    }
};

struct FlattenedLayerSummary {
    int16_t layer = 0;
    int16_t datatype = 0;
    size_t raw_polygon_count = 0;
    size_t canonical_rect_count = 0;
    core::U128 union_area_dbu2{};
    Box64 bbox{};
};

// Exact GDSII 8-byte Excess-64 real conversion.
std::array<uint8_t, 8> double_to_gds_real(double value) noexcept;
double gds_real_to_double(const uint8_t bytes[8]) noexcept;

// Binary GDSII stream serialization (deterministic fixed timestamp for reproducible SHA-256).
std::vector<uint8_t> write_gds_bytes(const GdsLibrary& lib);
bool write_gds_file(const GdsLibrary& lib, const std::string& path, std::string* err_out = nullptr);

// Binary GDSII stream deserialization.
bool read_gds_bytes(const uint8_t* data, size_t size, GdsLibrary& out_lib,
                    std::string* err_out = nullptr);
bool read_gds_file(const std::string& path, GdsLibrary& out_lib, std::string* err_out = nullptr);

// Flatten Manhattan/bounding-box geometry of `top_cell` (or first top cell if empty)
// by (layer, datatype) using exact integer transforms across SREF/AREF hierarchy.
std::map<LayerKey, std::vector<Box64>> flatten_boxes_by_layer(
    const GdsLibrary& lib,
    const std::string& top_cell = "");

// Compute canonical PolygonSet union area & bbox per layer for a cell hierarchy.
std::vector<FlattenedLayerSummary> summarize_gds_layers(
    const GdsLibrary& lib,
    const std::string& top_cell = "");

}  // namespace clay_pnr::gds
