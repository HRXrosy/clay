#pragma once

// Pure C++20 Hierarchical Layout Database, Manufacturing-Grid Snapper,
// Per-Net Polygon Merger, and Multi-Layer Physical Connectivity Extractor.
// Strictly free of <Python.h> and KLayout.

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/polygon_set.hpp"
#include "clay_pnr/gds/gds_stream.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace clay_pnr::layout {

using clay_pnr::core::Box64;
using clay_pnr::core::Point64;
using clay_pnr::core::PolygonSet;

struct LayerKey {
    int16_t layer = 0;
    int16_t datatype = 0;

    auto operator<=>(const LayerKey&) const noexcept = default;
};

struct LayerPoly {
    LayerKey key{};
    std::vector<Point64> vertices;
};

struct LayerBox {
    LayerKey key{};
    Box64 box{};
};

// Exact nearest manufacturing-grid snap (ties round UP, identical for +/- coords).
inline int64_t snap_coord_nearest_up(int64_t value_nm, int64_t grid_nm) noexcept {
    if (grid_nm <= 1) return value_nm;
    // Mathematical floor division toward -inf:
    int64_t q = value_nm / grid_nm;
    int64_t r = value_nm % grid_nm;
    if (r < 0) {
        r += grid_nm;
        --q;
    }
    const int64_t floor_v = q * grid_nm;
    return (2 * r < grid_nm) ? floor_v : (floor_v + grid_nm);
}

struct LayoutText {
    LayerKey key{};
    std::string text;
    Point64 pos{};
};

struct LayoutInstance {
    std::string cell_name;
    Point64 origin{};
    bool mirror_x = false;
    double angle_deg = 0.0;
};

struct LayoutCell {
    std::string name;
    std::vector<LayerPoly> polygons;
    std::vector<LayoutText> texts;
    std::vector<LayoutInstance> instances;

    void add_box(LayerKey key, Box64 b);
    void add_polygon(LayerKey key, std::vector<Point64> verts);
    void add_text(LayerKey key, std::string text, Point64 pos);
    void translate(int64_t dx, int64_t dy);
    size_t snap_to_grid(int64_t grid_nm);
    bool compute_bbox(Box64& out) const noexcept;
};

class LayoutDB {
public:
    std::string name = "CLAY";
    double user_unit = 1e-3;
    double dbu_meters = 1e-9;
    std::vector<LayoutCell> cells;

    LayoutCell& add_cell(std::string cell_name);
    LayoutCell* find_cell(const std::string& cell_name);
    const LayoutCell* find_cell(const std::string& cell_name) const;

    // Snap every polygon vertex across all cells to `grid_nm` (labels untouched).
    size_t snap_all_to_grid(int64_t grid_nm);

    // Export directly to binary GDSII v6.0 via clay_pnr::gds (zero KLayout, deterministic).
    clay_pnr::gds::GdsLibrary to_gds_library() const;
    bool write_gds(const std::string& path, std::string* err = nullptr) const;
};

// Per-net x per-layer orthogonal merge returning GDS-safe single-contour orthogonal polygons.
// Never unions across nets.
struct NetLayerBoxes {
    std::string net;
    LayerKey key{};
    std::vector<Box64> boxes;
};

struct MergedNetPoly {
    std::string net;
    LayerKey key{};
    std::vector<Point64> vertices;  // hole-free orthogonal boundary or slice
};

// Merges boxes per (net, layer, datatype) via PolygonSet::from_boxes.
// Note: if a merged island contains holes, returns exact canonical non-overlapping
// slice boxes (hole-free GDS-safe) so every emitted polygon is simple & exact area!
std::vector<LayerPoly> merge_net_boxes_exact(const std::vector<NetLayerBoxes>& items);

// ---------------------------------------------------------------------------
// Multi-Layer Physical Connectivity Extractor (pure C++20, zero KLayout L2N)
// ---------------------------------------------------------------------------

struct StackConnection {
    LayerKey lower{};
    LayerKey cut{};
    LayerKey upper{};
};

struct MimCarveRule {
    LayerKey bottom{};
    LayerKey riser{};
    LayerKey top_marker{};
    int64_t oversize_dbu = 0;
    bool marker_only = false;
};

struct ConnectivityComponent {
    uint32_t component_id = 0;
    std::vector<LayerBox> shapes;
};

// Extracts exact connected components over a multi-layer orthogonal conductor/via/MIM stack.
// Shapes touching by edge or corner on the same conductor layer merge; cut shapes join
// overlapping lower & upper conductor shapes (respecting MIM carve isolation on bottom plate).
std::vector<ConnectivityComponent> extract_stack_connectivity(
    const std::vector<LayerBox>& shapes,
    const std::vector<LayerKey>& conductors,
    const std::vector<StackConnection>& connections,
    const std::vector<MimCarveRule>& mim_carves = {});

// ---------------------------------------------------------------------------
// Orthogonal Polygon Decomposition & Exact Union Coverage (zero KLayout)
// ---------------------------------------------------------------------------

// Validates a closed rectilinear polygon (rejecting < 3 vertices, diagonal edges,
// or zero area) and appends exact non-overlapping horizontal-strip boxes covering it.
bool decompose_orthogonal_polygon_boxes(
    const std::vector<Point64>& raw_ring,
    std::vector<Box64>& out_boxes,
    std::string* err = nullptr);

// Exact check: true iff every point of `actual_boxes` is covered by the union of
// `authority_boxes` (exact equivalent to `(Region(actual) - Region(authority)).is_empty()`).
bool boxes_covered_by_union_exact(
    const std::vector<Box64>& actual_boxes,
    const std::vector<Box64>& authority_boxes);

// ---------------------------------------------------------------------------
// Exact Net-Connectivity & Net-Wirelength Kernels (drop-in for clay_layout)
// ---------------------------------------------------------------------------

struct NetShapeBox {
    uint32_t net_id = 0;
    LayerKey key{};
    Box64 box{};
};

struct NetConnectivityResult {
    // Pairs (net_id, min_component_root) sorted by net_id.
    std::vector<std::pair<uint32_t, uint32_t>> component_of;
    // Sorted unique (net_a, net_b) shorted pairs with net_a < net_b.
    std::vector<std::pair<uint32_t, uint32_t>> merged_pairs;
    bool distinct = true;
    std::vector<uint32_t> shape_root;
    size_t short_count = 0;
};

// Exact mirror of clay/layout/_rs_bridge_net_connectivity.py:py_connected_components
// (path-halving DSU with union-toward-second-root in (i, j) order).
NetConnectivityResult verify_net_connectivity_exact(
    const std::vector<NetShapeBox>& shapes,
    const std::vector<LayerKey>& via_layers);

struct NetWirelengthResult {
    std::vector<int64_t> net_total_span;
    std::vector<int64_t> net_max_pin_span;
    int64_t netlist_total_hpwl = 0;
    int64_t overlapping_pairs = 0;
};

// Exact mirror of clay/layout/_rs_bridge_net_wirelength.py.
NetWirelengthResult compute_net_wirelength_exact(
    const std::vector<std::vector<Point64>>& nets,
    const std::vector<Box64>& bboxes);

}  // namespace clay_pnr::layout

