// Pure C++20 Hierarchical Layout Database, Grid Snapper, Per-Net Merger, and
// Multi-Layer Physical Connectivity Extractor.
// Strictly free of <Python.h> and KLayout.

#include "clay_pnr/layout/layout_db.hpp"

#include "clay_pnr/route/gridless_spatial.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace clay_pnr::layout {

namespace {

struct Dsu {
    std::vector<uint32_t> parent;
    explicit Dsu(size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0U); }
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

inline bool boxes_touch_closed(const Box64& a, const Box64& b) noexcept {
    return a.x0 <= b.x1 && b.x0 <= a.x1 && a.y0 <= b.y1 && b.y0 <= a.y1;
}

}  // namespace

void LayoutCell::add_box(LayerKey key, Box64 b) {
    if (b.x1 <= b.x0 || b.y1 <= b.y0) return;
    LayerPoly lp{};
    lp.key = key;
    lp.vertices = {
        Point64{b.x0, b.y0},
        Point64{b.x1, b.y0},
        Point64{b.x1, b.y1},
        Point64{b.x0, b.y1},
    };
    polygons.push_back(std::move(lp));
}

void LayoutCell::add_polygon(LayerKey key, std::vector<Point64> verts) {
    if (verts.size() < 3) return;
    polygons.push_back(LayerPoly{key, std::move(verts)});
}

void LayoutCell::add_text(LayerKey key, std::string text, Point64 pos) {
    texts.push_back(LayoutText{key, std::move(text), pos});
}

void LayoutCell::translate(int64_t dx, int64_t dy) {
    if (dx == 0 && dy == 0) return;
    for (auto& lp : polygons) {
        for (auto& pt : lp.vertices) {
            pt.x += dx;
            pt.y += dy;
        }
    }
    for (auto& t : texts) {
        t.pos.x += dx;
        t.pos.y += dy;
    }
    for (auto& inst : instances) {
        inst.origin.x += dx;
        inst.origin.y += dy;
    }
}

size_t LayoutCell::snap_to_grid(int64_t grid_nm) {
    if (grid_nm <= 1) return 0;
    size_t moved = 0;
    for (auto& lp : polygons) {
        bool changed = false;
        for (auto& pt : lp.vertices) {
            const int64_t nx = snap_coord_nearest_up(pt.x, grid_nm);
            const int64_t ny = snap_coord_nearest_up(pt.y, grid_nm);
            if (nx != pt.x || ny != pt.y) {
                pt.x = nx;
                pt.y = ny;
                changed = true;
            }
        }
        if (changed) ++moved;
    }
    return moved;
}

bool LayoutCell::compute_bbox(Box64& out) const noexcept {
    bool any = false;
    int64_t x0 = std::numeric_limits<int64_t>::max();
    int64_t y0 = std::numeric_limits<int64_t>::max();
    int64_t x1 = std::numeric_limits<int64_t>::min();
    int64_t y1 = std::numeric_limits<int64_t>::min();
    for (const auto& lp : polygons) {
        for (const auto& pt : lp.vertices) {
            any = true;
            if (pt.x < x0) x0 = pt.x;
            if (pt.y < y0) y0 = pt.y;
            if (pt.x > x1) x1 = pt.x;
            if (pt.y > y1) y1 = pt.y;
        }
    }
    if (!any) return false;
    out = Box64{x0, y0, x1, y1};
    return true;
}

LayoutCell& LayoutDB::add_cell(std::string cell_name) {
    cells.push_back(LayoutCell{std::move(cell_name), {}, {}, {}});
    return cells.back();
}

LayoutCell* LayoutDB::find_cell(const std::string& cell_name) {
    for (auto& c : cells) {
        if (c.name == cell_name) return &c;
    }
    return nullptr;
}

const LayoutCell* LayoutDB::find_cell(const std::string& cell_name) const {
    for (const auto& c : cells) {
        if (c.name == cell_name) return &c;
    }
    return nullptr;
}

size_t LayoutDB::snap_all_to_grid(int64_t grid_nm) {
    if (grid_nm <= 1) return 0;
    size_t total = 0;
    for (auto& c : cells) total += c.snap_to_grid(grid_nm);
    return total;
}

clay_pnr::gds::GdsLibrary LayoutDB::to_gds_library() const {
    clay_pnr::gds::GdsLibrary lib;
    lib.name = name;
    lib.user_unit = user_unit;
    lib.dbu_meters = dbu_meters;
    for (const auto& cell : cells) {
        auto& gc = lib.add_cell(cell.name);
        for (const auto& lp : cell.polygons) {
            gc.add_polygon(lp.key.layer, lp.key.datatype, lp.vertices);
        }
        for (const auto& t : cell.texts) {
            gc.add_text(t.key.layer, t.key.datatype, t.text, t.pos);
        }
        for (const auto& inst : cell.instances) {
            clay_pnr::gds::GdsSref ref{};
            ref.cell_name = inst.cell_name;
            ref.x = static_cast<int32_t>(inst.origin.x);
            ref.y = static_cast<int32_t>(inst.origin.y);
            ref.strans = inst.mirror_x ? static_cast<uint16_t>(0x8000U) : static_cast<uint16_t>(0U);
            ref.angle = inst.angle_deg;
            gc.srefs.push_back(std::move(ref));
        }
    }
    return lib;
}

bool LayoutDB::write_gds(const std::string& path, std::string* err) const {
    const auto lib = to_gds_library();
    return clay_pnr::gds::write_gds_file(lib, path, err);
}

std::vector<LayerPoly> merge_net_boxes_exact(const std::vector<NetLayerBoxes>& items) {
    // Group deterministically by (net, LayerKey) so cross-net boxes are NEVER merged.
    std::map<std::pair<std::string, LayerKey>, std::vector<Box64>> grouped;
    for (const auto& item : items) {
        auto& dst = grouped[{item.net, item.key}];
        dst.insert(dst.end(), item.boxes.begin(), item.boxes.end());
    }

    std::vector<LayerPoly> out;
    for (const auto& [nk, boxes] : grouped) {
        const LayerKey& key = nk.second;
        const PolygonSet ps = PolygonSet::from_boxes(boxes);
        if (ps.empty()) continue;

        // Decompose into connected components. For any hole-free component, emit a single
        // merged orthogonal contour; for components with holes, emit canonical hole-free
        // horizontal slices (exact area & zero GDS self-intersection!).
        const auto comps = ps.connected_components(false);
        for (const auto& comp : comps) {
            const auto contours = comp.to_contours();
            bool has_hole = false;
            for (const auto& c : contours) {
                if (c.is_hole) {
                    has_hole = true;
                    break;
                }
            }
            if (!has_hole && contours.size() == 1 && contours[0].points.size() >= 4) {
                out.push_back(LayerPoly{key, contours[0].points});
            } else {
                for (const auto& r : comp.rects()) {
                    out.push_back(LayerPoly{
                        key,
                        {Point64{r.x0, r.y0}, Point64{r.x1, r.y0},
                         Point64{r.x1, r.y1}, Point64{r.x0, r.y1}},
                    });
                }
            }
        }
    }
    return out;
}

std::vector<ConnectivityComponent> extract_stack_connectivity(
    const std::vector<LayerBox>& shapes,
    const std::vector<LayerKey>& conductors,
    const std::vector<StackConnection>& connections,
    const std::vector<MimCarveRule>& mim_carves) {
    if (shapes.empty()) return {};

    // Bucket raw boxes by LayerKey.
    std::map<LayerKey, std::vector<Box64>> by_layer;
    for (const auto& sb : shapes) {
        if (sb.box.x1 > sb.box.x0 && sb.box.y1 > sb.box.y0) by_layer[sb.key].push_back(sb.box);
    }

    // Build lookup of MIM carve rules keyed by (bottom, riser).
    std::map<std::pair<LayerKey, LayerKey>, MimCarveRule> carves_by_conn;
    std::set<LayerKey> marker_layers;
    for (const auto& mc : mim_carves) {
        carves_by_conn[{mc.bottom, mc.riser}] = mc;
        marker_layers.insert(mc.top_marker);
    }

    // Build canonical PolygonSets per layer.
    std::map<LayerKey, PolygonSet> layer_ps;
    for (const auto& [lk, boxes] : by_layer) {
        layer_ps[lk] = PolygonSet::from_boxes(boxes);
    }

    // Each node in the connectivity graph is one canonical box on a logical layer slot.
    // Logical layer slots include real LayerKeys plus virtual carved bottom-metal slots.
    struct Node {
        LayerKey output_layer{};
        Box64 box{};
        uint32_t slot_id = 0;
    };

    std::map<LayerKey, uint32_t> layer_to_slot;
    uint32_t next_slot = 0;
    auto get_slot = [&](LayerKey lk) -> uint32_t {
        auto it = layer_to_slot.find(lk);
        if (it != layer_to_slot.end()) return it->second;
        const uint32_t id = next_slot++;
        layer_to_slot[lk] = id;
        return id;
    };

    std::vector<Node> nodes;
    std::map<uint32_t, clay_pnr::route::PureSpatialIndex> slot_index;
    std::map<uint32_t, std::vector<uint32_t>> slot_nodes;

    auto add_ps_to_slot = [&](uint32_t slot, LayerKey output_lk, const PolygonSet& ps) {
        auto& idx = slot_index[slot];
        auto& lst = slot_nodes[slot];
        for (const auto& r : ps.rects()) {
            const uint32_t nid = static_cast<uint32_t>(nodes.size());
            nodes.push_back(Node{output_lk, r, slot});
            idx.add_entry(r, nid, static_cast<uint64_t>(nid), static_cast<uint64_t>(nid));
            lst.push_back(nid);
        }
    };

    // Populate base layer slots.
    for (const auto& [lk, ps] : layer_ps) {
        add_ps_to_slot(get_slot(lk), lk, ps);
    }

    // Populate virtual MIM carve slots where both bottom and top_marker are present.
    struct VirtualCarveSlots {
        uint32_t lower_slot = 0;
        uint32_t mim_virtual_slot = 0;
        uint32_t metal_ncap_slot = 0;
        uint32_t marker_slot = 0;
        uint32_t cut_slot = 0;
        bool active = false;
    };
    std::vector<VirtualCarveSlots> active_carves;
    std::set<std::pair<LayerKey, LayerKey>> carved_pairs;

    for (const auto& conn : connections) {
        auto mc_it = carves_by_conn.find({conn.lower, conn.cut});
        if (mc_it == carves_by_conn.end()) continue;
        auto low_it = layer_ps.find(conn.lower);
        auto mrk_it = layer_ps.find(mc_it->second.top_marker);
        auto cut_it = layer_ps.find(conn.cut);
        if (low_it == layer_ps.end() || mrk_it == layer_ps.end() || cut_it == layer_ps.end()) continue;

        const auto& carve = mc_it->second;
        const PolygonSet& lower = low_it->second;
        const PolygonSet& marker = mrk_it->second;

        // Interacting components of `lower` that touch `marker`.
        std::vector<Box64> interacting_boxes;
        for (const auto& comp : lower.connected_components(true)) {
            if (comp.touches(marker)) {
                const auto& cr = comp.rects();
                interacting_boxes.insert(interacting_boxes.end(), cr.begin(), cr.end());
            }
        }
        const PolygonSet interacting = PolygonSet::from_boxes(interacting_boxes);
        PolygonSet mim_virtual;
        if (carve.marker_only) {
            mim_virtual = PolygonSet::boolean_op(marker, interacting, clay_pnr::core::BoolOp::And);
        } else if (carve.oversize_dbu > 0) {
            const PolygonSet grown = marker.sized(carve.oversize_dbu, carve.oversize_dbu);
            mim_virtual = PolygonSet::boolean_op(grown, interacting, clay_pnr::core::BoolOp::And);
        } else {
            mim_virtual = interacting;
        }
        const PolygonSet metal_ncap = PolygonSet::boolean_op(lower, mim_virtual, clay_pnr::core::BoolOp::Not);

        VirtualCarveSlots vcs{};
        vcs.lower_slot = get_slot(conn.lower);
        vcs.marker_slot = get_slot(carve.top_marker);
        vcs.cut_slot = get_slot(conn.cut);
        vcs.mim_virtual_slot = next_slot++;
        vcs.metal_ncap_slot = next_slot++;
        vcs.active = true;
        add_ps_to_slot(vcs.mim_virtual_slot, conn.lower, mim_virtual);
        add_ps_to_slot(vcs.metal_ncap_slot, conn.lower, metal_ncap);
        active_carves.push_back(vcs);
        carved_pairs.insert({conn.lower, conn.cut});
    }

    Dsu dsu(nodes.size());

    auto connect_slot_pair = [&](uint32_t s1, uint32_t s2) {
        auto it1 = slot_nodes.find(s1);
        auto it2 = slot_index.find(s2);
        if (it1 == slot_nodes.end() || it2 == slot_index.end()) return;
        const auto& n1 = it1->second;
        const auto& idx2 = it2->second;
        for (uint32_t u : n1) {
            const Box64& bu = nodes[u].box;
            idx2.for_each_intersecting(bu, [&](const clay_pnr::route::BucketItem& item) {
                const uint32_t v = item.owner_id;
                if (u != v && boxes_touch_closed(bu, nodes[v].box)) {
                    dsu.unite(u, v);
                }
            });
        }
    };

    // 1. Same-layer connectivity on all conductor layers.
    for (const auto& c : conductors) {
        auto it = layer_to_slot.find(c);
        if (it != layer_to_slot.end()) {
            connect_slot_pair(it->second, it->second);
        }
    }

    // 2. Vertical via connections (plus MIM carve isolation links).
    for (const auto& vcs : active_carves) {
        connect_slot_pair(vcs.lower_slot, vcs.mim_virtual_slot);
        connect_slot_pair(vcs.metal_ncap_slot, vcs.mim_virtual_slot);
        connect_slot_pair(vcs.metal_ncap_slot, vcs.lower_slot);
        connect_slot_pair(vcs.metal_ncap_slot, vcs.cut_slot);
        connect_slot_pair(vcs.marker_slot, vcs.cut_slot);
    }

    for (const auto& conn : connections) {
        auto cut_it = layer_to_slot.find(conn.cut);
        if (cut_it == layer_to_slot.end()) continue;
        const uint32_t cut_s = cut_it->second;
        if (carved_pairs.find({conn.lower, conn.cut}) == carved_pairs.end()) {
            auto low_it = layer_to_slot.find(conn.lower);
            if (low_it != layer_to_slot.end()) connect_slot_pair(low_it->second, cut_s);
        }
        auto up_it = layer_to_slot.find(conn.upper);
        if (up_it != layer_to_slot.end()) connect_slot_pair(cut_s, up_it->second);
    }

    // Collect non-virtual shapes by DSU root (only base slots correspond to real shapes).
    const uint32_t base_slot_limit = static_cast<uint32_t>(layer_to_slot.size());
    std::map<uint32_t, std::vector<LayerBox>> comp_map;
    for (uint32_t nid = 0; nid < nodes.size(); ++nid) {
        if (nodes[nid].slot_id >= base_slot_limit) continue;
        const uint32_t root = dsu.find(nid);
        comp_map[root].push_back(LayerBox{nodes[nid].output_layer, nodes[nid].box});
    }

    std::vector<ConnectivityComponent> result;
    result.reserve(comp_map.size());
    uint32_t cid = 0;
    for (auto& [root, list] : comp_map) {
        (void)root;
        std::sort(list.begin(), list.end(), [](const LayerBox& a, const LayerBox& b) {
            if (a.key != b.key) return a.key < b.key;
            if (a.box.x0 != b.box.x0) return a.box.x0 < b.box.x0;
            if (a.box.y0 != b.box.y0) return a.box.y0 < b.box.y0;
            if (a.box.x1 != b.box.x1) return a.box.x1 < b.box.x1;
            return a.box.y1 < b.box.y1;
        });
        result.push_back(ConnectivityComponent{cid++, std::move(list)});
    }
    return result;
}

// ---------------------------------------------------------------------------
// Orthogonal Polygon Decomposition & Exact Union Coverage
// ---------------------------------------------------------------------------

bool decompose_orthogonal_polygon_boxes(
    const std::vector<Point64>& raw_ring,
    std::vector<Box64>& out_boxes,
    std::string* err) {
    if (raw_ring.size() < 3) {
        if (err) *err = "placed layer rectangle authority has a degenerate polygon";
        return false;
    }

    std::vector<Point64> pts;
    pts.reserve(raw_ring.size());
    for (const auto& p : raw_ring) {
        if (pts.empty() || pts.back().x != p.x || pts.back().y != p.y) {
            pts.push_back(p);
        }
    }
    while (pts.size() > 1 && pts.front().x == pts.back().x && pts.front().y == pts.back().y) {
        pts.pop_back();
    }
    if (pts.size() < 3) {
        if (err) *err = "placed layer rectangle authority has a zero-area polygon";
        return false;
    }

    const size_t n = pts.size();
    int64_t area2 = 0;
    const int64_t ox = pts[0].x;
    const int64_t oy = pts[0].y;
    for (size_t i = 0; i < n; ++i) {
        const Point64& a = pts[i];
        const Point64& b = pts[(i + 1) % n];
        if (a.x != b.x && a.y != b.y) {
            if (err) *err = "placed layer rectangle authority requires Manhattan artwork";
            return false;
        }
        area2 += (a.x - ox) * (b.y - oy) - (b.x - ox) * (a.y - oy);
    }
    if (area2 == 0) {
        if (err) *err = "placed layer rectangle authority has a zero-area polygon";
        return false;
    }

    std::vector<Point64> corners;
    corners.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const Point64& prev = pts[(i + n - 1) % n];
        const Point64& cur = pts[i];
        const Point64& next = pts[(i + 1) % n];
        const bool same_x = (prev.x == cur.x && cur.x == next.x);
        const bool same_y = (prev.y == cur.y && cur.y == next.y);
        if (!same_x && !same_y) {
            corners.push_back(cur);
        }
    }
    if (corners.size() < 4) {
        if (err) *err = "placed layer rectangle authority has a zero-area polygon";
        return false;
    }

    if (corners.size() == 4) {
        int64_t x0 = corners[0].x, x1 = corners[0].x;
        int64_t y0 = corners[0].y, y1 = corners[0].y;
        for (size_t i = 1; i < 4; ++i) {
            if (corners[i].x < x0) x0 = corners[i].x;
            if (corners[i].x > x1) x1 = corners[i].x;
            if (corners[i].y < y0) y0 = corners[i].y;
            if (corners[i].y > y1) y1 = corners[i].y;
        }
        if (x1 <= x0 || y1 <= y0) {
            if (err) *err = "placed layer rectangle authority has a zero-area polygon";
            return false;
        }
        out_boxes.push_back(Box64{x0, y0, x1, y1});
        return true;
    }

    struct VEdge {
        int64_t x;
        int64_t y0;
        int64_t y1;
    };
    std::vector<VEdge> vedges;
    std::vector<int64_t> ys;
    vedges.reserve(corners.size() / 2 + 1);
    ys.reserve(corners.size());
    const size_t nc = corners.size();
    for (size_t i = 0; i < nc; ++i) {
        const Point64& a = corners[i];
        const Point64& b = corners[(i + 1) % nc];
        if (a.x == b.x && a.y != b.y) {
            const int64_t lo = std::min(a.y, b.y);
            const int64_t hi = std::max(a.y, b.y);
            vedges.push_back(VEdge{a.x, lo, hi});
            ys.push_back(lo);
            ys.push_back(hi);
        }
    }
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    if (ys.size() < 2) {
        if (err) *err = "placed layer rectangle authority has a zero-area polygon";
        return false;
    }

    std::vector<int64_t> xs;
    xs.reserve(vedges.size());
    const size_t before_count = out_boxes.size();
    for (size_t k = 0; k + 1 < ys.size(); ++k) {
        const int64_t y0 = ys[k];
        const int64_t y1 = ys[k + 1];
        if (y1 <= y0) continue;
        xs.clear();
        for (const auto& e : vedges) {
            if (e.y0 <= y0 && e.y1 >= y1) {
                xs.push_back(e.x);
            }
        }
        if (xs.empty()) continue;
        if ((xs.size() & 1U) != 0U) {
            if (err) *err = "placed layer rectangle authority requires Manhattan artwork";
            return false;
        }
        std::sort(xs.begin(), xs.end());
        for (size_t j = 0; j < xs.size(); j += 2) {
            if (xs[j + 1] > xs[j]) {
                out_boxes.push_back(Box64{xs[j], y0, xs[j + 1], y1});
            }
        }
    }
    if (out_boxes.size() == before_count) {
        if (err) *err = "placed layer rectangle authority has a zero-area polygon";
        return false;
    }
    return true;
}

bool boxes_covered_by_union_exact(
    const std::vector<Box64>& actual_boxes,
    const std::vector<Box64>& authority_boxes) {
    if (actual_boxes.empty()) return true;
    if (authority_boxes.empty()) return false;

    std::vector<Box64> uncovered;
    for (const auto& a : actual_boxes) {
        if (a.x1 <= a.x0 || a.y1 <= a.y0) continue;
        bool contained = false;
        for (const auto& auth : authority_boxes) {
            if (auth.x0 <= a.x0 && auth.y0 <= a.y0 && auth.x1 >= a.x1 && auth.y1 >= a.y1) {
                contained = true;
                break;
            }
        }
        if (!contained) uncovered.push_back(a);
    }
    if (uncovered.empty()) return true;

    const PolygonSet ps_act = PolygonSet::from_boxes(uncovered);
    if (ps_act.empty()) return true;
    const PolygonSet ps_auth = PolygonSet::from_boxes(authority_boxes);
    return PolygonSet::op_not(ps_act, ps_auth).empty();
}

// ---------------------------------------------------------------------------
// Exact Net-Connectivity & Net-Wirelength Kernels
// ---------------------------------------------------------------------------

NetConnectivityResult verify_net_connectivity_exact(
    const std::vector<NetShapeBox>& shapes,
    const std::vector<LayerKey>& via_layers) {
    NetConnectivityResult res;
    const size_t n = shapes.size();
    if (n == 0) {
        res.distinct = true;
        res.short_count = 0;
        return res;
    }

    std::set<LayerKey> via_set(via_layers.begin(), via_layers.end());
    std::vector<Box64> norm_boxes(n);
    std::vector<uint8_t> is_via(n, 0);
    for (size_t i = 0; i < n; ++i) {
        norm_boxes[i] = clay_pnr::core::normalize_box(
            shapes[i].box.x0, shapes[i].box.y0, shapes[i].box.x1, shapes[i].box.y1);
        is_via[i] = (via_set.find(shapes[i].key) != via_set.end()) ? 1U : 0U;
    }

    std::vector<uint32_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0U);
    auto find_root = [&](uint32_t x) noexcept {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite_to_second = [&](uint32_t i, uint32_t j) noexcept {
        const uint32_t ri = find_root(i);
        const uint32_t rj = find_root(j);
        if (ri != rj) parent[ri] = rj;
    };

    for (size_t i = 0; i < n; ++i) {
        const Box64& bi = norm_boxes[i];
        const LayerKey ki = shapes[i].key;
        const uint8_t vi = is_via[i];
        for (size_t j = i + 1; j < n; ++j) {
            const Box64& bj = norm_boxes[j];
            if (bi.x0 > bj.x1 || bj.x0 > bi.x1 || bi.y0 > bj.y1 || bj.y0 > bi.y1) continue;
            if (ki == shapes[j].key || vi != 0U || is_via[j] != 0U) {
                unite_to_second(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
            }
        }
    }

    res.shape_root.resize(n);
    std::map<uint32_t, uint32_t> net_min_root;
    std::map<uint32_t, std::set<uint32_t>> root_to_nets;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t r = find_root(static_cast<uint32_t>(i));
        res.shape_root[i] = r;
        const uint32_t net_id = shapes[i].net_id;
        auto it = net_min_root.find(net_id);
        if (it == net_min_root.end() || r < it->second) {
            net_min_root[net_id] = r;
        }
        root_to_nets[r].insert(net_id);
    }

    res.component_of.reserve(net_min_root.size());
    for (const auto& [nid, root] : net_min_root) {
        res.component_of.emplace_back(nid, root);
    }

    std::set<std::pair<uint32_t, uint32_t>> pair_ids;
    for (const auto& [root, nets] : root_to_nets) {
        (void)root;
        if (nets.size() <= 1) continue;
        std::vector<uint32_t> ordered(nets.begin(), nets.end());
        for (size_t a = 0; a < ordered.size(); ++a) {
            for (size_t b = a + 1; b < ordered.size(); ++b) {
                pair_ids.emplace(ordered[a], ordered[b]);
            }
        }
    }
    res.merged_pairs.assign(pair_ids.begin(), pair_ids.end());
    res.short_count = res.merged_pairs.size();
    res.distinct = res.merged_pairs.empty();
    return res;
}

NetWirelengthResult compute_net_wirelength_exact(
    const std::vector<std::vector<Point64>>& nets,
    const std::vector<Box64>& bboxes) {
    NetWirelengthResult res;
    res.net_total_span.reserve(nets.size());
    res.net_max_pin_span.reserve(nets.size());

    std::vector<Point64> pts;
    for (const auto& net : nets) {
        pts = net;
        std::sort(pts.begin(), pts.end(), [](const Point64& a, const Point64& b) {
            return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
        });
        pts.erase(std::unique(pts.begin(), pts.end(), [](const Point64& a, const Point64& b) {
            return a.x == b.x && a.y == b.y;
        }), pts.end());

        if (pts.size() < 2) {
            res.net_total_span.push_back(0);
            res.net_max_pin_span.push_back(0);
            continue;
        }
        int64_t min_x = pts[0].x, max_x = pts[0].x;
        int64_t min_y = pts[0].y, max_y = pts[0].y;
        for (size_t i = 1; i < pts.size(); ++i) {
            if (pts[i].x < min_x) min_x = pts[i].x;
            if (pts[i].x > max_x) max_x = pts[i].x;
            if (pts[i].y < min_y) min_y = pts[i].y;
            if (pts[i].y > max_y) max_y = pts[i].y;
        }
        const int64_t hpwl = (max_x - min_x) + (max_y - min_y);
        res.net_total_span.push_back(hpwl);
        res.netlist_total_hpwl += hpwl;

        int64_t min_s = pts[0].x + pts[0].y, max_s = min_s;
        int64_t min_d = pts[0].x - pts[0].y, max_d = min_d;
        for (size_t i = 1; i < pts.size(); ++i) {
            const int64_t s = pts[i].x + pts[i].y;
            const int64_t d = pts[i].x - pts[i].y;
            if (s < min_s) min_s = s;
            if (s > max_s) max_s = s;
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
        }
        res.net_max_pin_span.push_back(std::max(max_s - min_s, max_d - min_d));
    }

    int64_t ov = 0;
    const size_t nb = bboxes.size();
    for (size_t i = 0; i < nb; ++i) {
        const Box64& a = bboxes[i];
        for (size_t j = i + 1; j < nb; ++j) {
            const Box64& b = bboxes[j];
            if (a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1) {
                ++ov;
            }
        }
    }
    res.overlapping_pairs = ov;
    return res;
}

}  // namespace clay_pnr::layout
