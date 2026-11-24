// Thin CPython RAII binding for Clay's unified C++20 Main Engine (`clay._native._engine_cpp`).
// Exposes pure C++20:
//   - PolygonSet boolean/morphology/contour/component ops
//   - Native binary GDSII v6.0 reader/writer (boxes, polygons, text labels)
//   - Difference-Constraint 1D & 2D Compactors
//   - Exact-Constraint 2D Envelope-Min Analog Placer
//   - Per-Net Polygon Merger & Multi-Layer Physical Connectivity Extractor (with MIM carve)
//   - Rule-Driven MOS PCell & Shared-Diffusion Row Generators

#include "clay_pnr/core/py_raii.hpp"
#include "clay_pnr/core/polygon_set.hpp"
#include "clay_pnr/engine/standalone_synth.hpp"
#include "clay_pnr/gds/gds_stream.hpp"
#include "clay_pnr/ir/clay_ir.hpp"
#include "clay_pnr/layout/layout_db.hpp"
#include "clay_pnr/pcell/mos_pcell.hpp"
#include "clay_pnr/place/banded_placer.hpp"
#include "clay_pnr/place/compactor.hpp"
#include "clay_pnr/place/envelope_placer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using clay_pnr::core::BoolOp;
using clay_pnr::core::Box64;
using clay_pnr::core::Point64;
using clay_pnr::core::PolygonSet;
using clay_pnr::gds::GdsLibrary;
using clay_pnr::layout::LayerBox;
using clay_pnr::layout::LayerKey;
using clay_pnr::layout::MimCarveRule;
using clay_pnr::layout::NetLayerBoxes;
using clay_pnr::layout::StackConnection;
using clay_pnr::place::CompactStatus;
using clay_pnr::place::EnvelopeAxisGroup;
using clay_pnr::place::EnvelopeCluster;
using clay_pnr::place::EnvelopeGroup;
using clay_pnr::place::EnvelopeNet;
using clay_pnr::place::EnvelopeOrder;
using clay_pnr::place::EnvelopeProblem;
using clay_pnr::place::EnvelopeRailAffinity;
using clay_pnr::place::EnvelopeStatus;
using clay_pnr::place::IntDiffEdge;
using clay_pnr::place::SymmetryPairSpec;
using clay_pnr::py::PyRef;
using clay_pnr::py::make_box64_tuple;
using clay_pnr::py::parse_int64;
using clay_pnr::py::parse_rect_list;
using clay_pnr::py::parse_rect_norm;

PyObject* make_box_list(const std::vector<Box64>& boxes) {
    PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(boxes.size())));
    if (!lst) return nullptr;
    for (size_t i = 0; i < boxes.size(); ++i) {
        PyObject* t = make_box64_tuple(boxes[i]);
        if (!t) return nullptr;
        PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), t);
    }
    return lst.release();
}

PyObject* make_point_list(const std::vector<Point64>& pts) {
    PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pts.size())));
    if (!lst) return nullptr;
    for (size_t i = 0; i < pts.size(); ++i) {
        PyRef pt = PyRef::adopt(Py_BuildValue("(LL)", static_cast<long long>(pts[i].x),
                                              static_cast<long long>(pts[i].y)));
        if (!pt) return nullptr;
        PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), pt.release());
    }
    return lst.release();
}

PyObject* make_u128_pylong(clay_pnr::core::U128 v) {
    if (v.hi == 0) {
        return PyLong_FromUnsignedLongLong(v.lo);
    }
    PyRef hi = PyRef::adopt(PyLong_FromUnsignedLongLong(v.hi));
    PyRef shift = PyRef::adopt(PyLong_FromLong(64));
    PyRef hi_shifted = PyRef::adopt(PyNumber_Lshift(hi.get(), shift.get()));
    PyRef lo = PyRef::adopt(PyLong_FromUnsignedLongLong(v.lo));
    if (!hi || !shift || !hi_shifted || !lo) return nullptr;
    return PyNumber_Or(hi_shifted.get(), lo.get());
}

// ---------------------------------------------------------------------------
// Phase 0: PolygonSet & GDS Stream
// ---------------------------------------------------------------------------

PyObject* py_polygon_boolean(PyObject* /*self*/, PyObject* args) {
    PyObject* a_obj = nullptr;
    PyObject* b_obj = nullptr;
    const char* op_str = nullptr;
    if (!PyArg_ParseTuple(args, "OOs", &a_obj, &b_obj, &op_str)) return nullptr;
    std::vector<Box64> a_boxes, b_boxes;
    if (!parse_rect_list(a_obj, a_boxes) || !parse_rect_list(b_obj, b_boxes)) return nullptr;
    BoolOp op = BoolOp::Or;
    const std::string ops(op_str);
    if (ops == "or" || ops == "union") op = BoolOp::Or;
    else if (ops == "and" || ops == "intersect") op = BoolOp::And;
    else if (ops == "not" || ops == "diff") op = BoolOp::Not;
    else if (ops == "xor") op = BoolOp::Xor;
    else {
        PyErr_SetString(PyExc_ValueError, "op must be one of 'or', 'and', 'not', 'xor'");
        return nullptr;
    }
    const PolygonSet res = PolygonSet::boolean_op(
        PolygonSet::from_boxes(a_boxes), PolygonSet::from_boxes(b_boxes), op);
    return make_box_list(res.rects());
}

PyObject* py_polygon_sized(PyObject* /*self*/, PyObject* args) {
    PyObject* a_obj = nullptr;
    long long dx = 0, dy = 0;
    if (!PyArg_ParseTuple(args, "OLL", &a_obj, &dx, &dy)) return nullptr;
    std::vector<Box64> boxes;
    if (!parse_rect_list(a_obj, boxes)) return nullptr;
    const PolygonSet res = PolygonSet::from_boxes(boxes).sized(static_cast<int64_t>(dx), static_cast<int64_t>(dy));
    return make_box_list(res.rects());
}

PyObject* py_polygon_components(PyObject* /*self*/, PyObject* args) {
    PyObject* a_obj = nullptr;
    int include_corner = 1;
    if (!PyArg_ParseTuple(args, "O|p", &a_obj, &include_corner)) return nullptr;
    std::vector<Box64> boxes;
    if (!parse_rect_list(a_obj, boxes)) return nullptr;
    const auto comps = PolygonSet::from_boxes(boxes).connected_components(include_corner != 0);
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(comps.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < comps.size(); ++i) {
        PyObject* sub = make_box_list(comps[i].rects());
        if (!sub) return nullptr;
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), sub);
    }
    return out.release();
}

PyObject* py_polygon_contours(PyObject* /*self*/, PyObject* args) {
    PyObject* a_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &a_obj)) return nullptr;
    std::vector<Box64> boxes;
    if (!parse_rect_list(a_obj, boxes)) return nullptr;
    const auto contours = PolygonSet::from_boxes(boxes).to_contours();
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(contours.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < contours.size(); ++i) {
        PyRef pts = PyRef::adopt(make_point_list(contours[i].points));
        if (!pts) return nullptr;
        PyRef pair = PyRef::adopt(PyTuple_New(2));
        if (!pair) return nullptr;
        PyTuple_SET_ITEM(pair.get(), 0, pts.release());
        PyTuple_SET_ITEM(pair.get(), 1, PyBool_FromLong(contours[i].is_hole ? 1 : 0));
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), pair.release());
    }
    return out.release();
}

PyObject* py_gds_write_boxes(PyObject* /*self*/, PyObject* args) {
    const char* path = nullptr;
    const char* lib_name = nullptr;
    const char* cell_name = nullptr;
    PyObject* items_obj = nullptr;
    double user_unit = 1e-3, dbu_meters = 1e-9;
    if (!PyArg_ParseTuple(args, "sssO|dd", &path, &lib_name, &cell_name, &items_obj, &user_unit, &dbu_meters)) {
        return nullptr;
    }
    PyRef seq = PyRef::adopt(PySequence_Fast(items_obj, "items must be a sequence"));
    if (!seq) return nullptr;
    GdsLibrary lib;
    lib.name = lib_name;
    lib.user_unit = user_unit;
    lib.dbu_meters = dbu_meters;
    auto& cell = lib.add_cell(cell_name);
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (Py_ssize_t i = 0; i < n; ++i) {
        int layer = 0, dt = 0;
        long long x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        if (!PyArg_ParseTuple(items[i], "iiLLLL", &layer, &dt, &x0, &y0, &x1, &y1)) return nullptr;
        cell.add_box(static_cast<int16_t>(layer), static_cast<int16_t>(dt),
                     Box64{static_cast<int64_t>(x0), static_cast<int64_t>(y0),
                           static_cast<int64_t>(x1), static_cast<int64_t>(y1)});
    }
    std::string err;
    if (!clay_pnr::gds::write_gds_file(lib, path, &err)) {
        PyErr_SetString(PyExc_OSError, err.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_gds_write_polygons(PyObject* /*self*/, PyObject* args) {
    const char* path = nullptr;
    const char* lib_name = nullptr;
    const char* cell_name = nullptr;
    PyObject* polys_obj = nullptr;
    PyObject* texts_obj = Py_None;
    double user_unit = 1e-3, dbu_meters = 1e-9;
    if (!PyArg_ParseTuple(args, "sssO|Odd", &path, &lib_name, &cell_name, &polys_obj,
                          &texts_obj, &user_unit, &dbu_meters)) {
        return nullptr;
    }
    PyRef pseq = PyRef::adopt(PySequence_Fast(polys_obj, "polygons must be sequence of (layer, dt, [(x,y),...])"));
    if (!pseq) return nullptr;
    GdsLibrary lib;
    lib.name = lib_name;
    lib.user_unit = user_unit;
    lib.dbu_meters = dbu_meters;
    auto& cell = lib.add_cell(cell_name);

    const Py_ssize_t np = PySequence_Fast_GET_SIZE(pseq.get());
    PyObject** pitems = PySequence_Fast_ITEMS(pseq.get());
    for (Py_ssize_t i = 0; i < np; ++i) {
        int layer = 0, dt = 0;
        PyObject* pts_obj = nullptr;
        if (!PyArg_ParseTuple(pitems[i], "iiO", &layer, &dt, &pts_obj)) return nullptr;
        PyRef vseq = PyRef::adopt(PySequence_Fast(pts_obj, "vertices must be sequence of (x, y)"));
        if (!vseq) return nullptr;
        const Py_ssize_t nv = PySequence_Fast_GET_SIZE(vseq.get());
        PyObject** vitems = PySequence_Fast_ITEMS(vseq.get());
        std::vector<Point64> verts;
        verts.reserve(static_cast<size_t>(nv));
        for (Py_ssize_t j = 0; j < nv; ++j) {
            long long vx = 0, vy = 0;
            if (!PyArg_ParseTuple(vitems[j], "LL", &vx, &vy)) return nullptr;
            verts.push_back(Point64{static_cast<int64_t>(vx), static_cast<int64_t>(vy)});
        }
        cell.add_polygon(static_cast<int16_t>(layer), static_cast<int16_t>(dt), std::move(verts));
    }

    if (texts_obj != Py_None) {
        PyRef tseq = PyRef::adopt(PySequence_Fast(texts_obj, "texts must be sequence of (layer, dt, text, x, y)"));
        if (!tseq) return nullptr;
        const Py_ssize_t nt = PySequence_Fast_GET_SIZE(tseq.get());
        PyObject** titems = PySequence_Fast_ITEMS(tseq.get());
        for (Py_ssize_t i = 0; i < nt; ++i) {
            int layer = 0, dt = 0;
            const char* txt = nullptr;
            long long tx = 0, ty = 0;
            if (!PyArg_ParseTuple(titems[i], "iisLL", &layer, &dt, &txt, &tx, &ty)) return nullptr;
            cell.add_text(static_cast<int16_t>(layer), static_cast<int16_t>(dt), txt,
                          Point64{static_cast<int64_t>(tx), static_cast<int64_t>(ty)});
        }
    }

    std::string err;
    if (!clay_pnr::gds::write_gds_file(lib, path, &err)) {
        PyErr_SetString(PyExc_OSError, err.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_gds_inspect_layers(PyObject* /*self*/, PyObject* args) {
    const char* path = nullptr;
    const char* top_cell = "";
    if (!PyArg_ParseTuple(args, "s|s", &path, &top_cell)) return nullptr;
    GdsLibrary lib;
    std::string err;
    if (!clay_pnr::gds::read_gds_file(path, lib, &err)) {
        PyErr_SetString(PyExc_OSError, err.c_str());
        return nullptr;
    }
    const auto sums = clay_pnr::gds::summarize_gds_layers(lib, top_cell);
    PyRef out = PyRef::adopt(PyDict_New());
    if (!out) return nullptr;
    for (const auto& s : sums) {
        PyRef key = PyRef::adopt(Py_BuildValue("(ii)", static_cast<int>(s.layer), static_cast<int>(s.datatype)));
        PyRef area = PyRef::adopt(make_u128_pylong(s.union_area_dbu2));
        PyRef bb = PyRef::adopt(make_box64_tuple(s.bbox));
        if (!key || !area || !bb) return nullptr;
        PyRef val = PyRef::adopt(Py_BuildValue("(KKOO)",
                                               static_cast<unsigned long long>(s.raw_polygon_count),
                                               static_cast<unsigned long long>(s.canonical_rect_count),
                                               area.get(), bb.get()));
        if (!val) return nullptr;
        if (PyDict_SetItem(out.get(), key.get(), val.get()) < 0) return nullptr;
    }
    return out.release();
}

// ---------------------------------------------------------------------------
// Phase 3A: Pure C++20 Difference-Constraint Compactors & 2D Envelope Placer
// ---------------------------------------------------------------------------

PyObject* py_compact_axis(PyObject* /*self*/, PyObject* args) {
    PyObject* extents_obj = nullptr;
    PyObject* edges_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &extents_obj, &edges_obj)) return nullptr;
    PyRef eseq = PyRef::adopt(PySequence_Fast(extents_obj, "extents must be sequence of positive ints"));
    PyRef cseq = PyRef::adopt(PySequence_Fast(edges_obj, "edges must be sequence of (before, after, min_delta)"));
    if (!eseq || !cseq) return nullptr;

    const Py_ssize_t n = PySequence_Fast_GET_SIZE(eseq.get());
    std::vector<int64_t> extents(static_cast<size_t>(n));
    PyObject** eitems = PySequence_Fast_ITEMS(eseq.get());
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long long v = PyLong_AsLongLong(eitems[i]);
        if (v == -1 && PyErr_Occurred()) return nullptr;
        extents[static_cast<size_t>(i)] = static_cast<int64_t>(v);
    }

    const Py_ssize_t m = PySequence_Fast_GET_SIZE(cseq.get());
    std::vector<IntDiffEdge> edges;
    edges.reserve(static_cast<size_t>(m));
    PyObject** citems = PySequence_Fast_ITEMS(cseq.get());
    for (Py_ssize_t i = 0; i < m; ++i) {
        unsigned long u = 0, v = 0;
        long long d = 0;
        if (!PyArg_ParseTuple(citems[i], "kkL", &u, &v, &d)) return nullptr;
        edges.push_back(IntDiffEdge{static_cast<uint32_t>(u), static_cast<uint32_t>(v), static_cast<int64_t>(d)});
    }

    const auto res = clay_pnr::place::compact_axis_dag(extents, edges);
    if (res.status != CompactStatus::Ok) {
        PyErr_SetString(PyExc_ValueError, res.message.c_str());
        return nullptr;
    }

    PyRef plist = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.positions.size())));
    if (!plist) return nullptr;
    for (size_t i = 0; i < res.positions.size(); ++i) {
        PyObject* pv = PyLong_FromLongLong(static_cast<long long>(res.positions[i]));
        if (!pv) return nullptr;
        PyList_SET_ITEM(plist.get(), static_cast<Py_ssize_t>(i), pv);
    }
    return Py_BuildValue("(OL)", plist.get(), static_cast<long long>(res.extent));
}

PyObject* py_compact_2d(PyObject* /*self*/, PyObject* args) {
    PyObject* w_obj = nullptr;
    PyObject* h_obj = nullptr;
    PyObject* he_obj = nullptr;
    PyObject* ve_obj = nullptr;
    PyObject* sym_obj = Py_None;
    long long grid_step = 1;
    if (!PyArg_ParseTuple(args, "OOOO|OL", &w_obj, &h_obj, &he_obj, &ve_obj, &sym_obj, &grid_step)) {
        return nullptr;
    }
    auto parse_i64_seq = [](PyObject* obj, std::vector<int64_t>& out) -> bool {
        PyRef s = PyRef::adopt(PySequence_Fast(obj, "expected integer sequence"));
        if (!s) return false;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        out.resize(static_cast<size_t>(k));
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        for (Py_ssize_t i = 0; i < k; ++i) {
            const long long v = PyLong_AsLongLong(items[i]);
            if (v == -1 && PyErr_Occurred()) return false;
            out[static_cast<size_t>(i)] = static_cast<int64_t>(v);
        }
        return true;
    };
    auto parse_edge_seq = [](PyObject* obj, std::vector<IntDiffEdge>& out) -> bool {
        PyRef s = PyRef::adopt(PySequence_Fast(obj, "expected edge sequence"));
        if (!s) return false;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        out.reserve(static_cast<size_t>(k));
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        for (Py_ssize_t i = 0; i < k; ++i) {
            unsigned long u = 0, v = 0;
            long long d = 0;
            if (!PyArg_ParseTuple(items[i], "kkL", &u, &v, &d)) return false;
            out.push_back(IntDiffEdge{static_cast<uint32_t>(u), static_cast<uint32_t>(v), static_cast<int64_t>(d)});
        }
        return true;
    };

    std::vector<int64_t> widths, heights;
    std::vector<IntDiffEdge> h_edges, v_edges;
    std::vector<SymmetryPairSpec> syms;
    if (!parse_i64_seq(w_obj, widths) || !parse_i64_seq(h_obj, heights) ||
        !parse_edge_seq(he_obj, h_edges) || !parse_edge_seq(ve_obj, v_edges)) {
        return nullptr;
    }
    if (sym_obj != Py_None) {
        PyRef s = PyRef::adopt(PySequence_Fast(sym_obj, "symmetry_pairs must be sequence of (a, b)"));
        if (!s) return nullptr;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        for (Py_ssize_t i = 0; i < k; ++i) {
            unsigned long a = 0, b = 0;
            if (!PyArg_ParseTuple(items[i], "kk", &a, &b)) return nullptr;
            syms.push_back(SymmetryPairSpec{static_cast<uint32_t>(a), static_cast<uint32_t>(b)});
        }
    }

    const auto res = clay_pnr::place::compact_2d_topology(
        widths, heights, h_edges, v_edges, syms, static_cast<int64_t>(grid_step));
    if (res.status != CompactStatus::Ok) {
        PyErr_SetString(PyExc_ValueError, res.message.c_str());
        return nullptr;
    }
    PyRef xl = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.x.size())));
    PyRef yl = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.y.size())));
    if (!xl || !yl) return nullptr;
    for (size_t i = 0; i < res.x.size(); ++i) {
        PyList_SET_ITEM(xl.get(), static_cast<Py_ssize_t>(i), PyLong_FromLongLong(res.x[i]));
        PyList_SET_ITEM(yl.get(), static_cast<Py_ssize_t>(i), PyLong_FromLongLong(res.y[i]));
    }
    return Py_BuildValue("(OOLLL)", xl.get(), yl.get(),
                         static_cast<long long>(res.width),
                         static_cast<long long>(res.height),
                         static_cast<long long>(res.shared_axis2));
}

PyObject* py_solve_envelope_min(PyObject* /*self*/, PyObject* args) {
    PyObject* spec = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &spec)) return nullptr;

    EnvelopeProblem prob{};
    std::unordered_map<std::string, uint32_t> key_to_idx;

    // groups: [(key, raw_w, raw_h, fw, fh), ...]
    PyObject* groups_obj = PyDict_GetItemString(spec, "groups");
    if (!groups_obj) {
        PyErr_SetString(PyExc_KeyError, "spec missing 'groups'");
        return nullptr;
    }
    PyRef gseq = PyRef::adopt(PySequence_Fast(groups_obj, "groups must be sequence"));
    if (!gseq) return nullptr;
    const Py_ssize_t ng = PySequence_Fast_GET_SIZE(gseq.get());
    PyObject** gitems = PySequence_Fast_ITEMS(gseq.get());
    prob.groups.reserve(static_cast<size_t>(ng));
    for (Py_ssize_t i = 0; i < ng; ++i) {
        const char* key = nullptr;
        double rw = 0.0, rh = 0.0;
        long long fw = 1, fh = 1;
        if (!PyArg_ParseTuple(gitems[i], "sddLL", &key, &rw, &rh, &fw, &fh)) return nullptr;
        key_to_idx[key] = static_cast<uint32_t>(prob.groups.size());
        prob.groups.push_back(EnvelopeGroup{key, rw, rh, static_cast<int64_t>(fw), static_cast<int64_t>(fh)});
    }

    auto get_double = [&](const char* k, double def) -> double {
        PyObject* o = PyDict_GetItemString(spec, k);
        if (!o || o == Py_None) return def;
        return PyFloat_AsDouble(o);
    };
    auto get_i64 = [&](const char* k, int64_t def) -> int64_t {
        PyObject* o = PyDict_GetItemString(spec, k);
        if (!o || o == Py_None) return def;
        return static_cast<int64_t>(PyLong_AsLongLong(o));
    };

    prob.gap = get_double("gap", 0.5);
    prob.wl_weight = get_i64("wl_weight", 1);
    prob.maxspan_weight = get_i64("maxspan_weight", 3000);
    prob.aspect_target = get_double("aspect_target", 0.0);
    prob.aspect_q = (prob.aspect_target > 0.0)
                        ? std::max<int64_t>(1, static_cast<int64_t>(std::llround(prob.aspect_target * 100.0)))
                        : 0;
    prob.max_time_s = get_double("max_time_s", 5.0);
    prob.max_iterations = get_i64("max_iterations", 0);
    prob.seed = static_cast<uint64_t>(get_i64("seed", 0));
    PyObject* hnw = PyDict_GetItemString(spec, "has_net_weights");
    prob.has_net_weights = (hnw != nullptr && PyObject_IsTrue(hnw) == 1);
    PyObject* acs = PyDict_GetItemString(spec, "axis_centered_soft");
    prob.axis_centered_soft = (acs != nullptr && PyObject_IsTrue(acs) == 1);
    if (PyErr_Occurred()) return nullptr;

    // nets: [(member_keys_seq, weight_scaled_int, is_signal_bool), ...]
    if (PyObject* nets_obj = PyDict_GetItemString(spec, "nets")) {
        PyRef nseq = PyRef::adopt(PySequence_Fast(nets_obj, "nets must be sequence"));
        if (!nseq) return nullptr;
        const Py_ssize_t nn = PySequence_Fast_GET_SIZE(nseq.get());
        PyObject** nitems = PySequence_Fast_ITEMS(nseq.get());
        for (Py_ssize_t i = 0; i < nn; ++i) {
            PyObject* mobj = nullptr;
            long long wscaled = 1;
            int is_sig = 0;
            if (!PyArg_ParseTuple(nitems[i], "OLp", &mobj, &wscaled, &is_sig)) return nullptr;
            PyRef mseq = PyRef::adopt(PySequence_Fast(mobj, "net members must be sequence"));
            if (!mseq) return nullptr;
            EnvelopeNet en{};
            en.weight_scaled = static_cast<int64_t>(wscaled);
            en.is_signal = (is_sig != 0);
            const Py_ssize_t nm = PySequence_Fast_GET_SIZE(mseq.get());
            PyObject** mitems = PySequence_Fast_ITEMS(mseq.get());
            for (Py_ssize_t j = 0; j < nm; ++j) {
                const char* mk = PyUnicode_AsUTF8(mitems[j]);
                if (!mk) return nullptr;
                auto it = key_to_idx.find(mk);
                if (it != key_to_idx.end()) en.members.push_back(it->second);
            }
            if (en.members.size() >= 2) prob.nets.push_back(std::move(en));
        }
    }

    // symmetry_pairs: [(a, b), ...]
    if (PyObject* sp_obj = PyDict_GetItemString(spec, "symmetry_pairs")) {
        PyRef sseq = PyRef::adopt(PySequence_Fast(sp_obj, "symmetry_pairs must be sequence"));
        if (!sseq) return nullptr;
        const Py_ssize_t ns = PySequence_Fast_GET_SIZE(sseq.get());
        PyObject** sitems = PySequence_Fast_ITEMS(sseq.get());
        for (Py_ssize_t i = 0; i < ns; ++i) {
            const char* a = nullptr;
            const char* b = nullptr;
            if (!PyArg_ParseTuple(sitems[i], "ss", &a, &b)) return nullptr;
            auto ia = key_to_idx.find(a);
            auto ib = key_to_idx.find(b);
            if (ia != key_to_idx.end() && ib != key_to_idx.end() && ia->second != ib->second) {
                prob.symmetry_pairs.push_back({ia->second, ib->second});
            }
        }
    }

    // axis_groups: [(key, offset_rounded_int, is_primary_bool), ...]
    if (PyObject* ag_obj = PyDict_GetItemString(spec, "axis_groups")) {
        PyRef aseq = PyRef::adopt(PySequence_Fast(ag_obj, "axis_groups must be sequence"));
        if (!aseq) return nullptr;
        const Py_ssize_t na = PySequence_Fast_GET_SIZE(aseq.get());
        PyObject** aitems = PySequence_Fast_ITEMS(aseq.get());
        for (Py_ssize_t i = 0; i < na; ++i) {
            const char* k = nullptr;
            long long off = 0;
            int prim = 0;
            if (!PyArg_ParseTuple(aitems[i], "sLp", &k, &off, &prim)) return nullptr;
            auto it = key_to_idx.find(k);
            if (it != key_to_idx.end()) {
                prob.axis_groups.push_back(EnvelopeAxisGroup{it->second, static_cast<int64_t>(off), prim != 0});
            }
        }
    }

    // clusters: [(member_keys_seq, pad_int), ...]
    if (PyObject* cl_obj = PyDict_GetItemString(spec, "clusters")) {
        PyRef cseq = PyRef::adopt(PySequence_Fast(cl_obj, "clusters must be sequence"));
        if (!cseq) return nullptr;
        const Py_ssize_t nc = PySequence_Fast_GET_SIZE(cseq.get());
        PyObject** citems = PySequence_Fast_ITEMS(cseq.get());
        for (Py_ssize_t i = 0; i < nc; ++i) {
            PyObject* mobj = nullptr;
            long long pad = 0;
            if (!PyArg_ParseTuple(citems[i], "OL", &mobj, &pad)) return nullptr;
            PyRef mseq = PyRef::adopt(PySequence_Fast(mobj, "cluster members must be sequence"));
            if (!mseq) return nullptr;
            EnvelopeCluster ec{};
            ec.pad = static_cast<int64_t>(pad);
            const Py_ssize_t nm = PySequence_Fast_GET_SIZE(mseq.get());
            PyObject** mitems = PySequence_Fast_ITEMS(mseq.get());
            for (Py_ssize_t j = 0; j < nm; ++j) {
                const char* mk = PyUnicode_AsUTF8(mitems[j]);
                if (!mk) return nullptr;
                auto it = key_to_idx.find(mk);
                if (it != key_to_idx.end()) ec.members.push_back(it->second);
            }
            if (!ec.members.empty()) prob.clusters.push_back(std::move(ec));
        }
    }

    // band_of_group: dict {key: int} (optional)
    if (PyObject* bg_obj = PyDict_GetItemString(spec, "band_of_group")) {
        if (bg_obj != Py_None && PyDict_Check(bg_obj) && PyDict_Size(bg_obj) > 0) {
            prob.band_of_group.assign(prob.groups.size(), std::numeric_limits<int32_t>::min());
            PyObject* k = nullptr;
            PyObject* v = nullptr;
            Py_ssize_t pos = 0;
            while (PyDict_Next(bg_obj, &pos, &k, &v)) {
                const char* ks = PyUnicode_AsUTF8(k);
                if (!ks) return nullptr;
                const long bv = PyLong_AsLong(v);
                if (bv == -1 && PyErr_Occurred()) return nullptr;
                auto it = key_to_idx.find(ks);
                if (it != key_to_idx.end()) prob.band_of_group[it->second] = static_cast<int32_t>(bv);
            }
        }
    }

    // orders: [(a, b, is_above_bool), ...]
    if (PyObject* ord_obj = PyDict_GetItemString(spec, "orders")) {
        PyRef oseq = PyRef::adopt(PySequence_Fast(ord_obj, "orders must be sequence"));
        if (!oseq) return nullptr;
        const Py_ssize_t no = PySequence_Fast_GET_SIZE(oseq.get());
        PyObject** oitems = PySequence_Fast_ITEMS(oseq.get());
        for (Py_ssize_t i = 0; i < no; ++i) {
            const char* a = nullptr;
            const char* b = nullptr;
            int is_above = 0;
            if (!PyArg_ParseTuple(oitems[i], "ssp", &a, &b, &is_above)) return nullptr;
            auto ia = key_to_idx.find(a);
            auto ib = key_to_idx.find(b);
            if (ia != key_to_idx.end() && ib != key_to_idx.end() && ia->second != ib->second) {
                prob.orders.push_back(EnvelopeOrder{ia->second, ib->second, is_above != 0});
            }
        }
    }

    // abut_pairs: [(a, b), ...]
    if (PyObject* ab_obj = PyDict_GetItemString(spec, "abut_pairs")) {
        PyRef aseq = PyRef::adopt(PySequence_Fast(ab_obj, "abut_pairs must be sequence"));
        if (!aseq) return nullptr;
        const Py_ssize_t na = PySequence_Fast_GET_SIZE(aseq.get());
        PyObject** aitems = PySequence_Fast_ITEMS(aseq.get());
        for (Py_ssize_t i = 0; i < na; ++i) {
            const char* a = nullptr;
            const char* b = nullptr;
            if (!PyArg_ParseTuple(aitems[i], "ss", &a, &b)) return nullptr;
            auto ia = key_to_idx.find(a);
            auto ib = key_to_idx.find(b);
            if (ia != key_to_idx.end() && ib != key_to_idx.end() && ia->second != ib->second) {
                prob.abut_pairs.push_back({ia->second, ib->second});
            }
        }
    }

    // rail_affinities: [(key, is_top_bool), ...]
    if (PyObject* ra_obj = PyDict_GetItemString(spec, "rail_affinities")) {
        PyRef rseq = PyRef::adopt(PySequence_Fast(ra_obj, "rail_affinities must be sequence"));
        if (!rseq) return nullptr;
        const Py_ssize_t nr = PySequence_Fast_GET_SIZE(rseq.get());
        PyObject** ritems = PySequence_Fast_ITEMS(rseq.get());
        for (Py_ssize_t i = 0; i < nr; ++i) {
            const char* k = nullptr;
            int is_top = 0;
            if (!PyArg_ParseTuple(ritems[i], "sp", &k, &is_top)) return nullptr;
            auto it = key_to_idx.find(k);
            if (it != key_to_idx.end()) {
                prob.rail_affinities.push_back(EnvelopeRailAffinity{it->second, is_top != 0});
            }
        }
    }

    const auto sol = clay_pnr::place::solve_envelope_min(prob);

    const char* status_str = "UNKNOWN";
    if (sol.status == EnvelopeStatus::Optimal) status_str = "OPTIMAL";
    else if (sol.status == EnvelopeStatus::Feasible) status_str = "FEASIBLE";
    else if (sol.status == EnvelopeStatus::Infeasible) status_str = "INFEASIBLE";

    PyRef xs_dict = PyRef::adopt(PyDict_New());
    PyRef ys_dict = PyRef::adopt(PyDict_New());
    if (!xs_dict || !ys_dict) return nullptr;
    if (sol.xs.size() == prob.groups.size() && sol.ys.size() == prob.groups.size()) {
        for (size_t i = 0; i < prob.groups.size(); ++i) {
            PyRef xv = PyRef::adopt(PyLong_FromLongLong(sol.xs[i]));
            PyRef yv = PyRef::adopt(PyLong_FromLongLong(sol.ys[i]));
            if (!xv || !yv) return nullptr;
            if (PyDict_SetItemString(xs_dict.get(), prob.groups[i].key.c_str(), xv.get()) < 0) return nullptr;
            if (PyDict_SetItemString(ys_dict.get(), prob.groups[i].key.c_str(), yv.get()) < 0) return nullptr;
        }
    }

    return Py_BuildValue(
        "{s:s,s:O,s:O,s:L,s:L,s:L,s:L,s:K,s:s}",
        "status", status_str,
        "xs", xs_dict.get(),
        "ys", ys_dict.get(),
        "W", static_cast<long long>(sol.W),
        "H", static_cast<long long>(sol.H),
        "objective", static_cast<long long>(sol.objective),
        "lower_bound", static_cast<long long>(sol.lower_bound),
        "evaluations", static_cast<unsigned long long>(sol.evaluations),
        "detail", sol.detail.c_str());
}

// ---------------------------------------------------------------------------
// Phase 2: LayoutDB Per-Net Merge, Multi-Layer Connectivity & MOS PCell
// ---------------------------------------------------------------------------

PyObject* py_merge_net_boxes(PyObject* /*self*/, PyObject* args) {
    PyObject* items_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &items_obj)) return nullptr;
    PyRef seq = PyRef::adopt(PySequence_Fast(items_obj, "expected [(net, layer, dt, boxes), ...]"));
    if (!seq) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    std::vector<NetLayerBoxes> input;
    input.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* net = nullptr;
        int layer = 0, dt = 0;
        PyObject* b_obj = nullptr;
        if (!PyArg_ParseTuple(items[i], "siiO", &net, &layer, &dt, &b_obj)) return nullptr;
        std::vector<Box64> boxes;
        if (!parse_rect_list(b_obj, boxes)) return nullptr;
        input.push_back(NetLayerBoxes{net, LayerKey{static_cast<int16_t>(layer), static_cast<int16_t>(dt)}, std::move(boxes)});
    }
    const auto merged = clay_pnr::layout::merge_net_boxes_exact(input);
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(merged.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < merged.size(); ++i) {
        PyRef pts = PyRef::adopt(make_point_list(merged[i].vertices));
        if (!pts) return nullptr;
        PyRef tup = PyRef::adopt(Py_BuildValue("(iiO)", static_cast<int>(merged[i].key.layer),
                                               static_cast<int>(merged[i].key.datatype), pts.get()));
        if (!tup) return nullptr;
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    return out.release();
}

PyObject* py_extract_connectivity(PyObject* /*self*/, PyObject* args) {
    PyObject* shapes_obj = nullptr;
    PyObject* cond_obj = nullptr;
    PyObject* conn_obj = nullptr;
    PyObject* carve_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|O", &shapes_obj, &cond_obj, &conn_obj, &carve_obj)) return nullptr;

    std::vector<LayerBox> shapes;
    {
        PyRef s = PyRef::adopt(PySequence_Fast(shapes_obj, "shapes must be [(layer, dt, x0, y0, x1, y1), ...]"));
        if (!s) return nullptr;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        shapes.reserve(static_cast<size_t>(k));
        for (Py_ssize_t i = 0; i < k; ++i) {
            int l = 0, dt = 0;
            long long x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            if (!PyArg_ParseTuple(items[i], "iiLLLL", &l, &dt, &x0, &y0, &x1, &y1)) return nullptr;
            shapes.push_back(LayerBox{LayerKey{static_cast<int16_t>(l), static_cast<int16_t>(dt)},
                                      Box64{static_cast<int64_t>(x0), static_cast<int64_t>(y0),
                                            static_cast<int64_t>(x1), static_cast<int64_t>(y1)}});
        }
    }
    std::vector<LayerKey> conductors;
    {
        PyRef s = PyRef::adopt(PySequence_Fast(cond_obj, "conductors must be [(layer, dt), ...]"));
        if (!s) return nullptr;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        for (Py_ssize_t i = 0; i < k; ++i) {
            int l = 0, dt = 0;
            if (!PyArg_ParseTuple(items[i], "ii", &l, &dt)) return nullptr;
            conductors.push_back(LayerKey{static_cast<int16_t>(l), static_cast<int16_t>(dt)});
        }
    }
    std::vector<StackConnection> conns;
    {
        PyRef s = PyRef::adopt(PySequence_Fast(conn_obj, "connections must be [((l,d), (l,d), (l,d)), ...]"));
        if (!s) return nullptr;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        for (Py_ssize_t i = 0; i < k; ++i) {
            int l1 = 0, d1 = 0, l2 = 0, d2 = 0, l3 = 0, d3 = 0;
            if (!PyArg_ParseTuple(items[i], "(ii)(ii)(ii)", &l1, &d1, &l2, &d2, &l3, &d3)) return nullptr;
            conns.push_back(StackConnection{
                LayerKey{static_cast<int16_t>(l1), static_cast<int16_t>(d1)},
                LayerKey{static_cast<int16_t>(l2), static_cast<int16_t>(d2)},
                LayerKey{static_cast<int16_t>(l3), static_cast<int16_t>(d3)}});
        }
    }
    std::vector<MimCarveRule> carves;
    if (carve_obj != Py_None) {
        PyRef s = PyRef::adopt(PySequence_Fast(carve_obj, "mim_carves must be sequence"));
        if (!s) return nullptr;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(s.get());
        PyObject** items = PySequence_Fast_ITEMS(s.get());
        for (Py_ssize_t i = 0; i < k; ++i) {
            int bl = 0, bd = 0, rl = 0, rd = 0, ml = 0, md = 0, mo = 0;
            long long ov = 0;
            if (!PyArg_ParseTuple(items[i], "(ii)(ii)(ii)Lp", &bl, &bd, &rl, &rd, &ml, &md, &ov, &mo)) return nullptr;
            carves.push_back(MimCarveRule{
                LayerKey{static_cast<int16_t>(bl), static_cast<int16_t>(bd)},
                LayerKey{static_cast<int16_t>(rl), static_cast<int16_t>(rd)},
                LayerKey{static_cast<int16_t>(ml), static_cast<int16_t>(md)},
                static_cast<int64_t>(ov), mo != 0});
        }
    }

    const auto comps = clay_pnr::layout::extract_stack_connectivity(shapes, conductors, conns, carves);
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(comps.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < comps.size(); ++i) {
        const auto& c = comps[i].shapes;
        PyRef clist = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(c.size())));
        if (!clist) return nullptr;
        for (size_t j = 0; j < c.size(); ++j) {
            PyRef row = PyRef::adopt(Py_BuildValue(
                "(iiLLLL)", static_cast<int>(c[j].key.layer), static_cast<int>(c[j].key.datatype),
                static_cast<long long>(c[j].box.x0), static_cast<long long>(c[j].box.y0),
                static_cast<long long>(c[j].box.x1), static_cast<long long>(c[j].box.y1)));
            if (!row) return nullptr;
            PyList_SET_ITEM(clist.get(), static_cast<Py_ssize_t>(j), row.release());
        }
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), clist.release());
    }
    return out.release();
}

PyObject* py_generate_mos_pcell(PyObject* /*self*/, PyObject* args) {
    const char* name = nullptr;
    int is_pmos = 0;
    long long w_nm = 1000, l_nm = 280;
    int fingers = 2, dl = 0, dr = 0;
    const char* pdk_id = "gf180mcu";
    if (!PyArg_ParseTuple(args, "spLLi|iis", &name, &is_pmos, &w_nm, &l_nm, &fingers, &dl, &dr, &pdk_id)) {
        return nullptr;
    }
    const auto rules = (std::string(pdk_id) == "sky130") ? clay_pnr::pcell::sky130_mos_rules()
                                                         : clay_pnr::pcell::gf180mcu_mos_rules();
    clay_pnr::pcell::MosSpec spec{name, is_pmos != 0, static_cast<int64_t>(w_nm),
                                  static_cast<int64_t>(l_nm), fingers, dl, dr};
    const auto res = clay_pnr::pcell::generate_mos_pcell(spec, rules);

    PyRef polys = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.cell.polygons.size())));
    if (!polys) return nullptr;
    for (size_t i = 0; i < res.cell.polygons.size(); ++i) {
        const auto& lp = res.cell.polygons[i];
        PyRef pts = PyRef::adopt(make_point_list(lp.vertices));
        if (!pts) return nullptr;
        PyRef item = PyRef::adopt(Py_BuildValue("(iiO)", static_cast<int>(lp.key.layer),
                                                static_cast<int>(lp.key.datatype), pts.get()));
        if (!item) return nullptr;
        PyList_SET_ITEM(polys.get(), static_cast<Py_ssize_t>(i), item.release());
    }
    PyRef bb = PyRef::adopt(make_box64_tuple(res.terminals.bbox));
    PyRef abb = PyRef::adopt(make_box64_tuple(res.active_bbox));
    PyRef srects = PyRef::adopt(make_box_list(res.terminals.source_rects));
    PyRef drects = PyRef::adopt(make_box_list(res.terminals.drain_rects));
    PyRef grects = PyRef::adopt(make_box_list(res.terminals.gate_rects));
    if (!bb || !abb || !srects || !drects || !grects) return nullptr;
    return Py_BuildValue("{s:O,s:O,s:O,s:L,s:O,s:O,s:O}",
                         "polygons", polys.get(),
                         "bbox", bb.get(),
                         "active_bbox", abb.get(),
                         "gate_pitch_nm", static_cast<long long>(res.gate_pitch_nm),
                         "source_rects", srects.get(),
                         "drain_rects", drects.get(),
                         "gate_rects", grects.get());
}

// ---------------------------------------------------------------------------
// Phase 3B & 4A: Banded 2D Area Placer, .clay DSL Parser & Standalone Synth
// ---------------------------------------------------------------------------

PyObject* py_solve_banded_area(PyObject* /*self*/, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {
        "blocks", "gap_nm", "grid_nm", "w_cap_nm", "h_bound_nm",
        "bottom_edge_keepout_nm", "top_edge_keepout_nm",
        "strict_width_cap", "exact_outline_width", "enforce_band_order",
        "objective_area", "dense_2d",
        "aspect_obligation", "aspect_lower", "aspect_upper", "aspect_scale",
        "order_pairs", "vertical_order_pairs", "pair_clearances", "pull_pairs",
        "max_evals", "seed", nullptr
    };
    PyObject* blocks_obj = nullptr;
    long long gap_nm = 0, grid_nm = 1, w_cap_nm = 0, h_bound_nm = 0;
    long long bottom_keepout = 0, top_keepout = 0;
    int strict_w = 0, exact_w = 0, enforce_band = 1, obj_area = 1, dense_2d = 0;
    int aspect_obl = 0;
    long long aspect_lo = 0, aspect_hi = 0, aspect_sc = 1000000;
    PyObject *ord_obj = Py_None, *vord_obj = Py_None, *pclr_obj = Py_None, *pull_obj = Py_None;
    int max_evals = 2500;
    unsigned long long seed = 0x434C41593342ULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|LLLLLLppppppLLLOOOOiK", const_cast<char**>(kwlist),
            &blocks_obj, &gap_nm, &grid_nm, &w_cap_nm, &h_bound_nm,
            &bottom_keepout, &top_keepout,
            &strict_w, &exact_w, &enforce_band, &obj_area, &dense_2d,
            &aspect_obl, &aspect_lo, &aspect_hi, &aspect_sc,
            &ord_obj, &vord_obj, &pclr_obj, &pull_obj,
            &max_evals, &seed)) {
        return nullptr;
    }

    clay_pnr::place::BandedProblem prob{};
    prob.gap_nm = static_cast<int64_t>(gap_nm);
    prob.grid_nm = static_cast<int64_t>(grid_nm);
    prob.w_cap_nm = static_cast<int64_t>(w_cap_nm);
    prob.h_bound_nm = static_cast<int64_t>(h_bound_nm);
    prob.bottom_edge_keepout_nm = static_cast<int64_t>(bottom_keepout);
    prob.top_edge_keepout_nm = static_cast<int64_t>(top_keepout);
    prob.strict_width_cap = (strict_w != 0);
    prob.exact_outline_width = (exact_w != 0);
    prob.enforce_band_order = (enforce_band != 0);
    prob.objective_area = (obj_area != 0);
    prob.dense_2d = (dense_2d != 0);
    prob.aspect_obligation = (aspect_obl != 0);
    prob.aspect_lower = static_cast<int64_t>(aspect_lo);
    prob.aspect_upper = static_cast<int64_t>(aspect_hi);
    prob.aspect_scale = static_cast<int64_t>(aspect_sc);
    prob.max_evals = max_evals;
    prob.seed = static_cast<uint64_t>(seed);

    {
        PyRef bseq = PyRef::adopt(PySequence_Fast(blocks_obj, "blocks must be sequence"));
        if (!bseq) return nullptr;
        const Py_ssize_t nb = PySequence_Fast_GET_SIZE(bseq.get());
        PyObject** items = PySequence_Fast_ITEMS(bseq.get());
        prob.blocks.reserve(static_cast<size_t>(nb));
        for (Py_ssize_t i = 0; i < nb; ++i) {
            const char* name = nullptr;
            long long w = 0, h = 0, ymin_k = 0;
            int band = 0, lat = 0, le = 0, re = 0, lp = 0, rp = 0, tk = 0;
            if (!PyArg_ParseTuple(items[i], "sLLi|pppppLp",
                                  &name, &w, &h, &band, &lat, &le, &re, &lp, &rp, &ymin_k, &tk)) {
                return nullptr;
            }
            prob.blocks.push_back(clay_pnr::place::BandedBlock{
                name, static_cast<int64_t>(w), static_cast<int64_t>(h), band,
                lat != 0, le != 0, re != 0, lp != 0, rp != 0,
                static_cast<int64_t>(ymin_k), tk != 0
            });
        }
    }

    auto parse_pairs = [](PyObject* obj, std::vector<std::pair<int32_t, int32_t>>& dst) -> bool {
        if (!obj || obj == Py_None) return true;
        PyRef seq = PyRef::adopt(PySequence_Fast(obj, "expected sequence of (i, j) pairs"));
        if (!seq) return false;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(seq.get());
        PyObject** items = PySequence_Fast_ITEMS(seq.get());
        dst.reserve(static_cast<size_t>(k));
        for (Py_ssize_t i = 0; i < k; ++i) {
            int a = 0, b = 0;
            if (!PyArg_ParseTuple(items[i], "ii", &a, &b)) return false;
            dst.emplace_back(a, b);
        }
        return true;
    };
    if (!parse_pairs(ord_obj, prob.order_pairs) ||
        !parse_pairs(vord_obj, prob.vertical_order_pairs) ||
        !parse_pairs(pull_obj, prob.pull_pairs)) {
        return nullptr;
    }
    if (pclr_obj && pclr_obj != Py_None) {
        PyRef seq = PyRef::adopt(PySequence_Fast(pclr_obj, "pair_clearances must be [(a, b, gap_nm), ...]"));
        if (!seq) return nullptr;
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(seq.get());
        PyObject** items = PySequence_Fast_ITEMS(seq.get());
        prob.pair_clearances.reserve(static_cast<size_t>(k));
        for (Py_ssize_t i = 0; i < k; ++i) {
            int a = 0, b = 0;
            long long gnm = 0;
            if (!PyArg_ParseTuple(items[i], "iiL", &a, &b, &gnm)) return nullptr;
            prob.pair_clearances.push_back(clay_pnr::place::BandedPairClearance{a, b, static_cast<int64_t>(gnm)});
        }
    }

    clay_pnr::place::BandedSolution sol;
    Py_BEGIN_ALLOW_THREADS
    sol = clay_pnr::place::solve_banded_area(prob);
    Py_END_ALLOW_THREADS

    PyRef xs = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(sol.xs.size())));
    PyRef ys = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(sol.ys.size())));
    if (!xs || !ys) return nullptr;
    for (size_t i = 0; i < sol.xs.size(); ++i) {
        PyList_SET_ITEM(xs.get(), static_cast<Py_ssize_t>(i), PyLong_FromLongLong(sol.xs[i]));
        PyList_SET_ITEM(ys.get(), static_cast<Py_ssize_t>(i), PyLong_FromLongLong(sol.ys[i]));
    }
    const char* st_str = "INFEASIBLE";
    if (sol.status == clay_pnr::place::BandedStatus::Optimal) st_str = "OPTIMAL";
    else if (sol.status == clay_pnr::place::BandedStatus::Feasible) st_str = "FEASIBLE";
    else if (sol.status == clay_pnr::place::BandedStatus::InvalidInput) st_str = "INVALID";

    return Py_BuildValue(
        "{s:s,s:O,s:O,s:L,s:L,s:L,s:L,s:L,s:K,s:s}",
        "status", st_str,
        "xs", xs.get(),
        "ys", ys.get(),
        "W", static_cast<long long>(sol.W),
        "H", static_cast<long long>(sol.H),
        "primary", static_cast<long long>(sol.primary),
        "soft_cost", static_cast<long long>(sol.soft_cost),
        "lower_bound", static_cast<long long>(sol.lower_bound),
        "evaluations", static_cast<unsigned long long>(sol.evaluations),
        "detail", sol.detail.c_str());
}

PyObject* py_parse_clay_ir(PyObject* /*self*/, PyObject* args) {
    const char* src_cstr = nullptr;
    Py_ssize_t src_len = 0;
    if (!PyArg_ParseTuple(args, "s#", &src_cstr, &src_len)) return nullptr;
    const std::string src(src_cstr, static_cast<size_t>(src_len));
    clay_pnr::ir::ParseResult pr;
    Py_BEGIN_ALLOW_THREADS
    pr = clay_pnr::ir::parse_clay_source(src);
    Py_END_ALLOW_THREADS

    if (!pr.ok) {
        PyErr_Format(PyExc_ValueError, "Clay C++20 parse error (%d:%d): %s",
                     pr.line, pr.col, pr.error.c_str());
        return nullptr;
    }

    PyRef py_cells = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pr.file.cells.size())));
    if (!py_cells) return nullptr;
    for (size_t ci = 0; ci < pr.file.cells.size(); ++ci) {
        const auto& c = pr.file.cells[ci];
        PyRef py_ports = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(c.ports.size())));
        PyRef py_devs = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(c.devices.size())));
        PyRef py_fgs = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(c.functional_groups.size())));
        PyRef py_sizing = PyRef::adopt(PyDict_New());
        if (!py_ports || !py_devs || !py_fgs || !py_sizing) return nullptr;

        for (size_t i = 0; i < c.ports.size(); ++i) {
            PyObject* p = Py_BuildValue("(ss)", c.ports[i].name.c_str(), c.ports[i].role.c_str());
            if (!p) return nullptr;
            PyList_SET_ITEM(py_ports.get(), static_cast<Py_ssize_t>(i), p);
        }
        for (size_t i = 0; i < c.devices.size(); ++i) {
            const auto& d = c.devices[i];
            PyRef pins_d = PyRef::adopt(PyDict_New());
            PyRef params_d = PyRef::adopt(PyDict_New());
            PyRef strp_d = PyRef::adopt(PyDict_New());
            if (!pins_d || !params_d || !strp_d) return nullptr;
            for (const auto& [k, v] : d.pins) {
                PyRef pv = PyRef::adopt(PyUnicode_FromString(v.c_str()));
                if (!pv || PyDict_SetItemString(pins_d.get(), k.c_str(), pv.get()) < 0) return nullptr;
            }
            for (const auto& [k, v] : d.params) {
                PyRef pv = PyRef::adopt(PyFloat_FromDouble(v));
                if (!pv || PyDict_SetItemString(params_d.get(), k.c_str(), pv.get()) < 0) return nullptr;
            }
            for (const auto& [k, v] : d.str_params) {
                PyRef pv = PyRef::adopt(PyUnicode_FromString(v.c_str()));
                if (!pv || PyDict_SetItemString(strp_d.get(), k.c_str(), pv.get()) < 0) return nullptr;
            }
            PyObject* dev_d = Py_BuildValue(
                "{s:s,s:s,s:s,s:O,s:O,s:O,s:(ii)}",
                "decl_kind", d.decl_kind.c_str(),
                "name", d.name.c_str(),
                "device_type", d.device_type.c_str(),
                "pins", pins_d.get(),
                "params", params_d.get(),
                "str_params", strp_d.get(),
                "span", static_cast<int>(d.span.line), static_cast<int>(d.span.col));
            if (!dev_d) return nullptr;
            PyList_SET_ITEM(py_devs.get(), static_cast<Py_ssize_t>(i), dev_d);
        }
        for (size_t i = 0; i < c.functional_groups.size(); ++i) {
            const auto& fg = c.functional_groups[i];
            PyRef dlist = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(fg.devices.size())));
            if (!dlist) return nullptr;
            for (size_t j = 0; j < fg.devices.size(); ++j) {
                PyList_SET_ITEM(dlist.get(), static_cast<Py_ssize_t>(j),
                                PyUnicode_FromString(fg.devices[j].c_str()));
            }
            PyObject* fgd = Py_BuildValue("{s:s,s:s,s:O}",
                                          "kind", fg.kind.c_str(),
                                          "name", fg.name.c_str(),
                                          "devices", dlist.get());
            if (!fgd) return nullptr;
            PyList_SET_ITEM(py_fgs.get(), static_cast<Py_ssize_t>(i), fgd);
        }
        for (const auto& [k, v] : c.sizing) {
            PyRef fv = PyRef::adopt(PyFloat_FromDouble(v));
            if (!fv || PyDict_SetItemString(py_sizing.get(), k.c_str(), fv.get()) < 0) return nullptr;
        }
        PyObject* cd = Py_BuildValue(
            "{s:s,s:(ii),s:O,s:O,s:O,s:O}",
            "name", c.name.c_str(),
            "span", static_cast<int>(c.span.line), static_cast<int>(c.span.col),
            "ports", py_ports.get(),
            "devices", py_devs.get(),
            "functional_groups", py_fgs.get(),
            "sizing", py_sizing.get());
        if (!cd) return nullptr;
        PyList_SET_ITEM(py_cells.get(), static_cast<Py_ssize_t>(ci), cd);
    }

    PyRef py_intents = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pr.file.layout_intents.size())));
    if (!py_intents) return nullptr;
    for (size_t i = 0; i < pr.file.layout_intents.size(); ++i) {
        const auto& li = pr.file.layout_intents[i];
        PyRef ents = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(li.entries.size())));
        if (!ents) return nullptr;
        for (size_t j = 0; j < li.entries.size(); ++j) {
            const auto& e = li.entries[j];
            PyRef pos_l = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(e.positional.size())));
            if (!pos_l) return nullptr;
            for (size_t k = 0; k < e.positional.size(); ++k) {
                PyList_SET_ITEM(pos_l.get(), static_cast<Py_ssize_t>(k),
                                PyUnicode_FromString(e.positional[k].c_str()));
            }
            PyObject* ed = Py_BuildValue("{s:s,s:O,s:(ii)}",
                                         "kind", e.kind.c_str(),
                                         "positional", pos_l.get(),
                                         "span", static_cast<int>(e.span.line), static_cast<int>(e.span.col));
            if (!ed) return nullptr;
            PyList_SET_ITEM(ents.get(), static_cast<Py_ssize_t>(j), ed);
        }
        PyObject* lid = Py_BuildValue("{s:s,s:O,s:(ii)}",
                                      "cell_name", li.cell_name.c_str(),
                                      "entries", ents.get(),
                                      "span", static_cast<int>(li.span.line), static_cast<int>(li.span.col));
        if (!lid) return nullptr;
        PyList_SET_ITEM(py_intents.get(), static_cast<Py_ssize_t>(i), lid);
    }

    PyRef py_annots = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pr.file.annotations.size())));
    if (!py_annots) return nullptr;
    for (size_t i = 0; i < pr.file.annotations.size(); ++i) {
        const auto& ab = pr.file.annotations[i];
        PyObject* abd = Py_BuildValue("{s:s,s:n,s:(ii)}",
                                      "cell_name", ab.cell_name.c_str(),
                                      "count", static_cast<Py_ssize_t>(ab.entries.size()),
                                      "span", static_cast<int>(ab.span.line), static_cast<int>(ab.span.col));
        if (!abd) return nullptr;
        PyList_SET_ITEM(py_annots.get(), static_cast<Py_ssize_t>(i), abd);
    }

    auto make_str_list = [](const std::vector<std::string>& items) -> PyObject* {
        PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(items.size())));
        if (!lst) return nullptr;
        for (size_t i = 0; i < items.size(); ++i) {
            PyObject* s = PyUnicode_FromStringAndSize(items[i].data(), static_cast<Py_ssize_t>(items[i].size()));
            if (!s) return nullptr;
            PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), s);
        }
        return lst.release();
    };
    PyRef py_imports = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pr.file.import_decls.size())));
    if (!py_imports) return nullptr;
    for (size_t i = 0; i < pr.file.import_decls.size(); ++i) {
        const auto& imp = pr.file.import_decls[i];
        PyObject* row = Py_BuildValue("(ss(ii))", imp.path.c_str(), imp.alias.c_str(),
                                      static_cast<int>(imp.span.line), static_cast<int>(imp.span.col));
        if (!row) return nullptr;
        PyList_SET_ITEM(py_imports.get(), static_cast<Py_ssize_t>(i), row);
    }
    PyRef py_aliases = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pr.file.cell_aliases.size())));
    if (!py_aliases) return nullptr;
    for (size_t i = 0; i < pr.file.cell_aliases.size(); ++i) {
        const auto& al = pr.file.cell_aliases[i];
        PyObject* row = Py_BuildValue("(ss(ii))", al.name.c_str(), al.target.c_str(),
                                      static_cast<int>(al.span.line), static_cast<int>(al.span.col));
        if (!row) return nullptr;
        PyList_SET_ITEM(py_aliases.get(), static_cast<Py_ssize_t>(i), row);
    }
    PyRef py_policy_names = PyRef::adopt(make_str_list(pr.file.policy_names));
    PyRef py_quality_names = PyRef::adopt(make_str_list(pr.file.quality_names));
    PyRef py_verify_names = PyRef::adopt(make_str_list(pr.file.verify_names));
    PyRef py_fn_names = PyRef::adopt(make_str_list(pr.file.fn_names));
    if (!py_policy_names || !py_quality_names || !py_verify_names || !py_fn_names) return nullptr;

    return Py_BuildValue(
        "{s:s,s:s,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:n,s:n,s:n,s:n}",
        "package", pr.file.package_name.c_str(),
        "pdk", pr.file.pdk.c_str(),
        "imports", py_imports.get(),
        "cell_aliases", py_aliases.get(),
        "cells", py_cells.get(),
        "layout_intents", py_intents.get(),
        "annotations", py_annots.get(),
        "policy_names", py_policy_names.get(),
        "quality_names", py_quality_names.get(),
        "verify_names", py_verify_names.get(),
        "fn_names", py_fn_names.get(),
        "flows", static_cast<Py_ssize_t>(pr.file.flows.size()),
        "policies", static_cast<Py_ssize_t>(pr.file.policy_count),
        "qualities", static_cast<Py_ssize_t>(pr.file.quality_count),
        "verifies", static_cast<Py_ssize_t>(pr.file.verify_count));
}

PyObject* py_synthesize_clay_to_gds(PyObject* /*self*/, PyObject* args) {
    const char* clay_path = nullptr;
    const char* out_gds = nullptr;
    const char* pdk_override = "";
    int emit_routes = 1;
    if (!PyArg_ParseTuple(args, "ss|sp", &clay_path, &out_gds, &pdk_override, &emit_routes)) {
        return nullptr;
    }
    clay_pnr::engine::SynthOptions opts{};
    opts.pdk_override = pdk_override ? pdk_override : "";
    opts.emit_routes = (emit_routes != 0);

    clay_pnr::engine::StandaloneSynthResult res;
    Py_BEGIN_ALLOW_THREADS
    res = clay_pnr::engine::synthesize_clay_file_to_gds(clay_path, out_gds, opts);
    Py_END_ALLOW_THREADS

    if (!res.ok) {
        PyErr_SetString(PyExc_RuntimeError, res.error.c_str());
        return nullptr;
    }
    PyRef bb = PyRef::adopt(make_box64_tuple(res.die_bbox_nm));
    if (!bb) return nullptr;
    return Py_BuildValue(
        "{s:s,s:s,s:n,s:n,s:n,s:n,s:O}",
        "cell", res.cell_name.c_str(),
        "pdk", res.pdk_id.c_str(),
        "devices", static_cast<Py_ssize_t>(res.device_count),
        "nets", static_cast<Py_ssize_t>(res.net_count),
        "routed_nets", static_cast<Py_ssize_t>(res.routed_net_count),
        "polygons", static_cast<Py_ssize_t>(res.polygon_count),
        "bbox_nm", bb.get());
}

// ---------------------------------------------------------------------------
// Phase 5: Multi-Cell GDS Writer, Net-Connectivity, Net-Wirelength,
//          Production Stamp Geometry/Decomposition, Union Cover & Row Digest
// ---------------------------------------------------------------------------

inline bool extract_double_fast(PyObject* obj, double& out) {
    if (PyFloat_CheckExact(obj)) {
        out = PyFloat_AS_DOUBLE(obj);
        return true;
    }
    if (PyLong_CheckExact(obj)) {
        out = PyLong_AsDouble(obj);
        return !(out == -1.0 && PyErr_Occurred());
    }
    out = PyFloat_AsDouble(obj);
    return !(out == -1.0 && PyErr_Occurred());
}

inline bool extract_xy_pair_double(PyObject* obj, double& x_out, double& y_out) {
    if (PyTuple_CheckExact(obj) && PyTuple_GET_SIZE(obj) >= 2) {
        return extract_double_fast(PyTuple_GET_ITEM(obj, 0), x_out) &&
               extract_double_fast(PyTuple_GET_ITEM(obj, 1), y_out);
    }
    if (PyList_CheckExact(obj) && PyList_GET_SIZE(obj) >= 2) {
        return extract_double_fast(PyList_GET_ITEM(obj, 0), x_out) &&
               extract_double_fast(PyList_GET_ITEM(obj, 1), y_out);
    }
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "expected 2-element coordinate pair"));
    if (!seq) return false;
    if (PySequence_Fast_GET_SIZE(seq.get()) < 2) {
        PyErr_SetString(PyExc_ValueError, "coordinate pair requires 2 items");
        return false;
    }
    return extract_double_fast(PySequence_Fast_GET_ITEM(seq.get(), 0), x_out) &&
           extract_double_fast(PySequence_Fast_GET_ITEM(seq.get(), 1), y_out);
}

inline bool placed_coord_nm_py_exact(double local_um, double placement_um, int64_t& out) {
    if (clay_pnr::core::placed_coord_nm_try_fast(local_um, placement_um, out)) {
        return true;
    }
    const double composed_nm = (local_um + placement_um) * 1000.0;
    PyObject* builtins = PyEval_GetBuiltins();
    if (!builtins) return false;
    PyObject* round_fn = PyDict_GetItemString(builtins, "round");
    if (!round_fn) {
        PyErr_SetString(PyExc_RuntimeError, "builtins.round unavailable");
        return false;
    }
    PyRef py_c = PyRef::adopt(PyFloat_FromDouble(composed_nm));
    PyRef py_nine = PyRef::adopt(PyLong_FromLong(9));
    if (!py_c || !py_nine) return false;
    PyRef stable = PyRef::adopt(PyObject_CallFunctionObjArgs(round_fn, py_c.get(), py_nine.get(), nullptr));
    if (!stable) return false;
    const double s = PyFloat_AsDouble(stable.get());
    if (s == -1.0 && PyErr_Occurred()) return false;
    if (!std::isfinite(s)) {
        PyErr_SetString(std::isnan(s) ? PyExc_ValueError : PyExc_OverflowError,
                        "non-finite placed coordinate");
        return false;
    }
    out = static_cast<int64_t>(std::floor(s + 0.5));
    return true;
}

PyObject* py_gds_write_library(PyObject* /*self*/, PyObject* args) {
    const char* path = nullptr;
    const char* lib_name = nullptr;
    PyObject* cells_obj = nullptr;
    double user_unit = 1e-3;
    double dbu_meters = 1e-9;
    if (!PyArg_ParseTuple(args, "ssO|dd", &path, &lib_name, &cells_obj, &user_unit, &dbu_meters)) {
        return nullptr;
    }
    GdsLibrary lib;
    lib.name = lib_name;
    lib.user_unit = user_unit;
    lib.dbu_meters = dbu_meters;

    PyRef cseq = PyRef::adopt(PySequence_Fast(cells_obj, "cells must be a sequence"));
    if (!cseq) return nullptr;
    const Py_ssize_t nc = PySequence_Fast_GET_SIZE(cseq.get());
    for (Py_ssize_t i = 0; i < nc; ++i) {
        PyObject* citem = PySequence_Fast_GET_ITEM(cseq.get(), i);
        PyRef ctuple = PyRef::adopt(PySequence_Fast(citem, "cell must be (name, polygons)"));
        if (!ctuple || PySequence_Fast_GET_SIZE(ctuple.get()) < 2) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "cell must be (name, polygons)");
            return nullptr;
        }
        const char* cname = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(ctuple.get(), 0));
        if (!cname) return nullptr;
        auto& cell = lib.add_cell(cname);
        PyRef pseq = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(ctuple.get(), 1), "polygons must be sequence"));
        if (!pseq) return nullptr;
        const Py_ssize_t np = PySequence_Fast_GET_SIZE(pseq.get());
        for (Py_ssize_t j = 0; j < np; ++j) {
            PyObject* pitem = PySequence_Fast_GET_ITEM(pseq.get(), j);
            PyRef ptup = PyRef::adopt(PySequence_Fast(pitem, "poly must be (layer, dt, pts)"));
            if (!ptup || PySequence_Fast_GET_SIZE(ptup.get()) != 3) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "poly must be (layer, dt, pts)");
                return nullptr;
            }
            int64_t l = 0, d = 0;
            if (!parse_int64(PySequence_Fast_GET_ITEM(ptup.get(), 0), l) ||
                !parse_int64(PySequence_Fast_GET_ITEM(ptup.get(), 1), d)) return nullptr;
            PyRef vseq = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(ptup.get(), 2), "pts must be sequence"));
            if (!vseq) return nullptr;
            const Py_ssize_t nv = PySequence_Fast_GET_SIZE(vseq.get());
            std::vector<Point64> pts;
            pts.reserve(static_cast<size_t>(nv));
            for (Py_ssize_t k = 0; k < nv; ++k) {
                PyObject* v = PySequence_Fast_GET_ITEM(vseq.get(), k);
                int64_t x = 0, y = 0;
                if (PyTuple_CheckExact(v) && PyTuple_GET_SIZE(v) == 2) {
                    if (!parse_int64(PyTuple_GET_ITEM(v, 0), x) || !parse_int64(PyTuple_GET_ITEM(v, 1), y)) return nullptr;
                } else {
                    PyRef vs = PyRef::adopt(PySequence_Fast(v, "vertex must be (x, y)"));
                    if (!vs || PySequence_Fast_GET_SIZE(vs.get()) != 2) return nullptr;
                    if (!parse_int64(PySequence_Fast_GET_ITEM(vs.get(), 0), x) ||
                        !parse_int64(PySequence_Fast_GET_ITEM(vs.get(), 1), y)) return nullptr;
                }
                pts.push_back(Point64{x, y});
            }
            if (pts.size() >= 3) {
                cell.add_polygon(static_cast<int16_t>(l), static_cast<int16_t>(d), std::move(pts));
            }
        }
    }

    std::string err;
    bool ok = false;
    Py_BEGIN_ALLOW_THREADS
    ok = clay_pnr::gds::write_gds_file(lib, path, &err);
    Py_END_ALLOW_THREADS
    if (!ok) {
        PyErr_SetString(PyExc_RuntimeError, err.c_str());
        return nullptr;
    }
    return PyUnicode_FromString(path);
}

PyObject* py_verify_net_connectivity(PyObject* /*self*/, PyObject* args) {
    PyObject* shapes_obj = nullptr;
    PyObject* vias_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &shapes_obj, &vias_obj)) return nullptr;

    PyRef sseq = PyRef::adopt(PySequence_Fast(shapes_obj, "shapes must be a sequence"));
    PyRef vseq = PyRef::adopt(PySequence_Fast(vias_obj, "via_layers must be a sequence"));
    if (!sseq || !vseq) return nullptr;

    PyRef label_to_id = PyRef::adopt(PyDict_New());
    if (!label_to_id) return nullptr;
    std::vector<PyObject*> labels;  // borrowed refs alive in sseq

    const Py_ssize_t ns = PySequence_Fast_GET_SIZE(sseq.get());
    std::vector<clay_pnr::layout::NetShapeBox> shapes;
    shapes.reserve(static_cast<size_t>(ns));

    for (Py_ssize_t i = 0; i < ns; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(sseq.get(), i);
        PyRef itup = PyRef::adopt(PySequence_Fast(item, "shape must be (net, (layer, dt), bbox)"));
        if (!itup || PySequence_Fast_GET_SIZE(itup.get()) != 3) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "shape must be 3-tuple");
            return nullptr;
        }
        PyObject* net_label = PySequence_Fast_GET_ITEM(itup.get(), 0);
        PyObject* existing = PyDict_GetItemWithError(label_to_id.get(), net_label);
        uint32_t net_id = 0;
        if (existing) {
            net_id = static_cast<uint32_t>(PyLong_AsUnsignedLong(existing));
        } else {
            if (PyErr_Occurred()) return nullptr;
            net_id = static_cast<uint32_t>(labels.size());
            labels.push_back(net_label);
            PyRef py_id = PyRef::adopt(PyLong_FromUnsignedLong(net_id));
            if (!py_id || PyDict_SetItem(label_to_id.get(), net_label, py_id.get()) < 0) return nullptr;
        }
        PyRef ltup = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(itup.get(), 1), "layer must be (l, d)"));
        if (!ltup || PySequence_Fast_GET_SIZE(ltup.get()) != 2) return nullptr;
        int64_t l = 0, d = 0;
        if (!parse_int64(PySequence_Fast_GET_ITEM(ltup.get(), 0), l) ||
            !parse_int64(PySequence_Fast_GET_ITEM(ltup.get(), 1), d)) return nullptr;
        Box64 box{};
        if (!parse_rect_norm(PySequence_Fast_GET_ITEM(itup.get(), 2), box)) return nullptr;
        shapes.push_back(clay_pnr::layout::NetShapeBox{
            net_id, LayerKey{static_cast<int16_t>(l), static_cast<int16_t>(d)}, box});
    }

    const Py_ssize_t nv = PySequence_Fast_GET_SIZE(vseq.get());
    std::vector<LayerKey> via_layers;
    via_layers.reserve(static_cast<size_t>(nv));
    for (Py_ssize_t i = 0; i < nv; ++i) {
        PyRef vt = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(vseq.get(), i), "via layer must be (l, d)"));
        if (!vt || PySequence_Fast_GET_SIZE(vt.get()) != 2) return nullptr;
        int64_t l = 0, d = 0;
        if (!parse_int64(PySequence_Fast_GET_ITEM(vt.get(), 0), l) ||
            !parse_int64(PySequence_Fast_GET_ITEM(vt.get(), 1), d)) return nullptr;
        via_layers.push_back(LayerKey{static_cast<int16_t>(l), static_cast<int16_t>(d)});
    }

    clay_pnr::layout::NetConnectivityResult res;
    Py_BEGIN_ALLOW_THREADS
    res = clay_pnr::layout::verify_net_connectivity_exact(shapes, via_layers);
    Py_END_ALLOW_THREADS

    PyRef comp_dict = PyRef::adopt(PyDict_New());
    PyRef pairs_list = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.merged_pairs.size())));
    PyRef roots_list = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.shape_root.size())));
    if (!comp_dict || !pairs_list || !roots_list) return nullptr;

    for (const auto& [nid, root] : res.component_of) {
        PyRef py_r = PyRef::adopt(PyLong_FromUnsignedLong(root));
        if (!py_r || PyDict_SetItem(comp_dict.get(), labels[nid], py_r.get()) < 0) return nullptr;
    }
    for (size_t i = 0; i < res.merged_pairs.size(); ++i) {
        PyRef pair = PyRef::adopt(PyList_New(2));
        if (!pair) return nullptr;
        PyObject* la = labels[res.merged_pairs[i].first];
        PyObject* lb = labels[res.merged_pairs[i].second];
        Py_INCREF(la);
        Py_INCREF(lb);
        PyList_SET_ITEM(pair.get(), 0, la);
        PyList_SET_ITEM(pair.get(), 1, lb);
        PyList_SET_ITEM(pairs_list.get(), static_cast<Py_ssize_t>(i), pair.release());
    }
    if (PyList_Sort(pairs_list.get()) < 0) return nullptr;

    for (size_t i = 0; i < res.shape_root.size(); ++i) {
        PyRef py_r = PyRef::adopt(PyLong_FromUnsignedLong(res.shape_root[i]));
        if (!py_r) return nullptr;
        PyList_SET_ITEM(roots_list.get(), static_cast<Py_ssize_t>(i), py_r.release());
    }

    PyRef out = PyRef::adopt(PyDict_New());
    PyRef sc = PyRef::adopt(PyLong_FromSize_t(res.short_count));
    if (!out || !sc) return nullptr;
    if (PyDict_SetItemString(out.get(), "component_of", comp_dict.get()) < 0 ||
        PyDict_SetItemString(out.get(), "merged_pairs", pairs_list.get()) < 0 ||
        PyDict_SetItemString(out.get(), "distinct", res.distinct ? Py_True : Py_False) < 0 ||
        PyDict_SetItemString(out.get(), "shape_root", roots_list.get()) < 0 ||
        PyDict_SetItemString(out.get(), "short_count", sc.get()) < 0) {
        return nullptr;
    }
    return out.release();
}

PyObject* py_compute_net_wirelength(PyObject* /*self*/, PyObject* args) {
    PyObject* nets_obj = nullptr;
    PyObject* bboxes_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &nets_obj, &bboxes_obj)) return nullptr;

    PyRef nseq = PyRef::adopt(PySequence_Fast(nets_obj, "nets must be a sequence"));
    PyRef bseq = PyRef::adopt(PySequence_Fast(bboxes_obj, "bboxes must be a sequence"));
    if (!nseq || !bseq) return nullptr;

    const Py_ssize_t nn = PySequence_Fast_GET_SIZE(nseq.get());
    std::vector<std::vector<Point64>> nets;
    nets.reserve(static_cast<size_t>(nn));
    for (Py_ssize_t i = 0; i < nn; ++i) {
        PyRef pseq = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(nseq.get(), i), "net must be sequence of points"));
        if (!pseq) return nullptr;
        const Py_ssize_t np = PySequence_Fast_GET_SIZE(pseq.get());
        std::vector<Point64> pts;
        pts.reserve(static_cast<size_t>(np));
        for (Py_ssize_t j = 0; j < np; ++j) {
            PyObject* pt = PySequence_Fast_GET_ITEM(pseq.get(), j);
            int64_t x = 0, y = 0;
            if (PyTuple_CheckExact(pt) && PyTuple_GET_SIZE(pt) == 2) {
                if (!parse_int64(PyTuple_GET_ITEM(pt, 0), x) || !parse_int64(PyTuple_GET_ITEM(pt, 1), y)) return nullptr;
            } else {
                PyRef ps = PyRef::adopt(PySequence_Fast(pt, "point must be (x, y)"));
                if (!ps || PySequence_Fast_GET_SIZE(ps.get()) != 2) return nullptr;
                if (!parse_int64(PySequence_Fast_GET_ITEM(ps.get(), 0), x) ||
                    !parse_int64(PySequence_Fast_GET_ITEM(ps.get(), 1), y)) return nullptr;
            }
            pts.push_back(Point64{x, y});
        }
        nets.push_back(std::move(pts));
    }

    const Py_ssize_t nb = PySequence_Fast_GET_SIZE(bseq.get());
    std::vector<Box64> bboxes;
    bboxes.reserve(static_cast<size_t>(nb));
    for (Py_ssize_t i = 0; i < nb; ++i) {
        PyObject* bo = PySequence_Fast_GET_ITEM(bseq.get(), i);
        PyRef bs = PyRef::adopt(PySequence_Fast(bo, "bbox must be 4-sequence"));
        if (!bs || PySequence_Fast_GET_SIZE(bs.get()) != 4) return nullptr;
        Box64 b{};
        if (!parse_int64(PySequence_Fast_GET_ITEM(bs.get(), 0), b.x0) ||
            !parse_int64(PySequence_Fast_GET_ITEM(bs.get(), 1), b.y0) ||
            !parse_int64(PySequence_Fast_GET_ITEM(bs.get(), 2), b.x1) ||
            !parse_int64(PySequence_Fast_GET_ITEM(bs.get(), 3), b.y1)) return nullptr;
        bboxes.push_back(b);
    }

    clay_pnr::layout::NetWirelengthResult res;
    Py_BEGIN_ALLOW_THREADS
    res = clay_pnr::layout::compute_net_wirelength_exact(nets, bboxes);
    Py_END_ALLOW_THREADS

    PyRef py_span = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.net_total_span.size())));
    PyRef py_max = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.net_max_pin_span.size())));
    if (!py_span || !py_max) return nullptr;
    for (size_t i = 0; i < res.net_total_span.size(); ++i) {
        PyRef v1 = PyRef::adopt(PyLong_FromLongLong(res.net_total_span[i]));
        PyRef v2 = PyRef::adopt(PyLong_FromLongLong(res.net_max_pin_span[i]));
        if (!v1 || !v2) return nullptr;
        PyList_SET_ITEM(py_span.get(), static_cast<Py_ssize_t>(i), v1.release());
        PyList_SET_ITEM(py_max.get(), static_cast<Py_ssize_t>(i), v2.release());
    }
    return Py_BuildValue(
        "{s:O,s:O,s:L,s:L}",
        "net_total_span", py_span.get(),
        "net_max_pin_span", py_max.get(),
        "netlist_total_hpwl", static_cast<long long>(res.netlist_total_hpwl),
        "overlapping_pairs", static_cast<long long>(res.overlapping_pairs));
}

PyObject* py_placed_stamp_bboxes_nm(PyObject* /*self*/, PyObject* args) {
    PyObject* geom_obj = nullptr;
    PyObject* place_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &geom_obj, &place_obj)) return nullptr;
    if (!PyDict_Check(geom_obj) || !PyDict_Check(place_obj)) {
        PyErr_SetString(PyExc_TypeError, "geometry and placement must be dicts");
        return nullptr;
    }

    PyRef out = PyRef::adopt(PyDict_New());
    if (!out) return nullptr;
    Py_ssize_t pos_idx = 0;
    PyObject *name_obj = nullptr, *cg_obj = nullptr;
    while (PyDict_Next(geom_obj, &pos_idx, &name_obj, &cg_obj)) {
        PyObject* pos_val = PyDict_GetItemWithError(place_obj, name_obj);
        if (!pos_val) {
            if (PyErr_Occurred()) return nullptr;
            continue;
        }
        if (pos_val == Py_None) continue;
        double px = 0.0, py = 0.0;
        if (!extract_xy_pair_double(pos_val, px, py)) return nullptr;

        PyRef raw_layers = PyRef::adopt(PyObject_GetAttrString(cg_obj, "polys_by_layer"));
        if (!raw_layers) {
            PyErr_Clear();
            continue;
        }
        if (!PyDict_Check(raw_layers.get())) continue;

        bool any = false;
        int64_t min_x = std::numeric_limits<int64_t>::max();
        int64_t min_y = std::numeric_limits<int64_t>::max();
        int64_t max_x = std::numeric_limits<int64_t>::min();
        int64_t max_y = std::numeric_limits<int64_t>::min();

        Py_ssize_t lidx = 0;
        PyObject *lkey = nullptr, *rings_obj = nullptr;
        while (PyDict_Next(raw_layers.get(), &lidx, &lkey, &rings_obj)) {
            PyRef rseq = PyRef::adopt(PySequence_Fast(rings_obj, "rings must be sequence"));
            if (!rseq) return nullptr;
            const Py_ssize_t nr = PySequence_Fast_GET_SIZE(rseq.get());
            for (Py_ssize_t ri = 0; ri < nr; ++ri) {
                PyRef ring = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(rseq.get(), ri), "ring must be sequence"));
                if (!ring) return nullptr;
                const Py_ssize_t nv = PySequence_Fast_GET_SIZE(ring.get());
                PyObject** vitems = PySequence_Fast_ITEMS(ring.get());
                for (Py_ssize_t vi = 0; vi < nv; ++vi) {
                    double lx = 0.0, ly = 0.0;
                    if (!extract_xy_pair_double(vitems[vi], lx, ly)) return nullptr;
                    int64_t ix = 0, iy = 0;
                    if (!placed_coord_nm_py_exact(lx, px, ix) || !placed_coord_nm_py_exact(ly, py, iy)) return nullptr;
                    if (!any) {
                        min_x = max_x = ix;
                        min_y = max_y = iy;
                        any = true;
                    } else {
                        if (ix < min_x) min_x = ix;
                        if (ix > max_x) max_x = ix;
                        if (iy < min_y) min_y = iy;
                        if (iy > max_y) max_y = iy;
                    }
                }
            }
        }
        if (any) {
            PyRef bb = PyRef::adopt(make_box64_tuple(Box64{min_x, min_y, max_x, max_y}));
            if (!bb || PyDict_SetItem(out.get(), name_obj, bb.get()) < 0) return nullptr;
        }
    }
    return out.release();
}

PyObject* py_placed_stamp_layer_bboxes_nm(PyObject* /*self*/, PyObject* args) {
    PyObject* geom_obj = nullptr;
    PyObject* place_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &geom_obj, &place_obj)) return nullptr;
    if (!PyDict_Check(geom_obj) || !PyDict_Check(place_obj)) {
        PyErr_SetString(PyExc_TypeError, "geometry and placement must be dicts");
        return nullptr;
    }

    PyRef out = PyRef::adopt(PyDict_New());
    if (!out) return nullptr;
    Py_ssize_t pos_idx = 0;
    PyObject *name_obj = nullptr, *cg_obj = nullptr;
    while (PyDict_Next(geom_obj, &pos_idx, &name_obj, &cg_obj)) {
        PyObject* pos_val = PyDict_GetItemWithError(place_obj, name_obj);
        if (!pos_val) {
            if (PyErr_Occurred()) return nullptr;
            continue;
        }
        if (pos_val == Py_None) continue;
        double px = 0.0, py = 0.0;
        if (!extract_xy_pair_double(pos_val, px, py)) return nullptr;

        PyRef raw_layers = PyRef::adopt(PyObject_GetAttrString(cg_obj, "polys_by_layer"));
        if (!raw_layers) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError, "placed layer bbox authority must be an exact mapping");
            return nullptr;
        }
        if (!PyDict_CheckExact(raw_layers.get())) {
            PyErr_SetString(PyExc_TypeError, "placed layer bbox authority must be an exact mapping");
            return nullptr;
        }

        PyRef layer_dict = PyRef::adopt(PyDict_New());
        if (!layer_dict) return nullptr;
        Py_ssize_t lidx = 0;
        PyObject *lkey = nullptr, *rings_obj = nullptr;
        while (PyDict_Next(raw_layers.get(), &lidx, &lkey, &rings_obj)) {
            if (!PyTuple_CheckExact(lkey) || PyTuple_GET_SIZE(lkey) != 2 ||
                !PyLong_CheckExact(PyTuple_GET_ITEM(lkey, 0)) ||
                !PyLong_CheckExact(PyTuple_GET_ITEM(lkey, 1))) {
                PyErr_SetString(PyExc_TypeError, "placed layer bbox authority has a malformed layer key");
                return nullptr;
            }
            if (!PyList_CheckExact(rings_obj)) {
                PyErr_SetString(PyExc_TypeError, "placed layer bbox authority has a malformed polygon list");
                return nullptr;
            }
            bool any = false;
            int64_t min_x = 0, min_y = 0, max_x = 0, max_y = 0;
            const Py_ssize_t nr = PyList_GET_SIZE(rings_obj);
            for (Py_ssize_t ri = 0; ri < nr; ++ri) {
                PyRef ring = PyRef::adopt(PySequence_Fast(PyList_GET_ITEM(rings_obj, ri), "ring must be sequence"));
                if (!ring) return nullptr;
                const Py_ssize_t nv = PySequence_Fast_GET_SIZE(ring.get());
                PyObject** vitems = PySequence_Fast_ITEMS(ring.get());
                for (Py_ssize_t vi = 0; vi < nv; ++vi) {
                    double lx = 0.0, ly = 0.0;
                    if (!extract_xy_pair_double(vitems[vi], lx, ly)) return nullptr;
                    int64_t ix = 0, iy = 0;
                    if (!placed_coord_nm_py_exact(lx, px, ix) || !placed_coord_nm_py_exact(ly, py, iy)) return nullptr;
                    if (!any) {
                        min_x = max_x = ix;
                        min_y = max_y = iy;
                        any = true;
                    } else {
                        if (ix < min_x) min_x = ix;
                        if (ix > max_x) max_x = ix;
                        if (iy < min_y) min_y = iy;
                        if (iy > max_y) max_y = iy;
                    }
                }
            }
            if (any) {
                PyRef bb = PyRef::adopt(make_box64_tuple(Box64{min_x, min_y, max_x, max_y}));
                if (!bb || PyDict_SetItem(layer_dict.get(), lkey, bb.get()) < 0) return nullptr;
            }
        }
        if (PyDict_GET_SIZE(layer_dict.get()) > 0) {
            PyRef str_name = PyRef::adopt(PyObject_Str(name_obj));
            if (!str_name || PyDict_SetItem(out.get(), str_name.get(), layer_dict.get()) < 0) return nullptr;
        }
    }
    return out.release();
}

PyObject* py_placed_stamp_layer_rectangles_nm(PyObject* /*self*/, PyObject* args) {
    PyObject* geom_obj = nullptr;
    PyObject* place_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &geom_obj, &place_obj)) return nullptr;
    if (!PyDict_Check(geom_obj) || !PyDict_Check(place_obj)) {
        PyErr_SetString(PyExc_TypeError, "geometry and placement must be dicts");
        return nullptr;
    }

    PyRef out = PyRef::adopt(PyDict_New());
    if (!out) return nullptr;
    std::vector<Point64> ring_pts;
    std::vector<Box64> raw_boxes;

    Py_ssize_t pos_idx = 0;
    PyObject *name_obj = nullptr, *cg_obj = nullptr;
    while (PyDict_Next(geom_obj, &pos_idx, &name_obj, &cg_obj)) {
        PyObject* pos_val = PyDict_GetItemWithError(place_obj, name_obj);
        if (!pos_val) {
            if (PyErr_Occurred()) return nullptr;
            continue;
        }
        if (pos_val == Py_None) continue;
        double px = 0.0, py = 0.0;
        if (!extract_xy_pair_double(pos_val, px, py)) return nullptr;

        PyRef raw_layers = PyRef::adopt(PyObject_GetAttrString(cg_obj, "polys_by_layer"));
        if (!raw_layers) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError, "placed layer rectangle authority must be an exact mapping");
            return nullptr;
        }
        if (!PyDict_CheckExact(raw_layers.get())) {
            PyErr_SetString(PyExc_TypeError, "placed layer rectangle authority must be an exact mapping");
            return nullptr;
        }

        PyRef layer_dict = PyRef::adopt(PyDict_New());
        if (!layer_dict) return nullptr;
        Py_ssize_t lidx = 0;
        PyObject *lkey = nullptr, *rings_obj = nullptr;
        while (PyDict_Next(raw_layers.get(), &lidx, &lkey, &rings_obj)) {
            if (!PyTuple_CheckExact(lkey) || PyTuple_GET_SIZE(lkey) != 2 ||
                !PyLong_CheckExact(PyTuple_GET_ITEM(lkey, 0)) ||
                !PyLong_CheckExact(PyTuple_GET_ITEM(lkey, 1))) {
                PyErr_SetString(PyExc_TypeError, "placed layer rectangle authority has a malformed layer key");
                return nullptr;
            }
            if (!PyList_CheckExact(rings_obj)) {
                PyErr_SetString(PyExc_TypeError, "placed layer rectangle authority has a malformed polygon list");
                return nullptr;
            }

            raw_boxes.clear();
            const Py_ssize_t nr = PyList_GET_SIZE(rings_obj);
            for (Py_ssize_t ri = 0; ri < nr; ++ri) {
                PyRef ring = PyRef::adopt(PySequence_Fast(PyList_GET_ITEM(rings_obj, ri), "ring must be sequence"));
                if (!ring) return nullptr;
                const Py_ssize_t nv = PySequence_Fast_GET_SIZE(ring.get());
                if (nv < 3) {
                    PyErr_SetString(PyExc_ValueError, "placed layer rectangle authority has a degenerate polygon");
                    return nullptr;
                }
                ring_pts.clear();
                ring_pts.reserve(static_cast<size_t>(nv));
                PyObject** vitems = PySequence_Fast_ITEMS(ring.get());
                for (Py_ssize_t vi = 0; vi < nv; ++vi) {
                    double lx = 0.0, ly = 0.0;
                    if (!extract_xy_pair_double(vitems[vi], lx, ly)) return nullptr;
                    int64_t ix = 0, iy = 0;
                    if (!placed_coord_nm_py_exact(lx, px, ix) || !placed_coord_nm_py_exact(ly, py, iy)) return nullptr;
                    ring_pts.push_back(Point64{ix, iy});
                }
                std::string err;
                if (!clay_pnr::layout::decompose_orthogonal_polygon_boxes(ring_pts, raw_boxes, &err)) {
                    PyErr_SetString(PyExc_ValueError, err.c_str());
                    return nullptr;
                }
            }
            if (raw_boxes.empty()) continue;
            const PolygonSet merged = PolygonSet::from_boxes(raw_boxes);
            const auto& rects = merged.rects();
            if (rects.empty()) continue;
            PyRef tup = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(rects.size())));
            if (!tup) return nullptr;
            for (size_t i = 0; i < rects.size(); ++i) {
                PyRef bt = PyRef::adopt(make_box64_tuple(rects[i]));
                if (!bt) return nullptr;
                PyTuple_SET_ITEM(tup.get(), static_cast<Py_ssize_t>(i), bt.release());
            }
            if (PyDict_SetItem(layer_dict.get(), lkey, tup.get()) < 0) return nullptr;
        }
        if (PyDict_GET_SIZE(layer_dict.get()) > 0) {
            PyRef str_name = PyRef::adopt(PyObject_Str(name_obj));
            if (!str_name || PyDict_SetItem(out.get(), str_name.get(), layer_dict.get()) < 0) return nullptr;
        }
    }
    return out.release();
}

PyObject* py_boxes_covered_by_union(PyObject* /*self*/, PyObject* args) {
    PyObject* act_obj = nullptr;
    PyObject* auth_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &act_obj, &auth_obj)) return nullptr;
    std::vector<Box64> act, auth;
    if (!parse_rect_list(act_obj, act) || !parse_rect_list(auth_obj, auth)) return nullptr;
    const bool ok = clay_pnr::layout::boxes_covered_by_union_exact(act, auth);
    return PyBool_FromLong(ok ? 1 : 0);
}

PyObject* py_canonical_row_polygon_records(PyObject* /*self*/, PyObject* args) {
    PyObject* raw_polys_obj = nullptr;
    long long max_polys = 0;
    long long max_verts = 0;
    long long max_cum_verts = 0;
    long long max_dim = 0;
    if (!PyArg_ParseTuple(args, "OLLLL", &raw_polys_obj, &max_polys, &max_verts, &max_cum_verts, &max_dim)) {
        return nullptr;
    }

    PyRef iter = PyRef::adopt(PyObject_GetIter(raw_polys_obj));
    if (!iter) return nullptr;

    struct PolyRec {
        int64_t layer = 0;
        int64_t datatype = 0;
        std::vector<std::pair<int64_t, int64_t>> verts;
    };
    std::vector<PolyRec> recs;
    int64_t cum_verts = 0;
    const int64_t neg_max_dim = -static_cast<int64_t>(max_dim);
    const int64_t pos_max_dim = static_cast<int64_t>(max_dim);

    while (PyRef lp = PyRef::adopt(PyIter_Next(iter.get()))) {
        if (static_cast<long long>(recs.size()) >= max_polys) {
            PyErr_Format(PyExc_OverflowError, "CAP_POLYGONS:%lld", max_polys);
            return nullptr;
        }
        PyRef layer_obj = PyRef::adopt(PyObject_GetAttrString(lp.get(), "layer"));
        if (!layer_obj) PyErr_Clear();
        PyRef dt_obj = PyRef::adopt(PyObject_GetAttrString(lp.get(), "datatype"));
        if (!dt_obj) PyErr_Clear();
        if (!layer_obj || !dt_obj || !PyLong_CheckExact(layer_obj.get()) || !PyLong_CheckExact(dt_obj.get())) {
            PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon layer/datatype must be exact integers");
            return nullptr;
        }
        int overflow = 0;
        const int64_t layer = PyLong_AsLongLongAndOverflow(layer_obj.get(), &overflow);
        if (overflow != 0 || (layer == -1 && PyErr_Occurred())) {
            PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon layer/datatype must be exact integers");
            return nullptr;
        }
        const int64_t dt = PyLong_AsLongLongAndOverflow(dt_obj.get(), &overflow);
        if (overflow != 0 || (dt == -1 && PyErr_Occurred())) {
            PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon layer/datatype must be exact integers");
            return nullptr;
        }

        PyRef poly_obj = PyRef::adopt(PyObject_GetAttrString(lp.get(), "polygon"));
        if (!poly_obj) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon must have at least three vertices");
            return nullptr;
        }
        PyRef verts_obj = PyRef::adopt(PyObject_GetAttrString(poly_obj.get(), "vertices"));
        if (!verts_obj) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon must have at least three vertices");
            return nullptr;
        }

        PolyRec rec;
        rec.layer = layer;
        rec.datatype = dt;

        if (PyList_CheckExact(verts_obj.get()) || PyTuple_CheckExact(verts_obj.get())) {
            const bool is_list = PyList_CheckExact(verts_obj.get());
            const Py_ssize_t nv = is_list ? PyList_GET_SIZE(verts_obj.get()) : PyTuple_GET_SIZE(verts_obj.get());
            if (nv > static_cast<Py_ssize_t>(max_verts)) {
                PyErr_Format(PyExc_OverflowError, "CAP_VERTS:%lld", max_verts);
                return nullptr;
            }
            if (nv < 3) {
                PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon must have at least three vertices");
                return nullptr;
            }
            rec.verts.reserve(static_cast<size_t>(nv));
            for (Py_ssize_t i = 0; i < nv; ++i) {
                PyObject* v = is_list ? PyList_GET_ITEM(verts_obj.get(), i) : PyTuple_GET_ITEM(verts_obj.get(), i);
                PyRef xo = PyRef::adopt(PyObject_GetAttrString(v, "x"));
                if (!xo) PyErr_Clear();
                if (!xo || !PyLong_CheckExact(xo.get())) {
                    PyErr_Format(PyExc_ValueError,
                                 "MAT:materialized polygon x coordinate must be an exact integer in [-%lld, %lld]",
                                 max_dim, max_dim);
                    return nullptr;
                }
                int ovx = 0;
                const int64_t x = PyLong_AsLongLongAndOverflow(xo.get(), &ovx);
                if (ovx != 0 || x < neg_max_dim || x > pos_max_dim) {
                    PyErr_Format(PyExc_ValueError,
                                 "MAT:materialized polygon x coordinate must be an exact integer in [-%lld, %lld]",
                                 max_dim, max_dim);
                    return nullptr;
                }
                PyRef yo = PyRef::adopt(PyObject_GetAttrString(v, "y"));
                if (!yo) PyErr_Clear();
                if (!yo || !PyLong_CheckExact(yo.get())) {
                    PyErr_Format(PyExc_ValueError,
                                 "MAT:materialized polygon y coordinate must be an exact integer in [-%lld, %lld]",
                                 max_dim, max_dim);
                    return nullptr;
                }
                int ovy = 0;
                const int64_t y = PyLong_AsLongLongAndOverflow(yo.get(), &ovy);
                if (ovy != 0 || y < neg_max_dim || y > pos_max_dim) {
                    PyErr_Format(PyExc_ValueError,
                                 "MAT:materialized polygon y coordinate must be an exact integer in [-%lld, %lld]",
                                 max_dim, max_dim);
                    return nullptr;
                }
                rec.verts.emplace_back(x, y);
            }
        } else {
            PyRef vit = PyRef::adopt(PyObject_GetIter(verts_obj.get()));
            if (!vit) return nullptr;
            while (PyRef v = PyRef::adopt(PyIter_Next(vit.get()))) {
                if (static_cast<long long>(rec.verts.size()) >= max_verts) {
                    PyErr_Format(PyExc_OverflowError, "CAP_VERTS:%lld", max_verts);
                    return nullptr;
                }
                PyRef xo = PyRef::adopt(PyObject_GetAttrString(v.get(), "x"));
                if (!xo) PyErr_Clear();
                int ovx = 0;
                const int64_t x = (xo && PyLong_CheckExact(xo.get()))
                    ? PyLong_AsLongLongAndOverflow(xo.get(), &ovx) : (ovx = 1, 0);
                if (ovx != 0 || x < neg_max_dim || x > pos_max_dim) {
                    PyErr_Format(PyExc_ValueError,
                                 "MAT:materialized polygon x coordinate must be an exact integer in [-%lld, %lld]",
                                 max_dim, max_dim);
                    return nullptr;
                }
                PyRef yo = PyRef::adopt(PyObject_GetAttrString(v.get(), "y"));
                if (!yo) PyErr_Clear();
                int ovy = 0;
                const int64_t y = (yo && PyLong_CheckExact(yo.get()))
                    ? PyLong_AsLongLongAndOverflow(yo.get(), &ovy) : (ovy = 1, 0);
                if (ovy != 0 || y < neg_max_dim || y > pos_max_dim) {
                    PyErr_Format(PyExc_ValueError,
                                 "MAT:materialized polygon y coordinate must be an exact integer in [-%lld, %lld]",
                                 max_dim, max_dim);
                    return nullptr;
                }
                rec.verts.emplace_back(x, y);
            }
            if (PyErr_Occurred()) return nullptr;
            if (rec.verts.size() < 3) {
                PyErr_SetString(PyExc_ValueError, "MAT:materialized polygon must have at least three vertices");
                return nullptr;
            }
        }

        cum_verts += static_cast<int64_t>(rec.verts.size());
        if (cum_verts > static_cast<int64_t>(max_cum_verts)) {
            PyErr_SetString(PyExc_ValueError, "MAT:materialization exceeds cumulative vertex complexity cap");
            return nullptr;
        }
        recs.push_back(std::move(rec));
    }
    if (PyErr_Occurred()) return nullptr;

    std::sort(recs.begin(), recs.end(), [](const PolyRec& a, const PolyRec& b) {
        if (a.layer != b.layer) return a.layer < b.layer;
        if (a.datatype != b.datatype) return a.datatype < b.datatype;
        return a.verts < b.verts;
    });

    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(recs.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < recs.size(); ++i) {
        const auto& r = recs[i];
        PyRef vt = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(r.verts.size())));
        if (!vt) return nullptr;
        for (size_t j = 0; j < r.verts.size(); ++j) {
            PyRef pt = PyRef::adopt(PyTuple_New(2));
            PyRef ox = PyRef::adopt(PyLong_FromLongLong(r.verts[j].first));
            PyRef oy = PyRef::adopt(PyLong_FromLongLong(r.verts[j].second));
            if (!pt || !ox || !oy) return nullptr;
            PyTuple_SET_ITEM(pt.get(), 0, ox.release());
            PyTuple_SET_ITEM(pt.get(), 1, oy.release());
            PyTuple_SET_ITEM(vt.get(), static_cast<Py_ssize_t>(j), pt.release());
        }
        PyRef row = PyRef::adopt(PyTuple_New(3));
        PyRef ol = PyRef::adopt(PyLong_FromLongLong(r.layer));
        PyRef od = PyRef::adopt(PyLong_FromLongLong(r.datatype));
        if (!row || !ol || !od) return nullptr;
        PyTuple_SET_ITEM(row.get(), 0, ol.release());
        PyTuple_SET_ITEM(row.get(), 1, od.release());
        PyTuple_SET_ITEM(row.get(), 2, vt.release());
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), row.release());
    }
    return out.release();
}

PyObject* py_canonical_emitted_device_snapshot_records(PyObject* /*self*/, PyObject* args) {
    PyObject* geom_obj = nullptr;
    PyObject* place_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &geom_obj, &place_obj)) return nullptr;
    if (!PyDict_Check(geom_obj) || !PyDict_Check(place_obj)) {
        PyErr_SetString(PyExc_TypeError, "emitted device snapshot requires geometry and placement mappings");
        return nullptr;
    }

    struct SnapRec {
        std::string name;
        std::vector<std::string> members;
        int64_t layer = 0;
        int64_t datatype = 0;
        std::vector<std::pair<int64_t, int64_t>> ring;
    };
    std::vector<SnapRec> recs;
    std::vector<std::pair<int64_t, int64_t>> raw_ring, best_ring, cand_ring;

    Py_ssize_t pos_idx = 0;
    PyObject *raw_name_obj = nullptr, *stamp_obj = nullptr;
    while (PyDict_Next(geom_obj, &pos_idx, &raw_name_obj, &stamp_obj)) {
        PyRef name_str_obj = PyRef::adopt(PyObject_Str(raw_name_obj));
        if (!name_str_obj) return nullptr;
        const char* name_cstr = PyUnicode_AsUTF8(name_str_obj.get());
        if (!name_cstr) return nullptr;
        const std::string name(name_cstr);

        std::vector<std::string> members;
        PyRef raw_members = PyRef::adopt(PyObject_GetAttrString(stamp_obj, "composite_members"));
        if (!raw_members) {
            PyErr_Clear();
        } else if (raw_members.get() != Py_None) {
            if (!PyTuple_CheckExact(raw_members.get())) {
                PyErr_Format(PyExc_ValueError, "emitted device snapshot '%s' has malformed composite members", name_cstr);
                return nullptr;
            }
            const Py_ssize_t nm = PyTuple_GET_SIZE(raw_members.get());
            members.reserve(static_cast<size_t>(nm));
            for (Py_ssize_t mi = 0; mi < nm; ++mi) {
                PyObject* mo = PyTuple_GET_ITEM(raw_members.get(), mi);
                if (!PyUnicode_CheckExact(mo) || PyUnicode_GET_LENGTH(mo) == 0) {
                    PyErr_Format(PyExc_ValueError, "emitted device snapshot '%s' has malformed composite members", name_cstr);
                    return nullptr;
                }
                const char* ms = PyUnicode_AsUTF8(mo);
                if (!ms) return nullptr;
                std::string mstr(ms);
                for (const auto& prev : members) {
                    if (prev == mstr) {
                        PyErr_Format(PyExc_ValueError, "emitted device snapshot '%s' has malformed composite members", name_cstr);
                        return nullptr;
                    }
                }
                members.push_back(std::move(mstr));
            }
        }

        PyObject* pos_val = PyDict_GetItemWithError(place_obj, raw_name_obj);
        if (!pos_val) {
            if (PyErr_Occurred()) return nullptr;
            pos_val = PyDict_GetItemWithError(place_obj, name_str_obj.get());
            if (!pos_val && PyErr_Occurred()) return nullptr;
        }
        if (!pos_val || pos_val == Py_None) continue;
        if ((!PyTuple_CheckExact(pos_val) && !PyList_CheckExact(pos_val)) ||
            PySequence_Fast_GET_SIZE(pos_val) != 2) {
            PyErr_SetString(PyExc_ValueError, "emitted device snapshot has invalid placement");
            return nullptr;
        }
        double px = 0.0, py = 0.0;
        if (!extract_xy_pair_double(pos_val, px, py)) {
            PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains an invalid polygon");
            return nullptr;
        }

        PyRef layers_obj = PyRef::adopt(PyObject_GetAttrString(stamp_obj, "polys_by_layer"));
        if (!layers_obj || !PyDict_Check(layers_obj.get())) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError, "emitted device snapshot stamp lacks polys_by_layer");
            return nullptr;
        }

        Py_ssize_t lidx = 0;
        PyObject *lkey = nullptr, *rings_obj = nullptr;
        while (PyDict_Next(layers_obj.get(), &lidx, &lkey, &rings_obj)) {
            PyRef kt = PyRef::adopt(PySequence_Fast(lkey, "invalid layer key"));
            if (!kt || PySequence_Fast_GET_SIZE(kt.get()) < 2) {
                PyErr_Clear();
                PyErr_SetString(PyExc_ValueError, "emitted device snapshot has invalid layer key");
                return nullptr;
            }
            int64_t l_val = 0, d_val = 0;
            if (!parse_int64(PySequence_Fast_GET_ITEM(kt.get(), 0), l_val) ||
                !parse_int64(PySequence_Fast_GET_ITEM(kt.get(), 1), d_val)) {
                PyErr_Clear();
                PyErr_SetString(PyExc_ValueError, "emitted device snapshot has invalid layer key");
                return nullptr;
            }
            if (!PyTuple_Check(rings_obj) && !PyList_Check(rings_obj)) {
                PyErr_SetString(PyExc_TypeError, "emitted device snapshot rings must be a sequence");
                return nullptr;
            }
            const bool rings_is_list = PyList_Check(rings_obj);
            const Py_ssize_t nr = rings_is_list ? PyList_GET_SIZE(rings_obj) : PyTuple_GET_SIZE(rings_obj);
            for (Py_ssize_t ri = 0; ri < nr; ++ri) {
                PyObject* ring_obj = rings_is_list ? PyList_GET_ITEM(rings_obj, ri) : PyTuple_GET_ITEM(rings_obj, ri);
                PyRef rseq = PyRef::adopt(PySequence_Fast(ring_obj, "invalid ring"));
                if (!rseq) {
                    PyErr_Clear();
                    PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains an invalid polygon");
                    return nullptr;
                }
                const Py_ssize_t nv = PySequence_Fast_GET_SIZE(rseq.get());
                raw_ring.clear();
                raw_ring.reserve(static_cast<size_t>(nv));
                PyObject** vitems = PySequence_Fast_ITEMS(rseq.get());
                for (Py_ssize_t vi = 0; vi < nv; ++vi) {
                    double lx = 0.0, ly = 0.0;
                    if (!extract_xy_pair_double(vitems[vi], lx, ly)) {
                        PyErr_Clear();
                        PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains an invalid polygon");
                        return nullptr;
                    }
                    int64_t ix = 0, iy = 0;
                    if (!placed_coord_nm_py_exact(lx, px, ix) || !placed_coord_nm_py_exact(ly, py, iy)) {
                        PyErr_Clear();
                        PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains an invalid polygon");
                        return nullptr;
                    }
                    raw_ring.emplace_back(ix, iy);
                }
                if (raw_ring.size() > 1 && raw_ring.front() == raw_ring.back()) {
                    raw_ring.pop_back();
                }
                const size_t rn = raw_ring.size();
                if (rn < 3) {
                    PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains a degenerate polygon");
                    return nullptr;
                }
                // Count unique vertices and exact twice signed area.
                size_t unique_cnt = 0;
                for (size_t a = 0; a < rn && unique_cnt < 3; ++a) {
                    bool seen = false;
                    for (size_t b = 0; b < a; ++b) {
                        if (raw_ring[a] == raw_ring[b]) { seen = true; break; }
                    }
                    if (!seen) ++unique_cnt;
                }
                if (unique_cnt < 3) {
                    PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains a degenerate polygon");
                    return nullptr;
                }
                int64_t area2 = 0;
                const int64_t ox = raw_ring[0].first;
                const int64_t oy = raw_ring[0].second;
                for (size_t a = 0; a < rn; ++a) {
                    const auto& p0 = raw_ring[a];
                    const auto& p1 = raw_ring[(a + 1) % rn];
                    area2 += (p0.first - ox) * (p1.second - oy) - (p1.first - ox) * (p0.second - oy);
                }
                if (area2 == 0) {
                    PyErr_SetString(PyExc_ValueError, "emitted device snapshot contains a degenerate polygon");
                    return nullptr;
                }
                // Canonical minimal rotation across forward and reversed orientations.
                best_ring = raw_ring;
                cand_ring.resize(rn);
                for (int rev = 0; rev < 2; ++rev) {
                    for (size_t shift = 0; shift < rn; ++shift) {
                        if (rev == 0 && shift == 0) continue;
                        for (size_t k = 0; k < rn; ++k) {
                            const size_t idx = rev ? ((rn - 1 - ((shift + k) % rn))) : ((shift + k) % rn);
                            cand_ring[k] = raw_ring[idx];
                        }
                        if (cand_ring < best_ring) best_ring = cand_ring;
                    }
                }
                recs.push_back(SnapRec{name, members, l_val, d_val, best_ring});
            }
        }
    }

    std::sort(recs.begin(), recs.end(), [](const SnapRec& a, const SnapRec& b) {
        if (a.name != b.name) return a.name < b.name;
        if (a.members != b.members) return a.members < b.members;
        if (a.layer != b.layer) return a.layer < b.layer;
        if (a.datatype != b.datatype) return a.datatype < b.datatype;
        return a.ring < b.ring;
    });

    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(recs.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < recs.size(); ++i) {
        const auto& r = recs[i];
        PyRef py_name = PyRef::adopt(PyUnicode_FromStringAndSize(r.name.data(), static_cast<Py_ssize_t>(r.name.size())));
        PyRef py_mem = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(r.members.size())));
        PyRef py_key = PyRef::adopt(PyTuple_New(2));
        PyRef py_l = PyRef::adopt(PyLong_FromLongLong(r.layer));
        PyRef py_d = PyRef::adopt(PyLong_FromLongLong(r.datatype));
        PyRef py_ring = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(r.ring.size())));
        PyRef row = PyRef::adopt(PyTuple_New(4));
        if (!py_name || !py_mem || !py_key || !py_l || !py_d || !py_ring || !row) return nullptr;
        for (size_t m = 0; m < r.members.size(); ++m) {
            PyRef ms = PyRef::adopt(PyUnicode_FromStringAndSize(r.members[m].data(), static_cast<Py_ssize_t>(r.members[m].size())));
            if (!ms) return nullptr;
            PyTuple_SET_ITEM(py_mem.get(), static_cast<Py_ssize_t>(m), ms.release());
        }
        PyTuple_SET_ITEM(py_key.get(), 0, py_l.release());
        PyTuple_SET_ITEM(py_key.get(), 1, py_d.release());
        for (size_t k = 0; k < r.ring.size(); ++k) {
            PyRef pt = PyRef::adopt(PyTuple_New(2));
            PyRef ox = PyRef::adopt(PyLong_FromLongLong(r.ring[k].first));
            PyRef oy = PyRef::adopt(PyLong_FromLongLong(r.ring[k].second));
            if (!pt || !ox || !oy) return nullptr;
            PyTuple_SET_ITEM(pt.get(), 0, ox.release());
            PyTuple_SET_ITEM(pt.get(), 1, oy.release());
            PyTuple_SET_ITEM(py_ring.get(), static_cast<Py_ssize_t>(k), pt.release());
        }
        PyTuple_SET_ITEM(row.get(), 0, py_name.release());
        PyTuple_SET_ITEM(row.get(), 1, py_mem.release());
        PyTuple_SET_ITEM(row.get(), 2, py_key.release());
        PyTuple_SET_ITEM(row.get(), 3, py_ring.release());
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), row.release());
    }
    return out.release();
}

PyMethodDef kEngineMethods[] = {
    {"polygon_boolean_cpp", py_polygon_boolean, METH_VARARGS,
     "Exact C++20 sweep-line Manhattan boolean op ('or', 'and', 'not', 'xor') returning canonical boxes."},
    {"polygon_sized_cpp", py_polygon_sized, METH_VARARGS,
     "Exact C++20 L-infinity morphological grow/shrink returning canonical boxes."},
    {"polygon_components_cpp", py_polygon_components, METH_VARARGS,
     "Exact C++20 connected components over canonical Manhattan boxes."},
    {"polygon_contours_cpp", py_polygon_contours, METH_VARARGS,
     "Exact C++20 orthogonal boundary contour extraction returning [(points, is_hole), ...]."},
    {"gds_write_boxes_cpp", py_gds_write_boxes, METH_VARARGS,
     "Pure C++20 binary GDSII v6.0 box writer."},
    {"gds_write_polygons_cpp", py_gds_write_polygons, METH_VARARGS,
     "Pure C++20 binary GDSII v6.0 arbitrary polygon + text label writer."},
    {"gds_write_library_cpp", py_gds_write_library, METH_VARARGS,
     "Pure C++20 multi-cell binary GDSII v6.0 library writer."},
    {"gds_inspect_layers_cpp", py_gds_inspect_layers, METH_VARARGS,
     "Pure C++20 binary GDSII v6.0 reader + hierarchical layer area/bbox flattener."},
    {"compact_axis_cpp", py_compact_axis, METH_VARARGS,
     "Exact O(V+E) pure C++20 integer difference-constraint 1D DAG compactor."},
    {"compact_2d_cpp", py_compact_2d, METH_VARARGS,
     "Pure C++20 2D difference-constraint compactor with symmetry axis and manufacturing grid."},
    {"solve_envelope_min_cpp", py_solve_envelope_min, METH_VARARGS,
     "Pure C++20 deterministic 2D envelope-min analog placer (zero OR-Tools)."},
    {"solve_banded_area_cpp", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(py_solve_banded_area)), METH_VARARGS | METH_KEYWORDS,
     "Pure C++20 deterministic 2D banded area/height macro placer (zero OR-Tools)."},
    {"parse_clay_ir_cpp", py_parse_clay_ir, METH_VARARGS,
     "Pure C++20 zero-Python .clay DSL lexer & recursive-descent IR parser."},
    {"synthesize_clay_to_gds_cpp", py_synthesize_clay_to_gds, METH_VARARGS,
     "Pure C++20 end-to-end zero-Python .clay -> PCell -> BandedPlace -> Route -> GDSII synthesis."},
    {"merge_net_boxes_cpp", py_merge_net_boxes, METH_VARARGS,
     "Pure C++20 per-net x per-layer orthogonal polygon merger (GDS-safe hole-free)."},
    {"extract_connectivity_cpp", py_extract_connectivity, METH_VARARGS,
     "Pure C++20 multi-layer physical connectivity extractor with MIM carve support."},
    {"verify_net_connectivity_cpp", py_verify_net_connectivity, METH_VARARGS,
     "Pure C++20 in-process net connectivity & inter-net short checker (drop-in for Rust net-connectivity)."},
    {"compute_net_wirelength_cpp", py_compute_net_wirelength, METH_VARARGS,
     "Pure C++20 in-process net HPWL, max pin span, and overlap counter (drop-in for Rust net-wirelength)."},
    {"placed_stamp_bboxes_nm_cpp", py_placed_stamp_bboxes_nm, METH_VARARGS,
     "Exact IEEE-754 parity C++20 placed stamp bounding box computation."},
    {"placed_stamp_layer_bboxes_nm_cpp", py_placed_stamp_layer_bboxes_nm, METH_VARARGS,
     "Exact IEEE-754 parity C++20 placed per-layer stamp bounding box computation."},
    {"placed_stamp_layer_rectangles_nm_cpp", py_placed_stamp_layer_rectangles_nm, METH_VARARGS,
     "Exact C++20 Manhattan decomposition + PolygonSet merge for placed stamp layer rectangles (zero KLayout)."},
    {"boxes_covered_by_union_cpp", py_boxes_covered_by_union, METH_VARARGS,
     "Exact C++20 PolygonSet union coverage check (zero KLayout)."},
    {"canonical_row_polygon_records_cpp", py_canonical_row_polygon_records, METH_VARARGS,
     "Exact C++20 validation + canonical sorting of LayerPolygon records for row geometry digests."},
    {"canonical_emitted_device_snapshot_records_cpp", py_canonical_emitted_device_snapshot_records, METH_VARARGS,
     "Exact C++20 canonicalized emitted device polygon records for SHA-256 snapshot digests."},
    {"generate_mos_pcell_cpp", py_generate_mos_pcell, METH_VARARGS,
     "Pure C++20 rule-driven multi-finger MOS PCell generator."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kEngineModule = {
    PyModuleDef_HEAD_INIT,
    "_engine_cpp",
    "Unified Clay C++20 Main Engine bindings.",
    -1,
    kEngineMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__engine_cpp(void) {
    return PyModule_Create(&kEngineModule);
}
