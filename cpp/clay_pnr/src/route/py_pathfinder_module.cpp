// Thin RAII CPython 3.12 binding module for `clay._native._pathfinder_cpp`.
// Exposes NativePathFinderGrid and NativeRectClipper backed by pure C++20
// clay_pnr::route::{PathFinderGridCore, RectClipperCore}.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)  // CPython 3.12 cpython/unicodeobject.h _unused_op
#endif
#include "clay_pnr/core/py_raii.hpp"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "clay_pnr/route/pathfinder_core.hpp"

#include <algorithm>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace {

using clay_pnr::core::Box64;
using clay_pnr::py::PyCheckpointAdapter;
using clay_pnr::py::PyRef;
using clay_pnr::py::make_box64_tuple;
using clay_pnr::route::DirPrefEntry;
using clay_pnr::route::FinishRoundResult;
using clay_pnr::route::PathFinderGridCore;
using clay_pnr::route::RectClipperCore;
using clay_pnr::route::RectUm;
using clay_pnr::route::kDefaultCheckpointInterval;

struct PyNativePathFinderGrid {
    PyObject_HEAD
    PathFinderGridCore* core;
};

static PathFinderGridCore* require_grid_core(PyNativePathFinderGrid* self) {
    if (!self || !self->core) {
        PyErr_SetString(PyExc_RuntimeError, "NativePathFinderGrid is not initialized");
        return nullptr;
    }
    return self->core;
}

static bool parse_cell_tuple(PyObject* obj, const PathFinderGridCore& core, int32_t& out_idx) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) != 3) {
        PyErr_SetString(PyExc_TypeError, "expected 3-tuple (ix, iy, gi)");
        return false;
    }
    const long ix = PyLong_AsLong(PyTuple_GET_ITEM(obj, 0));
    const long iy = PyLong_AsLong(PyTuple_GET_ITEM(obj, 1));
    const long gi = PyLong_AsLong(PyTuple_GET_ITEM(obj, 2));
    if (PyErr_Occurred()) return false;
    if (ix < 0 || ix >= core.nx || iy < 0 || iy >= core.ny || gi < 0 || gi >= core.n_layers) {
        out_idx = -1;
        return true;
    }
    out_idx = core.cell_idx(static_cast<int32_t>(ix), static_cast<int32_t>(iy), static_cast<int32_t>(gi));
    return true;
}

static PyObject* make_cell_tuple(const PathFinderGridCore& core, int32_t idx) {
    int32_t ix = 0, iy = 0, gi = 0;
    core.unpack_idx(idx, ix, iy, gi);
    PyRef t = PyRef::adopt(PyTuple_New(3));
    if (!t) return nullptr;
    PyRef v0 = PyRef::adopt(PyLong_FromLong(ix));
    PyRef v1 = PyRef::adopt(PyLong_FromLong(iy));
    PyRef v2 = PyRef::adopt(PyLong_FromLong(gi));
    if (!v0 || !v1 || !v2) return nullptr;
    PyTuple_SET_ITEM(t.get(), 0, v0.release());
    PyTuple_SET_ITEM(t.get(), 1, v1.release());
    PyTuple_SET_ITEM(t.get(), 2, v2.release());
    return t.release();
}

static PyObject* make_xy_pair_list(const std::vector<std::pair<int32_t, int32_t>>& pairs) {
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pairs.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < pairs.size(); ++i) {
        PyRef tup = PyRef::adopt(PyTuple_New(2));
        if (!tup) return nullptr;
        PyRef vx = PyRef::adopt(PyLong_FromLong(pairs[i].first));
        PyRef vy = PyRef::adopt(PyLong_FromLong(pairs[i].second));
        if (!vx || !vy) return nullptr;
        PyTuple_SET_ITEM(tup.get(), 0, vx.release());
        PyTuple_SET_ITEM(tup.get(), 1, vy.release());
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    return out.release();
}

static bool parse_int_seq(PyObject* seq_obj, std::vector<int32_t>& out) {
    PyRef seq = PyRef::adopt(PySequence_Fast(seq_obj, "expected sequence of ints"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    out.resize(static_cast<size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long v = PyLong_AsLong(items[i]);
        if (v == -1 && PyErr_Occurred()) {
            return false;
        }
        out[static_cast<size_t>(i)] = static_cast<int32_t>(v);
    }
    return true;
}

static bool parse_bool_seq(PyObject* seq_obj, std::vector<uint8_t>& out) {
    PyRef seq = PyRef::adopt(PySequence_Fast(seq_obj, "expected sequence of bools"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    out.resize(static_cast<size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (Py_ssize_t i = 0; i < n; ++i) {
        const int b = PyObject_IsTrue(items[i]);
        if (b < 0) {
            return false;
        }
        out[static_cast<size_t>(i)] = static_cast<uint8_t>(b != 0 ? 1 : 0);
    }
    return true;
}

static bool parse_layer_mask(PyObject* seq_obj, int32_t n_layers, uint32_t& mask) {
    mask = 0;
    PyRef iter = PyRef::adopt(PyObject_GetIter(seq_obj));
    if (!iter) return false;
    for (;;) {
        PyRef item = PyRef::adopt(PyIter_Next(iter.get()));
        if (!item) break;
        const long gi = PyLong_AsLong(item.get());
        if (gi == -1 && PyErr_Occurred()) {
            return false;
        }
        if (gi >= 0 && gi < n_layers) {
            mask |= (1U << static_cast<uint32_t>(gi));
        }
    }
    return !PyErr_Occurred();
}

static bool parse_cell_iter(
    PyObject* iter_obj, const PathFinderGridCore& core,
    std::vector<int32_t>& out, bool skip_oob = false
) {
    out.clear();
    if (iter_obj == nullptr || iter_obj == Py_None) return true;
    PyRef iter = PyRef::adopt(PyObject_GetIter(iter_obj));
    if (!iter) return false;
    for (;;) {
        PyRef item = PyRef::adopt(PyIter_Next(iter.get()));
        if (!item) break;
        int32_t idx = -1;
        if (!parse_cell_tuple(item.get(), core, idx)) {
            return false;
        }
        if (idx >= 0) {
            out.push_back(idx);
        } else if (!skip_oob) {
            PyErr_SetString(PyExc_ValueError, "cell coordinate out of grid bounds");
            return false;
        }
    }
    return !PyErr_Occurred();
}

static bool parse_dir_pref(
    PyObject* obj, int32_t n_layers,
    bool& has_dir_pref, std::vector<DirPrefEntry>& out
) {
    has_dir_pref = false;
    out.assign(static_cast<size_t>(n_layers), DirPrefEntry{});
    if (obj == nullptr || obj == Py_None) return true;
    if (!PyDict_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "dir_pref must be dict or None");
        return false;
    }
    if (PyDict_Size(obj) == 0) return true;
    PyObject* key = nullptr;
    PyObject* val = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &val)) {
        const long gi = PyLong_AsLong(key);
        if (gi == -1 && PyErr_Occurred()) return false;
        if (!PyTuple_Check(val) || PyTuple_GET_SIZE(val) != 2) {
            PyErr_SetString(PyExc_TypeError, "dir_pref value must be (axis, weight)");
            return false;
        }
        PyObject* ax_obj = PyTuple_GET_ITEM(val, 0);
        PyObject* w_obj = PyTuple_GET_ITEM(val, 1);
        const char* ax_str = PyUnicode_AsUTF8(ax_obj);
        if (!ax_str) return false;
        const double w = PyFloat_AsDouble(w_obj);
        if (PyErr_Occurred()) return false;
        if (gi >= 0 && gi < n_layers) {
            out[static_cast<size_t>(gi)] = DirPrefEntry{true, ax_str[0] == 'H', w};
            has_dir_pref = true;
        }
    }
    return true;
}

static void PyGrid_dealloc(PyNativePathFinderGrid* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* PyGrid_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativePathFinderGrid*>(type->tp_alloc(type, 0));
    if (self != nullptr) {
        self->core = nullptr;
    }
    return reinterpret_cast<PyObject*>(self);
}

static int PyGrid_init(PyNativePathFinderGrid* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {
        const_cast<char*>("nx"),
        const_cast<char*>("ny"),
        const_cast<char*>("n_layers"),
        const_cast<char*>("x0_nm"),
        const_cast<char*>("y0_nm"),
        const_cast<char*>("pitch_nm"),
        const_cast<char*>("preferred_horiz"),
        const_cast<char*>("min_width_nm"),
        const_cast<char*>("min_space_nm"),
        const_cast<char*>("via_cut_size_nm"),
        const_cast<char*>("via_enc_below_nm"),
        const_cast<char*>("via_enc_above_nm"),
        const_cast<char*>("via_cut_space_nm"),
        nullptr
    };
    int nx = 0, ny = 0, n_layers = 0;
    long long x0_nm = 0, y0_nm = 0, pitch_nm = 0;
    PyObject *ph_obj = nullptr, *mw_obj = nullptr, *ms_obj = nullptr;
    PyObject *vc_obj = nullptr, *veb_obj = nullptr, *vea_obj = nullptr, *vsp_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "iiiLLLOOOOOOO", kwlist,
            &nx, &ny, &n_layers, &x0_nm, &y0_nm, &pitch_nm,
            &ph_obj, &mw_obj, &ms_obj, &vc_obj, &veb_obj, &vea_obj, &vsp_obj)) {
        return -1;
    }

    std::vector<uint8_t> ph;
    std::vector<int32_t> mw, ms, vc, veb, vea, vsp;
    if (!parse_bool_seq(ph_obj, ph) ||
        !parse_int_seq(mw_obj, mw) ||
        !parse_int_seq(ms_obj, ms) ||
        !parse_int_seq(vc_obj, vc) ||
        !parse_int_seq(veb_obj, veb) ||
        !parse_int_seq(vea_obj, vea) ||
        !parse_int_seq(vsp_obj, vsp)) {
        return -1;
    }
    const size_t nl = static_cast<size_t>(std::max(0, n_layers));
    const size_t nv = (nl > 0) ? (nl - 1) : 0;
    if (ph.size() != nl || mw.size() != nl || ms.size() != nl ||
        vc.size() != nv || veb.size() != nv || vea.size() != nv || vsp.size() != nv) {
        PyErr_SetString(PyExc_ValueError, "layer/via rule array lengths do not match n_layers");
        return -1;
    }

    delete self->core;
    self->core = new (std::nothrow) PathFinderGridCore();
    if (!self->core) {
        PyErr_NoMemory();
        return -1;
    }
    if (!self->core->init(nx, ny, n_layers, x0_nm, y0_nm, pitch_nm, ph, mw, ms, vc, veb, vea, vsp)) {
        delete self->core;
        self->core = nullptr;
        PyErr_SetString(PyExc_ValueError, "invalid grid dimensions or overflow");
        return -1;
    }
    return 0;
}

static PyObject* PyGrid_set_over_active(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject* rects_obj = nullptr;
    int gi_floor = 0;
    double penalty = 0.0;
    if (!PyArg_ParseTuple(args, "Oid", &rects_obj, &gi_floor, &penalty)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    if (penalty <= 0.0 || gi_floor >= c.n_layers || rects_obj == Py_None) {
        c.set_over_active({}, gi_floor, penalty);
        Py_RETURN_NONE;
    }
    PyRef seq = PyRef::adopt(PySequence_Fast(rects_obj, "rects_um must be sequence"));
    if (!seq) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    std::vector<RectUm> rects;
    rects.reserve(static_cast<size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (Py_ssize_t i = 0; i < n; ++i) {
        double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
        if (!PyArg_ParseTuple(items[i], "dddd", &x0, &y0, &x1, &y1)) {
            return nullptr;
        }
        rects.push_back(RectUm{x0, y0, x1, y1});
    }
    c.set_over_active(rects, gi_floor, penalty);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_static_obstacles(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject *obs_obj = nullptr, *vobs_obj = nullptr, *mim_clr_obj = nullptr;
    PyObject *mim_cut_obj = nullptr, *env_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOOOO", &obs_obj, &vobs_obj, &mim_clr_obj, &mim_cut_obj, &env_obj)) {
        return nullptr;
    }
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    c.clear_static_obstacles();

    if (env_obj != Py_None) {
        long long ex0 = 0, ey0 = 0, ex1 = 0, ey1 = 0;
        if (!PyArg_ParseTuple(env_obj, "LLLL", &ex0, &ey0, &ex1, &ey1)) return nullptr;
        c.set_envelope(ex0, ey0, ex1, ey1);
    }

    if (obs_obj != Py_None) {
        PyRef seq = PyRef::adopt(PySequence_Fast(obs_obj, "obs_rects must be sequence"));
        if (!seq) return nullptr;
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
        PyObject** items = PySequence_Fast_ITEMS(seq.get());
        for (Py_ssize_t i = 0; i < n; ++i) {
            int gi = 0, owner_id = -1;
            double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
            if (!PyArg_ParseTuple(items[i], "iddddi", &gi, &x0, &y0, &x1, &y1, &owner_id)) {
                return nullptr;
            }
            c.add_obs_rect(gi, x0, y0, x1, y1, owner_id);
        }
    }

    if (vobs_obj != Py_None) {
        PyRef seq = PyRef::adopt(PySequence_Fast(vobs_obj, "via_obs_rects must be sequence"));
        if (!seq) return nullptr;
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
        PyObject** items = PySequence_Fast_ITEMS(seq.get());
        for (Py_ssize_t i = 0; i < n; ++i) {
            int gl = 0, owner_id = -1, is_cap = 0;
            double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
            if (!PyArg_ParseTuple(items[i], "iddddip", &gl, &x0, &y0, &x1, &y1, &owner_id, &is_cap)) {
                return nullptr;
            }
            c.add_via_obs_rect(gl, x0, y0, x1, y1, owner_id, is_cap != 0);
        }
    }

    if (mim_clr_obj != Py_None) {
        PyRef seq = PyRef::adopt(PySequence_Fast(mim_clr_obj, "mim_clear_rects must be sequence"));
        if (!seq) return nullptr;
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
        PyObject** items = PySequence_Fast_ITEMS(seq.get());
        for (Py_ssize_t i = 0; i < n; ++i) {
            int gi = 0, owner_id = -1;
            long long bx0 = 0, by0 = 0, bx1 = 0, by1 = 0, clr = 0;
            if (!PyArg_ParseTuple(items[i], "iLLLLLi", &gi, &bx0, &by0, &bx1, &by1, &clr, &owner_id)) {
                return nullptr;
            }
            c.add_mim_clear_rect(gi, bx0, by0, bx1, by1, clr, owner_id);
        }
    }

    if (mim_cut_obj != Py_None) {
        PyRef seq = PyRef::adopt(PySequence_Fast(mim_cut_obj, "mim_forbidden_cut_rects must be sequence"));
        if (!seq) return nullptr;
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
        PyObject** items = PySequence_Fast_ITEMS(seq.get());
        for (Py_ssize_t i = 0; i < n; ++i) {
            int gl = 0;
            long long bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
            if (!PyArg_ParseTuple(items[i], "iLLLL", &gl, &bx0, &by0, &bx1, &by1)) {
                return nullptr;
            }
            c.add_mim_forbidden_cut_rect(gl, bx0, by0, bx1, by1);
        }
    }
    Py_RETURN_NONE;
}

static PyObject* PyGrid_prepare_net(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, route_w_nm = 0, own_width_floor = 0, trunk_gi = 0;
    int add_unowned_pad_extra = 1, build_via_blocked = 1;
    PyObject *lg_obj = nullptr, *pins_obj = nullptr, *pin_xy_obj = nullptr;
    if (!PyArg_ParseTuple(
            args, "iiiOOOi|pp",
            &net_id, &route_w_nm, &own_width_floor, &lg_obj, &pins_obj, &pin_xy_obj,
            &trunk_gi, &add_unowned_pad_extra, &build_via_blocked)) {
        return nullptr;
    }
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;

    uint32_t allowed_mask = 0;
    if (!parse_layer_mask(lg_obj, c.n_layers, allowed_mask)) return nullptr;
    std::vector<int32_t> pin_idxs;
    if (!parse_cell_iter(pins_obj, c, pin_idxs, true)) return nullptr;

    std::vector<std::vector<std::pair<int64_t, int64_t>>> pin_xy_by_gi(static_cast<size_t>(c.n_layers));
    if (pin_xy_obj != Py_None && PyDict_Check(pin_xy_obj)) {
        PyObject* k = nullptr;
        PyObject* v = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(pin_xy_obj, &pos, &k, &v)) {
            const long gi = PyLong_AsLong(k);
            if (gi == -1 && PyErr_Occurred()) return nullptr;
            if (gi < 0 || gi >= c.n_layers) continue;
            PyRef seq = PyRef::adopt(PySequence_Fast(v, "pin_xy list"));
            if (!seq) return nullptr;
            const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
            PyObject** items = PySequence_Fast_ITEMS(seq.get());
            for (Py_ssize_t i = 0; i < n; ++i) {
                long long px = 0, py = 0;
                if (!PyArg_ParseTuple(items[i], "LL", &px, &py)) {
                    return nullptr;
                }
                pin_xy_by_gi[static_cast<size_t>(gi)].emplace_back(px, py);
            }
        }
    }

    c.prepare_net(net_id, route_w_nm, own_width_floor, allowed_mask, pin_idxs, pin_xy_by_gi, trunk_gi,
                  add_unowned_pad_extra != 0, build_via_blocked != 0);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_net_trunk(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, trunk_gi = 0;
    if (!PyArg_ParseTuple(args, "ii", &net_id, &trunk_gi)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    cp->set_net_trunk(net_id, trunk_gi);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_net_blocked_cells(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0;
    PyObject* cells_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iO", &net_id, &cells_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    if (net_id < 0) {
        PyErr_SetString(PyExc_ValueError, "net_id must be >= 0");
        return nullptr;
    }
    std::vector<int32_t> idxs;
    if (!parse_cell_iter(cells_obj, *cp, idxs, true)) return nullptr;
    cp->set_net_blocked_cells(net_id, idxs);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_net_via_blocked_cells(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0;
    PyObject* cells_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iO", &net_id, &cells_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    if (net_id < 0) {
        PyErr_SetString(PyExc_ValueError, "net_id must be >= 0");
        return nullptr;
    }
    if (c.n_layers <= 1 || cells_obj == Py_None) {
        c.set_net_via_blocked_cells(net_id, {});
        Py_RETURN_NONE;
    }
    PyRef iter = PyRef::adopt(PyObject_GetIter(cells_obj));
    if (!iter) return nullptr;
    std::vector<int32_t> vidxs;
    for (;;) {
        PyRef item = PyRef::adopt(PyIter_Next(iter.get()));
        if (!item) break;
        int ix = 0, iy = 0, gl = 0;
        if (!PyArg_ParseTuple(item.get(), "iii", &ix, &iy, &gl)) {
            return nullptr;
        }
        if (c.in_bounds(ix, iy) && gl >= 0 && gl < c.n_layers - 1) {
            vidxs.push_back(gl * c.nxy + iy * c.nx + ix);
        }
    }
    if (PyErr_Occurred()) return nullptr;
    c.set_net_via_blocked_cells(net_id, vidxs);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_is_blocked_cell(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, ix = 0, iy = 0, gi = 0, check_committed = 0;
    if (!PyArg_ParseTuple(args, "iiii|p", &net_id, &ix, &iy, &gi, &check_committed)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    if (cp->is_blocked_cell(net_id, ix, iy, gi, check_committed != 0)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject* PyGrid_is_committed_foreign(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, ix = 0, iy = 0, gi = 0;
    if (!PyArg_ParseTuple(args, "iiii", &net_id, &ix, &iy, &gi)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    if (cp->is_committed_foreign(net_id, ix, iy, gi)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject* PyGrid_foreign_committed_rects(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0;
    if (!PyArg_ParseTuple(args, "i", &net_id)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    const auto by_layer = cp->foreign_committed_rects(net_id);
    PyRef out = PyRef::adopt(PyDict_New());
    if (!out) return nullptr;
    for (size_t gi = 0; gi < by_layer.size(); ++gi) {
        const auto& rects = by_layer[gi];
        if (rects.empty()) continue;
        PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(rects.size())));
        if (!lst) return nullptr;
        for (size_t i = 0; i < rects.size(); ++i) {
            PyRef tup = PyRef::adopt(make_box64_tuple(rects[i]));
            if (!tup) return nullptr;
            PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), tup.release());
        }
        PyRef k = PyRef::adopt(PyLong_FromLong(static_cast<long>(gi)));
        if (!k || PyDict_SetItem(out.get(), k.get(), lst.get()) < 0) return nullptr;
    }
    return out.release();
}

static PyObject* PyGrid_get_blocked_cells(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, gi = 0;
    if (!PyArg_ParseTuple(args, "ii", &net_id, &gi)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    return make_xy_pair_list(cp->get_blocked_cells(net_id, gi));
}

static PyObject* PyGrid_get_via_infeasible_cells(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, gl = 0;
    if (!PyArg_ParseTuple(args, "ii", &net_id, &gl)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    return make_xy_pair_list(cp->get_via_infeasible_cells(net_id, gl));
}

static PyObject* PyGrid_pin_is_trapped(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, ix = 0, iy = 0, gi = 0;
    PyObject* lg_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iiiiO", &net_id, &ix, &iy, &gi, &lg_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    uint32_t mask = 0;
    if (!parse_layer_mask(lg_obj, cp->n_layers, mask)) return nullptr;
    if (cp->pin_is_trapped(net_id, ix, iy, gi, mask)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject* PyGrid_reset_history(PyNativePathFinderGrid* self, PyObject*) {
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    cp->reset_history();
    Py_RETURN_NONE;
}

static PyObject* PyGrid_begin_round(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject* active_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &active_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    std::vector<int32_t> active_ids;
    if (!parse_int_seq(active_obj, active_ids)) return nullptr;
    cp->begin_round(active_ids);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_mark_net_starting(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0;
    if (!PyArg_ParseTuple(args, "i", &net_id)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    cp->mark_net_starting(net_id);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_commit_net_route(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, ok = 1;
    PyObject *wire_obj = nullptr, *seed_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iOOp", &net_id, &wire_obj, &seed_obj, &ok)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    std::vector<int32_t> wires, seeds;
    if (!parse_cell_iter(wire_obj, *cp, wires, true)) return nullptr;
    if (!parse_cell_iter(seed_obj, *cp, seeds, true)) return nullptr;
    cp->commit_net_route(net_id, wires, seeds, ok != 0);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_reserve_mirror_corridor(PyNativePathFinderGrid* self, PyObject* args) {
    int net_b_id = 0;
    PyObject* refl_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iO", &net_b_id, &refl_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    PyRef out = PyRef::adopt(PySet_New(nullptr));
    if (!out) return nullptr;
    PyRef iter = PyRef::adopt(PyObject_GetIter(refl_obj));
    if (!iter) return nullptr;
    for (;;) {
        PyRef item = PyRef::adopt(PyIter_Next(iter.get()));
        if (!item) break;
        int32_t idx = -1;
        if (!parse_cell_tuple(item.get(), c, idx)) {
            return nullptr;
        }
        if (idx >= 0) {
            if (c.try_reserve_mirror_cell(net_b_id, idx)) {
                if (PySet_Add(out.get(), item.get()) < 0) return nullptr;
            }
        } else {
            // Out-of-bounds cell: preserve identical returned set for mirror_partial accounting
            if (PySet_Add(out.get(), item.get()) < 0) return nullptr;
        }
    }
    if (PyErr_Occurred()) return nullptr;
    return out.release();
}

static PyObject* PyGrid_finish_round(PyNativePathFinderGrid* self, PyObject* args) {
    double hist_fac = 1.0;
    if (!PyArg_ParseTuple(args, "d", &hist_fac)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;

    const FinishRoundResult fr = c.finish_round(hist_fac);

    PyRef py_overused = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(fr.overused.size())));
    if (!py_overused) return nullptr;
    for (size_t i = 0; i < fr.overused.size(); ++i) {
        PyRef tup = PyRef::adopt(make_cell_tuple(c, fr.overused[i]));
        if (!tup) return nullptr;
        PyList_SET_ITEM(py_overused.get(), static_cast<Py_ssize_t>(i), tup.release());
    }

    PyRef py_pad = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(fr.pad_contended.size())));
    if (!py_pad) return nullptr;
    for (size_t i = 0; i < fr.pad_contended.size(); ++i) {
        const auto& row = fr.pad_contended[i];
        PyRef cell_t = PyRef::adopt(make_cell_tuple(c, row.cell_idx));
        PyRef owners_t = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(row.foreign_owners.size())));
        PyRef owner_id_py = PyRef::adopt(PyLong_FromLong(row.metal_owner_id));
        PyRef entry = PyRef::adopt(PyTuple_New(3));
        if (!cell_t || !owners_t || !owner_id_py || !entry) return nullptr;
        for (size_t k = 0; k < row.foreign_owners.size(); ++k) {
            PyRef fo = PyRef::adopt(PyLong_FromLong(row.foreign_owners[k]));
            if (!fo) return nullptr;
            PyTuple_SET_ITEM(owners_t.get(), static_cast<Py_ssize_t>(k), fo.release());
        }
        PyTuple_SET_ITEM(entry.get(), 0, cell_t.release());
        PyTuple_SET_ITEM(entry.get(), 1, owner_id_py.release());
        PyTuple_SET_ITEM(entry.get(), 2, owners_t.release());
        PyList_SET_ITEM(py_pad.get(), static_cast<Py_ssize_t>(i), entry.release());
    }

    PyRef res = PyRef::adopt(PyTuple_New(3));
    if (!res) return nullptr;
    PyTuple_SET_ITEM(res.get(), 0, PyBool_FromLong(fr.converged ? 1 : 0));
    PyTuple_SET_ITEM(res.get(), 1, py_overused.release());
    PyTuple_SET_ITEM(res.get(), 2, py_pad.release());
    return res.release();
}

static PyObject* PyGrid_route_one_net_soft(PyNativePathFinderGrid* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {
        const_cast<char*>("net_id"),
        const_cast<char*>("pin_cells"),
        const_cast<char*>("layer_gis"),
        const_cast<char*>("seed_cells"),
        const_cast<char*>("pres_fac"),
        const_cast<char*>("hist_fac"),
        const_cast<char*>("trunk_gi"),
        const_cast<char*>("use_via_cost"),
        const_cast<char*>("dir_pref"),
        const_cast<char*>("mirror_cells"),
        const_cast<char*>("mirror_penalty"),
        const_cast<char*>("max_expansions"),
        const_cast<char*>("window_ladder"),
        const_cast<char*>("use_pf_cost"),
        const_cast<char*>("check_committed"),
        const_cast<char*>("checkpoint_cb"),
        const_cast<char*>("checkpoint_interval"),
        nullptr
    };
    int net_id = 0, trunk_gi = 0, use_via_cost = 0;
    int window_ladder = 1, use_pf_cost = 1, check_committed = 0;
    double pres_fac = 0.5, hist_fac = 1.0, mirror_penalty = 50.0;
    long long max_expansions = 2000000;
    long long checkpoint_interval = kDefaultCheckpointInterval;
    PyObject *pins_obj = nullptr, *lg_obj = nullptr, *seeds_obj = Py_None;
    PyObject *dir_pref_obj = Py_None, *mirror_obj = Py_None, *chk_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "iOO|OddipOOdLpppOL", kwlist,
            &net_id, &pins_obj, &lg_obj, &seeds_obj,
            &pres_fac, &hist_fac, &trunk_gi, &use_via_cost,
            &dir_pref_obj, &mirror_obj, &mirror_penalty,
            &max_expansions, &window_ladder, &use_pf_cost, &check_committed,
            &chk_obj, &checkpoint_interval)) {
        return nullptr;
    }

    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;

    uint32_t allowed_mask = 0;
    if (!parse_layer_mask(lg_obj, c.n_layers, allowed_mask)) return nullptr;

    std::vector<int32_t> pin_cells, seed_cells, mirror_cells;
    if (!parse_cell_iter(pins_obj, c, pin_cells, false)) return nullptr;
    if (!parse_cell_iter(seeds_obj, c, seed_cells, true)) return nullptr;
    const bool has_mirror = (mirror_obj != nullptr && mirror_obj != Py_None);
    if (has_mirror && !parse_cell_iter(mirror_obj, c, mirror_cells, true)) return nullptr;

    bool has_dir_pref = false;
    std::vector<DirPrefEntry> dir_pref;
    if (!parse_dir_pref(dir_pref_obj, c.n_layers, has_dir_pref, dir_pref)) return nullptr;

    PyCheckpointAdapter chk_adapter;
    if (!chk_adapter.init(chk_obj)) return nullptr;

    std::vector<int32_t> wire_idxs;
    const int rc = c.route_one_net(
        net_id, check_committed != 0, use_pf_cost != 0,
        pin_cells, allowed_mask, seed_cells,
        pres_fac, hist_fac, trunk_gi, use_via_cost != 0,
        has_dir_pref, dir_pref,
        has_mirror, mirror_cells, mirror_penalty,
        window_ladder != 0, max_expansions,
        chk_adapter.hook(), checkpoint_interval, wire_idxs
    );
    if (rc < 0) return nullptr;  // Python exception already set by checkpoint callback

    PyRef wire_set = PyRef::adopt(PySet_New(nullptr));
    if (!wire_set) return nullptr;
    if (rc > 0) {
        for (int32_t idx : wire_idxs) {
            PyRef tup = PyRef::adopt(make_cell_tuple(c, idx));
            if (!tup || PySet_Add(wire_set.get(), tup.get()) < 0) return nullptr;
        }
    }
    PyRef ret = PyRef::adopt(PyTuple_New(2));
    if (!ret) return nullptr;
    PyTuple_SET_ITEM(ret.get(), 0, wire_set.release());
    PyTuple_SET_ITEM(ret.get(), 1, PyBool_FromLong(rc > 0 ? 1 : 0));
    return ret.release();
}

static PyObject* PyGrid_astar(PyNativePathFinderGrid* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {
        const_cast<char*>("net_id"),
        const_cast<char*>("sources"),
        const_cast<char*>("target"),
        const_cast<char*>("layer_gis"),
        const_cast<char*>("pres_fac"),
        const_cast<char*>("hist_fac"),
        const_cast<char*>("trunk_gi"),
        const_cast<char*>("use_via_cost"),
        const_cast<char*>("dir_pref"),
        const_cast<char*>("dir_runs"),
        const_cast<char*>("mirror_cells"),
        const_cast<char*>("mirror_penalty"),
        const_cast<char*>("window_slack"),
        const_cast<char*>("max_expansions"),
        const_cast<char*>("use_pf_cost"),
        const_cast<char*>("check_committed"),
        const_cast<char*>("checkpoint_cb"),
        const_cast<char*>("checkpoint_interval"),
        nullptr
    };
    int net_id = 0, trunk_gi = 0, use_via_cost = 0, use_pf_cost = 1, check_committed = 0;
    double pres_fac = 0.5, hist_fac = 1.0, mirror_penalty = 50.0;
    long long max_expansions = 2000000;
    long long checkpoint_interval = kDefaultCheckpointInterval;
    PyObject *sources_obj = nullptr, *target_obj = nullptr, *lg_obj = nullptr;
    PyObject *dir_pref_obj = Py_None, *dir_runs_obj = Py_None, *mirror_obj = Py_None;
    PyObject *slack_obj = Py_None, *chk_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "iOOO|ddipOOOdOLppOL", kwlist,
            &net_id, &sources_obj, &target_obj, &lg_obj,
            &pres_fac, &hist_fac, &trunk_gi, &use_via_cost,
            &dir_pref_obj, &dir_runs_obj, &mirror_obj, &mirror_penalty,
            &slack_obj, &max_expansions, &use_pf_cost, &check_committed,
            &chk_obj, &checkpoint_interval)) {
        return nullptr;
    }

    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;

    uint32_t allowed_mask = 0;
    if (!parse_layer_mask(lg_obj, c.n_layers, allowed_mask)) return nullptr;
    std::vector<int32_t> sources;
    if (!parse_cell_iter(sources_obj, c, sources, false)) return nullptr;
    int32_t target_idx = -1;
    if (!parse_cell_tuple(target_obj, c, target_idx) || target_idx < 0) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "target out of bounds");
        return nullptr;
    }

    bool has_dir_pref = false;
    std::vector<DirPrefEntry> dir_pref;
    if (!parse_dir_pref(dir_pref_obj, c.n_layers, has_dir_pref, dir_pref)) return nullptr;

    uint32_t dr_gen = 0;
    bool has_dir_runs = false;
    if (has_dir_pref && dir_runs_obj != nullptr && dir_runs_obj != Py_None &&
        PyDict_Check(dir_runs_obj) && PyDict_Size(dir_runs_obj) > 0) {
        if (c.dir_runs_stamp.empty()) c.next_dir_gen();
        dr_gen = c.next_gen(c.dir_runs_stamp, c.dir_runs_gen);
        has_dir_runs = true;
        PyObject* k = nullptr;
        PyObject* v = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dir_runs_obj, &pos, &k, &v)) {
            int32_t idx = -1;
            if (!parse_cell_tuple(k, c, idx)) return nullptr;
            const long rv = PyLong_AsLong(v);
            if (rv == -1 && PyErr_Occurred()) return nullptr;
            if (idx >= 0) {
                c.dir_runs_stamp[static_cast<size_t>(idx)] = dr_gen;
                c.dir_runs_arr[static_cast<size_t>(idx)] = static_cast<int32_t>(rv);
            }
        }
    }

    const bool has_mirror = (mirror_obj != nullptr && mirror_obj != Py_None);
    uint32_t mgen = 0;
    if (has_mirror) {
        std::vector<int32_t> mcells;
        if (!parse_cell_iter(mirror_obj, c, mcells, true)) return nullptr;
        mgen = c.next_gen(c.mirror_stamp, c.mirror_gen);
        for (int32_t mc : mcells) c.mirror_stamp[static_cast<size_t>(mc)] = mgen;
    }

    bool has_win = false;
    int32_t win_x0 = 0, win_y0 = 0, win_x1 = 0, win_y1 = 0;
    if (slack_obj != nullptr && slack_obj != Py_None) {
        const long sl = PyLong_AsLong(slack_obj);
        if (sl == -1 && PyErr_Occurred()) return nullptr;
        has_win = true;
        int32_t tx = 0, ty = 0;
        c.unpack_xy(target_idx, tx, ty);
        int32_t min_x = tx, max_x = tx, min_y = ty, max_y = ty;
        for (int32_t s : sources) {
            int32_t sx = 0, sy = 0;
            c.unpack_xy(s, sx, sy);
            if (sx < min_x) min_x = sx;
            if (sx > max_x) max_x = sx;
            if (sy < min_y) min_y = sy;
            if (sy > max_y) max_y = sy;
        }
        win_x0 = min_x - static_cast<int32_t>(sl);
        win_y0 = min_y - static_cast<int32_t>(sl);
        win_x1 = max_x + static_cast<int32_t>(sl);
        win_y1 = max_y + static_cast<int32_t>(sl);
    }

    PyCheckpointAdapter chk_adapter;
    if (!chk_adapter.init(chk_obj)) return nullptr;

    std::vector<int32_t> path;
    int rc = 0;
    if (has_dir_pref) {
        rc = c.astar_directed(
            net_id, check_committed != 0, sources, target_idx, allowed_mask,
            dir_pref, has_dir_runs, dr_gen,
            use_pf_cost != 0, pres_fac, hist_fac, trunk_gi, use_via_cost != 0,
            false, 0, has_mirror, mgen, mirror_penalty, 0,
            has_win, win_x0, win_y0, win_x1, win_y1,
            max_expansions, chk_adapter.hook(), checkpoint_interval, path
        );
    } else {
        rc = c.astar_undirected(
            net_id, check_committed != 0, sources, target_idx, allowed_mask,
            use_pf_cost != 0, pres_fac, hist_fac, trunk_gi, use_via_cost != 0,
            false, 0, has_mirror, mgen, mirror_penalty, 0,
            has_win, win_x0, win_y0, win_x1, win_y1,
            max_expansions, chk_adapter.hook(), checkpoint_interval, path
        );
    }
    if (rc < 0) return nullptr;
    if (rc == 0) Py_RETURN_NONE;

    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(path.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < path.size(); ++i) {
        PyRef tup = PyRef::adopt(make_cell_tuple(c, path[i]));
        if (!tup) return nullptr;
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    return out.release();
}

static PyObject* PyGrid_reset_committed(PyNativePathFinderGrid* self, PyObject*) {
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    cp->reset_committed();
    Py_RETURN_NONE;
}

static PyObject* PyGrid_commit_maze_cells(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, overwrite = 1;
    PyObject* cells_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iO|p", &net_id, &cells_obj, &overwrite)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    std::vector<int32_t> idxs;
    if (!parse_cell_iter(cells_obj, *cp, idxs, true)) return nullptr;
    cp->commit_maze_cells(net_id, idxs, overwrite != 0);
    Py_RETURN_NONE;
}

static PyObject* PyGrid_commit_maze_halo(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, wire_w_nm = 0, extra_sp_nm = 0, max_other_w_nm = 0;
    int use_taper = 0, via_partner_half_nm = 0;
    int overwrite_wires = 1, include_via_halo = 1;
    PyObject* wire_obj = nullptr;
    if (!PyArg_ParseTuple(
            args, "iOiiipi|pp",
            &net_id, &wire_obj, &wire_w_nm, &extra_sp_nm, &max_other_w_nm,
            &use_taper, &via_partner_half_nm, &overwrite_wires, &include_via_halo)) {
        return nullptr;
    }
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    std::vector<int32_t> wires;
    if (!parse_cell_iter(wire_obj, *cp, wires, true)) return nullptr;
    cp->commit_maze_halo(
        net_id, wires, wire_w_nm, extra_sp_nm, max_other_w_nm,
        use_taper != 0, via_partner_half_nm, overwrite_wires != 0, include_via_halo != 0
    );
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_occupancy(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject* dict_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &dict_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    c.clear_occupancy();
    if (dict_obj == Py_None) Py_RETURN_NONE;
    if (!PyDict_Check(dict_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected dict");
        return nullptr;
    }
    PyObject* k = nullptr;
    PyObject* v = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict_obj, &pos, &k, &v)) {
        int32_t idx = -1;
        if (!parse_cell_tuple(k, c, idx)) return nullptr;
        const long cnt = PyLong_AsLong(v);
        if (cnt == -1 && PyErr_Occurred()) return nullptr;
        if (idx >= 0 && cnt != 0) {
            c.set_occupancy_cell(idx, static_cast<int32_t>(cnt));
        }
    }
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_history(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject* dict_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &dict_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    c.clear_history_values();
    if (dict_obj == Py_None) Py_RETURN_NONE;
    if (!PyDict_Check(dict_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected dict");
        return nullptr;
    }
    PyObject* k = nullptr;
    PyObject* v = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict_obj, &pos, &k, &v)) {
        int32_t idx = -1;
        if (!parse_cell_tuple(k, c, idx)) return nullptr;
        const double hv = PyFloat_AsDouble(v);
        if (PyErr_Occurred()) return nullptr;
        if (idx >= 0) c.history[static_cast<size_t>(idx)] = hv;
    }
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_via_pad_history(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject* dict_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &dict_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    c.clear_via_pad_history_values();
    if (dict_obj == Py_None) Py_RETURN_NONE;
    if (!PyDict_Check(dict_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected dict");
        return nullptr;
    }
    PyObject* k = nullptr;
    PyObject* v = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict_obj, &pos, &k, &v)) {
        int ix = 0, iy = 0, gl = 0;
        if (!PyArg_ParseTuple(k, "iii", &ix, &iy, &gl)) return nullptr;
        const double hv = PyFloat_AsDouble(v);
        if (PyErr_Occurred()) return nullptr;
        if (c.in_bounds(ix, iy) && gl >= 0 && gl < c.n_layers - 1) {
            c.via_pad_history[static_cast<size_t>(gl * c.nxy + iy * c.nx + ix)] = hv;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* PyGrid_set_pad_keepout_counts(PyNativePathFinderGrid* self, PyObject* args) {
    PyObject* dict_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &dict_obj)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    c.clear_pad_keepout_counts();
    if (dict_obj == Py_None) Py_RETURN_NONE;
    if (!PyDict_Check(dict_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected dict");
        return nullptr;
    }
    PyObject* k = nullptr;
    PyObject* v = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict_obj, &pos, &k, &v)) {
        int32_t idx = -1;
        if (!parse_cell_tuple(k, c, idx)) return nullptr;
        const long cnt = PyLong_AsLong(v);
        if (cnt == -1 && PyErr_Occurred()) return nullptr;
        if (idx >= 0 && cnt != 0) {
            c.set_pad_keepout_count_cell(idx, static_cast<int32_t>(cnt));
        }
    }
    Py_RETURN_NONE;
}

static PyObject* PyGrid_net_footprint(PyNativePathFinderGrid* self, PyObject* args) {
    int net_id = 0, route_w_nm = 0, use_neck = 1;
    PyObject* wire_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iOi|p", &net_id, &wire_obj, &route_w_nm, &use_neck)) return nullptr;
    PathFinderGridCore* cp = require_grid_core(self);
    if (!cp) return nullptr;
    PathFinderGridCore& c = *cp;
    std::vector<int32_t> wires;
    if (!parse_cell_iter(wire_obj, c, wires, true)) return nullptr;
    const std::vector<int32_t> fp = c.net_footprint(net_id, wires, route_w_nm, use_neck != 0);
    PyRef out = PyRef::adopt(PySet_New(nullptr));
    if (!out) return nullptr;
    for (int32_t kidx : fp) {
        PyRef tup = PyRef::adopt(make_cell_tuple(c, kidx));
        if (!tup || PySet_Add(out.get(), tup.get()) < 0) return nullptr;
    }
    return out.release();
}

static PyMethodDef PyGrid_methods[] = {
    {"set_over_active", reinterpret_cast<PyCFunction>(PyGrid_set_over_active), METH_VARARGS,
     "Rasterize over-active diffusion rectangles into 2D bitset."},
    {"set_static_obstacles", reinterpret_cast<PyCFunction>(PyGrid_set_static_obstacles), METH_VARARGS,
     "Upload static metal/via/MIM obstacles and optional floorplan envelope."},
    {"prepare_net", reinterpret_cast<PyCFunction>(PyGrid_prepare_net), METH_VARARGS,
     "Build per-net blocked bitset, via-infeasible bitset, neck bits, and stencils."},
    {"set_net_trunk", reinterpret_cast<PyCFunction>(PyGrid_set_net_trunk), METH_VARARGS,
     "Set trunk target graph layer for a net."},
    {"set_net_blocked_cells", reinterpret_cast<PyCFunction>(PyGrid_set_net_blocked_cells), METH_VARARGS,
     "Directly set blocked cells for a net."},
    {"set_net_via_blocked_cells", reinterpret_cast<PyCFunction>(PyGrid_set_net_via_blocked_cells), METH_VARARGS,
     "Directly set infeasible via transitions (ix, iy, gl) for a net."},
    {"is_blocked_cell", reinterpret_cast<PyCFunction>(PyGrid_is_blocked_cell), METH_VARARGS,
     "Return True iff (ix, iy, gi) is hard-blocked, reserved, or (optionally) committed by another net."},
    {"is_committed_foreign", reinterpret_cast<PyCFunction>(PyGrid_is_committed_foreign), METH_VARARGS,
     "Return True iff (ix, iy, gi) is committed to a foreign net."},
    {"foreign_committed_rects", reinterpret_cast<PyCFunction>(PyGrid_foreign_committed_rects), METH_VARARGS,
     "Return {gi: [(x0, y0, x1, y1), ...]} of foreign committed metal keep-out rects."},
    {"get_blocked_cells", reinterpret_cast<PyCFunction>(PyGrid_get_blocked_cells), METH_VARARGS,
     "Return list of blocked (ix, iy) cells for (net_id, gi)."},
    {"get_via_infeasible_cells", reinterpret_cast<PyCFunction>(PyGrid_get_via_infeasible_cells), METH_VARARGS,
     "Return list of infeasible (ix, iy) via cells for (net_id, gl)."},
    {"pin_is_trapped", reinterpret_cast<PyCFunction>(PyGrid_pin_is_trapped), METH_VARARGS,
     "Exact 64-cell pocket BFS matching PathFinderEngine._pin_is_trapped."},
    {"reset_history", reinterpret_cast<PyCFunction>(PyGrid_reset_history), METH_NOARGS,
     "Reset PathFinder history and via_pad_history arrays."},
    {"begin_round", reinterpret_cast<PyCFunction>(PyGrid_begin_round), METH_VARARGS,
     "Sparsely reset per-round occupancy/keepout/reservation and seed unrouted pin counts."},
    {"mark_net_starting", reinterpret_cast<PyCFunction>(PyGrid_mark_net_starting), METH_VARARGS,
     "Decrement unrouted pin counts for a net entering search."},
    {"commit_net_route", reinterpret_cast<PyCFunction>(PyGrid_commit_net_route), METH_VARARGS,
     "Commit routed wire cells into occupancy, metal_owner, and via-pad keepout."},
    {"reserve_mirror_corridor", reinterpret_cast<PyCFunction>(PyGrid_reserve_mirror_corridor), METH_VARARGS,
     "Reserve unoccupied reflected cells as hard keep-out corridor for net_b."},
    {"finish_round", reinterpret_cast<PyCFunction>(PyGrid_finish_round), METH_VARARGS,
     "Collect overused & pad-contended cells and bump history if unconverged."},
    {"route_one_net_soft", reinterpret_cast<PyCFunction>(PyGrid_route_one_net_soft), METH_VARARGS | METH_KEYWORDS,
     "Run full sink-by-sink multi-source 3D A* with 4-level window ladder and checkpoints."},
    {"astar", reinterpret_cast<PyCFunction>(PyGrid_astar), METH_VARARGS | METH_KEYWORDS,
     "Run single multi-source 3D A* (undirected or directed comb contract)."},
    {"reset_committed", reinterpret_cast<PyCFunction>(PyGrid_reset_committed), METH_NOARGS,
     "Sparsely clear Maze committed owner map."},
    {"commit_maze_cells", reinterpret_cast<PyCFunction>(PyGrid_commit_maze_cells), METH_VARARGS,
     "Commit cells into Maze committed owner map."},
    {"commit_maze_halo", reinterpret_cast<PyCFunction>(PyGrid_commit_maze_halo), METH_VARARGS,
     "Commit wire cells + width/spacing halo + via-pad halo into Maze committed map."},
    {"set_occupancy", reinterpret_cast<PyCFunction>(PyGrid_set_occupancy), METH_VARARGS,
     "Directly populate occupancy map for testing."},
    {"set_history", reinterpret_cast<PyCFunction>(PyGrid_set_history), METH_VARARGS,
     "Directly populate history map for testing."},
    {"set_via_pad_history", reinterpret_cast<PyCFunction>(PyGrid_set_via_pad_history), METH_VARARGS,
     "Directly populate via_pad_history map for testing."},
    {"set_pad_keepout_counts", reinterpret_cast<PyCFunction>(PyGrid_set_pad_keepout_counts), METH_VARARGS,
     "Directly populate pad_keepout_count map for testing."},
    {"net_footprint", reinterpret_cast<PyCFunction>(PyGrid_net_footprint), METH_VARARGS,
     "Compute tapered clearance footprint cells for a net."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject make_native_grid_type() {
    PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
    t.tp_name = "clay._native._pathfinder_cpp.NativePathFinderGrid";
    t.tp_basicsize = sizeof(PyNativePathFinderGrid);
    t.tp_dealloc = reinterpret_cast<destructor>(PyGrid_dealloc);
    t.tp_flags = Py_TPFLAGS_DEFAULT;
    t.tp_doc = "C++20 flat 3D grid PathFinder & Maze A* engine kernel.";
    t.tp_methods = PyGrid_methods;
    t.tp_init = reinterpret_cast<initproc>(PyGrid_init);
    t.tp_new = PyGrid_new;
    return t;
}

static PyTypeObject PyNativePathFinderGridType = make_native_grid_type();

// ===================== NativeRectClipper =====================

struct PyNativeRectClipper {
    PyObject_HEAD
    RectClipperCore* core;
};

static PyObject* PyClipper_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeRectClipper*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->core = new (std::nothrow) RectClipperCore();
    if (!self->core) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return reinterpret_cast<PyObject*>(self);
}

static void PyClipper_dealloc(PyNativeRectClipper* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static int PyClipper_init(PyNativeRectClipper* self, PyObject* args, PyObject*) {
    PyObject* dict_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &dict_obj)) return -1;
    if (!PyDict_Check(dict_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected dict of {(layer, dt): [(x0, y0, x1, y1), ...]}");
        return -1;
    }
    RectClipperCore& c = *self->core;
    c.clear();
    PyObject* k = nullptr;
    PyObject* v = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict_obj, &pos, &k, &v)) {
        if (!PyTuple_Check(k) || PyTuple_GET_SIZE(k) != 2) {
            PyErr_SetString(PyExc_TypeError, "layer key must be 2-tuple (layer, datatype)");
            return -1;
        }
        const int64_t lyr = PyLong_AsLongLong(PyTuple_GET_ITEM(k, 0));
        const int64_t dt = PyLong_AsLongLong(PyTuple_GET_ITEM(k, 1));
        if (PyErr_Occurred()) return -1;
        PyRef seq = PyRef::adopt(PySequence_Fast(v, "holes must be a sequence"));
        if (!seq) return -1;
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
        if (n > 0) {
            std::vector<Box64> holes;
            holes.reserve(static_cast<size_t>(n));
            PyObject** items = PySequence_Fast_ITEMS(seq.get());
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject* item = items[i];
                if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 4) {
                    PyErr_SetString(PyExc_TypeError, "hole rect must be 4-tuple");
                    return -1;
                }
                const int64_t hx0 = PyLong_AsLongLong(PyTuple_GET_ITEM(item, 0));
                const int64_t hy0 = PyLong_AsLongLong(PyTuple_GET_ITEM(item, 1));
                const int64_t hx1 = PyLong_AsLongLong(PyTuple_GET_ITEM(item, 2));
                const int64_t hy1 = PyLong_AsLongLong(PyTuple_GET_ITEM(item, 3));
                if (PyErr_Occurred()) {
                    return -1;
                }
                holes.push_back(Box64{hx0, hy0, hx1, hy1});
            }
            c.add_layer_holes(lyr, dt, std::move(holes));
        }
    }
    return 0;
}

static PyObject* make_rect_list(const std::vector<Box64>& rects) {
    PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(rects.size())));
    if (!lst) return nullptr;
    for (size_t i = 0; i < rects.size(); ++i) {
        PyRef tup = PyRef::adopt(make_box64_tuple(rects[i]));
        if (!tup) return nullptr;
        PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    return lst.release();
}

static PyObject* PyClipper_clip_emit(PyNativeRectClipper* self, PyObject* args) {
    long long lyr = 0, dt = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0, min_w = 1;
    if (!PyArg_ParseTuple(args, "(LL)LLLLL", &lyr, &dt, &x0, &y0, &x1, &y1, &min_w)) return nullptr;
    const auto& rects = self->core->clip_emit(
        static_cast<int64_t>(lyr), static_cast<int64_t>(dt),
        static_cast<int64_t>(x0), static_cast<int64_t>(y0),
        static_cast<int64_t>(x1), static_cast<int64_t>(y1),
        static_cast<int64_t>(min_w)
    );
    return make_rect_list(rects);
}

static PyObject* PyClipper_subtract_rects(PyNativeRectClipper* self, PyObject* args) {
    long long lyr = 0, dt = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!PyArg_ParseTuple(args, "(LL)LLLL", &lyr, &dt, &x0, &y0, &x1, &y1)) return nullptr;
    const auto& rects = self->core->subtract_rects(
        static_cast<int64_t>(lyr), static_cast<int64_t>(dt),
        static_cast<int64_t>(x0), static_cast<int64_t>(y0),
        static_cast<int64_t>(x1), static_cast<int64_t>(y1)
    );
    return make_rect_list(rects);
}

static PyMethodDef PyClipper_methods[] = {
    {"clip_emit", reinterpret_cast<PyCFunction>(PyClipper_clip_emit), METH_VARARGS,
     "Clip emitted rectangle against layer foreign holes with exact sub-min filter and spine retry."},
    {"subtract_rects", reinterpret_cast<PyCFunction>(PyClipper_subtract_rects), METH_VARARGS,
     "Exact axis-aligned rectangle subtraction against layer foreign holes."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject make_native_clipper_type() {
    PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
    t.tp_name = "clay._native._pathfinder_cpp.NativeRectClipper";
    t.tp_basicsize = sizeof(PyNativeRectClipper);
    t.tp_dealloc = reinterpret_cast<destructor>(PyClipper_dealloc);
    t.tp_flags = Py_TPFLAGS_DEFAULT;
    t.tp_doc = "C++20 axis-aligned rectangle clipper matching via_model.subtract_rects and engine_maze._emit.";
    t.tp_methods = PyClipper_methods;
    t.tp_init = reinterpret_cast<initproc>(PyClipper_init);
    t.tp_new = PyClipper_new;
    return t;
}

static PyTypeObject PyNativeRectClipperType = make_native_clipper_type();

static PyModuleDef pathfinder_cpp_module = {
    PyModuleDef_HEAD_INIT,
    "_pathfinder_cpp",
    "C++20 native 3D grid PathFinder & Maze routing kernels.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__pathfinder_cpp(void) {
    if (PyType_Ready(&PyNativePathFinderGridType) < 0) {
        return nullptr;
    }
    if (PyType_Ready(&PyNativeRectClipperType) < 0) {
        return nullptr;
    }
    PyRef m = PyRef::adopt(PyModule_Create(&pathfinder_cpp_module));
    if (!m) {
        return nullptr;
    }
    Py_INCREF(&PyNativePathFinderGridType);
    if (PyModule_AddObject(m.get(), "NativePathFinderGrid",
                           reinterpret_cast<PyObject*>(&PyNativePathFinderGridType)) < 0) {
        Py_DECREF(&PyNativePathFinderGridType);
        return nullptr;
    }
    Py_INCREF(&PyNativeRectClipperType);
    if (PyModule_AddObject(m.get(), "NativeRectClipper",
                           reinterpret_cast<PyObject*>(&PyNativeRectClipperType)) < 0) {
        Py_DECREF(&PyNativeRectClipperType);
        return nullptr;
    }
    return m.release();
}
