// CPython binding TU for Gridless spatial index, legality, congestion,
// 2D potential / layer hmin, distance transform, history grid, and NativeEdgeContext.

#include "clay_pnr/route/gridless_py_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clay_pnr::route::py_bind {

// ---------------------------------------------------------------------------
// 1. SpatialIndexCore & PyNativeSpatialIndex
// ---------------------------------------------------------------------------

SpatialIndexCore::SpatialIndexCore() {
    owner_sort_keys.emplace_back("");  // owner_id 0 == None -> ""
}

SpatialIndexCore::~SpatialIndexCore() {
    for (auto& e : py_entries) {
        Py_XDECREF(e.owner_obj);
        Py_XDECREF(e.rect_obj);
        Py_XDECREF(e.item_tuple);
        Py_XDECREF(e.norm_tuple);
    }
    for (auto& kv : custom_owners) {
        Py_XDECREF(kv.first);
    }
}

uint32_t SpatialIndexCore::intern_owner(PyObject* owner, bool& err) {
    err = false;
    if (owner == Py_None) return 0;
    if (PyUnicode_CheckExact(owner)) {
        Py_ssize_t sz = 0;
        const char* data = PyUnicode_AsUTF8AndSize(owner, &sz);
        if (!data) {
            err = true;
            return 0;
        }
        std::string s(data, static_cast<size_t>(sz));
        auto it = str_owner_to_id.find(s);
        if (it != str_owner_to_id.end()) return it->second;
        const uint32_t id = next_owner_id++;
        str_owner_to_id.emplace(s, id);
        owner_sort_keys.push_back(std::move(s));
        return id;
    }
    for (const auto& kv : custom_owners) {
        const int eq = PyObject_RichCompareBool(kv.first, owner, Py_EQ);
        if (eq < 0) {
            err = true;
            return 0;
        }
        if (eq == 1) return kv.second;
    }
    PyRef s_obj = PyRef::adopt(PyObject_Str(owner));
    if (!s_obj) {
        err = true;
        return 0;
    }
    Py_ssize_t sz = 0;
    const char* data = PyUnicode_AsUTF8AndSize(s_obj.get(), &sz);
    if (!data) {
        err = true;
        return 0;
    }
    const uint32_t id = next_owner_id++;
    Py_INCREF(owner);
    custom_owners.emplace_back(owner, id);
    owner_sort_keys.emplace_back(data, static_cast<size_t>(sz));
    return id;
}

uint32_t SpatialIndexCore::lookup_owner(PyObject* owner, bool& err) const {
    err = false;
    if (!owner) return UINT32_MAX;
    if (owner == Py_None) return 0;
    if (PyUnicode_CheckExact(owner)) {
        Py_ssize_t sz = 0;
        const char* data = PyUnicode_AsUTF8AndSize(owner, &sz);
        if (!data) {
            err = true;
            return UINT32_MAX;
        }
        std::string s(data, static_cast<size_t>(sz));
        auto it = str_owner_to_id.find(s);
        return (it == str_owner_to_id.end()) ? UINT32_MAX : it->second;
    }
    for (const auto& kv : custom_owners) {
        const int eq = PyObject_RichCompareBool(kv.first, owner, Py_EQ);
        if (eq < 0) {
            err = true;
            return UINT32_MAX;
        }
        if (eq == 1) return kv.second;
    }
    return UINT32_MAX;
}

bool SpatialIndexCore::add(PyObject* owner, PyObject* rect_obj) {
    Box64 nb;
    if (!parse_rect_norm(rect_obj, nb)) return false;

    bool err = false;
    const uint32_t owner_id = intern_owner(owner, err);
    if (err) return false;

    PyRef item_tuple = PyRef::adopt(PyTuple_New(2));
    if (!item_tuple) return false;
    Py_INCREF(owner);
    Py_INCREF(rect_obj);
    PyTuple_SET_ITEM(item_tuple.get(), 0, owner);
    PyTuple_SET_ITEM(item_tuple.get(), 1, rect_obj);

    PyRef norm_tuple;
    if (PyTuple_CheckExact(rect_obj) && PyTuple_GET_SIZE(rect_obj) == 4) {
        const int64_t rx0 = PyLong_AsLongLong(PyTuple_GET_ITEM(rect_obj, 0));
        const int64_t ry0 = PyLong_AsLongLong(PyTuple_GET_ITEM(rect_obj, 1));
        const int64_t rx1 = PyLong_AsLongLong(PyTuple_GET_ITEM(rect_obj, 2));
        const int64_t ry1 = PyLong_AsLongLong(PyTuple_GET_ITEM(rect_obj, 3));
        if (rx0 == nb.x0 && ry0 == nb.y0 && rx1 == nb.x1 && ry1 == nb.y1) {
            norm_tuple = PyRef::borrow(rect_obj);
        }
    }
    if (!norm_tuple) {
        norm_tuple = PyRef::adopt(make_box64_tuple(nb));
        if (!norm_tuple) return false;
    }

    Py_INCREF(owner);
    Py_INCREF(rect_obj);

    add_entry(nb, owner_id, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rect_obj)),
              py_to_tag(owner));
    py_entries.push_back(PySpatialSidecar{
        owner, rect_obj, item_tuple.release(), norm_tuple.release()
    });
    return true;
}

namespace {

void NativeSpatialIndex_dealloc(PyNativeSpatialIndex* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeSpatialIndex_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeSpatialIndex*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->core = new SpatialIndexCore();
    return reinterpret_cast<PyObject*>(self);
}

int NativeSpatialIndex_init(PyNativeSpatialIndex* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"cell", nullptr};
    long long cell = 6000;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|L", const_cast<char**>(kwlist), &cell)) {
        return -1;
    }
    self->core->cell = (cell < 1) ? 1 : static_cast<int64_t>(cell);
    return 0;
}

PyObject* NativeSpatialIndex_get_cell(PyNativeSpatialIndex* self, void*) {
    return PyLong_FromLongLong(self->core->cell);
}

PyObject* NativeSpatialIndex_get_size(PyNativeSpatialIndex* self, void*) {
    return PyLong_FromSize_t(self->core->entries.size());
}

PyObject* NativeSpatialIndex_add(PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "add(owner, rect) expects exactly 2 arguments");
        return nullptr;
    }
    if (!self->core->add(args[0], args[1])) return nullptr;
    Py_RETURN_NONE;
}

PyObject* NativeSpatialIndex_export_buckets(PyNativeSpatialIndex* self, PyObject*) {
    const auto& core = *self->core;
    PyRef dict = PyRef::adopt(PyDict_New());
    if (!dict) return nullptr;
    for (const auto& b : core.buckets) {
        PyRef key = PyRef::adopt(make_xy_tuple(b.bx, b.by));
        if (!key) return nullptr;

        PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(b.items.size())));
        if (!lst) return nullptr;
        for (size_t i = 0; i < b.items.size(); ++i) {
            PyObject* tup = core.py_entries[b.items[i].entry_idx].item_tuple;
            Py_INCREF(tup);
            PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), tup);
        }
        if (PyDict_SetItem(dict.get(), key.get(), lst.get()) < 0) return nullptr;
    }
    return dict.release();
}

PyObject* NativeSpatialIndex_query(PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    Box64 q;
    if (nargs == 1) {
        if (!parse_rect_norm(args[0], q)) return nullptr;
    } else if (nargs == 4) {
        int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
            !parse_int64(args[2], x1) || !parse_int64(args[3], y1)) {
            return nullptr;
        }
        q = normalize_box(x0, y0, x1, y1);
    } else {
        PyErr_SetString(PyExc_TypeError, "query expects (bbox) or (x0, y0, x1, y1)");
        return nullptr;
    }

    const auto& core = *self->core;
    std::vector<uint32_t> hits;
    hits.reserve(32);
    core.for_each_intersecting(q, [&](const BucketItem& item) {
        hits.push_back(item.entry_idx);
    });

    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(hits.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < hits.size(); ++i) {
        PyObject* tup = core.py_entries[hits[i]].item_tuple;
        Py_INCREF(tup);
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup);
    }
    return out.release();
}

PyObject* NativeSpatialIndex_scan_metal_conflicts(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs < 6 || nargs > 7) {
        PyErr_SetString(PyExc_TypeError,
            "scan_metal_conflicts(x0, y0, x1, y1, space_nm, own_owner, exclude_owner=None)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, space_nm = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], space_nm)) {
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t own_id = core.lookup_owner(args[5], err);
    if (err) return nullptr;
    uint32_t excl_id = UINT32_MAX;
    if (nargs == 7 && args[6] != Py_None) {
        excl_id = core.lookup_owner(args[6], err);
        if (err) return nullptr;
    }

    const Box64 box = normalize_box(x0, y0, x1, y1);
    const auto hits = core.scan_metal_conflicts_pure(box, space_nm, own_id, excl_id);

    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(hits.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& e = core.py_entries[hits[i].entry_idx];
        PyRef tup = PyRef::adopt(PyTuple_New(3));
        if (!tup) return nullptr;
        Py_INCREF(e.owner_obj);
        Py_INCREF(e.norm_tuple);
        PyObject* b_obj = hits[i].is_touch ? Py_True : Py_False;
        Py_INCREF(b_obj);
        PyTuple_SET_ITEM(tup.get(), 0, e.owner_obj);
        PyTuple_SET_ITEM(tup.get(), 1, e.norm_tuple);
        PyTuple_SET_ITEM(tup.get(), 2, b_obj);
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    return out.release();
}

PyObject* NativeSpatialIndex_scan_committed_metal(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError,
            "scan_committed_metal(x0, y0, x1, y1, space_nm, res_cell, pres_scaled, own_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, sp = 0, rc = 0, pres = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], sp) || !parse_int64(args[5], rc) ||
        !parse_int64(args[6], pres)) {
        return nullptr;
    }
    if (rc <= 0) {
        PyErr_SetString(PyExc_ValueError, "res_cell must be positive");
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t own_id = core.lookup_owner(args[7], err);
    if (err) return nullptr;

    const Box64 box = normalize_box(x0, y0, x1, y1);
    const auto scanned = core.scan_committed_metal_pure(box, sp, rc, pres, own_id);

    PyRef res = PyRef::adopt(PyTuple_New(2));
    PyRef tot_obj = PyRef::adopt(PyLong_FromLongLong(scanned.present_total));
    if (!res || !tot_obj) return nullptr;
    PyTuple_SET_ITEM(res.get(), 0, tot_obj.release());

    if (scanned.hits.empty()) {
        PyRef empty = PyRef::adopt(PyTuple_New(0));
        if (!empty) return nullptr;
        PyTuple_SET_ITEM(res.get(), 1, empty.release());
        return res.release();
    }

    PyRef vlist = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(scanned.hits.size())));
    if (!vlist) return nullptr;
    for (size_t i = 0; i < scanned.hits.size(); ++i) {
        const auto& e = core.py_entries[scanned.hits[i].entry_idx];
        PyRef tup = PyRef::adopt(PyTuple_New(3));
        if (!tup) return nullptr;
        Py_INCREF(e.owner_obj);
        Py_INCREF(e.norm_tuple);
        PyObject* b_obj = scanned.hits[i].is_touch ? Py_True : Py_False;
        Py_INCREF(b_obj);
        PyTuple_SET_ITEM(tup.get(), 0, e.owner_obj);
        PyTuple_SET_ITEM(tup.get(), 1, e.norm_tuple);
        PyTuple_SET_ITEM(tup.get(), 2, b_obj);
        PyList_SET_ITEM(vlist.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    PyTuple_SET_ITEM(res.get(), 1, vlist.release());
    return res.release();
}

PyObject* NativeSpatialIndex_scan_obstacle_conflicts(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError,
            "scan_obstacle_conflicts(x0, y0, x1, y1, space_nm, own_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, sp = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], sp)) {
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t own_id = core.lookup_owner(args[5], err);
    if (err) return nullptr;

    const auto hits = core.scan_obstacle_conflicts_pure(normalize_box(x0, y0, x1, y1), sp, own_id);
    if (hits.empty()) return PyTuple_New(0);
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(hits.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < hits.size(); ++i) {
        PyObject* tup = core.py_entries[hits[i]].item_tuple;
        Py_INCREF(tup);
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup);
    }
    return out.release();
}

using CutSortItem = PureSpatialIndex::CutSortItem;

void sort_cut_items(const SpatialIndexCore& core, std::vector<CutSortItem>& v) {
    if (v.size() <= 1) return;
    std::sort(v.begin(), v.end(), [&](const CutSortItem& a, const CutSortItem& b) {
        const auto& sa = core.owner_sort_keys[a.owner_id];
        const auto& sb = core.owner_sort_keys[b.owner_id];
        if (sa != sb) return sa < sb;
        return a.box < b.box;
    });
}

PyObject* cut_items_to_tuple(const SpatialIndexCore& core, const std::vector<CutSortItem>& v) {
    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(v.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < v.size(); ++i) {
        const auto& e = core.py_entries[v[i].entry_idx];
        PyRef pair = PyRef::adopt(PyTuple_New(2));
        if (!pair) return nullptr;
        Py_INCREF(e.owner_obj);
        Py_INCREF(e.norm_tuple);
        PyTuple_SET_ITEM(pair.get(), 0, e.owner_obj);
        PyTuple_SET_ITEM(pair.get(), 1, e.norm_tuple);
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), pair.release());
    }
    return out.release();
}

PyObject* NativeSpatialIndex_scan_cut_conflicts(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError,
            "scan_cut_conflicts(x0, y0, x1, y1, via_space_nm, own_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, vsp = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], vsp)) {
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t own_id = core.lookup_owner(args[5], err);
    if (err) return nullptr;

    auto scanned = core.scan_cut_conflicts_pure(normalize_box(x0, y0, x1, y1), vsp, own_id);
    sort_cut_items(core, scanned.foreign_contacts);
    sort_cut_items(core, scanned.spacing_violations);

    PyRef res = PyRef::adopt(PyTuple_New(3));
    PyRef ps_obj = PyRef::adopt(PyLong_FromLongLong(scanned.present_same));
    PyRef fc_obj = PyRef::adopt(cut_items_to_tuple(core, scanned.foreign_contacts));
    PyRef sv_obj = PyRef::adopt(cut_items_to_tuple(core, scanned.spacing_violations));
    if (!res || !ps_obj || !fc_obj || !sv_obj) return nullptr;
    PyTuple_SET_ITEM(res.get(), 0, ps_obj.release());
    PyTuple_SET_ITEM(res.get(), 1, fc_obj.release());
    PyTuple_SET_ITEM(res.get(), 2, sv_obj.release());
    return res.release();
}

PyObject* NativeSpatialIndex_scan_committed_cuts(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError,
            "scan_committed_cuts(x0, y0, x1, y1, via_space_nm, res_cell, pres_scaled, own_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, vsp = 0, rc = 0, pres = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], vsp) || !parse_int64(args[5], rc) ||
        !parse_int64(args[6], pres)) {
        return nullptr;
    }
    if (rc <= 0) {
        PyErr_SetString(PyExc_ValueError, "res_cell must be positive");
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t own_id = core.lookup_owner(args[7], err);
    if (err) return nullptr;

    auto scanned = core.scan_committed_cuts_pure(normalize_box(x0, y0, x1, y1), vsp, rc, pres, own_id);
    sort_cut_items(core, scanned.foreign_contacts);
    sort_cut_items(core, scanned.spacing_violations);

    PyRef res = PyRef::adopt(PyTuple_New(3));
    PyRef tot_obj = PyRef::adopt(PyLong_FromLongLong(scanned.present_total));
    PyRef fc_obj = PyRef::adopt(cut_items_to_tuple(core, scanned.foreign_contacts));
    PyRef sv_obj = PyRef::adopt(cut_items_to_tuple(core, scanned.spacing_violations));
    if (!res || !tot_obj || !fc_obj || !sv_obj) return nullptr;
    PyTuple_SET_ITEM(res.get(), 0, tot_obj.release());
    PyTuple_SET_ITEM(res.get(), 1, fc_obj.release());
    PyTuple_SET_ITEM(res.get(), 2, sv_obj.release());
    return res.release();
}

PyObject* NativeSpatialIndex_scan_fixed_cut_conflicts(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError,
            "scan_fixed_cut_conflicts(x0, y0, x1, y1, via_space_nm)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, vsp = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], vsp)) {
        return nullptr;
    }
    const auto& core = *self->core;
    const auto hits = core.scan_fixed_cut_conflicts_pure(normalize_box(x0, y0, x1, y1), vsp);
    if (hits.empty()) return PyTuple_New(0);
    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(hits.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < hits.size(); ++i) {
        PyObject* nt = core.py_entries[hits[i].entry_idx].norm_tuple;
        Py_INCREF(nt);
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), nt);
    }
    return out.release();
}

PyObject* NativeSpatialIndex_scan_same_net_band_count(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError,
            "scan_same_net_band_count(x0, y0, x1, y1, space_nm, target_owner, filter_by_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, sp = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], sp)) {
        return nullptr;
    }
    const int filter_by_owner = PyObject_IsTrue(args[6]);
    if (filter_by_owner < 0) return nullptr;

    const auto& core = *self->core;
    uint32_t target_id = UINT32_MAX;
    if (filter_by_owner) {
        bool err = false;
        target_id = core.lookup_owner(args[5], err);
        if (err) return nullptr;
    }
    return PyLong_FromLongLong(core.scan_same_net_band_count_pure(
        normalize_box(x0, y0, x1, y1), sp, target_id, filter_by_owner != 0));
}

PyObject* NativeSpatialIndex_scan_same_net_band_rects(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError,
            "scan_same_net_band_rects(x0, y0, x1, y1, space_nm, target_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, sp = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], sp)) {
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t target_id = core.lookup_owner(args[5], err);
    if (err) return nullptr;
    const auto hits = core.scan_same_net_band_rects_pure(normalize_box(x0, y0, x1, y1), sp, target_id);
    if (hits.empty()) return PyTuple_New(0);
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(hits.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& e = core.py_entries[hits[i].entry_idx];
        PyRef tup = PyRef::adopt(PyTuple_New(5));
        PyRef osx = PyRef::adopt(PyLong_FromLongLong(hits[i].sx));
        PyRef osy = PyRef::adopt(PyLong_FromLongLong(hits[i].sy));
        if (!tup || !osx || !osy) return nullptr;
        Py_INCREF(e.owner_obj);
        Py_INCREF(e.rect_obj);
        Py_INCREF(e.norm_tuple);
        PyTuple_SET_ITEM(tup.get(), 0, e.owner_obj);
        PyTuple_SET_ITEM(tup.get(), 1, e.rect_obj);
        PyTuple_SET_ITEM(tup.get(), 2, e.norm_tuple);
        PyTuple_SET_ITEM(tup.get(), 3, osx.release());
        PyTuple_SET_ITEM(tup.get(), 4, osy.release());
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), tup.release());
    }
    return out.release();
}

PyObject* NativeSpatialIndex_has_foreign_contact(
    PyNativeSpatialIndex* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError,
            "has_foreign_contact(x0, y0, x1, y1, own_owner)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1)) {
        return nullptr;
    }
    const auto& core = *self->core;
    bool err = false;
    const uint32_t own_id = core.lookup_owner(args[4], err);
    if (err) return nullptr;

    const Box64 nb = normalize_box(x0, y0, x1, y1);
    if (spatial_has_foreign_intersecting_fast(core, nb, own_id)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* NativeSpatialIndex_export_entries(PyNativeSpatialIndex* self, PyObject*) {
    const auto* core = self->core;
    PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(core->py_entries.size())));
    if (!lst) return nullptr;
    for (size_t i = 0; i < core->py_entries.size(); ++i) {
        PyObject* item = core->py_entries[i].item_tuple;
        Py_INCREF(item);
        PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), item);
    }
    return lst.release();
}

PyObject* NativeSpatialIndex_clone(PyNativeSpatialIndex* self, PyObject*) {
    const auto* src = self->core;
    if (!src->custom_owners.empty()) {
        Py_RETURN_NONE;
    }
    auto* dst_obj = reinterpret_cast<PyNativeSpatialIndex*>(
        Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0));
    if (!dst_obj) return nullptr;
    auto* dst = new SpatialIndexCore();
    dst_obj->core = dst;

    static_cast<PureSpatialIndex&>(*dst) = src->clone_geometry();
    dst->py_entries.reserve(src->py_entries.size());
    for (const auto& e : src->py_entries) {
        Py_XINCREF(e.owner_obj);
        Py_XINCREF(e.rect_obj);
        Py_XINCREF(e.item_tuple);
        Py_XINCREF(e.norm_tuple);
        dst->py_entries.push_back(e);
    }
    dst->str_owner_to_id = src->str_owner_to_id;
    dst->owner_sort_keys = src->owner_sort_keys;
    dst->next_owner_id = src->next_owner_id;
    return reinterpret_cast<PyObject*>(dst_obj);
}

PyMethodDef NativeSpatialIndex_methods[] = {
    {"add", reinterpret_cast<PyCFunction>(NativeSpatialIndex_add), METH_FASTCALL,
     "Add (owner, rect) into the spatial index."},
    {"export_buckets", reinterpret_cast<PyCFunction>(NativeSpatialIndex_export_buckets), METH_NOARGS,
     "Materialize Python buckets dict preserving exact (owner, rect) object identities."},
    {"export_entries", reinterpret_cast<PyCFunction>(NativeSpatialIndex_export_entries), METH_NOARGS,
     "Return list of (owner, rect) pairs in insertion order for serialization."},
    {"clone", reinterpret_cast<PyCFunction>(NativeSpatialIndex_clone), METH_NOARGS,
     "Fast native clone of NativeSpatialIndex (or None if non-string custom owners exist)."},
    {"query", reinterpret_cast<PyCFunction>(NativeSpatialIndex_query), METH_FASTCALL,
     "Exact closed-bbox query returning deduplicated list of (owner, rect)."},
    {"scan_metal_conflicts", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_metal_conflicts), METH_FASTCALL,
     "Zero-alloc Euclidean + touch conflict scan returning [(owner, norm_rect, is_touch), ...]."},
    {"scan_committed_metal", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_committed_metal), METH_FASTCALL,
     "Single-pass committed metal scan returning (present_total, [(owner, norm_rect, is_contact), ...])."},
    {"scan_obstacle_conflicts", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_obstacle_conflicts), METH_FASTCALL,
     "Zero-alloc L-infinity fixed-obstacle conflict scan returning [(owner, orig_rect), ...]."},
    {"scan_cut_conflicts", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_cut_conflicts), METH_FASTCALL,
     "Single-pass cut scan returning (present_same, foreign_contacts, spacing_violations)."},
    {"scan_committed_cuts", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_committed_cuts), METH_FASTCALL,
     "Single-pass committed cut scan returning (present_total, foreign_contacts, spacing_violations)."},
    {"scan_fixed_cut_conflicts", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_fixed_cut_conflicts), METH_FASTCALL,
     "Zero-alloc fixed cut conflict scan returning sorted conflicting norm rects."},
    {"scan_same_net_band_count", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_same_net_band_count), METH_FASTCALL,
     "Zero-alloc count of entries in Euclidean open band (0, space_nm)."},
    {"scan_same_net_band_rects", reinterpret_cast<PyCFunction>(NativeSpatialIndex_scan_same_net_band_rects), METH_FASTCALL,
     "Zero-alloc scan of own-net entries in Euclidean open band (0, space_nm)."},
    {"has_foreign_contact", reinterpret_cast<PyCFunction>(NativeSpatialIndex_has_foreign_contact), METH_FASTCALL,
     "Fast boolean check for closed contact with any foreign rectangle."},
    {nullptr, nullptr, 0, nullptr}
};

PyGetSetDef NativeSpatialIndex_getset[] = {
    {"cell", reinterpret_cast<getter>(NativeSpatialIndex_get_cell), nullptr, "Bucket cell size (nm).", nullptr},
    {"size", reinterpret_cast<getter>(NativeSpatialIndex_get_size), nullptr, "Number of indexed rectangles.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

}  // namespace

PyTypeObject NativeSpatialIndexType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.NativeSpatialIndex",
    sizeof(PyNativeSpatialIndex),
    0,
    reinterpret_cast<destructor>(NativeSpatialIndex_dealloc),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "C++20 uniform bucket spatial index with O(1) generation dedup and specialized scans.",
    0, 0, 0, 0, 0, 0,
    NativeSpatialIndex_methods,
    0,
    NativeSpatialIndex_getset,
    0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(NativeSpatialIndex_init),
    0,
    NativeSpatialIndex_new,
};

// Fast spatial helpers & raw_gap_metallised_cpp live in pure C++20 `gridless_spatial.cpp`
// and are forwarded inline from `gridless_py_internal.hpp`.

// ---------------------------------------------------------------------------
// 3. batch_filter_via_sites_cpp (thin wrapper over batch_filter_via_sites_pure)
// ---------------------------------------------------------------------------

PyObject* py_batch_filter_via_sites_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {
        "xs", "ys", "extra_pts", "sites",
        "atom_rect", "atom_min_width",
        "lo", "bh", "ah", "ext",
        "sp_lo", "sp_hi", "vsp", "grid", "net",
        "obs_lo", "obs_hi", "obs_cut_lo", "fixed_cuts_fallback",
        "metal_lo", "metal_hi", "cut_lo",
        "attempt_cuts", "attempt_metal_lo", "attempt_metal_hi",
        "abstract_lo", "abstract_hi",
        "envelope_bbox", "local_envelope_bbox",
        "forbidden_xy",
        "fallback_halos_lo", "fallback_halos_hi", "fallback_halos_cut",
        "mirror_active",
        nullptr
    };

    PyObject *xs_obj = Py_None, *ys_obj = Py_None, *extra_obj = Py_None, *sites_obj = Py_None;
    PyObject *atom_rect_obj = Py_None;
    long long atom_min_width = 0;
    long long lo_ll = 0, bh_ll = 0, ah_ll = 0, ext_ll = 0;
    long long sp_lo_ll = 0, sp_hi_ll = 0, vsp_ll = 0, grid_ll = 1;
    PyObject *net_obj = nullptr;
    PyObject *obs_lo_obj = Py_None, *obs_hi_obj = Py_None, *obs_cut_lo_obj = Py_None;
    PyObject *fixed_cuts_obj = Py_None;
    PyObject *metal_lo_obj = Py_None, *metal_hi_obj = Py_None, *cut_lo_obj = Py_None;
    PyObject *attempt_cuts_obj = Py_None, *attempt_metal_lo_obj = Py_None, *attempt_metal_hi_obj = Py_None;
    PyObject *abstract_lo_obj = Py_None, *abstract_hi_obj = Py_None;
    PyObject *env_obj = Py_None, *local_env_obj = Py_None;
    PyObject *forbidden_obj = Py_None;
    PyObject *fh_lo_obj = Py_None, *fh_hi_obj = Py_None, *fh_cut_obj = Py_None;
    int mirror_active = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds,
            "|OOOOOLLLLLLLLLOOOOOOOOOOOOOOOOOOOp",
            const_cast<char**>(kwlist),
            &xs_obj, &ys_obj, &extra_obj, &sites_obj,
            &atom_rect_obj, &atom_min_width,
            &lo_ll, &bh_ll, &ah_ll, &ext_ll,
            &sp_lo_ll, &sp_hi_ll, &vsp_ll, &grid_ll, &net_obj,
            &obs_lo_obj, &obs_hi_obj, &obs_cut_lo_obj, &fixed_cuts_obj,
            &metal_lo_obj, &metal_hi_obj, &cut_lo_obj,
            &attempt_cuts_obj, &attempt_metal_lo_obj, &attempt_metal_hi_obj,
            &abstract_lo_obj, &abstract_hi_obj,
            &env_obj, &local_env_obj,
            &forbidden_obj,
            &fh_lo_obj, &fh_hi_obj, &fh_cut_obj,
            &mirror_active)) {
        return nullptr;
    }

    if (!net_obj) {
        PyErr_SetString(PyExc_TypeError, "net argument is required");
        return nullptr;
    }

    bool ok = true;
    const SpatialIndexCore* obs_lo = extract_spatial_core(obs_lo_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_hi = extract_spatial_core(obs_hi_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_cut_lo = extract_spatial_core(obs_cut_lo_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* metal_lo = extract_spatial_core(metal_lo_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* metal_hi = extract_spatial_core(metal_hi_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* cut_lo = extract_spatial_core(cut_lo_obj, ok); if (!ok) return nullptr;

    ViaSiteFilterRequest req{};
    req.has_atom = (atom_rect_obj != Py_None);
    if (req.has_atom && !parse_rect_norm(atom_rect_obj, req.atom_rect)) return nullptr;
    req.atom_min_width = static_cast<int64_t>(atom_min_width);
    req.has_env = (env_obj != Py_None);
    if (req.has_env && !parse_rect_norm(env_obj, req.env_box)) return nullptr;
    req.has_local_env = (local_env_obj != Py_None);
    if (req.has_local_env && !parse_rect_norm(local_env_obj, req.local_env_box)) return nullptr;

    std::vector<int64_t> xs, ys;
    std::vector<std::pair<int64_t, int64_t>> extra_pts, explicit_sites;
    req.use_explicit_sites = (sites_obj != Py_None);
    if (req.use_explicit_sites) {
        if (!parse_xy_list(sites_obj, explicit_sites)) return nullptr;
        req.explicit_sites = &explicit_sites;
    } else {
        if (!parse_int64_list(xs_obj, xs)) return nullptr;
        if (!parse_int64_list(ys_obj, ys)) return nullptr;
        if (!parse_xy_list(extra_obj, extra_pts)) return nullptr;
        req.xs = &xs;
        req.ys = &ys;
        req.extra_pts = &extra_pts;
    }

    std::vector<Box64> fixed_cuts, attempt_cuts, attempt_metal_lo, attempt_metal_hi;
    std::vector<Box64> fh_lo, fh_hi, fh_cut;
    if (!parse_rect_list(fixed_cuts_obj, fixed_cuts)) return nullptr;
    if (!parse_rect_list(attempt_cuts_obj, attempt_cuts)) return nullptr;
    if (!parse_rect_list(attempt_metal_lo_obj, attempt_metal_lo)) return nullptr;
    if (!parse_rect_list(attempt_metal_hi_obj, attempt_metal_hi)) return nullptr;
    if (!parse_rect_list(fh_lo_obj, fh_lo)) return nullptr;
    if (!parse_rect_list(fh_hi_obj, fh_hi)) return nullptr;
    if (!parse_rect_list(fh_cut_obj, fh_cut)) return nullptr;

    std::unordered_set<Box64, Box64Hash> abstract_lo_set, abstract_hi_set;
    if (abstract_lo_obj != Py_None) {
        std::vector<Box64> tmp;
        if (!parse_rect_list(abstract_lo_obj, tmp)) return nullptr;
        for (const auto& b : tmp) abstract_lo_set.insert(b);
    }
    if (abstract_hi_obj != Py_None) {
        std::vector<Box64> tmp;
        if (!parse_rect_list(abstract_hi_obj, tmp)) return nullptr;
        for (const auto& b : tmp) abstract_hi_set.insert(b);
    }

    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> forbidden_set;
    if (forbidden_obj != Py_None) {
        std::vector<std::pair<int64_t, int64_t>> ftmp;
        if (!parse_xy_list(forbidden_obj, ftmp)) return nullptr;
        for (const auto& p : ftmp) forbidden_set.insert(p);
    }

    bool err = false;
    req.net_obs_lo = obs_lo ? obs_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_obs_hi = obs_hi ? obs_hi->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_metal_lo = metal_lo ? metal_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_metal_hi = metal_hi ? metal_hi->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_cut_lo = cut_lo ? cut_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;

    req.lo = static_cast<int32_t>(lo_ll);
    req.bh = static_cast<int64_t>(bh_ll);
    req.ah = static_cast<int64_t>(ah_ll);
    req.ext = static_cast<int64_t>(ext_ll);
    req.sp_lo = static_cast<int64_t>(sp_lo_ll);
    req.sp_hi = static_cast<int64_t>(sp_hi_ll);
    req.vsp = static_cast<int64_t>(vsp_ll);
    req.grid = std::max<int64_t>(1, static_cast<int64_t>(grid_ll));
    req.net_owner_tag = py_to_tag(net_obj);
    req.obs_lo = obs_lo;
    req.obs_hi = obs_hi;
    req.obs_cut_lo = obs_cut_lo;
    req.metal_lo = metal_lo;
    req.metal_hi = metal_hi;
    req.cut_lo = cut_lo;
    req.fixed_cuts = &fixed_cuts;
    req.attempt_cuts = &attempt_cuts;
    req.attempt_metal_lo = &attempt_metal_lo;
    req.attempt_metal_hi = &attempt_metal_hi;
    req.abstract_lo_set = &abstract_lo_set;
    req.abstract_hi_set = &abstract_hi_set;
    req.forbidden_set = &forbidden_set;
    req.fh_lo = &fh_lo;
    req.fh_hi = &fh_hi;
    req.fh_cut = &fh_cut;
    req.mirror_active = (mirror_active != 0);

    const auto res = batch_filter_via_sites_pure(req);

    auto build_xy_list = [](const std::vector<std::pair<int64_t, int64_t>>& pts) -> PyObject* {
        PyRef lst = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pts.size())));
        if (!lst) return nullptr;
        for (size_t i = 0; i < pts.size(); ++i) {
            PyRef tup = PyRef::adopt(make_xy_tuple(pts[i].first, pts[i].second));
            if (!tup) return nullptr;
            PyList_SET_ITEM(lst.get(), static_cast<Py_ssize_t>(i), tup.release());
        }
        return lst.release();
    };

    PyRef py_legal = PyRef::adopt(build_xy_list(res.legal_sites));
    PyRef py_fallback = PyRef::adopt(build_xy_list(res.fallback_sites));
    PyRef py_count = PyRef::adopt(PyLong_FromLongLong(res.candidate_count));
    PyRef py_rejs = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.agg_rejs.size())));
    if (!py_legal || !py_fallback || !py_count || !py_rejs) return nullptr;

    Py_ssize_t ridx = 0;
    for (const auto& kv : res.agg_rejs) {
        const PureAggRejKey& k = kv.first;
        const PureAggRejVal& v = kv.second;
        PyRef rec = PyRef::adopt(PyTuple_New(8));
        PyRef o_cat = PyRef::adopt(PyLong_FromLong(k.cat));
        PyRef o_owner = PyRef::borrow(tag_to_py_borrowed(v.owner_tag));
        PyRef o_gi = PyRef::adopt(PyLong_FromLong(k.gi));
        PyRef o_half = PyRef::adopt(PyLong_FromLongLong(k.candidate_half));
        PyRef o_req = (k.required_space < 0)
            ? PyRef::borrow(Py_None)
            : PyRef::adopt(PyLong_FromLongLong(k.required_space));
        PyRef o_blk = (!k.has_blocker)
            ? PyRef::borrow(Py_None)
            : PyRef::adopt(make_box64_tuple(k.blocker));
        PyRef o_cnt = PyRef::adopt(PyLong_FromLongLong(v.count));
        PyRef o_ext = PyRef::adopt(make_box64_tuple(v.extent));
        if (!rec || !o_cat || !o_owner || !o_gi || !o_half || !o_req || !o_blk || !o_cnt || !o_ext) {
            return nullptr;
        }
        PyTuple_SET_ITEM(rec.get(), 0, o_cat.release());
        PyTuple_SET_ITEM(rec.get(), 1, o_owner.release());
        PyTuple_SET_ITEM(rec.get(), 2, o_gi.release());
        PyTuple_SET_ITEM(rec.get(), 3, o_half.release());
        PyTuple_SET_ITEM(rec.get(), 4, o_req.release());
        PyTuple_SET_ITEM(rec.get(), 5, o_blk.release());
        PyTuple_SET_ITEM(rec.get(), 6, o_cnt.release());
        PyTuple_SET_ITEM(rec.get(), 7, o_ext.release());
        PyList_SET_ITEM(py_rejs.get(), ridx++, rec.release());
    }

    PyRef out = PyRef::adopt(PyTuple_New(4));
    if (!out) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, py_legal.release());
    PyTuple_SET_ITEM(out.get(), 1, py_fallback.release());
    PyTuple_SET_ITEM(out.get(), 2, py_count.release());
    PyTuple_SET_ITEM(out.get(), 3, py_rejs.release());
    return out.release();
}

// ---------------------------------------------------------------------------
// 4. PyNative2DPotential, PyNativeLayerHmin & exact_distance_transform_cpp
// ---------------------------------------------------------------------------

namespace {

bool parse_2d_points(PyObject* obj, std::vector<Pt2D>& out) {
    if (PyDict_Check(obj)) {
        PyObject *key = nullptr, *val = nullptr;
        Py_ssize_t pos = 0;
        out.reserve(static_cast<size_t>(PyDict_GET_SIZE(obj)));
        while (PyDict_Next(obj, &pos, &key, &val)) {
            int64_t gx = 0, gy = 0, c = 0;
            if (PyTuple_CheckExact(key) && PyTuple_GET_SIZE(key) >= 2) {
                if (!parse_int64(PyTuple_GET_ITEM(key, 0), gx) ||
                    !parse_int64(PyTuple_GET_ITEM(key, 1), gy) ||
                    !parse_int64(val, c)) {
                    return false;
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "projected keys must be (gx, gy, ...) tuples");
                return false;
            }
            out.push_back(Pt2D{gx, gy, c});
        }
        return true;
    }
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "points must be a dict or sequence of (gx, gy, cost)"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* row = items[i];
        int64_t gx = 0, gy = 0, c = 0;
        if (PyTuple_CheckExact(row) && PyTuple_GET_SIZE(row) == 3) {
            if (!parse_int64(PyTuple_GET_ITEM(row, 0), gx) ||
                !parse_int64(PyTuple_GET_ITEM(row, 1), gy) ||
                !parse_int64(PyTuple_GET_ITEM(row, 2), c)) {
                return false;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "each point must be a (gx, gy, cost) 3-tuple");
            return false;
        }
        out.push_back(Pt2D{gx, gy, c});
    }
    return true;
}

PyObject* Native2DPotential_vectorcall(
    PyObject* callable, PyObject* const* args, size_t nargsf, PyObject* kwnames) {
    if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
        PyErr_SetString(PyExc_TypeError, "Native2DPotential() takes no keyword arguments");
        return nullptr;
    }
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Native2DPotential(x, y) expects exactly 2 arguments");
        return nullptr;
    }
    int64_t x = 0, y = 0;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y)) return nullptr;
    auto* self = reinterpret_cast<PyNative2DPotential*>(callable);
    return PyLong_FromLongLong(self->core->eval(x, y));
}

void Native2DPotential_dealloc(PyNative2DPotential* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* Native2DPotential_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNative2DPotential*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->vectorcall = Native2DPotential_vectorcall;
    self->core = new Potential2DCore();
    return reinterpret_cast<PyObject*>(self);
}

int Native2DPotential_init(PyNative2DPotential* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"points", "rate", nullptr};
    PyObject* pts_obj = nullptr;
    long long rate = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OL", const_cast<char**>(kwlist), &pts_obj, &rate)) {
        return -1;
    }
    std::vector<Pt2D> pts;
    if (!parse_2d_points(pts_obj, pts)) return -1;
    if (pts.empty()) {
        PyErr_SetString(PyExc_ValueError, "Native2DPotential requires at least one point");
        return -1;
    }
    self->core->init(std::move(pts), static_cast<int64_t>(rate));
    return 0;
}

PyObject* Native2DPotential_value(PyNative2DPotential* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "value(x, y) expects 2 arguments");
        return nullptr;
    }
    int64_t x = 0, y = 0;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y)) return nullptr;
    return PyLong_FromLongLong(self->core->eval(x, y));
}

PyMethodDef Native2DPotential_methods[] = {
    {"value", reinterpret_cast<PyCFunction>(Native2DPotential_value), METH_FASTCALL,
     "Evaluate exact 2D L1 potential at (x, y)."},
    {nullptr, nullptr, 0, nullptr}
};

PyObject* NativeLayerHmin_vectorcall(
    PyObject* callable, PyObject* const* args, size_t nargsf, PyObject* kwnames) {
    if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
        PyErr_SetString(PyExc_TypeError, "NativeLayerHmin() takes no keyword arguments");
        return nullptr;
    }
    if (PyVectorcall_NARGS(nargsf) != 3) {
        PyErr_SetString(PyExc_TypeError, "NativeLayerHmin(x, y, gi) expects 3 arguments");
        return nullptr;
    }
    int64_t x = 0, y = 0, gi = 0;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y) || !parse_int64(args[2], gi)) {
        return nullptr;
    }
    auto* self = reinterpret_cast<PyNativeLayerHmin*>(callable);
    if (gi < 0 || gi >= self->holder->core.n_gi) {
        PyErr_SetString(PyExc_IndexError, "layer index out of range");
        return nullptr;
    }
    return PyLong_FromLongLong(self->holder->core.eval(x, y, static_cast<int32_t>(gi)));
}

void NativeLayerHmin_dealloc(PyNativeLayerHmin* self) {
    delete self->holder;
    self->holder = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeLayerHmin_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeLayerHmin*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->vectorcall = NativeLayerHmin_vectorcall;
    self->holder = new LayerHminPyHolder();
    return reinterpret_cast<PyObject*>(self);
}

int NativeLayerHmin_init(PyNativeLayerHmin* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {
        "layer_specs", "target_boxes", "via_cost", "n_gi", "max_cache", nullptr
    };
    PyObject *specs_obj = nullptr, *boxes_obj = nullptr;
    long long via_cost = 40000, n_gi = 0, max_cache = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOLLL", const_cast<char**>(kwlist),
            &specs_obj, &boxes_obj, &via_cost, &n_gi, &max_cache)) {
        return -1;
    }
    if (n_gi <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_gi must be positive");
        return -1;
    }
    auto* holder = self->holder;
    holder->clear_refs();
    auto& core = holder->core;
    core.specs.clear();
    core.target_boxes.clear();
    core.cache.clear();
    core.n_gi = static_cast<int32_t>(n_gi);
    core.via_cost = static_cast<int64_t>(via_cost);
    core.max_cache = (max_cache > 0) ? static_cast<size_t>(max_cache) : 0;

    if (!parse_rect_list(boxes_obj, core.target_boxes) || core.target_boxes.empty()) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "target_boxes must be non-empty");
        }
        return -1;
    }

    PyRef seq = PyRef::adopt(PySequence_Fast(specs_obj, "layer_specs must be a sequence"));
    if (!seq) return -1;
    const Py_ssize_t ns = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    core.specs.reserve(static_cast<size_t>(ns));
    holder->pot_refs.reserve(static_cast<size_t>(ns));
    for (Py_ssize_t i = 0; i < ns; ++i) {
        PyObject* row = items[i];
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 4) {
            PyErr_SetString(PyExc_TypeError, "each layer_spec must be (layer, rate, pot, via_tail)");
            return -1;
        }
        int64_t layer = 0, rate = 0, via_tail = 0;
        PyObject* pot_obj = PyTuple_GET_ITEM(row, 2);
        if (!parse_int64(PyTuple_GET_ITEM(row, 0), layer) ||
            !parse_int64(PyTuple_GET_ITEM(row, 1), rate) ||
            !parse_int64(PyTuple_GET_ITEM(row, 3), via_tail)) {
            return -1;
        }
        if (Py_TYPE(pot_obj) != &Native2DPotentialType) {
            PyErr_SetString(PyExc_TypeError, "layer_spec potential must be Native2DPotential");
            return -1;
        }
        if (layer < 0 || layer >= n_gi) {
            PyErr_SetString(PyExc_IndexError, "layer out of range");
            return -1;
        }
        Py_INCREF(pot_obj);
        holder->pot_refs.push_back(pot_obj);
        core.specs.push_back(LayerHminSpec{
            static_cast<int32_t>(layer),
            rate,
            via_tail,
            reinterpret_cast<PyNative2DPotential*>(pot_obj)->core
        });
    }
    return 0;
}

}  // namespace

PyTypeObject Native2DPotentialType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.Native2DPotential",
    sizeof(PyNative2DPotential),
    0,
    reinterpret_cast<destructor>(Native2DPotential_dealloc),
    offsetof(PyNative2DPotential, vectorcall),
    0, 0, 0, 0, 0, 0, 0, 0,
    PyVectorcall_Call,
    0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    "C++20 4-quadrant / 1D-Pareto / 2D KD-tree exact L1 potential evaluator.",
    0, 0, 0, 0, 0, 0,
    Native2DPotential_methods,
    0, 0, 0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(Native2DPotential_init),
    0,
    Native2DPotential_new,
};

PyTypeObject NativeLayerHminType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.NativeLayerHmin",
    sizeof(PyNativeLayerHmin),
    0,
    reinterpret_cast<destructor>(NativeLayerHmin_dealloc),
    offsetof(PyNativeLayerHmin, vectorcall),
    0, 0, 0, 0, 0, 0, 0, 0,
    PyVectorcall_Call,
    0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    "C++20 multi-layer region A* hmin lower-bound evaluator.",
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(NativeLayerHmin_init),
    0,
    NativeLayerHmin_new,
};

PyObject* py_exact_distance_transform_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError,
            "exact_distance_transform_cpp(xs, ys, layers, goal_terminal_cost, step_cost, via_cost)");
        return nullptr;
    }
    std::vector<int64_t> xs, ys, layers;
    if (!parse_int64_list(args[0], xs) ||
        !parse_int64_list(args[1], ys) ||
        !parse_int64_list(args[2], layers)) {
        return nullptr;
    }
    int64_t step_cost = 1, via_cost = 40000;
    if (!parse_int64(args[4], step_cost) || !parse_int64(args[5], via_cost)) {
        return nullptr;
    }

    const size_t nx = xs.size();
    const size_t ny = ys.size();
    const size_t nl = layers.size();
    if (nx == 0 || ny == 0 || nl == 0) {
        return PyList_New(0);
    }

    auto validate_axis = [](const std::vector<int64_t>& v,
                            std::unordered_map<int64_t, size_t>& idx) -> bool {
        idx.reserve(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0 && v[i] <= v[i - 1]) {
                PyErr_SetString(PyExc_ValueError,
                    "multigoal potential axes must be strictly increasing");
                return false;
            }
            idx.emplace(v[i], i);
        }
        return true;
    };

    std::unordered_map<int64_t, size_t> x_idx, y_idx, l_idx;
    if (!validate_axis(xs, x_idx) || !validate_axis(ys, y_idx) || !validate_axis(layers, l_idx)) {
        return nullptr;
    }

    std::vector<DistanceTransformSeed> seeds;
    const size_t plane = nx * ny;
    bool has_max = false;
    int64_t max_term = 0;

    PyObject* goals_obj = args[3];
    if (PyDict_Check(goals_obj)) {
        const Py_ssize_t ng = PyDict_GET_SIZE(goals_obj);
        if (ng == 0) return PyList_New(0);
        seeds.reserve(static_cast<size_t>(ng));
        PyObject *key = nullptr, *val = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(goals_obj, &pos, &key, &val)) {
            int64_t gx = 0, gy = 0, gl = 0, cost = 0;
            if (!PyTuple_CheckExact(key) || PyTuple_GET_SIZE(key) != 3 ||
                !parse_int64(PyTuple_GET_ITEM(key, 0), gx) ||
                !parse_int64(PyTuple_GET_ITEM(key, 1), gy) ||
                !parse_int64(PyTuple_GET_ITEM(key, 2), gl) ||
                !parse_int64(val, cost)) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_TypeError, "goal key must be (gx, gy, gl) int tuple");
                }
                return nullptr;
            }
            auto ix = x_idx.find(gx);
            auto iy = y_idx.find(gy);
            auto il = l_idx.find(gl);
            if (ix == x_idx.end() || iy == y_idx.end() || il == l_idx.end()) {
                PyErr_SetString(PyExc_KeyError, "goal coordinate not found in axes");
                return nullptr;
            }
            if (!has_max || cost > max_term) { max_term = cost; has_max = true; }
            seeds.push_back(DistanceTransformSeed{il->second * plane + iy->second * nx + ix->second, cost});
        }
    } else {
        PyRef items_iter = PyRef::adopt(PyObject_GetIter(goals_obj));
        if (!items_iter) return nullptr;
        while (true) {
            PyRef item = PyRef::adopt(PyIter_Next(items_iter.get()));
            if (!item) break;
            if (!PyTuple_CheckExact(item.get()) || PyTuple_GET_SIZE(item.get()) != 2) {
                PyErr_SetString(PyExc_TypeError, "goal_items must yield ((gx, gy, gl), cost)");
                return nullptr;
            }
            PyObject* key = PyTuple_GET_ITEM(item.get(), 0);
            PyObject* val = PyTuple_GET_ITEM(item.get(), 1);
            int64_t gx = 0, gy = 0, gl = 0, cost = 0;
            if (!PyTuple_CheckExact(key) || PyTuple_GET_SIZE(key) != 3 ||
                !parse_int64(PyTuple_GET_ITEM(key, 0), gx) ||
                !parse_int64(PyTuple_GET_ITEM(key, 1), gy) ||
                !parse_int64(PyTuple_GET_ITEM(key, 2), gl) ||
                !parse_int64(val, cost)) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_TypeError, "goal key must be (gx, gy, gl) int tuple");
                }
                return nullptr;
            }
            auto ix = x_idx.find(gx);
            auto iy = y_idx.find(gy);
            auto il = l_idx.find(gl);
            if (ix == x_idx.end() || iy == y_idx.end() || il == l_idx.end()) {
                PyErr_SetString(PyExc_KeyError, "goal coordinate not found in axes");
                return nullptr;
            }
            if (!has_max || cost > max_term) { max_term = cost; has_max = true; }
            seeds.push_back(DistanceTransformSeed{il->second * plane + iy->second * nx + ix->second, cost});
        }
        if (PyErr_Occurred()) return nullptr;
        if (seeds.empty()) return PyList_New(0);
    }

    std::vector<int64_t> pot;
    const DistanceTransformStatus st = exact_distance_transform_from_seeds(
        xs, ys, layers, seeds, max_term, step_cost, via_cost, pot);
    if (st == DistanceTransformStatus::Overflow) {
        PyErr_SetString(PyExc_OverflowError, "distance transform exceeds int64 range");
        return nullptr;
    }
    if (st == DistanceTransformStatus::BadAlloc) {
        return PyErr_NoMemory();
    }
    if (st != DistanceTransformStatus::Ok) {
        return PyList_New(0);
    }

    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(pot.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < pot.size(); ++i) {
        PyRef v = PyRef::adopt(PyLong_FromLongLong(pot[i]));
        if (!v) return nullptr;
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), v.release());
    }
    return out.release();
}

// ---------------------------------------------------------------------------
// 5. compute_throat_events_cpp & batch_via_congestion_cpp
// ---------------------------------------------------------------------------

PyObject* py_compute_throat_events_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 9) {
        PyErr_SetString(PyExc_TypeError,
            "compute_throat_events_cpp(base_xs, base_ys, x0, y0, x1, y1, bh, min_width, grid)");
        return nullptr;
    }
    std::vector<int64_t> base_xs, base_ys;
    if (!parse_int64_list(args[0], base_xs) || !parse_int64_list(args[1], base_ys)) {
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, bh = 0, min_width = 0, grid = 1;
    if (!parse_int64(args[2], x0) || !parse_int64(args[3], y0) ||
        !parse_int64(args[4], x1) || !parse_int64(args[5], y1) ||
        !parse_int64(args[6], bh) || !parse_int64(args[7], min_width) ||
        !parse_int64(args[8], grid)) {
        return nullptr;
    }

    const ThroatEventsResult res = compute_throat_events_core(
        base_xs, base_ys, x0, y0, x1, y1, bh, min_width, grid);

    auto make_int_tuple = [](const std::vector<int64_t>& vec) -> PyObject* {
        PyRef t = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(vec.size())));
        if (!t) return nullptr;
        for (size_t i = 0; i < vec.size(); ++i) {
            PyRef v = PyRef::adopt(PyLong_FromLongLong(vec[i]));
            if (!v) return nullptr;
            PyTuple_SET_ITEM(t.get(), static_cast<Py_ssize_t>(i), v.release());
        }
        return t.release();
    };

    PyRef py_xs = PyRef::adopt(make_int_tuple(res.xs));
    PyRef py_ys = PyRef::adopt(make_int_tuple(res.ys));
    PyRef py_th = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(res.thresh.size())));
    if (!py_xs || !py_ys || !py_th) return nullptr;
    for (size_t i = 0; i < res.thresh.size(); ++i) {
        PyRef pt = PyRef::adopt(make_xy_tuple(res.thresh[i].first, res.thresh[i].second));
        if (!pt) return nullptr;
        PyTuple_SET_ITEM(py_th.get(), static_cast<Py_ssize_t>(i), pt.release());
    }

    PyRef out = PyRef::adopt(PyTuple_New(3));
    if (!out) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, py_xs.release());
    PyTuple_SET_ITEM(out.get(), 1, py_ys.release());
    PyTuple_SET_ITEM(out.get(), 2, py_th.release());
    return out.release();
}

namespace {

bool parse_hist_entries(PyObject* obj, HistMap& out) {
    if (!obj || obj == Py_None) return true;
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "history entries must be a sequence of (cx, cy, val)"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.reserve(static_cast<size_t>(n) * 2 + 1);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* row = items[i];
        int64_t cx = 0, cy = 0, hv = 0;
        if (PyTuple_CheckExact(row) && PyTuple_GET_SIZE(row) == 3) {
            if (!parse_int64(PyTuple_GET_ITEM(row, 0), cx) ||
                !parse_int64(PyTuple_GET_ITEM(row, 1), cy) ||
                !parse_int64(PyTuple_GET_ITEM(row, 2), hv)) {
                return false;
            }
        } else {
            PyRef rseq = PyRef::adopt(PySequence_Fast(row, "history entry must be 3-sequence"));
            if (!rseq) return false;
            const bool ok = PySequence_Fast_GET_SIZE(rseq.get()) == 3 &&
                            parse_int64(PySequence_Fast_GET_ITEM(rseq.get(), 0), cx) &&
                            parse_int64(PySequence_Fast_GET_ITEM(rseq.get(), 1), cy) &&
                            parse_int64(PySequence_Fast_GET_ITEM(rseq.get(), 2), hv);
            if (!ok) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "history entry must have length 3");
                return false;
            }
        }
        out.emplace(std::make_pair(cx, cy), hv);
    }
    return true;
}

}  // namespace

PyObject* py_batch_via_congestion_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {
        "sites",
        "bh", "ah", "ext",
        "sp_lo", "sp_hi", "vsp",
        "rc", "pres", "mul_scale", "cost_factor_scale",
        "net",
        "metal_lo", "metal_hi",
        "obs_lo", "obs_hi",
        "attempt_lo", "attempt_hi",
        "cut_lo",
        "hist_lo", "hist_hi", "hist_cut",
        "wide_halos_lo", "wide_halos_hi",
        nullptr
    };

    PyObject* sites_obj = nullptr;
    long long bh_ll = 0, ah_ll = 0, ext_ll = 0;
    long long sp_lo_ll = 0, sp_hi_ll = 0, vsp_ll = 0;
    long long rc_ll = 1, pres_ll = 0, mul_scale_ll = 0, scale_ll = 1000;
    PyObject* net_obj = nullptr;
    PyObject *metal_lo_obj = Py_None, *metal_hi_obj = Py_None;
    PyObject *obs_lo_obj = Py_None, *obs_hi_obj = Py_None;
    PyObject *att_lo_obj = Py_None, *att_hi_obj = Py_None;
    PyObject *cut_lo_obj = Py_None;
    PyObject *hist_lo_obj = Py_None, *hist_hi_obj = Py_None, *hist_cut_obj = Py_None;
    PyObject *wh_lo_obj = Py_None, *wh_hi_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds,
            "OLLLLLLLLLLO|OOOOOOOOOOOO",
            const_cast<char**>(kwlist),
            &sites_obj,
            &bh_ll, &ah_ll, &ext_ll,
            &sp_lo_ll, &sp_hi_ll, &vsp_ll,
            &rc_ll, &pres_ll, &mul_scale_ll, &scale_ll,
            &net_obj,
            &metal_lo_obj, &metal_hi_obj,
            &obs_lo_obj, &obs_hi_obj,
            &att_lo_obj, &att_hi_obj,
            &cut_lo_obj,
            &hist_lo_obj, &hist_hi_obj, &hist_cut_obj,
            &wh_lo_obj, &wh_hi_obj)) {
        return nullptr;
    }

    std::vector<std::pair<int64_t, int64_t>> sites;
    if (!parse_xy_list(sites_obj, sites)) return nullptr;

    bool ok = true;
    const SpatialIndexCore* metal_lo = extract_spatial_core(metal_lo_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* metal_hi = extract_spatial_core(metal_hi_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_lo = extract_spatial_core(obs_lo_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_hi = extract_spatial_core(obs_hi_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* att_lo = extract_spatial_core(att_lo_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* att_hi = extract_spatial_core(att_hi_obj, ok); if (!ok) return nullptr;
    const SpatialIndexCore* cut_lo = extract_spatial_core(cut_lo_obj, ok); if (!ok) return nullptr;

    HistMap hist_lo, hist_hi, hist_cut;
    if (!parse_hist_entries(hist_lo_obj, hist_lo) ||
        !parse_hist_entries(hist_hi_obj, hist_hi) ||
        !parse_hist_entries(hist_cut_obj, hist_cut)) {
        return nullptr;
    }

    std::vector<Box64> wh_lo, wh_hi;
    if (!parse_rect_list(wh_lo_obj, wh_lo) || !parse_rect_list(wh_hi_obj, wh_hi)) {
        return nullptr;
    }

    bool err = false;
    BatchViaCongestionRequest req{};
    req.sites = &sites;
    req.bh = static_cast<int64_t>(bh_ll);
    req.ah = static_cast<int64_t>(ah_ll);
    req.ext = static_cast<int64_t>(ext_ll);
    req.sp_lo = static_cast<int64_t>(sp_lo_ll);
    req.sp_hi = static_cast<int64_t>(sp_hi_ll);
    req.vsp = static_cast<int64_t>(vsp_ll);
    req.rc = std::max<int64_t>(1, static_cast<int64_t>(rc_ll));
    req.pres = static_cast<int64_t>(pres_ll);
    req.mul_scale = static_cast<int64_t>(mul_scale_ll);
    req.cost_scale = static_cast<int64_t>(scale_ll);
    req.metal_lo = metal_lo;
    req.metal_hi = metal_hi;
    req.obs_lo = obs_lo;
    req.obs_hi = obs_hi;
    req.att_lo = att_lo;
    req.att_hi = att_hi;
    req.cut_lo = cut_lo;
    req.net_metal_lo = metal_lo ? metal_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_metal_hi = metal_hi ? metal_hi->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_obs_lo = obs_lo ? obs_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_obs_hi = obs_hi ? obs_hi->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.net_cut_lo = cut_lo ? cut_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.hist_lo = &hist_lo;
    req.hist_hi = &hist_hi;
    req.hist_cut = &hist_cut;
    req.wh_lo = &wh_lo;
    req.wh_hi = &wh_hi;

    const auto vals = batch_via_congestion_pure(req);
    PyRef out = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(vals.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < vals.size(); ++i) {
        PyRef pv = PyRef::adopt(PyLong_FromLongLong(vals[i]));
        if (!pv) return nullptr;
        PyList_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), pv.release());
    }
    return out.release();
}

// ---------------------------------------------------------------------------
// 6. PyNativeHistoryGrid, check_wire/via_legal_fast_cpp & box_congestion_fast_cpp
// ---------------------------------------------------------------------------

namespace {

void NativeHistoryGrid_dealloc(PyNativeHistoryGrid* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeHistoryGrid_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeHistoryGrid*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->core = new HistoryGridCore();
    return reinterpret_cast<PyObject*>(self);
}

int NativeHistoryGrid_init(PyNativeHistoryGrid* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"history", "hist_fac", "cost_factor_scale", nullptr};
    PyObject* hist_obj = nullptr;
    double hist_fac = 1.0;
    long long cost_scale_ll = 1000;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "Od|L", const_cast<char**>(kwlist),
            &hist_obj, &hist_fac, &cost_scale_ll)) {
        return -1;
    }
    if (!PyDict_CheckExact(hist_obj)) {
        PyErr_SetString(PyExc_TypeError, "NativeHistoryGrid requires an exact dict");
        return -1;
    }
    const double scale_d = static_cast<double>(cost_scale_ll);

    std::unordered_map<int64_t, std::vector<HistoryRawCell>> by_layer;

    PyObject *key = nullptr, *val = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(hist_obj, &pos, &key, &val)) {
        const double hv = PyFloat_AsDouble(val);
        if (hv == -1.0 && PyErr_Occurred()) return -1;
        if (hv == 0.0) continue;
        // Exact Python left-to-right IEEE-754 order: (hv * hist_fac) * _COST_FACTOR_SCALE
        const double prod = (hv * hist_fac) * scale_d;
        if (!std::isfinite(prod) || std::fabs(prod) > 4.0e15) {
            PyErr_SetString(PyExc_OverflowError, "history value out of int64 SAT range");
            return -1;
        }
        const int64_t iv = round_half_even_i64(prod);
        if (iv == 0) continue;

        int64_t cx = 0, cy = 0, rgi = 0;
        if (PyTuple_CheckExact(key) && PyTuple_GET_SIZE(key) == 3) {
            if (!parse_int64(PyTuple_GET_ITEM(key, 0), cx) ||
                !parse_int64(PyTuple_GET_ITEM(key, 1), cy) ||
                !parse_int64(PyTuple_GET_ITEM(key, 2), rgi)) {
                return -1;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "history keys must be (cx, cy, resource_gi) 3-tuples");
            return -1;
        }
        by_layer[rgi].push_back(HistoryRawCell{cx, cy, iv});
    }

    self->core->build(by_layer);
    return 0;
}

PyObject* NativeHistoryGrid_eval_units(
    PyNativeHistoryGrid* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "eval_units(resource_gi, x0, y0, x1, y1, rc) expects 6 arguments");
        return nullptr;
    }
    int64_t rgi = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0, rc = 1;
    if (!parse_int64(args[0], rgi) || !parse_int64(args[1], x0) ||
        !parse_int64(args[2], y0) || !parse_int64(args[3], x1) ||
        !parse_int64(args[4], y1) || !parse_int64(args[5], rc)) {
        return nullptr;
    }
    if (rc < 1) rc = 1;
    const Box64 nb = normalize_box(x0, y0, x1, y1);
    return PyLong_FromLongLong(self->core->eval_units(rgi, nb, rc));
}

PyObject* NativeHistoryGrid_eval_pending_lb(
    PyNativeHistoryGrid* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 9) {
        PyErr_SetString(PyExc_TypeError,
            "eval_pending_lb(gi, ax, ay, bx, by, pad, rc, hist_scale_num, cost_factor_scale)");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0, pad = 0, rc = 1, hsn = 0, cfs = 1000;
    if (!parse_int64(args[0], gi) || !parse_int64(args[1], ax) ||
        !parse_int64(args[2], ay) || !parse_int64(args[3], bx) ||
        !parse_int64(args[4], by) || !parse_int64(args[5], pad) ||
        !parse_int64(args[6], rc) || !parse_int64(args[7], hsn) ||
        !parse_int64(args[8], cfs)) {
        return nullptr;
    }
    if (rc < 1) rc = 1;
    bool ovf = false;
    const int64_t ans = self->core->eval_pending_lb(gi, ax, ay, bx, by, pad, rc, hsn, cfs, ovf);
    if (ovf) Py_RETURN_NONE;
    return PyLong_FromLongLong(ans);
}

PyMethodDef NativeHistoryGrid_methods[] = {
    {"eval_units", reinterpret_cast<PyCFunction>(NativeHistoryGrid_eval_units), METH_FASTCALL,
     "Evaluate exact fixed-point history units intersecting (x0, y0, x1, y1)."},
    {"eval_pending_lb", reinterpret_cast<PyCFunction>(NativeHistoryGrid_eval_pending_lb), METH_FASTCALL,
     "Evaluate exact pending_history_lb in O(1) via 2D SAT."},
    {nullptr, nullptr, 0, nullptr}
};

}  // namespace

PyTypeObject NativeHistoryGridType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.NativeHistoryGrid",
    sizeof(PyNativeHistoryGrid),
    0,
    reinterpret_cast<destructor>(NativeHistoryGrid_dealloc),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "C++20 2D Summed-Area Table history grid for O(1) history congestion queries.",
    0, 0, 0, 0, 0, 0,
    NativeHistoryGrid_methods,
    0, 0, 0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(NativeHistoryGrid_init),
    0,
    NativeHistoryGrid_new,
};

namespace {

bool parse_wide_entries_py(PyObject* wobj, PyObject* net_obj, std::vector<FastWideCheck>& out) {
    if (!wobj || wobj == Py_None) return true;
    PyRef wseq = PyRef::adopt(PySequence_Fast(wobj, "wide_entries must be a sequence"));
    if (!wseq) return false;
    const Py_ssize_t nw = PySequence_Fast_GET_SIZE(wseq.get());
    PyObject** witems = PySequence_Fast_ITEMS(wseq.get());
    out.reserve(static_cast<size_t>(nw));
    bool ok = true, err = false;
    for (Py_ssize_t i = 0; i < nw; ++i) {
        PyObject* row = witems[i];
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 3) {
            PyErr_SetString(PyExc_TypeError, "wide entry must be (w_sp, core_idx, att_idx)");
            return false;
        }
        int64_t w_sp = 0;
        if (!parse_int64(PyTuple_GET_ITEM(row, 0), w_sp)) return false;
        const SpatialIndexCore* core_idx = extract_spatial_core(PyTuple_GET_ITEM(row, 1), ok);
        if (!ok) return false;
        const SpatialIndexCore* att_idx = extract_spatial_core(PyTuple_GET_ITEM(row, 2), ok);
        if (!ok) return false;
        const uint32_t own_c = core_idx ? core_idx->lookup_owner(net_obj, err) : UINT32_MAX;
        if (err) return false;
        const uint32_t own_a = att_idx ? att_idx->lookup_owner(net_obj, err) : UINT32_MAX;
        if (err) return false;
        out.push_back(FastWideCheck{w_sp, core_idx, own_c, att_idx, own_a});
    }
    return true;
}

struct PyGapLoaderCtx {
    PyObject* att_m_obj = nullptr;
    PyObject* abs_obj = nullptr;
};

bool load_gap_from_py(void* raw, std::vector<Box64>& att_m,
                      std::unordered_set<Box64, Box64Hash>& abs_set) {
    auto* c = static_cast<PyGapLoaderCtx*>(raw);
    if (!parse_rect_list(c->att_m_obj, att_m)) return false;
    if (c->abs_obj && c->abs_obj != Py_None) {
        std::vector<Box64> tmp;
        if (!parse_rect_list(c->abs_obj, tmp)) return false;
        for (const auto& b : tmp) abs_set.insert(b);
    }
    return true;
}

}  // namespace

PyObject* py_check_wire_legal_fast_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 12) {
        PyErr_SetString(PyExc_TypeError,
            "check_wire_legal_fast_cpp(x0, y0, x1, y1, sp, rc, pres, net, obs_idx, metal_idx, env_bbox, wide_entries)");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, sp = 0, rc = 1, pres = 0;
    if (!parse_int64(args[0], x0) || !parse_int64(args[1], y0) ||
        !parse_int64(args[2], x1) || !parse_int64(args[3], y1) ||
        !parse_int64(args[4], sp) || !parse_int64(args[5], rc) ||
        !parse_int64(args[6], pres)) {
        return nullptr;
    }
    PyObject* net_obj = args[7];
    bool ok = true, err = false;
    const SpatialIndexCore* obs_idx = extract_spatial_core(args[8], ok); if (!ok) return nullptr;
    const SpatialIndexCore* metal_idx = extract_spatial_core(args[9], ok); if (!ok) return nullptr;

    WireLegalFastRequest req{};
    req.wb = normalize_box(x0, y0, x1, y1);
    req.sp = sp;
    req.rc = std::max<int64_t>(1, rc);
    req.pres = pres;
    req.has_env = (args[10] != Py_None);
    if (req.has_env && !parse_rect_norm(args[10], req.env)) return nullptr;
    req.obs_idx = obs_idx;
    req.own_obs = obs_idx ? obs_idx->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.metal_idx = metal_idx;
    req.own_metal = metal_idx ? metal_idx->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;

    std::vector<FastWideCheck> wide_vec;
    if (!parse_wide_entries_py(args[11], net_obj, wide_vec)) return nullptr;
    req.wide_entries = wide_vec.empty() ? nullptr : &wide_vec;

    int64_t present_total = 0;
    if (!check_wire_legal_fast_pure(req, present_total)) Py_RETURN_NONE;
    return PyLong_FromLongLong(present_total);
}

PyObject* py_check_via_legal_fast_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 26) {
        PyErr_SetString(PyExc_TypeError, "check_via_legal_fast_cpp expects 26 arguments");
        return nullptr;
    }
    ViaLegalFastRequest req{};
    if (!parse_int64(args[0], req.x) || !parse_int64(args[1], req.y) ||
        !parse_int64(args[2], req.bh) || !parse_int64(args[3], req.ah) ||
        !parse_int64(args[4], req.ext) || !parse_int64(args[5], req.sp_lo) ||
        !parse_int64(args[6], req.sp_hi) || !parse_int64(args[7], req.vsp) ||
        !parse_int64(args[8], req.grid) || !parse_int64(args[9], req.rc) ||
        !parse_int64(args[10], req.pres)) {
        return nullptr;
    }
    if (req.grid < 1) req.grid = 1;
    if (req.rc < 1) req.rc = 1;
    if ((req.x % req.grid) != 0 || (req.y % req.grid) != 0) Py_RETURN_NONE;

    PyObject* net_obj = args[11];
    bool ok = true, err = false;
    const SpatialIndexCore* obs_lo = extract_spatial_core(args[12], ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_hi = extract_spatial_core(args[13], ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_cut_lo = extract_spatial_core(args[14], ok); if (!ok) return nullptr;
    const SpatialIndexCore* metal_lo = extract_spatial_core(args[15], ok); if (!ok) return nullptr;
    const SpatialIndexCore* metal_hi = extract_spatial_core(args[16], ok); if (!ok) return nullptr;
    const SpatialIndexCore* cut_lo = extract_spatial_core(args[17], ok); if (!ok) return nullptr;

    req.has_env = (args[18] != Py_None);
    if (req.has_env && !parse_rect_norm(args[18], req.env)) return nullptr;

    req.obs_lo = obs_lo;
    req.obs_hi = obs_hi;
    req.obs_cut_lo = obs_cut_lo;
    req.metal_lo = metal_lo;
    req.metal_hi = metal_hi;
    req.cut_lo = cut_lo;
    req.own_obs_lo = obs_lo ? obs_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.own_obs_hi = obs_hi ? obs_hi->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.own_metal_lo = metal_lo ? metal_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.own_metal_hi = metal_hi ? metal_hi->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;
    req.own_cut_lo = cut_lo ? cut_lo->lookup_owner(net_obj, err) : UINT32_MAX; if (err) return nullptr;

    std::vector<FastWideCheck> wlo, whi;
    if (!parse_wide_entries_py(args[19], net_obj, wlo)) return nullptr;
    if (!parse_wide_entries_py(args[20], net_obj, whi)) return nullptr;
    req.wide_lo = wlo.empty() ? nullptr : &wlo;
    req.wide_hi = whi.empty() ? nullptr : &whi;

    std::vector<Box64> att_cuts;
    if (args[21] != Py_None) {
        if (!parse_rect_list(args[21], att_cuts)) return nullptr;
        req.att_cuts = &att_cuts;
    }

    PyGapLoaderCtx gctx_lo{args[22], args[24]};
    PyGapLoaderCtx gctx_hi{args[23], args[25]};
    req.load_gap_lo = load_gap_from_py;
    req.gap_lo_ctx = &gctx_lo;
    req.load_gap_hi = load_gap_from_py;
    req.gap_hi_ctx = &gctx_hi;

    int64_t pres_lo = 0, pres_hi = 0, pres_cut = 0;
    const int status = check_via_legal_fast_pure(req, pres_lo, pres_hi, pres_cut);
    if (status < 0) return nullptr;
    if (status == 0) Py_RETURN_NONE;

    PyRef out = PyRef::adopt(PyTuple_New(3));
    PyRef o0 = PyRef::adopt(PyLong_FromLongLong(pres_lo));
    PyRef o1 = PyRef::adopt(PyLong_FromLongLong(pres_hi));
    PyRef o2 = PyRef::adopt(PyLong_FromLongLong(pres_cut));
    if (!out || !o0 || !o1 || !o2) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o0.release());
    PyTuple_SET_ITEM(out.get(), 1, o1.release());
    PyTuple_SET_ITEM(out.get(), 2, o2.release());
    return out.release();
}

PyObject* py_box_congestion_fast_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 16) {
        PyErr_SetString(PyExc_TypeError, "box_congestion_fast_cpp expects 16 arguments");
        return nullptr;
    }
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    BoxCongestionFastRequest req{};
    if (!parse_int64(args[0], req.gi) || !parse_int64(args[1], x0) ||
        !parse_int64(args[2], y0) || !parse_int64(args[3], x1) ||
        !parse_int64(args[4], y1) || !parse_int64(args[5], req.sp) ||
        !parse_int64(args[6], req.rc) || !parse_int64(args[7], req.pres) ||
        !parse_int64(args[8], req.mul_scale) || !parse_int64(args[9], req.cfs) ||
        !parse_int64(args[15], req.present_hint)) {
        return nullptr;
    }
    req.box = normalize_box(x0, y0, x1, y1);
    PyObject* net_obj = args[10];
    bool ok = true, err = false;
    const SpatialIndexCore* metal_idx = extract_spatial_core(args[11], ok); if (!ok) return nullptr;
    const SpatialIndexCore* obs_idx = extract_spatial_core(args[12], ok); if (!ok) return nullptr;
    const SpatialIndexCore* att_idx = extract_spatial_core(args[13], ok); if (!ok) return nullptr;
    const HistoryGridCore* hist_grid = extract_history_core(args[14], ok); if (!ok) return nullptr;

    req.metal_idx = metal_idx;
    req.own_metal = (req.present_hint < 0 && metal_idx) ? metal_idx->lookup_owner(net_obj, err) : UINT32_MAX;
    if (err) return nullptr;
    req.obs_idx = obs_idx;
    req.own_obs = obs_idx ? obs_idx->lookup_owner(net_obj, err) : UINT32_MAX;
    if (err) return nullptr;
    req.att_idx = att_idx;
    req.hist_grid = hist_grid;

    bool ovf = false;
    const int64_t ans = box_congestion_fast_pure(req, ovf);
    if (ovf) Py_RETURN_NONE;
    return PyLong_FromLongLong(ans);
}

// ---------------------------------------------------------------------------
// 7. EdgeContextCore & PyNativeEdgeContext (thin wrapper over PureEdgeContext)
// ---------------------------------------------------------------------------

void EdgeContextCore::release_py_refs() noexcept {
    mirror_ctx = nullptr;
    hist_grid = nullptr;
    net_tag = 0;
    layers.clear();
    vias.clear();
    for (PyObject* obj : kept_refs) {
        Py_XDECREF(obj);
    }
    kept_refs.clear();
    next_extra_kind_id = 1;
}

EdgeContextCore::~EdgeContextCore() {
    release_py_refs();
}

bool EdgeContextCore::append_rejs_to_list(
    std::unordered_map<EdgeRejKey, PureEdgeRejVal, EdgeRejKeyHash>& map,
    PyObject* py_rejs, Py_ssize_t& ridx) {
    for (const auto& kv : map) {
        const EdgeRejKey& k = kv.first;
        const PureEdgeRejVal& v = kv.second;
        PyRef rec = PyRef::adopt(PyTuple_New(9));
        PyRef o_cat = PyRef::adopt(PyLong_FromLong(k.cat));
        PyRef o_owner = PyRef::borrow(tag_to_py_borrowed(v.owner_tag));
        PyRef o_gi = PyRef::adopt(PyLong_FromLong(k.gi));
        PyRef o_half = PyRef::adopt(PyLong_FromLongLong(k.candidate_half));
        PyRef o_req = (k.required_space < 0)
            ? PyRef::borrow(Py_None)
            : PyRef::adopt(PyLong_FromLongLong(k.required_space));
        PyRef o_blk = (!k.has_blocker)
            ? PyRef::borrow(Py_None)
            : PyRef::adopt(make_box64_tuple(k.blocker));
        PyRef o_cnt = PyRef::adopt(PyLong_FromLongLong(v.count));
        PyRef o_ext = PyRef::adopt(make_box64_tuple(v.extent));
        PyRef o_kind = PyRef::borrow(tag_to_py_borrowed(v.extra_kind_tag));
        if (!rec || !o_cat || !o_owner || !o_gi || !o_half ||
            !o_req || !o_blk || !o_cnt || !o_ext || !o_kind) {
            return false;
        }
        PyTuple_SET_ITEM(rec.get(), 0, o_cat.release());
        PyTuple_SET_ITEM(rec.get(), 1, o_owner.release());
        PyTuple_SET_ITEM(rec.get(), 2, o_gi.release());
        PyTuple_SET_ITEM(rec.get(), 3, o_half.release());
        PyTuple_SET_ITEM(rec.get(), 4, o_req.release());
        PyTuple_SET_ITEM(rec.get(), 5, o_blk.release());
        PyTuple_SET_ITEM(rec.get(), 6, o_cnt.release());
        PyTuple_SET_ITEM(rec.get(), 7, o_ext.release());
        PyTuple_SET_ITEM(rec.get(), 8, o_kind.release());
        PyList_SET_ITEM(py_rejs, ridx++, rec.release());
    }
    map.clear();
    return true;
}

PyObject* EdgeContextCore::flush_rejections_py() {
    const size_t total_sz = agg_rejs.size() + (mirror_ctx ? mirror_ctx->agg_rejs.size() : 0);
    PyRef py_rejs = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(total_sz)));
    if (!py_rejs) return nullptr;
    Py_ssize_t ridx = 0;
    if (!append_rejs_to_list(agg_rejs, py_rejs.get(), ridx)) return nullptr;
    pending_rej_count = 0;
    if (mirror_ctx) {
        if (!append_rejs_to_list(mirror_ctx->agg_rejs, py_rejs.get(), ridx)) return nullptr;
        mirror_ctx->pending_rej_count = 0;
    }
    return py_rejs.release();
}

namespace {

void NativeEdgeContext_dealloc(PyNativeEdgeContext* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeEdgeContext_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeEdgeContext*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->core = new EdgeContextCore();
    return reinterpret_cast<PyObject*>(self);
}

int NativeEdgeContext_init(PyNativeEdgeContext* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {
        "net", "n_layers", "grid", "rc", "pres", "mul_scale", "cost_factor_scale",
        "turn_cost", "via_cost", "step_cost", "off_pref_cost",
        "single_escape_ok", "single_via_penalty", "via_succ_dirs",
        "env_bbox", "hist_grid", "neg_pending_hist", "hist_scale_num",
        "layers", "vias",
        "mirror_ctx", "mirror_axis2", "mirror_intent_space",
        nullptr
    };
    PyObject *net_obj = nullptr, *via_dirs_obj = nullptr, *env_obj = nullptr;
    PyObject *hist_obj = Py_None, *layers_obj = nullptr, *vias_obj = nullptr;
    PyObject *mirror_ctx_obj = Py_None;
    int n_layers = 0, single_escape_ok = 0, neg_pending_hist = 0;
    long long grid_ll = 1, rc_ll = 1, pres_ll = 0, mul_ll = 0, cfs_ll = 1000;
    long long turn_ll = 0, via_ll = 40000, step_ll = 100, off_ll = 100;
    long long svp_ll = 0, hsn_ll = 0;
    long long mirror_axis2_ll = 0, mirror_intent_ll = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OiLLLLLLLLLpLOOOpLOO|OLL", const_cast<char**>(kwlist),
            &net_obj, &n_layers, &grid_ll, &rc_ll, &pres_ll, &mul_ll, &cfs_ll,
            &turn_ll, &via_ll, &step_ll, &off_ll,
            &single_escape_ok, &svp_ll, &via_dirs_obj,
            &env_obj, &hist_obj, &neg_pending_hist, &hsn_ll,
            &layers_obj, &vias_obj,
            &mirror_ctx_obj, &mirror_axis2_ll, &mirror_intent_ll)) {
        return -1;
    }

    auto* c = self->core;
    c->release_py_refs();
    c->agg_rejs.clear();
    c->pending_rej_count = 0;
    c->total_rej_gen = 0;
    c->mirror_pair_conflict_seen = false;

    c->net_tag = c->keep_ref(net_obj);
    c->n_layers = n_layers;
    c->grid = std::max<int64_t>(1, static_cast<int64_t>(grid_ll));
    c->rc = std::max<int64_t>(1, static_cast<int64_t>(rc_ll));
    c->pres = static_cast<int64_t>(pres_ll);
    c->mul_scale = static_cast<int64_t>(mul_ll);
    c->cost_factor_scale = std::max<int64_t>(1, static_cast<int64_t>(cfs_ll));
    c->turn_cost = static_cast<int64_t>(turn_ll);
    c->via_cost = static_cast<int64_t>(via_ll);
    c->step_cost = static_cast<int64_t>(step_ll);
    c->off_pref_cost = static_cast<int64_t>(off_ll);
    c->single_escape_ok = (single_escape_ok != 0);
    c->single_via_penalty = static_cast<int64_t>(svp_ll);
    c->neg_pending_hist = (neg_pending_hist != 0);
    c->hist_scale_num = static_cast<int64_t>(hsn_ll);

    {
        PyRef dseq = PyRef::adopt(PySequence_Fast(via_dirs_obj, "via_succ_dirs must be a 3-sequence"));
        if (!dseq) return -1;
        if (PySequence_Fast_GET_SIZE(dseq.get()) != 3) {
            PyErr_SetString(PyExc_ValueError, "via_succ_dirs must have length 3");
            return -1;
        }
        for (int i = 0; i < 3; ++i) {
            int64_t dv = 0;
            if (!parse_int64(PySequence_Fast_GET_ITEM(dseq.get(), i), dv)) return -1;
            c->via_succ_dirs[i] = static_cast<int32_t>(dv);
        }
    }

    c->has_env = (env_obj != Py_None);
    if (c->has_env && !parse_rect_norm(env_obj, c->env_box)) return -1;

    bool ok = true;
    const HistoryGridCore* hg = extract_history_core(hist_obj, ok);
    if (!ok) return -1;
    c->keep_ref(hist_obj);
    c->hist_grid = hg;

    PyRef lseq = PyRef::adopt(PySequence_Fast(layers_obj, "layers must be a sequence"));
    if (!lseq) return -1;
    if (PySequence_Fast_GET_SIZE(lseq.get()) != n_layers) {
        PyErr_SetString(PyExc_ValueError, "len(layers) must match n_layers");
        return -1;
    }
    c->layers.resize(static_cast<size_t>(n_layers));
    bool err = false;
    for (int32_t gi = 0; gi < n_layers; ++gi) {
        PyObject* row = PySequence_Fast_GET_ITEM(lseq.get(), gi);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 19) {
            PyErr_SetString(PyExc_TypeError, "each layer spec must be a 19-tuple");
            return -1;
        }
        PureEdgeLayerData& L = c->layers[static_cast<size_t>(gi)];
        if (!parse_int64(PyTuple_GET_ITEM(row, 0), L.hw) ||
            !parse_int64(PyTuple_GET_ITEM(row, 1), L.min_hw) ||
            !parse_int64(PyTuple_GET_ITEM(row, 2), L.space) ||
            !parse_int64(PyTuple_GET_ITEM(row, 3), L.base_rate)) {
            return -1;
        }
        const int ph_tr = PyObject_IsTrue(PyTuple_GET_ITEM(row, 4));
        if (ph_tr < 0) return -1;
        L.pref_h = (ph_tr == 1);
        L.off_pref_weight = PyFloat_AsDouble(PyTuple_GET_ITEM(row, 5));
        if (L.off_pref_weight == -1.0 && PyErr_Occurred()) return -1;
        L.off_pref_weight_is_one = (L.off_pref_weight == 1.0);
        L.net_is_supra_min = (L.hw > L.min_hw);

        PyObject* obs_o = PyTuple_GET_ITEM(row, 6);
        PyObject* met_o = PyTuple_GET_ITEM(row, 7);
        PyObject* att_o = PyTuple_GET_ITEM(row, 8);
        const SpatialIndexCore* py_obs_idx = extract_spatial_core(obs_o, ok); if (!ok) return -1;
        const SpatialIndexCore* py_met_idx = extract_spatial_core(met_o, ok); if (!ok) return -1;
        const SpatialIndexCore* py_att_idx = extract_spatial_core(att_o, ok); if (!ok) return -1;
        c->keep_ref(obs_o);
        c->keep_ref(met_o);
        c->keep_ref(att_o);
        L.obs_idx = py_obs_idx;
        L.metal_idx = py_met_idx;
        L.att_idx = py_att_idx;
        L.net_obs_id = py_obs_idx ? py_obs_idx->lookup_owner(net_obj, err) : UINT32_MAX;
        if (err) return -1;
        L.net_metal_id = py_met_idx ? py_met_idx->lookup_owner(net_obj, err) : UINT32_MAX;
        if (err) return -1;

        if (!parse_rect_list(PyTuple_GET_ITEM(row, 9), L.attempt_metal)) return -1;

        PyObject* abs_o = PyTuple_GET_ITEM(row, 10);
        if (abs_o != Py_None) {
            std::vector<Box64> tmp;
            if (!parse_rect_list(abs_o, tmp)) return -1;
            L.abstract_set.reserve(tmp.size() * 2 + 1);
            for (const auto& b : tmp) L.abstract_set.insert(b);
        }

        PyObject* vp_o = PyTuple_GET_ITEM(row, 11);
        if (py_met_idx && vp_o != Py_None) {
            const Py_ssize_t vp_len = PyObject_Length(vp_o);
            if (vp_len < 0) return -1;
            if (vp_len > 0) {
                PyRef it = PyRef::adopt(PyObject_GetIter(vp_o));
                if (!it) return -1;
                while (true) {
                    PyRef item = PyRef::adopt(PyIter_Next(it.get()));
                    if (!item) break;
                    if (!PyTuple_CheckExact(item.get()) || PyTuple_GET_SIZE(item.get()) != 2) {
                        PyErr_SetString(PyExc_TypeError, "via_pad_provenance item must be (owner, rect)");
                        return -1;
                    }
                    const uint32_t oid = py_met_idx->lookup_owner(PyTuple_GET_ITEM(item.get(), 0), err);
                    if (err) return -1;
                    if (oid != UINT32_MAX) {
                        Box64 b;
                        if (!parse_rect_norm(PyTuple_GET_ITEM(item.get(), 1), b)) return -1;
                        L.via_pad_set.insert(OwnerBoxKey{oid, b});
                    }
                }
                if (PyErr_Occurred()) return -1;
            }
        }

        PyObject* wo_o = PyTuple_GET_ITEM(row, 12);
        if (py_met_idx && wo_o != Py_None) {
            PyRef woseq = PyRef::adopt(PySequence_Fast(wo_o, "widened_owners must be a sequence"));
            if (!woseq) return -1;
            const Py_ssize_t nwo = PySequence_Fast_GET_SIZE(woseq.get());
            if (nwo > 0) {
                L.metal_owner_supra_min.assign(py_met_idx->next_owner_id, 0);
                for (Py_ssize_t i = 0; i < nwo; ++i) {
                    const uint32_t oid = py_met_idx->lookup_owner(PySequence_Fast_GET_ITEM(woseq.get(), i), err);
                    if (err) return -1;
                    if (oid < L.metal_owner_supra_min.size()) L.metal_owner_supra_min[oid] = 1;
                }
            }
        }

        PyObject* po_o = PyTuple_GET_ITEM(row, 13);
        if (py_met_idx && po_o != Py_None) {
            PyRef poseq = PyRef::adopt(PySequence_Fast(po_o, "persistent_owners must be a sequence"));
            if (!poseq) return -1;
            const Py_ssize_t npo = PySequence_Fast_GET_SIZE(poseq.get());
            if (npo > 0) {
                L.metal_owner_persistent.assign(py_met_idx->next_owner_id, 0);
                for (Py_ssize_t i = 0; i < npo; ++i) {
                    PyObject* pr = PySequence_Fast_GET_ITEM(poseq.get(), i);
                    if (!PyTuple_CheckExact(pr) || PyTuple_GET_SIZE(pr) != 2) {
                        PyErr_SetString(PyExc_TypeError, "persistent_owner entry must be (owner, always_hard)");
                        return -1;
                    }
                    const uint32_t oid = py_met_idx->lookup_owner(PyTuple_GET_ITEM(pr, 0), err);
                    if (err) return -1;
                    if (oid < L.metal_owner_persistent.size()) {
                        const int ah_tr = PyObject_IsTrue(PyTuple_GET_ITEM(pr, 1));
                        if (ah_tr < 0) return -1;
                        L.metal_owner_persistent[oid] = (ah_tr == 1) ? 1 : 2;
                    }
                }
            }
        }

        PyObject* mwg_o = PyTuple_GET_ITEM(row, 14);
        if (mwg_o != Py_None) {
            L.has_wide_rule = true;
            if (!parse_int64(mwg_o, L.min_wide_ge)) return -1;
        }
        PyObject* we_o = PyTuple_GET_ITEM(row, 15);
        if (we_o != Py_None) {
            PyRef weseq = PyRef::adopt(PySequence_Fast(we_o, "wide_entries must be a sequence"));
            if (!weseq) return -1;
            const Py_ssize_t nwe = PySequence_Fast_GET_SIZE(weseq.get());
            L.wide_entries.reserve(static_cast<size_t>(nwe));
            for (Py_ssize_t i = 0; i < nwe; ++i) {
                PyObject* wr = PySequence_Fast_GET_ITEM(weseq.get(), i);
                if (!PyTuple_CheckExact(wr) || PyTuple_GET_SIZE(wr) != 3) {
                    PyErr_SetString(PyExc_TypeError, "wide entry must be (w_sp, core_idx, att_idx)");
                    return -1;
                }
                PureWideCheckEntry we;
                if (!parse_int64(PyTuple_GET_ITEM(wr, 0), we.w_sp)) return -1;
                PyObject* co = PyTuple_GET_ITEM(wr, 1);
                PyObject* ao = PyTuple_GET_ITEM(wr, 2);
                const SpatialIndexCore* py_co = extract_spatial_core(co, ok); if (!ok) return -1;
                const SpatialIndexCore* py_ao = extract_spatial_core(ao, ok); if (!ok) return -1;
                c->keep_ref(co);
                c->keep_ref(ao);
                we.core_idx = py_co;
                we.att_idx = py_ao;
                we.own_core = py_co ? py_co->lookup_owner(net_obj, err) : UINT32_MAX;
                if (err) return -1;
                we.own_att = py_ao ? py_ao->lookup_owner(net_obj, err) : UINT32_MAX;
                if (err) return -1;
                L.wide_entries.push_back(we);
            }
        }

        PyObject* mb_o = PyTuple_GET_ITEM(row, 16);
        if (mb_o != Py_None) {
            PyRef s = PyRef::adopt(PySequence_Fast(mb_o, "mim_bottom must be a sequence"));
            if (!s) return -1;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(s.get()); ++i) {
                PyObject* r = PySequence_Fast_GET_ITEM(s.get(), i);
                if (!PyTuple_CheckExact(r) || PyTuple_GET_SIZE(r) != 3) {
                    PyErr_SetString(PyExc_TypeError, "mim_bottom entry must be 3-tuple");
                    return -1;
                }
                PureMimClearanceItem m;
                if (!parse_rect_norm(PyTuple_GET_ITEM(r, 1), m.blocker) ||
                    !parse_int64(PyTuple_GET_ITEM(r, 2), m.clearance)) {
                    return -1;
                }
                m.owner_tag = c->keep_ref(PyTuple_GET_ITEM(r, 0));
                L.mim_bottom.push_back(m);
            }
        }

        PyObject* md_o = PyTuple_GET_ITEM(row, 17);
        if (md_o != Py_None) {
            PyRef s = PyRef::adopt(PySequence_Fast(md_o, "mim_derived must be a sequence"));
            if (!s) return -1;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(s.get()); ++i) {
                PyObject* r = PySequence_Fast_GET_ITEM(s.get(), i);
                if (!PyTuple_CheckExact(r) || PyTuple_GET_SIZE(r) != 5) {
                    PyErr_SetString(PyExc_TypeError, "mim_derived entry must be 5-tuple");
                    return -1;
                }
                PureMimDerivedItem m;
                if (!parse_rect_norm(PyTuple_GET_ITEM(r, 1), m.blocker) ||
                    !parse_int64(PyTuple_GET_ITEM(r, 2), m.clearance)) {
                    return -1;
                }
                const int eu_tr = PyObject_IsTrue(PyTuple_GET_ITEM(r, 3));
                if (eu_tr < 0) return -1;
                m.euclidean = (eu_tr == 1);
                m.owner_tag = c->keep_ref(PyTuple_GET_ITEM(r, 0));
                PyObject* rule_o = PyTuple_GET_ITEM(r, 4);
                m.rule_tag = c->keep_ref(rule_o);
                m.kind_id = c->intern_extra_kind(rule_o);
                L.mim_derived.push_back(m);
            }
        }

        PyObject* mc_o = PyTuple_GET_ITEM(row, 18);
        if (mc_o != Py_None) {
            PyRef s = PyRef::adopt(PySequence_Fast(mc_o, "mim_carve must be a sequence"));
            if (!s) return -1;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(s.get()); ++i) {
                PyObject* r = PySequence_Fast_GET_ITEM(s.get(), i);
                if (!PyTuple_CheckExact(r) || PyTuple_GET_SIZE(r) != 3) {
                    PyErr_SetString(PyExc_TypeError, "mim_carve entry must be 3-tuple");
                    return -1;
                }
                PureMimCarveItem m;
                if (!parse_rect_norm(PyTuple_GET_ITEM(r, 1), m.blocker)) return -1;
                m.owner_tag = c->keep_ref(PyTuple_GET_ITEM(r, 0));
                PyObject* kind_o = PyTuple_GET_ITEM(r, 2);
                m.kind_tag = c->keep_ref(kind_o);
                m.kind_id = c->intern_extra_kind(kind_o);
                L.mim_carve.push_back(m);
            }
        }
    }

    const int32_t n_vias = (n_layers > 0) ? (n_layers - 1) : 0;
    PyRef vseq = PyRef::adopt(PySequence_Fast(vias_obj, "vias must be a sequence"));
    if (!vseq) return -1;
    if (PySequence_Fast_GET_SIZE(vseq.get()) != n_vias) {
        PyErr_SetString(PyExc_ValueError, "len(vias) must match max(0, n_layers - 1)");
        return -1;
    }
    c->vias.resize(static_cast<size_t>(n_vias));
    for (int32_t lo = 0; lo < n_vias; ++lo) {
        PyObject* row = PySequence_Fast_GET_ITEM(vseq.get(), lo);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 15) {
            PyErr_SetString(PyExc_TypeError, "each via spec must be a 15-tuple");
            return -1;
        }
        PureEdgeViaData& V = c->vias[static_cast<size_t>(lo)];
        if (!parse_int64(PyTuple_GET_ITEM(row, 0), V.bh0) ||
            !parse_int64(PyTuple_GET_ITEM(row, 1), V.ah0) ||
            !parse_int64(PyTuple_GET_ITEM(row, 2), V.ext0) ||
            !parse_int64(PyTuple_GET_ITEM(row, 3), V.bh1) ||
            !parse_int64(PyTuple_GET_ITEM(row, 4), V.ah1) ||
            !parse_int64(PyTuple_GET_ITEM(row, 5), V.ext1) ||
            !parse_int64(PyTuple_GET_ITEM(row, 6), V.vsp)) {
            return -1;
        }
        PyObject* oc_o = PyTuple_GET_ITEM(row, 7);
        const SpatialIndexCore* py_oc = extract_spatial_core(oc_o, ok); if (!ok) return -1;
        c->keep_ref(oc_o);
        V.obs_cut_idx = py_oc;
        if (!parse_rect_list(PyTuple_GET_ITEM(row, 8), V.fixed_cuts_fallback)) return -1;

        PyObject* c_o = PyTuple_GET_ITEM(row, 9);
        const SpatialIndexCore* py_cut = extract_spatial_core(c_o, ok); if (!ok) return -1;
        c->keep_ref(c_o);
        V.cut_idx = py_cut;
        V.net_cut_id = py_cut ? py_cut->lookup_owner(net_obj, err) : UINT32_MAX;
        if (err) return -1;

        if (!parse_rect_list(PyTuple_GET_ITEM(row, 10), V.attempt_cuts)) return -1;

        PyObject* mf_o = PyTuple_GET_ITEM(row, 11);
        if (mf_o != Py_None) {
            PyRef s = PyRef::adopt(PySequence_Fast(mf_o, "mim_forb_cut must be a sequence"));
            if (!s) return -1;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(s.get()); ++i) {
                PyObject* r = PySequence_Fast_GET_ITEM(s.get(), i);
                if (!PyTuple_CheckExact(r) || PyTuple_GET_SIZE(r) != 2) {
                    PyErr_SetString(PyExc_TypeError, "mim_forb_cut entry must be 2-tuple");
                    return -1;
                }
                PureMimCutTouchItem m;
                if (!parse_rect_norm(PyTuple_GET_ITEM(r, 1), m.blocker)) return -1;
                m.owner_tag = c->keep_ref(PyTuple_GET_ITEM(r, 0));
                V.mim_forb_cut.push_back(m);
            }
        }

        PyObject* mt_o = PyTuple_GET_ITEM(row, 12);
        if (mt_o != Py_None) {
            PyRef s = PyRef::adopt(PySequence_Fast(mt_o, "mim_top_marker must be a sequence"));
            if (!s) return -1;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(s.get()); ++i) {
                PyObject* r = PySequence_Fast_GET_ITEM(s.get(), i);
                if (!PyTuple_CheckExact(r) || PyTuple_GET_SIZE(r) != 3) {
                    PyErr_SetString(PyExc_TypeError, "mim_top_marker entry must be 3-tuple");
                    return -1;
                }
                PureMimClearanceItem m;
                if (!parse_rect_norm(PyTuple_GET_ITEM(r, 1), m.blocker) ||
                    !parse_int64(PyTuple_GET_ITEM(r, 2), m.clearance)) {
                    return -1;
                }
                m.owner_tag = c->keep_ref(PyTuple_GET_ITEM(r, 0));
                V.mim_top_marker.push_back(m);
            }
        }

        PyObject* f0_o = PyTuple_GET_ITEM(row, 13);
        if (f0_o != Py_None) {
            std::vector<std::pair<int64_t, int64_t>> pts;
            if (!parse_xy_list(f0_o, pts)) return -1;
            for (const auto& p : pts) V.forb_xy_0.insert(p);
        }
        PyObject* f1_o = PyTuple_GET_ITEM(row, 14);
        if (f1_o != Py_None) {
            std::vector<std::pair<int64_t, int64_t>> pts;
            if (!parse_xy_list(f1_o, pts)) return -1;
            for (const auto& p : pts) V.forb_xy_1.insert(p);
        }
    }

    if (mirror_ctx_obj != Py_None) {
        if (Py_TYPE(mirror_ctx_obj) != Py_TYPE(self)) {
            PyErr_SetString(PyExc_TypeError, "mirror_ctx must be NativeEdgeContext or None");
            return -1;
        }
        auto* mobj = reinterpret_cast<PyNativeEdgeContext*>(mirror_ctx_obj);
        if (mobj->core->n_layers != n_layers) {
            PyErr_SetString(PyExc_ValueError, "mirror_ctx n_layers mismatch");
            return -1;
        }
        c->keep_ref(mirror_ctx_obj);
        c->mirror_ctx = mobj->core;
        c->mirror_axis2 = static_cast<int64_t>(mirror_axis2_ll);
        c->mirror_intent_space = static_cast<int64_t>(mirror_intent_ll);
        c->mirror_pair_conflict_seen = false;
    }
    return 0;
}

PyObject* NativeEdgeContext_check_wire_legal_single(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "check_wire_legal_single(gi, ax, ay, bx, by) expects 5 arguments");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], gi) || !parse_int64(args[1], ax) ||
        !parse_int64(args[2], ay) || !parse_int64(args[3], bx) || !parse_int64(args[4], by)) {
        return nullptr;
    }
    int64_t pres_out = 0;
    const int st = self->core->check_wire_legal_single(static_cast<int32_t>(gi), ax, ay, bx, by, pres_out);
    if (st < 0) Py_RETURN_NONE;
    if (st == 0) return PyLong_FromLong(-1);
    return PyLong_FromLongLong(pres_out);
}

PyObject* NativeEdgeContext_check_via_legal_single(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "check_via_legal_single(x, y, lo, n_flag) expects 4 arguments");
        return nullptr;
    }
    int64_t x = 0, y = 0, lo = 0, n_flag = 0;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y) ||
        !parse_int64(args[2], lo) || !parse_int64(args[3], n_flag)) {
        return nullptr;
    }
    int64_t p_lo = 0, p_hi = 0, p_cut = 0;
    const int st = self->core->check_via_legal_single(x, y, static_cast<int32_t>(lo),
                                                      static_cast<int32_t>(n_flag), p_lo, p_hi, p_cut);
    if (st < 0) Py_RETURN_NONE;
    if (st == 0) Py_RETURN_FALSE;
    PyRef out = PyRef::adopt(PyTuple_New(3));
    PyRef o0 = PyRef::adopt(PyLong_FromLongLong(p_lo));
    PyRef o1 = PyRef::adopt(PyLong_FromLongLong(p_hi));
    PyRef o2 = PyRef::adopt(PyLong_FromLongLong(p_cut));
    if (!out || !o0 || !o1 || !o2) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o0.release());
    PyTuple_SET_ITEM(out.get(), 1, o1.release());
    PyTuple_SET_ITEM(out.get(), 2, o2.release());
    return out.release();
}

PyObject* NativeEdgeContext_check_wire_legal(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "check_wire_legal(gi, ax, ay, bx, by) expects 5 arguments");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], gi) || !parse_int64(args[1], ax) ||
        !parse_int64(args[2], ay) || !parse_int64(args[3], bx) || !parse_int64(args[4], by)) {
        return nullptr;
    }
    int64_t pres_out = 0;
    const int st = self->core->check_wire_legal(static_cast<int32_t>(gi), ax, ay, bx, by, pres_out);
    if (st < 0) Py_RETURN_NONE;
    if (st == 0) return PyLong_FromLong(-1);
    return PyLong_FromLongLong(pres_out);
}

PyObject* NativeEdgeContext_check_via_legal(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "check_via_legal(x, y, lo, n_flag) expects 4 arguments");
        return nullptr;
    }
    int64_t x = 0, y = 0, lo = 0, n_flag = 0;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y) ||
        !parse_int64(args[2], lo) || !parse_int64(args[3], n_flag)) {
        return nullptr;
    }
    int64_t p_lo = 0, p_hi = 0, p_cut = 0;
    const int st = self->core->check_via_legal(x, y, static_cast<int32_t>(lo),
                                               static_cast<int32_t>(n_flag), p_lo, p_hi, p_cut);
    if (st < 0) Py_RETURN_NONE;
    if (st == 0) Py_RETURN_FALSE;
    PyRef out = PyRef::adopt(PyTuple_New(3));
    PyRef o0 = PyRef::adopt(PyLong_FromLongLong(p_lo));
    PyRef o1 = PyRef::adopt(PyLong_FromLongLong(p_hi));
    PyRef o2 = PyRef::adopt(PyLong_FromLongLong(p_cut));
    if (!out || !o0 || !o1 || !o2) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o0.release());
    PyTuple_SET_ITEM(out.get(), 1, o1.release());
    PyTuple_SET_ITEM(out.get(), 2, o2.release());
    return out.release();
}

PyObject* NativeEdgeContext_box_congestion(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "box_congestion(gi, x0, y0, x1, y1, half, present_hint)");
        return nullptr;
    }
    int64_t gi = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0, half = 0, phint = -1;
    if (!parse_int64(args[0], gi) || !parse_int64(args[1], x0) ||
        !parse_int64(args[2], y0) || !parse_int64(args[3], x1) ||
        !parse_int64(args[4], y1) || !parse_int64(args[5], half) ||
        !parse_int64(args[6], phint)) {
        return nullptr;
    }
    int64_t out_c = 0;
    const int st = self->core->eval_box_congestion(
        static_cast<int32_t>(gi), normalize_box(x0, y0, x1, y1), half, phint, out_c);
    if (st < 0) Py_RETURN_NONE;
    return PyLong_FromLongLong(out_c);
}

PyObject* NativeEdgeContext_via_congestion(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "via_congestion(x, y, lo, n_flag, pres_lo, pres_hi, pres_cut)");
        return nullptr;
    }
    int64_t x = 0, y = 0, lo = 0, nf = 0, pl = -1, ph = -1, pc = -1;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y) ||
        !parse_int64(args[2], lo) || !parse_int64(args[3], nf) ||
        !parse_int64(args[4], pl) || !parse_int64(args[5], ph) ||
        !parse_int64(args[6], pc)) {
        return nullptr;
    }
    int64_t out_c = 0;
    const int st = self->core->eval_via_congestion(
        x, y, static_cast<int32_t>(lo), static_cast<int32_t>(nf), pl, ph, pc, out_c);
    if (st < 0) Py_RETURN_NONE;
    return PyLong_FromLongLong(out_c);
}

PyObject* NativeEdgeContext_wire_static_cost(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "wire_static_cost(gi, ax, ay, bx, by) expects 5 arguments");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], gi) || !parse_int64(args[1], ax) ||
        !parse_int64(args[2], ay) || !parse_int64(args[3], bx) || !parse_int64(args[4], by)) {
        return nullptr;
    }
    int64_t pres_out = 0;
    const int st = self->core->check_wire_legal(static_cast<int32_t>(gi), ax, ay, bx, by, pres_out);
    if (st < 0) Py_RETURN_NONE;
    if (st == 0) return PyLong_FromLong(-1);
    int64_t cost = 0;
    if (self->core->eval_wire_price(static_cast<int32_t>(gi), ax, ay, bx, by, pres_out, cost) < 0) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLongLong(cost);
}

PyObject* NativeEdgeContext_via_static_cost(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "via_static_cost(x, y, lo) expects 3 arguments");
        return nullptr;
    }
    int64_t x = 0, y = 0, lo = 0;
    if (!parse_int64(args[0], x) || !parse_int64(args[1], y) || !parse_int64(args[2], lo)) {
        return nullptr;
    }
    int64_t cost = 0;
    int32_t model_n = 0;
    const int st = self->core->eval_via_static_cost_fresh(x, y, static_cast<int32_t>(lo), cost, model_n);
    if (st < 0) Py_RETURN_NONE;
    if (st == 0) Py_RETURN_FALSE;
    PyRef out = PyRef::adopt(PyTuple_New(2));
    PyRef o0 = PyRef::adopt(PyLong_FromLongLong(cost));
    PyRef o1 = PyRef::adopt(PyLong_FromLong(model_n));
    if (!out || !o0 || !o1) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o0.release());
    PyTuple_SET_ITEM(out.get(), 1, o1.release());
    return out.release();
}

PyObject* NativeEdgeContext_pending_history_lb(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "pending_history_lb(gi, ax, ay, bx, by) expects 5 arguments");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], gi) || !parse_int64(args[1], ax) ||
        !parse_int64(args[2], ay) || !parse_int64(args[3], bx) || !parse_int64(args[4], by)) {
        return nullptr;
    }
    auto* c = self->core;
    if (!c->neg_pending_hist || !c->hist_grid || (ax == bx && ay == by)) {
        return PyLong_FromLong(0);
    }
    if (gi < 0 || gi >= c->n_layers) Py_RETURN_NONE;
    const PureEdgeLayerData& L = c->layers[static_cast<size_t>(gi)];
    const int64_t pad = L.hw + floor_div_pos(L.space, 2);
    bool ovf = false;
    const int64_t val = c->hist_grid->eval_pending_lb(
        gi, ax, ay, bx, by, pad, c->rc, c->hist_scale_num, c->cost_factor_scale, ovf);
    if (ovf) Py_RETURN_NONE;
    return PyLong_FromLongLong(val);
}

PyObject* NativeEdgeContext_flush_rejections(PyNativeEdgeContext* self, PyObject*) {
    return self->core->flush_rejections_py();
}

PyObject* NativeEdgeContext_take_mirror_conflict_seen(PyNativeEdgeContext* self, PyObject*) {
    const bool seen = self->core->mirror_pair_conflict_seen;
    self->core->mirror_pair_conflict_seen = false;
    if (seen) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* NativeEdgeContext_get_pending_rej_count(PyNativeEdgeContext* self, void*) {
    return PyLong_FromLongLong(self->core->total_pending_rej());
}

PyObject* NativeEdgeContext_get_total_rej_gen(PyNativeEdgeContext* self, void*) {
    return PyLong_FromLongLong(self->core->total_rej_generation());
}

PyObject* NativeEdgeContext_get_has_mirror(PyNativeEdgeContext* self, void*) {
    if (self->core->mirror_ctx != nullptr) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyMethodDef NativeEdgeContext_methods[] = {
    {"check_wire_legal", reinterpret_cast<PyCFunction>(NativeEdgeContext_check_wire_legal), METH_FASTCALL,
     "Check wire legality (including active mirror) in C++; returns present_total (>=0), -1 (illegal), or None (fallback)."},
    {"check_wire_legal_single", reinterpret_cast<PyCFunction>(NativeEdgeContext_check_wire_legal_single), METH_FASTCALL,
     "Check single-net wire legality in C++; returns present_total (>=0), -1 (illegal), or None (fallback)."},
    {"check_via_legal", reinterpret_cast<PyCFunction>(NativeEdgeContext_check_via_legal), METH_FASTCALL,
     "Check via legality (including active mirror) in C++; returns (pres_lo, pres_hi, pres_cut), False (illegal), or None (fallback)."},
    {"check_via_legal_single", reinterpret_cast<PyCFunction>(NativeEdgeContext_check_via_legal_single), METH_FASTCALL,
     "Check single-net via legality in C++; returns (pres_lo, pres_hi, pres_cut), False (illegal), or None (fallback)."},
    {"box_congestion", reinterpret_cast<PyCFunction>(NativeEdgeContext_box_congestion), METH_FASTCALL,
     "Compute exact box congestion via NativeEdgeContext."},
    {"via_congestion", reinterpret_cast<PyCFunction>(NativeEdgeContext_via_congestion), METH_FASTCALL,
     "Compute exact via congestion via NativeEdgeContext."},
    {"wire_static_cost", reinterpret_cast<PyCFunction>(NativeEdgeContext_wire_static_cost), METH_FASTCALL,
     "Compute wire legality + static cost in one C++ call."},
    {"via_static_cost", reinterpret_cast<PyCFunction>(NativeEdgeContext_via_static_cost), METH_FASTCALL,
     "Compute via legality + default/single selection + static cost in one C++ call."},
    {"pending_history_lb", reinterpret_cast<PyCFunction>(NativeEdgeContext_pending_history_lb), METH_FASTCALL,
     "Compute exact pending_history_lb in O(1) via NativeHistoryGrid SAT."},
    {"flush_rejections", reinterpret_cast<PyCFunction>(NativeEdgeContext_flush_rejections), METH_NOARGS,
     "Drain buffered aggregated rejections as 9-tuples and clear buffer."},
    {"take_mirror_conflict_seen", reinterpret_cast<PyCFunction>(NativeEdgeContext_take_mirror_conflict_seen), METH_NOARGS,
     "Return and reset mirror_pair_conflict_seen flag."},
    {"run_hard_connectivity_preflight", reinterpret_cast<PyCFunction>(NativeEdgeContext_run_hard_connectivity_preflight), METH_FASTCALL,
     "Run exact hard_connectivity_preflight BFS in C++20, populating shared packed edge cache."},
    {"run_cartesian_astar", reinterpret_cast<PyCFunction>(NativeEdgeContext_run_cartesian_astar), METH_FASTCALL,
     "Run exact Cartesian 4-tuple A* in C++20 with shared packed edge cache."},
    {"advance_region_step_ctx", reinterpret_cast<PyCFunction>(NativeEdgeContext_advance_region_step_ctx), METH_FASTCALL,
     "Fused C++20 _advance_planar_region_step with NativeEdgeContext and NativeAxisPriceCache."},
    {nullptr, nullptr, 0, nullptr}
};

PyGetSetDef NativeEdgeContext_getset[] = {
    {"pending_rej_count", reinterpret_cast<getter>(NativeEdgeContext_get_pending_rej_count), nullptr,
     "Number of buffered rejection events awaiting flush.", nullptr},
    {"total_rej_gen", reinterpret_cast<getter>(NativeEdgeContext_get_total_rej_gen), nullptr,
     "Monotonic rejection event count recorded on this context.", nullptr},
    {"has_mirror", reinterpret_cast<getter>(NativeEdgeContext_get_has_mirror), nullptr,
     "True when active mirror legality partner context is attached.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

}  // namespace

PyTypeObject NativeEdgeContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.NativeEdgeContext",
    sizeof(PyNativeEdgeContext),
    0,
    reinterpret_cast<destructor>(NativeEdgeContext_dealloc),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "Per-_route_edge C++20 immutable world snapshot for legality, pricing, preflight, A*, and region steps.",
    0, 0, 0, 0, 0, 0,
    NativeEdgeContext_methods,
    0,
    NativeEdgeContext_getset,
    0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(NativeEdgeContext_init),
    0,
    NativeEdgeContext_new,
};

// ---------------------------------------------------------------------------
// 8. Module-level geometry functions
// ---------------------------------------------------------------------------

PyObject* py_rects_clear_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "rects_clear_cpp(a, b, space_nm) expects 3 arguments");
        return nullptr;
    }
    Box64 a, b;
    int64_t sp = 0;
    if (!parse_rect_norm(args[0], a) || !parse_rect_norm(args[1], b) || !parse_int64(args[2], sp)) {
        return nullptr;
    }
    if (rects_clear_linf(a, b, sp)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* py_rects_clear_euclidean_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "rects_clear_euclidean_cpp(a, b, space_nm) expects 3 arguments");
        return nullptr;
    }
    Box64 a, b;
    int64_t sp = 0;
    if (!parse_rect_norm(args[0], a) || !parse_rect_norm(args[1], b) || !parse_int64(args[2], sp)) {
        return nullptr;
    }
    if (rects_clear_euclid_exact(a, b, sp)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* py_closed_rects_touch_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "closed_rects_touch_cpp(a, b) expects 2 arguments");
        return nullptr;
    }
    Box64 a, b;
    if (!parse_rect_norm(args[0], a) || !parse_rect_norm(args[1], b)) return nullptr;
    if (closed_rects_touch(a, b)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* py_sep_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "sep_cpp(a, b) expects 2 arguments");
        return nullptr;
    }
    Box64 a, b;
    if (!parse_rect_norm(args[0], a) || !parse_rect_norm(args[1], b)) return nullptr;
    int64_t sx = 0, sy = 0;
    box_sep(a, b, sx, sy);
    return make_xy_tuple(sx, sy);
}

}  // namespace clay_pnr::route::py_bind
