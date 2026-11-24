#pragma once

// Internal CPython binding declarations shared across Clay's modular _gridless_cpp TUs:
//   - gridless_spatial_py.cpp
//   - gridless_search_py.cpp
//   - py_gridless_module.cpp
//
// Pure C++20 core headers/sources MUST NEVER include this header.
// All core algorithms & geometry live in pure C++20 `gridless_core.hpp`,
// `gridless_spatial.hpp`, and `gridless_search.hpp`.

#include "clay_pnr/core/py_raii.hpp"
#include "clay_pnr/route/gridless_core.hpp"
#include "clay_pnr/route/gridless_search.hpp"
#include "clay_pnr/route/gridless_spatial.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clay_pnr::route::py_bind {

using clay_pnr::py::PyRef;
using clay_pnr::py::make_box64_tuple;
using clay_pnr::py::parse_int64;
using clay_pnr::py::parse_int64_list;
using clay_pnr::py::parse_rect_list;
using clay_pnr::py::parse_rect_norm;
using clay_pnr::py::parse_xy_list;

inline PyObject* make_xy_tuple(int64_t x, int64_t y) {
    PyRef t = PyRef::adopt(PyTuple_New(2));
    if (!t) return nullptr;
    PyRef ox = PyRef::adopt(PyLong_FromLongLong(x));
    PyRef oy = PyRef::adopt(PyLong_FromLongLong(y));
    if (!ox || !oy) return nullptr;
    PyTuple_SET_ITEM(t.get(), 0, ox.release());
    PyTuple_SET_ITEM(t.get(), 1, oy.release());
    return t.release();
}

inline bool parse_int32_list(PyObject* obj, std::vector<int32_t>& out) {
    std::vector<int64_t> tmp;
    if (!parse_int64_list(obj, tmp)) return false;
    out.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        out[i] = static_cast<int32_t>(tmp[i]);
    }
    return true;
}

inline uint64_t py_to_tag(PyObject* obj) noexcept {
    if (!obj || obj == Py_None) return 0ULL;
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(obj));
}

inline PyObject* tag_to_py_borrowed(uint64_t tag) noexcept {
    return (tag == 0ULL) ? Py_None : reinterpret_cast<PyObject*>(static_cast<uintptr_t>(tag));
}

// ---------------------------------------------------------------------------
// 1. SpatialIndexCore (thin Python sidecar over PureSpatialIndex)
// ---------------------------------------------------------------------------

struct PySpatialSidecar {
    PyObject* owner_obj = nullptr;   // owned ref
    PyObject* rect_obj = nullptr;    // owned ref
    PyObject* item_tuple = nullptr;  // owned ref: (owner_obj, rect_obj)
    PyObject* norm_tuple = nullptr;  // owned ref: normalized 4-tuple
};

struct SpatialIndexCore : public PureSpatialIndex {
    std::vector<PySpatialSidecar> py_entries;  // parallel to PureSpatialIndex::entries

    // Owner interning: 0 == Py_None; 1..N == interned UTF-8 strings or custom objects
    std::unordered_map<std::string, uint32_t> str_owner_to_id;
    std::vector<std::pair<PyObject*, uint32_t>> custom_owners;  // owned refs
    std::vector<std::string> owner_sort_keys;  // owner_id -> "" for None else str(owner)
    uint32_t next_owner_id = 1;

    SpatialIndexCore();
    ~SpatialIndexCore();
    SpatialIndexCore(const SpatialIndexCore&) = delete;
    SpatialIndexCore& operator=(const SpatialIndexCore&) = delete;

    uint32_t intern_owner(PyObject* owner, bool& err);
    uint32_t lookup_owner(PyObject* owner, bool& err) const;
    bool add(PyObject* owner, PyObject* rect_obj);
};

struct PyNativeSpatialIndex {
    PyObject_HEAD
    SpatialIndexCore* core;
};

extern PyTypeObject NativeSpatialIndexType;

inline const SpatialIndexCore* extract_spatial_core(PyObject* obj, bool& ok) noexcept {
    ok = true;
    if (!obj || obj == Py_None) return nullptr;
    if (Py_TYPE(obj) != &NativeSpatialIndexType) {
        PyErr_SetString(PyExc_TypeError, "expected NativeSpatialIndex or None");
        ok = false;
        return nullptr;
    }
    return reinterpret_cast<PyNativeSpatialIndex*>(obj)->core;
}

inline bool spatial_has_foreign_intersecting_fast(
    const SpatialIndexCore& core, const Box64& q, uint32_t own_id) noexcept {
    return core.has_foreign_intersecting(q, own_id);
}

inline bool spatial_has_foreign_linf_conflict_fast(
    const SpatialIndexCore& core, const Box64& box, int64_t sp, uint32_t own_id) noexcept {
    return core.has_foreign_linf_conflict(box, sp, own_id);
}

inline bool spatial_has_any_intersecting_fast(
    const SpatialIndexCore& core, const Box64& q) noexcept {
    return core.has_any_intersecting(q);
}

inline bool raw_gap_metallised_cpp(
    const Box64& candidate,
    const Box64& blocker,
    int64_t sx,
    int64_t sy,
    const std::vector<Box64>& attempt_metal,
    const SpatialIndexCore* obs_idx,
    const std::unordered_set<Box64, Box64Hash>& abstract_set,
    const SpatialIndexCore* metal_idx,
    std::vector<Box64>& cover_buf) {
    return raw_gap_metallised_pure(candidate, blocker, sx, sy, attempt_metal,
                                   obs_idx, abstract_set, metal_idx, cover_buf);
}

// ---------------------------------------------------------------------------
// 2. PyNative2DPotential, PyNativeLayerHmin & PyNativeHistoryGrid
// ---------------------------------------------------------------------------

struct PyNative2DPotential {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    Potential2DCore* core;
};

extern PyTypeObject Native2DPotentialType;

struct LayerHminPyHolder {
    LayerHminCore core;
    std::vector<PyObject*> pot_refs;

    void clear_refs() noexcept {
        for (PyObject* p : pot_refs) Py_XDECREF(p);
        pot_refs.clear();
    }

    ~LayerHminPyHolder() {
        clear_refs();
    }
};

struct PyNativeLayerHmin {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    LayerHminPyHolder* holder;
};

extern PyTypeObject NativeLayerHminType;

struct PyNativeHistoryGrid {
    PyObject_HEAD
    HistoryGridCore* core;
};

extern PyTypeObject NativeHistoryGridType;

inline const HistoryGridCore* extract_history_core(PyObject* obj, bool& ok) noexcept {
    ok = true;
    if (!obj || obj == Py_None) return nullptr;
    if (Py_TYPE(obj) != &NativeHistoryGridType) {
        PyErr_SetString(PyExc_TypeError, "expected NativeHistoryGrid or None");
        ok = false;
        return nullptr;
    }
    return reinterpret_cast<PyNativeHistoryGrid*>(obj)->core;
}

// ---------------------------------------------------------------------------
// 3. EdgeContextCore (thin Python lifetime holder over PureEdgeContext)
// ---------------------------------------------------------------------------

struct EdgeContextCore : public PureEdgeContext {
    std::vector<PyObject*> kept_refs;
    uint32_t next_extra_kind_id = 1;

    EdgeContextCore() = default;
    ~EdgeContextCore();
    EdgeContextCore(const EdgeContextCore&) = delete;
    EdgeContextCore& operator=(const EdgeContextCore&) = delete;

    void release_py_refs() noexcept;

    inline uint64_t keep_ref(PyObject* obj) {
        if (!obj || obj == Py_None) return 0ULL;
        Py_INCREF(obj);
        kept_refs.push_back(obj);
        return py_to_tag(obj);
    }

    inline uint32_t intern_extra_kind(PyObject* obj) {
        if (!obj || obj == Py_None) return 0u;
        keep_ref(obj);
        return next_extra_kind_id++;
    }

    bool append_rejs_to_list(
        std::unordered_map<EdgeRejKey, PureEdgeRejVal, EdgeRejKeyHash>& map,
        PyObject* py_rejs, Py_ssize_t& ridx);
    PyObject* flush_rejections_py();
};

struct PyNativeEdgeContext {
    PyObject_HEAD
    EdgeContextCore* core;
};

extern PyTypeObject NativeEdgeContextType;

// ---------------------------------------------------------------------------
// 4. PyNativeAxisPriceCache & PyNativeViaTargetProjector
// ---------------------------------------------------------------------------

struct AxisPriceCacheCore : public PureAxisPriceCache {
    PyObject* str_price_segment = nullptr;
    PyObject* str_price_range = nullptr;

    AxisPriceCacheCore();
    ~AxisPriceCacheCore();
    AxisPriceCacheCore(const AxisPriceCacheCore&) = delete;
    AxisPriceCacheCore& operator=(const AxisPriceCacheCore&) = delete;

    int call_charge(PyObject* charge_fn, PyObject* kind_str) const;
    bool call_price(PyObject* price_fn, int64_t gi, int64_t ax, int64_t ay,
                    int64_t bx, int64_t by, int64_t& out_val) const;
    int eval_partial(PyObject* price_fn, PyObject* charge_fn,
                     int64_t gi, bool horizontal, int64_t fixed,
                     int64_t a, int64_t b, int64_t& out) const;
    int eval_range(PyObject* price_fn, PyObject* charge_fn,
                   int64_t gi, bool horizontal, int64_t fixed, int8_t sign,
                   int32_t begin, int32_t end, int64_t& out);
    int run(PyObject* price_fn, PyObject* charge_fn,
            int64_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
            int64_t& out);
};

struct PyNativeAxisPriceCache {
    PyObject_HEAD
    AxisPriceCacheCore* core;
};

extern PyTypeObject NativeAxisPriceCacheType;

struct ViaTargetProjectorCore : public PureViaTargetProjector {
    PyObject* hmin_obj = nullptr;
    PyObject* empty_tuple = nullptr;
    std::unordered_map<OwnerCacheKey, PyObject*, OwnerCacheKeyHash> owner_cache;
    std::vector<PyObject*> stored_u_bases;

    ViaTargetProjectorCore();
    ~ViaTargetProjectorCore();
    ViaTargetProjectorCore(const ViaTargetProjectorCore&) = delete;
    ViaTargetProjectorCore& operator=(const ViaTargetProjectorCore&) = delete;

    void clear_frontier();
    const std::vector<int32_t>* lookup_owner_ids(const SpatialIndexCore* other_idx, int64_t epx, int64_t epy);
    PyObject* lookup_owners_borrowed(const SpatialIndexCore* other_idx, int64_t epx, int64_t epy);
};

struct PyNativeViaTargetProjector {
    PyObject_HEAD
    ViaTargetProjectorCore* core;
};

extern PyTypeObject NativeViaTargetProjectorType;

// ---------------------------------------------------------------------------
// 5. Cross-TU methods & exported module-level CPython functions
// ---------------------------------------------------------------------------

PyObject* NativeEdgeContext_run_hard_connectivity_preflight(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* NativeEdgeContext_run_cartesian_astar(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* NativeEdgeContext_advance_region_step_ctx(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* py_rects_clear_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_rects_clear_euclidean_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_closed_rects_touch_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_sep_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_batch_filter_via_sites_cpp(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* py_compute_throat_events_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_batch_via_congestion_cpp(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* py_check_wire_legal_fast_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_check_via_legal_fast_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_box_congestion_fast_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_exact_distance_transform_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* py_project_portal_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_project_target_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_settled_axis_prefix_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_batch_project_portals_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
PyObject* py_batch_project_goal_targets_cpp(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

}  // namespace clay_pnr::route::py_bind
