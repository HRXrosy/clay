// CPython extension module entrypoint for clay._native._gridless_cpp.
// Registers all 7 native types, 16 module functions, and 20 REJ_* constants.

#include "clay_pnr/route/gridless_py_internal.hpp"

namespace {

using namespace clay_pnr::route;
using namespace clay_pnr::route::py_bind;

static PyMethodDef module_methods[] = {
    {"rects_clear_cpp", reinterpret_cast<PyCFunction>(py_rects_clear_cpp), METH_FASTCALL,
     "Exact L-infinity clearance check matching gridless_types._rects_clear(a, b, space_nm)."},
    {"rects_clear_euclidean_cpp", reinterpret_cast<PyCFunction>(py_rects_clear_euclidean_cpp), METH_FASTCALL,
     "Exact Euclidean clearance check (dx^2 + dy^2 >= space_nm^2)."},
    {"closed_rects_touch_cpp", reinterpret_cast<PyCFunction>(py_closed_rects_touch_cpp), METH_FASTCALL,
     "Exact closed axis-aligned rectangle contact predicate."},
    {"sep_cpp", reinterpret_cast<PyCFunction>(py_sep_cpp), METH_FASTCALL,
     "Exact axis-aligned edge-to-edge separation (sx, sy)."},
    {"batch_filter_via_sites_cpp", reinterpret_cast<PyCFunction>(py_batch_filter_via_sites_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "Batch C++20 owned-component via legality pre-screener with exact aggregated rejections."},
    {"compute_throat_events_cpp", reinterpret_cast<PyCFunction>(py_compute_throat_events_cpp), METH_FASTCALL,
     "Exact C++20 owned-atom throat event & Euclidean threshold candidate generator."},
    {"batch_via_congestion_cpp", reinterpret_cast<PyCFunction>(py_batch_via_congestion_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "Batch C++20 exact _via_congestion evaluator (metal + obs + attempt + cut + history)."},
    {"check_wire_legal_fast_cpp", reinterpret_cast<PyCFunction>(py_check_wire_legal_fast_cpp), METH_FASTCALL,
     "Fast C++20 wire legality verifier returning exact present_total on clean legal wires or None."},
    {"check_via_legal_fast_cpp", reinterpret_cast<PyCFunction>(py_check_via_legal_fast_cpp), METH_FASTCALL,
     "Fast C++20 via legality verifier returning (pres_lo, pres_hi, pres_cut) on clean legal vias or None."},
    {"box_congestion_fast_cpp", reinterpret_cast<PyCFunction>(py_box_congestion_fast_cpp), METH_FASTCALL,
     "Fast C++20 exact _box_congestion evaluator using NativeSpatialIndex and NativeHistoryGrid."},
    {"exact_distance_transform_cpp", reinterpret_cast<PyCFunction>(py_exact_distance_transform_cpp), METH_FASTCALL,
     "Exact 3D separable L1 min-plus distance transform."},
    {"project_portal_cpp", reinterpret_cast<PyCFunction>(py_project_portal_cpp), METH_FASTCALL,
     "Exact C++20 region_event_geometry.project_portal."},
    {"project_target_cpp", reinterpret_cast<PyCFunction>(py_project_target_cpp), METH_FASTCALL,
     "Exact C++20 region_event_geometry.project_target."},
    {"settled_axis_prefix_cpp", reinterpret_cast<PyCFunction>(py_settled_axis_prefix_cpp), METH_FASTCALL,
     "Exact C++20 region_event_geometry.settled_axis_prefix."},
    {"batch_project_portals_cpp", reinterpret_cast<PyCFunction>(py_batch_project_portals_cpp), METH_FASTCALL,
     "Batch C++20 portal projection and bucket-key generation for region A*."},
    {"batch_project_goal_targets_cpp", reinterpret_cast<PyCFunction>(py_batch_project_goal_targets_cpp), METH_FASTCALL,
     "Batch C++20 goal target projection for region A*."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef gridless_cpp_module = {
    PyModuleDef_HEAD_INIT,
    "_gridless_cpp",
    "C++20 native acceleration kernels for Clay's GridlessRouteEngine.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__gridless_cpp(void) {
    using namespace clay_pnr::route;
    using namespace clay_pnr::route::py_bind;

    if (PyType_Ready(&NativeSpatialIndexType) < 0) return nullptr;
    if (PyType_Ready(&Native2DPotentialType) < 0) return nullptr;
    if (PyType_Ready(&NativeLayerHminType) < 0) return nullptr;
    if (PyType_Ready(&NativeAxisPriceCacheType) < 0) return nullptr;
    if (PyType_Ready(&NativeViaTargetProjectorType) < 0) return nullptr;
    if (PyType_Ready(&NativeHistoryGridType) < 0) return nullptr;
    if (PyType_Ready(&NativeEdgeContextType) < 0) return nullptr;

    PyRef m = PyRef::adopt(PyModule_Create(&gridless_cpp_module));
    if (!m) return nullptr;

    auto add_type = [&](const char* name, PyTypeObject* tp) -> bool {
        Py_INCREF(tp);
        if (PyModule_AddObject(m.get(), name, reinterpret_cast<PyObject*>(tp)) < 0) {
            Py_DECREF(tp);
            return false;
        }
        return true;
    };

    if (!add_type("NativeSpatialIndex", &NativeSpatialIndexType) ||
        !add_type("Native2DPotential", &Native2DPotentialType) ||
        !add_type("NativeLayerHmin", &NativeLayerHminType) ||
        !add_type("NativeAxisPriceCache", &NativeAxisPriceCacheType) ||
        !add_type("NativeViaTargetProjector", &NativeViaTargetProjectorType) ||
        !add_type("NativeHistoryGrid", &NativeHistoryGridType) ||
        !add_type("NativeEdgeContext", &NativeEdgeContextType)) {
        return nullptr;
    }

    if (PyModule_AddIntConstant(m.get(), "REJ_CONFLICT_BRANCH_VIA", REJ_CONFLICT_BRANCH_VIA) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MANUFACTURING_GRID", REJ_MANUFACTURING_GRID) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_GROUP_ROUTE_LOCAL_ENVELOPE", REJ_GROUP_ROUTE_LOCAL_ENVELOPE) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_ROUTING_ENVELOPE", REJ_ROUTING_ENVELOPE) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_FIXED_METAL", REJ_FIXED_METAL) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_FIXED_SAME_NET_VIA_PAD", REJ_FIXED_SAME_NET_VIA_PAD) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_FIXED_CUT", REJ_FIXED_CUT) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_COMMITTED_METAL_CONTACT", REJ_COMMITTED_METAL_CONTACT) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_COMMITTED_METAL_SPACING", REJ_COMMITTED_METAL_SPACING) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_COMMITTED_CUT_CONTACT", REJ_COMMITTED_CUT_CONTACT) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_COMMITTED_CUT_SPACING", REJ_COMMITTED_CUT_SPACING) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_COMMITTED_SELF_CUT", REJ_COMMITTED_SELF_CUT) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_ATTEMPT_CUT", REJ_ATTEMPT_CUT) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIM_BOTTOM_ROUTING", REJ_MIM_BOTTOM_ROUTING) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIM_DERIVED_UNRELATED", REJ_MIM_DERIVED_UNRELATED) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIM_CARVE_BOTTOM", REJ_MIM_CARVE_BOTTOM) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIM_FORBIDDEN_CUT_TOUCH", REJ_MIM_FORBIDDEN_CUT_TOUCH) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIM_TOP_VIA_MARKER", REJ_MIM_TOP_VIA_MARKER) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIRROR_PAIR_METAL", REJ_MIRROR_PAIR_METAL) < 0 ||
        PyModule_AddIntConstant(m.get(), "REJ_MIRROR_PAIR_CUT", REJ_MIRROR_PAIR_CUT) < 0) {
        return nullptr;
    }

    return m.release();
}
