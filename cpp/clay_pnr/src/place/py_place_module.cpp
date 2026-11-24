// Thin RAII CPython binding module for clay._native._place_cpp.
// Delegates all algorithmic work to pure C++20 kernels in clay_pnr::place::*.
//
// Exports (100% identical API to legacy placement_kernels.cpp):
//   - rectangle_union_cover_cpp(boxes, checkpoint=None)
//   - forbidden_box_cover_cpp(left_atoms, right_atoms, required_nm, checkpoint=None)
//   - NativeRectangleIndex(boxes, checkpoint=None)
//       .conflicts(boxes, clearance_nm, checkpoint=None)
//       .intersecting(box, include_touching=False)
//   - exact_forbidden_union_cpp(boxes)
//   - longest_path_positions_cpp(nodes, edges, weight)
//   - pack_band_sp_cpp(gamma_plus, gamma_minus, blocks, gap)
//   - search_block_layout_cpp(block_sizes, dev_gap, aspect_target=None)
//   - search_band_dev_layout_cpp(row_half_ws, row_heights, dev_gap, aspect_target=None)
//   - placed_coord_nm_cpp(local_um, placement_um)
//   - placed_point_nm_cpp(x_um, y_um, placement_um)
//   - placed_ring_nm_cpp(ring, placement_um)

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)  // CPython 3.12 cpython/unicodeobject.h _unused_op
#endif
#include <Python.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "clay_pnr/core/geometry.hpp"
#include "clay_pnr/core/py_raii.hpp"
#include "clay_pnr/core/status.hpp"
#include "clay_pnr/place/bvh_index.hpp"
#include "clay_pnr/place/seqpair.hpp"
#include "clay_pnr/place/sweep_union.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

using clay_pnr::core::Box64;
using clay_pnr::core::CheckpointPoller;
using clay_pnr::core::IntStatus;
using clay_pnr::core::kCoordLimit60;
using clay_pnr::core::kCoordLimit61;
using clay_pnr::place::AtomRec;
using clay_pnr::place::BvhIndex;
using clay_pnr::place::ConflictResult;
using clay_pnr::place::DagEdge;
using clay_pnr::place::ExactUnionStatus;
using clay_pnr::place::ForbiddenCoverResult;
using clay_pnr::place::ForbiddenStatus;
using clay_pnr::place::LongestPathResult;
using clay_pnr::place::LongestPathStatus;
using clay_pnr::place::PackBandSpResult;
using clay_pnr::place::SearchLayoutResult;
using clay_pnr::place::SweepWorkspace;
using clay_pnr::place::exact_forbidden_union;
using clay_pnr::place::forbidden_box_cover;
using clay_pnr::place::longest_path_positions;
using clay_pnr::place::pack_band_sp;
using clay_pnr::place::rectangle_union_cover;
using clay_pnr::place::search_band_dev_layout;
using clay_pnr::place::search_block_layout;
using clay_pnr::py::PyCheckpointAdapter;
using clay_pnr::py::PyRef;
using clay_pnr::py::boxes_to_pytuple;
using clay_pnr::py::extract_exact_i64;
using clay_pnr::py::extract_exact_i64_bounded;
using clay_pnr::py::parse_box64_strict;

// ============================================================================
// 1. rectangle_union_cover_cpp
// ============================================================================

static PyObject* py_rectangle_union_cover_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"boxes", "checkpoint", nullptr};
    PyObject* boxes_obj = nullptr;
    PyObject* checkpoint_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:rectangle_union_cover_cpp",
                                     const_cast<char**>(kwlist),
                                     &boxes_obj, &checkpoint_obj)) {
        return nullptr;
    }

    PyCheckpointAdapter ckpt;
    if (!ckpt.init(checkpoint_obj)) {
        return nullptr;
    }
    CheckpointPoller poller = ckpt.poller();
    if (!poller.call_now()) {
        return nullptr;
    }

    std::vector<Box64> boxes;
    const char* kTypeErr = "rectangle union requires four exact integers";
    const char* kValErr = "rectangle union requires positive rectangles";

    if (PyTuple_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyTuple_GET_SIZE(boxes_obj);
        boxes.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!poller.poll_index(static_cast<size_t>(i))) return nullptr;
            Box64 b;
            if (!parse_box64_strict(PyTuple_GET_ITEM(boxes_obj, i), b, kTypeErr, kValErr, true, 0)) {
                return nullptr;
            }
            boxes.push_back(b);
        }
    } else if (PyList_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyList_GET_SIZE(boxes_obj);
        boxes.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!poller.poll_index(static_cast<size_t>(i))) return nullptr;
            Box64 b;
            if (!parse_box64_strict(PyList_GET_ITEM(boxes_obj, i), b, kTypeErr, kValErr, true, 0)) {
                return nullptr;
            }
            boxes.push_back(b);
        }
    } else {
        PyRef it = PyRef::adopt(PyObject_GetIter(boxes_obj));
        if (!it) return nullptr;
        size_t idx = 0;
        while (PyRef row = PyRef::adopt(PyIter_Next(it.get()))) {
            if (!poller.poll_index(idx++)) return nullptr;
            Box64 b;
            if (!parse_box64_strict(row.get(), b, kTypeErr, kValErr, true, 0)) {
                return nullptr;
            }
            boxes.push_back(b);
        }
        if (PyErr_Occurred()) return nullptr;
    }

    if (boxes.empty()) {
        return PyTuple_New(0);
    }

    SweepWorkspace ws;
    std::vector<Box64> cover;
    if (!rectangle_union_cover(boxes, poller, cover, ws)) {
        return nullptr;
    }
    if (!poller.call_now()) {
        return nullptr;
    }
    return boxes_to_pytuple(cover);
}

// ============================================================================
// 2. forbidden_box_cover_cpp
// ============================================================================

bool parse_atom_sequence(PyObject* seq_obj, PyObject* layer_map, uint32_t& next_layer_id,
                         std::vector<AtomRec>& out, CheckpointPoller& poller) {
    const char* kTypeErr = "rectangle union requires four exact integers";
    const char* kValErr = "rectangle union requires positive rectangles";

    auto parse_one = [&](PyObject* pair_obj, size_t idx) -> bool {
        if (!poller.poll_index(idx)) return false;
        PyObject* layer_obj = nullptr;
        PyObject* rect_obj = nullptr;
        PyRef owned_layer;
        PyRef owned_rect;

        if (PyTuple_CheckExact(pair_obj)) {
            if (PyTuple_GET_SIZE(pair_obj) != 2) {
                PyErr_SetString(PyExc_ValueError, "atom entry must be a (layer, box) pair");
                return false;
            }
            layer_obj = PyTuple_GET_ITEM(pair_obj, 0);
            rect_obj = PyTuple_GET_ITEM(pair_obj, 1);
        } else if (PyList_CheckExact(pair_obj)) {
            if (PyList_GET_SIZE(pair_obj) != 2) {
                PyErr_SetString(PyExc_ValueError, "atom entry must be a (layer, box) pair");
                return false;
            }
            layer_obj = PyList_GET_ITEM(pair_obj, 0);
            rect_obj = PyList_GET_ITEM(pair_obj, 1);
        } else {
            PyRef it = PyRef::adopt(PyObject_GetIter(pair_obj));
            if (!it) return false;
            owned_layer = PyRef::adopt(PyIter_Next(it.get()));
            if (owned_layer) {
                owned_rect = PyRef::adopt(PyIter_Next(it.get()));
            }
            PyRef extra = owned_rect ? PyRef::adopt(PyIter_Next(it.get())) : PyRef{};
            if (extra || !owned_layer || !owned_rect) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError, "atom entry must be a (layer, box) pair");
                }
                return false;
            }
            layer_obj = owned_layer.get();
            rect_obj = owned_rect.get();
        }

        uint32_t lid = 0;
        if (layer_obj != Py_None) {
            PyObject* existing = PyDict_GetItemWithError(layer_map, layer_obj);
            if (existing) {
                lid = static_cast<uint32_t>(PyLong_AsUnsignedLong(existing));
            } else if (PyErr_Occurred()) {
                return false;
            } else {
                lid = next_layer_id++;
                PyRef id_obj = PyRef::adopt(PyLong_FromUnsignedLong(lid));
                if (!id_obj) return false;
                if (PyDict_SetItem(layer_map, layer_obj, id_obj.get()) < 0) {
                    return false;
                }
            }
        }

        Box64 box;
        // Positivity is validated on derived Minkowski difference boxes, matching Python.
        if (!parse_box64_strict(rect_obj, box, kTypeErr, kValErr, false, kCoordLimit60)) {
            return false;
        }
        out.push_back(AtomRec{lid, box});
        return true;
    };

    if (PyTuple_CheckExact(seq_obj)) {
        const Py_ssize_t n = PyTuple_GET_SIZE(seq_obj);
        out.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!parse_one(PyTuple_GET_ITEM(seq_obj, i), static_cast<size_t>(i))) {
                return false;
            }
        }
    } else if (PyList_CheckExact(seq_obj)) {
        const Py_ssize_t n = PyList_GET_SIZE(seq_obj);
        out.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!parse_one(PyList_GET_ITEM(seq_obj, i), static_cast<size_t>(i))) {
                return false;
            }
        }
    } else {
        PyRef it = PyRef::adopt(PyObject_GetIter(seq_obj));
        if (!it) return false;
        size_t idx = 0;
        while (PyRef item = PyRef::adopt(PyIter_Next(it.get()))) {
            if (!parse_one(item.get(), idx++)) {
                return false;
            }
        }
        if (PyErr_Occurred()) return false;
    }
    return true;
}

PyObject* make_forbidden_result_tuple(const std::vector<Box64>& cover,
                                      uint64_t raw_count,
                                      uint64_t unique_count) {
    PyRef cover_tup = PyRef::adopt(boxes_to_pytuple(cover));
    PyRef py_raw = PyRef::adopt(PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(raw_count)));
    PyRef py_uniq = PyRef::adopt(PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(unique_count)));
    PyRef out = PyRef::adopt(PyTuple_New(3));
    if (!cover_tup || !py_raw || !py_uniq || !out) {
        return nullptr;
    }
    PyTuple_SET_ITEM(out.get(), 0, cover_tup.release());
    PyTuple_SET_ITEM(out.get(), 1, py_raw.release());
    PyTuple_SET_ITEM(out.get(), 2, py_uniq.release());
    return out.release();
}

static PyObject* py_forbidden_box_cover_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"left_atoms", "right_atoms", "required_nm", "checkpoint", nullptr};
    PyObject* left_obj = nullptr;
    PyObject* right_obj = nullptr;
    PyObject* req_obj = nullptr;
    PyObject* checkpoint_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|O:forbidden_box_cover_cpp",
                                     const_cast<char**>(kwlist),
                                     &left_obj, &right_obj, &req_obj, &checkpoint_obj)) {
        return nullptr;
    }

    int64_t required_nm = 0;
    const IntStatus req_st = extract_exact_i64(req_obj, required_nm);
    if (req_st == IntStatus::NotExactInt) {
        PyErr_SetString(PyExc_ValueError, "pair collision clearance must be one non-negative integer");
        return nullptr;
    }
    if (req_st == IntStatus::Overflow) {
        PyRef zero = PyRef::adopt(PyLong_FromLong(0));
        if (!zero) return nullptr;
        const int is_neg = PyObject_RichCompareBool(req_obj, zero.get(), Py_LT);
        if (is_neg < 0) return nullptr;
        if (is_neg) {
            PyErr_SetString(PyExc_ValueError, "pair collision clearance must be one non-negative integer");
            return nullptr;
        }
        PyErr_SetString(PyExc_OverflowError, "required_nm exceeds C++ int64 fast-path range");
        return nullptr;
    }
    if (required_nm < 0) {
        PyErr_SetString(PyExc_ValueError, "pair collision clearance must be one non-negative integer");
        return nullptr;
    }
    if (required_nm > kCoordLimit60) {
        PyErr_SetString(PyExc_OverflowError, "required_nm exceeds C++ int64 fast-path range");
        return nullptr;
    }

    PyCheckpointAdapter ckpt;
    if (!ckpt.init(checkpoint_obj)) {
        return nullptr;
    }
    CheckpointPoller poller = ckpt.poller();
    if (!poller.call_now()) {
        return nullptr;
    }

    PyRef layer_map = PyRef::adopt(PyDict_New());
    if (!layer_map) return nullptr;
    uint32_t next_layer_id = 1;

    std::vector<AtomRec> left_atoms;
    std::vector<AtomRec> right_atoms;
    // Parse right_atoms first so ObservedBox cancellation order matches Python's _ordered_pairs.
    const bool right_ok = parse_atom_sequence(right_obj, layer_map.get(), next_layer_id, right_atoms, poller);
    const bool left_ok = right_ok && parse_atom_sequence(left_obj, layer_map.get(), next_layer_id, left_atoms, poller);
    layer_map.reset();
    if (!left_ok) {
        return nullptr;
    }

    const ForbiddenCoverResult res = forbidden_box_cover(left_atoms, right_atoms, required_nm, poller);
    switch (res.status) {
        case ForbiddenStatus::Ok:
            return make_forbidden_result_tuple(res.cover, res.raw_count, res.unique_count);
        case ForbiddenStatus::Cancelled:
            return nullptr;
        case ForbiddenStatus::NonPositiveBox:
            PyErr_SetString(PyExc_ValueError, "rectangle union requires positive rectangles");
            return nullptr;
        case ForbiddenStatus::LostAllGeometry:
            PyErr_SetString(PyExc_ValueError, "pair collision forbidden union lost all geometry");
            return nullptr;
    }
    return nullptr;
}

// ============================================================================
// 3. NativeRectangleIndex (flat 2D BVH wrapper)
// ============================================================================

struct NativeRectangleIndexObject {
    PyObject_HEAD
    BvhIndex* impl;
};

static void NativeRectangleIndex_dealloc(NativeRectangleIndexObject* self) {
    delete self->impl;
    self->impl = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* NativeRectangleIndex_new(PyTypeObject* type, PyObject* /*args*/,
                                          PyObject* /*kwargs*/) {
    auto* self = reinterpret_cast<NativeRectangleIndexObject*>(type->tp_alloc(type, 0));
    if (self) {
        self->impl = nullptr;
    }
    return reinterpret_cast<PyObject*>(self);
}

static int NativeRectangleIndex_init(NativeRectangleIndexObject* self,
                                     PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"boxes", "checkpoint", nullptr};
    PyObject* boxes_obj = nullptr;
    PyObject* checkpoint_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:NativeRectangleIndex",
                                     const_cast<char**>(kwlist),
                                     &boxes_obj, &checkpoint_obj)) {
        return -1;
    }

    PyCheckpointAdapter ckpt;
    if (!ckpt.init(checkpoint_obj)) {
        return -1;
    }
    CheckpointPoller poller = ckpt.poller();
    if (!poller.call_now()) {
        return -1;
    }

    std::vector<Box64> parsed_boxes;
    const char* kTypeErr = "collision index requires four exact integers";
    const char* kValErr = "collision index requires positive rectangles";

    if (PyTuple_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyTuple_GET_SIZE(boxes_obj);
        parsed_boxes.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!poller.poll_index(static_cast<size_t>(i))) return -1;
            Box64 b;
            if (!parse_box64_strict(PyTuple_GET_ITEM(boxes_obj, i), b, kTypeErr, kValErr, true, kCoordLimit61)) {
                return -1;
            }
            parsed_boxes.push_back(b);
        }
    } else if (PyList_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyList_GET_SIZE(boxes_obj);
        parsed_boxes.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!poller.poll_index(static_cast<size_t>(i))) return -1;
            Box64 b;
            if (!parse_box64_strict(PyList_GET_ITEM(boxes_obj, i), b, kTypeErr, kValErr, true, kCoordLimit61)) {
                return -1;
            }
            parsed_boxes.push_back(b);
        }
    } else {
        PyRef it = PyRef::adopt(PyObject_GetIter(boxes_obj));
        if (!it) return -1;
        size_t idx = 0;
        while (PyRef row = PyRef::adopt(PyIter_Next(it.get()))) {
            if (!poller.poll_index(idx++)) return -1;
            Box64 b;
            if (!parse_box64_strict(row.get(), b, kTypeErr, kValErr, true, kCoordLimit61)) {
                return -1;
            }
            parsed_boxes.push_back(b);
        }
        if (PyErr_Occurred()) return -1;
    }

    auto impl = std::make_unique<BvhIndex>();
    if (!parsed_boxes.empty()) {
        if (!impl->build(std::move(parsed_boxes), poller)) {
            return -1;
        }
    }
    if (!poller.call_now()) {
        return -1;
    }

    delete self->impl;
    self->impl = impl.release();
    return 0;
}

static PyObject* NativeRectangleIndex_conflicts(NativeRectangleIndexObject* self,
                                                PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"boxes", "clearance_nm", "checkpoint", nullptr};
    PyObject* boxes_obj = nullptr;
    PyObject* gap_obj = nullptr;
    PyObject* checkpoint_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:conflicts",
                                     const_cast<char**>(kwlist),
                                     &boxes_obj, &gap_obj, &checkpoint_obj)) {
        return nullptr;
    }

    PyCheckpointAdapter ckpt;
    if (!ckpt.init(checkpoint_obj)) {
        return nullptr;
    }
    CheckpointPoller poller = ckpt.poller();
    if (!poller.call_now()) {
        return nullptr;
    }
    if (!self->impl || self->impl->empty()) {
        Py_RETURN_FALSE;
    }

    int64_t gap = 0;
    IntStatus gst = IntStatus::Ok;
    if (!extract_exact_i64_bounded(gap_obj, kCoordLimit61, gap, gst)) {
        if (gst == IntStatus::Overflow) {
            PyErr_SetString(PyExc_OverflowError, "clearance_nm exceeds C++ int64 fast-path range");
        } else {
            PyErr_SetString(PyExc_TypeError, "clearance_nm must be an exact integer");
        }
        return nullptr;
    }

    CheckpointPoller visit_poller = ckpt.poller();
    const char* kTypeErr = "collision query requires four exact integers";
    const char* kValErr = "collision query box invalid";
    bool conflict = false;

    if (PyTuple_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyTuple_GET_SIZE(boxes_obj);
        for (Py_ssize_t i = 0; i < n; ++i) {
            Box64 q;
            if (!parse_box64_strict(PyTuple_GET_ITEM(boxes_obj, i), q, kTypeErr, kValErr, false, kCoordLimit61)) {
                return nullptr;
            }
            const ConflictResult rc = self->impl->query_conflict_one(q, gap, visit_poller);
            if (rc == ConflictResult::Cancelled) return nullptr;
            if (rc == ConflictResult::Conflict) {
                conflict = true;
                break;
            }
        }
    } else if (PyList_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyList_GET_SIZE(boxes_obj);
        for (Py_ssize_t i = 0; i < n; ++i) {
            Box64 q;
            if (!parse_box64_strict(PyList_GET_ITEM(boxes_obj, i), q, kTypeErr, kValErr, false, kCoordLimit61)) {
                return nullptr;
            }
            const ConflictResult rc = self->impl->query_conflict_one(q, gap, visit_poller);
            if (rc == ConflictResult::Cancelled) return nullptr;
            if (rc == ConflictResult::Conflict) {
                conflict = true;
                break;
            }
        }
    } else {
        PyRef it = PyRef::adopt(PyObject_GetIter(boxes_obj));
        if (!it) return nullptr;
        while (PyRef row = PyRef::adopt(PyIter_Next(it.get()))) {
            Box64 q;
            if (!parse_box64_strict(row.get(), q, kTypeErr, kValErr, false, kCoordLimit61)) {
                return nullptr;
            }
            const ConflictResult rc = self->impl->query_conflict_one(q, gap, visit_poller);
            if (rc == ConflictResult::Cancelled) return nullptr;
            if (rc == ConflictResult::Conflict) {
                conflict = true;
                break;
            }
        }
        if (!conflict && PyErr_Occurred()) return nullptr;
    }

    if (!poller.call_now()) {
        return nullptr;
    }
    return PyBool_FromLong(conflict ? 1 : 0);
}

static PyObject* NativeRectangleIndex_intersecting(NativeRectangleIndexObject* self,
                                                   PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"box", "include_touching", nullptr};
    PyObject* box_obj = nullptr;
    int include_touching = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:intersecting",
                                     const_cast<char**>(kwlist),
                                     &box_obj, &include_touching)) {
        return nullptr;
    }
    if (!self->impl || self->impl->empty()) {
        return PyTuple_New(0);
    }
    Box64 q;
    if (!parse_box64_strict(box_obj, q, "collision query requires four exact integers",
                            "collision query box invalid", false, kCoordLimit61)) {
        return nullptr;
    }
    std::vector<Box64> hits;
    self->impl->query_intersecting(q, include_touching != 0, hits);
    return boxes_to_pytuple(hits);
}

static PyMethodDef NativeRectangleIndex_methods[] = {
    {"conflicts", reinterpret_cast<PyCFunction>(NativeRectangleIndex_conflicts),
     METH_VARARGS | METH_KEYWORDS,
     "Return True iff any query box has clearance < clearance_nm against an indexed box."},
    {"intersecting", reinterpret_cast<PyCFunction>(NativeRectangleIndex_intersecting),
     METH_VARARGS | METH_KEYWORDS,
     "Return tuple of indexed boxes intersecting (or touching if include_touching=True) box."},
    {nullptr, nullptr, 0, nullptr},
};

static PyTypeObject NativeRectangleIndexType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "clay._native._place_cpp.NativeRectangleIndex",
    sizeof(NativeRectangleIndexObject),
    0,
    reinterpret_cast<destructor>(NativeRectangleIndex_dealloc),
    0, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr,
    Py_TPFLAGS_DEFAULT,
    "Flat contiguous-array 2D BVH for exact integer rectangle clearance and intersection queries.",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    NativeRectangleIndex_methods,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
    reinterpret_cast<initproc>(NativeRectangleIndex_init),
    nullptr,
    NativeRectangleIndex_new,
};

// ============================================================================
// 4. exact_forbidden_union_cpp
// ============================================================================

static PyObject* py_exact_forbidden_union_cpp(PyObject* /*self*/, PyObject* args) {
    PyObject* boxes_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:exact_forbidden_union_cpp", &boxes_obj)) {
        return nullptr;
    }

    std::vector<Box64> boxes;
    const char* kTypeErr = "full-2D forbidden box must contain four exact integers";
    const char* kValErr = "full-2D forbidden box must be positive";

    if (PyTuple_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyTuple_GET_SIZE(boxes_obj);
        boxes.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            Box64 b;
            if (!parse_box64_strict(PyTuple_GET_ITEM(boxes_obj, i), b, kTypeErr, kValErr, true, 0)) {
                return nullptr;
            }
            boxes.push_back(b);
        }
    } else if (PyList_CheckExact(boxes_obj)) {
        const Py_ssize_t n = PyList_GET_SIZE(boxes_obj);
        boxes.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            Box64 b;
            if (!parse_box64_strict(PyList_GET_ITEM(boxes_obj, i), b, kTypeErr, kValErr, true, 0)) {
                return nullptr;
            }
            boxes.push_back(b);
        }
    } else {
        PyRef it = PyRef::adopt(PyObject_GetIter(boxes_obj));
        if (!it) return nullptr;
        while (PyRef row = PyRef::adopt(PyIter_Next(it.get()))) {
            Box64 b;
            if (!parse_box64_strict(row.get(), b, kTypeErr, kValErr, true, 0)) {
                return nullptr;
            }
            boxes.push_back(b);
        }
        if (PyErr_Occurred()) return nullptr;
    }

    std::vector<Box64> out;
    const ExactUnionStatus st = exact_forbidden_union(std::move(boxes), out);
    if (st == ExactUnionStatus::LostAllGeometry) {
        PyErr_SetString(PyExc_ValueError, "full-2D pair forbidden union lost all geometry");
        return nullptr;
    }
    return boxes_to_pytuple(out);
}

// ============================================================================
// 5. Sequence-Pair Constraint Graph Longest Path & Band/Block Layout Search
// ============================================================================

static PyObject* py_longest_path_positions_cpp(PyObject* /*self*/, PyObject* args) {
    PyObject* nodes_obj = nullptr;
    PyObject* edges_obj = nullptr;
    PyObject* weight_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO:longest_path_positions_cpp",
                          &nodes_obj, &edges_obj, &weight_obj)) {
        return nullptr;
    }
    if (!PyDict_Check(weight_obj)) {
        PyErr_SetString(PyExc_TypeError, "weight must be a dict");
        return nullptr;
    }

    PyRef nodes_seq = PyRef::adopt(PySequence_Fast(nodes_obj, "nodes must be a sequence"));
    if (!nodes_seq) return nullptr;
    const Py_ssize_t n_py = PySequence_Fast_GET_SIZE(nodes_seq.get());
    PyObject** node_items = PySequence_Fast_ITEMS(nodes_seq.get());

    const size_t n = static_cast<size_t>(n_py);
    // Flat open-addressing node table avoids PyDict_New + PyLong allocations.
    struct NodeSlot {
        PyObject* key = nullptr;
        Py_hash_t hash = 0;
        uint32_t id = 0;
    };
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    const size_t mask = cap - 1;
    std::vector<NodeSlot> table(cap);

    auto find_node_id = [&](PyObject* key, uint32_t& out_id) -> int {
        const Py_hash_t h = PyObject_Hash(key);
        if (h == -1 && PyErr_Occurred()) return -1;
        size_t slot = static_cast<size_t>(h) & mask;
        while (true) {
            const NodeSlot& s = table[slot];
            if (!s.key) return 0;
            if (s.key == key) {
                out_id = s.id;
                return 1;
            }
            if (s.hash == h) {
                const int eq = PyObject_RichCompareBool(s.key, key, Py_EQ);
                if (eq < 0) return -1;
                if (eq) {
                    out_id = s.id;
                    return 1;
                }
            }
            slot = (slot + 1) & mask;
        }
    };

    for (size_t i = 0; i < n; ++i) {
        PyObject* key = node_items[i];
        const Py_hash_t h = PyObject_Hash(key);
        if (h == -1 && PyErr_Occurred()) return nullptr;
        size_t slot = static_cast<size_t>(h) & mask;
        while (true) {
            NodeSlot& s = table[slot];
            if (!s.key) {
                s.key = key;
                s.hash = h;
                s.id = static_cast<uint32_t>(i);
                break;
            }
            bool dup = (s.key == key);
            if (!dup && s.hash == h) {
                const int eq = PyObject_RichCompareBool(s.key, key, Py_EQ);
                if (eq < 0) return nullptr;
                dup = (eq != 0);
            }
            if (dup) {
                PyErr_SetString(PyExc_OverflowError, "duplicate node in longest_path_positions");
                return nullptr;
            }
            slot = (slot + 1) & mask;
        }
    }

    PyRef edges_seq = PyRef::adopt(PySequence_Fast(edges_obj, "edges must be a sequence"));
    if (!edges_seq) return nullptr;
    const Py_ssize_t m_py = PySequence_Fast_GET_SIZE(edges_seq.get());
    PyObject** edge_items = PySequence_Fast_ITEMS(edges_seq.get());

    std::vector<DagEdge> dag_edges;
    dag_edges.reserve(static_cast<size_t>(m_py));

    for (Py_ssize_t i = 0; i < m_py; ++i) {
        PyObject* edge = edge_items[i];
        PyObject* u_obj = nullptr;
        PyObject* v_obj = nullptr;
        PyObject* lookup_key = edge;
        PyRef temp_key;

        if (PyTuple_CheckExact(edge) && PyTuple_GET_SIZE(edge) == 2) {
            u_obj = PyTuple_GET_ITEM(edge, 0);
            v_obj = PyTuple_GET_ITEM(edge, 1);
        } else {
            PyRef pair_fast = PyRef::adopt(PySequence_Fast(edge, "edge must be a 2-element pair"));
            if (!pair_fast) return nullptr;
            if (PySequence_Fast_GET_SIZE(pair_fast.get()) != 2) {
                PyErr_SetString(PyExc_ValueError, "edge must have length 2");
                return nullptr;
            }
            u_obj = PySequence_Fast_GET_ITEM(pair_fast.get(), 0);
            v_obj = PySequence_Fast_GET_ITEM(pair_fast.get(), 1);
            temp_key = PyRef::adopt(PyTuple_Pack(2, u_obj, v_obj));
            if (!temp_key) return nullptr;
            lookup_key = temp_key.get();
        }

        uint32_t u = 0;
        uint32_t v = 0;
        const int found_u = find_node_id(u_obj, u);
        if (found_u < 0) return nullptr;
        const int found_v = find_node_id(v_obj, v);
        if (found_v < 0) return nullptr;
        if (found_v == 0) {
            PyErr_SetObject(PyExc_KeyError, v_obj);
            return nullptr;
        }
        if (found_u == 0) {
            PyErr_SetString(PyExc_OverflowError, "edge source node not in nodes list");
            return nullptr;
        }

        PyObject* w_obj = PyDict_GetItemWithError(weight_obj, lookup_key);
        temp_key.reset();
        if (PyErr_Occurred()) return nullptr;

        double w_val = 0.0;
        bool has_w = false;
        if (w_obj) {
            w_val = PyFloat_AsDouble(w_obj);
            if (w_val == -1.0 && PyErr_Occurred()) return nullptr;
            has_w = true;
        }

        dag_edges.push_back(DagEdge{u, v, w_val, has_w, static_cast<uint32_t>(i)});
    }

    const LongestPathResult lp = longest_path_positions(n, dag_edges);
    if (lp.status == LongestPathStatus::MissingWeight) {
        PyErr_SetObject(PyExc_KeyError, edge_items[lp.missing_edge_index]);
        return nullptr;
    }
    edges_seq.reset();

    if (lp.status == LongestPathStatus::CycleDetected) {
        PyErr_Format(
            PyExc_ValueError,
            "constraint graph is not a DAG (a cycle in the relative order) -- resolved %zu/%zu nodes",
            lp.resolved_count, n);
        return nullptr;
    }

    PyRef out_dict = PyRef::adopt(PyDict_New());
    if (!out_dict) return nullptr;
    for (size_t i = 0; i < n; ++i) {
        PyRef val_obj = PyRef::adopt(PyFloat_FromDouble(lp.positions[i]));
        if (!val_obj) return nullptr;
        if (PyDict_SetItem(out_dict.get(), node_items[i], val_obj.get()) < 0) {
            return nullptr;
        }
    }
    return out_dict.release();
}

PyObject* build_rel_dict(const int* pos_plus, const int* pos_minus, int n) {
    PyRef rel = PyRef::adopt(PyDict_New());
    PyRef s_ilj = PyRef::adopt(PyUnicode_FromString("i_left_j"));
    PyRef s_jli = PyRef::adopt(PyUnicode_FromString("j_left_i"));
    PyRef s_ibj = PyRef::adopt(PyUnicode_FromString("i_below_j"));
    PyRef s_jbi = PyRef::adopt(PyUnicode_FromString("j_below_i"));
    if (!rel || !s_ilj || !s_jli || !s_ibj || !s_jbi) {
        return nullptr;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const bool i_bp = pos_plus[i] < pos_plus[j];
            const bool i_bm = pos_minus[i] < pos_minus[j];
            PyObject* val = nullptr;
            if (i_bp && i_bm) val = s_ilj.get();
            else if (!i_bp && !i_bm) val = s_jli.get();
            else if (i_bp && !i_bm) val = s_jbi.get();
            else val = s_ibj.get();

            PyRef key = PyRef::adopt(PyTuple_New(2));
            PyRef pi = PyRef::adopt(PyLong_FromLong(i));
            PyRef pj = PyRef::adopt(PyLong_FromLong(j));
            if (!key || !pi || !pj) return nullptr;
            PyTuple_SET_ITEM(key.get(), 0, pi.release());
            PyTuple_SET_ITEM(key.get(), 1, pj.release());
            if (PyDict_SetItem(rel.get(), key.get(), val) < 0) {
                return nullptr;
            }
        }
    }
    return rel.release();
}

bool parse_blocks_wh(PyObject* blocks_obj, std::vector<double>& bw, std::vector<double>& bh) {
    PyRef seq = PyRef::adopt(PySequence_Fast(blocks_obj, "blocks must be a sequence"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    bw.resize(static_cast<size_t>(n));
    bh.resize(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyRef pair = PyRef::adopt(PySequence_Fast(items[i], "block size must be (w, h)"));
        if (!pair) return false;
        if (PySequence_Fast_GET_SIZE(pair.get()) != 2) {
            PyErr_SetString(PyExc_ValueError, "block size must have length 2");
            return false;
        }
        bw[static_cast<size_t>(i)] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pair.get(), 0));
        bh[static_cast<size_t>(i)] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pair.get(), 1));
        if (PyErr_Occurred()) return false;
    }
    return true;
}

bool parse_perm(PyObject* obj, int n, std::vector<int>& out) {
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "permutation must be a sequence"));
    if (!seq) return false;
    if (PySequence_Fast_GET_SIZE(seq.get()) != n) {
        PyErr_SetString(PyExc_ValueError, "permutation length mismatch");
        return false;
    }
    out.resize(static_cast<size_t>(n));
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (int i = 0; i < n; ++i) {
        const long val = PyLong_AsLong(items[i]);
        if (val == -1 && PyErr_Occurred()) return false;
        if (val < 0 || val >= n || seen[static_cast<size_t>(val)]) {
            PyErr_SetString(PyExc_ValueError, "invalid permutation element");
            return false;
        }
        seen[static_cast<size_t>(val)] = 1;
        out[static_cast<size_t>(i)] = static_cast<int>(val);
    }
    return true;
}

static PyObject* py_pack_band_sp_cpp(PyObject* /*self*/, PyObject* args) {
    PyObject* gp_obj = nullptr;
    PyObject* gm_obj = nullptr;
    PyObject* blocks_obj = nullptr;
    double gap = 0.0;
    if (!PyArg_ParseTuple(args, "OOOd:pack_band_sp_cpp", &gp_obj, &gm_obj, &blocks_obj, &gap)) {
        return nullptr;
    }
    std::vector<double> bw;
    std::vector<double> bh;
    if (!parse_blocks_wh(blocks_obj, bw, bh)) {
        return nullptr;
    }
    const int n = static_cast<int>(bw.size());
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "pack_band_sp requires at least one block");
        return nullptr;
    }
    std::vector<int> gp;
    std::vector<int> gm;
    if (!parse_perm(gp_obj, n, gp) || !parse_perm(gm_obj, n, gm)) {
        return nullptr;
    }

    const PackBandSpResult packed = pack_band_sp(gp, gm, bw, bh, gap);
    PyRef py_w = PyRef::adopt(PyFloat_FromDouble(packed.width));
    PyRef py_h = PyRef::adopt(PyFloat_FromDouble(packed.height));
    PyRef rel = PyRef::adopt(build_rel_dict(packed.pos_plus.data(), packed.pos_minus.data(), n));
    PyRef out = PyRef::adopt(PyTuple_New(3));
    if (!py_w || !py_h || !rel || !out) {
        return nullptr;
    }
    PyTuple_SET_ITEM(out.get(), 0, py_w.release());
    PyTuple_SET_ITEM(out.get(), 1, py_h.release());
    PyTuple_SET_ITEM(out.get(), 2, rel.release());
    return out.release();
}

static PyObject* py_search_block_layout_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"block_sizes", "dev_gap", "aspect_target", nullptr};
    PyObject* blocks_obj = nullptr;
    double dev_gap = 0.0;
    PyObject* aspect_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|O:search_block_layout_cpp",
                                     const_cast<char**>(kwlist),
                                     &blocks_obj, &dev_gap, &aspect_obj)) {
        return nullptr;
    }
    (void)aspect_obj;  // Kept for signature parity with _search_block_layout.

    std::vector<double> bw;
    std::vector<double> bh;
    if (!parse_blocks_wh(blocks_obj, bw, bh)) {
        return nullptr;
    }
    if (bw.size() < 2) {
        return PyDict_New();
    }

    const SearchLayoutResult res = search_block_layout(bw, bh, dev_gap);
    return build_rel_dict(res.pos_plus.data(), res.pos_minus.data(), res.n);
}

static PyObject* py_search_band_dev_layout_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"row_half_ws", "row_heights", "dev_gap", "aspect_target", nullptr};
    PyObject* rhw_obj = nullptr;
    PyObject* rht_obj = nullptr;
    double dev_gap = 0.0;
    PyObject* aspect_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOd|O:search_band_dev_layout_cpp",
                                     const_cast<char**>(kwlist),
                                     &rhw_obj, &rht_obj, &dev_gap, &aspect_obj)) {
        return nullptr;
    }
    (void)aspect_obj;

    PyRef rhw_seq = PyRef::adopt(PySequence_Fast(rhw_obj, "row_half_ws must be a sequence"));
    if (!rhw_seq) return nullptr;
    PyRef rht_seq = PyRef::adopt(PySequence_Fast(rht_obj, "row_heights must be a sequence"));
    if (!rht_seq) return nullptr;

    const Py_ssize_t n_py = PySequence_Fast_GET_SIZE(rhw_seq.get());
    if (PySequence_Fast_GET_SIZE(rht_seq.get()) != n_py) {
        PyErr_SetString(PyExc_ValueError, "row_half_ws and row_heights length mismatch");
        return nullptr;
    }
    const int n = static_cast<int>(n_py);
    if (n < 2) {
        return PyDict_New();
    }

    std::vector<uint32_t> band_off(static_cast<size_t>(n) + 1, 0);
    std::vector<double> half_w;
    std::vector<double> half_h;
    for (int r = 0; r < n; ++r) {
        PyRef hw_row = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(rhw_seq.get(), r), "row must be sequence"));
        if (!hw_row) return nullptr;
        PyRef ht_row = PyRef::adopt(PySequence_Fast(PySequence_Fast_GET_ITEM(rht_seq.get(), r), "row must be sequence"));
        if (!ht_row) return nullptr;

        const Py_ssize_t len = PySequence_Fast_GET_SIZE(hw_row.get());
        if (PySequence_Fast_GET_SIZE(ht_row.get()) != len) {
            PyErr_SetString(PyExc_ValueError, "band row length mismatch");
            return nullptr;
        }
        for (Py_ssize_t k = 0; k < len; ++k) {
            const double w = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(hw_row.get(), k));
            const double h = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(ht_row.get(), k));
            if (PyErr_Occurred()) return nullptr;
            half_w.push_back(w);
            half_h.push_back(h * 0.5);
        }
        band_off[static_cast<size_t>(r) + 1] = static_cast<uint32_t>(half_w.size());
    }
    rhw_seq.reset();
    rht_seq.reset();

    if (half_w.empty()) {
        return PyDict_New();
    }

    const SearchLayoutResult res = search_band_dev_layout(band_off, half_w, half_h, dev_gap);
    if (res.n < 2) {
        return PyDict_New();
    }
    return build_rel_dict(res.pos_plus.data(), res.pos_minus.data(), res.n);
}

// ============================================================================
// 6. Rigid Placed Geometry Coordinate Quantization (placed_geometry.py)
// ============================================================================

inline bool extract_double(PyObject* obj, double& out) noexcept {
    if (PyFloat_CheckExact(obj)) {
        out = PyFloat_AS_DOUBLE(obj);
        return true;
    }
    out = PyFloat_AsDouble(obj);
    return !(out == -1.0 && PyErr_Occurred());
}

using clay_pnr::core::placed_coord_nm_try_fast;

static PyObject* placed_coord_nm_slow_py(double local_um, double placement_um) {
    const double composed_nm = (local_um + placement_um) * 1000.0;
    PyObject* builtins = PyEval_GetBuiltins();
    if (!builtins) return nullptr;
    PyObject* round_fn = PyDict_GetItemString(builtins, "round");
    if (!round_fn) {
        PyErr_SetString(PyExc_RuntimeError, "builtins.round unavailable");
        return nullptr;
    }
    PyRef py_c = PyRef::adopt(PyFloat_FromDouble(composed_nm));
    PyRef py_nine = PyRef::adopt(PyLong_FromLong(9));
    if (!py_c || !py_nine) return nullptr;
    PyRef stable = PyRef::adopt(PyObject_CallFunctionObjArgs(round_fn, py_c.get(), py_nine.get(), nullptr));
    if (!stable) return nullptr;
    const double s = PyFloat_AsDouble(stable.get());
    if (s == -1.0 && PyErr_Occurred()) return nullptr;
    if (!std::isfinite(s)) {
        if (std::isnan(s)) {
            PyErr_SetString(PyExc_ValueError, "cannot convert float NaN to integer");
        } else {
            PyErr_SetString(PyExc_OverflowError, "cannot convert float infinity to integer");
        }
        return nullptr;
    }
    const double fl = std::floor(s + 0.5);
    if (fl >= -9e18 && fl <= 9e18) {
        return PyLong_FromLongLong(static_cast<long long>(fl));
    }
    return PyLong_FromDouble(fl);
}

inline PyObject* placed_coord_nm_to_py(double local_um, double placement_um) {
    int64_t fast_val = 0;
    if (placed_coord_nm_try_fast(local_um, placement_um, fast_val)) {
        return PyLong_FromLongLong(static_cast<long long>(fast_val));
    }
    return placed_coord_nm_slow_py(local_um, placement_um);
}

inline bool extract_xy_pair(PyObject* obj, double& x_out, double& y_out) {
    if (PyTuple_CheckExact(obj)) {
        if (PyTuple_GET_SIZE(obj) < 2) {
            PyErr_SetString(PyExc_IndexError, "point/placement pair requires at least 2 items");
            return false;
        }
        return extract_double(PyTuple_GET_ITEM(obj, 0), x_out) &&
               extract_double(PyTuple_GET_ITEM(obj, 1), y_out);
    }
    if (PyList_CheckExact(obj)) {
        if (PyList_GET_SIZE(obj) < 2) {
            PyErr_SetString(PyExc_IndexError, "point/placement pair requires at least 2 items");
            return false;
        }
        return extract_double(PyList_GET_ITEM(obj, 0), x_out) &&
               extract_double(PyList_GET_ITEM(obj, 1), y_out);
    }
    PyRef item0 = PyRef::adopt(PySequence_GetItem(obj, 0));
    if (!item0) return false;
    PyRef item1 = PyRef::adopt(PySequence_GetItem(obj, 1));
    if (!item1) return false;
    return extract_double(item0.get(), x_out) && extract_double(item1.get(), y_out);
}

static PyObject* py_placed_coord_nm_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "placed_coord_nm_cpp expects exactly 2 arguments");
        return nullptr;
    }
    double a = 0.0;
    double b = 0.0;
    if (!extract_double(args[0], a) || !extract_double(args[1], b)) {
        return nullptr;
    }
    return placed_coord_nm_to_py(a, b);
}

static PyObject* py_placed_point_nm_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "placed_point_nm_cpp expects exactly 3 arguments");
        return nullptr;
    }
    double x = 0.0;
    double y = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    if (!extract_double(args[0], x) || !extract_double(args[1], y) ||
        !extract_xy_pair(args[2], cx, cy)) {
        return nullptr;
    }
    PyRef px = PyRef::adopt(placed_coord_nm_to_py(x, cx));
    PyRef py = px ? PyRef::adopt(placed_coord_nm_to_py(y, cy)) : PyRef{};
    PyRef tup = py ? PyRef::adopt(PyTuple_New(2)) : PyRef{};
    if (!tup) return nullptr;
    PyTuple_SET_ITEM(tup.get(), 0, px.release());
    PyTuple_SET_ITEM(tup.get(), 1, py.release());
    return tup.release();
}

static PyObject* py_placed_ring_nm_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "placed_ring_nm_cpp expects exactly 2 arguments");
        return nullptr;
    }
    PyObject* ring_obj = args[0];
    double cx = 0.0;
    double cy = 0.0;
    if (!extract_xy_pair(args[1], cx, cy)) {
        return nullptr;
    }

    auto build_pair = [cx, cy](PyObject* pt_obj) -> PyObject* {
        double x = 0.0;
        double y = 0.0;
        if (!extract_xy_pair(pt_obj, x, y)) {
            return nullptr;
        }
        PyRef px = PyRef::adopt(placed_coord_nm_to_py(x, cx));
        PyRef py = px ? PyRef::adopt(placed_coord_nm_to_py(y, cy)) : PyRef{};
        PyRef tup = py ? PyRef::adopt(PyTuple_New(2)) : PyRef{};
        if (!tup) return nullptr;
        PyTuple_SET_ITEM(tup.get(), 0, px.release());
        PyTuple_SET_ITEM(tup.get(), 1, py.release());
        return tup.release();
    };

    if (PyList_CheckExact(ring_obj)) {
        const Py_ssize_t n = PyList_GET_SIZE(ring_obj);
        PyRef out = PyRef::adopt(PyList_New(n));
        if (!out) return nullptr;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyRef pair = PyRef::adopt(build_pair(PyList_GET_ITEM(ring_obj, i)));
            if (!pair) return nullptr;
            PyList_SET_ITEM(out.get(), i, pair.release());
        }
        return out.release();
    }
    if (PyTuple_CheckExact(ring_obj)) {
        const Py_ssize_t n = PyTuple_GET_SIZE(ring_obj);
        PyRef out = PyRef::adopt(PyList_New(n));
        if (!out) return nullptr;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyRef pair = PyRef::adopt(build_pair(PyTuple_GET_ITEM(ring_obj, i)));
            if (!pair) return nullptr;
            PyList_SET_ITEM(out.get(), i, pair.release());
        }
        return out.release();
    }
    PyRef it = PyRef::adopt(PyObject_GetIter(ring_obj));
    if (!it) return nullptr;
    PyRef out = PyRef::adopt(PyList_New(0));
    if (!out) return nullptr;
    while (PyRef item = PyRef::adopt(PyIter_Next(it.get()))) {
        PyRef pair = PyRef::adopt(build_pair(item.get()));
        if (!pair) return nullptr;
        if (PyList_Append(out.get(), pair.get()) < 0) {
            return nullptr;
        }
    }
    if (PyErr_Occurred()) return nullptr;
    return out.release();
}

static PyMethodDef PlaceCppMethods[] = {
    {"rectangle_union_cover_cpp", reinterpret_cast<PyCFunction>(py_rectangle_union_cover_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "Exact int64 segment-tree sweep-line rectangle union."},
    {"forbidden_box_cover_cpp", reinterpret_cast<PyCFunction>(py_forbidden_box_cover_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "Exact layered Minkowski-difference streaming forbidden box cover."},
    {"exact_forbidden_union_cpp", py_exact_forbidden_union_cpp,
     METH_VARARGS,
     "Exact segment-tree rectangle union for full2d_pair_cover._exact_forbidden_union."},
    {"longest_path_positions_cpp", py_longest_path_positions_cpp,
     METH_VARARGS,
     "Kahn topological-sort longest path on a weighted constraint DAG."},
    {"pack_band_sp_cpp", py_pack_band_sp_cpp,
     METH_VARARGS,
     "Pack a band/block-level sequence-pair (gamma_plus, gamma_minus) to (W, H, rel)."},
    {"search_block_layout_cpp", reinterpret_cast<PyCFunction>(py_search_block_layout_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "Deterministic exhaustive/hill-climb sequence-pair block layout search."},
    {"search_band_dev_layout_cpp", reinterpret_cast<PyCFunction>(py_search_band_dev_layout_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "Deterministic exhaustive/hill-climb device-level band sequence-pair search."},
    {"placed_coord_nm_cpp", reinterpret_cast<PyCFunction>(py_placed_coord_nm_cpp),
     METH_FASTCALL,
     "Exact IEEE-754 parity rigid coordinate quantization to integer nm."},
    {"placed_point_nm_cpp", reinterpret_cast<PyCFunction>(py_placed_point_nm_cpp),
     METH_FASTCALL,
     "Exact IEEE-754 parity rigid 2D point quantization to integer nm."},
    {"placed_ring_nm_cpp", reinterpret_cast<PyCFunction>(py_placed_ring_nm_cpp),
     METH_FASTCALL,
     "Exact IEEE-754 parity batch polygon ring quantization to integer nm pairs."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef PlaceCppModule = {
    PyModuleDef_HEAD_INIT,
    "_place_cpp",
    "C++20 native placement and spatial geometry kernels for Clay.",
    -1,
    PlaceCppMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__place_cpp(void) {
    if (PyType_Ready(&NativeRectangleIndexType) < 0) {
        return nullptr;
    }
    PyRef m = PyRef::adopt(PyModule_Create(&PlaceCppModule));
    if (!m) {
        return nullptr;
    }
    Py_INCREF(&NativeRectangleIndexType);
    if (PyModule_AddObject(m.get(), "NativeRectangleIndex",
                           reinterpret_cast<PyObject*>(&NativeRectangleIndexType)) < 0) {
        Py_DECREF(&NativeRectangleIndexType);
        return nullptr;
    }
    return m.release();
}
