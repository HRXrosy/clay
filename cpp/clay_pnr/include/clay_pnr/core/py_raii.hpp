#pragma once

// C++20 CPython RAII & parsing utilities for Clay PnR binding translation units.
// ONLY include this header from Python binding TUs (py_*_module.cpp / *_py.cpp);
// pure C++20 core headers/sources must remain completely free of <Python.h>.

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
#include "clay_pnr/core/status.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace clay_pnr::py {

using core::Box64;
using core::CheckpointHook;
using core::CheckpointPoller;
using core::IntStatus;

// Move-only owning reference to a PyObject*, preventing Py_DECREF leaks on early returns.
class PyRef {
public:
    constexpr PyRef() noexcept = default;
    constexpr explicit PyRef(PyObject* owned_ptr) noexcept : ptr_(owned_ptr) {}

    ~PyRef() noexcept {
        Py_XDECREF(ptr_);
    }

    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;

    PyRef(PyRef&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    PyRef& operator=(PyRef&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    static PyRef adopt(PyObject* owned_ptr) noexcept {
        return PyRef(owned_ptr);
    }

    static PyRef borrow(PyObject* borrowed_ptr) noexcept {
        Py_XINCREF(borrowed_ptr);
        return PyRef(borrowed_ptr);
    }

    PyObject* get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    PyObject* release() noexcept {
        PyObject* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

    void reset(PyObject* owned_ptr = nullptr) noexcept {
        if (ptr_ != owned_ptr) {
            Py_XDECREF(ptr_);
            ptr_ = owned_ptr;
        }
    }

private:
    PyObject* ptr_ = nullptr;
};

// Bridges a Python callable (or None/falsy object) into a pure C++20 CheckpointHook.
// Lifetime: must outlive any CheckpointHook / CheckpointPoller referencing it.
struct PyCheckpointAdapter {
    PyObject* cb = nullptr;  // borrowed reference

    bool init(PyObject* callback) noexcept {
        cb = nullptr;
        if (!callback || callback == Py_None) {
            return true;
        }
        const int truthy = PyObject_IsTrue(callback);
        if (truthy < 0) {
            return false;
        }
        if (truthy) {
            cb = callback;
        }
        return true;
    }

    static bool trampoline(void* ctx) {
        auto* self = static_cast<PyCheckpointAdapter*>(ctx);
        if (!self || !self->cb) return true;
        PyRef res = PyRef::adopt(PyObject_CallNoArgs(self->cb));
        return static_cast<bool>(res);
    }

    CheckpointHook hook() noexcept {
        if (!cb) return CheckpointHook{nullptr, nullptr};
        return CheckpointHook{&PyCheckpointAdapter::trampoline, this};
    }

    CheckpointPoller poller(uint64_t poll_mask = 255ULL) noexcept {
        return CheckpointPoller{hook(), poll_mask};
    }

    bool call_now() const {
        if (!cb) return true;
        PyRef res = PyRef::adopt(PyObject_CallNoArgs(cb));
        return static_cast<bool>(res);
    }
};

// Exact Python `int` extraction (rejects bool and non-int subclasses).
inline IntStatus extract_exact_i64(PyObject* obj, int64_t& out) noexcept {
    if (!PyLong_CheckExact(obj)) {
        return IntStatus::NotExactInt;
    }
    int overflow = 0;
    const long long val = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0) {
        return IntStatus::Overflow;
    }
    if (val == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return IntStatus::NotExactInt;
    }
    out = static_cast<int64_t>(val);
    return IntStatus::Ok;
}

inline bool extract_exact_i64_bounded(PyObject* obj, int64_t limit, int64_t& out,
                                      IntStatus& status) noexcept {
    status = extract_exact_i64(obj, out);
    if (status != IntStatus::Ok) {
        return false;
    }
    if (out < -limit || out > limit) {
        status = IntStatus::Overflow;
        return false;
    }
    return true;
}

// Strict 4-int Box64 parser preserving exact placement_kernels.cpp semantics:
// fast-paths exact tuple/list, supports arbitrary 4-element iterables/subclasses,
// rejects non-exact ints with TypeError, out-of-range with OverflowError,
// and optionally requires x0 < x1 and y0 < y1 (ValueError).
inline bool parse_box64_strict(PyObject* row, Box64& out, const char* type_err_msg,
                               const char* val_err_msg, bool require_positive,
                               int64_t limit = 0) {
    PyObject* items[4] = {nullptr, nullptr, nullptr, nullptr};
    PyRef owned_items[4];

    if (PyTuple_CheckExact(row)) {
        if (PyTuple_GET_SIZE(row) != 4) {
            PyErr_SetString(PyExc_TypeError, type_err_msg);
            return false;
        }
        for (int i = 0; i < 4; ++i) items[i] = PyTuple_GET_ITEM(row, i);
    } else if (PyList_CheckExact(row)) {
        if (PyList_GET_SIZE(row) != 4) {
            PyErr_SetString(PyExc_TypeError, type_err_msg);
            return false;
        }
        for (int i = 0; i < 4; ++i) items[i] = PyList_GET_ITEM(row, i);
    } else {
        const Py_ssize_t len = PyObject_Length(row);
        if (len < 0) return false;
        if (len != 4) {
            PyErr_SetString(PyExc_TypeError, type_err_msg);
            return false;
        }
        PyRef it = PyRef::adopt(PyObject_GetIter(row));
        if (!it) return false;
        for (int i = 0; i < 4; ++i) {
            PyRef elem = PyRef::adopt(PyIter_Next(it.get()));
            if (!elem) {
                if (PyErr_Occurred()) return false;
                PyErr_SetString(PyExc_TypeError, type_err_msg);
                return false;
            }
            items[i] = elem.get();
            owned_items[i] = std::move(elem);
        }
        PyRef extra = PyRef::adopt(PyIter_Next(it.get()));
        if (extra) {
            PyErr_SetString(PyExc_TypeError, type_err_msg);
            return false;
        }
        if (PyErr_Occurred()) return false;
    }

    int64_t vals[4] = {0, 0, 0, 0};
    bool has_overflow = false;
    bool has_type_err = false;
    for (int i = 0; i < 4; ++i) {
        const IntStatus st = extract_exact_i64(items[i], vals[i]);
        if (st == IntStatus::NotExactInt) {
            has_type_err = true;
            break;
        }
        if (st == IntStatus::Overflow || (limit > 0 && (vals[i] < -limit || vals[i] > limit))) {
            has_overflow = true;
        }
    }

    if (has_type_err) {
        PyErr_SetString(PyExc_TypeError, type_err_msg);
        return false;
    }
    if (has_overflow) {
        PyErr_SetString(PyExc_OverflowError, "integer coordinate exceeds C++ int64 fast-path range");
        return false;
    }
    if (require_positive && (vals[0] >= vals[2] || vals[1] >= vals[3])) {
        PyErr_SetString(PyExc_ValueError, val_err_msg);
        return false;
    }
    out = Box64{vals[0], vals[1], vals[2], vals[3]};
    return true;
}

// Gridless-style integer & normalized-box parsers.
inline bool parse_int64(PyObject* obj, int64_t& out) noexcept {
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0) {
        PyErr_SetString(PyExc_OverflowError, "coordinate exceeds int64");
        return false;
    }
    if (v == -1 && PyErr_Occurred()) return false;
    out = static_cast<int64_t>(v);
    return true;
}

inline bool parse_rect_norm(PyObject* obj, Box64& out) noexcept {
    if (PyTuple_CheckExact(obj) && PyTuple_GET_SIZE(obj) == 4) {
        int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        if (!parse_int64(PyTuple_GET_ITEM(obj, 0), x0) ||
            !parse_int64(PyTuple_GET_ITEM(obj, 1), y0) ||
            !parse_int64(PyTuple_GET_ITEM(obj, 2), x1) ||
            !parse_int64(PyTuple_GET_ITEM(obj, 3), y1)) {
            return false;
        }
        out = core::normalize_box(x0, y0, x1, y1);
        return true;
    }
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "rect must be a 4-sequence of ints"));
    if (!seq) return false;
    if (PySequence_Fast_GET_SIZE(seq.get()) != 4) {
        PyErr_SetString(PyExc_ValueError, "rect must have length 4");
        return false;
    }
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!parse_int64(items[0], x0) || !parse_int64(items[1], y0) ||
        !parse_int64(items[2], x1) || !parse_int64(items[3], y1)) {
        return false;
    }
    out = core::normalize_box(x0, y0, x1, y1);
    return true;
}

inline PyObject* make_box64_tuple(const Box64& b) {
    PyRef t = PyRef::adopt(PyTuple_New(4));
    if (!t) return nullptr;
    PyRef v0 = PyRef::adopt(PyLong_FromLongLong(b.x0));
    PyRef v1 = PyRef::adopt(PyLong_FromLongLong(b.y0));
    PyRef v2 = PyRef::adopt(PyLong_FromLongLong(b.x1));
    PyRef v3 = PyRef::adopt(PyLong_FromLongLong(b.y1));
    if (!v0 || !v1 || !v2 || !v3) return nullptr;
    PyTuple_SET_ITEM(t.get(), 0, v0.release());
    PyTuple_SET_ITEM(t.get(), 1, v1.release());
    PyTuple_SET_ITEM(t.get(), 2, v2.release());
    PyTuple_SET_ITEM(t.get(), 3, v3.release());
    return t.release();
}

inline PyObject* boxes_to_pytuple(const std::vector<Box64>& boxes) {
    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(boxes.size())));
    if (!out) return nullptr;
    for (size_t i = 0; i < boxes.size(); ++i) {
        PyRef item = PyRef::adopt(make_box64_tuple(boxes[i]));
        if (!item) return nullptr;
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), item.release());
    }
    return out.release();
}

inline bool parse_rect_list(PyObject* obj, std::vector<Box64>& out) {
    if (!obj || obj == Py_None) return true;
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "expected sequence of rects"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.reserve(out.size() + static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        Box64 b;
        if (!parse_rect_norm(items[i], b)) return false;
        out.push_back(b);
    }
    return true;
}

inline bool parse_int64_list(PyObject* obj, std::vector<int64_t>& out) {
    if (!obj || obj == Py_None) return true;
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "expected sequence of ints"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.reserve(out.size() + static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        int64_t v = 0;
        if (!parse_int64(items[i], v)) return false;
        out.push_back(v);
    }
    return true;
}

inline bool parse_xy_list(PyObject* obj, std::vector<std::pair<int64_t, int64_t>>& out) {
    if (!obj || obj == Py_None) return true;
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "expected sequence of (x, y) pairs"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.reserve(out.size() + static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* pair = items[i];
        int64_t x = 0, y = 0;
        if (PyTuple_CheckExact(pair) && PyTuple_GET_SIZE(pair) == 2) {
            if (!parse_int64(PyTuple_GET_ITEM(pair, 0), x) ||
                !parse_int64(PyTuple_GET_ITEM(pair, 1), y)) {
                return false;
            }
        } else {
            PyRef pseq = PyRef::adopt(PySequence_Fast(pair, "expected 2-sequence (x, y)"));
            if (!pseq) return false;
            if (PySequence_Fast_GET_SIZE(pseq.get()) != 2 ||
                !parse_int64(PySequence_Fast_GET_ITEM(pseq.get(), 0), x) ||
                !parse_int64(PySequence_Fast_GET_ITEM(pseq.get(), 1), y)) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError, "expected (x, y) pair");
                }
                return false;
            }
        }
        out.emplace_back(x, y);
    }
    return true;
}

}  // namespace clay_pnr::py

