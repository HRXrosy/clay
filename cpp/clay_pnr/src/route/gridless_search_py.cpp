// CPython binding TU for Gridless search & projection kernels:
//   - NativeAxisPriceCache (run & run_ctx)
//   - NativeViaTargetProjector (project & native region frontier heap)
//   - NativeEdgeContext.run_hard_connectivity_preflight
//   - NativeEdgeContext.run_cartesian_astar
//   - NativeEdgeContext.advance_region_step_ctx
//   - Region-event geometry projection functions

#include "clay_pnr/route/gridless_py_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clay_pnr::route::py_bind {

// ---------------------------------------------------------------------------
// 1. AxisPriceCacheCore & PyNativeAxisPriceCache
// ---------------------------------------------------------------------------

AxisPriceCacheCore::AxisPriceCacheCore() {
    str_price_segment = PyUnicode_InternFromString("price_segment");
    str_price_range = PyUnicode_InternFromString("price_range");
}

AxisPriceCacheCore::~AxisPriceCacheCore() {
    Py_XDECREF(str_price_segment);
    Py_XDECREF(str_price_range);
}

int AxisPriceCacheCore::call_charge(PyObject* charge_fn, PyObject* kind_str) const {
    PyObject* argv[1] = {kind_str};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(charge_fn, argv, 1, nullptr));
    if (!res) return -1;
    return PyObject_IsTrue(res.get());
}

bool AxisPriceCacheCore::call_price(PyObject* price_fn, int64_t gi, int64_t ax, int64_t ay,
                                    int64_t bx, int64_t by, int64_t& out_val) const {
    PyRef a0 = PyRef::adopt(PyLong_FromLongLong(gi));
    PyRef a1 = PyRef::adopt(PyLong_FromLongLong(ax));
    PyRef a2 = PyRef::adopt(PyLong_FromLongLong(ay));
    PyRef a3 = PyRef::adopt(PyLong_FromLongLong(bx));
    PyRef a4 = PyRef::adopt(PyLong_FromLongLong(by));
    if (!a0 || !a1 || !a2 || !a3 || !a4) return false;
    PyObject* argv[5] = {a0.get(), a1.get(), a2.get(), a3.get(), a4.get()};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(price_fn, argv, 5, nullptr));
    if (!res) return false;
    return parse_int64(res.get(), out_val);
}

int AxisPriceCacheCore::eval_partial(PyObject* price_fn, PyObject* charge_fn,
                                     int64_t gi, bool horizontal, int64_t fixed,
                                     int64_t a, int64_t b, int64_t& out) const {
    const int ch = call_charge(charge_fn, str_price_segment);
    if (ch <= 0) return ch;
    const bool ok = horizontal
        ? call_price(price_fn, gi, a, fixed, b, fixed, out)
        : call_price(price_fn, gi, fixed, a, fixed, b, out);
    return ok ? 1 : -1;
}

int AxisPriceCacheCore::eval_range(PyObject* price_fn, PyObject* charge_fn,
                                   int64_t gi, bool horizontal, int64_t fixed, int8_t sign,
                                   int32_t begin, int32_t end, int64_t& out) {
    if (begin >= end) {
        out = 0;
        return 1;
    }
    const AxisLineKey lk{gi, fixed, static_cast<int8_t>(horizontal ? 1 : 0), sign};
    const AxisRangeKey rk{lk, begin, end};
    auto rit = range_totals.find(rk);
    if (rit != range_totals.end()) {
        const int ch = call_charge(charge_fn, str_price_range);
        if (ch <= 0) return ch;
        out = rit->second;
        return 1;
    }
    const auto& axes = horizontal ? xs : ys;
    auto& line = lines[lk];
    if (line.valid.empty() && axes.size() > 1) {
        line.vals.assign(axes.size() - 1, 0);
        line.valid.assign(axes.size() - 1, 0);
    }
    int64_t total = 0;
    for (int32_t i = begin; i < end; ++i) {
        const size_t ui = static_cast<size_t>(i);
        if (!line.valid[ui]) {
            const int ch = call_charge(charge_fn, str_price_segment);
            if (ch <= 0) return ch;
            int64_t a = axes[ui];
            int64_t b = axes[ui + 1];
            if (sign < 0) std::swap(a, b);
            int64_t val = 0;
            const bool ok = horizontal
                ? call_price(price_fn, gi, a, fixed, b, fixed, val)
                : call_price(price_fn, gi, fixed, a, fixed, b, val);
            if (!ok) return -1;
            line.vals[ui] = val;
            line.valid[ui] = 1;
        }
        total += line.vals[ui];
    }
    range_totals[rk] = total;
    const int ch = call_charge(charge_fn, str_price_range);
    if (ch <= 0) return ch;
    out = total;
    return 1;
}

int AxisPriceCacheCore::run(PyObject* price_fn, PyObject* charge_fn,
                            int64_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
                            int64_t& out) {
    if (ax == bx && ay == by) {
        out = 0;
        return 1;
    }
    if (ax != bx && ay != by) {
        PyErr_SetString(PyExc_ValueError, "cost run must be orthogonal");
        return -1;
    }
    const bool horizontal = (ay == by);
    const int64_t first_coord = horizontal ? ax : ay;
    const int64_t last_coord = horizontal ? bx : by;
    const int64_t fixed = horizontal ? ay : ax;
    const int8_t sign = (last_coord > first_coord) ? 1 : -1;
    const auto& axes = horizontal ? xs : ys;
    const int64_t lo = (first_coord < last_coord) ? first_coord : last_coord;
    const int64_t hi = (first_coord < last_coord) ? last_coord : first_coord;

    const auto it_first = std::upper_bound(axes.begin(), axes.end(), lo);
    const auto it_stop = std::lower_bound(axes.begin(), axes.end(), hi);
    const int32_t first = static_cast<int32_t>(it_first - axes.begin());
    const int32_t stop = static_cast<int32_t>(it_stop - axes.begin());

    if (first == stop) {
        return eval_partial(price_fn, charge_fn, gi, horizontal, fixed,
                            first_coord, last_coord, out);
    }
    const int64_t leading = (sign > 0) ? axes[static_cast<size_t>(first)]
                                       : axes[static_cast<size_t>(stop - 1)];
    const int64_t trailing = (sign > 0) ? axes[static_cast<size_t>(stop - 1)]
                                        : axes[static_cast<size_t>(first)];
    int64_t a_val = 0;
    int st = eval_partial(price_fn, charge_fn, gi, horizontal, fixed,
                          first_coord, leading, a_val);
    if (st <= 0) return st;

    int64_t mid_val = 0;
    st = eval_range(price_fn, charge_fn, gi, horizontal, fixed, sign,
                    first, stop - 1, mid_val);
    if (st <= 0) return st;

    int64_t b_val = 0;
    st = eval_partial(price_fn, charge_fn, gi, horizontal, fixed,
                      trailing, last_coord, b_val);
    if (st <= 0) return st;

    out = a_val + mid_val + b_val;
    return 1;
}

namespace {

struct PyAxisPriceHookCtx {
    AxisPriceCacheCore* cache = nullptr;
    PyObject* watchdog_cb = nullptr;
    PyObject* fallback_price_fn = nullptr;
};

bool py_axis_watchdog_trampoline(void* raw, int32_t kind_id) {
    const auto* h = static_cast<const PyAxisPriceHookCtx*>(raw);
    if (!h->watchdog_cb || h->watchdog_cb == Py_None) return true;
    PyRef a0 = PyRef::adopt(PyLong_FromLong(kind_id));
    if (!a0) return false;
    PyObject* arg = a0.get();
    PyRef r = PyRef::adopt(PyObject_Vectorcall(h->watchdog_cb, &arg, 1, nullptr));
    return static_cast<bool>(r);
}

bool py_axis_fallback_price_trampoline(void* raw, int64_t gi, int64_t ax, int64_t ay,
                                       int64_t bx, int64_t by, int64_t& out_val) {
    const auto* h = static_cast<const PyAxisPriceHookCtx*>(raw);
    return h->cache->call_price(h->fallback_price_fn, gi, ax, ay, bx, by, out_val);
}

int run_axis_price_ctx_core(
    AxisPriceCacheCore& cache,
    const EdgeContextCore& ctx,
    int64_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
    int64_t remaining_budget,
    int64_t& steps_charged,
    int64_t& static_price_steps,
    int64_t& price_cache_hits,
    size_t max_run_cache,
    PyObject* watchdog_cb,
    PyObject* fallback_price_fn,
    int64_t& out_cost,
    bool& budget_stopped) {
    PyAxisPriceHookCtx hctx{&cache, watchdog_cb, fallback_price_fn};
    AxisPriceHooks hooks{&hctx, &py_axis_watchdog_trampoline, &py_axis_fallback_price_trampoline};
    const int st = run_axis_price_ctx_pure(
        cache, ctx, gi, ax, ay, bx, by,
        remaining_budget, steps_charged, static_price_steps, price_cache_hits,
        max_run_cache, out_cost, budget_stopped, hooks);
    if (st == -2) {
        PyErr_SetString(PyExc_ValueError, "cost run must be orthogonal");
        return -1;
    }
    return st;
}

void NativeAxisPriceCache_dealloc(PyNativeAxisPriceCache* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeAxisPriceCache_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeAxisPriceCache*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->core = new AxisPriceCacheCore();
    return reinterpret_cast<PyObject*>(self);
}

bool parse_strict_int_axis(PyObject* obj, std::vector<int64_t>& out) {
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "pricing axes must be strictly ordered integer coordinates"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.clear();
    out.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!PyLong_CheckExact(items[i])) {
            PyErr_SetString(PyExc_ValueError, "pricing axes must be strictly ordered integer coordinates");
            return false;
        }
        int64_t v = 0;
        if (!parse_int64(items[i], v)) return false;
        if (i > 0 && out.back() >= v) {
            PyErr_SetString(PyExc_ValueError, "pricing axes must be strictly ordered integer coordinates");
            return false;
        }
        out.push_back(v);
    }
    return true;
}

int NativeAxisPriceCache_init(PyNativeAxisPriceCache* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"xs", "ys", nullptr};
    PyObject *xs_obj = nullptr, *ys_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &xs_obj, &ys_obj)) {
        return -1;
    }
    if (!parse_strict_int_axis(xs_obj, self->core->xs) ||
        !parse_strict_int_axis(ys_obj, self->core->ys)) {
        return -1;
    }
    return 0;
}

PyObject* NativeAxisPriceCache_run(
    PyNativeAxisPriceCache* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "run(gi, start, end, price, charge) expects 5 arguments");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], gi)) return nullptr;

    PyObject* start = args[1];
    PyObject* end = args[2];
    if (PyTuple_CheckExact(start) && PyTuple_GET_SIZE(start) == 2 &&
        PyTuple_CheckExact(end) && PyTuple_GET_SIZE(end) == 2) {
        if (!parse_int64(PyTuple_GET_ITEM(start, 0), ax) ||
            !parse_int64(PyTuple_GET_ITEM(start, 1), ay) ||
            !parse_int64(PyTuple_GET_ITEM(end, 0), bx) ||
            !parse_int64(PyTuple_GET_ITEM(end, 1), by)) {
            return nullptr;
        }
    } else {
        PyRef sseq = PyRef::adopt(PySequence_Fast(start, "start must be a 2-sequence"));
        if (!sseq) return nullptr;
        PyRef eseq = PyRef::adopt(PySequence_Fast(end, "end must be a 2-sequence"));
        if (!eseq) return nullptr;
        const bool ok = PySequence_Fast_GET_SIZE(sseq.get()) == 2 &&
                        PySequence_Fast_GET_SIZE(eseq.get()) == 2 &&
                        parse_int64(PySequence_Fast_GET_ITEM(sseq.get(), 0), ax) &&
                        parse_int64(PySequence_Fast_GET_ITEM(sseq.get(), 1), ay) &&
                        parse_int64(PySequence_Fast_GET_ITEM(eseq.get(), 0), bx) &&
                        parse_int64(PySequence_Fast_GET_ITEM(eseq.get(), 1), by);
        if (!ok) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "start/end must be 2-tuples");
            return nullptr;
        }
    }

    int64_t out_val = 0;
    const int st = self->core->run(args[3], args[4], gi, ax, ay, bx, by, out_val);
    if (st < 0) return nullptr;
    if (st == 0) Py_RETURN_NONE;
    return PyLong_FromLongLong(out_val);
}

PyObject* NativeAxisPriceCache_run_ctx(
    PyNativeAxisPriceCache* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 10) {
        PyErr_SetString(PyExc_TypeError, "run_ctx expects 10 arguments");
        return nullptr;
    }
    int64_t gi = 0, ax = 0, ay = 0, bx = 0, by = 0;
    int64_t rem_budget = 0, sp_steps = 0, pc_hits = 0, max_rc = 0;
    if (!parse_int64(args[0], gi) || !parse_int64(args[4], rem_budget) ||
        !parse_int64(args[5], sp_steps) || !parse_int64(args[6], pc_hits) ||
        !parse_int64(args[7], max_rc)) {
        return nullptr;
    }
    PyObject* start = args[1];
    PyObject* end = args[2];
    if (!PyTuple_CheckExact(start) || PyTuple_GET_SIZE(start) != 2 ||
        !PyTuple_CheckExact(end) || PyTuple_GET_SIZE(end) != 2 ||
        !parse_int64(PyTuple_GET_ITEM(start, 0), ax) ||
        !parse_int64(PyTuple_GET_ITEM(start, 1), ay) ||
        !parse_int64(PyTuple_GET_ITEM(end, 0), bx) ||
        !parse_int64(PyTuple_GET_ITEM(end, 1), by)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "start/end must be 2-tuples");
        return nullptr;
    }
    if (Py_TYPE(args[3]) != &NativeEdgeContextType) {
        PyErr_SetString(PyExc_TypeError, "edge_ctx must be NativeEdgeContext");
        return nullptr;
    }
    const auto* ctx = reinterpret_cast<PyNativeEdgeContext*>(args[3])->core;
    int64_t steps_charged = 0;
    int64_t out_cost = 0;
    bool budget_stopped = false;
    const int st = run_axis_price_ctx_core(
        *self->core, *ctx, gi, ax, ay, bx, by,
        rem_budget, steps_charged, sp_steps, pc_hits,
        static_cast<size_t>(std::max<int64_t>(0, max_rc)),
        args[8], args[9], out_cost, budget_stopped);
    if (st < 0) return nullptr;

    PyRef out = PyRef::adopt(PyTuple_New(5));
    PyRef o0 = (st == 1) ? PyRef::adopt(PyLong_FromLongLong(out_cost)) : PyRef::borrow(Py_None);
    PyRef o1 = PyRef::adopt(PyLong_FromLongLong(steps_charged));
    PyRef o2 = PyRef::adopt(PyLong_FromLongLong(sp_steps));
    PyRef o3 = PyRef::adopt(PyLong_FromLongLong(pc_hits));
    PyRef o4 = PyRef::borrow(budget_stopped ? Py_True : Py_False);
    if (!out || !o0 || !o1 || !o2 || !o3 || !o4) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o0.release());
    PyTuple_SET_ITEM(out.get(), 1, o1.release());
    PyTuple_SET_ITEM(out.get(), 2, o2.release());
    PyTuple_SET_ITEM(out.get(), 3, o3.release());
    PyTuple_SET_ITEM(out.get(), 4, o4.release());
    return out.release();
}

PyMethodDef NativeAxisPriceCache_methods[] = {
    {"run", reinterpret_cast<PyCFunction>(NativeAxisPriceCache_run), METH_FASTCALL,
     "Evaluate cached orthogonal range price preserving exact charge order."},
    {"run_ctx", reinterpret_cast<PyCFunction>(NativeAxisPriceCache_run_ctx), METH_FASTCALL,
     "Evaluate cached orthogonal range price natively via NativeEdgeContext."},
    {nullptr, nullptr, 0, nullptr}
};

}  // namespace

PyTypeObject NativeAxisPriceCacheType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.NativeAxisPriceCache",
    sizeof(PyNativeAxisPriceCache),
    0,
    reinterpret_cast<destructor>(NativeAxisPriceCache_dealloc),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "C++20 exact orthogonal axis price range-sum cache.",
    0, 0, 0, 0, 0, 0,
    NativeAxisPriceCache_methods,
    0, 0, 0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(NativeAxisPriceCache_init),
    0,
    NativeAxisPriceCache_new,
};

// ---------------------------------------------------------------------------
// 2. ViaTargetProjectorCore & PyNativeViaTargetProjector
// ---------------------------------------------------------------------------

ViaTargetProjectorCore::ViaTargetProjectorCore() {
    empty_tuple = PyTuple_New(0);
}

ViaTargetProjectorCore::~ViaTargetProjectorCore() {
    clear_frontier();
    Py_XDECREF(hmin_obj);
    Py_XDECREF(empty_tuple);
    for (auto& kv : owner_cache) {
        Py_XDECREF(kv.second);
    }
}

void ViaTargetProjectorCore::clear_frontier() {
    clear_frontier_pure();
    for (PyObject* u : stored_u_bases) Py_XDECREF(u);
    stored_u_bases.clear();
}

const std::vector<int32_t>* ViaTargetProjectorCore::lookup_owner_ids(
    const SpatialIndexCore* other_idx, int64_t epx, int64_t epy) {
    const OwnerCacheKey okey{reinterpret_cast<uintptr_t>(other_idx), epx, epy};
    auto it = owner_ids_cache.find(okey);
    if (it != owner_ids_cache.end()) return &it->second;

    std::vector<int32_t> ids;
    const Box64 q{epx, epy, epx, epy};
    bool parse_err = false;
    other_idx->for_each_intersecting(q, [&](const BucketItem& item) {
        if (parse_err) return;
        PyObject* owner_obj = other_idx->py_entries[item.entry_idx].owner_obj;
        if (PyLong_CheckExact(owner_obj)) {
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(owner_obj, &overflow);
            if (overflow || (v == -1 && PyErr_Occurred())) { parse_err = true; return; }
            ids.push_back(static_cast<int32_t>(v));
        } else if (PyUnicode_CheckExact(owner_obj) && PyUnicode_IS_ASCII(owner_obj)) {
            const Py_ssize_t len = PyUnicode_GET_LENGTH(owner_obj);
            const char* s = reinterpret_cast<const char*>(PyUnicode_1BYTE_DATA(owner_obj));
            if (len > 0 && len <= 10) {
                int64_t val = 0;
                bool ok_ascii = true;
                Py_ssize_t idx = 0;
                bool neg = false;
                if (s[0] == '-') { neg = true; idx = 1; if (len == 1) ok_ascii = false; }
                for (; idx < len; ++idx) {
                    const char ch = s[idx];
                    if (ch < '0' || ch > '9') { ok_ascii = false; break; }
                    val = val * 10 + (ch - '0');
                }
                if (ok_ascii) {
                    ids.push_back(static_cast<int32_t>(neg ? -val : val));
                    return;
                }
            }
            PyRef oid = PyRef::adopt(PyNumber_Long(owner_obj));
            if (!oid) { parse_err = true; return; }
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(oid.get(), &overflow);
            if (overflow || (v == -1 && PyErr_Occurred())) { parse_err = true; return; }
            ids.push_back(static_cast<int32_t>(v));
        } else {
            PyRef oid = PyRef::adopt(PyNumber_Long(owner_obj));
            if (!oid) { parse_err = true; return; }
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(oid.get(), &overflow);
            if (overflow || (v == -1 && PyErr_Occurred())) { parse_err = true; return; }
            ids.push_back(static_cast<int32_t>(v));
        }
    });
    if (parse_err) return nullptr;
    auto ins = owner_ids_cache.emplace(okey, std::move(ids));
    return &ins.first->second;
}

PyObject* ViaTargetProjectorCore::lookup_owners_borrowed(
    const SpatialIndexCore* other_idx, int64_t epx, int64_t epy) {
    const OwnerCacheKey okey{reinterpret_cast<uintptr_t>(other_idx), epx, epy};
    auto it = owner_cache.find(okey);
    if (it != owner_cache.end()) return it->second;

    std::vector<uint32_t> eidxs;
    const Box64 q{epx, epy, epx, epy};
    other_idx->for_each_intersecting(q, [&](const BucketItem& item) {
        eidxs.push_back(item.entry_idx);
    });
    if (eidxs.empty()) {
        Py_INCREF(empty_tuple);
        owner_cache.emplace(okey, empty_tuple);
        return empty_tuple;
    }
    PyRef tup = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(eidxs.size())));
    if (!tup) return nullptr;
    for (size_t i = 0; i < eidxs.size(); ++i) {
        PyObject* owner_obj = other_idx->py_entries[eidxs[i]].owner_obj;
        PyRef oid = PyLong_CheckExact(owner_obj)
            ? PyRef::borrow(owner_obj)
            : PyRef::adopt(PyNumber_Long(owner_obj));
        if (!oid) return nullptr;
        PyTuple_SET_ITEM(tup.get(), static_cast<Py_ssize_t>(i), oid.release());
    }
    PyObject* raw = tup.release();
    owner_cache.emplace(okey, raw);
    return raw;
}

namespace {

void NativeViaTargetProjector_dealloc(PyNativeViaTargetProjector* self) {
    delete self->core;
    self->core = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeViaTargetProjector_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<PyNativeViaTargetProjector*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->core = new ViaTargetProjectorCore();
    return reinterpret_cast<PyObject*>(self);
}

int NativeViaTargetProjector_init(
    PyNativeViaTargetProjector* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {
        "goal_set", "target_node", "target_singleton_none",
        "target_regions", "target_excluded_xy", "hmin",
        "grid", "turn_cost", "via_cost", "rect_bucket_nm",
        "neg_bucket_dedup", "via_succ_dir_1", "via_succ_dir_2",
        nullptr
    };
    PyObject *goal_set_obj = nullptr, *target_node_obj = nullptr;
    int target_singleton_none = 0;
    PyObject *target_regions_obj = nullptr, *target_excluded_obj = nullptr, *hmin_obj = nullptr;
    long long grid_ll = 1, turn_cost_ll = 0, via_cost_ll = 40000, rect_bucket_ll = 140;
    int neg_bucket_dedup = 0;
    int via_succ_dir_1 = 0, via_succ_dir_2 = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOpOOOLLLLpii", const_cast<char**>(kwlist),
            &goal_set_obj, &target_node_obj, &target_singleton_none,
            &target_regions_obj, &target_excluded_obj, &hmin_obj,
            &grid_ll, &turn_cost_ll, &via_cost_ll, &rect_bucket_ll,
            &neg_bucket_dedup, &via_succ_dir_1, &via_succ_dir_2)) {
        return -1;
    }
    if (Py_TYPE(hmin_obj) != &NativeLayerHminType) {
        PyErr_SetString(PyExc_TypeError, "hmin must be NativeLayerHmin");
        return -1;
    }

    auto* core = self->core;
    {
        PyRef seq = PyRef::adopt(PySequence_Fast(target_node_obj, "target_node must be (x, y, gi)"));
        if (!seq) return -1;
        int64_t tx = 0, ty = 0, tgi = 0;
        const bool ok = PySequence_Fast_GET_SIZE(seq.get()) == 3 &&
                        parse_int64(PySequence_Fast_GET_ITEM(seq.get(), 0), tx) &&
                        parse_int64(PySequence_Fast_GET_ITEM(seq.get(), 1), ty) &&
                        parse_int64(PySequence_Fast_GET_ITEM(seq.get(), 2), tgi);
        if (!ok) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "target_node must have length 3");
            return -1;
        }
        core->target_node_x = tx;
        core->target_node_y = ty;
        core->target_node_gi = static_cast<int32_t>(tgi);
    }

    {
        PyRef it = PyRef::adopt(PyObject_GetIter(goal_set_obj));
        if (!it) return -1;
        while (true) {
            PyRef item = PyRef::adopt(PyIter_Next(it.get()));
            if (!item) break;
            int64_t gx = 0, gy = 0, gl = 0;
            bool ok = false;
            if (PyTuple_CheckExact(item.get()) && PyTuple_GET_SIZE(item.get()) == 3) {
                ok = parse_int64(PyTuple_GET_ITEM(item.get(), 0), gx) &&
                     parse_int64(PyTuple_GET_ITEM(item.get(), 1), gy) &&
                     parse_int64(PyTuple_GET_ITEM(item.get(), 2), gl);
            } else {
                PyRef s = PyRef::adopt(PySequence_Fast(item.get(), "goal must be (x, y, gi)"));
                if (s) {
                    ok = PySequence_Fast_GET_SIZE(s.get()) == 3 &&
                         parse_int64(PySequence_Fast_GET_ITEM(s.get(), 0), gx) &&
                         parse_int64(PySequence_Fast_GET_ITEM(s.get(), 1), gy) &&
                         parse_int64(PySequence_Fast_GET_ITEM(s.get(), 2), gl);
                }
            }
            if (!ok) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "goal must be 3-tuple");
                return -1;
            }
            core->goal_by_layer[static_cast<int32_t>(gl)].emplace(gx, gy);
        }
        if (PyErr_Occurred()) return -1;
    }

    core->target_singleton_none = (target_singleton_none != 0);
    if (!parse_rect_list(target_regions_obj, core->target_regions)) return -1;
    {
        std::vector<std::pair<int64_t, int64_t>> ex_list;
        if (!parse_xy_list(target_excluded_obj, ex_list)) return -1;
        for (const auto& p : ex_list) core->target_excluded_xy.insert(p);
    }

    Py_XDECREF(core->hmin_obj);
    core->hmin_obj = Py_NewRef(hmin_obj);
    core->hmin_core = &reinterpret_cast<PyNativeLayerHmin*>(hmin_obj)->holder->core;
    core->grid = std::max<int64_t>(1, static_cast<int64_t>(grid_ll));
    core->turn_cost = static_cast<int64_t>(turn_cost_ll);
    core->via_cost = static_cast<int64_t>(via_cost_ll);
    core->rect_bucket_nm = std::max<int64_t>(1, static_cast<int64_t>(rect_bucket_ll));
    core->neg_bucket_dedup = (neg_bucket_dedup != 0);
    core->via_succ_dir_1 = static_cast<int32_t>(via_succ_dir_1);
    core->via_succ_dir_2 = static_cast<int32_t>(via_succ_dir_2);
    return 0;
}

bool parse_clipped_boxes(PyObject* clipped_obj, std::vector<Box64>& out) {
    PyRef clipped_seq = PyRef::adopt(PySequence_Fast(clipped_obj, "clipped must be a sequence of 4-tuples"));
    if (!clipped_seq) return false;
    const Py_ssize_t n_clipped = PySequence_Fast_GET_SIZE(clipped_seq.get());
    PyObject** c_items = PySequence_Fast_ITEMS(clipped_seq.get());
    out.clear();
    out.reserve(static_cast<size_t>(n_clipped));
    for (Py_ssize_t i = 0; i < n_clipped; ++i) {
        PyObject* box_obj = c_items[i];
        int64_t lx = 0, ly = 0, hx = 0, hy = 0;
        if (PyTuple_CheckExact(box_obj) && PyTuple_GET_SIZE(box_obj) == 4) {
            if (!parse_int64(PyTuple_GET_ITEM(box_obj, 0), lx) ||
                !parse_int64(PyTuple_GET_ITEM(box_obj, 1), ly) ||
                !parse_int64(PyTuple_GET_ITEM(box_obj, 2), hx) ||
                !parse_int64(PyTuple_GET_ITEM(box_obj, 3), hy)) {
                return false;
            }
            out.push_back(Box64{lx, ly, hx, hy});
        } else {
            Box64 tmp;
            if (!parse_rect_norm(box_obj, tmp)) return false;
            out.push_back(tmp);
        }
    }
    return true;
}

struct PyOwnerLookupCtx {
    ViaTargetProjectorCore* core = nullptr;
    const SpatialIndexCore* other_idx = nullptr;
};

const std::vector<int32_t>* py_owner_ids_trampoline(void* raw, int64_t end_x, int64_t end_y) {
    auto* h = static_cast<PyOwnerLookupCtx*>(raw);
    return h->core->lookup_owner_ids(h->other_idx, end_x, end_y);
}

PyObject* NativeViaTargetProjector_project(
    PyNativeViaTargetProjector* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 10) {
        PyErr_SetString(PyExc_TypeError,
            "project(ul, other, px, py, rx0, ry0, rx1, ry1, clipped, other_region_idx)");
        return nullptr;
    }
    int64_t ul_64 = 0, other_64 = 0, px = 0, py = 0, rx0 = 0, ry0 = 0, rx1 = 0, ry1 = 0;
    if (!parse_int64(args[0], ul_64) || !parse_int64(args[1], other_64) ||
        !parse_int64(args[2], px) || !parse_int64(args[3], py) ||
        !parse_int64(args[4], rx0) || !parse_int64(args[5], ry0) ||
        !parse_int64(args[6], rx1) || !parse_int64(args[7], ry1)) {
        return nullptr;
    }
    bool ok = true;
    const SpatialIndexCore* other_idx = extract_spatial_core(args[9], ok);
    if (!ok) return nullptr;

    std::vector<Box64> clipped;
    if (!parse_clipped_boxes(args[8], clipped)) return nullptr;

    auto* core = self->core;
    NativeViaTargetCacheEntry projected;
    const int pst = core->project_clipped_pure(
        static_cast<int32_t>(ul_64), static_cast<int32_t>(other_64),
        px, py, rx0, ry0, rx1, ry1, clipped,
        false, nullptr, nullptr, projected);
    if (pst == -1) {
        PyErr_SetString(PyExc_ValueError, "target projection starts outside its rectangle");
        return nullptr;
    }
    if (pst < 0) return nullptr;

    PyRef py_start_pt = PyRef::adopt(make_xy_tuple(px, py));
    if (!py_start_pt) return nullptr;

    const size_t n_recs = projected.recs.size();
    PyRef py_recs = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(n_recs)));
    if (!py_recs) return nullptr;

    for (size_t i = 0; i < n_recs; ++i) {
        const NativeViaTargetRec& r = projected.recs[i];
        const ViaPathKey& pk = r.pk;
        const int64_t end_x = (pk.len == 2) ? pk.x1 : pk.x2;
        const int64_t end_y = (pk.len == 2) ? pk.y1 : pk.y2;

        PyObject* owners_borrowed = core->empty_tuple;
        if (r.goal_waypoint < 0 && other_idx != nullptr) {
            owners_borrowed = core->lookup_owners_borrowed(other_idx, end_x, end_y);
            if (!owners_borrowed) return nullptr;
        }

        PyRef py_path = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(pk.len)));
        PyRef pt1 = PyRef::adopt(make_xy_tuple(pk.x1, pk.y1));
        if (!py_path || !pt1) return nullptr;
        Py_INCREF(py_start_pt.get());
        PyTuple_SET_ITEM(py_path.get(), 0, py_start_pt.get());
        PyTuple_SET_ITEM(py_path.get(), 1, pt1.release());
        if (pk.len == 3) {
            PyRef pt2 = PyRef::adopt(make_xy_tuple(pk.x2, pk.y2));
            if (!pt2) return nullptr;
            PyTuple_SET_ITEM(py_path.get(), 2, pt2.release());
        }

        PyRef rec = PyRef::adopt(PyTuple_New(12));
        PyRef o_ex = PyRef::adopt(PyLong_FromLongLong(r.ex));
        PyRef o_ey = PyRef::adopt(PyLong_FromLongLong(r.ey));
        PyRef o_p1y = PyRef::adopt(PyLong_FromLongLong(r.p1_y_nm));
        PyRef o_gw = (r.goal_waypoint < 0) ? PyRef::borrow(Py_None) : PyRef::adopt(PyLong_FromLong(r.goal_waypoint));
        PyRef o_ieg = PyRef::borrow(r.is_end_goal ? Py_True : Py_False);
        PyRef o_oulast = PyRef::adopt(PyLong_FromLong(r.other_ulast));
        PyRef o_owners = PyRef::borrow(owners_borrowed);
        PyRef o_ebx = PyRef::adopt(PyLong_FromLongLong(r.ebx));
        PyRef o_eby = PyRef::adopt(PyLong_FromLongLong(r.eby));
        PyRef o_eturn = PyRef::adopt(PyLong_FromLongLong(r.extra_turn));
        PyRef o_tlb = r.has_tail_lb ? PyRef::adopt(PyLong_FromLongLong(r.tail_lb)) : PyRef::borrow(Py_None);
        if (!rec || !o_ex || !o_ey || !o_p1y || !o_gw || !o_ieg || !o_oulast ||
            !o_owners || !o_ebx || !o_eby || !o_eturn || !o_tlb) {
            return nullptr;
        }
        PyTuple_SET_ITEM(rec.get(), 0, py_path.release());
        PyTuple_SET_ITEM(rec.get(), 1, o_ex.release());
        PyTuple_SET_ITEM(rec.get(), 2, o_ey.release());
        PyTuple_SET_ITEM(rec.get(), 3, o_p1y.release());
        PyTuple_SET_ITEM(rec.get(), 4, o_gw.release());
        PyTuple_SET_ITEM(rec.get(), 5, o_ieg.release());
        PyTuple_SET_ITEM(rec.get(), 6, o_oulast.release());
        PyTuple_SET_ITEM(rec.get(), 7, o_owners.release());
        PyTuple_SET_ITEM(rec.get(), 8, o_ebx.release());
        PyTuple_SET_ITEM(rec.get(), 9, o_eby.release());
        PyTuple_SET_ITEM(rec.get(), 10, o_eturn.release());
        PyTuple_SET_ITEM(rec.get(), 11, o_tlb.release());
        PyTuple_SET_ITEM(py_recs.get(), static_cast<Py_ssize_t>(i), rec.release());
    }

    PyRef out = PyRef::adopt(PyTuple_New(2));
    if (!out) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, Py_NewRef(projected.has_cross ? Py_True : Py_False));
    PyTuple_SET_ITEM(out.get(), 1, py_recs.release());
    return out.release();
}

PyObject* NativeViaTargetProjector_has_cached_target(
    PyNativeViaTargetProjector* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "has_cached_target(ul, other, rectangle, px, py, with_owners)");
        return nullptr;
    }
    int64_t ul = 0, other = 0, rect = 0, px = 0, py = 0;
    if (!parse_int64(args[0], ul) || !parse_int64(args[1], other) ||
        !parse_int64(args[2], rect) || !parse_int64(args[3], px) || !parse_int64(args[4], py)) {
        return nullptr;
    }
    const int wo_tr = PyObject_IsTrue(args[5]);
    if (wo_tr < 0) return nullptr;
    if (self->core->has_cached_target(
            static_cast<int32_t>(ul), static_cast<int32_t>(other),
            static_cast<int32_t>(rect), px, py, wo_tr == 1)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* NativeViaTargetProjector_enqueue_via_targets(
    PyNativeViaTargetProjector* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 20) {
        PyErr_SetString(PyExc_TypeError, "enqueue_via_targets expects 20 arguments");
        return nullptr;
    }
    int64_t ul_64 = 0, other_64 = 0, rect_64 = 0, px = 0, py = 0;
    int64_t rx0 = 0, ry0 = 0, rx1 = 0, ry1 = 0;
    int64_t u_du = 0, ulast_64 = 0, u_rate = 0, base_lb = 0;
    if (!parse_int64(args[0], ul_64) || !parse_int64(args[1], other_64) ||
        !parse_int64(args[2], rect_64) || !parse_int64(args[3], px) || !parse_int64(args[4], py) ||
        !parse_int64(args[5], rx0) || !parse_int64(args[6], ry0) ||
        !parse_int64(args[7], rx1) || !parse_int64(args[8], ry1) ||
        !parse_int64(args[12], u_du) || !parse_int64(args[13], ulast_64) ||
        !parse_int64(args[14], u_rate) || !parse_int64(args[15], base_lb)) {
        return nullptr;
    }
    bool ok = true;
    const SpatialIndexCore* other_idx = extract_spatial_core(args[10], ok);
    if (!ok) return nullptr;

    bool has_cutoff = false;
    int64_t cutoff = 0;
    if (args[16] != Py_None) {
        if (!parse_int64(args[16], cutoff)) return nullptr;
        has_cutoff = true;
    }
    bool has_pattern_cutoff = false;
    int64_t pattern_cutoff = 0;
    if (args[17] != Py_None) {
        if (!parse_int64(args[17], pattern_cutoff)) return nullptr;
        has_pattern_cutoff = true;
    }
    int64_t other_cap = -1;
    if (args[18] != Py_None) {
        if (!parse_int64(args[18], other_cap)) return nullptr;
    }
    const int ebs_tr = PyObject_IsTrue(args[19]);
    if (ebs_tr < 0) return nullptr;

    auto* core = self->core;
    const int32_t ul = static_cast<int32_t>(ul_64);
    const int32_t other = static_cast<int32_t>(other_64);
    const int32_t rect_id = static_cast<int32_t>(rect_64);
    const bool with_owners = (other_idx != nullptr);

    std::vector<Box64> clipped;
    const std::vector<Box64>* clipped_ptr = nullptr;
    if (!core->has_cached_target(ul, other, rect_id, px, py, with_owners)) {
        if (!parse_clipped_boxes(args[9], clipped)) return nullptr;
        clipped_ptr = &clipped;
    }

    PyOwnerLookupCtx octx{core, other_idx};
    const uint32_t cand_u_idx = static_cast<uint32_t>(core->stored_u_bases.size());
    bool u_used = false;
    const int mask = core->enqueue_via_targets_pure(
        ul, other, rect_id, px, py, rx0, ry0, rx1, ry1,
        clipped_ptr, with_owners,
        with_owners ? &py_owner_ids_trampoline : nullptr,
        &octx,
        cand_u_idx, u_du, static_cast<int32_t>(ulast_64), u_rate, base_lb,
        has_cutoff, cutoff, has_pattern_cutoff, pattern_cutoff,
        other_cap, ebs_tr == 1, u_used);
    if (mask == -1) {
        PyErr_SetString(PyExc_ValueError, "target projection starts outside its rectangle");
        return nullptr;
    }
    if (mask < 0) return nullptr;
    if (u_used) {
        Py_INCREF(args[11]);
        core->stored_u_bases.push_back(args[11]);
    }
    return PyLong_FromLong(mask);
}

PyObject* NativeViaTargetProjector_frontier_top_f(PyNativeViaTargetProjector* self, PyObject*) {
    int64_t top_f = 0;
    if (!self->core->frontier_top_f(top_f)) Py_RETURN_NONE;
    return PyLong_FromLongLong(top_f);
}

PyObject* NativeViaTargetProjector_frontier_pop(
    PyNativeViaTargetProjector* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "frontier_pop(dist_dict, restrict_rect_entries, other_caps)");
        return nullptr;
    }
    auto* core = self->core;
    RegionFrontierEntry e{};
    if (!core->pop_frontier_raw(e)) Py_RETURN_FALSE;

    PyObject* u_base = core->stored_u_bases[e.u_idx];
    if (PyDict_CheckExact(args[0])) {
        PyObject* d_obj = PyDict_GetItemWithError(args[0], u_base);
        if (!d_obj) {
            if (PyErr_Occurred()) return nullptr;
        } else {
            int64_t dv = 0;
            if (parse_int64(d_obj, dv) && e.u_du > dv) {
                Py_RETURN_FALSE;
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
                PyRef py_udu = PyRef::adopt(PyLong_FromLongLong(e.u_du));
                if (!py_udu) return nullptr;
                const int cmp = PyObject_RichCompareBool(py_udu.get(), d_obj, Py_GT);
                if (cmp < 0) return nullptr;
                if (cmp > 0) Py_RETURN_FALSE;
            }
        }
    }

    bool sat_pruned = false;
    const NativeViaTargetRec& rec = e.cache_entry->recs[e.rec_idx];
    const int re_tr = PyObject_IsTrue(args[1]);
    if (re_tr < 0) return nullptr;
    const bool restrict_entries = (re_tr == 1);
    if (restrict_entries && e.r_dest < -1 && args[2] != Py_None &&
        core->neg_bucket_dedup && e.cache_entry->has_owners) {
        const int32_t r_other = -e.r_dest - 2;
        PyRef py_rother = PyRef::adopt(PyLong_FromLong(r_other));
        if (!py_rother) return nullptr;
        PyObject* cap_obj = PyDict_GetItemWithError(args[2], py_rother.get());
        if (!cap_obj && PyErr_Occurred()) return nullptr;
        if (cap_obj && cap_obj != Py_None) {
            int64_t other_cap = -1;
            if (!parse_int64(cap_obj, other_cap)) return nullptr;
            if (!core->check_frontier_entry_open(e, r_other, other_cap, sat_pruned)) {
                if (sat_pruned) Py_RETURN_TRUE;
                Py_RETURN_FALSE;
            }
        }
    }

    const ViaPathKey& pk = rec.pk;
    PyRef py_path = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(pk.len)));
    PyRef pt0 = PyRef::adopt(make_xy_tuple(pk.x0, pk.y0));
    PyRef pt1 = PyRef::adopt(make_xy_tuple(pk.x1, pk.y1));
    if (!py_path || !pt0 || !pt1) return nullptr;
    PyTuple_SET_ITEM(py_path.get(), 0, pt0.release());
    PyTuple_SET_ITEM(py_path.get(), 1, pt1.release());
    if (pk.len == 3) {
        PyRef pt2 = PyRef::adopt(make_xy_tuple(pk.x2, pk.y2));
        if (!pt2) return nullptr;
        PyTuple_SET_ITEM(py_path.get(), 2, pt2.release());
    }

    PyRef out = PyRef::adopt(PyTuple_New(5));
    PyRef o_du = PyRef::adopt(PyLong_FromLongLong(e.u_du));
    PyRef o_dest = PyRef::adopt(PyLong_FromLong(e.r_dest));
    if (!out || !o_du || !o_dest) return nullptr;
    Py_INCREF(u_base);
    PyTuple_SET_ITEM(out.get(), 0, u_base);
    PyTuple_SET_ITEM(out.get(), 1, o_du.release());
    PyTuple_SET_ITEM(out.get(), 2, py_path.release());
    PyTuple_SET_ITEM(out.get(), 3, o_dest.release());
    PyTuple_SET_ITEM(out.get(), 4, Py_NewRef(sat_pruned ? Py_True : Py_False));
    return out.release();
}

PyObject* NativeViaTargetProjector_inc_rect_entry(
    PyNativeViaTargetProjector* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "inc_rect_entry(layer, rect_id, ulast, bx, by)");
        return nullptr;
    }
    int64_t layer = 0, rect_id = 0, ulast = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], layer) || !parse_int64(args[1], rect_id) ||
        !parse_int64(args[2], ulast) || !parse_int64(args[3], bx) || !parse_int64(args[4], by)) {
        return nullptr;
    }
    self->core->inc_rect_entry(
        static_cast<int32_t>(layer), static_cast<int32_t>(rect_id),
        static_cast<int32_t>(ulast), bx, by);
    Py_RETURN_NONE;
}

PyObject* NativeViaTargetProjector_clear_frontier(PyNativeViaTargetProjector* self, PyObject*) {
    self->core->clear_frontier();
    Py_RETURN_NONE;
}

PyMethodDef NativeViaTargetProjector_methods[] = {
    {"project", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_project), METH_FASTCALL,
     "Project clipped via target regions to exact sorted candidate records."},
    {"has_cached_target", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_has_cached_target), METH_FASTCALL,
     "Return True if via targets for (ul, other, rectangle, px, py, with_owners) are cached natively."},
    {"enqueue_via_targets", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_enqueue_via_targets), METH_FASTCALL,
     "Project/cache via targets natively and push surviving entries directly into the C++20 region_frontier heap."},
    {"frontier_top_f", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_frontier_top_f), METH_NOARGS,
     "Return minimum lb_f on the native region_frontier heap, or None if empty."},
    {"frontier_pop", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_frontier_pop), METH_FASTCALL,
     "Pop top entry from native region_frontier heap with integrated staleness and saturation pruning."},
    {"inc_rect_entry", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_inc_rect_entry), METH_FASTCALL,
     "Increment native rect_entry_expansions counter for saturation parity."},
    {"clear_frontier", reinterpret_cast<PyCFunction>(NativeViaTargetProjector_clear_frontier), METH_NOARGS,
     "Clear native region_frontier heap and rect_entry_expansions counters."},
    {nullptr, nullptr, 0, nullptr}
};

}  // namespace

PyTypeObject NativeViaTargetProjectorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_gridless_cpp.NativeViaTargetProjector",
    sizeof(PyNativeViaTargetProjector),
    0,
    reinterpret_cast<destructor>(NativeViaTargetProjector_dealloc),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "C++20 region A* via-target projector with integrated hmin and owner cache.",
    0, 0, 0, 0, 0, 0,
    NativeViaTargetProjector_methods,
    0, 0, 0, 0, 0, 0, 0,
    reinterpret_cast<initproc>(NativeViaTargetProjector_init),
    0,
    NativeViaTargetProjector_new,
};

// ---------------------------------------------------------------------------
// 3. Watchdog & wide-metal callbacks for Preflight & Cartesian A*
// ---------------------------------------------------------------------------

namespace {

bool call_preflight_watchdog(
    PyObject* cb, int phase_id, int64_t nodes, int64_t edges, bool packed) {
    if (!cb || cb == Py_None) return true;
    PyRef a0 = PyRef::adopt(PyLong_FromLong(phase_id));
    PyRef a1 = PyRef::adopt(PyLong_FromLongLong(nodes));
    PyRef a2 = PyRef::adopt(PyLong_FromLongLong(edges));
    PyRef a3 = PyRef::borrow(packed ? Py_True : Py_False);
    if (!a0 || !a1 || !a2 || !a3) return false;
    PyObject* argv[4] = {a0.get(), a1.get(), a2.get(), a3.get()};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(cb, argv, 4, nullptr));
    return static_cast<bool>(res);
}

bool call_astar_watchdog(
    PyObject* cb, int phase_id, int64_t expansions, int64_t term_priced, bool packed) {
    if (!cb || cb == Py_None) return true;
    PyRef a0 = PyRef::adopt(PyLong_FromLong(phase_id));
    PyRef a1 = PyRef::adopt(PyLong_FromLongLong(expansions));
    PyRef a2 = PyRef::adopt(PyLong_FromLongLong(term_priced));
    PyRef a3 = PyRef::borrow(packed ? Py_True : Py_False);
    if (!a0 || !a1 || !a2 || !a3) return false;
    PyObject* argv[4] = {a0.get(), a1.get(), a2.get(), a3.get()};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(cb, argv, 4, nullptr));
    return static_cast<bool>(res);
}

bool call_wide_wire_cb(
    PyObject* cb, int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
    bool& out_legal, int64_t& out_present, bool& out_rej_inc) {
    PyRef a0 = PyRef::adopt(PyLong_FromLong(gi));
    PyRef a1 = PyRef::adopt(PyLong_FromLongLong(ax));
    PyRef a2 = PyRef::adopt(PyLong_FromLongLong(ay));
    PyRef a3 = PyRef::adopt(PyLong_FromLongLong(bx));
    PyRef a4 = PyRef::adopt(PyLong_FromLongLong(by));
    if (!a0 || !a1 || !a2 || !a3 || !a4) return false;
    PyObject* argv[5] = {a0.get(), a1.get(), a2.get(), a3.get(), a4.get()};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(cb, argv, 5, nullptr));
    if (!res) return false;
    if (!PyTuple_CheckExact(res.get()) || PyTuple_GET_SIZE(res.get()) != 3) {
        PyErr_SetString(PyExc_TypeError, "wide_wire_cb must return (legal, present, rej_inc)");
        return false;
    }
    const int leg_tr = PyObject_IsTrue(PyTuple_GET_ITEM(res.get(), 0));
    if (leg_tr < 0) return false;
    out_legal = (leg_tr == 1);
    if (!parse_int64(PyTuple_GET_ITEM(res.get(), 1), out_present)) return false;
    const int rej_tr = PyObject_IsTrue(PyTuple_GET_ITEM(res.get(), 2));
    if (rej_tr < 0) return false;
    out_rej_inc = (rej_tr == 1);
    return true;
}

bool call_wide_via_cb(
    PyObject* cb, int64_t x, int64_t y, int32_t lo, int32_t n_flag,
    bool& out_legal, int64_t& out_plo, int64_t& out_phi, int64_t& out_pcut, bool& out_rej_inc) {
    PyRef a0 = PyRef::adopt(PyLong_FromLongLong(x));
    PyRef a1 = PyRef::adopt(PyLong_FromLongLong(y));
    PyRef a2 = PyRef::adopt(PyLong_FromLong(lo));
    PyRef a3 = PyRef::adopt(PyLong_FromLong(n_flag));
    if (!a0 || !a1 || !a2 || !a3) return false;
    PyObject* argv[4] = {a0.get(), a1.get(), a2.get(), a3.get()};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(cb, argv, 4, nullptr));
    if (!res) return false;
    if (!PyTuple_CheckExact(res.get()) || PyTuple_GET_SIZE(res.get()) != 5) {
        PyErr_SetString(PyExc_TypeError, "wide_via_cb must return (legal, plo, phi, pcut, rej_inc)");
        return false;
    }
    const int leg_tr = PyObject_IsTrue(PyTuple_GET_ITEM(res.get(), 0));
    if (leg_tr < 0) return false;
    out_legal = (leg_tr == 1);
    if (!parse_int64(PyTuple_GET_ITEM(res.get(), 1), out_plo) ||
        !parse_int64(PyTuple_GET_ITEM(res.get(), 2), out_phi) ||
        !parse_int64(PyTuple_GET_ITEM(res.get(), 3), out_pcut)) {
        return false;
    }
    const int rej_tr = PyObject_IsTrue(PyTuple_GET_ITEM(res.get(), 4));
    if (rej_tr < 0) return false;
    out_rej_inc = (rej_tr == 1);
    return true;
}

bool call_wide_cong_cb(
    PyObject* cb, int kind, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e,
    int64_t& out_val) {
    PyRef p0 = PyRef::adopt(PyLong_FromLong(kind));
    PyRef p1 = PyRef::adopt(PyLong_FromLongLong(a));
    PyRef p2 = PyRef::adopt(PyLong_FromLongLong(b));
    PyRef p3 = PyRef::adopt(PyLong_FromLongLong(c));
    PyRef p4 = PyRef::adopt(PyLong_FromLongLong(d));
    PyRef p5 = PyRef::adopt(PyLong_FromLongLong(e));
    if (!p0 || !p1 || !p2 || !p3 || !p4 || !p5) return false;
    PyObject* argv[6] = {p0.get(), p1.get(), p2.get(), p3.get(), p4.get(), p5.get()};
    PyRef res = PyRef::adopt(PyObject_Vectorcall(cb, argv, 6, nullptr));
    if (!res) return false;
    return parse_int64(res.get(), out_val);
}

bool parse_endpoint_specs(PyObject* obj, std::vector<AStarEndpointSpec>& out) {
    PyRef seq = PyRef::adopt(PySequence_Fast(obj, "endpoint specs must be a sequence"));
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    out.resize(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* row = items[i];
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 9) {
            PyErr_SetString(PyExc_TypeError, "each endpoint spec must be a 9-tuple");
            return false;
        }
        int64_t xp = 0, yp = 0, lp = 0, vx = 0, vy = 0, vlo = 0, vn = 0, lb = 0;
        if (!parse_int64(PyTuple_GET_ITEM(row, 0), xp) ||
            !parse_int64(PyTuple_GET_ITEM(row, 1), yp) ||
            !parse_int64(PyTuple_GET_ITEM(row, 2), lp) ||
            !parse_int64(PyTuple_GET_ITEM(row, 4), vx) ||
            !parse_int64(PyTuple_GET_ITEM(row, 5), vy) ||
            !parse_int64(PyTuple_GET_ITEM(row, 6), vlo) ||
            !parse_int64(PyTuple_GET_ITEM(row, 7), vn) ||
            !parse_int64(PyTuple_GET_ITEM(row, 8), lb)) {
            return false;
        }
        const int hv_tr = PyObject_IsTrue(PyTuple_GET_ITEM(row, 3));
        if (hv_tr < 0) return false;
        out[static_cast<size_t>(i)] = AStarEndpointSpec{
            static_cast<int32_t>(xp), static_cast<int32_t>(yp), static_cast<int32_t>(lp),
            hv_tr == 1,
            vx, vy, static_cast<int32_t>(vlo), static_cast<int32_t>(vn), lb
        };
    }
    return true;
}

PyObject* make_small_path_tuple(const SmallPath& sp) {
    PyRef t = PyRef::adopt(PyTuple_New(sp.len));
    if (!t) return nullptr;
    for (int i = 0; i < sp.len; ++i) {
        PyRef pt = PyRef::adopt(make_xy_tuple(sp.pts[i].first, sp.pts[i].second));
        if (!pt) return nullptr;
        PyTuple_SET_ITEM(t.get(), i, pt.release());
    }
    return t.release();
}

}  // namespace

// ---------------------------------------------------------------------------
// 4. NativeEdgeContext.run_hard_connectivity_preflight (delegates to pure C++20)
// ---------------------------------------------------------------------------

namespace {

struct PySearchHookCtx {
    PyObject* wd_cb = nullptr;
    PyObject* ww_cb = nullptr;
    PyObject* wv_cb = nullptr;
    PyObject* wc_cb = nullptr;
};

bool py_preflight_wd_trampoline(void* raw, int32_t kind, int64_t nodes, int64_t edges, bool packed) {
    const auto* h = static_cast<const PySearchHookCtx*>(raw);
    return call_preflight_watchdog(h->wd_cb, kind, nodes, edges, packed);
}

bool py_astar_wd_trampoline(void* raw, int32_t kind, int64_t exps, int64_t tap, bool packed) {
    const auto* h = static_cast<const PySearchHookCtx*>(raw);
    return call_astar_watchdog(h->wd_cb, kind, exps, tap, packed);
}

bool py_wide_wire_trampoline(void* raw, int32_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by,
                             bool& out_legal, int64_t& out_pres, bool& out_rej_inc) {
    const auto* h = static_cast<const PySearchHookCtx*>(raw);
    return call_wide_wire_cb(h->ww_cb, gi, ax, ay, bx, by, out_legal, out_pres, out_rej_inc);
}

bool py_wide_via_trampoline(void* raw, int64_t x, int64_t y, int32_t lo, int32_t nf,
                            bool& out_legal, int64_t& out_plo, int64_t& out_phi,
                            int64_t& out_pcut, bool& out_rej_inc) {
    const auto* h = static_cast<const PySearchHookCtx*>(raw);
    return call_wide_via_cb(h->wv_cb, x, y, lo, nf, out_legal, out_plo, out_phi, out_pcut, out_rej_inc);
}

bool py_wide_cong_trampoline(void* raw, int32_t kind, int64_t a0, int64_t a1, int64_t a2,
                             int64_t a3, int64_t a4, int64_t& out_cost) {
    const auto* h = static_cast<const PySearchHookCtx*>(raw);
    return call_wide_cong_cb(h->wc_cb, kind, a0, a1, a2, a3, a4, out_cost);
}

}  // namespace

PyObject* NativeEdgeContext_run_hard_connectivity_preflight(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 10) {
        PyErr_SetString(PyExc_TypeError, "run_hard_connectivity_preflight expects 10 arguments");
        return nullptr;
    }
    std::vector<int64_t> xs, ys, start_ids, goal_ids;
    std::vector<int32_t> layers;
    int64_t max_exp = 0;
    if (!parse_int64_list(args[0], xs) || !parse_int64_list(args[1], ys) ||
        !parse_int32_list(args[2], layers) || !parse_int64_list(args[3], start_ids) ||
        !parse_int64_list(args[4], goal_ids) || !parse_int64(args[5], max_exp)) {
        return nullptr;
    }
    const int wc_tr = PyObject_IsTrue(args[6]);
    if (wc_tr < 0) return nullptr;

    PySearchHookCtx hctx{args[7], args[8], args[9], nullptr};
    GridlessSearchHooks hooks{
        &hctx, &py_preflight_wd_trampoline, nullptr,
        &py_wide_wire_trampoline, &py_wide_via_trampoline, nullptr
    };

    const PreflightResult res = run_hard_connectivity_preflight_pure(
        *self->core, xs, ys, layers, std::move(start_ids), std::move(goal_ids),
        max_exp, wc_tr == 1, hooks);
    if (res.status == -2) return nullptr;

    PyRef out = PyRef::adopt(PyTuple_New(4));
    PyRef o0 = PyRef::adopt(PyLong_FromLong(res.status));
    PyRef o1 = PyRef::adopt(PyLong_FromLongLong(res.connectivity_nodes));
    PyRef o2 = PyRef::adopt(PyLong_FromLongLong(res.connectivity_edges));
    if (!out || !o0 || !o1 || !o2) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o0.release());
    PyTuple_SET_ITEM(out.get(), 1, o1.release());
    PyTuple_SET_ITEM(out.get(), 2, o2.release());
    PyTuple_SET_ITEM(out.get(), 3, Py_NewRef(res.packed_enabled ? Py_True : Py_False));
    return out.release();
}

// ---------------------------------------------------------------------------
// 5. NativeEdgeContext.run_cartesian_astar (delegates to pure C++20)
// ---------------------------------------------------------------------------

PyObject* NativeEdgeContext_run_cartesian_astar(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 15) {
        PyErr_SetString(PyExc_TypeError, "run_cartesian_astar expects 15 arguments");
        return nullptr;
    }
    std::vector<int64_t> xs, ys;
    std::vector<int32_t> layers;
    std::vector<AStarEndpointSpec> starts, goals;
    int64_t budget = 0, solve_exp_total = 0, solve_exp_budget = 0, max_exp = 0;
    if (!parse_int64_list(args[0], xs) || !parse_int64_list(args[1], ys) ||
        !parse_int32_list(args[2], layers) || !parse_endpoint_specs(args[3], starts) ||
        !parse_endpoint_specs(args[4], goals) || !parse_int64(args[5], budget) ||
        !parse_int64(args[6], solve_exp_total) || !parse_int64(args[7], solve_exp_budget) ||
        !parse_int64(args[8], max_exp)) {
        return nullptr;
    }
    const int wc_tr = PyObject_IsTrue(args[9]);
    const int igd_tr = PyObject_IsTrue(args[10]);
    const int fp_tr = PyObject_IsTrue(args[12]);
    if (wc_tr < 0 || igd_tr < 0 || fp_tr < 0) return nullptr;
    const bool has_incumbent = (args[11] != Py_None);
    int64_t incumbent_score = 0;
    if (has_incumbent && !parse_int64(args[11], incumbent_score)) return nullptr;
    if (!PyTuple_CheckExact(args[14]) || PyTuple_GET_SIZE(args[14]) != 3) {
        PyErr_SetString(PyExc_TypeError, "fallbacks must be a 3-tuple (wide_wire_cb, wide_via_cb, wide_cong_cb)");
        return nullptr;
    }

    PySearchHookCtx hctx{
        args[13],
        PyTuple_GET_ITEM(args[14], 0),
        PyTuple_GET_ITEM(args[14], 1),
        PyTuple_GET_ITEM(args[14], 2),
    };
    GridlessSearchHooks hooks{
        &hctx, nullptr, &py_astar_wd_trampoline,
        &py_wide_wire_trampoline, &py_wide_via_trampoline, &py_wide_cong_trampoline
    };

    const AStarResult res = run_cartesian_astar_pure(
        *self->core, xs, ys, layers, starts, goals,
        budget, solve_exp_total, solve_exp_budget, max_exp,
        wc_tr == 1, igd_tr == 1, has_incumbent, incumbent_score, fp_tr == 1, hooks);
    if (res.status == -2) return nullptr;

    PyRef py_path;
    if (res.status == 0) {
        py_path = PyRef::adopt(PyList_New(static_cast<Py_ssize_t>(res.path.size())));
        if (!py_path) return nullptr;
        for (size_t i = 0; i < res.path.size(); ++i) {
            const auto& pn = res.path[i];
            PyRef row = PyRef::adopt(PyTuple_New(6));
            PyRef o0 = PyRef::adopt(PyLong_FromLongLong(pn.x));
            PyRef o1 = PyRef::adopt(PyLong_FromLongLong(pn.y));
            PyRef o2 = PyRef::adopt(PyLong_FromLong(pn.gi));
            PyRef o3 = PyRef::adopt(PyLong_FromLong(pn.last_dir));
            PyRef o4 = PyRef::adopt(PyLong_FromLong(pn.via_lo));
            PyRef o5 = PyRef::adopt(PyLong_FromLong(pn.via_n));
            if (!row || !o0 || !o1 || !o2 || !o3 || !o4 || !o5) return nullptr;
            PyTuple_SET_ITEM(row.get(), 0, o0.release());
            PyTuple_SET_ITEM(row.get(), 1, o1.release());
            PyTuple_SET_ITEM(row.get(), 2, o2.release());
            PyTuple_SET_ITEM(row.get(), 3, o3.release());
            PyTuple_SET_ITEM(row.get(), 4, o4.release());
            PyTuple_SET_ITEM(row.get(), 5, o5.release());
            PyList_SET_ITEM(py_path.get(), static_cast<Py_ssize_t>(i), row.release());
        }
    } else {
        py_path = PyRef::borrow(Py_None);
    }

    PyRef out = PyRef::adopt(PyTuple_New(6));
    PyRef o_st = PyRef::adopt(PyLong_FromLong(res.status));
    PyRef o_exp = PyRef::adopt(PyLong_FromLongLong(res.expansions));
    PyRef o_tap = PyRef::adopt(PyLong_FromLongLong(res.terminal_access_priced));
    PyRef o_rsi = PyRef::adopt(PyLong_FromLong(res.root_start_idx));
    if (!out || !o_st || !o_exp || !o_tap || !o_rsi) return nullptr;
    PyTuple_SET_ITEM(out.get(), 0, o_st.release());
    PyTuple_SET_ITEM(out.get(), 1, o_exp.release());
    PyTuple_SET_ITEM(out.get(), 2, o_tap.release());
    PyTuple_SET_ITEM(out.get(), 3, Py_NewRef(res.packed_enabled ? Py_True : Py_False));
    PyTuple_SET_ITEM(out.get(), 4, o_rsi.release());
    PyTuple_SET_ITEM(out.get(), 5, py_path.release());
    return out.release();
}

// ---------------------------------------------------------------------------
// 6. NativeEdgeContext.advance_region_step_ctx (delegates to pure C++20)
// ---------------------------------------------------------------------------

namespace {

struct PyRegionStepHookCtx {
    AxisPriceCacheCore* cache = nullptr;
    PyObject* wd_cb = nullptr;
    PyObject* fb_wire_fn = nullptr;
    PyObject* fb_price_fn = nullptr;
};

bool py_region_wd_trampoline(void* raw, int32_t kind_id) {
    const auto* h = static_cast<const PyRegionStepHookCtx*>(raw);
    if (!h->wd_cb || h->wd_cb == Py_None) return true;
    PyRef a0 = PyRef::adopt(PyLong_FromLong(kind_id));
    if (!a0) return false;
    PyObject* arg = a0.get();
    PyRef r = PyRef::adopt(PyObject_Vectorcall(h->wd_cb, &arg, 1, nullptr));
    return static_cast<bool>(r);
}

bool py_region_fb_wire_trampoline(
    void* raw, int64_t ul, int64_t ux, int64_t uy, int64_t vx, int64_t vy, bool& out_legal) {
    const auto* h = static_cast<const PyRegionStepHookCtx*>(raw);
    PyRef a0 = PyRef::adopt(PyLong_FromLongLong(ul));
    PyRef a1 = PyRef::adopt(PyLong_FromLongLong(ux));
    PyRef a2 = PyRef::adopt(PyLong_FromLongLong(uy));
    PyRef a3 = PyRef::adopt(PyLong_FromLongLong(vx));
    PyRef a4 = PyRef::adopt(PyLong_FromLongLong(vy));
    if (!a0 || !a1 || !a2 || !a3 || !a4) return false;
    PyObject* argv[5] = {a0.get(), a1.get(), a2.get(), a3.get(), a4.get()};
    PyRef r = PyRef::adopt(PyObject_Vectorcall(h->fb_wire_fn, argv, 5, nullptr));
    if (!r) return false;
    const int is_leg = PyObject_IsTrue(r.get());
    if (is_leg < 0) return false;
    out_legal = (is_leg == 1);
    return true;
}

bool py_region_fb_price_trampoline(
    void* raw, int64_t gi, int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t& out_val) {
    const auto* h = static_cast<const PyRegionStepHookCtx*>(raw);
    return h->cache->call_price(h->fb_price_fn, gi, ax, ay, bx, by, out_val);
}

}  // namespace

PyObject* NativeEdgeContext_advance_region_step_ctx(
    PyNativeEdgeContext* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 16) {
        PyErr_SetString(PyExc_TypeError, "advance_region_step_ctx expects 16 arguments");
        return nullptr;
    }
    PyObject* u_obj = args[0];
    PyObject* path_obj = args[1];
    int64_t next_rect = 0, grid = 1;
    int64_t expansions = 0, budget = 0, event_work = 0, event_budget = 0;
    int64_t rem_solve = 0, sp_steps = 0, pc_hits = 0, max_rc = 0;
    if (!parse_int64(args[2], next_rect) || !parse_int64(args[3], grid) ||
        !parse_int64(args[5], expansions) || !parse_int64(args[6], budget) ||
        !parse_int64(args[7], event_work) || !parse_int64(args[8], event_budget) ||
        !parse_int64(args[9], rem_solve) || !parse_int64(args[10], sp_steps) ||
        !parse_int64(args[11], pc_hits) || !parse_int64(args[12], max_rc)) {
        return nullptr;
    }
    if (Py_TYPE(args[4]) != &NativeAxisPriceCacheType) {
        PyErr_SetString(PyExc_TypeError, "axis_price_cache must be NativeAxisPriceCache");
        return nullptr;
    }
    auto* acache = reinterpret_cast<PyNativeAxisPriceCache*>(args[4])->core;

    auto make_ret = [](PyRef step_obj, int64_t ev_ch, int64_t pr_ch,
                       int64_t sps, int64_t pch, bool bstop) -> PyObject* {
        PyRef out = PyRef::adopt(PyTuple_New(6));
        PyRef o1 = PyRef::adopt(PyLong_FromLongLong(ev_ch));
        PyRef o2 = PyRef::adopt(PyLong_FromLongLong(pr_ch));
        PyRef o3 = PyRef::adopt(PyLong_FromLongLong(sps));
        PyRef o4 = PyRef::adopt(PyLong_FromLongLong(pch));
        if (!out || !o1 || !o2 || !o3 || !o4) return nullptr;
        PyTuple_SET_ITEM(out.get(), 0, step_obj.release());
        PyTuple_SET_ITEM(out.get(), 1, o1.release());
        PyTuple_SET_ITEM(out.get(), 2, o2.release());
        PyTuple_SET_ITEM(out.get(), 3, o3.release());
        PyTuple_SET_ITEM(out.get(), 4, o4.release());
        PyTuple_SET_ITEM(out.get(), 5, Py_NewRef(bstop ? Py_True : Py_False));
        return out.release();
    };

    if (!PyTuple_CheckExact(path_obj)) {
        PyErr_SetString(PyExc_TypeError, "path must be a tuple");
        return nullptr;
    }
    const Py_ssize_t plen = PyTuple_GET_SIZE(path_obj);
    if (plen < 2) {
        return make_ret(PyRef::borrow(Py_None), 0, 0, sp_steps, pc_hits, false);
    }

    if (expansions >= budget || expansions + event_work >= event_budget || rem_solve <= 0) {
        return make_ret(PyRef::borrow(Py_None), 0, 0, sp_steps, pc_hits, true);
    }

    if (!PyTuple_CheckExact(u_obj) || PyTuple_GET_SIZE(u_obj) < 10) {
        PyErr_SetString(PyExc_TypeError, "state u must be a >=10-tuple");
        return nullptr;
    }
    RegionStepState u_state{};
    if (!parse_int64(PyTuple_GET_ITEM(u_obj, 0), u_state.ux) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 1), u_state.uy) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 2), u_state.ul) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 3), u_state.ulast) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 4), u_state.rect) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 7), u_state.run_x) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 8), u_state.run_y) ||
        !parse_int64(PyTuple_GET_ITEM(u_obj, 9), u_state.run_dir)) {
        return nullptr;
    }

    PyObject* p1 = PyTuple_GET_ITEM(path_obj, 1);
    int64_t p1x = 0, p1y = 0;
    if (!PyTuple_CheckExact(p1) || PyTuple_GET_SIZE(p1) != 2 ||
        !parse_int64(PyTuple_GET_ITEM(p1, 0), p1x) ||
        !parse_int64(PyTuple_GET_ITEM(p1, 1), p1y)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "path[1] must be a 2-tuple");
        return nullptr;
    }

    PyRegionStepHookCtx hctx{acache, args[13], args[14], args[15]};
    RegionStepHooks hooks{
        &hctx, &py_region_wd_trampoline, &py_region_fb_wire_trampoline, &py_region_fb_price_trampoline
    };
    const size_t max_rc_sz = static_cast<size_t>(std::max<int64_t>(0, max_rc));
    const RegionStepResult res = advance_region_step_pure(
        *self->core, *acache, u_state, p1x, p1y, static_cast<int64_t>(plen - 2),
        next_rect, grid, expansions, budget, event_work, event_budget, rem_solve,
        sp_steps, pc_hits, max_rc_sz, hooks);
    if (res.status == -1) return nullptr;
    if (res.status == -2) {
        PyErr_SetString(PyExc_ValueError, "cost run must be orthogonal");
        return nullptr;
    }
    if (res.status == 0) {
        return make_ret(PyRef::borrow(Py_None), res.ev_charged, res.price_charged,
                        sp_steps, pc_hits, res.budget_stopped);
    }

    const Py_ssize_t rest_len = plen - 2;
    PyRef py_rest = PyRef::adopt(PyTuple_New(rest_len));
    if (!py_rest) return nullptr;
    for (Py_ssize_t i = 0; i < rest_len; ++i) {
        PyObject* pt = PyTuple_GET_ITEM(path_obj, i + 2);
        Py_INCREF(pt);
        PyTuple_SET_ITEM(py_rest.get(), i, pt);
    }

    PyRef rv = PyRef::adopt(PyTuple_New(10));
    PyRef f0 = PyRef::adopt(PyLong_FromLongLong(res.vx));
    PyRef f1 = PyRef::adopt(PyLong_FromLongLong(res.vy));
    PyRef f2 = PyRef::adopt(PyLong_FromLongLong(u_state.ul));
    PyRef f3 = PyRef::adopt(PyLong_FromLongLong(res.vdir));
    PyRef f4 = PyRef::adopt(PyLong_FromLongLong(res.out_rect));
    PyRef f6 = PyRef::adopt(PyLong_FromLongLong(next_rect));
    PyRef f7 = PyRef::adopt(PyLong_FromLongLong(res.start_x));
    PyRef f8 = PyRef::adopt(PyLong_FromLongLong(res.start_y));
    PyRef f9 = PyRef::adopt(PyLong_FromLongLong(res.signed_dir));
    if (!rv || !f0 || !f1 || !f2 || !f3 || !f4 || !f6 || !f7 || !f8 || !f9) return nullptr;
    PyTuple_SET_ITEM(rv.get(), 0, f0.release());
    PyTuple_SET_ITEM(rv.get(), 1, f1.release());
    PyTuple_SET_ITEM(rv.get(), 2, f2.release());
    PyTuple_SET_ITEM(rv.get(), 3, f3.release());
    PyTuple_SET_ITEM(rv.get(), 4, f4.release());
    PyTuple_SET_ITEM(rv.get(), 5, py_rest.release());
    PyTuple_SET_ITEM(rv.get(), 6, f6.release());
    PyTuple_SET_ITEM(rv.get(), 7, f7.release());
    PyTuple_SET_ITEM(rv.get(), 8, f8.release());
    PyTuple_SET_ITEM(rv.get(), 9, f9.release());

    PyRef step_pair = PyRef::adopt(PyTuple_New(2));
    PyRef o_cost = PyRef::adopt(PyLong_FromLongLong(res.step_cost));
    if (!step_pair || !o_cost) return nullptr;
    PyTuple_SET_ITEM(step_pair.get(), 0, rv.release());
    PyTuple_SET_ITEM(step_pair.get(), 1, o_cost.release());
    return make_ret(std::move(step_pair), res.ev_charged, res.price_charged, sp_steps, pc_hits, false);
}

// ---------------------------------------------------------------------------
// 7. Region-Event geometry projection CPython wrappers
// ---------------------------------------------------------------------------

PyObject* py_project_portal_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 10) {
        PyErr_SetString(PyExc_TypeError, "project_portal_cpp expects 10 arguments");
        return nullptr;
    }
    PyObject* rects_obj = args[0];
    int64_t src = 0, tgt = 0, dir = 0, bnd = 0, lo = 0, hi = 0, px = 0, py = 0;
    if (!parse_int64(args[1], src) || !parse_int64(args[2], tgt) ||
        !parse_int64(args[4], dir) || !parse_int64(args[5], bnd) ||
        !parse_int64(args[6], lo) || !parse_int64(args[7], hi) ||
        !parse_int64(args[8], px) || !parse_int64(args[9], py)) {
        return nullptr;
    }
    const int ax_tr = PyObject_IsTrue(args[3]);
    if (ax_tr < 0) return nullptr;
    const bool axis_is_x = (ax_tr == 1);
    if (!PyTuple_CheckExact(rects_obj)) {
        PyErr_SetString(PyExc_TypeError, "rectangles must be a tuple");
        return nullptr;
    }
    const Py_ssize_t n_rects = PyTuple_GET_SIZE(rects_obj);
    if (src < 0 || src >= n_rects || tgt < 0 || tgt >= n_rects ||
        (dir != -1 && dir != 1) || lo >= hi) {
        PyErr_SetString(PyExc_ValueError, "invalid portal");
        return nullptr;
    }
    Box64 sr, tr;
    if (!parse_rect_norm(PyTuple_GET_ITEM(rects_obj, static_cast<Py_ssize_t>(src)), sr) ||
        !parse_rect_norm(PyTuple_GET_ITEM(rects_obj, static_cast<Py_ssize_t>(tgt)), tr)) {
        return nullptr;
    }
    SmallPath paths[2];
    int np = 0;
    const RegionGeomError gerr = compute_portal_paths_core(
        sr, tr, axis_is_x, dir, bnd, lo, hi, px, py, paths, np);
    if (gerr == RegionGeomError::EntryOutsideSource) {
        PyErr_SetString(PyExc_ValueError, "portal entry lies outside its source rectangle");
        return nullptr;
    }
    if (gerr == RegionGeomError::PortalMismatch) {
        PyErr_SetString(PyExc_ValueError, "portal does not join its declared rectangles");
        return nullptr;
    }
    PyRef out = PyRef::adopt(PyTuple_New(np));
    if (!out) return nullptr;
    for (int i = 0; i < np; ++i) {
        PyRef pt = PyRef::adopt(make_small_path_tuple(paths[i]));
        if (!pt) return nullptr;
        PyTuple_SET_ITEM(out.get(), i, pt.release());
    }
    return out.release();
}

PyObject* py_project_target_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError, "project_target_cpp expects 8 arguments");
        return nullptr;
    }
    PyObject* rects_obj = args[0];
    int64_t src = 0, px = 0, py = 0, tx0 = 0, ty0 = 0, tx1 = 0, ty1 = 0;
    if (!parse_int64(args[1], src) || !parse_int64(args[2], px) || !parse_int64(args[3], py) ||
        !parse_int64(args[4], tx0) || !parse_int64(args[5], ty0) ||
        !parse_int64(args[6], tx1) || !parse_int64(args[7], ty1)) {
        return nullptr;
    }
    if (!PyTuple_CheckExact(rects_obj)) {
        PyErr_SetString(PyExc_TypeError, "rectangles must be a tuple");
        return nullptr;
    }
    const Py_ssize_t n_rects = PyTuple_GET_SIZE(rects_obj);
    if (src < 0 || src >= n_rects) {
        PyErr_SetString(PyExc_ValueError, "invalid source rectangle");
        return nullptr;
    }
    Box64 sr;
    if (!parse_rect_norm(PyTuple_GET_ITEM(rects_obj, static_cast<Py_ssize_t>(src)), sr)) return nullptr;
    SmallPath paths[2];
    int np = 0;
    const RegionGeomError gerr = compute_target_paths_core(sr, Box64{tx0, ty0, tx1, ty1}, px, py, paths, np);
    if (gerr == RegionGeomError::EntryOutsideSource) {
        PyErr_SetString(PyExc_ValueError, "target projection starts outside its rectangle");
        return nullptr;
    }
    PyRef out = PyRef::adopt(PyTuple_New(np));
    if (!out) return nullptr;
    for (int i = 0; i < np; ++i) {
        PyRef pt = PyRef::adopt(make_small_path_tuple(paths[i]));
        if (!pt) return nullptr;
        PyTuple_SET_ITEM(out.get(), i, pt.release());
    }
    return out.release();
}

PyObject* py_settled_axis_prefix_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "settled_axis_prefix_cpp expects 6 arguments");
        return nullptr;
    }
    int64_t ax = 0, ay = 0, bx = 0, by = 0;
    if (!parse_int64(args[0], ax) || !parse_int64(args[1], ay) ||
        !parse_int64(args[2], bx) || !parse_int64(args[3], by)) {
        return nullptr;
    }
    if (ax == bx && ay == by) Py_RETURN_NONE;
    if (ax != bx && ay != by) {
        PyErr_SetString(PyExc_ValueError, "cost run must be orthogonal");
        return nullptr;
    }
    std::vector<int64_t> xs, ys;
    const bool horizontal = (ay == by);
    if (horizontal) {
        if (!parse_int64_list(args[4], xs)) return nullptr;
    } else {
        if (!parse_int64_list(args[5], ys)) return nullptr;
    }
    int64_t ox = 0, oy = 0;
    if (!settled_axis_prefix_core(ax, ay, bx, by, xs, ys, ox, oy)) Py_RETURN_NONE;
    return make_xy_tuple(ox, oy);
}

PyObject* py_batch_project_portals_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError, "batch_project_portals_cpp expects 8 arguments");
        return nullptr;
    }
    PyObject* rects_obj = args[0];
    if (!PyTuple_CheckExact(rects_obj)) {
        PyErr_SetString(PyExc_TypeError, "rectangles must be a tuple");
        return nullptr;
    }
    const Py_ssize_t n_rects = PyTuple_GET_SIZE(rects_obj);
    int64_t ul = 0, grid = 1, bucket_nm = 140, px = 0, py = 0;
    if (!parse_int64(args[2], ul) || !parse_int64(args[3], grid) ||
        !parse_int64(args[4], bucket_nm) || !parse_int64(args[6], px) ||
        !parse_int64(args[7], py)) {
        return nullptr;
    }
    const int nd_tr = PyObject_IsTrue(args[5]);
    if (nd_tr < 0) return nullptr;
    const bool neg_dedup = (nd_tr == 1);

    PyRef pseq = PyRef::adopt(PySequence_Fast(args[1], "portal_rows must be a sequence"));
    if (!pseq) return nullptr;
    const Py_ssize_t n_portals = PySequence_Fast_GET_SIZE(pseq.get());
    PyObject** pitems = PySequence_Fast_ITEMS(pseq.get());

    struct StepKey {
        SmallPath path;
        int64_t ptarget;
        bool operator==(const StepKey& o) const noexcept { return ptarget == o.ptarget && path == o.path; }
    };
    struct StepKeyHash {
        size_t operator()(const StepKey& k) const noexcept {
            return static_cast<size_t>(mix64(
                static_cast<uint64_t>(SmallPathHash{}(k.path)) ^ static_cast<uint64_t>(k.ptarget)));
        }
    };
    std::unordered_set<StepKey, StepKeyHash> seen;
    seen.reserve(static_cast<size_t>(n_portals) * 3 + 4);

    std::vector<PyObject*> out_recs;
    out_recs.reserve(static_cast<size_t>(n_portals) * 2);
    auto cleanup = [&]() {
        for (PyObject* o : out_recs) Py_DECREF(o);
        out_recs.clear();
    };

    for (Py_ssize_t i = 0; i < n_portals; ++i) {
        PyObject* row = pitems[i];
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 7) {
            cleanup();
            PyErr_SetString(PyExc_TypeError, "portal row must be 7-tuple");
            return nullptr;
        }
        int64_t src = 0, tgt = 0, dir = 0, bnd = 0, lo = 0, hi = 0;
        if (!parse_int64(PyTuple_GET_ITEM(row, 0), src) ||
            !parse_int64(PyTuple_GET_ITEM(row, 1), tgt) ||
            !parse_int64(PyTuple_GET_ITEM(row, 3), dir) ||
            !parse_int64(PyTuple_GET_ITEM(row, 4), bnd) ||
            !parse_int64(PyTuple_GET_ITEM(row, 5), lo) ||
            !parse_int64(PyTuple_GET_ITEM(row, 6), hi)) {
            cleanup();
            return nullptr;
        }
        const int ax_tr = PyObject_IsTrue(PyTuple_GET_ITEM(row, 2));
        if (ax_tr < 0) {
            cleanup();
            return nullptr;
        }
        const bool axis_is_x = (ax_tr == 1);
        if (src < 0 || src >= n_rects || tgt < 0 || tgt >= n_rects || (dir != -1 && dir != 1) || lo >= hi) {
            cleanup();
            PyErr_SetString(PyExc_ValueError, "invalid portal");
            return nullptr;
        }
        Box64 sr, tr;
        if (!parse_rect_norm(PyTuple_GET_ITEM(rects_obj, static_cast<Py_ssize_t>(src)), sr) ||
            !parse_rect_norm(PyTuple_GET_ITEM(rects_obj, static_cast<Py_ssize_t>(tgt)), tr)) {
            cleanup();
            return nullptr;
        }
        SmallPath paths[2];
        int np = 0;
        const RegionGeomError gerr = compute_portal_paths_core(
            sr, tr, axis_is_x, dir, bnd, lo, hi, px, py, paths, np);
        if (gerr == RegionGeomError::EntryOutsideSource) {
            cleanup();
            PyErr_SetString(PyExc_ValueError, "portal entry lies outside its source rectangle");
            return nullptr;
        }
        if (gerr == RegionGeomError::PortalMismatch) {
            cleanup();
            PyErr_SetString(PyExc_ValueError, "portal does not join its declared rectangles");
            return nullptr;
        }
        for (int pi = 0; pi < np; ++pi) {
            const SmallPath& sp = paths[pi];
            if (!seen.insert(StepKey{sp, tgt}).second) continue;
            const int64_t fx = sp.pts[sp.len - 1].first * grid;
            const int64_t fy = sp.pts[sp.len - 1].second * grid;
            const int64_t end_vdir = (sp.pts[sp.len - 1].second == sp.pts[sp.len - 2].second) ? 1 : 2;
            PyRef py_path = PyRef::adopt(make_small_path_tuple(sp));
            PyRef py_tgt = PyRef::adopt(PyLong_FromLongLong(tgt));
            PyRef py_rkey = PyRef::adopt(PyTuple_New(5));
            PyRef rk0 = PyRef::adopt(PyLong_FromLongLong(ul));
            PyRef rk1 = PyRef::adopt(PyLong_FromLongLong(tgt));
            PyRef rk2 = PyRef::adopt(neg_dedup ? PyLong_FromLongLong(end_vdir) : PyLong_FromLong(0));
            PyRef rk3 = PyRef::adopt(neg_dedup ? PyLong_FromLongLong(floor_div_pos(fx, bucket_nm)) : PyLong_FromLong(0));
            PyRef rk4 = PyRef::adopt(neg_dedup ? PyLong_FromLongLong(floor_div_pos(fy, bucket_nm)) : PyLong_FromLong(0));
            PyRef rec = PyRef::adopt(PyTuple_New(3));
            if (!py_path || !py_tgt || !py_rkey || !rk0 || !rk1 || !rk2 || !rk3 || !rk4 || !rec) {
                cleanup();
                return nullptr;
            }
            PyTuple_SET_ITEM(py_rkey.get(), 0, rk0.release());
            PyTuple_SET_ITEM(py_rkey.get(), 1, rk1.release());
            PyTuple_SET_ITEM(py_rkey.get(), 2, rk2.release());
            PyTuple_SET_ITEM(py_rkey.get(), 3, rk3.release());
            PyTuple_SET_ITEM(py_rkey.get(), 4, rk4.release());
            PyTuple_SET_ITEM(rec.get(), 0, py_path.release());
            PyTuple_SET_ITEM(rec.get(), 1, py_tgt.release());
            PyTuple_SET_ITEM(rec.get(), 2, py_rkey.release());
            out_recs.push_back(rec.release());
        }
    }

    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(out_recs.size())));
    if (!out) {
        cleanup();
        return nullptr;
    }
    for (size_t i = 0; i < out_recs.size(); ++i) {
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), out_recs[i]);
    }
    return out.release();
}

PyObject* py_batch_project_goal_targets_cpp(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "batch_project_goal_targets_cpp expects 5 arguments");
        return nullptr;
    }
    PyObject* rects_obj = args[0];
    if (!PyTuple_CheckExact(rects_obj)) {
        PyErr_SetString(PyExc_TypeError, "rectangles must be a tuple");
        return nullptr;
    }
    int64_t src = 0, px = 0, py = 0;
    if (!parse_int64(args[1], src) || !parse_int64(args[3], px) || !parse_int64(args[4], py)) {
        return nullptr;
    }
    const Py_ssize_t n_rects = PyTuple_GET_SIZE(rects_obj);
    if (src < 0 || src >= n_rects) {
        PyErr_SetString(PyExc_ValueError, "invalid source rectangle");
        return nullptr;
    }
    Box64 sr;
    if (!parse_rect_norm(PyTuple_GET_ITEM(rects_obj, static_cast<Py_ssize_t>(src)), sr)) return nullptr;
    std::vector<Box64> targets;
    if (!parse_rect_list(args[2], targets)) return nullptr;

    std::unordered_set<SmallPath, SmallPathHash> seen;
    seen.reserve(targets.size() * 2 + 4);
    std::vector<PyObject*> out_paths;
    out_paths.reserve(targets.size() * 2);
    auto cleanup = [&]() {
        for (PyObject* o : out_paths) Py_DECREF(o);
        out_paths.clear();
    };
    for (const auto& t : targets) {
        SmallPath paths[2];
        int np = 0;
        const RegionGeomError gerr = compute_target_paths_core(sr, t, px, py, paths, np);
        if (gerr == RegionGeomError::EntryOutsideSource) {
            cleanup();
            PyErr_SetString(PyExc_ValueError, "target projection starts outside its rectangle");
            return nullptr;
        }
        for (int pi = 0; pi < np; ++pi) {
            if (!seen.insert(paths[pi]).second) continue;
            PyRef pt = PyRef::adopt(make_small_path_tuple(paths[pi]));
            if (!pt) {
                cleanup();
                return nullptr;
            }
            out_paths.push_back(pt.release());
        }
    }
    PyRef out = PyRef::adopt(PyTuple_New(static_cast<Py_ssize_t>(out_paths.size())));
    if (!out) {
        cleanup();
        return nullptr;
    }
    for (size_t i = 0; i < out_paths.size(); ++i) {
        PyTuple_SET_ITEM(out.get(), static_cast<Py_ssize_t>(i), out_paths[i]);
    }
    return out.release();
}

}  // namespace clay_pnr::route::py_bind
