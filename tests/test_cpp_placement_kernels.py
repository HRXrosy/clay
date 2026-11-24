"""Comprehensive A/B differential tests (CLAY_DISABLE_CPP_PNR=0 vs 1) for clay._native._place_cpp."""
from __future__ import annotations

import builtins
import random
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import pytest

from clay._native import clear_native_cache, load_native


@contextmanager
def _native_mode(monkeypatch, *, enabled: bool):
    if enabled:
        monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    else:
        monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    clear_native_cache()
    try:
        yield
    finally:
        clear_native_cache()


def _grid_points(boxes):
    return {
        (x, y)
        for x0, y0, x1, y1 in boxes
        for x in range(x0, x1)
        for y in range(y0, y1)
    }


# ============================================================================
# 0. Module loading & env toggle
# ============================================================================


def test_place_cpp_module_loads_and_respects_disable_flag(monkeypatch):
    with _native_mode(monkeypatch, enabled=True):
        mod = load_native("_place_cpp")
        assert mod is not None, "_place_cpp extension must be compiled and loadable"
        for sym in (
            "rectangle_union_cover_cpp",
            "forbidden_box_cover_cpp",
            "NativeRectangleIndex",
            "exact_forbidden_union_cpp",
            "longest_path_positions_cpp",
            "pack_band_sp_cpp",
            "search_block_layout_cpp",
            "search_band_dev_layout_cpp",
        ):
            assert hasattr(mod, sym), f"missing exported symbol {sym}"

    with _native_mode(monkeypatch, enabled=False):
        assert load_native("_place_cpp") is None


# ============================================================================
# 1. rectangle_union_cover A/B differential tests
# ============================================================================


def test_rectangle_union_ab_random_adversarial_and_holes(monkeypatch):
    from clay.layout.spatial.rectangle_union import rectangle_union_cover

    cases = [
        (),
        ((0, 0, 1, 1),),
        ((0, 0, 2, 2), (0, 0, 2, 2), (0, 0, 2, 2)),
        # Ring with interior hole + touching ear
        ((-3, -3, 3, -2), (-3, 2, 3, 3), (-3, -2, -2, 2), (2, -2, 3, 2), (3, -1, 5, 1)),
        # Swiss cheese: two separate holes inside one outer frame
        (
            (0, 0, 10, 2), (0, 6, 10, 8),
            (0, 2, 2, 6), (4, 2, 6, 6), (8, 2, 10, 6),
        ),
        # Corner-touching checkerboard
        ((0, 0, 1, 1), (1, 1, 2, 2), (2, 0, 3, 1), (0, 2, 1, 3), (2, 2, 3, 3)),
        # Large 64-bit signed coordinates
        tuple((a + (1 << 50), b - (1 << 48), c + (1 << 50), d - (1 << 48))
              for a, b, c, d in ((-3, -3, 3, -2), (-3, 2, 3, 3), (-3, -2, -2, 2), (2, -2, 3, 2))),
        # Near INT64 limits (safe inside int64_t)
        (((1 << 62), -(1 << 62), (1 << 62) + 5, -(1 << 62) + 5),
         ((1 << 62) + 3, -(1 << 62) + 2, (1 << 62) + 9, -(1 << 62) + 7)),
        # Beyond int64_t (triggers seamless Python fallback)
        (((1 << 80), -(1 << 80), (1 << 80) + 10, -(1 << 80) + 10),
         ((1 << 80) + 5, -(1 << 80) + 5, (1 << 80) + 15, -(1 << 80) + 15)),
        ((-(1 << 63) - 1, 0, -(1 << 63) + 5, 4), (-(1 << 63) + 2, 1, -(1 << 63) + 8, 5)),
        (((1 << 63) - 2, 0, (1 << 63) + 3, 4), ((1 << 63) - 5, 0, (1 << 63) - 1, 4)),
    ]

    rng = random.Random(20261122)
    for _ in range(50):
        n = rng.randint(1, 40)
        boxes = []
        for _ in range(n):
            a, c = sorted(rng.sample(range(-12, 13), 2))
            b, d = sorted(rng.sample(range(-12, 13), 2))
            boxes.append((a, b, c, d))
        cases.append(tuple(boxes))

    for boxes in cases:
        with _native_mode(monkeypatch, enabled=True):
            cpp_res = rectangle_union_cover(boxes)
            cpp_gen = rectangle_union_cover(b for b in boxes)
        with _native_mode(monkeypatch, enabled=False):
            py_res = rectangle_union_cover(boxes)
        assert cpp_res == py_res
        assert cpp_gen == py_res


@pytest.mark.parametrize(
    "bad_boxes,exc_type",
    [
        (((0, 0, 0, 1),), ValueError),
        (((1, 0, 0, 1),), ValueError),
        (((0, 2, 1, 2),), ValueError),
        (((0, 3, 1, 2),), ValueError),
        (((0, 0, 1.0, 1),), TypeError),
        (((False, 0, 1, 1),), TypeError),
        (((0, True, 1, 1),), TypeError),
        (((0, 0, 1),), TypeError),
        (((0, 0, 1, 1, 2),), TypeError),
        ((5,), TypeError),
        # Overflow box mixed with invalid box must still raise exact error
        ((((1 << 80), 0, (1 << 80) + 1, 1), (0, 0, 0, 1)), ValueError),
        ((((1 << 80), 0, (1 << 80) + 1, 1), (0, 0, 1.5, 1)), TypeError),
    ],
)
def test_rectangle_union_ab_invalid_inputs(monkeypatch, bad_boxes, exc_type):
    from clay.layout.spatial.rectangle_union import rectangle_union_cover

    with _native_mode(monkeypatch, enabled=True):
        with pytest.raises(exc_type):
            rectangle_union_cover(bad_boxes)
    with _native_mode(monkeypatch, enabled=False):
        with pytest.raises(exc_type):
            rectangle_union_cover(bad_boxes)


def test_rectangle_union_cpp_checkpoint_propagates_base_exception(monkeypatch):
    from clay.layout.spatial.rectangle_union import rectangle_union_cover

    class CustomAbort(BaseException):
        pass

    boxes = tuple((0, i * 2, 10, i * 2 + 1) for i in range(4000))
    for enabled in (True, False):
        calls = 0

        def checkpoint():
            nonlocal calls
            calls += 1
            if calls == 7:
                raise CustomAbort("aborted mid-sweep")

        with _native_mode(monkeypatch, enabled=enabled):
            with pytest.raises(CustomAbort, match="aborted mid-sweep"):
                rectangle_union_cover(boxes, checkpoint=checkpoint)
        assert calls == 7


# ============================================================================
# 2. forbidden_box_cover A/B differential tests
# ============================================================================


def test_forbidden_box_cover_ab_random_multilayer_and_batches(monkeypatch):
    from clay.layout.spatial.collision_union import forbidden_box_cover

    rng = random.Random(9001)
    layers = (None, (1, 0), (2, 0), (3, 1), "metal1")
    for _ in range(30):
        def make_atoms():
            out = []
            for _ in range(rng.randint(0, 22)):
                x = rng.randrange(-15, 15)
                y = rng.randrange(-15, 15)
                out.append((
                    rng.choice(layers),
                    (x, y, x + rng.randrange(1, 6), y + rng.randrange(1, 6)),
                ))
            return tuple(out + out[:2])

        left, right = make_atoms(), make_atoms()
        for gap in (0, 1, 4):
            with _native_mode(monkeypatch, enabled=True):
                cpp_res = forbidden_box_cover(left, right, required_nm=gap)
            with _native_mode(monkeypatch, enabled=False):
                py_res = forbidden_box_cover(left, right, required_nm=gap)
            assert cpp_res == py_res

    # Multi-level binary tree batch (> 2048 unique boxes) + witness containment
    left_big = tuple((None, (i * 100, 0, i * 100 + 5000, 5000)) for i in range(70))
    right_big = tuple((None, (j, j * 100, j + 5000, j * 100 + 5000)) for j in range(70))
    with _native_mode(monkeypatch, enabled=True):
        cpp_big = forbidden_box_cover(left_big, right_big, required_nm=10)
    with _native_mode(monkeypatch, enabled=False):
        py_big = forbidden_box_cover(left_big, right_big, required_nm=10)
    assert cpp_big == py_big

    # 8-witness containment with 1 huge enclosing box + 6000 tiny contained boxes
    outer = (-500, -500, 15000, 15000)
    left_w = ((None, outer),) + tuple((None, (i, i, i + 2, i + 2)) for i in range(6000))
    right_w = ((None, (0, 0, 1, 1)),)
    with _native_mode(monkeypatch, enabled=True):
        cpp_w = forbidden_box_cover(left_w, right_w, required_nm=0)
    with _native_mode(monkeypatch, enabled=False):
        py_w = forbidden_box_cover(left_w, right_w, required_nm=0)
    assert cpp_w == py_w == ((outer,), 6001, 6001)


@pytest.mark.parametrize(
    "left,right,gap",
    [
        # Wide 32-bit boundary
        (((None, (-(2**31), 0, -(2**31) + 1, 1)), (None, (-(2**31) + 1, 0, -(2**31) + 2, 1))),
         ((None, (0, 0, 1, 1)),), 0),
        (((None, (-(2**31) - 1, 0, -(2**31), 1)), (None, (-(2**31), 0, -(2**31) + 1, 1))),
         ((None, (0, 0, 1, 1)),), 0),
        (((None, (2**31 - 3, 0, 2**31 - 2, 1)), (None, (2**31 - 2, 0, 2**31 - 1, 1))),
         ((None, (0, 0, 1, 1)),), 0),
        (((None, (2**31 - 2, 0, 2**31 - 1, 1)), (None, (2**31 - 1, 0, 2**31, 1))),
         ((None, (0, 0, 1, 1)),), 0),
        # Beyond 2**60 / int64_t overflow seamless fallback
        (((None, (2**75, 0, 2**75 + 5, 5)), (None, (2**75 + 2, 1, 2**75 + 7, 6))),
         ((None, (0, 0, 3, 3)),), 0),
        (((None, (0, 0, 2, 2)),), ((None, (0, 0, 2, 2)),), 2**70),
        # Non-shrinking cover fallback (len(exact) >= unique_count)
        (((None, (0, 0, 1, 1)),),
         ((None, (-4, -1, 1, 0)), (None, (-2, -2, -1, 1))), 0),
    ],
)
def test_forbidden_box_cover_ab_wide_overflow_and_fallback(monkeypatch, left, right, gap):
    from clay.layout.spatial.collision_union import forbidden_box_cover

    with _native_mode(monkeypatch, enabled=True):
        cpp_res = forbidden_box_cover(left, right, required_nm=gap)
    with _native_mode(monkeypatch, enabled=False):
        py_res = forbidden_box_cover(left, right, required_nm=gap)
    assert cpp_res == py_res


@pytest.mark.parametrize("bad_req", [-1, -2**80, False, True, 0.0, "0"])
def test_forbidden_box_cover_ab_invalid_required_nm(monkeypatch, bad_req):
    from clay.layout.spatial.collision_union import forbidden_box_cover

    atoms = ((None, (0, 0, 2, 2)),)
    for enabled in (True, False):
        with _native_mode(monkeypatch, enabled=enabled):
            with pytest.raises(ValueError, match="non-negative integer"):
                forbidden_box_cover(atoms, atoms, required_nm=bad_req)


def test_forbidden_box_cover_cpp_checkpoint_cancellation(monkeypatch):
    from clay.layout.spatial.collision_union import forbidden_box_cover

    class Cancelled(BaseException):
        pass

    atoms = tuple((None, (i * 10, 0, i * 10 + 3, 5)) for i in range(400))
    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 25:
            raise Cancelled("stop forbidden")

    with _native_mode(monkeypatch, enabled=True):
        with pytest.raises(Cancelled, match="stop forbidden"):
            forbidden_box_cover(atoms, atoms, required_nm=1, checkpoint=checkpoint)
    assert calls == 25


# ============================================================================
# 3. RectangleIndex / NativeRectangleIndex A/B differential tests
# ============================================================================


def test_rectangle_index_ab_conflicts_and_intersecting(monkeypatch):
    from clay.layout.spatial.collision_queries import RectangleIndex

    rng = random.Random(4242)
    for _ in range(50):
        indexed = []
        for _ in range(rng.randint(0, 60)):
            x, y = rng.randrange(-50, 50), rng.randrange(-50, 50)
            indexed.append((x, y, x + rng.randrange(1, 8), y + rng.randrange(1, 8)))
        # Inject exact duplicates to verify duplicate retention in intersecting()
        if indexed:
            indexed.extend(indexed[:3])
        indexed = tuple(indexed)

        queries = tuple(
            (x, y, x + rng.randrange(1, 8), y + rng.randrange(1, 8))
            for x, y in ((rng.randrange(-50, 50), rng.randrange(-50, 50)) for _ in range(25))
        )

        with _native_mode(monkeypatch, enabled=True):
            cpp_idx = RectangleIndex(indexed)
            assert cpp_idx._native is not None
        with _native_mode(monkeypatch, enabled=False):
            py_idx = RectangleIndex(indexed)
            assert py_idx._native is None

        for gap in (-2, -1, 0, 1, 5, 20):
            with _native_mode(monkeypatch, enabled=True):
                cpp_c = cpp_idx.conflicts(queries, clearance_nm=gap)
            with _native_mode(monkeypatch, enabled=False):
                py_c = py_idx.conflicts(queries, clearance_nm=gap)
            assert cpp_c == py_c

        for q in queries[:6]:
            for touch in (False, True):
                with _native_mode(monkeypatch, enabled=True):
                    cpp_hits = sorted(cpp_idx.intersecting(q, include_touching=touch))
                with _native_mode(monkeypatch, enabled=False):
                    py_hits = sorted(py_idx.intersecting(q, include_touching=touch))
                assert cpp_hits == py_hits


def test_rectangle_index_ab_touching_boundaries_and_bigints(monkeypatch):
    from clay.layout.spatial.collision_queries import RectangleIndex

    # Edge-touching, corner-touching, duplicate retention
    boxes = ((0, 0, 4, 4), (0, 0, 4, 4), (4, 0, 8, 4), (4, 4, 8, 8), (10, 10, 12, 12))
    q = (2, 2, 4, 4)
    with _native_mode(monkeypatch, enabled=True):
        idx_cpp = RectangleIndex(boxes)
        hits_strict_cpp = sorted(idx_cpp.intersecting(q, include_touching=False))
        hits_touch_cpp = sorted(idx_cpp.intersecting(q, include_touching=True))
    with _native_mode(monkeypatch, enabled=False):
        idx_py = RectangleIndex(boxes)
        hits_strict_py = sorted(idx_py.intersecting(q, include_touching=False))
        hits_touch_py = sorted(idx_py.intersecting(q, include_touching=True))
    assert hits_strict_cpp == hits_strict_py == [(0, 0, 4, 4), (0, 0, 4, 4)]
    assert hits_touch_cpp == hits_touch_py == [
        (0, 0, 4, 4), (0, 0, 4, 4), (4, 0, 8, 4), (4, 4, 8, 8)
    ]

    # Big integers beyond int64_t (fallback in both __init__ and queries)
    big_indexed = (((1 << 80), 0, (1 << 80) + 10, 10),)
    big_q = (((1 << 80) + 11, 0, (1 << 80) + 20, 10),)
    for enabled in (True, False):
        with _native_mode(monkeypatch, enabled=enabled):
            idx = RectangleIndex(big_indexed)
            assert not idx.conflicts(big_q, clearance_nm=1)
            assert idx.conflicts(big_q, clearance_nm=2)
            assert idx.intersecting(((1 << 80) + 10, 0, (1 << 80) + 12, 5), include_touching=True) == big_indexed
            assert idx.intersecting(((1 << 80) + 10, 0, (1 << 80) + 12, 5), include_touching=False) == ()

    # Normal indexed + >2^61 query box / huge clearance_nm
    small_indexed = ((0, 0, 5, 5),)
    with _native_mode(monkeypatch, enabled=True):
        idx = RectangleIndex(small_indexed)
        assert not idx.conflicts((((1 << 70), 0, (1 << 70) + 5, 5),), clearance_nm=1)
        assert idx.conflicts(((10, 0, 15, 5),), clearance_nm=(1 << 70))


@pytest.mark.parametrize(
    "bad_boxes,exc_type",
    [
        (((0, 0, 0, 2),), ValueError),
        (((3, 0, 2, 2),), ValueError),
        (((0, 5, 2, 5),), ValueError),
        (((0, 0, 2.0, 2),), TypeError),
        (((False, 0, 2, 2),), TypeError),
        (((0, 0, 2),), TypeError),
    ],
)
def test_rectangle_index_ab_invalid_inputs(monkeypatch, bad_boxes, exc_type):
    from clay.layout.spatial.collision_queries import RectangleIndex

    for enabled in (True, False):
        with _native_mode(monkeypatch, enabled=enabled):
            with pytest.raises(exc_type):
                RectangleIndex(bad_boxes)


def test_rectangle_index_cpp_checkpoint_cancellation_and_reuse(monkeypatch):
    from clay.layout.spatial.collision_queries import RectangleIndex

    class Aborted(BaseException):
        pass

    left = tuple((i * 4, 0, i * 4 + 1, 1) for i in range(4000))
    right = tuple((i * 4 + 2, 0, i * 4 + 3, 1) for i in range(4000))

    # Cancel during build
    build_calls = 0

    def cancel_build():
        nonlocal build_calls
        build_calls += 1
        if build_calls == 5:
            raise Aborted("build aborted")

    with _native_mode(monkeypatch, enabled=True):
        with pytest.raises(Aborted, match="build aborted"):
            RectangleIndex(left, checkpoint=cancel_build)
        assert build_calls == 5

        # Build cleanly, then cancel during conflicts, then reuse cleanly
        idx = RectangleIndex(left)
        q_calls = 0

        def cancel_query():
            nonlocal q_calls
            q_calls += 1
            if q_calls == 6:
                raise Aborted("query aborted")

        with pytest.raises(Aborted, match="query aborted"):
            idx.conflicts(right, clearance_nm=1, checkpoint=cancel_query)
        assert q_calls == 6
        assert not idx.conflicts(right, clearance_nm=1)
        assert idx.conflicts(right, clearance_nm=2)


# ============================================================================
# 4. full2d_pair_cover._exact_forbidden_union A/B differential tests
# ============================================================================


def test_exact_forbidden_union_ab_equivalence_and_no_klayout(monkeypatch):
    from clay.layout.place.full2d_pair_cover import _exact_forbidden_union

    # Dense grid with a hole
    dense = {
        (3 * x, 5 * y, 3 * x + 17, 5 * y + 13)
        for x in range(30)
        for y in range(20)
        if not (8 < x < 18 and 5 < y < 13)
    }

    # Verify C++ path does not import klayout
    orig_import = builtins.__import__

    def no_klayout_import(name, *args, **kwargs):
        assert not name.startswith("klayout"), "C++ exact_forbidden_union must not import klayout"
        return orig_import(name, *args, **kwargs)

    with _native_mode(monkeypatch, enabled=True):
        monkeypatch.setattr(builtins, "__import__", no_klayout_import)
        cpp_dense = _exact_forbidden_union(dense)
    monkeypatch.setattr(builtins, "__import__", orig_import)

    with _native_mode(monkeypatch, enabled=False):
        py_dense = _exact_forbidden_union(dense)

    assert _grid_points(cpp_dense) == _grid_points(py_dense) == _grid_points(dense)
    assert sum((c - a) * (d - b) for a, b, c, d in cpp_dense) == len(_grid_points(dense))
    assert len(cpp_dense) < len(dense)

    # Exact edge cases: empty, single, crossing non-shrinking, 32-bit boundary, span > 2^31-1, >int64
    edge_cases = [
        set(),
        {(0, 0, 2, 2)},
        {(-2, -1, 1, 0), (-1, -2, 0, 1)},  # non-shrinking -> canonical
        {(-(2**31) - 1, -1, 0, 2), (-(2**31), -1, 1, 2)},  # out of int32
        {(-2_000_000_000, -1, 1, 2), (0, -1, 2_000_000_000, 2)},  # span > 2^31 - 1
        {(2**80, 0, 2**80 + 2, 2), (2**80 + 1, 0, 2**80 + 3, 2)},  # > int64 fallback
    ]
    for boxes in edge_cases:
        with _native_mode(monkeypatch, enabled=True):
            cpp_r = _exact_forbidden_union(boxes)
        with _native_mode(monkeypatch, enabled=False):
            py_r = _exact_forbidden_union(boxes)
        assert cpp_r == py_r


# ============================================================================
# 5. seqpair_placement A/B differential tests
# ============================================================================


def test_longest_path_positions_ab_dags_and_cycles(monkeypatch):
    from clay.layout.seqpair_placement import _longest_path_positions

    # Empty, isolated, chain, parallel edges
    for nodes, edges, weights in [
        ([], [], {}),
        (["A", "B"], [], {}),
        (["A", "B", "C"], [("A", "B"), ("B", "C"), ("A", "C")],
         {("A", "B"): 12.5, ("B", "C"): 7.25, ("A", "C"): 15.0}),
        (["A", "B"], [("A", "B"), ("A", "B")], {("A", "B"): 5.0}),
    ]:
        with _native_mode(monkeypatch, enabled=True):
            cpp_pos = _longest_path_positions(nodes, edges, weights)
        with _native_mode(monkeypatch, enabled=False):
            py_pos = _longest_path_positions(nodes, edges, weights)
        assert cpp_pos == py_pos

    # Random 150-node DAGs
    rng = random.Random(1337)
    for _ in range(20):
        n = rng.randint(10, 150)
        nodes = [f"M{i}" for i in range(n)]
        edges = []
        weights = {}
        for i in range(n):
            for j in range(i + 1, min(n, i + 12)):
                if rng.random() < 0.35:
                    edges.append((nodes[i], nodes[j]))
                    weights[(nodes[i], nodes[j])] = rng.uniform(0.5, 5000.0)
        rng.shuffle(edges)
        with _native_mode(monkeypatch, enabled=True):
            cpp_pos = _longest_path_positions(nodes, edges, weights)
        with _native_mode(monkeypatch, enabled=False):
            py_pos = _longest_path_positions(nodes, edges, weights)
        assert cpp_pos == py_pos

    # Cycles: self-loop, 2-cycle, 3-cycle with acyclic prefix
    cycle_cases = [
        (["A"], [("A", "A")], {("A", "A"): 1.0}),
        (["A", "B"], [("A", "B"), ("B", "A")], {("A", "B"): 1.0, ("B", "A"): 1.0}),
        (["S", "A", "B", "C"], [("S", "A"), ("A", "B"), ("B", "C"), ("C", "A")],
         {("S", "A"): 1.0, ("A", "B"): 2.0, ("B", "C"): 3.0, ("C", "A"): 4.0}),
    ]
    for nodes, edges, weights in cycle_cases:
        with _native_mode(monkeypatch, enabled=True):
            with pytest.raises(ValueError) as cpp_exc:
                _longest_path_positions(nodes, edges, weights)
        with _native_mode(monkeypatch, enabled=False):
            with pytest.raises(ValueError) as py_exc:
                _longest_path_positions(nodes, edges, weights)
        assert str(cpp_exc.value) == str(py_exc.value)


def test_pack_band_sp_and_searches_ab_exact_parity(monkeypatch):
    from clay.layout.seqpair_placement import (
        _pack_band_sp,
        _search_band_layout,
        _search_block_layout,
    )

    rng = random.Random(2026)

    # _pack_band_sp differential tests
    for n in (1, 2, 4, 6, 10):
        for _ in range(6):
            gp = list(range(n))
            gm = list(range(n))
            rng.shuffle(gp)
            rng.shuffle(gm)
            blocks = [(rng.uniform(500.0, 12000.0), rng.uniform(500.0, 9000.0)) for _ in range(n)]
            gap = rng.choice([0.0, 460.0, 1400.0])
            with _native_mode(monkeypatch, enabled=True):
                cpp_r = _pack_band_sp(gp, gm, blocks, gap)
            with _native_mode(monkeypatch, enabled=False):
                py_r = _pack_band_sp(gp, gm, blocks, gap)
            assert cpp_r == py_r

    # _search_block_layout: both exhaustive (n <= 6) and hill-climb (n > 6)
    for n in (2, 3, 4, 5, 7, 9, 12):
        blocks = [(rng.uniform(1000.0, 15000.0), rng.uniform(1000.0, 10000.0)) for _ in range(n)]
        gap = 1400.0
        with _native_mode(monkeypatch, enabled=True):
            cpp_rel = _search_block_layout(blocks, gap, aspect_target=1.6)
        with _native_mode(monkeypatch, enabled=False):
            py_rel = _search_block_layout(blocks, gap, aspect_target=1.6)
        assert cpp_rel == py_rel

    # _search_band_layout (no dev_cost)
    row_seqs = [("A0", "A1"), ("B0",), ("C0", "C1", "C2"), ("D0", "D1")]
    half_w = {d: rng.uniform(400.0, 4000.0) for seq in row_seqs for d in seq}
    heights = {d: rng.uniform(800.0, 6000.0) for seq in row_seqs for d in seq}
    with _native_mode(monkeypatch, enabled=True):
        cpp_band = _search_band_layout(row_seqs, half_w, heights, 1400.0)
    with _native_mode(monkeypatch, enabled=False):
        py_band = _search_band_layout(row_seqs, half_w, heights, 1400.0)
    assert cpp_band == py_band


def test_search_band_dev_layout_ab_exact_parity(monkeypatch):
    from clay.layout.seqpair_placement import (
        _cost_from_dims,
        _longest_path_positions,
        _search_band_layout,
    )

    rng = random.Random(7777)
    for n_bands in (3, 4, 5, 8):
        row_seqs = []
        half_w = {}
        heights = {}
        for bi in range(n_bands):
            seq = tuple(f"B{bi}_D{di}" for di in range(rng.randint(1, 4)))
            row_seqs.append(seq)
            for d in seq:
                half_w[d] = rng.uniform(300.0, 5000.0)
                heights[d] = rng.uniform(600.0, 7000.0)
        dev_gap = 1400.0
        nodes = [d for seq in row_seqs for d in seq]
        base_left = [(u, v) for seq in row_seqs for u, v in zip(seq, seq[1:])]

        def _dev_cost(cand_rel):
            lo_edges = list(base_left)
            be_edges = []
            slo = set(lo_edges)
            sbe = set()
            for (ci, cj), crel in cand_rel.items():
                lb, hb = (ci, cj) if crel.startswith("i_") else (cj, ci)
                ax = "below" if "below" in crel else "left_of"
                for uu in row_seqs[lb]:
                    for vv in row_seqs[hb]:
                        if ax == "below":
                            if (uu, vv) not in sbe:
                                be_edges.append((uu, vv))
                                sbe.add((uu, vv))
                        else:
                            if (uu, vv) not in slo:
                                lo_edges.append((uu, vv))
                                slo.add((uu, vv))
            hw = {(u, v): half_w[u] + half_w[v] + dev_gap for (u, v) in lo_edges}
            vw = {(u, v): heights[u] / 2.0 + heights[v] / 2.0 + dev_gap for (u, v) in be_edges}
            xc = _longest_path_positions(nodes, lo_edges, hw)
            yc = _longest_path_positions(nodes, be_edges, vw)
            W = max(xc[d] + half_w[d] for d in nodes) - min(xc[d] - half_w[d] for d in nodes)
            H = max(yc[d] + heights[d] / 2.0 for d in nodes) - min(yc[d] - heights[d] / 2.0 for d in nodes)
            return _cost_from_dims(W, H, 1.6)

        with _native_mode(monkeypatch, enabled=True):
            mod = load_native("_place_cpp")
            row_half_ws = [[half_w[d] for d in seq] for seq in row_seqs]
            row_heights = [[heights[d] for d in seq] for seq in row_seqs]
            cpp_rel = mod.search_band_dev_layout_cpp(row_half_ws, row_heights, dev_gap, 1.6)
        with _native_mode(monkeypatch, enabled=False):
            py_rel = _search_band_layout(row_seqs, half_w, heights, dev_gap, dev_cost=_dev_cost)
        assert cpp_rel == py_rel


def test_place_seqpair_end_to_end_ab_parity(monkeypatch):
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.seqpair_placement import place_seqpair
    from clay.parser import parse_clay_file

    root = Path(__file__).resolve().parents[1]
    clay_path = root / "benchmarks/gf180mcu/two_stage_ota/source/design.clay"
    cir_path = None
    cf = parse_clay_file(str(clay_path))
    cd = next((c for c in cf.cell_defs if c.__dict__.get("_layout_annotations")), cf.cell_defs[0])
    intent = layout_intent_from_cell_def(cd)

    with _native_mode(monkeypatch, enabled=True):
        pl_cpp = place_seqpair(cd, intent, source_cir=cir_path)
    with _native_mode(monkeypatch, enabled=False):
        pl_py = place_seqpair(cd, intent, source_cir=cir_path)

    assert {k: v.bbox for k, v in pl_cpp.devices.items()} == {
        k: v.bbox for k, v in pl_py.devices.items()
    }
    assert pl_cpp.axis_x == pl_py.axis_x
    assert pl_cpp.symmetry_score == pl_py.symmetry_score
