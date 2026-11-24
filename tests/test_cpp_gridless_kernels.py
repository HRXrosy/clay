"""Differential and unit tests for C++20 Gridless PnR kernels (clay._native._gridless_cpp).

Every kernel is differentially validated against the pure-Python reference path
(``CLAY_DISABLE_CPP_PNR=1`` vs ``0``) for 100% bit/geometry/diagnostic equivalence.
"""
from __future__ import annotations

import itertools
import random
from types import MappingProxyType

import pytest

from clay._native import clear_native_cache, load_native
from clay.layout.route import engine_gridless as gridless
from clay.layout.route.axis_price_cache import AxisPriceCache
from clay.layout.route.engine import Obstacle, Pin
from clay.layout.route.engine_gridless import (
    GridlessEngine,
    _ExactGoalPotentialIndex,
    _RoutedNet,
    _SpatialIndex,
    _ViaSite,
    _WireSeg,
    _closed_rects_touch,
    _norm,
    _rects_clear,
    _sep,
)
from clay.layout.route.gridless_planar_search import _build_exact_2d_potential_fn
from clay.layout.route.region_event_geometry import iter_axis_cost_segments
from clay.layout.tech import LayoutTech


def _require_cpp():
    clear_native_cache()
    mod = load_native("_gridless_cpp")
    assert mod is not None, "_gridless_cpp extension must be built and loadable"
    return mod


# ---------------------------------------------------------------------------
# 1. Exact geometry primitives (L-inf + Euclidean diagonal + 128-bit)
# ---------------------------------------------------------------------------


def test_rects_clear_and_euclidean_diagonal_boundaries():
    cpp = _require_cpp()

    # Exact Euclidean diagonal boundaries:
    # 3-4-5 right triangle: dist == 5 -> clear for 5, not clear for 6.
    a = (0, 0, 10, 10)
    b = (13, 14, 23, 24)  # dx=3, dy=4 -> dist=5
    assert cpp.rects_clear_euclidean_cpp(a, b, 5) is True
    assert cpp.rects_clear_euclidean_cpp(a, b, 6) is False
    # L-inf: max(3, 4) = 4 -> clear for 4, not clear for 5!
    assert cpp.rects_clear_cpp(a, b, 4) is True
    assert cpp.rects_clear_cpp(a, b, 5) is False
    assert cpp.rects_clear_cpp(a, b, 4) == _rects_clear(a, b, 4)
    assert cpp.rects_clear_cpp(a, b, 5) == _rects_clear(a, b, 5)

    # Classic 140nm PDK diagonal boundary:
    # 99^2 + 99^2 = 19602 >= 19600 = 140^2 -> Euclidean clear, L-inf NOT clear.
    c1 = (100 + 99, 100 + 99, 300, 300)
    base = (0, 0, 100, 100)
    assert cpp.rects_clear_euclidean_cpp(base, c1, 140) is True
    assert cpp.rects_clear_cpp(base, c1, 140) is False
    assert _rects_clear(base, c1, 140) is False

    # 98^2 + 99^2 = 19405 < 19600 = 140^2 -> NOT Euclidean clear.
    c2 = (100 + 98, 100 + 99, 300, 300)
    assert cpp.rects_clear_euclidean_cpp(base, c2, 140) is False

    # 139^2 + 17^2 = 19321 + 289 = 19610 >= 19600 -> clear.
    # 139^2 + 16^2 = 19321 + 256 = 19577 < 19600 -> not clear.
    assert cpp.rects_clear_euclidean_cpp(base, (239, 117, 300, 200), 140) is True
    assert cpp.rects_clear_euclidean_cpp(base, (239, 116, 300, 200), 140) is False

    # Touching / overlapping:
    touch = (100, 100, 200, 200)
    assert cpp.closed_rects_touch_cpp(base, touch) is True
    assert cpp.rects_clear_euclidean_cpp(base, touch, 1) is False
    assert cpp.rects_clear_cpp(base, touch, 1) is False
    assert cpp.rects_clear_cpp(base, touch, 0) is True
    assert _rects_clear(base, touch, 0) is True
    assert cpp.rects_clear_euclidean_cpp(base, touch, 0) is True  # edges touch, no area overlap
    assert cpp.rects_clear_euclidean_cpp(base, (99, 99, 200, 200), 0) is False

    # Unnormalized input rectangles + negative coordinates
    rng = random.Random(20261122)
    for _ in range(500):
        ra = tuple(rng.randint(-5000, 5000) for _ in range(4))
        rb = tuple(rng.randint(-5000, 5000) for _ in range(4))
        sp = rng.choice([-10, 0, 1, 50, 140, 230, 500])
        assert cpp.sep_cpp(ra, rb) == _sep(_norm(ra), _norm(rb))
        assert cpp.closed_rects_touch_cpp(ra, rb) == _closed_rects_touch(ra, rb)
        assert cpp.rects_clear_cpp(ra, rb, sp) == _rects_clear(ra, rb, sp)
        sx, sy = _sep(_norm(ra), _norm(rb))
        if sp > 0:
            assert cpp.rects_clear_euclidean_cpp(ra, rb, sp) == (sx * sx + sy * sy >= sp * sp)

    # Large > 2^31 128-bit exactness check (must not lose precision in double!)
    sp_big = (1 << 32) + 7  # 4294967303
    # dx = 3037000505, dy = 3037000505 -> test exact square comparison
    dx_big = 3037000505
    dy_big = 3037000504
    expected = (dx_big * dx_big + dy_big * dy_big) >= (sp_big * sp_big)
    assert cpp.rects_clear_euclidean_cpp(
        (0, 0, 10, 10), (10 + dx_big, 10 + dy_big, 20 + dx_big, 20 + dy_big), sp_big
    ) is expected
    dy_big2 = 3037000505
    expected2 = (dx_big * dx_big + dy_big2 * dy_big2) >= (sp_big * sp_big)
    assert cpp.rects_clear_euclidean_cpp(
        (0, 0, 10, 10), (10 + dx_big, 10 + dy_big2, 20 + dx_big, 20 + dy_big2), sp_big
    ) is expected2


# ---------------------------------------------------------------------------
# 2. _SpatialIndex & NativeSpatialIndex differential parity
# ---------------------------------------------------------------------------


def _build_pair_indexes(entries, cell=6000, monkeypatch=None):
    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    idx_cpp = _SpatialIndex(cell=cell)
    assert idx_cpp._native is not None

    monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    clear_native_cache()
    idx_py = _SpatialIndex(cell=cell)
    assert idx_py._native is None

    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()

    for owner, rect in entries:
        idx_cpp.add(owner, rect)
        idx_py.add(owner, rect)
    return idx_cpp, idx_py


def test_spatial_index_multi_owner_dedup_negative_coords_and_sparse(monkeypatch):
    shared = (-6001, -6000, 6000, 6001)
    same_coords_new_obj = tuple([-6001, -6000, 6000, 6001])
    entries = [
        ("netA", shared),
        ("netB", shared),          # different owner, SAME rect object -> MUST stay distinct
        ("netA", shared),          # duplicate (owner, id(rect)) -> MUST be deduped
        ("netA", same_coords_new_obj),  # same coords, DIFFERENT rect object -> distinct claim
        (None, (-1, -1, 0, 0)),
        ("netC", (5999, 5999, 6000, 6000)),
        ("netD", (12000, -12000, 6000, -6000)),  # unnormalized rect!
    ]
    idx_cpp, idx_py = _build_pair_indexes(entries, cell=6000, monkeypatch=monkeypatch)

    probes = [
        shared,
        (-10000, -10000, 10000, 10000),
        (-1, -1, 0, 0),
        (-6001, -6001, -6000, -6000),
        (5999, 5999, 6001, 6001),
        (7000, -9000, 11000, -7000),
        (50000, 50000, 51000, 51000),
    ]
    for q in probes:
        assert list(idx_cpp.query(q)) == list(idx_py.query(q)), f"mismatch on query {q}"

    # Verify lazy buckets export preserves exact dict type, keys, and object identities
    b_cpp = idx_cpp.buckets
    assert type(b_cpp) is dict
    assert b_cpp == idx_py.buckets
    assert idx_cpp.buckets is b_cpp
    for k, rows_cpp in b_cpp.items():
        rows_py = idx_py.buckets[k]
        assert len(rows_cpp) == len(rows_py)
        for (o1, r1), (o2, r2) in zip(rows_cpp, rows_py):
            assert o1 == o2 and r1 is r2

    # Verify MappingProxyType sealing keeps native queries working while blocking add()
    sealed = MappingProxyType({k: tuple(v) for k, v in b_cpp.items()})
    idx_cpp.buckets = sealed
    assert idx_cpp.buckets is sealed
    assert idx_cpp._native is not None
    assert list(idx_cpp.query(shared)) == list(idx_py.query(shared))
    with pytest.raises(AttributeError):
        idx_cpp.add("netZ", (0, 0, 10, 10))
    idx_cpp.buckets = b_cpp
    assert idx_cpp.buckets is b_cpp
    assert idx_cpp._native is not None

    # Sparse index (> 65536 bucket span triggers hash-table fallback cleanly)
    sparse_entries = [
        ("a", (0, 0, 100, 100)),
        ("b", (2_000_000_000, 2_000_000_000, 2_000_000_100, 2_000_000_100)),
    ]
    s_cpp, s_py = _build_pair_indexes(sparse_entries, cell=6000, monkeypatch=monkeypatch)
    assert list(s_cpp.query((0, 0, 50, 50))) == list(s_py.query((0, 0, 50, 50)))
    assert list(s_cpp.query((1_999_999_950, 1_999_999_950, 2_000_000_050, 2_000_000_050))) == list(
        s_py.query((1_999_999_950, 1_999_999_950, 2_000_000_050, 2_000_000_050))
    )


def test_spatial_index_metal_and_cut_scans_match_python(monkeypatch):
    rng = random.Random(20261123)
    owners = [None, "VDD", "GND", "sig0", "sig1", "sig2"]
    entries = []
    for _ in range(150):
        x0 = rng.randint(-4000, 4000)
        y0 = rng.randint(-4000, 4000)
        w = rng.randint(50, 900)
        h = rng.randint(50, 900)
        entries.append((rng.choice(owners), (x0, y0, x0 + w, y0 + h)))
    # Add exact identical cuts to test present_same
    entries.append(("VDD", (100, 100, 300, 300)))
    entries.append(("GND", (100, 100, 300, 300)))
    entries.append(("VDD", (400, 100, 600, 300)))

    idx_cpp, idx_py = _build_pair_indexes(entries, cell=1000, monkeypatch=monkeypatch)

    for _ in range(80):
        qx0 = rng.randint(-4200, 4200)
        qy0 = rng.randint(-4200, 4200)
        qx1 = qx0 + rng.randint(40, 600)
        qy1 = qy0 + rng.randint(40, 600)
        sp = rng.choice([0, 140, 230, 350])
        own = rng.choice(owners)
        excl = rng.choice([None, "sig0"])

        assert idx_cpp.scan_metal_conflicts(qx0, qy0, qx1, qy1, sp, own, excl) == (
            idx_py.scan_metal_conflicts(qx0, qy0, qx1, qy1, sp, own, excl)
        )
        assert idx_cpp.scan_cut_conflicts(qx0, qy0, qx1, qy1, max(1, sp), own) == (
            idx_py.scan_cut_conflicts(qx0, qy0, qx1, qy1, max(1, sp), own)
        )

    # Exact present_same check
    ps_c, fc_c, sv_c = idx_cpp.scan_cut_conflicts(100, 100, 300, 300, 200, "VDD")
    ps_p, fc_p, sv_p = idx_py.scan_cut_conflicts(100, 100, 300, 300, 200, "VDD")
    assert (ps_c, fc_c, sv_c) == (ps_p, fc_p, sv_p)
    assert ps_c >= 1
    assert ("GND", (100, 100, 300, 300)) in fc_c
    assert ("VDD", (400, 100, 600, 300)) in sv_c


# ---------------------------------------------------------------------------
# 3. batch_filter_via_sites_cpp differential parity with full GridlessEngine
# ---------------------------------------------------------------------------


def _make_rich_escape_world(disable_cpp: bool, monkeypatch):
    if disable_cpp:
        monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    else:
        monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()

    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))
    from clay.layout.route.engine_maze import _Tech
    via1 = tuple(_Tech(tech).via_key_for(0))

    obstacles = [
        # Owned GND component with a bridge and a nearby unbridged same-net finger
        Obstacle(0.0, 0.0, 6.0, 0.8, m1, owner_net="GND", kind="device_terminal",
                 terminal_conductor_ref="tc:gnd_bar"),
        Obstacle(0.0, 0.95, 1.5, 1.5, m1, owner_net="GND", kind="device_terminal",
                 terminal_conductor_ref="tc:gnd_unbridged"),
        Obstacle(2.0, 0.8, 2.6, 1.5, m1, owner_net="GND", kind="device_terminal",
                 terminal_conductor_ref="tc:gnd_bridge"),
        Obstacle(2.6, 0.95, 3.8, 1.5, m1, owner_net="GND", kind="device_terminal",
                 terminal_conductor_ref="tc:gnd_bridged"),
        # Foreign M1 & M2 blockers
        Obstacle(0.8, -0.8, 1.8, -0.1, m1, owner_net="VDD", kind="device_terminal",
                 terminal_conductor_ref="tc:vdd_m1"),
        Obstacle(1.0, -0.5, 2.5, 1.2, m2, owner_net="sig_a", kind="device_terminal",
                 terminal_conductor_ref="tc:siga_m2"),
        # Fixed cut
        Obstacle(3.2, 0.2, 3.46, 0.46, via1, owner_net="VDD", kind="device_cut",
                 terminal_conductor_ref="tc:vdd_cut"),
    ]
    engine = GridlessEngine(time_budget_sec=30.0, max_iters=1)
    out = engine.route({}, obstacles, [], tech, resource_plan=None)
    assert out.success

    # Commit foreign wires + via pads + cuts + attempt cuts to exercise ALL rejection categories
    committed = {}
    engine._commit(
        engine._book,
        committed,
        _RoutedNet(
            name="sig_b",
            segs=[_WireSeg(1, 4000, 0, 4000, 1200)],
            vias=[_ViaSite(4800, 400, 0, 0)],
            connected=True,
        ),
    )
    engine._rebuild_metal_index()
    ext = engine._via_cut_extent(0, None)
    engine._attempt_cuts = {0: [(5300 - ext, 400 - ext, 5300 + ext, 400 + ext)]}
    engine._search_forbidden_via_sites = {(500, 400, 0, 0)}
    engine._hard_rejections = {}
    engine._hard_rejection_samples = {}
    engine._hard_rejection_generation = 0
    return engine


def test_owned_component_escape_cpp_matches_python_bit_for_bit(monkeypatch):
    eng_cpp = _make_rich_escape_world(False, monkeypatch)
    eng_py = _make_rich_escape_world(True, monkeypatch)

    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    states_cpp = eng_cpp._owned_component_escape_states("GND", 200, 200, 0)
    rej_cpp = dict(eng_cpp._hard_rejections)
    samp_cpp = dict(eng_cpp._hard_rejection_samples)
    gen_cpp = eng_cpp._hard_rejection_generation
    ev_cpp = eng_cpp._hard_rejection_evidence()
    sev_cpp = eng_cpp._hard_rejection_sample_evidence()
    det_cpp = dict(eng_cpp._route_failure_details)

    monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    clear_native_cache()
    states_py = eng_py._owned_component_escape_states("GND", 200, 200, 0)
    rej_py = dict(eng_py._hard_rejections)
    samp_py = dict(eng_py._hard_rejection_samples)
    gen_py = eng_py._hard_rejection_generation
    ev_py = eng_py._hard_rejection_evidence()
    sev_py = eng_py._hard_rejection_sample_evidence()
    det_py = dict(eng_py._route_failure_details)

    assert states_cpp == states_py
    assert len(states_cpp) > 0
    assert rej_cpp == rej_py
    assert samp_cpp == samp_py
    assert gen_cpp == sum(rej_cpp.values())
    assert gen_cpp >= gen_py > 0
    assert ev_cpp == ev_py
    assert sev_cpp == sev_py
    assert det_cpp == det_py
    # Confirm multiple distinct rejection categories were exercised!
    cats = {cat for cat, _owner in rej_cpp}
    assert {"fixed_metal", "fixed_cut", "committed_metal_contact", "committed_via_pad_spacing"}.issubset(cats)


def test_monkeypatched_rects_clear_and_sep_are_respected(monkeypatch):
    eng = _make_rich_escape_world(False, monkeypatch)
    rc_calls = 0
    orig_rc = gridless._rects_clear

    def counted_rc(a, b, sp):
        nonlocal rc_calls
        rc_calls += 1
        return orig_rc(a, b, sp)

    monkeypatch.setattr(gridless, "_rects_clear", counted_rc)
    eng._clear_obstacles(0, (900, -200, 1200, 100), 230, "GND")
    assert rc_calls > 0

    sep_calls = 0
    orig_sep = gridless._sep

    def counted_sep(a, b):
        nonlocal sep_calls
        sep_calls += 1
        return orig_sep(a, b)

    monkeypatch.setattr(gridless, "_sep", counted_sep)
    eng._scan_committed_metal(1, (3900, 0, 4100, 500), "GND")
    assert sep_calls > 0


# ---------------------------------------------------------------------------
# 4. Native2DPotential, NativeLayerHmin & exact_distance_transform_cpp
# ---------------------------------------------------------------------------


def test_native_2d_potential_matches_kd_tree_small_and_large():
    cpp = _require_cpp()
    rng = random.Random(20261124)
    for n_pts in (1, 12, 48, 120, 300):
        rate = rng.choice([1, 51, 73, 125])
        projected = {}
        for _ in range(n_pts):
            gx = rng.randint(-15000, 15000)
            gy = rng.randint(-12000, 12000)
            c = rng.randint(0, 500000)
            projected[(gx, gy, 0)] = min(projected.get((gx, gy, 0), c), c)
        kd = _ExactGoalPotentialIndex(projected, step_cost=rate)
        nat = cpp.Native2DPotential(projected, rate)
        built = _build_exact_2d_potential_fn(projected, rate, kd)
        assert isinstance(built, cpp.Native2DPotential)

        xs = sorted({gx for gx, _, _ in projected})
        ys = sorted({gy for _, gy, _ in projected})
        probe_xs = [-50000, xs[0] - 1, xs[0], (xs[0] + xs[-1]) // 2, xs[-1], xs[-1] + 1, 50000]
        probe_xs += [rng.randint(xs[0] - 500, xs[-1] + 500) for _ in range(25)]
        probe_ys = [-50000, ys[0] - 1, ys[0], (ys[0] + ys[-1]) // 2, ys[-1], ys[-1] + 1, 50000]
        probe_ys += [rng.randint(ys[0] - 500, ys[-1] + 500) for _ in range(25)]
        for x in probe_xs:
            for y in probe_ys:
                expected = kd.value(x, y, 0)
                assert nat(x, y) == expected
                assert nat.value(x, y) == expected
                assert built(x, y) == expected


def test_exact_distance_transform_cpp_matches_python(monkeypatch):
    cpp = _require_cpp()
    rng = random.Random(20261125)
    xs = sorted(rng.sample(range(-10000, 10000), 35))
    ys = sorted(rng.sample(range(-8000, 8000), 27))
    layers = [0, 1, 2, 4]
    goals = {}
    for _ in range(18):
        g = (rng.choice(xs), rng.choice(ys), rng.choice(layers))
        c = rng.randint(0, 90000)
        goals[g] = min(goals.get(g, c), c)

    eng = GridlessEngine(time_budget_sec=30.0)
    eng._watchdog_deadline = None

    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    pot_cpp = eng._exact_multigoal_potential(xs, ys, layers, goals, "sig")

    monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    clear_native_cache()
    pot_py = eng._exact_multigoal_potential(xs, ys, layers, goals, "sig")

    assert pot_cpp == pot_py
    assert len(pot_cpp) == len(xs) * len(ys) * len(layers)

    with pytest.raises(ValueError, match="strictly increasing"):
        cpp.exact_distance_transform_cpp([10, 10], [0], [0], {(10, 0, 0): 0}, 1, 40000)


# ---------------------------------------------------------------------------
# 5. AxisPriceCache differential parity (charge order, refusal, resume)
# ---------------------------------------------------------------------------


def test_axis_price_cache_cpp_matches_python_charges_and_refusals(monkeypatch):
    xs = (-25, -10, 0, 7, 20, 45, 100)
    ys = (-15, -5, 4, 18, 30, 75)

    def make_price(log):
        def price(gi, ax, ay, bx, by):
            log.append((gi, ax, ay, bx, by))
            return gi * 17 + ax * 3 + ay * 5 + bx * 7 + by * 11 + abs(ax - bx) * 13 + abs(ay - by) * 19
        return price

    # 1) Exhaustive coordinate pairs & directions
    p_log_cpp, c_log_cpp = [], []
    p_log_py, c_log_py = [], []

    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    cache_cpp = AxisPriceCache(xs, ys, make_price(p_log_cpp), lambda k: c_log_cpp.append(k) or True)
    assert cache_cpp._native is not None

    monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    clear_native_cache()
    cache_py = AxisPriceCache(xs, ys, make_price(p_log_py), lambda k: c_log_py.append(k) or True)
    assert cache_py._native is None

    coords = (-30, -25, -12, -10, 0, 7, 15, 20, 45, 100, 120)
    for a, b in itertools.product(coords, repeat=2):
        for start, end in (((a, 4), (b, 4)), ((7, a), (7, b))):
            v_cpp = cache_cpp.run(1, start, end)
            v_py = cache_py.run(1, start, end)
            assert v_cpp == v_py
    assert p_log_cpp == p_log_py
    assert c_log_cpp == c_log_py

    # 2) Refused budget at every cutoff k = 0..12, then dynamic cache.charge replacement & resume
    for cutoff in range(13):
        monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()
        cnt_c, pl_c, cl_c = [0], [], []
        def ch_c(kind):
            cl_c.append(kind)
            cnt_c[0] += 1
            return cnt_c[0] <= cutoff
        cc = AxisPriceCache(xs, ys, make_price(pl_c), ch_c)

        monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        clear_native_cache()
        cnt_p, pl_p, cl_p = [0], [], []
        def ch_p(kind):
            cl_p.append(kind)
            cnt_p[0] += 1
            return cnt_p[0] <= cutoff
        cp = AxisPriceCache(xs, ys, make_price(pl_p), ch_p)

        r_c = cc.run(0, (-30, 0), (120, 0))
        r_p = cp.run(0, (-30, 0), (120, 0))
        assert r_c == r_p
        assert cl_c == cl_p
        assert pl_c == pl_p

        # Resume with unlimited charge
        cc.charge = lambda k: cl_c.append(k) or True
        cp.charge = lambda k: cl_p.append(k) or True
        assert cc.run(0, (-30, 0), (120, 0)) == cp.run(0, (-30, 0), (120, 0))
        assert cl_c == cl_p
        assert pl_c == pl_p


# ---------------------------------------------------------------------------
# 6. End-to-end multi-net GridlessEngine route differential parity
# ---------------------------------------------------------------------------


def test_end_to_end_gridless_route_bit_identical_cpp_vs_python(monkeypatch):
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))

    obstacles = [
        Obstacle(-0.3, -0.3, 0.3, 0.3, m1, owner_net="netA", kind="device_terminal", terminal_conductor_ref="tc:a0"),
        Obstacle(7.7, -0.3, 8.3, 0.3, m1, owner_net="netA", kind="device_terminal", terminal_conductor_ref="tc:a1"),
        Obstacle(-0.3, 3.7, 0.3, 4.3, m1, owner_net="netB", kind="device_terminal", terminal_conductor_ref="tc:b0"),
        Obstacle(7.7, 3.7, 8.3, 4.3, m1, owner_net="netB", kind="device_terminal", terminal_conductor_ref="tc:b1"),
        Obstacle(3.0, -1.0, 5.0, 5.0, m1, owner_net="blk", kind="device_terminal", terminal_conductor_ref="tc:blk"),
        Obstacle(3.5, 1.0, 4.5, 3.0, m2, owner_net="blk2", kind="device_terminal", terminal_conductor_ref="tc:blk2"),
    ]
    pins = {
        "netA": [
            Pin("netA", "M0", "d", (0.0, 0.0), m1, xy_nm=(0, 0)),
            Pin("netA", "M1", "d", (8.0, 0.0), m1, xy_nm=(8000, 0)),
        ],
        "netB": [
            Pin("netB", "M2", "s", (0.0, 4.0), m1, xy_nm=(0, 4000)),
            Pin("netB", "M3", "s", (8.0, 4.0), m1, xy_nm=(8000, 4000)),
        ],
    }

    results = []
    for disable in (False, True):
        if disable:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()
        eng = GridlessEngine(time_budget_sec=30.0, max_iters=3)
        out = eng.route(pins, obstacles, [], tech, resource_plan=None)
        assert out.success
        snapshot = {
            net: (tuple(r.segs), tuple(r.vias), r.connected)
            for net, r in sorted(eng._best_committed.items())
        }
        results.append(snapshot)

    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# 7. v2 kernels: throat events, batch via congestion, via-target projector, fast wire legal
# ---------------------------------------------------------------------------


def test_compute_throat_events_cpp_matches_python():
    cpp = _require_cpp()
    import math as _math

    eng = GridlessEngine(time_budget_sec=10.0)
    rng = random.Random(20261126)

    for grid in (1, 5, 10, 25):
        eng._manufacturing_grid_nm = grid
        for _ in range(12):
            x0 = rng.randint(-2000, 1000)
            y0 = rng.randint(-2000, 1000)
            x1 = x0 + rng.randint(100, 1600)
            y1 = y0 + rng.randint(100, 1600)
            bh = rng.choice([80, 150, 260, 420])
            min_w = rng.choice([120, 230, 280, 500])
            cx0, cy0, cx1, cy1 = x0 - bh, y0 - bh, x1 + bh, y1 + bh
            base_xs = sorted({rng.randint(cx0 - 100, cx1 + 100) for _ in range(18)})
            base_ys = sorted({rng.randint(cy0 - 100, cy1 + 100) for _ in range(18)})

            # Reference Python logic from gridless_owned_escape.py
            def py_axis_events(vals, dlo, dhi, alo, ahi):
                seeds = {alo - bh, alo + bh, ahi - bh, ahi + bh} | {int(v) for v in vals}
                exp = set()
                for s in sorted(seeds):
                    for d in (-1, 0, 1):
                        v = s + d
                        if dlo <= v <= dhi:
                            exp.add(v)
                return tuple(eng._on_manufacturing_grid_lines(exp, dlo, dhi, preserve_frontier_sides=True))

            pxs = py_axis_events(base_xs, cx0, cx1, x0, x1)
            pys = py_axis_events(base_ys, cy0, cy1, y0, y1)
            wsq = min_w * min_w

            def ov(c, lo, hi):
                return max(0, min(hi, c + bh) - max(lo, c - bh))

            def csqrt(val):
                r = _math.isqrt(max(0, int(val)))
                return r if r * r == val else r + 1

            th = set()
            for y in pys:
                rx = csqrt(wsq - ov(y, y0, y1) ** 2)
                if rx <= min(x1 - x0, 2 * bh):
                    for b in (x0 - bh + rx, x1 + bh - rx):
                        for d in (-1, 0, 1):
                            for gx in eng._on_manufacturing_grid_lines((b + d,), cx0, cx1, preserve_frontier_sides=True):
                                for gy in eng._on_manufacturing_grid_lines((y,), cy0, cy1, preserve_frontier_sides=True):
                                    th.add((gx, gy))
            for x in pxs:
                ry = csqrt(wsq - ov(x, x0, x1) ** 2)
                if ry <= min(y1 - y0, 2 * bh):
                    for b in (y0 - bh + ry, y1 + bh - ry):
                        for d in (-1, 0, 1):
                            for gx in eng._on_manufacturing_grid_lines((x,), cx0, cx1, preserve_frontier_sides=True):
                                for gy in eng._on_manufacturing_grid_lines((b + d,), cy0, cy1, preserve_frontier_sides=True):
                                    th.add((gx, gy))

            cxs, cys, cth = cpp.compute_throat_events_cpp(
                base_xs, base_ys, x0, y0, x1, y1, bh, min_w, grid
            )
            assert cxs == pxs
            assert cys == pys
            assert cth == tuple(sorted(th))


def test_batch_via_congestion_and_single_via_escape_match_python(monkeypatch):
    cpp = _require_cpp()
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))

    obstacles = [
        Obstacle(0.0, 0.0, 3.0, 1.2, m1, owner_net="GND", kind="device_terminal", terminal_conductor_ref="tc:g0"),
        Obstacle(3.2, 0.0, 5.0, 1.2, m1, owner_net="GND", kind="device_terminal", terminal_conductor_ref="tc:g1"),
        Obstacle(0.5, 1.45, 2.5, 2.2, m1, owner_net="foreignA", kind="device_terminal", terminal_conductor_ref="tc:f0"),
        Obstacle(1.0, -0.6, 2.2, -0.2, m2, owner_net="foreignB", kind="device_terminal", terminal_conductor_ref="tc:f1"),
    ]
    pins = {
        "GND": [Pin("GND", "M0", "s", (0.5, 0.5), m1, xy_nm=(500, 500))],
        "foreignA": [Pin("foreignA", "M1", "d", (1.0, 1.8), m1, xy_nm=(1000, 1800))],
    }

    def run_mode(disable_cpp: bool):
        if disable_cpp:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()
        eng = GridlessEngine(
            time_budget_sec=30.0,
            via_redundancy=2,
            allow_single_via_escape=True,
        )
        out = eng.route({}, obstacles, [], tech, resource_plan=None)
        assert out.success
        eng._pres_fac = 2.5
        eng.hist_fac = 1.75
        # Populate committed metal, committed cuts, attempt metal, and PathFinder history
        eng._metal_idx.setdefault(0, _SpatialIndex()).add("foreignA", (600, 200, 1400, 700))
        eng._metal_idx.setdefault(1, _SpatialIndex()).add("foreignB", (1200, 100, 2200, 900))
        eng._cut_idx.setdefault(0, _SpatialIndex()).add("foreignA", (900, 350, 1160, 610))
        eng._attempt_idx.setdefault(0, _SpatialIndex()).add("GND", (100, 950, 1500, 1250))
        rc = eng._res_cell
        eng._history = {
            (500 // rc, 500 // rc, 0): 1.5,
            (1000 // rc, 500 // rc, 1): 2.25,
            (1000 // rc, 500 // rc, eng._n_layers + 0): 3.0,
        }
        eng._watchdog_deadline = None

        # 1) Direct batch_via_congestion_cpp vs _via_congestion across 100 sites
        if not disable_cpp:
            test_pts = [(x, y) for x in range(-200, 3200, 250) for y in range(-200, 1400, 250)]
            hist_by_rgi = {}
            hfac = eng.hist_fac * 1000
            for (hcx, hcy, hrgi), hv in eng._history.items():
                if hv:
                    hist_by_rgi.setdefault(int(hrgi), []).append((int(hcx), int(hcy), int(round(hv * hfac))))
            for model_n in (None, 1):
                batched = eng._batch_via_congestion_for_sites(test_pts, 0, "GND", model_n, cpp, hist_by_rgi)
                direct = [eng._via_congestion(x, y, 0, "GND", model_n) for x, y in test_pts]
                assert batched == direct

        # 2) Full _compute_owned_component_escape_states with allow_single_via_escape=True
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        states = eng._compute_owned_component_escape_states("GND", 500, 500, 0)
        return (
            states,
            dict(eng._hard_rejections),
            dict(eng._hard_rejection_samples),
        )

    states_cpp, rej_cpp, samp_cpp = run_mode(False)
    states_py, rej_py, samp_py = run_mode(True)
    assert len(states_cpp) > 0
    assert states_cpp == states_py
    assert rej_cpp == rej_py
    assert samp_cpp == samp_py


def test_native_via_target_projector_matches_python():
    cpp = _require_cpp()
    from clay.layout.route.gridless_types import (
        _OwnedTerminalAtom,
        _PlanarEndpointDomain,
        _VIA_COST,
        _region_via_goal_waypoint,
    )
    from clay.layout.route.region_event_geometry import RegionGeometry, project_target

    rng = random.Random(20261127)
    grid = 5
    turn_cost = 1500
    rect_bucket_nm = 140
    source_rect = (0, 0, 200, 200)
    geom = RegionGeometry((source_rect,), ())

    # Build NativeLayerHmin
    pot0 = cpp.Native2DPotential({(500, 500, 0): 0, (800, 200, 0): 1200}, 51)
    pot1 = cpp.Native2DPotential({(500, 500, 1): 40000, (900, 800, 1): 500}, 1)
    hmin = cpp.NativeLayerHmin(
        [(0, 51, pot0, 0), (1, 1, pot1, _VIA_COST)],
        [(500, 500, 600, 600)],
        _VIA_COST,
        3,
        4096,
    )

    atom = _OwnedTerminalAtom(0, (400, 400, 600, 600), "sig", "device_terminal", "tc:goal")
    target_domain = _PlanarEndpointDomain(
        node=(500, 500, 0),
        grid=grid,
        singleton=None,
        regions=((atom, (400, 400, 600, 600)),),
        excluded=frozenset({(450, 450, 0, 0, 0, 0, 0, 0)}),
    )
    goal_set = {(500, 500, 0), (150, 250, 0), (700, 300, 1)}

    other_idx = _SpatialIndex(cell=20)
    for rid in range(15):
        x0 = rng.randint(0, 150)
        y0 = rng.randint(0, 150)
        other_idx.add(rid, (x0, y0, x0 + rng.randint(10, 60), y0 + rng.randint(10, 60)))

    clipped = tuple(
        (x0, y0, x0 + rng.randint(1, 30), y0 + rng.randint(1, 30))
        for x0, y0 in [(rng.randint(0, 169), rng.randint(0, 169)) for _ in range(40)]
    )

    for neg_dedup in (False, True):
        proj = cpp.NativeViaTargetProjector(
            goal_set=goal_set,
            target_node=target_domain.node,
            target_singleton_none=True,
            target_regions=tuple(r for _, r in target_domain.regions),
            target_excluded_xy=((450, 450),),
            hmin=hmin,
            grid=grid,
            turn_cost=turn_cost,
            via_cost=_VIA_COST,
            rect_bucket_nm=rect_bucket_nm,
            neg_bucket_dedup=neg_dedup,
            via_succ_dir_1=0,
            via_succ_dir_2=0,
        )
        for ul, other in ((0, 1), (1, 0)):
            for px, py in ((10, 10), (90, 90), (100, 100), (180, 20), (30, 50)):
                for check_sat in (False, True):
                    # Reference Python loop
                    has_cross = False
                    seen_paths = set()
                    ends_seen = {(px, py)}
                    py_recs = []
                    for lx, ly, hx, hy in clipped:
                        if lx <= px < hx and ly <= py < hy:
                            has_cross = True
                            continue
                        ex_g = lx if px < lx else (hx - 1 if px >= hx else px)
                        ey_g = ly if py < ly else (hy - 1 if py >= hy else py)
                        if (ex_g, ey_g) in ends_seen:
                            continue
                        ends_seen.add((ex_g, ey_g))
                        for path in project_target(geom, 0, (px, py), (lx, ly, hx, hy)):
                            if len(path) == 1:
                                has_cross = True
                            elif path not in seen_paths:
                                seen_paths.add(path)
                                epx, epy = path[-1]
                                ex, ey = epx * grid, epy * grid
                                end3 = (ex, ey, ul)
                                if neg_dedup:
                                    gw = _region_via_goal_waypoint(
                                        path, grid=grid, layer=ul, goal_set=goal_set, target_domain=target_domain
                                    )
                                else:
                                    is_ep = end3 in goal_set or (ul == target_domain.node[2] and bool(target_domain.owning_atoms(end3)))
                                    gw = len(path) - 1 if is_ep else None
                                is_eg = gw == len(path) - 1
                                owners = (
                                    tuple(int(o) for o, _ in other_idx.query((epx, epy, epx, epy)))
                                    if gw is None and check_sat
                                    else ()
                                )
                                extra_turn = turn_cost if len(path) > 2 else 0
                                if gw is not None and not is_eg:
                                    tail_lb = None
                                elif is_eg:
                                    tail_lb = hmin(ex, ey, ul)
                                else:
                                    tail_lb = _VIA_COST + hmin(ex, ey, other)
                                py_recs.append((
                                    path, ex, ey, path[1][1] * grid, gw, is_eg, 0,
                                    owners, ex // rect_bucket_nm, ey // rect_bucket_nm, extra_turn, tail_lb,
                                ))

                    c_cross, c_recs = proj.project(
                        ul, other, px, py,
                        source_rect[0], source_rect[1], source_rect[2], source_rect[3],
                        clipped,
                        other_idx._native if check_sat else None,
                    )
                    assert c_cross == has_cross
                    assert c_recs == tuple(py_recs)


def test_check_wire_legal_fast_cpp_matches_python(monkeypatch):
    cpp = _require_cpp()
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    obstacles = [
        Obstacle(0.0, 0.0, 0.5, 0.5, m1, owner_net="sig", kind="device_terminal", terminal_conductor_ref="tc:s0"),
        Obstacle(2.0, 0.0, 2.5, 0.5, m1, owner_net="blk", kind="device_terminal", terminal_conductor_ref="tc:b0"),
    ]
    pins = {"sig": [Pin("sig", "M0", "d", (0.2, 0.2), m1, xy_nm=(200, 200))]}

    def eval_wires(disable_cpp: bool):
        if disable_cpp:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()
        eng = GridlessEngine(time_budget_sec=10.0)
        out = eng.route({}, obstacles, [], tech, resource_plan=None)
        assert out.success
        eng._metal_idx.setdefault(0, _SpatialIndex()).add("other", (4000, 0, 4500, 500))
        eng._pres_fac = 2.0
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        edges = [
            (0, 0, 2000, 1000, 2000),      # clean legal
            (0, 0, 200, 2200, 200),        # hits foreign obstacle blk
            (0, 3500, 200, 4600, 200),     # hits foreign committed metal contact
            (0, 3500, 620, 4600, 620),     # enters foreign committed metal spacing band
            (0, 4700, 680, 5200, 680),     # diagonal corner clear of sp, check present
        ]
        res = []
        for gi, ax, ay, bx, by in edges:
            eng._active_static_metal_scans = {}
            ok = eng._wire_legal_single(gi, ax, ay, bx, by, "sig")
            cong = eng._edge_congestion(gi, ax, ay, bx, by, "sig")
            res.append((ok, cong))
        return res, dict(eng._hard_rejections), dict(eng._hard_rejection_samples)

    res_cpp, rej_cpp, samp_cpp = eval_wires(False)
    res_py, rej_py, samp_py = eval_wires(True)
    assert res_cpp == res_py
    assert rej_cpp == rej_py
    assert samp_cpp == samp_py


def test_spatial_index_deepcopy_and_pickle_roundtrip():
    import copy
    import pickle
    from types import MappingProxyType

    _require_cpp()
    clear_native_cache()

    idx = _SpatialIndex(cell=1000)
    idx.add("net_a", (0, 0, 500, 500))
    idx.add("net_b", (400, 400, 1200, 1200))
    idx.add(None, (-500, -500, -100, -100))

    # 1) Native deepcopy preserves independence and native acceleration
    dc = copy.deepcopy(idx)
    assert dc._native is not None
    assert dc.query((0, 0, 600, 600)) == idx.query((0, 0, 600, 600))
    dc.add("net_c", (100, 100, 200, 200))
    assert len(idx.query((0, 0, 600, 600))) == 2
    assert len(dc.query((0, 0, 600, 600))) == 3

    # 2) Pickle round-trip produces bit-identical buckets and rehydrates native
    pk = pickle.loads(pickle.dumps(idx))
    assert pk._native is not None
    assert pk.cell == idx.cell
    assert pk.buckets == idx.buckets
    assert pk.query((-600, -600, 1500, 1500)) == idx.query((-600, -600, 1500, 1500))

    # 3) Sealed MappingProxyType buckets deepcopy + pickle cleanly
    sealed = _SpatialIndex(cell=1000)
    sealed.add("net_a", (0, 0, 500, 500))
    sealed.buckets = MappingProxyType(sealed.buckets)
    sealed_dc = copy.deepcopy(sealed)
    assert list(sealed_dc.query((0, 0, 500, 500))) == list(sealed.query((0, 0, 500, 500)))
    sealed_pk = pickle.loads(pickle.dumps(sealed))
    assert sealed_pk._native is not None
    assert list(sealed_pk.query((0, 0, 500, 500))) == list(sealed.query((0, 0, 500, 500)))


def test_native_history_grid_sat_sparse_and_subclass_guard(monkeypatch):
    cpp = _require_cpp()
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    seed_obs = [Obstacle(0.0, 0.0, 0.5, 0.5, m1, owner_net="seed", kind="device_terminal", terminal_conductor_ref="tc:seed")]

    eng = GridlessEngine(time_budget_sec=10.0)
    out = eng.route({}, seed_obs, [], tech, resource_plan=None)
    assert out.success
    rc = eng._res_cell
    eng.hist_fac = 1.25

    # Dense grid with negative, zero, positive, and round-half-to-even values
    dense_hist = {
        (-3, -2, 0): 0.002,
        (-1, 0, 0): 0.004,
        (0, 0, 0): 0.5,
        (1, 0, 0): 1.5,
        (2, 1, 0): 2.5,
        (5, 3, 1): 3.75,
        (0, 0, eng._n_layers + 0): 1.25,
    }
    hgrid = cpp.NativeHistoryGrid(dense_hist, eng.hist_fac, 1000)
    eng._history = dense_hist

    test_boxes = [
        (-4 * rc, -3 * rc, 4 * rc, 3 * rc),
        (-rc, 0, 0, rc),
        (0, 0, 0, 0),                                   # degenerate on boundary
        (rc // 2, rc // 2, rc // 2, rc // 2),           # degenerate inside cell
        (-rc // 2, -rc // 2, -rc // 2, -rc // 2),       # negative degenerate inside cell
        (rc, 0, rc, rc),                                # zero-width on boundary
        (rc // 2, 0, 2 * rc + 1, rc + 1),
        (100 * rc, 100 * rc, 105 * rc, 105 * rc),       # disjoint outside bbox
    ]
    for rgi in (0, 1, 2, eng._n_layers + 0):
        for bx in test_boxes:
            eng._active_history_grid = None
            py_val = eng._history_congestion_units(rgi, bx)
            cpp_val = hgrid.eval_units(rgi, bx[0], bx[1], bx[2], bx[3], rc)
            assert cpp_val == py_val, f"rgi={rgi} box={bx}: {cpp_val} != {py_val}"

    # Sparse fallback (> 4096 cell span)
    sparse_hist = {
        (0, 0, 0): 1.5,
        (10000, 10000, 0): 2.25,
        (-5000, 200, 0): 3.0,
    }
    sgrid = cpp.NativeHistoryGrid(sparse_hist, eng.hist_fac, 1000)
    eng._history = sparse_hist
    for bx in (
        (-5002 * rc, 198 * rc, -4998 * rc, 202 * rc),
        (-2 * rc, -2 * rc, 2 * rc, 2 * rc),
        (9999 * rc, 9999 * rc, 10002 * rc, 10002 * rc),
    ):
        assert sgrid.eval_units(0, bx[0], bx[1], bx[2], bx[3], rc) == eng._history_congestion_units(0, bx)
    # Huge span query directly on sparse NativeHistoryGrid (instant in C++: O(|sparse|))
    assert sgrid.eval_units(0, -10000 * rc, -10000 * rc, 20000 * rc, 20000 * rc, rc) == (
        round(1.5 * eng.hist_fac * 1000) + round(2.25 * eng.hist_fac * 1000) + round(3.0 * eng.hist_fac * 1000)
    )

    # Subclass guard: _NoGlobalScan(dict) MUST NOT enter NativeHistoryGrid
    class _TrackedDict(dict):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.gets = 0

        def get(self, k, default=None):
            self.gets += 1
            return super().get(k, default)

    tracked = _TrackedDict(dense_hist)
    eng._history = tracked
    eng._active_history_grid = hgrid
    val = eng._history_congestion_units(0, (0, 0, rc, rc))
    assert tracked.gets > 0
    assert val == hgrid.eval_units(0, 0, 0, rc, rc, rc)
    eng._active_history_grid = None


def test_check_via_legal_fast_and_box_congestion_match_python(monkeypatch):
    cpp = _require_cpp()
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))

    obstacles = [
        Obstacle(0.0, 0.0, 1.0, 1.0, m1, owner_net="sig", kind="device_terminal", terminal_conductor_ref="tc:s0"),
        Obstacle(3.0, 0.0, 4.0, 1.0, m1, owner_net="blk", kind="device_terminal", terminal_conductor_ref="tc:b0"),
        Obstacle(6.0, 0.0, 7.0, 1.0, m2, owner_net="blk2", kind="device_terminal", terminal_conductor_ref="tc:b1"),
    ]

    def run_mode(disable_cpp: bool):
        if disable_cpp:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()
        eng = GridlessEngine(time_budget_sec=10.0, via_redundancy=2, allow_single_via_escape=True)
        out = eng.route({}, obstacles, [], tech, resource_plan=None)
        assert out.success
        eng._metal_idx.setdefault(0, _SpatialIndex()).add("other", (8000, 0, 8600, 600))
        eng._metal_idx.setdefault(1, _SpatialIndex()).add("other", (9500, 0, 10200, 600))
        eng._cut_idx.setdefault(0, _SpatialIndex()).add("other", (11000, 200, 11260, 460))
        eng._attempt_cuts.setdefault(0, []).append((12500, 200, 12760, 460))
        eng._pres_fac = 2.0
        eng.hist_fac = 1.5
        rc = eng._res_cell
        eng._history = {(5000 // rc, 500 // rc, 0): 2.0, (5000 // rc, 500 // rc, eng._n_layers): 1.5}
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0

        pts = [
            (5000, 500, 0, None),   # clean legal
            (5000, 500, 0, 1),      # clean legal single
            (5003, 500, 0, None),   # off manufacturing grid
            (3500, 500, 0, None),   # fixed metal lo
            (6500, 500, 0, None),   # fixed metal hi
            (1300, 500, 0, None),   # own-net open band
            (8300, 300, 0, None),   # committed metal contact lo
            (9800, 300, 0, None),   # committed metal contact hi
            (11130, 330, 0, None),  # committed cut contact
            (12630, 330, 0, None),  # attempt cut overlap
        ]
        res = []
        for x, y, lo, n in pts:
            ok = eng._via_legal(x, y, lo, "sig", n=n)
            cong = eng._via_congestion(x, y, lo, "sig", n=n)
            res.append((ok, cong))
        return res, dict(eng._hard_rejections), dict(eng._hard_rejection_samples), eng._hard_rejection_generation

    r_cpp, rej_cpp, samp_cpp, gen_cpp = run_mode(False)
    r_py, rej_py, samp_py, gen_py = run_mode(True)
    assert r_cpp == r_py
    assert rej_cpp == rej_py
    assert samp_cpp == samp_py
    assert gen_cpp == gen_py

    # Direct box_congestion_fast_cpp vs Python _box_congestion
    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    eng = GridlessEngine(time_budget_sec=10.0)
    eng.route({}, obstacles, [], tech, resource_plan=None)
    eng._metal_idx.setdefault(0, _SpatialIndex()).add("other", (8000, 0, 8600, 600))
    eng._attempt_idx.setdefault(0, _SpatialIndex()).add("sig", (4800, 800, 5400, 1100))
    eng._pres_fac = 2.0
    eng.hist_fac = 1.5
    rc = eng._res_cell
    eng._history = {(5000 // rc, 500 // rc, 0): 2.0}
    from clay.layout.route.gridless_types import _CONG_WEIGHT, _COST_FACTOR_SCALE, _STEP_COST_PER_NM
    hgrid = cpp.NativeHistoryGrid(eng._history, eng.hist_fac, _COST_FACTOR_SCALE)
    for box in ((4800, 400, 5400, 700), (7900, 100, 8200, 400), (1100, 200, 1400, 600)):
        py_c = eng._box_congestion(0, box, "sig")
        cpp_c = cpp.box_congestion_fast_cpp(
            0, box[0], box[1], box[2], box[3],
            eng._space[0], rc, int(round(eng._pres_fac * _COST_FACTOR_SCALE)),
            rc * _STEP_COST_PER_NM * _CONG_WEIGHT, _COST_FACTOR_SCALE, "sig",
            eng._metal_idx[0]._native, eng._obs_idx[0]._native,
            eng._attempt_idx[0]._native, hgrid, -1,
        )
        assert cpp_c == py_c


def test_preflight_and_cartesian_astar_cpp_exact_parity(monkeypatch):
    _require_cpp()
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))

    # Build a maze with obstacles, committed metal, MIM bottom clearance, history, and multiple starts/goals
    obstacles = [
        Obstacle(2.0, -2.0, 2.6, 4.0, m1, owner_net="blk", kind="device_terminal", terminal_conductor_ref="tc:wall"),
        Obstacle(4.0, 1.0, 4.6, 5.0, m2, owner_net="blk2", kind="device_terminal", terminal_conductor_ref="tc:wall2"),
    ]

    def run_scenarios(disable_cpp: bool):
        if disable_cpp:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()

        results = {}

        # Scenario 1: Connected multi-layer detour with MIM bottom clearance & history
        eng = GridlessEngine(time_budget_sec=20.0, via_redundancy=2, allow_single_via_escape=True)
        eng.route({}, obstacles, [], tech, resource_plan=None)
        eng._pres_fac = 2.0
        eng.hist_fac = 1.5
        rc = eng._res_cell
        eng._history = {(3000 // rc, 1000 // rc, 1): 2.0, (3000 // rc, 2000 // rc, 1): 1.5}
        eng._mim_bottom_routing_clearance_by_gi = {1: (("mim_bot", (2800, 1500, 3400, 2200), 300),)}
        eng._metal_idx.setdefault(1, _SpatialIndex()).add("other", (3000, -500, 3500, 400))
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        xs = list(range(0, 7000, 500))
        ys = list(range(-2000, 5500, 500))
        routed = eng._route_edge(
            [((0, 0, 0), None), ((0, 500, 1), _ViaSite(0, 500, 0, 1))],
            {(6000, 2000, 0): None, (6000, 2500, 1): _ViaSite(6000, 2500, 0, 0)},
            "sig", xs, ys, [0, 1, 2], trunk=1,
        )
        results["connected_mim"] = (
            routed,
            eng._solve_expansions_total,
            eng._selected_access_pair,
            dict(eng._hard_rejections),
            dict(eng._hard_rejection_samples),
        )

        # Scenario 2: Static disconnected (sealed cage around goal)
        sealed_obs = [
            Obstacle(5.0, -1.0, 7.0, 1.0, m1, owner_net="cage", kind="device_terminal", terminal_conductor_ref="tc:c1"),
            Obstacle(5.0, -1.0, 7.0, 1.0, m2, owner_net="cage", kind="device_terminal", terminal_conductor_ref="tc:c2"),
        ]
        eng2 = GridlessEngine(time_budget_sec=20.0)
        eng2.route({}, sealed_obs, [], tech, resource_plan=None)
        eng2._hard_rejections = {}
        eng2._hard_rejection_samples = {}
        eng2._hard_rejection_generation = 0
        r2 = eng2._route_edge(
            [((0, 0, 0), None)],
            {(6000, 0, 0): None},
            "sig", list(range(0, 7000, 1000)), list(range(-1000, 2000, 1000)), [0, 1],
        )
        results["disconnected"] = (
            r2,
            eng2._edge_fail,
            dict(eng2._hard_rejections),
            dict(eng2._hard_rejection_samples),
        )

        # Scenario 3: Many goals (> 16, triggers Potential2DCore + search_reversed)
        eng3 = GridlessEngine(time_budget_sec=20.0)
        eng3.route({}, obstacles, [], tech, resource_plan=None)
        many_starts = [((0, y, 0), None) for y in range(-1500, 4500, 250)]  # 24 starts -> reversed!
        single_goal = {(6000, 1000, 1): None}
        r3 = eng3._route_edge(
            many_starts,
            single_goal,
            "sig",
            list(range(0, 7000, 500)),
            list(range(-1500, 5000, 250)),
            [0, 1, 2],
        )
        results["reversed_many"] = (r3, eng3._solve_expansions_total, eng3._selected_access_pair)

        # Scenario 4: Tight budget stop parity
        eng4 = GridlessEngine(time_budget_sec=20.0, max_expansions=5)
        eng4.route({}, obstacles, [], tech, resource_plan=None)
        eng4._solve_expansion_budget = 5
        r4 = eng4._route_edge(
            [((0, 0, 0), None)],
            {(6000, 4000, 0): None},
            "sig", xs, ys, [0, 1, 2],
        )
        results["budget_stop"] = (r4, eng4._edge_fail, eng4._solve_expansions_total)

        return results

    cpp_res = run_scenarios(False)
    py_res = run_scenarios(True)
    for k in py_res:
        assert cpp_res[k] == py_res[k], f"Mismatch in scenario {k}:\nCPP={cpp_res[k]}\nPY ={py_res[k]}"


def test_region_geometry_and_axis_price_run_ctx_parity(monkeypatch):
    cpp = _require_cpp()
    from clay.layout.route.axis_price_cache import AxisPriceCache
    from clay.layout.route.region_event_geometry import (
        Portal,
        RegionGeometry,
        project_portal,
        project_target,
        settled_axis_prefix,
    )

    rects = ((0, 0, 10, 10), (10, 2, 20, 12), (2, 10, 8, 18))
    portals = (
        Portal(0, 1, "x", 1, 10, 2, 10),
        Portal(1, 0, "x", -1, 10, 2, 10),
        Portal(0, 2, "y", 1, 10, 2, 8),
        Portal(2, 0, "y", -1, 10, 2, 8),
    )
    geom = RegionGeometry(rects, portals)

    # Compare project_portal & project_target in Python vs C++
    for p in portals:
        sr = rects[p.source]
        for px in (sr[0], (sr[0] + sr[2]) // 2, sr[2] - 1):
            for py in (sr[1], (sr[1] + sr[3]) // 2, sr[3] - 1):
                monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
                clear_native_cache()
                py_p = project_portal(geom, p, (px, py))
                monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
                clear_native_cache()
                cpp_p = project_portal(geom, p, (px, py))
                assert cpp_p == py_p

    for target in ((5, 5, 12, 12), (0, 0, 1, 1), (15, 15, 20, 20)):
        for px, py in ((0, 0), (5, 5), (9, 9), (2, 8)):
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
            clear_native_cache()
            py_t = project_target(geom, 0, (px, py), target)
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
            clear_native_cache()
            cpp_t = project_target(geom, 0, (px, py), target)
            assert cpp_t == py_t

    # ValueError parity on invalid portal / out-of-bounds entry
    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    with pytest.raises(ValueError):
        project_portal(geom, portals[0], (10, 5))
    with pytest.raises(ValueError):
        project_target(geom, 0, (-1, 5), (0, 0, 5, 5))

    # settled_axis_prefix_cpp parity
    xs = (0, 100, 300, 700, 1000)
    ys = (-500, 0, 250, 800)
    for start, end in (
        ((50, 0), (650, 0)),
        ((650, 0), (50, 0)),
        ((100, 0), (1000, 0)),
        ((10, 0), (50, 0)),
        ((100, -400), (100, 750)),
        ((100, 750), (100, -400)),
        ((100, 0), (100, 0)),
    ):
        assert cpp.settled_axis_prefix_cpp(start[0], start[1], end[0], end[1], xs, ys) == (
            settled_axis_prefix(start, end, xs, ys)
        )

    # AxisPriceCache.run_ctx vs pure Python AxisPriceCache.run parity (including budget refusal)
    from clay.layout.route.gridless_types import _CONG_WEIGHT, _STEP_COST_PER_NM
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    seed_obs = [Obstacle(0.0, 0.0, 0.5, 0.5, m1, owner_net="seed", kind="device_terminal", terminal_conductor_ref="tc:seed")]
    eng = GridlessEngine(time_budget_sec=10.0)
    eng.route({}, seed_obs, [], tech, resource_plan=None)
    eng._pres_fac = 2.0
    eng.hist_fac = 1.5
    rc = eng._res_cell
    eng._history = {(200 // rc, 0, 0): 2.0, (500 // rc, 0, 1): 1.25}
    eng._metal_idx.setdefault(0, _SpatialIndex()).add("other", (150, -50, 350, 50))
    _cpp, edge_ctx, _hg = eng._build_native_edge_context("sig", 1, True, rc * _STEP_COST_PER_NM * _CONG_WEIGHT)
    assert edge_ctx is not None

    monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
    clear_native_cache()
    py_steps = [0]
    py_cache = AxisPriceCache(
        xs, ys,
        lambda gi, ax, ay, bx, by: eng._edge_objective("sig", gi, ax, ay, bx, by, 1) + eng._edge_congestion(gi, ax, ay, bx, by, "sig"),
        lambda _k: (py_steps.__setitem__(0, py_steps[0] + 1) or True) if py_steps[0] < 15 else False,
    )
    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()
    cpp_cache = AxisPriceCache(xs, ys, lambda *a: 0, lambda _k: True)
    rem = 15
    sps = 0
    pch = 0
    for gi, s, e in (
        (0, (50, 0), (950, 0)),
        (0, (50, 0), (950, 0)),    # cached hit in run_cache
        (1, (950, 0), (50, 0)),
        (1, (100, -400), (100, 750)),
        (0, (0, 250), (1000, 250)),  # hits budget limit
    ):
        py_v = py_cache.run(gi, s, e)
        # Note: py_cache doesn't have outer event_run_cache; test run_ctx with max_run_cache=0 for exact step parity
        cpp_v, ch, sps, pch, _bs = cpp_cache.run_ctx(
            gi, s, e, edge_ctx, rem, sps, pch, 0, None,
            lambda g, a, b, c, d: eng._edge_objective("sig", g, a, b, c, d, 1) + eng._edge_congestion(g, a, b, c, d, "sig"),
        )
        rem -= ch
        assert cpp_v == py_v
    assert 15 - rem == py_steps[0]


def test_region_event_end_to_end_astar_and_multipin_escape_parity(monkeypatch):
    _require_cpp()
    from clay.layout.route.gridless_types import _OwnedTerminalAtom, _PlanarEndpointDomain

    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))

    # Multi-atom component + obstacles + committed metal + history
    obstacles = [
        Obstacle(0.0, 0.0, 2.0, 1.0, m1, owner_net="sig", kind="device_terminal", terminal_conductor_ref="tc:s0"),
        Obstacle(2.0, 0.0, 4.0, 1.0, m1, owner_net="sig", kind="device_terminal", terminal_conductor_ref="tc:s1"),
        Obstacle(8.0, 2.0, 11.0, 4.0, m1, owner_net="sig", kind="device_terminal", terminal_conductor_ref="tc:g0"),
        Obstacle(5.0, -1.0, 5.8, 3.0, m1, owner_net="blk", kind="device_terminal", terminal_conductor_ref="tc:b0"),
    ]

    def run_mode(disable_cpp: bool):
        if disable_cpp:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()

        eng = GridlessEngine(time_budget_sec=20.0, via_redundancy=2, allow_single_via_escape=True)
        out = eng.route({}, obstacles, [], tech, resource_plan=None)
        assert out.success
        eng._pres_fac = 2.0
        eng.hist_fac = 1.5
        rc = eng._res_cell
        eng._history = {(6000 // rc, 2000 // rc, 1): 2.0, (7000 // rc, 3000 // rc, 0): 1.5}
        eng._metal_idx.setdefault(1, _SpatialIndex()).add("other", (6000, 0, 6600, 1200))

        # 1) Multi-pin owned-component escape parity (tests component shared cache + stack cache)
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        esc_a = eng._compute_owned_component_escape_states("sig", 500, 500, 0)
        esc_b = eng._compute_owned_component_escape_states("sig", 3500, 500, 0)
        esc_rej = dict(eng._hard_rejections)
        esc_samp = dict(eng._hard_rejection_samples)
        esc_gen = eng._hard_rejection_generation

        # 2) End-to-end Region-Event A* (event_graph is not None)
        grid = max(1, int(getattr(eng, "_manufacturing_grid_nm", 5)))
        goal_atom = _OwnedTerminalAtom(0, (8000, 2000, 11000, 4000), "sig", "device_terminal", "tc:g0")
        target_domain = _PlanarEndpointDomain(
            node=(9000, 3000, 0),
            grid=grid,
            singleton=None,
            regions=((goal_atom, (8000, 2000, 11000, 4000)),),
            excluded=frozenset(),
        )
        eng._active_endpoint_access_domains = (None, target_domain)
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        eng._solve_expansions_total = 0
        xs = list(range(0, 12000, 1000))
        ys = list(range(-1000, 5000, 1000))
        routed = eng._route_edge(
            [((500, 500, 0), None)],
            {(9000, 3000, 0): None},
            "sig", xs, ys, [0, 1, 2], trunk=1,
        )
        return (
            esc_a, esc_b, esc_rej, esc_samp, esc_gen,
            routed,
            dict(eng._hard_rejections), dict(eng._hard_rejection_samples),
        )

    assert run_mode(False) == run_mode(True)

    # Also verify exact solve_expansions_total parity when isolating _gridless_cpp alone
    import clay.layout.route.axis_price_cache as _apc_mod
    import clay.layout.route.gridless_legality as _gl_mod
    import clay.layout.route.gridless_owned_escape as _goe_mod
    import clay.layout.route.gridless_planar_search as _gps_mod
    import clay.layout.route.gridless_types as _gt_mod
    import clay.layout.route.region_event_geometry as _reg_mod

    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()

    def run_isolated(disable_gridless: bool):
        orig_load = load_native
        fake_load = (lambda name: None if name == "_gridless_cpp" else orig_load(name)) if disable_gridless else orig_load
        for mod in (_apc_mod, _gl_mod, _goe_mod, _gps_mod, _gt_mod, _reg_mod):
            monkeypatch.setattr(mod, "load_native", fake_load)
        eng = GridlessEngine(time_budget_sec=20.0, via_redundancy=2, allow_single_via_escape=True)
        eng.route({}, obstacles, [], tech, resource_plan=None)
        eng._pres_fac = 2.0
        eng.hist_fac = 1.5
        rc = eng._res_cell
        eng._history = {(6000 // rc, 2000 // rc, 1): 2.0, (7000 // rc, 3000 // rc, 0): 1.5}
        eng._metal_idx.setdefault(1, _SpatialIndex()).add("other", (6000, 0, 6600, 1200))
        grid = max(1, int(getattr(eng, "_manufacturing_grid_nm", 5)))
        goal_atom = _OwnedTerminalAtom(0, (8000, 2000, 11000, 4000), "sig", "device_terminal", "tc:g0")
        target_domain = _PlanarEndpointDomain(
            node=(9000, 3000, 0),
            grid=grid,
            singleton=None,
            regions=((goal_atom, (8000, 2000, 11000, 4000)),),
            excluded=frozenset(),
        )
        eng._active_endpoint_access_domains = (None, target_domain)
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        eng._solve_expansions_total = 0
        routed = eng._route_edge(
            [((500, 500, 0), None)],
            {(9000, 3000, 0): None},
            "sig", list(range(0, 12000, 1000)), list(range(-1000, 5000, 1000)), [0, 1, 2], trunk=1,
        )
        return routed, eng._solve_expansions_total

    assert run_isolated(False) == run_isolated(True)


def test_watchdog_and_monkeypatch_fallback_safety(monkeypatch):
    from clay.layout.route.gridless_types import NegotiationWatchdogStop

    _require_cpp()
    monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
    clear_native_cache()

    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    obs = [Obstacle(2.0, -1.0, 2.5, 2.0, m1, owner_net="blk", kind="device_terminal", terminal_conductor_ref="tc:b")]

    # 1) Watchdog stop inside C++ preflight/A* still raises NegotiationWatchdogStop cleanly
    eng = GridlessEngine(time_budget_sec=10.0)
    eng.route({}, obs, [], tech, resource_plan=None)
    eng._watchdog_deadline = 0.0
    with pytest.raises(NegotiationWatchdogStop):
        eng._route_edge(
            [((0, 0, 0), None)],
            {(5000, 0, 0): None},
            "sig", list(range(0, 6000, 500)), list(range(-1000, 2500, 500)), [0, 1],
        )
    assert "_active_edge_ctx" not in eng.__dict__

    # 2) Monkeypatching _via_legal cleanly bypasses NativeEdgeContext and runs custom hook
    eng2 = GridlessEngine(time_budget_sec=10.0)
    eng2.route({}, obs, [], tech, resource_plan=None)
    eng2._watchdog_deadline = None
    via_calls = []
    orig_via = eng2._via_legal

    def spy_via(x, y, lo, net, n=None):
        via_calls.append((x, y, lo))
        return orig_via(x, y, lo, net, n=n)

    monkeypatch.setattr(eng2, "_via_legal", spy_via)
    routed = eng2._route_edge(
        [((0, 0, 0), None)],
        {(5000, 0, 1): None},
        "sig", list(range(0, 6000, 1000)), list(range(-1000, 2000, 1000)), [0, 1],
    )
    assert routed is not None
    assert len(via_calls) > 0
    assert "_active_edge_ctx" not in eng2.__dict__


def test_native_edge_context_mirror_legality_parity(monkeypatch):
    from clay.layout.route.constraints import Mirror

    _require_cpp()
    tech = LayoutTech.resolve("gf180mcu")
    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))

    # Axis at x = 5000 nm (axis2 = 10000 nm):
    # - obstacle on primary side at x=[2200, 2600]
    # - obstacle on partner side at x=[6800, 7200] (reflects to primary x=[2800, 3200]!)
    # - crossing near x=5000 triggers mirror_pair_in_transaction_conflict (metal & cut)
    obstacles = [
        Obstacle(0.0, 0.0, 0.4, 0.4, m1, owner_net="inp", kind="device_terminal", terminal_conductor_ref="tc:p0"),
        Obstacle(9.6, 0.0, 10.0, 0.4, m1, owner_net="inn", kind="device_terminal", terminal_conductor_ref="tc:n0"),
        Obstacle(2.2, 0.8, 2.6, 1.4, m1, owner_net="blk_a", kind="device_terminal", terminal_conductor_ref="tc:ba"),
        Obstacle(6.8, -0.2, 7.2, 0.6, m1, owner_net="blk_b", kind="device_terminal", terminal_conductor_ref="tc:bb"),
        Obstacle(6.8, 1.8, 7.2, 2.4, m2, owner_net="blk_m2", kind="device_terminal", terminal_conductor_ref="tc:bm2"),
    ]
    mirror = Mirror(net_a="inp", net_b="inn", axis_x_um=5.0)

    def run_mirror(disable_cpp: bool):
        if disable_cpp:
            monkeypatch.setenv("CLAY_DISABLE_CPP_PNR", "1")
        else:
            monkeypatch.delenv("CLAY_DISABLE_CPP_PNR", raising=False)
        clear_native_cache()
        eng = GridlessEngine(time_budget_sec=20.0, via_redundancy=2, allow_single_via_escape=True)
        eng.route({}, obstacles, [], tech, resource_plan=None)
        eng._active_mirror_legality = mirror
        eng._mirror_pair_conflict_seen = False
        eng._pres_fac = 2.0
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        eng._solve_expansions_total = 0

        # 1) Direct wire & via mirror legality checks through NativeEdgeContext
        _cpp_m, edge_ctx, _hg = eng._build_native_edge_context("inp", 1, False, 1000, [0, 1])
        if not disable_cpp:
            assert edge_ctx is not None
            assert edge_ctx.has_mirror is True
            eng._active_edge_ctx = edge_ctx
            eng._active_edge_ctx_net = "inp"
            eng._active_edge_ctx_cpp = _cpp_m
            eng._active_edge_ctx_gen_bumped = 0
            eng._active_edge_ctx_bypass = False

        wire_probes = [
            (0, 1000, 200, 1500, 200),   # clean symmetric wire
            (0, 2000, 1000, 2800, 1000), # hits blk_a on primary side
            (0, 2600, 200, 3400, 200),   # reflects to x=[6600, 7400], hits blk_b on partner side
            (0, 4800, 200, 5200, 200),   # crosses axis -> mirror pair metal conflict
        ]
        via_probes = [
            (1000, 200, 0, None),        # clean symmetric via
            (3000, 2100, 0, None),       # partner M2 hits blk_m2
            (4950, 200, 0, None),        # near axis -> mirror pair pad/cut conflict
            (4950, 200, 0, 1),           # single via near axis -> mirror pair cut/pad conflict
        ]
        w_out = [eng._wire_legal(gi, ax, ay, bx, by, "inp") for gi, ax, ay, bx, by in wire_probes]
        v_out = [eng._via_legal(x, y, lo, "inp", n=n) for x, y, lo, n in via_probes]
        if not disable_cpp:
            eng._flush_active_edge_ctx()
            for k in ("_active_edge_ctx", "_active_edge_ctx_net", "_active_edge_ctx_cpp",
                      "_active_edge_ctx_gen_bumped", "_active_edge_ctx_bypass"):
                eng.__dict__.pop(k, None)
        seen_direct = bool(eng._mirror_pair_conflict_seen)
        rej_direct = dict(eng._hard_rejections)
        samp_direct = dict(eng._hard_rejection_samples)

        # 2) Full _route_edge with active mirror legality (detours around partner blocker!)
        eng._hard_rejections = {}
        eng._hard_rejection_samples = {}
        eng._hard_rejection_generation = 0
        eng._mirror_pair_conflict_seen = False
        eng._solve_expansions_total = 0
        xs = list(range(200, 5001, 400))
        ys = list(range(-400, 2801, 400))
        routed = eng._route_edge(
            [((200, 200, 0), None)],
            {(4200, 200, 0): None},
            "inp", xs, ys, [0, 1], trunk=1,
        )
        return (
            w_out, v_out, seen_direct, rej_direct, samp_direct,
            routed, eng._solve_expansions_total,
            bool(eng._mirror_pair_conflict_seen),
            dict(eng._hard_rejections), dict(eng._hard_rejection_samples),
        )

    res_cpp = run_mirror(False)
    res_py = run_mirror(True)
    assert res_cpp == res_py
    assert res_cpp[2] is True  # mirror_pair_conflict_seen triggered
    assert res_cpp[5] is not None  # mirrored route succeeded


def test_native_via_target_projector_frontier_parity():
    import heapq
    import random
    from clay.layout.route.gridless_types import (
        _OwnedTerminalAtom,
        _PlanarEndpointDomain,
        _VIA_COST,
        _region_via_goal_prefix_lower_bound,
        _region_via_goal_waypoint,
    )
    from clay.layout.route.region_event_geometry import RegionGeometry

    cpp = _require_cpp()
    rng = random.Random(202607)
    grid = 5
    turn_cost = 1200
    rect_bucket_nm = 140
    source_rect = (0, 0, 200, 200)
    geom = RegionGeometry(rectangles=(source_rect,), portals=())
    target_atom = _OwnedTerminalAtom(0, (400, 400, 600, 600), "sig", "device_terminal", "tc:t")
    target_domain = _PlanarEndpointDomain(
        node=(100, 100, 0),
        grid=grid,
        singleton=None,
        regions=((target_atom, (400, 400, 600, 600)),),
        excluded=frozenset(),
    )
    goal_set = {(500, 500, 0), (150, 500, 0), (800, 200, 1)}
    p0 = cpp.Native2DPotential({(500, 500, 0): 0, (150, 500, 0): 0, (800, 200, 0): _VIA_COST}, 51)
    p1 = cpp.Native2DPotential({(500, 500, 0): _VIA_COST, (150, 500, 0): _VIA_COST, (800, 200, 0): 0}, 1)
    hmin = cpp.NativeLayerHmin(
        ((0, 51, p0, 0), (1, 1, p1, _VIA_COST)),
        ((400, 400, 600, 600),),
        _VIA_COST,
        2,
        10000,
    )
    other_idx = _SpatialIndex(cell=64)
    for rid in range(25):
        x0 = rng.randint(0, 150)
        y0 = rng.randint(0, 150)
        other_idx.add(str(rid), (x0, y0, x0 + rng.randint(15, 65), y0 + rng.randint(15, 65)))

    clipped = tuple(
        (x0, y0, x0 + rng.randint(2, 25), y0 + rng.randint(2, 25))
        for x0, y0 in [(rng.randint(0, 170), rng.randint(0, 170)) for _ in range(35)]
    )

    proj = cpp.NativeViaTargetProjector(
        goal_set=goal_set,
        target_node=target_domain.node,
        target_singleton_none=True,
        target_regions=((400, 400, 600, 600),),
        target_excluded_xy=(),
        hmin=hmin,
        grid=grid,
        turn_cost=turn_cost,
        via_cost=_VIA_COST,
        rect_bucket_nm=rect_bucket_nm,
        neg_bucket_dedup=True,
        via_succ_dir_1=1,
        via_succ_dir_2=2,
    )

    py_frontier = []
    py_seq = 0
    rect_counts = {}
    other_cap = 2
    # Pre-saturate a few buckets so both enqueue-time and pop-time pruning fire
    for rid in range(10):
        for ulast in (1, 2):
            for bx in range(0, 8):
                for by in range(0, 8):
                    if (rid + bx + by) % 3 == 0:
                        k = (1, rid, ulast, bx, by)
                        rect_counts[k] = 2
                        proj.inc_rect_entry(*k)
                        proj.inc_rect_entry(*k)

    dist = {}
    calls = [
        (0, 1, 30, 50, 1000, 1, 51, 1500, None, None),
        (0, 1, 30, 50, 2000, 2, 51, 2200, 900000, None),   # cache hit + cutoff
        (0, 1, 100, 100, 500, 0, 51, 500, None, 800000),   # has_cross + pattern_cutoff
    ]
    for idx_c, (ul, other, px, py, u_du, ulast, u_rate, base_lb, cutoff, pcutoff) in enumerate(calls):
        u_base = (px * grid, py * grid, ul, ulast, 0, (), -1, px * grid, py * grid, 0)
        # Make first u_base stale on second push to exercise dist staleness check
        dist[u_base] = u_du if idx_c != 0 else u_du - 10

        has_cached_before = proj.has_cached_target(ul, other, 0, px, py, True)
        assert has_cached_before == (idx_c == 1)
        fmask = proj.enqueue_via_targets(
            ul, other, 0, px, py, 0, 0, 200, 200,
            None if has_cached_before else clipped,
            other_idx._native,
            u_base, u_du, ulast, u_rate, base_lb,
            cutoff, pcutoff, other_cap, False,
        )
        c_cross, c_sat = bool(fmask & 1), bool(fmask & 2)

        # Reference Python enqueue using proj.project()
        py_cross, py_recs = proj.project(ul, other, px, py, 0, 0, 200, 200, clipped, other_idx._native)
        py_sat = False
        ux, uy = px * grid, py * grid
        for (path, ex, ey, p1_y_nm, gw, is_eg, oulast, owners, ebx, eby, eturn, tlb) in py_recs:
            if gw is None:
                any_open = any(rect_counts.get((other, oid, oulast, ebx, eby), 0) < other_cap for oid in owners)
                if not any_open:
                    if owners:
                        py_sat = True
                    continue
            if tlb is None:
                lb_f = _region_via_goal_prefix_lower_bound(
                    path, gw, grid=grid, ux=ux, uy=uy, ulast=ulast,
                    rate=u_rate, turn_cost=turn_cost, base=base_lb, pending=0,
                )
            else:
                fv = 1 if p1_y_nm == uy else 2
                t1 = turn_cost if (ulast != 0 and fv != ulast) else 0
                lb_f = base_lb + (abs(ex - ux) + abs(ey - uy)) * u_rate + t1 + eturn + tlb
            if cutoff is not None and lb_f > cutoff:
                continue
            if pcutoff is not None and lb_f >= pcutoff:
                continue
            py_seq += 1
            heapq.heappush(py_frontier, (lb_f, py_seq, "via_target", u_base, u_du, path, -other - 2))
        assert c_cross == py_cross
        assert c_sat == py_sat

    # Now saturate one more bucket mid-flight to test pop-time saturation pruning!
    for rid in range(10, 18):
        for ulast in (1, 2):
            for bx in range(0, 8):
                for by in range(0, 8):
                    k = (1, rid, ulast, bx, by)
                    rect_counts[k] = rect_counts.get(k, 0) + 2
                    proj.inc_rect_entry(*k)
                    proj.inc_rect_entry(*k)

    # Drain both frontiers and assert bit-exact popped sequence & sat_pruned!
    py_popped = []
    py_pop_sat = False
    while py_frontier:
        top_f = py_frontier[0][0]
        assert proj.frontier_top_f() == top_f
        _f, _s, _k, pu, pdu, ppath, pdest = heapq.heappop(py_frontier)
        c_pop = proj.frontier_pop(dist, True, {1: other_cap})

        py_skip = False
        if pdu > dist.get(pu, float("inf")):
            py_skip = True
        else:
            r_other = -pdest - 2
            gw = _region_via_goal_waypoint(
                ppath, grid=grid, layer=pu[2], goal_set=goal_set, target_domain=target_domain
            )
            if gw is None:
                epx, epy = ppath[-1]
                vdir = 1 if ppath[-1][1] == ppath[-2][1] else 2
                ebx, eby = (epx * grid) // rect_bucket_nm, (epy * grid) // rect_bucket_nm
                any_open = False
                for owner, _box in other_idx.query((epx, epy, epx, epy)):
                    if rect_counts.get((r_other, int(owner), vdir, ebx, eby), 0) >= other_cap:
                        py_pop_sat = True
                    else:
                        any_open = True
                        break
                if not any_open:
                    py_skip = True

        if py_skip:
            assert isinstance(c_pop, bool)
        else:
            assert isinstance(c_pop, tuple)
            cu, cdu, cpath, cdest, _csat = c_pop
            assert (cu, cdu, cpath, cdest) == (pu, pdu, ppath, pdest)
            py_popped.append((pu, pdu, ppath, pdest))

    assert proj.frontier_top_f() is None
    assert len(py_popped) > 0
    proj.clear_frontier()
    assert proj.frontier_top_f() is None


def test_localized_candidate_deck_connects_probes_matches_full_layout_with_mim_carve():
    """Verify localized _fixed_deck_components_cache matches brute-force full layout."""
    import klayout.db as kdb
    from clay.layout.route.deck_isolation import (
        DeckConnectStack,
        NetProbe,
        check_deck_net_isolation,
    )
    from clay.layout.route.gridless_types import _ViaSite, _WireSeg

    tech = LayoutTech.resolve("gf180mcu", mim_option="B", metal_level="5LM")
    stack = DeckConnectStack.from_tech(tech)
    assert stack.mim_carves, "GF180 5LM Option-B must define MIM carve"
    carve = stack.mim_carves[0]

    m1 = tuple(tech.metal_key(1))
    m2 = tuple(tech.metal_key(2))
    v1 = tuple(tech.via_key(1))
    m4 = carve.bottom
    m5 = tuple(tech.metal_key(5))
    v4 = carve.riser
    fusetop = carve.top_marker

    eng = GridlessEngine(time_budget_sec=10.0)
    out = eng.route({}, [], [], tech, resource_plan=None)
    assert out.success
    rtech = eng._tech
    eng._deck_connect_stack = stack

    # Build > 64 fixed rectangles so localization triggers, including:
    # - 70 unrelated dummy fixed shapes
    # - a fixed M1-V1-M2 chain
    # - an isolated fixed V1 cut
    # - a GF180 5LM Option-B MIM capacitor (M4 bottom + fusetop marker + V4 on top plate + M5 top)
    fixed: list[tuple[tuple[int, int], str | None, tuple[int, int, int, int]]] = []
    for i in range(70):
        x0 = 100000 + i * 2000
        fixed.append((m1, f"dummy_{i}", (x0, 0, x0 + 500, 500)))
    # Fixed M1 -> V1 -> M2 bridge at x=0..4000
    fixed.append((m1, "sig_a", (0, 0, 1200, 600)))
    fixed.append((v1, "sig_a", (800, 150, 1060, 410)))
    fixed.append((m2, "sig_a", (700, 100, 4000, 500)))
    # Isolated fixed V1 cut at (6000, 200)
    fixed.append((v1, None, (6000, 200, 6260, 460)))
    # Fixed Option-B MIM capacitor: M4 plate (0..10000, 10000..20000),
    # fusetop top marker (2000..8000, 12000..18000), fixed V4+M5 on top marker,
    # and M4 bottom-plate tab extending outside fusetop to x=0..1500.
    fixed.append((m4, "mim_bot", (0, 10000, 10000, 20000)))
    fixed.append((fusetop, "mim_top", (2000, 12000, 8000, 18000)))
    fixed.append((v4, "mim_top", (4800, 14800, 5060, 15060)))
    fixed.append((m5, "mim_top", (4500, 14500, 9000, 15500)))

    eng._fixed_deck_literal_rects = fixed

    def brute_force(net, segs, vias, probe_states):
        layout = kdb.Layout()
        layout.dbu = float(rtech.pdk.rules.dbu_nm) / 1000.0
        cell = layout.create_cell("BRUTE")
        li_map = {}

        def _li(layer):
            if layer not in li_map:
                li_map[layer] = layout.layer(*layer)
            return li_map[layer]

        for raw_layer, _owner, rect in fixed:
            cell.shapes(_li(tuple(raw_layer))).insert(kdb.Box(*rect))
        from clay.layout.route.gridless_types import _RoutedNet

        emitted = eng._emit_net(_RoutedNet(name=net, segs=list(segs), vias=list(vias)))
        for poly in emitted.wires:
            layer = (int(poly.layer), int(poly.datatype))
            pts = [kdb.Point(int(pt.x), int(pt.y)) for pt in poly.polygon.vertices]
            cell.shapes(_li(layer)).insert(kdb.Polygon(pts))
        probes = tuple(
            NetProbe("CANDIDATE_ACCESS", tuple(rtech.gi_to_metal_key(gi)), x, y)
            for x, y, gi in probe_states
        )
        return check_deck_net_isolation(layout=layout, cell=cell, stack=stack, probes=probes).ok

    gi_m1, gi_m2, gi_m4, gi_m5 = 0, 1, 3, 4
    assert tuple(rtech.gi_to_metal_key(gi_m1)) == m1
    assert tuple(rtech.gi_to_metal_key(gi_m4)) == m4
    assert tuple(rtech.gi_to_metal_key(gi_m5)) == m5

    cases = [
        # 1) Connected through fixed M1->V1->M2 bridge
        ("sig_a", [_WireSeg(gi_m2, 3800, 300, 5000, 300, 300)], [], [(200, 300, gi_m1), (4800, 300, gi_m2)], True),
        # 2) Disconnected probes
        ("sig_a", [_WireSeg(gi_m1, 20000, 20000, 22000, 20000, 300)], [], [(200, 300, gi_m1), (21000, 20000, gi_m1)], False),
        # 3) Connected across isolated fixed V1 cut
        ("sig_iso", [_WireSeg(gi_m1, 5500, 330, 6500, 330, 300), _WireSeg(gi_m2, 5800, 330, 7200, 330, 300)], [], [(5600, 330, gi_m1), (7000, 330, gi_m2)], True),
        # 4) Option-B MIM isolation: bottom M4 tab vs top M5 plate MUST NOT connect
        ("mim_bot", [_WireSeg(gi_m4, 500, 15000, 1200, 15000, 400)], [], [(600, 15000, gi_m4), (8500, 15000, gi_m5)], False),
        # 5) Option-B MIM top plate: candidate V4 on fixed fusetop connects to fixed M5 top plate
        ("mim_top", [_WireSeg(gi_m5, 2500, 15000, 3500, 15000, 500)], [_ViaSite(3000, 15000, gi_m4, 1)], [(2600, 15000, gi_m5), (8500, 15000, gi_m5)], True),
        # 6) Option-B MIM carve: candidate V4 inside fusetop does NOT connect M4 bottom tab to M5
        ("mim_bad", [_WireSeg(gi_m5, 2500, 15000, 3500, 15000, 500)], [_ViaSite(3000, 15000, gi_m4, 1)], [(600, 15000, gi_m4), (2600, 15000, gi_m5)], False),
        # 7) Option-B MIM outside carve: candidate V4 on uncarved M4 tab DOES connect M4 to M5
        ("mim_out", [_WireSeg(gi_m5, 400, 15000, 1000, 15000, 500)], [_ViaSite(700, 15000, gi_m4, 1)], [(500, 15000, gi_m4), (900, 15000, gi_m5)], True),
    ]

    for net, segs, vias, probes, expected in cases:
        got_loc = eng._candidate_deck_connects_probes(net, segs, vias, probes, include_fixed=True)
        got_bf = brute_force(net, segs, vias, probes)
        assert got_loc == got_bf == expected, (net, got_loc, got_bf, expected)





