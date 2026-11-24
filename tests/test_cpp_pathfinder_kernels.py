"""Differential, budget-stop, and regression tests for C++20 PathFinder & Maze kernels."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple

import klayout.db as kdb
import pytest

from clay._native import clear_native_cache, load_native
from clay.flow.execution_budget import (
    EXECUTION_CANCELLED,
    WALL_BUDGET_EXCEEDED,
    CancellationToken,
    ExecutionBudget,
    ExecutionStopped,
)
from clay.flow.execution_session import ExecutionSession
from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.route.constraints import Direction, LayerSet, Mirror, Spacing, Width
from clay.layout.route.engine import Obstacle, Pin
from clay.layout.route.engine_maze import MazeRouteEngine, _Tech
from clay.layout.route.engine_pathfinder import PathFinderEngine
from clay.layout.tech import LayoutTech

pytestmark = pytest.mark.unit


@contextmanager
def _cpp_mode(enabled: bool):
    prev = os.environ.get("CLAY_DISABLE_CPP_PNR")
    try:
        os.environ["CLAY_DISABLE_CPP_PNR"] = "0" if enabled else "1"
        clear_native_cache()
        yield
    finally:
        if prev is None:
            os.environ.pop("CLAY_DISABLE_CPP_PNR", None)
        else:
            os.environ["CLAY_DISABLE_CPP_PNR"] = prev
        clear_native_cache()


@pytest.fixture(scope="module")
def tech():
    return LayoutTech.resolve("gf180mcu")


@pytest.fixture(scope="module")
def pitch(tech):
    return _Tech(tech).pitch_nm / 1000.0


def _exact_signature(rc):
    """Exact geometry + metadata signature for A/B differential checks."""
    nets = []
    for n in sorted(rc.nets):
        nr = rc.nets[n]
        wires = tuple(
            (
                int(w.layer),
                int(w.datatype),
                tuple((int(v.x), int(v.y)) for v in w.polygon.vertices),
            )
            for w in nr.wires
        )
        nets.append((n, bool(nr.connected), round(float(nr.length_um), 6), wires))
    shorts = tuple(sorted((s["cell"], s["nets"]) for s in rc.shorts))
    mirrors = tuple(
        (
            r.get("net_a"),
            r.get("net_b"),
            bool(r.get("satisfied")),
            r.get("n_reflection_cells_blocked"),
            r.get("n_reflection_cells"),
        )
        for r in (getattr(rc, "mirror_reports", None) or [])
    )
    return (
        bool(rc.success),
        int(getattr(rc, "negotiation_rounds", 0)),
        tuple(getattr(rc, "best_overused_cells", ())),
        tuple(getattr(rc, "best_pad_contended_cells", ())),
        shorts,
        mirrors,
        tuple(nets),
    )


def _run_ab(engine_factory, pins, obstacles, constraints, tech, **route_kwargs):
    with _cpp_mode(False):
        py_res = engine_factory().route(pins, obstacles, constraints, tech, **route_kwargs)
    with _cpp_mode(True):
        cpp_res = engine_factory().route(pins, obstacles, constraints, tech, **route_kwargs)
    assert _exact_signature(cpp_res) == _exact_signature(py_res)
    return py_res, cpp_res


def test_native_pathfinder_module_loads_and_toggles():
    with _cpp_mode(True):
        mod = load_native("_pathfinder_cpp")
        assert mod is not None
        assert hasattr(mod, "NativePathFinderGrid")
    with _cpp_mode(False):
        assert load_native("_pathfinder_cpp") is None


def test_ab_undirected_pathfinder_congestion_and_widen(tech, pitch):
    """Multi-round negotiated congestion (undirected A*, 4-level window ladder, widen)."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    P = pitch
    pins = {
        f"n{k}": [
            Pin(f"n{k}", "A", "d", (k * 2.0 * P, 0.0), m1),
            Pin(f"n{k}", "B", "d", ((4 - k) * 2.0 * P, 14.0 * P), m1),
        ]
        for k in range(5)
    }
    obstacles = [
        # Horizontal divider on M1 with a narrow 2-track gap forcing negotiation + M2 climbs
        Obstacle(-4.0 * P, 6.5 * P, 2.5 * P, 7.5 * P, m1),
        Obstacle(5.5 * P, 6.5 * P, 14.0 * P, 7.5 * P, m1),
        # Partial M2 blockage over the middle to force mixed-layer detours and tie-breaking
        Obstacle(1.5 * P, 5.5 * P, 6.5 * P, 8.5 * P, m2),
    ]
    cons = [
        Width("n0", 0.50),
        Width("n4", 0.50),
        LayerSet("n2", frozenset({m1, m2})),
    ]
    for widen in (False, True):
        py_r, cpp_r = _run_ab(
            lambda _w=widen: PathFinderEngine(widen=_w, max_iters=16),
            pins, obstacles, cons, tech,
        )
        assert cpp_r.success is True
        assert cpp_r.shorts == []
        assert py_r.negotiation_rounds == cpp_r.negotiation_rounds


def test_ab_directed_astar_comb_contract(tech, pitch):
    """Directed 3D A* with Direction constraints, multi-pin dir_runs tree memory, and via resets."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    m3 = tech.metal_key(3)
    P = pitch
    pins = {
        "bus_a": [
            Pin("bus_a", "D0", "d", (0.0, 0.0), m1),
            Pin("bus_a", "D1", "d", (12.0 * P, 2.0 * P), m1),
            Pin("bus_a", "D2", "d", (6.0 * P, 10.0 * P), m2),
            Pin("bus_a", "D3", "d", (14.0 * P, 12.0 * P), m1),
        ],
        "bus_b": [
            Pin("bus_b", "E0", "s", (2.0 * P, 12.0 * P), m1),
            Pin("bus_b", "E1", "s", (10.0 * P, 0.0), m1),
            Pin("bus_b", "E2", "s", (14.0 * P, 6.0 * P), m3),
        ],
    }
    obstacles = [
        Obstacle(4.5 * P, 3.5 * P, 8.5 * P, 6.5 * P, m2),
        Obstacle(7.5 * P, 7.5 * P, 10.5 * P, 9.5 * P, m1),
    ]
    cons = [
        Direction(1, "H", weight=2.0),
        Direction(2, "V", weight=2.5),
        Direction(3, "H", weight=1.5),
    ]
    py_r, cpp_r = _run_ab(
        lambda: PathFinderEngine(widen=True, max_iters=12),
        pins, obstacles, cons, tech,
    )
    assert cpp_r.success is True
    assert cpp_r.shorts == []


def test_ab_via_blocked_via_cost_and_envelope(tech, pitch):
    """via_blocked (device-pad-intrusion, via cuts, envelope) + Round>=1 via_cost parity."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    m3 = tech.metal_key(3)
    v1 = tech.via_key(1)
    P = pitch
    pins = {
        "n_hi": [
            Pin("n_hi", "M1", "d", (1.0 * P, 2.0 * P), m1),
            Pin("n_hi", "M2", "d", (11.0 * P, 2.0 * P), m1),
        ],
        "n_lo": [
            Pin("n_lo", "M3", "s", (1.0 * P, 3.0 * P), m1),
            Pin("n_lo", "M4", "s", (11.0 * P, 3.0 * P), m1),
        ],
        "n_cross": [
            Pin("n_cross", "M5", "g", (6.0 * P, 0.0), m1),
            Pin("n_cross", "M6", "g", (6.0 * P, 6.0 * P), m1),
        ],
    }
    obstacles = [
        # Foreign device M1 pads near pins & mid-channel (triggers via_infeasible device-pad-intrusion)
        Obstacle(4.5 * P, 1.5 * P, 5.2 * P, 4.5 * P, m1, owner_net="vdd", kind="device"),
        Obstacle(7.0 * P, 1.5 * P, 7.7 * P, 4.5 * P, m1, owner_net="vss", kind="device"),
        # Device via cut obstacle on V1 (triggers via_obs_by_gi check)
        Obstacle(5.85 * P, 1.85 * P, 6.15 * P, 2.15 * P, v1, owner_net="vdd", kind="device_cut"),
        # Cap plate via obstacle
        Obstacle(8.85 * P, 2.85 * P, 9.15 * P, 3.15 * P, v1, owner_net="n_hi", kind="cap_plate_via"),
    ]
    env = (-2000, -2000, int(round(14.0 * P * 1000)), int(round(9.0 * P * 1000)))
    py_r, cpp_r = _run_ab(
        lambda: PathFinderEngine(widen=True, max_iters=16),
        pins, obstacles, [], tech, routing_envelope_nm=env,
    )
    assert cpp_r.success is True
    assert cpp_r.shorts == []
    assert py_r.best_pad_contended_cells == cpp_r.best_pad_contended_cells


def test_ab_mirror_exact_and_partial_corridors(tech, pitch):
    """Mirror reserved corridors: both exact adoption and partial fallback with hard-prefer bias."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    P = pitch
    axis_x = 10.0 * P

    # Case 1: symmetric unobstructed world -> exact mirror adoption
    pins_exact = {
        "inp": [
            Pin("inp", "M1", "g", (4.0 * P, 2.0 * P), m1),
            Pin("inp", "M3", "d", (7.0 * P, 10.0 * P), m1),
        ],
        "inn": [
            Pin("inn", "M2", "g", (16.0 * P, 2.0 * P), m1),
            Pin("inn", "M4", "d", (13.0 * P, 10.0 * P), m1),
        ],
        "bias": [
            Pin("bias", "M5", "g", (2.0 * P, 6.0 * P), m1),
            Pin("bias", "M6", "g", (18.0 * P, 6.0 * P), m1),
        ],
    }
    cons_exact = [Mirror("inp", "inn", axis_x_um=axis_x)]
    py_e, cpp_e = _run_ab(
        lambda: PathFinderEngine(widen=False, max_iters=12),
        pins_exact, [], cons_exact, tech,
    )
    assert cpp_e.success is True
    assert cpp_e.mirror_reports and cpp_e.mirror_reports[0]["satisfied"] is True

    # Case 2: asymmetric fixed obstacle on inn side -> partial mirror fallback
    obs_partial = [
        Obstacle(12.5 * P, 4.5 * P, 16.5 * P, 7.5 * P, m1),
        Obstacle(12.5 * P, 4.5 * P, 16.5 * P, 7.5 * P, m2),
    ]
    py_p, cpp_p = _run_ab(
        lambda: PathFinderEngine(widen=False, max_iters=12),
        pins_exact, obs_partial, cons_exact, tech,
    )
    assert cpp_p.nets["inp"].connected and cpp_p.nets["inn"].connected
    assert cpp_p.mirror_reports == py_p.mirror_reports


def test_ab_over_active_penalty_and_trunk_layers(tech, pitch):
    """over_active_penalty + trunk_layers differential parity and active-detour check."""
    m1 = tech.metal_key(1)
    P = pitch
    pins = {
        "long_sig": [
            Pin("long_sig", "A", "d", (0.0, 5.0 * P), m1),
            Pin("long_sig", "B", "d", (26.0 * P, 5.0 * P), m1),
        ],
        "short_sig": [
            Pin("short_sig", "C", "s", (4.0 * P, 5.0 * P), m1),
            Pin("short_sig", "D", "s", (10.0 * P, 5.0 * P), m1),
        ],
    }
    active_rects = [(6.0 * P, 2.0 * P, 20.0 * P, 8.0 * P)]
    py_r, cpp_r = _run_ab(
        lambda: PathFinderEngine(
            widen=True,
            trunk_layers=True,
            over_active_penalty=18.0,
            over_active_min_level=3,
            over_active_rects_um=active_rects,
            max_iters=12,
        ),
        pins, [], [], tech,
    )
    assert cpp_r.success is True
    assert cpp_r.shorts == []


def test_ab_maze_engine_taper_reorder_mirror_direction_preroute(tech, pitch):
    """MazeRouteEngine C++20 vs pure Python exact parity across taper/reorder/Mirror/Direction/preroute."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    P = pitch
    pins = {
        "a": [
            Pin("a", "M1", "d", (2.0 * P, 1.0 * P), m1),
            Pin("a", "M2", "d", (6.0 * P, 9.0 * P), m1),
        ],
        "b": [
            Pin("b", "M3", "d", (14.0 * P, 1.0 * P), m1),
            Pin("b", "M4", "d", (10.0 * P, 9.0 * P), m1),
        ],
        "rail": [
            Pin("rail", "M5", "s", (1.0 * P, 5.0 * P), m1),
            Pin("rail", "M6", "s", (15.0 * P, 5.0 * P), m1),
        ],
    }
    pre_box = Polygon.rectangle(
        int(round(0.5 * P * 1000)),
        int(round(4.75 * P * 1000)),
        int(round(15.5 * P * 1000)),
        int(round(5.25 * P * 1000)),
    )
    preroutes = {"rail": [LayerPolygon(layer=m2[0], datatype=m2[1], polygon=pre_box)]}
    cons = [
        Mirror("a", "b", axis_x_um=8.0 * P),
        Direction(1, "H", weight=1.5),
        Direction(2, "V", weight=1.5),
        Spacing("a", "b", min_um=0.4),
        Width("a", 0.45),
    ]
    py_r, cpp_r = _run_ab(
        lambda: MazeRouteEngine(taper=True, max_reorder=2),
        pins, [], cons, tech, preroutes=preroutes,
    )
    assert cpp_r.success is True
    assert cpp_r.shorts == []


def test_ab_net_footprint_native_matches_python(tech, pitch):
    """NativePathFinderGrid.net_footprint matches PathFinderEngine._net_footprint."""
    with _cpp_mode(True):
        mod = load_native("_pathfinder_cpp")
        assert mod is not None
        eng = PathFinderEngine(widen=True)
        tview = _Tech(tech)
        m1 = tech.metal_key(1)
        pins = {
            "n0": [
                Pin("n0", "A", "d", (0.0, 0.0), m1),
                Pin("n0", "B", "d", (10.0 * pitch, 6.0 * pitch), m1),
            ]
        }
        grid = eng._make_grid(tview, pins, [])
        from clay.layout.route.engine_pathfinder import _build_native_grid

        ng = _build_native_grid(grid, tview, tech, eng._via_env_rule_nm)
        assert ng is not None
        wire_cells = {
            (8, 8, 0), (9, 8, 0), (10, 8, 0), (10, 8, 1), (10, 9, 1), (10, 10, 1),
            (10, 10, 2), (11, 10, 2),
        }
        pin_cells = [(8, 8, 0), (11, 10, 2)]
        trunk_w = 600
        width_at = eng._make_width_at(grid, tview, pin_cells, trunk_w)
        ng.prepare_net(0, trunk_w, 0, list(range(tview.n_layers)), pin_cells, {}, 0, True, False)
        cpp_fp = set(ng.net_footprint(0, list(wire_cells), trunk_w, True))
        py_fp = eng._net_footprint(grid, tview, wire_cells, width_at)
        assert cpp_fp == py_fp


def test_custom_python_callbacks_on_astar_and_route_one_net_soft(tech, pitch):
    """Direct _astar / _route_one_net_soft calls with custom lambdas stay 100% compatible."""
    eng = PathFinderEngine()
    tview = _Tech(tech)
    m1 = tech.metal_key(1)
    pins = {"n": [Pin("n", "A", "d", (0.0, 0.0), m1), Pin("n", "B", "d", (6.0 * pitch, 0.0), m1)]}
    grid = eng._make_grid(tview, pins, [])
    src = (grid.snap_index(0.0, 0.0)[0], grid.snap_index(0.0, 0.0)[1], 0)
    dst = (grid.snap_index(6.0 * pitch, 0.0)[0], grid.snap_index(6.0 * pitch, 0.0)[1], 0)

    wall_x = (src[0] + dst[0]) // 2
    called = {"blk": 0, "cost": 0, "vcost": 0}

    def my_blocked(ix, iy, gi):
        called["blk"] += 1
        return gi == 0 and ix == wall_x

    def my_cost(ix, iy, gi):
        called["cost"] += 1
        return 2.0 if iy == src[1] else 0.0

    def my_via_cost(ix, iy, gl):
        called["vcost"] += 1
        return 0.5

    with _cpp_mode(True):
        path = eng._astar(
            grid=grid, tech=tview, sources={src}, target=dst, layer_gis=[0, 1],
            is_blocked=my_blocked, cell_cost=my_cost, via_cost=my_via_cost,
        )
        assert path is not None and path[0] == src and path[-1] == dst
        assert all(not my_blocked(*c) for c in path[1:-1])
        assert called["blk"] > 0 and called["cost"] > 0 and called["vcost"] > 0

        wcells, ok = eng._route_one_net_soft(
            grid=grid, tech=tview, pin_cells=[src, dst], layer_gis=[0, 1],
            is_blocked=my_blocked, cell_cost=my_cost, via_cost=my_via_cost,
        )
        assert ok is True and src in wcells and dst in wcells


@pytest.mark.parametrize("cpp_enabled", [True, False])
def test_execution_session_wall_budget_interrupts_quickly(tmp_path, tech, pitch, cpp_enabled):
    """Hard Fix #1: wall budget stop interrupts A* search cleanly in < 0.5s (C++20 and Python)."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    m3 = tech.metal_key(3)
    P = pitch
    # Large 3D grid with target completely walled off on all allowed layers so A* must
    # explore > 250k states across iterations unless interrupted by checkpoint() every 4096 expansions.
    far = 260.0 * P if cpp_enabled else 160.0 * P
    pins = {
        "n0": [
            Pin("n0", "A", "d", (0.0, 0.0), m1),
            Pin("n0", "B", "d", (far, far), m1),
        ],
    }
    obstacles = []
    layers_for_wall = (m1, m2, m3) if cpp_enabled else (m1,)
    for mk in layers_for_wall:
        obstacles.extend([
            Obstacle(far - 2.0 * P, far - 2.0 * P, far + 2.0 * P, far - 1.0 * P, mk),
            Obstacle(far - 2.0 * P, far + 1.0 * P, far + 2.0 * P, far + 2.0 * P, mk),
            Obstacle(far - 2.0 * P, far - 2.0 * P, far - 1.0 * P, far + 2.0 * P, mk),
            Obstacle(far + 1.0 * P, far - 2.0 * P, far + 2.0 * P, far + 2.0 * P, mk),
        ])
    cons = [LayerSet("n0", frozenset(layers_for_wall))]
    ev_path = tmp_path / f"events_wall_{int(cpp_enabled)}.jsonl"
    session = ExecutionSession(
        project_id="test_pf_budget",
        run_id="r1",
        invocation_id=f"inv_wall_{int(cpp_enabled)}",
        event_path=ev_path,
        budget=ExecutionBudget(wall_seconds=0.06),
    )

    with _cpp_mode(cpp_enabled):
        eng = PathFinderEngine(max_iters=80, max_expansions=50_000_000)
        t0 = time.perf_counter()
        with pytest.raises(ExecutionStopped) as exc_info:
            with session.activate():
                eng.route(pins, obstacles, cons, tech)
        elapsed = time.perf_counter() - t0
        assert exc_info.value.reason == WALL_BUDGET_EXCEEDED
        assert elapsed < 0.5, f"budget stop took {elapsed:.3f}s (expected < 0.5s)"

        # Verify engine remains completely reusable after interruption (no leaked C++ state)
        ok_pins = {
            "ok": [
                Pin("ok", "A", "d", (0.0, 0.0), m1),
                Pin("ok", "B", "d", (5.0 * P, 0.0), m1),
            ]
        }
        res_after = eng.route(ok_pins, [], [], tech)
        assert res_after.success is True
        assert res_after.nets["ok"].connected is True


def test_execution_session_cancellation_interrupts_cpp_astar(tmp_path, tech, pitch):
    """Hard Fix #1: thread-safe CancellationToken interrupts C++20 PathFinder search cleanly."""
    m1 = tech.metal_key(1)
    m2 = tech.metal_key(2)
    m3 = tech.metal_key(3)
    P = pitch
    far = 260.0 * P
    pins = {
        "n0": [
            Pin("n0", "A", "d", (0.0, 0.0), m1),
            Pin("n0", "B", "d", (far, far), m1),
        ],
    }
    obstacles = []
    for mk in (m1, m2, m3):
        obstacles.extend([
            Obstacle(far - 2.0 * P, far - 2.0 * P, far + 2.0 * P, far - 1.0 * P, mk),
            Obstacle(far - 2.0 * P, far + 1.0 * P, far + 2.0 * P, far + 2.0 * P, mk),
            Obstacle(far - 2.0 * P, far - 2.0 * P, far - 1.0 * P, far + 2.0 * P, mk),
            Obstacle(far + 1.0 * P, far - 2.0 * P, far + 2.0 * P, far + 2.0 * P, mk),
        ])
    cons = [LayerSet("n0", frozenset({m1, m2, m3}))]
    token = CancellationToken()
    session = ExecutionSession(
        project_id="test_pf_cancel",
        run_id="r1",
        invocation_id="inv_cancel",
        event_path=tmp_path / "events_cancel.jsonl",
        cancellation=token,
    )
    timer = threading.Timer(0.05, lambda: token.cancel("user_stop"))
    timer.start()
    try:
        with _cpp_mode(True):
            eng = PathFinderEngine(max_iters=80, max_expansions=50_000_000)
            t0 = time.perf_counter()
            with pytest.raises(ExecutionStopped) as exc_info:
                with session.activate():
                    eng.route(pins, obstacles, cons, tech)
            elapsed = time.perf_counter() - t0
            assert exc_info.value.reason == EXECUTION_CANCELLED
            assert elapsed < 0.5, f"cancel stop took {elapsed:.3f}s (expected < 0.5s)"
    finally:
        timer.cancel()


def test_native_rect_clipper_matches_python_subtract_rects_and_negative_coords():
    """NativeRectClipper matches via_model.subtract_rects bit-for-bit incl. negative coords."""
    from clay.layout.route.via_model import subtract_rects

    with _cpp_mode(True):
        mod = load_native("_pathfinder_cpp")
        assert mod is not None and hasattr(mod, "NativeRectClipper")
        m1 = (34, 0)
        holes = [
            (-500, -300, -100, 100),
            (0, -150, 250, 250),
            (200, 100, 600, 400),
            (900, 900, 1200, 1200),
        ]
        clipper = mod.NativeRectClipper({m1: holes})
        test_rects = [
            (-700, -500, 700, 500),
            (-200, -200, 100, 100),
            (-999, -999, -600, -600),
            (100, -100, 350, 300),
            (-455, -255, 55, 55),
        ]
        for rx0, ry0, rx1, ry1 in test_rects:
            assert clipper.subtract_rects(m1, rx0, ry0, rx1, ry1) == subtract_rects(
                (rx0, ry0, rx1, ry1), holes
            )


def test_maze_engine_relocates_colliding_230nm_adjacent_foreign_m1_pads():
    """Hard Fix #2 on MazeRouteEngine: colliding foreign M1 device pads separated by exact
    230nm min_space on 5LM 900nm grid disambiguate cleanly with zero shorts and >= 230nm spacing."""
    tech = LayoutTech.resolve("gf180mcu", metal_level="5LM")
    m1 = tuple(tech.metal_key(1))
    b_box = (280780, 232720, 281160, 233100)
    s_box = (278760, 233330, 283180, 238010)
    vdd_far = (280610, 230210, 280990, 230590)
    sig_far = (275210, 232910, 275590, 233290)
    anchor = (270000, 225000, 270380, 225380)

    def _ob(b, own):
        return Obstacle(b[0] / 1000.0, b[1] / 1000.0, b[2] / 1000.0, b[3] / 1000.0, m1, owner_net=own, kind="device")

    def _pin(net, dev, term, xy):
        return Pin(net, dev, term, (xy[0] / 1000.0, xy[1] / 1000.0), m1)

    obstacles = [_ob(b_box, "VDD"), _ob(s_box, "net0206"), _ob(vdd_far, "VDD"), _ob(sig_far, "net0206"), _ob(anchor, None)]
    pins = {
        "VDD": [_pin("VDD", "M38", "b", (280970, 232910)), _pin("VDD", "R1", "b", (280800, 230400))],
        "net0206": [_pin("net0206", "M38", "s", (280970, 233500)), _pin("net0206", "R2", "s", (275400, 233100))],
    }
    py_r, cpp_r = _run_ab(
        lambda: MazeRouteEngine(taper=True),
        pins, obstacles, [Width("VDD", 0.8), Width("net0206", 0.44)], tech,
    )
    assert cpp_r.success is True
    assert cpp_r.shorts == []
    assert cpp_r.nets["VDD"].connected is True and cpp_r.nets["net0206"].connected is True


@pytest.mark.parametrize(
    "case_name,s_box",
    [
        ("m38_exact_wide_source_and_body_tap", (278760, 233330, 283180, 238010)),
        ("m38_dual_minimal_380nm_pads", (280780, 233330, 281160, 233710)),
    ],
)
def test_adjacent_230nm_foreign_m1_pads_zero_bridging_and_finalization_clean(case_name, s_box):
    """Hard Fix #2 (Attempt 8 M38 regression): two foreign M1 device pads separated by exact
    GF180 min_space = 230nm on 5LM 900nm grid must converge short-free, maintain >= 230nm
    M1 separation, pass materialize_router_owned_metal_finalization with unique owners, and
    preserve exact C++20 vs pure-Python parity."""
    from clay.layout.route.owned_metal_finalization import (
        materialize_router_owned_metal_finalization,
    )

    tech = LayoutTech.resolve("gf180mcu", metal_level="5LM")
    m1 = tuple(tech.metal_key(1))
    b_box = (280780, 232720, 281160, 233100)  # M38.b VDD (380x380), 230nm below s_box
    vdd_far = (280610, 230210, 280990, 230590)
    sig_far = (275210, 232910, 275590, 233290)
    anchor = (270000, 225000, 270380, 225380)

    def _ob(b, own):
        return Obstacle(b[0] / 1000.0, b[1] / 1000.0, b[2] / 1000.0, b[3] / 1000.0, m1, owner_net=own, kind="device")

    def _pin(net, dev, term, xy):
        return Pin(net, dev, term, (xy[0] / 1000.0, xy[1] / 1000.0), m1)

    obstacles = [
        _ob(b_box, "VDD"),
        _ob(s_box, "net0206"),
        _ob(vdd_far, "VDD"),
        _ob(sig_far, "net0206"),
        _ob(anchor, None),
    ]
    pins = {
        "VDD": [_pin("VDD", "M38", "b", (280970, 232910)), _pin("VDD", "R1", "b", (280800, 230400))],
        "net0206": [_pin("net0206", "M38", "s", (280970, 233500)), _pin("net0206", "R2", "s", (275400, 233100))],
    }
    cons = [Width("VDD", 0.8), Width("net0206", 0.44)]

    py_r, cpp_r = _run_ab(
        lambda: PathFinderEngine(widen=True, max_iters=8),
        pins, obstacles, cons, tech,
    )
    assert cpp_r.success is True, case_name
    assert cpp_r.shorts == [], case_name
    assert cpp_r.nets["VDD"].connected is True
    assert cpp_r.nets["net0206"].connected is True

    geom, pl = {}, {}
    for name, boxes in [("M38", [b_box, s_box]), ("R1", [vdd_far]), ("R2", [sig_far])]:
        rings = [
            [(b[0] / 1000.0, b[1] / 1000.0), (b[2] / 1000.0, b[1] / 1000.0),
             (b[2] / 1000.0, b[3] / 1000.0), (b[0] / 1000.0, b[3] / 1000.0)]
            for b in boxes
        ]
        geom[name] = type("Stamp", (), {"polys_by_layer": {m1: rings}})()
        pl[name] = (0.0, 0.0)

    # Must NOT raise ValueError('assembled-metal repair has non-unique route owner')
    materialize_router_owned_metal_finalization(cpp_r, geom, pl, pins, tech)

    r_vdd = kdb.Region()
    r_sig = kdb.Region()
    r_vdd.insert(kdb.Box(*b_box))
    r_vdd.insert(kdb.Box(*vdd_far))
    r_sig.insert(kdb.Box(*s_box))
    r_sig.insert(kdb.Box(*sig_far))
    for w in cpp_r.nets["VDD"].wires:
        if (int(w.layer), int(w.datatype)) == m1:
            r_vdd.insert(kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in w.polygon.vertices]))
    for w in cpp_r.nets["net0206"].wires:
        if (int(w.layer), int(w.datatype)) == m1:
            r_sig.insert(kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in w.polygon.vertices]))
    r_vdd.merge()
    r_sig.merge()
    assert (r_vdd & r_sig).is_empty(), f"{case_name}: M1 short between VDD and net0206"
    assert (r_vdd.sized(229) & r_sig).is_empty(), f"{case_name}: M1 spacing < 230nm between VDD and net0206"

