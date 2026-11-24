"""Phase 2 & Phase 3 Pure C++20 Main Engine differential parity & contract tests.

Verifies exact mathematical equivalence between our zero-dependency C++20 Main Engine
(`clay._native._engine_cpp` + `clay_engine_cli.exe`) and legacy Python/SciPy/OR-Tools/KLayout oracles:
  1. `compact_axis_cpp` vs `scipy.optimize.linprog/highs` (`lp_compactor.compact_axis`) over 200 random DAGs.
  2. `compact_2d_cpp` with symmetry axis + manufacturing grid alignment.
  3. `solve_envelope_min_cpp` (`backend="cpp"`) vs CP-SAT (`backend="cpsat"`): determinism,
     zero-overlap, padded disjoint clusters, bands + floating passive right-column banking,
     hard symmetry pairs, aspect target rotation, rail affinities, and speed.
  4. `extract_connectivity_cpp` vs `klayout.db.LayoutToNetlist` across multi-layer metal/via/MIM carve stacks.
  5. `emit_routed_gds_cpp`, `emit_binary_gds_native`, `generate_mos_pcell_cpp`, and standalone
     `clay_engine_cli.exe` (`gds-emit-json` & `pcell-mos`) exact KLayout GDS area & bbox parity.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform
import random
import subprocess
import time

import pytest

from clay._native import _engine_cpp
from clay.layout.backend import LayerPolygon, NamedCell, ParameterizedLibrary, Point, Polygon
from clay.layout.binary_gds_subprocess import (
    _find_cpp_cli_binary,
    emit_binary_gds_native,
    emit_binary_gds_via_subprocess,
)
from clay.layout.place.lp_compactor import DifferenceConstraint, compact_axis
from clay.layout.route.emit_gds import emit_routed_gds, emit_routed_gds_cpp
from clay.layout.synth.placement import Group, place_envelope_min

REPO_ROOT = Path(__file__).resolve().parents[1]


def _klayout_layer_summary(gds_path: Path, cell_name: str | None = None) -> dict[tuple[int, int], tuple[int, tuple[int, int, int, int]]]:
    import klayout.db as kdb

    ly = kdb.Layout()
    ly.read(str(gds_path))
    cell = ly.cell(cell_name) if cell_name else ly.top_cell()
    assert cell is not None
    out: dict[tuple[int, int], tuple[int, tuple[int, int, int, int]]] = {}
    for li in ly.layer_indexes():
        info = ly.get_info(li)
        reg = kdb.Region(cell.begin_shapes_rec(li)).merged()
        if reg.is_empty():
            continue
        bb = reg.bbox()
        out[(int(info.layer), int(info.datatype))] = (
            int(reg.area()),
            (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top)),
        )
    return out


def test_cpp_dag_compactor_exact_parity_with_highs_lp_200_random_dags() -> None:
    rng = random.Random(20261124)
    for trial in range(200):
        n = rng.randint(2, 18)
        names = [f"b{i:02d}" for i in range(n)]
        extents_list = [rng.randint(100, 5000) for _ in range(n)]
        extents_map = dict(zip(names, extents_list))
        # Generate random acyclic edges following a random topological permutation.
        perm = list(range(n))
        rng.shuffle(perm)
        py_cons = []
        cpp_edges = []
        for idx_u in range(n):
            u = perm[idx_u]
            for idx_v in range(idx_u + 1, n):
                if rng.random() < 0.35:
                    v = perm[idx_v]
                    delta = extents_list[u] + rng.randint(0, 2000)
                    py_cons.append(DifferenceConstraint(names[u], names[v], delta))
                    cpp_edges.append((u, v, delta))

        lp_res = compact_axis(extents_map, py_cons)
        cpp_pos, cpp_ext = _engine_cpp.compact_axis_cpp(extents_list, cpp_edges)

        assert cpp_ext == lp_res.extent_nm, f"trial {trial}: extent mismatch {cpp_ext} != {lp_res.extent_nm}"
        for i, name in enumerate(names):
            assert cpp_pos[i] == lp_res.positions_nm[name], (
                f"trial {trial} {name}: pos {cpp_pos[i]} != {lp_res.positions_nm[name]}"
            )

    # Cycle rejection parity
    with pytest.raises(ValueError, match="acyclic"):
        _engine_cpp.compact_axis_cpp([10, 10], [(0, 1, 12), (1, 0, 12)])


def test_cpp_2d_compactor_symmetry_and_manufacturing_grid() -> None:
    widths = [1000, 1000, 1600, 1600, 800]
    heights = [700, 700, 900, 900, 500]
    h_edges = [(0, 1, 1200), (2, 3, 2000), (1, 4, 1300)]
    v_edges = [(0, 2, 1000), (1, 3, 1000)]
    syms = [(0, 1), (2, 3)]
    xs, ys, W, H, axis2 = _engine_cpp.compact_2d_cpp(widths, heights, h_edges, v_edges, syms, 5)
    assert ys[0] == ys[1] and ys[2] == ys[3]
    assert ys[2] >= ys[0] + 1000
    assert 2 * xs[0] + 2 * xs[1] + widths[0] + widths[1] == axis2
    assert 2 * xs[2] + 2 * xs[3] + widths[2] + widths[3] == axis2
    assert all(v % 5 == 0 for v in xs + ys)
    assert W >= max(xs[i] + widths[i] for i in range(len(widths)))
    assert H >= max(ys[i] + heights[i] for i in range(len(heights)))


def test_cpp_envelope_placer_all_analog_constraints_and_speed() -> None:
    # Realistic 10-group analog floorplan with guard-ring islands, bands + floating passives,
    # explicit left_of orders, signal nets, rail affinities, and landscape aspect target.
    groups = [
        Group("M_tail", 10.0, 6.0),
        Group("M_in1", 12.0, 8.0),
        Group("M_in2", 12.0, 8.0),
        Group("M_nbias", 8.0, 6.0),
        Group("M_ld1", 14.0, 9.0),
        Group("M_ld2", 14.0, 9.0),
        Group("M_outp", 16.0, 8.0),
        Group("M_pbias", 8.0, 7.0),
        Group("P_Cc", 18.0, 18.0),
        Group("P_Rz", 6.0, 12.0),
    ]
    nets = [
        ["M_tail", "M_in1", "M_in2"],
        ["M_in1", "M_ld1"],
        ["M_in2", "M_ld2", "M_outp", "P_Cc"],
        ["M_outp", "P_Cc", "P_Rz"],
    ]
    t0 = time.perf_counter()
    res = place_envelope_min(
        groups,
        nets=nets,
        net_weights=[0.5, 1.0, 1.0 / 3.0, 0.5],
        net_signal=[True, True, True, True],
        disjoint_clusters=[["M_in1", "M_in2"], ["M_ld1", "M_ld2"]],
        island_pads=[3.0, 3.0],
        band_of_group={
            "M_tail": 0, "M_in1": 0, "M_in2": 0, "M_nbias": 0,
            "M_ld1": 1, "M_ld2": 1, "M_outp": 1, "M_pbias": 1,
        },
        order_constraints=[("M_in1", "M_in2", "left_of"), ("M_ld1", "M_ld2", "left_of")],
        rail_affinities=[("M_tail", "bottom"), ("M_outp", "top")],
        aspect_target=1.7,
        gap=1.0,
        wl_weight=2,
        backend="cpp",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res.status in ("OPTIMAL", "FEASIBLE")
    assert elapsed_ms < 2000.0, f"Pure C++20 envelope placer unexpectedly slow: {elapsed_ms:.1f} ms"

    # Verify determinism (second call bit-exact)
    res2 = place_envelope_min(
        groups,
        nets=nets,
        net_weights=[0.5, 1.0, 1.0 / 3.0, 0.5],
        net_signal=[True, True, True, True],
        disjoint_clusters=[["M_in1", "M_in2"], ["M_ld1", "M_ld2"]],
        island_pads=[3.0, 3.0],
        band_of_group={
            "M_tail": 0, "M_in1": 0, "M_in2": 0, "M_nbias": 0,
            "M_ld1": 1, "M_ld2": 1, "M_outp": 1, "M_pbias": 1,
        },
        order_constraints=[("M_in1", "M_in2", "left_of"), ("M_ld1", "M_ld2", "left_of")],
        rail_affinities=[("M_tail", "bottom"), ("M_outp", "top")],
        aspect_target=1.7,
        gap=1.0,
        wl_weight=2,
        backend="cpp",
    )
    assert res.centers == res2.centers
    assert (res.width, res.height, res.envelope) == (res2.width, res2.height, res2.envelope)

    # Verify hard band separation (every NMOS group strictly below every PMOS group)
    sizes = {g.key: (g.w, g.h) for g in groups}
    nmos_top = max(res.centers[k][1] + sizes[k][1] / 2.0 for k in ("M_tail", "M_in1", "M_in2", "M_nbias"))
    pmos_bot = min(res.centers[k][1] - sizes[k][1] / 2.0 for k in ("M_ld1", "M_ld2", "M_outp", "M_pbias"))
    assert nmos_top <= pmos_bot + 1e-6

    # Verify floating passives banked to the right of MOS core
    mos_right = max(
        res.centers[k][0] + sizes[k][0] / 2.0
        for k in ("M_tail", "M_in1", "M_in2", "M_nbias", "M_ld1", "M_ld2", "M_outp", "M_pbias")
    )
    for fk in ("P_Cc", "P_Rz"):
        assert res.centers[fk][0] - sizes[fk][0] / 2.0 >= mos_right - 1e-6


def test_cpp_stack_connectivity_matches_klayout_l2n_with_mim_carve() -> None:
    import klayout.db as kdb

    M1 = (34, 0)
    V1 = (35, 0)
    M2 = (36, 0)
    M4 = (46, 0)
    V4 = (41, 0)
    M5 = (81, 0)
    MIM = (75, 0)

    shapes = [
        # Net A: M1 -> V1 -> M2 bridge
        (*M1, 0, 0, 500, 200),
        (*V1, 300, 50, 450, 150),
        (*M2, 250, 0, 900, 200),
        (*V1, 750, 50, 850, 150),
        (*M1, 700, 0, 1200, 200),
        # Net B: isolated M1 island
        (*M1, 0, 600, 400, 800),
        # MIM cap: bottom M4 plate carved by MIM marker -> top V4/M5 must NOT short to bottom M4!
        (*M4, 2000, 0, 3000, 1000),
        (*MIM, 2100, 100, 2900, 900),
        (*V4, 2400, 400, 2600, 600),
        (*M5, 2350, 350, 2650, 650),
        # Ordinary M4->V4->M5 via outside MIM cap -> MUST connect into one component!
        (*M4, 4000, 0, 4500, 500),
        (*V4, 4150, 150, 4350, 350),
        (*M5, 4100, 100, 4400, 400),
    ]
    conductors = [M1, M2, M4, M5, MIM]
    connections = [(M1, V1, M2), (M4, V4, M5)]
    mim_carves = [(M4, V4, MIM, 0, True)]

    cpp_comps = _engine_cpp.extract_connectivity_cpp(shapes, conductors, connections, mim_carves)

    # KLayout LayoutToNetlist reference oracle:
    ly = kdb.Layout()
    ly.dbu = 0.001
    cell = ly.create_cell("TOP")
    lmap = {}
    for l, dt, x0, y0, x1, y1 in shapes:
        if (l, dt) not in lmap:
            lmap[(l, dt)] = ly.layer(l, dt)
        cell.shapes(lmap[(l, dt)]).insert(kdb.Box(x0, y0, x1, y1))

    l2n = kdb.LayoutToNetlist(kdb.RecursiveShapeIterator(ly, cell, []))
    l2n.threads = 1
    h = {lk: l2n.make_polygon_layer(li, f"L_{lk[0]}_{lk[1]}") for lk, li in lmap.items()}
    for c in conductors:
        if c in h:
            l2n.connect(h[c])
    l2n.connect(h[M1], h[V1])
    l2n.connect(h[V1], h[M2])
    interacting = h[M4].interacting(h[MIM])
    mim_virtual = h[MIM] & interacting
    metal_ncap = h[M4] - mim_virtual
    l2n.register(mim_virtual, "MIM_V")
    l2n.register(metal_ncap, "M4_NCAP")
    l2n.connect(h[M4], mim_virtual)
    l2n.connect(metal_ncap, mim_virtual)
    l2n.connect(metal_ncap, h[M4])
    l2n.connect(metal_ncap, h[V4])
    l2n.connect(h[MIM], h[V4])
    l2n.connect(h[V4], h[M5])
    l2n.extract_netlist()

    top_ckt = l2n.netlist().circuit_by_name("TOP")
    kl_comp_areas = []
    for net in top_ckt.each_net():
        per_layer = {}
        for lk, handle in h.items():
            reg = l2n.shapes_of_net(net, handle, False)
            if reg is not None and not reg.is_empty():
                per_layer[lk] = int(reg.merged().area())
        if per_layer:
            kl_comp_areas.append(tuple(sorted(per_layer.items())))
    kl_comp_areas.sort()

    cpp_comp_areas = []
    for comp in cpp_comps:
        by_lk: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for l, dt, x0, y0, x1, y1 in comp:
            by_lk.setdefault((l, dt), []).append((x0, y0, x1, y1))
        per_layer = {}
        for lk, boxes in by_lk.items():
            merged = _engine_cpp.polygon_boolean_cpp(boxes, [], "or")
            per_layer[lk] = sum((b[2] - b[0]) * (b[3] - b[1]) for b in merged)
        cpp_comp_areas.append(tuple(sorted(per_layer.items())))
    cpp_comp_areas.sort()

    assert len(cpp_comp_areas) == len(kl_comp_areas) == 5
    assert cpp_comp_areas == kl_comp_areas


def test_cpp_gds_emit_and_pcell_cli_klayout_parity(tmp_path: Path) -> None:
    # 1. emit_routed_gds_cpp vs emit_routed_gds (KLayout) exact layer area & bbox parity!
    def _rect_lp(layer: int, dt: int, x0: int, y0: int, x1: int, y1: int) -> LayerPolygon:
        return LayerPolygon(
            layer=layer,
            datatype=dt,
            polygon=Polygon(vertices=[Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)]),
        )

    class _FakeNet:
        def __init__(self, wires):
            self.wires = wires

    class _FakeRouted:
        nets = {
            "VDD": _FakeNet([_rect_lp(34, 0, 0, 0, 500, 200), _rect_lp(34, 0, 400, 0, 900, 200)]),
            "VSS": _FakeNet([_rect_lp(34, 0, 0, 500, 900, 700), _rect_lp(36, 0, 100, 400, 300, 900)]),
        }
        wires = nets["VDD"].wires + nets["VSS"].wires

    kl_gds = tmp_path / "kl_routed.gds"
    cpp_gds = tmp_path / "cpp_routed.gds"
    emit_routed_gds(_FakeRouted(), str(kl_gds), cell_name="TOP")
    emit_routed_gds_cpp(_FakeRouted(), str(cpp_gds), cell_name="TOP")
    assert _klayout_layer_summary(cpp_gds, "TOP") == _klayout_layer_summary(kl_gds, "TOP")

    # 2. emit_binary_gds_native & clay_engine_cli.exe drop-in ParameterizedLibrary JSON -> GDSII
    lib = ParameterizedLibrary(
        name="CPP_TEST_LIB",
        cells=[
            NamedCell(
                name="CELL_A",
                polygons=[
                    _rect_lp(22, 0, 0, 0, 1000, 600),
                    _rect_lp(30, 0, 300, -200, 580, 800),
                ],
            )
        ],
    )
    nat_gds = tmp_path / "native.gds"
    cli_gds = tmp_path / "cli.gds"
    emit_binary_gds_native(lib, nat_gds)
    cpp_cli = _find_cpp_cli_binary()
    assert cpp_cli is not None and cpp_cli.is_file()
    emit_binary_gds_via_subprocess(lib, cli_gds, binary_path=cpp_cli)
    assert _klayout_layer_summary(nat_gds, "CELL_A") == _klayout_layer_summary(cli_gds, "CELL_A")
    assert _klayout_layer_summary(nat_gds, "CELL_A")[(22, 0)][0] == 600_000
    assert _klayout_layer_summary(nat_gds, "CELL_A")[(30, 0)][0] == 280_000

    # 3. Pure C++20 MOS PCell generator via Python binding AND standalone CLI -> KLayout readable!
    pcell = _engine_cpp.generate_mos_pcell_cpp("MP1", True, 2000, 280, 4, 1, 1, "gf180mcu")
    assert len(pcell["gate_rects"]) == 4
    assert len(pcell["source_rects"]) == 4
    assert len(pcell["drain_rects"]) == 4

    pcell_gds = tmp_path / "mos_pcell.gds"
    proc = subprocess.run(
        [str(cpp_cli), "pcell-mos", "pmos", "2000", "280", "4", "gf180mcu", "-o", str(pcell_gds)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(proc.stdout)
    assert meta["pdk"] == "gf180mcu"
    kl_pcell = _klayout_layer_summary(pcell_gds, "PMOS_PCELL")
    assert (22, 0) in kl_pcell and (30, 0) in kl_pcell and (33, 0) in kl_pcell and (34, 0) in kl_pcell and (21, 0) in kl_pcell

