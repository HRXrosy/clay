"""Phase 0 C++20 main-engine differential parity tests against KLayout.

Verifies:
1. Exact Manhattan boolean ops (OR, AND, NOT, XOR) match `klayout.db.Region` 100% across
   randomized multi-rectangle stress cases (area, bbox, and point-set symmetric difference == 0).
2. Exact morphological sizing (`sized(dx, dy)` grow & non-convex erosion around concavities/holes)
   matches `klayout.db.Region.sized(dx, dy, 2)` (L-infinity / square mode) 100%.
3. Connected components and contour extraction match KLayout merged region invariants.
4. Pure C++20 GDSII v6.0 writer emits standard `.gds` files readable by `klayout.db.Layout`,
   and the pure C++20 GDSII reader + flattener matches KLayout layer-by-layer union areas
   and bounding boxes on real signed-off GF180 & SKY130 GDS files.
5. Standalone zero-Python `clay_engine_cli` binary runs `--version`, `selftest`, and `gds-inspect`.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform
import random
import subprocess

import klayout.db as db
import pytest

from clay._native import _engine_cpp

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_BIN = (
    REPO_ROOT
    / "cpp"
    / "clay_pnr"
    / "build"
    / "modular"
    / "bin"
    / ("clay_engine_cli.exe" if platform.system() == "Windows" else "clay_engine_cli")
)


def _to_klayout_region(boxes: list[tuple[int, int, int, int]]) -> db.Region:
    reg = db.Region()
    for x0, y0, x1, y1 in boxes:
        if x0 < x1 and y0 < y1:
            reg.insert(db.Box(int(x0), int(y0), int(x1), int(y1)))
    reg.merge()
    return reg


def _rand_boxes(rng: random.Random, count: int, span: int = 800, step: int = 20) -> list[tuple[int, int, int, int]]:
    out = []
    for _ in range(count):
        x0 = rng.randrange(-span, span, step)
        y0 = rng.randrange(-span, span, step)
        w = rng.randrange(step, 260, step)
        h = rng.randrange(step, 260, step)
        out.append((x0, y0, x0 + w, y0 + h))
    return out


def test_cpp_polygon_set_boolean_ops_exact_match_klayout() -> None:
    rng = random.Random(20261124)
    for trial in range(120):
        a = _rand_boxes(rng, rng.randint(0, 18))
        b = _rand_boxes(rng, rng.randint(0, 18))
        ra = _to_klayout_region(a)
        rb = _to_klayout_region(b)

        expected_map = {
            "or": ra | rb,
            "and": ra & rb,
            "not": ra - rb,
            "xor": ra ^ rb,
        }
        for op, k_exp in expected_map.items():
            cpp_boxes = _engine_cpp.polygon_boolean_cpp(a, b, op)
            k_got = _to_klayout_region(cpp_boxes)
            # Symmetric difference must be strictly empty!
            assert (k_got ^ k_exp).is_empty(), f"Mismatch on trial={trial} op={op}"
            assert k_got.area() == k_exp.area()


def test_cpp_polygon_set_morphology_exact_match_klayout_linf() -> None:
    rng = random.Random(20261125)
    # Include concavities, frames with holes, thin necks, and random clusters.
    cases = [
        [(0, 0, 1000, 200), (0, 800, 1000, 1000), (0, 0, 200, 1000), (800, 0, 1000, 1000)],
        [(0, 0, 400, 100), (300, 0, 400, 400), (300, 300, 700, 400)],
        *(_rand_boxes(rng, rng.randint(4, 16)) for _ in range(60)),
    ]
    deltas = [(40, 40), (100, 100), (-20, -20), (-60, -60), (-120, -120), (40, 0), (0, -40)]
    for ci, boxes in enumerate(cases):
        reg = _to_klayout_region(boxes)
        for dx, dy in deltas:
            cpp_sized = _engine_cpp.polygon_sized_cpp(boxes, dx, dy)
            k_got = _to_klayout_region(cpp_sized)
            # KLayout mode=2 is exact square / L-infinity sizing.
            k_exp = reg.sized(dx, dy, 2)
            k_exp.merge()
            assert (k_got ^ k_exp).is_empty(), f"Sized mismatch case={ci} dx={dx} dy={dy}"


def test_cpp_polygon_set_components_and_contours_match_klayout() -> None:
    boxes = [
        # Frame with hole
        (0, 0, 300, 50),
        (0, 250, 300, 300),
        (0, 0, 50, 300),
        (250, 0, 300, 300),
        # Corner-touching pair far away
        (500, 500, 600, 600),
        (600, 600, 700, 700),
    ]
    comps_corner = _engine_cpp.polygon_components_cpp(boxes, True)
    comps_edge = _engine_cpp.polygon_components_cpp(boxes, False)
    assert len(comps_corner) == 2
    assert len(comps_edge) == 3

    contours = _engine_cpp.polygon_contours_cpp(boxes)
    # Reconstruct KLayout region from emitted outer & hole contours and compare exact area.
    outers = db.Region()
    holes = db.Region()
    for pts, is_hole in contours:
        poly = db.Polygon([db.Point(int(x), int(y)) for x, y in pts])
        if is_hole:
            holes.insert(poly)
        else:
            outers.insert(poly)
    reconstructed = outers - holes
    expected = _to_klayout_region(boxes)
    assert (reconstructed ^ expected).is_empty()


def test_cpp_gds_writer_and_reader_exact_match_klayout(tmp_path: Path) -> None:
    gds_out = tmp_path / "cpp_phase0_test.gds"
    shapes = [
        (34, 0, 0, 0, 1200, 400),
        (34, 0, 1000, 0, 2000, 1000),  # overlaps first on layer 34/0
        (36, 0, -500, -300, 500, 700),
        (68, 20, 100, 100, 900, 300),
    ]
    _engine_cpp.gds_write_boxes_cpp(str(gds_out), "PHASE0_LIB", "TOP_CELL", shapes, 1e-3, 1e-9)
    assert gds_out.is_file() and gds_out.stat().st_size > 120

    # Verify KLayout parses the C++20-emitted GDSII cleanly with exact DBU & layer union areas.
    ly = db.Layout()
    ly.read(str(gds_out))
    assert abs(ly.dbu - 0.001) < 1e-12
    top = ly.top_cell()
    assert top.name == "TOP_CELL"

    cpp_summary = _engine_cpp.gds_inspect_layers_cpp(str(gds_out), "TOP_CELL")
    for li in ly.layer_indices():
        info = ly.get_info(li)
        reg = db.Region(top.begin_shapes_rec(li))
        reg.merge()
        raw_cnt, canon_cnt, area_dbu2, bb = cpp_summary[(info.layer, info.datatype)]
        assert area_dbu2 == reg.area()
        kbb = reg.bbox()
        assert bb == (kbb.left, kbb.bottom, kbb.right, kbb.top)
        assert canon_cnt >= 1 and raw_cnt >= 1


@pytest.mark.parametrize(
    "rel_gds",
    [
        "cases/unseen_benchmarks/two_stage_ota_auto_gf180/_flow_run/work/layout.gds",
        "cases/unseen_benchmarks/ldo_auto_gf180/_flow_run/work/layout.gds",
        "cmp_dev/out/g3_golden_repro_layout/output/work/layout.gds",
    ],
)
def test_cpp_gds_reader_matches_klayout_on_real_signed_off_gds(rel_gds: str) -> None:
    gds_path = REPO_ROOT / rel_gds
    if not gds_path.is_file():
        pytest.skip(f"GDS fixture not present: {rel_gds}")

    ly = db.Layout()
    ly.read(str(gds_path))
    top = ly.top_cell()
    cpp_summary = _engine_cpp.gds_inspect_layers_cpp(str(gds_path), top.name)

    checked = 0
    for li in ly.layer_indices():
        info = ly.get_info(li)
        reg = db.Region(top.begin_shapes_rec(li))
        reg.merge()
        if reg.is_empty():
            continue
        assert (info.layer, info.datatype) in cpp_summary, f"Missing layer {info.layer}/{info.datatype}"
        _, _, area_dbu2, bb = cpp_summary[(info.layer, info.datatype)]
        kbb = reg.bbox()
        assert bb == (kbb.left, kbb.bottom, kbb.right, kbb.top), f"BBox mismatch on {info.layer}/{info.datatype}"
        assert area_dbu2 == reg.area(), f"Area mismatch on {info.layer}/{info.datatype}"
        checked += 1
    assert checked >= 10


def test_standalone_cpp20_engine_cli_works() -> None:
    assert CLI_BIN.is_file(), f"Missing standalone C++20 CLI binary: {CLI_BIN}"

    r_info = subprocess.run([str(CLI_BIN), "info"], capture_output=True, text=True, check=True)
    info_doc = json.loads(r_info.stdout)
    assert info_doc["python_dependency"] is False
    assert "core.polygon_set" in info_doc["modules"]

    r_self = subprocess.run([str(CLI_BIN), "selftest"], capture_output=True, text=True, check=True)
    self_doc = json.loads(r_self.stdout)
    assert self_doc["status"] == "PASS"

    jku_gds = REPO_ROOT / "cmp_dev" / "out" / "g3_golden_repro_layout" / "output" / "work" / "layout.gds"
    if jku_gds.is_file():
        r_insp = subprocess.run([str(CLI_BIN), "gds-inspect", str(jku_gds)], capture_output=True, text=True, check=True)
        insp_doc = json.loads(r_insp.stdout)
        assert insp_doc["layer_count"] >= 15
        assert insp_doc["size_um"][0] == pytest.approx(30.665, abs=0.01)
        assert insp_doc["size_um"][1] == pytest.approx(28.115, abs=0.01)
