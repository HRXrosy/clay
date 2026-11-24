"""Grade the GENERIC Clay layout path against the four probe circuits.

This harness drives the SAME generic place+route+emit path the flow's
:class:`clay.flow.step.LayoutStep` uses when a design selects ``layout =
generic`` -- i.e. the Clay-core placer (:func:`clay.layout.generic_placement.
place_generic`, real device geometry read from the bound source ``.cir``) feeding
the Clay-core general router (:func:`clay.layout.route.route_cell.route_cell` ->
``emit_routed_cell``) -- for each of the four probes (fc_ota / ldo / two_stage /
bandgap), then reports, PER CIRCUIT and HONESTLY:

  * did it PLACE + ROUTE + EMIT a GDS (and is the routed net set short-free);
  * the DRC violation count on the generic GDS (real gf180mcu KLayout deck);
  * the AREA + wire-polygon delta versus the committed hand-generator GDS.

It is EXPECTED that the first generic layout is NOT yet DRC-clean / LVS-matching
/ area-competitive with the hand generators. The point is the PRECISE per-circuit
gap (the engine roadmap), reported as-measured -- never patched per-circuit to
look better, never a fabricated DRC-clean verdict.

Usage (real Docker / KLayout for the DRC pass)::

    PYTHONPATH=. python tools/clay_layout_grade.py             # all four probes
    PYTHONPATH=. python tools/clay_layout_grade.py fc_ota ldo  # a subset
    PYTHONPATH=. python tools/clay_layout_grade.py --no-drc     # skip the DRC pass

The generic layout path takes circuit specifics ONLY from the parsed cell + the
derived intent + the bound ``.cir``; there is NO per-circuit code here.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROBES = ("fc_ota", "ldo", "two_stage", "bandgap")


# --------------------------------------------------------------------------- #
# GDS geometry measurement (bbox area + total metal-polygon area), via klayout #
# --------------------------------------------------------------------------- #
def _gds_geometry(gds_path: Path, cell_name: Optional[str] = None) -> dict:
    """Return ``{bbox_area_um2, poly_area_um2, n_polys, width_um, height_um}``.

    ``bbox_area`` is the cell bounding-box area (the floorplan footprint);
    ``poly_area`` is the summed area of EVERY drawn polygon (a wire/active-usage
    proxy). Both in um^2 using the layout's dbu. Returns zeros when the GDS holds
    no geometry (an empty placement still emits a valid but empty cell).
    """
    import klayout.db as db

    ly = db.Layout()
    ly.read(str(gds_path))
    dbu = ly.dbu
    top = None
    if cell_name is not None:
        top = ly.cell(cell_name)
    if top is None:
        tops = ly.top_cells()
        top = tops[0] if tops else None
    if top is None:
        return {
            "bbox_area_um2": 0.0,
            "poly_area_um2": 0.0,
            "n_polys": 0,
            "width_um": 0.0,
            "height_um": 0.0,
        }
    bbox = top.bbox()
    width_um = bbox.width() * dbu
    height_um = bbox.height() * dbu
    poly_area_dbu2 = 0.0
    n_polys = 0
    for li in ly.layer_indexes():
        for shape in top.shapes(li).each():
            if shape.is_polygon() or shape.is_box() or shape.is_path():
                poly = shape.polygon
                if poly is not None:
                    poly_area_dbu2 += poly.area()
                    n_polys += 1
    return {
        "bbox_area_um2": round(width_um * height_um, 4),
        "poly_area_um2": round(poly_area_dbu2 * dbu * dbu, 4),
        "n_polys": n_polys,
        "width_um": round(width_um, 4),
        "height_um": round(height_um, 4),
    }


# --------------------------------------------------------------------------- #
# Per-circuit grade row                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class GradeRow:
    key: str
    cell_name: str = ""
    placed: bool = False
    n_placed: int = 0
    solver_success: bool = False
    used_default_geometry: bool = True
    symmetry_score: float = 0.0
    routed: bool = False
    n_polys: int = 0
    routed_nets_distinct: bool = False
    n_routed_shorts: int = 0
    emitted: bool = False
    gds_path: str = ""
    # DRC (real gf180 deck)
    drc_ran: bool = False
    drc_violations: Optional[int] = None
    drc_clean: Optional[bool] = None
    drc_by_rule: dict = field(default_factory=dict)
    # area vs hand generator
    generic_geom: dict = field(default_factory=dict)
    hand_geom: dict = field(default_factory=dict)
    area_ratio_vs_hand: Optional[float] = None
    poly_area_ratio_vs_hand: Optional[float] = None
    note: str = ""
    error: str = ""

    def headline(self) -> str:
        if self.error:
            return f"{self.key}: ERROR {self.error}"
        bits = []
        bits.append("places" if self.placed else "NO-PLACE")
        bits.append("routes" if self.routed else "NO-ROUTE")
        bits.append("emits" if self.emitted else "NO-EMIT")
        stage = "+".join(bits)
        drc = (
            f"DRC {self.drc_violations} violations"
            if self.drc_violations is not None
            else ("DRC clean" if self.drc_clean else "DRC not-run")
        )
        area = (
            f"{self.area_ratio_vs_hand:.2f}x area"
            if self.area_ratio_vs_hand is not None
            else "area n/a"
        )
        short = "" if self.routed_nets_distinct else f", {self.n_routed_shorts} net short(s)"
        return f"{self.key}: {stage}, {drc}, {area}{short}"


def grade_circuit(key: str, run_drc: bool = True) -> GradeRow:
    """Run the generic place+route+emit path for one probe and grade it."""
    from clay.flow.spec import discover_registry
    from clay.flow.step import LayoutStep, gf180_klayout_decks
    from clay.layout.generic_placement import parse_cir_mos_records, place_generic
    from clay.layout.routed_cell_types import emit_routed_cell
    from clay.layout.route.route_cell import route_cell
    from clay.layout.intent import layout_intent_from_cell_def

    row = GradeRow(key=key)
    try:
        registry = discover_registry()
    except Exception as exc:  # noqa: BLE001
        row.error = f"registry discovery failed: {type(exc).__name__}: {exc}"
        return row
    spec = registry.get(key)
    if spec is None:
        row.error = f"no circuit {key!r} in the discovered registry"
        return row
    row.cell_name = spec.gds_cell_name

    cell_def = LayoutStep._parse_cell_def(spec)
    if cell_def is None:
        row.error = f"cell {spec.top_cell!r} not found under {spec.case_dir}"
        return row

    source_cir = Path(spec.source_cir) if Path(spec.source_cir).is_file() else None

    # 1) place (real device geometry from the .cir).
    intent = layout_intent_from_cell_def(cell_def)
    placement = place_generic(cell_def, intent, source_cir=source_cir)
    row.n_placed = placement.n_placed
    row.solver_success = placement.solver_success
    row.used_default_geometry = placement.used_default_geometry
    row.symmetry_score = round(placement.symmetry_score, 4)
    row.placed = placement.n_placed > 0 and placement.solver_success
    row.note = placement.note

    # 2) route + 3) emit through the GENERAL production router (route_cell), the
    # SAME engine the flow's LayoutStep uses (route_generic is retired). The net
    # binding comes from cell_def.devices when materialised, else the bound .cir
    # records -- no per-circuit branch.
    records = parse_cir_mos_records(source_cir) if source_cir else {}

    # grading reports the honest residual (shorts/opens), so it must RECEIVE the
    # cell even when incomplete -> opt out of the net-completeness gate's raise.
    routed = route_cell(
        placement, cell_def, intent, spec.pdk,
        require_complete=False, device_records=records,
    ).to_routed_cell()
    row.n_polys = len(routed.polys)
    row.routed = row.n_polys > 0
    shorts = routed.shorts
    row.routed_nets_distinct = bool(getattr(shorts, "distinct", False))
    row.n_routed_shorts = len(getattr(shorts, "merged_pairs", []) or [])

    work = Path(tempfile.mkdtemp(prefix=f"clay_grade_{key}_"))
    gds = work / f"{spec.gds_cell_name}.generic.gds"
    emit_routed_cell(routed, str(gds), cell_name=spec.gds_cell_name)
    row.emitted = gds.is_file()
    row.gds_path = str(gds)

    # geometry of the generic GDS + the committed hand-generator GDS.
    try:
        row.generic_geom = _gds_geometry(gds, spec.gds_cell_name)
    except Exception as exc:  # noqa: BLE001
        row.generic_geom = {"error": f"{type(exc).__name__}: {exc}"}
    hand_gds = Path(spec.fallback_gds)
    if hand_gds.is_file():
        try:
            row.hand_geom = _gds_geometry(hand_gds, spec.gds_cell_name)
        except Exception as exc:  # noqa: BLE001
            row.hand_geom = {"error": f"{type(exc).__name__}: {exc}"}
    g_area = row.generic_geom.get("bbox_area_um2") or 0.0
    h_area = row.hand_geom.get("bbox_area_um2") or 0.0
    if h_area > 0 and g_area > 0:
        row.area_ratio_vs_hand = round(g_area / h_area, 3)
    g_poly = row.generic_geom.get("poly_area_um2") or 0.0
    h_poly = row.hand_geom.get("poly_area_um2") or 0.0
    if h_poly > 0 and g_poly > 0:
        row.poly_area_ratio_vs_hand = round(g_poly / h_poly, 3)

    # 4) DRC on the generic GDS (real gf180 deck).
    if run_drc and row.emitted and row.n_polys > 0:
        try:
            from tools.klayout_drc_signoff_v139.run import run_gf180_drc

            report = run_gf180_drc(
                gds,
                report_path=work / f"{key}.generic.lyrdb",
                deck=gf180_klayout_decks()["drc"],
                topcell=spec.gds_cell_name,
                timeout_sec=600,
            )
            row.drc_ran = True
            row.drc_violations = report.total_violations
            row.drc_clean = report.is_clean()
            row.drc_by_rule = report.counts_by_rule()
        except Exception as exc:  # noqa: BLE001
            row.note = (row.note + f" | DRC failed: {type(exc).__name__}: {exc}").strip()
    return row


def run_grade(keys=PROBES, run_drc: bool = True) -> list[GradeRow]:
    return [grade_circuit(k, run_drc=run_drc) for k in keys]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuits", nargs="*", default=list(PROBES),
                        help="probe keys to grade (default: all four)")
    parser.add_argument("--no-drc", action="store_true",
                        help="skip the real-deck DRC pass (place+route+emit + area only)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the full grade rows to this JSON path")
    args = parser.parse_args(argv)

    keys = args.circuits or list(PROBES)
    rows = run_grade(keys, run_drc=not args.no_drc)

    print("=" * 78)
    print("GENERIC Clay layout path -- per-probe grade (honest gap report)")
    print("=" * 78)
    for row in rows:
        print(row.headline())
        if not row.error:
            print(
                f"    placed={row.n_placed} (solver_ok={row.solver_success}, "
                f"real_geom={not row.used_default_geometry}, "
                f"symmetry={row.symmetry_score}); polys={row.n_polys}; "
                f"nets_distinct={row.routed_nets_distinct}"
            )
            gg, hg = row.generic_geom, row.hand_geom
            print(
                f"    generic bbox={gg.get('bbox_area_um2')} um^2 "
                f"({gg.get('width_um')}x{gg.get('height_um')}); "
                f"hand bbox={hg.get('bbox_area_um2')} um^2; "
                f"poly_area_ratio={row.poly_area_ratio_vs_hand}"
            )
            if row.drc_by_rule:
                top_rules = sorted(row.drc_by_rule.items(), key=lambda kv: -kv[1])[:6]
                print("    top DRC rules: " + ", ".join(f"{r}={c}" for r, c in top_rules))
    print("=" * 78)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
