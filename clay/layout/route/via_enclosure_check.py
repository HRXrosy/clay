"""VIA METAL-ENCLOSURE obligation -- an INDEPENDENT checker read from emitted GDS.

THE OBLIGATION
==============
For every via cut drawn on a cut layer (contact / via1 / via2 / ...), the routed
metal landing on each adjacent routing metal must OVERHANG the cut by >= the PDK
enclosure on every side. This is the positive contract the router owes after it
drops a cut; it is named here as a checkable obligation so a breakage is a RED
verdict, not a silent geometric defect.

INDEPENDENCE (checker not the checked)
======================================
This checker does NOT consult the router's internal state. It reads the EMITTED
GDS (the canonical artifact every oracle reads), resolves the cut + adjacent metal
layers + the per-side enclosure from the PDK ``tech`` contract
(:mod:`clay.layout.route.via_model`), and measures coverage with klayout region
geometry -- the same geometry the foundry DRC deck measures. It mirrors the
foundry check ``metal.enclosing(cut, e)``: a cut whose surrounding metal does not
extend ``e`` past every cut edge fires. The verdict aligns with the real gf180
DRC V1.3*/V1.4*/V2.3*/V2.4* enclosure rules; a divergence between this checker and
the foundry oracle is a bug in this checker, never grounds to trust it over the
oracle.

GENERIC, NO PER-PDK BRANCH
==========================
The cut/metal layer pairing and the enclosure values come entirely from the PDK
(``tech.via_key`` / ``tech.metal_key`` / ``tech.via_rule``). A PDK with a different
stack or looser enclosure is checked by the same code with its own numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from clay.layout.route.via_model import via_enclosure_nm, via_cut_size_nm

_LayerKey = Tuple[int, int]


@dataclass
class ViaEnclosureViolation:
    """One via cut whose metal landing under-encloses it on a side.

    ``side`` is ``"below"`` (lower metal) or ``"above"`` (upper metal); ``cut_xy_um``
    is the cut centroid (um); ``required_nm`` is the rule enclosure; ``cut_layer`` /
    ``metal_layer`` are the GDS keys involved."""

    via_name: str
    side: str
    cut_layer: _LayerKey
    metal_layer: _LayerKey
    required_nm: int
    cut_xy_um: Tuple[float, float]


@dataclass
class ViaEnclosureReport:
    """Result of :func:`check_via_enclosure` on one GDS.

    ``status`` is ``"pass"`` (no firing), ``"fail"`` (>=1 firing), or
    ``"unmeasurable"`` (no cut layers / klayout unavailable -- never silently
    pass). ``violations`` lists every under-enclosed (cut, side). ``n_cuts``
    counts the cuts examined; ``per_rule`` is ``{f"{via}.{side}": count}``."""

    status: str = "unmeasurable"
    violations: List[ViaEnclosureViolation] = field(default_factory=list)
    n_cuts: int = 0
    per_rule: Dict[str, int] = field(default_factory=dict)
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.violations)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _adjacent_metal_keys(tech: Any, via_n: int) -> Optional[Tuple[_LayerKey, _LayerKey]]:
    """The (lower, upper) routing-metal GDS keys a via at index ``via_n`` joins.

    ``via_n == 0`` (contact) joins the diffusion/poly stack to metal1; its LOWER
    landing is not a routing metal (it is comp/poly), so contact is handled by the
    metal1 side only -- this returns ``None`` for contact and the caller skips its
    below side (the comp/poly contact enclosure CO.4 is a device-cell concern, not
    a routed-wire landing)."""
    if via_n < 1:
        return None
    try:
        lower = tuple(tech.metal_key(via_n))
        upper = tuple(tech.metal_key(via_n + 1))
    except Exception:
        return None
    return lower, upper


def _via_name_for_index(tech: Any, via_n: int) -> Optional[str]:
    """The PDK via NAME for cut index ``via_n`` (``via1`` / ``vi1`` / ``v1`` ...)."""
    for cand in (f"via{via_n}", f"vi{via_n}", f"v{via_n}"):
        try:
            names = set(tech.via_names())
        except Exception:
            names = set()
        if cand in names:
            return cand
    return None


def check_via_enclosure(
    gds_path: str,
    tech: Any,
    *,
    top_cell: Optional[str] = None,
    max_via_index: int = 6,
) -> ViaEnclosureReport:
    """Measure the via metal-enclosure obligation on ``gds_path`` against ``tech``.

    For each routing via (``via1`` .. ``via{max_via_index}``) the PDK declares: find
    its cut polygons in the GDS and, on each adjacent routing metal, flag every cut
    the metal does not enclose by >= the PDK enclosure on all sides. Mirrors the
    foundry ``metal.enclosing(cut, e)`` rule with klayout region geometry.

    Returns a :class:`ViaEnclosureReport`. ``unmeasurable`` (never silent pass) when
    klayout is unavailable or no via cut layers are present.
    """
    try:
        import klayout.db as kdb
    except Exception as exc:  # pragma: no cover - klayout always present in CI
        return ViaEnclosureReport(status="unmeasurable", reason=f"klayout import: {exc}")

    ly = kdb.Layout()
    try:
        ly.read(gds_path)
    except Exception as exc:
        return ViaEnclosureReport(status="unmeasurable", reason=f"read {gds_path}: {exc}")

    if top_cell is not None:
        cell = ly.cell(top_cell)
        if cell is None:
            return ViaEnclosureReport(
                status="unmeasurable", reason=f"top cell {top_cell!r} not found"
            )
    else:
        tops = ly.top_cells()
        if not tops:
            return ViaEnclosureReport(status="unmeasurable", reason="no top cell")
        cell = tops[0]

    dbu = ly.dbu  # um per dbu
    nm_per_dbu = 1000.0 * dbu  # nm per dbu (== 1.0 when dbu == 0.001)

    def _region(key: _LayerKey):
        li = ly.find_layer(int(key[0]), int(key[1]))
        if li is None:
            return None
        return kdb.Region(cell.begin_shapes_rec(li))

    report = ViaEnclosureReport(violations=[])
    examined_any_layer = False

    for via_n in range(1, max_via_index + 1):
        try:
            cut_key = tuple(tech.via_key(via_n))
        except Exception:
            continue
        adj = _adjacent_metal_keys(tech, via_n)
        if adj is None:
            continue
        cut_reg = _region(cut_key)
        if cut_reg is None or cut_reg.is_empty():
            continue
        examined_any_layer = True
        via_name = _via_name_for_index(tech, via_n)
        if via_name is None:
            continue
        rb, ra = via_enclosure_nm(tech, via_name)
        lower_key, upper_key = adj
        n_cuts_here = cut_reg.count()
        report.n_cuts += n_cuts_here

        for (side, enc_nm, mkey) in (
            ("below", rb, lower_key),
            ("above", ra, upper_key),
        ):
            if enc_nm <= 0:
                continue  # PDK declares no enclosure on this side -> nothing to owe
            metal_reg = _region(mkey)
            if metal_reg is None:
                metal_reg = kdb.Region()
            # FOUNDRY-FAITHFUL measurement: the via metal-enclosure rule is the
            # KLayout DRC idiom ``metal.enclosing(cut, dist)`` -- the per-EDGE
            # enclosure check (``Region.enclosing_check``) that flags cut edges the
            # metal does not overhang by ``dist``. This is exactly what BOTH oracles
            # run: the synthesized deck's ``inner.enclosed(outer, v)`` and the real
            # gf180 deck's V1.3/V1.4 enclosure. A wire running THROUGH a via is
            # enclosed along its run by the wire itself; only the perpendicular
            # (end-of-line) edges must carry ``dist`` -- the edge check captures that
            # directionality, so a wide-enough wire passes (matching external_pdk DRC 0)
            # while a min-width gf180 wire fails. (A naive ``cut.sized(enc) - metal``
            # square test over-counts -- it demands enclosure on all four sides even
            # where the wire body already supplies it -- and disagrees with the
            # oracle; the edge check is the native form of the same obligation.)
            dist_dbu = int(round(enc_nm / nm_per_dbu))
            ec = metal_reg.enclosing_check(cut_reg, dist_dbu)
            # ``enclosing_check`` returns ONE EdgePair per violating edge. Group by
            # the cut polygon the violating edge belongs to so the count is per-cut
            # (comparable to a DRC instance), and record a representative centroid.
            flagged_cuts: Dict[Tuple[int, int], None] = {}
            for ep in ec.each():
                e = ep.second  # the cut edge (the enclosed shape's edge)
                key_pt = (
                    int(round((e.x1 + e.x2) / 2.0)),
                    int(round((e.y1 + e.y2) / 2.0)),
                )
                flagged_cuts[key_pt] = None
            for (mx, my) in flagged_cuts:
                report.violations.append(
                    ViaEnclosureViolation(
                        via_name=via_name,
                        side=side,
                        cut_layer=cut_key,
                        metal_layer=tuple(mkey),
                        required_nm=enc_nm,
                        cut_xy_um=(mx * dbu, my * dbu),
                    )
                )
                rk = f"{via_name}.{side}"
                report.per_rule[rk] = report.per_rule.get(rk, 0) + 1

    if not examined_any_layer:
        report.status = "unmeasurable"
        report.reason = report.reason or "no via cut layers present in GDS"
        return report
    report.status = "fail" if report.violations else "pass"
    return report


__all__ = [
    "ViaEnclosureViolation",
    "ViaEnclosureReport",
    "check_via_enclosure",
]

"""Router-cut scope; fixed MIM cuts remain in the complete GDS.

This does not change the existing all-cut checker or substitute for full DRC.
The caller must pass the actual live engine and actual route baselines.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math

from clay.layout.route.via_model import via_enclosure_nm, landing_rect_for_cut
from pathlib import Path


def _via_rule_authority_error(tech, index, name):
    raw_rules = tech.drc_rules()
    parsed = {} if raw_rules is None else raw_rules
    if not isinstance(parsed, dict):
        return 'PDK parsed rule authority is malformed'
    try:
        declared = tech.via_rule(name)
    except Exception:
        declared = None
    keys = {f'via{index}_size', f'via{index}_width', f'via{index}_space',
            f'via{index}_enclosure_metal{index}', f'via{index}_enclosure_metal{index+1}'}
    for key in keys & parsed.keys():
        fact = parsed[key]
        if not isinstance(fact, dict):
            return 'malformed PDK via rule ' + key
        values = [fact[field] for field in ('value_nm', 'value_um') if field in fact]
        if not values or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            return 'malformed PDK via rule ' + key
        if key in {f'via{index}_size', f'via{index}_width'} and any(value <= 0 for value in values):
            return 'malformed PDK via rule ' + key
    known = any(isinstance(parsed.get(k), dict) and
                any(type(parsed[k].get(f)) in (int, float) for f in ('value_nm', 'value_um'))
                for k in keys)
    if declared is None and not known:
        return 'present cut has no PDK enclosure authority'
    if not parsed:
        for face in ('enclosure_below', 'enclosure_above'):
            raw = getattr(declared, face, None)
            value = getattr(raw, 'value', raw)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return 'present cut has unknown PDK ' + face
    return None


@dataclass(frozen=True)
class RouterCutReport:
    status: str
    artifact_sha256: str
    checked: int
    cuts: tuple
    violations: tuple
    reason: str = ''
    scope: str = 'router-owned cuts only; fixed artwork and full DRC remain separate'
    manifest_sha256: str = ''


def actual_router_cut_manifest(engine, baselines):
    rows = []
    for net in sorted(baselines):
        for via in baselines[net].vias:
            engine._require_policy_via(via)
            shapes = engine._via_polys(via)[2:]
            cuts = []
            for shape in shapes:
                points = tuple((int(p.x), int(p.y)) for p in shape.polygon.vertices)
                xs, ys = zip(*points)
                box = (min(xs), min(ys), max(xs), max(ys))
                if len(points) != 4 or set(points) != {(box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3])}:
                    raise ValueError('router cut is not an exact rectangle')
                cuts.append((str(net), (int(shape.layer), int(shape.datatype)), box))
            allowed_cuts = (
                (1, engine.via_redundancy ** 2)
                if bool(getattr(engine, "_single_via_escape_allowed", lambda: False)())
                else (engine.via_redundancy ** 2,)
            )
            if len(cuts) not in allowed_cuts or len(set(cuts)) != len(cuts):
                raise ValueError('router via array is incomplete')
            expected_key = tuple(engine._tech.via_key_for(via.lo))
            expected = {
                (str(net), expected_key, tuple(landing_rect_for_cut(x, y, engine._via_cut[via.lo], 0)))
                for x, y in engine._via_cut_centers(via.x, via.y, via.lo, engine._site_n(via))
            }
            if set(cuts) != expected:
                raise ValueError('router via cut geometry differs from selected site and policy')
            rows.extend(cuts)
    if len(rows) != len({(layer, box) for _net, layer, box in rows}):
        raise ValueError('duplicate router cut manifest entry')
    return tuple(sorted(rows))


def check_router_cuts(path, tech, manifest):
    try:
        digest = sha256(Path(path).read_bytes()).hexdigest()
    except Exception as exc:
        return RouterCutReport('unmeasurable', '', 0, (), (), 'cannot read GDS: ' + str(exc))
    def unavailable(reason):
        return RouterCutReport('unmeasurable', digest, 0, (), (), reason)
    if manifest is None:
        return unavailable('router cut manifest unavailable')
    try:
        import klayout.db as db
        from clay.layout.route.deck_isolation import DeckConnectStack
        if type(manifest) not in (tuple, list):
            return unavailable('router cut manifest is malformed')
        seen = set()
        for row in manifest:
            if type(row) not in (tuple, list) or len(row) != 3:
                return unavailable('router cut manifest row is malformed')
            net, layer, box = row
            if (type(net) is not str or not net or type(layer) not in (tuple, list) or len(layer) != 2
                or type(box) not in (tuple, list) or len(box) != 4
                or any(type(x) is not int for x in (*layer, *box))
                or box[0] >= box[2] or box[1] >= box[3]):
                return unavailable('router cut manifest identity is malformed')
            identity = (tuple(layer), tuple(box))
            if identity in seen:
                return unavailable('router cut manifest contains duplicate geometry')
            seen.add(identity)
        manifest_digest = sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        layout = db.Layout()
        layout.read(str(path))
        tops = layout.top_cells()
        if len(tops) != 1 or layout.dbu != 0.001:
            return unavailable('expected one top and integer-nm GDS')
        top = tops[0]
        stack = DeckConnectStack.from_tech(tech)
        regions = {}
        def region(key):
            key = tuple(key)
            if key not in regions:
                idx = layout.find_layer(*key)
                regions[key] = db.Region() if idx is None else db.Region(top.begin_shapes_rec(idx)).merged()
            return regions[key]
        by_cut = {tuple(c.cut): c for c in stack.connections}
        if len(by_cut) != len(stack.connections):
            return unavailable('duplicate PDK via layers')
        errors = []
        for net, key, box in manifest:
            connection = by_cut.get(tuple(key))
            if connection is None:
                return unavailable('manifest cut is outside the full PDK stack')
            matches = [i for i in range(1, len(stack.conductors)) if tuple(tech.via_key(i)) == tuple(key)]
            if len(matches) != 1:
                return unavailable('PDK cut index is not unique')
            name = _via_name_for_index(tech, matches[0])
            if name is None:
                return unavailable('present cut has no recognized via rule name')
            authority_error = _via_rule_authority_error(tech, matches[0], name)
            if authority_error:
                return unavailable(authority_error)
            cut = db.Region(db.Box(*box))
            if not (cut - region(key)).is_empty():
                errors.append((net, tuple(key), tuple(box), 'cut absent from GDS'))
            lower, upper = region(connection.lower), region(connection.upper)
            for carve in stack.mim_carves:
                if (carve.bottom, carve.riser) != (connection.lower, connection.cut):
                    continue
                marker = region(carve.top_marker)
                interacting = lower.interacting(marker)
                if carve.marker_only:
                    virtual = marker & interacting
                elif carve.oversize_um > 0:
                    virtual = marker.sized(int(round(carve.oversize_um / layout.dbu))) & interacting
                else:
                    virtual = interacting
                if not cut.interacting(marker | virtual).is_empty():
                    errors.append((net, tuple(key), tuple(box), 'router cut intersects MIM domain'))
            enclosures = via_enclosure_nm(tech, name)
            for side, metal, distance in (('below', lower, enclosures[0]), ('above', upper, enclosures[1])):
                if not (cut - metal).is_empty():
                    errors.append((net, tuple(key), tuple(box), side + ' cut coverage missing'))
                elif distance > 0 and not metal.enclosing_check(cut, int(distance)).is_empty():
                    errors.append((net, tuple(key), tuple(box), side + ' enclosure violation'))
        return RouterCutReport('fail' if errors else 'pass', digest, len(manifest), tuple(manifest), tuple(errors), manifest_sha256=manifest_digest)
    except Exception as exc:
        return unavailable(str(exc))

__all__ += ["RouterCutReport", "actual_router_cut_manifest", "check_router_cuts"]
