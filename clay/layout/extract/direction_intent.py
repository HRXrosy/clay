"""Reference-layout per-layer DIRECTION style extraction (plan S3, intent face).

THE STYLE != LEAKAGE BOUNDARY (iron rule, same class as style_order.py): the
style channel may carry direction STATISTICS as pure RELATIVE facts. What
crosses this function's boundary is, per metal level, the H/V edge-length
RATIO (a dimensionless number in [0,1]) and the majority axis -- never a
coordinate, a length total, or a spacing value. The reference GDS is read
INSIDE the extractor (the migration-legal reference read) and the absolute
geometry is consumed and discarded.

HONEST DEGRADATION: a layer whose majority-axis purity is below ``min_purity``
(default 0.6) yields NO direction intent (``axis=None`` in the evidence) -- a
mixed-direction reference layer commands nothing; we never invent a preference
the reference does not exhibit.

PHASE-CONSISTENT EXTENSION (the constraint-construction root cause of the
test4 purity collapse): the measured pure layers fix the ALTERNATION PHASE of
the stack (M1=V, M2=H, M3=V => odd metals V, even metals H). A consumer
routing on layers the reference does not command (below the purity gate, or
carrying no metal) needs those layers filled with the SAME phase -- filling
them from a parity convention (lower.CONVENTIONAL_AXES: odd=H, even=V) can be
phase-OPPOSED to the measured stack (it commands M4=V where the V,H,V
continuation is M4=H), so wrong-axis overflow gets no penalty-free drain.
``extend_layers`` (opt-in) emits, for each requested routable metal level with
no measured command, the phase-continuation axis with a self-declared degraded
``source='phase-extension'`` (measured rows say ``source='measured'``). The
extension consumes ONLY the phase (axis strings + small metal indices) -- zero
coordinates, the same legitimacy class as the ratios. If the measured layers
do not fix a single consistent phase (non-alternating majorities), ``phase``
is None and NO extension is emitted -- we never invent a continuation the
reference does not exhibit.

WHY A PARAMETER (not consumer-side extension from the reported phase): the
extractor does not know the routing TARGET's stack (``tech_id`` here is the
reference's PDK), so the target's routable metal levels must come from the
caller either way. Doing the extension INSIDE the extractor keeps the
degraded-provenance marking (``source='phase-extension'``) in the one place
that owns direction provenance -- every consumer gets the self-declaration for
free instead of re-implementing the alternation arithmetic and inventing its
own marking. The ``phase`` field is still reported for transparency.

WEIGHT: fixed at 1.0 for every emitted direction. Direction is a SOFT
preference (an extra step-cost factor in the engine, never a hard block);
CALIBRATING the weight against routed-quality outcomes is explicitly out of
scope here -- the weight is a design-side cost knob, not a measured reference
value, so a constant is the honest choice.

MEASUREMENT: per metal level, the summed |dx| vs |dy| of every polygon edge on
that GDS layer (boxes/paths/polygons, instances flattened with their
transforms). For a w x h rectangle this contributes 2w horizontal and 2h
vertical, so long wires dominate their run axis while square pads/vias split
evenly (diluting purity -- a conservative, direction-neutral effect). The
ratio is an OUTPUT-side relative statistic of the reference, the same
legitimacy class as the row partition in style_order.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

_MIN_PURITY = 0.6
_STYLE_WEIGHT = 1.0
_MAX_METALS = 10  # matches lower.CONVENTIONAL_AXES coverage


def _flip(axis: str) -> str:
    return "H" if axis == "V" else "V"


def extract_direction_intent(
    ref_gds_path: str,
    tech_id: str = "gf180mcu",
    *,
    cell: Optional[str] = None,
    min_purity: float = _MIN_PURITY,
    extend_layers: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Measure the reference's per-layer H/V composition -> direction intent.

    Parameters
    ----------
    ref_gds_path : str
        The reference GDS (read-only; consumed inside, no geometry crosses out).
    tech_id : str
        The REFERENCE's PDK id -- resolves each metal level's GDS (layer,
        datatype) key. This is the reference-side tech (the layout being
        measured), not the routing target; the emitted intent speaks in metal
        LEVELS (layer_n), which transfer across PDKs.
    cell : str, optional
        Cell to measure. Default: the single top cell (or, with several top
        cells, the one with the largest bounding box).
    min_purity : float
        The honesty gate: a layer emits a direction only if its majority-axis
        ratio is >= this value.
    extend_layers : iterable of int, optional
        The TARGET's routable metal levels (e.g. the routing engine's fine
        stack ``[1, 2, 3, 4]``). Each requested level with NO measured
        direction gets the phase-continuation axis at the fixed weight, with
        ``source='phase-extension'`` (self-declared degradation). Default
        None extends nothing -- byte-stable report shape aside from the
        always-present ``source`` / ``phase`` fields. Levels outside
        ``1..10`` are ignored (matches the convention-table coverage).

    Returns
    -------
    dict with keys
        ``directions`` -- ``[{'layer_n': int, 'axis': 'H'|'V', 'weight': 1.0,
        'source': 'measured'|'phase-extension'}]`` -- measured rows (purity
        gate passed) first, then any phase extensions (ready to become
        :class:`clay.layout.route.constraints.Direction` constraints);
        ``evidence`` -- ``{layer_n: {'h_ratio','v_ratio','purity','axis'}}``
        for every non-empty metal layer (``axis`` None below the gate; the
        evidence is the MEASUREMENT face -- an extension never touches it);
        ``phase`` -- ``{'odd': 'H'|'V', 'even': 'H'|'V'}`` when the measured
        layers fix one consistent alternation phase, else None;
        ``min_purity`` -- the gate used; ``provenance`` -- 'style-transfer'.
        Every numeric leaf is a ratio / index / weight -- ZERO coordinates.
    """
    import klayout.db as kdb

    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)

    ly = kdb.Layout()
    ly.read(str(ref_gds_path))
    if cell is not None:
        top = ly.cell(str(cell))
        if top is None:
            raise KeyError(f"cell {cell!r} not found in {ref_gds_path}")
    else:
        tops = ly.top_cells()
        if not tops:
            raise ValueError(f"no top cell in {ref_gds_path}")
        top = max(tops, key=lambda c: c.bbox().area())

    evidence: Dict[int, Dict[str, Any]] = {}
    directions: List[Dict[str, Any]] = []
    for n in range(1, _MAX_METALS + 1):
        try:
            l, d = tech.metal_key(n)
        except KeyError:
            break
        li = ly.find_layer(int(l), int(d))
        if li is None:
            continue  # layer absent from the file -> nothing to measure
        h_len = 0
        v_len = 0
        it = top.begin_shapes_rec(li)
        while not it.at_end():
            sh = it.shape()
            if sh.is_box() or sh.is_polygon() or sh.is_path():
                poly = sh.polygon.transformed(it.trans())
                for e in poly.each_edge():
                    h_len += abs(e.dx())
                    v_len += abs(e.dy())
            it.next()
        tot = h_len + v_len
        if tot == 0:
            continue  # empty layer -> no statistic, no intent
        h_ratio = h_len / tot
        v_ratio = v_len / tot
        purity = max(h_ratio, v_ratio)
        majority = "H" if h_ratio >= v_ratio else "V"
        passes = purity >= float(min_purity)
        evidence[n] = {
            "h_ratio": h_ratio,
            "v_ratio": v_ratio,
            "purity": purity,
            # axis is the COMMANDED axis: None below the gate (honest
            # degradation -- a mixed layer commands nothing).
            "axis": majority if passes else None,
        }
        if passes:
            directions.append(
                {
                    "layer_n": n,
                    "axis": majority,
                    "weight": _STYLE_WEIGHT,
                    "source": "measured",
                }
            )

    # PHASE: the single alternation phase the measured layers fix, if any.
    # Each measured (n, axis) votes for the odd-level axis (axis itself on an
    # odd level, the flip on an even level); one unanimous vote = a phase.
    phase: Optional[Dict[str, str]] = None
    odd_votes = {
        d["axis"] if d["layer_n"] % 2 == 1 else _flip(d["axis"])
        for d in directions
    }
    if len(odd_votes) == 1:
        odd_axis = next(iter(odd_votes))
        phase = {"odd": odd_axis, "even": _flip(odd_axis)}

    # PHASE-CONSISTENT EXTENSION (opt-in): fill each requested routable level
    # that the reference does not command with the phase continuation, at the
    # same fixed soft weight, self-declaring the degraded source. Consumes
    # ONLY the phase -- zero coordinates. No phase -> no extension.
    if extend_layers is not None and phase is not None:
        measured_levels = {d["layer_n"] for d in directions}
        for n in sorted({int(x) for x in extend_layers}):
            if n < 1 or n > _MAX_METALS or n in measured_levels:
                continue
            directions.append(
                {
                    "layer_n": n,
                    "axis": phase["odd"] if n % 2 == 1 else phase["even"],
                    "weight": _STYLE_WEIGHT,
                    "source": "phase-extension",
                }
            )

    return {
        "directions": directions,
        "evidence": evidence,
        "phase": phase,
        "min_purity": float(min_purity),
        "provenance": "style-transfer",
    }


__all__ = ["extract_direction_intent"]
