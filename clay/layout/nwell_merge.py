"""R144 (BL-144A) — merge same-potential PMOS Nwells (kills the NW.2b residual).

After R140 E (:mod:`clay.layout.gf180_pmos_geom`) builds each PMOS as a tight
Nwell box around its own PCOMP, a 5T-OTA places its two mirror-load PMOS devices
side by side. The real ``gf180mcu.drc`` deck then reports the last residual::

    NW.2b_3.3V   x1   Min. Nwell Space (Outside DNWELL) [Different potential].
                      (nwell.drc / gf180mcu.drc line 871, 1.4um = 1400 nm)

R142 (:mod:`clay.layout.m1_spacing_nwell_fix`) repairs different-potential wells
by *nudging* them apart to the 1400 nm ``NW.2b`` floor. But in a 5T-OTA the
mirror-load PMOS pair both connect their source/well to **VDD** — they are the
**SAME potential**. ``NW.2b`` only penalizes Nwells of DIFFERENT potential; two
SAME-potential wells are governed by the much smaller equi-potential rule
``NW.2a`` (600 nm), and two same-potential wells that touch/overlap simply MERGE
into one well (the deck unions same-layer geometry before spacing checks, so a
single merged well has no internal spacing edge at all).

So the correct repair for the 5T-OTA NW.2b residual is NOT to push the two
mirror wells apart (that wastes area and is electrically meaningless for a shared
node) but to **merge same-potential wells into one Nwell** whose bbox is the
union of both. After the merge there is exactly one Nwell covering both PCOMPs:

* ``NW.2b`` (different potential, 1400 nm) no longer applies — the two wells are
  now one shape, so there is no different-potential well pair to check.
* the merged well still encloses BOTH PCOMPs by >= ``DF.4c`` (430 nm) on every
  side, because a bbox union only ever GROWS the boundary outward — each PCOMP
  keeps at least the enclosure margin its original (tight R140-E) well gave it.
* any residual same-potential adjacency falls under ``NW.2a`` (600 nm), which a
  bbox union trivially satisfies (the merged well is contiguous).

Rule values used (all transcribed verbatim from the real gf180mcu KLayout DRC
deck; nothing invented):

* ``NW.2b_3.3V``  Min. Nwell Space [Outside DNWELL][Different potential]
             = 1.4 um = 1400 nm   (gf180mcu.drc line 871; via
             :data:`gf180_rules.NWELL_SPACE_DIFF_POT`).
* ``NW.2a_3.3V``  Min. Nwell Space (equi-potential) = 0.6 um = 600 nm
             (gf180mcu.drc line 859; via ``gf180_rules.min_spacing_nm("NWELL")``).
* ``DF.4c_3.3V``  Min. (Nwell overlap of PCOMP) outside DNWELL = 0.43 um = 430 nm
             (comp.drc line 366; via
             :data:`gf180_pmos_geom.GF180_NWELL_PCOMP_ENCLOSURE_NM`, re-exported
             through :func:`nwell_enclosure_guard.nwell_pcomp_enclosure_nm`).

This module is pure geometry. It imports the committed backend IR, the cited
``gf180_rules`` accessors and the R143 B enclosure helper read-only and emits
only merged copies of the input Nwell ``LayerPolygon`` shapes. All coordinates
are GDS database units (dbu = 1 nm for gf180mcu).

Public API
----------
- :func:`nwell_diff_potential_spacing_nm` — the cited ``NW.2b`` accessor (1400).
- :func:`nwell_equi_potential_spacing_nm` — the cited ``NW.2a`` accessor (600).
- :func:`merge_nwell_boxes` — union the bboxes of a set of Nwell shapes into one.
- :func:`group_wells_by_potential` — bucket Nwell shapes by their potential label.
- :func:`merge_same_potential_nwells` — the top-level fix: merge every group of
  same-potential Nwells into one well, leaving different-potential / other shapes
  in place. Returns a structured :class:`NwellMergeResult`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import LayerMap, LayerPolygon, LogicalLayer, Polygon
from .gf180_rules import NWELL_SPACE_DIFF_POT, min_spacing_nm

# --- cited rule accessors (verbatim from gf180_rules / the real deck) --------

#: ``NW.2b_3.3V`` different-potential Nwell space — 1.4 um = 1400 nm (line 871).
NWELL_DIFF_POT_RULE_ID = "NW.2b_3.3V"
#: ``NW.2a_3.3V`` equi-potential Nwell space — 0.6 um = 600 nm (deck line 859).
NWELL_EQUI_POT_RULE_ID = "NW.2a_3.3V"
#: ``DF.4c_3.3V`` Nwell overlap of PCOMP — 0.43 um = 430 nm (comp.drc line 366).
DF4C_RULE_ID = "DF.4c_3.3V"


def nwell_diff_potential_spacing_nm() -> int:
    """Return ``NW.2b`` different-potential Nwell spacing in nm (1400).

    The rule a same-potential merge REMOVES: with the two mirror wells unioned
    into one, there is no different-potential well pair left for ``NW.2b`` to
    check. Sourced from ``gf180_rules.NWELL_SPACE_DIFF_POT`` (line 871, 1.4 um).
    """
    return NWELL_SPACE_DIFF_POT.value_nm


def nwell_equi_potential_spacing_nm() -> int:
    """Return ``NW.2a`` equi-potential Nwell spacing in nm (600).

    The rule that GOVERNS same-potential wells; a bbox union (contiguous well)
    trivially satisfies it. Sourced from ``gf180_rules`` NWELL min_spacing
    (deck line 859, 0.6 um).
    """
    return min_spacing_nm("NWELL")


# --- rect view over a LayerPolygon -------------------------------------------


@dataclass(frozen=True, eq=True)
class _Rect:
    """Axis-aligned bbox of a LayerPolygon (database units)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @staticmethod
    def of(lp: LayerPolygon) -> "_Rect":
        xs = [v.x for v in lp.polygon.vertices]
        ys = [v.y for v in lp.polygon.vertices]
        return _Rect(min(xs), min(ys), max(xs), max(ys))


def _layer_key(pdk_layers: LayerMap, logical: LogicalLayer) -> Tuple[int, int]:
    g = pdk_layers.gds_layer(logical)
    if g is None:
        raise ValueError(f"PDK must define {logical.name} for Nwell merge")
    return (g.layer, g.datatype)


# --- merge primitive ---------------------------------------------------------


def merge_nwell_boxes(wells: Sequence[LayerPolygon]) -> LayerPolygon:
    """Return ONE Nwell whose bbox is the union of every well in ``wells``.

    All inputs must share the same GDS (layer, datatype) — they are the same
    Nwell layer for one potential group. The result is a single rectangle on that
    layer covering every input box; because a bbox union only ever extends the
    boundary outward, the merged well encloses everything each input well
    enclosed (so per-PCOMP ``DF.4c`` enclosure is preserved, never reduced).

    Raises:
        ValueError: if ``wells`` is empty or its members are on different layers.
    """
    if not wells:
        raise ValueError("merge_nwell_boxes requires at least one well")
    key = (wells[0].layer, wells[0].datatype)
    rects = []
    for w in wells:
        if (w.layer, w.datatype) != key:
            raise ValueError(
                f"cannot merge Nwells on different layers "
                f"({key}) vs ({w.layer}/{w.datatype})"
            )
        rects.append(_Rect.of(w))
    x_min = min(r.x_min for r in rects)
    y_min = min(r.y_min for r in rects)
    x_max = max(r.x_max for r in rects)
    y_max = max(r.y_max for r in rects)
    return LayerPolygon(
        layer=key[0],
        datatype=key[1],
        polygon=Polygon.rectangle(x_min, y_min, x_max, y_max),
    )


# --- potential grouping ------------------------------------------------------


def group_wells_by_potential(
    nwell_indices: Sequence[int],
    potential_groups: Dict[int, str],
) -> Dict[str, List[int]]:
    """Bucket Nwell shape-indices by their potential label.

    ``potential_groups`` maps shape-index -> potential/net label (e.g. the two
    5T-OTA mirror-load wells both labelled ``"VDD"``). Only indices that appear in
    ``nwell_indices`` AND carry a label are grouped; an unlabelled well is left
    out (it is not part of any known same-potential merge and is preserved as-is
    by the caller). Returns ``{label: [sorted shape indices]}``; insertion order
    of first appearance is preserved across labels.
    """
    out: Dict[str, List[int]] = {}
    nw_set = set(nwell_indices)
    for i in nwell_indices:
        label = potential_groups.get(i)
        if label is None:
            continue
        out.setdefault(label, []).append(i)
    # keep each bucket sorted by shape index for deterministic output
    for label in out:
        out[label] = sorted(set(out[label]) & nw_set)
    return out


# --- structured result -------------------------------------------------------


@dataclass(frozen=True)
class NwellMergeResult:
    """Result of a same-potential Nwell merge pass.

    Attributes:
        shapes: the full shape list AFTER the merge. Non-Nwell shapes and
            unlabelled / singleton wells keep their position; each merged group
            is collapsed to ONE well placed at the position of the group's first
            (lowest-index) well, and the other members of that group are dropped.
        rule_id: the deck rule the merge REMOVES ("NW.2b_3.3V" different
            potential — there is no different-potential well pair after merging).
        diff_pot_rule_nm: the ``NW.2b`` value (nm; 1400) the merge sidesteps.
        equi_pot_rule_nm: the ``NW.2a`` value (nm; 600) governing the merged
            same-potential well (trivially satisfied by a contiguous box).
        merged_groups: ``{label: [shape indices that were merged]}`` for every
            potential group that had >= 2 wells (only these actually merged).
        merged_count: number of wells that were folded away by merging (== sum of
            (group size - 1) over merged groups); 0 == nothing to merge.
    """

    shapes: List[LayerPolygon]
    rule_id: str
    diff_pot_rule_nm: int
    equi_pot_rule_nm: int
    merged_groups: Dict[str, List[int]]
    merged_count: int

    @property
    def merged(self) -> bool:
        """True iff at least one same-potential group was merged."""
        return self.merged_count > 0


def merge_same_potential_nwells(
    shapes: Sequence[LayerPolygon],
    potential_groups: Dict[int, str],
    pdk_layers: LayerMap,
) -> NwellMergeResult:
    """Merge every group of SAME-potential Nwells into one well (kills NW.2b).

    The last 5T-OTA real-deck residual is ``NW.2b`` x1: two mirror-load PMOS
    Nwells sit within the 1400 nm different-potential spacing. But those two wells
    share VDD — they are the SAME potential, so ``NW.2b`` should never apply.
    This pass MERGES every set of same-potential Nwells (those carrying the same
    ``potential_groups`` label) into a single Nwell whose bbox is the union of the
    group (:func:`merge_nwell_boxes`). After the merge:

    * ``NW.2b`` (different potential, 1400 nm) has no well PAIR to check inside a
      merged group — the group is one shape.
    * the merged well still encloses every PCOMP it covers by >= ``DF.4c``
      (430 nm), because a bbox union only grows the boundary outward.
    * the merged (contiguous) well trivially clears the same-potential ``NW.2a``
      (600 nm) floor.

    Only shapes on the PDK's NWELL layer are considered for merging. Non-Nwell
    shapes (PCOMP, poly, contacts, metal) pass through untouched and keep their
    relative order. Wells with no label in ``potential_groups`` — and singleton
    groups — are preserved unchanged (there is nothing to merge). Each merged
    group is collapsed to one well at the slot of its lowest-index member; the
    other members are removed.

    Different-potential wells are NEVER merged (that would short distinct nodes);
    only wells the caller explicitly marks as the same potential are unioned.

    Args:
        shapes: the shape list containing the PMOS Nwell boxes plus other layers.
        potential_groups: shape-index -> potential/net label. Wells sharing a
            label are the same potential and get merged (e.g. ``{0: "VDD",
            1: "VDD"}`` for a 5T-OTA mirror pair).
        pdk_layers: a gf180mcu LayerMap (defines NWELL 21/0).

    Returns:
        A :class:`NwellMergeResult` with the merged shape list, the merged groups,
        and the cited NW.2b / NW.2a rule values.
    """
    nw_key = _layer_key(pdk_layers, LogicalLayer.NWELL)

    # Copy every shape (input order preserved); only same-potential Nwell groups
    # collapse to a single merged well.
    work: List[Optional[LayerPolygon]] = [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=list(lp.polygon.vertices)),
        )
        for lp in shapes
    ]

    nwell_idx = [
        i for i, lp in enumerate(shapes) if (lp.layer, lp.datatype) == nw_key
    ]
    groups = group_wells_by_potential(nwell_idx, potential_groups)

    merged_groups: Dict[str, List[int]] = {}
    merged_count = 0
    for label, members in groups.items():
        if len(members) < 2:
            # a single well at a potential has nothing to merge with.
            continue
        anchor = members[0]
        merged_well = merge_nwell_boxes([shapes[i] for i in members])
        work[anchor] = merged_well
        for other in members[1:]:
            work[other] = None  # removed; folded into the anchor's merged well
        merged_groups[label] = list(members)
        merged_count += len(members) - 1

    out_shapes = [lp for lp in work if lp is not None]

    return NwellMergeResult(
        shapes=out_shapes,
        rule_id=NWELL_DIFF_POT_RULE_ID,
        diff_pot_rule_nm=nwell_diff_potential_spacing_nm(),
        equi_pot_rule_nm=nwell_equi_potential_spacing_nm(),
        merged_groups=merged_groups,
        merged_count=merged_count,
    )


__all__ = [
    "NWELL_DIFF_POT_RULE_ID",
    "NWELL_EQUI_POT_RULE_ID",
    "DF4C_RULE_ID",
    "nwell_diff_potential_spacing_nm",
    "nwell_equi_potential_spacing_nm",
    "merge_nwell_boxes",
    "group_wells_by_potential",
    "NwellMergeResult",
    "merge_same_potential_nwells",
]
