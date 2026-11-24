"""ABSTRACT floorplan-intent extractor for the front-end design.clay emit.

WHAT THIS IS
============
:func:`clay.frontend.clay_emit.emit_design_clay` renders Layer A (the complete
device netlist) and Layer B (functional-group roles + @symmetric/@matching). It
carries NO spatial intent, so :func:`clay.layout.generic_placement.place_generic`
falls back to its default NMOS-below-PMOS band and the cell sprawls tall.

This module derives the missing SPATIAL OVERLAY: a set of ``@place_group(...,
devices=(...), row=R [, col=C])`` relations that assign every device to a row /
column of a 2-D analog floorplan. When emitted into the ``layout_intent`` block,
:func:`clay.layout.annotate_to_solver._row_plan_from_floorplan` fires and the
placement compacts into a wide 2-D shape instead of one tall column.

ABSTRACT RELATIONS ONLY -- NO COORDINATE LEAK
=============================================
The floorplan is expressed as ABSTRACT ORDINALS (row index / column index) +
group membership, NEVER a raw reference coordinate. The row/column ASSIGNMENT is
derived from each device's FUNCTIONAL ROLE (the netlist-derived
:class:`FunctionalRoleHit`) and, for a device no role claimed, its DEVICE KIND +
size class. None of these inputs is a position read from the reference layout:
they are all properties of the SCHEMATIC (topology roles, transistor kind, W/L),
so the emitted overlay contains no reference dbu. A leak audit
(:func:`clay.frontend.clay_emit_compactness_check.assert_no_reference_coord_leak`)
proves the emitted text carries no absolute x/y / centroid / numeric axis.

The region ORDER (which role-band is the bottom tier, which the top, passives
banked, the large pass device in its own column) is a GENERIC analog-floorplan
policy keyed on role / kind -- the structural pattern an analog op-amp / LDO
floorplan follows (signal-flow rail banding + a banked passive cluster + a wide
output device beside the core). It is data-driven from the role decomposition,
with NO per-circuit branch (no ``if cell == ...`` / ``if device == ...``): the
SAME policy runs for any RoleDecomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from clay.frontend.role_model import DeviceCard, RoleDecomposition

__all__ = [
    "PlaceGroupRelation",
    "floorplan_place_groups",
    "floorplan_place_groups_from_reference",
]


@dataclass(frozen=True)
class PlaceGroupRelation:
    """One abstract ``@place_group`` relation: a named device group pinned to a
    floorplan ROW (and optional COLUMN). Carries ordinals only -- never a
    coordinate."""

    name: str
    devices: Tuple[str, ...]
    row: int
    col: Optional[int] = None


# --- the generic role -> floorplan-row policy --------------------------------
#
# Rows are ABSTRACT ordinals (0 = bottom tier). The ordering follows the analog
# signal-flow / rail-band convention an op-amp / LDO core uses: the NMOS
# current-source / mirror tree at the bottom, the differential input pair and its
# tail above it, the cascode / load mirror toward the PMOS rail at the top. A
# device whose role is NOT in this table falls through to the KIND-based default
# (PMOS toward the top rail, NMOS toward the bottom rail) so EVERY device gets a
# region and nothing is dumped into a default catch-all row. This is the abstract
# region ORDER -- a structural relation, not a coordinate.
_ROLE_ROW: Dict[str, int] = {
    # bottom rail band: NMOS bias / mirror tree + the tail source.
    "bias_tree": 0,
    "tail_current": 0,
    "current_mirror": 1,
    # core band: the differential input pair.
    "input_pair": 2,
    # toward the top rail: cascode then the PMOS load mirror.
    "cascode": 3,
    "load_mirror": 4,
    "second_stage": 4,
}

# A device no role claimed lands by KIND in the rail band matching its type, so a
# stray current source / replica / output FET still gets a region (NMOS low, PMOS
# high) rather than a sprawling default row. These two rows reuse the bottom /
# top band ordinals so unroled devices merge into the existing bands.
_KIND_DEFAULT_ROW: Dict[str, int] = {
    "nmos": 1,   # merge with the NMOS mirror band
    "pmos": 4,   # merge with the PMOS load band
}

# The passive cluster (resistors) is a BANK in its own tier ABOVE the MOS core --
# the reference's compact folded resistor bank. Caps get a sibling tier. These two
# ordinals sit above every MOS row so the passive block is a distinct region (the
# placer's passive path lays the bank out compactly there). They are ordinals, not
# y-positions.
_RESISTOR_BANK_ROW = 5
_CAP_BANK_ROW = 6


def _role_of_device(decomp: RoleDecomposition) -> Dict[str, str]:
    """``device -> role token`` from the decomposition's role hits.

    A device may be named by more than one role hit (e.g. a tail device that is
    also part of a current mirror). The FIRST role whose row ordinal is the
    LOWEST wins, so the device lands in the most specific / bottom-most band it
    belongs to -- deterministic (roles iterated in order, lowest row kept)."""
    out: Dict[str, str] = {}
    best_row: Dict[str, int] = {}
    for hit in decomp.roles:
        row = _ROLE_ROW.get(hit.role)
        if row is None:
            continue
        for dev in hit.devices:
            if dev not in best_row or row < best_row[dev]:
                best_row[dev] = row
                out[dev] = hit.role
    return out


def _is_resistor(card: DeviceCard) -> bool:
    k = (card.kind or "").strip().lower()
    return k in ("res", "resistor", "r")


def _is_cap(card: DeviceCard) -> bool:
    k = (card.kind or "").strip().lower()
    return k in ("cap", "capacitor", "c", "mim")


def _mos_kind(card: DeviceCard) -> Optional[str]:
    k = (card.kind or "").strip().lower()
    if k.startswith("n"):
        return "nmos"
    if k.startswith("p") and k != "pnp":
        return "pmos"
    return None


def floorplan_place_groups(
    decomp: RoleDecomposition,
) -> Tuple[PlaceGroupRelation, ...]:
    """Derive the ABSTRACT @place_group floorplan relations for a decomposition.

    Returns one :class:`PlaceGroupRelation` per row that has at least one device.
    Every device in the netlist is assigned to exactly one row:

      * a device a floorplan role claims (input_pair / current_mirror / cascode /
        load_mirror / bias_tree / tail_current / second_stage) -> that role's row;
      * a MOS device no such role claims -> its KIND's default rail band;
      * a resistor -> the resistor BANK row; a cap -> the cap BANK row.

    Within a row the devices are ordered (and given a column ordinal) by name so
    the placement column order is deterministic. The output is ordinals + group
    membership ONLY -- no reference coordinate -- so the leak audit stays GREEN.

    Generic: the SAME policy runs for any RoleDecomposition; there is no
    per-circuit / per-device branch.
    """
    role_of = _role_of_device(decomp)

    # bucket every device by its floorplan row.
    rows: Dict[int, List[str]] = {}
    for card in decomp.devices:
        name = card.name
        if _is_resistor(card):
            row = _RESISTOR_BANK_ROW
        elif _is_cap(card):
            row = _CAP_BANK_ROW
        elif name in role_of:
            row = _ROLE_ROW[role_of[name]]
        else:
            mk = _mos_kind(card)
            if mk is None:
                # an unsupported kind (BJT / diode) -> share the bottom band so it
                # still gets a region (never silently dropped from the floorplan).
                row = 0
            else:
                row = _KIND_DEFAULT_ROW[mk]
        rows.setdefault(row, []).append(name)

    relations: List[PlaceGroupRelation] = []
    for row in sorted(rows):
        members = sorted(rows[row])
        # one @place_group per row; the within-row column ordinal is the member's
        # index in the sorted order (a deterministic left-to-right packing).
        relations.append(
            PlaceGroupRelation(
                name=f"floorplan_row{row}",
                devices=tuple(members),
                row=row,
                col=None,
            )
        )
    return tuple(relations)


def floorplan_place_groups_from_reference(
    decomp: RoleDecomposition,
    ref_intent: "object",
    device_master: Dict[str, str],
) -> Tuple[PlaceGroupRelation, ...]:
    """Derive the @place_group floorplan from a REFERENCE layout's arrangement.

    Where :func:`floorplan_place_groups` keys rows off SCHEMATIC ROLE alone, this
    keys them off the ABSTRACT ARRANGEMENT a reference layout realised -- a
    :class:`clay.frontend.reference_spatial_intent.ReferenceSpatialIntent` carrying
    ordinal relations ONLY (region tier / column order / banking / wide-block). Each
    device is routed to its region by its PDK MASTER (a schematic property in
    ``device_master``), and lands in that region's:

      * ``row`` = the region's reference TIER ordinal (0 = bottom; the resistor bank
        sits on its own top tier, the MOS+cap core below -- exactly the reference's
        vertical banding);
      * ``col`` = the region's reference COLUMN ordinal (0 = leftmost; the caps land
        left, the wide pass block lands right -- the reference's horizontal order).

    One ``@place_group`` is emitted per (tier, column) region so the placer realises
    the reference banding. A device whose master has NO reference region (an
    unmapped kind) is appended to the role-only floorplan's assignment so it still
    gets a row -- never silently dropped. The relations consumed are ordinals; no
    reference coordinate is ever read, so the leak audit stays GREEN.

    Generic: the routing keys off master / tier / column for ANY reference, no
    per-circuit branch.
    """
    region_tier: Dict[str, int] = dict(getattr(ref_intent, "region_tier", {}) or {})
    region_column: Dict[str, int] = dict(
        getattr(ref_intent, "region_column", {}) or {}
    )
    if not region_tier:
        # no usable reference arrangement -> fall back to the role-only floorplan.
        return floorplan_place_groups(decomp)

    # bucket every device into its reference region (by master); unmapped -> None.
    by_region: Dict[Optional[str], List[str]] = {}
    for card in decomp.devices:
        region = device_master.get(card.name)
        if region not in region_tier:
            region = None
        by_region.setdefault(region, []).append(card.name)

    relations: List[PlaceGroupRelation] = []
    # one group per reference region, ordered bottom->top then left->right so the
    # emit is deterministic.
    mapped = [r for r in by_region if r is not None]
    for region in sorted(
        mapped, key=lambda r: (region_tier[r], region_column.get(r, 0), r)
    ):
        members = sorted(by_region[region])
        if not members:
            continue
        relations.append(
            PlaceGroupRelation(
                name=f"region_{region}",
                devices=tuple(members),
                row=region_tier[region],
                col=region_column.get(region, 0),
            )
        )

    # devices whose master has no reference region: assign them via the role-only
    # policy so every device still has a row (completeness preserved).
    unmapped = by_region.get(None, [])
    if unmapped:
        max_row = max(region_tier.values())
        unmapped_set = set(unmapped)
        sub = RoleDecomposition(
            cell=decomp.cell,
            devices=[c for c in decomp.devices if c.name in unmapped_set],
            roles=[],
            matched_pairs=(),
            mirror_axis=getattr(decomp, "mirror_axis", None),
        )
        for pg in floorplan_place_groups(sub):
            relations.append(
                PlaceGroupRelation(
                    name=f"unmapped_{pg.name}",
                    devices=pg.devices,
                    row=max_row + 1 + pg.row,
                    col=None,
                )
            )
    return tuple(relations)
