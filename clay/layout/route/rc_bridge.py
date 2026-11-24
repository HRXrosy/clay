"""Shared-diffusion/common-centroid array resolution, placement-to-route bridge, and pin relocation."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
from dataclasses import dataclass, field, is_dataclass, replace
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.layout.extract.realize_render import (
    build_device_geometry,
    resolve_device_fold_target,
)
from clay.layout.nonmos_placement_inputs import replay_native_nonmos_placement
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route.pins import extract_pins as _base_extract_pins
from clay.layout.route.rc_devices import (
    _BridgedDevice,
    _BridgedIntent,
    _BridgedPassive,
    _NM_PER_UM,
    _PlacedBjtStamp,
    _PlacedPassiveStamp,
    _PlacedStamp,
    _ROLE_TO_TERM,
    _XY,
    _declared_ground_ports,
    _rc_attr,
    bridge_bjts_from_cell_def,
    bridge_devices_from_cell_def as _base_bridge_devices_from_cell_def,
)
from clay.layout.tech import LayoutTech

@dataclass
class _ArrayConsumption:
    """A declared ``@array(style=shared_diffusion)`` resolved to its ordered
    bridged members + the built ONE-merged-OD :class:`SharedDiffusionArray`.

    ``array is None`` (with a non-empty ``fallback`` reason) means the declared
    array could NOT be materialized as one shared-OD strip on this Stage
    (an odd interior finger count, a non-``ratio``/`''` order -- an ABBA
    common-centroid interleave is Stage 2 -- unequal member W/L, or a build
    error). Those members stay on the per-device stamp path, byte-identical to a
    cell that declared no array for them. Stage 2 (``build_common_centroid_array``
    + ``@dummy_flank`` separators) handles the interleaved / odd cases."""

    group: str
    members: List[str]                       # ordered member device names
    order: str
    kind: str                                # 'pmos' | 'nmos'
    array: Optional[Any] = None              # SharedDiffusionArray | None (fallback)
    fallback: str = ""
    flank_dummies: List[str] = field(default_factory=list)  # folded rail-dummy device names
    # Canonical stored-row consumers carry the already-certified projection
    # authority.  Legacy array resolvers leave these unset and retain the
    # geometry-derived fallback below.
    certified_member_offsets_nm: Optional[Dict[str, Tuple[int, int]]] = None
    certified_tile_center_nm: Optional[Tuple[int, int]] = None
    certified_translation_nm: Optional[Tuple[int, int]] = None
    body_tap_owner_sha256: str = ""
    body_net: str = ""


def _resolve_rail_flank_dummies(group, members, kind, by_name, intent):
    """Find the fully-off RAIL dummy device(s) for ``@dummy_flank(target=group,
    tie=rail, share_diffusion=true)``: a ``gate==drain==source`` device of the
    array's kind sharing ``group``'s ``@guard_ring`` island (the dummies sit in the
    ring group, not the functional group). Returns ``([(dummy9tuple, per_end)],
    [names])`` for :func:`build_shared_diffusion_array`'s ``flank_dummies``. General
    -- keys only on intent + net equality, no per-circuit branch."""
    flanks = [
        df for df in (getattr(intent, "dummy_flanks", ()) or ())
        if str(getattr(df, "target", "")) == group
        and str(getattr(df, "tie", "")).strip().lower() == "rail"
        and bool(getattr(df, "share_diffusion", True))
    ]
    if not flanks:
        return [], []
    island: tuple = ()
    for gr in (getattr(intent, "guard_rings", ()) or ()):
        if set(members) & set(getattr(gr, "members", ())):
            island = tuple(getattr(gr, "members", ()))
            break
    ref = by_name.get(members[0]) if members else None
    specs = []
    names = []
    for df in flanks:
        per_end = max(1, int(getattr(df, "per_end", 1) or 1))
        for m in island:
            if m in members or m not in by_name:
                continue
            d = by_name[m]
            g = d.nets.get("gate", ""); s = d.nets.get("source", ""); dr = d.nets.get("drain", "")
            fully_off = bool(g) and g == s == dr
            same_geom = (str(d.kind) == kind and ref is not None
                         and abs(float(d.W) - float(ref.W)) < 1e-9
                         and abs(float(d.L) - float(ref.L)) < 1e-9)
            if fully_off and same_geom and m not in names:
                specs.append((
                    (
                        d.name, str(d.kind), float(d.W), float(d.L), int(d.M),
                        g, dr, s, d.nets.get("body", ""),
                    ),
                    per_end,
                ))
                names.append(d.name)
    return specs, names


def _resolve_cc_flank_dummies(group, members, kind, by_name, intent, drains, source_net):
    """Find the common-centroid END-CAP dummy device(s) for ``@dummy_flank(target=
    group, share_diffusion=true)``, sitting in ``group``'s ``@guard_ring`` island (the
    dummies live in the ring group, not the functional group). Accepts TWO kinds, both
    of the pair's kind/W/L:

      * DRAIN-EXPOSED (``tie=gate_off``): gate == source (fully off) and drain is ONE
        of the pair's two ``drains`` -- a matched gate-off end-cap.
      * FULLY-OFF RAIL (``tie=rail``): gate == source == drain == the pair's
        ``source_net`` -- an ALL-SOURCE rail end-cap (the CC analogue of a shared-
        diffusion strip's rail dummy).

    Returns a list of 9-tuple member specs ``[(name, kind, W, L, M, gate, drain,
    source, body), ...]`` for :func:`build_common_centroid_array`'s ``flank_dummies``
    (which classifies + derives the per-end count itself). GENERAL -- keys only on
    intent + net equality, no per-circuit branch. Empty when none is declared/found."""
    flanks = [
        df for df in (getattr(intent, "dummy_flanks", ()) or ())
        if str(getattr(df, "target", "")) == group
        and str(getattr(df, "tie", "")).strip().lower() in ("gate_off", "rail")
        and bool(getattr(df, "share_diffusion", True))
    ]
    if not flanks:
        return []
    island: tuple = ()
    for gr in (getattr(intent, "guard_rings", ()) or ()):
        if set(members) & set(getattr(gr, "members", ())):
            island = tuple(getattr(gr, "members", ()))
            break
    ref = by_name.get(members[0]) if members else None
    out = []
    seen = set()
    for _df in flanks:
        for m in island:
            if m in members or m not in by_name:
                continue
            d = by_name[m]
            g = d.nets.get("gate", ""); s = d.nets.get("source", ""); dr = d.nets.get("drain", "")
            same_geom = (str(d.kind) == kind and ref is not None
                         and abs(float(d.W) - float(ref.W)) < 1e-9
                         and abs(float(d.L) - float(ref.L)) < 1e-9)
            drain_exposed = bool(g) and g == s and bool(dr) and dr != s and dr in drains
            rail = bool(g) and g == s == dr and bool(source_net) and dr == source_net
            if same_geom and (drain_exposed or rail) and d.name not in seen:
                out.append(
                    (
                        d.name, str(d.kind), float(d.W), float(d.L), int(d.M),
                        g, dr, s, d.nets.get("body", ""),
                    )
                )
                seen.add(d.name)
    return out


def resolve_shared_diffusion_arrays(
    cell_def: Any,
    intent: Any,
    tech: Any,
    *,
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[_ArrayConsumption], Dict[str, str]]:
    """Resolve every declared ``@array(style=shared_diffusion)`` to its members +
    built geometry -- the SINGLE source both the compact placer (via
    ``clay.layout.envelope_place``) and the route emit path consume.

    GENERAL, no per-circuit branch: keys ONLY on ``intent.arrays`` (the declared
    ``ArrayIntent`` whose ``style == 'shared_diffusion'``) and the topology-derived
    ``clay.frontend.craft_infer.group_members`` map (group label -> ordered device
    members). The member sizing / nets / oxide come from the SAME
    :func:`bridge_devices_from_cell_def` universe the router bridge reads.

    Returns ``(consumptions, member_group)`` where ``consumptions`` is one
    :class:`_ArrayConsumption` per declared shared-diffusion array (a VALID one
    carries the built ``SharedDiffusionArray``; an unmaterializable one carries a
    ``fallback`` reason and ``array=None``), and ``member_group`` maps each device
    that belongs to a VALID array to its group label (a fallback array's members
    are NOT in the map -> they keep the per-device path). Empty ``([], {})`` when
    the cell declares no shared-diffusion array (byte-identical-off)."""
    bridge_devices_from_cell_def = _rc_attr("bridge_devices_from_cell_def", _base_bridge_devices_from_cell_def)
    declared = [
        a for a in (getattr(intent, "arrays", ()) or ())
        if str(getattr(a, "style", "")).strip().lower() == "shared_diffusion"
    ]
    if not declared:
        return [], {}

    from clay.frontend.craft_infer import group_members
    from clay.layout.device_cell import build_shared_diffusion_array

    gm = group_members(cell_def)
    devices, _passives = bridge_devices_from_cell_def(cell_def, device_records)
    by_name = {d.name: d for d in devices}

    consumptions: List[_ArrayConsumption] = []
    member_group: Dict[str, str] = {}
    for a in declared:
        group = str(getattr(a, "group", "") or "")
        order = str(getattr(a, "order", "") or "")
        members = [str(m) for m in (gm.get(group) or ())]
        if not members or any(m not in by_name for m in members):
            consumptions.append(
                _ArrayConsumption(group=group, members=members, order=order,
                                  kind="", array=None,
                                  fallback="group not in group_members / member unbridged")
            )
            continue
        kind = str(by_name[members[0]].kind)
        member_tuples = []
        oxide = ""
        for m in members:
            d = by_name[m]
            oxide = str(getattr(d, "oxide", "thin") or "thin")
            member_tuples.append((
                d.name, str(d.kind), float(d.W), float(d.L), int(d.M),
                d.nets.get("gate", ""), d.nets.get("drain", ""),
                d.nets.get("source", ""),
                d.nets.get("body", ""),
            ))
        rows = max(1, int(getattr(a, "rows", 1) or 1))  # declared @array(rows=N) fold
        # @dummy_flank(tie=rail, share_diffusion=true) -> fold the rail dummy into
        # the strip as fully-off end-caps (only for a 1-row strip; a folded-row
        # end-cap tie is Stage 2). OPT-IN (default OFF): the merged end-cap GROWS the
        # array tile, which makes the CP envelope solve INFEASIBLE under the hard
        # cross-group symmetry constraint (proven: CP-SAT status=INFEASIBLE, not a
        # timeout) -> the placer drops to a shelf-pack that discards aspect+symmetry
        # (a NET regression). The geometry is correct in isolation (see
        # test_shared_diffusion_flank_dummy) and re-enables once the envelope solve
        # is made symmetry-robust (soft-symmetry re-solve instead of shelf-pack).
        # Default OFF -> flank_specs empty -> byte-identical to the un-folded strip.
        # See memory project_ota_dummy_fold_breaks_cp_symmetry.
        flank_specs, flank_names = ([], [])
        if rows == 1 and os.environ.get("CLAY_OTA_MERGED_DUMMY_FOLD") == "1":
            flank_specs, flank_names = _resolve_rail_flank_dummies(
                group, members, kind, by_name, intent)
        try:
            arr = build_shared_diffusion_array(
                tech, member_tuples, order=(order or "ratio"), oxide=oxide,
                fold_rows=rows, flank_dummies=flank_specs,
            )
        except ValueError as e:  # odd interior nf / non-ratio order / unequal W|L /
            # distinct-drain rows>1 SHORT refusal / bad flank dummy -> honest fallback.
            # Retry once WITHOUT the flank dummies so a bad end-cap does not lose the
            # whole array fold (the dummy then stays a free device -> checker flags it).
            if flank_specs:
                try:
                    arr = build_shared_diffusion_array(
                        tech, member_tuples, order=(order or "ratio"), oxide=oxide,
                        fold_rows=rows, flank_dummies=[],
                    )
                    flank_names = []
                except ValueError as e2:
                    e = e2
                    arr = None
            else:
                arr = None
            if arr is None:
                consumptions.append(
                    _ArrayConsumption(group=group, members=members, order=order,
                                      kind=kind, array=None, fallback=str(e))
                )
                continue
        consumptions.append(
            _ArrayConsumption(group=group, members=members, order=order,
                              kind=kind, array=arr, fallback="",
                              flank_dummies=list(flank_names))
        )
        for m in members:
            member_group[m] = group
        for dn in flank_names:  # folded dummy is consumed into this tile
            member_group[dn] = group
    return consumptions, member_group


def resolve_common_centroids(
    cell_def: Any,
    intent: Any,
    tech: Any,
    *,
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List["_ArrayConsumption"], Dict[str, str]]:
    """Resolve every declared ``@common_centroid`` to its 2-member matched pair + built
    :class:`CommonCentroidArray` (the point-reflected interleave). Same
    ``(consumptions, member_group)`` shape as :func:`resolve_shared_diffusion_arrays`,
    so ``_bridge_placement_to_route_inputs`` / ``array_footprint_tiles`` stamp the
    common-centroid tile exactly like an ``@array`` tile (ONE atomic footprint). Empty
    when no ``@common_centroid`` is declared (byte-identical-off)."""
    _resolve_cc_flank_dummies = _rc_attr("_resolve_cc_flank_dummies", globals()["_resolve_cc_flank_dummies"])
    bridge_devices_from_cell_def = _rc_attr("bridge_devices_from_cell_def", _base_bridge_devices_from_cell_def)
    declared = list(getattr(intent, "common_centroids", ()) or ())
    if not declared:
        return [], {}
    from clay.frontend.craft_infer import group_members
    from clay.layout.common_centroid_array import build_common_centroid_array

    gm = group_members(cell_def)
    devices, _passives = bridge_devices_from_cell_def(cell_def, device_records)
    by_name = {d.name: d for d in devices}

    consumptions: List[_ArrayConsumption] = []
    member_group: Dict[str, str] = {}
    for c in declared:
        group = str(getattr(c, "group", "") or "")
        rows = int(getattr(c, "rows", 2) or 2)
        members = [str(m) for m in (gm.get(group) or ())]
        if len(members) != 2 or any(m not in by_name for m in members):
            consumptions.append(
                _ArrayConsumption(group=group, members=members, order="", kind="",
                                  array=None,
                                  fallback="common_centroid needs exactly 2 bridged members")
            )
            continue
        first_member = str(getattr(c, "first_member", "") or "")
        if first_member:
            if first_member not in members:
                consumptions.append(
                    _ArrayConsumption(
                        group=group,
                        members=members,
                        order="",
                        kind="",
                        array=None,
                        fallback=(
                            "common_centroid first_member must name one bridged "
                            "group member"
                        ),
                    )
                )
                continue
            members = [first_member, *(m for m in members if m != first_member)]
        kind = str(by_name[members[0]].kind)
        member_tuples = []
        oxide = ""
        for m in members:
            d = by_name[m]
            oxide = str(getattr(d, "oxide", "thin") or "thin")
            member_tuples.append((
                d.name, str(d.kind), float(d.W), float(d.L), int(d.M),
                d.nets.get("gate", ""), d.nets.get("drain", ""),
                d.nets.get("source", ""),
                d.nets.get("body", ""),
            ))
        # @dummy_flank(tie=gate_off, share_diffusion=true) -> fold the gate-off input-
        # pair dummies INTO the common-centroid tile as gap=0 end-cap fingers, so the
        # @guard_ring hugs ONE dense block. OPT-IN behind the SAME env flag as the rail
        # case (default OFF -> flank empty -> byte-identical to the un-folded tile);
        # folding SHRINKS the ring-enclosed footprint (removes two stranded free
        # dummies) rather than adding free area, but the CP envelope symmetry solve is
        # gated on the same flag until it is symmetry-robust.
        flank_specs = []
        if os.environ.get("CLAY_OTA_MERGED_DUMMY_FOLD") == "1":
            drains = {by_name[m].nets.get("drain", "") for m in members}
            src_nets = {by_name[m].nets.get("source", "") for m in members}
            source_net = next(iter(src_nets)) if len(src_nets) == 1 else ""
            flank_specs = _resolve_cc_flank_dummies(
                group, members, kind, by_name, intent, drains, source_net)
        try:
            arr = build_common_centroid_array(
                tech, member_tuples, rows=rows, oxide=oxide,
                flank_dummies=flank_specs,
                orientation=str(getattr(c, "orientation", "R0") or "R0"))
        except ValueError as e:
            # Retry once WITHOUT the flank dummies so a bad end-cap does not lose the
            # whole CC fold (the dummy then stays a free device -> a checker flags it).
            if flank_specs:
                try:
                    arr = build_common_centroid_array(
                        tech, member_tuples, rows=rows, oxide=oxide,
                        orientation=str(getattr(c, "orientation", "R0") or "R0"))
                    flank_specs = []
                except ValueError as e2:
                    e = e2
                    arr = None
            else:
                arr = None
            if arr is None:
                consumptions.append(
                    _ArrayConsumption(group=group, members=members, order="",
                                      kind=kind, array=None, fallback=str(e))
                )
                continue
        flank_names = [f[0] for f in flank_specs]
        consumptions.append(
            _ArrayConsumption(group=group, members=members, order="", kind=kind,
                              array=arr, fallback="", flank_dummies=flank_names)
        )
        for m in members:
            member_group[m] = group
        for dn in flank_names:  # folded dummy is consumed into this tile (the fill fix)
            member_group[dn] = group
    return consumptions, member_group


@dataclass(frozen=True)
class _ArrayExtentTranslation:
    """One exact local-array frame plus its certified rigid translation.

    A local extent may have a half-nm geometric centre.  That centre remains an
    exact :class:`Fraction` fact; it is never independently rounded by the stamp,
    pins, or structural expectation.  The emitted CellGeometry uses the integer
    lower centre as its local origin, while ``translation_nm`` is the one rigid
    local-to-cell delta.  Consequently every integer local vertex and terminal
    lands on an integer global nm coordinate even for an odd local extent.
    """

    exact_center_nm: Tuple[Fraction, Fraction]
    stamp_origin_nm: Tuple[int, int]
    translation_nm: Tuple[int, int]

    def __post_init__(self) -> None:
        expected_origin = tuple(
            value.numerator // value.denominator
            for value in self.exact_center_nm
        )
        if self.stamp_origin_nm != expected_origin:
            raise ValueError(
                "array stamp origin must be the lower integer point of its exact centre"
            )

    @property
    def stamp_position_nm(self) -> Tuple[int, int]:
        return (
            self.stamp_origin_nm[0] + self.translation_nm[0],
            self.stamp_origin_nm[1] + self.translation_nm[1],
        )

    def local_to_cell_nm(self, xy_nm: Tuple[int, int]) -> Tuple[int, int]:
        return (
            int(xy_nm[0]) + self.translation_nm[0],
            int(xy_nm[1]) + self.translation_nm[1],
        )


def _array_extent_translation(
    arr: Any,
    *,
    translation_nm: Tuple[int, int],
) -> _ArrayExtentTranslation:
    """Return the sole exact extent frame used by array stamp/pin/expectation.

    Canonical row macros pass their certificate's rigid ``translation_nm``.
    ``exact_center_nm`` records odd extents without turning the half-nm into an
    independently rounded global coordinate.  There is deliberately no mode that
    infers the delta from a mutable placed centre.
    """
    if (
        type(translation_nm) is not tuple
        or len(translation_nm) != 2
        or any(type(value) is not int for value in translation_nm)
    ):
        from clay.layout.generic_placement import RowFinalPlacementResidualError

        raise RowFinalPlacementResidualError(
            ("ROW_ARRAY_TRANSLATION_CERTIFICATE_MALFORMED",)
        )
    x0, y0, x1, y1 = arr.extent()
    exact_center = (Fraction(int(x0) + int(x1), 2),
                    Fraction(int(y0) + int(y1), 2))
    stamp_origin = (
        exact_center[0].numerator // exact_center[0].denominator,
        exact_center[1].numerator // exact_center[1].denominator,
    )
    return _ArrayExtentTranslation(
        exact_center_nm=exact_center,
        stamp_origin_nm=stamp_origin,
        translation_nm=translation_nm,
    )


def array_member_offsets_um(arr: Any) -> Dict[str, Tuple[float, float]]:
    """``{member -> (dx, dy)}`` -- each member's centroid offset (um) from the
    array's lower integer extent origin, derived from the built
    :class:`SharedDiffusionArray`.

    The member x-centre is the midpoint of that member's finger x-span; the y is
    the shared channel centre (all members share the one OD strip). Both the
    compact placer (to co-place the members within the one array footprint) and
    the emit path (to seat the ONE array stamp so its per-member fingers land on
    the per-member pin probes) read the SAME integer-origin offsets from this one
    function -- no second derivation."""
    x0, y0, x1, y1 = arr.extent()
    ecx, ecy = (x0 + x1) // 2, (y0 + y1) // 2
    chan_cy = float(getattr(arr, "channel_width_nm", 0)) / 2.0
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for f in getattr(arr, "fingers", []) or []:
        spans.setdefault(str(f.member), []).append((int(f.x0), int(f.x1)))
    out: Dict[str, Tuple[float, float]] = {}
    for m, sp in spans.items():
        mcx = (min(s[0] for s in sp) + max(s[1] for s in sp)) / 2.0
        out[m] = (
            int(round(mcx - ecx)) / _NM_PER_UM,
            int(round(chan_cy - ecy)) / _NM_PER_UM,
        )
    return out


def _array_cell_geometry(
    arr: Any,
    name: str,
    kind: str,
    members: Sequence[str] = (),
    *,
    translation: Optional[_ArrayExtentTranslation] = None,
    body_tap_owner_sha256: str = "",
    body_net: str = "",
) -> Any:
    """Convert absolute-nm array polygons into the common placed CellGeometry.

    The stamp is expressed about the array's lower integer extent origin (the same
    origin :func:`array_member_offsets_um` measures from).  Seating that origin at
    the certificate-derived position applies one rigid integer translation even
    when the geometric extent centre lies on a half-nm coordinate.

    PROVENANCE (Gap A): the stamp declares its fused member names on
    ``composite_members`` -- the SAME provenance the interdigitate composite stamp
    carries -- so every downstream consumer that resolves a DECLARED member back to
    its placed footprint (``group_guard_rings`` ring clustering, per-member pins,
    LVS) maps the member THROUGH this stamp instead of reporting it "no geometry
    stamp" and silently dropping its guard ring."""
    from clay.layout.device_cell import _validate_body_tap_owner_sha256
    from clay.layout.extract.library_device import CellGeometry

    if translation is None:
        from clay.layout.generic_placement import RowFinalPlacementResidualError

        raise RowFinalPlacementResidualError(
            ("ROW_ARRAY_TRANSLATION_CERTIFICATE_MISSING",)
        )
    frame = translation
    body_tap_owner_sha256 = _validate_body_tap_owner_sha256(
        body_tap_owner_sha256
    )
    if type(body_net) is not str or body_net != body_net.strip():
        raise ValueError("array body_net must be an exact canonical string")
    if bool(body_tap_owner_sha256) != bool(body_net):
        raise ValueError(
            "array body-tap owner requires one exact non-empty body net"
        )
    ecx, ecy = frame.stamp_origin_nm
    pbl: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
    for lp in arr.polygons:
        key = (int(lp.layer), int(lp.datatype))
        ring = [((float(v.x) - ecx) / _NM_PER_UM, (float(v.y) - ecy) / _NM_PER_UM)
                for v in lp.polygon.vertices]
        pbl.setdefault(key, []).append(ring)
    cg = CellGeometry(
        name=str(name), kind=str(kind), ref_centroid=(0.0, 0.0),
        polys_by_layer=pbl, clean=True,
    )
    cg.composite_members = tuple(str(m) for m in members)
    from clay.layout.owned_artwork import bind_row_owned_artwork_stamp
    bind_row_owned_artwork_stamp(arr, cg, native_origin_nm=frame.stamp_origin_nm, consumed_devices=tuple(members))
    cg.body_tap_owner_sha256 = body_tap_owner_sha256
    cg.body_net = body_net
    fixed_pcell = getattr(arr, "fixed_pcell", None)
    if fixed_pcell:
        if type(fixed_pcell) is not dict:
            raise TypeError("array fixed PCell authority is malformed")
        cg.fixed_pcell = dict(fixed_pcell)
    if getattr(arr, "mosaic_guest_slot_binding", None) is not None:
        from clay.layout.mosaic_guest_slot_stamp import bind_mosaic_guest_slot_stamp
        bind_mosaic_guest_slot_stamp(arr, cg, native_origin_nm=frame.stamp_origin_nm)
    if not hasattr(cg, "physical_occurrences_by_member"):
        from clay.layout.native_row_occurrences import native_row_occurrences_by_member

        cg.physical_occurrences_by_member = native_row_occurrences_by_member(
            arr,
            consumed_devices=tuple(members),
            native_origin_nm=frame.stamp_origin_nm,
        )
    return cg


def _stored_row_consumptions(
    placement_obj: Any,
    intent: Any,
    devices: Sequence[_BridgedDevice],
) -> Tuple[List[_ArrayConsumption], Dict[str, str]]:
    """Project the canonical Placement row master without rebuilding arrays."""
    _array_extent_translation = _rc_attr("_array_extent_translation", globals()["_array_extent_translation"])
    declared = bool(
        (getattr(intent, "arrays", ()) or ())
        or (getattr(intent, "common_centroids", ()) or ())
        or (getattr(intent, "discrete_mosaics", ()) or ())
    )
    program = getattr(placement_obj, "row_program", None)
    materialization = getattr(placement_obj, "row_materialization", None)
    if not declared and program is None and materialization is None:
        return [], {}
    from clay.layout.generic_placement import (
        RowFinalPlacementResidualError,
        row_macro_member_offsets_nm,
        verify_stored_row_projection,
    )

    if program is None or materialization is None:
        raise RowFinalPlacementResidualError(("ROW_MASTER_METADATA_INCOMPLETE",))
    verify_stored_row_projection(placement_obj)
    if tuple(getattr(program, "residuals", ()) or ()) or tuple(
        getattr(materialization, "residuals", ()) or ()
    ):
        raise RowFinalPlacementResidualError(("ROW_MASTER_HAS_RESIDUALS",))
    certificate = getattr(materialization, "certificate", None)
    if (
        certificate is None
        or getattr(certificate, "program_sha256", None) != getattr(program, "sha256", None)
    ):
        raise RowFinalPlacementResidualError(("ROW_MASTER_CERTIFICATE_MISMATCH",))
    final_certificate = getattr(placement_obj, "row_final_certificate", None)
    if (
        final_certificate is None
        or getattr(final_certificate, "program_sha256", None) != getattr(program, "sha256", None)
    ):
        raise RowFinalPlacementResidualError(("ROW_FINAL_CERTIFICATE_MISMATCH",))
    bindings = {
        item.entry_id: item
        for item in tuple(getattr(final_certificate, "macro_bindings", ()) or ())
    }
    if len(bindings) != len(tuple(getattr(program, "entries", ()) or ())):
        raise RowFinalPlacementResidualError(("ROW_FINAL_BINDING_CARDINALITY",))
    realizations = {
        item.entry_id: item
        for item in tuple(getattr(materialization, "realizations", ()) or ())
    }
    if len(realizations) != len(tuple(getattr(program, "entries", ()) or ())):
        raise RowFinalPlacementResidualError(("ROW_REALIZATION_CARDINALITY",))
    by_name = {device.name: device for device in devices}
    consumptions: List[_ArrayConsumption] = []
    member_group: Dict[str, str] = {}
    for entry in program.entries:
        realization = realizations.get(entry.entry_id)
        if realization is None or realization.owner != entry.owner:
            raise RowFinalPlacementResidualError(("ROW_REALIZATION_MISSING",))
        if tuple(realization.consumed_devices) != tuple(entry.consumed_devices):
            raise RowFinalPlacementResidualError(("ROW_REALIZATION_CONSUMPTION_MISMATCH",))
        binding = bindings.get(entry.entry_id)
        if (
            binding is None
            or binding.entry_sha256 != entry.sha256
            or binding.materialization_sha256 != materialization.certificate.sha256
            or binding.owner != entry.owner
            or tuple(binding.consumed_devices) != tuple(entry.consumed_devices)
            or type(binding.artwork_sha256) is not str
            or len(binding.artwork_sha256) != 64
            or type(binding.bbox_nm) is not tuple
            or len(binding.bbox_nm) != 4
        ):
            raise RowFinalPlacementResidualError(("ROW_FINAL_BINDING_MISMATCH",))
        kinds = {
            str(by_name[name].kind)
            for name in entry.consumed_devices
            if name in by_name
        }
        if len(kinds) != 1 or any(name not in by_name for name in entry.consumed_devices):
            raise RowFinalPlacementResidualError(("ROW_MEMBER_BRIDGE_MISMATCH",))
        member_offsets_nm = row_macro_member_offsets_nm(entry, realization)
        if any(name not in member_offsets_nm for name in entry.active_devices):
            raise RowFinalPlacementResidualError(("ROW_ACTIVE_MEMBER_OFFSET_MISSING",))
        lx0, ly0, lx1, ly1 = realization.local_bbox_nm
        bx0, by0, bx1, by1 = binding.bbox_nm
        translations = {
            (bx0 - lx0, by0 - ly0),
            (bx1 - lx1, by1 - ly1),
        }
        if len(translations) != 1:
            raise RowFinalPlacementResidualError(("ROW_FINAL_BINDING_NOT_RIGID",))
        dx_nm, dy_nm = next(iter(translations))
        translation = _array_extent_translation(
            realization.geometry, translation_nm=(dx_nm, dy_nm)
        )
        tile_center_nm = translation.stamp_position_nm
        if tile_center_nm != ((bx0 + bx1) // 2, (by0 + by1) // 2):
            raise RowFinalPlacementResidualError(("ROW_FINAL_BINDING_ORIGIN_MISMATCH",))
        delegation = getattr(entry, "body_tap_delegation", None)
        body_tap_owner_sha256 = (
            str(delegation.declaration_sha256) if delegation is not None else ""
        )
        body_net = str(delegation.tie_net) if delegation is not None else ""
        if (
            getattr(realization.geometry, "body_tap_owner_sha256", "")
            != body_tap_owner_sha256
            or (
                body_tap_owner_sha256
                and getattr(realization.geometry, "body_net", "") != body_net
            )
        ):
            raise RowFinalPlacementResidualError(("ROW_BODY_TAP_OWNER_MISMATCH",))
        consumption = _ArrayConsumption(
            group=entry.group,
            members=list(entry.active_devices),
            order=entry.order,
            kind=next(iter(kinds)),
            array=realization.geometry,
            fallback="",
            flank_dummies=list(entry.dummy_devices),
            certified_member_offsets_nm=member_offsets_nm,
            certified_tile_center_nm=tile_center_nm,
            certified_translation_nm=translation.translation_nm,
            body_tap_owner_sha256=body_tap_owner_sha256,
            body_net=body_net,
        )
        consumptions.append(consumption)
        for name in entry.consumed_devices:
            if name in member_group:
                raise RowFinalPlacementResidualError(("ROW_MEMBER_CONSUMED_TWICE",))
            member_group[name] = entry.group
    return consumptions, member_group


def _snap_matching_stamp_positions_to_manufacturing_grid(
    geometry: Dict[str, Any], placement_um: Dict[str, _XY], tech: Any,
    matching_pair_axes: Dict[Any, Any],
) -> None:
    """Select one lattice phase for a direct pair's exact artwork axis.

    Opposite sub-grid centroid residues cannot be snapped independently after
    per-polygon normalization: the two identical devices may acquire different
    widths. Normalize the pair on a common phase, then translate only whole
    grid steps to restore the exact measured artwork axis. Joint-row
    members have no separate stamp and remain owned by their native master.
    """
    import math
    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    from clay.layout.route.emit_gds import _nearest_on_grid, pdk_manufacturing_grid_nm
    grid = pdk_manufacturing_grid_nm(tech)
    if grid <= 1:
        return
    for pair, (dimension, axis) in sorted(matching_pair_axes.items()):
        if len(pair) != 2 or any(name not in geometry or name not in placement_um for name in pair):
            continue
        along = 0 if dimension == 'x' else 1
        across = 1 - along
        # A subsequent bridge may already carry the final normalized stamp;
        # replay its actual vertices rather than snapping its local origin again.
        pair_geometry = {name: geometry[name] for name in pair}
        current_boxes = _placed_stamp_bboxes_nm(pair_geometry, placement_um)
        if all(name in current_boxes for name in pair):
            a, b = (current_boxes[name] for name in pair)
            points = [((x + placement_um[name][0])*_NM_PER_UM,
                       (y + placement_um[name][1])*_NM_PER_UM)
                      for name in pair
                      for rings in geometry[name].polys_by_layer.values()
                      for ring in rings for x, y in ring]
            if ((a[2]-a[0], a[3]-a[1]) == (b[2]-b[0], b[3]-b[1])
                and a[along]+a[along+2]+b[along]+b[along+2] == 4*float(axis)
                and a[across]+a[across+2] == b[across]+b[across+2]
                and all(abs(value - _nearest_on_grid(round(value), grid)) < 1e-6
                        for point in points for value in point)):
                continue
        orig_pos = {name: tuple(float(v) for v in placement_um[name]) for name in pair}
        positions = [[float(value) * _NM_PER_UM for value in placement_um[name]] for name in pair]
        if (dimension not in ('x', 'y') or
            not math.isclose(positions[0][along] + positions[1][along], 2 * float(axis), abs_tol=1e-6, rel_tol=0.) or
            not math.isclose(positions[0][across], positions[1][across], abs_tol=1e-6, rel_tol=0.)):
            raise ValueError('matching position does not replay its declared axis before grid projection')
        twice_axis = round(2 * float(axis))
        if not math.isclose(twice_axis, 2 * float(axis), abs_tol=1e-6, rel_tol=0.) or twice_axis % grid:
            raise ValueError('matching position axis has no exact manufacturing-grid phase')
        b0 = current_boxes.get(pair[0])
        if b0 is not None and ((b0[along + 2] - b0[along]) % 2) == 1:
            low_along = _nearest_on_grid(round(b0[along]), grid)
            positions[0][along] = positions[0][along] + (low_along - b0[along])
        else:
            positions[0][along] = _nearest_on_grid(round(positions[0][along]), grid)
        positions[1][along] = twice_axis - positions[0][along]
        if b0 is not None and ((b0[across + 2] - b0[across]) % 2) == 1:
            low_across = _nearest_on_grid(round(b0[across]), grid)
            perpendicular = positions[0][across] + (low_across - b0[across])
        else:
            perpendicular = _nearest_on_grid(round(positions[0][across]), grid)
        positions[0][across] = positions[1][across] = perpendicular
        for name, position in zip(pair, positions):
            placement_um[name] = tuple(value / _NM_PER_UM for value in position)
        pair_geometry = {name: geometry[name] for name in pair}
        _normalize_materialized_geometry_to_manufacturing_grid(pair_geometry, placement_um, tech)
        boxes = _placed_stamp_bboxes_nm(pair_geometry, placement_um)
        if any(name not in boxes for name in pair):
            raise ValueError('matching position lacks measurable normalized artwork')
        a, b = (boxes[name] for name in pair)
        if (a[2] - a[0], a[3] - a[1]) != (b[2] - b[0], b[3] - b[1]):
            raise ValueError('matching position normalization changed pair sizes unequally')
        centers = [((box[0] + box[2]) / 2., (box[1] + box[3]) / 2.) for box in (a,b)]
        delta = [0., 0.]
        delta[along] = twice_axis - centers[0][along] - centers[1][along]
        delta[across] = centers[0][across] - centers[1][across]
        if any(value % grid for value in delta):
            raise ValueError('matching position artwork cannot replay its axis by a rigid grid translation')
        placement_um[pair[1]] = tuple(placement_um[pair[1]][i] + delta[i] / _NM_PER_UM for i in range(2))
        for name in pair:
            stamp = geometry[name]
            orig_u = getattr(stamp, "passive_artwork_origin_um", None)
            if orig_u is not None:
                dx_um = placement_um[name][0] - orig_pos[name][0]
                dy_um = placement_um[name][1] - orig_pos[name][1]
                updated_origin = (float(orig_u[0]) + dx_um, float(orig_u[1]) + dy_um)
                if is_dataclass(stamp):
                    geometry[name] = replace(stamp, passive_artwork_origin_um=updated_origin)
                else:
                    stamp.passive_artwork_origin_um = updated_origin


def _normalize_materialized_geometry_to_manufacturing_grid(
    geometry: Dict[str, Any],
    placement_um: Dict[str, _XY],
    tech: Any,
) -> int:
    """Resolve stamp-centroid half-DBU residue before routing and exact emit.

    Device artwork is authored in integer nanometres, but ``CellGeometry`` stores
    it relative to a centroid in micrometres.  An odd-nanometre extent can therefore
    create half-nanometre local coordinates; adding an independently integer-rounded
    placement centre does not necessarily reconstruct the original manufacturing-
    grid vertex.  Final emit historically rounded those vertices, which is a real
    geometry transformation and is forbidden by the exact-artifact contract.

    Quantize each *absolute* stamp vertex with the same nearest-grid rule before
    obstacle construction, pin landing, routing, placement certification, and emit.
    The stamp remains centroid-relative and connectivity/layer ownership are
    unchanged.  Reapplying this function is byte-idempotent.
    """

    from clay.layout.route.emit_gds import (
        _nearest_on_grid,
        pdk_manufacturing_grid_nm,
    )

    grid_nm = pdk_manufacturing_grid_nm(tech)
    if grid_nm <= 1:
        return 0
    changed = 0
    for name, stamp in (geometry or {}).items():
        position = (placement_um or {}).get(name)
        polygons = getattr(stamp, "polys_by_layer", None)
        if position is None or not isinstance(polygons, dict):
            continue
        cx_um, cy_um = float(position[0]), float(position[1])
        if getattr(stamp, 'source', '') in {'shared_diffusion_array', 'discrete_mosaic'}:
            from clay.layout.placed_geometry import placed_ring_nm

            # Native rows already bind every mask and occurrence to one local
            # origin. Rebasing their polygons here silently invalidates that
            # origin (GF180 odd extents exposed a uniform 2-nm discrepancy).
            # Move the origin and logical aliases by one common lattice phase;
            # never repair individual vertices of certified row artwork.
            first_x: Optional[int] = None
            first_y: Optional[int] = None
            rx0 = 0
            ry0 = 0
            n_points = 0
            for rings in polygons.values():
                for ring in rings:
                    if not ring:
                        continue
                    pts_nm = placed_ring_nm(ring, position)
                    n_points += len(pts_nm)
                    for x_nm, y_nm in pts_nm:
                        if first_x is None:
                            first_x = x_nm
                            first_y = y_nm
                            rx0 = x_nm % grid_nm
                            ry0 = y_nm % grid_nm
                        elif x_nm % grid_nm != rx0 or y_nm % grid_nm != ry0:
                            raise ValueError('native row manufacturing-grid phase requires non-rigid vertex changes')
            if first_x is None or first_y is None:
                raise ValueError('native row manufacturing-grid phase has no artwork')
            dx_nm = _nearest_on_grid(first_x, grid_nm) - first_x
            dy_nm = _nearest_on_grid(first_y, grid_nm) - first_y
            if dx_nm or dy_nm:
                placement_um[name] = (cx_um + dx_nm / _NM_PER_UM,
                                      cy_um + dy_nm / _NM_PER_UM)
                for member in getattr(stamp, 'composite_members', ()):
                    if member != name and member in placement_um and member not in geometry:
                        mx, my = placement_um[member]
                        placement_um[member] = (mx + dx_nm / _NM_PER_UM, my + dy_nm / _NM_PER_UM)
                changed += n_points
            continue
        half_g = grid_nm // 2
        nm_per_um = _NM_PER_UM
        normalized: Dict[Any, Any] = {}
        for layer, rings in polygons.items():
            normalized_rings = []
            for ring in rings:
                if not ring:
                    normalized_rings.append([])
                    continue
                r0x, r0y = ring[0]
                abs0_x = (float(r0x) + cx_um) * nm_per_um
                abs0_y = (float(r0y) + cy_um) * nm_per_um
                anchor_x_nm = round(abs0_x)
                anchor_y_nm = round(abs0_y)
                delta_x_nm = ((anchor_x_nm + half_g) // grid_nm) * grid_nm - abs0_x
                delta_y_nm = ((anchor_y_nm + half_g) // grid_nm) * grid_nm - abs0_y
                normalized_ring = []
                for raw_x_um, raw_y_um in ring:
                    abs_x = (float(raw_x_um) + cx_um) * nm_per_um
                    abs_y = (float(raw_y_um) + cy_um) * nm_per_um
                    abs_x_nm = round(abs_x)
                    abs_y_nm = round(abs_y)
                    snapped_x_nm = ((round(abs_x + delta_x_nm) + half_g) // grid_nm) * grid_nm
                    snapped_y_nm = ((round(abs_y + delta_y_nm) + half_g) // grid_nm) * grid_nm
                    if snapped_x_nm != abs_x_nm or snapped_y_nm != abs_y_nm:
                        changed += 1
                    normalized_ring.append((
                        snapped_x_nm / nm_per_um - cx_um,
                        snapped_y_nm / nm_per_um - cy_um,
                    ))
                normalized_rings.append(normalized_ring)
            normalized[layer] = normalized_rings
        stamp.polys_by_layer = normalized
    return changed


def _manufacturing_grid_bbox_access(
    bbox_nm: Sequence[int], tech: Any
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
    """Return an on-grid conductor bbox and one contained on-grid access point."""

    from clay.layout.route.emit_gds import (
        _nearest_on_grid,
        pdk_manufacturing_grid_nm,
    )

    if len(tuple(bbox_nm)) != 4:
        raise ValueError("materialized terminal bbox must contain four coordinates")
    grid_nm = pdk_manufacturing_grid_nm(tech)
    bbox = tuple(
        _nearest_on_grid(int(value), grid_nm) for value in tuple(bbox_nm)
    )
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("manufacturing-grid terminal bbox must have positive area")
    access = (
        _nearest_on_grid((x0 + x1) // 2, grid_nm),
        _nearest_on_grid((y0 + y1) // 2, grid_nm),
    )
    if not (x0 <= access[0] <= x1 and y0 <= access[1] <= y1):
        raise ValueError("manufacturing-grid terminal access escaped its bbox")
    return bbox, access


_MIRROR_X_ORIENT_MAP = {
    "R0": "MR180",
    "R90": "MR90",
    "R180": "MR0",
    "R270": "MR270",
    "MR0": "R180",
    "MR90": "R90",
    "MR180": "R0",
    "MR270": "R270",
}


def _placed_poly_rel_inventory(
    pd: Any, *, mirror_x: bool = False, layers: Optional[set] = None
) -> Counter:
    box = tuple(getattr(pd, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
    cx2 = int(box[0]) + int(box[2])
    cy2 = int(box[1]) + int(box[3])
    sign = -1 if mirror_x else 1
    inventory: Counter = Counter()
    for lp in getattr(pd, "polys", ()) or ():
        layer_key = (int(lp.layer), int(lp.datatype))
        if layers is not None and layer_key not in layers:
            continue
        pts = tuple(
            (sign * (2 * int(v.x) - cx2), 2 * int(v.y) - cy2)
            for v in lp.polygon.vertices
        )
        if pts and pts[-1] == pts[0]:
            pts = pts[:-1]
        if not pts:
            continue
        rev = tuple(reversed(pts))
        key = min(seq[i:] + seq[:i] for seq in (pts, rev) for i in range(len(seq)))
        inventory[(layer_key[0], layer_key[1], key)] += 1
    return inventory


def _terminal_x_signature_nm(pd: Any, *, mirror_x: bool = False) -> tuple:
    box = tuple(getattr(pd, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
    cx2 = int(box[0]) + int(box[2])
    sign = -1 if mirror_x else 1
    accesses = dict(getattr(pd, "terminal_accesses", {}) or {})
    if not accesses:
        accesses = {k: (v,) for k, v in (getattr(pd, "terminals", {}) or {}).items()}
    return tuple(
        (role, tuple(sorted(sign * (2 * int(pt[0]) - cx2) for pt in pts)))
        for role, pts in sorted(accesses.items())
        if role in {"s", "d", "g"}
    )


def _stamp_x_signature_nm(stamp: Any) -> tuple:
    accesses = dict(getattr(stamp, "terminal_accesses_um", {}) or {})
    if not accesses:
        accesses = {
            k: (v,)
            for k, v in (getattr(stamp, "terminal_offsets_um", {}) or {}).items()
        }
    return tuple(
        (role, tuple(sorted(round(2.0 * float(pt[0]) * _NM_PER_UM) for pt in pts)))
        for role, pts in sorted(accesses.items())
        if role in {"s", "d", "g"}
    )


def _mirror_cell_geometry_x(stamp: Any) -> None:
    stamp.polys_by_layer = {
        layer: [[(-float(x), float(y)) for x, y in ring] for ring in rings]
        for layer, rings in (getattr(stamp, "polys_by_layer", {}) or {}).items()
    }
    accesses = getattr(stamp, "terminal_accesses_um", None)
    if isinstance(accesses, dict):
        stamp.terminal_accesses_um = {
            role: tuple((-float(x), float(y)) for x, y in pts)
            for role, pts in accesses.items()
        }
    offsets = getattr(stamp, "terminal_offsets_um", None)
    if isinstance(offsets, dict):
        stamp.terminal_offsets_um = {
            role: (-float(pt[0]), float(pt[1]))
            for role, pt in offsets.items()
        }
    occs = getattr(stamp, "physical_occurrences", None)
    if occs:
        stamp.physical_occurrences = tuple(
            sorted(
                (-float(x), float(y), _MIRROR_X_ORIENT_MAP.get(str(orient), str(orient)))
                for x, y, orient in occs
            )
        )


def _sync_matching_pair_stamp_mirrors(
    placed: Mapping[str, Any],
    geometry: Dict[str, Any],
    matching_pair_axes: Mapping[Any, Any],
) -> None:
    """Retain placer X-mirroring on CellGeometry stamps rebuilt for matched pairs."""
    for raw_pair, axis_spec in sorted((matching_pair_axes or {}).items()):
        if (
            len(raw_pair) != 2
            or type(axis_spec) is not tuple
            or len(axis_spec) != 2
            or axis_spec[0] != "x"
        ):
            continue
        a, b = raw_pair
        if a not in placed or b not in placed or a not in geometry or b not in geometry:
            continue
        if type(geometry[a]) is _PlacedBjtStamp or type(geometry[b]) is _PlacedBjtStamp:
            continue
        pa, pb = placed[a], placed[b]
        inv_a = _placed_poly_rel_inventory(pa, mirror_x=False)
        inv_b = _placed_poly_rel_inventory(pb, mirror_x=False)
        if not inv_a or inv_a == inv_b or _placed_poly_rel_inventory(pa, mirror_x=True) != inv_b:
            continue
        for member in (a, b):
            pm = placed[member]
            sig_id = _terminal_x_signature_nm(pm, mirror_x=False)
            sig_mir = _terminal_x_signature_nm(pm, mirror_x=True)
            sig_st = _stamp_x_signature_nm(geometry[member])
            if sig_id != sig_mir and sig_st == sig_mir and sig_st != sig_id:
                _mirror_cell_geometry_x(geometry[member])


def _bridge_placement_to_route_inputs(
    placement_obj: Any, cell_def: Any, tech_id: str,
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
    dense_contacts: bool = False, intent: Any = None,
) -> Tuple[_BridgedIntent, Dict[str, _XY], Dict[str, Any]]:
    from clay.layout.place.row_replay_session import row_replay_invocation, bind_current_row_materialization
    tech = LayoutTech.resolve(str(tech_id))
    with row_replay_invocation(source_cell=cell_def, tech=tech):
        bind_current_row_materialization(getattr(placement_obj, 'row_materialization', None),
            program=getattr(placement_obj, 'row_program', None), tech=tech)
        return _bridge_placement_to_route_inputs_impl(placement_obj, cell_def, tech_id,
            device_records=device_records, dense_contacts=dense_contacts, intent=intent)


def _bridge_placement_to_route_inputs_impl(
    placement_obj: Any,
    cell_def: Any,
    tech_id: str,
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
    dense_contacts: bool = False,
    intent: Any = None,
) -> Tuple[_BridgedIntent, Dict[str, _XY], Dict[str, Any]]:
    """Bridge a :class:`Placement` (generic placer output) + parsed ``CellDef`` into
    ``(intent, placement_um, geometry)`` -- exactly the triple ``extract_pins`` /
    ``build_layout_model`` consume.

    * ``intent``       -- a ``_BridgedIntent`` (MOS devices with net bindings + W/L/M).
    * ``placement_um`` -- ``{device_name -> (x_um, y_um)}`` centroid, NM->um from the
      placed bbox centre (so a mirrored member's centroid is on the right side of
      the axis).
    * ``geometry``     -- ``{device_name -> CellGeometry}`` LEAKAGE-FREE parametric
      stamps (:func:`build_device_geometry` from the netlist kind/W/L/M + the PDK),
      never a reference capture.

    The device net graph + sizing come from the parsed ``CellDef.devices`` when it
    is populated (a ``design.clay`` whose devices are materialised in the IR -- the
    comparator path, byte-unchanged). When that list is EMPTY -- a topology-only
    ``design.clay`` whose net bindings live solely in the bound source ``.cir``
    (the production folded-cascode / two-stage cells) -- the bridge falls back to
    ``device_records`` (the SAME ``parse_cir_mos_records`` + ``parse_cir_passive_records``
    map the placer + the router netlist consume). This is the netlist source, never a
    per-circuit branch; without it ``route_cell`` extracts ZERO nets for a netlist-bound
    cell and routes nothing.

    A device the placer did not place (absent from ``placement_obj.devices``) is
    skipped (its terminals surface as honest gaps in the pin report downstream)."""
    # cell_def -> bridged MOS/passive lists (materialised devices, else the bound-.cir
    # records fallback). SINGLE source shared with the compaction placer bridge.
    _array_cell_geometry = _rc_attr("_array_cell_geometry", globals()["_array_cell_geometry"])
    _array_extent_translation = _rc_attr("_array_extent_translation", globals()["_array_extent_translation"])
    _stored_row_consumptions = _rc_attr("_stored_row_consumptions", globals()["_stored_row_consumptions"])
    bridge_devices_from_cell_def = _rc_attr("bridge_devices_from_cell_def", _base_bridge_devices_from_cell_def)
    from clay.layout.place.row_replay_session import row_phase_call
    native_geometry = getattr(placement_obj, "native_geometry", None)
    if native_geometry is not None:
        from clay.layout.native_sky130 import NativeGeometryBundle, native_route_devices
        from clay.port_roles import resolve_positive_supply_net, resolve_substrate_net
        if type(native_geometry) is not NativeGeometryBundle or str(tech_id) != "sky130":
            raise ValueError("native route bridge requires the bound SKY130 geometry provider")
        has_rows = getattr(placement_obj, "row_program", None) is not None
        if has_rows:
            native_geometry.require_partitioned_placement(placement_obj, cell_def)
        else:
            native_geometry.require_placement(placement_obj, cell_def)
        bridged = _BridgedIntent(native_route_devices(cell_def, native_geometry),
            substrate_net=resolve_substrate_net(cell_def, fallback=""),
            deep_nwell_net=resolve_positive_supply_net(cell_def, fallback=""))
        rows, row_members = _stored_row_consumptions(placement_obj, intent, bridged.devices) if has_rows else ([], {})
        positions, geometry = {}, {}
        for name, placed in placement_obj.devices.items():
            x0, y0, x1, y1 = placed.bbox
            cx, cy = (x0+x1)/2, (y0+y1)/2
            positions[name] = (cx/_NM_PER_UM, cy/_NM_PER_UM)
            if name not in row_members:
                stamp = _PlacedBjtStamp(placed, cx, cy)
                native = native_geometry.geometries[name]
                nodes = native.native_evidence['terminal_nodes']
                if (type(nodes) is not dict or set(nodes) != set(native.terminal_accesses) or
                        set(nodes) != set(stamp.terminal_accesses_um)):
                    raise ValueError('native terminal node role inventory changed: ' + name)
                ledger = {}
                for role in sorted(nodes):
                    names = nodes[role]
                    if (type(names) is not list or not names or names != sorted(set(names)) or
                            any(type(node) is not str or not node for node in names) or
                            len(names) != len(stamp.terminal_accesses_um[role])):
                        raise ValueError('native terminal node/access inventory changed: ' + name + '.' + role)
                    ledger[role] = tuple(names)
                stamp.native_terminal_nodes = ledger
                device = next(item for item in bridged.devices if item.name == name)
                stamp.native_terminal_authority = tuple(
                    (role, ledger[role], tuple(stamp.terminal_accesses_um[role]),
                     tuple(stamp.terminal_layers[role]), device.native_terminals[role])
                    for role in sorted(ledger))
                geometry[name] = stamp
        for ac in rows:
            from clay.layout.generic_placement import RowFinalPlacementResidualError
            if ac.certified_tile_center_nm is None or ac.certified_translation_nm is None:
                raise RowFinalPlacementResidualError(("ROW_ARRAY_TRANSLATION_CERTIFICATE_MISSING",))
            translation = _array_extent_translation(ac.array, translation_nm=ac.certified_translation_nm)
            if translation.stamp_position_nm != ac.certified_tile_center_nm:
                raise RowFinalPlacementResidualError(("ROW_CERTIFIED_TRANSLATION_MISMATCH",))
            positions[ac.group] = tuple(value / _NM_PER_UM for value in translation.stamp_position_nm)
            geometry[ac.group] = row_phase_call('route_bridge.array_stamp', _array_cell_geometry,
                ac.array, ac.group, ac.kind, list(ac.members) + list(ac.flank_dummies),
                translation=translation, body_tap_owner_sha256=ac.body_tap_owner_sha256,
                body_net=ac.body_net)
        return bridged, positions, geometry
    if any(getattr(device, "device_class", None) == "native"
           for device in getattr(cell_def, "devices", ())):
        raise ValueError("native route bridge is missing its source-bound geometry")
    devices, passives = bridge_devices_from_cell_def(cell_def, device_records)
    bjts = bridge_bjts_from_cell_def(cell_def)
    _tech_resolved = LayoutTech.resolve(str(tech_id))
    row_phase_call('route_bridge.nonmos', replay_native_nonmos_placement,
        placement_obj, source_cell=cell_def, tech=_tech_resolved, intent=intent)
    from clay.port_roles import resolve_positive_supply_net, resolve_substrate_net
    from clay.well_isolation import declared_well_isolated_devices

    _isolated = declared_well_isolated_devices(cell_def)
    bridged = _BridgedIntent(
        devices,
        passives,
        bjts,
        well_isolated_devices=_isolated,
        substrate_net=resolve_substrate_net(
            cell_def, fallback=_tech_resolved.substrate_net()
        ),
        deep_nwell_net=resolve_positive_supply_net(cell_def, fallback=""),
    )
    substrate_rail_nets = _declared_ground_ports(cell_def)

    # DECLARED shared-diffusion @array consumption (Stage-1 craft): a device that
    # belongs to a VALID declared shared-diffusion array is NOT stamped as its own
    # per-device geometry; instead the whole array is stamped ONCE below, keyed on
    # the array's GROUP label, as ONE continuous merged-OD strip. ``array_member_
    # group`` maps each such member -> its group. A cell that declares no
    # (materializable) shared-diffusion array leaves this EMPTY -> every device
    # takes the per-device path below, BYTE-IDENTICAL. ``intent`` is optional so
    # the historical positional callers (byte-identical) still work.
    array_consumptions: List[_ArrayConsumption] = []
    array_member_group: Dict[str, str] = {}
    if intent is not None:
        array_consumptions, array_member_group = row_phase_call('route_bridge.stored_rows', _stored_row_consumptions,
            placement_obj, intent, devices
        )
    # A direct singleton ``@array(M, style=discrete)`` is one logical MOS whose
    # multiplicity was materialized as independent unit cells by the placer.  It
    # is deliberately not a shared-diffusion row macro, so the row-consumption
    # inventory above does not own it.  Preserve that exact placed artwork at the
    # bridge instead of rebuilding the same W/L/M as one continuous finger strip.
    device_names = {device.name for device in devices}
    direct_discrete_mos = {
        str(declaration.group)
        for declaration in tuple(getattr(intent, "arrays", ()) or ())
        if str(getattr(declaration, "style", "")).strip().lower() == "discrete"
        and str(getattr(declaration, "group", "")) in device_names
    } if intent is not None else set()
    # A certified ordered-row construction already owns exact native fingers,
    # contact population, mirrors and terminal accesses.  The generic compact
    # stamp builder may choose another strip aspect even when W/M are equal.
    # Carry the existing exact placed-MOS path across this boundary instead.
    ordered_row_mos = set()
    ordered_evidence = getattr(placement_obj, "ordered_row_evidence", None)
    if ordered_evidence is not None:
        from clay.layout.place.ordered_mirror_rows import verify_ordered_row_source
        verify_ordered_row_source(placement_obj, cell_def)
        ordered_row_mos = {name for _row, members in ordered_evidence.program.rows
                           for name in members}

    # Per-device ``@dense_contacts`` intent DECLARED in the .clay annotate block: the
    # rebuilt route stamp must honor it exactly as the placer does (one canonical
    # geometry). Read the parsed device-level DenseContacts annotations off the
    # CellDef (device_id -> list[LayoutAnnotation]); a device with none stays sparse.
    _dense_set: set = set()
    # An explicit orthogonal transform owns final artwork.  Such a MOS must cross
    # this bridge as the placer's exact polygons/accesses; scalar reconstruction
    # would silently reset its orientation to R0.
    _orient_set: set = set()
    # Per-device ``@fold`` intent DECLARED in the .clay: the route stamp must fold the
    # device IDENTICALLY to the placer (one canonical geometry) -- ``label -> h_max_um``
    # (or None -> module-default finger-height bound).
    _fold_map: Dict[str, Any] = {}
    try:
        from clay.layout.annotations import DenseContacts, Fold, Orient
        for _tgt, _anns in (getattr(cell_def, "__dict__", {}).get(
                "_layout_annotations", {}) or {}).items():
            if any(isinstance(_a, DenseContacts) for _a in (_anns or ())):
                _dense_set.add(str(_tgt))
            for _a in (_anns or ()):
                if isinstance(_a, Orient):
                    _orient_set.add(str(_tgt))
                if isinstance(_a, Fold):
                    _fold_map[str(_tgt)] = _a.h_max_um
    except Exception:  # noqa: BLE001 - annotations are optional
        _dense_set = set()
        _orient_set = set()
        _fold_map = {}

    # DECLARED ``@fold`` is applied ONCE, HERE, by REWRITING the bridged device's
    # per-finger W and finger count M to the SHARED :func:`folded_finger_plan` split
    # (W_eff = M*W conserved EXACTLY). Every downstream route consumer then reads the
    # SAME folded device -- the geometry stamp (build_device_geometry below), the
    # FOLD-AWARE pin-probe (extract_pins(per_device_fold=True) re-derives the probe
    # from dev.W/dev.M, so it MUST see the folded values or the gate pin lands on the
    # UNFOLDED single-finger coordinate and floats: the LVS gate-open this closes),
    # and the obstacle model. This is the one-canonical-fold contract: fold in ONE
    # place, not per consumer. The LVS reference netlist is built from ``cell_def``
    # (unfolded W=40 M=1); KLayout combines the folded fingers back to the same total
    # W, so the device stays electrically identical and LVS-matches.
    from clay.layout.extract.fold import DEFAULT_H_MAX_UM, folded_finger_plan

    for dev in devices:
        if dev.name in _fold_map:
            _hmax = _fold_map[dev.name]
            _hmax = float(_hmax) if _hmax is not None else DEFAULT_H_MAX_UM
            w_finger_um, n_fingers = folded_finger_plan(
                dev.W, dev.L, max(1, int(dev.M)), h_max_um=_hmax
            )
            dev.W = w_finger_um
            dev.M = n_fingers

    placement_um: Dict[str, _XY] = {}
    geometry: Dict[str, Any] = {}
    placed = getattr(placement_obj, "devices", {}) or {}
    # When either member of a matched pair requires exact placer artwork
    # (@orient, discrete MOS, ordered row, physical occurrences), both halves
    # must cross via the same placed-artwork path so substrate-tap defaults and
    # orientations cannot diverge across the pair (rc_bridge normalization).
    placed_exact_mos = set(direct_discrete_mos) | set(ordered_row_mos) | set(_orient_set)
    for dev in devices:
        pd = placed.get(dev.name)
        if pd is None or dev.name in array_member_group:
            continue
        orient_val = str(getattr(pd, "orientation", "") or "").upper()
        if tuple(getattr(pd, "physical_occurrences", ()) or ()) or orient_val not in ("", "R0"):
            placed_exact_mos.add(dev.name)
    for raw_pair in (getattr(placement_obj, "matching_pair_axes", {}) or {}):
        if len(raw_pair) != 2:
            continue
        pair_a, pair_b = raw_pair
        if (
            (pair_a in placed_exact_mos or pair_b in placed_exact_mos)
            and pair_a not in array_member_group
            and pair_b not in array_member_group
            and pair_a in placed
            and pair_b in placed
            and tuple(getattr(placed[pair_a], "polys", ()) or ())
            and tuple(getattr(placed[pair_b], "polys", ()) or ())
            and dict(getattr(placed[pair_a], "terminal_accesses", {}) or {})
            and dict(getattr(placed[pair_b], "terminal_accesses", {}) or {})
        ):
            placed_exact_mos.add(pair_a)
            placed_exact_mos.add(pair_b)
    for dev in devices:
        pd = placed.get(dev.name)
        if pd is None:
            continue
        x0, y0, x1, y1 = pd.bbox  # NM
        placement_um[dev.name] = (
            (x0 + x1) / 2.0 / _NM_PER_UM,
            (y0 + y1) / 2.0 / _NM_PER_UM,
        )
        # DECLARED shared-diffusion array member: DO NOT stamp its own per-device
        # geometry -- the whole array is stamped ONCE below (one merged OD). The
        # member's placement centroid is KEPT (above) so extract_pins still probes
        # its per-terminal pins on the array's M1 (the MOS pin probe is built from
        # the netlist device + centroid, independent of the geometry dict). A cell
        # with no declared array leaves array_member_group EMPTY -> this never fires.
        if dev.name in array_member_group:
            continue
        fixed_pcell = getattr(pd, "fixed_pcell", {}) or {}
        if fixed_pcell:
            required = {
                "reference_gds_sha256",
                "reference_master",
                "artwork_sha256",
                "parameter_key",
                "unit_occurrences",
                "position_authority",
                "supported_scope",
            }
            has_collision_rectangles = "collision_rectangles_nm" in fixed_pcell
            collision_rectangles = fixed_pcell.get("collision_rectangles_nm", ())
            if (
                type(fixed_pcell) is not dict
                or not required <= set(fixed_pcell) <= required | {
                    "collision_rectangles_nm"
                }
                or any(
                    type(fixed_pcell[field]) is not str
                    or len(fixed_pcell[field]) != 64
                    or any(char not in "0123456789abcdef" for char in fixed_pcell[field])
                    for field in ("reference_gds_sha256", "artwork_sha256")
                )
                or type(fixed_pcell["reference_master"]) is not str
                or not fixed_pcell["reference_master"]
                or type(fixed_pcell["parameter_key"]) is not tuple
                or len(fixed_pcell["parameter_key"]) != 4
                or type(fixed_pcell["unit_occurrences"]) is not int
                or fixed_pcell["unit_occurrences"] <= 0
                or fixed_pcell["position_authority"] != "excluded"
                or fixed_pcell["supported_scope"]
                != "bound_ldo_exact_parameter_key_only"
                or (
                    has_collision_rectangles
                    and (
                        type(collision_rectangles) is not tuple
                        or len(collision_rectangles)
                        != fixed_pcell["unit_occurrences"]
                        or any(
                            type(rectangle) is not tuple
                            or len(rectangle) != 4
                            or any(type(value) is not int for value in rectangle)
                            or not (
                                0 <= rectangle[0] < rectangle[2]
                                and 0 <= rectangle[1] < rectangle[3]
                            )
                            for rectangle in collision_rectangles
                        )
                    )
                )
            ):
                raise ValueError("placed fixed PCell authority is malformed")
            geometry[dev.name] = _PlacedBjtStamp(pd, (x0 + x1) / 2.0, (y0 + y1) / 2.0)
            continue
        # Canonical evaluated occurrences own the placed artwork also when the
        # array was declared through a functional-group alias. Rebuilding W/L/M
        # here would erase that array's unit grid (historically kept only for
        # direct device targets). Consume the placer's witness unchanged.
        if dev.name in placed_exact_mos:
            if not tuple(getattr(pd, "polys", ()) or ()) or not dict(
                getattr(pd, "terminal_accesses", {}) or {}
            ):
                raise ValueError(
                    "placed exact MOS lacks artwork or terminal access"
                )
            geometry[dev.name] = _PlacedBjtStamp(
                pd, (x0 + x1) / 2.0, (y0 + y1) / 2.0
            )
            continue
        # ROLE-keyed (source/drain/gate/body), NOT term-keyed: build_device_geometry
        # / _build_device_nets take the device's declared ROLE nets and do the
        # role->term (s/d/g/b) mapping themselves. A term-keyed dict here silently
        # nulled every declared net (_build_device_nets(term_dict) matched no known
        # role key and returned None) -- the F2 incident. See _build_device_nets'
        # loud contract in clay.layout.extract.realize_render.
        role_nets = {role: dev.nets.get(role, "") for role in _ROLE_TO_TERM}
        # ONE canonical fold: the route geometry stamp must match the artwork the
        # PLACER drew. The placer folds a large strip MOS (e.g. the LDO pass FET
        # MM28, M=40) wide via the serpentine 2D-fold; re-rendering it here with the
        # default (no target) draws it TALL instead, so the stamp overruns the
        # placer-reserved footprint and its M1 drain plate covers every foreign M1
        # pad in its column (the cross-net-short / LVS-no-match root). The SHARED
        # gate ``resolve_device_fold_target`` returns the SAME per-device fold target
        # the placer used (or None -> single-row), so the stamp the obstacles, the
        # short oracle, and the emitted GDS all read is identical to the placement.
        # ONE canonical fold: a declared ``@fold`` device already had its dev.W/dev.M
        # rewritten to the multi-finger split above, so it renders SINGLE-ROW
        # multi-finger here (its serpentine target is suppressed -- the two fold
        # mechanisms are mutually exclusive). A non-@fold strip MOS (e.g. the LDO pass
        # FET) still folds via the SHARED serpentine gate ``resolve_device_fold_target``
        # so the stamp matches the placer-reserved footprint (else its M1 drain plate
        # overruns and shorts foreign pads).
        if dev.name in _fold_map:
            fold_target = None
            fold_max_rows = 1
        else:
            fold_target = resolve_device_fold_target(
                _tech_resolved, dev.kind, dev.W * 1e-6, dev.L * 1e-6, dev.M
            )
            fold_max_rows = 4
        # DENSE CONTACTS must ride the SAME canonical stamp as the fold: contact_rows
        # is single-row-only (unsupported with a serpentine fold), so apply it only
        # when this device is NOT serpentine-folded (fold_target is None). A @fold
        # device is SINGLE-ROW multi-finger (fold_target None), so its declared dense
        # column still applies. One canonical geometry, no second layout path.
        stamp_geom = build_device_geometry(
            tech_id, dev.kind, dev.W, dev.L, dev.M, nets=role_nets,
            substrate_rail_nets=substrate_rail_nets,
            row_height_target_um=fold_target,
            max_rows=fold_max_rows,
            contact_rows=((dense_contacts or dev.name in _dense_set)
                          and fold_target is None),
            oxide=dev.oxide,
        )
        if dev.name in _fold_map:
            stamp_geom.single_row_fold = True
        geometry[dev.name] = stamp_geom

    # PASSIVES: seat each placed resistor / MIM cap as a net-aware route node.
    # Its geometry stamp is the placer's OWN drawn artwork (PlacedDevice.polys)
    # re-centred to centroid-relative um -- the SAME one-canonical-geometry rule the
    # MOS fold authority uses, so the obstacle builder, the short oracle, and the
    # emitted GDS all read EXACTLY what the placer drew (no rebuild mismatch). A
    # passive the placer did not seat is skipped (its terminals surface as honest
    # pin-report gaps downstream).
    from clay.layout.source_resistor_placement import bridge_current_resistor_assemblies
    resistor_members = bridge_current_resistor_assemblies(
        placement_obj, placed, passives, geometry, placement_um)
    for psv in passives:
        if psv.name in resistor_members:
            continue
        pd = placed.get(psv.name)
        if pd is None:
            continue
        x0, y0, x1, y1 = pd.bbox  # NM
        cx_nm = (x0 + x1) / 2.0
        cy_nm = (y0 + y1) / 2.0
        placement_um[psv.name] = (cx_nm / _NM_PER_UM, cy_nm / _NM_PER_UM)
        from clay.layout.passive_terminal_binding import bind_placed_passive_access
        stamp = _PlacedPassiveStamp(pd, cx_nm, cy_nm)
        bind_placed_passive_access(stamp, pd, passive=psv,
                                  center_nm=(cx_nm, cy_nm), tech=_tech_resolved)
        geometry[psv.name] = stamp

    # BJT fixed cells: carry the placer's exact foundry artwork and C/B/E access
    # into the same geometry/placement currency as every other route obstacle.
    for bjt in bjts:
        pd = placed.get(bjt.name)
        if pd is None:
            continue
        x0, y0, x1, y1 = pd.bbox
        cx_nm = (x0 + x1) / 2.0
        cy_nm = (y0 + y1) / 2.0
        placement_um[bjt.name] = (cx_nm / _NM_PER_UM, cy_nm / _NM_PER_UM)
        geometry[bjt.name] = _PlacedBjtStamp(pd, cx_nm, cy_nm)

    # STAMP EACH VALID SHARED-DIFFUSION ARRAY ONCE, keyed on its GROUP label. The
    # array is ONE continuous merged-OD strip (adjacent fingers share diffusion);
    # its members were skipped in the device loop above, so this is the SINGLE
    # geometry the emit loop / obstacle builder / short oracle read for the whole
    # cluster (no per-member stamps -> no gap-separated OD islands, no duplicate
    # artwork). The array stamp is seated at the tile centre that carries each
    # member's finger x-centre onto the member's own placement sub-position (so the
    # per-member pin probes land on the array metal). Canonical row macros use the
    # RowFinalCertificate origin and the materializer's certified member offsets.
    # An array whose members were not placed is skipped honestly (no fabricated
    # stamp); a materialized array without translation authority is a typed stop.
    for ac in array_consumptions:
        if ac.array is None:
            continue  # fallback -> members already took the per-device path
        certified = ac.certified_tile_center_nm
        certified_translation = ac.certified_translation_nm
        from clay.layout.generic_placement import RowFinalPlacementResidualError

        if certified is None or certified_translation is None:
            raise RowFinalPlacementResidualError(
                ("ROW_ARRAY_TRANSLATION_CERTIFICATE_MISSING",)
            )
        if ac.certified_member_offsets_nm is None:
            raise RowFinalPlacementResidualError(
                ("ROW_CERTIFIED_MEMBER_OFFSETS_MISSING",)
            )
        # Verify that the final member placements still realize the same rigid
        # row-master translation, then seat the route stamp directly from the
        # RowFinalCertificate.  No member-centre inference is permitted.
        translation = _array_extent_translation(
            ac.array, translation_nm=certified_translation
        )
        if translation.stamp_position_nm != certified:
            raise RowFinalPlacementResidualError(
                ("ROW_CERTIFIED_TRANSLATION_MISMATCH",)
            )
        tcx = translation.stamp_position_nm[0] / _NM_PER_UM
        tcy = translation.stamp_position_nm[1] / _NM_PER_UM
        for m in ac.members:
            p = placement_um.get(m)
            offset_nm = ac.certified_member_offsets_nm.get(m)
            if p is None or offset_nm is None:
                raise RowFinalPlacementResidualError(
                    (f"ROW_CERTIFIED_MEMBER_PLACEMENT_MISSING:{m}",)
                )
            inferred = (
                p[0] - offset_nm[0] / _NM_PER_UM,
                p[1] - offset_nm[1] / _NM_PER_UM,
            )
            if (
                abs(inferred[0] - tcx) > 0.5 / _NM_PER_UM
                or abs(inferred[1] - tcy) > 0.5 / _NM_PER_UM
            ):
                raise RowFinalPlacementResidualError(
                    (f"ROW_CERTIFIED_MEMBER_TRANSLATION_MISMATCH:{m}",)
                )
        placement_um[ac.group] = (tcx, tcy)
        geometry[ac.group] = row_phase_call('route_bridge.array_stamp', _array_cell_geometry,
            ac.array, ac.group, ac.kind,
            list(ac.members) + list(ac.flank_dummies),
            translation=translation,
            body_tap_owner_sha256=ac.body_tap_owner_sha256,
            body_net=ac.body_net,
        )

    _sync_matching_pair_stamp_mirrors(
        placed,
        geometry,
        getattr(placement_obj, "matching_pair_axes", {}) or {},
    )
    _snap_matching_stamp_positions_to_manufacturing_grid(
        geometry, placement_um, _tech_resolved,
        getattr(placement_obj, "matching_pair_axes", {}) or {},
    )
    _normalize_materialized_geometry_to_manufacturing_grid(
        geometry, placement_um, _tech_resolved
    )
    return bridged, placement_um, geometry


@dataclass
class _AdjacentNativeRowPins:
    """Pin inventory from the immediately preceding native extraction."""
    operands: tuple
    pins: tuple
    members: frozenset
    snapshot: object
    facts: tuple
    report_facts: tuple
    used: bool = False


def _adjacent_native_row_snapshot(*, tech, geometry, placement_um, row_program,
                                  row_materialization, row_final_certificate, intent, source_cell):
    """Detach the existing source, PDK, row and placed-stamp replay operands."""
    from pathlib import Path
    from clay.layout.extract.library_device import CellGeometry
    from clay.layout.place.row_replay_session import _pdk_state
    from clay.layout.place.row_replay_snapshot import (
        freeze_row_values, geometry_snapshot, materialization_snapshot, source_projection_snapshot,
    )
    projection = row_materialization.source_projection
    witness = getattr(projection, 'native_witness', None)
    if (witness is None or source_cell is not witness.cell
            or any(type(stamp) not in (CellGeometry, _PlacedBjtStamp) for stamp in geometry.values())
            or not row_materialization.realizations
            or any(getattr(row.geometry, 'joint_row_binding', None) is None
                   for row in row_materialization.realizations)):
        raise TypeError('unsupported adjacent native pin replay operands')
    paths = [source_cell.source_span['file'], projection.authoring_source_cell.source_span['file']]
    if projection.source_cir:
        paths.append(projection.source_cir)
    stamps = []
    for name,stamp in sorted(geometry.items()):
        # The native route bridge uses the same slotted stamp class for free
        # native devices that the rail snapshot already handles.
        raw = (vars(stamp) if type(stamp) is CellGeometry else
               {slot: getattr(stamp, slot) for cls in type(stamp).__mro__
                for slot in cls.__dict__.get('__slots__', ()) if hasattr(stamp, slot)})
        stamps.append((name, id(stamp), freeze_row_values(raw)))
    return (freeze_row_values(vars(source_cell)), freeze_row_values(intent), freeze_row_values(row_program),
        row_final_certificate.to_canonical_bytes(), materialization_snapshot(row_materialization),
        tuple(geometry_snapshot(row.geometry, row.consumed_devices) for row in row_materialization.realizations),
        source_projection_snapshot(projection, validated_native_program=row_program),
        freeze_row_values(placement_um),
        tuple(stamps),
        tuple((str(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in paths),
        _pdk_state(tech, refresh=True))


def _adjacent_pin_facts(pins_by_net):
    from clay.layout.place.row_replay_snapshot import freeze_row_values
    return freeze_row_values(pins_by_net)


def _extract_native_row_pin_report(
    bridged_intent, placement_um, *, tech, geometry, intent, source_cell,
    placement_obj, route_engine, include_power, extra_power_nets, _reuse_capture=None,
):
    """Extract ordinary native pins and source-owned joint-row components once.

    A joint row has a single physical stamp rather than one stamp per source
    member.  Replay the retained row's complete component inventory before the
    ordinary native extractor asks for individual artwork.  Other native
    devices still use exactly the existing terminal extraction path.
    """
    extract_pins = _rc_attr("extract_pins", _base_extract_pins)
    declared_fold_devices: list[str] = []
    try:
        from clay.layout.annotations import Fold
        for _tgt, _anns in (getattr(source_cell, "__dict__", {}).get(
            "_layout_annotations", {}) or {}).items():
            if any(isinstance(_a, Fold) for _a in (_anns or ())):
                declared_fold_devices.append(str(_tgt))
    except Exception:
        declared_fold_devices = []
    extra_pin_kw: Dict[str, Any] = (
        {"fold_devices": tuple(sorted(declared_fold_devices))}
        if declared_fold_devices and extract_pins is _base_extract_pins
        else {}
    )
    materialization = getattr(placement_obj, "row_materialization", None)
    realizations = getattr(materialization, "realizations", ()) or ()
    has_joint_row = any(
        getattr(row.geometry, "joint_row_binding", None) is not None
        for row in realizations
    )
    from clay.layout.row_terminal_probes import _retained_typed_geometry
    if not _retained_typed_geometry(materialization):
        return extract_pins(
            bridged_intent, placement_um, tech=tech, return_report=True,
            include_power=include_power, geometry=geometry, per_device_fold=True,
            extra_power_nets=extra_power_nets,
            **extra_pin_kw,
        )
    # The retained-row replay checks the entire physical component denominator,
    # including supply components.  Do not silently expose those components as
    # signal-only routing when the caller has disabled power routing.
    if not include_power:
        raise ValueError("native source-owned row requires power routing")
    if has_joint_row and not getattr(route_engine, "consumes_single_component_joint_terminals", False):
        raise ValueError("JOINT_ROUTING_ACCESS_UNSUPPORTED")

    operands = dict(tech=tech, geometry=geometry, placement_um=placement_um,
        row_program=getattr(placement_obj, "row_program", None), row_materialization=materialization,
        row_final_certificate=getattr(placement_obj, "row_final_certificate", None),
        intent=intent, source_cell=source_cell)
    before = None
    if _reuse_capture is not None and has_joint_row:
        try:
            before = _adjacent_native_row_snapshot(**operands)
        except (TypeError, ValueError, AttributeError, KeyError, OSError):
            pass  # Unsupported operands keep the original independent replay.
    from clay.layout.route.joint_component_pins import build_current_joint_component_pins
    row_pins, members = build_current_joint_component_pins(
        allow_fixed_multilayer=bool(getattr(route_engine, 'consumes_fixed_multilayer_joint_terminals', False)),
        multilayer_engine=route_engine,
        **operands,
    )
    native_names = {str(device.name) for device in bridged_intent.devices}
    if not members <= native_names:
        raise ValueError("joint row members differ from native source devices")
    ordinary = _BridgedIntent(
        [device for device in bridged_intent.devices if str(device.name) not in members],
        bridged_intent.passives, bridged_intent.bjts,
        well_isolated_devices=bridged_intent.well_isolated_devices,
        substrate_net=bridged_intent.substrate_net,
        deep_nwell_net=bridged_intent.deep_nwell_net,
    )
    report = extract_pins(
        ordinary, placement_um, tech=tech, return_report=True,
        include_power=include_power, geometry=geometry, per_device_fold=True,
        extra_power_nets=extra_power_nets,
        **extra_pin_kw,
    )
    for pin in row_pins:
        report.pins_by_net.setdefault(pin.net, []).append(pin)
    if before is not None:
        if _adjacent_native_row_snapshot(**operands) != before:
            raise ValueError('native pin replay operands changed during extraction')
        from clay.layout.place.row_replay_snapshot import freeze_row_values
        _reuse_capture.append(_AdjacentNativeRowPins(
            (tech, geometry, placement_um, getattr(placement_obj, 'row_program', None),
             materialization, getattr(placement_obj, 'row_final_certificate', None), intent, source_cell),
            tuple(row_pins), frozenset(members), before,
            freeze_row_values(tuple(row_pins)),
            _adjacent_pin_facts(report.pins_by_net)))
    return report


def _relocate_array_member_pins(
    pins_by_net: Dict[str, List[Any]],
    bridged_intent: Any,
    cell_def: Any,
    intent: Any,
    tech: Any,
    placement_um: Dict[str, _XY],
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
    placement_obj: Any = None,
    geometry: Optional[Dict[str, Any]] = None,
    allow_single_component_joint: bool = False,
    allow_fixed_multilayer_joint: bool = False,
    multilayer_engine: Any = None,
    _adjacent_native_row_pins: Optional[_AdjacentNativeRowPins] = None,
) -> None:
    """S1 ARRAY PIN-LANDING: land each shared-diffusion @array member's router pins
    on its REAL interleaved STRAPS, replacing the stale sub-centroid pins.

    ``extract_pins`` is MOS-geometry-blind and device-keyed: it probes each array
    member as a STANDALONE folded MOS at the member's own placement centroid (the
    sub-centroid the bridge keeps so the whole array can stamp once). But the member
    has NO per-device artwork -- the array is ONE merged-OD strip whose drain columns
    are INTERLEAVED, so the standalone probe lands each member's drain net OFF the
    array's own drain strap. The measured result: the per-member drain nets collapse
    (routing 7/15, our MOS un-extractable under LVS).

    This replaces, IN PLACE, each member's pins with one Pin per terminal at
    ``SharedDiffusionArray.member_pins(name)`` carried from the array's build frame
    into cell um by the SAME lower-integer-origin + certified-delta transform that
    :func:`_array_cell_geometry` uses for the polygons. So a member's
    drain pin lands on its own drain-strap column, its source pin on a shared-source
    column, and its gate pin on the array's one gate-contact M1 pad.  When the one
    shared N-well tap is isolated because body != source, ``member_pins`` also
    publishes ``b`` on the declared body net.  All pins land on the array's OWN
    emitted M1, the single merged-OD stamp untouched.

    GENERAL, no per-circuit branch: keys only on the declared arrays -- BOTH
    shared-diffusion (:func:`resolve_shared_diffusion_arrays`) AND common-centroid
    (:func:`resolve_common_centroids`; ``CommonCentroidArray`` subclasses
    ``SharedDiffusionArray`` so it exposes the SAME ``member_pins`` / ``extent``
    interleaved-strap interface) + the members' role nets. A ``@common_centroid``
    member's terminals otherwise stay at the MOS-geometry-blind sub-centroid (6-12um
    off the array's own interleaved strap), so the strap emits owner_net=None and
    HARD keeps out the very net it carries -> the member pin self-seals (measured:
    input-pair / bias-mirror nets fail with A* ``expansions=1``). Landing them on
    ``member_pins`` closes that. A cell that declares no (materializable) array
    resolves to nothing and this is a NO-OP -> byte-identical pin path for every
    non-array device."""
    _array_extent_translation = _rc_attr("_array_extent_translation", globals()["_array_extent_translation"])
    _stored_row_consumptions = _rc_attr("_stored_row_consumptions", globals()["_stored_row_consumptions"])
    if intent is None:
        return
    valid, _member_group = _stored_row_consumptions(
        placement_obj, intent, tuple(getattr(bridged_intent, "devices", ()) or ())
    )
    if not valid:
        return

    from clay.layout.route.engine import Pin as _ArrPin

    by_name = {
        str(getattr(d, "name", "")): d
        for d in (getattr(bridged_intent, "devices", []) or [])
    }
    m1_layer = tuple(tech.metal_key(1))
    # member_pins terminal role -> the device's netlist ROLE key (_BridgedDevice.nets).
    role_to_term = {"d": "drain", "s": "source", "g": "gate", "b": "body"}

    typed_groups = {
        ac.group for ac in valid
        if (getattr(ac.array, "physical_terminal_inventory", None) is not None
            or getattr(ac.array, "fixed_occurrence_sources", None) is not None
            or bool(getattr(ac.array, "fixed_pcell", None))
            or (allow_single_component_joint and getattr(ac.array, "joint_row_binding", None) is not None))
    }
    component_pins = ()
    if typed_groups:
        from clay.layout.row_terminal_probes import build_row_terminal_pins

        if geometry is None:
            raise ValueError("row terminal component relocation requires actual route geometry")
        pin_operands = dict(
            tech=tech, geometry=geometry, placement_um=placement_um,
            row_program=getattr(placement_obj, "row_program", None),
            row_materialization=getattr(placement_obj, "row_materialization", None),
            row_final_certificate=getattr(placement_obj, "row_final_certificate", None),
            intent=intent, source_cell=cell_def,
        )
        if allow_single_component_joint:
            from clay.layout.route.joint_component_pins import build_current_joint_component_pins
            candidate = _adjacent_native_row_pins
            values = tuple(pin_operands[name] for name in ('tech','geometry','placement_um',
                'row_program','row_materialization','row_final_certificate','intent','source_cell'))
            if (type(candidate) is _AdjacentNativeRowPins and not candidate.used
                    and all(a is b for a,b in zip(candidate.operands, values))):
                from collections import Counter
                from clay.layout.route.terminal_identity import pin_fact, pin_source_aliases
                from clay.layout.place.row_replay_snapshot import freeze_row_values
                if (_adjacent_native_row_snapshot(**pin_operands) != candidate.snapshot
                        or _adjacent_pin_facts(pins_by_net) != candidate.report_facts
                        or freeze_row_values(candidate.pins) != candidate.facts):
                    raise ValueError('native pin replay operands changed before relocation')
                actual = [pin_fact(pin) for net,pins in pins_by_net.items() for pin in pins
                          if any(member in candidate.members for member,_role in pin_source_aliases(pin))
                          and pin.net == net]
                if Counter(actual) == Counter(pin_fact(pin) for pin in candidate.pins):
                    component_pins = candidate.pins
                    candidate.used = True
                else:
                    component_pins, _members = build_current_joint_component_pins(allow_fixed_multilayer=allow_fixed_multilayer_joint, multilayer_engine=multilayer_engine, **pin_operands)
            else:
                component_pins, _members = build_current_joint_component_pins(allow_fixed_multilayer=allow_fixed_multilayer_joint, multilayer_engine=multilayer_engine, **pin_operands)
        else:
            component_pins = build_row_terminal_pins(**pin_operands)

    for ac in valid:
        if ac.group in typed_groups:
            consumed = set(ac.members) | set(ac.flank_dummies)
            for net in list(pins_by_net):
                kept = [pin for pin in pins_by_net[net] if pin.device not in consumed]
                if kept:
                    pins_by_net[net] = kept
                else:
                    del pins_by_net[net]
            for pin in component_pins:
                if pin.device in consumed:
                    pins_by_net.setdefault(pin.net, []).append(pin)
            continue
        tile = placement_um.get(ac.group)
        if tile is None:
            continue  # array not seated (no member placed) -> nothing to relocate
        # A consumed flank dummy has no standalone device artwork: its fingers are
        # already part of the signed row-macro stamp.  ``extract_pins`` is unaware
        # of that ownership and otherwise leaves stale pins at the dummy's former
        # free-device centroid.  Those points are neither terminals of the macro nor
        # backed by emitted metal, so remove them before publishing the final world.
        for dummy in ac.flank_dummies:
            for net in list(pins_by_net):
                kept = [
                    pin for pin in pins_by_net[net]
                    if str(getattr(pin, "device", "")) != dummy
                ]
                if kept:
                    pins_by_net[net] = kept
                else:
                    del pins_by_net[net]
        certified_translation = getattr(ac, "certified_translation_nm", None)
        if certified_translation is None:
            from clay.layout.generic_placement import RowFinalPlacementResidualError

            raise RowFinalPlacementResidualError(
                ("ROW_ARRAY_TRANSLATION_CERTIFICATE_MISSING",)
            )
        translation = _array_extent_translation(
            ac.array, translation_nm=certified_translation
        )
        stamp_position = translation.stamp_position_nm
        if (
            abs(float(tile[0]) - stamp_position[0] / _NM_PER_UM)
            > 0.5 / _NM_PER_UM
            or abs(float(tile[1]) - stamp_position[1] / _NM_PER_UM)
            > 0.5 / _NM_PER_UM
        ):
            from clay.layout.generic_placement import RowFinalPlacementResidualError

            raise RowFinalPlacementResidualError(
                ("ROW_CERTIFIED_ARRAY_PIN_TRANSLATION_MISMATCH",)
            )
        for m in ac.members:
            dev = by_name.get(m)
            if dev is None or placement_um.get(m) is None:
                continue
            try:
                mp = ac.array.member_pins(m)
            except Exception:  # noqa: BLE001 - a member with no drain column is skipped
                continue
            nets = getattr(dev, "nets", {}) or {}
            # 1) drop THIS member's stale (sub-centroid) pins from every net.
            for net in list(pins_by_net):
                kept = [
                    p for p in pins_by_net[net]
                    if str(getattr(p, "device", "")) != m
                ]
                if kept:
                    pins_by_net[net] = kept
                else:
                    del pins_by_net[net]
            # 2) emit one strap pin per terminal, on the member's role net, at the
            #    array-frame member pin carried into cell um (tile centre + offset).
            for role, (mx, my) in mp.items():
                net = str(nets.get(role_to_term.get(role, ""), ""))
                if not net:
                    continue
                x_nm, y_nm = translation.local_to_cell_nm((mx, my))
                xy = (x_nm / _NM_PER_UM, y_nm / _NM_PER_UM)
                pins_by_net.setdefault(net, []).append(
                    _ArrPin(
                        net=net,
                        device=m,
                        terminal=role,
                        xy_um=xy,
                        layer=m1_layer,
                        xy_nm=(x_nm, y_nm),
                    )
                )
