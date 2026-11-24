"""Envelope-min device placement as a re-positioning of a built ``Placement``.

This wires the validated geometric-synthesis core
(:func:`clay.layout.synth.placement.place_envelope_min`) as an alternative
PLACEMENT for a cell that already has per-device geometry built by
:func:`clay.layout.generic_placement.place_generic`.

It does NOT build any device geometry and it does NOT re-implement a placer:

  1. take the device CELL geometry place_generic already produced (each
     :class:`PlacedDevice` carries its drawn polys + terminal pads + bbox);
  2. read each device's CURRENT footprint (w,h) off its placed bbox ->
     one :class:`~clay.layout.synth.placement.Group` per device (key = label);
  3. build the device-level nets from the parsed ``CellDef`` pin map and the
     declared symmetry pairs from the ``LayoutIntent`` (topology only -- no
     reference coordinate is read);
  4. call ``place_envelope_min`` (CP-SAT envelope-min + wirelength + L-inf
     max-span + hard symmetry + ``aspect_target``);
  5. RE-POSITION every device's bbox / terminals / polys by the rigid (dx, dy)
     that carries its place_generic centroid onto the envelope-min centre.

The output is the SAME canonical :class:`Placement` type ``route_cell`` already
consumes -- a different device arrangement, not a second materialization path.
No per-circuit branch: the only inputs are the built geometry, the parsed pin
map, the declared symmetry pairs, and an ``aspect_target`` scalar.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.backend import _translate
from clay.layout.generic_placement import (
    Placement,
    PlacedDevice,
    RigidNonmosMacroProjectionError,
    RowFinalPlacementResidualError,
    attach_physical_macro_projection,
    _cell_geom,
    _polys_bbox,
    placement_auxiliary_polys,
    project_stored_row_master,
    row_macro_member_offsets_nm,
    verify_rigid_nonmos_macro_projection,
    project_rigid_nonmos_matching_axes,
)
from clay.layout.synth.placement import (
    ArrayTile,
    Group,
    PlacementResult,
    place_envelope_min,
)


def array_footprint_tiles(cell_def, intent, tech_id: str, device_records=None) -> List[ArrayTile]:
    """DECLARED shared-diffusion ``@array`` tiles for the compact placer.

    Each VALID ``@array(style=shared_diffusion)`` becomes ONE :class:`ArrayTile`
    (footprint = the built ``SharedDiffusionArray`` extent, members co-placed by the
    array's own per-member offsets) so the placer seats the whole cluster as ONE
    atomic footprint instead of N free devices. Reuses the SINGLE array-resolution
    source (:func:`clay.layout.route.route_cell.resolve_shared_diffusion_arrays`) the
    emit path also consumes -- no second derivation. A cell that declares no
    (materializable) shared-diffusion array returns ``[]`` -> byte-identical pack."""
    try:
        from clay.layout.route.route_cell import (
            array_member_offsets_um,
            resolve_common_centroids,
            resolve_shared_diffusion_arrays,
        )
        from clay.layout.tech import LayoutTech
    except Exception:  # noqa: BLE001 - array consumption is an optional density path
        return []
    if not (getattr(intent, "arrays", ()) or ()) and not (
        getattr(intent, "common_centroids", ()) or ()
    ):
        return []
    try:
        tech = LayoutTech.resolve(str(tech_id))
    except Exception:  # noqa: BLE001
        return []

    def _tile_from(ac) -> ArrayTile:
        x0, y0, x1, y1 = ac.array.extent()
        offs = array_member_offsets_um(ac.array)
        return ArrayTile(
            key=str(ac.group), kind=str(ac.kind),
            members=tuple(str(m) for m in ac.members),
            w_um=(x1 - x0) / 1000.0, h_um=(y1 - y0) / 1000.0,
            rel={str(m): offs.get(m, (0.0, 0.0)) for m in ac.members},
        )

    tiles: List[ArrayTile] = []
    # shared-diffusion @array tiles (the SINGLE resolution the emit path also consumes)
    sd_cons, _sd_mg = resolve_shared_diffusion_arrays(
        cell_def, intent, tech, device_records=device_records
    )
    # common-centroid @common_centroid tiles -- seated as ONE atomic footprint too, so
    # the placer keeps the interleaved pair compact (the input diff pair) instead of
    # scattering its members as free devices (which left the 2-row fold un-materialised).
    cc_cons, _cc_mg = resolve_common_centroids(
        cell_def, intent, tech, device_records=device_records
    )
    for ac in list(sd_cons) + list(cc_cons):
        if ac.array is None:
            continue  # fallback -> members stay on the per-device path
        tiles.append(_tile_from(ac))
    return tiles


def rigid_nonmos_footprint_tiles(base: Placement) -> List[ArrayTile]:
    """Project stored passive/BJT macro plans into generic atomic CP tiles."""

    result = []
    plans = getattr(getattr(base, "rigid_nonmos_macro_plans", None), "plans", ())
    for plan in plans:
        x0, y0, x1, y1 = plan.local_bbox_nm
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        result.append(ArrayTile(
            key=f"nonrow:{plan.macro_id}",
            kind="passive",
            members=plan.members,
            w_um=(x1 - x0) / 1000.0,
            h_um=(y1 - y0) / 1000.0,
            rel={
                member.member: (
                    ((member.local_bbox_nm[0] + member.local_bbox_nm[2]) / 2.0 - cx)
                    / 1000.0,
                    ((member.local_bbox_nm[1] + member.local_bbox_nm[3]) / 2.0 - cy)
                    / 1000.0,
                )
                for member in plan.member_plans
            },
        ))
    return result


def certified_matching_footprint_tiles(
    base: Placement,
    *,
    existing_tiles: Sequence[ArrayTile],
) -> List[ArrayTile]:
    """Project standalone materializer-certified resistor pairs as atomic tiles.

    Stored row and rigid-non-MOS plans remain the stronger owners and are excluded
    through ``existing_tiles``.  A remaining pair is accepted only when both
    members are resistor views with equal footprints and their current bboxes
    replay the certified local axis and perpendicular alignment exactly.  The tile
    preserves those current relative offsets; it does not repack from a final
    centroid or a place-group token.
    """

    occupied = {
        str(member) for tile in existing_tiles for member in tuple(tile.members)
    }
    devices = dict(getattr(base, "devices", {}) or {})
    raw_axes = dict(getattr(base, "matching_pair_axes", {}) or {})
    candidates: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for raw_pair, raw_axis in raw_axes.items():
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or raw_pair[0] == raw_pair[1]
            or type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or raw_axis[0] not in {"x", "y"}
        ):
            continue
        pair = tuple(sorted((str(raw_pair[0]), str(raw_pair[1]))))
        try:
            coordinate = float(raw_axis[1])
        except (TypeError, ValueError):
            continue
        candidates[pair] = (str(raw_axis[0]), coordinate)
    degree: Dict[str, int] = {}
    for pair in candidates:
        for member in pair:
            degree[member] = degree.get(member, 0) + 1

    tiles: List[ArrayTile] = []
    for pair, (axis, coordinate) in sorted(candidates.items()):
        if (
            any(member in occupied or degree.get(member) != 1 for member in pair)
            or any(member not in devices for member in pair)
            or any(str(devices[member].kind).strip().lower() != "res" for member in pair)
        ):
            continue
        boxes = [tuple(devices[member].bbox) for member in pair]
        sizes = [
            (box[2] - box[0], box[3] - box[1]) for box in boxes
        ]
        axis_index = 0 if axis == "x" else 1
        perpendicular_index = 1 - axis_index
        centres = [
            (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
            )
            for box in boxes
        ]
        if (
            sizes[0] != sizes[1]
            or centres[0][perpendicular_index] != centres[1][perpendicular_index]
            or (centres[0][axis_index] + centres[1][axis_index]) / 2.0
            != coordinate
        ):
            continue
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        tiles.append(ArrayTile(
            key=f"matching:{pair[0]}:{pair[1]}",
            kind="passive",
            members=pair,
            w_um=(x1 - x0) / 1000.0,
            h_um=(y1 - y0) / 1000.0,
            rel={
                member: (
                    (centre[0] - cx) / 1000.0,
                    (centre[1] - cy) / 1000.0,
                )
                for member, centre in zip(pair, centres)
            },
        ))
        occupied.update(pair)
    return tiles


def stored_row_footprint_tiles(base: Placement, intent) -> List[ArrayTile]:
    """Project the exact-once RowProgram materialization into CP atomic tiles.

    This is deliberately a projection of ``base`` metadata.  It never imports the
    router's legacy array resolvers or invokes a rigid geometry builder again.
    """
    program = getattr(base, "row_program", None)
    materialization = getattr(base, "row_materialization", None)
    declared = bool(
        (getattr(intent, "arrays", ()) or ())
        or (getattr(intent, "common_centroids", ()) or ())
    )
    if program is None and materialization is None:
        if declared:
            raise RowFinalPlacementResidualError(("ROW_MASTER_METADATA_MISSING",))
        return []
    if program is None or materialization is None:
        raise RowFinalPlacementResidualError(("ROW_MASTER_METADATA_INCOMPLETE",))
    from clay.layout.place.row_replay_session import require_current_row_materialization

    try:
        require_current_row_materialization(
            materialization, program=program, tech_id=program.tech_id
        )
    except Exception as exc:
        raise RowFinalPlacementResidualError((
            f"ROW_MATERIALIZATION_REPLAY_MISMATCH:{exc}",
        )) from exc
    by_entry = {item.entry_id: item for item in materialization.realizations}
    by_member = {item.device: item for item in program.members}
    witness = getattr(getattr(materialization, 'source_projection', None), 'native_witness', None)
    from clay.layout.native_diffusion_source import NativePCellWitness
    native_groups = ({name for name, _geometry, _folder in witness.rows}
                     if type(witness) is NativePCellWitness else set())
    native_grid = None
    if native_groups:
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        from clay.layout.tech import LayoutTech
        native_grid = pdk_manufacturing_grid_nm(LayoutTech.resolve(program.tech_id))
    result: List[ArrayTile] = []
    for entry in program.entries:
        realization = by_entry.get(entry.entry_id)
        if realization is None:
            raise RowFinalPlacementResidualError(("ROW_REALIZATION_MISSING",))
        offsets = row_macro_member_offsets_nm(entry, realization)
        x0, y0, x1, y1 = realization.local_bbox_nm
        active = by_member[entry.active_devices[0]]
        result.append(ArrayTile(
            key=entry.group,
            kind=active.kind,
            members=entry.consumed_devices,
            w_um=(x1 - x0) / 1000.0,
            h_um=(y1 - y0) / 1000.0,
            rel={name: (point[0] / 1000.0, point[1] / 1000.0)
                 for name, point in offsets.items()},
            native_bbox_nm=(realization.local_bbox_nm if entry.group in native_groups else None),
            native_grid_nm=(native_grid if entry.group in native_groups else None),
        ))
    return result


def _device_footprint_um(pd: PlacedDevice) -> Tuple[float, float]:
    x0, y0, x1, y1 = pd.bbox
    return (x1 - x0) / 1000.0, (y1 - y0) / 1000.0  # nm -> um


def _active_x_insets_um(
    placed: Dict[str, PlacedDevice], tech_id: str
) -> Dict[str, float]:
    """``{device -> per-side bbox->COMP x-inset (um)}`` for every built MOS.

    A declared ``@abut`` pair packs COMP-flush (the two active/diffusion rectangles meet
    into ONE continuous OD island). To collapse the inter-member pitch to that shared-OD
    spacing the placer needs each device's NON-diffusion x-margin -- the distance from
    the device footprint (bbox) edge to the active (COMP) edge, which the gap must remove
    on BOTH facing sides. Derive it from the BUILT device geometry: the active layer key
    comes from the SAME device_cell resolution the materializer draws (no second source),
    and the WIDEST active poly is the channel COMP (robust to a small tap-COMP square).
    The per-side inset is half the bbox->COMP width difference (matched devices are
    x-symmetric, so both facing sides share it). Returns ``{}`` when the tech / active
    layer cannot be resolved (honest -> the placer falls back to a gap-collapsed
    edge-to-edge pitch, never a fabricated inset)."""
    try:
        from clay.layout.tech import LayoutTech
        from clay.layout.device_cell import _resolve_layer_keys
    except Exception:
        return {}
    try:
        tech = LayoutTech.resolve(tech_id)
    except Exception:
        return {}
    out: Dict[str, float] = {}
    active_key_by_kind: Dict[str, object] = {}
    for lbl, pd in placed.items():
        kind = str(getattr(pd, "kind", "") or "")
        if kind not in ("nmos", "pmos"):
            continue
        if kind not in active_key_by_kind:
            try:
                active_key_by_kind[kind] = _resolve_layer_keys(tech, kind).active
            except Exception:
                active_key_by_kind[kind] = None
        akey = active_key_by_kind[kind]
        if akey is None:
            continue
        comp_w = 0
        for lp in (getattr(pd, "polys", None) or []):
            if (getattr(lp, "layer", None), getattr(lp, "datatype", None)) != akey:
                continue
            b = _polys_bbox([lp])
            if b is not None:
                comp_w = max(comp_w, b[2] - b[0])  # widest active poly = channel COMP
        if comp_w <= 0:
            continue
        x0, _y0, x1, _y1 = pd.bbox
        inset_nm = (x1 - x0 - comp_w) / 2.0
        if inset_nm > 0:
            out[str(lbl)] = inset_nm / 1000.0  # nm -> um, per side
    return out


def _nets_from_cell_def(cell_def, placed_labels: Sequence[str]) -> List[List[str]]:
    """``[[group_key, ...], ...]`` -- one entry per net, the device labels it joins.

    Group keys are device labels. Only placed labels participate (a net pin on an
    un-placed device is dropped from that net). Power/ground nets stay in (the
    envelope-min L-inf term excludes high-fanout nets itself)."""
    labset = set(str(x) for x in placed_labels)
    by_net: Dict[str, List[str]] = {}
    for dev in getattr(cell_def, "devices", []) or []:
        lbl = str(getattr(dev, "instance_name", ""))
        if lbl not in labset:
            continue
        for _pin, net in (getattr(dev, "pins", {}) or {}).items():
            net = str(net)
            by_net.setdefault(net, [])
            if lbl not in by_net[net]:
                by_net[net].append(lbl)
    # nets that touch >= 2 distinct groups carry placement signal
    return [grp for grp in by_net.values() if len(grp) >= 2]


def _symmetry_pairs(intent, placed_labels: Sequence[str]) -> List[Tuple[str, str]]:
    """Declared mirror / diff pairs restricted to placed labels (topology only)."""
    labset = set(str(x) for x in placed_labels)
    out: List[Tuple[str, str]] = []
    seen = set()
    for attr in ("mirror_pairs", "diff_pairs"):
        for p in (getattr(intent, attr, ()) or ()):
            try:
                a, b = str(p[0]), str(p[1])
            except Exception:  # noqa: BLE001
                continue
            if a in labset and b in labset and a != b and (a, b) not in seen and (b, a) not in seen:
                out.append((a, b))
                seen.add((a, b))
    return out


def _placement_symmetry_pairs(
    base: Placement,
    intent,
    placed_labels: Sequence[str],
) -> List[Tuple[str, str]]:
    """Return the already-lowered device pairs that envelope packing must keep.

    ``LayoutIntent.diff_pairs`` contains net identities, not device labels.  The
    canonical base placement has already combined annotations with net-side
    polarity into ordered device pairs; discarding that result here let the
    envelope pass independently reshuffle matched islands.  Legacy device-named
    mirror declarations remain additive after the base pairs.
    """
    allowed = {str(label) for label in placed_labels}
    result: List[Tuple[str, str]] = []
    seen = set()

    constraints = getattr(base, "constraints", None)
    for pair in tuple(getattr(constraints, "symmetry_pairs", ()) or ()):
        first, second = str(pair[0]), str(pair[1])
        key = frozenset((first, second))
        if (
            first == second
            or first not in allowed
            or second not in allowed
            or key in seen
        ):
            continue
        result.append((first, second))
        seen.add(key)

    for first, second in _symmetry_pairs(intent, placed_labels):
        key = frozenset((first, second))
        if key in seen:
            continue
        result.append((first, second))
        seen.add(key)
    return result


def _symmetry_order_constraints(
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Tuple[str, str, str], ...]:
    """Materialize pair orientation; symmetry alone leaves left/right ambiguous."""
    return tuple(
        (str(first), str(second), "left_of")
        for first, second in pairs
        if str(first) != str(second)
    )


def _compact_order_constraints(
    base: Placement,
    intent,
    placed_labels: Sequence[str],
    *,
    cell_def=None,
    stored_row_tiles=(),
) -> Tuple[Tuple[str, str, str], ...]:
    """Compose explicit source order with oriented matched-pair polarity.

    A source ``@order`` owns its pair and is never overridden.  For every other
    already-declared symmetry pair, the base placement's ordered identity becomes
    a hard ``left_of`` relation in the CP compactor so group folding cannot reverse
    the net-level differential polarity.
    """
    allowed = {str(label) for label in placed_labels}
    declared = []
    owned_pairs = set()
    for raw in tuple(getattr(intent, "order_constraints", ()) or ()):
        if len(raw) != 3:
            continue
        first, second, relation = map(str, raw)
        if first not in allowed or second not in allowed:
            continue
        declared.append((first, second, relation))
        owned_pairs.add(frozenset((first, second)))
    # A native row is one rigid physical tile, even when an order and a matching
    # name different member devices.  Matching declarations have no orientation:
    # their canonical pair order is merely name-normalization.  At the actual
    # row-owner boundary, an explicit horizontal order owns that direction.
    # Preserve explicit mirror and differential-net orientation; never infer an
    # electrical pair from a geometric matching declaration.
    native_owners = {}
    for tile in stored_row_tiles:
        if tile.native_bbox_nm is None or tile.native_grid_nm is None:
            continue
        for member in tile.members:
            native_owners[str(member)] = str(tile.key)
    explicit_row_orders = {
        (native_owners[first], native_owners[second])
        for first, second, relation in declared
        if relation == 'left_of' and first in native_owners and second in native_owners
        and native_owners[first] != native_owners[second]
    }
    undirected_matching = set()
    directed_pairs = set()
    diff_members = set()
    directional_owners = set()
    directional_owner_pairs = set()
    if cell_def is not None and explicit_row_orders:
        from clay.layout.annotations import Matching
        from clay.layout.net_match_scope import placement_diff_pairs
        for target, annotations in (cell_def.__dict__.get('_layout_annotations', {}) or {}).items():
            for annotation in annotations:
                if isinstance(annotation, Matching):
                    undirected_matching.add(frozenset((str(target), str(annotation.other))))
        constraints = getattr(base, 'constraints', None)
        for group in tuple(getattr(constraints, 'declared_mirror_groups', ()) or ()):
            directed_pairs.update(frozenset(map(str, pair)) for pair in group.pairs)
        # The legacy compact view also accepts device-named mirror pairs.  A
        # mirror can instead name asymmetric electrical nodes; preserve the
        # canonical direction of any matched member touching those nodes too.
        mirror_pairs = tuple(getattr(intent, 'mirror_pairs', ()) or ())
        directed_pairs.update(frozenset(map(str, pair)) for pair in mirror_pairs)
        diff_nets = {str(net) for pair in (*placement_diff_pairs(intent), *mirror_pairs) for net in pair}
        diff_members = {str(device.instance_name)
                        for device in getattr(cell_def, 'devices', ())
                        if diff_nets.intersection(str(net) for net in (device.pins or {}).values())}
        diff_members.update(str(member) for member in
                            (getattr(constraints, 'diff_net_sides', {}) or {}))
        # Every member moves with its complete native row.  A directional
        # constraint on any member (or another pair across these owners) must
        # therefore protect the owners, not just the selected matching endpoints.
        directional_owners = {native_owners[member] for member in diff_members
                              if member in native_owners}
        directional_owners.update(native_owners[member] for pair in directed_pairs
                                  for member in pair if member in native_owners)
        directional_owner_pairs = {
            frozenset(native_owners[member] for member in pair)
            for pair in directed_pairs if len(pair) == 2 and all(member in native_owners for member in pair)
            and len({native_owners[member] for member in pair}) == 2
        }
    for first, second, relation in _symmetry_order_constraints(
        _placement_symmetry_pairs(base, intent, placed_labels)
    ):
        if frozenset((first, second)) in owned_pairs:
            continue
        pair = frozenset((first, second))
        if (pair in undirected_matching and pair not in directed_pairs
                and not pair.intersection(diff_members)
                and first in native_owners and second in native_owners
                and not {native_owners[first], native_owners[second]}.intersection(directional_owners)
                and frozenset((native_owners[first], native_owners[second])) not in directional_owner_pairs
                and (native_owners[second], native_owners[first]) in explicit_row_orders):
            continue
        declared.append((first, second, relation))
        owned_pairs.add(frozenset((first, second)))
    return tuple(declared)


def _inherit_placement_pair_identity(ext, base: Placement, intent, labels) -> None:
    """Keep canonical source pair identities in the temporary compact view.

    The mechanical netlist extractor groups shared-gate devices.  Such an
    inferred pair can cross two already-declared matched pairs (for example a
    cross-coupled latch and its output inverters).  Re-grouping those devices
    would turn consistent device orders into a cycle between compact groups,
    and would also introduce a different hard group symmetry.

    Source pairs have already been lowered by the canonical base placement.
    They own the physical pairing of their members here, just as they own its
    directed order in ``_compact_order_constraints``.  Preserve the original
    inference for uncovered devices; do not relabel source symmetry as an
    electrical differential pair or an equal-size match.  The source intent,
    base constraints, and all explicit orders remain untouched.
    """
    pairs = _placement_symmetry_pairs(base, intent, labels)
    if not pairs:
        return
    owned = {member for pair in pairs for member in pair}
    pair_keys = {frozenset(pair) for pair in pairs}

    def compatible(pair):
        members = frozenset((str(pair[0]), str(pair[1])))
        return not members.intersection(owned) or members in pair_keys

    ext.mirror_pairs = list(pairs) + [
        pair for pair in ext.mirror_pairs
        if not {str(pair[0]), str(pair[1])}.intersection(owned)
    ]
    ext.diff_pairs = [pair for pair in ext.diff_pairs if compatible(pair)]
    ext.matched_pairs = [pair for pair in ext.matched_pairs if compatible(pair)]


def _reposition_device(
    pd: PlacedDevice,
    cx_nm: int,
    cy_nm: int,
    *,
    grid_nm: int = 1,
) -> PlacedDevice:
    """Rigid-translate a placed device so its bbox centre lands on (cx, cy) nm.

    When ``grid_nm > 1`` and a device extent is odd in nanometres (half-nm
    centroid), preserve the manufacturing-grid parity of its bbox edges instead
    of shifting odd-extent edges by half a nanometre off-grid.
    """
    x0, y0, x1, y1 = pd.bbox
    cur_cx = (x0 + x1) // 2
    cur_cy = (y0 + y1) // 2
    dx = int(cx_nm - cur_cx)
    dy = int(cy_nm - cur_cy)
    if grid_nm > 1:
        if ((x1 - x0) % 2) == 1:
            rem_x = (x0 + dx) % grid_nm
            if rem_x:
                dx += (grid_nm - rem_x) if (2 * rem_x >= grid_nm) else -rem_x
        if ((y1 - y0) % 2) == 1:
            rem_y = (y0 + dy) % grid_nm
            if rem_y:
                dy += (grid_nm - rem_y) if (2 * rem_y >= grid_nm) else -rem_y
    new_bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    new_terms = {k: (int(px + dx), int(py + dy)) for k, (px, py) in (pd.terminals or {}).items()}
    new_accesses = {
        role: tuple((int(px + dx), int(py + dy)) for px, py in accesses)
        for role, accesses in (pd.terminal_accesses or {}).items()
    }
    new_polys = _translate(list(pd.polys or []), dx, dy)
    return replace(
        pd,
        bbox=new_bbox,
        terminals=new_terms,
        terminal_accesses=new_accesses,
        polys=new_polys,
    )


def _reposition_native_exact(pd: PlacedDevice, center_um, grid_nm: int) -> PlacedDevice:
    """Preserve an exact solver center and the native rigid translation phase."""
    if type(grid_nm) is not int or grid_nm <= 0:
        raise ValueError('native compaction requires the PDK manufacturing grid')
    offsets=[]
    for center,low,high in zip(center_um,pd.bbox[:2],pd.bbox[2:]):
        doubled=float(center)*2000.0
        target2=round(doubled)
        if not math.isfinite(doubled) or abs(doubled-target2)>1e-6:
            raise ValueError('native compaction center is not an exact half-nanometre')
        left2=target2-(high-low)
        if left2 % 2:
            raise ValueError('native compaction center is incompatible with the rigid extent')
        delta=left2//2-low
        if delta % grid_nm:
            raise ValueError('native compaction violates the rigid manufacturing-grid phase')
        offsets.append(delta)
    dx,dy=offsets
    return replace(pd,
        bbox=(pd.bbox[0]+dx,pd.bbox[1]+dy,pd.bbox[2]+dx,pd.bbox[3]+dy),
        terminals={key:(int(x+dx),int(y+dy)) for key,(x,y) in (pd.terminals or {}).items()},
        terminal_accesses={role:tuple((int(x+dx),int(y+dy)) for x,y in values)
                           for role,values in (pd.terminal_accesses or {}).items()},
        polys=_translate(list(pd.polys or ()),dx,dy))


def _project_cp_owned_matching_axes(
    base: Placement,
    devices: Dict[str, PlacedDevice],
    result: PlacementResult,
    *,
    stored_row_tiles: Tuple[ArrayTile, ...] = (),
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Replay CP-owned pair axes at the quantized device positions.

    ``cp_group_placement_pos`` is the structural authority: it emits an owner only
    when the canonical materializer pair is wholly contained by one atomic CP group,
    has equal member footprints, and shares the perpendicular group-local offset.
    This seam merely converts that certificate from the solver's um placement to the
    integer-nm stamp coordinates.  It never creates a pair from final geometry.

    Stored-row and rigid-non-MOS array tiles retain their stronger replay paths.  CP
    owns only its native compact groups (``G_``) and the standalone certified matching
    tiles created above (``A_matching:``).
    """

    projected = project_rigid_nonmos_matching_axes(base, devices)
    base_axes = dict(getattr(base, "matching_pair_axes", {}) or {})
    base_keys = {
        tuple(sorted((str(raw_pair[0]), str(raw_pair[1])))): raw_pair
        for raw_pair in base_axes
        if type(raw_pair) is tuple
        and len(raw_pair) == 2
        and raw_pair[0] != raw_pair[1]
    }
    owners = dict(getattr(result, "matching_pair_owners", {}) or {})
    for pair, owner_proof in sorted(owners.items()):
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(owner_proof) is not tuple
            or len(owner_proof) != 2
        ):
            continue
        axis, owner = str(owner_proof[0]), str(owner_proof[1])
        if axis not in {"x", "y"} or not (
            owner.startswith("G_") or owner.startswith("A_matching:")
        ):
            continue
        base_pair = base_keys.get(tuple(sorted(pair)))
        prior = base_axes.get(base_pair)
        if prior is None or len(prior) != 2 or str(prior[0]) != axis:
            continue
        first = devices.get(pair[0])
        second = devices.get(pair[1])
        if first is None or second is None:
            continue
        boxes = (tuple(first.bbox), tuple(second.bbox))
        sizes = tuple(
            (box[2] - box[0], box[3] - box[1]) for box in boxes
        )
        if sizes[0] != sizes[1]:
            continue
        centres = tuple(
            (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
            )
            for box in boxes
        )
        axis_index = 0 if axis == "x" else 1
        perpendicular_index = 1 - axis_index
        if centres[0][perpendicular_index] != centres[1][perpendicular_index]:
            continue
        projected[base_pair] = (
            axis,
            (centres[0][axis_index] + centres[1][axis_index]) / 2.0,
        )
    if type(stored_row_tiles) is not tuple:
        raise TypeError("stored row tile inventory must be an exact tuple")
    if stored_row_tiles:
        centres_by_owner = getattr(result, "centers", None)
        if type(centres_by_owner) is not dict:
            raise TypeError("stored row CP owner centres must be an exact dict")
        validated_tiles = []
        consumed_members = set()
        for tile in stored_row_tiles:
            members = getattr(tile, "members", None)
            rel = getattr(tile, "rel", None)
            key = getattr(tile, "key", None)
            if (
                type(tile) is not ArrayTile
                or type(key) is not str
                or not key
                or type(members) is not tuple
                or not members
                or any(type(member) is not str or not member for member in members)
                or len(set(members)) != len(members)
                or consumed_members & set(members)
                or type(rel) is not dict
                or set(rel) != set(members)
                or any(
                    type(offset) is not tuple
                    or len(offset) != 2
                    or any(
                        type(value) not in {int, float}
                        or not math.isfinite(float(value))
                        for value in offset
                    )
                    for offset in rel.values()
                )
            ):
                raise TypeError("stored row CP tile authority is malformed")
            owner = f"A_{key}"
            centre = centres_by_owner.get(owner)
            if (
                type(centre) is not tuple
                or len(centre) != 2
                or any(
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    for value in centre
                )
            ):
                raise ValueError(
                    f"stored row CP tile {key!r} lacks one exact solver owner"
                )
            consumed_members.update(members)
            validated_tiles.append((tile, centre))
        for canonical_pair, base_pair in sorted(base_keys.items()):
            matching_tiles = tuple(
                (tile, centre)
                for tile, centre in validated_tiles
                if set(canonical_pair) <= set(tile.members)
            )
            if not matching_tiles:
                continue
            if len(matching_tiles) != 1:
                raise ValueError("stored row matching pair has ambiguous CP owner")
            tile, centre = matching_tiles[0]
            prior = base_axes.get(base_pair)
            if (
                type(prior) is not tuple
                or len(prior) != 2
                or type(prior[0]) is not str
                or prior[0] not in {"x", "y"}
            ):
                raise TypeError("stored row matching axis authority is malformed")
            # This row tile is the unique final owner of the pair.  Remove the
            # pre-row coordinate before replay so a topology that does not
            # realize one simple aligned pair cannot retain a stale authority.
            # Downstream exact intent then observes a missing proof and blocks
            # promotion; valid aligned rows are re-added below.
            projected.pop(base_pair, None)
            boxes = []
            for member in base_pair:
                view = devices.get(member)
                bbox = getattr(view, "bbox", None)
                if (
                    type(bbox) is not tuple
                    or len(bbox) != 4
                    or any(type(value) is not int for value in bbox)
                ):
                    raise TypeError(
                        f"stored row CP member {member!r} lacks exact logical bbox"
                    )
                expected = (
                    int(round((float(centre[0]) + float(tile.rel[member][0])) * 1000.0)),
                    int(round((float(centre[1]) + float(tile.rel[member][1])) * 1000.0)),
                )
                actual = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                if actual != expected:
                    raise ValueError(
                        f"stored row CP member {member!r} does not replay owner offset"
                    )
                boxes.append(bbox)
            axis_index = 0 if prior[0] == "x" else 1
            perpendicular_index = 1 - axis_index
            actual_centres = tuple(
                (
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                )
                for bbox in boxes
            )
            if (
                actual_centres[0][perpendicular_index]
                != actual_centres[1][perpendicular_index]
            ):
                continue
            projected[base_pair] = (
                prior[0],
                (
                    actual_centres[0][axis_index]
                    + actual_centres[1][axis_index]
                )
                / 2.0,
            )
    return projected


def _requantize_rigid_nonmos_members(
    base: Placement,
    placed: Dict[str, PlacedDevice],
) -> Dict[str, PlacedDevice]:
    """Apply one integer-nm translation to every member of each rigid plan.

    CP/envelope coordinates use floating-point micrometres.  Independently
    rounding two member centres can change their relative offset by one nm when
    their footprints have different parity.  Anchor the macro on its first
    member after solver projection, then derive every member centre from the
    canonical local bbox with one shared integer translation.
    """

    result = dict(placed)
    plans = getattr(getattr(base, "rigid_nonmos_macro_plans", None), "plans", ())
    for plan in plans:
        anchor = plan.member_plans[0]
        anchor_view = result.get(anchor.member)
        if anchor_view is None:
            raise RigidNonmosMacroProjectionError(
                f"RIGID_NONMOS_MEMBER_MISSING:{anchor.member}"
            )
        dx = int(anchor_view.bbox[0]) - int(anchor.local_bbox_nm[0])
        dy = int(anchor_view.bbox[1]) - int(anchor.local_bbox_nm[1])
        for member in plan.member_plans:
            x0, y0, x1, y1 = member.local_bbox_nm
            target_cx = dx + (x0 + x1) // 2
            target_cy = dy + (y0 + y1) // 2
            view = result.get(member.member)
            if view is None:
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MEMBER_MISSING:{member.member}"
                )
            result[member.member] = _reposition_device(
                view, target_cx, target_cy
            )
    return result


def envelope_min_replace_placement(
    base: Placement,
    cell_def,
    intent,
    *,
    aspect_target: Optional[float] = 2.0,
    gap_um: float = 1.5,
    wl_weight: int = 4,
    max_time_s: float = 25.0,
) -> Tuple[Placement, PlacementResult]:
    """Re-place the devices of ``base`` with envelope-min; keep its built geometry.

    Returns ``(new_placement, cp_result)``. ``new_placement`` is a fresh
    :class:`Placement` carrying every device re-positioned + a rebuilt cell geom.
    """
    # Validate the exact-once structural authority even though this legacy
    # envelope solver has no native ArrayTile input.  The final projection below
    # seats the stored rigid macro; missing metadata is a typed stop, never a
    # route-resolver fallback.
    row_tiles = stored_row_footprint_tiles(base, intent)
    atomic_tiles = [*row_tiles, *rigid_nonmos_footprint_tiles(base)]
    atomic_tiles.extend(certified_matching_footprint_tiles(
        base, existing_tiles=atomic_tiles
    ))
    placed = dict(base.devices)
    if not placed:
        return base, PlacementResult(centers={}, width=0.0, height=0.0, envelope=0.0, status="EMPTY")

    labels = list(placed.keys())
    if atomic_tiles:
        row_keys = {tile.key for tile in row_tiles}
        row_owner = {
            tile.key: (
                f"row:{tile.key}"
                if tile.key in row_keys
                else f"atomic:{tile.key}"
            )
            for tile in atomic_tiles
        }
        member_owner = {
            member: row_owner[tile.key]
            for tile in atomic_tiles for member in tile.members
        }
        groups = [
            Group(key=lbl, w=max(_device_footprint_um(placed[lbl])[0], 0.1),
                  h=max(_device_footprint_um(placed[lbl])[1], 0.1))
            for lbl in labels if lbl not in member_owner
        ] + [
            Group(key=row_owner[tile.key], w=max(tile.w_um, 0.1), h=max(tile.h_um, 0.1))
            for tile in atomic_tiles
        ]

        def _lower_groups(rows):
            lowered = []
            for row in rows:
                mapped = []
                for name in row:
                    owner = member_owner.get(name, name)
                    if owner not in mapped:
                        mapped.append(owner)
                if len(mapped) >= 2:
                    lowered.append(mapped)
            return lowered

        nets = _lower_groups(_nets_from_cell_def(cell_def, labels))
        sym = [
            tuple(pair)
            for pair in _lower_groups(
                _placement_symmetry_pairs(base, intent, labels)
            )
        ]
    else:
        # Literal legacy branch: no declaration means no extra grouping, vars, or
        # ordering differences in the historical envelope model.
        member_owner = {}
        row_owner = {}
        groups = [Group(key=lbl, w=max(_device_footprint_um(placed[lbl])[0], 0.1),
                        h=max(_device_footprint_um(placed[lbl])[1], 0.1)) for lbl in labels]
        nets = _nets_from_cell_def(cell_def, labels)
        sym = _placement_symmetry_pairs(base, intent, labels)

    res = place_envelope_min(
        groups,
        nets=nets,
        symmetry_pairs=sym,
        order_constraints=_symmetry_order_constraints(sym),
        aspect_target=aspect_target,
        wl_weight=wl_weight,
        gap=gap_um,
        max_time_s=max_time_s,
    )

    # re-position every device onto its envelope-min centre (um -> nm).
    new_placed: Dict[str, PlacedDevice] = {}
    all_polys: List = []
    auxiliary_polys = placement_auxiliary_polys(base)
    tile_by_member = {
        member: tile for tile in atomic_tiles for member in tile.members
    }
    for lbl in labels:
        tile = tile_by_member.get(lbl)
        if tile is None:
            c = res.centers.get(lbl)
        else:
            centre = res.centers.get(row_owner[tile.key])
            rel = tile.rel.get(lbl)
            c = (
                (centre[0] + rel[0], centre[1] + rel[1])
                if centre is not None and rel is not None
                else None
            )
        if c is None:
            new_placed[lbl] = placed[lbl]
            all_polys += list(placed[lbl].polys or [])
            continue
        cx_nm = int(round(c[0] * 1000.0))
        cy_nm = int(round(c[1] * 1000.0))
        npd = _reposition_device(placed[lbl], cx_nm, cy_nm)
        new_placed[lbl] = npd
        all_polys += list(npd.polys or [])
    new_placed = _requantize_rigid_nonmos_members(base, new_placed)
    all_polys = [
        poly for device in new_placed.values() for poly in (device.polys or ())
    ] + auxiliary_polys

    full_extent = (
        min(p.bbox[0] for p in new_placed.values()),
        min(p.bbox[1] for p in new_placed.values()),
        max(p.bbox[2] for p in new_placed.values()),
        max(p.bbox[3] for p in new_placed.values()),
    )
    geom = _cell_geom(new_placed, full_extent)
    axis_x = int(round(((full_extent[0] + full_extent[2]) / 2)))

    new_placement = Placement(
        devices=new_placed,
        axis_x=axis_x,
        geom=geom,
        polys=all_polys,
        quality=base.quality,
        solver_status=f"envelope_min:{res.status}",
        solver_success=res.status in ("OPTIMAL", "FEASIBLE"),
        solve_seconds=base.solve_seconds,
        constraints=base.constraints,
        used_default_geometry=base.used_default_geometry,
        note=(base.note + f" | envelope_min re-place: {res.width:.1f}x{res.height:.1f}um "
              f"env={res.envelope:.0f} status={res.status}"),
        row_program=base.row_program,
        row_materialization=base.row_materialization,
        row_final_certificate=base.row_final_certificate,
        row_residuals=base.row_residuals,
        row_generated_nwell_bridge_keys=base.row_generated_nwell_bridge_keys,
        physical_macro_graph=base.physical_macro_graph,
        physical_macro_scope_token=base.physical_macro_scope_token,
        row_macro_contracts=base.row_macro_contracts,
        physical_macro_contracts=base.physical_macro_contracts,
        physical_macro_compile_inputs=base.physical_macro_compile_inputs,
        physical_macro_placement_certificate=None,
        rigid_nonmos_macro_inputs=base.rigid_nonmos_macro_inputs,
        rigid_nonmos_macro_plans=base.rigid_nonmos_macro_plans,
        rigid_nonmos_macro_interfaces=base.rigid_nonmos_macro_interfaces,
        matching_pair_axes=project_rigid_nonmos_matching_axes(base, new_placed),
        dropped_nonmos_matched=base.dropped_nonmos_matched,
        native_geometry=base.native_geometry,
    )
    projected = attach_physical_macro_projection(
        project_stored_row_master(new_placement)
    )
    if base.physical_macro_graph is not None:
        verify_rigid_nonmos_macro_projection(projected)
    return projected, res


def _extracted_intent_from_cell_def(cell_def, device_records=None, *, native_geometry=None):
    """Build an ``ExtractedIntent`` (devices + passives + derived topology pairs) from
    a parsed ``CellDef`` -- the inputs :func:`cp_group_placement_pos` consumes.

    REUSE, not hand-roll: the MOS/passive device universe comes from the SAME
    :func:`clay.layout.route.route_cell.bridge_devices_from_cell_def` the router bridge
    reads (materialised ``cell_def.devices`` else the bound-``.cir`` records), and the
    matched / diff / mirror pairs are derived by the GENERAL mechanical extractor
    :func:`clay.layout.extract.netlist_intent._derive_topology` -- the same topology
    derivation the specialized OTA path uses. Pure topology; no coordinate, no
    per-circuit branch."""
    from clay.layout.extract.netlist_intent import _derive_topology
    from clay.layout.extract.types import ExtractedIntent
    from clay.layout.route.route_cell import bridge_devices_from_cell_def

    if native_geometry is None:
        devices, passives = bridge_devices_from_cell_def(cell_def, device_records)
    else:
        from clay.layout.native_sky130 import native_route_devices

        native = native_route_devices(cell_def, native_geometry)
        devices = [device for device in native
                   if native_geometry.geometries[device.name].native_spec['family'] == 'mos']
        passives = [device for device in native
                    if native_geometry.geometries[device.name].native_spec['family'] != 'mos']
    # _derive_topology reads d.nets["source"/"gate"/"drain"] with bracket access; make
    # every MOS carry the four role keys so a device missing a pin is a "" net (never a
    # KeyError). The bridged nets are a plain dict, so setdefault is safe.
    for d in devices:
        for role in ("source", "drain", "gate", "body"):
            d.nets.setdefault(role, "")
    ext = ExtractedIntent(cell=str(getattr(cell_def, "local_name", "cell")))
    ext.devices = list(devices)
    ext.passives = list(passives)
    _derive_topology(ext)  # fills diff_pairs / mirror_pairs / matched_pairs
    if native_geometry is not None:
        # Shared gate connectivity is also common in clocked switches and NOR
        # logic.  Only the native role graph's source/PDK-grounded current
        # mirrors may turn that inference into a compact-placement pair.  The
        # canonical placement's explicit matching and symmetry are restored by
        # _inherit_placement_pair_identity at the actual compact caller.
        from clay.frontend.native_role_overlay import infer_native_role_overlay

        overlay = infer_native_role_overlay(cell_def, pdk_id='sky130')
        supported = {
            frozenset((left, right))
            for role in overlay.decomposition.roles if role.role == 'current_mirror'
            for i, left in enumerate(role.devices)
            for right in role.devices[i + 1:]
        }
        inferred_mirrors = {frozenset(pair) for pair in ext.mirror_pairs}
        ext.mirror_pairs = [pair for pair in ext.mirror_pairs if frozenset(pair) in supported]
        ext.matched_pairs = [pair for pair in ext.matched_pairs
                             if frozenset(pair) not in inferred_mirrors
                             or frozenset(pair) in supported]
    return ext


def declared_device_bands(intent) -> Dict[str, int]:
    """{device -> row ordinal} from the DECLARED @place_group rows -- the ONE
    source of the compaction's tier-band constraints.

    Only a @place_group pinned to a SINGLE floorplan row (``row_min == row_max``,
    i.e. the .clay declares ``row=N``) pins its devices to a tier; a group with
    no declared row (or a range) contributes nothing. A design that declares NO
    rows therefore packs in FREE 2D -- there is no flag that adds or drops
    bands, the declaration alone decides (leakage-free: declared ordinals only,
    never a reference coordinate)."""
    bands: Dict[str, int] = {}
    for pg in (getattr(intent, "place_groups", ()) or ()):
        r = getattr(pg, "row_min", None)
        if r is None or getattr(pg, "row_max", None) != r:
            continue  # only a single-row @place_group pins a device to one tier
        for d in (getattr(pg, "devices", ()) or ()):
            bands[str(d)] = int(r)
    return bands


def _require_bands_declaration_driven(respect_place_group_bands: bool) -> None:
    """NAMED error when a caller tries the retired band-override flag.

    ``respect_place_group_bands=False`` used to DROP the declared @place_group
    tier bands (the R6/R7 probe lever) -- a caller parameter silently
    overriding the design's .clay, which violates the declared-intent rule.
    Bands are now respected IFF the .clay declares ``row=``/``col=`` on a
    @place_group: to get the free-2D pack, REMOVE the row declarations from the
    design source (see ota_dev/source/two_stage_tsmc.clay). The parameter is
    kept for signature compatibility only."""
    if not respect_place_group_bands:
        raise ValueError(
            "respect_place_group_bands=False is retired: band constraints are "
            "DECLARATION-driven (a @place_group with row=/col= in the .clay). "
            "Declare free-2D intent by removing row=/col= from the design's "
            "@place_group rows instead of overriding them from a caller."
        )


def cp_compact_replace_placement(
    base: Placement,
    cell_def,
    intent,
    *,
    tech_id: str,
    aspect_target: Optional[float] = None,
    gap_um: float = 1.0,
    passive_gap_um: float = 2.0,
    max_time_s: float = 25.0,
    device_records=None,
    channel_relax_nets: Optional[Sequence[str]] = None,
    channel_pad_um: float = 0.0,
    channel_relax_passive_only: bool = True,
    mirror_pairs_rigid: bool = False,
    mirror_axis_mode: str = "shared_hard",
    respect_place_group_bands: bool = True,
    backend: str = "auto",
) -> Tuple[Placement, Optional[PlacementResult]]:
    """Re-place ``base`` compact through the ROUTABILITY-AWARE CP placer, keeping its
    built geometry.

    The routability-blind :func:`envelope_min_replace_placement` packs raw device
    footprints + a scalar gap with NO reservation for the router's emergent via
    landing pad -- so a routed layer-transition's via pad overhangs a neighbour's
    device metal, the via is suppressed, and nets short through the device-cell via
    stack (the measured ota_dev 7-net merge). This path routes the SAME compaction
    through :func:`clay.layout.synth.placement.cp_group_placement_pos` with ``tech_id``
    supplied, so the R2 via-landing + min-space budget (and the group ring pad) is
    RESERVED around every placed element by construction -- the shared obligation the
    bare placer dropped. Compact matched groups + passives are packed as first-class
    tiles; the returned device-level positions re-seat ``base``'s devices exactly as
    :func:`envelope_min_replace_placement` does (same rigid-translate mechanics, same
    canonical :class:`Placement`, no second materialization path).

    Returns ``(new_placement, cp_result)`` (``cp_result`` is the group-level
    :class:`PlacementResult`), or ``(base, None)`` when the cell exposes no device
    universe to place from (honest -- never a silent fake)."""
    from clay.layout.synth.placement import cp_group_placement_pos

    _require_bands_declaration_driven(respect_place_group_bands)
    placed = dict(base.devices)
    if not placed:
        return base, PlacementResult(centers={}, width=0.0, height=0.0,
                                     envelope=0.0, status="EMPTY")

    ext = _extracted_intent_from_cell_def(cell_def, device_records,
        native_geometry=getattr(base, 'native_geometry', None))
    if not ext.devices and not ext.passives:
        # No device universe (topology-only cell_def with no bound records) -> honest
        # no-op rather than a fabricated placement.
        return base, None

    # RING-INTEGRITY SEAM: the cell-def-derived ExtractedIntent drops the declared
    # @guard_ring islands, so the compact grouping (well_islands.compact_matched_groups)
    # would fold singletons across island boundaries (the straddle root: ours 2 frames
    # + 12 stranded taps vs the reference 9 frames + 0). Carry the declared islands onto
    # ext.guard_islands from the LayoutIntent in scope so declared_guard_islands()/
    # island_of() refine-never-straddle. No declaration -> empty -> byte-identical fold.
    ext.guard_islands = [
        {"members": tuple(str(m) for m in (getattr(g, "members", ()) or ())),
         "well": str(getattr(g, "well", "") or ""),
         "tie": str(getattr(g, "tie_net", "") or "")}
        for g in (getattr(intent, "guard_rings", ()) or ())
    ]

    labels = list(placed.keys())
    _inherit_placement_pair_identity(ext, base, intent, labels)
    sizes: Dict[str, Tuple[float, float]] = {
        lbl: (max(_device_footprint_um(placed[lbl])[0], 0.1),
              max(_device_footprint_um(placed[lbl])[1], 0.1))
        for lbl in labels
    }
    # Reserve BOTH de-siloed placement-time budgets by passing tech_id (mirrors the
    # specialized OTA call anchor_ota makes):
    #   * the R2 VIA-LANDING + min-space budget, so the router's emergent via pad clears
    #     foreign device metal by construction (the via-suppression open cure); and
    #   * the per-group RING pad (ring_pad_by_kind derived from tech_id), so a guard
    #     ring's outward WELL extension clears the neighbouring group -> the pmos nwell
    #     ring stops merging with an adjacent nmos substrate. MEASURED load-bearing on
    #     the external_pdk OTA: with the ring pad the input-stage 7-net via-stack short
    #     (GND,NET10,NET15,NET7,VDD,VIN,VIP) COLLAPSES; without it that short persists
    #     and the pmos ring shorts (gnd,vdd). General over any ringed compaction; no
    #     per-circuit branch.
    # DECLARED @guard_ring KEEP-TOGETHER: a ring is materialized as ONE frame around
    # its member cluster, so the members must be placed CONTIGUOUSLY (else the frame
    # sprawls) and two different-net ring frames must stay disjoint (else the rails
    # short -- the measured GND,NET048,NET07,NET4,VDD merge). The declared islands
    # live on the LayoutIntent (the SAME intent route_cell materializes the rings
    # from); thread them into the CP placer so the pack honors the ring topology by
    # construction. A cell that declares no ring passes None -> byte-identical.
    guard_islands = [list(getattr(g, "members", ()) or ())
                     for g in (getattr(intent, "guard_rings", ()) or ())]
    guard_islands = [isl for isl in guard_islands if isl]

    # DECLARED @place_group TIER BANDS (device -> row ordinal): a @place_group pinned
    # to a SINGLE floorplan row (the .clay declares row=N) assigns its devices to that
    # tier; the declared rows thread into the CP compaction so the pack compacts WITHIN
    # each band and preserves the band Y-ORDER instead of INTERLEAVING the tiers.
    # PURELY DECLARATION-DRIVEN (``declared_device_bands``): a cell that declares no
    # rows leaves this EMPTY -> no band constraint -> the groups pack in FREE 2D by
    # wirelength / functional adjacency (the 2D-NESTED pack -- the Miller/load caps
    # float in the wirelength mosaic instead of a side column, because the passive
    # side-bank constraint lives INSIDE place_envelope_min's band block). Band-vs-2D is
    # therefore a .clay statement (declare or remove row=), never a caller flag -- the
    # retired ``respect_place_group_bands=False`` override is a NAMED error above.
    # Intent-derived ordinals only; no reference coordinate, leakage-free.
    device_bands: Dict[str, int] = declared_device_bands(intent)

    # DECLARED @order / @abut (cell-level placement intent carried on the LayoutIntent):
    # thread them into the CP placer so the compact pack realises the declared relative
    # ORDER (a hard left_of/above) and ADJACENCY (a tight pull), and so a DECLARED @abut
    # pair of matched devices shares ONE continuous OD (diffusion abut) instead of two
    # gap-separated OD stamps -- the arraying-density fix. For the shared-OD pitch the
    # placer needs each device's per-side bbox->COMP inset; derive it from the BUILT
    # device geometry ONLY when an @abut is declared (else skip -> zero overhead). A cell
    # that declares NEITHER leaves both empty -> byte-identical (no threading, no press).
    abut_pairs = tuple(getattr(intent, "abut_pairs", ()) or ())
    od_inset_by_member: Optional[Dict[str, float]] = None
    if abut_pairs:
        od_inset_by_member = _active_x_insets_um(placed, str(tech_id))

    # DECLARED shared-diffusion/common-centroid rows: place each stored realization
    # as ONE atomic footprint, not as N free devices.  This is a projection of the
    # exact-once RowProgram master attached by place_generic; no second resolver or
    # geometry builder is invoked here.  Empty -> byte-identical legacy pack.
    stored_row_tiles = stored_row_footprint_tiles(base, intent)
    order_constraints = _compact_order_constraints(
        base, intent, labels, cell_def=cell_def, stored_row_tiles=stored_row_tiles
    )
    native_cross_row_inventory = None
    native_cross_row_groups = None
    native_cross_row_directions = None
    if base.row_program is not None and base.physical_macro_graph is not None:
        row_owners = {member: entry.entry_id for entry in base.row_program.entries
                      for member in entry.consumed_devices}
        cross_pairs = set()
        for macro in base.physical_macro_graph.macros:
            if macro.kind != "matching_relation":
                continue
            pair = tuple(sorted(macro.members))
            if len(pair) != 2 or any(member not in row_owners for member in pair):
                raise ValueError("native cross-row matching source partition is incomplete")
            if row_owners[pair[0]] != row_owners[pair[1]]:
                cross_pairs.add(pair)
        if cross_pairs:
            from clay.layout.place.cross_row_matching import native_cross_row_inventory_from_projection
            from clay.layout.place.row_source_projection import replay_row_source_projection
            from clay.layout.tech import LayoutTech

            cross_row_tech = LayoutTech.resolve(str(tech_id))
            replay_row_source_projection(
                base.row_materialization.source_projection, cell_def, cross_row_tech,
            )
            native_cross_row_inventory = native_cross_row_inventory_from_projection(
                base.row_program, base.row_materialization, cross_row_tech,
            )
            if (native_cross_row_inventory is None
                    or {row.members for row in native_cross_row_inventory.pairs} != cross_pairs):
                raise ValueError("native cross-row source matching inventory differs")
            entries = {entry.entry_id: entry for entry in base.row_program.entries}
            tiles = {tile.key: tile for tile in stored_row_tiles}
            native_cross_row_groups = {}
            for row, members in native_cross_row_inventory.row_source_members:
                entry = entries.get(row)
                tile = None if entry is None else tiles.get(entry.group)
                if tile is None or set(tile.members) != set(members):
                    raise ValueError("native cross-row source row does not own one CP tile")
                native_cross_row_groups[row] = f"A_{tile.key}"
            raw_axes = getattr(base, "matching_pair_axes", {})
            if type(raw_axes) is not dict or any(pair not in raw_axes for pair in cross_pairs):
                raise ValueError("native cross-row source matching axis identity is missing")
            native_cross_row_directions = {pair: raw_axes[pair][0] for pair in sorted(cross_pairs)}
    array_tiles = [
        *stored_row_tiles,
        *rigid_nonmos_footprint_tiles(base),
    ]
    array_tiles.extend(certified_matching_footprint_tiles(
        base, existing_tiles=array_tiles
    ))

    # COMMON-CENTROID PRESS (opt-in): thread the declared-symmetry press the CP
    # placer ALREADY implements (mirror_pairs_rigid + mirror_axis_mode) into the
    # compact generic path -- the SAME centered_soft press the OTA fold (anchor_ota)
    # reaches to place matched/mirror pairs about ONE shared centred axis. The
    # compact seam previously dropped these knobs, so a compact cell got no centroid
    # press (near-zero symmetry axis). Default OFF (mirror_pairs_rigid=False) adds NO
    # CP axis term -> byte-identical to the un-pressed compact pack. Topology-derived
    # (mirror_pairs); no reference coordinate, no per-circuit branch.
    from clay.layout.place.native_matching_well import (
        select_native_matching_well_pairs,
        verify_native_matching_group_wells,
    )
    from clay.layout.tech import LayoutTech
    native_matching_well_pairs = (
        select_native_matching_well_pairs(
            base, cell_def, intent, LayoutTech.resolve(str(tech_id))
        )
        if getattr(base, 'native_geometry', None) is not None else ()
    )
    well_pair_gaps = {
        frozenset(pair.names): pair.member_gap_nm / 1000.0
        for pair in native_matching_well_pairs
    }
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    layout_grid = pdk_manufacturing_grid_nm(LayoutTech.resolve(str(tech_id)))
    native_bundle = getattr(base, 'native_geometry', None)
    native_row_members = {name for entry in getattr(getattr(base, 'row_program', None), 'entries', ())
                          for name in entry.consumed_devices}
    native_grid = None
    native_device_frames = None
    native_device_names = set()
    if native_bundle is not None:
        native_grid = layout_grid
        native_device_names = set(native_bundle.geometries)
        native_device_frames = {name: tuple(pd.bbox) for name, pd in placed.items()
                                if name in native_device_names and name not in native_row_members}
    pos, res = cp_group_placement_pos(
        ext, sizes, tech_id=str(tech_id),
        gap=float(gap_um), passive_gap=float(passive_gap_um),
        max_time_s=float(max_time_s), aspect_target=aspect_target,
        keep_together_islands=(guard_islands or None),
        channel_relax_nets=channel_relax_nets, channel_pad_um=channel_pad_um,
        channel_relax_passive_only=channel_relax_passive_only,
        mirror_pairs_rigid=bool(mirror_pairs_rigid),
        mirror_axis_mode=str(mirror_axis_mode),
        device_bands=(device_bands or None),
        order_constraints=order_constraints,
        abut_pairs=abut_pairs,
        od_inset_by_member=od_inset_by_member,
        arrays=(array_tiles or None),
        matching_pair_axes=getattr(base, "matching_pair_axes", {}) or {},
        matching_pair_min_gap_um=well_pair_gaps,
        explicit_symmetry_pairs=_placement_symmetry_pairs(base, intent, labels),
        native_cross_row_inventory=native_cross_row_inventory,
        native_cross_row_groups=native_cross_row_groups,
        native_cross_row_directions=native_cross_row_directions,
        native_device_frames=native_device_frames,
        native_device_grid_nm=native_grid,
        backend=backend,
    )

    # re-position every device onto its CP centre (um -> nm). A device with no CP
    # position (not in the placed element set) keeps its base position.
    new_placed: Dict[str, PlacedDevice] = {}
    all_polys: List = []
    auxiliary_polys = placement_auxiliary_polys(base)
    for lbl in labels:
        c = pos.get(lbl)
        if c is None:
            new_placed[lbl] = placed[lbl]
            all_polys += list(placed[lbl].polys or [])
            continue
        cx_nm = int(round(c[0] * 1000.0))
        cy_nm = int(round(c[1] * 1000.0))
        npd = (_reposition_native_exact(placed[lbl], c, native_grid)
               if lbl in native_device_names and lbl not in native_row_members
               else _reposition_device(placed[lbl], cx_nm, cy_nm, grid_nm=layout_grid))
        new_placed[lbl] = npd
        all_polys += list(npd.polys or [])
    new_placed = _requantize_rigid_nonmos_members(base, new_placed)
    # The selected source masks are translated by the native rigid placement
    # above. Replay the PDK well rule on the actual pair before its offsets can
    # become one atomic matching owner.
    verify_native_matching_group_wells(native_matching_well_pairs, new_placed, res)
    native_row_cp_frame_proof = None
    if res.native_row_cp_capture is not None:
        from clay.layout.place.native_row_cp_frames import capture_native_row_cp_frame_proof
        from clay.layout.place.native_row_group_authority import capture_native_row_group_authority
        from clay.layout.tech import LayoutTech
        frame_tech = LayoutTech.resolve(str(tech_id))
        group_authority = capture_native_row_group_authority(
            base.row_program, base.row_materialization, frame_tech,
            gap_um=gap_um, channel_relax_nets=channel_relax_nets,
            channel_pad_um=channel_pad_um, channel_relax_passive_only=channel_relax_passive_only)
        native_row_cp_frame_proof = capture_native_row_cp_frame_proof(
            *res.native_row_cp_capture, base.row_program, base.row_materialization,
            frame_tech, group_authority=group_authority)
    if res.native_cross_row_cp_capture is not None:
        from clay.layout.place.cross_row_matching import NativeCrossRowCpProof, replay_native_cross_row_cp_proof
        if native_row_cp_frame_proof is None:
            raise ValueError('native cross-row CP capture lacks source-bound row frames')
        proof = NativeCrossRowCpProof(*res.native_cross_row_cp_capture,
                                      native_row_frame_proof=native_row_cp_frame_proof)
        replay_native_cross_row_cp_proof(native_cross_row_inventory, proof, native_cross_row_directions,
            row_program=base.row_program, row_materialization=base.row_materialization,
            tech=LayoutTech.resolve(str(tech_id)))
        res = replace(res, native_cross_row_cp_proof=proof)
    all_polys = [
        poly for device in new_placed.values() for poly in (device.polys or ())
    ] + auxiliary_polys

    full_extent = (
        min(p.bbox[0] for p in new_placed.values()),
        min(p.bbox[1] for p in new_placed.values()),
        max(p.bbox[2] for p in new_placed.values()),
        max(p.bbox[3] for p in new_placed.values()),
    )
    geom = _cell_geom(new_placed, full_extent)
    axis_x = int(round(((full_extent[0] + full_extent[2]) / 2)))

    new_placement = Placement(
        devices=new_placed,
        axis_x=axis_x,
        geom=geom,
        polys=all_polys,
        quality=base.quality,
        solver_status=f"cp_compact:{res.status}",
        solver_success=res.status in ("OPTIMAL", "FEASIBLE"),
        solve_seconds=base.solve_seconds,
        constraints=base.constraints,
        used_default_geometry=base.used_default_geometry,
        note=(base.note + f" | cp_compact re-place: {res.width:.1f}x{res.height:.1f}um "
              f"env={res.envelope:.0f} status={res.status} (via/ring reserved)"),
        row_program=base.row_program,
        row_materialization=base.row_materialization,
        row_final_certificate=base.row_final_certificate,
        row_residuals=base.row_residuals,
        row_generated_nwell_bridge_keys=base.row_generated_nwell_bridge_keys,
        physical_macro_graph=base.physical_macro_graph,
        physical_macro_scope_token=base.physical_macro_scope_token,
        row_macro_contracts=base.row_macro_contracts,
        physical_macro_contracts=base.physical_macro_contracts,
        physical_macro_compile_inputs=base.physical_macro_compile_inputs,
        physical_macro_placement_certificate=None,
        rigid_nonmos_macro_inputs=base.rigid_nonmos_macro_inputs,
        rigid_nonmos_macro_plans=base.rigid_nonmos_macro_plans,
        rigid_nonmos_macro_interfaces=base.rigid_nonmos_macro_interfaces,
        matching_pair_axes=_project_cp_owned_matching_axes(
            base,
            new_placed,
            res,
            stored_row_tiles=tuple(stored_row_tiles),
        ),
        native_cross_row_cp_proof=res.native_cross_row_cp_proof,
        native_row_cp_frame_proof=native_row_cp_frame_proof,
        dropped_nonmos_matched=base.dropped_nonmos_matched,
        native_geometry=base.native_geometry,
    )
    projected = attach_physical_macro_projection(
        project_stored_row_master(new_placement)
    )
    if base.physical_macro_graph is not None:
        verify_rigid_nonmos_macro_projection(projected)
    return projected, res


def maybe_envelope_replace(
    base: Placement,
    cell_def,
    intent,
    *,
    aspect_target: Optional[float] = None,
    gap_um: float = 1.5,
    tech_id: Optional[str] = None,
    device_records=None,
    channel_relax_nets: Optional[Sequence[str]] = None,
    channel_pad_um: float = 0.0,
    channel_relax_passive_only: bool = True,
    mirror_pairs_rigid: bool = False,
    mirror_axis_mode: str = "shared_hard",
    respect_place_group_bands: bool = True,
    **kw,
) -> Tuple[Placement, Optional[PlacementResult]]:
    """Gated compact re-placement of a built ``Placement``.

    This is the SINGLE gate the production generic flow (``_run_generic``) and the
    ota_dev build both call so there is no per-circuit branch: opting in is a scalar
    ``aspect_target`` supplied by the caller (a flow / build parameter, or the config
    fidelity band center), never a circuit name.

    * ``aspect_target is None`` -> NO-OP: returns ``(base, None)`` unchanged. The
      placement is byte-identical for every cell that does not opt in.
    * ``aspect_target`` set AND ``tech_id`` supplied -> re-packs ``base`` compact via
      :func:`cp_compact_replace_placement`, the ROUTABILITY-AWARE CP placer that
      RESERVES the R2 via-landing + min-space budget (mirrors the specialized OTA
      call ``anchor_ota`` makes) so the router's emergent via pad clears foreign
      device metal by construction (the de-siloed shared obligation).
    * ``aspect_target`` set with NO ``tech_id`` -> falls back to the historical
      routability-blind :func:`envelope_min_replace_placement` (unchanged).

    ``respect_place_group_bands`` is retired (signature-compat only): tier bands
    engage IFF the design's .clay declares ``row=`` on a @place_group
    (``declared_device_bands``); passing False is a NAMED error, never a silent
    override of the declaration.
    """
    _require_bands_declaration_driven(respect_place_group_bands)
    if aspect_target is None:
        return base, None
    if tech_id is not None:
        return cp_compact_replace_placement(
            base, cell_def, intent, tech_id=str(tech_id),
            aspect_target=aspect_target, gap_um=gap_um,
            device_records=device_records,
            channel_relax_nets=channel_relax_nets, channel_pad_um=channel_pad_um,
            channel_relax_passive_only=channel_relax_passive_only,
            mirror_pairs_rigid=mirror_pairs_rigid,
            mirror_axis_mode=mirror_axis_mode,
            respect_place_group_bands=respect_place_group_bands,
            **kw
        )
    # the routability-blind fallback has no per-net corridor obligation; the channel
    # feedback is only meaningful on the via/ring-reserving cp path (tech_id given).
    # The common-centroid press is likewise a cp-path capability (mirror_pairs_rigid /
    # mirror_axis_mode) -- NOT forwarded to the bare fallback (byte-identical there).
    return envelope_min_replace_placement(
        base, cell_def, intent, aspect_target=aspect_target, gap_um=gap_um, **kw
    )
