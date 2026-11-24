"""PDK-derived well, guard-ring, rail keepout, and ring-padding geometry."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.compact_blocks import _NM_PER_UM, extract_blocks
from clay.layout.place.constraints_spatial import (
    _stable_hard_order,
    assign_bands,
    common_centroid_devices,
    identify_left_column,
)
from clay.layout.place.placement_constants import (
    _BAND_CENTER,
    _BAND_TOP,
    _Bbox,
)


def _effective_block_gap_nm(intent: Any, tech: Any, declared_gap_nm: int) -> int:
    """Return the legal clearance common to every compact-block pair.

    ``@place_policy(gap=...)`` is an optimization target, not a waiver of PDK
    geometry rules.  COMP, M1 and implant spacing apply to every materialized ring
    boundary, so the scalar solve gap is the larger of those PDK-derived rules and
    the declared target.  N-well spacing is deliberately excluded here: it is
    lowered by :func:`_ring_pair_clearances_nm` only between two N-well ring blocks.

    A cell without a declared ring keeps the caller's gap exactly, preserving the
    historical unringed placement path.  The derivation is structural and PDK-driven:
    it does not inspect circuit, device, or net names.
    """

    if type(declared_gap_nm) is not int or declared_gap_nm < 0:
        raise ValueError("declared block gap must be a non-negative integer nm")
    if not tuple(getattr(intent, "guard_rings", ()) or ()):
        return declared_gap_nm

    from clay.layout.extract.realize_render import _group_ring_base_clearance_nm

    return max(declared_gap_nm, int(_group_ring_base_clearance_nm(tech)))

def _ring_pair_clearances_nm(
    intent: Any,
    tech: Any,
    *,
    block_ids: Sequence[str],
    block_member_devices: Optional[Mapping[str, Sequence[str]]] = None,
    known_device_names: Any = (),
    well_island_share_plans: Any = (),
    base_gap_nm: int,
) -> Dict[Tuple[str, str], int]:
    """Lower N-well spacing onto exactly the materialized N-ring block pairs.

    ``extract_blocks`` assigns declared guard-ring island ``i`` the structural id
    ``ISL{i}``.  Ring islands are rigid and never bundled, so this identity remains
    valid through the initial solve, exact-materialization replay and route-feedback
    replay.  No circuit, device or net name participates in the decision.
    """

    if type(base_gap_nm) is not int or base_gap_nm < 0:
        raise ValueError("base block gap must be a non-negative integer nm")
    present = {str(block_id) for block_id in block_ids}
    nwell_blocks = tuple(
        sorted(
            f"ISL{index}"
            for index, ring in enumerate(
                tuple(getattr(intent, "guard_rings", ()) or ())
            )
            if str(getattr(ring, "well", "")).strip().lower() == "n"
            and f"ISL{index}" in present
        )
    )
    if len(nwell_blocks) < 2:
        return {}

    from clay.layout.extract.realize_render import _group_ring_nwell_clearance_nm

    required = int(_group_ring_nwell_clearance_nm(tech))
    if required <= base_gap_nm:
        return {}
    shared_pairs: set[Tuple[str, str]] = set()
    for group in _well_share_ring_cohesion_groups(
        intent,
        tech,
        block_ids=tuple(sorted(present)),
        block_member_devices=block_member_devices,
        known_device_names=known_device_names,
        well_island_share_plans=well_island_share_plans,
    ):
        nwell_group = tuple(block_id for block_id in group if block_id in nwell_blocks)
        shared_pairs.update(
            (left, right)
            for index, left in enumerate(nwell_group)
            for right in nwell_group[index + 1 :]
        )
    return {
        (left, right): required
        for index, left in enumerate(nwell_blocks)
        for right in nwell_blocks[index + 1 :]
        if (left, right) not in shared_pairs
    }

def _well_share_ring_cohesion_groups(
    intent: Any,
    tech: Any,
    *,
    block_ids: Any,
    block_member_devices: Any,
    known_device_names: Any,
    well_island_share_plans: Any,
) -> Tuple[Tuple[str, ...], ...]:
    """Project each declared shared-well domain onto its exact ring masters.

    Well sharing relaxes a spacing constraint; it does not merge the independent
    ring masters.  The constructive placer nevertheless needs the complete master
    inventory as a soft cohesion authority, otherwise a finite-budget solve can
    scatter the relaxed islands more widely than the strict incumbent.  This
    projection consumes only canonical device ownership and the installed PDK
    plan.  Missing, mixed, or duplicated ownership is a typed failure rather than
    a silently weakened objective.
    """

    from clay.layout.well_island_sharing import replay_well_island_share_plans

    plans = replay_well_island_share_plans(well_island_share_plans, tech)
    if not plans:
        return ()
    if type(block_ids) is not tuple:
        raise TypeError("shared well master inventory must be an exact tuple")
    if (
        any(type(block_id) is not str or not block_id for block_id in block_ids)
        or block_ids != tuple(sorted(set(block_ids)))
    ):
        raise ValueError("shared well master inventory must be unique and canonical")
    if not isinstance(block_member_devices, Mapping):
        raise TypeError("shared well ring ownership must be one mapping")
    if type(known_device_names) is not tuple:
        raise TypeError("shared well device universe must be an exact tuple")
    if (
        any(type(name) is not str or not name for name in known_device_names)
        or known_device_names != tuple(sorted(set(known_device_names)))
    ):
        raise ValueError("shared well device universe must be unique and canonical")
    raw_rings = getattr(intent, "guard_rings", ())
    if type(raw_rings) is not tuple:
        raise TypeError("shared well ring declaration inventory must be an exact tuple")
    # A shared well is legal without a guard-ring declaration.  This consumer
    # projects ring masters only; device-level sharing is constructed and
    # checked by the well-island consumer.  With no rings it has no masters to
    # constrain, and must not invent a missing-ring obligation.
    if not raw_rings:
        return ()

    present = set(block_ids)
    known = set(known_device_names)
    ring_well_by_block = {
        f"ISL{index}": (
            "nwell"
            if str(getattr(ring, "well", "")).strip().lower() == "n"
            else "pwell"
            if str(getattr(ring, "well", "")).strip().lower() == "p"
            else ""
        )
        for index, ring in enumerate(raw_rings)
        if f"ISL{index}" in present
    }
    groups: list[Tuple[str, ...]] = []
    consumed_blocks: set[str] = set()
    for plan in plans:
        planned = set(plan.members)
        consumed_members: set[str] = set()
        owned_blocks: list[str] = []
        for block_id in sorted(ring_well_by_block):
            if ring_well_by_block[block_id] != plan.well:
                continue
            raw_members = block_member_devices.get(block_id, ())
            try:
                members = set(raw_members)
            except TypeError as exc:
                raise TypeError(
                    "shared well ring ownership must contain iterable member sets"
                ) from exc
            if any(type(member) is not str or not member for member in members):
                raise TypeError(
                    "shared well ring ownership contains a non-string member"
                )
            device_members = members & known
            intersection = device_members & planned
            if not intersection:
                continue
            if device_members - planned or consumed_members & device_members:
                raise ValueError(
                    f"shared well {plan.group!r} lacks exact ring ownership"
                )
            owned_blocks.append(block_id)
            consumed_members.update(device_members)
        if consumed_members != planned or not owned_blocks:
            raise ValueError(
                f"shared well {plan.group!r} lacks exact ring ownership"
            )
        group = tuple(owned_blocks)
        if consumed_blocks.intersection(group):
            raise ValueError("shared well cohesion groups overlap on ring masters")
        consumed_blocks.update(group)
        if len(group) >= 2:
            groups.append(group)
    return tuple(groups)

def _native_rail_landing_pad_halves_nm(
    tech: Any,
    *,
    block_stamps: Mapping[str, Sequence[str]],
    geometry: Mapping[str, Any],
    device_nets: Mapping[str, Mapping[str, str]],
    rail_specs: Sequence[Any],
    placement_um: Optional[Mapping[str, Tuple[float, float]]],
    ruled_conductors: Sequence[Tuple[int, int]],
) -> Dict[Tuple[str, Tuple[int, int]], int]:
    """Reserve only missing native-port climbs to an actual compiled upper rail.

    The materializer supplies the terminal inventory and complete native artwork.
    A same-net port on another disconnected component is not an escape for this
    one. Conversely, a real native cut joining this component to the next metal
    discharges the new-landing obligation. Coordinates use a half-nm local lattice
    so centroid-relative half phases are not rounded into false connections.
    """
    if not rail_specs or not device_nets:
        return {}
    import klayout.db as kdb

    rail_layers: Dict[str, set[int]] = {}
    for spec in rail_specs:
        rail_layers.setdefault(str(spec.net), set()).add(int(spec.layer_n))
    maximum = max(n for values in rail_layers.values() for n in values)
    climbs = []
    for upper_n in range(2, maximum + 1):
        upper = tuple(tech.metal_key(upper_n))
        if upper not in ruled_conductors:
            continue
        via = tech.via_rule(f"via{upper_n - 1}")
        size = int(via.size.value)
        spacing = int(via.spacing.value)
        enclosure = int(via.enclosure_above.value)
        # The router's installed redundant 2-by-2 landing footprint. Reserving
        # this extent also admits its smaller, legally chosen single-cut model.
        half_pad = (2 * size + spacing + 2 * enclosure + 1) // 2
        climbs.append((upper_n, tuple(tech.metal_key(upper_n - 1)),
                       upper, tuple(tech.via_key(upper_n - 1)), half_pad))
    result: Dict[Tuple[str, Tuple[int, int]], int] = {}
    for block, stamps in sorted(block_stamps.items()):
        regions: Dict[Tuple[int, int], Any] = {}
        ports = []
        for stamp in sorted(stamps):
            geom = geometry.get(stamp)
            if geom is None:
                continue
            dx, dy = (placement_um or {}).get(stamp, (0.0, 0.0))
            for layer, rings in (getattr(geom, "polys_by_layer", {}) or {}).items():
                region = regions.setdefault(tuple(layer), kdb.Region())
                for ring in rings:
                    region.insert(kdb.Polygon([
                        kdb.Point(round((float(x) + dx) * 2000),
                                  round((float(y) + dy) * 2000))
                        for x, y in ring]))
            nets = device_nets.get(stamp, {})
            layers = getattr(geom, "terminal_layers", {}) or {}
            for terminal, points in (
                getattr(geom, "terminal_accesses_um", {}) or {}
            ).items():
                long_role = {"s": "source", "d": "drain", "g": "gate", "b": "body"}.get(terminal)
                bindings = {nets[key] for key in (terminal, long_role)
                            if key is not None and key in nets}
                if len(bindings) > 1:
                    raise ValueError("native terminal has conflicting source binding aliases")
                net = next(iter(bindings), None)
                if net not in rail_layers or terminal not in layers:
                    continue
                for x, y in points:
                    ports.append((str(net), tuple(layers[terminal]),
                                  kdb.Point(round((float(x) + dx) * 2000),
                                            round((float(y) + dy) * 2000))))
        if not ports:
            continue
        for upper_n, lower, upper, cut_layer, half_pad in climbs:
            components = tuple(regions.get(lower, kdb.Region()).merged().each())
            cuts = regions.get(cut_layer, kdb.Region())
            upper_metal = regions.get(upper, kdb.Region())
            for net, layer, point in ports:
                if layer != lower or not any(n >= upper_n for n in rail_layers[net]):
                    continue
                containing = [polygon for polygon in components if polygon.inside(point)]
                if len(containing) != 1:
                    raise ValueError("native rail terminal lacks one exact metal component")
                component = kdb.Region(containing[0])
                native_cuts = cuts.interacting(component)
                if not upper_metal.interacting(native_cuts).is_empty():
                    continue
                result[(str(block), upper)] = max(
                    result.get((str(block), upper), 0), half_pad)
    return result


def _mim_pair_clearances_nm(
    tech: Any,
    *,
    block_stamps: Mapping[str, Sequence[str]],
    geometry: Mapping[str, Any],
    base_gap_nm: int,
    device_nets: Optional[Mapping[str, Mapping[str, str]]] = None,
    rail_specs: Sequence[Any] = (),
    placement_um: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[Tuple[str, str], int]:
    """Lower PDK MIM bottom-metal spacing onto exact rigid-block pairs.

    A foundry MIM rule selects a bottom conductor through its marker and then
    requires clearance to *any* other conductor on that layer.  The compact
    placer therefore cannot treat ``@place_policy(gap=...)`` as a waiver, but it
    also must not inflate every pair globally.  This projection inspects the
    materialized stamp layers in each rigid block and raises only a pair where
    one side contains the marker-selected bottom stack and the other side can
    draw the ruled conductor.  Circuit, device and PDK names never participate.

    Exact within-cell MIM geometry remains the device materializer's contract;
    this function owns only separation between independently movable blocks.
    """

    if type(base_gap_nm) is not int or base_gap_nm < 0:
        raise ValueError("base block gap must be a non-negative integer nm")
    rules = tuple(
        tech.mim_bottom_routing_clearance_rules()
        if hasattr(tech, "mim_bottom_routing_clearance_rules")
        else ()
    )
    if not rules:
        return {}

    native_landings = _native_rail_landing_pad_halves_nm(
        tech, block_stamps=block_stamps, geometry=geometry,
        device_nets=device_nets or {}, rail_specs=rail_specs,
        placement_um=placement_um,
        ruled_conductors=tuple(tuple(rule[0]) for rule in rules),
    )

    layers_by_block: Dict[str, set[Tuple[int, int]]] = {}
    for block_id, stamps in sorted(block_stamps.items()):
        layers: set[Tuple[int, int]] = set()
        for stamp in sorted(str(value) for value in (stamps or ())):
            geom = geometry.get(stamp)
            for raw_layer, polygons in (
                getattr(geom, "polys_by_layer", {}) or {}
            ).items():
                if polygons:
                    layers.add((int(raw_layer[0]), int(raw_layer[1])))
        layers_by_block[str(block_id)] = layers

    obligations: Dict[Tuple[str, str], int] = {}
    block_ids = tuple(sorted(layers_by_block))
    for index, left in enumerate(block_ids):
        left_layers = layers_by_block[left]
        for right in block_ids[index + 1 :]:
            right_layers = layers_by_block[right]
            required = base_gap_nm
            for conductor, marker, clearance_nm in rules:
                conductor_key = (int(conductor[0]), int(conductor[1]))
                marker_key = (int(marker[0]), int(marker[1]))
                clearance = int(clearance_nm)
                left_bottom = (
                    conductor_key in left_layers and marker_key in left_layers
                )
                right_bottom = (
                    conductor_key in right_layers and marker_key in right_layers
                )
                if (
                    left_bottom and conductor_key in right_layers
                ) or (
                    right_bottom and conductor_key in left_layers
                ):
                    required = max(required, clearance)
                if left_bottom and (right, conductor_key) in native_landings:
                    required = max(required, clearance + native_landings[(right, conductor_key)])
                if right_bottom and (left, conductor_key) in native_landings:
                    required = max(required, clearance + native_landings[(left, conductor_key)])
            if required > base_gap_nm:
                obligations[(left, right)] = required
    return obligations

def _mim_native_access_pair_clearances_nm(tech: Any, **kwargs: Any) -> Dict[Tuple[str, str], int]:
    """Return only future landing reservations, separate from native mask facts."""
    physical = _mim_pair_clearances_nm(
        tech, block_stamps=kwargs["block_stamps"], geometry=kwargs["geometry"],
        base_gap_nm=kwargs["base_gap_nm"])
    complete = _mim_pair_clearances_nm(tech, **kwargs)
    return {pair: gap for pair, gap in complete.items()
            if gap > physical.get(pair, kwargs["base_gap_nm"])}


def _layer_disjoint_overlap_compatible_pairs(
    *,
    block_stamps: Mapping[str, Sequence[str]],
    geometry: Mapping[str, Any],
    master_members: Mapping[str, Sequence[str]],
    excluded_layers: Tuple[Tuple[int, int], ...] = (),
) -> Tuple[Tuple[str, str], ...]:
    """Prove which rigid masters may share XY because their artwork layers do not.

    This is deliberately a local-geometry proof, not a device-kind heuristic.  A
    master is eligible only when every owned materialized block has a non-empty,
    resolvable stamp inventory and all observed non-empty ``polys_by_layer`` keys
    are exact integer GDS layer/datatype pairs.  Missing or malformed evidence is
    fail-closed and therefore leaves the ordinary 2-D collision rule in force.
    """

    if not isinstance(block_stamps, Mapping):
        raise TypeError("overlap compatibility block stamps must be a mapping")
    if not isinstance(geometry, Mapping):
        raise TypeError("overlap compatibility geometry must be a mapping")
    if not isinstance(master_members, Mapping):
        raise TypeError("overlap compatibility master members must be a mapping")

    inventories: Dict[str, frozenset[Tuple[int, int]]] = {}
    for raw_master, raw_members in sorted(master_members.items()):
        if type(raw_master) is not str or not raw_master:
            raise TypeError("overlap compatibility master name must be exact")
        try:
            members = tuple(raw_members)
        except TypeError as exc:
            raise TypeError(
                "overlap compatibility master members must be finite"
            ) from exc
        if (
            not members
            or any(type(member) is not str or not member for member in members)
            or len(set(members)) != len(members)
        ):
            raise ValueError(
                "overlap compatibility master members must be non-empty and unique"
            )
        layers: set[Tuple[int, int]] = set()
        complete = True
        for member in members:
            try:
                stamps = tuple(block_stamps.get(member, ()))
            except TypeError:
                complete = False
                break
            if (
                not stamps
                or any(type(stamp) is not str or not stamp for stamp in stamps)
            ):
                complete = False
                break
            for stamp_name in stamps:
                stamp = geometry.get(stamp_name)
                polygons_by_layer = getattr(stamp, "polys_by_layer", None)
                if stamp is None or not isinstance(polygons_by_layer, Mapping):
                    complete = False
                    break
                for raw_layer, polygons in polygons_by_layer.items():
                    if not polygons:
                        continue
                    if raw_layer in excluded_layers:
                        continue
                    if (
                        type(raw_layer) is not tuple
                        or len(raw_layer) != 2
                        or any(type(value) is not int for value in raw_layer)
                    ):
                        complete = False
                        break
                    layers.add(raw_layer)
                if not complete:
                    break
            if not complete:
                break
        if complete and layers:
            inventories[raw_master] = frozenset(layers)

    compatible = []
    masters = tuple(sorted(inventories))
    for index, left in enumerate(masters):
        for right in masters[index + 1 :]:
            if inventories[left].isdisjoint(inventories[right]):
                compatible.append((left, right))
    return tuple(compatible)

def _project_pair_clearances_to_masters(
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    *,
    master_members: Mapping[str, Sequence[str]],
    base_gap_nm: int,
) -> Dict[Tuple[str, str], int]:
    """Project materialized-block obligations through rigid master bundling."""

    member_to_master: Dict[str, str] = {}
    for master, members in sorted(master_members.items()):
        for member in members:
            value = str(member)
            if value in member_to_master and member_to_master[value] != str(master):
                raise ValueError("one materialized block belongs to multiple masters")
            member_to_master[value] = str(master)
    out: Dict[Tuple[str, str], int] = {}
    for raw_pair, raw_required in sorted(pair_clearances_nm.items()):
        left = member_to_master.get(str(raw_pair[0]))
        right = member_to_master.get(str(raw_pair[1]))
        if left is None or right is None:
            raise ValueError("pair clearance references an unbound materialized block")
        if left == right:
            continue  # consumed inside the rigid composite by its bundler
        pair = tuple(sorted((left, right)))
        out[pair] = max(out.get(pair, base_gap_nm), int(raw_required))
    return dict(sorted(out.items()))

def _project_pair_clearances_through_composites(
    obligations, *, live_ids, composites, base_gap_nm,
):
    """Project leaf-owned rules after each intermediate rigid bundling stage."""
    def leaves(name, active=()):
        if name in active:
            raise ValueError("pair-clearance composite membership contains a cycle")
        if name not in composites:
            return (name,)
        return tuple(leaf for member in composites[name]
                     for leaf in leaves(member, (*active, name)))
    return _project_pair_clearances_to_masters(
        obligations,
        master_members={name:leaves(name) for name in live_ids},
        base_gap_nm=base_gap_nm,
    )


def _project_native_access_pair_clearances_to_masters(
    obligations: Mapping[Tuple[str, str], int], *,
    master_members: Mapping[str, Sequence[str]], base_gap_nm: int,
) -> Dict[Tuple[str, str], int]:
    """Project a reserved interface without pretending it is native mask artwork."""
    result = _project_pair_clearances_to_masters(
        obligations, master_members=master_members, base_gap_nm=base_gap_nm)
    owners = {member: master for master, members in master_members.items()
              for member in members}
    for left, right in obligations:
        if owners[left] == owners[right]:
            raise ValueError("native access reservation inside one rigid master needs a local clearance proof")
    return result


def _merge_pair_clearances_nm(
    *obligations: Mapping[Tuple[str, str], int],
) -> Dict[Tuple[str, str], int]:
    """Combine independently derived pair obligations without weakening either."""

    out: Dict[Tuple[str, str], int] = {}
    for table in obligations:
        for raw_pair, required in table.items():
            pair = tuple(sorted((str(raw_pair[0]), str(raw_pair[1]))))
            out[pair] = max(out.get(pair, 0), int(required))
    return dict(sorted(out.items()))

def _canonical_pair_clearances_nm(
    pair_clearances_nm: Any,
    *,
    block_ids: Sequence[str],
    base_gap_nm: int,
) -> Dict[Tuple[str, str], int]:
    """Validate and canonicalize exact pair-clearance obligations."""

    present = {str(block_id) for block_id in block_ids}
    try:
        raw_rows = dict(pair_clearances_nm or {})
    except (TypeError, ValueError) as exc:
        raise ValueError("pair clearances must be a mapping") from exc
    out: Dict[Tuple[str, str], int] = {}
    for raw_pair, required in raw_rows.items():
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            raise ValueError("pair clearance keys must contain two block ids")
        left, right = (str(raw_pair[0]), str(raw_pair[1]))
        pair = tuple(sorted((left, right)))
        if (
            not left
            or not right
            or left == right
            or left not in present
            or right not in present
            or type(required) is not int
            or required < base_gap_nm
        ):
            raise ValueError(
                "pair clearance has an invalid block or distance: "
                f"pair={pair!r}, required={required!r}, base_gap_nm={base_gap_nm!r}, "
                f"missing={tuple(name for name in pair if name not in present)!r}"
            )
        if pair in out and out[pair] != required:
            raise ValueError("pair clearance contains conflicting distances")
        out[pair] = int(required)
    return dict(sorted(out.items()))

# --------------------------------------------------------------------------- #
# BLOCK-MODEL adapters: resolve member devices, bundle passive composites, assign
# band ordinals. PURE (no geometry / CP) -- unit-testable on synthetic block data.
# --------------------------------------------------------------------------- #
def _resolve_block_member_devices(
    block_stamps: Dict[str, set], geom: Dict[str, Any]
) -> Dict[str, set]:
    """``{block_id -> {member device names}}``. A stamp names its device directly;
    an array / common-centroid TILE stamp lists its consumed devices on
    ``composite_members`` (the SAME provenance the rest of the pipeline reads)."""
    out: Dict[str, set] = {}
    for bid, stamps in block_stamps.items():
        devs: set = set()
        for s in stamps:
            devs.add(s)
            devs |= set(getattr(geom.get(s), "composite_members", ()) or ())
        out[bid] = devs
    return out

def _resolved_device_master_rows(
    block_member_devices: Mapping[str, set],
    live_master_ids: Tuple[str, ...],
) -> Tuple[Tuple[str, str], ...]:
    """Freeze device ownership using only masters admitted to this solver call."""

    if not isinstance(block_member_devices, Mapping):
        raise TypeError("solver master ownership must be a mapping")
    if (
        type(live_master_ids) is not tuple
        or not live_master_ids
        or live_master_ids != tuple(sorted(set(live_master_ids)))
        or any(type(master) is not str or not master for master in live_master_ids)
    ):
        raise TypeError("live solver masters must be one exact sorted unique tuple")

    device_owner: Dict[str, str] = {}
    for master in live_master_ids:
        if master not in block_member_devices:
            raise ValueError(f"live solver master {master!r} lacks device ownership")
        members = block_member_devices[master]
        if (
            type(members) is not set
            or not members
            or any(type(device) is not str or not device for device in members)
        ):
            raise TypeError(
                f"live solver master {master!r} must own one nonempty device set"
            )
        for device in sorted(members):
            prior = device_owner.get(device)
            if prior is not None:
                raise ValueError(
                    f"device {device!r} belongs to multiple live solver masters: "
                    f"{prior!r}, {master!r}"
                )
            device_owner[device] = master

    return tuple(sorted(device_owner.items()))

def _installed_adjacency_pitch_nm(tech: Any) -> int:
    """Return the installed poly pitch used by ``@adjacent`` ratios."""

    from clay.layout.device_cell import _poly_name

    rules = tech.rules
    poly_name = _poly_name(rules)
    width = rules.min_width(poly_name)
    spacing = rules.min_spacing(poly_name)
    if (
        type(width) is not int
        or width <= 0
        or type(spacing) is not int
        or spacing <= 0
    ):
        raise ValueError("@adjacent requires installed poly width and spacing rules")
    return width + spacing

def _shared_well_passive_gap_nm(tech: Any) -> int:
    """PDK floor between passive devices already enclosed by one well island."""

    from clay.layout.device_cell import _active_name, _poly_name

    resolved = []
    for layer_name in (_active_name(tech), _poly_name(tech)):
        try:
            spacing = tech.min_space(layer_name)
        except KeyError:
            spacing = None
        if type(spacing) is not int or spacing <= 0:
            raise ValueError(
                "same-well passive adjacency requires positive integer active "
                "and poly spacing authority"
            )
        resolved.append(spacing)
    return max(resolved)

def _pdk_layer_clearances_nm(tech: Any) -> Dict[Tuple[int, int], int]:
    """Return same-layer spacing indexed by the installed GDS layer key.

    Several rule-container names may resolve to one drawing layer.  The maximum
    declared spacing is retained so aliases cannot weaken the foundry rule.
    Layers without a spacing rule are deliberately omitted; callers retain
    their conservative block-clearance fallback for those layers.
    """

    clearances: Dict[Tuple[int, int], int] = {}

    def install(layer_name: str, spacing: object) -> None:
        if type(spacing) is not int or spacing < 0:
            return
        try:
            key = tuple(tech.rules.layer(layer_name))
        except KeyError:
            return
        if len(key) != 2 or any(type(value) is not int or value < 0 for value in key):
            raise ValueError("PDK spacing names a malformed GDS layer key")
        clearances[key] = max(clearances.get(key, 0), spacing)

    for layer_name in tech.rules.layer_names():
        try:
            spacing = tech.min_space(layer_name)
        except KeyError:
            continue
        install(layer_name, spacing)

    for via_name in tech.rules.via_names():
        try:
            install(via_name, tech.via_spacing(via_name))
        except KeyError:
            continue

    implant = tech.implant_well()
    install("nplus", getattr(implant, "nplus_min_space", None))
    install("pplus", getattr(implant, "pplus_min_space", None))

    declared = getattr(tech.adapter, "placement_layer_spacing_nm", {}) or {}
    if not isinstance(declared, Mapping):
        raise TypeError("PDK placement layer spacing authority must be a mapping")
    for layer_name, spacing in sorted(declared.items()):
        install(str(layer_name), spacing)
    return clearances

def _same_bulk_nwell_pair_clearances_nm(
    *,
    block_member_devices: Mapping[str, set],
    device_nets: Mapping[str, Mapping[str, str]],
    tech: Any,
) -> Dict[Tuple[str, str, Tuple[int, int]], int]:
    """Bind the parsed same-potential N-well rule to proven same-bulk owners."""

    declared = (
        getattr(tech.adapter, "placement_same_net_layer_spacing_nm", {}) or {}
    )
    if not isinstance(declared, Mapping):
        raise TypeError("PDK same-net layer spacing authority must be a mapping")
    spacing = declared.get("nwell")
    nwell_key = tech.build_layer("nwell")
    if type(spacing) is not int or spacing < 0 or nwell_key is None:
        return {}
    key = tuple(nwell_key)
    bulk_by_owner: Dict[str, str] = {}
    for owner, members in sorted(block_member_devices.items()):
        bulks = {
            str(nets.get("body") or nets.get("b") or nets.get("bulk") or "")
            for member in members
            if (nets := device_nets.get(str(member))) is not None
        }
        bulks.discard("")
        if len(bulks) == 1:
            bulk_by_owner[str(owner)] = next(iter(bulks))
    rows: Dict[Tuple[str, str, Tuple[int, int]], int] = {}
    owners = tuple(sorted(bulk_by_owner))
    for index, left in enumerate(owners):
        for right in owners[index + 1 :]:
            if bulk_by_owner[left] == bulk_by_owner[right]:
                rows[(left, right, key)] = spacing
    return rows


def _minimum_passive_layer_gap_nm(
    tech, lower_layers, upper_layers, *, axis, current_gap_nm, minimum_nm, maximum_nm,
):
    """Smallest grid-legal gap proved on the two actual native inventories.

    The owners are already ordered along one disjoint full-bbox axis. Moving
    the upper owner away is monotone, so an integer binary search preserves
    compactness without charging an inset implant to a metal overhang.
    """
    import klayout.db as kdb
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    if type(axis) is not int or axis not in (0, 1):
        raise ValueError("passive spacing requires one exact placement axis")
    if any(type(value) is not int or value < 0
           for value in (current_gap_nm, minimum_nm, maximum_nm)):
        raise ValueError("passive spacing bounds must be nonnegative integer nm")
    grid = max(1, int(pdk_manufacturing_grid_nm(tech)))
    clearances = _pdk_layer_clearances_nm(tech)

    def region(rectangles):
        result = kdb.Region()
        for box in rectangles:
            if (type(box) is not tuple or len(box) != 4
                    or any(type(value) is not int for value in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError("passive spacing requires canonical native rectangles")
            result.insert(kdb.Box(*box))
        result.merge()
        return result

    pairs = [(region(lower_layers[key]), region(upper_layers[key]), clearances[key])
             for key in sorted(set(lower_layers) & set(upper_layers) & set(clearances))
             if clearances[key] > 0]

    def legal(gap):
        shift = gap - current_gap_nm
        dx, dy = (shift, 0) if axis == 0 else (0, shift)
        return all(left.separation_check(right.moved(dx, dy), spacing).is_empty()
                   for left, right, spacing in pairs)

    low, high = (minimum_nm + grid - 1) // grid, maximum_nm // grid
    if low > high or not legal(high * grid):
        raise ValueError("same-ring adjacency is below actual passive layer spacing")
    while low < high:
        middle = (low + high) // 2
        if legal(middle * grid):
            high = middle
        else:
            low = middle + 1
    return low * grid

def _same_bulk_mergeable_pair_layers(
    *,
    block_member_devices: Mapping[str, set],
    device_nets: Mapping[str, Mapping[str, str]],
    tech: Any,
) -> Tuple[Tuple[str, str, Tuple[int, int]], ...]:
    """Return mergeable enclosure layers for owners in one proven bulk domain."""

    declared = getattr(
        getattr(tech, "adapter", None), "placement_same_domain_mergeable_layers", ()
    )
    if (
        type(declared) is not tuple
        or declared != tuple(sorted(set(declared)))
        or any(type(name) is not str or not name for name in declared)
    ):
        raise TypeError("PDK mergeable placement layers must be canonical")
    layer_keys = []
    for layer_name in declared:
        try:
            raw_key = tuple(tech.rules.layer(layer_name))
        except KeyError:
            continue
        if len(raw_key) != 2 or any(
            type(value) is not int or value < 0 for value in raw_key
        ):
            raise ValueError("PDK mergeable placement layer key is malformed")
        layer_keys.append(raw_key)

    bulk_by_owner: Dict[str, str] = {}
    for owner, members in sorted(block_member_devices.items()):
        bulks = {
            str(nets.get("body") or nets.get("b") or nets.get("bulk") or "")
            for member in members
            if (nets := device_nets.get(str(member))) is not None
        }
        bulks.discard("")
        if len(bulks) == 1:
            bulk_by_owner[str(owner)] = next(iter(bulks))
    rows = []
    owners = tuple(sorted(bulk_by_owner))
    for index, left in enumerate(owners):
        for right in owners[index + 1 :]:
            if bulk_by_owner[left] != bulk_by_owner[right]:
                continue
            rows.extend((left, right, key) for key in layer_keys)
    return tuple(sorted(rows))

def _materialize_same_ring_passive_adjacency(
    geom: Mapping[str, Any],
    pum: Mapping[str, Tuple[float, float]],
    *,
    intent: Any,
    tech: Any,
    tech_id: str,
    bridged_intent: Any,
    declarations: Sequence[Any],
    device_kind: Mapping[str, str],
    block_gring: Mapping[str, str],
    gring_names: set[str],
    block_member_devices: Mapping[str, set[str]],
    rigid_nonmos_macro_plans: Any,
) -> tuple[Dict[str, Any], Dict[str, Tuple[float, float]]]:
    """Close one declared passive pair inside a shared guard-ring island."""

    rigid_members = {
        member
        for plan in tuple(getattr(rigid_nonmos_macro_plans, "plans", ()) or ())
        for member in plan.members
    }
    member_block = {
        member: block
        for block, members in block_member_devices.items()
        for member in members
    }
    candidates = []
    for declaration in declarations:
        left = declaration.source
        right = declaration.target
        block = member_block.get(left)
        if block is None or block != member_block.get(right) or block not in block_gring:
            continue
        if {
            str(device_kind.get(left, "")).lower(),
            str(device_kind.get(right, "")).lower(),
        } - {"res", "resistor", "cap", "capacitor"}:
            continue
        if left in rigid_members or right in rigid_members:
            continue
        if set(block_member_devices[block]) != {left, right}:
            raise ValueError(
                "same-ring passive adjacency requires one exact two-member island"
            )
        candidates.append((block, declaration))
    if not candidates:
        return geom, pum
    if len({block for block, _declaration in candidates}) != len(candidates):
        raise ValueError("same-ring passive island has ambiguous adjacency declarations")
    if bridged_intent is None:
        raise ValueError("same-ring passive adjacency requires bridged source intent")

    from clay.layout.extract.realize_render import (
        _placed_stamp_bboxes_nm, _placed_stamp_layer_rectangles_nm,
    )

    bboxes = _placed_stamp_bboxes_nm(geom, pum)
    layer_rectangles = _placed_stamp_layer_rectangles_nm(geom, pum)

    def physical_owner(member: str) -> str:
        owners = tuple(sorted(
            name
            for name, stamp in geom.items()
            if name == member
            or member in tuple(getattr(stamp, "composite_members", ()) or ())
        ))
        if len(owners) != 1:
            raise ValueError(
                f"same-ring passive member {member!r} lacks unique geometry ownership"
            )
        return owners[0]

    pitch = _installed_adjacency_pitch_nm(tech)
    minimum_gap = _shared_well_passive_gap_nm(tech)
    stamp_deltas: Dict[str, Tuple[int, int]] = {}
    for _block, declaration in candidates:
        ratio = declaration.max_separation_multiplier
        maximum = (
            pitch * ratio.numerator + ratio.denominator - 1
        ) // ratio.denominator
        if minimum_gap > maximum:
            raise ValueError(
                "same-ring passive adjacency is below installed PDK spacing"
            )
        left_owner = physical_owner(declaration.source)
        right_owner = physical_owner(declaration.target)
        if left_owner == right_owner:
            raise ValueError("same-ring passive endpoints share one geometry owner")
        left = bboxes[left_owner]
        right = bboxes[right_owner]
        x_overlap = min(left[2], right[2]) - max(left[0], right[0])
        y_overlap = min(left[3], right[3]) - max(left[1], right[1])
        if y_overlap > 0 and (left[2] <= right[0] or right[2] <= left[0]):
            lower, upper = (
                (left_owner, right_owner)
                if left[0] <= right[0]
                else (right_owner, left_owner)
            )
            current = bboxes[upper][0] - bboxes[lower][2]
            axis = 0
        elif x_overlap > 0 and (left[3] <= right[1] or right[3] <= left[1]):
            lower, upper = (
                (left_owner, right_owner)
                if left[1] <= right[1]
                else (right_owner, left_owner)
            )
            current = bboxes[upper][1] - bboxes[lower][3]
            axis = 1
        else:
            raise ValueError(
                "same-ring passive adjacency lacks a single shared-edge axis"
            )
        if current < 0:
            raise ValueError("same-ring passive members overlap")
        legal_gap = _minimum_passive_layer_gap_nm(
            tech, layer_rectangles[lower], layer_rectangles[upper], axis=axis,
            current_gap_nm=current, minimum_nm=minimum_gap, maximum_nm=maximum,
        )
        if legal_gap <= current <= maximum:
            continue
        delta = (legal_gap - current, 0) if axis == 0 else (0, legal_gap - current)
        prior = stamp_deltas.get(upper)
        if prior is not None and prior != delta:
            raise ValueError("same-ring passive member has conflicting adjacency moves")
        stamp_deltas[upper] = delta

    if not stamp_deltas:
        return geom, pum
    adjusted_pum = dict(pum)
    for stamp, (dx, dy) in stamp_deltas.items():
        px, py = adjusted_pum[stamp]
        adjusted_pum[stamp] = (
            float(px) + dx / _NM_PER_UM,
            float(py) + dy / _NM_PER_UM,
        )
    adjusted_geom = dict(geom)
    for ring_name in gring_names:
        adjusted_geom.pop(ring_name, None)
        adjusted_pum.pop(ring_name, None)
    guard_rings = tuple(getattr(intent, "guard_rings", ()) or ())
    if guard_rings:
        from clay.layout.route.route_cell import _materialize_guard_rings

        _materialize_guard_rings(
            guard_rings,
            bridged_intent,
            adjusted_pum,
            adjusted_geom,
            tech,
            tech_id,
        )
    return adjusted_geom, adjusted_pum

def _resolved_adjacency_limits_nm(
    declarations: object,
    *,
    device_blocks: Mapping[str, str],
    tech: Any,
    minimum_gap_nm: int,
) -> Tuple[Tuple[str, str, int], ...]:
    """Bind source members to rigid blocks and consume the declared ratio."""

    from clay.layout.intent import replay_adjacent_inventory

    typed = replay_adjacent_inventory(declarations)
    if not typed:
        return ()
    pitch = _installed_adjacency_pitch_nm(tech)
    limits: Dict[Tuple[str, str], int] = {}
    for declaration in typed:
        left = device_blocks.get(declaration.source)
        right = device_blocks.get(declaration.target)
        if left is None or right is None:
            raise ValueError("@adjacent endpoint has no materialized rigid block")
        if left == right:
            continue
        ratio = declaration.max_separation_multiplier
        maximum = (pitch * ratio.numerator + ratio.denominator - 1) // ratio.denominator
        if maximum < minimum_gap_nm:
            raise ValueError(
                "@adjacent maximum separation is below the legal placement gap"
            )
        pair = tuple(sorted((left, right)))
        limits[pair] = min(limits.get(pair, maximum), maximum)
    return tuple((left, right, limit) for (left, right), limit in sorted(limits.items()))

def _adjacency_compatible_bottom_exemptions(
    exemptions: Sequence[str],
    *,
    adjacency_limits_nm: Sequence[Tuple[str, str, int]],
    bands: Mapping[str, int],
) -> Tuple[str, ...]:
    """Keep rail carriers fixed only when no cross-band adjacency must move them."""

    movable = {
        endpoint
        for left, right, _maximum in adjacency_limits_nm
        if bands.get(left) != bands.get(right)
        for endpoint in (left, right)
    }
    return tuple(sorted(set(exemptions) - movable))

def _hard_placement_compatible_bottom_exemptions(
    exemptions: Sequence[str],
    *,
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    bands: Mapping[str, int],
) -> Tuple[str, ...]:
    """Choose non-overlapping bottom carriers under hard band/axis constraints."""

    retained = {str(block_id) for block_id in exemptions}
    if bands:
        bottom_band = min(int(value) for value in bands.values())
        retained = {
            block_id
            for block_id in retained
            if bands.get(block_id) == bottom_band
        }
    for group in axis_alignment_groups:
        aligned = sorted(
            str(block_id)
            for block_id, _offset2 in group
            if str(block_id) in retained
        )
        retained.difference_update(aligned[1:])
    return tuple(sorted(retained))

def _compiled_edge_rail_specs_for_placement(
    intent: Any,
    tech: Any,
    bridged_intent: Any,
) -> Tuple[Any, ...]:
    """Compile the router's exact edge-rail specs for placement replay."""

    edge_trunks = tuple(
        trunk
        for trunk in (getattr(intent, "trunks", ()) or ())
        if str(getattr(trunk, "edge", "") or "").strip()
    )
    if not edge_trunks:
        return ()

    from clay.layout.route.route_cell import (
        _compile_edge_rail_specs_from_intent,
        _routing_intent_for,
    )

    # An explicit edge width needs no derived net-width model.  For an omitted
    # width, reuse the router's one design/PDK-derived routing intent rather than
    # inventing a placement-side fallback.
    routing_intent = (
        _routing_intent_for(intent, bridged_intent, tech)
        if any(
            float(getattr(trunk, "width_um", 0.0) or 0.0) <= 0.0
            for trunk in edge_trunks
        )
        else SimpleNamespace(widths={})
    )
    return tuple(_compile_edge_rail_specs_from_intent(
        intent,
        routing_intent,
        tech,
    ))

def _compiled_edge_rail_keepouts_nm(
    intent: Any,
    tech: Any,
    bridged_intent: Any,
) -> Tuple[
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], Tuple[str, ...]],
]:
    """Return exact keepouts and owner nets keyed by ``(edge, axis)``.

    Placement and routing consume one compiled edge-rail obligation.  The compact
    materializer reserves its exact compiled width plus the same-layer spacing rule
    before routing starts.  No circuit, net, or PDK name participates.
    """

    from clay.layout.route.rail_plan import _required_metal_rule

    demands = _compiled_edge_rail_specs_for_placement(intent, tech, bridged_intent)
    keepouts: Dict[Tuple[str, str], int] = {}
    nets: Dict[Tuple[str, str], set[str]] = {}
    for demand in demands:
        key = (str(demand.edge), str(demand.actual_axis))
        keepouts[key] = max(
            keepouts.get(key, 0),
            int(demand.width_nm) + int(demand.offset_nm)
            + int(_required_metal_rule(tech, demand.layer_n, "min_space")),
        )
        nets.setdefault(key, set()).add(str(demand.net))
    return keepouts, {
        key: tuple(sorted(values)) for key, values in sorted(nets.items())
    }

def _edge_keepout_clear_blocks(
    intent: Any,
    tech: Any,
    bridged_intent: Any,
    *,
    edge: str,
    geometry: Mapping[str, Any],
    block_stamps: Mapping[str, Sequence[str]],
    block_gring: Mapping[str, str],
    master_members: Mapping[str, Sequence[str]],
    bands: Mapping[str, int],
) -> Tuple[str, ...]:
    """Return masters proven disjoint from one horizontal edge rail's layers.

    A rail on metal ``n`` may overlap a rigid master only when that master's exact
    pre-route artwork contains neither metal ``n`` nor either via cut incident on
    it.  Center-band masters retain the compiled bottom keepout independently, so
    only the bottom-edge proof excludes them.  Missing layer/stamp/member evidence
    excludes only that master; malformed rail-layer authority fails the entire
    proof closed.
    """

    if type(edge) is not str or edge not in {"bottom", "top"}:
        raise ValueError("edge rail layer-clear proof requires 'bottom' or 'top'")

    specs = tuple(
        spec
        for spec in _compiled_edge_rail_specs_for_placement(
            intent, tech, bridged_intent
        )
        if str(getattr(spec, "edge", "")) == edge
        and str(getattr(spec, "actual_axis", "")) == "H"
    )
    if not specs or not master_members:
        return ()

    def exact_layer_key(raw: Any) -> Tuple[int, int]:
        if (
            type(raw) is not tuple
            or len(raw) != 2
            or any(type(value) is not int for value in raw)
        ):
            raise ValueError("layer key must be an exact (int, int) tuple")
        return raw

    blocked_layers: set[Tuple[int, int]] = set()
    for spec in specs:
        try:
            layer_n = getattr(spec, "layer_n")
            if type(layer_n) is not int:
                raise ValueError("rail layer number must be an exact int")
            declared_layer_key = exact_layer_key(spec.layer_key)
            canonical_layer_key = exact_layer_key(tech.metal_key(layer_n))
            if declared_layer_key != canonical_layer_key:
                raise ValueError("rail layer key disagrees with PDK metal authority")
            blocked_layers.add(canonical_layer_key)
        except (AttributeError, KeyError, TypeError, ValueError):
            return ()
        for via_n in (layer_n - 1, layer_n):
            if via_n < 0:
                continue
            try:
                blocked_layers.add(exact_layer_key(tech.via_key(via_n)))
            except (AttributeError, KeyError, TypeError, ValueError):
                return ()

    clear: List[str] = []
    for master, raw_members in sorted(master_members.items()):
        if master not in bands or (
            edge == "bottom" and bands.get(master) == _BAND_CENTER
        ):
            continue
        members = tuple(str(member) for member in raw_members)
        if not members:
            continue
        master_clear = True
        stamp_names: set[str] = set()
        for member in members:
            stamps = tuple(str(stamp) for stamp in block_stamps.get(member, ()))
            if not stamps:
                master_clear = False
                break
            stamp_names.update(stamps)
            ring_stamp = block_gring.get(member)
            if ring_stamp is not None:
                stamp_names.add(str(ring_stamp))
        if not master_clear or not stamp_names:
            continue
        for stamp_name in sorted(stamp_names):
            stamp = geometry.get(stamp_name)
            layers = getattr(stamp, "polys_by_layer", None)
            if stamp is None or not isinstance(layers, Mapping):
                master_clear = False
                break
            try:
                present_layers = {
                    exact_layer_key(layer)
                    for layer, polygons in layers.items()
                    if polygons
                }
            except (TypeError, ValueError):
                master_clear = False
                break
            if present_layers & blocked_layers:
                master_clear = False
                break
        if master_clear:
            clear.append(str(master))
    return tuple(clear)

def _bottom_edge_keepout_clear_blocks(
    intent: Any,
    tech: Any,
    bridged_intent: Any,
    **kwargs: Any,
) -> Tuple[str, ...]:
    """Backward-compatible bottom-edge specialization of the layer proof."""

    return _edge_keepout_clear_blocks(
        intent,
        tech,
        bridged_intent,
        edge="bottom",
        **kwargs,
    )

def _top_edge_keepout_clear_blocks(
    intent: Any,
    tech: Any,
    bridged_intent: Any,
    **kwargs: Any,
) -> Tuple[str, ...]:
    """Return exact masters whose artwork cannot obstruct the top edge rail."""

    return _edge_keepout_clear_blocks(
        intent,
        tech,
        bridged_intent,
        edge="top",
        **kwargs,
    )

def replay_member_ring_padding(
    *,
    blocks: Sequence[Tuple[str, _Bbox]],
    block_stamps: Dict[str, set],
    block_member_devices: Dict[str, set],
    bands: Dict[str, int],
    left_column: Optional[str],
    guard_rings: Sequence[Any],
    guard_ring_outer_clearances_nm: Sequence[int],
    top_edge_keepout_nm: int,
    top_edge_rail_nets: Sequence[str],
    bottom_edge_keepout_nm: int,
    bottom_edge_rail_nets: Sequence[str],
) -> Tuple[Dict[str, int], Dict[str, int], set[str]]:
    """Replay the canonical ring-padding facts from one unpadded block model.

    The compact materializer and exact-GDS checker share this pure calculation.
    The checker still rebuilds ``blocks`` independently from canonical source
    members and PDK-minimum rings; caller-supplied padding never enters here.
    """

    rings = tuple(guard_rings)
    clearances = tuple(guard_ring_outer_clearances_nm)
    if len(clearances) != len(rings):
        raise ValueError("guard-ring outer clearances must match declarations")
    if any(type(value) is not int or value < 0 for value in clearances):
        raise TypeError("guard-ring outer clearances must be non-negative integers")

    bbox_by_id = dict(blocks)
    ring_by_block_id = {
        f"ISL{index}": (ring, clearances[index])
        for index, ring in enumerate(rings)
    }
    top_nets = {str(net) for net in top_edge_rail_nets}
    top_ids = tuple(
        block_id
        for block_id, _bbox in blocks
        if bands.get(block_id) == _BAND_TOP and block_id != left_column
    )
    top_padding: Dict[str, int] = {}
    if top_ids:
        for block_id in top_ids:
            rail_reserve_nm = int(top_edge_keepout_nm)
            ring_and_clearance = ring_by_block_id.get(block_id)
            tie_net = (
                str(getattr(ring_and_clearance[0], "tie_net", "") or "")
                if ring_and_clearance
                else ""
            )
            if (
                ring_and_clearance is not None
                and top_nets == {tie_net}
            ):
                # The final rail candidate is replayed against exact same-layer
                # conductor ownership.  A sole same-net ring can therefore be
                # the intrinsic boundary carrier: foreign M2 removes the edge
                # candidate and fails closed, while same-net contact is legal.
                rail_reserve_nm = 0
            pad_nm = rail_reserve_nm
            if pad_nm <= 0:
                continue
            for name in block_stamps.get(block_id, ()):
                top_padding[str(name)] = pad_nm
            for name in block_member_devices.get(block_id, ()):
                top_padding[str(name)] = pad_nm

    bottom_padding: Dict[str, int] = {}
    padded_bottom_blocks: set[str] = set()
    bottom_nets = set(str(net) for net in bottom_edge_rail_nets)
    if blocks and int(bottom_edge_keepout_nm) > 0 and bottom_nets:
        floor_y = min(bbox[1] for _block_id, bbox in blocks)
        for index, ring in enumerate(rings):
            block_id = f"ISL{index}"
            bbox = bbox_by_id.get(block_id)
            tie_net = str(getattr(ring, "tie_net", "") or "")
            if (
                bbox is None
                or bbox[1] != floor_y
                or tie_net not in bottom_nets
            ):
                continue
            rail_reserve_nm = int(bottom_edge_keepout_nm)
            if bottom_nets == {tie_net}:
                # As on the top edge, exact rail planning owns the conductor
                # collision proof; do not manufacture an empty ring cavity.
                rail_reserve_nm = 0
            if rail_reserve_nm > 0:
                for name in block_stamps.get(block_id, ()):
                    bottom_padding[str(name)] = rail_reserve_nm
                for name in block_member_devices.get(block_id, ()):
                    bottom_padding[str(name)] = rail_reserve_nm
            padded_bottom_blocks.add(block_id)
    return top_padding, bottom_padding, padded_bottom_blocks

def _exact_member_ring_padding_rows(value: Any, label: str) -> Dict[str, int]:
    """Freeze one signed padding inventory without mutable/coercive aliases."""

    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    if any(
        type(row) is not tuple
        or len(row) != 2
        or type(row[0]) is not str
        or not row[0]
        or type(row[1]) is not int
        or row[1] < 0
        for row in value
    ):
        raise TypeError(f"{label} must contain exact pairs")
    if tuple(sorted(value)) != value or len(dict(value)) != len(value):
        raise ValueError(f"{label} must be sorted and unique")
    return dict(value)

def _signed_member_ring_padding_names(value: Any, label: str) -> set[str]:
    """Return members carried by an exact padding inventory, including zero."""

    return set(_exact_member_ring_padding_rows(value, label))

def _replay_ring_padding_on_positioned_master(
    *,
    materialized_blocks: Sequence[Tuple[str, _Bbox]],
    master_blocks: Sequence[Tuple[str, _Bbox]],
    master_members: Mapping[str, Sequence[str]],
    positions: Mapping[str, Tuple[int, int]],
    ring_block_ids: set[str],
    block_stamps: Mapping[str, Sequence[str]],
    block_member_devices: Mapping[str, Sequence[str]],
    prior_top_padding: Mapping[str, int],
    prior_bottom_padding: Mapping[str, int],
    bands: Mapping[str, int],
    left_column: Optional[str],
    guard_rings: Sequence[Any],
    guard_ring_outer_clearances_nm: Sequence[int],
    top_edge_keepout_nm: int,
    top_edge_rail_nets: Sequence[str],
    bottom_edge_keepout_nm: int,
    bottom_edge_rail_nets: Sequence[str],
) -> Tuple[Dict[str, int], Dict[str, int], set[str]]:
    """Deconvolve old ring padding, then measure the solved exact master."""

    materialized = dict(materialized_blocks)
    masters = dict(master_blocks)
    positioned_raw_masters = []
    master_stamps: Dict[str, set[str]] = {}
    master_devices: Dict[str, set[str]] = {}
    for master, members in sorted(master_members.items()):
        if master not in masters or master not in positions or not members:
            raise ValueError("ring-padding master partition is incomplete")
        old_master = masters[master]
        dx = positions[master][0] - old_master[0]
        dy = positions[master][1] - old_master[1]
        raw_members = []
        stamps: set[str] = set()
        devices: set[str] = set()
        for member in members:
            if member not in materialized:
                raise ValueError("ring-padding master member is missing")
            x0, y0, x1, y1 = materialized[member]
            member_stamps = {
                str(name) for name in block_stamps.get(member, ())
            }
            member_devices = {
                str(name) for name in block_member_devices.get(member, ())
            }
            stamps.update(member_stamps)
            devices.update(member_devices)
            if member in ring_block_ids:
                owned_names = member_stamps | member_devices
                y0 += max(
                    (prior_bottom_padding.get(name, 0) for name in owned_names),
                    default=0,
                )
                y1 -= max(
                    (prior_top_padding.get(name, 0) for name in owned_names),
                    default=0,
                )
            if y1 <= y0:
                raise ValueError(
                    f"feedback ring padding consumes block {member!r}"
                )
            raw_members.append((x0 + dx, y0 + dy, x1 + dx, y1 + dy))
        master_stamps[master] = stamps
        master_devices[master] = devices
        positioned_raw_masters.append(
            (
                master,
                (
                    min(row[0] for row in raw_members),
                    min(row[1] for row in raw_members),
                    max(row[2] for row in raw_members),
                    max(row[3] for row in raw_members),
                ),
            )
        )
    return replay_member_ring_padding(
        blocks=tuple(positioned_raw_masters),
        block_stamps=master_stamps,
        block_member_devices=master_devices,
        bands=bands,
        left_column=left_column,
        guard_rings=guard_rings,
        guard_ring_outer_clearances_nm=guard_ring_outer_clearances_nm,
        top_edge_keepout_nm=top_edge_keepout_nm,
        top_edge_rail_nets=top_edge_rail_nets,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        bottom_edge_rail_nets=bottom_edge_rail_nets,
    )

def replay_canonical_member_ring_padding(
    *,
    source_cell: Any,
    intent: Any,
    geometry: Dict[str, Any],
    placement_um: Dict[str, Tuple[float, float]],
    tech: Any,
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]]:
    """Independently rebuild the unpadded source block model and its pad rows."""

    from copy import copy

    from clay.layout.compact_blocks import extract_blocks
    from clay.layout.route.route_cell import (
        _BridgedIntent,
        _materialize_guard_rings,
        bridge_bjts_from_cell_def,
        bridge_devices_from_cell_def,
    )
    from clay.layout.tech import LayoutTech
    from clay.port_roles import resolve_positive_supply_net, resolve_substrate_net
    from clay.well_isolation import declared_well_isolated_devices

    resolved_tech = tech if isinstance(tech, LayoutTech) else LayoutTech.resolve(str(tech))
    devices, passives = bridge_devices_from_cell_def(source_cell)
    bjts = bridge_bjts_from_cell_def(source_cell)
    bridged = _BridgedIntent(
        devices,
        passives,
        bjts,
        well_isolated_devices=declared_well_isolated_devices(source_cell),
        substrate_net=resolve_substrate_net(
            source_cell, fallback=resolved_tech.substrate_net()
        ),
        deep_nwell_net=resolve_positive_supply_net(source_cell, fallback=""),
    )
    canonical_names = {
        str(device.name)
        for device in (*tuple(devices), *tuple(passives), *tuple(bjts))
    }
    base_geometry: Dict[str, Any] = {}
    base_placement: Dict[str, Tuple[float, float]] = {}
    for raw_name, stamp in geometry.items():
        name = str(raw_name)
        members = tuple(
            str(member)
            for member in (getattr(stamp, "composite_members", ()) or ())
        )
        if members and not set(members) <= canonical_names:
            raise ValueError(
                f"emitter composite {name!r} has foreign source members"
            )
        if name not in canonical_names and not members:
            continue
        if name not in placement_um:
            raise ValueError(f"emitter source stamp {name!r} lacks placement")
        canonical_stamp = copy(stamp)
        if hasattr(canonical_stamp, "source"):
            setattr(canonical_stamp, "source", "canonical_source_member")
        base_geometry[name] = canonical_stamp
        base_placement[name] = placement_um[name]

    declarations = tuple(getattr(intent, "guard_rings", ()) or ())
    from clay.layout.extract.realize_render import _ring_extension_nm

    ring_outer_clearances_nm = tuple(
        _ring_extension_nm(resolved_tech, str(declaration.well))
        for declaration in declarations
    )
    if declarations:
        _materialize_guard_rings(
            declarations,
            bridged,
            base_placement,
            base_geometry,
            resolved_tech,
            str(getattr(resolved_tech, "pdk_id", "") or resolved_tech),
        )
    blocks, block_stamps, _block_rings, _ring_names = extract_blocks(
        base_geometry, base_placement, intent
    )
    block_members = _resolve_block_member_devices(block_stamps, base_geometry)
    device_kind = {
        str(device.name): str(device.kind)
        for device in (*tuple(devices), *tuple(passives))
    }
    for bjt in tuple(bjts):
        device_kind[str(bjt.name)] = str(getattr(bjt, "kind", "bjt") or "bjt")
    device_nets = {
        str(device.name): dict(device.nets)
        for device in (*tuple(devices), *tuple(passives))
    }
    block_ids = tuple(block_id for block_id, _bbox in blocks)
    bands = assign_bands(
        block_ids,
        block_members,
        device_kind,
        device_nets=device_nets,
        place_groups=tuple(getattr(intent, "place_groups", ()) or ()),
        rail_affinities=tuple(getattr(intent, "rail_affinities", ()) or ()),
        band_intents=tuple(getattr(intent, "bands", ()) or ()),
    )
    common_centroid_members = common_centroid_devices(intent, base_geometry)
    source_file = getattr(source_cell, "_clay_file", None)
    declared_left_group = ""
    if source_file is not None:
        from clay.layout.policy import policy_for_cell

        declared_left_group = str(
            policy_for_cell(source_file, source_cell).left_column_group or ""
        )
    left_column = identify_left_column(
        block_ids,
        block_members,
        common_centroid_members,
        declared_left_groups=(declared_left_group,) if declared_left_group else (),
        require_unique=bool(getattr(intent, "channels", ()) or ()),
    )
    edge_keepouts, edge_nets = _compiled_edge_rail_keepouts_nm(
        intent, resolved_tech, bridged
    )
    top, bottom, _bottom_blocks = replay_member_ring_padding(
        blocks=blocks,
        block_stamps=block_stamps,
        block_member_devices=block_members,
        bands=bands,
        left_column=left_column,
        guard_rings=declarations,
        guard_ring_outer_clearances_nm=ring_outer_clearances_nm,
        top_edge_keepout_nm=edge_keepouts.get(("top", "H"), 0),
        top_edge_rail_nets=edge_nets.get(("top", "H"), ()),
        bottom_edge_keepout_nm=edge_keepouts.get(("bottom", "H"), 0),
        bottom_edge_rail_nets=edge_nets.get(("bottom", "H"), ()),
    )
    return tuple(sorted(top.items())), tuple(sorted(bottom.items()))

def bundle_passive_composites(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Dict[str, set],
    device_kind: Dict[str, str],
    place_groups: Sequence[Any],
    gap_nm: int,
    island_block_ids: set,
    order_constraints: Sequence[Tuple[str, str, str]] = (),
    pair_clearances_nm: Any = None,
    matching_pair_axes: Mapping[
        Tuple[str, str], Tuple[str, float]
    ] | None = None,
    partitioned_passive_groups: Any = (),
    edge_alignments: Sequence[Any] = (),
) -> Tuple[
    List[Tuple[str, _Bbox]],
    Dict[str, List[str]],
    Dict[str, Tuple[str, Tuple[int, int]]],
]:
    """Preserve certified two-resistor atoms, then merge standalone capacitor groups.

    A resistor region normally remains freely packable.  The sole exception is an
    exact two-member resistor ``@place_group`` whose same pair is present in the
    canonical materializer's ``matching_pair_axes`` certificate.  Those two proofs
    own one rigid atom: retain the already-materialized relative offsets, so the
    compact packer can only translate the pair.  No axis is inferred from final
    geometry.

    STANDALONE capacitor blocks that share an ALL-CAP ``@place_group`` merge
    into one rigid composite block (members laid side-by-side, in declared
    ``@order(..., left_of)`` order with deterministic lexical tie-breaking, sharing
    ``gap_nm``; composite bbox = their union). GENERIC -- keys on "the place_group's
    devices are all capacitors" + "the block is standalone (not a guard-ring
    island)", never a cap name.  A resistor region remains a collection of movable
    blocks for the global 2-D packer; treating it as one horizontal macro creates
    width proportional to resistor count and erases the declared floorplan region.
    An already-islanded passive is also left untouched.

    Returns ``(new_blocks, composite_members, member_offsets)``:
      * ``new_blocks``        -- ``blocks`` with each merged member removed and one
        ``PGRP_<group>`` composite ``(id, bbox)`` appended.
      * ``composite_members`` -- ``{composite_id -> [member block ids, placed order]}``.
      * ``member_offsets``    -- ``{member block id -> (composite_id, (dx, dy))}`` the
        member's origin offset inside the composite bbox.
    """
    if type(partitioned_passive_groups) is not tuple:
        raise TypeError("partitioned passive groups must be an exact tuple")
    canonical_partition_groups: list[frozenset[str]] = []
    partition_members: set[str] = set()
    for raw_group in partitioned_passive_groups:
        if type(raw_group) is not tuple or not raw_group:
            raise TypeError(
                "partitioned passive groups must contain non-empty exact tuples"
            )
        if any(type(member) is not str or not member for member in raw_group):
            raise TypeError(
                "partitioned passive group members must be non-empty exact strings"
            )
        group = frozenset(raw_group)
        if len(group) != len(raw_group):
            raise ValueError("partitioned passive group members must be unique")
        overlap = partition_members.intersection(group)
        if overlap:
            raise ValueError(
                "partitioned passive groups reuse member "
                f"{min(overlap)!r}"
            )
        partition_members.update(group)
        canonical_partition_groups.append(group)

    gap = int(gap_nm)
    bb_of = dict(blocks)
    pair_clearances = _canonical_pair_clearances_nm(
        pair_clearances_nm,
        block_ids=tuple(bb_of),
        base_gap_nm=gap,
    )
    merged: set = set()
    composites: List[Tuple[str, _Bbox, List[str], Dict[str, Tuple[int, int]]]] = []
    certified_pairs: set[frozenset[str]] = set()
    for raw_pair, raw_axis in (matching_pair_axes or {}).items():
        if (
            type(raw_pair) is tuple
            and len(raw_pair) == 2
            and raw_pair[0] != raw_pair[1]
            and type(raw_axis) is tuple
            and len(raw_axis) == 2
            and raw_axis[0] in {"x", "y"}
        ):
            try:
                coordinate = float(raw_axis[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(coordinate):
                certified_pairs.add(frozenset(str(member) for member in raw_pair))

    for pg in place_groups or ():
        declared_members = tuple(str(member) for member in (
            getattr(pg, "devices", ()) or ()
        ))
        member_set = frozenset(declared_members)
        if (
            len(declared_members) != 2
            or len(member_set) != 2
            or member_set not in certified_pairs
            or any(device_kind.get(member) != "res" for member in member_set)
        ):
            continue
        owners: List[str] = []
        for member in declared_members:
            hits = [
                block_id
                for block_id, _bbox in blocks
                if block_member_devices.get(block_id, set()) == {member}
            ]
            if len(hits) != 1:
                owners = []
                break
            owners.append(hits[0])
        if (
            len(owners) != 2
            or owners[0] == owners[1]
            or any(owner in island_block_ids or owner in merged for owner in owners)
        ):
            continue
        x0 = min(bb_of[owner][0] for owner in owners)
        y0 = min(bb_of[owner][1] for owner in owners)
        x1 = max(bb_of[owner][2] for owner in owners)
        y1 = max(bb_of[owner][3] for owner in owners)
        offsets = {
            owner: (bb_of[owner][0] - x0, bb_of[owner][1] - y0)
            for owner in owners
        }
        group = str(getattr(pg, "group", ""))
        master = f"MPAIR_{group}"
        if not group or master in bb_of:
            raise ValueError("certified resistor pair has an invalid master identity")
        composites.append((master, (0, 0, x1 - x0, y1 - y0), owners, offsets))
        merged.update(owners)

    aligned_pairs: set[frozenset[str]] = set()
    for decl in edge_alignments or ():
        first = str(getattr(decl, "first", "") or "")
        second = str(getattr(decl, "second", "") or "")
        if first and second and first != second:
            aligned_pairs.add(frozenset((first, second)))

    for pg in place_groups or ():
        gdevs = set(getattr(pg, "devices", ()) or ())
        if not gdevs or not all(device_kind.get(d) == "cap" for d in gdevs):
            continue  # resistor regions are globally packed, not rigidly bundled
        if getattr(pg, "packing", None) in {"deformable", "relational"}:
            continue
        if any(pair <= gdevs for pair in aligned_pairs):
            continue  # concentrically or edge-aligned passives are bundled by edge alignment
        owned_partition_groups = tuple(
            group for group in canonical_partition_groups if group <= gdevs
        )
        if (
            len(owned_partition_groups) >= 2
            and set().union(*owned_partition_groups) == gdevs
        ):
            continue
        gname = str(getattr(pg, "group", ""))
        elig = []
        for bid, _bb in blocks:
            if bid in island_block_ids or bid in merged:
                continue
            mdev = block_member_devices.get(bid, set())
            if (
                mdev
                and mdev <= gdevs
                and all(device_kind.get(d) == "cap" for d in mdev)
            ):
                elig.append(bid)
        if len(elig) < 2:
            continue  # nothing to bundle (0 or 1 standalone member)
        member_to_block = {
            member: bid for bid in elig for member in block_member_devices.get(bid, ())
        }
        internal_order_pairs = []
        for left, right, relation in order_constraints or ():
            if str(relation) != "left_of":
                continue
            left_block = member_to_block.get(str(left))
            right_block = member_to_block.get(str(right))
            if (
                left_block is not None
                and right_block is not None
                and left_block != right_block
            ):
                internal_order_pairs.append((left_block, right_block))
        elig = _stable_hard_order(elig, internal_order_pairs)
        cur = 0
        hmax = 0
        offsets: Dict[str, Tuple[int, int]] = {}
        prior: Optional[str] = None
        for bid in elig:
            bb = bb_of[bid]
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            if prior is not None:
                cur += int(
                    pair_clearances.get(tuple(sorted((prior, bid))), gap)
                )
            offsets[bid] = (cur, 0)
            cur += w
            hmax = max(hmax, h)
            prior = bid
        width = cur
        composites.append((f"PGRP_{gname}", (0, 0, width, hmax), elig, offsets))
        merged.update(elig)

    new_blocks: List[Tuple[str, _Bbox]] = [
        (bid, bb) for bid, bb in blocks if bid not in merged
    ]
    composite_members: Dict[str, List[str]] = {}
    member_offsets: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    for comp_id, bb, parts, offsets in composites:
        new_blocks.append((comp_id, bb))
        composite_members[comp_id] = parts
        for bid, off in offsets.items():
            member_offsets[bid] = (comp_id, off)
    return new_blocks, composite_members, member_offsets
