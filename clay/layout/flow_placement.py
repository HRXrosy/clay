"""Declared structure-aware placement preparation for the canonical flow.
This module owns the one generic lowering from declared band/anchor/channel
intent to the route-time compact placement.  A selected place policy supplies
optimization targets; it is not the activation switch.  The reference runner
may report the result, but it must not be the semantic owner of this decision.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple
from weakref import ref as _weakref
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.place.collision_geometry_session import collision_query_scope
from clay.layout.place.layered_placement_boundary import restore_layered_compaction_inputs, final_layered_compaction_view
from clay.layout.place.layered_placement_boundary import compaction_source_groups, InvalidCompactionGroupOwnership
from clay.flow.execution_budget import ExecutionStopped
from clay.layout.placement_infeasibility import (
    PlacementInfeasibilityError,
    SufficientCoreDiagnostic,
)

if TYPE_CHECKING:
    from clay.layout.generic_placement import Placement
    from clay.layout.intent_basis import IntentBasis
    from clay.layout.place_blocks_2d import FinalPlacementCertificate


class DeclaredCompactionError(RuntimeError):
    """The declared structure-aware compact placement could not be materialized."""

    def __init__(
        self,
        message: str,
        *,
        sufficient_core: SufficientCoreDiagnostic | None = None,
        diagnostic_preservation: Mapping[str, object] | None = None,
    ) -> None:
        self.sufficient_core = sufficient_core
        self.diagnostic_preservation = diagnostic_preservation
        super().__init__(message)


class UnsupportedCraftCompactionError(DeclaredCompactionError):
    """A declared craft variant has no implementation in the canonical lowering."""


def _declared_compaction_error(exc: Exception) -> DeclaredCompactionError:
    """Wrap one implementation failure while preserving typed diagnostics."""

    return DeclaredCompactionError(
        f"declared craft compaction failed: {type(exc).__name__}: {exc}",
        diagnostic_preservation=getattr(exc, "diagnostic_preservation", None),
    )


def _canonical_compaction_policy(cell_def: Any, supplied_policy: Any) -> Any:
    """Replay a parsed cell's source policy; constructed test IR uses its input."""

    source_file = getattr(cell_def, "_clay_file", None)
    if source_file is None:
        return supplied_policy
    from clay.layout.policy import policy_for_cell

    return policy_for_cell(source_file, cell_def)


def _exact_craft_declarations(intent: Any, field: str) -> tuple:
    values = getattr(intent, field, ())
    if type(values) is not tuple:
        raise UnsupportedCraftCompactionError(
            f"canonical craft compaction requires {field} as an exact tuple"
        )
    return values


def _declared_minimum_structural_occupancy(intent: Any) -> Optional[float]:
    densities = _exact_craft_declarations(intent, "densities")
    if len(densities) > 1:
        raise UnsupportedCraftCompactionError(
            "canonical craft compaction supports exactly one @density declaration"
        )
    if not densities:
        return None
    from clay.layout.intent import replay_density_intent

    try:
        density = replay_density_intent(densities[0])
    except (TypeError, ValueError) as exc:
        raise UnsupportedCraftCompactionError(
            f"canonical craft compaction has invalid @density: {exc}"
        ) from exc
    return 1.0 - density.max_whitespace_fraction


def _declared_minimum_functional_od_occupancy(intent: Any) -> Optional[float]:
    declarations = _exact_craft_declarations(intent, "od_fills")
    if len(declarations) > 1:
        raise UnsupportedCraftCompactionError(
            "canonical craft compaction supports exactly one @od_fill declaration"
        )
    if not declarations:
        return None
    from clay.layout.intent import replay_od_fill_intent

    try:
        return replay_od_fill_intent(declarations[0]).min_fraction
    except (TypeError, ValueError) as exc:
        raise UnsupportedCraftCompactionError(
            f"canonical craft compaction has invalid @od_fill: {exc}"
        ) from exc


def _bind_functional_od_placement_certificate(
    intent: Any,
    certificate: object,
    *,
    functional_od_area_nm2: int,
    active_layer_keys: tuple[tuple[int, int], ...],
) -> object:
    """Attach the functional-active materializer inputs to one certificate."""

    from clay.layout.place.final_placement_state import functional_od_policy_binding

    target = _declared_minimum_functional_od_occupancy(intent)
    target, area, layers, digest = functional_od_policy_binding(
        target,
        functional_od_area_nm2 if target is not None else 0,
        active_layer_keys if target is not None else (),
    )
    return replace(
        certificate,
        declared_minimum_functional_od_occupancy=target,
        functional_od_area_nm2=area,
        functional_od_layer_keys=layers,
        functional_od_policy_sha256=digest,
    )


def _bind_group_route_placement_certificate(
    cell_def: Any,
    intent: Any,
    certificate: object,
    *,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
) -> object:
    """Bind declared route-group topology to exact final member geometry."""

    raw_declarations = intent.route_groups
    if type(raw_declarations) is not tuple:
        raise TypeError("route-group placement boundary requires an exact tuple")
    if not raw_declarations:
        return certificate
    from clay.layout.route.group_ports import (
        bind_group_route_placement,
        build_group_route_port_inventory,
        verify_group_route_placement,
    )

    inventory = build_group_route_port_inventory(
        cell_def,
        intent,
        fact_set=intent_fact_set,
        source_inventory=source_inventory,
    )
    placed = bind_group_route_placement(inventory, certificate)
    signed = replace(
        certificate,
        group_route_placement_certificate=placed,
    )
    verify_group_route_placement(inventory, signed, placed)
    return signed


def _require_od_fill_bound_placement_certificate(
    intent: Any,
    certificate: object,
    *,
    functional_od_area_nm2: int | None = None,
    active_layer_keys: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Replay canonical functional OD intent before placement enters routing."""

    expected = _declared_minimum_functional_od_occupancy(intent)
    observed = getattr(
        certificate, "declared_minimum_functional_od_occupancy", None
    )
    if observed != expected:
        raise DeclaredCompactionError(
            "final placement differs from canonical @od_fill intent: "
            f"expected={expected!r} observed={observed!r}"
        )
    if expected is None:
        return
    if functional_od_area_nm2 is None or active_layer_keys is None:
        raise DeclaredCompactionError(
            "canonical @od_fill route boundary lacks replayed artwork evidence"
        )
    from clay.layout.place.final_placement_state import functional_od_policy_binding

    _target, replayed_area, replayed_layers, _digest = functional_od_policy_binding(
        expected,
        functional_od_area_nm2,
        active_layer_keys,
    )
    if (
        getattr(certificate, "functional_od_area_nm2", None) != replayed_area
        or getattr(certificate, "functional_od_layer_keys", None)
        != replayed_layers
    ):
        raise DeclaredCompactionError(
            "final placement differs from replayed canonical @od_fill artwork"
        )
    from clay.layout.place_blocks_2d import placement_certificate_is_signable

    if not placement_certificate_is_signable(certificate):
        raise DeclaredCompactionError(
            "final placement carrying canonical @od_fill is not signable"
        )


def _functional_od_artwork_nm2(
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Any],
    tech: Any,
    source_mos_names: tuple[str, ...],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Measure source-owned active artwork used as materializer input.

    This is not final evidence.  It binds the compact solver's causal input;
    the exact checker independently rereads and uniquely claims the emitted GDS.
    """

    import klayout.db as kdb

    from clay.layout.placed_geometry import placed_point_nm

    from clay.layout.completeness.gds_witness import functional_active_layer_keys

    try:
        keys = functional_active_layer_keys(tech)
    except (TypeError, ValueError) as exc:
        raise UnsupportedCraftCompactionError(
            "installed PDK exposes no functional active/OD layer"
        ) from exc
    owners = set()
    for member in source_mos_names:
        aggregate_owners = tuple(
            sorted(
                str(name)
                for name, stamp in geometry.items()
                if str(name) != member
                and member in tuple(getattr(stamp, "composite_members", ()) or ())
            )
        )
        if member in geometry:
            if aggregate_owners:
                raise DeclaredCompactionError(
                    f"functional OD member {member!r} has duplicate emitter owners"
                )
            owners.add(member)
        elif len(aggregate_owners) == 1:
            owners.add(aggregate_owners[0])
        else:
            detail = "no" if not aggregate_owners else "ambiguous"
            raise DeclaredCompactionError(
                f"functional OD member {member!r} has {detail} emitter owner"
            )
    if not owners:
        raise DeclaredCompactionError("functional OD source MOS inventory is empty")
    region = kdb.Region()
    for owner in sorted(owners):
        stamp = geometry.get(owner)
        position = placement_um.get(owner)
        if stamp is None or position is None:
            raise DeclaredCompactionError(
                f"functional OD owner {owner!r} lacks geometry or placement"
            )
        owner_has_active = False
        for key in keys:
            for ring in tuple(
                (getattr(stamp, "polys_by_layer", {}) or {}).get(key, ()) or ()
            ):
                points = [
                    kdb.Point(*placed_point_nm(float(x), float(y), tuple(position)))
                    for x, y in tuple(ring)
                ]
                if len(points) < 3:
                    raise DeclaredCompactionError(
                        f"functional OD owner {owner!r} has degenerate active artwork"
                    )
                region.insert(kdb.Polygon(points))
                owner_has_active = True
        if not owner_has_active:
            raise DeclaredCompactionError(
                f"functional OD owner {owner!r} has no installed active artwork"
            )
    region.merge()
    area_nm2 = int(region.area())
    if area_nm2 <= 0:
        raise DeclaredCompactionError("functional OD artwork has zero area")
    return area_nm2, keys


def _require_density_bound_placement_certificate(
    intent: Any,
    certificate: object,
) -> None:
    """Replay canonical density intent before a placement enters the router."""

    from clay.layout.place.final_placement_state import (
        canonical_minimum_structural_occupancy,
    )
    from clay.layout.place_blocks_2d import placement_certificate_is_signable

    expected = _declared_minimum_structural_occupancy(intent)
    try:
        observed = canonical_minimum_structural_occupancy(
            getattr(certificate, "declared_minimum_structural_occupancy", None)
        )
    except (TypeError, ValueError) as exc:
        raise DeclaredCompactionError(
            f"final placement has invalid canonical @density evidence: {exc}"
        ) from exc
    if observed != expected:
        raise DeclaredCompactionError(
            "final placement differs from canonical @density intent: "
            f"expected={expected!r} observed={observed!r}"
        )
    if expected is None:
        return
    if not placement_certificate_is_signable(certificate):
        raise DeclaredCompactionError(
            "final placement carrying canonical @density is not signable"
        )


@dataclass(frozen=True)
class DeclaredCompactOverride:
    """Exact route override derived from policy and declared placement structure."""

    override: Tuple[Dict[str, Any], Dict[str, Any]]
    ring_inclusive: Tuple[Dict[str, Any], Dict[str, Any]]
    logical_placement_basis: Placement
    width_um: float
    height_um: float
    loose_width_um: float
    loose_height_um: float
    n_rings: int
    n_declared_rings: int
    final_placement_certificate: FinalPlacementCertificate
    member_ring_pad_top_nm: Tuple[Tuple[str, int], ...] = ()
    member_ring_pad_bottom_nm: Tuple[Tuple[str, int], ...] = ()
    n_source_guard_ring_declarations: int = 0
    shared_well_guard_envelope_plans: tuple = ()
    shared_well_guard_ring_inventory: Any = None
    shared_well_guard_child_plans: tuple = ()
    shared_well_flexible_content_bboxes_nm: tuple = ()
    source_mos_names: Optional[Tuple[str, ...]] = None

    @property
    def placement_state_digest(self) -> str:
        return str(self.final_placement_certificate.placement_state_digest)


@dataclass(frozen=True)
class PreparedSharedGuardHierarchy:
    """Exact shared-ring ownership plus optional deformable solve authority."""

    physical_intent: Any
    placement_um: Dict[str, Tuple[float, float]]
    geometry: Dict[str, Any]
    envelope_plans: tuple
    ring_inventory: Any
    child_plans: tuple
    flexible_envelope_constraints: tuple = ()
    flexible_content_bboxes_nm: tuple = ()


def declared_craft_compaction_requested(effective_policy: Any, intent: Any) -> bool:
    """Return whether declarations require the structure-aware compact lowering.

    Bands, legacy membership/range groups, explicit column floorplans, edge
    anchors, channels, power planes, and route groups require the final-master lowering. Exact
    row-only groups remain authoritative in the topology-aware initial placer;
    replaying those rows here would discard that placer's connectivity objective.
    A selected policy supplies optimization targets and never decides whether
    declared structure exists. Rail affinities keep their independent lowerer.
    """

    del effective_policy  # decision is declaration-driven; kept for call symmetry
    from clay.layout.intent import place_group_floorplan_row

    place_groups = _exact_craft_declarations(intent, "place_groups")
    has_final_master_place_group = any(
        place_group_floorplan_row(group) is None
        or getattr(group, "col", None) is not None
        for group in place_groups
    )
    declarations = tuple(
        _exact_craft_declarations(intent, field)
        for field in ("bands", "anchors", "channels")
    )
    return (
        any(bool(values) for values in declarations)
        or has_final_master_place_group
        or bool(_exact_craft_declarations(intent, "route_groups"))
        or bool(_exact_craft_declarations(intent, "group_center_alignments"))
        # A plane's actual footprint is the final placement envelope. Its GDS
        # observer requires that same canonical final-placement certificate,
        # including when no band/anchor/place_group happens to activate it.
        or bool(_exact_craft_declarations(intent, "power_planes"))
        # Edge rails can pad a declared guard. The exact guard observer must
        # replay that padding from the canonical final placement certificate.
        or (
            bool(_exact_craft_declarations(intent, "guard_rings"))
            and any(
                str(getattr(trunk, "edge", "") or "").strip()
                for trunk in _exact_craft_declarations(intent, "trunks")
            )
        )
    ) or (
        _declared_minimum_structural_occupancy(intent) is not None
    ) or (
        _declared_minimum_functional_od_occupancy(intent) is not None
    )


def declared_compact_aspect_bounds(
    cell_def: object,
) -> Optional[Tuple[float, float]]:
    """Return the author-declared compact search range, when optimization is live."""

    data = dict(getattr(cell_def, "sizing_kwargs", {}) or {})
    budget = data.get("opt_layout_budget")
    if (
        not isinstance(budget, (int, float))
        or isinstance(budget, bool)
        or float(budget) <= 0.0
    ):
        return None
    from clay.layout.compact_intent_domain import compact_intent_domain_spec

    domain = compact_intent_domain_spec(cell_def)
    return (float(domain.aspect_bounds[0]), float(domain.aspect_bounds[1]))


def certify_final_macro_consumption(
    graph: Any,
    candidate: Any,
    intent: Any,
    route_result: Any,
) -> Any:
    """Certify graph partitions against the actual shared-router placement."""

    from clay.layout.macro_consumption import certify_macro_consumption

    return certify_macro_consumption(
        graph,
        candidate,
        intent,
        getattr(route_result, "placement_um", {}) or {},
        getattr(route_result, "geometry", {}) or {},
    )


def _validate_supported_craft_variants(intent: Any, cell_def: Any = None) -> None:
    """Fail closed on grammar variants the current materializer does not honor."""

    _declared_minimum_structural_occupancy(intent)
    _declared_minimum_functional_od_occupancy(intent)
    bands = _exact_craft_declarations(intent, "bands")
    if len(bands) > 1:
        raise UnsupportedCraftCompactionError(
            "canonical craft compaction supports exactly one @band declaration"
        )
    if bands:
        from clay.layout.intent import BandIntent

        declaration = bands[0]
        if type(declaration) is not BandIntent:
            raise UnsupportedCraftCompactionError(
                "canonical craft compaction requires an exact BandIntent"
            )
        center = str(getattr(bands[0], "center", "") or "")
        if center != "passives":
            groups = {
                str(getattr(group, "group", "")): tuple(
                    str(device)
                    for device in tuple(getattr(group, "devices", ()) or ())
                )
                for group in tuple(getattr(intent, "place_groups", ()) or ())
            }
            if not groups.get(center):
                raise UnsupportedCraftCompactionError(
                    f"band center group {center!r} is not a non-empty place_group"
                )
            source_classes = {
                str(getattr(device, "instance_name", "")): str(
                    getattr(device, "device_class", "") or ""
                ).lower()
                for device in tuple(getattr(cell_def, "devices", ()) or ())
            }
            if any(
                source_classes.get(member) not in {"res", "cap"}
                for member in groups[center]
            ):
                raise UnsupportedCraftCompactionError(
                    f"band center group {center!r} must contain only source-bound "
                    "passive devices"
                )
        if cell_def is not None:
            pmos_classes = {"pmos", "pmos3v", "pch", "pch3", "pd"}
            nmos_classes = {"nmos", "nmos3v", "nch", "nch3", "nd"}
            connected = {"PMOS": 0, "NMOS": 0}
            for device in tuple(getattr(cell_def, "devices", ()) or ()):
                device_class = str(
                    getattr(device, "device_class", "") or ""
                ).lower()
                if device_class in pmos_classes:
                    family, rail = "PMOS", str(declaration.top)
                elif device_class in nmos_classes:
                    family, rail = "NMOS", str(declaration.bottom)
                else:
                    continue
                pins = dict(getattr(device, "pins", {}) or {})
                if rail in {str(pins.get("s", "")), str(pins.get("b", ""))}:
                    connected[family] += 1
            for family, side in (("PMOS", "top"), ("NMOS", "bottom")):
                if connected[family] == 0:
                    rail = str(
                        declaration.top if family == "PMOS" else declaration.bottom
                    )
                    raise UnsupportedCraftCompactionError(
                        f"band requires a non-empty {family} source/body connection "
                        f"to {side} rail {rail!r}"
                    )
    anchors = _exact_craft_declarations(intent, "anchors")
    if len(anchors) > 64:
        raise UnsupportedCraftCompactionError("too many anchor declarations")
    from clay.layout.intent import (
        AnchorIntent,
        ChannelIntent,
        PlaceGroup,
        replay_channel_intent,
    )

    for anchor in anchors:
        if type(anchor) is not AnchorIntent:
            raise UnsupportedCraftCompactionError(
                "canonical craft compaction requires exact AnchorIntent values"
            )
    anchor_sha256s = tuple(anchor.sha256 for anchor in anchors)
    if len(set(anchor_sha256s)) != len(anchor_sha256s):
        raise UnsupportedCraftCompactionError(
            "canonical craft compaction rejects duplicate anchor declarations"
        )

    raw_place_groups = getattr(intent, "place_groups", ())
    if type(raw_place_groups) is not tuple:
        raise UnsupportedCraftCompactionError(
            "canonical craft compaction requires place_groups as an exact tuple"
        )
    if len(raw_place_groups) > 4096:
        raise UnsupportedCraftCompactionError(
            "anchor place-group inventory is too large"
        )
    group_names: list[str] = []
    group_members: Dict[str, Tuple[str, ...]] = {}
    for group in raw_place_groups:
        if type(group) is not PlaceGroup or type(group.devices) is not tuple:
            raise UnsupportedCraftCompactionError(
                "canonical craft compaction requires exact tuple-backed PlaceGroup values"
            )
        name = group.group
        members = group.devices
        if type(name) is not str or not name or name != name.strip():
            raise UnsupportedCraftCompactionError(
                "anchor place-group names must be canonical non-empty strings"
            )
        if len(name) > 256:
            raise UnsupportedCraftCompactionError(
                f"anchor place-group {name!r} name is too long"
            )
        if not members:
            raise UnsupportedCraftCompactionError(
                f"anchor place-group {name!r} must be non-empty"
            )
        if len(members) > 4096:
            raise UnsupportedCraftCompactionError(
                f"anchor place-group {name!r} membership is too large"
            )
        if any(
            type(member) is not str or not member or member != member.strip()
            for member in members
        ):
            raise UnsupportedCraftCompactionError(
                f"anchor place-group {name!r} has a noncanonical member"
            )
        if any(len(member) > 256 for member in members):
            raise UnsupportedCraftCompactionError(
                f"anchor place-group {name!r} member is too long"
            )
        if len(set(members)) != len(members):
            raise UnsupportedCraftCompactionError(
                f"anchor place-group {name!r} has duplicate members"
            )
        group_names.append(name)
        group_members[name] = members
    if len(set(group_names)) != len(group_names):
        raise UnsupportedCraftCompactionError(
            "anchor declarations reject duplicate place-group names"
        )
    if cell_def is not None:
        source_devices = tuple(getattr(cell_def, "devices", ()) or ())
        source_names = {
            str(getattr(device, "instance_name", "")) for device in source_devices
        }
        source_nets = {
            str(net)
            for device in source_devices
            for net in dict(getattr(device, "pins", {}) or {}).values()
        }
        source_groups = set(group_names)
        for group_name, members in group_members.items():
            missing_members = tuple(
                member for member in members if member not in source_names
            )
            if missing_members:
                raise UnsupportedCraftCompactionError(
                    f"anchor place-group {group_name!r} has absent source members "
                    f"{missing_members}"
                )
    else:
        source_names = source_nets = source_groups = set()

    for anchor in anchors:
        edge = str(getattr(anchor, "edge", "") or "").strip().lower()
        if edge not in {"left", "right"}:
            raise UnsupportedCraftCompactionError(
                f"anchor edge {edge!r} is unsupported by the canonical craft "
                "compaction materializer (implemented: left/right)"
            )
        target = str(anchor.target)
        if cell_def is not None:
            resolutions = sum(
                target in inventory
                for inventory in (source_names, source_nets, source_groups)
            )
            if resolutions == 0:
                raise UnsupportedCraftCompactionError(
                    f"anchor target {target!r} is absent from source topology/groups"
                )
            if resolutions > 1:
                raise UnsupportedCraftCompactionError(
                    f"anchor target {target!r} is ambiguous across net/device/group"
                )
    channels = _exact_craft_declarations(intent, "channels")
    if len(channels) > 1:
        raise UnsupportedCraftCompactionError(
            "canonical craft compaction requires exactly one channel declaration"
        )
    for channel in channels:
        if type(channel) is not ChannelIntent:
            raise UnsupportedCraftCompactionError(
                "canonical craft compaction requires exact ChannelIntent values"
            )
        try:
            channel = replay_channel_intent(channel)
        except (TypeError, ValueError) as exc:
            raise UnsupportedCraftCompactionError(str(exc)) from exc
        orient = str(getattr(channel, "orient", "") or "").strip().lower()
        if orient != "vertical":
            raise UnsupportedCraftCompactionError(
                f"channel orientation {orient!r} is unsupported by the canonical "
                "craft compaction materializer (implemented: vertical)"
            )
        if float(getattr(channel, "width_um", 0.0) or 0.0) <= 0.0:
            raise UnsupportedCraftCompactionError(
                "channel width must be positive for canonical craft compaction"
            )
    if channels and not tuple(getattr(intent, "common_centroids", ()) or ()):
        raise UnsupportedCraftCompactionError(
            "vertical channel compaction requires a declared common-centroid "
            "left-column structure"
        )


def _adjacency_from_cell(cell_def: Any) -> tuple[object, ...]:
    """Replay the complete typed device-level ``@adjacent`` inventory."""

    from clay.layout.intent import adjacent_intents_from_cell_def

    return adjacent_intents_from_cell_def(cell_def)


def _well_island_share_plans(cell_def: Any, intent: Any, tech: Any) -> tuple:
    """Compile canonical topology declarations into PDK-bound placer authority."""

    from clay.layout.well_island_sharing import compile_well_island_share_plans

    declarations = _exact_craft_declarations(intent, "well_island_shares")
    return compile_well_island_share_plans(cell_def, declarations, tech)


_INTENT_MIM_ATTRS = ("mim_option", "metal_level", "mim_cap", "pex_backend")


def _copy_intent_mim_attrs(source: Any, target: Any) -> Any:
    """Propagate attached GF180 MIM configuration attributes across intent copies."""
    if source is None or target is None:
        return target
    src_dict = getattr(source, "__dict__", None)
    dst_dict = getattr(target, "__dict__", None)
    if not isinstance(src_dict, dict) or not isinstance(dst_dict, dict):
        return target
    for attr in _INTENT_MIM_ATTRS:
        if attr in src_dict and src_dict[attr] is not None and attr not in dst_dict:
            dst_dict[attr] = src_dict[attr]
    return target


def _resolve_layout_tech(pdk: Any, cell_def: Any = None, intent: Any = None):
    """Resolve ``LayoutTech`` while preserving option-bound MIM configuration."""
    from clay.layout.tech import LayoutTech

    tech = pdk if isinstance(pdk, LayoutTech) else LayoutTech.resolve(str(pdk))
    if str(getattr(tech, "pdk_id", "") or "").lower() != "gf180mcu":
        return tech
    mim_opt = (
        getattr(tech, "mim_option", None)
        or getattr(intent, "mim_option", None)
        or getattr(cell_def, "mim_option", None)
    )
    cap_opt = (
        getattr(tech, "cap_option", None)
        or getattr(intent, "mim_cap", None)
        or getattr(cell_def, "mim_cap", None)
    )
    lvl = (
        getattr(tech, "metal_level", None)
        or getattr(intent, "metal_level", None)
        or getattr(cell_def, "metal_level", None)
    )
    pex = (
        getattr(tech, "pex_backend", None)
        or getattr(intent, "pex_backend", None)
        or getattr(cell_def, "pex_backend", None)
    )
    if any(
        val is not None and str(val).strip()
        for val in (mim_opt, cap_opt, lvl, pex)
    ):
        return tech.with_mim_config(
            mim_option=mim_opt,
            cap_option=cap_opt,
            metal_level=lvl,
            pex_backend=pex,
        )
    return tech


def _physical_shared_guard_intent(intent: Any, inventory: Any) -> Any:
    """Derive the physical ring inventory without rewriting source intent."""

    if inventory is None:
        return intent
    from clay.layout.intent import LayoutIntent
    from clay.layout.well_island_sharing import SharedWellGuardRingInventory

    if type(intent) is not LayoutIntent:
        raise TypeError("shared guard lowering requires one exact LayoutIntent")
    if type(inventory) is not SharedWellGuardRingInventory:
        raise TypeError(
            "shared guard lowering requires one exact physical ring inventory"
        )
    if not inventory.envelope_bindings:
        raise ValueError("shared guard lowering requires a non-empty binding inventory")
    return _copy_intent_mim_attrs(
        intent,
        replace(intent, guard_rings=inventory.effective_guard_rings),
    )


def _prepare_shared_guard_hierarchy(
    intent: Any,
    well_island_share_plans: Any,
    placement_um: Any,
    geometry: Any,
    bridged_intent: Any,
    tech: Any,
    tech_id: str,
    *,
    gap_nm: int,
    rigid_nonmos_macro_plans: Any = None,
) -> PreparedSharedGuardHierarchy:
    """Compile one generic shared-well hierarchy without freezing its members."""

    from clay.layout.compact_blocks import extract_blocks
    from clay.layout.extract.realize_render import (
        _placed_stamp_bboxes_nm,
        _ring_extension_nm,
    )
    from clay.layout.place_blocks_2d import (
        FlexibleEnvelopeConstraint,
        _materialized_adjacency_scope_bboxes_nm,
        _resolve_block_member_devices,
    )
    from clay.layout.shared_well_hierarchy import (
        compile_shared_well_guard_child_plans,
    )
    from clay.layout.well_island_sharing import (
        compile_shared_well_guard_envelopes,
        compile_shared_well_guard_ring_inventory,
    )

    guard_rings = getattr(intent, "guard_rings", ())
    if type(guard_rings) is not tuple:
        raise TypeError("shared guard hierarchy requires an exact ring tuple")
    envelopes = compile_shared_well_guard_envelopes(
        well_island_share_plans,
        guard_rings,
    )
    if not envelopes:
        raise ValueError("shared guard hierarchy requires a non-empty envelope plan")
    ring_inventory = compile_shared_well_guard_ring_inventory(
        envelopes,
        guard_rings,
    )
    blocks, block_stamps, block_gring, _gring_names = extract_blocks(
        geometry,
        placement_um,
        intent,
    )
    block_ids = {block_id for block_id, _bbox_nm in blocks}
    source_names = {
        str(member.name)
        for field in ("devices", "passives", "bjts")
        for member in tuple(getattr(bridged_intent, field, ()) or ())
    }
    if not source_names:
        raise ValueError("shared guard hierarchy has no canonical source members")
    resolved_ownership = _resolve_block_member_devices(block_stamps, geometry)
    placed = _placed_stamp_bboxes_nm(geometry, placement_um)
    ownership_rows: list[tuple[str, tuple[str, ...]]] = []
    content_bbox_rows: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, declaration in enumerate(guard_rings):
        block_id = f"ISL{index}"
        if block_id not in block_ids:
            continue
        exact_source_members = tuple(
            sorted(set(resolved_ownership.get(block_id, ())) & source_names)
        )
        if exact_source_members != tuple(sorted(declaration.members)):
            raise ValueError(
                f"shared guard hierarchy owner {block_id!r} differs from declaration"
            )
        stamps = tuple(sorted(block_stamps.get(block_id, ())))
        if not stamps or any(stamp not in placed for stamp in stamps):
            raise ValueError(
                f"shared guard hierarchy owner {block_id!r} lacks placed stamps"
            )
        boxes = tuple(placed[stamp] for stamp in stamps)
        ownership_rows.append((block_id, exact_source_members))
        content_bbox_rows.append(
            (
                block_id,
                (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ),
            )
        )
    extension_rows = (
        ("nwell", int(_ring_extension_nm(tech, "n"))),
        ("pwell", int(_ring_extension_nm(tech, "p"))),
    )
    child_plans = compile_shared_well_guard_child_plans(
        envelopes,
        guard_rings,
        tuple(sorted(ownership_rows)),
        tuple(sorted(content_bbox_rows)),
        gap_nm=gap_nm,
        ring_extension_nm_by_well=extension_rows,
    )
    canonical_blocks = tuple(sorted(blocks))
    master_members = {
        block_id: (block_id,) for block_id, _bbox_nm in canonical_blocks
    }
    adjacency_scope_rows = _materialized_adjacency_scope_bboxes_nm(
        master_blocks=canonical_blocks,
        master_members=master_members,
        materialized_stamps=block_stamps,
        materialized_rings=block_gring,
        geometry=geometry,
        placement_um=placement_um,
        tech=tech,
        source_members=tuple(sorted(source_names)),
        rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
    )
    physical_bboxes = dict(canonical_blocks)
    content_bboxes = dict(content_bbox_rows)
    adjacency_bboxes = dict(adjacency_scope_rows)
    flexible_constraints: list[FlexibleEnvelopeConstraint] = []
    flexible_member_ids: set[str] = set()
    for plan in child_plans:
        members = tuple(master.block_id for master in plan.masters)
        if any(
            member not in physical_bboxes
            or member not in content_bboxes
            or member not in adjacency_bboxes
            for member in members
        ):
            raise ValueError(
                f"shared guard envelope {plan.group!r} lacks exact physical views"
            )
        if flexible_member_ids.intersection(members):
            raise ValueError("shared guard envelopes overlap one physical owner")
        flexible_member_ids.update(members)

        def _offset(
            bbox: tuple[int, int, int, int],
            content: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            return (
                bbox[0] - content[0],
                bbox[1] - content[1],
                bbox[2] - content[0],
                bbox[3] - content[1],
            )

        flexible_constraints.append(
            FlexibleEnvelopeConstraint(
                group=plan.group,
                members=members,
                extension_nm=plan.ring_extension_nm,
                foreign_clearance_nm=gap_nm,
                foreign_exclusion_mode="envelope",
                member_physical_offsets_nm=tuple(
                    (
                        member,
                        _offset(physical_bboxes[member], content_bboxes[member]),
                    )
                    for member in members
                ),
                member_adjacency_offsets_nm=tuple(
                    (
                        member,
                        _offset(adjacency_bboxes[member], content_bboxes[member]),
                    )
                    for member in members
                ),
            )
        )
    return PreparedSharedGuardHierarchy(
        physical_intent=intent,
        placement_um=placement_um,
        geometry=geometry,
        envelope_plans=envelopes,
        ring_inventory=ring_inventory,
        child_plans=child_plans,
        flexible_envelope_constraints=tuple(sorted(flexible_constraints)),
        flexible_content_bboxes_nm=tuple(
            row
            for row in sorted(content_bbox_rows)
            if row[0] in flexible_member_ids
        ),
    )


def _ring_names(geometry: Mapping[str, Any]) -> set[str]:
    return {
        str(name)
        for name, geom in geometry.items()
        if str(getattr(geom, "source", "")) == "group_guard_ring"
        or str(name).startswith("GRING")
    }


def _canonicalize_free_dummy_ring_islands(
    *,
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    intent: Any,
    bridged_intent: Any,
    group_members_by_label: Mapping[str, Tuple[str, ...]],
    tech: Any,
    declared_gap_nm: int,
    matching_pair_axes: Any,
) -> Dict[str, Tuple[float, float]]:
    """Normalize one proven matching-pair + free-dummy ring locally.

    A separately owned ``share_diffusion=false`` flank dummy can be repacked.
    A complete joint row keeps its rigid pose after current artwork/source replay.
    When its target is exactly one materializer-certified two-device MOS pair and
    one declared guard ring owns that pair plus one rail-tied residual dummy, treat
    the already-symmetric pair as a rigid local master.  Compare placing the dummy
    beside or above that master with the source/PDK-derived gap and the installed
    ring extension, then select the smaller ring-inclusive rectangle.  The pair's
    relative geometry is never re-solved and the incoming island union supplies
    only a translation origin; no reference coordinate enters the decision.

    The optimization is deliberately narrow.  A declared free-dummy topology with
    missing, ambiguous, or malformed ownership is a named stop rather than an
    invitation to repack an unrelated ring.
    """

    from clay.layout.extract.realize_render import (
        _placed_stamp_bboxes_nm,
        _ring_extension_nm,
    )
    from clay.layout.intent import DummyFlankIntent, GuardRingIntent
    from clay.layout.place_blocks_2d import _effective_block_gap_nm

    raw_flanks = _exact_craft_declarations(intent, "dummy_flanks")
    raw_rings = _exact_craft_declarations(intent, "guard_rings")
    if type(group_members_by_label) is not dict:
        raise UnsupportedCraftCompactionError(
            "canonical free-dummy packing requires an exact group map"
        )
    if type(matching_pair_axes) is not dict:
        raise UnsupportedCraftCompactionError(
            "canonical free-dummy packing requires exact matching-axis authority"
        )
    if type(declared_gap_nm) is not int or declared_gap_nm < 0:
        raise UnsupportedCraftCompactionError(
            "canonical free-dummy packing requires a non-negative integer gap"
        )

    devices: Dict[str, Any] = {}
    for device in tuple(getattr(bridged_intent, "devices", ()) or ()):
        name = getattr(device, "name", None)
        if type(name) is not str or not name or name in devices:
            raise UnsupportedCraftCompactionError(
                "canonical free-dummy packing has ambiguous MOS device authority"
            )
        devices[name] = device

    out = dict(placement_um)
    consumed_members: set[str] = set()
    for declaration in raw_flanks:
        if type(declaration) is not DummyFlankIntent:
            raise UnsupportedCraftCompactionError(
                "canonical dummy-flank inventory contains a foreign declaration"
            )
        if (
            type(declaration.target) is not str
            or not declaration.target
            or declaration.target != declaration.target.strip()
            or type(declaration.per_end) is not int
            or declaration.per_end <= 0
            or type(declaration.tie) is not str
            or declaration.tie != declaration.tie.strip().lower()
            or type(declaration.share_diffusion) is not bool
        ):
            raise UnsupportedCraftCompactionError(
                "canonical dummy-flank declaration is malformed"
            )
        if declaration.share_diffusion:
            continue
        if declaration.tie != "rail":
            raise UnsupportedCraftCompactionError(
                "free dummy ring packing supports only a rail-tied flank"
            )

        raw_pair = group_members_by_label.get(declaration.target)
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or any(type(member) is not str or not member for member in raw_pair)
            or len(set(raw_pair)) != 2
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy target must resolve to one exact two-member group"
            )
        pair = tuple(sorted(raw_pair))
        raw_axis = matching_pair_axes.get(pair)
        if (
            type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or raw_axis[0] not in {"x", "y"}
            or type(raw_axis[0]) is not str
            or type(raw_axis[1]) not in {int, float}
            or not math.isfinite(float(raw_axis[1]))
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy target lacks one exact finite matching axis"
            )

        matching_rings = []
        for ring in raw_rings:
            if type(ring) is not GuardRingIntent:
                raise UnsupportedCraftCompactionError(
                    "canonical guard-ring inventory contains a foreign declaration"
                )
            try:
                replayed_ring = GuardRingIntent(
                    members=ring.members,
                    well=ring.well,
                    tie_net=ring.tie_net,
                    hug=ring.hug,
                    role=ring.role,
                    enclose=ring.enclose,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise UnsupportedCraftCompactionError(
                    "canonical guard-ring declaration is malformed"
                ) from exc
            members = replayed_ring.members
            if set(pair) <= set(members):
                matching_rings.append(replayed_ring)
        if len(matching_rings) != 1:
            raise UnsupportedCraftCompactionError(
                "free dummy target must belong to one exact guard ring"
            )
        ring = matching_rings[0]
        residual = tuple(member for member in ring.members if member not in pair)
        if len(residual) != 1 or len(set(ring.members)) != 3:
            raise UnsupportedCraftCompactionError(
                "free dummy ring must contain its pair and one residual member"
            )
        dummy = residual[0]
        island_members = set(pair) | {dummy}
        overlap = consumed_members & island_members
        if overlap:
            raise UnsupportedCraftCompactionError(
                f"free dummy ring member {min(overlap)!r} is multiply consumed"
            )

        from clay.layout.joint_ring_ownership import resolve_dummy_ring_owners
        try:
            owner_by_member, joint_owner = resolve_dummy_ring_owners(geometry, island_members)
        except ValueError as exc:
            raise UnsupportedCraftCompactionError(str(exc)) from exc

        try:
            pair_devices = tuple(devices[member] for member in pair)
            dummy_device = devices[dummy]
        except KeyError as exc:
            raise UnsupportedCraftCompactionError(
                "free dummy ring lacks canonical MOS device authority"
            ) from exc
        raw_kinds = tuple(getattr(device, "kind", None) for device in pair_devices)
        if any(
            type(kind) is not str or kind != kind.strip().lower()
            for kind in raw_kinds
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy MOS kind authority is malformed"
            )
        kinds = raw_kinds
        if kinds[0] not in {"nmos", "pmos"} or kinds[0] != kinds[1]:
            raise UnsupportedCraftCompactionError(
                "free dummy matching pair must be one MOS polarity"
            )
        dummy_kind = getattr(dummy_device, "kind", None)
        if (
            type(dummy_kind) is not str
            or dummy_kind != dummy_kind.strip().lower()
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy MOS kind authority is malformed"
            )
        if dummy_kind != kinds[0]:
            raise UnsupportedCraftCompactionError(
                "free dummy must have the matching pair's MOS polarity"
            )
        def device_geometry(device: Any) -> Tuple[float, float, int]:
            try:
                width = getattr(device, "W")
                length = getattr(device, "L")
                multiplicity = getattr(device, "M")
            except AttributeError as exc:
                raise UnsupportedCraftCompactionError(
                    "free dummy geometry authority is malformed"
                ) from exc
            if (
                type(width) not in {int, float}
                or type(length) not in {int, float}
                or type(multiplicity) is not int
                or not math.isfinite(float(width))
                or not math.isfinite(float(length))
                or float(width) <= 0.0
                or float(length) <= 0.0
                or multiplicity <= 0
            ):
                raise UnsupportedCraftCompactionError(
                    "free dummy geometry authority is malformed"
                )
            return float(width), float(length), multiplicity

        pair_geometry = tuple(device_geometry(device) for device in pair_devices)
        dummy_geometry = device_geometry(dummy_device)
        if (
            pair_geometry[0] != pair_geometry[1]
            or dummy_geometry[0] != pair_geometry[0][0]
            or dummy_geometry[1] != pair_geometry[0][1]
            or dummy_geometry[2] != 2 * declaration.per_end * pair_geometry[0][2]
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy geometry does not replay the declared pair flank count"
            )
        dummy_nets = getattr(dummy_device, "nets", None)
        if (
            type(dummy_nets) is not dict
            or any(
                type(role) is not str or type(net) is not str
                for role, net in dummy_nets.items()
            )
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy net authority is malformed"
            )
        rail = dummy_nets.get("gate", "")
        if (
            not rail
            or any(dummy_nets.get(role, "") != rail for role in ("drain", "source", "body"))
            or str(ring.tie_net) != rail
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy is not fully rail-tied to its guard ring"
            )

        if joint_owner is None:
            bboxes = _placed_stamp_bboxes_nm(dict(geometry), out)
        else:
            from clay.layout.joint_ring_ownership import require_joint_ring_artwork
            try:
                bboxes = require_joint_ring_artwork(geometry[joint_owner],
                    placement_um=out.get(joint_owner), members=tuple(sorted(island_members)),
                    devices=devices, tech=tech)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise UnsupportedCraftCompactionError(f"free dummy joint ring ownership is invalid: {exc}") from exc
        if any(member not in bboxes for member in island_members):
            raise UnsupportedCraftCompactionError(
                "free dummy ring lacks measurable placed artwork"
            )
        pair_boxes = tuple(bboxes[member] for member in pair)
        pair_sizes = tuple(
            (box[2] - box[0], box[3] - box[1]) for box in pair_boxes
        )
        pair_centres = tuple(
            ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            for box in pair_boxes
        )
        axis_index = 0 if raw_axis[0] == "x" else 1
        perpendicular = 1 - axis_index
        if (
            pair_sizes[0] != pair_sizes[1]
            or pair_centres[0][perpendicular] != pair_centres[1][perpendicular]
            or (
                pair_centres[0][axis_index] + pair_centres[1][axis_index]
            )
            / 2.0
            != float(raw_axis[1])
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy matching pair artwork does not replay its axis"
            )
        pair_bbox = (
            min(box[0] for box in pair_boxes),
            min(box[1] for box in pair_boxes),
            max(box[2] for box in pair_boxes),
            max(box[3] for box in pair_boxes),
        )
        dummy_bbox = bboxes[dummy]
        island_bbox = (
            min(pair_bbox[0], dummy_bbox[0]),
            min(pair_bbox[1], dummy_bbox[1]),
            max(pair_bbox[2], dummy_bbox[2]),
            max(pair_bbox[3], dummy_bbox[3]),
        )
        pair_width = pair_bbox[2] - pair_bbox[0]
        pair_height = pair_bbox[3] - pair_bbox[1]
        dummy_width = dummy_bbox[2] - dummy_bbox[0]
        dummy_height = dummy_bbox[3] - dummy_bbox[1]
        if min(pair_width, pair_height, dummy_width, dummy_height) <= 0:
            raise UnsupportedCraftCompactionError(
                "free dummy ring contains degenerate artwork"
            )
        gap_nm = _effective_block_gap_nm(intent, tech, declared_gap_nm)
        extension_nm = _ring_extension_nm(tech, str(ring.well))
        if (
            type(gap_nm) is not int
            or gap_nm < 0
            or type(extension_nm) is not int
            or extension_nm <= 0
        ):
            raise UnsupportedCraftCompactionError(
                "free dummy ring lacks positive installed-PDK geometry authority"
            )

        if joint_owner is not None:
            # A complete current native row already owns the pair and both
            # dummy flanks. Preserve that one rigid master after the same source,
            # axis and PDK checks; splitting its member views would move raw parts.
            consumed_members.update(island_members)
            continue

        candidates = (
            (
                "horizontal",
                pair_width + gap_nm + dummy_width,
                max(pair_height, dummy_height),
            ),
            (
                "vertical",
                max(pair_width, dummy_width),
                pair_height + gap_nm + dummy_height,
            ),
        )
        mode, selected_width, selected_height = min(
            candidates,
            key=lambda item: (
                (item[1] + 2 * extension_nm)
                * (item[2] + 2 * extension_nm),
                item[0],
            ),
        )
        current_width = island_bbox[2] - island_bbox[0]
        current_height = island_bbox[3] - island_bbox[1]
        current_area = (current_width + 2 * extension_nm) * (
            current_height + 2 * extension_nm
        )
        selected_area = (selected_width + 2 * extension_nm) * (
            selected_height + 2 * extension_nm
        )
        if selected_area >= current_area:
            consumed_members.update(island_members)
            continue

        origin_x, origin_y = island_bbox[0], island_bbox[1]
        pair_delta = (origin_x - pair_bbox[0], origin_y - pair_bbox[1])
        if mode == "horizontal":
            dummy_origin = (origin_x + pair_width + gap_nm, origin_y)
        else:
            dummy_origin = (origin_x, origin_y + pair_height + gap_nm)
        dummy_delta = (
            dummy_origin[0] - dummy_bbox[0],
            dummy_origin[1] - dummy_bbox[1],
        )
        for member in pair:
            x_um, y_um = out[owner_by_member[member]]
            out[owner_by_member[member]] = (
                float(x_um) + pair_delta[0] / 1000.0,
                float(y_um) + pair_delta[1] / 1000.0,
            )
        x_um, y_um = out[owner_by_member[dummy]]
        out[owner_by_member[dummy]] = (
            float(x_um) + dummy_delta[0] / 1000.0,
            float(y_um) + dummy_delta[1] / 1000.0,
        )
        consumed_members.update(island_members)

    return out


def materialize_declared_compact_override(
    placement: Any,
    cell_def: Any,
    intent: Any,
    pdk: Any,
    *,
    effective_policy: Any,
    intent_basis: IntentBasis | None = None,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
    device_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    generated_incumbent: object | None = None,
    area_target: object | None = None,
    layered_placement_assemblies: tuple = (),
    corridor_expand_um: float = 0.0,
) -> Optional[DeclaredCompactOverride]:
    """Materialize the policy-selected craft floorplan for canonical routing.
    Returns ``None`` when the declaration does not select this lowering.  When
    selected, failure is named and fail-closed: no caller may silently fall back
    to the loose placement.  Guard rings are included while solving so their
    footprint is reserved, then removed from the returned route override because
    :func:`route_cell` is their sole materializer.
    """
    from contextlib import ExitStack
    from clay.layout.place.row_replay_session import shared_row_replay_session

    if type(layered_placement_assemblies) is not tuple:
        raise TypeError("layered assembly inventory must be an exact tuple")
    if generated_incumbent is not None:
        from clay.layout.place.generated_incumbent import GeneratedIncumbentSeed
        if type(generated_incumbent) is not GeneratedIncumbentSeed:
            raise TypeError("declared compaction incumbent must be immutable native seed data")
    if not declared_craft_compaction_requested(effective_policy, intent):
        if layered_placement_assemblies:
            raise DeclaredCompactionError("layered assembly inventory requires declared compact materialization")
        if generated_incumbent is not None:
            raise DeclaredCompactionError("incumbent requested without declared compact materialization")
        return None
    effective_policy = _canonical_compaction_policy(cell_def, effective_policy)
    _validate_supported_craft_variants(intent, cell_def)
    from clay.layout.compact_blocks import _envelope
    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    from clay.layout.place_blocks_2d import RepositionPum2DResult, reposition_pum_2d
    from clay.layout.route.route_cell import (
        _bridge_placement_to_route_inputs,
        _materialize_guard_rings,
    )
    from clay.layout.tech import LayoutTech
    tech = _resolve_layout_tech(pdk, cell_def=cell_def, intent=intent)
    tech_id = getattr(tech, "pdk_id", None) or str(pdk)
    with ExitStack() as _replay_stack:
        if getattr(placement, "native_geometry", None) is not None:
            _replay_stack.enter_context(
                shared_row_replay_session(
                    program=getattr(placement, "row_program", None),
                    materialization=getattr(placement, "row_materialization", None),
                    tech=tech,
                    source_cell=cell_def,
                )
            )
        from clay.layout.place.native_matching_well import (
            select_native_matching_well_pairs,
            verify_native_matching_emitted_wells,
        )
        native_matching_well_pairs = select_native_matching_well_pairs(
            placement, cell_def, intent, tech
        )
        well_island_share_plans = _well_island_share_plans(cell_def, intent, tech)
        gap = getattr(effective_policy, "gap_um", None)
        gap_um = (1.5 if gap is None else float(gap)) + max(0.0, float(corridor_expand_um))
        guard_rings = getattr(intent, "guard_rings", ())
        if type(guard_rings) is not tuple:
            raise TypeError("guard-ring inventory must be an exact tuple")
        prepack_guard_rings = guard_rings
        if well_island_share_plans and guard_rings:
            from clay.layout.well_island_sharing import (
                compile_shared_well_guard_envelopes,
                compile_shared_well_guard_ring_inventory,
            )

            prepack_envelopes = compile_shared_well_guard_envelopes(
                well_island_share_plans,
                guard_rings,
            )
            prepack_guard_rings = compile_shared_well_guard_ring_inventory(
                prepack_envelopes,
                guard_rings,
            ).effective_guard_rings
        hierarchical_rings = tuple(
            ring
            for ring in prepack_guard_rings
            if str(getattr(ring, "pack", "") or "") == "hierarchical"
        )
        if hierarchical_rings:
            # A merged ring is one physical hull.  Seeding that hull from the loose
            # device solve can freeze cell-scale whitespace inside the ring before
            # the exact block master ever sees it.  The deterministic hierarchy-aware
            # shelf pack is only a seed: the downstream block solver and its final
            # certificate remain the sole placement authority.  Legacy rings never
            # enter this branch, preserving their placement bytes exactly.
            if (getattr(placement, 'native_geometry', None) is not None
                    and getattr(placement, 'row_program', None) is not None):
                # A shelf-pack result has no native CP frame proof and cannot become
                # a source-row placement certificate.  The existing native seed is
                # already source-bound and grid-verified; the unchanged downstream
                # materialized solver still owns the final hierarchical placement.
                from clay.layout.generic_placement import verify_stored_row_projection
                if getattr(placement, 'native_row_cp_frame_proof', None) is None:
                    raise DeclaredCompactionError('hierarchical native row seed lacks its CP frame proof')
                verify_stored_row_projection(placement)
            else:
                from clay.layout.envelope_place import cp_compact_replace_placement
                placement, seed_result = cp_compact_replace_placement(
                    placement, cell_def, intent, tech_id=str(tech_id), gap_um=gap_um,
                    max_time_s=0.0, device_records=(dict(device_records) if device_records else None),
                    backend="fallback",
                )
                if seed_result is None or seed_result.status != "FALLBACK":
                    raise DeclaredCompactionError(
                        "hierarchical guard-ring prepack returned no deterministic seed"
                    )
        if getattr(placement, "native_geometry", None) is None:
            _replay_stack.enter_context(
                shared_row_replay_session(
                    program=getattr(placement, "row_program", None),
                    materialization=getattr(placement, "row_materialization", None),
                    tech=tech,
                    source_cell=cell_def,
                )
            )
        return _materialize_declared_compact_override_post_seed(
            placement=placement,
            cell_def=cell_def,
            intent=intent,
            tech=tech,
            tech_id=tech_id,
            effective_policy=effective_policy,
            intent_basis=intent_basis,
            intent_fact_set=intent_fact_set,
            source_inventory=source_inventory,
            device_records=device_records,
            generated_incumbent=generated_incumbent,
            area_target=area_target,
            layered_placement_assemblies=layered_placement_assemblies,
            native_matching_well_pairs=native_matching_well_pairs,
            well_island_share_plans=well_island_share_plans,
            gap_um=gap_um,
            guard_rings=guard_rings,
        )


def _materialize_declared_compact_override_post_seed(
    *,
    placement: Any,
    cell_def: Any,
    intent: Any,
    tech: Any,
    tech_id: Any,
    effective_policy: Any,
    intent_basis: IntentBasis | None,
    intent_fact_set: Any,
    source_inventory: Any,
    device_records: Optional[Mapping[str, Mapping[str, Any]]],
    generated_incumbent: object | None,
    area_target: object | None,
    layered_placement_assemblies: tuple,
    native_matching_well_pairs: Any,
    well_island_share_plans: Any,
    gap_um: float,
    guard_rings: tuple,
) -> DeclaredCompactOverride:
    from clay.layout.compact_blocks import _envelope
    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    from clay.layout.place.native_matching_well import verify_native_matching_emitted_wells
    from clay.layout.place_blocks_2d import RepositionPum2DResult, reposition_pum_2d
    from clay.layout.route.route_cell import (
        _bridge_placement_to_route_inputs,
        _materialize_guard_rings,
    )
    _binding, placement_um, geometry = _bridge_placement_to_route_inputs(
        placement,
        cell_def,
        tech_id,
        device_records=(dict(device_records) if device_records else None),
        intent=intent,
    )
    if not geometry:
        raise DeclaredCompactionError(
            "declared craft compaction has no bridged placement geometry"
        )
    bridged_mos_names = tuple(
        sorted(str(device.name) for device in tuple(_binding.devices))
    )

    try:
        canonical_group_members = compaction_source_groups(cell_def)
    except InvalidCompactionGroupOwnership as error:
        raise UnsupportedCraftCompactionError(str(error)) from error
    matching_pair_axes = getattr(placement, "matching_pair_axes", {})
    from clay.layout.place.row_replay_session import row_replay_phase
    with row_replay_phase('declared_compact.canonicalize_islands'):
        placement_um = _canonicalize_free_dummy_ring_islands(
            geometry=geometry,
            placement_um=placement_um,
            intent=intent,
            bridged_intent=_binding,
            group_members_by_label=canonical_group_members,
            tech=tech,
            declared_gap_nm=int(round(gap_um * 1000.0)),
            matching_pair_axes=matching_pair_axes,
        )

    loose_w, loose_h, _ = _envelope(_placed_stamp_bboxes_nm(geometry, placement_um))
    if guard_rings:
        _materialize_guard_rings(
            guard_rings,
            _binding,
            placement_um,
            geometry,
            tech,
            tech_id,
        )

    physical_intent = intent
    logical_placement_basis = placement
    shared_guard_envelope_plans: tuple = ()
    shared_guard_ring_inventory = None
    shared_guard_child_plans: tuple = ()
    flexible_envelope_constraints: tuple = ()
    flexible_content_bboxes_nm: tuple = ()
    if well_island_share_plans and guard_rings:
        prepared_shared_guard = _prepare_shared_guard_hierarchy(
            intent,
            well_island_share_plans,
            placement_um,
            geometry,
            _binding,
            tech,
            str(tech_id),
            gap_nm=int(round(gap_um * 1000.0)),
            rigid_nonmos_macro_plans=getattr(
                logical_placement_basis, "rigid_nonmos_macro_plans", None
            ),
        )
        physical_intent = prepared_shared_guard.physical_intent
        placement_um = prepared_shared_guard.placement_um
        geometry = prepared_shared_guard.geometry
        shared_guard_envelope_plans = prepared_shared_guard.envelope_plans
        shared_guard_ring_inventory = prepared_shared_guard.ring_inventory
        shared_guard_child_plans = prepared_shared_guard.child_plans
        flexible_envelope_constraints = (
            prepared_shared_guard.flexible_envelope_constraints
        )
        flexible_content_bboxes_nm = (
            prepared_shared_guard.flexible_content_bboxes_nm
        )
        if shared_guard_child_plans and not flexible_envelope_constraints:
            from clay.layout.shared_well_hierarchy import (
                project_shared_well_guard_logical_placement,
            )

            logical_placement_basis = project_shared_well_guard_logical_placement(
                placement,
                shared_guard_child_plans,
            )
            matching_pair_axes = logical_placement_basis.matching_pair_axes
    physical_guard_rings = tuple(getattr(physical_intent, "guard_rings", ()) or ())

    geometry, placement_um, layered_placement_assemblies, loose_w, loose_h = restore_layered_compaction_inputs(
        layered_placement_assemblies, source_cell=cell_def, tech=tech, placement=logical_placement_basis,
        geometry=geometry, placement_um=placement_um, loose_size_um=(loose_w, loose_h))

    aspect_value = getattr(effective_policy, "aspect_target", None)
    # With no declared aspect target, use the incoming envelope only as a solver
    # heuristic.  Policy absence must not become a hard signoff obligation.
    aspect = (
        float(aspect_value)
        if aspect_value is not None
        else float(loose_w) / max(float(loose_h), 1e-12)
    )
    minimum_structural_occupancy = _declared_minimum_structural_occupancy(intent)
    minimum_functional_od_occupancy = (
        _declared_minimum_functional_od_occupancy(intent)
    )
    pack_mode = "dense_2d" if (
        minimum_structural_occupancy is not None
        or minimum_functional_od_occupancy is not None
    ) else (
        "dense_2d"
        if str(getattr(effective_policy, "pack", "rows")).strip().lower() == "dense_2d"
        else "rows"
    )
    # Numeric bounds describe the optimizer's fixed-aspect search domain.  A
    # natural-aspect policy intentionally preserves the incoming envelope and
    # must not be made infeasible merely because that envelope lies outside the
    # sampled fixed-target interval.
    aspect_bounds = (
        declared_compact_aspect_bounds(cell_def)
        if aspect_value is not None
        else None
    )
    try:
        with collision_query_scope():
            with row_replay_phase('declared_compact.reposition'):
                repositioned = reposition_pum_2d(
                    geometry,
                    placement_um,
                    physical_intent,
                    tech,
                    tech_id,
                    gap_um=gap_um,
                    bi=_binding,
                    cell_def=cell_def,
                    row_program=getattr(logical_placement_basis, "row_program", None),
                    row_materialization=getattr(logical_placement_basis, "row_materialization", None),
                    **({"native_cross_row_cp_proof": logical_placement_basis.native_cross_row_cp_proof}
                       if getattr(logical_placement_basis, "native_cross_row_cp_proof", None) is not None else {}),
                    native_geometry=getattr(logical_placement_basis, "native_geometry", None),
                    layered_placement_assemblies=layered_placement_assemblies,
                    aspect_target=aspect,
                    aspect_bounds=aspect_bounds,
                    aspect_obligation=aspect_value is not None,
                    outline_width_um=getattr(effective_policy, "outline_width_um", None),
                    area_target=area_target,
                    adjacency=_adjacency_from_cell(cell_def),
                    pack_mode=pack_mode,
                    minimum_structural_occupancy=minimum_structural_occupancy,
                    left_column_group=str(
                        getattr(effective_policy, "left_column_group", "") or ""
                    ),
                    channel_strategy=str(
                        getattr(effective_policy, "channel_strategy", "declared")
                    ),
                    rigid_nonmos_macro_plans=getattr(
                        logical_placement_basis, "rigid_nonmos_macro_plans", None
                    ),
                    rigid_nonmos_placed_devices=getattr(
                        logical_placement_basis, "devices", None
                    ),
                    matching_pair_axes=matching_pair_axes,
                    well_island_share_plans=well_island_share_plans,
                    shared_well_guard_envelope_plans=shared_guard_envelope_plans,
                    shared_well_guard_child_plans=shared_guard_child_plans,
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    flexible_content_bboxes_nm=flexible_content_bboxes_nm,
                    intent_basis=intent_basis,
                    intent_fact_set=intent_fact_set,
                    source_inventory=source_inventory,
                    **({"generated_incumbent": generated_incumbent} if generated_incumbent is not None else {}),
                )
    except ExecutionStopped:
        raise
    except PlacementInfeasibilityError as exc:
        raise DeclaredCompactionError(
            f"declared craft compaction failed: {type(exc).__name__}: {exc}",
            sufficient_core=exc.diagnostic,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - named fail-closed boundary
        raise _declared_compaction_error(exc) from exc
    if not isinstance(repositioned, RepositionPum2DResult):
        raise DeclaredCompactionError(
            "declared craft compaction returned no typed final placement certificate"
        )
    compact_um, compact_geometry = repositioned

    if minimum_functional_od_occupancy is not None:
        source_mos_names = bridged_mos_names
        functional_area, active_keys = _functional_od_artwork_nm2(
            compact_geometry,
            compact_um,
            tech,
            source_mos_names,
        )
        repositioned = replace(
            repositioned,
            final_placement_certificate=_bind_functional_od_placement_certificate(
                intent,
                repositioned.final_placement_certificate,
                functional_od_area_nm2=functional_area,
                active_layer_keys=active_keys,
            ),
        )

    repositioned = replace(
        repositioned,
        final_placement_certificate=_bind_group_route_placement_certificate(
            cell_def,
            intent,
            repositioned.final_placement_certificate,
            intent_fact_set=intent_fact_set,
            source_inventory=source_inventory,
        ),
    )

    names = _ring_names(compact_geometry)
    if len(names) != len(physical_guard_rings):
        raise DeclaredCompactionError(
            "declared craft compaction materialized "
            f"{len(names)}/{len(physical_guard_rings)} guard rings"
        )
    width, height, _ = _envelope(_placed_stamp_bboxes_nm(compact_geometry, compact_um))
    if width <= 0.0 or height <= 0.0:
        raise DeclaredCompactionError(
            f"declared craft compaction returned degenerate {width}x{height}um envelope"
        )
    from clay.layout.place_blocks_2d import placement_certificate_is_signable

    if not placement_certificate_is_signable(repositioned.final_placement_certificate):
        certificate = repositioned.final_placement_certificate
        raise DeclaredCompactionError(
            "declared craft compaction returned an unsignable final placement: "
            f"status={certificate.status} fallback={certificate.used_fallback} "
            f"geometry={certificate.geometry_legal} cuts={certificate.cuts_satisfied} "
            f"aspect={certificate.aspect_satisfied}"
        )
    _require_density_bound_placement_certificate(
        intent, repositioned.final_placement_certificate
    )
    _require_od_fill_bound_placement_certificate(
        intent,
        repositioned.final_placement_certificate,
        functional_od_area_nm2=(
            functional_area if minimum_functional_od_occupancy is not None else None
        ),
        active_layer_keys=(
            active_keys if minimum_functional_od_occupancy is not None else None
        ),
    )

    route_um, route_geometry = final_layered_compaction_view(repositioned.final_placement_certificate,
        bindings=layered_placement_assemblies, source_cell=cell_def, tech=tech,
        placement=logical_placement_basis, intent=physical_intent, geometry=compact_geometry,
        placement_um=compact_um, ring_names=names)
    # Match members become one solver master, so inter-master collision replay
    # cannot check the well spacing within that owner. Independently replay the
    # source-owned final stamp masks before publishing the route override.
    verify_native_matching_emitted_wells(native_matching_well_pairs, route_geometry, route_um)
    return DeclaredCompactOverride(
        override=(route_um, route_geometry),
        ring_inclusive=(compact_um, compact_geometry),
        logical_placement_basis=logical_placement_basis,
        width_um=float(width),
        height_um=float(height),
        loose_width_um=float(loose_w),
        loose_height_um=float(loose_h),
        n_rings=len(names),
        n_declared_rings=len(physical_guard_rings),
        final_placement_certificate=repositioned.final_placement_certificate,
        member_ring_pad_top_nm=repositioned.member_ring_pad_top_nm,
        member_ring_pad_bottom_nm=repositioned.member_ring_pad_bottom_nm,
        n_source_guard_ring_declarations=len(guard_rings),
        shared_well_guard_envelope_plans=shared_guard_envelope_plans,
        shared_well_guard_ring_inventory=shared_guard_ring_inventory,
        shared_well_guard_child_plans=shared_guard_child_plans,
        shared_well_flexible_content_bboxes_nm=flexible_content_bboxes_nm,
        source_mos_names=bridged_mos_names,
    )


@dataclass(frozen=True)
class _AuthenticatedPreparedCompact:
    source_sha256: str
    topology_sha256: Optional[str]
    pdk_id: str
    pdk_provenance_sha256: str
    snapshot_value: Any


_AUTHENTICATED_PREPARED_COMPACTS: Dict[
    int, Tuple[_weakref[DeclaredCompactOverride], _AuthenticatedPreparedCompact]
] = {}


def register_authenticated_prepared_compact(
    compact: DeclaredCompactOverride,
    *,
    source_sha256: str,
    topology_sha256: Optional[str],
    pdk_id: str,
    pdk_provenance_sha256: str,
    snapshot_value: Any,
) -> None:
    """Internal registration boundary for prepared_scene_io after full replay."""
    if type(compact) is not DeclaredCompactOverride:
        raise TypeError("authenticated prepared compact requires DeclaredCompactOverride")
    obj_id = id(compact)
    record = _AuthenticatedPreparedCompact(
        source_sha256=source_sha256,
        topology_sha256=topology_sha256,
        pdk_id=pdk_id,
        pdk_provenance_sha256=pdk_provenance_sha256,
        snapshot_value=snapshot_value,
    )
    _AUTHENTICATED_PREPARED_COMPACTS[obj_id] = (
        _weakref(
            compact,
            lambda _r, key=obj_id: _AUTHENTICATED_PREPARED_COMPACTS.pop(key, None),
        ),
        record,
    )


def _declared_route_result(routed: Any) -> Any:
    """Read native route state without replacing its convergence wrapper."""
    from clay.layout.route_refine import ConvergedRouteCell

    return routed.result if type(routed) is ConvergedRouteCell else routed


def _require_prepared_declared_compact(
    compact: object,
    *,
    cell_def: Any = None,
    pdk: Any = None,
) -> None:
    """Replay the typed boundary (live in-process OR authenticated snapshot)."""
    from clay.layout.place.final_placement_state import final_placement_state_payload
    from clay.layout.place_blocks_2d import (
        FinalPlacementCertificate,
        placement_certificate_is_signable,
    )

    if type(compact) is not DeclaredCompactOverride:
        raise TypeError("prepared_compact requires an exact DeclaredCompactOverride")
    certificate = compact.final_placement_certificate
    if type(certificate) is not FinalPlacementCertificate:
        raise TypeError("prepared compact requires an exact final placement certificate")
    final_placement_state_payload(certificate)
    if not placement_certificate_is_signable(certificate):
        raise DeclaredCompactionError("prepared final placement certificate is not signable")
    entry = _AUTHENTICATED_PREPARED_COMPACTS.get(id(compact))
    authenticated = (
        entry[1] if (entry is not None and entry[0]() is compact) else None
    )
    if authenticated is not None:
        from clay.layout.place.prepared_scene_io import (
            prepared_compact_snapshot_digest,
        )
        from clay.layout.route.resource_world import pdk_provenance_digest
        from clay.layout.tech import LayoutTech

        try:
            current_snapshot = prepared_compact_snapshot_digest(compact)
        except Exception as exc:
            raise DeclaredCompactionError(
                f"authenticated prepared_compact snapshot is unreadable: {exc}"
            ) from exc
        if current_snapshot != authenticated.snapshot_value:
            raise DeclaredCompactionError(
                "authenticated prepared_compact was mutated after load"
            )
        if pdk is not None:
            tech = _resolve_layout_tech(pdk, cell_def=cell_def, intent=intent)
            if (
                tech.pdk_id != authenticated.pdk_id
                or pdk_provenance_digest(tech) != authenticated.pdk_provenance_sha256
            ):
                raise DeclaredCompactionError(
                    "authenticated prepared_compact PDK or rule provenance differs"
                )
        span = getattr(cell_def, "source_span", None)
        if cell_def is not None and isinstance(span, dict) and span.get("file"):
            from clay.frontend.source_revision import reopen_complete_source

            live_source = reopen_complete_source(cell_def)
            if live_source["source_sha256"] != authenticated.source_sha256:
                raise DeclaredCompactionError(
                    "authenticated prepared_compact .clay source SHA-256 differs"
                )


def _bounded_preroute_gate_and_expand(
    *,
    compact: Optional[DeclaredCompactOverride],
    placement: Any,
    cell_def: Any,
    intent: Any,
    pdk: Any,
    effective_policy: Any,
    intent_basis: IntentBasis | None,
    intent_fact_set: Any,
    source_inventory: Any,
    device_records: Optional[Mapping[str, Mapping[str, Any]]],
    engine: Any,
    prepared_compact: DeclaredCompactOverride | None,
    allow_placement_repair: bool,
    diagnostic_observer: Any,
    pre_route_gate: Any,
    enable_preroute_gate: Optional[bool],
    budget_sec: float,
) -> Optional[DeclaredCompactOverride]:
    """Run the <5s pre-route static gate with up to 2 bounded corridor expansions.

    Preserves ``pre_route_gate`` (``LiveRoutingDiagnostic``),
    ``diagnostic_observer``, ``prepared_compact``, ``allow_placement_repair=False``,
    and mock/dummy unit test callers untouched.
    """
    if (
        enable_preroute_gate is False
        or pre_route_gate is not None
        or diagnostic_observer is not None
        or prepared_compact is not None
        or not allow_placement_repair
        or cell_def is None
        or not getattr(placement, "devices", None)
    ):
        return compact

    from clay.layout.route.engine_gridless import GridlessEngine
    from clay.layout.route.engine_registry import route_engine_from_policy
    from clay.layout.tech import LayoutTech

    # Guard toy/mock PDK tokens cleanly while preserving Option-B / metal-level variants.
    try:
        tech = _resolve_layout_tech(pdk, cell_def=cell_def, intent=intent)
    except Exception:
        return compact

    selected_engine = engine
    if selected_engine is None and effective_policy is not None:
        try:
            selected_engine = route_engine_from_policy(effective_policy)
        except Exception:
            selected_engine = None

    if enable_preroute_gate is None and not isinstance(selected_engine, GridlessEngine):
        return compact

    import inspect
    from clay.core_contract import VerificationStatus
    from clay.layout.preroute_gate import run_preroute_gate
    from clay.layout.route.rc_devices import RoutePlanningStop

    supports_expand = (
        "corridor_expand_um" in inspect.signature(materialize_declared_compact_override).parameters
    )
    can_recompact = (
        compact is not None
        and supports_expand
        and declared_craft_compaction_requested(effective_policy, intent)
    )

    current = compact
    total_expand_um = 0.0
    max_expand_rounds = 2

    for round_idx in range(max_expand_rounds + 1):
        gate_placement = (
            getattr(current, "logical_placement_basis", placement)
            if current is not None
            else placement
        )
        gate_place_um, gate_geom = (
            current.override if current is not None and getattr(current, "override", None) else (None, None)
        )
        gate = run_preroute_gate(
            gate_placement,
            cell_def,
            intent,
            tech,
            device_records=device_records,
            effective_policy=effective_policy,
            geometry=gate_geom,
            placement_um=gate_place_um,
            engine=selected_engine,
            final_placement_certificate=getattr(current, "final_placement_certificate", None),
            member_ring_pad_top_nm=getattr(current, "member_ring_pad_top_nm", ()),
            member_ring_pad_bottom_nm=getattr(current, "member_ring_pad_bottom_nm", ()),
            shared_well_guard_envelope_plans=getattr(current, "shared_well_guard_envelope_plans", ()),
            shared_well_guard_ring_inventory=getattr(current, "shared_well_guard_ring_inventory", None),
            budget_sec=max(0.5, float(budget_sec) / (max_expand_rounds + 1)),
        )
        if gate.passed:
            return current

        # In implicit auto-gate mode, enforce proven T1 mismatches and hard T2 blocks.
        # Heuristic corridor_congestion (which estimates inter-device FEOL gap tracks
        # without full over-the-cell 3D routing) still triggers bounded expansion when
        # can_recompact is available, but never fatally aborts an otherwise T1/pin-clean
        # Gridless design when implicit auto-gate cannot recompact (e.g. proven-routable
        # over-the-cell StrongARM layouts).
        hard_t2 = [
            b
            for b in gate.blockers
            if b.tier == "T2" and b.category != "corridor_congestion"
        ]
        expandable_t2 = [
            b
            for b in gate.blockers
            if b.tier == "T2" and b.min_expand_nm > 0
        ]
        if (
            enable_preroute_gate is None
            and gate.t1_passed
            and not hard_t2
            and (not can_recompact or round_idx >= max_expand_rounds)
        ):
            return current
        if (
            not gate.t1_passed
            or not expandable_t2
            or not can_recompact
            or round_idx >= max_expand_rounds
        ):
            state = "PREROUTE_T1_BLOCKED" if not gate.t1_passed else "PREROUTE_T2_BLOCKED"
            raise RoutePlanningStop(
                phase="PRE_ROUTE_GATE",
                state=state,
                evidence=(
                    gate.summary,
                    tuple(b.to_dict() for b in gate.blockers),
                ),
                status=VerificationStatus.FAIL,
            )

        delta_um = max(b.min_expand_nm for b in expandable_t2) / 1000.0
        total_expand_um += delta_um
        rebuilt = materialize_declared_compact_override(
            placement,
            cell_def,
            intent,
            pdk,
            effective_policy=effective_policy,
            intent_basis=intent_basis,
            intent_fact_set=intent_fact_set,
            source_inventory=source_inventory,
            device_records=device_records,
            corridor_expand_um=total_expand_um,
        )
        if rebuilt is None:
            raise RoutePlanningStop(
                phase="PRE_ROUTE_GATE",
                state="PREROUTE_T2_BLOCKED",
                evidence=(
                    gate.summary,
                    tuple(b.to_dict() for b in gate.blockers),
                ),
                status=VerificationStatus.FAIL,
            )
        current = rebuilt

    return current


def route_with_declared_compaction(
    placement: Any,
    cell_def: Any,
    intent: Any,
    pdk: Any,
    *,
    effective_policy: Any,
    intent_basis: IntentBasis | None = None,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
    device_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_iters: int = 1,
    engine: Any = None,
    execution_scope: Any = None,
    prepared_compact: DeclaredCompactOverride | None = None,
    allow_placement_repair: bool = True,
    group_fallback_evidence_dir: Any = None,
    diagnostic_observer: Any = None,
    pre_route_gate: Any = None,
    diagnostic_source_ticket: Any = None,
    enable_preroute_gate: Optional[bool] = None,
    preroute_gate_budget_sec: Optional[float] = None,
) -> Tuple[Any, Optional[DeclaredCompactOverride]]:
    """Route through the one declaration-driven compact flow seam.

    Both ``LayoutStep`` and thin experiment runners call this helper.  It owns
    the lowering from placement declarations to the ring-free route override,
    then hands that result to the canonical ``route_cell_to_intent`` entry.
    A caller may retain a live ``prepared_compact`` and reuse this same native
    handoff. Its origin and mutable artwork must still belong to the checked
    invocation; this option is not a serialized-placement loader. Density, OD
    and native geometry-binding checks remain active. Disabling placement repair
    preserves the original route-planning stop before any placement retry.
    """
    if type(allow_placement_repair) is not bool:
        raise TypeError("allow_placement_repair must be an exact bool")
    if diagnostic_observer is not None and pre_route_gate is not None:
        raise ValueError('pre-route gate and diagnostic observer are mutually exclusive')
    if pre_route_gate is not None:
        from clay.layout.route.live_diagnostic import PreRouteGate
        if type(pre_route_gate) is not PreRouteGate:
            raise TypeError('pre-route gate must be an exact PreRouteGate')
    if (diagnostic_observer is not None or pre_route_gate is not None) and engine is None:
        from clay.layout.route.live_diagnostic import DiagnosticRoutingUnavailable
        raise DiagnosticRoutingUnavailable('diagnostic requires a source-declared route engine')
    if (diagnostic_observer is not None or pre_route_gate is not None) and max_iters != effective_policy.route_iters:
        raise ValueError("diagnostic routing iterations differ from source policy")
    if prepared_compact is None:
        from clay.layout.place.row_replay_session import row_replay_phase
        with row_replay_phase('declared_route.materialize_compact'):
            compact = materialize_declared_compact_override(
                placement,
                cell_def,
                intent,
                pdk,
                effective_policy=effective_policy,
                intent_basis=intent_basis,
                intent_fact_set=intent_fact_set,
                source_inventory=source_inventory,
                device_records=device_records,
            )
    else:
        _require_prepared_declared_compact(
            prepared_compact, cell_def=cell_def, pdk=pdk
        )
        compact = prepared_compact
    from clay.layout.place.row_replay_session import shared_row_replay_session
    from clay.layout.route_refine import route_cell_to_intent

    route_kwargs = dict(
        device_records=(dict(device_records) if device_records else None),
        max_iters=max_iters,
        engine=engine,
    )
    if group_fallback_evidence_dir is not None:
        route_kwargs["group_fallback_evidence_dir"] = group_fallback_evidence_dir
    if execution_scope is not None:
        route_kwargs["execution_scope"] = execution_scope
    if diagnostic_observer is not None:
        route_kwargs.update(diagnostic_observer=diagnostic_observer,
                            diagnostic_compact=compact, diagnostic_policy=effective_policy)
        if diagnostic_source_ticket is not None:
            route_kwargs['diagnostic_source_ticket'] = diagnostic_source_ticket
    elif diagnostic_source_ticket is not None:
        raise ValueError('source handoff requires a diagnostic observer')
    if pre_route_gate is not None:
        route_kwargs.update(pre_route_gate=pre_route_gate,
                            diagnostic_compact=compact, diagnostic_policy=effective_policy)

    def route(prepared: Optional[DeclaredCompactOverride]) -> Any:
        route_placement = (
            placement if prepared is None else prepared.logical_placement_basis
        )
        with shared_row_replay_session(
            program=getattr(route_placement, "row_program", None),
            materialization=getattr(route_placement, "row_materialization", None),
            tech=pdk,
            source_cell=cell_def,
        ):
            if prepared is not None:
                _require_density_bound_placement_certificate(
                    intent, prepared.final_placement_certificate
                )
                od_target = _declared_minimum_functional_od_occupancy(intent)
                if od_target is not None:
                    from clay.layout.tech import LayoutTech

                    route_tech = _resolve_layout_tech(pdk, cell_def=cell_def, intent=intent)
                    if prepared.source_mos_names is not None:
                        mos_names = prepared.source_mos_names
                    else:
                        from clay.layout.route.route_cell import (
                            _bridge_placement_to_route_inputs,
                        )
                        route_tech_id = getattr(route_tech, "pdk_id", None) or str(pdk)
                        route_binding, _unused_pum, _unused_geometry = (
                            _bridge_placement_to_route_inputs(
                                route_placement,
                                cell_def,
                                route_tech_id,
                                device_records=(
                                    dict(device_records) if device_records else None
                                ),
                                intent=intent,
                            )
                        )
                        mos_names = tuple(
                            sorted(str(device.name) for device in route_binding.devices)
                        )
                    route_od_area, route_active_keys = _functional_od_artwork_nm2(
                        prepared.ring_inclusive[1],
                        prepared.ring_inclusive[0],
                        route_tech,
                        mos_names,
                    )
                else:
                    route_od_area, route_active_keys = None, None
                _require_od_fill_bound_placement_certificate(
                    intent,
                    prepared.final_placement_certificate,
                    functional_od_area_nm2=route_od_area,
                    active_layer_keys=route_active_keys,
                )
            from clay.layout.place.row_replay_session import row_replay_phase
            with row_replay_phase('declared_route.route_cell'), collision_query_scope():
                return route_cell_to_intent(
                    route_placement,
                    cell_def,
                    intent,
                    pdk,
                    **route_kwargs,
                    placement_override=None if prepared is None else prepared.override,
                    final_placement_certificate=(
                        None if prepared is None else prepared.final_placement_certificate
                    ),
                    member_ring_pad_top_nm=(
                        () if prepared is None else prepared.member_ring_pad_top_nm
                    ),
                    member_ring_pad_bottom_nm=(
                        ()
                        if prepared is None
                        else prepared.member_ring_pad_bottom_nm
                    ),
                    shared_well_guard_envelope_plans=(
                        ()
                        if prepared is None
                        else prepared.shared_well_guard_envelope_plans
                    ),
                    shared_well_guard_ring_inventory=(
                        None
                        if prepared is None
                        else prepared.shared_well_guard_ring_inventory
                    ),
                    intent_fact_set=intent_fact_set,
                    source_inventory=source_inventory,
                )

    from clay.layout.route.route_cell import RoutePlanningStop
    from clay.layout.place.resource_feedback import (
        CutLedger,
        Unlowerable,
        final_placement_state_digest,
    )

    effective_preroute_budget = (
        float(preroute_gate_budget_sec)
        if preroute_gate_budget_sec is not None
        else max(4.5, min(90.0, 0.6 * len(getattr(placement, "devices", ()) or ())))
    )
    compact = _bounded_preroute_gate_and_expand(
        compact=compact,
        placement=placement,
        cell_def=cell_def,
        intent=intent,
        pdk=pdk,
        effective_policy=effective_policy,
        intent_basis=intent_basis,
        intent_fact_set=intent_fact_set,
        source_inventory=source_inventory,
        device_records=device_records,
        engine=engine,
        prepared_compact=prepared_compact,
        allow_placement_repair=allow_placement_repair,
        diagnostic_observer=diagnostic_observer,
        pre_route_gate=pre_route_gate,
        enable_preroute_gate=enable_preroute_gate,
        budget_sec=effective_preroute_budget,
    )
    try:
        routed = route(compact)
        return _close_declared_resource_feedback(routed, compact, cut_ledger=None)
    except RoutePlanningStop as stopped:
        if diagnostic_observer is not None or pre_route_gate is not None or not allow_placement_repair:
            raise
        initial_feedback_context = getattr(stopped, "feedback_context", None)
        initial_ledger = CutLedger.start(
            pool=getattr(initial_feedback_context, "pool", None),
            final_placement_certificate=(
                None if compact is None else compact.final_placement_certificate
            ),
        )
        if isinstance(initial_ledger, CutLedger):
            stopped.cut_ledger = initial_ledger
        else:
            stopped.cut_ledger_residual = initial_ledger
        repair = _lower_declared_route_repair(stopped, compact)
        if repair is None or compact is None:
            raise
        solve_cut, solve_cut_proof = repair

    def require_ledger(value: object, phase: str) -> CutLedger:
        if isinstance(value, CutLedger):
            return value
        if isinstance(value, Unlowerable):
            raise DeclaredCompactionError(
                "declared capacity repair cut ledger "
                f"{phase}: {value.reason_code}: {value.detail}"
            )
        raise DeclaredCompactionError(
            f"declared capacity repair cut ledger {phase}: foreign result"
        )

    ledger = require_ledger(
        initial_ledger,
        "start",
    )
    ledger = require_ledger(
        ledger.record_cut(solve_cut, proof=solve_cut_proof),
        "record_cut",
    )

    repaired = _repair_declared_compact_override(
        compact,
        compact.logical_placement_basis,
        cell_def,
        intent,
        pdk,
        effective_policy=effective_policy,
        intent_basis=intent_basis,
        intent_fact_set=intent_fact_set,
        source_inventory=source_inventory,
        device_records=device_records,
        solve_cut=solve_cut,
        solve_cut_proof=solve_cut_proof,
    )
    # Reject a metadata-only or otherwise geometry-identical repair before it
    # consumes the second route attempt.  ``record_progress`` is intentionally
    # production-owned here rather than a unit-only CutLedger capability.
    ledger = require_ledger(
        ledger.record_progress(
            final_placement_state_digest(repaired.final_placement_certificate)
        ),
        "record_progress",
    )
    # The canonical bound is two route-planning attempts and one final-master
    # repair.  A repeated conflict, UNKNOWN, or any other typed stop escapes
    # unchanged from this second call; no convergence-price loop is hidden here.
    try:
        routed = route(repaired)
    except RoutePlanningStop as repeated:
        repeated_context = getattr(repeated, "feedback_context", None)
        advanced_or_residual = ledger.advance_scope(
            pool=getattr(repeated_context, "pool", None),
            final_placement_certificate=repaired.final_placement_certificate,
        )
        if isinstance(advanced_or_residual, CutLedger):
            repeated.cut_ledger = advanced_or_residual
        else:
            repeated.cut_ledger = ledger
            repeated.cut_ledger_residual = advanced_or_residual
        raise
    ledger = require_ledger(
        ledger.advance_scope(
            pool=getattr(_declared_route_result(routed), "candidate_pool", None),
            final_placement_certificate=repaired.final_placement_certificate,
        ),
        "advance_scope",
    )
    return _close_declared_resource_feedback(routed, repaired, cut_ledger=ledger)


def _close_declared_resource_feedback(
    routed: Any,
    compact: Optional[DeclaredCompactOverride],
    *,
    cut_ledger: Any,
) -> Tuple[Any, Optional[DeclaredCompactOverride]]:
    """Persist final capacity/Hall closure on result and placement certificate."""

    if compact is None:
        return routed, compact
    route_result = _declared_route_result(routed)
    pool = getattr(route_result, "candidate_pool", None)
    plan = getattr(route_result, "resource_plan", None)
    if pool is None and plan is None:
        return routed, compact
    if pool is None or plan is None:
        raise DeclaredCompactionError("declared resource closure has partial pool/plan lineage")
    from clay.layout.place.final_placement_state import final_placement_state_payload

    original_state = final_placement_state_payload(compact.final_placement_certificate)
    route_certificate = getattr(route_result, "final_placement_certificate", None)
    if (route_certificate is not None
            and final_placement_state_payload(route_certificate) != original_state):
        raise DeclaredCompactionError("routed placement state differs from prepared compact")
    from clay.layout.place.resource_feedback import (
        CutLedger,
        Unlowerable,
        certify_proven_feasible_resources,
    )

    ledger = cut_ledger
    if ledger is None:
        ledger = CutLedger.start(
            pool=pool,
            final_placement_certificate=compact.final_placement_certificate,
        )
    if isinstance(ledger, Unlowerable) or not isinstance(ledger, CutLedger):
        detail = getattr(ledger, "detail", "foreign CutLedger")
        raise DeclaredCompactionError(
            f"declared resource closure cannot start CutLedger: {detail}"
        )
    closure = certify_proven_feasible_resources(
        pool=pool,
        plan=plan,
        final_placement_certificate=compact.final_placement_certificate,
        cut_ledger=ledger,
    )
    signed_certificate = replace(
        compact.final_placement_certificate,
        resource_status="PROVEN_FEASIBLE",
        resource_closure_certificate=closure,
    )
    if final_placement_state_payload(signed_certificate) != original_state:
        raise DeclaredCompactionError("resource closure changed canonical placement state")
    route_result.final_placement_certificate = signed_certificate
    route_result.placement_resource_closure = closure
    return routed, replace(
        compact,
        final_placement_certificate=signed_certificate,
    )


def _lower_declared_route_repair(
    stopped: object,
    compact: Optional[DeclaredCompactOverride],
) -> Optional[Tuple[Any, Any]]:
    """Return one replay-proved repair cut, never infer one from text.

    A pre-signed capacity separation remains the stronger option.  The live v1
    fallback for replayed Capacity/Hall conflicts excludes only the exact
    final-master origin vector and therefore makes no geometric-causality claim.
    """

    if compact is None:
        return None
    from clay.layout.place.resource_feedback import (
        ExactPlacementNoGoodProof,
        PlacementCausalityPayload,
        PlacementCutReplayProof,
        PlacementRepairWitness,
        PlacementSolveCut,
        RoutePlanningFeedbackContext,
        lower_exact_placement_nogood,
        lower_route_planning_feedback,
        lower_placement_conflict,
    )
    from clay.layout.route.resource_plan import CapacityConflict

    context = getattr(stopped, "feedback_context", None)
    if (
        not isinstance(context, RoutePlanningFeedbackContext)
        or context.final_placement_certificate
        is not compact.final_placement_certificate
    ):
        return None
    if (
        getattr(context.final_placement_certificate, "placement_causality_basis", None)
        is not None
    ):
        lowered = lower_route_planning_feedback(context)
        if (
            isinstance(lowered, tuple)
            and len(lowered) == 2
            and isinstance(lowered[0], PlacementSolveCut)
        ):
            return lowered
    payloads = getattr(
        context.final_placement_certificate,
        "placement_causality_payloads",
        (),
    )
    try:
        candidates = tuple(
            sorted(
                (
                    payload
                    for payload in tuple(payloads or ())
                    if isinstance(payload, PlacementCausalityPayload)
                ),
                key=lambda payload: payload.payload_digest,
            )
        )
    except (TypeError, ValueError):
        return None
    for payload in candidates if isinstance(context.conflict, CapacityConflict) else ():
        try:
            witness = PlacementRepairWitness(
                witness_id=(
                    "canonical-route-capacity:"
                    f"{context.conflict_hash}:{payload.payload_digest}"
                ),
                conflict_hash=context.conflict_hash,
                candidate_pool_hash=context.candidate_pool_hash,
                placement_hash=context.placement_state_digest,
                repair_kind=payload.repair_kind,
                movable_block_ids=payload.resource_owner_block_ids,
                movable_site_ids=payload.movable_site_ids,
                corridor_relation=payload.corridor_relation,
                required_clearance_nm=payload.target_required_clearance_nm,
                declaration_refs=context.conflict.declaration_refs,
                causality_payload_digest=payload.payload_digest,
            )
            solve_cut = lower_placement_conflict(
                pool=context.pool,
                conflict=context.conflict,
                witness=witness,
                final_placement_certificate=context.final_placement_certificate,
            )
        except (TypeError, ValueError):
            continue
        if isinstance(solve_cut, PlacementSolveCut):
            return solve_cut, PlacementCutReplayProof(
                pool=context.pool,
                conflict=context.conflict,
                witness=witness,
                final_placement_certificate=context.final_placement_certificate,
            )
    solve_cut = lower_exact_placement_nogood(context)
    if isinstance(solve_cut, PlacementSolveCut):
        return solve_cut, ExactPlacementNoGoodProof(context)
    return None


def _repair_declared_compact_override(
    compact: DeclaredCompactOverride,
    placement: Any,
    cell_def: Any,
    intent: Any,
    pdk: Any,
    *,
    effective_policy: Any,
    intent_basis: IntentBasis | None = None,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
    device_records: Optional[Mapping[str, Mapping[str, Any]]],
    solve_cut: Any,
    solve_cut_proof: Any,
) -> DeclaredCompactOverride:
    """Re-enter the exact materialized compact master with one certified cut."""

    from clay.layout.compact_blocks import _envelope
    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    from clay.layout.place_blocks_2d import RepositionPum2DResult, reposition_pum_2d
    from clay.layout.route.route_cell import _bridge_placement_to_route_inputs
    from clay.layout.tech import LayoutTech

    effective_policy = _canonical_compaction_policy(cell_def, effective_policy)
    tech = _resolve_layout_tech(pdk, cell_def=cell_def, intent=intent)
    tech_id = getattr(tech, "pdk_id", None) or str(pdk)
    well_island_share_plans = _well_island_share_plans(cell_def, intent, tech)
    source_placement = project_final_route_placement(
        placement,
        compact.ring_inclusive[0],
        compact.ring_inclusive[1],
        final_placement_certificate=compact.final_placement_certificate,
        intent=intent,
    )
    binding, _unused_pum, _unused_geometry = _bridge_placement_to_route_inputs(
        source_placement,
        cell_def,
        tech_id,
        device_records=(dict(device_records) if device_records else None),
        intent=intent,
    )
    repair_matching_axes = _repair_matching_pair_axes(
        source_placement,
        compact.final_placement_certificate,
    )
    if repair_matching_axes != getattr(
        source_placement, "matching_pair_axes", None
    ):
        raise DeclaredCompactionError(
            "declared capacity repair logical basis does not replay final axes"
        )
    aspect_value = getattr(effective_policy, "aspect_target", None)
    aspect = (
        float(aspect_value)
        if aspect_value is not None
        else float(compact.loose_width_um) / max(float(compact.loose_height_um), 1e-12)
    )
    gap_value = getattr(effective_policy, "gap_um", None)
    gap_um = 1.5 if gap_value is None else float(gap_value)
    minimum_structural_occupancy = _declared_minimum_structural_occupancy(intent)
    minimum_functional_od_occupancy = (
        _declared_minimum_functional_od_occupancy(intent)
    )
    pack_mode = "dense_2d" if (
        minimum_structural_occupancy is not None
        or minimum_functional_od_occupancy is not None
    ) else (
        "dense_2d"
        if str(getattr(effective_policy, "pack", "rows")).strip().lower() == "dense_2d"
        else "rows"
    )
    shared_well_child_groups = {
        plan.group for plan in compact.shared_well_guard_child_plans
    }
    repositioned = reposition_pum_2d(
        dict(compact.ring_inclusive[1]),
        dict(compact.ring_inclusive[0]),
        intent,
        tech,
        tech_id,
        gap_um=gap_um,
        bi=binding,
        cell_def=cell_def,
        row_program=getattr(source_placement, "row_program", None),
        row_materialization=getattr(source_placement, "row_materialization", None),
        native_geometry=getattr(source_placement, "native_geometry", None),
        aspect_target=aspect,
        aspect_bounds=(
            declared_compact_aspect_bounds(cell_def)
            if aspect_value is not None
            else None
        ),
        aspect_obligation=aspect_value is not None,
        outline_width_um=getattr(effective_policy, "outline_width_um", None),
        adjacency=_adjacency_from_cell(cell_def),
        pack_mode=pack_mode,
        minimum_structural_occupancy=minimum_structural_occupancy,
        left_column_group=str(
            getattr(effective_policy, "left_column_group", "") or ""
        ),
        channel_strategy=str(
            getattr(effective_policy, "channel_strategy", "declared")
        ),
        solve_cuts=(solve_cut,),
        solve_cut_proofs=(solve_cut_proof,),
        solve_cut_source_certificate=compact.final_placement_certificate,
        source_member_ring_pad_top_nm=compact.member_ring_pad_top_nm,
        source_member_ring_pad_bottom_nm=compact.member_ring_pad_bottom_nm,
        rigid_nonmos_macro_plans=getattr(
            source_placement, "rigid_nonmos_macro_plans", None
        ),
        rigid_nonmos_placed_devices=getattr(source_placement, "devices", None),
        matching_pair_axes=repair_matching_axes,
        well_island_share_plans=well_island_share_plans,
        shared_well_guard_envelope_plans=(
            compact.shared_well_guard_envelope_plans
        ),
        shared_well_guard_child_plans=compact.shared_well_guard_child_plans,
        flexible_envelope_constraints=tuple(
            projection.constraint
            for projection in (
                compact.final_placement_certificate.flexible_envelope_projections
            )
            if projection.constraint.group in shared_well_child_groups
        ),
        flexible_content_bboxes_nm=(
            compact.shared_well_flexible_content_bboxes_nm
        ),
        intent_basis=intent_basis,
        intent_fact_set=intent_fact_set,
        source_inventory=source_inventory,
    )
    if not isinstance(repositioned, RepositionPum2DResult):
        raise DeclaredCompactionError(
            "declared capacity repair returned no typed final placement certificate"
        )
    compact_um, compact_geometry = repositioned
    if minimum_functional_od_occupancy is not None:
        source_mos_names = tuple(
            sorted(str(device.name) for device in tuple(binding.devices))
        )
        functional_area, active_keys = _functional_od_artwork_nm2(
            compact_geometry,
            compact_um,
            tech,
            source_mos_names,
        )
        repositioned = replace(
            repositioned,
            final_placement_certificate=_bind_functional_od_placement_certificate(
                intent,
                repositioned.final_placement_certificate,
                functional_od_area_nm2=functional_area,
                active_layer_keys=active_keys,
            ),
        )
    repositioned = replace(
        repositioned,
        final_placement_certificate=_bind_group_route_placement_certificate(
            cell_def,
            intent,
            repositioned.final_placement_certificate,
            intent_fact_set=intent_fact_set,
            source_inventory=source_inventory,
        ),
    )
    from clay.layout.place_blocks_2d import placement_certificate_is_signable

    if not placement_certificate_is_signable(repositioned.final_placement_certificate):
        raise DeclaredCompactionError(
            "declared capacity repair returned an unsignable final placement"
        )
    _require_density_bound_placement_certificate(
        intent, repositioned.final_placement_certificate
    )
    _require_od_fill_bound_placement_certificate(
        intent,
        repositioned.final_placement_certificate,
        functional_od_area_nm2=(
            functional_area if minimum_functional_od_occupancy is not None else None
        ),
        active_layer_keys=(
            active_keys if minimum_functional_od_occupancy is not None else None
        ),
    )
    names = _ring_names(compact_geometry)
    if len(names) != compact.n_declared_rings:
        raise DeclaredCompactionError(
            "declared capacity repair materialized "
            f"{len(names)}/{compact.n_declared_rings} guard rings"
        )
    width, height, _unused_aspect = _envelope(
        _placed_stamp_bboxes_nm(compact_geometry, compact_um)
    )
    if width <= 0.0 or height <= 0.0:
        raise DeclaredCompactionError(
            f"declared capacity repair returned degenerate {width}x{height}um envelope"
        )
    route_um = {name: value for name, value in compact_um.items() if name not in names}
    route_geometry = {
        name: value for name, value in compact_geometry.items() if name not in names
    }
    return DeclaredCompactOverride(
        override=(route_um, route_geometry),
        ring_inclusive=(compact_um, compact_geometry),
        logical_placement_basis=source_placement,
        width_um=float(width),
        height_um=float(height),
        loose_width_um=compact.loose_width_um,
        loose_height_um=compact.loose_height_um,
        n_rings=len(names),
        n_declared_rings=compact.n_declared_rings,
        final_placement_certificate=repositioned.final_placement_certificate,
        member_ring_pad_top_nm=repositioned.member_ring_pad_top_nm,
        member_ring_pad_bottom_nm=repositioned.member_ring_pad_bottom_nm,
        source_mos_names=tuple(
            sorted(str(device.name) for device in tuple(binding.devices))
        ),
    )


def _project_device_to_stamp_bbox(device: Any, stamp_bbox: Tuple[int, int, int, int]) -> Any:
    """Rigidly translate one placed device onto an exact emitted-stamp bbox."""

    from clay.layout.envelope_place import _reposition_device

    width = int(device.bbox[2]) - int(device.bbox[0])
    height = int(device.bbox[3]) - int(device.bbox[1])
    stamp_width = int(stamp_bbox[2]) - int(stamp_bbox[0])
    stamp_height = int(stamp_bbox[3]) - int(stamp_bbox[1])
    if (width, height) != (stamp_width, stamp_height):
        raise DeclaredCompactionError(
            "final route stamp size differs from placed device: "
            f"{device.label} placed={width}x{height}nm "
            f"stamp={stamp_width}x{stamp_height}nm"
        )
    # Odd-sized integer-nm primitives have a legitimate half-nm geometric
    # centre.  Rounding that centre independently can move the projected
    # certificate one nanometre away from the exact emitted stamp.  Seat the
    # existing rigid device by the stamp's integer lower-left instead.
    projected = _reposition_device(
        device,
        int(stamp_bbox[0]) + width // 2,
        int(stamp_bbox[1]) + height // 2,
    )
    if projected.bbox != stamp_bbox:
        raise DeclaredCompactionError(
            f"final route stamp bbox projection failed: {device.label}"
        )
    return projected


def _final_master_matching_pairs(
    final_placement_certificate: Any,
) -> Tuple[Tuple[str, str], ...]:
    """Return exact two-member pairs owned by a certified ``MPAIR`` master.

    ``MPAIR_`` is created only by ``bundle_passive_composites`` after intersecting
    one exact two-resistor ``@place_group`` with the canonical materializer's
    matching-axis certificate.  The final placement certificate then binds that
    master to the exact materialized block partition routed and emitted below.
    """

    rows = tuple(
        getattr(final_placement_certificate, "materialized_block_members", ()) or ()
    )
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            continue
        master, members = row
        if not isinstance(master, str) or not master.startswith("MPAIR_"):
            continue
        if not isinstance(members, (tuple, list)) or len(members) != 2:
            continue
        # ``extract_blocks`` names a singleton canonical device block ``BLK_<id>``.
        # MPAIR ownership is over those exact block ids; project them back through
        # that fixed extractor namespace to the canonical device identities.
        pair = tuple(sorted(
            raw[4:] if raw.startswith("BLK_") else raw
            for raw in (str(member) for member in members)
        ))
        if pair[0] == pair[1] or any(not member for member in pair):
            continue
        pairs.append(pair)
    return tuple(sorted(set(pairs)))


def _final_mos_matching_projection_rows(
    final_placement_certificate: Any,
) -> tuple[Any, ...]:
    """Replay the exact inventory of signed rigid MOS-axis projection rows."""

    from clay.layout.place.final_placement_state import MatchingPairAxisProjection

    raw = getattr(final_placement_certificate, "matching_pair_axis_projections", ())
    if type(raw) is not tuple:
        raise TypeError("matching-axis projection inventory must be an exact tuple")
    if any(type(row) is not MatchingPairAxisProjection for row in raw):
        raise TypeError("matching-axis projection inventory contains a foreign row")
    if raw != tuple(sorted(raw)) or len({row.pair for row in raw}) != len(raw):
        raise ValueError("matching-axis projections must be canonical and unique")
    return raw


def _repair_matching_pair_axes(
    placement: Any,
    final_placement_certificate: Any,
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Replay the final logical MOS axis into one route-feedback re-entry.

    The physical PCell coordinate is a separate footprint witness and must never
    be fed back into the solver's logical-device axis inventory.
    """

    raw_axes = getattr(placement, "matching_pair_axes", {})
    if type(raw_axes) is not dict:
        raise TypeError("repair matching-axis inventory must be an exact dict")
    result = dict(raw_axes)
    for projection in _final_mos_matching_projection_rows(
        final_placement_certificate
    ):
        result[projection.pair] = (
            projection.axis,
            projection.final_logical_coordinate4_nm / 4.0,
        )
    same = getattr(final_placement_certificate, "native_same_row_final_proof", None)
    if same is not None:
        from clay.layout.place.native_same_row_matching import NativeSameRowFinalProof
        if type(same) is not NativeSameRowFinalProof:
            raise TypeError("repair native same-row proof is foreign")
        same.__post_init__()
        for pair, (axis, coordinate4) in same.final_logical_axes4_nm:
            if pair not in result:
                raise ValueError("repair native same-row source axis is missing")
            result[pair] = (axis, coordinate4 / 4.0)
    return result


def _final_mos_matching_physical_bboxes(
    final_placement_certificate: Any,
    exact_device_bboxes: Any,
) -> Dict[Tuple[str, str], Tuple[Tuple[int, int, int, int], ...]]:
    """Bind signed MOS pair footprints to the exact stamps routed and emitted."""

    rows = _final_mos_matching_projection_rows(final_placement_certificate)
    if not rows:
        return {}
    if type(exact_device_bboxes) is not dict:
        raise TypeError(
            "matching-axis projection requires exact final stamp bbox authority"
        )
    result = {}
    for row in rows:
        actual = []
        for member in row.pair:
            bbox = exact_device_bboxes.get(member)
            if (
                type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
            ):
                raise TypeError(
                    f"matching-axis projection member {member!r} lacks exact final stamp"
                )
            actual.append(bbox)
        pair_boxes = tuple(actual)
        if pair_boxes != row.final_member_bboxes_nm:
            raise ValueError(
                "matching-axis projection does not bind exact final stamp bboxes"
            )
        result[row.pair] = pair_boxes
    return result


def _project_final_master_matching_axes(
    base_placement: Any,
    projected_devices: Mapping[str, Any],
    final_placement_certificate: Any,
    matching_pair_physical_bboxes: Mapping[
        Tuple[str, str], Tuple[Tuple[int, int, int, int], ...]
    ],
    *, native_cross_row_axes: Mapping[Tuple[str, str], Tuple[str, float]] | None = None,
    native_same_row_axes: Mapping[Tuple[str, str], Tuple[str, float]] | None = None,
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Replay certified non-MOS and rigid-block MOS matching-axis transport.

    MPAIR keeps its one-nm quantization replay below.  MOS axes are never inferred
    from final geometry: an exact final-placement row must bind the canonical source
    axis to one source master and its rigid delta; source and projected logical
    bboxes only replay that signed arithmetic.
    """

    from clay.layout.generic_placement import project_rigid_nonmos_matching_axes
    from clay.layout.place.final_placement_state import (
        MatchingPairAxisProjection,
        verify_matching_pair_projection_owners,
    )

    result = project_rigid_nonmos_matching_axes(
        base_placement, projected_devices
    )
    raw_mos_projections = getattr(
        final_placement_certificate, "matching_pair_axis_projections", ()
    )
    if type(raw_mos_projections) is not tuple:
        raise TypeError("matching-axis projection inventory must be an exact tuple")
    if any(type(row) is not MatchingPairAxisProjection for row in raw_mos_projections):
        raise TypeError("matching-axis projection inventory contains a foreign row")
    if (
        raw_mos_projections != tuple(sorted(raw_mos_projections))
        or len({row.pair for row in raw_mos_projections}) != len(raw_mos_projections)
    ):
        raise ValueError("matching-axis projections must be canonical and unique")
    verify_matching_pair_projection_owners(final_placement_certificate)
    base_devices = dict(getattr(base_placement, "devices", {}) or {})
    raw_base_axes = getattr(base_placement, "matching_pair_axes", {})
    if type(raw_base_axes) is not dict:
        raise TypeError("base matching-axis inventory must be an exact dict")
    base_mos_records: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for raw_pair, raw_axis in raw_base_axes.items():
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or any(type(member) is not str or not member for member in raw_pair)
            or raw_pair[0] == raw_pair[1]
        ):
            raise TypeError("base matching-axis pair inventory is malformed")
        if raw_pair != tuple(sorted(raw_pair)):
            raise ValueError("base matching-axis pair inventory is not canonical")
        if (
            type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or type(raw_axis[0]) is not str
            or raw_axis[0] not in {"x", "y"}
            or type(raw_axis[1]) not in {int, float}
            or not math.isfinite(float(raw_axis[1]))
            or not (4.0 * float(raw_axis[1])).is_integer()
        ):
            raise TypeError(
                "base matching-axis value must be an exact finite quarter-nm tuple"
            )
        if all(
            member in base_devices
            and type(getattr(base_devices[member], "kind", None)) is str
            and getattr(base_devices[member], "kind") in {"nmos", "pmos"}
            for member in raw_pair
        ):
            base_mos_records[raw_pair] = (
                raw_axis[0],
                int(4.0 * float(raw_axis[1])),
            )
    cross_axes = dict(native_cross_row_axes or {})
    if not set(cross_axes) <= set(base_mos_records):
        raise ValueError("native cross-row final axes name foreign base pairs")
    same_axes = dict(native_same_row_axes or {})
    same_proof = getattr(final_placement_certificate, "native_same_row_final_proof", None)
    if not set(same_axes) <= set(base_mos_records) or set(same_axes) & set(cross_axes):
        raise ValueError("native same-row final axes name foreign or multiply owned pairs")
    if same_proof is not None:
        signed_logical = dict(same_proof.source_logical_axes4_nm)
        if set(signed_logical) != set(same_axes) or any(
                signed_logical[pair] != base_mos_records[pair] for pair in same_axes):
            raise ValueError("native same-row source logical axes differ from placement")
    elif same_axes:
        raise ValueError("native same-row axes lack final proof")
    base_mos_pairs = set(base_mos_records) - set(cross_axes) - set(same_axes)
    projected_mos_records = {
        row.pair: (row.axis, row.source_logical_coordinate4_nm)
        for row in raw_mos_projections
    }
    projected_mos_pairs = set(projected_mos_records)
    missing_mos_pairs = tuple(sorted(base_mos_pairs - projected_mos_pairs))
    if missing_mos_pairs:
        raise ValueError(
            "matching-axis projection inventory omits base MOS pair: "
            f"{missing_mos_pairs}"
        )
    undeclared_mos_pairs = tuple(sorted(projected_mos_pairs - base_mos_pairs))
    if undeclared_mos_pairs:
        raise ValueError(
            "matching-axis projection inventory adds undeclared MOS pair: "
            f"{undeclared_mos_pairs}"
        )
    mismatched_mos_pairs = tuple(sorted(
        pair
        for pair in base_mos_pairs
        if projected_mos_records[pair] != base_mos_records[pair]
    ))
    if mismatched_mos_pairs:
        raise ValueError(
            "base matching-axis values do not replay MOS projections: "
            f"{mismatched_mos_pairs}"
        )
    if raw_mos_projections:
        if type(matching_pair_physical_bboxes) is not dict or set(
            matching_pair_physical_bboxes
        ) != {row.pair for row in raw_mos_projections}:
            raise TypeError(
                "matching-axis projection physical bbox inventory is incomplete"
            )
        for row in raw_mos_projections:
            if any(
                member not in base_devices or member not in projected_devices
                for member in row.pair
            ):
                raise ValueError("matching-axis projection pair lacks device geometry")
            if any(
                type(getattr(base_devices[member], "kind", None)) is not str
                or getattr(base_devices[member], "kind") not in {"nmos", "pmos"}
                for member in row.pair
            ):
                raise ValueError("matching-axis projection is not a MOS pair")
            if matching_pair_physical_bboxes[row.pair] != row.final_member_bboxes_nm:
                raise ValueError(
                    "matching-axis projection does not replay physical bboxes"
                )
            logical_views = []
            for label, devices, expected_coordinate4 in (
                ("source", base_devices, row.source_logical_coordinate4_nm),
                ("final", projected_devices, row.final_logical_coordinate4_nm),
            ):
                boxes = []
                for member in row.pair:
                    bbox = getattr(devices[member], "bbox", None)
                    if (
                        type(bbox) is not tuple
                        or len(bbox) != 4
                        or any(type(value) is not int for value in bbox)
                        or bbox[2] <= bbox[0]
                        or bbox[3] <= bbox[1]
                    ):
                        raise TypeError(
                            f"matching-axis projection {label} logical bbox is not exact"
                        )
                    boxes.append(bbox)
                pair_boxes = tuple(boxes)
                axis_indices = (0, 2) if row.axis == "x" else (1, 3)
                perpendicular_indices = (1, 3) if row.axis == "x" else (0, 2)
                coordinate4 = sum(
                    bbox[index]
                    for bbox in pair_boxes
                    for index in axis_indices
                )
                if coordinate4 != expected_coordinate4:
                    raise ValueError(
                        f"matching-axis projection {label} logical coordinate "
                        "does not replay device bboxes"
                    )
                if sum(
                    pair_boxes[0][index] for index in perpendicular_indices
                ) != sum(pair_boxes[1][index] for index in perpendicular_indices):
                    raise ValueError(
                        f"matching-axis projection {label} logical pair is not aligned"
                    )
                logical_views.append(pair_boxes)
            source_logical_boxes, final_logical_boxes = logical_views
            source_sizes = tuple(
                (bbox[2] - bbox[0], bbox[3] - bbox[1])
                for bbox in source_logical_boxes
            )
            final_sizes = tuple(
                (bbox[2] - bbox[0], bbox[3] - bbox[1])
                for bbox in final_logical_boxes
            )
            if final_sizes != source_sizes:
                raise ValueError(
                    "matching-axis projection logical bboxes change member sizes"
                )
            for source_bbox, final_bbox in zip(
                source_logical_boxes, final_logical_boxes
            ):
                if final_bbox != (
                    source_bbox[0] + row.delta_nm[0],
                    source_bbox[1] + row.delta_nm[1],
                    source_bbox[2] + row.delta_nm[0],
                    source_bbox[3] + row.delta_nm[1],
                ):
                    raise ValueError(
                        "matching-axis projection logical bboxes do not replay rigid delta"
                    )
            result[row.pair] = (
                row.axis,
                row.final_coordinate4_nm / 4.0,
            )
    projected_mos_owners = {
        frozenset(row.pair) for row in raw_mos_projections
    }
    owners = {
        frozenset(pair)
        for pair in _final_master_matching_pairs(final_placement_certificate)
        if frozenset(pair) not in projected_mos_owners
    }
    if not owners:
        result.update(cross_axes)
        result.update(same_axes)
        return result
    base_devices = dict(getattr(base_placement, "devices", {}) or {})
    for raw_pair, raw_axis in tuple(sorted(
        dict(getattr(base_placement, "matching_pair_axes", {}) or {}).items()
    )):
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or frozenset(str(member) for member in raw_pair) not in owners
            or type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or raw_axis[0] not in {"x", "y"}
        ):
            continue
        pair = tuple(str(member) for member in raw_pair)
        if any(member not in base_devices for member in pair) or any(
            member not in projected_devices for member in pair
        ):
            continue
        old_boxes = tuple(tuple(base_devices[member].bbox) for member in pair)
        new_boxes = tuple(tuple(projected_devices[member].bbox) for member in pair)
        old_sizes = tuple(
            (box[2] - box[0], box[3] - box[1]) for box in old_boxes
        )
        new_sizes = tuple(
            (box[2] - box[0], box[3] - box[1]) for box in new_boxes
        )
        if old_sizes[0] != old_sizes[1] or new_sizes != old_sizes:
            continue
        old_centres = tuple(
            ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            for box in old_boxes
        )
        new_centres = tuple(
            ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            for box in new_boxes
        )
        axis_index = 0 if raw_axis[0] == "x" else 1
        perpendicular_index = 1 - axis_index
        old_coordinate = (
            old_centres[0][axis_index] + old_centres[1][axis_index]
        ) / 2.0
        if (
            old_centres[0][perpendicular_index]
            != old_centres[1][perpendicular_index]
            or old_coordinate != float(raw_axis[1])
            or new_centres[0][perpendicular_index]
            != new_centres[1][perpendicular_index]
        ):
            continue
        translations = tuple(
            (
                new_box[0] - old_box[0],
                new_box[1] - old_box[1],
            )
            for old_box, new_box in zip(old_boxes, new_boxes)
        )
        if (
            translations[0][perpendicular_index]
            != translations[1][perpendicular_index]
            or abs(
                translations[0][axis_index] - translations[1][axis_index]
            ) > 1
        ):
            continue
        result[raw_pair] = (
            str(raw_axis[0]),
            (
                new_centres[0][axis_index] + new_centres[1][axis_index]
            ) / 2.0,
        )
    result.update(cross_axes)
    result.update(same_axes)
    return result


def _require_collision_source_pdk(certificate, program, rigid_nonmos_plans):
    exclusion_pdk = getattr(certificate, 'collision_exclusion_pdk_id', '')
    source_pdk = (program.tech_id if program is not None else
                  getattr(rigid_nonmos_plans, 'tech_id', ''))
    if exclusion_pdk and source_pdk != exclusion_pdk:
        raise ValueError('collision exclusion differs from source placement PDK')


def project_final_route_placement(
    base_placement: Any,
    placement_um: Mapping[str, Tuple[float, float]],
    geometry: Optional[Mapping[str, Any]] = None,
    *,
    final_placement_certificate: Any = None,
    intent: Any = None,
    retained_row_bindings: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Project one final route view under its existing live native row scope."""
    program = getattr(base_placement, 'row_program', None)
    materialization = getattr(base_placement, 'row_materialization', None)
    projection = getattr(materialization, 'source_projection', None)
    witness = getattr(projection, 'native_witness', None)
    if program is not None and witness is not None:
        from clay.layout.common_centroid_array import CommonCentroidArray
        from clay.layout.joint_row_geometry import JointRowGeometryBinding
        from clay.layout.mosaic_geometry import MosaicGeometry
        realizations = getattr(materialization, 'realizations', ())
        if realizations and all(
            type(row.geometry) in (CommonCentroidArray, MosaicGeometry)
            and type(getattr(row.geometry, 'joint_row_binding', None)) is JointRowGeometryBinding
            for row in realizations
        ):
            from clay.layout.place.row_replay_session import native_matching_row_scope
            from clay.layout.tech import LayoutTech
            tech = LayoutTech.resolve(program.tech_id)
            with native_matching_row_scope(program=program, materialization=materialization, tech=tech):
                return _project_final_route_placement_impl(
                    base_placement, placement_um, geometry,
                    final_placement_certificate=final_placement_certificate, intent=intent,
                    retained_row_bindings=retained_row_bindings)
    return _project_final_route_placement_impl(
        base_placement, placement_um, geometry,
        final_placement_certificate=final_placement_certificate, intent=intent,
        retained_row_bindings=retained_row_bindings)


def _project_final_route_placement_impl(
    base_placement: Any,
    placement_um: Mapping[str, Tuple[float, float]],
    geometry: Optional[Mapping[str, Any]] = None,
    *,
    final_placement_certificate: Any = None,
    intent: Any = None,
    retained_row_bindings: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Project routed device centres into a downstream ``Placement`` view.

    ``route_cell`` routes and emits from ``placement_um``/``geometry``.  Structural
    diagnostics still consume the generic :class:`Placement` interface, so leaving
    the pre-override object in ``FlowContext`` would make them measure a different
    floorplan than the emitted GDS.  Free devices move directly.  Stored row
    members move from the compact macro centre and are re-signed through the row
    final-master projector; a loose-placement certificate is never carried across.

    The exact routed stamp geometry remains separately authoritative on the route
    result; this adapter exists only for consumers that require ``Placement``.
    """

    from collections import Counter

    from clay.layout.envelope_place import _cell_geom, _reposition_device
    from clay.layout.generic_placement import (
        RowFinalPlacementResidualError,
        placement_auxiliary_polys,
        project_rigid_nonmos_matching_axes,
        project_stored_row_master,
        row_macro_member_offsets_nm,
    )

    devices = dict(getattr(base_placement, "devices", {}) or {})
    if not devices:
        return base_placement
    stamp_bboxes = {}
    if geometry is not None:
        from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm

        stamp_bboxes = _placed_stamp_bboxes_nm(
            dict(geometry), dict(placement_um)
        )
    rigid_nonmos_members = {
        str(member)
        for plan in tuple(getattr(
            getattr(base_placement, "rigid_nonmos_macro_plans", None),
            "plans",
            (),
        ) or ())
        for member in tuple(getattr(plan, "members", ()) or ())
    }
    rigid_source_bboxes = {}
    if geometry is not None and rigid_nonmos_members:
        from clay.layout.place_blocks_2d import _source_device_bboxes_nm

        # A source member can be emitted by one shared composite stamp whose
        # name differs from that member. Its logical/body centre is not the
        # origin of the full retained PCell artwork. Use the existing complete
        # source-to-emitter bbox projection for the rigid translation.
        rigid_source_bboxes = _source_device_bboxes_nm(
            dict(geometry), dict(placement_um), tuple(sorted(rigid_nonmos_members))
        )
        stamp_bboxes.update(rigid_source_bboxes)
    mos_projection_rows = _final_mos_matching_projection_rows(
        final_placement_certificate
    )
    mos_projection_members = {
        member for row in mos_projection_rows for member in row.pair
    }
    mos_projection_delta_by_member = {}
    for row in mos_projection_rows:
        for member in row.pair:
            if member in mos_projection_delta_by_member:
                raise ValueError(
                    "matching-axis projection member has multiple rigid owners"
                )
            mos_projection_delta_by_member[member] = row.delta_nm
    final_master_matching_members = {
        member
        for pair in _final_master_matching_pairs(final_placement_certificate)
        for member in pair
    } - mos_projection_members
    matching_stamp_bboxes = {}
    if geometry is not None and mos_projection_members:
        from clay.layout.place_blocks_2d import _source_device_bboxes_nm

        matching_stamp_bboxes = _source_device_bboxes_nm(
            dict(geometry),
            dict(placement_um),
            tuple(sorted(mos_projection_members)),
        )
    matching_pair_physical_bboxes = _final_mos_matching_physical_bboxes(
        final_placement_certificate,
        matching_stamp_bboxes if geometry is not None else None,
    )
    program = getattr(base_placement, "row_program", None)
    materialization = getattr(base_placement, "row_materialization", None)
    cross_final = getattr(final_placement_certificate, "native_cross_row_final_proof", None)
    cross_cp = getattr(base_placement, "native_cross_row_cp_proof", None)
    cross_final_axes = {}
    if cross_cp is not None or cross_final is not None:
        if getattr(base_placement, "ordered_row_evidence", None) is not None:
            raise ValueError("native cross-row matching with ordered-row projection is unsupported")
        if cross_cp is None or cross_final is None or geometry is None or program is None or materialization is None:
            raise ValueError("native cross-row final source or artwork proof is incomplete")
        from clay.layout.place.cross_row_matching import (
            native_cross_row_inventory_from_projection, native_cross_row_stamp_centres,
            replay_native_cross_row_final_proof,
        )
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        from clay.layout.tech import LayoutTech
        tech = LayoutTech.resolve(program.tech_id)
        inventory = native_cross_row_inventory_from_projection(program, materialization, tech)
        if inventory is None or cross_final.cp_proof != cross_cp:
            raise ValueError("native cross-row final source or CP differs")
        directions = {pair.members: base_placement.matching_pair_axes[pair.members][0]
                      for pair in inventory.pairs}
        actual_centres = native_cross_row_stamp_centres(inventory, program, materialization,
                                                       geometry, placement_um)
        signed_axes = replay_native_cross_row_final_proof(
            inventory, cross_final, directions, actual_centres, pdk_manufacturing_grid_nm(tech),
            row_program=program, row_materialization=materialization, tech=tech)
        # Replay declarations from both captured source roles, then check the
        # actual routed occurrence geometry against the original authoring source.
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.place.materialized_placement import (
            _replay_cross_row_source_bottom_alignments,
            _require_cross_row_source_alignment_authority,
        )
        projection = materialization.source_projection
        source_intent = layout_intent_from_cell_def(projection.authoring_source_cell)
        _require_cross_row_source_alignment_authority(
            source_intent.edge_alignments, inventory, projection, program)
        _replay_cross_row_source_bottom_alignments(
            source_intent.edge_alignments, inventory, geometry, placement_um)
        cross_final_axes = {pair: (axis, coordinate / 4.0)
                            for pair, (axis, coordinate) in signed_axes.items()}
    source_projection = getattr(materialization, 'source_projection', None)
    source_witness = getattr(source_projection, 'native_witness', None)
    if source_witness is not None and program is not None and geometry is not None:
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.place.row_replay_session import current_row_source_program
        from clay.layout.place.materialized_placement import (
            _require_native_left_alignment_authority, _witness_singleton_native_left_alignments,
            _replay_native_left_alignments,
        )
        from clay.layout.tech import LayoutTech
        current_row_source_program(
            source_projection,
            source_witness.cell,
            LayoutTech.resolve(program.tech_id),
            program=program,
            materialization=materialization,
        )
        source_intent = layout_intent_from_cell_def(source_projection.authoring_source_cell)
        _require_native_left_alignment_authority(source_intent, source_projection, program)
        source_left = _witness_singleton_native_left_alignments(source_intent, program, source_witness)
        _replay_native_left_alignments(source_left, program, geometry, placement_um)
    same_final = getattr(final_placement_certificate, "native_same_row_final_proof", None)
    same_final_axes = {}
    native_declared = any(getattr(row, "style", None) == "native_diffusion"
                          for row in tuple(getattr(intent, "arrays", ()) or ()))
    native_declared = native_declared or getattr(
        getattr(materialization, "source_projection", None), "native_witness", None) is not None
    if native_declared or same_final is not None:
        if geometry is None or program is None or materialization is None:
            raise ValueError("native same-row final source or artwork proof is incomplete")
        from fractions import Fraction
        from clay.layout.generic_placement import verify_stored_row_projection
        from clay.layout.place.native_same_row_matching import (
            native_same_row_inventory_from_projection, native_same_row_stamp_centres,
            native_same_row_occurrence_axes, replay_native_same_row_final_proof,
        )
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        from clay.layout.tech import LayoutTech
        tech = LayoutTech.resolve(program.tech_id)
        inventory = native_same_row_inventory_from_projection(program, materialization, tech)
        if inventory is None:
            if same_final is not None:
                raise ValueError("native same-row proof names no declared source pairs")
        else:
            if same_final is None:
                raise ValueError("native same-row final proof is missing")
            verify_stored_row_projection(base_placement)
            row_cert = getattr(base_placement, "row_final_certificate", None)
            if row_cert is None:
                raise ValueError("native same-row source row certificate is missing")
            bindings = {row.entry_id: row for row in row_cert.macro_bindings}
            source_centres = {}
            for owner, members in inventory.row_source_members:
                binding = bindings.get(owner)
                if binding is None or binding.consumed_devices != members:
                    raise ValueError("native same-row source row ownership differs")
                box = binding.bbox_nm
                source_centres[owner] = (box[0] + box[2], box[1] + box[3])
            base_axes = getattr(base_placement, "matching_pair_axes", {})
            logical_axes = {}
            for pair in inventory.pairs:
                raw = base_axes.get(pair.members)
                if type(raw) is not tuple or len(raw) != 2 or raw[0] not in ("x", "y"):
                    raise ValueError("native same-row source logical axis is missing")
                coordinate4 = Fraction(str(raw[1])) * 4
                if coordinate4.denominator != 1:
                    raise ValueError("native same-row source logical axis is not quarter-nm exact")
                logical_axes[pair.members] = (raw[0], int(coordinate4))
            source_axes = native_same_row_occurrence_axes(inventory, source_centres, logical_axes)
            actual_centres = native_same_row_stamp_centres(
                inventory, program, materialization, geometry, placement_um)
            signed_axes = replay_native_same_row_final_proof(
                inventory, same_final, source_centres, source_axes, actual_centres,
                pdk_manufacturing_grid_nm(tech), logical_axes=logical_axes)
            same_final_axes = {pair: (axis, float(coordinate / 4))
                               for pair, (axis, coordinate) in signed_axes.items()}
    consumed = set(getattr(program, "consumed_devices", ()) or ())
    row_free_physical_bboxes = {}
    if program is not None:
        if geometry is None:
            raise RowFinalPlacementResidualError(
                ("ROW_OVERRIDE_FREE_PHYSICAL_BBOX_EVIDENCE_MISSING",)
            )
        from clay.layout.place_blocks_2d import _source_device_bboxes_nm

        row_free_physical_bboxes = _source_device_bboxes_nm(
            dict(geometry),
            dict(placement_um),
            tuple(sorted(set(devices) - consumed)),
        )
    projected = {}
    for name, device in devices.items():
        if name in consumed:
            projected[name] = device
            continue
        centre = placement_um.get(str(name))
        if centre is None:
            projected[name] = device
            continue
        stamp_bbox = (
            rigid_source_bboxes.get(str(name), stamp_bboxes.get(str(name)))
            if str(name) in rigid_nonmos_members
            or str(name) in final_master_matching_members
            else None
        )
        if stamp_bbox is None:
            projected[name] = _reposition_device(
                device,
                int(round(float(centre[0]) * 1000.0)),
                int(round(float(centre[1]) * 1000.0)),
            )
            continue
        projected[name] = _project_device_to_stamp_bbox(device, stamp_bbox)
    if program is not None or materialization is not None:
        if program is None or materialization is None or geometry is None:
            raise RowFinalPlacementResidualError(
                ("ROW_OVERRIDE_ARTWORK_EVIDENCE_MISSING",)
            )
        by_entry = {item.entry_id: item for item in materialization.realizations}

        def canonical_ring(points):
            raw = tuple((int(x), int(y)) for x, y in points)
            if len(raw) > 1 and raw[0] == raw[-1]:
                raw = raw[:-1]
            if len(raw) < 3:
                raise RowFinalPlacementResidualError(
                    ("ROW_OVERRIDE_ARTWORK_DEGENERATE",)
                )
            candidates = []
            for values in (raw, tuple(reversed(raw))):
                candidates.extend(
                    values[index:] + values[:index] for index in range(len(values))
                )
            return min(candidates)

        for entry in program.entries:
            realization = by_entry.get(entry.entry_id)
            stamp = geometry.get(entry.group)
            centre = placement_um.get(entry.group)
            if realization is None or stamp is None or centre is None:
                raise RowFinalPlacementResidualError(
                    (f"ROW_OVERRIDE_ARTWORK_MISSING:{entry.entry_id}",)
                )
            cx_nm = int(round(float(centre[0]) * 1000.0))
            cy_nm = int(round(float(centre[1]) * 1000.0))
            lx0, ly0, lx1, ly1 = realization.local_bbox_nm
            expected = Counter(
                (
                    (int(poly.layer), int(poly.datatype)),
                    canonical_ring(
                        tuple(
                            placed_point_nm(
                                (float(point.x) - (lx0 + lx1) // 2) / 1000.0,
                                (float(point.y) - (ly0 + ly1) // 2) / 1000.0,
                                centre,
                            )
                            for point in poly.polygon.vertices
                        )
                    ),
                )
                for poly in realization.geometry.polygons
            )
            actual = Counter(
                (
                    (int(layer[0]), int(layer[1])),
                    canonical_ring(
                        tuple(
                            placed_point_nm(float(x), float(y), centre) for x, y in ring
                        )
                    ),
                )
                for layer, rings in (getattr(stamp, "polys_by_layer", {}) or {}).items()
                for ring in rings
            )
            if actual != expected:
                raise RowFinalPlacementResidualError(
                    (f"ROW_OVERRIDE_ARTWORK_MISMATCH:{entry.entry_id}",)
                )
            offsets = row_macro_member_offsets_nm(entry, realization)
            for name in entry.consumed_devices:
                ox, oy = offsets[name]
                projected[name] = _reposition_device(
                    devices[name], cx_nm + ox, cy_nm + oy
                )
    # A signed MOS projection owns the logical solver view independently of the
    # PCell's local origin.  Translate that logical view by the certificate's
    # exact block delta after every ordinary/row projector has run; the final
    # consumer below independently replays the same delta and the physical view.
    for name, delta_nm in mos_projection_delta_by_member.items():
        if name not in devices:
            raise ValueError(
                f"matching-axis projection member {name!r} lacks a logical device"
            )
        bbox = getattr(devices[name], "bbox", None)
        if (
            type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
        ):
            raise TypeError(
                f"matching-axis projection member {name!r} lacks an exact logical bbox"
            )
        projected[name] = _reposition_device(
            devices[name],
            (bbox[0] + bbox[2]) // 2 + delta_nm[0],
            (bbox[1] + bbox[3]) // 2 + delta_nm[1],
        )
    x0 = min(device.bbox[0] for device in projected.values())
    y0 = min(device.bbox[1] for device in projected.values())
    x1 = max(device.bbox[2] for device in projected.values())
    y1 = max(device.bbox[3] for device in projected.values())
    free_polys = [
        poly
        for name, device in projected.items()
        if name not in consumed
        for poly in (device.polys or ())
    ]
    auxiliary = placement_auxiliary_polys(base_placement) if program is not None else []
    final = replace(
        base_placement,
        devices=projected,
        axis_x=int(round((x0 + x1) / 2.0)),
        geom=_cell_geom(projected, (x0, y0, x1, y1)),
        polys=[*free_polys, *auxiliary],
        row_final_certificate=(
            None if program is not None else base_placement.row_final_certificate
        ),
        row_free_physical_bboxes=row_free_physical_bboxes,
        row_native_geometry_binding=None,
        native_cross_row_final_proof=cross_final,
        native_same_row_final_proof=same_final,
        native_row_final_transported=(base_placement.native_row_cp_frame_proof is not None),
        rigid_nonmos_native_geometry_binding=None,
        row_residuals=(() if program is not None else base_placement.row_residuals),
        matching_pair_axes=(base_placement.matching_pair_axes
            if getattr(base_placement, "ordered_row_evidence", None) is not None
            else _project_final_master_matching_axes(
            base_placement,
            projected,
            final_placement_certificate,
            matching_pair_physical_bboxes,
            native_cross_row_axes=cross_final_axes,
            native_same_row_axes=same_final_axes,
        )),
        matching_pair_physical_bboxes=matching_pair_physical_bboxes,
        note=(
            str(getattr(base_placement, "note", ""))
            + " | final route placement projected from canonical placement_override"
        ),
    )
    if getattr(base_placement, "ordered_row_evidence", None) is not None:
        from clay.layout.place.ordered_mirror_rows import (
            project_ordered_row_placement,
            verify_ordered_row_route_artwork,
        )

        certified_compaction = (
            final_placement_certificate is not None
            and (
                intent is None
                or declared_craft_compaction_requested(None, intent)
            )
        )
        final = project_ordered_row_placement(
            base_placement,
            final,
            final_placement_certificate=(
                final_placement_certificate if certified_compaction else None
            ),
            intent=intent,
        )
        if geometry is not None:
            verify_ordered_row_route_artwork(final, geometry, placement_um)
    from clay.layout.generic_placement import (
        attach_physical_macro_projection,
        verify_rigid_nonmos_macro_projection,
    )

    if program is not None or rigid_nonmos_members:
        binding_field = ("row_native_geometry_binding" if program is not None
                         else "rigid_nonmos_native_geometry_binding")
        retained_native = (
            dict(retained_row_bindings).get(binding_field)
            if retained_row_bindings is not None
            else None
        )
        if retained_native is None:
            retained_native = getattr(base_placement, binding_field, None)
        if retained_native is not None:
            from clay.layout.place.row_native_geometry import RowNativeGeometryBinding

            if type(retained_native) is not RowNativeGeometryBinding:
                raise TypeError("final row projection requires exact native geometry binding")
        if (retained_native is not None
                and retained_native.certificate is final_placement_certificate):
            # This proof covers the original device stamps. Flow-added rails
            # and guards remain in the full geometry for emission and signoff.
            # Every original stamp and position is still required for replay.
            retained_native.require_actual_route_geometry(
                geometry={name: geometry[name] for name in retained_native.stamp_names},
                placement_um={name: placement_um[name] for name in retained_native.stamp_names},
                intent=intent,
            )
            final = replace(final, **{binding_field: retained_native})
        elif (getattr(final_placement_certificate, "full2d_compaction_graph", None) is not None
                and getattr(final_placement_certificate, "materialized_block_source_members", ())):
            # Carry the complete native proof before any envelope predicate.
            # A passive bank can contain a foreign envelope in an internal gap
            # even when every individual row/free bbox is disjoint.
            if intent is None:
                raise ValueError("native row geometry requires the actual layout intent")
            from clay.layout.place.row_native_geometry import RowNativeGeometryBinding

            _require_collision_source_pdk(final_placement_certificate, program,
                getattr(base_placement, 'rigid_nonmos_macro_plans', None))
            native_binding = RowNativeGeometryBinding.from_route_geometry(
                certificate=final_placement_certificate, geometry=geometry,
                placement_um=placement_um, intent=intent,
            )
            final = replace(final, **{binding_field: native_binding})
    if program is not None:
        # Without complete native source evidence the historical solid-envelope
        # guard remains mandatory; incomplete evidence never grants an exemption.
        projected_final = project_stored_row_master(final)
    else:
        projected_final = final
    result = attach_physical_macro_projection(projected_final)
    if getattr(base_placement, "physical_macro_graph", None) is not None:
        verify_rigid_nonmos_macro_projection(
            result,
            exact_device_bboxes=(stamp_bboxes if geometry is not None else None),
        )
    return result


def final_route_device_polygon_count(geometry: Mapping[str, Any]) -> int:
    """Count exact routed device-stamp polygons, excluding flow-owned frames/rails."""

    total = 0
    for stamp in geometry.values():
        if str(getattr(stamp, "source", "")) in {
            "group_guard_ring",
            "power_rail",
            "power_plane",
        }:
            continue
        total += sum(
            len(rings)
            for rings in (getattr(stamp, "polys_by_layer", {}) or {}).values()
        )
    return total


__all__ = [
    "DeclaredCompactOverride",
    "DeclaredCompactionError",
    "UnsupportedCraftCompactionError",
    "declared_craft_compaction_requested",
    "materialize_declared_compact_override",
    "route_with_declared_compaction",
    "project_final_route_placement",
    "final_route_device_polygon_count",
]
