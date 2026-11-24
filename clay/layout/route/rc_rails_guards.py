"""Guard-ring materialization, routing envelopes, anchor edge pins, and power-rail materialization."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route import constraints as _C
from clay.layout.route.engine import NetRoute, Obstacle
from clay.layout.route.rc_bridge import (
    _manufacturing_grid_bbox_access,
    _normalize_materialized_geometry_to_manufacturing_grid,
)
from clay.layout.route.rc_devices import (
    RoutePlanningStop,
    _BridgedIntent,
    _BringupRailConstructionEvidence,
    _NM_PER_UM,
    _XY,
    _cell_bbox_um,
    _metal_index_of,
)
from clay.layout.route.rc_placement_evidence import FinalPlacementBindingError
from clay.layout.route.rc_planning import _compile_edge_rail_specs_from_intent
from clay.layout.tech import LayoutTech


def _validate_guard_ring_member_padding(
    raw_padding: Any,
    *,
    placement_override: Any,
    final_placement_certificate: Any,
) -> Tuple[Tuple[str, int], ...]:
    """Validate top or bottom guard-ring member padding rows (deduplicated helper)."""
    ring_pad_error = (
        "guard-ring member padding must be unique non-negative integer facts"
    )
    try:
        raw_rows = tuple(raw_padding)
    except TypeError:
        raise FinalPlacementBindingError(ring_pad_error) from None
    malformed = any(
        not isinstance(row, (tuple, list)) or len(row) != 2
        for row in raw_rows
    )
    rows = () if malformed else tuple((row[0], row[1]) for row in raw_rows)
    if (
        malformed
        or any(
            not isinstance(member, str)
            or not member
            or type(pad_nm) is not int
            or pad_nm < 0
            for member, pad_nm in rows
        )
        or len({member for member, _pad_nm in rows}) != len(rows)
    ):
        raise FinalPlacementBindingError(ring_pad_error)
    if rows and (
        placement_override is None or final_placement_certificate is None
    ):
        raise FinalPlacementBindingError(
            "guard-ring member padding requires its exact override certificate"
        )
    return rows


def _effective_guard_ring_intents(
    intent: Any,
    *,
    shared_well_guard_envelope_plans: Any = (),
    shared_well_guard_ring_inventory: Any = None,
) -> Tuple[Any, ...]:
    """Replay an optional shared-well lowering against source ring ownership."""

    source_rings = getattr(intent, "guard_rings", ())
    if type(source_rings) is not tuple:
        raise TypeError("source guard-ring inventory must be one exact tuple")
    if (
        shared_well_guard_envelope_plans == ()
        and shared_well_guard_ring_inventory is None
    ):
        return source_rings
    from clay.layout.well_island_sharing import (
        SharedWellGuardRingInventory,
        compile_shared_well_guard_ring_inventory,
    )

    if type(shared_well_guard_envelope_plans) is not tuple:
        raise TypeError("shared guard envelope plans must be one exact tuple")
    if type(shared_well_guard_ring_inventory) is not SharedWellGuardRingInventory:
        raise TypeError("route requires one exact shared guard-ring inventory")
    replayed = compile_shared_well_guard_ring_inventory(
        shared_well_guard_envelope_plans,
        source_rings,
    )
    if replayed != shared_well_guard_ring_inventory:
        raise ValueError("shared guard-ring inventory does not replay source intent")
    return replayed.effective_guard_rings


def _append_guard_ring_tap_pins(pins_by_net, ring_taps, *, route_power=True,
                                edge_rail_nets=()):
    """Use one native tap-to-Pin adapter in placement and routing models."""
    from clay.layout.route.engine import Pin as _RingPin

    for tap in ring_taps:
        net = str(tap["net"])
        if not net or (not route_power and net.strip().lower() in edge_rail_nets):
            continue
        access_xy_nm = tuple(int(v) for v in tap["access_xy_nm"])
        pins_by_net.setdefault(net, []).append(
            _RingPin(net=net, device=tap["ring"],
                     terminal=str(tap.get("terminal", "tap") or "tap"),
                     xy_um=tuple(v / _NM_PER_UM for v in access_xy_nm),
                     layer=tuple(int(v) for v in tap["layer"]),
                     xy_nm=access_xy_nm)
        )


def _guard_ring_taps_from_stamps(geometry, placement_um):
    """Replay native terminal facts, never infer ring ownership from net names."""
    from clay.layout.placed_geometry import placed_point_nm

    taps = []
    for name, stamp in geometry.items():
        if str(getattr(stamp, "source", "")) != "group_guard_ring":
            continue
        native = getattr(stamp, "guard_ring_tap_terminals", None)
        if not native:
            raise ValueError(f"guard ring {name!r} lacks native tap terminal facts")
        position = placement_um[name]
        for record in native:
            if record["ring"] != name:
                raise ValueError(f"guard ring {name!r} has a foreign terminal owner")
            tap = dict(record)
            x, y = record["access_xy_nm"]
            tap["access_xy_nm"] = placed_point_nm(x / _NM_PER_UM, y / _NM_PER_UM, position)
            x0, y0, x1, y1 = record["bbox_nm"]
            tap["bbox_nm"] = (*placed_point_nm(x0 / _NM_PER_UM, y0 / _NM_PER_UM, position),
                              *placed_point_nm(x1 / _NM_PER_UM, y1 / _NM_PER_UM, position))
            taps.append(tap)
    return tuple(taps)


def _materialize_guard_rings(
    guard_ring_intents: Sequence[Any],
    bridged_intent: Any,
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    tech: LayoutTech,
    tech_id: str,
    member_ring_pad_top_nm: Optional[Dict[str, int]] = None,
    member_ring_pad_bottom_nm: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Stamp each declared ring into ``geometry`` as a placement-absolute ring stamp
    (``GRING{i}`` at position (0,0)), replicating the anchor_ota stamp pattern, and
    return the routable tap-access list (one ``{ring, net, bbox_nm}`` per ring).

    The rings are built by the SHARED :func:`group_guard_rings` core, clustered by
    the DECLARED member groups (``island_clusters``) -- so the drawn polarity + tap
    net come from pure design data (device kind + body net) and the declared ``well``
    / ``tie`` become the checker's OBLIGATION, verified against the drawn ring. NO
    per-device / per-net branch."""
    from clay.layout.extract.library_device import CellGeometry as _CG
    from clay.layout.extract.realize_render import (
        group_guard_rings,
        group_ring_tap_terminals,
    )

    # A row macro may omit its local PMOS N-well tap only when one canonical
    # outer ring owns that exact tap obligation.  The claim is carried through
    # the certified row projection, then replayed here against both the source
    # declaration and the PDK materializer output.  It is never inferred from a
    # nearby same-net polygon or from placement coordinates.
    owner_claims: List[Tuple[str, str, str, Tuple[str, ...], Any]] = []
    raw_claims: List[Tuple[str, str, str, Tuple[str, ...]]] = []
    from clay.layout.device_cell import _validate_body_tap_owner_sha256

    for stamp_name, stamp in geometry.items():
        raw_owner = getattr(stamp, "body_tap_owner_sha256", "")
        if raw_owner == "":
            continue
        try:
            owner = _validate_body_tap_owner_sha256(raw_owner)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"body-tap owner on {stamp_name!r} is not canonical"
            ) from exc
        body_net = getattr(stamp, "body_net", "")
        members = getattr(stamp, "composite_members", ())
        if (
            type(body_net) is not str
            or not body_net
            or body_net != body_net.strip()
            or type(members) is not tuple
            or not members
            or any(
                type(member) is not str
                or not member
                or member != member.strip()
                for member in members
            )
            or len(set(members)) != len(members)
        ):
            raise ValueError(
                f"body-tap owner on {stamp_name!r} has malformed body/member authority"
            )
        raw_claims.append((str(stamp_name), owner, body_net, tuple(members)))

    if raw_claims:
        from clay.layout.intent import GuardRingIntent

        if type(guard_ring_intents) is not tuple:
            raise ValueError(
                "body-tap owner replay requires an exact guard-ring declaration tuple"
            )
        declarations_by_sha: Dict[str, List[GuardRingIntent]] = {}
        for raw_declaration in guard_ring_intents:
            if type(raw_declaration) is not GuardRingIntent:
                raise ValueError(
                    "body-tap owner replay requires exact GuardRingIntent declarations"
                )
            declaration = GuardRingIntent(
                members=raw_declaration.members,
                well=raw_declaration.well,
                tie_net=raw_declaration.tie_net,
                hug=raw_declaration.hug,
                role=raw_declaration.role,
                enclose=raw_declaration.enclose,
            )
            declarations_by_sha.setdefault(declaration.sha256, []).append(declaration)
        seen_owners: set[str] = set()
        for stamp_name, owner, body_net, members in raw_claims:
            matches = declarations_by_sha.get(owner, ())
            if len(matches) != 1 or owner in seen_owners:
                raise ValueError(
                    f"body-tap owner {owner!r} on {stamp_name!r} is not unique and present"
                )
            seen_owners.add(owner)
            declaration = matches[0]
            if (
                tuple(sorted(declaration.members)) != tuple(sorted(members))
                or declaration.well != "n"
                or declaration.tie_net != body_net
                or declaration.hug != "tight"
                or declaration.role != "tap"
                or declaration.enclose
            ):
                raise ValueError(
                    f"body-tap owner {owner!r} on {stamp_name!r} changed authority"
                )
            owner_claims.append(
                (stamp_name, owner, body_net, tuple(sorted(members)), declaration)
            )

    # This is the executable declaration boundary, not a parser-side hint.  The
    # existing builder's zero extra clearance is exactly its PDK-derived minimum,
    # so ``hug=tight`` selects that rule surface explicitly.  Unknown spacing and
    # as-yet-unimplemented absolute enclosure requests fail closed instead of
    # silently inheriting the legacy default.
    for declaration in guard_ring_intents:
        hug = str(getattr(declaration, "hug", "") or "").strip().lower()
        role = str(getattr(declaration, "role", "") or "").strip().lower()
        enclose = str(getattr(declaration, "enclose", "") or "").strip().lower()
        if hug not in ("", "tight"):
            raise ValueError(f"unsupported @guard_ring hug={hug!r}")
        if role not in ("", "tap", "isolation"):
            raise ValueError(f"unsupported @guard_ring role={role!r}")
        if enclose:
            raise ValueError(
                f"unsupported @guard_ring enclose={enclose!r}; "
                "no absolute enclosure materializer is registered"
            )

    islands = [tuple(gr.members) for gr in guard_ring_intents]
    # The declared well/tie per member (the @guard_ring records carry it, the bridged
    # intent's __slots__ cannot) -- threaded so a PASSIVE island (resistor/cap, no MOS
    # kind) rings on its declared well instead of being dropped.
    member_well_tie: Dict[str, Tuple[str, str]] = {}
    for gr in guard_ring_intents:
        well = str(getattr(gr, "well", "") or "")
        tie = str(getattr(gr, "tie_net", getattr(gr, "tie", "")) or "")
        for m in getattr(gr, "members", ()) or ():
            member_well_tie[str(m)] = (well, tie)
    rings, _notes = group_guard_rings(
        geometry, placement_um, bridged_intent, tech_id,
        extra_clearance_nm=0,
        island_clusters=islands or None,
        member_well_tie=member_well_tie or None,
        member_ring_pad_top_nm=member_ring_pad_top_nm or None,
        member_ring_pad_bottom_nm=member_ring_pad_bottom_nm or None,
    )

    def expanded_members(materialized_ring: Any) -> Tuple[str, ...]:
        expanded: List[str] = []
        for member in tuple(getattr(materialized_ring, "members", ()) or ()):
            composite = tuple(
                getattr(geometry.get(str(member)), "composite_members", ()) or ()
            )
            expanded.extend(str(value) for value in (composite or (member,)))
        return tuple(sorted(set(expanded)))

    # ``role`` is executable too: a tap ring must be a MOS body-tap ring and an
    # isolation ring must enclose a passive island.  Resolve this from the rings
    # the PDK builder actually returned, not from parser spelling.  A declaration
    # that split or vanished is not silently accepted as partial materialization.
    for declaration in guard_ring_intents:
        declared = tuple(
            sorted(set(str(value) for value in declaration.members))
        )
        matches = tuple(
            ring for ring in rings if expanded_members(ring) == declared
        )
        if len(matches) != 1:
            raise ValueError(
                f"@guard_ring members {declared} materialized as "
                f"{len(matches)} exact rings"
            )
        role = str(getattr(declaration, "role", "") or "").strip().lower()
        materialized_is_passive = str(getattr(matches[0], "kind", "")) == "passive"
        if role == "tap" and materialized_is_passive:
            raise ValueError("@guard_ring role='tap' cannot materialize a passive ring")
        if role == "isolation" and not materialized_is_passive:
            raise ValueError("@guard_ring role='isolation' cannot materialize a MOS ring")

    for stamp_name, owner, body_net, members, _declaration in owner_claims:
        matches = tuple(
            ring for ring in rings if expanded_members(ring) == members
        )
        if (
            len(matches) != 1
            or str(getattr(matches[0], "kind", "")) == "passive"
            or str(getattr(matches[0], "well", "")) != "n"
            or str(getattr(matches[0], "tap_net", "")) != body_net
        ):
            raise ValueError(
                f"body-tap owner {owner!r} on {stamp_name!r} did not materialize "
                "one exact same-net N-well tap ring"
            )

    taps = group_ring_tap_terminals(rings)
    out: List[Dict[str, Any]] = []
    for i, gr in enumerate(rings):
        key = f"GRING{i}"
        ring_record_start = len(out)
        exact_members = expanded_members(gr)
        matching_declarations = tuple(
            declaration
            for declaration in guard_ring_intents
            if tuple(sorted(getattr(declaration, "members", ()) or ()))
            == exact_members
        )
        if len(matching_declarations) != 1:
            raise ValueError(
                f"materialized guard ring {i} lacks one exact declaration owner"
            )
        declaration = matching_declarations[0]
        polys: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
        for lp in gr.ring.polygons:
            polys.setdefault((int(lp.layer), int(lp.datatype)), []).append(
                [(float(v.x) / _NM_PER_UM, float(v.y) / _NM_PER_UM)
                 for v in lp.polygon.vertices]
            )
        cg = _CG(name=key, kind="guard_ring", ref_centroid=(0.0, 0.0),
                 polys_by_layer=polys)
        cg.source = "group_guard_ring"
        cg.guard_ring_intent_sha256 = str(declaration.sha256)
        cg.guard_ring_declared_members = exact_members
        geometry[key] = cg
        placement_um[key] = (0.0, 0.0)
        _normalize_materialized_geometry_to_manufacturing_grid(
            {key: cg}, {key: placement_um[key]}, tech
        )
        # Resolve a composite array stamp back to the declaration's member
        # devices.  The topology compiler can then pair taps to GuardRingIntent
        # structurally, independent of group_guard_rings output ordering and net
        # spelling.  Keep the materialized member keys too for audit provenance.
        ring_taps = [tap for tap in taps if int(tap.get("ring_index", -1)) == i]
        if not ring_taps:
            raise ValueError(f"materialized guard ring {i} exposes no routable tap")
        materialized_members = tuple(str(m) for m in gr.members)
        declared_members: List[str] = []
        for member in materialized_members:
            composite = tuple(
                getattr(geometry.get(member), "composite_members", ()) or ()
            )
            declared_members.extend(str(m) for m in (composite or (member,)))
        for tap in ring_taps:
            terminal = str(tap.get("terminal", "tap") or "tap")
            bbox_nm, access_xy_nm = _manufacturing_grid_bbox_access(
                tuple(int(v) for v in tap["bbox_nm"]), tech
            )
            out.append({
                "ring": key,
                "terminal": terminal,
                "net": str(tap["net"]),
                "bbox_nm": bbox_nm,
                "access_xy_nm": access_xy_nm,
                "layer": tuple(int(v) for v in tap["layer"]),
                "well": str(tap.get("well", "") or ""),
                "members": materialized_members,
                # Auxiliary deep/substrate taps are routable obligations but do
                # not masquerade as the declared body-ring identity.
                "declared_members": (
                    tuple(sorted(set(declared_members)))
                    if terminal == "tap"
                    else ()
                ),
            })
        # These are the actual native tap accesses in this stamp's local frame.
        # Preserve them with the artwork so source placement can seed precisely
        # the same M1 ownership as the downstream router, including auxiliary taps.
        cg.guard_ring_tap_terminals = tuple(dict(tap) for tap in out[ring_record_start:])
    return out



def _routing_envelope_from_geometry(
    placement_um: Dict[str, _XY], geometry: Dict[str, Any]
) -> Any:
    """Final pre-route rectangle in the canonical placed integer-nm currency.

    The final-placement certificate is built from ``_placed_stamp_bboxes_nm``.
    Re-observing the same vertices as a float-um union and quantizing only the
    extrema can differ by one nanometre at an exact half-nm tie, so the routing
    envelope must consume that same per-vertex observer rather than duplicate it.
    """

    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    from clay.layout.route.floorplan_containment import RoutingEnvelope

    stamp_bboxes = tuple(
        _placed_stamp_bboxes_nm(geometry, placement_um).values()
    )
    if not stamp_bboxes:
        raise ValueError("routing envelope requires non-empty final geometry")
    return RoutingEnvelope(
        min(bbox[0] for bbox in stamp_bboxes),
        min(bbox[1] for bbox in stamp_bboxes),
        max(bbox[2] for bbox in stamp_bboxes),
        max(bbox[3] for bbox in stamp_bboxes),
    )


def _routing_envelope_from_final_placement(
    geometry_envelope: Any,
    final_placement_certificate: Any,
) -> Any:
    """Replay certified empty bottom/right edge-rail bands without enlarging other edges."""

    if final_placement_certificate is None:
        return geometry_envelope
    from clay.layout.route.floorplan_containment import RoutingEnvelope

    raw_bbox = getattr(final_placement_certificate, "bbox_nm", None)
    if (
        type(raw_bbox) is not tuple
        or len(raw_bbox) != 4
        or any(type(value) is not int for value in raw_bbox)
    ):
        raise FinalPlacementBindingError(
            "final placement floorplan bbox must be one exact integer tuple"
        )
    certified = RoutingEnvelope.coerce(raw_bbox)
    raw_keepout = getattr(
        final_placement_certificate, "declared_bottom_edge_keepout_nm", 0
    )
    if type(raw_keepout) is not int or raw_keepout < 0:
        raise FinalPlacementBindingError(
            "final placement bottom-edge keepout must be exact non-negative nm"
        )
    edge_rail_reservation = getattr(
        final_placement_certificate, "edge_rail_reservation", None
    )
    raw_right_margin = (
        getattr(edge_rail_reservation, "right_margin_nm", 0)
        if edge_rail_reservation is not None
        else 0
    )
    if raw_right_margin is None:
        raw_right_margin = 0
    if type(raw_right_margin) is not int or raw_right_margin < 0:
        raise FinalPlacementBindingError(
            "final placement right-edge rail margin must be exact non-negative nm"
        )
    raw_top_margin = (
        getattr(edge_rail_reservation, "top_margin_nm", 0)
        if edge_rail_reservation is not None
        else 0
    )
    if raw_top_margin is None:
        raw_top_margin = 0
    if type(raw_top_margin) is not int or raw_top_margin < 0:
        raise FinalPlacementBindingError(
            "final placement top-edge rail margin must be exact non-negative nm"
        )
    for field in (
        "bottom_edge_keepout_carrier_blocks",
        "bottom_edge_keepout_clear_blocks",
    ):
        raw_names = getattr(final_placement_certificate, field, ())
        if type(raw_names) is not tuple:
            raise FinalPlacementBindingError(
                "final placement bottom-edge ownership must be exact tuples"
            )
    core = geometry_envelope.bbox_nm
    expected = (
        core[0],
        (0 if raw_keepout > 0 else core[1]),
        core[2] + raw_right_margin,
        core[3] + raw_top_margin,
    )
    if certified.bbox_nm != expected:
        raise FinalPlacementBindingError(
            "final placement floorplan may add only its declared bottom/right/top edge-rail bands: "
            f"certificate={certified.bbox_nm}, geometry={core}, "
            f"keepout_nm={raw_keepout}, right_margin_nm={raw_right_margin}, "
            f"top_margin_nm={raw_top_margin}"
        )
    if raw_keepout > 0:
        allowed = set(final_placement_certificate.bottom_edge_keepout_carrier_blocks)
        allowed.update(final_placement_certificate.bottom_edge_keepout_clear_blocks)
        if not allowed and core[1] < raw_keepout:
            raise FinalPlacementBindingError(
                "empty bottom-edge rail band is occupied by uncertified geometry"
            )
    return certified


def _materialize_declared_anchor_edge_pins(
    intent: Any,
    pins_by_net: Dict[str, List[Any]],
    routing_intent: Any,
    tech: Any,
    routing_envelope: Any,
    *,
    square_wire_ends: bool = True,
) -> Tuple[Pin, ...]:
    """Lower net ``@anchor`` declarations to PDK-derived upper-metal edge pins."""

    from clay.layout.route.engine import Pin as _AnchorPin
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    declarations = tuple(getattr(intent, "anchors", ()) or ())
    if not declarations:
        return ()
    from clay.layout.route.topology_compile import (
        _pdk_transport_layer_closure,
        _routing_intent_allowed_layers,
        _routing_intent_trunk_layers,
    )

    closure = _pdk_transport_layer_closure(tech)
    hard_layers = _routing_intent_allowed_layers(routing_intent, tech)
    trunk_layers = _routing_intent_trunk_layers(routing_intent, tech)

    # The edge rail materializer consumes these same compiled widths/layers.
    # Its future fixed metal is already an endpoint-placement obligation.
    anchor_edge_rails = _compile_edge_rail_specs_from_intent(
        intent, routing_intent, tech,
    )

    created: List[Pin] = []
    for declaration in declarations:
        edge = str(getattr(declaration, "edge", "") or "").strip().lower()
        target = str(getattr(declaration, "target", "") or "")
        normalized = target.strip().rstrip("!").casefold()
        matches = tuple(
            net
            for net in pins_by_net
            if net.strip().rstrip("!").casefold() == normalized
        )
        if edge not in {"left", "right"} or not matches:
            continue  # device/group anchors are already placement-owned
        if len(matches) != 1:
            raise ValueError(f"anchor net {target!r} has ambiguous routed aliases")
        net = matches[0]
        hard = hard_layers.get(net)
        preferred = tuple(
            layer_n
            for layer_n in trunk_layers.get(net, ())
            if str(tech.metal_preferred_axis(layer_n)).upper() == "H"
        )
        eligible = hard or tuple(dict.fromkeys(preferred + closure[1:]))
        horizontal = tuple(
            layer_n
            for layer_n in eligible
            if str(tech.metal_preferred_axis(layer_n)).upper() == "H"
        )
        if not horizontal:
            raise ValueError(
                f"anchor net {target!r} has no allowed upper horizontal metal"
            )
        layer_n = horizontal[0]
        layer_key = tuple(tech.metal_key(layer_n))
        width_decl = dict(getattr(routing_intent, "widths", {}) or {}).get(net)
        width_nm = max(
            int(tech.min_width(f"metal{layer_n}") or 0),
            int(round(1000.0 * float(getattr(width_decl, "value", 0.0) or 0.0))),
        )
        if width_nm <= 0:
            raise ValueError(f"anchor net {target!r} has no PDK-legal wire width")
        half_width = (width_nm + 1) // 2
        grid_nm = pdk_manufacturing_grid_nm(tech)
        if grid_nm <= 0 or half_width % grid_nm:
            raise ValueError(f"anchor net {target!r} has no legal on-grid wire domain")
        source = (min if edge == "left" else max)(
            pins_by_net[net],
            key=lambda pin: int(
                pin.xy_nm[0]
                if pin.xy_nm is not None
                else round(float(pin.xy_um[0]) * _NM_PER_UM)
            ),
        )
        source_y = int(
            source.xy_nm[1]
            if source.xy_nm is not None
            else round(float(source.xy_um[1]) * _NM_PER_UM)
        )
        # The negotiated Gridless producer emits a square half-width end cap;
        # exact resource-plan tracks instead stop flush at their endpoint.
        # Put the access center inside by the ACTUAL producer end extension,
        # so emitted route metal, rather than a virtual pin, reaches the edge.
        # This does not enlarge the certified floorplan or clip any route.
        end_extension = half_width if square_wire_ends else 0
        x_nm = (
            int(routing_envelope.x0) + end_extension
            if edge == "left"
            else int(routing_envelope.x1) - end_extension
        )
        lower_y = -(-(int(routing_envelope.y0) + half_width) // grid_nm) * grid_nm
        upper_y = ((int(routing_envelope.y1) - half_width) // grid_nm) * grid_nm
        for rail in anchor_edge_rails:
            if (int(rail.layer_n) != layer_n
                    or str(rail.net).strip().rstrip("!").casefold() == normalized):
                continue
            spacing_nm = int(tech.min_space(f"metal{layer_n}"))
            if spacing_nm < 0:
                raise ValueError(f"anchor net {target!r} has invalid PDK spacing")
            clearance_nm = int(rail.width_nm) + spacing_nm + half_width
            if rail.actual_axis == "H" and rail.edge == "bottom":
                lower_y = max(lower_y,
                    -(-(int(routing_envelope.y0) + clearance_nm) // grid_nm) * grid_nm)
            elif rail.actual_axis == "H" and rail.edge == "top":
                upper_y = min(upper_y,
                    ((int(routing_envelope.y1) - clearance_nm) // grid_nm) * grid_nm)
            elif rail.actual_axis == "V" and rail.edge == edge:
                raise ValueError(f"anchor net {target!r} has no legal edge-rail wire domain")
        if (lower_y > upper_y or x_nm % grid_nm
                or int(routing_envelope.x1) - int(routing_envelope.x0) < 2 * half_width):
            raise ValueError(f"anchor net {target!r} has no legal on-grid wire domain")
        source_grid_y = ((2 * source_y + grid_nm) // (2 * grid_nm)) * grid_nm
        y_nm = min(max(source_grid_y, lower_y), upper_y)
        pin = _AnchorPin(
            net=net,
            device="__declared_anchor__",
            terminal=edge,
            xy_um=(x_nm / _NM_PER_UM, y_nm / _NM_PER_UM),
            layer=layer_key,
            xy_nm=(x_nm, y_nm),
            requires_wire_endpoint=bool(square_wire_ends),
        )
        pins_by_net[net].append(pin)
        created.append(pin)
    return tuple(created)


def _legacy_materialize_power_rails(
    trunks: Sequence[Any],
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    tech: LayoutTech,
    *,
    pins_by_net: Optional[Dict[str, List[Any]]] = None,
) -> List[Dict[str, Any]]:
    """Materialize each EDGE-rail ``@trunk(net, layer=Mk, orient=H, width=W, edge=
    top|bottom)`` into ``geometry`` as a full-cell-width horizontal Mk rail along the
    named cell edge (the reference power-rail structure: VDD a wide M2 rail on the
    TOP edge, GND on the BOTTOM).

    Stamped as a ``PRAIL_<net>`` :class:`CellGeometry` (same emit path as guard
    rings, so the ONE emit boundary draws it). Returns one ``{rail, net, layer,
    center_um, bbox_um}`` per rail so the caller can inject a same-net rail Pin: the
    rail then seeds the gridless upper-artwork model as OWNED metal (foreign nets
    route around it; the supply net drops onto it) and is a REAL tie, not dead
    artwork. Coordinate-free: edge + width + layer come from the declaration,
    never a reference coordinate. Canonical Gridless routes use the independently
    certified rail plan and emit it through :func:`_materialize_power_rails` below.
    """
    from clay.layout.extract.library_device import CellGeometry as _CG

    bbox = _cell_bbox_um(placement_um, geometry)
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    out: List[Dict[str, Any]] = []
    for t in trunks or ():
        net = str(getattr(t, "net", "") or "")
        edge = str(getattr(t, "edge", "") or "").strip().lower()
        if not net or edge not in ("top", "bottom"):
            continue
        k = _metal_index_of(str(getattr(t, "layer", "") or "metal2"), tech) or 2
        try:
            mk = tuple(int(v) for v in tech.metal_key(k))
        except Exception:
            continue
        w_um = float(getattr(t, "width_um", 0.0) or 0.0)
        if w_um <= 0.0:
            w_um = 8.0  # default supply-rail width when the .clay declares none
        offset_um = float(getattr(t, "offset_um", 0.0) or 0.0)
        rx0, rx1 = x0, x1
        if edge == "top":
            ry0, ry1 = y1 - w_um - offset_um, y1 - offset_um
        else:
            ry0, ry1 = y0 + offset_um, y0 + offset_um + w_um
        rail = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
        key = f"PRAIL_{net}"
        cg = _CG(name=key, kind="power_rail", ref_centroid=(0.0, 0.0),
                 polys_by_layer={mk: [rail]})
        cg.source = "power_rail"
        geometry[key] = cg
        placement_um[key] = (0.0, 0.0)
        out.append({
            "rail": key, "net": net, "layer": mk,
            "center_um": ((rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5),
            "bbox_um": (rx0, ry0, rx1, ry1),
        })
    return out


def _materialize_power_rails(
    pool: Any,
    plan: Any,
    replay_receipt: Any,
    tech: LayoutTech,
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    pins_by_net: Dict[str, List[Any]],
) -> Tuple[List[Dict[str, Any]], Tuple[Any, ...]]:
    """Pure emitter for one already-verified rail pool/plan.

    Candidate discovery, PDK lookup, scoring, and fallback are deliberately absent.
    The selected integer-nm bbox and access point are copied exactly once into the
    canonical geometry/pin stores.  Duplicate invocation is a construction error.
    """
    from clay.layout.route.rail_plan import _verify_rail_placement_optimality_replay

    _verify_rail_placement_optimality_replay(
        pool, plan, replay_receipt, tech=tech
    )
    demands = {item.demand_id: item for item in pool.demands}
    prepared: List[_PhysicalRailWriteFact] = []
    for selected in plan.selected:
        demand = demands[selected.demand_id]
        identity_payload = json.dumps(
            {
                "candidate_pool_hash": pool.sha256(),
                "candidate_id": selected.candidate_id,
                "demand_id": demand.demand_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        content_id = "PRAIL_" + hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest()[:20]
        prepared.append(_PhysicalRailWriteFact(
            demand=demand,
            content_id=content_id,
            bbox_nm=selected.bbox_nm,
            access_xy_nm=selected.access_xy_nm,
            candidate_id=selected.candidate_id,
        ))
    _require_no_new_wide_rail_spacing(tuple(prepared), tech, placement_um, geometry)
    return _write_physical_power_rails(
        tuple(prepared), placement_um, geometry, pins_by_net=pins_by_net
    )


def _require_no_new_wide_rail_spacing(facts, tech, placement_um, geometry):
    """Check the emitted rail batch against actual merged metal before writes.

    Reservation uses provable rectangular rail clearances to let placement
    move source artwork.  This final literal observer also covers narrow rails
    entering existing wide metal, attachments, and other nonrectangular cases.
    Compare with the pre-rail world so unrelated pre-existing markers are not
    attributed to this transaction.
    """
    query = getattr(tech, 'width_spacing_entries', None)
    if not callable(query):
        return
    layers = {}
    for fact in facts:
        demand = fact.demand
        name = f'metal{demand.layer_n}'
        if any(getattr(band, 'opening_nm', None) is not None for band in query(name)):
            key = tuple(demand.layer_key)
            if key in layers and layers[key] != name:
                raise ValueError('wide rail layers have conflicting metal identities')
            layers[key] = name
    if not layers:
        return
    import klayout.db as kdb
    from clay.layout.owned_artwork import _stamp_records
    from clay.layout.route.wide_metal_spacing import wide_spacing_checks
    from clay.layout.route.rail_plan import RailPlacementError

    before = {key: kdb.Region() for key in layers}
    for name, stamp in geometry.items():
        if name not in placement_um:
            raise ValueError('wide rail observer found artwork without placement')
        for layer, datatype, points in _stamp_records(stamp, placement_um[name]):
            key = (layer, datatype)
            if key in before:
                before[key].insert(kdb.Polygon([kdb.Point(x, y) for x, y in points]))
    after = {key: region.dup() for key, region in before.items()}
    for fact in facts:
        key = tuple(fact.demand.layer_key)
        if key in after:
            after[key].insert(kdb.Box(*fact.bbox_nm))
    for key, layer_name in layers.items():
        old = wide_spacing_checks(before[key], tech, layer_name)
        new = wide_spacing_checks(after[key], tech, layer_name)
        if len(old) != len(new):
            raise ValueError('wide rail PDK check inventory changed during materialization')
        for prior, current in zip(old, new):
            if (prior.kind, prior.spacing_nm) != (current.kind, current.spacing_nm):
                raise ValueError('wide rail PDK check authority changed during materialization')
            added = current.errors.polygons(0) - prior.errors.polygons(0)
            if not added.is_empty():
                raise RailPlacementError(
                    f'rail batch introduces {layer_name} {current.kind} spacing '
                    f'violation (required {current.spacing_nm} nm)')


@dataclass(frozen=True)
class _PhysicalRailWriteFact:
    """One normalized rail write, independent of how its location was selected."""

    demand: Any
    content_id: str
    bbox_nm: Tuple[int, int, int, int]
    access_xy_nm: Tuple[int, int]
    candidate_id: Optional[str] = None




def _write_physical_power_rails(
    physical_facts: Sequence[_PhysicalRailWriteFact],
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    pins_by_net: Dict[str, List[Any]],
) -> Tuple[List[Dict[str, Any]], Tuple[Any, ...]]:
    """Atomically preflight, then write normalized rail geometry and access pins."""

    from clay.layout.extract.library_device import CellGeometry as _CG
    from clay.layout.route.engine import Pin as _RailPin
    from clay.layout.route.rail_plan import MaterializedRailFact

    facts = tuple(physical_facts)
    prepared: List[_PhysicalRailWriteFact] = []
    content_ids = set()
    demand_ids = set()
    for fact in facts:
        if type(fact) is not _PhysicalRailWriteFact:
            raise TypeError("rail physical facts must be _PhysicalRailWriteFact")
        demand = fact.demand
        demand_id = str(getattr(demand, "demand_id", "") or "")
        net = str(getattr(demand, "net", "") or "")
        layer_key = getattr(demand, "layer_key", None)
        preferred_axis = str(
            getattr(demand, "preferred_axis", "") or ""
        ).upper()
        width_nm = getattr(demand, "width_nm", None)
        if not demand_id or not net:
            raise ValueError("rail physical fact requires named demand and net")
        if (
            type(fact.content_id) is not str
            or not fact.content_id.startswith("PRAIL_")
        ):
            raise ValueError("rail physical fact requires a PRAIL_ content id")
        if (
            fact.content_id in content_ids
            or demand_id in demand_ids
        ):
            raise ValueError(
                "duplicate rail physical fact in one atomic write batch"
            )
        content_ids.add(fact.content_id)
        demand_ids.add(demand_id)
        if (
            type(layer_key) is not tuple
            or len(layer_key) != 2
            or not all(type(value) is int for value in layer_key)
        ):
            raise TypeError("rail physical fact layer_key must be an exact int tuple")
        if (
            type(fact.bbox_nm) is not tuple
            or len(fact.bbox_nm) != 4
            or not all(type(value) is int for value in fact.bbox_nm)
        ):
            raise TypeError("rail physical bbox must be an exact four-int tuple")
        if (
            type(fact.access_xy_nm) is not tuple
            or len(fact.access_xy_nm) != 2
            or not all(type(value) is int for value in fact.access_xy_nm)
        ):
            raise TypeError("rail physical access must be an exact two-int tuple")
        x0, y0, x1, y1 = fact.bbox_nm
        ax, ay = fact.access_xy_nm
        if x1 <= x0 or y1 <= y0:
            raise ValueError("rail physical bbox must have positive area")
        if not (x0 <= ax <= x1 and y0 <= ay <= y1):
            raise ValueError("rail physical access must lie inside its bbox")
        if type(width_nm) is not int or width_nm <= 0:
            raise ValueError("rail physical demand width must be a positive integer")
        if (
            (preferred_axis == "H" and y1 - y0 != width_nm)
            or (preferred_axis == "V" and x1 - x0 != width_nm)
            or preferred_axis not in ("H", "V")
        ):
            raise ValueError("rail physical bbox does not preserve demand width")
        if fact.content_id in geometry or fact.content_id in placement_um:
            raise ValueError(
                f"rail plan content {fact.content_id!r} was already emitted"
            )
        if any(
            str(getattr(stamp, "rail_demand_id", "")) == demand_id
            for stamp in geometry.values()
        ):
            raise ValueError(
                "duplicate rail physical fact for already emitted demand "
                f"{demand_id!r}"
            )
        if any(
            str(getattr(pin, "device", "")) == fact.content_id
            for net_pins in pins_by_net.values()
            for pin in net_pins
        ):
            raise ValueError(
                f"rail plan access {fact.content_id!r} was already emitted"
            )
        prepared.append(fact)

    prepared.sort(key=lambda item: (
        str(item.demand.demand_id), item.content_id
    ))
    write_rows: List[Tuple[Any, Any, Any, Dict[str, Any], Any]] = []
    for fact in prepared:
        demand = fact.demand
        x0, y0, x1, y1 = fact.bbox_nm
        bbox_um = tuple(value / _NM_PER_UM for value in fact.bbox_nm)
        access_um = tuple(value / _NM_PER_UM for value in fact.access_xy_nm)
        ring = [
            (x0 / _NM_PER_UM, y0 / _NM_PER_UM),
            (x1 / _NM_PER_UM, y0 / _NM_PER_UM),
            (x1 / _NM_PER_UM, y1 / _NM_PER_UM),
            (x0 / _NM_PER_UM, y1 / _NM_PER_UM),
        ]
        stamp = _CG(
            name=fact.content_id,
            kind="power_rail",
            ref_centroid=(0.0, 0.0),
            polys_by_layer={demand.layer_key: [ring]},
        )
        stamp.source = "power_rail"
        stamp.rail_demand_id = demand.demand_id
        rail_pin = _RailPin(
            net=demand.net,
            device=fact.content_id,
            terminal="rail",
            xy_um=access_um,
            layer=demand.layer_key,
            xy_nm=fact.access_xy_nm,
        )
        emitted = {
            "rail": fact.content_id,
            "net": demand.net,
            "layer": demand.layer_key,
            "center_um": access_um,
            "bbox_um": bbox_um,
            "demand_id": demand.demand_id,
        }
        if fact.candidate_id is not None:
            emitted["candidate_id"] = fact.candidate_id
        materialized = MaterializedRailFact(
            content_id=fact.content_id,
            demand_id=demand.demand_id,
            net=demand.net,
            layer_key=demand.layer_key,
            bbox_nm=fact.bbox_nm,
            access_xy_nm=fact.access_xy_nm,
            axis_authority=demand.axis_authority,
        )
        write_rows.append((fact, stamp, rail_pin, emitted, materialized))

    out: List[Dict[str, Any]] = []
    materialized_facts: List[Any] = []
    for fact, stamp, rail_pin, emitted, materialized in write_rows:
        geometry[fact.content_id] = stamp
        placement_um[fact.content_id] = (0.0, 0.0)
        pins_by_net.setdefault(fact.demand.net, []).append(rail_pin)
        out.append(emitted)
        materialized_facts.append(materialized)
    return out, tuple(materialized_facts)


def _bringup_rail_construction_stop(demand: Any, reason: str) -> RoutePlanningStop:
    evidence = _BringupRailConstructionEvidence(
        demand_id=str(getattr(demand, "demand_id", "") or "<unknown>"),
        net=str(getattr(demand, "net", "") or "<unknown>"),
        reason=str(reason),
    )
    return RoutePlanningStop(
        phase="RAIL_PLACEMENT",
        state="BRINGUP_RAIL_CONSTRUCTION_FAILED",
        evidence=evidence,
    )


def _build_bringup_outside_edge_rail_facts(
    demands: Sequence[Any],
    cell_bbox_nm: Tuple[int, int, int, int],
    tech: LayoutTech,
) -> Tuple[_PhysicalRailWriteFact, ...]:
    """Construct deterministic non-optimal rails outside the final cell bbox."""

    ordered = tuple(sorted(
        tuple(demands), key=lambda item: str(getattr(item, "demand_id", ""))
    ))
    if not ordered:
        return ()
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.rail_plan import _verify_demand_against_tech

    try:
        grid_nm = pdk_manufacturing_grid_nm(tech)
    except Exception as exc:
        raise _bringup_rail_construction_stop(
            ordered[0], "PDK manufacturing-grid lookup failed"
        ) from exc
    if type(grid_nm) is not int or grid_nm <= 0:
        raise _bringup_rail_construction_stop(
            ordered[0], "PDK manufacturing grid is not a positive exact integer"
        )
    if (
        type(cell_bbox_nm) is not tuple
        or len(cell_bbox_nm) != 4
        or not all(type(value) is int for value in cell_bbox_nm)
        or cell_bbox_nm[2] <= cell_bbox_nm[0]
        or cell_bbox_nm[3] <= cell_bbox_nm[1]
    ):
        raise _bringup_rail_construction_stop(
            ordered[0], "final cell bbox is unavailable or degenerate"
    )
    x0, y0, x1, y1 = cell_bbox_nm
    offsets: Dict[Tuple[Tuple[int, int], str], int] = {}
    layer_axes: Dict[Tuple[int, int], set] = {}
    facts: List[_PhysicalRailWriteFact] = []
    seen_demands = set()
    for demand in ordered:
        if getattr(demand, "offset_nm", 0):
            raise _bringup_rail_construction_stop(
                demand, "inward offset is unsupported by outside-edge bringup"
            )
        demand_id = str(getattr(demand, "demand_id", "") or "")
        net = str(getattr(demand, "net", "") or "")
        layer_key = getattr(demand, "layer_key", None)
        layer_n = getattr(demand, "layer_n", None)
        axis = str(getattr(demand, "preferred_axis", "") or "").upper()
        edge = str(getattr(demand, "edge", "") or "").lower()
        width_nm = getattr(demand, "width_nm", None)
        if not demand_id or demand_id in seen_demands:
            raise _bringup_rail_construction_stop(
                demand, "demand ids must be non-empty and unique"
            )
        seen_demands.add(demand_id)
        if not net:
            raise _bringup_rail_construction_stop(demand, "rail net is unnamed")
        if (
            type(layer_key) is not tuple
            or len(layer_key) != 2
            or not all(type(value) is int for value in layer_key)
            or type(layer_n) is not int
        ):
            raise _bringup_rail_construction_stop(
                demand, "rail metal layer facts are unavailable"
            )
        if type(width_nm) is not int or width_nm <= 0:
            raise _bringup_rail_construction_stop(
                demand, "compiled rail width is not a positive integer"
            )
        if (
            (axis == "H" and edge not in ("top", "bottom"))
            or (axis == "V" and edge not in ("left", "right"))
            or axis not in ("H", "V")
        ):
            raise _bringup_rail_construction_stop(
                demand, f"preferred axis {axis!r} is incompatible with edge {edge!r}"
            )
        prior_axes = layer_axes.setdefault(layer_key, set())
        if any(prior_axis != axis for prior_axis in prior_axes):
            raise _bringup_rail_construction_stop(
                demand,
                "mixed-axis rails on one layer would create an outside-edge "
                "corner spacing/notch violation",
            )
        prior_axes.add(axis)
        try:
            _verify_demand_against_tech(demand, tech)
        except Exception as exc:
            raise _bringup_rail_construction_stop(
                demand, f"rail demand does not match current PDK: {exc}"
            ) from exc
        if width_nm % grid_nm:
            raise _bringup_rail_construction_stop(
                demand,
                f"compiled rail width is off the {grid_nm}-nm manufacturing grid",
            )
        run_bounds = (x0, x1) if axis == "H" else (y0, y1)
        if any(value % grid_nm for value in run_bounds):
            raise _bringup_rail_construction_stop(
                demand,
                "final cell run-axis bounds are off the manufacturing grid",
            )
        try:
            min_space_nm = tech.min_space(f"metal{layer_n}")
        except Exception as exc:
            raise _bringup_rail_construction_stop(
                demand, "PDK metal min-space lookup failed"
            ) from exc
        if type(min_space_nm) is not int or min_space_nm <= 0:
            raise _bringup_rail_construction_stop(
                demand, "PDK metal min-space is unavailable"
            )
        spacing_step_nm = -((-min_space_nm) // grid_nm) * grid_nm
        group = (layer_key, edge)
        offset_nm = offsets.get(group, 0)
        if edge == "top":
            first_near_nm = -((-(y1 + min_space_nm)) // grid_nm) * grid_nm
            bbox_nm = (
                x0,
                first_near_nm + offset_nm,
                x1,
                first_near_nm + offset_nm + width_nm,
            )
        elif edge == "bottom":
            first_near_nm = ((y0 - min_space_nm) // grid_nm) * grid_nm
            bbox_nm = (
                x0,
                first_near_nm - offset_nm - width_nm,
                x1,
                first_near_nm - offset_nm,
            )
        elif edge == "right":
            first_near_nm = -((-(x1 + min_space_nm)) // grid_nm) * grid_nm
            bbox_nm = (
                first_near_nm + offset_nm,
                y0,
                first_near_nm + offset_nm + width_nm,
                y1,
            )
        else:
            first_near_nm = ((x0 - min_space_nm) // grid_nm) * grid_nm
            bbox_nm = (
                first_near_nm - offset_nm - width_nm,
                y0,
                first_near_nm - offset_nm,
                y1,
            )
        offsets[group] = offset_nm + width_nm + spacing_step_nm
        access_xy_nm = (
            bbox_nm[0]
            + (((bbox_nm[2] - bbox_nm[0]) // grid_nm) // 2) * grid_nm,
            bbox_nm[1]
            + (((bbox_nm[3] - bbox_nm[1]) // grid_nm) // 2) * grid_nm,
        )
        identity_payload = json.dumps({
            "bbox_nm": list(bbox_nm),
            "demand_id": demand_id,
            "schema": "clay.bringup-outside-edge-rail.v1",
        }, sort_keys=True, separators=(",", ":"))
        content_id = "PRAIL_" + hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest()[:20]
        facts.append(_PhysicalRailWriteFact(
            demand=demand,
            content_id=content_id,
            bbox_nm=bbox_nm,
            access_xy_nm=access_xy_nm,
        ))
    return tuple(facts)


def _materialize_bringup_power_rails(
    demands: Sequence[Any],
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    tech: LayoutTech,
    *,
    pins_by_net: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """Write stable outside-edge rail facts without planning or signoff evidence."""

    ordered = tuple(demands)
    if not ordered:
        return []
    named_demand = min(
        ordered, key=lambda item: str(getattr(item, "demand_id", ""))
    )
    if any(
        str(getattr(stamp, "source", "")) == "power_rail"
        or bool(str(getattr(stamp, "rail_demand_id", "") or ""))
        or str(name).startswith("PRAIL_")
        for name, stamp in geometry.items()
    ):
        raise _bringup_rail_construction_stop(
            named_demand,
            "bring-up rail materialization requires one complete atomic batch; "
            "existing power-rail geometry is forbidden",
        )
    bbox_um = _cell_bbox_um(placement_um, geometry)
    if bbox_um is None:
        raise _bringup_rail_construction_stop(
            named_demand, "final cell bbox is unavailable"
        )
    bbox_nm = tuple(round(value * _NM_PER_UM) for value in bbox_um)
    physical_facts = _build_bringup_outside_edge_rail_facts(
        ordered, bbox_nm, tech
    )
    rails, _materialized_facts = _write_physical_power_rails(
        physical_facts, placement_um, geometry, pins_by_net=pins_by_net
    )
    return rails
