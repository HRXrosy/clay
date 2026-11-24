"""Final-placement certificate binding and measurable resource-evidence construction."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route.rc_devices import _NM_PER_UM, _rc_attr


@dataclass(frozen=True)
class FinalPlacementResourceEvidence:
    status: str
    errors: Tuple[str, ...]
    cluster_index: Any = None
    resource_model: Any = None
    capacity_observations: Tuple[Any, ...] = ()
    unresolved_layer_resources: Tuple[str, ...] = ()

    @classmethod
    def unmeasurable(cls, *errors: str) -> "FinalPlacementResourceEvidence":
        named = tuple(str(error) for error in errors if str(error))
        return cls(status="UNMEASURABLE", errors=named or ("unknown observation error",))

    def canonical_json(self) -> str:
        payload = {
            "capacity_observations": [
                {
                    "clear_span_nm": item.clear_span_nm,
                    "corridor_id": item.corridor_id,
                    "layer_n": item.layer_n,
                    "nominal_track_capacity": item.nominal_track_capacity,
                    "pitch_nm": item.pitch_nm,
                    "preferred_axis": item.preferred_axis,
                }
                for item in self.capacity_observations
            ],
            "cluster_members": [
                {
                    "bbox_nm": list(cluster.bbox_nm),
                    "members": list(cluster.members),
                    "stamp": cluster.stamp,
                }
                for cluster in (
                    self.cluster_index.clusters if self.cluster_index is not None else ()
                )
            ],
            "errors": list(self.errors),
            "resource_model": (
                json.loads(self.resource_model.canonical_json())
                if self.resource_model is not None
                else None
            ),
            "schema": "clay.final-placement-resource-evidence.v1",
            "status": self.status,
            "unresolved_layer_resources": list(self.unresolved_layer_resources),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class FinalPlacementBindingError(ValueError):
    """A supplied final-master certificate does not describe the routed override."""


def _certificate_routed_master_blocks(
    final_placement_certificate: Any,
) -> Tuple[Tuple[str, Tuple[int, int, int, int]], ...]:
    """Project certificate masters into the exact physical routed view."""

    from clay.layout.place_blocks_2d import FlexibleEnvelopeProjection

    raw_blocks = getattr(final_placement_certificate, "blocks", None)
    if type(raw_blocks) is not tuple:
        raise FinalPlacementBindingError("certificate blocks must be one exact tuple")
    canonical_blocks: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for row in raw_blocks:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) is not tuple
            or len(row[1]) != 4
            or any(type(value) is not int for value in row[1])
            or row[1][2] <= row[1][0]
            or row[1][3] <= row[1][1]
        ):
            raise FinalPlacementBindingError(
                "certificate block inventory contains a malformed row"
            )
        canonical_blocks.append(row)
    if tuple(canonical_blocks) != tuple(sorted(canonical_blocks)) or len(
        {name for name, _bbox in canonical_blocks}
    ) != len(canonical_blocks):
        raise FinalPlacementBindingError(
            "certificate block inventory must be canonical and unique"
        )

    logical_by_name = dict(canonical_blocks)
    physical_by_name = dict(canonical_blocks)
    projections = getattr(
        final_placement_certificate, "flexible_envelope_projections", None
    )
    if type(projections) is not tuple:
        raise FinalPlacementBindingError(
            "certificate flexible projections must be one exact tuple"
        )
    if projections != tuple(sorted(projections)):
        raise FinalPlacementBindingError(
            "certificate flexible projections must be canonical"
        )
    consumed: set[str] = set()
    for projection in projections:
        if type(projection) is not FlexibleEnvelopeProjection:
            raise FinalPlacementBindingError(
                "certificate flexible projection has a foreign type"
            )
        offsets = dict(projection.constraint.member_physical_offsets_nm)
        for member in projection.constraint.members:
            if member in consumed or member not in logical_by_name or member not in offsets:
                raise FinalPlacementBindingError(
                    "certificate flexible projection ownership does not replay"
                )
            logical = logical_by_name[member]
            offset = offsets[member]
            physical_by_name[member] = (
                logical[0] + offset[0],
                logical[1] + offset[1],
                logical[0] + offset[2],
                logical[1] + offset[3],
            )
            consumed.add(member)
    return tuple(sorted(physical_by_name.items()))


def _final_placement_provenance_digest(
    resource_evidence: FinalPlacementResourceEvidence,
    final_placement_certificate: Any,
) -> str:
    """Bind an optional certificate to the exact pre-rail cluster inventory."""

    if final_placement_certificate is None:
        return resource_evidence.sha256
    from clay.layout.place.resource_feedback import final_placement_state_digest

    try:
        placement_digest = final_placement_state_digest(final_placement_certificate)
    except (TypeError, ValueError) as exc:
        raise FinalPlacementBindingError(
            f"invalid final placement certificate: {exc}"
        ) from exc
    if (
        resource_evidence.status != "MEASURABLE"
        or resource_evidence.cluster_index is None
    ):
        raise FinalPlacementBindingError(
            "final placement certificate cannot bind an unmeasurable routed override"
        )
    observed_blocks = tuple(
        sorted(
            (cluster.stamp, cluster.bbox_nm)
            for cluster in resource_evidence.cluster_index.clusters
        )
    )
    certificate_blocks = _certificate_routed_master_blocks(
        final_placement_certificate
    )
    materialized_members = tuple(
        getattr(final_placement_certificate, "materialized_block_members", ()) or ()
    )
    if materialized_members:
        observed_by_name = dict(observed_blocks)
        claimed_members = [
            member
            for _master, members in materialized_members
            for member in members
        ]
        if (
            len(claimed_members) != len(set(claimed_members))
            or set(claimed_members) != set(observed_by_name)
        ):
            raise FinalPlacementBindingError(
                "final placement certificate materialized members do not match "
                "the exact routed override"
            )
        certificate_member_blocks = tuple(
            sorted(final_placement_certificate.materialized_block_bboxes)
        )
        if certificate_member_blocks != observed_blocks:
            raise FinalPlacementBindingError(
                "final placement certificate materialized block bboxes do not "
                "match the exact routed override"
            )
        projected_blocks = []
        for master, members in materialized_members:
            boxes = [observed_by_name[member] for member in members]
            projected_blocks.append((
                master,
                (
                    min(bbox[0] for bbox in boxes),
                    min(bbox[1] for bbox in boxes),
                    max(bbox[2] for bbox in boxes),
                    max(bbox[3] for bbox in boxes),
                ),
            ))
        observed_master_blocks = tuple(sorted(projected_blocks))
    else:
        observed_master_blocks = observed_blocks
    if observed_master_blocks != certificate_blocks:
        observed_by_master = dict(observed_master_blocks)
        certificate_by_master = dict(certificate_blocks)
        missing = tuple(sorted(set(certificate_by_master) - set(observed_by_master)))
        extra = tuple(sorted(set(observed_by_master) - set(certificate_by_master)))
        mismatched = tuple(
            (name, certificate_by_master[name], observed_by_master[name])
            for name in sorted(set(certificate_by_master) & set(observed_by_master))
            if certificate_by_master[name] != observed_by_master[name]
        )
        raise FinalPlacementBindingError(
            "final placement certificate blocks do not match the exact routed "
            f"override: missing={missing}, extra={extra}, "
            f"mismatched={mismatched[:8]}"
        )
    if not observed_master_blocks:
        raise FinalPlacementBindingError(
            "final placement certificate cannot bind an empty routed override"
        )
    observed_bbox = (
        min(bbox[0] for _name, bbox in observed_master_blocks),
        min(bbox[1] for _name, bbox in observed_master_blocks),
        max(bbox[2] for _name, bbox in observed_master_blocks),
        max(bbox[3] for _name, bbox in observed_master_blocks),
    )
    certificate_bbox = (
        min(bbox[0] for _name, bbox in certificate_blocks),
        min(bbox[1] for _name, bbox in certificate_blocks),
        max(bbox[2] for _name, bbox in certificate_blocks),
        max(bbox[3] for _name, bbox in certificate_blocks),
    )
    if certificate_bbox != observed_bbox:
        raise FinalPlacementBindingError(
            "final placement certificate physical bbox does not match the exact routed override"
        )
    return placement_digest


def _resource_cluster_boxes(
    boxes: Mapping[str, Tuple[int, int, int, int]],
    flexible_envelope_projections: Any,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Collapse only certificate-owned flexible members for corridor capacity.

    Provenance retains the original cluster index.  The resource model instead
    sees one obstacle for a shared physical envelope whose member rings may
    overlap by construction.  No projection means the historical strict
    non-overlap model.
    """

    from clay.layout.place_blocks_2d import FlexibleEnvelopeProjection

    if type(flexible_envelope_projections) is not tuple:
        raise TypeError("flexible resource projections must be one exact tuple")
    if flexible_envelope_projections != tuple(sorted(flexible_envelope_projections)):
        raise ValueError("flexible resource projections must be canonical")
    resource_boxes = dict(boxes)
    consumed: set[str] = set()
    groups: set[str] = set()
    for projection in flexible_envelope_projections:
        if type(projection) is not FlexibleEnvelopeProjection:
            raise TypeError("flexible resource projection has a foreign type")
        group = projection.constraint.group
        members = projection.constraint.members
        if group in boxes or group in groups:
            raise ValueError("flexible resource envelope group collides")
        if consumed.intersection(members):
            raise ValueError("flexible resource projections repeat member ownership")
        missing = tuple(member for member in members if member not in boxes)
        if missing:
            raise ValueError(
                f"flexible resource envelope has absent observed members: {missing}"
            )
        observed = (
            min(boxes[member][0] for member in members),
            min(boxes[member][1] for member in members),
            max(boxes[member][2] for member in members),
            max(boxes[member][3] for member in members),
        )
        if observed != projection.bbox_nm:
            raise ValueError(
                "flexible resource envelope does not replay observed members"
            )
        member_union = _rectilinear_union_boxes(
            tuple(boxes[member] for member in members)
        )
        piece_names = (
            (group,)
            if len(member_union) == 1
            else tuple(
                f"{group}::{index:04d}" for index in range(len(member_union))
            )
        )
        collisions = tuple(name for name in piece_names if name in resource_boxes)
        if collisions:
            raise ValueError(
                f"flexible resource envelope pieces collide: {collisions}"
            )
        for member in members:
            del resource_boxes[member]
        resource_boxes.update(zip(piece_names, member_union))
        consumed.update(members)
        groups.add(group)
    return {name: resource_boxes[name] for name in sorted(resource_boxes)}


def _shared_guard_virtual_resource_boxes(
    boxes: Mapping[str, Tuple[int, int, int, int]],
    block_gring: Mapping[str, str],
    gring_names: Any,
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, _XY],
    flexible_envelope_projections: Any,
    shared_well_guard_ring_inventory: Any,
) -> Tuple[
    Dict[str, Tuple[int, int, int, int]],
    Dict[str, str],
]:
    """Replay a shared outer ring into the certificate's source-owner view.

    The physical scene contains one outer ring, while the placement certificate
    retains the independently movable source-local owners.  Resource/final-
    placement provenance uses virtual local bboxes reconstructed from the signed
    content offsets, but only after the actual provenance-tagged outer ring is
    observed at exactly the signed group envelope.
    """

    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    from clay.layout.place_blocks_2d import FlexibleEnvelopeProjection
    from clay.layout.well_island_sharing import SharedWellGuardRingInventory

    if type(shared_well_guard_ring_inventory) is not SharedWellGuardRingInventory:
        raise TypeError("shared resource observation requires one exact ring inventory")
    if type(flexible_envelope_projections) is not tuple or any(
        type(row) is not FlexibleEnvelopeProjection
        for row in flexible_envelope_projections
    ):
        raise TypeError("shared resource observation requires exact projections")
    projections = {
        projection.constraint.group: projection
        for projection in flexible_envelope_projections
    }
    bindings = {
        binding.group: binding
        for binding in shared_well_guard_ring_inventory.envelope_bindings
    }
    if set(projections) != set(bindings) or len(projections) != len(
        flexible_envelope_projections
    ):
        raise ValueError("shared resource projections differ from ring bindings")
    effective_by_sha = {
        declaration.sha256: declaration
        for declaration in shared_well_guard_ring_inventory.effective_guard_rings
    }
    placed = _placed_stamp_bboxes_nm(dict(geometry), dict(placement_um))
    ring_by_sha: Dict[str, str] = {}
    for name in sorted(gring_names):
        stamp = geometry.get(name)
        digest = getattr(stamp, "guard_ring_intent_sha256", "")
        members = getattr(stamp, "guard_ring_declared_members", ())
        if (
            type(digest) is not str
            or len(digest) != 64
            or type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
        ):
            raise ValueError("shared resource ring provenance is malformed")
        if digest in ring_by_sha:
            raise ValueError("shared resource ring provenance is not unique")
        ring_by_sha[digest] = name

    virtual = dict(boxes)
    group_ring_owner: Dict[str, str] = {}
    consumed_members: set[str] = set()
    for group in sorted(projections):
        projection = projections[group]
        binding = bindings[group]
        effective = effective_by_sha.get(binding.effective_declaration_sha256)
        ring_name = ring_by_sha.get(binding.effective_declaration_sha256)
        if effective is None or ring_name is None or ring_name not in placed:
            raise ValueError("shared resource envelope lacks its effective outer ring")
        if getattr(geometry[ring_name], "guard_ring_declared_members", ()) != tuple(
            effective.members
        ):
            raise ValueError("shared resource outer-ring members differ from inventory")
        if placed[ring_name] != projection.bbox_nm:
            raise ValueError("shared resource outer-ring bbox differs from projection")
        members = projection.constraint.members
        missing = tuple(member for member in members if member not in virtual)
        if missing:
            raise ValueError(
                f"flexible resource envelope has absent observed members: {missing}"
            )
        if consumed_members.intersection(members) or any(
            member in block_gring for member in members
        ):
            raise ValueError("shared resource source owners still carry local rings")
        offsets = dict(projection.constraint.member_physical_offsets_nm)
        projected_member_boxes: list[Tuple[int, int, int, int]] = []
        for member in members:
            content = virtual[member]
            offset = offsets[member]
            physical = (
                content[0] + offset[0],
                content[1] + offset[1],
                content[0] + offset[2],
                content[1] + offset[3],
            )
            if not (
                physical[0] <= content[0] < content[2] <= physical[2]
                and physical[1] <= content[1] < content[3] <= physical[3]
            ):
                raise ValueError("shared resource virtual ring does not enclose content")
            virtual[member] = physical
            projected_member_boxes.append(physical)
        observed = (
            min(bbox[0] for bbox in projected_member_boxes),
            min(bbox[1] for bbox in projected_member_boxes),
            max(bbox[2] for bbox in projected_member_boxes),
            max(bbox[3] for bbox in projected_member_boxes),
        )
        if observed != projection.bbox_nm:
            raise ValueError("shared resource virtual owners differ from projection")
        group_ring_owner[members[0]] = ring_name
        consumed_members.update(members)
    return virtual, group_ring_owner


def _rectilinear_union_boxes(
    boxes: Tuple[Tuple[int, int, int, int], ...],
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Return an exact, deterministic, non-overlapping rectangle union."""

    if type(boxes) is not tuple or not boxes:
        raise TypeError("resource union boxes must be one non-empty exact tuple")
    for bbox in boxes:
        if (
            type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError("resource union bbox must be one positive exact tuple")

    x_edges = tuple(sorted({bbox[index] for bbox in boxes for index in (0, 2)}))
    active: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    finished: List[Tuple[int, int, int, int]] = []
    for x0, x1 in zip(x_edges, x_edges[1:]):
        raw_intervals = sorted(
            (bbox[1], bbox[3])
            for bbox in boxes
            if bbox[0] <= x0 and bbox[2] >= x1
        )
        merged: List[Tuple[int, int]] = []
        for y0, y1 in raw_intervals:
            if merged and y0 <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], y1))
            else:
                merged.append((y0, y1))

        next_active: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
        for interval in merged:
            previous = active.get(interval)
            if previous is not None and previous[2] == x0:
                next_active[interval] = (
                    previous[0],
                    previous[1],
                    x1,
                    previous[3],
                )
            else:
                next_active[interval] = (x0, interval[0], x1, interval[1])
        finished.extend(
            rectangle
            for interval, rectangle in active.items()
            if interval not in next_active
        )
        active = next_active
    finished.extend(active.values())
    return tuple(sorted(finished))


def build_final_placement_resource_evidence(
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    intent: Any,
    tech: LayoutTech,
    *,
    final_placement_certificate: Any = None,
    shared_well_guard_ring_inventory: Any = None,
) -> FinalPlacementResourceEvidence:
    """Observe final clusters; observation failure can never abort routing."""
    try:
        projections = ()
        materialized_members = ()
        observe_envelope_union = False
        if final_placement_certificate is not None:
            from clay.layout.place_blocks_2d import FinalPlacementCertificate

            if type(final_placement_certificate) is not FinalPlacementCertificate:
                raise TypeError(
                    "resource observation requires one exact placement certificate"
                )
            projections = final_placement_certificate.flexible_envelope_projections
            materialized_members = final_placement_certificate.materialized_block_members
            observe_envelope_union = (
                final_placement_certificate.full2d_compaction_graph is not None
                or bool(final_placement_certificate.overlap_compatible_pairs)
                or any(len(members) > 1 for _master, members in
                       final_placement_certificate.materialized_block_members)
            )
        return _build_measurable_final_placement_resource_evidence(
            placement_um,
            geometry,
            intent,
            tech,
            flexible_envelope_projections=projections,
            materialized_block_members=materialized_members,
            shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
            observe_envelope_union=observe_envelope_union,
        )
    except Exception as exc:
        return FinalPlacementResourceEvidence.unmeasurable(
            f"{type(exc).__name__}: {exc}"
        )


def _build_measurable_final_placement_resource_evidence(
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    intent: Any,
    tech: LayoutTech,
    *,
    flexible_envelope_projections: Any = (),
    materialized_block_members: Any = (),
    shared_well_guard_ring_inventory: Any = None,
    observe_envelope_union: bool = False,
) -> FinalPlacementResourceEvidence:
    """Strict implementation behind the non-blocking observer boundary."""
    _resource_cluster_boxes = _rc_attr("_resource_cluster_boxes", globals()["_resource_cluster_boxes"])
    from clay.layout.compact_blocks import extract_blocks
    from clay.layout.placed_cluster_index import PlacedClusterIndex
    from clay.layout.route.resource_plan import (
        LayerResource,
        LayerResourceDeclarationError,
        RoutingResourceModel,
    )

    excluded = {
        name
        for name, stamp in geometry.items()
        if name.startswith("PRAIL_")
        or name.startswith("PPLANE_")
        or str(getattr(stamp, "source", "")) in ("power_rail", "power_plane")
    }
    observed_geometry = {
        name: stamp for name, stamp in geometry.items() if name not in excluded
    }
    observed_placement = {
        name: xy
        for name, xy in placement_um.items()
        if name in observed_geometry
    }
    blocks, block_stamps, block_gring, _gring_names = extract_blocks(
        observed_geometry, observed_placement, intent
    )
    boxes = {name: bbox for name, bbox in blocks}
    shared_ring_owners: Dict[str, str] = {}
    if shared_well_guard_ring_inventory is not None:
        boxes, shared_ring_owners = _shared_guard_virtual_resource_boxes(
            boxes,
            block_gring,
            _gring_names,
            observed_geometry,
            observed_placement,
            flexible_envelope_projections,
            shared_well_guard_ring_inventory,
        )
    composite_members: Dict[str, Tuple[str, ...]] = {}
    for block_name, stamps in block_stamps.items():
        members = set(stamps)
        for stamp_name in stamps:
            members.update(
                str(member)
                for member in (
                    getattr(observed_geometry.get(stamp_name), "composite_members", ())
                    or ()
                )
            )
        ring_name = block_gring.get(block_name)
        if ring_name:
            members.add(ring_name)
        shared_ring_name = shared_ring_owners.get(block_name)
        if shared_ring_name:
            members.add(shared_ring_name)
        composite_members[block_name] = tuple(sorted(members))

    cluster_index = PlacedClusterIndex.build(
        boxes=boxes, composite_members=composite_members
    )
    # Native layered placement permits overlapping envelopes. This observer
    # measures envelope-space union only; exact certificate/geometry binding and
    # physical collision legality remain independent mandatory consumers.
    resource_constructor = (
        RoutingResourceModel.from_cluster_envelope_union
        if observe_envelope_union else RoutingResourceModel.from_clusters
    )
    resource_boxes = boxes
    if flexible_envelope_projections and materialized_block_members:
        # Envelopes name solver masters; observation starts with the source
        # blocks.  Reconstruct each master from its actual materialized members.
        resource_boxes = {}
        for master, members in materialized_block_members:
            member_boxes = [boxes[member] for member in members]
            resource_boxes[master] = (
                min(box[0] for box in member_boxes), min(box[1] for box in member_boxes),
                max(box[2] for box in member_boxes), max(box[3] for box in member_boxes))
    resource_model = resource_constructor(
        _resource_cluster_boxes(resource_boxes, flexible_envelope_projections)
    )
    layer_numbers: List[int] = []
    layer_n = 1
    while True:
        try:
            tech.metal_key(layer_n)
        except KeyError:
            break
        layer_numbers.append(layer_n)
        layer_n += 1

    layers: List[Any] = []
    unresolved: List[str] = []
    for layer_n in layer_numbers:
        try:
            layers.append(LayerResource.from_layout_tech(tech, layer_n=layer_n))
        except LayerResourceDeclarationError as exc:
            unresolved.append(f"metal{layer_n}: {exc}")
    return FinalPlacementResourceEvidence(
        status="MEASURABLE",
        errors=(),
        cluster_index=cluster_index,
        resource_model=resource_model,
        capacity_observations=resource_model.capacity_observations(layers),
        unresolved_layer_resources=tuple(unresolved),
    )
