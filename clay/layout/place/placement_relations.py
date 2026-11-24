"""Project source placement declarations onto the actual materialized owner partition."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.intent_facts import EntityKind, IntentFactKind, IntentFactSet
from clay.layout.spatial.abstract_ir import SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
from clay.layout.spatial.intent_projection import replay_reference_conditioned_placement_obligation_inventory
from clay.layout.canonical_collision_projection import build_row_bounded_cavity_projection
from clay.layout.place.constraints_hierarchical import (
    _canonical_aggregate_envelope_orders,
    _canonical_hierarchical_macro_alignments,
    _canonical_hierarchical_macro_orders,
    _hierarchical_macro_alignment_parts,
)
from clay.layout.place.constraints_spatial import (
    _canonical_row_bounded_cavity_projections,
)
from clay.layout.place.placement_constants import (
    _AggregateEnvelopeOrder,
    _HierarchicalMacroAlignment,
    _HierarchicalMacroOrder,
    _RigidCompoundFootprint,
    _RowBoundedCavityProjection,
)


def _left_column_after_explicit_edge_anchor(
    inferred_left_column: Optional[str],
    hard_left_edge_blocks: Tuple[str, ...],
) -> Optional[str]:
    """Give an authored hard edge owner precedence over legacy column inference."""

    if inferred_left_column is not None and (
        type(inferred_left_column) is not str or not inferred_left_column
    ):
        raise TypeError("inferred left column must be one non-empty string")
    if type(hard_left_edge_blocks) is not tuple or any(
        type(block) is not str or not block for block in hard_left_edge_blocks
    ):
        raise TypeError("hard left-edge blocks must be one exact string tuple")
    if hard_left_edge_blocks != tuple(sorted(set(hard_left_edge_blocks))):
        raise ValueError("hard left-edge blocks must be sorted and unique")
    if (
        inferred_left_column is not None
        and hard_left_edge_blocks
        and inferred_left_column not in hard_left_edge_blocks
    ):
        return None
    return inferred_left_column

def _relational_place_group_envelope_edges(
    intent_fact_set: IntentFactSet,
    place_groups: Sequence[Any],
) -> Tuple[Tuple[str, str], ...]:
    """Lower horizontal group-envelope facts to coordinate-free solver edges.

    Only groups connected by an explicit ``envelope_separation_v1`` fact receive
    an edge; unrelated groups remain free.  No row, column, or observed reference
    coordinate is synthesized.  The caller enforces each edge over the complete
    materialized group envelopes rather than merely ordering their centroids.
    """

    if type(intent_fact_set) is not IntentFactSet:
        raise TypeError("relational floorplan requires one exact IntentFactSet")
    if type(place_groups) is not tuple:
        raise TypeError("relational floorplan place groups must be one exact tuple")

    groups_by_name: Dict[str, Any] = {}
    for group in place_groups:
        name = getattr(group, "group", None)
        members = getattr(group, "devices", None)
        if (
            type(name) is not str
            or not name
            or name != name.strip()
            or type(members) is not tuple
            or not members
            or any(type(member) is not str or not member for member in members)
        ):
            raise TypeError("relational floorplan place groups are malformed")
        if name in groups_by_name:
            raise ValueError(f"duplicate relational place group {name!r}")
        groups_by_name[name] = group

    edges: set[Tuple[str, str]] = set()
    for fact in intent_fact_set.facts:
        if fact.kind is not IntentFactKind.ORDER:
            continue
        payload = fact.payload
        if payload.get("semantic") != "envelope_separation_v1":
            continue
        if payload.get("axis") not in {"column", "horizontal", "x"}:
            raise ValueError(
                "envelope_separation_v1 requires a horizontal order axis"
            )
        if (
            len(fact.subjects) != 2
            or any(
                subject.kind is not EntityKind.PLACE_GROUP
                for subject in fact.subjects
            )
        ):
            raise ValueError(
                "envelope_separation_v1 requires two place-group subjects"
            )
        first, second = (subject.name for subject in fact.subjects)
        relation = payload.get("relation")
        if relation in {"left_of", "before"}:
            left, right = first, second
        elif relation in {"right_of", "after"}:
            left, right = second, first
        else:
            raise ValueError(
                "envelope_separation_v1 has no supported horizontal relation"
            )
        if left == right:
            raise ValueError("relational floorplan has a self order edge")
        unknown = {left, right} - set(groups_by_name)
        if unknown:
            raise ValueError(
                "relational floorplan references unknown place groups: "
                f"{tuple(sorted(unknown))!r}"
            )
        edges.add((left, right))

    if not edges:
        return ()

    involved = tuple(sorted({name for edge in edges for name in edge}))
    for left, right in sorted(edges):
        left_column = getattr(groups_by_name[left], "col", None)
        right_column = getattr(groups_by_name[right], "col", None)
        for column in (left_column, right_column):
            if column is not None and (type(column) is not int or column < 0):
                raise TypeError(
                    "relational floorplan columns must be non-negative exact ints"
                )
        if (
            left_column is not None
            and right_column is not None
            and left_column >= right_column
        ):
            raise ValueError(
                "relational envelope order conflicts with authored centroid "
                f"columns: {left!r} -> {right!r}"
            )

    outgoing: Dict[str, set[str]] = {name: set() for name in involved}
    incoming: Dict[str, set[str]] = {name: set() for name in involved}
    for left, right in sorted(edges):
        outgoing[left].add(right)
        incoming[right].add(left)

    indegree = {name: len(incoming[name]) for name in involved}
    ready = sorted(name for name in involved if indegree[name] == 0)
    visited: List[str] = []
    while ready:
        name = ready.pop(0)
        visited.append(name)
        for successor in sorted(outgoing[name]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(visited) != len(involved):
        raise ValueError("relational floorplan order contains a cycle")
    return tuple(sorted(edges))

def _relational_place_group_master_constraints(
    relational_group_edges: Tuple[Tuple[str, str], ...],
    place_groups: Tuple[Any, ...],
    device_blocks: Mapping[str, str],
    inferred_left_column: Optional[str],
) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...], Optional[str]]:
    """Project an acyclic group order graph onto materialized master blocks.

    Every edge becomes a complete bipartite master order.  Only terminal sink
    groups become lateral floorplan blocks; an intermediate group must retain its
    band structure.  An explicit predecessor also overrides the weaker legacy
    inference that one common-centroid master is the global leftmost column.
    """

    if type(relational_group_edges) is not tuple or type(place_groups) is not tuple:
        raise TypeError("relational master projection requires exact tuples")
    groups = {
        str(getattr(group, "group", "")): tuple(
            str(member) for member in tuple(getattr(group, "devices", ()) or ())
        )
        for group in place_groups
    }
    source_groups = {left for left, _right in relational_group_edges}
    order_pairs: set[Tuple[str, str]] = set()
    lateral_blocks: set[str] = set()
    resolved_left_column = inferred_left_column
    for left_group, right_group in relational_group_edges:
        left_members = groups[left_group]
        right_members = groups[right_group]
        missing = tuple(
            sorted(
                member
                for member in (*left_members, *right_members)
                if member not in device_blocks
            )
        )
        if missing:
            raise ValueError(
                "relational place-group members lack materialized masters: "
                f"{missing!r}"
            )
        left_blocks = {device_blocks[member] for member in left_members}
        right_blocks = {device_blocks[member] for member in right_members}
        crossing = left_blocks & right_blocks
        if crossing:
            raise ValueError(
                "one materialized master crosses a relational group envelope: "
                f"{tuple(sorted(crossing))!r}"
            )
        order_pairs.update(
            (left_block, right_block)
            for left_block in left_blocks
            for right_block in right_blocks
        )
        if right_group not in source_groups:
            lateral_blocks.update(right_blocks)
        if resolved_left_column in right_blocks:
            resolved_left_column = None
    return (
        tuple(sorted(order_pairs)),
        tuple(sorted(lateral_blocks)),
        resolved_left_column,
    )

def _nonrigid_place_group_blocks(
    place_groups: Tuple[Any, ...],
    device_blocks: Mapping[str, str],
) -> Tuple[str, ...]:
    """Return masters whose authored group may cross coarse global bands.

    ``packing=deformable`` or ``packing=relational`` means the members remain
    independently placeable as a relational 2-D structure. Keeping those masters
    in the polarity/passive band stack would still impose one full-width vertical
    tier and erase that meaning. This projection uses only source membership and
    current materialized ownership.
    """

    if type(place_groups) is not tuple or not isinstance(device_blocks, Mapping):
        raise TypeError("nonrigid place-group projection requires exact authority")
    owners: set[str] = set()
    for group in place_groups:
        if getattr(group, "packing", None) not in {"deformable", "relational"}:
            continue
        members = getattr(group, "devices", None)
        if (
            type(members) is not tuple
            or not members
            or any(type(member) is not str or not member for member in members)
        ):
            raise TypeError("nonrigid place-group members are malformed")
        missing = tuple(sorted(member for member in members if member not in device_blocks))
        if missing:
            raise ValueError(
                "nonrigid place-group members lack materialized masters: "
                f"{missing!r}"
            )
        owners.update(device_blocks[member] for member in members)
    return tuple(sorted(owners))

def _horizontal_order_is_materializable(
    first_block: str,
    second_block: str,
    bands: Mapping[str, int],
    lateral_blocks: Tuple[str, ...],
) -> bool:
    """Return whether one horizontal order survives coarse-band projection."""

    if (
        type(first_block) is not str
        or not first_block
        or type(second_block) is not str
        or not second_block
        or first_block == second_block
        or not isinstance(bands, Mapping)
        or type(lateral_blocks) is not tuple
    ):
        raise TypeError("horizontal order materialization authority is malformed")
    lateral = set(lateral_blocks)
    return (
        bands.get(first_block) == bands.get(second_block)
        or {first_block, second_block} <= lateral
    )

def _mosaic_center_alignment_pairs(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
) -> Tuple[Tuple[str, str, str], ...]:
    """Lower evidence-bound mosaic/anchor relations to live rigid masters."""

    from clay.layout.spatial.intent_projection import (
        replay_discrete_mosaic_inventory,
        replay_mosaic_center_alignment_inventory,
    )

    mosaics = replay_discrete_mosaic_inventory(
        tuple(getattr(intent, "discrete_mosaics", ()) or ())
    )
    alignments = replay_mosaic_center_alignment_inventory(
        tuple(getattr(intent, "mosaic_center_alignments", ()) or ())
    )
    if not alignments:
        return ()
    mosaic_by_id = {mosaic.mosaic_id: mosaic for mosaic in mosaics}
    owners: dict[str, list[str]] = {}
    for master, members in sorted(block_member_devices.items()):
        for member in sorted({str(member) for member in members}):
            owners.setdefault(member, []).append(str(master))

    pairs = []
    for alignment in alignments:
        mosaic = mosaic_by_id.get(alignment.mosaic_id)
        if mosaic is None:
            raise ValueError(
                "mosaic center alignment references an unknown discrete mosaic"
            )
        if (
            alignment.source_mosaic_sha256 != mosaic.sha256
            or alignment.source_asir_sha256 != mosaic.source_asir_sha256
        ):
            raise ValueError("mosaic center alignment provenance is stale")
        member_masters = set()
        for member in mosaic.members:
            candidates = tuple(owners.get(member, ()))
            if len(candidates) != 1:
                raise ValueError(
                    "mosaic center alignment member lacks unique master ownership"
                )
            member_masters.add(candidates[0])
        if len(member_masters) != 1:
            raise ValueError(
                "mosaic center alignment members are not one rigid master"
            )
        anchor_masters = tuple(owners.get(alignment.anchor_owner, ()))
        if len(anchor_masters) != 1:
            raise ValueError(
                "mosaic center alignment anchor lacks unique master ownership"
            )
        mosaic_master = next(iter(member_masters))
        anchor_master = anchor_masters[0]
        if mosaic_master == anchor_master:
            raise ValueError("mosaic center alignment collapsed onto one master")
        left, right = sorted((mosaic_master, anchor_master))
        pairs.append((left, right, alignment.axis))
    canonical = tuple(sorted(pairs))
    if len(canonical) != len(set(canonical)):
        raise ValueError("mosaic center alignment relations are redundant")
    return canonical

def _aggregate_scope_live_masters(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
    *,
    scope_ids: Sequence[str],
    relation_label: str,
) -> Dict[str, Tuple[str, ...]]:
    """Resolve semantic scopes through one exact live-master ownership authority."""

    from clay.layout.intent import replay_passive_bank_partition_intent
    from clay.layout.spatial.intent_projection import (
        replay_hierarchical_macro_dag_intent,
    )

    place_groups = tuple(getattr(intent, "place_groups", ()) or ())
    scope_members: dict[str, tuple[str, ...]] = {}

    def register_scope(scope: str, members: tuple[str, ...], label: str) -> None:
        if not scope or not members:
            raise ValueError(f"aggregate {label} scope ownership is incomplete")
        prior = scope_members.get(scope)
        if prior is not None and prior != members:
            raise ValueError(f"aggregate {label} scope ownership is ambiguous")
        scope_members[scope] = members

    for group in place_groups:
        scope = str(getattr(group, "group", ""))
        members = tuple(sorted(str(name) for name in getattr(group, "devices", ()) or ()))
        register_scope(scope, members, "place-group")
    raw_passive_partitions = tuple(
        getattr(intent, "passive_bank_partitions", ()) or ()
    )
    for raw_partition in raw_passive_partitions:
        partition = replay_passive_bank_partition_intent(raw_partition)
        scope = f"passive_macro:{partition.bank}"
        members = tuple(sorted(partition.members))
        register_scope(scope, members, "passive")
    hierarchy_declaration = getattr(intent, "hierarchical_macro_dag", None)
    if hierarchy_declaration is not None:
        hierarchy = replay_hierarchical_macro_dag_intent(
            hierarchy_declaration
        ).hierarchy
        for level in hierarchy.levels:
            for node in level.nodes:
                register_scope(
                    node.node_id,
                    tuple(sorted(node.member_nodes)),
                    "hierarchy-node",
                )
    owner_masters: dict[str, list[str]] = {}
    for master, members in sorted(block_member_devices.items()):
        for member in sorted({str(member) for member in members}):
            owner_masters.setdefault(member, []).append(str(master))

    resolved: Dict[str, Tuple[str, ...]] = {}
    for scope in sorted(set(scope_ids)):
        members = scope_members.get(scope)
        if members is None:
            raise ValueError(f"{relation_label} references an unknown scope")
        masters = set()
        for member in members:
            candidates = tuple(owner_masters.get(member, ()))
            if len(candidates) != 1:
                raise ValueError(
                    f"{relation_label} member lacks unique master ownership"
                )
            masters.add(candidates[0])
        resolved[scope] = tuple(sorted(masters))
    return resolved

def _aggregate_envelope_orders_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
    *,
    known_device_names: Optional[Sequence[str]] = None,
    composite_alias_members: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[_AggregateEnvelopeOrder, ...]:
    """Resolve projected scope relations onto the exact live master partition."""

    from clay.layout.spatial.intent_projection import replay_aggregate_order_inventory

    declarations = replay_aggregate_order_inventory(
        tuple(getattr(intent, "aggregate_orders", ()) or ())
    )
    scope_masters = _aggregate_scope_live_masters(
        intent,
        block_member_devices,
        scope_ids=tuple(
            scope
            for declaration in declarations
            for scope in (declaration.before_scope, declaration.after_scope)
        ),
        relation_label="aggregate envelope order",
    ) if declarations else {}

    rows = tuple(
        (
            declaration.before_scope,
            declaration.after_scope,
            declaration.axis,
            scope_masters[declaration.before_scope],
            scope_masters[declaration.after_scope],
        )
        for declaration in declarations
    )
    # Plain authored @order(place_group, ..., relation) is not a projected
    # aggregate_orders record. Device-only orders have their own lowerer, but
    # named groups are not live block IDs and used to disappear there. Resolve
    # their complete source membership through the same actual master partition
    # to measure each real envelope's center. Keep this established centroid
    # semantics; disjoint native placement needs explicit boundary-order facts.
    groups = {str(group.group): set(group.devices) for group in getattr(intent, "place_groups", ())}
    canonical_devices = None if known_device_names is None else set(known_device_names)
    owners_by_member: Dict[str, List[str]] = {}
    for owner, members in sorted(block_member_devices.items()):
        for member in set(members):
            owners_by_member.setdefault(member, []).append(owner)

    def logical_owner_members(owner: str) -> set[str]:
        owned = set(block_member_devices[owner])
        if canonical_devices is not None:
            if owner not in canonical_devices:
                owned.discard(owner)
            for alias in tuple(owned - canonical_devices):
                declared = set((composite_alias_members or {}).get(alias, ()))
                if declared and declared <= canonical_devices and declared <= owned:
                    owned.remove(alias)
        return owned

    def authored_scope_masters(scope: str) -> Tuple[str, ...]:
        members = groups.get(scope, {scope})
        masters: set[str] = set()
        for member in sorted(members):
            owners = owners_by_member.get(member, ())
            if len(owners) != 1:
                raise ValueError(
                    "authored group order member lacks unique master ownership: "
                    f"{scope!r}/{member!r}"
                )
            owner = owners[0]
            owner_members = logical_owner_members(owner)
            # Materialized inventories also carry a composite's OWN geometry
            # alias (e.g. its stamp name) beside actual schematic members.
            # Only the caller's exact topology inventory can certify that this
            # self-name is not another device. No foreign alias is discarded.
            if canonical_devices is not None and owner not in canonical_devices:
                owner_members.discard(owner)
            if canonical_devices is not None:
                for alias in tuple(owner_members - canonical_devices):
                    declared = set((composite_alias_members or {}).get(alias, ()))
                    if declared and declared <= canonical_devices and declared <= owner_members:
                        # This alias is a materialized geometry stamp whose
                        # exact canonical members are present in this owner.
                        # Preserve those members; only remove its representation
                        # name, never an arbitrary unknown logical device.
                        owner_members.remove(alias)
            if owner_members - members:
                raise ValueError(
                    "authored group order master owns undeclared scope members: "
                    f"{scope!r}/{owner!r}"
                )
            masters.add(owner)
        return tuple(sorted(masters))

    authored_rows = []
    for first, second, relation in tuple(getattr(intent, "order_constraints", ()) or ()):
        if first not in groups and second not in groups:
            continue
        if relation not in {"left_of", "above"}:
            raise ValueError("authored group order has an unsupported relation")
        before, after = (first, second) if relation == "left_of" else (second, first)
        authored_rows.append((
            before, after, "horizontal" if relation == "left_of" else "vertical",
            authored_scope_masters(before), authored_scope_masters(after),
        ))
    rows += tuple(authored_rows)
    # Rail affinity is checked on the group envelope versus the complete
    # artwork envelope, not on heuristic polarity bands. Lower that exact
    # half-plane relation into the certified envelope solver, including its
    # equality case. This remains active for relational/lateral groups.
    from clay.layout.intent import replay_rail_affinity_inventory

    universe = tuple(sorted(block_member_devices))
    rail_rows = []
    for declaration in replay_rail_affinity_inventory(tuple(getattr(intent, "rail_affinities", ()) or ())):
        members = groups.get(declaration.group, {declaration.group})
        owners = []
        covered = set()
        for owner, owned_members in sorted(block_member_devices.items()):
            owned_members = logical_owner_members(owner)
            overlap = set(owned_members) & members
            if not overlap:
                continue
            if set(owned_members) - members:
                raise ValueError("rail affinity master also owns undeclared group members: "
                    f"{declaration.group!r}/{owner!r}; "
                    f"foreign={sorted(set(owned_members) - members)!r}")
            if covered & overlap:
                raise ValueError("rail affinity member has duplicate master ownership")
            covered.update(overlap)
            owners.append(owner)
        if covered != members:
            raise ValueError("rail affinity member lacks unique live master ownership")
        prefix = "rail_affinity:" + declaration.group
        target = tuple(owners)
        if declaration.side == "top":
            rail_rows.append((prefix+":universe", prefix+":target", "vertical", universe, target))
        else:
            rail_rows.append((prefix+":target", prefix+":universe", "vertical", target, universe))
    rows += tuple(rail_rows)
    return _canonical_aggregate_envelope_orders(
        tuple(sorted(set(rows))),
        block_ids=tuple(sorted(block_member_devices)),
    )

def _aggregate_hierarchical_macro_alignments_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
) -> Tuple[_HierarchicalMacroAlignment, ...]:
    """Lower aggregate scope bands into the existing macro-alignment consumer."""

    from clay.layout.spatial.intent_projection import (
        replay_aggregate_alignment_inventory,
    )

    declarations = replay_aggregate_alignment_inventory(
        tuple(getattr(intent, "aggregate_alignments", ()) or ())
    )
    source_centers = tuple(getattr(intent, "group_center_alignments", ()) or ())
    if not declarations and not source_centers:
        return ()
    scope_masters = _aggregate_scope_live_masters(
        intent,
        block_member_devices,
        scope_ids=tuple(
            scope
            for declaration in declarations
            for scope in (declaration.first_scope, declaration.second_scope)
        ) + tuple(scope for declaration in source_centers
                  for scope in (declaration.target, *declaration.groups)),
        relation_label="aggregate alignment",
    )
    rows: Tuple[_HierarchicalMacroAlignment, ...] = tuple(
        (
            (
                SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
                + declaration.alignment_id
            ),
            declaration.axis,
            declaration.edge,
            declaration.tolerance_ppm,
            tuple(
                sorted(
                    (
                        (declaration.first_scope, scope_masters[declaration.first_scope]),
                        (declaration.second_scope, scope_masters[declaration.second_scope]),
                    )
                )
            ),
        )
        for declaration in declarations
    )
    for declaration in source_centers:
        union_masters = tuple(sorted({master for group in declaration.groups
                                      for master in scope_masters[group]}))
        group_pair = tuple(sorted((
            (declaration.target, scope_masters[declaration.target]),
            ('union:' + ','.join(declaration.groups), union_masters),
        )))
        rows += ((SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX + 'source_center_' + declaration.target,
                  'horizontal', group_pair),)
    return _canonical_hierarchical_macro_alignments(
        tuple(sorted(rows, key=_hierarchical_macro_alignment_parts)),
        block_ids=tuple(sorted(block_member_devices)),
    )

def _reference_conditioned_envelope_separations_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
) -> Tuple[_HierarchicalMacroOrder, ...]:
    """Lower typed reference obligations to the shared envelope authority."""

    from clay.layout.spatial.intent_projection import (
        SpatialIntentProjectionError,
        replay_reference_conditioned_placement_obligation_inventory,
    )

    all_declarations = replay_reference_conditioned_placement_obligation_inventory(
        tuple(
            getattr(
                intent,
                "reference_conditioned_placement_obligations",
                (),
            )
            or ()
        )
    )
    if not all_declarations:
        return ()
    unvalidated = tuple(
        declaration.obligation_id
        for declaration in all_declarations
        if declaration.witness_validation_status != "PASS"
    )
    if unvalidated:
        raise SpatialIntentProjectionError(
            "reference-conditioned placement requires passing witness validation: "
            f"{unvalidated[0]}"
        )
    declarations = tuple(
        declaration
        for declaration in all_declarations
        if declaration.relation_kind == "ordered_pair"
    )
    if not declarations:
        return ()
    scope_masters = _aggregate_scope_live_masters(
        intent,
        block_member_devices,
        scope_ids=tuple(
            scope
            for declaration in declarations
            for scope in (declaration.before_scope, declaration.after_scope)
        ),
        relation_label="reference-conditioned placement obligation",
    )
    rows = tuple(
        (
            f"reference_conditioned:{declaration.obligation_id}",
            declaration.before_scope,
            declaration.after_scope,
            declaration.axis,
            declaration.strength,
            scope_masters[declaration.before_scope],
            scope_masters[declaration.after_scope],
        )
        for declaration in declarations
    )
    return _canonical_hierarchical_macro_orders(
        tuple(sorted(rows)),
        block_ids=tuple(sorted(block_member_devices)),
    )

def _reference_conditioned_row_bounded_cavities_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
    *,
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...],
    dims: Mapping[str, Tuple[int, int]],
    host_scope_masters: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> Tuple[_RowBoundedCavityProjection, ...]:
    """Resolve witness-gated cavity declarations onto generated live geometry."""

    from clay.layout.intent import replay_row_bounded_cavity_inventory
    from clay.layout.spatial.intent_projection import (
        SpatialIntentProjectionError,
        replay_discrete_mosaic_inventory,
    )

    obligations = tuple(
        row
        for row in replay_reference_conditioned_placement_obligation_inventory(
            tuple(
                getattr(
                    intent,
                    "reference_conditioned_placement_obligations",
                    (),
                )
                or ()
            )
        )
        if row.relation_kind == "row_bounded_cavity"
    )
    if not obligations:
        return ()
    unvalidated = tuple(
        row.obligation_id
        for row in obligations
        if row.witness_validation_status != "PASS"
    )
    if unvalidated:
        raise SpatialIntentProjectionError(
            "reference-conditioned placement requires passing witness validation: "
            f"{unvalidated[0]}"
        )

    declarations = replay_row_bounded_cavity_inventory(
        tuple(getattr(intent, "row_bounded_cavities", ()) or ())
    )
    declarations_by_sha = {row.sha256: row for row in declarations}
    owner_masters: Dict[str, List[str]] = {}
    for master, members in sorted(block_member_devices.items()):
        for member in sorted(set(map(str, members))):
            owner_masters.setdefault(member, []).append(str(master))

    explicit_hosts = dict(host_scope_masters or {})
    mosaic_hosts: Dict[str, Tuple[str, ...]] = {}
    if not explicit_hosts:
        mosaics = replay_discrete_mosaic_inventory(
            tuple(getattr(intent, "discrete_mosaics", ()) or ())
        )
        for mosaic in mosaics:
            masters = {
                master
                for member in mosaic.members
                for master in owner_masters.get(member, ())
            }
            if len(masters) == 1:
                mosaic_hosts[mosaic.mosaic_id] = tuple(sorted(masters))

    footprints = dict(rigid_compound_footprints)
    rows = []
    for obligation in obligations:
        declaration = declarations_by_sha.get(obligation.source_relation_sha256)
        if declaration is None:
            raise SpatialIntentProjectionError(
                "reference-conditioned cavity lacks its source declaration"
            )
        if (
            declaration.host != obligation.host_scope
            or declaration.guests != obligation.guest_scopes
        ):
            raise SpatialIntentProjectionError(
                "reference-conditioned cavity differs from its source declaration"
            )
        host_masters = explicit_hosts.get(
            declaration.host,
            mosaic_hosts.get(declaration.host, ()),
        )
        if type(host_masters) is not tuple or len(host_masters) != 1:
            raise SpatialIntentProjectionError(
                "row-bounded cavity host lacks one live master"
            )
        host = host_masters[0]
        guest_masters = []
        for guest_scope in declaration.guests:
            candidates = tuple(owner_masters.get(guest_scope, ()))
            if len(candidates) != 1:
                raise SpatialIntentProjectionError(
                    "row-bounded cavity guest lacks one live master"
                )
            guest_masters.append(candidates[0])
        canonical_guests = tuple(sorted(set(guest_masters)))
        if len(canonical_guests) != len(guest_masters) or host in canonical_guests:
            raise SpatialIntentProjectionError(
                "row-bounded cavity live ownership collapses"
            )
        if host not in footprints:
            raise SpatialIntentProjectionError(
                "row-bounded cavity host lacks measurable row footprints"
            )
        rows.append(build_row_bounded_cavity_projection(
            obligation_id=obligation.obligation_id,
            source_obligation_sha256=obligation.sha256,
            host_master=host,
            guest_masters=canonical_guests,
            host_size_nm=dims[host],
            host_row_footprints_nm=footprints[host],
        ))
    return _canonical_row_bounded_cavity_projections(tuple(rows), dims=dims)

def _hierarchical_macro_constraints_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
    *,
    known_device_names: Sequence[str],
    tech: Any = None,
    cell_def: Any = None,
    base_gap_nm: int = 0,
) -> Tuple[
    Tuple[_HierarchicalMacroOrder, ...],
    Tuple[_HierarchicalMacroAlignment, ...],
]:
    """Lower one projected macro tree onto an exact, non-crossing live partition."""

    from clay.layout.spatial.intent_projection import (
        replay_hierarchical_macro_dag_intent,
    )

    aggregate_alignments = _aggregate_hierarchical_macro_alignments_from_intent(
        intent,
        block_member_devices,
    )
    reference_conditioned_orders = (
        _reference_conditioned_envelope_separations_from_intent(
            intent,
            block_member_devices,
        )
    )
    # Authored group relations bind whole live source partitions, never only a
    # convenient subset of a composite master. Synthetic stamp aliases are not
    # source devices and do not change ownership.
    source_alignments = tuple(getattr(intent, "group_edge_alignments", ()) or ())
    source_separations = tuple(getattr(intent, "group_separations", ()) or ())
    authored_orders, authored_alignments = [], []
    if source_alignments or source_separations:
        from clay.layout.intent import GroupEdgeAlignment, GroupSeparation
        if (any(type(row) is not GroupEdgeAlignment for row in source_alignments)
                or any(type(row) is not GroupSeparation for row in source_separations)):
            raise TypeError("authored group relation has a foreign declaration")
        groups = {group.group: set(group.devices) for group in getattr(intent, "place_groups", ())}
        scopes = tuple(scope for row in source_alignments for scope in (row.first, row.second)) + tuple(
            scope for row in source_separations for scope in (row.before, row.after))
        if any(scope not in groups for scope in scopes):
            raise ValueError("authored group relation references an unknown place group")
        resolved = _aggregate_scope_live_masters(intent, block_member_devices,
            scope_ids=scopes, relation_label="authored group relation")
        known = set(known_device_names)
        for scope in set(scopes):
            for master in resolved[scope]:
                if (set(block_member_devices[master]) & known) - groups[scope]:
                    raise ValueError("authored group relation master crosses source group ownership")
        for row in source_alignments:
            row.__post_init__()
            axis = 'horizontal' if row.edge in ('left', 'right') else 'vertical'
            edge = 'leading' if row.edge in ('left', 'bottom') else 'trailing'
            pair = tuple(sorted(((row.first, resolved[row.first]), (row.second, resolved[row.second]))))
            authored_alignments.append(('authored_group_align:' + row.first + ':' + row.second + ':' + row.edge,
                                        axis, edge, pair))
        for row in source_separations:
            row.__post_init__()
            from clay.layout.place.group_separation_strength import separation_strength
            authored_orders.append(('authored_group_separate:' + row.before + ':' + row.after + ':' + row.axis,
                                    row.before, row.after, row.axis, separation_strength(row.gap_nm),
                                    resolved[row.before], resolved[row.after]))
    aggregate_alignments = tuple(sorted((*aggregate_alignments, *authored_alignments),
                                        key=_hierarchical_macro_alignment_parts))
    from clay.layout.place.route_escape_reservation import (
        route_group_corridor_orders_from_intent,
    )
    corridor_orders = route_group_corridor_orders_from_intent(
        intent,
        block_member_devices,
        known_device_names=known_device_names,
        existing_orders=(*reference_conditioned_orders, *authored_orders),
        tech=tech,
        cell_def=cell_def,
        base_gap_nm=base_gap_nm,
    )
    reference_conditioned_orders = tuple(
        sorted((*reference_conditioned_orders, *authored_orders, *corridor_orders))
    )
    declaration = getattr(intent, "hierarchical_macro_dag", None)
    if declaration is None:
        return (_canonical_hierarchical_macro_orders(reference_conditioned_orders, block_ids=tuple(sorted(block_member_devices))),
                _canonical_hierarchical_macro_alignments(aggregate_alignments, block_ids=tuple(sorted(block_member_devices))))
    projected = replay_hierarchical_macro_dag_intent(declaration)
    hierarchy = projected.hierarchy
    root_members = set(hierarchy.root.member_nodes)
    known_members = {str(member) for member in known_device_names}
    if known_members != root_members:
        missing = tuple(sorted(known_members - root_members))
        foreign = tuple(sorted(root_members - known_members))
        raise ValueError(
            "hierarchical macro root must exactly cover canonical topology: "
            f"missing={missing!r} foreign={foreign!r}"
        )
    observed_members = {
        str(member)
        for members in block_member_devices.values()
        for member in members
        if str(member) in known_members
    }
    if observed_members != root_members:
        missing = tuple(sorted(root_members - observed_members))
        foreign = tuple(sorted(observed_members - root_members))
        raise ValueError(
            "hierarchical macro live masters must exactly cover projected devices: "
            f"missing={missing!r} foreign={foreign!r}"
        )
    owner_masters: dict[str, list[str]] = {}
    for master, members in sorted(block_member_devices.items()):
        for member in sorted(
            {str(member) for member in members if str(member) in known_members}
        ):
            owner_masters.setdefault(member, []).append(str(master))
    ambiguous = tuple(
        sorted(owner for owner, masters in owner_masters.items() if len(masters) != 1)
    )
    if ambiguous:
        raise ValueError(
            "hierarchical macro member lacks unique live-master ownership: "
            f"{ambiguous[0]}"
        )

    child_scopes = {
        level.scope_id for level in hierarchy.levels if level.parent_node is not None
    }
    leaf_by_owner = {}
    for level in hierarchy.levels:
        for node in level.nodes:
            if node.node_id in child_scopes:
                continue
            for owner in node.member_nodes:
                prior = leaf_by_owner.get(owner)
                if prior is not None and prior != node.node_id:
                    raise ValueError("hierarchical macro leaf ownership is ambiguous")
                leaf_by_owner[owner] = node.node_id
    if set(leaf_by_owner) != root_members:
        raise ValueError("hierarchical macro leaves do not exactly cover root devices")
    for master, members in sorted(block_member_devices.items()):
        leaves = {
            leaf_by_owner[str(member)]
            for member in members
            if str(member) in known_members
        }
        if not leaves:
            continue
        if len(leaves) != 1:
            raise ValueError(
                f"live master {master!r} crosses hierarchical leaf scopes"
            )

    def masters_for(node: Any) -> Tuple[str, ...]:
        return tuple(
            sorted({owner_masters[owner][0] for owner in node.member_nodes})
        )

    orders = []
    alignments: list[_HierarchicalMacroAlignment] = list(aggregate_alignments)
    for level in hierarchy.levels:
        nodes = {node.node_id: node for node in level.nodes}
        for order in level.orders:
            orders.append(
                (
                    level.scope_id,
                    order.before,
                    order.after,
                    order.axis,
                    order.strength.value,
                    masters_for(nodes[order.before]),
                    masters_for(nodes[order.after]),
                )
            )
        for alignment in level.alignments:
            groups = tuple(
                sorted(
                    (
                        node_id,
                        masters_for(nodes[node_id]),
                    )
                    for node_id in alignment.nodes
                )
            )
            alignments.append(
                (level.scope_id, alignment.axis, groups)
                if alignment.edge == "center" and not alignment.tolerance_ppm
                else (
                    level.scope_id,
                    alignment.axis,
                    alignment.edge,
                    alignment.tolerance_ppm,
                    groups,
                )
                if alignment.tolerance_ppm
                else (level.scope_id, alignment.axis, alignment.edge, groups)
            )
    block_ids = tuple(sorted(block_member_devices))
    return (
        _canonical_hierarchical_macro_orders(
            tuple(sorted((*orders, *reference_conditioned_orders))),
            block_ids=block_ids,
        ),
        _canonical_hierarchical_macro_alignments(
            tuple(
                sorted(
                    alignments,
                    key=lambda row: _hierarchical_macro_alignment_parts(row),
                )
            ),
            block_ids=block_ids,
        ),
    )

def _hierarchical_macro_cohesion_groups_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
    *,
    known_device_names: Sequence[str],
) -> Tuple[Tuple[str, ...], ...]:
    """Lower every multi-master hierarchy node to one soft cohesion group.

    The hard DAG continues to own relative order and separation.  This projection
    only minimizes whitespace inside declared node partitions; it never merges
    masters or invents a pairwise device relation.  Nested parent/child groups are
    both retained: this is the executable distinction between the two hierarchy
    levels, and prevents compact leaves from drifting apart inside their parent.
    """

    from clay.layout.spatial.intent_projection import (
        replay_hierarchical_macro_dag_intent,
    )

    declaration = getattr(intent, "hierarchical_macro_dag", None)
    if declaration is None:
        return ()
    _hierarchical_macro_constraints_from_intent(
        intent,
        block_member_devices,
        known_device_names=known_device_names,
    )
    hierarchy = replay_hierarchical_macro_dag_intent(declaration).hierarchy
    known = {str(member) for member in known_device_names}
    owner_master = {
        str(member): str(master)
        for master, members in sorted(block_member_devices.items())
        for member in members
        if str(member) in known
    }
    groups = []
    for level in hierarchy.levels:
        for node in level.nodes:
            group = tuple(sorted({owner_master[owner] for owner in node.member_nodes}))
            if len(group) < 2:
                continue
            groups.append(group)
    canonical = tuple(sorted(set(groups)))
    if not canonical:
        return ()
    return canonical
