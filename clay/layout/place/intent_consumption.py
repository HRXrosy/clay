"""Pure binding of selected floorplan atoms to normalized solver projections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

from clay.layout.intent_basis import (
    IntentBasis,
    IntentConsumptionCertificate,
)
from clay.layout.intent_closure import ConsumerProjection
from clay.layout.intent_facts import (
    EntityKind,
    EntityRef,
    IntentFactKind,
    IntentFactSet,
    REGISTERED_CONSUMERS,
    SourceInventory,
    verify_intent_fact_authority,
)


_FLOORPLAN_KIND = "group_floorplan.cell"
_GLOBAL_FLOORPLAN_COLUMN_SCOPE = -1


def _projection_sha256(rows: object) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_EMPTY_PROJECTION_SHA256 = _projection_sha256([])


@dataclass(frozen=True, order=True)
class _ObservedRelation:
    """One ID-free relation reconstructed from consumer-owned typed output."""

    kind: IntentFactKind
    subjects: tuple[EntityRef, ...]
    payload_json: str

    def __post_init__(self) -> None:
        if type(self.kind) is not IntentFactKind:
            raise TypeError("observed relation kind must be an exact IntentFactKind")
        if (
            type(self.subjects) is not tuple
            or any(type(subject) is not EntityRef for subject in self.subjects)
            or self.subjects != tuple(sorted(set(self.subjects)))
        ):
            raise ValueError("observed relation subjects must be sorted and unique")
        if type(self.payload_json) is not str:
            raise TypeError("observed relation payload must be canonical JSON text")
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("observed relation payload is not JSON") from exc
        if type(payload) is not dict or json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ) != self.payload_json:
            raise ValueError("observed relation payload is not canonical")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.observed-intent-relation/v1",
            "kind": self.kind.value,
            "subjects": [subject.canonical_data() for subject in self.subjects],
            "payload": json.loads(self.payload_json),
        }


def _relation_from_payload(
    kind: IntentFactKind,
    subjects: tuple[EntityRef, ...],
    payload: dict[str, object],
) -> _ObservedRelation:
    if type(payload) is not dict:
        raise TypeError("observed relation payload must be an exact dict")
    return _ObservedRelation(
        kind=kind,
        subjects=tuple(sorted(subjects)),
        payload_json=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )


def _ordinal_order_relations(
    ordinal_by_group: dict[str, tuple[int, int]],
) -> tuple[_ObservedRelation, ...]:
    """Lower registered ordinal cells to their coordinate-free order closure."""

    if type(ordinal_by_group) is not dict:
        raise TypeError("ordinal relation inventory must be an exact dict")
    names = tuple(sorted(ordinal_by_group))
    if len(set(names)) != len(names):
        raise ValueError("ordinal relation group names are not unique")
    for name, cell in ordinal_by_group.items():
        if (
            type(name) is not str
            or not name
            or type(cell) is not tuple
            or len(cell) != 2
            or any(type(value) is not int or value < 0 for value in cell)
        ):
            raise TypeError("ordinal relation cell is malformed")
    relations: list[_ObservedRelation] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            subjects = (
                EntityRef(EntityKind.PLACE_GROUP, left_name),
                EntityRef(EntityKind.PLACE_GROUP, right_name),
            )
            left_tier, left_column = ordinal_by_group[left_name]
            right_tier, right_column = ordinal_by_group[right_name]
            if left_tier == right_tier and left_column != right_column:
                before = left_column < right_column
                for axis, relation in (
                    ("column", "before" if before else "after"),
                    ("horizontal", "left_of" if before else "right_of"),
                    ("x", "left_of" if before else "right_of"),
                ):
                    relations.append(
                        _relation_from_payload(
                            IntentFactKind.ORDER,
                            subjects,
                            {"axis": axis, "relation": relation},
                        )
                    )
            if left_column == right_column and left_tier != right_tier:
                below = left_tier < right_tier
                for axis, relation in (
                    ("tier", "before" if below else "after"),
                    ("vertical", "below" if below else "above"),
                    ("y", "below" if below else "above"),
                ):
                    relations.append(
                        _relation_from_payload(
                            IntentFactKind.ORDER,
                            subjects,
                            {"axis": axis, "relation": relation},
                        )
                    )
    canonical = tuple(sorted(relations))
    if len(set(canonical)) != len(canonical):
        raise ValueError("ordinal relation closure contains duplicates")
    return canonical


def _bind_observed_relations(
    *,
    consumer: str,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
    observed_relations: tuple[_ObservedRelation, ...],
) -> ConsumerProjection:
    """Attach target IDs only after exact ID-free relations have matched."""

    if type(consumer) is not str or consumer not in REGISTERED_CONSUMERS:
        raise ValueError(f"consumer {consumer!r} has no closed registration")
    if type(fact_set) is not IntentFactSet:
        raise TypeError("fact_set must be an exact IntentFactSet")
    if type(source_inventory) is not SourceInventory:
        raise TypeError("source_inventory must be an exact SourceInventory")
    verify_intent_fact_authority(
        fact_set=fact_set,
        source_inventory=source_inventory,
    )
    if type(observed_relations) is not tuple or any(
        type(row) is not _ObservedRelation for row in observed_relations
    ):
        raise TypeError(
            "observed_relations must be an exact tuple of ID-free relations"
        )
    if tuple(sorted(observed_relations)) != observed_relations or len(
        set(observed_relations)
    ) != len(observed_relations):
        raise ValueError("observed relations must be sorted and unique")

    expected_by_relation = {
        (fact.kind, fact.subjects, fact.payload_json): fact.fact_id
        for fact in fact_set.facts
        if consumer in fact.required_consumers
    }
    if not expected_by_relation:
        raise ValueError(f"consumer {consumer!r} has no fact obligations")
    if len(expected_by_relation) != sum(
        consumer in fact.required_consumers for fact in fact_set.facts
    ):
        raise ValueError("consumer obligations contain duplicate semantic relations")
    observed_keys = {
        (row.kind, row.subjects, row.payload_json) for row in observed_relations
    }
    expected_keys = set(expected_by_relation)
    missing = tuple(
        sorted(expected_by_relation[key] for key in expected_keys - observed_keys)
    )
    if missing:
        raise ValueError(
            "observed fact projection differs from exact consumer obligations: "
            f"missing={missing}"
        )

    matched = tuple(
        row
        for row in observed_relations
        if (row.kind, row.subjects, row.payload_json) in expected_keys
    )
    normalized = [row.canonical_data() for row in matched]
    return ConsumerProjection(
        consumer=consumer,
        fact_ids=tuple(
            sorted(
                expected_by_relation[(row.kind, row.subjects, row.payload_json)]
                for row in matched
            )
        ),
        normalized_projection_sha256=_projection_sha256(normalized),
    )


def _canonical_projection(
    values: object,
) -> tuple[tuple[str, int, int, int, tuple[tuple[str, int, int], ...]], ...]:
    if type(values) is not tuple:
        raise TypeError("floorplan_column_groups must be an exact tuple")
    rows: list[
        tuple[str, int, int, int, tuple[tuple[str, int, int], ...]]
    ] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 5:
            raise TypeError("floorplan projection rows must be exact five-tuples")
        group, tier, column, member_count, raw_terms = raw
        if type(group) is not str or not group or group != group.strip():
            raise TypeError("floorplan projection group must be a canonical string")
        if type(tier) is not int or tier < _GLOBAL_FLOORPLAN_COLUMN_SCOPE:
            raise TypeError(
                "floorplan projection tier must be global (-1) or non-negative"
            )
        if type(column) is not int or column < 0:
            raise TypeError(
                "floorplan projection column must be a non-negative exact int"
            )
        if type(member_count) is not int or member_count <= 0:
            raise TypeError(
                "floorplan projection member_count must be a positive exact int"
            )
        if type(raw_terms) is not tuple or not raw_terms:
            raise TypeError("floorplan projection terms must be a non-empty exact tuple")
        terms: list[tuple[str, int, int]] = []
        for raw_term in raw_terms:
            if type(raw_term) is not tuple or len(raw_term) != 3:
                raise TypeError("floorplan projection terms must be exact three-tuples")
            master, count, center_offset2_sum = raw_term
            if type(master) is not str or not master or master != master.strip():
                raise TypeError("floorplan projection master must be a canonical string")
            if type(count) is not int or count <= 0:
                raise TypeError("floorplan projection term count must be positive")
            if type(center_offset2_sum) is not int or center_offset2_sum < 0:
                raise TypeError(
                    "floorplan projection center sum must be a non-negative exact int"
                )
            terms.append((master, count, center_offset2_sum))
        canonical_terms = tuple(sorted(terms))
        if tuple(raw_terms) != canonical_terms or len(
            {master for master, _count, _offset in terms}
        ) != len(terms):
            raise ValueError("floorplan projection terms must be sorted and unique")
        if sum(count for _master, count, _offset in terms) != member_count:
            raise ValueError("floorplan projection terms do not conserve members")
        rows.append((group, tier, column, member_count, canonical_terms))
    canonical = tuple(sorted(rows))
    if tuple(values) != canonical or len({row[0] for row in rows}) != len(rows):
        raise ValueError("floorplan projection rows must be sorted with unique groups")
    return canonical


def build_placement_fact_projection(
    *,
    consumer: str,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
    place_groups: tuple[object, ...],
    route_groups: tuple[object, ...],
    floorplan_column_groups: tuple[tuple[object, ...], ...],
    resolved_device_masters: tuple[tuple[str, str], ...],
    envelope_order_edges: tuple[tuple[str, str], ...] = (),
    rigid_nonmos_macro_plans: object = None,
    well_island_share_plans: tuple[object, ...] = (),
    shared_well_guard_envelope_plans: tuple[object, ...] = (),
    shared_well_guard_child_plans: tuple[object, ...] = (),
    flexible_envelope_constraints: tuple[object, ...] = (),
) -> ConsumerProjection:
    """Reconstruct one pre-solver placement relation projection.

    The current placement-group declarations and the already normalized solver
    membership rows are the evidence.  Signed fact IDs are attached only after
    an independently reconstructed relation body matches exactly.
    """

    if consumer not in {"placement_initial", "placement_replay"}:
        raise ValueError("placement projection requires an exact placement consumer")
    if type(place_groups) is not tuple or type(route_groups) is not tuple:
        raise TypeError("placement group declarations must be exact tuples")
    floorplan_rows = _canonical_projection(floorplan_column_groups)
    if type(resolved_device_masters) is not tuple:
        raise TypeError("resolved device-to-master membership must be an exact tuple")
    canonical_resolution: list[tuple[str, str]] = []
    for row in resolved_device_masters:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("resolved device-to-master row must be an exact pair")
        device, master = row
        if (
            type(device) is not str
            or not device
            or device != device.strip()
            or type(master) is not str
            or not master
            or master != master.strip()
        ):
            raise TypeError("resolved device-to-master row is malformed")
        canonical_resolution.append((device, master))
    if tuple(canonical_resolution) != tuple(sorted(canonical_resolution)) or len(
        {device for device, _master in canonical_resolution}
    ) != len(canonical_resolution):
        raise ValueError("resolved device-to-master rows must be sorted and unique")
    master_by_device = dict(canonical_resolution)

    from clay.layout.place_blocks_2d import FlexibleEnvelopeConstraint
    from clay.layout.shared_well_hierarchy import SharedWellGuardChildPlan
    from clay.layout.well_island_sharing import (
        SharedWellGuardEnvelopePlan,
        WellIslandSharePlan,
    )

    well_inputs = (
        well_island_share_plans,
        shared_well_guard_envelope_plans,
        shared_well_guard_child_plans,
        flexible_envelope_constraints,
    )
    if any(type(rows) is not tuple for rows in well_inputs):
        raise TypeError("placement shared-well evidence must use exact tuples")
    expected_types = (
        WellIslandSharePlan,
        SharedWellGuardEnvelopePlan,
        SharedWellGuardChildPlan,
        FlexibleEnvelopeConstraint,
    )
    for rows, expected_type in zip(well_inputs, expected_types, strict=True):
        if any(type(row) is not expected_type for row in rows):
            raise TypeError("placement shared-well evidence contains a foreign row")
    well_plan_inputs = well_inputs[:3]
    if any(well_plan_inputs) and not all(well_plan_inputs):
        raise ValueError(
            "placement shared-well evidence requires plans and exact local owners"
        )
    well_rows_by_group = []
    if all(well_plan_inputs):
        if not flexible_envelope_constraints:
            raise ValueError(
                "placement shared-well evidence lacks flexible constraints"
            )
        mappings = []
        for rows in well_inputs:
            groups = tuple(getattr(row, "group") for row in rows)
            if groups != tuple(sorted(set(groups))):
                raise ValueError(
                    "placement shared-well evidence must be canonical and unique"
                )
            mappings.append({getattr(row, "group"): row for row in rows})
        expected_groups = set(mappings[0])
        if any(set(mapping) != expected_groups for mapping in mappings[1:3]):
            raise ValueError("placement shared-well evidence inventories differ")
        if not expected_groups <= set(mappings[3]):
            raise ValueError(
                "placement shared-well evidence lacks flexible constraints"
            )
        for group in sorted(expected_groups):
            share = mappings[0][group]
            envelope = mappings[1][group]
            child = mappings[2][group]
            constraint = mappings[3][group]
            if (
                envelope.body != share.body
                or envelope.well != share.well
                or envelope.members != share.members
                or child.body != share.body
                or child.well != share.well
                or child.envelope_sha256 != envelope.sha256
            ):
                raise ValueError(
                    f"shared well {group!r} source and owner authority differ"
                )
            expected_local_owners = tuple(
                sorted(
                    zip(
                        envelope.local_ring_members,
                        envelope.local_ring_sha256s,
                        strict=True,
                    )
                )
            )
            observed_local_owners = tuple(
                sorted(
                    (master.members, master.source_ring_sha256)
                    for master in child.masters
                )
            )
            child_owner_ids = tuple(master.block_id for master in child.masters)
            if (
                observed_local_owners != expected_local_owners
                or tuple(sorted(child_owner_ids)) != child_owner_ids
                or constraint.members != child_owner_ids
                or constraint.extension_nm != child.ring_extension_nm
            ):
                raise ValueError(
                    f"shared well {group!r} lacks its exact local owner partition"
                )
            expected_master_by_device = {
                member: master.block_id
                for master in child.masters
                for member in master.members
            }
            if tuple(sorted(expected_master_by_device)) != share.members:
                raise ValueError(
                    f"shared well {group!r} local owners do not conserve members"
                )
            missing_members = tuple(
                member
                for member in share.members
                if master_by_device.get(member)
                != expected_master_by_device[member]
            )
            if missing_members:
                raise ValueError(
                    f"shared well {group!r} solver membership differs: "
                    f"members={missing_members}"
                )
            foreign_members = tuple(
                sorted(
                    device
                    for device, master in canonical_resolution
                    if master in set(child_owner_ids)
                    and device not in set(share.members)
                )
            )
            if foreign_members:
                raise ValueError(
                    f"shared well {group!r} has foreign local owner members: "
                    f"{foreign_members}"
                )
            well_rows_by_group.append((group, share))
    actual: set[_ObservedRelation] = set()

    def add(
        kind: IntentFactKind,
        subjects: tuple[EntityRef, ...],
        payload: dict[str, object],
    ) -> None:
        relation = _relation_from_payload(kind, subjects, payload)
        if relation in actual:
            raise ValueError("placement producer emitted a duplicate relation")
        actual.add(relation)

    if rigid_nonmos_macro_plans is not None:
        from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanSet

        if type(rigid_nonmos_macro_plans) is not RigidNonmosMacroPlanSet:
            raise TypeError(
                "placement rigid non-MOS authority has a foreign type"
            )
        for plan in rigid_nonmos_macro_plans.plans:
            missing = tuple(
                member for member in plan.members if member not in master_by_device
            )
            if missing:
                raise ValueError(
                    f"placement rigid macro {plan.macro_id!r} lacks solver "
                    f"members: {missing}"
                )
            add(
                IntentFactKind.MEMBER_PARTITION,
                (
                    EntityRef(EntityKind.MODULE, plan.macro_id),
                    *(
                        EntityRef(EntityKind.DEVICE, member)
                        for member in plan.members
                    ),
                ),
                {"complete": True},
            )

    for group, share in well_rows_by_group:
        domain = EntityRef(EntityKind.WELL_DOMAIN, group)
        add(
            IntentFactKind.WELL_DOMAIN,
            tuple(
                sorted(
                    (
                        domain,
                        *(
                            EntityRef(EntityKind.DEVICE, member)
                            for member in share.members
                        ),
                    )
                )
            ),
            {"body_relation": "same_body", "sharing": "shared"},
        )
        add(
            IntentFactKind.NO_FOREIGN_ENVELOPE,
            (domain,),
            {"scope": "members"},
        )

    members_by_group: dict[str, tuple[str, ...]] = {}
    all_place_members: set[str] = set()
    normalized_place_groups: list[tuple[str, tuple[str, ...]]] = []
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
            raise TypeError("placement place-group declaration is malformed")
        canonical_members = tuple(sorted(members))
        if len(set(members)) != len(members):
            raise ValueError("placement place-group members are not unique")
        normalized_place_groups.append((name, canonical_members))
    if len({name for name, _members in normalized_place_groups}) != len(
        normalized_place_groups
    ):
        raise ValueError("placement place-group declarations are not unique")
    for name, members in sorted(normalized_place_groups):
        overlap = tuple(sorted(all_place_members.intersection(members)))
        if overlap:
            raise ValueError(
                "placement place-group members are not an exact partition: "
                f"overlap={overlap}"
            )
        all_place_members.update(members)
        members_by_group[name] = members
        subjects = (
            EntityRef(EntityKind.PLACE_GROUP, name),
            *(EntityRef(EntityKind.DEVICE, member) for member in members),
        )
        for kind in (
            IntentFactKind.MEMBER_PARTITION,
            IntentFactKind.MEMBER_CONSERVATION,
        ):
            add(kind, subjects, {"complete": True})
    place_names = frozenset(members_by_group)
    missing_resolution = tuple(sorted(all_place_members - set(master_by_device)))
    if missing_resolution:
        raise ValueError(
            "solver membership is missing source group members: "
            f"{missing_resolution}"
        )
    if type(envelope_order_edges) is not tuple or any(
        type(edge) is not tuple
        or len(edge) != 2
        or any(type(name) is not str or not name for name in edge)
        for edge in envelope_order_edges
    ):
        raise TypeError("placement envelope order edges must be exact string pairs")
    if envelope_order_edges != tuple(sorted(set(envelope_order_edges))):
        raise ValueError("placement envelope order edges must be sorted and unique")
    for left, right in envelope_order_edges:
        if left == right or left not in place_names or right not in place_names:
            raise ValueError("placement envelope order edge names foreign groups")
        subjects = tuple(
            sorted(
                (
                    EntityRef(EntityKind.PLACE_GROUP, left),
                    EntityRef(EntityKind.PLACE_GROUP, right),
                )
            )
        )
        add(
            IntentFactKind.ORDER,
            subjects,
            {
                "axis": "horizontal",
                "relation": (
                    "left_of" if subjects[0].name == left else "right_of"
                ),
            },
        )

    for declaration in route_groups:
        name = getattr(declaration, "group", None)
        members = getattr(declaration, "place_groups", None)
        if type(name) is not str or not name or type(members) is not tuple:
            raise TypeError("placement route-group declaration is malformed")
        if not members:
            members = (name,)
        if (
            members != tuple(sorted(set(members)))
            or any(member not in place_names for member in members)
        ):
            raise ValueError("placement route-group members are not canonical owners")
        subjects = (
            EntityRef(EntityKind.ROUTE_GROUP, name),
            *(EntityRef(EntityKind.PLACE_GROUP, member) for member in members),
        )
        for kind in (
            IntentFactKind.MEMBER_PARTITION,
            IntentFactKind.MEMBER_CONSERVATION,
        ):
            add(kind, subjects, {"complete": True})

    ordinal_by_group: dict[str, tuple[int, int]] = {}
    global_column_by_group: dict[str, tuple[int, int]] = {}
    for name, tier, column, member_count, terms in floorplan_rows:
        if name not in place_names:
            raise ValueError("placement floorplan row names a foreign place group")
        source_members = members_by_group[name]
        if member_count != len(source_members):
            raise ValueError(
                f"solver membership count differs for place group {name!r}"
            )
        expected_counts = tuple(
            sorted(Counter(master_by_device[item] for item in source_members).items())
        )
        projected_counts = tuple((master, count) for master, count, _sum in terms)
        if projected_counts != expected_counts:
            raise ValueError(
                f"solver membership differs from source group {name!r}: "
                f"expected={expected_counts}, projected={projected_counts}"
            )
        if tier == _GLOBAL_FLOORPLAN_COLUMN_SCOPE:
            global_column_by_group[name] = (0, column)
        else:
            ordinal_by_group[name] = (tier, column)
            add(
                IntentFactKind.ORDINAL_CELL,
                (EntityRef(EntityKind.PLACE_GROUP, name),),
                {"tier": tier, "column": column},
            )

    actual.update(_ordinal_order_relations(ordinal_by_group))
    actual.update(_ordinal_order_relations(global_column_by_group))
    # A semantic order fact retains the same coordinate-free ordinal relation
    # at the placement-consumer boundary.  Its stronger geometric predicate is
    # measured separately by the G4 receipt against final group envelopes; the
    # semantic token must remain present here so centroid and envelope facts
    # cannot alias in the signed projection.
    base_actual = set(actual)
    for fact in fact_set.facts:
        if fact.kind is not IntentFactKind.ORDER or consumer not in fact.required_consumers:
            continue
        payload = fact.payload
        semantic = payload.get("semantic")
        if semantic is None:
            continue
        base = dict(payload)
        del base["semantic"]
        if _relation_from_payload(fact.kind, fact.subjects, base) in base_actual:
            add(fact.kind, fact.subjects, payload)
    for left_index, left_name in enumerate(sorted(ordinal_by_group)):
        for right_name in sorted(ordinal_by_group)[left_index + 1 :]:
            left_ref = EntityRef(EntityKind.PLACE_GROUP, left_name)
            right_ref = EntityRef(EntityKind.PLACE_GROUP, right_name)
            subjects = (left_ref, right_ref)
            left_tier, left_column = ordinal_by_group[left_name]
            right_tier, right_column = ordinal_by_group[right_name]
            if left_tier == right_tier and left_column != right_column:
                before = left_column < right_column
                if abs(left_column - right_column) == 1:
                    for axis, relation in (
                        ("column", "left_of" if before else "right_of"),
                        ("horizontal", "adjacent"),
                        ("x", "left_of" if before else "right_of"),
                    ):
                        add(
                            IntentFactKind.ADJACENCY,
                            subjects,
                            {"axis": axis, "relation": relation},
                        )
            if left_column == right_column and left_tier != right_tier:
                below = left_tier < right_tier
                if abs(left_tier - right_tier) == 1:
                    for axis, relation in (
                        ("tier", "below" if below else "above"),
                        ("vertical", "adjacent"),
                        ("y", "below" if below else "above"),
                    ):
                        add(
                            IntentFactKind.ADJACENCY,
                            subjects,
                            {"axis": axis, "relation": relation},
                        )

    return _bind_observed_relations(
        consumer=consumer,
        fact_set=fact_set,
        source_inventory=source_inventory,
        observed_relations=tuple(sorted(actual)),
    )


def _floorplan_atoms_for_stage(
    basis: IntentBasis,
    stage: str,
) -> tuple[tuple[str, str, int, int], ...]:
    rows: list[tuple[str, str, int, int]] = []
    for atom in basis.atoms:
        if atom.kind != _FLOORPLAN_KIND:
            raise ValueError(
                "floorplan consumption basis contains a non-floorplan intent atom"
            )
        payload = atom.payload
        if set(payload) != {"group", "tier", "column"}:
            raise ValueError("floorplan intent atom payload is not canonical")
        group = payload["group"]
        tier = payload["tier"]
        column = payload["column"]
        if (
            type(group) is not str
            or not group
            or type(tier) is not int
            or tier < 0
            or type(column) is not int
            or column < 0
        ):
            raise ValueError("floorplan intent atom payload has malformed ordinals")
        if stage in atom.required_stages:
            rows.append((atom.atom_id, group, tier, column))
    return tuple(rows)


def bind_floorplan_consumption(
    basis: IntentBasis,
    *,
    stage: str,
    floorplan_column_groups: tuple[tuple[object, ...], ...],
) -> IntentConsumptionCertificate:
    """Bind one solver boundary to every floorplan atom required at ``stage``."""

    if type(basis) is not IntentBasis:
        raise TypeError("basis must be an exact IntentBasis")
    replayed_basis = IntentBasis.from_data(basis.canonical_data())
    if replayed_basis != basis:
        raise ValueError("intent basis does not replay byte-equivalently")
    if type(stage) is not str or not stage:
        raise TypeError("stage must be an exact non-empty string")
    expected = _floorplan_atoms_for_stage(basis, stage)
    if not expected:
        raise ValueError(f"intent basis declares no floorplan atoms for stage {stage!r}")

    projection = _canonical_projection(floorplan_column_groups)
    if not projection:
        raise ValueError("selected intent atom has an empty no-op projection")
    projected_by_group = {
        group: (tier, column)
        for group, tier, column, _member_count, _terms in projection
    }
    expected_by_group = {
        group: (tier, column) for _atom_id, group, tier, column in expected
    }
    missing = sorted(set(expected_by_group) - set(projected_by_group))
    if missing:
        raise ValueError(f"floorplan consumption is missing selected atoms: {missing}")
    undeclared = sorted(set(projected_by_group) - set(expected_by_group))
    if undeclared:
        raise ValueError(
            f"floorplan consumption contains undeclared projected atoms: {undeclared}"
        )
    mismatched = {
        group: (expected_by_group[group], projected_by_group[group])
        for group in sorted(expected_by_group)
        if expected_by_group[group] != projected_by_group[group]
    }
    if mismatched:
        raise ValueError(
            "floorplan consumption changes selected atom ordinals: "
            f"{mismatched!r}"
        )

    normalized = [
        [
            group,
            tier,
            column,
            member_count,
            [list(term) for term in terms],
        ]
        for group, tier, column, member_count, terms in projection
    ]
    digest = _projection_sha256(normalized)
    if digest == _EMPTY_PROJECTION_SHA256:
        raise ValueError("selected intent atom has a no-op projection")
    return IntentConsumptionCertificate(
        stage=stage,
        basis_sha256=basis.sha256,
        consumed_atom_ids=tuple(atom_id for atom_id, _group, _tier, _column in expected),
        normalized_projection_sha256=digest,
    )


def require_consumer_parity(
    expected_basis: IntentBasis,
    certificates: tuple[IntentConsumptionCertificate, ...],
) -> None:
    """Require exact coverage and equal normalized projection at every consumer."""

    if type(expected_basis) is not IntentBasis:
        raise TypeError("expected_basis must be an exact IntentBasis")
    replayed_basis = IntentBasis.from_data(expected_basis.canonical_data())
    if replayed_basis != expected_basis:
        raise ValueError("expected intent basis does not replay byte-equivalently")
    if type(certificates) is not tuple:
        raise TypeError("certificates must be an exact tuple")
    if any(type(item) is not IntentConsumptionCertificate for item in certificates):
        raise TypeError(
            "certificates must contain exact IntentConsumptionCertificate values"
        )
    stages = tuple(certificate.stage for certificate in certificates)
    if len(set(stages)) != len(stages):
        raise ValueError("intent consumption requires one unique consumer stage")
    required_stages = tuple(
        sorted(
            {
                stage
                for atom in expected_basis.atoms
                for stage in atom.required_stages
            }
        )
    )
    if tuple(sorted(stages)) != required_stages:
        raise ValueError(
            "intent consumption does not cover required consumer stages: "
            f"required={required_stages}, actual={tuple(sorted(stages))}"
        )
    for certificate in certificates:
        if certificate.basis_sha256 != expected_basis.sha256:
            raise ValueError("intent consumption certificate binds a foreign basis")
        expected_ids = tuple(
            atom.atom_id
            for atom in expected_basis.atoms
            if certificate.stage in atom.required_stages
        )
        if certificate.consumed_atom_ids != expected_ids:
            raise ValueError(
                "intent consumption certificate does not bind exact selected atoms"
            )
        if (
            expected_ids
            and certificate.normalized_projection_sha256
            == _EMPTY_PROJECTION_SHA256
        ):
            raise ValueError("selected intent atom has a no-op projection")
    projection_digests = {
        certificate.normalized_projection_sha256 for certificate in certificates
    }
    if len(projection_digests) > 1:
        raise ValueError("intent consumers violate normalized projection parity")


__all__ = [
    "bind_floorplan_consumption",
    "build_placement_fact_projection",
    "require_consumer_parity",
]
