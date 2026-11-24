"""Compile coordinate-free shared-well declarations into PDK-bound plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from clay.layout.intent import (
    GuardRingIntent,
    WellIslandShareIntent,
    replay_well_island_share_intent,
)


class WellIslandSharingError(ValueError):
    """Typed fail-closed boundary for shared-well materialization."""


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise WellIslandSharingError(f"{field} must be one lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True, order=True)
class WellIslandSharePlan:
    """One canonical declaration resolved to physical members and a well layer."""

    group: str
    body: str
    well: str
    members: tuple[str, ...]
    layer_key: tuple[int, int]
    declaration_sha256: str

    def __post_init__(self) -> None:
        replay_well_island_share_intent(
            WellIslandShareIntent(group=self.group, body=self.body)
        )
        if self.well not in ("nwell", "pwell"):
            raise WellIslandSharingError("shared well must be nwell or pwell")
        if type(self.members) is not tuple:
            raise TypeError("shared well members must be an exact tuple")
        if not 2 <= len(self.members) <= 256:
            raise WellIslandSharingError(
                "shared well requires between 2 and 256 members"
            )
        if self.members != tuple(sorted(set(self.members))):
            raise WellIslandSharingError(
                "shared well members must be unique and canonical"
            )
        if (
            type(self.layer_key) is not tuple
            or len(self.layer_key) != 2
            or any(type(value) is not int or value < 0 for value in self.layer_key)
        ):
            raise TypeError("shared well layer_key must be an exact nonnegative tuple")
        _digest(self.declaration_sha256, "shared well declaration_sha256")

    def canonical_data(self) -> dict[str, object]:
        return {
            "group": self.group,
            "body": self.body,
            "well": self.well,
            "members": self.members,
            "layer_key": self.layer_key,
            "declaration_sha256": self.declaration_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class SharedWellGuardEnvelopePlan:
    """Coordinate-free proof that several local rings form one well envelope."""

    group: str
    body: str
    well: str
    members: tuple[str, ...]
    local_ring_members: tuple[tuple[str, ...], ...]
    local_ring_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        replay_well_island_share_intent(
            WellIslandShareIntent(group=self.group, body=self.body)
        )
        if self.well not in ("nwell", "pwell"):
            raise WellIslandSharingError("shared guard envelope has invalid well")
        if (
            type(self.members) is not tuple
            or self.members != tuple(sorted(set(self.members)))
        ):
            raise TypeError("shared guard envelope members must be canonical")
        if type(self.local_ring_members) is not tuple or len(
            self.local_ring_members
        ) < 2:
            raise TypeError(
                "shared guard envelope requires at least two exact local rings"
            )
        if self.local_ring_members != tuple(sorted(self.local_ring_members)):
            raise WellIslandSharingError(
                "shared guard envelope local rings must use canonical order"
            )
        consumed: set[str] = set()
        for ring in self.local_ring_members:
            if (
                type(ring) is not tuple
                or not ring
                or ring != tuple(sorted(set(ring)))
                or consumed.intersection(ring)
            ):
                raise WellIslandSharingError(
                    "shared guard envelope local rings must be exact and disjoint"
                )
            consumed.update(ring)
        if tuple(sorted(consumed)) != self.members:
            raise WellIslandSharingError(
                "shared guard envelope local rings do not cover exact members"
            )
        if (
            type(self.local_ring_sha256s) is not tuple
            or len(self.local_ring_sha256s) != len(self.local_ring_members)
        ):
            raise TypeError("shared guard envelope requires one SHA per local ring")
        for value in self.local_ring_sha256s:
            _digest(value, "shared guard envelope local ring SHA")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.shared-well-guard-envelope-plan/v1",
            "group": self.group,
            "body": self.body,
            "well": self.well,
            "members": self.members,
            "local_ring_members": self.local_ring_members,
            "local_ring_sha256s": self.local_ring_sha256s,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class WellDomainFactAuthority:
    """Source-replayed, coordinate-free authority for shared-well facts."""

    fact_set: Any
    source_inventory: Any
    share_plans: tuple[WellIslandSharePlan, ...]
    envelope_plans: tuple[SharedWellGuardEnvelopePlan, ...]
    atoms: tuple[Any, ...]
    entailments: tuple[Any, ...]

    def __post_init__(self) -> None:
        from clay.layout.intent_facts import (
            IntentFactSet,
            SourceInventory,
            verify_intent_fact_authority,
        )
        from clay.layout.intent_basis import IntentAtom
        from clay.layout.intent_closure import IntentEntailment

        if type(self.fact_set) is not IntentFactSet:
            raise TypeError("well-domain authority fact_set has a foreign type")
        if type(self.source_inventory) is not SourceInventory:
            raise TypeError("well-domain authority inventory has a foreign type")
        if (
            type(self.share_plans) is not tuple
            or not self.share_plans
            or any(type(plan) is not WellIslandSharePlan for plan in self.share_plans)
        ):
            raise TypeError("well-domain authority requires exact share plans")
        if (
            type(self.envelope_plans) is not tuple
            or any(
                type(plan) is not SharedWellGuardEnvelopePlan
                for plan in self.envelope_plans
            )
        ):
            raise TypeError("well-domain authority requires exact envelope plans")
        if tuple(plan.group for plan in self.share_plans) != tuple(
            sorted({plan.group for plan in self.share_plans})
        ):
            raise WellIslandSharingError(
                "well-domain authority share plans are not canonical"
            )
        envelope_by_group = {plan.group: plan for plan in self.envelope_plans}
        if len(envelope_by_group) != len(self.envelope_plans) or set(
            envelope_by_group
        ) != {plan.group for plan in self.share_plans}:
            raise WellIslandSharingError(
                "well-domain authority envelope inventory differs"
            )
        for share in self.share_plans:
            envelope = envelope_by_group[share.group]
            if (
                envelope.body != share.body
                or envelope.well != share.well
                or envelope.members != share.members
            ):
                raise WellIslandSharingError(
                    "well-domain authority envelope differs from source plan"
                )
        expected_facts = _well_domain_intent_facts(self.share_plans)
        if self.fact_set.facts != expected_facts:
            raise WellIslandSharingError(
                "well-domain authority facts differ from source plans"
            )
        if (
            type(self.atoms) is not tuple
            or any(type(atom) is not IntentAtom for atom in self.atoms)
            or self.atoms != _well_domain_intent_atoms(self.share_plans)
        ):
            raise WellIslandSharingError(
                "well-domain authority atoms differ from source plans"
            )
        expected_entailments = tuple(
            IntentEntailment(
                atom_id=atom.atom_id,
                fact_ids=tuple(
                    fact.fact_id
                    for fact in expected_facts
                    if any(
                        subject.kind.value == "well_domain"
                        and subject.name == atom.payload["group"]
                        for subject in fact.subjects
                    )
                ),
            )
            for atom in self.atoms
        )
        if (
            type(self.entailments) is not tuple
            or any(
                type(row) is not IntentEntailment for row in self.entailments
            )
            or self.entailments != expected_entailments
        ):
            raise WellIslandSharingError(
                "well-domain authority entailments differ from exact facts"
            )
        verify_intent_fact_authority(
            fact_set=self.fact_set,
            source_inventory=self.source_inventory,
        )


@dataclass(frozen=True, slots=True, order=True)
class SharedWellGuardRingBinding:
    """One outer-ring declaration bound to its exact local declarations."""

    group: str
    source_declaration_sha256s: tuple[str, ...]
    effective_declaration_sha256: str

    def __post_init__(self) -> None:
        if type(self.group) is not str or not self.group:
            raise TypeError("shared guard-ring binding group must be one string")
        if (
            type(self.source_declaration_sha256s) is not tuple
            or len(self.source_declaration_sha256s) < 2
            or len(set(self.source_declaration_sha256s))
            != len(self.source_declaration_sha256s)
        ):
            raise TypeError(
                "shared guard-ring binding requires exact unique source SHAs"
            )
        for value in self.source_declaration_sha256s:
            _digest(value, "shared guard-ring source declaration SHA")
        _digest(
            self.effective_declaration_sha256,
            "shared guard-ring effective declaration SHA",
        )


@dataclass(frozen=True, slots=True)
class SharedWellGuardRingInventory:
    """Effective ring inventory plus lossless local-to-outer provenance."""

    effective_guard_rings: tuple[GuardRingIntent, ...]
    envelope_bindings: tuple[SharedWellGuardRingBinding, ...]

    def __post_init__(self) -> None:
        if type(self.effective_guard_rings) is not tuple or any(
            type(value) is not GuardRingIntent
            for value in self.effective_guard_rings
        ):
            raise TypeError("effective guard rings must be an exact declaration tuple")
        if type(self.envelope_bindings) is not tuple or any(
            type(value) is not SharedWellGuardRingBinding
            for value in self.envelope_bindings
        ):
            raise TypeError("shared guard-ring bindings must be an exact tuple")
        if self.envelope_bindings != tuple(
            sorted(self.envelope_bindings, key=lambda value: value.group)
        ):
            raise WellIslandSharingError(
                "shared guard-ring bindings must use canonical group order"
            )
        effective_sha256s = tuple(
            value.sha256 for value in self.effective_guard_rings
        )
        for binding in self.envelope_bindings:
            if effective_sha256s.count(binding.effective_declaration_sha256) != 1:
                raise WellIslandSharingError(
                    "shared guard-ring binding lacks one effective declaration"
                )

    @property
    def source_declaration_sha256s(self) -> tuple[str, ...]:
        return tuple(
            digest
            for binding in self.envelope_bindings
            for digest in binding.source_declaration_sha256s
        )

    @property
    def envelope_ring_sha256s(self) -> tuple[str, ...]:
        return tuple(
            binding.effective_declaration_sha256
            for binding in self.envelope_bindings
        )


def replay_shared_well_guard_envelopes(
    envelope_plans: Any,
    guard_ring_intents: Any,
) -> tuple[SharedWellGuardEnvelopePlan, ...]:
    """Replay every local `(members, SHA)` row against source declarations."""

    if type(envelope_plans) is not tuple:
        raise TypeError("shared guard envelope inventory must be an exact tuple")
    if type(guard_ring_intents) is not tuple or any(
        type(value) is not GuardRingIntent for value in guard_ring_intents
    ):
        raise TypeError("shared guard envelope replay requires exact ring declarations")
    if any(type(value) is not SharedWellGuardEnvelopePlan for value in envelope_plans):
        raise TypeError("shared guard envelope inventory contains a foreign plan")
    if envelope_plans != tuple(sorted(envelope_plans, key=lambda value: value.group)):
        raise WellIslandSharingError(
            "shared guard envelope inventory must use canonical group order"
        )

    rings_by_sha: dict[str, list[GuardRingIntent]] = {}
    for declaration in guard_ring_intents:
        rings_by_sha.setdefault(declaration.sha256, []).append(declaration)
    consumed_sha256s: set[str] = set()
    replayed: list[SharedWellGuardEnvelopePlan] = []
    for value in envelope_plans:
        plan = SharedWellGuardEnvelopePlan(
            group=value.group,
            body=value.body,
            well=value.well,
            members=value.members,
            local_ring_members=value.local_ring_members,
            local_ring_sha256s=value.local_ring_sha256s,
        )
        expected_well = "n" if plan.well == "nwell" else "p"
        planned_members = set(plan.members)
        expected_rows = tuple(zip(plan.local_ring_members, plan.local_ring_sha256s))
        for members, declaration_sha256 in expected_rows:
            matches = rings_by_sha.get(declaration_sha256, ())
            if len(matches) != 1 or declaration_sha256 in consumed_sha256s:
                raise WellIslandSharingError(
                    f"shared well {plan.group!r} lacks an exact local guard-ring binding"
                )
            declaration = matches[0]
            if (
                tuple(sorted(declaration.members)) != members
                or declaration.well != expected_well
                or declaration.tie_net != plan.body
            ):
                raise WellIslandSharingError(
                    f"shared well {plan.group!r} lacks an exact local guard-ring binding"
                )
            consumed_sha256s.add(declaration_sha256)
        intersecting = tuple(
            sorted(
                declaration.sha256
                for declaration in guard_ring_intents
                if set(declaration.members).intersection(planned_members)
            )
        )
        if intersecting != tuple(sorted(plan.local_ring_sha256s)):
            raise WellIslandSharingError(
                f"shared well {plan.group!r} lacks an exact local guard-ring binding"
            )
        replayed.append(plan)
    return tuple(replayed)


def compile_shared_well_guard_ring_inventory(
    envelope_plans: Any,
    guard_ring_intents: Any,
) -> SharedWellGuardRingInventory:
    """Replace each proven local-ring partition by one effective outer ring."""

    plans = replay_shared_well_guard_envelopes(
        envelope_plans,
        guard_ring_intents,
    )
    plan_by_source_sha: dict[str, SharedWellGuardEnvelopePlan] = {}
    for plan in plans:
        for declaration_sha256 in plan.local_ring_sha256s:
            if declaration_sha256 in plan_by_source_sha:
                raise WellIslandSharingError(
                    "shared guard envelopes overlap on local declarations"
                )
            plan_by_source_sha[declaration_sha256] = plan

    emitted_groups: set[str] = set()
    effective: list[GuardRingIntent] = []
    bindings: list[SharedWellGuardRingBinding] = []
    declarations_by_sha = {
        declaration.sha256: declaration for declaration in guard_ring_intents
    }
    for declaration in guard_ring_intents:
        plan = plan_by_source_sha.get(declaration.sha256)
        if plan is None:
            effective.append(declaration)
            continue
        if plan.group in emitted_groups:
            continue
        local = tuple(
            declarations_by_sha[digest] for digest in plan.local_ring_sha256s
        )
        semantic_rows = {
            (item.hug, item.role, item.enclose) for item in local
        }
        if len(semantic_rows) != 1:
            raise WellIslandSharingError(
                f"shared well {plan.group!r} local rings differ in materialization semantics"
            )
        hug, role, enclose = next(iter(semantic_rows))
        outer = GuardRingIntent(
            members=plan.members,
            well="n" if plan.well == "nwell" else "p",
            tie_net=plan.body,
            hug=hug,
            role=role,
            enclose=enclose,
            # Combining multiple independently authored rings creates a new
            # aggregate hull.  Its member placement must therefore enter the
            # established hierarchy-aware prepack before this outer ring is
            # built, even when every source ring used the legacy empty pack.
            pack="hierarchical",
        )
        effective.append(outer)
        bindings.append(
            SharedWellGuardRingBinding(
                group=plan.group,
                source_declaration_sha256s=plan.local_ring_sha256s,
                effective_declaration_sha256=outer.sha256,
            )
        )
        emitted_groups.add(plan.group)
    if emitted_groups != {plan.group for plan in plans}:
        raise WellIslandSharingError("shared guard envelope did not lower exactly once")
    return SharedWellGuardRingInventory(
        effective_guard_rings=tuple(effective),
        envelope_bindings=tuple(sorted(bindings, key=lambda value: value.group)),
    )


def compile_shared_well_guard_envelopes(
    share_plans: Any,
    guard_ring_intents: Any,
) -> tuple[SharedWellGuardEnvelopePlan, ...]:
    """Bind shared wells to an exact partition of same-domain local rings.

    This is the ownership seam needed by a future flexible outer-ring
    materializer.  It deliberately contains no placement coordinate and does not
    merge the local rigid masters by itself.
    """

    from clay.layout.intent import GuardRingIntent

    if type(share_plans) is not tuple:
        raise TypeError("shared well guard envelopes require an exact plan tuple")
    if type(guard_ring_intents) is not tuple:
        raise TypeError("shared well guard envelopes require an exact ring tuple")
    if any(type(plan) is not WellIslandSharePlan for plan in share_plans):
        raise TypeError("shared well guard envelope contains a foreign plan")
    if any(type(ring) is not GuardRingIntent for ring in guard_ring_intents):
        raise TypeError("shared well guard envelope contains a foreign ring")

    envelopes: list[SharedWellGuardEnvelopePlan] = []
    consumed_ring_sha256s: set[str] = set()
    for plan in share_plans:
        expected_well = "n" if plan.well == "nwell" else "p"
        planned = set(plan.members)
        local: list[tuple[tuple[str, ...], str]] = []
        consumed_members: set[str] = set()
        for ring in guard_ring_intents:
            members = set(ring.members)
            if not members.intersection(planned):
                continue
            if (
                not members <= planned
                or ring.well != expected_well
                or ring.tie_net != plan.body
                or consumed_members.intersection(members)
                or ring.sha256 in consumed_ring_sha256s
            ):
                raise WellIslandSharingError(
                    f"shared well {plan.group!r} lacks an exact local guard-ring partition"
                )
            canonical_members = tuple(sorted(members))
            local.append((canonical_members, ring.sha256))
            consumed_members.update(members)
        if consumed_members != planned or len(local) < 2:
            raise WellIslandSharingError(
                f"shared well {plan.group!r} lacks an exact local guard-ring partition"
            )
        local.sort(key=lambda row: row[0])
        consumed_ring_sha256s.update(sha for _members, sha in local)
        envelopes.append(
            SharedWellGuardEnvelopePlan(
                group=plan.group,
                body=plan.body,
                well=plan.well,
                members=plan.members,
                local_ring_members=tuple(members for members, _sha in local),
                local_ring_sha256s=tuple(sha for _members, sha in local),
            )
        )
    return tuple(envelopes)


def compile_well_island_share_plans(
    cell: Any,
    declarations: Sequence[WellIslandShareIntent],
    tech: Any,
) -> tuple[WellIslandSharePlan, ...]:
    """Resolve shared-well declarations from canonical topology, without coordinates."""

    from clay.layout.annotate_to_solver import kind_polarity_oxide
    from clay.layout.functional_roles import functional_group_records_from_cell

    if type(declarations) is not tuple:
        raise TypeError("well island share declarations must be an exact tuple")
    if len(declarations) > 256:
        raise WellIslandSharingError("well island share inventory is too large")
    replayed = tuple(replay_well_island_share_intent(item) for item in declarations)
    if replayed != tuple(sorted(replayed, key=lambda item: item.group)):
        raise WellIslandSharingError(
            "well island share declarations must use canonical group order"
        )
    if len({item.group for item in replayed}) != len(replayed):
        raise WellIslandSharingError("duplicate well island share group")

    groups = {
        record.name: record.members
        for record in functional_group_records_from_cell(cell)
    }
    devices = {
        str(getattr(device, "instance_name", "") or ""): device
        for device in tuple(getattr(cell, "devices", ()) or ())
    }
    if "" in devices:
        raise WellIslandSharingError("canonical source contains an unnamed device")

    plans: list[WellIslandSharePlan] = []
    consumed: set[str] = set()
    for declaration in replayed:
        members = groups.get(declaration.group)
        if members is None:
            raise WellIslandSharingError(
                f"shared well references unknown functional group {declaration.group!r}"
            )
        members = tuple(sorted(str(member) for member in members))
        if not 2 <= len(members) <= 256:
            raise WellIslandSharingError(
                f"shared well group {declaration.group!r} requires 2..256 members"
            )
        foreign = tuple(sorted(set(members) - set(devices)))
        if foreign:
            raise WellIslandSharingError(
                f"shared well group contains unknown members {foreign!r}"
            )
        overlap = tuple(sorted(set(members) & consumed))
        if overlap:
            raise WellIslandSharingError(
                f"shared well groups overlap on members {overlap!r}"
            )

        polarities: set[str] = set()
        body_nets: set[str] = set()
        for member in members:
            device = devices[member]
            polarity, _oxide = kind_polarity_oxide(
                getattr(device, "device_class", "")
            )
            if str(getattr(device, "device_class", "")).lower() == "native":
                # The source role overlay already validates the PDK-native
                # contract and resolves polarity from PDK device bindings.
                # Limit the view to this member so mixed native/portable cells
                # do not reinterpret unrelated source devices.
                from types import SimpleNamespace
                from clay.frontend.native_role_overlay import _native_view

                cards, _infos, _diagnostics = _native_view(
                    SimpleNamespace(devices=(device,)), tech.pdk_id
                )
                polarity = cards[0].kind
            if polarity not in ("nmos", "pmos"):
                raise WellIslandSharingError(
                    f"shared well member {member!r} is not a supported MOS device"
                )
            polarities.add(polarity)
            pins = getattr(device, "pins", None)
            if not isinstance(pins, dict):
                try:
                    pins = dict(pins)
                except Exception as exc:
                    raise WellIslandSharingError(
                        f"shared well member {member!r} lacks canonical pins"
                    ) from exc
            body = pins.get("b")
            if type(body) is not str or not body:
                raise WellIslandSharingError(
                    f"shared well member {member!r} lacks a body net"
                )
            body_nets.add(body)
        if len(polarities) != 1:
            raise WellIslandSharingError(
                f"shared well group {declaration.group!r} crosses MOS polarities"
            )
        if body_nets != {declaration.body}:
            raise WellIslandSharingError(
                f"shared well group {declaration.group!r} body domain differs "
                f"from declaration {declaration.body!r}"
            )
        polarity = next(iter(polarities))
        well = "nwell" if polarity == "pmos" else "pwell"
        try:
            raw_layer = tech.build_layer(well)
            layer_key = (int(raw_layer[0]), int(raw_layer[1]))
        except Exception as exc:
            raise WellIslandSharingError(
                f"PDK {getattr(tech, 'pdk_id', '')!r} does not expose {well}"
            ) from exc
        plans.append(
            WellIslandSharePlan(
                group=declaration.group,
                body=declaration.body,
                well=well,
                members=members,
                layer_key=layer_key,
                declaration_sha256=declaration.sha256,
            )
        )
        consumed.update(members)
    return tuple(plans)


def _well_domain_intent_facts(
    share_plans: tuple[WellIslandSharePlan, ...],
) -> tuple[Any, ...]:
    """Derive the two coordinate-free semantic rows from source-bound plans."""

    from clay.layout.intent_facts import (
        EntityKind,
        EntityRef,
        FactAuthority,
        IntentFact,
        IntentFactKind,
        MINIMUM_REQUIRED_CONSUMERS,
    )

    required = tuple(
        sorted(MINIMUM_REQUIRED_CONSUMERS[IntentFactKind.WELL_DOMAIN])
    )
    facts = []
    for plan in share_plans:
        domain = EntityRef(EntityKind.WELL_DOMAIN, plan.group)
        members = tuple(
            EntityRef(EntityKind.DEVICE, member) for member in plan.members
        )
        facts.extend(
            (
                IntentFact.from_payload(
                    fact_id=f"well:{plan.group}",
                    kind=IntentFactKind.WELL_DOMAIN,
                    subjects=tuple(sorted((domain, *members))),
                    payload={
                        "body_relation": "same_body",
                        "sharing": "shared",
                    },
                    authority=FactAuthority.DESIGN,
                    required_consumers=required,
                ),
                IntentFact.from_payload(
                    fact_id=f"envelope:well_domain:{plan.group}",
                    kind=IntentFactKind.NO_FOREIGN_ENVELOPE,
                    subjects=(domain,),
                    payload={"scope": "members"},
                    authority=FactAuthority.DESIGN,
                    required_consumers=required,
                ),
            )
        )
    return tuple(sorted(facts, key=lambda fact: fact.fact_id))


def _well_domain_intent_atoms(
    share_plans: tuple[WellIslandSharePlan, ...],
) -> tuple[Any, ...]:
    from clay.layout.intent_basis import InformationOrigin, IntentAtom

    return tuple(
        IntentAtom.from_payload(
            atom_id=f"well_domain:{plan.group}:shared",
            kind="well_domain.shared",
            payload={"body": plan.body, "group": plan.group},
            origin=InformationOrigin.DESIGN_INPUT,
            cost_bits=(
                8
                + len(plan.group.encode("utf-8"))
                + len(plan.body.encode("utf-8"))
            ),
            required_stages=(
                "exact_materialization_replay",
                "final_gds",
                "initial_master",
            ),
        )
        for plan in share_plans
    )


def build_well_domain_fact_authority(
    *,
    source_cell: Any,
    tech: Any,
    circuit_id: str,
    pdk_snapshot_sha256: str,
    pdk_installed_authority_sha256: str,
) -> WellDomainFactAuthority:
    """Replay current source bytes and derive exact shared-well fact authority.

    No caller-supplied member or relation inventory enters this boundary.  The
    canonical source file is reread, its topology inventory is hashed, and both
    the shared-domain members and local-ring partition are recompiled before the
    immutable fact set is constructed.
    """

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.intent_basis import ClaimScope
    from clay.layout.intent_facts import (
        EntityKind,
        IntentFactSet,
        SourceInventory,
    )
    from clay.layout.intent_closure import IntentEntailment
    from clay.layout.route.exact_artifact_evidence import (
        _canonical_source_member_inventory,
        _replayed_source_cell_authority,
    )

    (
        source_sha256,
        topology_sha256,
        _member_classes,
        canonical_cell,
    ) = _replayed_source_cell_authority(source_cell)
    if canonical_cell is None:
        raise WellIslandSharingError(
            "well-domain authority requires canonical source bytes"
        )
    intent = layout_intent_from_cell_def(canonical_cell)
    plans = compile_well_island_share_plans(
        canonical_cell,
        tuple(intent.well_island_shares),
        tech,
    )
    if not plans:
        raise WellIslandSharingError(
            "well-domain authority requires a declared shared well"
        )
    envelopes = compile_shared_well_guard_envelopes(
        plans,
        tuple(intent.guard_rings),
    )
    source_members = tuple(
        sorted(str(row[0]) for row in _canonical_source_member_inventory(canonical_cell))
    )
    domains = tuple(plan.group for plan in plans)
    source_inventory = SourceInventory.from_inventory(
        circuit_id=circuit_id,
        source_sha256=source_sha256,
        topology_sha256=topology_sha256,
        inventory={
            EntityKind.DEVICE: source_members,
            EntityKind.WELL_DOMAIN: domains,
        },
    )
    fact_set = IntentFactSet(
        circuit_id=circuit_id,
        claim_scope=ClaimScope.REFERENCE_FREE_SINGLE_CIRCUIT,
        source_sha256=source_sha256,
        topology_sha256=topology_sha256,
        pdk_snapshot_sha256=pdk_snapshot_sha256,
        pdk_installed_authority_sha256=pdk_installed_authority_sha256,
        facts=_well_domain_intent_facts(plans),
    )
    atoms = _well_domain_intent_atoms(plans)
    entailments = tuple(
        IntentEntailment(
            atom_id=atom.atom_id,
            fact_ids=tuple(
                fact.fact_id
                for fact in fact_set.facts
                if any(
                    subject.kind is EntityKind.WELL_DOMAIN
                    and subject.name == atom.payload["group"]
                    for subject in fact.subjects
                )
            ),
        )
        for atom in atoms
    )
    return WellDomainFactAuthority(
        fact_set=fact_set,
        source_inventory=source_inventory,
        share_plans=plans,
        envelope_plans=envelopes,
        atoms=atoms,
        entailments=entailments,
    )


def replay_well_island_share_plans(
    plans: Any,
    tech: Any,
) -> tuple[WellIslandSharePlan, ...]:
    """Replay exact plan bytes against the installed PDK layer authority."""

    if type(plans) is not tuple:
        raise TypeError("well island share plans must be an exact tuple")
    replayed: list[WellIslandSharePlan] = []
    consumed: set[str] = set()
    for value in plans:
        if type(value) is not WellIslandSharePlan:
            raise TypeError(
                "well island share plan inventory requires exact "
                "WellIslandSharePlan values"
            )
        plan = WellIslandSharePlan(
            group=value.group,
            body=value.body,
            well=value.well,
            members=value.members,
            layer_key=value.layer_key,
            declaration_sha256=value.declaration_sha256,
        )
        raw_layer = tech.build_layer(plan.well)
        if (
            type(raw_layer) is not tuple
            or len(raw_layer) != 2
            or any(type(item) is not int or item < 0 for item in raw_layer)
        ):
            raise WellIslandSharingError(
                f"installed PDK exposes malformed {plan.well} layer authority"
            )
        if plan.layer_key != raw_layer:
            raise WellIslandSharingError(
                f"shared well {plan.group!r} layer differs from installed PDK"
            )
        overlap = tuple(sorted(set(plan.members) & consumed))
        if overlap:
            raise WellIslandSharingError(
                f"shared well plans overlap on members {overlap!r}"
            )
        consumed.update(plan.members)
        replayed.append(plan)
    canonical = tuple(sorted(replayed, key=lambda item: item.group))
    if tuple(replayed) != canonical:
        raise WellIslandSharingError(
            "well island share plans must use canonical group order"
        )
    if len({item.group for item in replayed}) != len(replayed):
        raise WellIslandSharingError("duplicate well island share plan group")
    return tuple(replayed)


def _snap_interval_inside(
    lo: int,
    hi: int,
    target_center: int,
    target_width: int,
    grid: int,
) -> tuple[int, int]:
    """Return a grid-aligned ``[s0, s1]`` interval strictly inside ``[lo, hi]``."""

    if grid <= 1:
        w = max(1, min(hi - lo, target_width))
        s0 = max(lo, min(hi - w, target_center - w // 2))
        return s0, s0 + w
    g_lo = ((lo + grid - 1) // grid) * grid
    g_hi = (hi // grid) * grid
    if g_hi - g_lo < grid:
        w = max(1, min(hi - lo, target_width))
        s0 = max(lo, min(hi - w, target_center - w // 2))
        return s0, s0 + w
    span = g_hi - g_lo
    rounded_w = ((target_width + grid // 2) // grid) * grid
    w = min(span, max(grid, rounded_w))
    ideal_s0 = target_center - w // 2
    snapped_s0 = ((ideal_s0 + grid // 2) // grid) * grid
    s0 = max(g_lo, min(g_hi - w, snapped_s0))
    return s0, s0 + w


def _interior_boxes(poly: Any, kdb: Any) -> tuple[tuple[int, int, int, int], ...]:
    """Return deterministic solid interior boxes contained within one polygon."""

    if poly.is_box():
        box = poly.bbox()
        return ((int(box.left), int(box.bottom), int(box.right), int(box.top)),)
    candidates: list[tuple[int, int, int, int]] = []
    poly_region = kdb.Region(poly)
    for trap in poly.decompose_trapezoids():
        if trap.is_box():
            tb = trap.bbox()
            candidates.append(
                (int(tb.left), int(tb.bottom), int(tb.right), int(tb.top))
            )
            continue
        pts = tuple((int(pt.x), int(pt.y)) for pt in trap.each_point_hull())
        if len(pts) < 3:
            continue
        ys = sorted({y for _, y in pts})
        y0, y1 = ys[0], ys[-1]
        if y1 <= y0:
            continue
        top_xs = [x for x, y in pts if y == y1]
        bot_xs = [x for x, y in pts if y == y0]
        if top_xs and bot_xs:
            x0 = max(min(top_xs), min(bot_xs))
            x1 = min(max(top_xs), max(bot_xs))
            if x1 > x0 and (kdb.Region(kdb.Box(x0, y0, x1, y1)) - poly_region).is_empty():
                candidates.append((x0, y0, x1, y1))
    if not candidates:
        raise WellIslandSharingError(
            "declared shared well polygon has no solid orthogonal interior"
        )
    max_min_dim = max(min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in candidates)
    core = tuple(
        sorted(
            rect
            for rect in candidates
            if min(rect[2] - rect[0], rect[3] - rect[1]) >= max_min_dim // 2
        )
    )
    return core or (max(candidates, key=lambda r: (min(r[2] - r[0], r[3] - r[1]), (r[2] - r[0]) * (r[3] - r[1]), r)),)


def _box_pair_priority(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    dx = max(0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0, max(ay0, by0) - min(ay1, by1))
    cx_diff = (ax0 + ax1) - (bx0 + bx1)
    cy_diff = (ay0 + ay1) - (by0 + by1)
    return (dx + dy, cx_diff * cx_diff + cy_diff * cy_diff, box_a, box_b)


def _grid_bridge_boxes(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    grid: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Build grid-aligned interior bridge boxes joining box_a and box_b."""

    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    min_dim = min(ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0)
    target_w = max(1, min_dim // 2)
    req_ov = max(1, min(min_dim, ((target_w + max(1, grid) // 2) // max(1, grid)) * max(1, grid) if grid > 1 else target_w))

    ax_s0, ax_s1 = _snap_interval_inside(ax0, ax1, (ax0 + ax1) // 2, target_w, grid)
    ay_s0, ay_s1 = _snap_interval_inside(ay0, ay1, (ay0 + ay1) // 2, target_w, grid)
    bx_s0, bx_s1 = _snap_interval_inside(bx0, bx1, (bx0 + bx1) // 2, target_w, grid)
    by_s0, by_s1 = _snap_interval_inside(by0, by1, (by0 + by1) // 2, target_w, grid)

    ov_x0, ov_x1 = max(ax0, bx0), min(ax1, bx1)
    if ov_x1 - ov_x0 >= req_ov:
        vx0, vx1 = _snap_interval_inside(
            ov_x0, ov_x1, (ov_x0 + ov_x1) // 2, target_w, grid
        )
        return ((vx0, min(ay_s0, by_s0), vx1, max(ay_s1, by_s1)),)

    ov_y0, ov_y1 = max(ay0, by0), min(ay1, by1)
    if ov_y1 - ov_y0 >= req_ov:
        hy0, hy1 = _snap_interval_inside(
            ov_y0, ov_y1, (ov_y0 + ov_y1) // 2, target_w, grid
        )
        return ((min(ax_s0, bx_s0), hy0, max(ax_s1, bx_s1), hy1),)

    horiz = (min(ax_s0, bx_s0), ay_s0, max(ax_s1, bx_s1), ay_s1)
    vert = (bx_s0, min(ay_s0, by_s0), bx_s1, max(ay_s1, by_s1))
    return (horiz, vert)


def materialize_well_island_share_region(
    region: Any,
    *,
    grid_nm: int = 5,
) -> Any:
    """Connect every declared member-well component with deterministic well straps.

    The input is already partitioned by one declaration, so this operation can
    never join another body domain.  It decomposes separated islands into real
    interior boxes, spans them with grid-snapped orthogonal straps, and verifies
    that the resulting physical region is one connected island.  The exact-GDS
    checker later measures that fact again from the written artifact; this
    function is generation, not signoff evidence.
    """

    import klayout.db as kdb

    if type(region) is not kdb.Region:
        raise TypeError("shared well materializer requires one exact KLayout Region")
    if isinstance(grid_nm, bool) or type(grid_nm) is not int or grid_nm <= 0:
        raise ValueError("shared well materializer grid_nm must be a positive integer")
    out = region.dup()
    out.merge()
    polygons = tuple(kdb.Polygon(poly) for poly in out.each_merged())
    if not polygons:
        raise WellIslandSharingError("shared well declaration has no well artwork")
    if len(polygons) == 1:
        return out

    def bbox_key(poly: Any) -> tuple[int, int, int, int]:
        box = poly.bbox()
        return (int(box.left), int(box.bottom), int(box.right), int(box.top))

    ordered = tuple(sorted(polygons, key=bbox_key))
    component_boxes = [_interior_boxes(poly, kdb) for poly in ordered]
    connected_boxes = list(component_boxes[0])
    remaining = set(range(1, len(ordered)))
    while remaining:
        best_choice = min(
            (
                (*_box_pair_priority(box_a, box_b), next_idx)
                for next_idx in sorted(remaining)
                for box_a in connected_boxes
                for box_b in component_boxes[next_idx]
            )
        )
        _, _, box_a, box_b, next_idx = best_choice
        bridge = _grid_bridge_boxes(box_a, box_b, grid_nm)
        for sx0, sy0, sx1, sy1 in bridge:
            out.insert(kdb.Box(sx0, sy0, sx1, sy1))
        connected_boxes.extend(component_boxes[next_idx])
        connected_boxes.extend(bridge)
        remaining.remove(next_idx)
    out.merge()
    if len(tuple(out.each_merged())) != 1:
        raise WellIslandSharingError(
            "declared shared well could not materialize one connected island"
        )
    return out


__all__ = [
    "SharedWellGuardEnvelopePlan",
    "SharedWellGuardRingBinding",
    "SharedWellGuardRingInventory",
    "WellDomainFactAuthority",
    "WellIslandSharePlan",
    "WellIslandSharingError",
    "build_well_domain_fact_authority",
    "compile_shared_well_guard_envelopes",
    "compile_shared_well_guard_ring_inventory",
    "compile_well_island_share_plans",
    "materialize_well_island_share_region",
    "replay_well_island_share_plans",
    "replay_shared_well_guard_envelopes",
]
