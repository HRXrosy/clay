"""Cheap, deterministic gates for removable group-floorplan intent atoms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from clay.layout.compact_intent import CompactIntentCandidate
from clay.layout.compact_intent_basis import select_group_floorplan_atoms
from clay.layout.intent_basis import IntentBasis, IntentConsumptionCertificate
from clay.layout.place.intent_consumption import require_consumer_parity


_SCHEMA = "clay.group-floorplan-static-gates/v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strings(data: dict[str, object], field: str) -> tuple[str, ...]:
    raw = data[field]
    if type(raw) is not list or any(type(item) is not str for item in raw):
        raise TypeError(f"{field} must be a list of strings")
    result = tuple(raw)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field} must contain sorted unique strings")
    return result


@dataclass(frozen=True)
class StaticGateReceipt:
    upper_basis_sha256: str
    selected_basis_sha256: str
    selected_atom_ids: tuple[str, ...]
    removed_atom_ids: tuple[str, ...]
    candidate_sha256: str
    replayed_candidate_sha256: str
    consumer_stages: tuple[str, ...]
    normalized_projection_sha256: str
    passed_gates: tuple[str, ...]
    status: str

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "upper_basis_sha256": self.upper_basis_sha256,
            "selected_basis_sha256": self.selected_basis_sha256,
            "selected_atom_ids": list(self.selected_atom_ids),
            "removed_atom_ids": list(self.removed_atom_ids),
            "candidate_sha256": self.candidate_sha256,
            "replayed_candidate_sha256": self.replayed_candidate_sha256,
            "consumer_stages": list(self.consumer_stages),
            "normalized_projection_sha256": self.normalized_projection_sha256,
            "passed_gates": list(self.passed_gates),
            "status": self.status,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "StaticGateReceipt":
        expected = {
            "schema", "upper_basis_sha256", "selected_basis_sha256",
            "selected_atom_ids", "removed_atom_ids", "candidate_sha256",
            "replayed_candidate_sha256", "consumer_stages",
            "normalized_projection_sha256", "passed_gates", "status",
        }
        if type(data) is not dict or set(data) != expected:
            raise ValueError("static gate receipt schema fields disagree")
        if data["schema"] != _SCHEMA:
            raise ValueError("static gate receipt schema differs")
        result = cls(
            upper_basis_sha256=data["upper_basis_sha256"],
            selected_basis_sha256=data["selected_basis_sha256"],
            selected_atom_ids=_strings(data, "selected_atom_ids"),
            removed_atom_ids=_strings(data, "removed_atom_ids"),
            candidate_sha256=data["candidate_sha256"],
            replayed_candidate_sha256=data["replayed_candidate_sha256"],
            consumer_stages=_strings(data, "consumer_stages"),
            normalized_projection_sha256=data["normalized_projection_sha256"],
            passed_gates=_strings(data, "passed_gates"),
            status=data["status"],
        )
        for digest in (
            result.upper_basis_sha256,
            result.selected_basis_sha256,
            result.candidate_sha256,
            result.replayed_candidate_sha256,
            result.normalized_projection_sha256,
        ):
            if type(digest) is not str or len(digest) != 64:
                raise ValueError("static gate receipt contains a malformed digest")
        if result.status != "pass" or result.passed_gates != ("G0", "G1", "G2", "G3"):
            raise ValueError("static gate receipt is not a complete pass")
        return result


def selected_intent_basis(
    upper_basis: IntentBasis,
    selected_atom_ids: tuple[str, ...],
) -> IntentBasis:
    selected = frozenset(selected_atom_ids)
    return IntentBasis(
        circuit_id=upper_basis.circuit_id,
        claim_scope=upper_basis.claim_scope,
        base_candidate_sha256=upper_basis.base_candidate_sha256,
        acceptance_policy_sha256=upper_basis.acceptance_policy_sha256,
        target_reference_receipt_sha256=upper_basis.target_reference_receipt_sha256,
        atoms=tuple(atom for atom in upper_basis.atoms if atom.atom_id in selected),
    )


def master_atom_closures(rows: object) -> tuple[tuple[str, ...], ...]:
    """Build implementation-local indivisible atom components from a projection."""

    if type(rows) is not list:
        raise TypeError("floorplan projection rows must be a list")
    master_groups: dict[str, set[str]] = {}
    all_groups: set[str] = set()
    for row in rows:
        if type(row) is not list or len(row) != 5 or type(row[0]) is not str:
            raise TypeError("floorplan projection row is malformed")
        group, _tier, _column, _member_count, terms = row
        if type(terms) is not list or not terms:
            raise TypeError("floorplan projection terms are malformed")
        all_groups.add(group)
        for term in terms:
            if type(term) is not list or len(term) != 3 or type(term[0]) is not str:
                raise TypeError("floorplan projection term is malformed")
            master_groups.setdefault(term[0], set()).add(group)
    neighbors = {group: {group} for group in all_groups}
    for groups in master_groups.values():
        for group in groups:
            neighbors[group].update(groups)
    closures: list[tuple[str, ...]] = []
    remaining = set(all_groups)
    while remaining:
        frontier = {min(remaining)}
        component: set[str] = set()
        while frontier:
            group = frontier.pop()
            if group in component:
                continue
            component.add(group)
            frontier.update(neighbors[group] - component)
        remaining.difference_update(component)
        if len(component) > 1:
            closures.append(tuple(sorted(
                f"group_floorplan:{group}:cell" for group in component
            )))
    return tuple(sorted(closures))


def selection_closure_violations(
    selected_atom_ids: tuple[str, ...],
    closures: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    selected = frozenset(selected_atom_ids)
    return tuple(
        closure
        for closure in closures
        if selected.intersection(closure) and not set(closure) <= selected
    )


def conditional_required_atoms(
    rows: object,
    axis_alignment_groups: object,
) -> tuple[str, ...]:
    """Atoms required whenever any column intent activates diff-axis partitioning."""

    if type(rows) is not list or type(axis_alignment_groups) is not list:
        raise TypeError("conditional closure inputs must be exact lists")
    groups_by_master: dict[str, set[str]] = {}
    for row in rows:
        if type(row) is not list or len(row) != 5 or type(row[0]) is not str:
            raise TypeError("floorplan projection row is malformed")
        for term in row[4]:
            if type(term) is not list or len(term) != 3 or type(term[0]) is not str:
                raise TypeError("floorplan projection term is malformed")
            groups_by_master.setdefault(term[0], set()).add(row[0])
    axis_masters: set[str] = set()
    for group in axis_alignment_groups:
        if type(group) is not list or not group:
            raise TypeError("axis alignment group is malformed")
        for term in group:
            if type(term) is not list or len(term) != 2 or type(term[0]) is not str:
                raise TypeError("axis alignment term is malformed")
            axis_masters.add(term[0])
    missing = tuple(sorted(axis_masters - set(groups_by_master)))
    if missing:
        raise ValueError(f"axis masters lack floorplan projection ownership: {missing}")
    return tuple(sorted({
        f"group_floorplan:{group}:cell"
        for master in axis_masters
        for group in groups_by_master[master]
    }))


def _projection_data(basis: IntentBasis) -> list[list[object]]:
    return [
        [atom.atom_id, atom.payload["group"], atom.payload["tier"], atom.payload["column"]]
        for atom in basis.atoms
    ]


def evaluate_group_floorplan_static_gates(
    *,
    source_path: Path,
    top_cell: str,
    upper_candidate: CompactIntentCandidate,
    upper_basis: IntentBasis,
    selected_atom_ids: tuple[str, ...],
    output_dir: Path,
) -> StaticGateReceipt:
    """Run G0 provenance, G1 replay, G2 parity, and G3 deletion causality."""

    from clay.flow.optimized_clay import (
        declared_compact_candidate,
        materialize_compact_intent_clay,
    )

    candidate = select_group_floorplan_atoms(
        upper_candidate, upper_basis, selected_atom_ids
    )
    selected_basis = selected_intent_basis(upper_basis, selected_atom_ids)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to reuse static gate output: {output}")
    output.mkdir(parents=True)
    materialized = output / "candidate.clay"
    materialize_compact_intent_clay(
        source_path=source_path,
        output_path=materialized,
        top_cell=top_cell,
        candidate=candidate,
        provenance={
            "truth": "staging",
            "verified": False,
            "optimality": "governed_static_gate_unevaluated",
            "upper_basis_sha256": upper_basis.sha256,
            "selected_basis_sha256": selected_basis.sha256,
        },
        promotion=False,
    )
    replayed = declared_compact_candidate(materialized.read_text(encoding="utf-8-sig"))
    if replayed is None or replayed.canonical_data() != candidate.canonical_data():
        raise ValueError("G1 materialize/reparse changed the selected candidate")
    selected = frozenset(selected_atom_ids)
    removed = tuple(atom.atom_id for atom in upper_basis.atoms if atom.atom_id not in selected)
    expected_rows = tuple(
        row for row in upper_candidate.group_floorplan
        if f"group_floorplan:{row[0]}:cell" in selected
    )
    if candidate.group_floorplan != expected_rows:
        raise ValueError("G1 changed fields differ from exact selected atoms")

    projection_sha256 = _sha256(_projection_data(selected_basis))
    stages = tuple(sorted({stage for atom in selected_basis.atoms for stage in atom.required_stages}))
    certificates = tuple(
        IntentConsumptionCertificate(
            stage=stage,
            basis_sha256=selected_basis.sha256,
            consumed_atom_ids=tuple(
                atom.atom_id for atom in selected_basis.atoms if stage in atom.required_stages
            ),
            normalized_projection_sha256=projection_sha256,
        )
        for stage in stages
    )
    require_consumer_parity(selected_basis, certificates)
    for atom in selected_basis.atoms:
        deletion_basis = selected_intent_basis(
            selected_basis,
            tuple(item.atom_id for item in selected_basis.atoms if item != atom),
        )
        if _sha256(_projection_data(deletion_basis)) == projection_sha256:
            raise ValueError(f"G3 deletion is a no-op for {atom.atom_id}")

    receipt = StaticGateReceipt(
        upper_basis_sha256=upper_basis.sha256,
        selected_basis_sha256=selected_basis.sha256,
        selected_atom_ids=selected_atom_ids,
        removed_atom_ids=removed,
        candidate_sha256=candidate.digest,
        replayed_candidate_sha256=replayed.digest,
        consumer_stages=stages,
        normalized_projection_sha256=projection_sha256,
        passed_gates=("G0", "G1", "G2", "G3"),
        status="pass",
    )
    (output / "static_gate_receipt.json").write_text(
        _canonical_json(receipt.canonical_data()) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "StaticGateReceipt",
    "conditional_required_atoms",
    "evaluate_group_floorplan_static_gates",
    "master_atom_closures",
    "selection_closure_violations",
    "selected_intent_basis",
]
