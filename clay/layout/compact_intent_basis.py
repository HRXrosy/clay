"""Adapter between compact floorplan candidates and removable intent atoms."""

from __future__ import annotations

from dataclasses import replace

from clay.layout.compact_intent import (
    CompactIntentCandidate,
    replay_compact_intent_candidate,
)
from clay.layout.intent_basis import (
    ClaimScope,
    InformationOrigin,
    IntentAtom,
    IntentBasis,
)


_FLOORPLAN_KIND = "group_floorplan.cell"
_FLOORPLAN_STAGES = ("exact_materialization_replay", "initial_master")


def _require_exact_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field} must be an exact non-empty string")
    return value


def _atom_id(group: str) -> str:
    return f"group_floorplan:{group}:cell"


def group_floorplan_basis(
    candidate: CompactIntentCandidate,
    *,
    circuit_id: str,
    acceptance_policy_sha256: str,
    reference_receipt_sha256: str,
) -> IntentBasis:
    """Atomize every reference floorplan cell without changing its ordinals."""

    if type(candidate) is not CompactIntentCandidate:
        raise TypeError("candidate must be an exact CompactIntentCandidate")
    replayed = replay_compact_intent_candidate(candidate)
    _require_exact_string(circuit_id, field="circuit_id")
    if not replayed.group_floorplan:
        raise ValueError("group_floorplan basis requires a non-empty upper bound")

    atoms = tuple(
        sorted(
            (
                IntentAtom.from_payload(
                    atom_id=_atom_id(group),
                    kind=_FLOORPLAN_KIND,
                    payload={"group": group, "tier": tier, "column": column},
                    origin=InformationOrigin.TARGET_REFERENCE_OBSERVATION,
                    cost_bits=(
                        8
                        + len(group.encode("utf-8"))
                        + (tier + 1).bit_length()
                        + (column + 1).bit_length()
                    ),
                    required_stages=_FLOORPLAN_STAGES,
                )
                for group, tier, column in replayed.group_floorplan
            ),
            key=lambda atom: atom.atom_id,
        )
    )
    return IntentBasis(
        circuit_id=circuit_id,
        claim_scope=ClaimScope.REFERENCE_CONDITIONED_SINGLE_CIRCUIT,
        base_candidate_sha256=replayed.digest,
        acceptance_policy_sha256=acceptance_policy_sha256,
        target_reference_receipt_sha256=reference_receipt_sha256,
        atoms=atoms,
    )


def select_group_floorplan_atoms(
    candidate: CompactIntentCandidate,
    basis: IntentBasis,
    selected_atom_ids: tuple[str, ...],
) -> CompactIntentCandidate:
    """Return the base candidate with exactly the selected floorplan cells."""

    if type(candidate) is not CompactIntentCandidate:
        raise TypeError("candidate must be an exact CompactIntentCandidate")
    if type(basis) is not IntentBasis:
        raise TypeError("basis must be an exact IntentBasis")
    if type(selected_atom_ids) is not tuple:
        raise TypeError("selected_atom_ids must be an exact tuple")
    if any(type(atom_id) is not str or not atom_id for atom_id in selected_atom_ids):
        raise TypeError("selected_atom_ids must contain exact non-empty strings")
    if tuple(sorted(set(selected_atom_ids))) != selected_atom_ids:
        raise ValueError("selected_atom_ids must contain sorted unique values")

    replayed_candidate = replay_compact_intent_candidate(candidate)
    replayed_basis = IntentBasis.from_data(basis.canonical_data())
    if replayed_basis != basis:
        raise ValueError("intent basis does not replay byte-equivalently")
    if basis.target_reference_receipt_sha256 is None:
        raise ValueError("group_floorplan basis lacks its target reference receipt")
    expected_basis = group_floorplan_basis(
        replayed_candidate,
        circuit_id=basis.circuit_id,
        acceptance_policy_sha256=basis.acceptance_policy_sha256,
        reference_receipt_sha256=basis.target_reference_receipt_sha256,
    )
    if basis != expected_basis:
        raise ValueError("group_floorplan basis is stale relative to its base candidate")

    known_ids = frozenset(atom.atom_id for atom in basis.atoms)
    unknown = sorted(set(selected_atom_ids) - known_ids)
    if unknown:
        raise ValueError(f"selected atom IDs are outside the basis: {unknown}")
    selected = frozenset(selected_atom_ids)
    return replace(
        replayed_candidate,
        group_floorplan=tuple(
            row
            for row in replayed_candidate.group_floorplan
            if _atom_id(row[0]) in selected
        ),
    )


__all__ = ["group_floorplan_basis", "select_group_floorplan_atoms"]
