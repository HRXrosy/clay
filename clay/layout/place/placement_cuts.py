"""Validated placement feedback cuts and their exact model predicates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.place.placement_constants import (
    _Bbox,
)


@dataclass(frozen=True)
class _PreparedSolveCut:
    """Runtime-neutral geometry carried from one exact typed solve cut."""

    cut_hash: str
    placement_hash: str
    candidate_pool_hash: str
    owner_a: str = ""
    owner_b: str = ""
    corridor_relation: str = ""
    required_clearance_nm: int = 0
    incumbent_origins: Tuple[Tuple[str, int, int], ...] = ()
    placement_basis_sha256: str = ""
    guarded_coordinates: Tuple[Tuple[str, str, int], ...] = ()

def _is_canonical_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

def _cut_satisfied_by_bboxes(
    cut: _PreparedSolveCut, bboxes: Mapping[str, _Bbox]
) -> bool:
    if cut.incumbent_origins:
        try:
            final_origins = tuple(
                (name, bboxes[name][0], bboxes[name][1])
                for name, _x, _y in cut.incumbent_origins
            )
        except KeyError:
            return False
        return final_origins != cut.incumbent_origins
    if cut.owner_a not in bboxes or cut.owner_b not in bboxes:
        return False
    if cut.guarded_coordinates and any(
        bboxes[name][0 if axis == "x" else 1] != incumbent
        for name, axis, incumbent in cut.guarded_coordinates
        if name in bboxes
    ):
        return True
    a = bboxes[cut.owner_a]
    b = bboxes[cut.owner_b]
    clearance = cut.required_clearance_nm
    if cut.corridor_relation == "horizontal_separation":
        if cut.guarded_coordinates:
            return a[2] + clearance <= b[0] or b[2] <= a[0]
        return a[2] + clearance <= b[0] or b[2] + clearance <= a[0]
    if cut.corridor_relation == "vertical_separation":
        if cut.guarded_coordinates:
            return a[3] + clearance <= b[1] or b[3] <= a[1]
        return a[3] + clearance <= b[1] or b[3] + clearance <= a[1]
    return False

def _exact_cut_input_blocks(
    blocks: Sequence[Tuple[str, _Bbox]],
) -> Tuple[Tuple[str, _Bbox], ...]:
    """Canonicalize a cut incumbent without accepting numeric coercion."""

    result: List[Tuple[str, _Bbox]] = []
    for entry in blocks:
        try:
            name, raw_bbox = entry
        except (TypeError, ValueError) as exc:
            raise ValueError("cut blocks must contain (name, bbox) pairs") from exc
        if not isinstance(name, str) or not name:
            raise ValueError("cut block ids must be non-empty strings")
        if not isinstance(raw_bbox, (tuple, list)):
            raise ValueError("cut block bboxes must be tuple/list values")
        bbox = tuple(raw_bbox)
        if (
            len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            raise ValueError("cut block bboxes must be ordered exact integer tuples")
        result.append((name, bbox))
    names = [name for name, _bbox in result]
    if len(names) != len(set(names)):
        raise ValueError("cut block ids must be unique")
    return tuple(sorted(result))

def _prepare_solve_cuts(
    solve_cuts: object,
    solve_cut_proofs: object,
    items: Sequence[Tuple[str, _Bbox]],
    source_certificate: object,
) -> Tuple[Tuple[_PreparedSolveCut, ...], Optional[str]]:
    """Validate the closed Task-7 cut boundary without a module import cycle.

    The feedback module imports :class:`FinalPlacementCertificate` under
    ``TYPE_CHECKING`` only.  This reverse dependency is therefore deliberately
    local and is reached only for a non-empty cut request.  Invalid requests are
    returned as named certificate residuals instead of being ignored or raised.
    """

    try:
        raw_cuts = tuple(solve_cuts or ())
    except TypeError:
        return (), "RESIDUAL_UNKNOWN_SOLVE_CUT"
    if not raw_cuts:
        return (), None

    from clay.layout.place.resource_feedback import (
        ExactPlacementNoGoodProof,
        FinalPlacementCertificateView,
        PlacementCutReplayProof,
        PlacementRepairKind,
        PlacementSolveCut,
        final_placement_state_digest,
        lower_exact_placement_nogood,
        lower_placement_conflict,
        lower_route_planning_feedback,
    )

    for raw_cut in raw_cuts:
        if type(raw_cut) is not PlacementSolveCut:
            return (), "RESIDUAL_UNKNOWN_SOLVE_CUT"
    if not isinstance(source_certificate, FinalPlacementCertificateView):
        return (), "RESIDUAL_STALE_SOLVE_CUT"
    try:
        source_blocks = _exact_cut_input_blocks(tuple(source_certificate.blocks))
        source_digest = final_placement_state_digest(source_certificate)
        source_applied_hashes = tuple(source_certificate.applied_cut_hashes)
    except (AttributeError, IndexError, TypeError, ValueError):
        return (), "RESIDUAL_STALE_SOLVE_CUT"
    if source_blocks != tuple(items):
        return (), "RESIDUAL_STALE_SOLVE_CUT"
    if (
        source_applied_hashes != tuple(sorted(source_applied_hashes))
        or len(source_applied_hashes) != len(set(source_applied_hashes))
        or not all(_is_canonical_sha256(value) for value in source_applied_hashes)
    ):
        return (), "RESIDUAL_STALE_SOLVE_CUT"

    incumbent = {name: bbox for name, bbox in items}
    prepared: List[_PreparedSolveCut] = []
    seen_hashes: set = set()
    current_scopes: set = set()
    prior_cut_hashes: set = set()
    for raw_cut in raw_cuts:
        if raw_cut.repair_kind not in (
            PlacementRepairKind.RELATIVE_BLOCK_ORDER,
            PlacementRepairKind.CERTIFIED_CHANNEL_SPAN,
            PlacementRepairKind.CANDIDATE_LANDING_ADJACENCY,
            PlacementRepairKind.EXACT_PLACEMENT_NOGOOD,
        ):
            return (), "RESIDUAL_UNKNOWN_SOLVE_CUT"
        is_nogood = raw_cut.repair_kind is PlacementRepairKind.EXACT_PLACEMENT_NOGOOD
        if is_nogood:
            if (
                raw_cut.corridor_relation is not None
                or raw_cut.required_clearance_nm != 0
                or raw_cut.movable_site_ids
                or raw_cut.incumbent_origins
                != tuple(
                    (name, incumbent[name][0], incumbent[name][1])
                    for name in raw_cut.movable_block_ids
                    if name in incumbent
                )
                or tuple(name for name, _x, _y in raw_cut.incumbent_origins)
                != raw_cut.movable_block_ids
            ):
                return (), "RESIDUAL_STALE_SOLVE_CUT"
        elif (
            raw_cut.conflict_kind not in ("capacity_conflict", "hall_deficit")
            or raw_cut.corridor_relation
            not in (
                "horizontal_separation",
                "vertical_separation",
            )
            or raw_cut.required_clearance_nm <= 0
            or len(raw_cut.movable_block_ids) != 2
            or raw_cut.movable_block_ids[0] == raw_cut.movable_block_ids[1]
        ):
            return (), "RESIDUAL_UNKNOWN_SOLVE_CUT"
        if not is_nogood and (
            (
                raw_cut.conflict_kind == "capacity_conflict"
                and (
                    raw_cut.repair_kind
                    is PlacementRepairKind.CANDIDATE_LANDING_ADJACENCY
                    or raw_cut.movable_site_ids
                )
            )
            or (
                raw_cut.conflict_kind == "hall_deficit"
                and (
                    raw_cut.repair_kind
                    is not PlacementRepairKind.CANDIDATE_LANDING_ADJACENCY
                    or not raw_cut.movable_site_ids
                )
            )
        ):
            return (), "RESIDUAL_UNKNOWN_SOLVE_CUT"
        missing = set(raw_cut.movable_block_ids) - set(incumbent)
        if missing:
            return (), "RESIDUAL_CUT_MEMBER_MISSING"
        if not all(
            _is_canonical_sha256(value)
            for value in (
                raw_cut.conflict_hash,
                raw_cut.candidate_pool_hash,
                raw_cut.placement_hash,
                raw_cut.cut_hash,
            )
        ):
            return (), "RESIDUAL_STALE_SOLVE_CUT"
        cut = _PreparedSolveCut(
            cut_hash=raw_cut.cut_hash,
            placement_hash=raw_cut.placement_hash,
            candidate_pool_hash=raw_cut.candidate_pool_hash,
            owner_a=(
                ""
                if is_nogood
                else (
                    raw_cut.oriented_owner_block_ids[0]
                    if raw_cut.guarded_coordinates
                    else raw_cut.movable_block_ids[0]
                )
            ),
            owner_b=(
                ""
                if is_nogood
                else (
                    raw_cut.oriented_owner_block_ids[1]
                    if raw_cut.guarded_coordinates
                    else raw_cut.movable_block_ids[1]
                )
            ),
            corridor_relation="" if is_nogood else raw_cut.corridor_relation,
            required_clearance_nm=raw_cut.required_clearance_nm,
            incumbent_origins=raw_cut.incumbent_origins,
            placement_basis_sha256=(raw_cut.placement_basis_sha256 or ""),
            guarded_coordinates=raw_cut.guarded_coordinates,
        )
        if not is_nogood and raw_cut.guarded_coordinates:
            repair_axis = (
                "y" if raw_cut.corridor_relation == "vertical_separation" else "x"
            )
            expected_guard_keys = {
                (name, axis)
                for name in incumbent
                for axis in ("x", "y")
                if (name, axis) != (cut.owner_b, repair_axis)
            }
            if (
                not _is_canonical_sha256(raw_cut.placement_basis_sha256)
                or set(raw_cut.oriented_owner_block_ids)
                != set(raw_cut.movable_block_ids)
                or {(name, axis) for name, axis, _value in raw_cut.guarded_coordinates}
                != expected_guard_keys
            ):
                return (), "RESIDUAL_STALE_SOLVE_CUT"
        if cut.cut_hash in seen_hashes:
            return (), "RESIDUAL_STALE_SOLVE_CUT"
        satisfied_by_incumbent = _cut_satisfied_by_bboxes(cut, incumbent)
        if cut.placement_hash == source_digest:
            if satisfied_by_incumbent:
                return (), "RESIDUAL_STALE_SOLVE_CUT"
            current_scopes.add((cut.placement_hash, cut.candidate_pool_hash))
        else:
            if cut.cut_hash not in source_applied_hashes or not satisfied_by_incumbent:
                return (), "RESIDUAL_STALE_SOLVE_CUT"
            prior_cut_hashes.add(cut.cut_hash)
        seen_hashes.add(cut.cut_hash)
        prepared.append(cut)

    if (
        len(current_scopes) != 1
        or tuple(sorted(prior_cut_hashes)) != source_applied_hashes
    ):
        return (), "RESIDUAL_STALE_SOLVE_CUT"

    try:
        raw_proofs = tuple(solve_cut_proofs or ())
    except TypeError:
        return (), "RESIDUAL_CUT_PROOF_MISMATCH"
    if not raw_proofs:
        return (), "RESIDUAL_CUT_PROOF_MISSING"
    if len({id(proof) for proof in raw_proofs}) != len(raw_proofs):
        return (), "RESIDUAL_CUT_PROOF_DUPLICATE"

    replayed_by_hash: Dict[str, object] = {}
    for proof in raw_proofs:
        if type(proof) not in (PlacementCutReplayProof, ExactPlacementNoGoodProof):
            return (), "RESIDUAL_CUT_PROOF_MISMATCH"
        try:
            if type(proof) is ExactPlacementNoGoodProof:
                replayed = lower_exact_placement_nogood(proof.feedback_context)
            elif proof.feedback_context is not None:
                lowered = lower_route_planning_feedback(proof.feedback_context)
                if not isinstance(lowered, tuple) or len(lowered) != 2:
                    return (), "RESIDUAL_CUT_PROOF_MISMATCH"
                replayed, expected_proof = lowered
                if expected_proof != proof:
                    return (), "RESIDUAL_CUT_PROOF_MISMATCH"
            else:
                replayed = lower_placement_conflict(
                    pool=proof.pool,
                    conflict=proof.conflict,
                    witness=proof.witness,
                    final_placement_certificate=proof.final_placement_certificate,
                )
        except Exception:
            return (), "RESIDUAL_CUT_PROOF_MISMATCH"
        if type(replayed) is not PlacementSolveCut:
            return (), "RESIDUAL_CUT_PROOF_MISMATCH"
        if replayed.cut_hash in replayed_by_hash:
            return (), "RESIDUAL_CUT_PROOF_DUPLICATE"
        replayed_by_hash[replayed.cut_hash] = replayed

    if len(raw_proofs) < len(prepared):
        return (), "RESIDUAL_CUT_PROOF_MISSING"
    if len(raw_proofs) > len(prepared):
        return (), "RESIDUAL_CUT_PROOF_MISMATCH"
    for raw_cut in raw_cuts:
        replayed = replayed_by_hash.get(raw_cut.cut_hash)
        if replayed is None:
            return (), "RESIDUAL_CUT_PROOF_MISSING"
        if replayed != raw_cut:
            return (), "RESIDUAL_CUT_PROOF_MISMATCH"
    return tuple(sorted(prepared, key=lambda cut: cut.cut_hash)), None

def _cut_extent_padding(
    cuts: Sequence[_PreparedSolveCut], relation: str, gap: int
) -> int:
    return sum(
        max(0, cut.required_clearance_nm - gap)
        for cut in cuts
        if cut.corridor_relation == relation and not cut.guarded_coordinates
    )

def _required_cut_axis_extent(
    cuts: Sequence[_PreparedSolveCut],
    dims: Mapping[str, Tuple[int, int]],
    relation: str,
) -> int:
    axis = 0 if relation == "horizontal_separation" else 1
    return max(
        (
            dims[cut.owner_a][axis]
            + cut.required_clearance_nm
            + dims[cut.owner_b][axis]
            for cut in cuts
            if cut.corridor_relation == relation and not cut.guarded_coordinates
        ),
        default=0,
    )

def _add_solve_cut_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    cuts: Sequence[_PreparedSolveCut],
) -> None:
    """Add one orientation-neutral, two-way hard disjunction per signed pair."""

    for index, cut in enumerate(cuts):
        if cut.incumbent_origins:
            different = []
            for block_id, incumbent_x, incumbent_y in cut.incumbent_origins:
                x_diff = model.NewBoolVar(f"cut_{index}_{block_id}_x_diff")
                y_diff = model.NewBoolVar(f"cut_{index}_{block_id}_y_diff")
                model.Add(xs[block_id] != incumbent_x).OnlyEnforceIf(x_diff)
                model.Add(xs[block_id] == incumbent_x).OnlyEnforceIf(x_diff.Not())
                model.Add(ys[block_id] != incumbent_y).OnlyEnforceIf(y_diff)
                model.Add(ys[block_id] == incumbent_y).OnlyEnforceIf(y_diff.Not())
                different.extend((x_diff, y_diff))
            model.AddBoolOr(different)
            continue
        a, b = cut.owner_a, cut.owner_b
        clearance = cut.required_clearance_nm
        forward = model.NewBoolVar(f"cut_{index}_forward")
        reverse = model.NewBoolVar(f"cut_{index}_reverse")
        different = []
        for block_id, axis, incumbent in cut.guarded_coordinates:
            changed = model.NewBoolVar(f"cut_{index}_{block_id}_{axis}_changed")
            variable = xs[block_id] if axis == "x" else ys[block_id]
            model.Add(variable != incumbent).OnlyEnforceIf(changed)
            model.Add(variable == incumbent).OnlyEnforceIf(changed.Not())
            different.append(changed)
        if cut.corridor_relation == "horizontal_separation":
            model.Add(xs[a] + dims[a][0] + clearance <= xs[b]).OnlyEnforceIf(forward)
            reverse_clearance = 0 if cut.guarded_coordinates else clearance
            model.Add(xs[b] + dims[b][0] + reverse_clearance <= xs[a]).OnlyEnforceIf(
                reverse
            )
        else:
            model.Add(ys[a] + dims[a][1] + clearance <= ys[b]).OnlyEnforceIf(forward)
            reverse_clearance = 0 if cut.guarded_coordinates else clearance
            model.Add(ys[b] + dims[b][1] + reverse_clearance <= ys[a]).OnlyEnforceIf(
                reverse
            )
        model.AddBoolOr(tuple(different) + (forward, reverse))
