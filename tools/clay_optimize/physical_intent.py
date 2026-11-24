"""Target-PDK cheap physical evaluation for canonical compact intent.

This module is circuit-neutral.  It materializes one
``CompactIntentCandidate`` back into Clay, runs the existing placement spine,
and reports deterministic physical proxies used to choose expensive signoff
candidates.  It owns no placer, router, PDK branch, or circuit-name policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

from clay.flow.optimized_clay import materialize_compact_intent_clay
from clay.layout.compact_intent import CompactIntentCandidate
from clay.parser import parse_clay_file


class PhysicalIntentInfeasible(ValueError):
    """A known, deterministic rejection of one materialized physical point."""


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GROUP_ID = re.compile(r"[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*")
_MAX_COORDINATE_NM = 10**15
_MAX_CHEAP_MEASURE = 1.0e18
_MAX_CHEAP_COUNT = 2**31 - 1
_CHEAP_PASS_KEYS = frozenset(
    {
        "schema",
        "input_sha256",
        "status",
        "candidate",
        "candidate_sha256",
        "candidate_source",
        "candidate_source_sha256",
        "metrics",
    }
)
_CHEAP_FAIL_KEYS = frozenset(
    {
        "schema",
        "input_sha256",
        "status",
        "outcome",
        "candidate",
        "candidate_sha256",
        "error_type",
        "error",
    }
)


def _cheap_measure(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an exact bounded number")
    try:
        measured = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an exact bounded number") from exc
    lower_ok = measured > 0.0 if positive else measured >= 0.0
    if not math.isfinite(measured) or not lower_ok or measured > _MAX_CHEAP_MEASURE:
        raise ValueError(f"{label} must be an exact bounded number")
    return 0.0 if measured == 0.0 else measured


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("physical-intent cache has a duplicate or invalid key")
        result[key] = value
    return result


@dataclass(frozen=True)
class PhysicalIntentFunctionalGroupBBox:
    """One candidate-derived functional-group envelope in integer nanometers."""

    group_id: str
    bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if (
            type(self.group_id) is not str
            or _GROUP_ID.fullmatch(self.group_id) is None
            or Path(self.group_id).suffix.casefold() == ".png"
        ):
            raise ValueError("functional-group bbox ID must be canonical")
        if type(self.bbox_nm) is not tuple or len(self.bbox_nm) != 4 or any(
            type(value) is not int or abs(value) > _MAX_COORDINATE_NM
            for value in self.bbox_nm
        ):
            raise ValueError("functional-group bbox must contain four bounded integers")
        if self.bbox_nm[2] <= self.bbox_nm[0] or self.bbox_nm[3] <= self.bbox_nm[1]:
            raise ValueError("functional-group bbox must have positive extent")

    def metrics_data(self) -> dict[str, object]:
        return {"group_id": self.group_id, "bbox_nm": list(self.bbox_nm)}


@dataclass(frozen=True)
class PhysicalIntentCheapResult:
    candidate_sha256: str
    candidate_source: Path
    bbox_nm: tuple[int, int, int, int]
    bbox_area_um2: float
    whitespace_um2: float
    wire_length_um: float
    via_count: int
    ring_perimeter_um: float
    ring_max_perimeter_um: float
    ring_span_waste_um2: float
    physical_signature_sha256: str
    macro_consumption_sha256: str = ""
    placement_state_sha256: str = ""
    functional_group_bboxes_nm: tuple[PhysicalIntentFunctionalGroupBBox, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.candidate_sha256) is not str
            or _SHA256.fullmatch(self.candidate_sha256) is None
        ):
            raise ValueError("candidate_sha256 must be a canonical SHA-256")
        if (
            not isinstance(self.candidate_source, Path)
            or self.candidate_source.suffix.casefold() != ".clay"
        ):
            raise ValueError("candidate_source must be a Clay source path")
        if type(self.bbox_nm) is not tuple or len(self.bbox_nm) != 4 or any(
            type(value) is not int or abs(value) > _MAX_COORDINATE_NM
            for value in self.bbox_nm
        ):
            raise ValueError("candidate bbox must contain four bounded integers")
        if self.bbox_nm[2] <= self.bbox_nm[0] or self.bbox_nm[3] <= self.bbox_nm[1]:
            raise ValueError("candidate bbox must have positive extent")
        for field in (
            "bbox_area_um2",
            "whitespace_um2",
            "wire_length_um",
            "ring_perimeter_um",
            "ring_max_perimeter_um",
            "ring_span_waste_um2",
        ):
            object.__setattr__(
                self,
                field,
                _cheap_measure(
                    getattr(self, field),
                    label=field,
                    positive=(field == "bbox_area_um2"),
                ),
            )
        expected_area = (
            (self.bbox_nm[2] - self.bbox_nm[0])
            * (self.bbox_nm[3] - self.bbox_nm[1])
            / 1_000_000.0
        )
        if self.bbox_area_um2 != expected_area:
            raise ValueError("bbox_area_um2 disagrees with candidate bbox")
        if self.whitespace_um2 > self.bbox_area_um2:
            raise ValueError("whitespace_um2 exceeds candidate bbox area")
        if self.ring_max_perimeter_um > self.ring_perimeter_um + 1e-9:
            raise ValueError("ring_max_perimeter_um exceeds total ring perimeter")
        if type(self.via_count) is not int or not 0 <= self.via_count <= _MAX_CHEAP_COUNT:
            raise ValueError("via_count must be a bounded non-negative integer")
        if (
            type(self.physical_signature_sha256) is not str
            or _SHA256.fullmatch(self.physical_signature_sha256) is None
        ):
            raise ValueError("physical_signature_sha256 must be a canonical SHA-256")
        if type(self.macro_consumption_sha256) is not str or (
            self.macro_consumption_sha256
            and _SHA256.fullmatch(self.macro_consumption_sha256) is None
        ):
            raise ValueError("macro_consumption_sha256 must be empty or a SHA-256")
        if type(self.placement_state_sha256) is not str or (
            self.placement_state_sha256
            and _SHA256.fullmatch(self.placement_state_sha256) is None
        ):
            raise ValueError("placement_state_sha256 must be empty or a SHA-256")
        if bool(self.macro_consumption_sha256) != bool(self.placement_state_sha256):
            raise ValueError("macro and placement evidence must be present together")
        if type(self.functional_group_bboxes_nm) is not tuple or any(
            type(item) is not PhysicalIntentFunctionalGroupBBox
            for item in self.functional_group_bboxes_nm
        ):
            raise ValueError("functional_group_bboxes_nm must be an exact typed tuple")
        ordered = tuple(
            sorted(self.functional_group_bboxes_nm, key=lambda item: item.group_id)
        )
        if len({item.group_id for item in ordered}) != len(ordered):
            raise ValueError("functional_group_bboxes_nm repeats a group ID")
        x0, y0, x1, y1 = self.bbox_nm
        if any(
            item.bbox_nm[0] < x0
            or item.bbox_nm[1] < y0
            or item.bbox_nm[2] > x1
            or item.bbox_nm[3] > y1
            for item in ordered
        ):
            raise ValueError("functional-group bbox lies outside the candidate bbox")
        object.__setattr__(self, "functional_group_bboxes_nm", ordered)

    def metrics(self) -> dict[str, object]:
        return {
            "bbox_nm": list(self.bbox_nm),
            "bbox_width_nm": self.bbox_nm[2] - self.bbox_nm[0],
            "bbox_height_nm": self.bbox_nm[3] - self.bbox_nm[1],
            "bbox_area_um2": self.bbox_area_um2,
            "whitespace_um2": self.whitespace_um2,
            "wire_length_um": self.wire_length_um,
            "via_count": self.via_count,
            "ring_perimeter_um": self.ring_perimeter_um,
            "ring_max_perimeter_um": self.ring_max_perimeter_um,
            "ring_span_waste_um2": self.ring_span_waste_um2,
            "physical_signature_sha256": self.physical_signature_sha256,
            "macro_consumption_sha256": self.macro_consumption_sha256,
            "placement_state_sha256": self.placement_state_sha256,
            "functional_group_bboxes_nm": [
                item.metrics_data() for item in self.functional_group_bboxes_nm
            ],
        }


class UnsignablePlacementError(RuntimeError):
    """The measured candidate lacks a signable solver-quality certificate."""


def _require_signable_placement_certificate(certificate: object) -> object:
    """Reject a cheap point whose own placement proof cannot reach signoff.

    A fallback can be a useful interactive picture, but it is not evidence that
    the declared compact point was solved.  Likewise geometry, solve-cut, and
    aspect residuals cannot be hidden by a low proxy area.  These outcomes are
    deliberately ordinary runtime failures rather than
    :class:`PhysicalIntentInfeasible`: a timeout/fallback is not a proof that
    the physical point can never solve, so the search records it as
    ``unmeasurable`` and never turns it into a permanent negative cache rule.
    """

    from clay.layout.place_blocks_2d import placement_certificate_is_signable

    if certificate is None:
        raise UnsignablePlacementError("placement certificate is missing")
    if placement_certificate_is_signable(certificate):
        return certificate
    if bool(getattr(certificate, "used_fallback", False)):
        kind = str(getattr(certificate, "fallback_kind", "") or "unspecified")
        raise UnsignablePlacementError(f"placement fallback is not signable: {kind}")
    status = str(getattr(certificate, "status", "") or "")
    if status not in {"OPTIMAL", "FEASIBLE"}:
        raise UnsignablePlacementError(
            f"placement status is not signable: {status or 'missing'}"
        )
    if getattr(certificate, "geometry_legal", None) is not True:
        raise UnsignablePlacementError("placement geometry is not certified legal")
    if getattr(certificate, "cuts_satisfied", None) is not True:
        raise UnsignablePlacementError("placement solve cut is not certified satisfied")
    if getattr(certificate, "aspect_satisfied", None) is not True:
        measured = getattr(certificate, "measured_aspect", None)
        declared = getattr(certificate, "declared_aspect", None)
        residual = getattr(certificate, "aspect_residual", None)
        raise UnsignablePlacementError(
            "placement aspect is not certified satisfied: "
            f"measured={measured} declared={declared} residual={residual}"
        )
    raise UnsignablePlacementError("placement certificate is not signable")


@dataclass(frozen=True)
class PhysicalIntentProbeBudget:
    """Declaration-owned bounded routing work used only for candidate ranking."""

    time_budget_sec: float
    max_iters: int
    max_expansions: int
    expansion_cap: int

    def canonical_data(self) -> dict[str, float | int]:
        return {
            "time_budget_sec": self.time_budget_sec,
            "max_iters": self.max_iters,
            "max_expansions": self.max_expansions,
            "expansion_cap": self.expansion_cap,
        }


@dataclass(frozen=True)
class PhysicalIntentProbeResult:
    """Typed non-signoff observation from a bounded canonical route attempt."""

    candidate_sha256: str
    status: str
    phase: str
    state: str
    reason: str
    net: str
    evidence_sha256: str
    details: tuple[tuple[str, object], ...] = ()

    def metrics(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "status": self.status,
            "phase": self.phase,
            "state": self.state,
            "reason": self.reason,
            "net": self.net,
            "evidence_sha256": self.evidence_sha256,
            "details": {str(key): value for key, value in self.details},
            "binding_truth": False,
            "evidence_scope": "bounded_route_probe",
        }


_PROBE_KEYS = (
    "opt_layout_probe_time_budget_s",
    "opt_layout_probe_max_iters",
    "opt_layout_probe_max_expansions",
    "opt_layout_probe_expansion_cap",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_intent_probe_budget(cell: object) -> PhysicalIntentProbeBudget | None:
    """Parse the existing ``sizing {}`` DATA surface into a typed probe budget."""

    sizing = dict(getattr(cell, "sizing_kwargs", {}) or {})
    present = tuple(key for key in _PROBE_KEYS if key in sizing)
    if not present:
        return None
    if len(present) != len(_PROBE_KEYS):
        missing = sorted(set(_PROBE_KEYS) - set(present))
        raise ValueError(
            "physical intent probe requires a complete declared contract; "
            f"missing {missing}"
        )

    def positive_number(key: str) -> float:
        value = sizing[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"physical intent probe {key} must be positive")
        return float(value)

    def positive_integer(key: str) -> int:
        value = positive_number(key)
        integer = int(round(value))
        if not math.isclose(value, integer, abs_tol=1e-12):
            raise ValueError(f"physical intent probe {key} must be an integer")
        return integer

    maximum = positive_integer("opt_layout_probe_max_expansions")
    cap = positive_integer("opt_layout_probe_expansion_cap")
    if cap < maximum:
        raise ValueError(
            "physical intent probe expansion cap must be at least max expansions"
        )
    return PhysicalIntentProbeBudget(
        time_budget_sec=positive_number("opt_layout_probe_time_budget_s"),
        max_iters=positive_integer("opt_layout_probe_max_iters"),
        max_expansions=maximum,
        expansion_cap=cap,
    )


def physical_intent_probe_rank(
    probe: PhysicalIntentProbeResult,
    *,
    cheap_metrics: Mapping[str, object],
) -> tuple[object, ...]:
    """Rank bounded route evidence without mistaking it for signoff truth."""

    details = {str(key): value for key, value in probe.details}

    def integer(key: str, default: int) -> int:
        value = details.get(key, default)
        return int(value) if isinstance(value, (int, float)) else default

    completed = probe.status == "complete"
    unmeasurable = probe.reason == "UNMEASURABLE" or probe.phase == "PROBE_ERROR"
    return (
        0 if completed else (2 if unmeasurable else 1),
        -integer("connected_net_count", 0),
        -integer("net_order_index", -1),
        integer("frontier_graph_states", 2**63 - 1),
        integer("endpoint_pair_count", 2**63 - 1),
        integer("solve_expansions_total", 2**63 - 1),
        float(cheap_metrics.get("bbox_area_um2", math.inf)),
        float(cheap_metrics.get("wire_length_um", math.inf)),
        probe.candidate_sha256,
    )


def load_physical_intent_cache(
    metrics_path: Path | str,
    *,
    candidate_source: Path | str,
    input_sha256: str,
    candidate: CompactIntentCandidate,
    expected_functional_group_ids: tuple[str, ...],
) -> dict[str, object] | None:
    """Load one state-bound cheap result, including deterministic red results.

    This cache contract belongs to the generic physical-intent evaluator rather
    than any circuit runner.  Both green and deterministic-red v2 rows bind the
    exact current candidate; red rows do not require a materialized source.
    """

    metrics_path = Path(metrics_path)
    candidate_source = Path(candidate_source)
    if type(candidate) is not CompactIntentCandidate:
        raise TypeError("physical-intent cache candidate must be exact typed")
    if type(input_sha256) is not str or _SHA256.fullmatch(input_sha256) is None:
        raise ValueError("physical-intent cache input_sha256 must be canonical")
    if candidate_source.suffix.casefold() != ".clay":
        return None
    if (
        type(expected_functional_group_ids) is not tuple
        or any(type(item) is not str for item in expected_functional_group_ids)
        or tuple(sorted(expected_functional_group_ids))
        != expected_functional_group_ids
        or len(set(expected_functional_group_ids))
        != len(expected_functional_group_ids)
    ):
        raise TypeError("cache functional-group IDs must be an exact sorted tuple")
    if not metrics_path.is_file():
        return None
    try:
        payload = json.loads(
            metrics_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
        if type(payload) is not dict:
            return None
        if payload.get("schema") != "clay.physical-compact-cheap/v2":
            return None
        if payload.get("input_sha256") != input_sha256:
            return None
        status = payload.get("status")
        if status == "pass":
            if set(payload) != _CHEAP_PASS_KEYS:
                return None
            if (
                payload.get("candidate") != candidate.canonical_data()
                or payload.get("candidate_sha256") != candidate.digest
                or payload.get("candidate_source") != str(candidate_source.resolve())
            ):
                return None
            if not candidate_source.is_file():
                return None
            source_sha256 = payload.get("candidate_source_sha256")
            if (
                type(source_sha256) is not str
                or _SHA256.fullmatch(source_sha256) is None
                or _file_sha256(candidate_source) != source_sha256
            ):
                return None
            metrics = payload.get("metrics")
            if type(metrics) is not dict:
                return None
            raw_groups = metrics.get("functional_group_bboxes_nm")
            if type(raw_groups) is not list or any(
                type(item) is not dict or type(item.get("group_id")) is not str
                for item in raw_groups
            ):
                return None
            group_ids = tuple(item["group_id"] for item in raw_groups)
            if group_ids != expected_functional_group_ids:
                return None
            return {"status": "pass", "metrics": dict(metrics)}
        if status == "fail":
            if set(payload) != _CHEAP_FAIL_KEYS:
                return None
            if (
                payload.get("outcome") != "infeasible"
                or payload.get("candidate") != candidate.canonical_data()
                or payload.get("candidate_sha256") != candidate.digest
            ):
                return None
            error_type = payload.get("error_type")
            error = payload.get("error")
            if type(error_type) is not str or type(error) is not str:
                return None
            return {
                "status": "fail",
                "error_type": error_type,
                "error": error,
            }
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def load_physical_intent_probe_cache(
    metrics_path: Path | str,
    *,
    input_sha256: str,
    candidate_sha256: str,
) -> PhysicalIntentProbeResult | None:
    """Load a state/budget/physical-signature-bound route observation."""

    path = Path(metrics_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "clay.physical-intent-route-probe/v1"
            or payload.get("input_sha256") != input_sha256
        ):
            return None
        probe = payload.get("probe")
        if not isinstance(probe, dict):
            return None
        raw_details = probe.get("details", {})
        if not isinstance(raw_details, dict):
            return None
        status = str(probe.get("status", ""))
        if status not in {"complete", "stopped"}:
            return None
        evidence = str(probe.get("evidence_sha256", ""))
        if len(evidence) != 64:
            return None
        return PhysicalIntentProbeResult(
            candidate_sha256=str(candidate_sha256),
            status=status,
            phase=str(probe.get("phase", "")),
            state=str(probe.get("state", "")),
            reason=str(probe.get("reason", "")),
            net=str(probe.get("net", "")),
            evidence_sha256=evidence,
            details=tuple(
                sorted((str(key), value) for key, value in raw_details.items())
            ),
        )
    except (OSError, ValueError, TypeError):
        return None


def _select_cell(path: Path, top_cell: str):
    parsed = parse_clay_file(path)
    cells = [cell for cell in parsed.cell_defs if cell.local_name == top_cell]
    if len(cells) != 1:
        raise ValueError(
            f"compact physical evaluation requires one {top_cell!r} cell; "
            f"found {len(cells)}"
        )
    return parsed, cells[0]


def _place_candidate_source(
    candidate_source: Path,
    *,
    top_cell: str,
    pdk: str,
    source_cir: Path,
):
    """Run the canonical pre-route placement stages for one materialized intent."""

    from clay.layout.envelope_place import maybe_envelope_replace
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place_refine import place_to_intent
    from clay.layout.policy import policy_for_cell

    parsed, cell = _select_cell(candidate_source, top_cell)
    intent = layout_intent_from_cell_def(cell)
    policy = policy_for_cell(parsed, cell)
    converged = place_to_intent(
        cell,
        intent,
        None,
        source_cir=source_cir,
        tech=str(pdk),
        max_iters=1,
        effective_policy=policy,
        engine=getattr(policy, "placement_engine", "milp"),
    )
    placement = converged.placement
    aspect = getattr(policy, "aspect_target", None)
    if aspect is not None:
        placement, _envelope = maybe_envelope_replace(
            placement,
            cell,
            intent,
            aspect_target=float(aspect),
            gap_um=float(getattr(policy, "gap_um", None) or 1.5),
            tech_id=str(pdk),
        )
    return parsed, cell, intent, policy, placement


def _bbox_area_nm2(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _physical_signature_sha256(
    *,
    pdk: str,
    bbox_nm: tuple[int, int, int, int],
    blocks: tuple[tuple[str, tuple[int, int, int, int]], ...],
    terminals: tuple[tuple[str, tuple[tuple[str, tuple[int, int]], ...]], ...],
) -> str:
    """Identify realized placement geometry, independent of intent spelling.

    A compact-intent digest must not enter this receipt: two structural points
    that materialize the same target-PDK blocks are physical aliases and should
    share placement/probe/signoff work.  Candidate provenance remains available
    separately on every ledger row.
    """

    payload = {
        "pdk": str(pdk),
        "bbox_nm": [int(value) for value in bbox_nm],
        "blocks": [
            [str(name), [int(value) for value in block_bbox]]
            for name, block_bbox in sorted(blocks)
        ],
        "terminals": [
            [
                str(name),
                [
                    [str(role), [int(point[0]), int(point[1])]]
                    for role, point in sorted(device_terminals)
                ],
            ]
            for name, device_terminals in sorted(terminals)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def deduplicate_physical_intent_candidates(
    candidates: Iterable[object],
    cheap_metrics: Mapping[str, Mapping[str, object]],
    *,
    rank: Callable[[object], tuple[object, ...]],
) -> tuple[object, ...]:
    """Keep one deterministic representative per realized placement.

    Candidate digests identify intent provenance, not physical work.  Once the
    cheap target-PDK evaluator has produced a physical signature, downstream
    probe/signoff work must be shared by aliases.  Rows without measurements
    are excluded; legacy rows without a signature remain distinct and cannot
    be silently coalesced.
    """

    representatives: dict[str, object] = {}
    for candidate in candidates:
        digest = str(getattr(candidate, "digest", ""))
        metrics = cheap_metrics.get(digest)
        if metrics is None:
            continue
        signature = metrics.get("physical_signature_sha256")
        key = (
            str(signature)
            if isinstance(signature, str) and signature
            else f"candidate:{digest}"
        )
        prior = representatives.get(key)
        if prior is None or rank(candidate) < rank(prior):
            representatives[key] = candidate
    return tuple(sorted(representatives.values(), key=rank))


def _hpwl_um(cell: object, placed: Mapping[str, object]) -> float:
    points: dict[str, list[tuple[int, int]]] = {}
    for device in getattr(cell, "devices", ()) or ():
        name = str(getattr(device, "instance_name", ""))
        physical = placed.get(name)
        if physical is None:
            continue
        bbox = tuple(int(value) for value in getattr(physical, "bbox"))
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        terminals = getattr(physical, "terminals", {}) or {}
        for role, net in (getattr(device, "pins", {}) or {}).items():
            raw = terminals.get(str(role), center)
            point = (int(raw[0]), int(raw[1]))
            points.setdefault(str(net), []).append(point)
    total_nm = 0
    for coordinates in points.values():
        if len(coordinates) < 2:
            continue
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        total_nm += max(xs) - min(xs) + max(ys) - min(ys)
    return total_nm / 1000.0


def _ring_perimeter_um(
    candidate: CompactIntentCandidate,
    placed: Mapping[str, object],
) -> float:
    perimeter_nm = 0
    for ring in candidate.guard_ring_groups:
        missing = sorted(set(ring.members) - set(placed))
        if missing:
            raise PhysicalIntentInfeasible(
                f"guard-ring members were not placed: {missing}"
            )
        boxes = [
            tuple(int(v) for v in getattr(placed[name], "bbox"))
            for name in ring.members
        ]
        left = min(box[0] for box in boxes)
        bottom = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        top = max(box[3] for box in boxes)
        perimeter_nm += 2 * ((right - left) + (top - bottom))
    return perimeter_nm / 1000.0


def _ring_span_waste_um2(
    candidate: CompactIntentCandidate,
    placed: Mapping[str, object],
) -> float:
    """Measure empty span enclosed by multi-member guard-ring obligations.

    Compatibility alone permits one ring to enclose remote devices.  Such a
    frame is electrically legal but cuts routing channels and creates narrow
    access throats.  The metric is purely geometric and circuit-neutral: for
    each declared ring it subtracts its member footprint from their enclosing
    box and sums the non-negative remainder.
    """

    waste_nm2 = 0
    for ring in candidate.guard_ring_groups:
        missing = sorted(set(ring.members) - set(placed))
        if missing:
            raise PhysicalIntentInfeasible(
                f"guard-ring members were not placed: {missing}"
            )
        boxes = [
            tuple(int(value) for value in getattr(placed[name], "bbox"))
            for name in ring.members
        ]
        left = min(box[0] for box in boxes)
        bottom = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        top = max(box[3] for box in boxes)
        enclosed_nm2 = max(0, right - left) * max(0, top - bottom)
        member_nm2 = sum(_bbox_area_nm2(box) for box in boxes)
        waste_nm2 += max(0, enclosed_nm2 - member_nm2)
    return waste_nm2 / 1_000_000.0


def _ring_max_perimeter_um(
    candidate: CompactIntentCandidate,
    placed: Mapping[str, object],
) -> float:
    """Return the longest single guard-ring frame perimeter.

    The maximum, unlike total perimeter, bounds the event-frontier workload on
    one owned ring atom during exact via-access enumeration.
    """

    maximum_nm = 0
    for ring in candidate.guard_ring_groups:
        missing = sorted(set(ring.members) - set(placed))
        if missing:
            raise PhysicalIntentInfeasible(
                f"guard-ring members were not placed: {missing}"
            )
        boxes = [
            tuple(int(value) for value in getattr(placed[name], "bbox"))
            for name in ring.members
        ]
        left = min(box[0] for box in boxes)
        bottom = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        top = max(box[3] for box in boxes)
        maximum_nm = max(maximum_nm, 2 * ((right - left) + (top - bottom)))
    return maximum_nm / 1000.0


def physical_intent_ring_metrics(
    candidate: CompactIntentCandidate,
    placed: Mapping[str, object],
) -> dict[str, float]:
    """Return the shared ring geometry used by every compact cohort policy."""

    return {
        "ring_perimeter_um": _ring_perimeter_um(candidate, placed),
        "ring_max_perimeter_um": _ring_max_perimeter_um(candidate, placed),
        "ring_span_waste_um2": _ring_span_waste_um2(candidate, placed),
    }


def _functional_group_bboxes(
    cell: object,
    placed: Mapping[str, object],
) -> tuple[PhysicalIntentFunctionalGroupBBox, ...]:
    """Project authored functional-group membership onto candidate geometry."""

    from clay.layout.functional_roles import functional_group_records_from_cell

    result: list[PhysicalIntentFunctionalGroupBBox] = []
    for group in functional_group_records_from_cell(cell):
        missing = tuple(member for member in group.members if member not in placed)
        if missing:
            raise PhysicalIntentInfeasible(
                f"functional group {group.name!r} has unplaced members: {missing}"
            )
        boxes: list[tuple[int, int, int, int]] = []
        for member in group.members:
            raw = getattr(placed[member], "bbox", None)
            if type(raw) is not tuple or len(raw) != 4 or any(
                type(value) is not int for value in raw
            ):
                raise PhysicalIntentInfeasible(
                    f"functional group {group.name!r} member {member!r} has invalid bbox"
                )
            boxes.append(raw)
        try:
            projected = PhysicalIntentFunctionalGroupBBox(
                group_id=group.name,
                bbox_nm=(
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ),
            )
        except ValueError as exc:
            raise PhysicalIntentInfeasible(
                f"functional group {group.name!r} has invalid bbox: {exc}"
            ) from exc
        result.append(projected)
    return tuple(sorted(result, key=lambda item: item.group_id))


def _candidate_device_bboxes(
    cell: object,
    placed: Mapping[str, object],
    *,
    certificate: object | None = None,
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Bind the exact cell inventory and optional signed geometry to one view."""

    expected = tuple(
        str(device.instance_name) for device in (getattr(cell, "devices", ()) or ())
    )
    if len(expected) != len(set(expected)):
        raise PhysicalIntentInfeasible("source cell has a duplicate device inventory")
    actual = tuple(placed)
    if (
        any(type(name) is not str or not name for name in actual)
        or len(actual) != len(set(actual))
        or set(actual) != set(expected)
    ):
        missing = tuple(sorted(set(expected) - set(actual)))
        foreign = tuple(sorted(set(actual) - set(expected)))
        raise PhysicalIntentInfeasible(
            "compact candidate did not place the exact device inventory: "
            f"missing={missing}, foreign={foreign}"
        )

    blocks: list[tuple[str, tuple[int, int, int, int]]] = []
    for name in sorted(actual):
        raw_bbox = getattr(placed[name], "bbox", None)
        if type(raw_bbox) is not tuple or len(raw_bbox) != 4 or any(
            type(value) is not int or abs(value) > _MAX_COORDINATE_NM
            for value in (raw_bbox if type(raw_bbox) is tuple else ())
        ):
            raise PhysicalIntentInfeasible(
                f"compact candidate device {name!r} has an invalid bbox"
            )
        bbox = tuple(raw_bbox)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise PhysicalIntentInfeasible(
                f"compact candidate device {name!r} has an invalid bbox"
            )
        blocks.append((name, bbox))
    canonical = tuple(blocks)

    if certificate is not None:
        raw_rows = getattr(certificate, "materialized_block_bboxes", None)
        if type(raw_rows) is not tuple:
            raise PhysicalIntentInfeasible(
                "compact candidate placement certificate has no exact device bbox map"
            )
        signed: list[tuple[str, tuple[int, int, int, int]]] = []
        for row in raw_rows:
            if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str:
                raise PhysicalIntentInfeasible(
                    "compact candidate placement certificate has an invalid device bbox map"
                )
            raw_bbox = row[1]
            if type(raw_bbox) is not tuple or len(raw_bbox) != 4 or any(
                type(value) is not int or abs(value) > _MAX_COORDINATE_NM
                for value in (raw_bbox if type(raw_bbox) is tuple else ())
            ):
                raise PhysicalIntentInfeasible(
                    "compact candidate placement certificate has an invalid device bbox map"
                )
            signed.append((row[0], tuple(raw_bbox)))
        signed_rows = tuple(sorted(signed))
        if len({name for name, _bbox in signed_rows}) != len(signed_rows):
            raise PhysicalIntentInfeasible(
                "compact candidate placement certificate repeats a device bbox"
            )
        if signed_rows != canonical:
            raise PhysicalIntentInfeasible(
                "compact candidate view disagrees with its placement certificate"
            )
    return canonical


def _candidate_route_device_bboxes(
    cell: object,
    placement_um: Mapping[str, tuple[float, float]],
    geometry: Mapping[str, object],
    *,
    certificate_bbox: tuple[int, int, int, int],
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Project exact routed stamps onto the canonical source-device inventory."""

    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm

    expected = {
        str(device.instance_name) for device in (getattr(cell, "devices", ()) or ())
    }
    if len(expected) != len(tuple(getattr(cell, "devices", ()) or ())):
        raise PhysicalIntentInfeasible("source cell has a duplicate device inventory")
    if (
        type(certificate_bbox) is not tuple
        or len(certificate_bbox) != 4
        or any(type(value) is not int for value in certificate_bbox)
    ):
        raise PhysicalIntentInfeasible("candidate certificate bbox is invalid")
    stamp_boxes = _placed_stamp_bboxes_nm(dict(geometry), dict(placement_um))
    projected: dict[str, tuple[int, int, int, int]] = {}
    for stamp_name, stamp in sorted(geometry.items()):
        bbox = stamp_boxes.get(stamp_name)
        if bbox is None:
            continue
        raw_members = getattr(stamp, "composite_members", ()) or ()
        if type(raw_members) is not tuple:
            raise PhysicalIntentInfeasible(
                f"route stamp {stamp_name!r} has non-canonical member ownership"
            )
        owners = raw_members or ((stamp_name,) if stamp_name in expected else ())
        foreign = tuple(sorted(set(owners) - expected))
        if foreign:
            raise PhysicalIntentInfeasible(
                f"route stamp {stamp_name!r} owns foreign members {foreign!r}"
            )
        for owner in owners:
            if owner in projected:
                raise PhysicalIntentInfeasible(
                    f"route member {owner!r} has multiple emitted stamp owners"
                )
            projected[owner] = tuple(int(value) for value in bbox)
    missing = tuple(sorted(expected - set(projected)))
    if missing:
        raise PhysicalIntentInfeasible(
            f"route geometry lacks exact source members {missing!r}"
        )
    cx0, cy0, cx1, cy1 = certificate_bbox
    outside = tuple(
        sorted(
            name
            for name, (x0, y0, x1, y1) in projected.items()
            if x0 < cx0 or y0 < cy0 or x1 > cx1 or y1 > cy1
        )
    )
    if outside:
        raise PhysicalIntentInfeasible(
            f"route geometry lies outside its placement certificate: {outside!r}"
        )
    return tuple(sorted(projected.items()))


def _candidate_placement_state_sha256(
    candidate: CompactIntentCandidate,
    certificate: object,
) -> str:
    if candidate.materialization_mode != "derived":
        return ""
    from clay.layout.place.final_placement_state import final_placement_state_digest

    return final_placement_state_digest(certificate)


def evaluate_compact_intent_candidate(
    *,
    source_path: Path | str,
    output_path: Path | str,
    top_cell: str,
    pdk: str,
    source_cir: Path | str,
    candidate: CompactIntentCandidate,
    seed: int,
) -> PhysicalIntentCheapResult:
    """Materialize and cheaply measure one declared compact candidate."""

    from clay.layout.flow_placement import (
        DeclaredCompactionError,
        materialize_declared_compact_override,
    )

    source_cir_path = Path(source_cir).resolve()
    if not source_cir_path.is_file():
        raise FileNotFoundError(
            f"compact physical evaluation source_cir is missing: {source_cir_path}"
        )
    artifact = materialize_compact_intent_clay(
        source_path=Path(source_path).resolve(),
        output_path=Path(output_path).resolve(),
        top_cell=top_cell,
        candidate=candidate,
        provenance={
            "seed": int(seed),
            "truth": "staging",
            "verified": False,
            "optimality": "unevaluated",
        },
        promotion=False,
    )
    if artifact.candidate_sha256 != candidate.digest:
        raise RuntimeError("materialized compact candidate digest mismatch")
    _parsed, cell, intent, policy, placement = _place_candidate_source(
        artifact.path,
        top_cell=top_cell,
        pdk=str(pdk),
        source_cir=source_cir_path,
    )
    try:
        compact = materialize_declared_compact_override(
            placement,
            cell,
            intent,
            str(pdk),
            effective_policy=policy,
        )
    except DeclaredCompactionError as exc:
        # The canonical compactor wraps unexpected implementation/environment
        # failures with ``raise ... from exc``.  Those remain unmeasurable and
        # must be retried; only its own cause-free structural rejections are a
        # stable property of this candidate.
        if exc.__cause__ is not None:
            raise
        raise PhysicalIntentInfeasible(str(exc)) from exc
    metric_placement = placement
    if compact is not None:
        from clay.layout.flow_placement import project_final_route_placement

        metric_placement = project_final_route_placement(
            placement,
            compact.override[0],
            compact.override[1],
        )
    placed = dict(metric_placement.devices)
    certificate = (
        placement.row_final_certificate
        if compact is None
        else compact.final_placement_certificate
    )
    if compact is None:
        blocks = _candidate_device_bboxes(
            cell,
            placed,
            certificate=(
                certificate if candidate.materialization_mode == "derived" else None
            ),
        )
    else:
        blocks = _candidate_route_device_bboxes(
            cell,
            compact.override[0],
            compact.override[1],
            certificate_bbox=tuple(compact.final_placement_certificate.bbox_nm),
        )
    block_map = dict(blocks)
    metric_devices = {
        name: SimpleNamespace(
            bbox=block_map[name],
            terminals=getattr(device, "terminals", {}) or {},
        )
        for name, device in placed.items()
    }
    if candidate.materialization_mode == "authoring" and certificate is None:
        # The untouched generic authoring path predates the 2-D compact solver
        # certificate.  Cheap ranking is explicitly non-binding, so measure its
        # complete, overlap-free device placement without pretending that a
        # solver certificate exists; exact route/DRC/LVS/PEX remains mandatory
        # before this anchor can win.
        if not blocks:
            raise PhysicalIntentInfeasible(
                "authoring candidate produced no placed device blocks"
            )
        bbox = (
            min(block[0] for _name, block in blocks),
            min(block[1] for _name, block in blocks),
            max(block[2] for _name, block in blocks),
            max(block[3] for _name, block in blocks),
        )
    else:
        _require_signable_placement_certificate(certificate)
        bbox = tuple(int(value) for value in certificate.bbox_nm)
        if not blocks:
            raise PhysicalIntentInfeasible(
                "compact candidate certificate has no materialized blocks"
            )
    if len(bbox) != 4 or _bbox_area_nm2(bbox) <= 0:
        raise PhysicalIntentInfeasible(
            "compact candidate produced an empty placement bbox"
        )
    occupied_nm2 = sum(
        _bbox_area_nm2(block_bbox) for block_bbox in set(block_map.values())
    )
    macro_consumption = None
    if candidate.materialization_mode == "derived":
        from clay.layout.compact_intent_domain import physical_macro_graph_for_cell
        from clay.layout.macro_consumption import certify_macro_consumption

        macro_consumption = certify_macro_consumption(
            physical_macro_graph_for_cell(cell),
            candidate,
            intent,
            placed,
            placed,
        )
        if macro_consumption.status != "pass":
            raise PhysicalIntentInfeasible(
                f"macro consumption {macro_consumption.status}: "
                f"{macro_consumption.reason}"
            )
    signature = _physical_signature_sha256(
        pdk=str(pdk),
        bbox_nm=bbox,
        blocks=blocks,
        terminals=tuple(
            (
                str(name),
                tuple(
                    sorted(
                        (
                            str(role),
                            (int(point[0]), int(point[1])),
                        )
                        for role, point in (
                            getattr(device, "terminals", {}) or {}
                        ).items()
                    )
                ),
            )
            for name, device in sorted(placed.items())
        ),
    )
    area_nm2 = _bbox_area_nm2(bbox)
    placement_state_sha256 = _candidate_placement_state_sha256(
        candidate,
        certificate,
    )
    ring_owner = candidate
    if candidate.materialization_mode == "authoring":
        ring_owner = SimpleNamespace(
            guard_ring_groups=tuple(getattr(intent, "guard_rings", ()) or ())
        )
    ring_metrics = physical_intent_ring_metrics(ring_owner, metric_devices)
    return PhysicalIntentCheapResult(
        candidate_sha256=candidate.digest,
        candidate_source=artifact.path,
        bbox_nm=bbox,
        bbox_area_um2=area_nm2 / 1_000_000.0,
        whitespace_um2=max(0, area_nm2 - occupied_nm2) / 1_000_000.0,
        wire_length_um=_hpwl_um(cell, metric_devices),
        via_count=sum(
            len(getattr(device, "terminals", {}) or {}) for device in placed.values()
        ),
        ring_perimeter_um=ring_metrics["ring_perimeter_um"],
        ring_max_perimeter_um=ring_metrics["ring_max_perimeter_um"],
        ring_span_waste_um2=ring_metrics["ring_span_waste_um2"],
        physical_signature_sha256=signature,
        macro_consumption_sha256=(
            "" if macro_consumption is None else macro_consumption.sha256
        ),
        placement_state_sha256=placement_state_sha256,
        functional_group_bboxes_nm=_functional_group_bboxes(cell, metric_devices),
    )


def evaluate_compact_intent_probe(
    *,
    candidate_source: Path | str,
    candidate_sha256: str,
    top_cell: str,
    pdk: str,
    budget: PhysicalIntentProbeBudget,
) -> PhysicalIntentProbeResult:
    """Run one bounded route observation on canonical signoff geometry.

    This deliberately keeps ``RouteExecutionScope.SIGNOFF`` (the default): the
    diagnostic scope owns different outside-edge rail geometry and therefore
    cannot rank the geometry that formal flow will route.  No GDS is emitted and
    a complete observation remains non-binding until DRC/full-extraction LVS/PEX.
    """

    from clay.layout.flow_placement import route_with_declared_compaction
    from clay.layout.route.engine_registry import make_route_engine
    from clay.layout.route.route_cell import RoutePlanningStop

    path = Path(candidate_source).resolve()
    _parsed, cell, intent, policy, placement = _place_candidate_source(
        path,
        top_cell=top_cell,
        pdk=str(pdk),
    )
    engine_name = str(getattr(policy, "route_engine", "") or "").strip().lower()
    if not engine_name:
        raise ValueError(
            "physical intent probe requires a declared deterministic route engine"
        )
    params = dict(getattr(policy, "route_engine_params", ()) or ())
    params.update(
        {
            "time_budget_sec": budget.time_budget_sec,
            "max_iters": budget.max_iters,
            "max_expansions": budget.max_expansions,
            "expansion_cap": budget.expansion_cap,
        }
    )
    engine = make_route_engine(engine_name, **params)
    if not bool(getattr(engine, "supports_deterministic_negotiation", False)):
        raise ValueError(
            "physical intent probe requires deterministic negotiated-congestion "
            f"evidence; engine={engine_name!r} does not provide it"
        )
    try:
        route_with_declared_compaction(
            placement,
            cell,
            intent,
            str(pdk),
            effective_policy=policy,
            max_iters=budget.max_iters,
            engine=engine,
        )
    except RoutePlanningStop as stopped:
        evidence = stopped.evidence
        raw_reason = getattr(evidence, "reason", "")
        reason = str(getattr(raw_reason, "value", raw_reason) or stopped.state)
        details = tuple(
            sorted(
                (
                    (str(key), value)
                    for key, value in tuple(getattr(evidence, "details", ()) or ())
                ),
                key=lambda item: item[0],
            )
        )
        return PhysicalIntentProbeResult(
            candidate_sha256=str(candidate_sha256),
            status="stopped",
            phase=str(stopped.phase),
            state=str(stopped.state),
            reason=reason,
            net=str(getattr(evidence, "net", "") or ""),
            evidence_sha256=str(stopped.evidence_hash),
            details=details,
        )
    return PhysicalIntentProbeResult(
        candidate_sha256=str(candidate_sha256),
        status="complete",
        phase="ROUTE_COMPLETE",
        state="NON_SIGNOFF_OBSERVATION",
        reason="NONE",
        net="",
        evidence_sha256=hashlib.sha256(
            f"{candidate_sha256}:{pdk}:complete".encode("utf-8")
        ).hexdigest(),
        details=(),
    )


__all__ = [
    "PhysicalIntentInfeasible",
    "PhysicalIntentFunctionalGroupBBox",
    "PhysicalIntentCheapResult",
    "PhysicalIntentProbeBudget",
    "PhysicalIntentProbeResult",
    "deduplicate_physical_intent_candidates",
    "evaluate_compact_intent_candidate",
    "evaluate_compact_intent_probe",
    "load_physical_intent_cache",
    "load_physical_intent_probe_cache",
    "physical_intent_probe_budget",
    "physical_intent_probe_rank",
    "physical_intent_ring_metrics",
]
