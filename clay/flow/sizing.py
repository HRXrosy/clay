"""Source-declared constrained sizing problem for the canonical Clay flow."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from clay.flow.spec import CircuitSpec, KpiSpec, OptimizationObjectiveSpec


_LARGE_VIOLATION = 1e6


def objective_metric_value(
    objective: OptimizationObjectiveSpec,
    metrics: Mapping[str, Any],
    *,
    post: bool = False,
) -> float:
    """Measure the source-declared objective in its declared metric units.

    The historical OTA objective is preserved. ``metric_<name>`` selects one
    measured KPI directly, for example ``metric_worst_delay_ns``. The caller
    applies minimize/maximize direction; missing or non-finite data is never
    an objective value.
    """
    suffix = "_post" if post else ""
    if objective.name == "fom_gbw_cl_over_idd":
        value = (float(metrics[f"gbw_hz{suffix}"]) * float(objective.cl_f)
                 / float(metrics[f"idd_a{suffix}"]))
        if value <= 0:
            raise ValueError("invalid OTA FOM")
    elif re.fullmatch(r"metric_[A-Za-z][A-Za-z0-9_]*", objective.name):
        key = objective.name[len("metric_"):]
        if key.endswith("_post"):
            raise ValueError("objective metric names must use the unsuffixed KPI name")
        value = float(metrics[key + suffix])
    else:
        raise ValueError(f"unsupported optimization objective {objective.name!r}")
    if not math.isfinite(value):
        raise ValueError("objective metric is non-finite")
    return value


def _rows(value: Any, dim: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=np.float32)
    return out.reshape(-1, dim)


def denormalize_sizing(spec: CircuitSpec, point: Any) -> dict[str, float]:
    intent = spec.optimization
    if intent is None:
        raise ValueError("denormalize_sizing requires optimization intent")
    vector = np.asarray(point, dtype=float).reshape(-1)
    if vector.shape != (len(intent.variables),):
        raise ValueError(f"expected {len(intent.variables)} normalized coordinates")
    domain_low, domain_high = intent.normalized_domain
    if np.any(vector < domain_low - 1e-7) or np.any(vector > domain_high + 1e-7):
        raise ValueError("normalized sizing lies outside the declared domain")
    span = domain_high - domain_low
    return {
        variable.name: float(
            variable.low
            + (vector[index] - domain_low) / span * (variable.high - variable.low)
        )
        for index, variable in enumerate(intent.variables)
    }


def normalize_sizing(spec: CircuitSpec, sizing: Mapping[str, float]) -> np.ndarray:
    """Map physical sizing DATA onto the intent-declared tSS-BO domain."""
    intent = spec.optimization
    if intent is None:
        raise ValueError("normalize_sizing requires optimization intent")
    domain_low, domain_high = intent.normalized_domain
    span = domain_high - domain_low
    point: list[float] = []
    for variable in intent.variables:
        if variable.name not in sizing:
            raise ValueError(f"missing sizing value {variable.name!r}")
        value = float(sizing[variable.name])
        if value < variable.low and math.isclose(value, variable.low, rel_tol=1e-12, abs_tol=1e-24):
            value = variable.low
        elif value > variable.high and math.isclose(value, variable.high, rel_tol=1e-12, abs_tol=1e-24):
            value = variable.high
        if not variable.low <= value <= variable.high:
            raise ValueError(
                f"sizing {variable.name}={value:g} lies outside "
                f"[{variable.low:g}, {variable.high:g}]"
            )
        point.append(
            domain_low
            + (value - variable.low) / (variable.high - variable.low) * span
        )
    return np.asarray(point, dtype=np.float32)


def _constraint(kpi: KpiSpec, raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _LARGE_VIOLATION
    if not math.isfinite(value) or kpi.bound is None:
        return _LARGE_VIOLATION
    op, reference = kpi.bound
    if op in {"ge", "le"}:
        scale = max(abs(float(reference)), 1e-30)
        return (
            (float(reference) - value) / scale
            if op == "ge"
            else (value - float(reference)) / scale
        )
    if op == "near":
        target, tolerance = reference
        return (abs(value - float(target)) - float(tolerance)) / max(
            abs(float(tolerance)), 1e-30
        )
    return _LARGE_VIOLATION


def _proxy_metric(base_key: str, pre: Mapping[str, float], limits: Mapping[str, float]):
    raw = pre.get(base_key)
    if raw is None:
        return None
    value = float(raw)
    if base_key == "dc_gain_db":
        value -= limits.get("gain_drop_db", 0.0)
    elif base_key == "gbw_hz":
        value *= 1.0 - limits.get("gbw_abs_rel", 0.0)
    elif base_key == "pm_deg":
        value -= limits.get("pm_drop_deg", 0.0)
    elif base_key == "slew_v_per_s":
        value *= 1.0 - limits.get("slew_drop_rel", 0.0)
    elif base_key == "settling_s":
        value *= 1.0 + limits.get("settling_increase_rel", 0.0)
    return value


def _degradation_values(
    pre: Mapping[str, float], post: Mapping[str, float], limits: Mapping[str, float]
) -> dict[str, float]:
    if not limits:
        return {}
    def rel(before: str, after: str, *, reverse: bool = False) -> float:
        p, q = float(pre[before]), float(post[after])
        return ((q - p) if reverse else (p - q)) / max(abs(p), 1e-30)

    measurements = {
        "gain_drop_db": lambda: float(pre["dc_gain_db"]) - float(post["dc_gain_db_post"]),
        "gbw_abs_rel": lambda: abs(float(post["gbw_hz_post"]) - float(pre["gbw_hz"]))
        / max(abs(float(pre["gbw_hz"])), 1e-30),
        "pm_drop_deg": lambda: float(pre["pm_deg"]) - float(post["pm_deg_post"]),
        "slew_drop_rel": lambda: rel("slew_v_per_s", "slew_v_per_s_post"),
        "settling_increase_rel": lambda: rel(
            "settling_s", "settling_s_post", reverse=True
        ),
    }
    return {name: measurements[name]() for name in limits}


class JointSizingProblem:
    """tSS-BO ``funct(X)`` backed by hierarchical pre/post evaluation."""

    def __init__(
        self,
        spec: CircuitSpec,
        evaluate: Callable[..., Mapping[str, Any]],
        *,
        journal_dir: Path,
    ) -> None:
        if spec.optimization is None:
            raise ValueError("JointSizingProblem requires optimization intent")
        self.spec = spec
        self.intent = spec.optimization
        self.evaluate = evaluate
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        # One problem instance is one deterministic evidence run.  Reusing an
        # out directory must not append current rows behind a stale optimizer
        # state and make the ledger appear to bind multiple code states.
        (self.journal_dir / "sizing_journal.jsonl").write_text(
            "", encoding="utf-8"
        )
        self.journal: list[dict[str, Any]] = []
        self.nominal_incumbent: dict[str, Any] | None = None
        self.verified_incumbent: dict[str, Any] | None = None
        self.outdim = 1 + len(spec.kpis) + len(self.intent.degradation_limits)

    def __call__(self, x: Any) -> np.ndarray:
        points = _rows(x, len(self.intent.variables))
        return np.asarray(
            [self.evaluate_point(point) for point in points], dtype=np.float32
        )

    def evaluate_point(
        self,
        point: Any,
        *,
        force_post: bool = False,
        post_reason: str | None = None,
    ) -> np.ndarray:
        """Evaluate one candidate under the deterministic hierarchical policy.

        The first point is the feasibility gate.  Thereafter POST is run only
        for periodic samples, a proxy-feasible improvement over the current
        verified incumbent, or an explicitly forced final check.
        """
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        if point.shape != (len(self.intent.variables),):
            raise ValueError("sizing point has the wrong dimension")
        index = len(self.journal)
        sizing = denormalize_sizing(self.spec, point)
        if force_post:
            reason = post_reason or "forced"
        elif index == 0:
            reason = "feasibility_gate"
        elif index % int(self.intent.post_every) == 0:
            reason = "periodic"
        else:
            reason = "pre_proxy"

        def run(run_post: bool) -> tuple[dict[str, Any], np.ndarray, float, dict[str, float], float]:
            try:
                evaluation = dict(
                    self.evaluate(
                        sizing,
                        evaluation_index=index,
                        run_post=run_post,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one BO sample is evidence
                from clay.flow.sizing_source import SizingSourceChanged
                from clay.flow.execution_budget import ExecutionStopped
                if isinstance(exc, (SizingSourceChanged, ExecutionStopped)):
                    raise
                evaluation = {
                    "truth": "unmeasurable",
                    "pre": {},
                    "post": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pre = dict(evaluation.get("pre") or {})
            post = dict(evaluation.get("post") or {})
            truth = str(evaluation.get("truth") or "pre_proxy")
            limits = dict(self.intent.degradation_limits)
            metric_values: dict[str, Any] = {}
            for kpi in self.spec.kpis:
                if truth == "post_pex":
                    metric_values[kpi.key] = post.get(kpi.key)
                else:
                    base_key = kpi.key[:-5] if kpi.key.endswith("_post") else kpi.key
                    metric_values[kpi.key] = _proxy_metric(base_key, pre, limits)
            constraints = [
                _constraint(kpi, metric_values[kpi.key]) for kpi in self.spec.kpis
            ]
            if truth == "post_pex":
                try:
                    degradation = _degradation_values(pre, post, limits)
                    degradation_constraints = [
                        (degradation[name] - limit) / max(abs(limit), 1e-30)
                        for name, limit in self.intent.degradation_limits
                    ]
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    degradation = {}
                    degradation_constraints = [_LARGE_VIOLATION] * len(limits)
            elif truth == "pre_proxy":
                degradation = {}
                degradation_constraints = [0.0] * len(limits)
            else:
                degradation = {}
                degradation_constraints = [_LARGE_VIOLATION] * len(limits)
            constraints.extend(degradation_constraints)
            metrics = post if truth == "post_pex" else pre
            try:
                fom = objective_metric_value(
                    self.intent.objective, metrics, post=truth == "post_pex"
                )
                objective = (
                    -fom if self.intent.objective.direction == "maximize" else fom
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                fom, objective = float("nan"), _LARGE_VIOLATION
                constraints = [_LARGE_VIOLATION] * len(constraints)
            try:
                proxy_fom = objective_metric_value(self.intent.objective, pre)
                proxy_objective = (
                    -proxy_fom
                    if self.intent.objective.direction == "maximize"
                    else proxy_fom
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                proxy_objective = _LARGE_VIOLATION
            return (
                evaluation,
                np.asarray([objective, *constraints], dtype=np.float32),
                fom,
                degradation,
                float(proxy_objective),
            )

        evaluation, row, fom, degradation, proxy_objective = run(
            reason != "pre_proxy"
        )
        truth = str(evaluation.get("truth") or "pre_proxy")
        if truth == "unmeasurable":
            # Fail closed, but do not collapse every simulator failure onto an
            # identical GP row.  A deterministic coordinate fingerprint gives
            # tSS-BO information about which invalid region it sampled while
            # every constraint remains strictly violated.
            fingerprint = sum(
                (coordinate + 1) * (float(value) + 0.5)
                for coordinate, value in enumerate(point)
            )
            row[0] = np.float32(1000.0 + fingerprint)
            for constraint_index in range(1, len(row)):
                coordinate = (constraint_index - 1) % len(point)
                row[constraint_index] = np.float32(
                    1.0
                    + abs(float(point[coordinate]))
                    + 0.001 * constraint_index * (fingerprint + 1.0)
                )
        if reason == "pre_proxy" and truth != "post_pex":
            proxy_feasible = bool(np.all(row[1:] <= 0.0))
            incumbent_proxy = (
                float(self.nominal_incumbent["proxy_objective"])
                if self.nominal_incumbent is not None
                else float("inf")
            )
            if proxy_feasible and proxy_objective < incumbent_proxy:
                reason = "incumbent"
                evaluation, row, fom, degradation, proxy_objective = run(True)
                truth = str(evaluation.get("truth") or "pre_proxy")

        pre = dict(evaluation.get("pre") or {})
        post = dict(evaluation.get("post") or {})
        nominal_post_verified = truth == "post_pex" and math.isfinite(fom) and bool(
            np.all(row[1:] <= 0.0)
        )
        record = {
            "evaluation_index": index,
            "normalized": [float(value) for value in point],
            "sizing": sizing,
            "truth": truth,
            "evaluation_state": (
                "measured" if truth in {"pre_proxy", "post_pex"} else truth
            ),
            "error": evaluation.get("error"),
            "pre": pre,
            "post": post or None,
            "fom": fom,
            "objective_name": self.intent.objective.name,
            "objective_direction": self.intent.objective.direction,
            "objective_value": fom,
            "proxy_objective": proxy_objective,
            "row": [float(value) for value in row],
            "degradation": degradation,
            # Nominal POST truth is enough to guide tSS-BO, but never enough
            # to promote the final incumbent.  ``verified`` becomes true only
            # after the source-declared full-PVT acceptance sweep below.
            "nominal_post_verified": nominal_post_verified,
            "verified": False,
            "post_reason": reason,
            "candidate_source": evaluation.get("candidate_source"),
            "candidate_pex": evaluation.get("candidate_pex"),
            "candidate_artifact_seal": evaluation.get("candidate_artifact_seal"),
            "candidate_design_clay": evaluation.get("candidate_design_clay"),
            "candidate_design_clay_sha256": evaluation.get("candidate_design_clay_sha256"),
            "authoring_design_clay": evaluation.get("authoring_design_clay"),
            "authoring_design_clay_sha256": evaluation.get("authoring_design_clay_sha256"),
        }
        self.journal.append(record)
        with (self.journal_dir / "sizing_journal.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if nominal_post_verified and (
            self.nominal_incumbent is None
            or float(row[0]) < float(self.nominal_incumbent["row"][0])
        ):
            self.nominal_incumbent = record
        return row

    def promote_acceptance(self) -> dict[str, Any] | None:
        """Promote the best candidate that passes full-PVT POST-PEX truth.

        Search-time POST samples deliberately use nominal PVT to keep the BO
        loop tractable.  Final promotion is a separate fail-closed contract:
        visit nominally feasible candidates in objective order and reuse each
        candidate's extracted netlist for the complete declared corner/temp
        sweep.  A faster nominal point can therefore never displace a slightly
        lower-FOM candidate that is the one actually robust across PVT.
        """
        validate = getattr(self.evaluate, "evaluate_acceptance", None)
        if not callable(validate):
            raise RuntimeError(
                "joint evaluator exposes no full-PVT POST-PEX acceptance driver"
            )
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()
        for record in sorted(
            (
                item
                for item in self.journal
                if item.get("nominal_post_verified") is True
            ),
            key=lambda item: float(item["row"][0]),
        ):
            identity = tuple(float(value) for value in record["normalized"])
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(record)

        self.verified_incumbent = None
        limits = dict(self.intent.degradation_limits)
        for record in candidates:
            try:
                measured = dict(validate(record))
                pvt = measured.get("__pvt__")
                if not isinstance(pvt, Mapping):
                    raise RuntimeError("acceptance driver returned no __pvt__ evidence")
                nominal_post = pvt.get("nominal_post")
                if not isinstance(nominal_post, Mapping):
                    raise RuntimeError(
                        "acceptance driver returned no nominal POST binding"
                    )
                aggregated = {
                    str(key): value
                    for key, value in measured.items()
                    if not str(key).startswith("__")
                }
                kpi_constraints = {
                    kpi.key: _constraint(kpi, aggregated.get(kpi.key))
                    for kpi in self.spec.kpis
                }
                degradation = _degradation_values(
                    dict(record.get("pre") or {}), dict(nominal_post), limits
                )
                degradation_constraints = {
                    name: (degradation[name] - limit) / max(abs(limit), 1e-30)
                    for name, limit in self.intent.degradation_limits
                }
                acceptance_objective = objective_metric_value(
                    self.intent.objective, aggregated, post=True
                )
                passed = all(
                    value <= 0.0
                    for value in (
                        *kpi_constraints.values(),
                        *degradation_constraints.values(),
                    )
                )
                record["acceptance"] = {
                    "truth": "full_pvt_post_pex",
                    "post": aggregated,
                    "pvt": dict(pvt),
                    "degradation": degradation,
                    "kpi_constraints": kpi_constraints,
                    "degradation_constraints": degradation_constraints,
                    "objective_name": self.intent.objective.name,
                    "objective_value": acceptance_objective,
                    "constraints_passed": passed,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 - candidate-level evidence
                from clay.flow.sizing_source import SizingSourceChanged
                from clay.flow.execution_budget import ExecutionStopped
                if isinstance(exc, (SizingSourceChanged, ExecutionStopped)):
                    raise
                passed = False
                record["acceptance"] = {
                    "truth": "unmeasurable",
                    "constraints_passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            record["verified"] = passed
            self._rewrite_journal()
            if passed:
                self.verified_incumbent = record
                return record
        return None

    def _rewrite_journal(self) -> None:
        path = self.journal_dir / "sizing_journal.jsonl"
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in self.journal),
            encoding="utf-8",
        )


__all__ = ["JointSizingProblem", "denormalize_sizing", "normalize_sizing", "objective_metric_value"]
