"""Materialize one verified sizing incumbent as a standalone Clay source.

The optimizer is allowed to choose values; it is not allowed to leave those
values as an invisible mutation of a parsed ``CellDef``.  This module preserves
the authoring source and deterministically rewrites only source-declared binding
targets in a derivative file.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from clay.flow.spec import OptimizationSpec
from clay.layout.compact_intent import (
    CompactIntentCandidate,
    CompactIntentTruth,
    compact_intent_candidate_from_data,
    compact_intent_evidence_from_provenance,
    load_compact_intent_provenance_json,
    replay_compact_intent_candidate,
)
from clay.parser import parse_clay_source


class UnresolvedCompactIntentError(ValueError):
    """A declared compact-layout search has no materialized candidate."""


class CompactIntentMaterializationError(ValueError):
    """A named failure to lower/replay graph-bound compact intent."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _validate_compact_candidate_graph(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
    allow_materialized_passive_arrays: bool = False,
) -> None:
    from clay.layout.compact_intent_domain import (
        GraphPartitionError,
        passive_bank_partition_intents_for_candidate,
        physical_macro_graph_for_candidate,
        validate_graph_partition_choice,
    )

    if candidate.physical_macro_graph_sha256 is None:
        return
    cell = _select_cell(parse_clay_source(source), top_cell)
    try:
        passive_bank_partition_intents_for_candidate(cell, candidate)
        graph = physical_macro_graph_for_candidate(
            cell,
            candidate,
            allow_materialized_passive_arrays=allow_materialized_passive_arrays,
        )
    except GraphPartitionError as exc:
        raise CompactIntentMaterializationError(exc.code, exc.detail) from exc
    if graph.sha256 != candidate.physical_macro_graph_sha256:
        raise CompactIntentMaterializationError(
            "STALE_PHYSICAL_MACRO_GRAPH",
            "candidate graph digest does not match effective source intent",
        )
    try:
        for choice in candidate.well_island_partitions:
            validate_graph_partition_choice(
                graph,
                cell,
                choice,
                expected_kind="well_island",
            )
    except GraphPartitionError as exc:
        raise CompactIntentMaterializationError(exc.code, exc.detail) from exc


def compact_intent_search_requested(cell: object) -> bool:
    """Return whether a cell declares an enabled compact-layout search.

    The decision is made only from the existing numeric ``sizing {}`` DATA.
    It deliberately knows neither circuit nor PDK identities.  A positive
    budget is the activation obligation. Axis completeness is validated by the
    typed domain builder; this function does not guess axis vocabulary from
    string prefixes.
    """

    data = dict(getattr(cell, "sizing_kwargs", {}) or {})
    raw_budget = data.get("opt_layout_budget")
    return (
        type(raw_budget) in (int, float)
        and math.isfinite(float(raw_budget))
        and float(raw_budget) > 0.0
    )


def require_resolved_compact_intent(
    cell: object,
    source: str,
) -> CompactIntentCandidate:
    """Return the selected candidate or fail closed on bypassed search intent.

    Candidate staging must be routable/signable before a search can decide its
    winner, so this seam requires canonical materialization but does not require
    promotion truth.  Promotion and incumbent truth remain ledger obligations.
    """

    try:
        provenance = _compact_intent_provenance(source)
        candidate = declared_compact_candidate(source)
    except ValueError as exc:
        raise UnresolvedCompactIntentError(str(exc)) from exc
    if candidate is None:
        name = str(getattr(cell, "local_name", "") or "<anonymous>")
        raise UnresolvedCompactIntentError(
            f"cell {name} declares opt_layout search intent but has no "
            "materialized compact_intent candidate"
        )

    name = str(getattr(cell, "local_name", "") or "<anonymous>")
    assert provenance is not None
    expected_schema = "clay.compact-intent/v2"
    if provenance.get("schema") != expected_schema:
        raise UnresolvedCompactIntentError(
            f"compact_intent provenance requires schema {expected_schema}"
        )
    if provenance.get("top_cell") != name:
        raise UnresolvedCompactIntentError(
            f"compact_intent provenance top_cell does not bind {name}"
        )
    if candidate.materialization_mode == "authoring":
        if provenance.get("optimality") != "unevaluated":
            raise UnresolvedCompactIntentError(
                "compact_intent authoring optimality must be unevaluated"
            )
        try:
            measured_authoring_sha256 = _replayed_authoring_source_sha256(source)
        except ValueError as exc:
            raise UnresolvedCompactIntentError(str(exc)) from exc
        if provenance.get("authoring_sha256") != measured_authoring_sha256:
            raise UnresolvedCompactIntentError(
                "compact_intent provenance does not bind the authoring source"
            )
    try:
        compact_intent_evidence_from_provenance(provenance)
    except ValueError as exc:
        raise UnresolvedCompactIntentError(str(exc)) from exc
    measured_body = _digest(_compact_intent_body(source).encode("utf-8"))
    if provenance.get("materialized_body_sha256") != measured_body:
        raise UnresolvedCompactIntentError(
            "compact_intent provenance does not bind the materialized source"
        )
    try:
        from clay.layout.compact_intent_domain import (
            validate_compact_intent_candidate,
        )

        validate_compact_intent_candidate(
            cell,
            candidate,
            allow_materialized_passive_arrays=True,
        )
    except ValueError as exc:
        raise UnresolvedCompactIntentError(
            f"compact_intent candidate violates its declared domain: {exc}"
        ) from exc
    try:
        replayed = _apply_compact_candidate_materialization(
            source,
            top_cell=name,
            candidate=candidate,
            allow_materialized_passive_arrays=True,
        )
        if _compact_intent_body(replayed) != _compact_intent_body(source):
            raise ValueError(
                "source differs from canonical materialization replay"
            )
        _validate_compact_candidate_round_trip(
            source,
            top_cell=name,
            candidate=candidate,
        )
    except ValueError as exc:
        raise UnresolvedCompactIntentError(
            f"compact_intent candidate is not materialized: {exc}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class OptimizedClayArtifact:
    path: Path
    sha256: str
    authoring_sha256: str
    sizing: Mapping[str, float]


@dataclass(frozen=True)
class CompactIntentClayArtifact:
    path: Path
    source_sha256: str
    authoring_sha256: str
    candidate_sha256: str


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_source_text(source: str) -> str:
    """Use one byte-independent newline form at every provenance boundary."""

    return source.replace("\r\n", "\n").replace("\r", "\n")


_COMPACT_INTENT_ROW = re.compile(
    r"(?m)^// compact_intent (?P<payload>\{[^\r\n]*\})\s*$"
)


def _replayed_authoring_source_sha256(source: str) -> str:
    """Rebuild the exact canonical source wrapped by an authoring marker."""

    normalized = _canonical_source_text(source)
    rows = list(_COMPACT_INTENT_ROW.finditer(normalized))
    if len(rows) != 1:
        raise ValueError("authoring compact intent requires exactly one marker")
    marker = rows[0]
    if normalized[marker.end() :].strip():
        raise ValueError("authoring compact intent marker must be the final record")
    prefix = normalized[: marker.start()]
    if not prefix.endswith("\n\n"):
        raise ValueError(
            "authoring compact intent marker lacks its canonical separator"
        )
    return _digest(prefix[:-1].encode("utf-8"))


def _compact_intent_provenance(source: str) -> dict[str, Any] | None:
    rows = list(_COMPACT_INTENT_ROW.finditer(source))
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("optimized Clay requires at most one compact_intent row")
    return load_compact_intent_provenance_json(rows[0].group("payload"))


def _compact_intent_body(source: str) -> str:
    """Canonical physical source body covered by the compact provenance row.

    Other evidence-only rows may be replaced by a later sizing promotion; they
    do not alter physical declarations and therefore are excluded from this
    structural binding.
    """

    body = _COMPACT_INTENT_ROW.sub("", _canonical_source_text(source))
    body = re.sub(
        r"(?m)^// (?:optimized_clay|sizing_candidate) \{[^\r\n]*\}\s*$",
        "",
        body,
    )
    return body.rstrip()


def declared_compact_candidate(source: str) -> CompactIntentCandidate | None:
    """Read the full structural DATA needed to survive a sizing derivative."""

    provenance = _compact_intent_provenance(source)
    if provenance is None:
        return None
    data = provenance.get("candidate")
    if not isinstance(data, Mapping):
        raise ValueError(
            "compact_intent provenance must carry canonical candidate DATA"
        )
    candidate = compact_intent_candidate_from_data(data)
    if provenance.get("candidate_sha256") != candidate.digest:
        raise ValueError("compact_intent candidate DATA digest mismatch")
    return candidate


def _refresh_compact_intent_provenance(
    source: str,
    *,
    top_cell: str,
    previous_body_sha256: str | None = None,
) -> str:
    """Rebind a preserved candidate marker after a canonical source rewrite."""

    source = _canonical_source_text(source)
    provenance = _compact_intent_provenance(source)
    if provenance is None:
        return source
    candidate = declared_compact_candidate(source)
    assert candidate is not None
    prior_was_authoritative = previous_body_sha256 is not None
    if previous_body_sha256 is None:
        declared_previous = provenance.get("materialized_body_sha256")
        if not isinstance(declared_previous, str):
            raise ValueError(
                "compact_intent provenance has no prior body binding"
            )
        previous_body_sha256 = declared_previous
    if provenance.get("materialized_body_sha256") != previous_body_sha256:
        raise ValueError(
            "compact_intent provenance is stale before sizing rewrite"
        )
    # An authoring anchor signs the declarations exactly as written.  Official
    # source rewrites (for example target-PDK binding) may change package/flow
    # bytes and therefore need a fresh body digest, but they must not reinterpret
    # ``natural`` as permission to delete an authored aspect or inject a pack
    # mode.  Only a derived candidate owns those structural rewrite axes.
    if candidate.materialization_mode == "derived":
        source = _apply_common_centroid_row_strategy(
            source,
            top_cell=top_cell,
            candidate=candidate,
        )
        source = _apply_compact_candidate_place_policy(
            source, top_cell=top_cell, candidate=candidate
        )
    evidence = compact_intent_evidence_from_provenance(provenance)
    rendered_body = _COMPACT_INTENT_ROW.sub("", source).rstrip()
    physical_body = _compact_intent_body(source)
    measured_body_sha256 = _digest(physical_body.encode("utf-8"))
    updated = dict(provenance)
    updated.update(
        schema="clay.compact-intent/v2",
        top_cell=str(top_cell),
        candidate=candidate.canonical_data(),
        candidate_sha256=candidate.digest,
        materialized_body_sha256=measured_body_sha256,
    )
    if candidate.materialization_mode == "authoring":
        # An authoring marker signs the complete marker-free source.  Official
        # rewrites such as target-PDK binding change those bytes, so retaining
        # the pre-rewrite authoring digest makes the otherwise valid anchor
        # fail its replay gate.  ``rendered_body + newline`` is exactly the
        # canonical prefix reconstructed by ``_replayed_authoring_source_sha256``.
        updated["authoring_sha256"] = _digest(
            (rendered_body + "\n").encode("utf-8")
        )
    if not prior_was_authoritative or measured_body_sha256 != previous_body_sha256:
        updated.update(truth="staging", verified=False)
        updated.pop("physical_pex_artifact", None)
        updated.pop("electrical_result_receipt", None)
    elif evidence.truth is not CompactIntentTruth.STAGING:
        updated.pop("verified", None)
        updated.update(evidence.canonical_data())
    rebound = rendered_body + "\n\n// compact_intent " + json.dumps(
        updated,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    _validate_compact_candidate_round_trip(
        rebound,
        top_cell=top_cell,
        candidate=candidate,
    )
    return rebound


def _select_cell(parsed: Any, top_cell: str) -> Any:
    matches = [
        cell
        for cell in parsed.cell_defs
        if getattr(cell, "local_name", getattr(cell, "name", None)) == top_cell
    ]
    if len(matches) != 1:
        raise ValueError(
            f"optimized Clay source requires exactly one cell {top_cell!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _block_end(text: str, open_brace: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    line_comment = False
    index = open_brace
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char in "\r\n":
                line_comment = False
            index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == "/" and nxt == "/":
            line_comment = True
            index += 2
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                break
        index += 1
    raise ValueError("unterminated Clay block while materializing optimized source")


def _cell_span(text: str, top_cell: str) -> tuple[int, int]:
    pattern = re.compile(rf"\bcell\s+{re.escape(top_cell)}\s*\(", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"authoring source requires exactly one cell {top_cell!r}; "
            f"found {len(matches)}"
        )
    open_brace = text.find("{", matches[0].end())
    if open_brace < 0:
        raise ValueError(f"cell {top_cell!r} has no body")
    return matches[0].start(), _block_end(text, open_brace)


def _top_level_named_block_span(
    text: str, *, keyword: str, name: str
) -> tuple[int, int, int]:
    pattern = re.compile(
        rf"\b{re.escape(keyword)}\s+{re.escape(name)}\s*\{{",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"authoring source requires exactly one {keyword} block for "
            f"{name!r}; found {len(matches)}"
        )
    open_brace = text.find("{", matches[0].start(), matches[0].end())
    return matches[0].start(), open_brace, _block_end(text, open_brace)


def _named_inner_block_span(
    text: str, *, outer: tuple[int, int], kind: str, name: str | None = None
) -> tuple[int, int]:
    start, end = outer
    if kind == "sizing":
        pattern = re.compile(r"\bsizing\s*\{")
    elif kind == "device":
        assert name is not None
        pattern = re.compile(
            rf"\b(?:device\s+{re.escape(name)}\s*:[^{{;]+|"
            rf"(?:res|cap)\s+{re.escape(name)})\s*\{{"
        )
    else:
        raise ValueError(f"unsupported Clay block kind {kind!r}")
    matches = list(pattern.finditer(text, start, end))
    if len(matches) != 1:
        label = name if name is not None else kind
        raise ValueError(
            f"authoring source requires exactly one {kind} block for {label!r}; "
            f"found {len(matches)}"
        )
    open_brace = text.find("{", matches[0].start(), matches[0].end())
    return matches[0].start(), _block_end(text, open_brace)


def _plain_number(value: float, scale: float) -> str:
    scaled = round(value / scale, 5)
    rounded = round(scaled)
    if math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-10):
        return str(int(rounded))
    return format(scaled, ".12g")


def _is_geometry_variable(variable: str) -> bool:
    """Recognize scalar and role-qualified MOS width/length DOFs."""
    upper = variable.upper()
    return upper in {"W", "L"} or upper.startswith(("W_", "L_"))


def _unit_scale(variable: str, value: float) -> float:
    upper = variable.upper()
    if _is_geometry_variable(variable):
        return 1e-6
    if upper.startswith("C"):
        return 1e-12
    if upper.startswith("I"):
        return 1e-6
    if upper.startswith("R") and abs(value) >= 1000.0:
        return 1000.0
    return 1.0


def canonicalize_sizing_value(variable: str, value: float) -> float:
    """Map one optimizer float to the deterministic physical source value.

    tSS-BO communicates through float32 normalized coordinates, so exact source
    values such as 80um return as 79.99999940395355um.  Clay geometry is integer
    nanometres; geometry variables therefore snap to the nearest integer nm.
    Non-geometric values retain five decimal places in their displayed source
    unit.  The joint evaluator and optimized-Clay materializer both call this
    function so schematic, pre-sim, layout, PEX, and the promoted source consume
    one value rather than adjacent float representations.
    """
    upper = variable.upper()
    if _is_geometry_variable(variable):
        return float(round(float(value) * 1e9) * 1e-9)
    scale = _unit_scale(variable, value)
    return float(round(value / scale, 5) * scale)


def canonicalize_variable_value(variable: Any, value: float, *, native: bool = False) -> float:
    """Quantize the optimizer coordinate once, before any simulation or geometry.

    A declared step is in the same SI units as the bounded variable. Native
    parameters never derive their units or resolution from a variable's name.
    """
    step = getattr(variable, "step", None)
    if step is not None:
        step = float(step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError(f"sizing variable {variable.name!r} has an invalid step")
        low = math.ceil(float(variable.low) / step - 1e-9)
        high = math.floor(float(variable.high) / step + 1e-9)
        if low > high:
            raise ValueError(f"sizing variable {variable.name!r} has no point on its step")
        index = max(low, min(high, int(round(float(value) / step))))
        return float(format(index * step, ".12g"))
    if native:
        return float(format(float(value), ".12g"))
    if any(
        str(b).startswith("device.") and str(b).upper().endswith((".W", ".L"))
        for b in (getattr(variable, "bindings", ()) or ())
    ):
        return float(round(float(value) * 1e9) * 1e-9)
    return canonicalize_sizing_value(variable.name, value)


def _format_clay_value(variable: str, value: float) -> str:
    upper = variable.upper()
    if _is_geometry_variable(variable):
        return _plain_number(value, 1e-6) + "u"
    if upper.startswith("C"):
        return _plain_number(value, 1e-12) + "p"
    if upper.startswith("I"):
        return _plain_number(value, 1e-6) + "u"
    if upper.startswith("R"):
        if abs(value) >= 1000.0:
            return _plain_number(value, 1000.0) + "k"
        return _plain_number(value, 1.0)
    return format(value, ".12g")


def _replace_assignment(
    text: str, *, span: tuple[int, int], field: str, replacement: str, label: str
) -> str:
    start, end = span
    body = text[start:end]
    pattern = re.compile(rf"(?m)(\b{re.escape(field)}\s*=\s*)([^;]+)(;)")
    matches = list(pattern.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            f"optimized Clay binding {label!r} requires exactly one assignment; "
            f"found {len(matches)}"
        )
    match = matches[0]
    rewritten = body[: match.start(2)] + replacement + body[match.end(2) :]
    return text[:start] + rewritten + text[end:]


def _bound_device_targets(optimization: OptimizationSpec) -> dict[str, list[tuple[str, str]]]:
    targets: dict[str, list[tuple[str, str]]] = {}
    for variable in optimization.variables:
        for binding in variable.bindings:
            parts = binding.split(".")
            if not parts or parts[0] != "device":
                continue
            if len(parts) != 3:
                raise ValueError(f"invalid device sizing binding {binding!r}")
            targets.setdefault(variable.name, []).append((parts[1], parts[2]))
    return targets


def _rewrite_bound_literals(
    source: str,
    *,
    top_cell: str,
    optimization: OptimizationSpec,
    sizing: Mapping[str, float],
) -> str:
    rewritten = source
    targets = _bound_device_targets(optimization)
    parsed = parse_clay_source(source)
    source_cell = _select_cell(parsed, top_cell)
    devices = {device.instance_name: device for device in source_cell.devices}
    from clay.flow.candidate_bindings import native_parameter_token, value_for_binding
    # Work one assignment at a time and re-locate blocks after each replacement;
    # formatted literals can change length, so stale source offsets are forbidden.
    for variable in optimization.variables:
        var_targets = targets.get(variable.name, ())
        is_geom_target = _is_geometry_variable(variable.name) or any(
            field.upper() in {"W", "L"} for _inst, field in var_targets
        )
        formatted = (
            _plain_number(sizing[variable.name], 1e-6) + "u"
            if is_geom_target
            else _format_clay_value(variable.name, sizing[variable.name])
        )
        if any(devices.get(instance) is not None and devices[instance].device_class == "native"
               for instance, _ in var_targets):
            formatted = format(sizing[variable.name], ".12g")
        for instance_name, field in var_targets:
            original = devices.get(instance_name)
            if original is None:
                raise ValueError(f"sizing binding names missing device {instance_name!r}")
            cell = _cell_span(rewritten, top_cell)
            device = _named_inner_block_span(
                rewritten, outer=cell, kind="device", name=instance_name
            )
            bound_value = value_for_binding(variable, f"device.{instance_name}.{field}", sizing[variable.name])
            replacement = (
                _plain_number(bound_value, 1e-6) + "u"
                if field.upper() in {"W", "L"}
                else _format_clay_value(variable.name, bound_value)
            )
            target_field = field
            if original.device_class == "native":
                target_field, token = native_parameter_token(
                    original, field, bound_value, pdk_id=parsed.pdk_id
                )
                replacement = '"' + token + '"'
            rewritten = _replace_assignment(
                rewritten,
                span=device,
                field=target_field,
                replacement=replacement,
                label=f"device.{instance_name}.{field}",
            )
        cell = _cell_span(rewritten, top_cell)
        sizing_block = _named_inner_block_span(
            rewritten, outer=cell, kind="sizing"
        )
        rewritten = _replace_assignment(
            rewritten,
            span=sizing_block,
            field=variable.name,
            replacement=formatted,
            label=f"sizing.{variable.name}",
        )
    return rewritten


def _apply_optimized_layout_style(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate | None = None,
) -> str:
    from clay.flow.optimized_layout_style import derive_optimized_layout_style

    cell = _select_cell(parse_clay_source(source), top_cell)
    style = derive_optimized_layout_style(cell, candidate=candidate)
    from clay.layout.intent import layout_intent_from_cell_def

    source_intent = layout_intent_from_cell_def(cell)
    route_groups = tuple(source_intent.route_groups)
    route_group_lines = tuple(
        "@route_group("
        + declaration.group
        + (
            ", groups=("
            + ", ".join(declaration.place_groups)
            + "), ports="
            if declaration.place_groups
            else ", ports="
        )
        + declaration.ports
        + ");"
        for declaration in route_groups
    )

    block_start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="layout_intent", name=top_cell
    )
    body = source[open_brace + 1 : block_end - 1]
    # Existing constructor declarations carry source-owned array topology,
    # transforms and pin access. Functional-role style is not an alternative
    # authority for those fields. Preserve the complete declarations and let
    # the explicit candidate-axis materializers update their selected values.
    constructor_key = re.compile(
        r"@(?P<kind>array|common_centroid)\s*\(\s*"
        r"(?P<target>[A-Za-z_][A-Za-z0-9_.]*)\s*[,)]"
    )
    constructor_lines = tuple(
        match.group(1).strip()
        for match in re.finditer(
            r"(?m)^[ \t]*(@(?:array|common_centroid)\s*\([^;]*\);)", body
        )
    )
    source_constructors = {}
    for line in constructor_lines:
        match = constructor_key.match(line)
        if match is None:
            raise ValueError("source array/common-centroid constructor has no explicit target")
        key = (match.group("kind"), match.group("target"))
        if key in source_constructors:
            raise ValueError(f"duplicate source constructor {key!r}")
        source_constructors[key] = line
    source_place_groups = {group.group for group in source_intent.place_groups}
    constructor_targets = {target for _kind, target in source_constructors}
    ordered_groups = set(() if candidate is None else candidate.group_order)
    unconstrained_place_group = re.compile(
        r"@place_group\((?P<target>[A-Za-z_][A-Za-z0-9_]*), "
        r"devices=\([^)]*\)\);$"
    )
    generated_lines = []
    consumed_constructors = set()
    for line in style.layout_lines:
        match = constructor_key.match(line)
        if match is not None:
            key = (match.group("kind"), match.group("target"))
            if key in source_constructors:
                # Keep the style slot as well as the source fields, so a
                # generated candidate array replays byte-for-byte after it
                # becomes an ordinary declaration in the derivative.
                generated_lines.append(source_constructors[key])
                consumed_constructors.add(key)
                continue
        place = unconstrained_place_group.fullmatch(line)
        if (place is not None
                and place.group("target") in constructor_targets
                and place.group("target") not in source_place_groups
                and place.group("target") not in ordered_groups):
            # An existing array constructor already owns this complete group.
            # Adding an otherwise empty placement declaration changes its
            # source graph identity without expressing a placement choice.
            continue
        generated_lines.append(line)
    remaining_constructors = tuple(
        line for key, line in source_constructors.items()
        if key not in consumed_constructors
    )
    generated_kinds = (
        "place_group|guard_ring|array|common_centroid|band|anchor|order|"
        "abut|well_island|channel|rail_affinity|split|trunk|route_group"
    )
    body = re.sub(
        r"(?m)^[ \t]*// Executable compact style derived from "
        r"functional-group DATA\.[ \t]*(?:\n)?",
        "",
        body,
    )
    body = re.sub(
        rf"(?ms)^[ \t]*@(?:{generated_kinds})\s*\([^;]*\);[ \t]*(?:\r?\n)?",
        "",
        body,
    ).rstrip()
    layout_lines = "\n".join(
        f"  {line}" for line in (*generated_lines, *remaining_constructors, *route_group_lines)
    )
    generated_layout = (
        body
        + "\n\n  // Executable compact style derived from functional-group DATA.\n"
        + layout_lines
        + "\n"
    )
    source = source[: open_brace + 1] + generated_layout + source[block_end - 1 :]

    _annotate_start, annotate_open, annotate_end = _top_level_named_block_span(
        source, keyword="annotate", name=top_cell
    )
    if candidate is not None:
        annotation_body = source[annotate_open + 1 : annotate_end - 1]
        annotation_body = re.sub(
            r"(?m)^[ \t]*// Fold wide optimized devices into compact "
            r"unit-finger geometry\.[ \t]*(?:\n)?",
            "",
            annotation_body,
        )
        oriented_passives = (
            [split.passive for split in candidate.passive_splits]
            + [passive for passive, _orientation in candidate.passive_orientations]
        )
        for passive in oriented_passives:
            annotation_body = re.sub(
                rf"(?m)^[ \t]*{re.escape(passive)}\s*:\s*"
                rf"@orient\s*\([^;]*\);[ \t]*(?:\r?\n)?",
                "",
                annotation_body,
            )
        if candidate.mos_fold_height_um is not None:
            annotation_body = re.sub(
                r"(?m)^[ \t]*[A-Za-z_][A-Za-z0-9_]*\s*:\s*"
                r"@fold\s*(?:\([^;]*\))?;[ \t]*(?:\r?\n)?",
                "",
                annotation_body,
            )
        annotation_body = annotation_body.rstrip()
        source = (
            source[: annotate_open + 1]
            + annotation_body
            + source[annotate_end - 1 :]
        )
        _annotate_start, annotate_open, annotate_end = _top_level_named_block_span(
            source, keyword="annotate", name=top_cell
        )
    if style.annotation_lines:
        annotations = "\n".join(
            f"  {line}" for line in style.annotation_lines
        )
        insertion = (
            "\n  // Fold wide optimized devices into compact unit-finger geometry.\n"
            + annotations
            + "\n"
        )
        source = source[: annotate_end - 1] + insertion + source[annotate_end - 1 :]
    return source


def _apply_candidate_fold_annotations(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Lower only the shared fold plan while retaining authored layout intent."""

    if candidate.mos_fold_height_um is None:
        return source
    from clay.flow.optimized_layout_style import derive_optimized_layout_style

    cell = _select_cell(parse_clay_source(source), top_cell)
    fold_lines = tuple(
        line
        for line in derive_optimized_layout_style(
            cell, candidate=candidate
        ).annotation_lines
        if re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*@fold\s*(?:\([^;]*\))?;",
            line,
        )
    )
    _annotate_start, annotate_open, annotate_end = _top_level_named_block_span(
        source, keyword="annotate", name=top_cell
    )
    body = source[annotate_open + 1 : annotate_end - 1]
    body = re.sub(
        r"(?m)^[ \t]*// Fold plan selected by compact intent\.[ \t]*(?:\r?\n)?",
        "",
        body,
    )
    body = re.sub(
        r"(?m)^[ \t]*[A-Za-z_][A-Za-z0-9_]*\s*:\s*"
        r"@fold\s*(?:\([^;]*\))?;[ \t]*(?:\r?\n)?",
        "",
        body,
    ).rstrip()
    insertion = "\n\n  // Fold plan selected by compact intent."
    if fold_lines:
        insertion += "\n" + "\n".join(f"  {line}" for line in fold_lines)
    insertion += "\n"
    return (
        source[: annotate_open + 1]
        + body
        + insertion
        + source[annotate_end - 1 :]
    )


def _preserve_authored_layout_intent(cell: object) -> bool:
    """Return the exact source-declared opt-in for ring-only derivatives."""

    raw = dict(getattr(cell, "sizing_kwargs", {}) or {}).get(
        "opt_layout_preserve_authored_intent", 0
    )
    if type(raw) not in (int, float) or not math.isfinite(float(raw)):
        raise ValueError(
            "opt_layout_preserve_authored_intent must be finite numeric DATA"
        )
    if float(raw) not in (0.0, 1.0):
        raise ValueError(
            "opt_layout_preserve_authored_intent must be exactly zero or one"
        )
    return float(raw) == 1.0


def _apply_guard_ring_strategy(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Replace only guard-ring ownership and retain every other declaration."""

    from clay.layout.intent import layout_intent_from_cell_def

    source_cell = _select_cell(parse_clay_source(source), top_cell)
    source_rings = tuple(layout_intent_from_cell_def(source_cell).guard_rings)
    source_owner = {
        member: index
        for index, ring in enumerate(source_rings)
        for member in ring.members
    }

    def ring_line(ring: object) -> str:
        members = tuple(getattr(ring, "members"))
        try:
            owners = tuple(sorted({source_owner[member] for member in members}))
        except KeyError as exc:
            raise CompactIntentMaterializationError(
                "UNKNOWN_GUARD_RING_MEMBER",
                f"candidate ring member {exc.args[0]!r} has no authored owner",
            ) from exc
        metadata = {
            (
                source_rings[index].hug,
                source_rings[index].role,
                source_rings[index].enclose,
            )
            for index in owners
        }
        if len(metadata) != 1:
            raise CompactIntentMaterializationError(
                "INCOMPATIBLE_GUARD_RING_CRAFT",
                "candidate ring cannot merge authored hug/role/enclose contracts",
            )
        hug, role, enclose = metadata.pop()
        pack = (
            "hierarchical"
            if len(owners) > 1
            else str(getattr(source_rings[owners[0]], "pack", "") or "")
        )
        craft = "".join(
            f", {name}={value}"
            for name, value in (
                ("hug", hug),
                ("role", role),
                ("enclose", enclose),
                ("pack", pack),
            )
            if value
        )
        return (
            "  @guard_ring(group=("
            + ", ".join(members)
            + f"), well={getattr(ring, 'well')}, "
            + f"tie={getattr(ring, 'tie_net')}{craft});"
        )

    _start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="layout_intent", name=top_cell
    )
    body = source[open_brace + 1 : block_end - 1]
    body = re.sub(
        r"(?m)^[ \t]*// Guard-ring strategy selected by compact intent\.[ \t]*"
        r"(?:\r?\n)?",
        "",
        body,
    )
    body = re.sub(
        r"(?ms)^[ \t]*@guard_ring\s*\([^;]*\);[ \t]*(?:\r?\n)?",
        "",
        body,
    )
    body = re.sub(r"(?:\r?\n){3,}", "\n\n", body).rstrip()
    ring_lines = "\n".join(ring_line(ring) for ring in candidate.guard_ring_groups)
    generated = body + "\n\n  // Guard-ring strategy selected by compact intent."
    if ring_lines:
        generated += "\n" + ring_lines
    generated += "\n"
    return source[: open_brace + 1] + generated + source[block_end - 1 :]


def _apply_passive_bank_strategy(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Replace graph-owned passive/well place groups in preserve mode."""

    choices = (
        *candidate.passive_bank_partitions,
        *candidate.well_island_partitions,
    )
    if not choices:
        return source
    from clay.flow.optimized_layout_style import _materialize_passive_partition_rows
    from clay.layout.intent import layout_intent_from_cell_def

    cell = _select_cell(parse_clay_source(source), top_cell)
    intent = layout_intent_from_cell_def(cell)
    owned = {
        member
        for choice in choices
        for group in choice.member_groups
        for member in group
    }
    affected = tuple(
        group for group in intent.place_groups if set(group.devices) & owned
    )
    if not affected:
        raise CompactIntentMaterializationError(
            "UNCONSUMED_MACRO_PARTITION",
            "macro partition has no authored place-group owner",
        )
    if any(group.col is not None for group in affected):
        raise CompactIntentMaterializationError(
            "UNSUPPORTED_MACRO_PARTITION",
            "macro partition cannot discard an authored column constraint",
        )
    rows = [
        (
            group.group,
            tuple(group.devices),
            group.row_min,
            group.row_max,
            group.priority,
        )
        for group in intent.place_groups
    ]
    try:
        materialized = _materialize_passive_partition_rows(rows, choices)
    except ValueError as exc:
        raise CompactIntentMaterializationError(
            "UNCONSUMED_MACRO_PARTITION", str(exc)
        ) from exc

    affected_names = {group.group for group in affected}
    band_owner_names = {
        str(getattr(declaration, "center", "") or "")
        for declaration in tuple(getattr(intent, "bands", ()) or ())
    }
    removable_names = affected_names - band_owner_names
    unaffected_names = {
        group.group for group in intent.place_groups if group.group not in affected_names
    }
    replacement_rows = tuple(
        row for row in materialized if row[0] not in unaffected_names
    )
    _start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="layout_intent", name=top_cell
    )
    body = source[open_brace + 1 : block_end - 1]
    body = re.sub(
        r"(?m)^[ \t]*// (?:Passive-bank|Macro-partition) strategy selected "
        r"by compact intent\.[ \t]*"
        r"(?:\r?\n)?",
        "",
        body,
    )
    if removable_names:
        names = "|".join(re.escape(name) for name in sorted(removable_names))
        body = re.sub(
            rf"(?ms)^[ \t]*@place_group\(\s*(?:{names})\s*,[^;]*\);"
            rf"[ \t]*(?:\r?\n)?",
            "",
            body,
        )
    body = re.sub(r"(?:\r?\n){3,}", "\n\n", body).rstrip()

    lines: list[str] = []
    for name, members, row_min, row_max, priority in replacement_rows:
        if (row_min is None) != (row_max is None):
            raise CompactIntentMaterializationError(
                "UNSUPPORTED_MACRO_PARTITION",
                f"place group {name!r} has an incomplete row bound",
            )
        arguments = [name, "devices=(" + ", ".join(members) + ")"]
        if row_min is not None and row_max is not None:
            arguments.append(
                f"row={row_min}"
                if row_min == row_max
                else f"rows={row_min}..{row_max}"
            )
        if priority:
            arguments.append(f"priority={priority}")
        lines.append("  @place_group(" + ", ".join(arguments) + ");")
    strategy_label = (
        "Macro-partition"
        if candidate.well_island_partitions
        else "Passive-bank"
    )
    generated = (
        body
        + f"\n\n  // {strategy_label} strategy selected by compact intent."
    )
    if lines:
        generated += "\n" + "\n".join(lines)
    generated += "\n"
    return source[: open_brace + 1] + generated + source[block_end - 1 :]


def _apply_passive_bank_declarations(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Materialize candidate partitions for both preserved and derived styles."""

    if not candidate.passive_bank_partitions:
        return source
    from clay.layout.compact_intent_domain import (
        passive_bank_partition_intents_for_candidate,
    )

    cell = _select_cell(parse_clay_source(source), top_cell)
    declarations = passive_bank_partition_intents_for_candidate(cell, candidate)
    parents = tuple(sorted({
        choice.macro_id.split(":")[1]
        for choice in candidate.passive_bank_partitions
    }))
    _start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="layout_intent", name=top_cell
    )
    body = source[open_brace + 1 : block_end - 1]
    body = re.sub(
        r"(?m)^[ \t]*// Passive-bank declarations selected by compact intent\."
        r"[ \t]*(?:\r?\n)?",
        "",
        body,
    )
    parent_pattern = "|".join(re.escape(parent) for parent in parents)
    body = re.sub(
        rf"(?m)^[ \t]*@passive_bank_partition\(\s*parent\s*=\s*"
        rf"(?:{parent_pattern})\s*,[^;]*\);[ \t]*(?:\r?\n)?",
        "",
        body,
    )
    body = re.sub(r"(?:\r?\n){3,}", "\n\n", body).rstrip()
    lines = tuple(
        "  @passive_bank_partition(parent="
        + declaration.parent
        + ", bank="
        + declaration.bank
        + ", members=("
        + ", ".join(declaration.members)
        + "));"
        for declaration in declarations
    )
    generated = body + "\n\n  // Passive-bank declarations selected by compact intent."
    if lines:
        generated += "\n" + "\n".join(lines)
    generated += "\n"
    return source[: open_brace + 1] + generated + source[block_end - 1 :]


def _apply_common_centroid_row_strategy(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Materialize the selected row count through existing row declarations.

    The row count is a structural axis.  Existing dummy capacity is conserved
    only when the resulting end-cap topology is physically realizable.  A
    drain-exposed shared-diffusion flank remains exactly one finger per row end;
    a fold that would require more is rejected rather than silently dropping or
    electrically changing edge-condition artwork.
    """

    candidate = replay_compact_intent_candidate(candidate)
    if not candidate.common_centroid_rows:
        return source
    from clay.layout.compact_intent_domain import (
        common_centroid_row_materialization_error,
    )
    from clay.layout.intent import layout_intent_from_cell_def

    cell = _select_cell(parse_clay_source(source), top_cell)
    intent = layout_intent_from_cell_def(cell)
    centroids = {item.group: item for item in intent.common_centroids}
    dummies = {item.target: item for item in intent.dummy_flanks}
    if len(centroids) != len(intent.common_centroids):
        raise CompactIntentMaterializationError(
            "AMBIGUOUS_COMMON_CENTROID", "duplicate common-centroid group"
        )
    if len(dummies) != len(intent.dummy_flanks):
        raise CompactIntentMaterializationError(
            "AMBIGUOUS_DUMMY_FLANK", "duplicate dummy-flank target"
        )

    selected = dict(candidate.common_centroid_rows)
    unknown = tuple(sorted(set(selected) - set(centroids)))
    if unknown:
        raise CompactIntentMaterializationError(
            "UNKNOWN_COMMON_CENTROID", f"unknown groups: {unknown!r}"
        )
    dummy_per_end: dict[str, int] = {}
    for group, rows in selected.items():
        dummy = dummies.get(group)
        error = common_centroid_row_materialization_error(
            centroids[group].rows,
            rows,
            dummy,
        )
        if error is not None:
            raise CompactIntentMaterializationError(
                "UNMATERIALIZABLE_DUMMY_TOPOLOGY",
                f"group {group!r}: {error}",
            )
        if dummy is None:
            continue
        budget = int(centroids[group].rows) * int(dummy.per_end)
        if budget % rows:
            raise CompactIntentMaterializationError(
                "UNMATERIALIZABLE_DUMMY_BUDGET",
                f"group {group!r} cannot conserve {budget} dummy fingers over "
                f"{rows} rows",
            )
        dummy_per_end[group] = budget // rows

    _start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="layout_intent", name=top_cell
    )
    body = source[open_brace + 1 : block_end - 1]

    def rewrite(kind: str, field: str, values: Mapping[str, int]) -> None:
        nonlocal body
        seen: dict[str, int] = {name: 0 for name in values}
        pattern = re.compile(
            rf"(?m)^(?P<indent>[ \t]*)@{kind}\((?P<args>[^;]*)\);"
        )

        def replace_call(match: re.Match[str]) -> str:
            arguments = _split_top_level_arguments(match.group("args"))
            if not arguments:
                return match.group(0)
            target = arguments[0]
            if "=" in target:
                key, value = (part.strip() for part in target.split("=", 1))
                if key not in {"group", "target"}:
                    return match.group(0)
                target = value
            target = target.strip()
            if target not in values:
                return match.group(0)
            seen[target] += 1
            rewritten = [
                argument
                for argument in arguments
                if re.match(rf"^{re.escape(field)}\s*=", argument) is None
            ]
            rewritten.append(f"{field}={values[target]}")
            return (
                match.group("indent")
                + f"@{kind}("
                + ", ".join(rewritten)
                + ");"
            )

        body = pattern.sub(replace_call, body)
        bad = tuple(sorted(name for name, count in seen.items() if count != 1))
        if bad:
            raise CompactIntentMaterializationError(
                "UNCONSUMED_COMMON_CENTROID_ROWS",
                f"{kind} declarations for {bad!r} were not consumed exactly once",
            )

    rewrite("common_centroid", "rows", selected)
    if dummy_per_end:
        rewrite("dummy_flank", "per_end", dummy_per_end)

    # A generated discrete @array is the same row master declaration.  Preserved
    # authoring sources may have no such redundant row, so rewrite it only when
    # present and never fabricate a second materialization path.
    array_groups = {
        item.group
        for item in intent.arrays
        if item.group in selected and str(item.style) == "discrete"
    }
    if array_groups:
        rewrite(
            "array", "rows", {group: selected[group] for group in array_groups}
        )
    return source[: open_brace + 1] + body + source[block_end - 1 :]


def _apply_compact_candidate_place_policy(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Materialize the candidate's natural/fixed aspect through existing policy."""

    if candidate.aspect_mode == "natural":
        rewritten = _apply_area_minimized_place_policy(source, top_cell=top_cell)
    else:
        ratio = float(candidate.aspect_mode.split(":", 1)[1])
        rewritten = source
        parsed = parse_clay_source(rewritten)
        cell = _select_cell(parsed, top_cell)
        flow_decl = getattr(cell, "flow_decl", None)
        policy_name = str(getattr(flow_decl, "policy_ref", "") or "")
        if not policy_name:
            raise ValueError("compact candidate aspect requires a selected place policy")
        _start, open_brace, block_end = _top_level_named_block_span(
            rewritten, keyword="policy", name=policy_name
        )
        body = rewritten[open_brace + 1 : block_end - 1]
        pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)@place_policy\s*\((?P<args>[^;]*)\);"
        )
        matches = list(pattern.finditer(body))
        if len(matches) != 1:
            raise ValueError(
                f"compact candidate requires exactly one @place_policy in "
                f"policy {policy_name!r}; found {len(matches)}"
            )
        match = matches[0]
        arguments = [
            argument
            for argument in _split_top_level_arguments(match.group("args"))
            if not re.match(r"^aspect\s*=", argument)
        ]
        arguments.insert(0, f"aspect = {format(ratio, '.12g')}")
        replacement = (
            match.group("indent") + "@place_policy(" + ", ".join(arguments) + ");"
        )
        rewritten_body = body[: match.start()] + replacement + body[match.end() :]
        rewritten = (
            rewritten[: open_brace + 1]
            + rewritten_body
            + rewritten[block_end - 1 :]
        )

    parsed = parse_clay_source(rewritten)
    cell = _select_cell(parsed, top_cell)
    flow_decl = getattr(cell, "flow_decl", None)
    policy_name = str(getattr(flow_decl, "policy_ref", "") or "")
    if not policy_name:
        raise ValueError("compact candidate pack requires a selected place policy")
    _start, open_brace, block_end = _top_level_named_block_span(
        rewritten, keyword="policy", name=policy_name
    )
    body = rewritten[open_brace + 1 : block_end - 1]
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)@place_policy\s*\((?P<args>[^;]*)\);"
    )
    matches = list(pattern.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            f"compact candidate pack requires exactly one @place_policy in "
            f"policy {policy_name!r}; found {len(matches)}"
        )
    match = matches[0]
    arguments = [
        argument
        for argument in _split_top_level_arguments(match.group("args"))
        if not re.match(r"^pack\s*=", argument)
        and not re.match(r"^left_column\s*=", argument)
        and not re.match(r"^channel_strategy\s*=", argument)
        and not (
            candidate.block_gap_um is not None
            and re.match(r"^gap\s*=", argument)
        )
    ]
    arguments.insert(0, f"pack = {candidate.pack_mode}")
    if candidate.left_column_group:
        arguments.insert(1, f"left_column = {candidate.left_column_group}")
    if candidate.block_gap_um is not None:
        arguments.insert(
            2 if candidate.left_column_group else 1,
            f"gap = {format(float(candidate.block_gap_um), '.12g')}",
        )
    if candidate.channel_strategy != "declared":
        arguments.append(f"channel_strategy = {candidate.channel_strategy}")
    replacement = (
        match.group("indent") + "@place_policy(" + ", ".join(arguments) + ");"
    )
    rewritten_body = body[: match.start()] + replacement + body[match.end() :]
    return (
        rewritten[: open_brace + 1]
        + rewritten_body
        + rewritten[block_end - 1 :]
    )


def _apply_compact_candidate_edge_rail_width(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> str:
    """Rewrite existing edge-trunk widths through the existing Clay intent slot."""

    if candidate.edge_rail_width_um is None:
        return source
    _start, open_brace, block_end = _top_level_named_block_span(
        source,
        keyword="layout_intent",
        name=top_cell,
    )
    body = source[open_brace + 1 : block_end - 1]
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)@trunk\((?P<args>[^;]*)\);"
    )
    replacements = 0

    def rewrite(match: re.Match[str]) -> str:
        nonlocal replacements
        arguments = _split_top_level_arguments(match.group("args"))
        if not any(re.match(r"^edge\s*=", argument) for argument in arguments):
            return match.group(0)
        width = format(float(candidate.edge_rail_width_um), ".12g")
        rewritten_arguments: list[str] = []
        width_seen = False
        for argument in arguments:
            if re.match(r"^width\s*=", argument):
                rewritten_arguments.append(f"width={width}")
                width_seen = True
            else:
                rewritten_arguments.append(argument)
        if not width_seen:
            rewritten_arguments.append(f"width={width}")
        replacements += 1
        return (
            match.group("indent")
            + "@trunk("
            + ", ".join(rewritten_arguments)
            + ");"
        )

    rewritten_body = pattern.sub(rewrite, body)
    if replacements == 0:
        raise ValueError(
            "compact candidate edge rail width requires at least one declared "
            f"edge @trunk in layout_intent {top_cell!r}"
        )
    return source[: open_brace + 1] + rewritten_body + source[block_end - 1 :]


def _apply_compact_candidate_materialization(
    source: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
    allow_materialized_passive_arrays: bool = False,
) -> str:
    """Return the marker-free canonical physical derivative for ``candidate``.

    This is deliberately shared by the writer and the admission gate.  Replaying
    it over an admitted candidate must be idempotent; a marker and a self-hash
    cannot substitute for materializing the declared axes in executable Clay.
    """

    candidate = replay_compact_intent_candidate(candidate)
    rewritten = _canonical_source_text(source)
    rewritten = _COMPACT_INTENT_ROW.sub("", rewritten).rstrip()
    if candidate.materialization_mode == "authoring":
        return rewritten
    _validate_compact_candidate_graph(
        rewritten,
        top_cell=top_cell,
        candidate=candidate,
        allow_materialized_passive_arrays=allow_materialized_passive_arrays,
    )
    cell = _select_cell(parse_clay_source(rewritten), top_cell)
    if candidate.channel_strategy == "block_gap_only":
        from clay.layout.intent import layout_intent_from_cell_def

        if not tuple(layout_intent_from_cell_def(cell).channels):
            raise CompactIntentMaterializationError(
                "MISSING_DECLARED_CHANNEL",
                "block_gap_only requires at least one authoring channel",
            )
    if _preserve_authored_layout_intent(cell):
        rewritten = _apply_guard_ring_strategy(
            rewritten,
            top_cell=top_cell,
            candidate=candidate,
        )
        rewritten = _apply_passive_bank_strategy(
            rewritten,
            top_cell=top_cell,
            candidate=candidate,
        )
        rewritten = _apply_candidate_fold_annotations(
            rewritten,
            top_cell=top_cell,
            candidate=candidate,
        )
    else:
        rewritten = _apply_optimized_layout_style(
            rewritten,
            top_cell=top_cell,
            candidate=candidate,
        )
    rewritten = _apply_common_centroid_row_strategy(
        rewritten,
        top_cell=top_cell,
        candidate=candidate,
    )
    rewritten = _apply_passive_bank_declarations(
        rewritten,
        top_cell=top_cell,
        candidate=candidate,
    )
    rewritten = _apply_compact_candidate_place_policy(
        rewritten,
        top_cell=top_cell,
        candidate=candidate,
    )
    rewritten = _apply_compact_candidate_edge_rail_width(
        rewritten,
        top_cell=top_cell,
        candidate=candidate,
    )
    return rewritten.rstrip()


def _split_top_level_arguments(text: str) -> list[str]:
    """Split one Clay annotation argument list without splitting nested tuples."""

    out: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{" :
            depth += 1
        elif char in ")]}" :
            depth -= 1
        elif char == "," and depth == 0:
            out.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _apply_area_minimized_place_policy(source: str, *, top_cell: str) -> str:
    """Let a declared area objective choose its natural compact aspect.

    A fixed aspect is a shape preference, not an area constraint.  Carrying it
    into the optimized derivative makes the second placement pass expand an
    already area-minimal floorplan.  Preserve every other policy setting and
    remove only ``aspect`` from the policy selected by this cell.
    """

    from clay.layout.intent import layout_intent_from_cell_def

    parsed = parse_clay_source(source)
    cell = _select_cell(parsed, top_cell)
    if layout_intent_from_cell_def(cell).objective != ("area", "minimize"):
        return source
    flow_decl = getattr(cell, "flow_decl", None)
    policy_name = str(getattr(flow_decl, "policy_ref", "") or "")
    if not policy_name:
        return source

    _start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="policy", name=policy_name
    )
    body = source[open_brace + 1 : block_end - 1]
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)@place_policy\s*\((?P<args>[^;]*)\);"
    )
    matches = list(pattern.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            f"area-minimized optimized Clay requires exactly one @place_policy "
            f"in policy {policy_name!r}; found {len(matches)}"
        )
    match = matches[0]
    arguments = _split_top_level_arguments(match.group("args"))
    kept = [
        argument
        for argument in arguments
        if not re.match(r"^aspect\s*=", argument)
    ]
    if len(kept) == len(arguments):
        return source
    replacement = ""
    if kept:
        replacement = (
            match.group("indent") + "@place_policy(" + ", ".join(kept) + ");"
        )
    rewritten_body = body[: match.start()] + replacement + body[match.end() :]
    return source[: open_brace + 1] + rewritten_body + source[block_end - 1 :]


def _apply_optimized_route_policy(source: str, *, top_cell: str) -> str:
    """Select the existing deck-aware analog router in an optimized derivative.

    An author-declared route policy remains authoritative.  When the authoring
    source left routing implicit, the optimized physical derivative declares the
    same gridless/via-redundancy profile used by the cross-PDK reference OTAs.
    This is policy DATA consumed by the existing engine registry; no router or
    circuit-specific path is introduced here.
    """

    parsed = parse_clay_source(source)
    cell = _select_cell(parsed, top_cell)
    flow_decl = getattr(cell, "flow_decl", None)
    policy_name = str(getattr(flow_decl, "policy_ref", "") or "")
    if not policy_name:
        return source
    _start, open_brace, block_end = _top_level_named_block_span(
        source, keyword="policy", name=policy_name
    )
    body = source[open_brace + 1 : block_end - 1]
    if re.search(r"(?m)^[ \t]*@route_policy\s*\(", body):
        return source
    place_matches = list(
        re.finditer(r"(?m)^(?P<indent>[ \t]*)@place_policy\s*\([^;]*\);", body)
    )
    if len(place_matches) != 1:
        raise ValueError(
            f"optimized route policy requires exactly one @place_policy in "
            f"policy {policy_name!r}; found {len(place_matches)}"
        )
    place = place_matches[0]
    declaration = (
        "\n"
        + place.group("indent")
        + "@route_policy(engine = gridless, time_budget_sec = 900, "
        + "via_redundancy = 2);"
    )
    rewritten_body = body[: place.end()] + declaration + body[place.end() :]
    return source[: open_brace + 1] + rewritten_body + source[block_end - 1 :]


def _semantic_snapshot(cell: Any) -> dict[str, Any]:
    return {
        "ports": tuple(
            (str(port), str(cell.port_roles.get(str(port), "")))
            for port in cell.ports
        ),
        "devices": {
            device.instance_name: {
                "class": device.device_class,
                "pins": dict(device.pins),
                "sizing": dict(device.sizing),
                "source_model": device.source_model,
                "pdk_variant": device.pdk_variant,
                "native_parameters": tuple(device.native_parameters),
            }
            for device in cell.devices
        },
    }


def _validate_semantics(
    source: str,
    rewritten: str,
    *,
    top_cell: str,
    optimization: OptimizationSpec,
    sizing: Mapping[str, float],
) -> None:
    from clay.flow.candidate_bindings import apply_candidate_device_bindings

    parsed_before, parsed_after = parse_clay_source(source), parse_clay_source(rewritten)
    if parsed_before.pdk_id != parsed_after.pdk_id:
        raise ValueError("optimized Clay rewrite changed the PDK")
    expected_cell = _select_cell(parsed_before, top_cell)
    if any(device.device_class == "native" for device in expected_cell.devices):
        from clay.flow.spec import _optimization_from_sizing

        if optimization != _optimization_from_sizing(expected_cell):
            raise ValueError("native optimization intent differs from the authoring source")
        expected_sizing = dict(expected_cell.sizing_kwargs)
        expected_sizing.update(sizing)
        actual_sizing = dict(_select_cell(parsed_after, top_cell).sizing_kwargs)
        if set(expected_sizing) != set(actual_sizing) or any(
            not math.isclose(float(actual_sizing[key]), float(value), rel_tol=1e-12, abs_tol=1e-24)
            for key, value in expected_sizing.items()
        ):
            raise ValueError("native candidate changed sizing declarations or failed shared variable binding")
    before = _semantic_snapshot(expected_cell)
    after = _semantic_snapshot(_select_cell(parsed_after, top_cell))
    if before["ports"] != after["ports"]:
        raise ValueError("optimized Clay rewrite changed cell terminals")
    if set(before["devices"]) != set(after["devices"]):
        raise ValueError("optimized Clay rewrite changed device identities")
    apply_candidate_device_bindings(expected_cell, optimization, sizing,
                                    pdk_id=parsed_before.pdk_id)
    if any(device.device_class == "native" for device in expected_cell.devices):
        from clay.flow.candidate_bindings import native_semantic_envelope

        expected_cell.sizing_kwargs.update(sizing)
        if native_semantic_envelope(parsed_before) != native_semantic_envelope(parsed_after):
            raise ValueError("native candidate changed unbound source, physical or flow intent")
    expected_snapshot = _semantic_snapshot(expected_cell)
    for name, original in before["devices"].items():
        candidate = after["devices"][name]
        if any(original[key] != candidate[key] for key in
               ("class", "pins", "source_model", "pdk_variant")):
            raise ValueError(f"optimized Clay rewrite changed topology for {name}")
        expected_device = expected_snapshot["devices"][name]
        if expected_device["native_parameters"] != candidate["native_parameters"]:
            raise ValueError(f"optimized Clay rewrite changed unbound native parameters or failed binding for {name}")
        if set(expected_device["sizing"]) != set(candidate["sizing"]):
            raise ValueError(f"optimized Clay rewrite changed sizing fields for {name}")
        for field, original_value in original["sizing"].items():
            expected = expected_device["sizing"][field]
            if not math.isclose(
                float(candidate["sizing"][field]),
                float(expected),
                rel_tol=1e-12,
                abs_tol=1e-24,
            ):
                raise ValueError(
                    f"optimized Clay rewrite failed binding device.{name}.{field}"
                )


def _canonical_optimization_sizing(
    optimization: OptimizationSpec | None,
    raw_sizing: Mapping[str, Any],
    *,
    native_variables: frozenset[str] = frozenset(),
) -> dict[str, float]:
    if optimization is None:
        raise ValueError("optimized Clay materialization requires optimization intent")
    expected_names = {variable.name for variable in optimization.variables}
    supplied_names = {str(name) for name in raw_sizing}
    missing = sorted(expected_names - supplied_names)
    unknown = sorted(supplied_names - expected_names)
    if missing:
        raise ValueError(f"optimized Clay incumbent has missing sizing: {missing}")
    if unknown:
        raise ValueError(f"optimized Clay incumbent has unknown sizing: {unknown}")
    raw_values = {name: float(raw_sizing[name]) for name in sorted(expected_names)}
    for variable in optimization.variables:
        value = raw_values[variable.name]
        if math.isfinite(value):
            if value < variable.low and math.isclose(value, variable.low, rel_tol=1e-12, abs_tol=1e-24):
                value = variable.low
                raw_values[variable.name] = value
            elif value > variable.high and math.isclose(value, variable.high, rel_tol=1e-12, abs_tol=1e-24):
                value = variable.high
                raw_values[variable.name] = value
        if not math.isfinite(value) or not variable.low <= value <= variable.high:
            raise ValueError(
                f"optimized Clay sizing {variable.name}={value!r} lies outside "
                f"[{variable.low:g}, {variable.high:g}]"
            )
    sizing: dict[str, float] = {
        variable.name: canonicalize_variable_value(
            variable, raw_values[variable.name], native=variable.name in native_variables
        )
        for variable in optimization.variables
    }
    for variable in optimization.variables:
        value = sizing[variable.name]
        if value < variable.low and math.isclose(value, variable.low, rel_tol=1e-12, abs_tol=1e-24):
            value = variable.low
            sizing[variable.name] = value
        elif value > variable.high and math.isclose(value, variable.high, rel_tol=1e-12, abs_tol=1e-24):
            value = variable.high
            sizing[variable.name] = value
        if not variable.low <= value <= variable.high:
            raise ValueError(
                f"canonical optimized Clay sizing {variable.name}={value!r} "
                f"lies outside [{variable.low:g}, {variable.high:g}]"
            )

    return sizing


def _resolve_source_grid_nm(parsed: Any) -> int:
    pdk_id = getattr(parsed, "pdk_id", None)
    if not pdk_id:
        return 1
    try:
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        from clay.layout.tech import LayoutTech

        return max(1, int(round(pdk_manufacturing_grid_nm(LayoutTech.resolve(str(pdk_id))))))
    except Exception:  # noqa: BLE001
        return 1


def _refresh_authored_resized_folds(
    source: str,
    *,
    top_cell: str,
    optimization: OptimizationSpec,
    fold_height_um: float,
    grid_nm: int,
) -> str:
    """Refresh ``@fold`` lines only for MOS devices whose W was resized by sizing."""
    parsed = parse_clay_source(source)
    cell = _select_cell(parsed, top_cell)
    resized_mos: set[str] = set()
    for variable in optimization.variables:
        for binding in variable.bindings:
            parts = str(binding).split(".")
            if len(parts) == 3 and parts[0] == "device" and parts[2].upper() == "W":
                resized_mos.add(parts[1])
    if not resized_mos:
        return source

    devices = {str(d.instance_name): d for d in cell.devices}
    common_centroid_members: set[str] = set()
    for role in getattr(cell, "functional_roles", ()) or ():
        attrs = dict(getattr(role, "attributes", {}) or {})
        if str(attrs.get("matching_pattern", "")).lower() == "common_centroid":
            common_centroid_members.update(str(n) for n in (attrs.get("devices", ()) or ()))
    height_nm = int(round(float(fold_height_um) * 1000.0))
    grid_step = max(1, int(grid_nm))
    new_folds: dict[str, str | None] = {}
    for dev_name in sorted(resized_mos):
        dev = devices.get(dev_name)
        if dev is None:
            continue
        cls = str(dev.device_class).lower()
        if not ("pmos" in cls or "nmos" in cls or cls.startswith(("pfet", "nfet"))):
            continue
        w_m = float(dev.sizing.get("W", 0.0))
        mult = max(1, int(round(float(dev.sizing.get("M", 1.0) or 1.0))))
        total_nm = int(round(w_m * 1e9)) * mult
        common = dev_name in common_centroid_members
        if common:
            fingers = max(2, int(math.ceil(total_nm / height_nm)), mult)
            if fingers % 2:
                fingers += 1
            needs_fold = mult < 2 or mult % 2 != 0 or total_nm > 2 * height_nm
        else:
            fingers = max(mult, int(math.ceil(total_nm / height_nm)))
            needs_fold = total_nm > 2 * height_nm
        if not needs_fold:
            new_folds[dev_name] = None
            continue
        if total_nm % (fingers * grid_step) == 0:
            wf_um = (total_nm // fingers) / 1000.0
            new_folds[dev_name] = f"{dev_name}: @fold({format(wf_um + 0.01, '.12g')});"
        else:
            new_folds[dev_name] = f"{dev_name}: @fold({format(fold_height_um, '.12g')});"

    if not new_folds:
        return source
    try:
        _start, open_b, end_b = _top_level_named_block_span(
            source, keyword="annotate", name=top_cell
        )
    except ValueError:
        return source
    body = source[open_b + 1 : end_b - 1]
    for dev_name in new_folds:
        body = re.sub(
            rf"(?m)^[ \t]*{re.escape(dev_name)}\s*:\s*@fold\s*(?:\([^;]*\))?;[ \t]*(?:\r?\n)?",
            "",
            body,
        )
    body = body.rstrip()
    fold_lines = [line for line in new_folds.values() if line is not None]
    if fold_lines:
        body = body + "\n" + "\n".join(f"  {line}" for line in fold_lines) + "\n"
    else:
        body = body + "\n"
    return source[: open_b + 1] + body + source[end_b - 1 :]


def _effective_sizing_fold_height_um(cell: object, candidate: CompactIntentCandidate | None) -> float:
    from clay.flow.optimized_layout_style import DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM

    if candidate is not None and candidate.mos_fold_height_um is not None:
        return float(candidate.mos_fold_height_um)
    raw = dict(getattr(cell, "sizing_kwargs", {}) or {}).get("opt_layout_fold_height_nominal")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)) and float(raw) > 0:
        return float(raw) * 1e6
    return float(DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM)


def _canonical_physical_sizing(
    source: str,
    *,
    top_cell: str,
    optimization: OptimizationSpec | None,
    raw_sizing: Mapping[str, Any],
) -> dict[str, float]:
    """Use the joint evaluator's exact source-derived width quantization.

    The BO coordinate is continuous, but both staging evaluation and verified
    promotion materialize integer-nanometre uniform fingers.  Applying this
    mapping in only the evaluator would let promotion create a neighbouring,
    unmeasured circuit revision.  Keep both materializers on one physical
    canonicalization contract, including a compact candidate's declared fold
    height when present.
    """

    from clay.flow.optimized_layout_style import canonicalize_fold_compatible_sizing

    parsed = parse_clay_source(source)
    cell = _select_cell(parsed, top_cell)
    from clay.flow.candidate_bindings import native_bound_variables

    native_variables = native_bound_variables(cell, optimization) if optimization is not None else set()
    if any(device.device_class == "native" for device in cell.devices):
        from clay.flow.spec import _optimization_from_sizing

        if optimization != _optimization_from_sizing(cell):
            raise ValueError("native optimization intent differs from the authoring source")
    sizing = _canonical_optimization_sizing(optimization, raw_sizing,
                                           native_variables=frozenset(native_variables))
    assert optimization is not None
    if any(device.device_class == "native" for device in cell.devices):
        # Native nf and W-total are source/PDK semantics. Portable folding may
        # not silently replace them; the native geometry provider checks them.
        return sizing
    candidate = declared_compact_candidate(source)
    fold_height_um = _effective_sizing_fold_height_um(cell, candidate)
    return canonicalize_fold_compatible_sizing(
        cell,
        optimization,
        sizing,
        fold_height_um=float(fold_height_um),
        grid_nm=_resolve_source_grid_nm(parsed),
    )


def _apply_sizing_derivative(
    source: str,
    *,
    top_cell: str,
    optimization: OptimizationSpec,
    sizing: Mapping[str, float],
) -> str:
    compact_candidate = declared_compact_candidate(source)
    previous_compact_body_sha256 = (
        None
        if compact_candidate is None
        else _digest(_compact_intent_body(source).encode("utf-8"))
    )
    rewritten = _rewrite_bound_literals(
        source,
        top_cell=top_cell,
        optimization=optimization,
        sizing=sizing,
    )
    parsed_src = parse_clay_source(source)
    cell = _select_cell(parsed_src, top_cell)
    if any(device.device_class == "native" for device in cell.devices):
        if not all(device.device_class == "native" for device in cell.devices):
            raise ValueError("native sizing derivatives require a complete native cell")
        # Preserve the actual native constructor/placement/routing declarations.
        # The portable style derives MOS folding from a different size contract.
        return rewritten
    if compact_candidate is None and _preserve_authored_layout_intent(cell):
        return _refresh_authored_resized_folds(
            rewritten,
            top_cell=top_cell,
            optimization=optimization,
            fold_height_um=_effective_sizing_fold_height_um(cell, None),
            grid_nm=_resolve_source_grid_nm(parsed_src),
        )
    if (
        compact_candidate is None
        or compact_candidate.materialization_mode == "derived"
    ):
        rewritten = _apply_optimized_layout_style(
            rewritten,
            top_cell=top_cell,
            candidate=compact_candidate,
        )
    if compact_candidate is None:
        rewritten = _apply_area_minimized_place_policy(
            rewritten, top_cell=top_cell
        )
    elif compact_candidate.materialization_mode == "derived":
        rewritten = _apply_compact_candidate_place_policy(
            rewritten,
            top_cell=top_cell,
            candidate=compact_candidate,
        )
        rewritten = _apply_compact_candidate_edge_rail_width(
            rewritten,
            top_cell=top_cell,
            candidate=compact_candidate,
        )
    rewritten = _apply_optimized_route_policy(rewritten, top_cell=top_cell)
    if compact_candidate is not None:
        assert previous_compact_body_sha256 is not None
        rewritten = _refresh_compact_intent_provenance(
            rewritten,
            top_cell=top_cell,
            previous_body_sha256=previous_compact_body_sha256,
        )
    return rewritten


def materialize_sizing_candidate_clay(
    *,
    source_path: Path,
    output_path: Path,
    top_cell: str,
    optimization: OptimizationSpec | None,
    sizing: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> OptimizedClayArtifact:
    """Write one unverified BO sizing point as executable staging Clay DATA."""

    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("sizing candidate output must not overwrite its source")
    source_bytes = source_path.read_bytes()
    authoring_sha = _digest(source_bytes)
    source = _canonical_source_text(source_bytes.decode("utf-8-sig"))
    canonical = _canonical_physical_sizing(
        source,
        top_cell=top_cell,
        optimization=optimization,
        raw_sizing=sizing,
    )
    assert optimization is not None
    rewritten = _apply_sizing_derivative(
        source,
        top_cell=top_cell,
        optimization=optimization,
        sizing=canonical,
    )
    rewritten = re.sub(
        r"(?m)^// (?:optimized_clay|sizing_candidate) \{[^\r\n]*\}\s*$",
        "",
        rewritten,
    ).rstrip()
    marker = {
        "authoring_sha256": authoring_sha,
        "engine": str(provenance.get("engine", optimization.engine)),
        "seed": int(provenance.get("seed", optimization.seed)),
        "sizing_sha256": hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "truth": "staging",
        "verified": False,
    }
    rewritten += "\n\n// sizing_candidate " + json.dumps(
        marker, sort_keys=True, separators=(",", ":")
    ) + "\n"
    _validate_semantics(
        source,
        rewritten,
        top_cell=top_cell,
        optimization=optimization,
        sizing=canonical,
    )
    if _digest(source_path.read_bytes()) != authoring_sha:
        raise RuntimeError("sizing candidate source changed during materialization")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(rewritten, encoding="utf-8", newline="\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return OptimizedClayArtifact(
        path=output_path,
        sha256=_digest(output_path.read_bytes()),
        authoring_sha256=authoring_sha,
        sizing=canonical,
    )


def materialize_optimized_clay(
    *,
    source_path: Path,
    output_path: Path,
    top_cell: str,
    optimization: OptimizationSpec | None,
    incumbent: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> OptimizedClayArtifact:
    """Write a verified incumbent into a derivative Clay source atomically."""
    if incumbent.get("truth") != "post_pex" or incumbent.get("verified") is not True:
        raise ValueError("optimized Clay materialization requires a verified POST-PEX incumbent")
    raw_sizing = incumbent.get("sizing")
    if not isinstance(raw_sizing, Mapping):
        raise ValueError("optimized Clay incumbent has missing sizing")
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("optimized Clay output must not overwrite the authoring source")
    source_bytes = source_path.read_bytes()
    authoring_sha = _digest(source_bytes)
    source = _canonical_source_text(source_bytes.decode("utf-8-sig"))
    sizing = _canonical_physical_sizing(
        source,
        top_cell=top_cell,
        optimization=optimization,
        raw_sizing=raw_sizing,
    )
    assert optimization is not None
    rewritten = _apply_sizing_derivative(
        source,
        top_cell=top_cell,
        optimization=optimization,
        sizing=sizing,
    )
    # Re-targeting is allowed to start from an earlier optimized derivative.
    # Its sizing values are rewritten above, so retaining the old evidence row
    # would falsely claim that two different incumbents are simultaneously the
    # source of truth.  Keep one current marker and preserve ancestry through
    # the new marker's authoring_sha256.
    rewritten = re.sub(
        r"(?m)^// (?:optimized_clay|sizing_candidate) \{[^\r\n]*\}\s*$",
        "",
        rewritten,
    ).rstrip()
    provenance_row = {
        "authoring_sha256": authoring_sha,
        "engine": str(provenance.get("engine", optimization.engine)),
        "seed": int(provenance.get("seed", optimization.seed)),
        "truth": "post_pex",
        "verified": True,
    }
    rewritten = rewritten.rstrip() + "\n\n// optimized_clay " + json.dumps(
        provenance_row, sort_keys=True, separators=(",", ":")
    ) + "\n"
    _validate_semantics(
        source,
        rewritten,
        top_cell=top_cell,
        optimization=optimization,
        sizing=sizing,
    )
    if _digest(source_path.read_bytes()) != authoring_sha:
        raise RuntimeError("authoring Clay source changed during optimized materialization")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(rewritten, encoding="utf-8", newline="\n")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_bytes = output_path.read_bytes()
    return OptimizedClayArtifact(
        path=output_path,
        sha256=_digest(output_bytes),
        authoring_sha256=authoring_sha,
        sizing=dict(sizing),
    )


def _validate_compact_candidate_round_trip(
    rewritten: str,
    *,
    top_cell: str,
    candidate: CompactIntentCandidate,
) -> None:
    if candidate.materialization_mode == "authoring":
        # Idempotent byte replay is checked by ``require_resolved_compact_intent``;
        # this anchor deliberately has no derived split/ring axes to compare
        # against declarations that it leaves untouched.
        return
    from clay.layout.annotations import Orient
    from clay.flow.optimized_layout_style import macro_partition_place_group_name
    from clay.layout.compact_intent_domain import (
        passive_bank_partition_intents_for_candidate,
        physical_macro_graph_for_cell,
    )
    from clay.layout.intent import layout_intent_from_cell_def

    parsed = parse_clay_source(rewritten)
    cell = _select_cell(parsed, top_cell)
    intent = layout_intent_from_cell_def(cell)
    from clay.layout.policy import policy_for_cell

    observed_channel_strategy = policy_for_cell(parsed, cell).channel_strategy
    if observed_channel_strategy != candidate.channel_strategy:
        raise CompactIntentMaterializationError(
            "UNCONSUMED_CHANNEL_STRATEGY",
            f"expected={candidate.channel_strategy!r}, "
            f"got={observed_channel_strategy!r}",
        )
    observed_left_column = policy_for_cell(parsed, cell).left_column_group
    declared_centroids = {item.group for item in intent.common_centroids}
    if candidate.left_column_group not in declared_centroids | {""}:
        raise CompactIntentMaterializationError(
            "UNKNOWN_LEFT_COLUMN_GROUP",
            f"{candidate.left_column_group!r} is not a declared common-centroid group",
        )
    if observed_left_column != candidate.left_column_group:
        raise CompactIntentMaterializationError(
            "UNCONSUMED_LEFT_COLUMN_GROUP",
            f"expected={candidate.left_column_group!r}, got={observed_left_column!r}",
        )
    observed_centroid_rows = {
        item.group: item.rows for item in intent.common_centroids
    }
    for group, rows in candidate.common_centroid_rows:
        if observed_centroid_rows.get(group) != rows:
            raise CompactIntentMaterializationError(
                "UNCONSUMED_COMMON_CENTROID_ROWS",
                f"expected {group!r} rows={rows}, got "
                f"{observed_centroid_rows.get(group)!r}",
            )
    passive_arrays = tuple(
        sorted(
            (
                str(array.group),
                str(array.style),
                int(array.rows),
                str(array.order),
            )
            for array in intent.arrays
            if str(array.group) in {
                choice.passive for choice in candidate.passive_arrays
            }
        )
    )
    expected_passive_arrays = tuple(
        sorted(
            (choice.passive, "discrete", choice.rows, "ratio")
            for choice in candidate.passive_arrays
        )
    )
    if passive_arrays != expected_passive_arrays:
        raise CompactIntentMaterializationError(
            "UNCONSUMED_PASSIVE_ARRAY",
            f"expected={expected_passive_arrays}, got={passive_arrays}",
        )
    if candidate.physical_macro_graph_sha256 is not None:
        measured_graph = physical_macro_graph_for_cell(cell)
        if measured_graph.sha256 != candidate.physical_macro_graph_sha256:
            raise CompactIntentMaterializationError(
                "STALE_PHYSICAL_MACRO_GRAPH",
                "reparsed derivative graph differs from selected candidate",
            )
        expected_partitions = passive_bank_partition_intents_for_candidate(
            cell, candidate
        )
        if tuple(intent.passive_bank_partitions) != expected_partitions:
            raise CompactIntentMaterializationError(
                "UNCONSUMED_MACRO_PARTITION",
                "reparsed derivative partition declarations differ from candidate",
            )
        place_groups = {group.group: group for group in intent.place_groups}
        for choice in (
            *candidate.passive_bank_partitions,
            *candidate.well_island_partitions,
        ):
            for index, expected_members in enumerate(choice.member_groups):
                group_name = macro_partition_place_group_name(choice, index)
                group = place_groups.get(group_name)
                if group is None or tuple(sorted(group.devices)) != expected_members:
                    raise CompactIntentMaterializationError(
                        "UNCONSUMED_MACRO_PARTITION",
                        f"place group {group_name!r} does not materialize "
                        f"{expected_members!r}",
                    )
    splits = tuple(
        sorted(
            (
                str(split.passive),
                int(split.n),
                str(split.match).lower(),
            )
            for split in intent.splits
        )
    )
    expected_splits = tuple(
        sorted(
            (split.passive, split.unit_count, split.connection)
            for split in candidate.passive_splits
        )
    )
    if splits != expected_splits:
        raise ValueError(
            f"compact candidate split round-trip mismatch: "
            f"expected={expected_splits}, got={splits}"
        )
    rings = tuple(
        sorted(
            (tuple(sorted(ring.members)), ring.well, ring.tie_net)
            for ring in intent.guard_rings
        )
    )
    expected_rings = tuple(
        sorted(
            (ring.members, ring.well, ring.tie_net)
            for ring in candidate.guard_ring_groups
        )
    )
    if rings != expected_rings:
        raise CompactIntentMaterializationError(
            "BODY_WELL_TIE_MISMATCH",
            f"expected={expected_rings}, got={rings}",
        )
    annotation_map = getattr(cell, "_layout_annotations", {}) or {}
    expected_orientations = {
        split.passive: split.orientation for split in candidate.passive_splits
    }
    expected_orientations.update(dict(candidate.passive_orientations))
    for passive, expected_orientation in sorted(expected_orientations.items()):
        orientations = [
            annotation.orientation.value
            for annotation in annotation_map.get(passive, ())
            if isinstance(annotation, Orient)
        ]
        if orientations != [expected_orientation]:
            raise ValueError(
                f"compact candidate passive orientation round-trip mismatch for "
                f"{passive}: expected={[expected_orientation]}, got={orientations}"
            )


def materialize_compact_intent_clay(
    *,
    source_path: Path,
    output_path: Path,
    top_cell: str,
    candidate: CompactIntentCandidate,
    provenance: Mapping[str, Any],
    promotion: bool,
) -> CompactIntentClayArtifact:
    """Write one explicit structural candidate without changing signed sizing."""

    evidence_input: dict[str, object] = {
        "truth": str(provenance.get("truth", "staging")),
    }
    for field in (
        "physical_pex_artifact",
        "electrical_result_receipt",
        "verified",
    ):
        if field in provenance:
            evidence_input[field] = provenance[field]
    evidence = compact_intent_evidence_from_provenance(evidence_input)
    if evidence.truth is CompactIntentTruth.PHYSICAL_EXACT_ARTIFACT:
        raise ValueError(
            "generic materialization cannot authenticate caller-asserted physical evidence"
        )
    if evidence.truth is not CompactIntentTruth.STAGING:
        boundary = "generic materialization" if promotion else "non-promotion"
        raise ValueError(
            f"compact intent {boundary} can only emit staging truth"
        )
    if promotion:
        raise ValueError(
            "generic materialization cannot promote staging truth"
        )
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError(
            "compact intent output must not overwrite the signed sizing source"
        )
    source_bytes = source_path.read_bytes()
    source = _canonical_source_text(source_bytes.decode("utf-8-sig"))
    candidate = replay_compact_intent_candidate(candidate)
    if (
        candidate.materialization_mode == "authoring"
        and provenance.get("optimality", "unevaluated") != "unevaluated"
    ):
        raise ValueError("compact_intent authoring optimality must be unevaluated")
    authoring_sha = _digest(
        source.encode("utf-8")
        if candidate.materialization_mode == "authoring"
        else source_bytes
    )
    authoring_cell = _select_cell(parse_clay_source(source), top_cell)
    before = _semantic_snapshot(authoring_cell)

    rewritten = _apply_compact_candidate_materialization(
        source,
        top_cell=top_cell,
        candidate=candidate,
    )
    provenance_row: dict[str, object] = {
        "schema": "clay.compact-intent/v2",
        "authoring_sha256": authoring_sha,
        "candidate": candidate.canonical_data(),
        "candidate_sha256": candidate.digest,
        "materialized_body_sha256": _digest(
            _compact_intent_body(rewritten).encode("utf-8")
        ),
        "optimality": str(provenance.get("optimality", "unevaluated")),
        "seed": int(provenance.get("seed", 0)),
        "top_cell": str(top_cell),
    }
    if evidence.truth is CompactIntentTruth.STAGING:
        provenance_row.update(truth="staging", verified=False)
    else:
        provenance_row.update(evidence.canonical_data())
    rewritten += "\n\n// compact_intent " + json.dumps(
        provenance_row,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"

    after = _semantic_snapshot(_select_cell(parse_clay_source(rewritten), top_cell))
    if before != after:
        raise ValueError(
            "compact intent rewrite changed terminals, topology, or fixed sizing"
        )
    _validate_compact_candidate_round_trip(
        rewritten,
        top_cell=top_cell,
        candidate=candidate,
    )
    if _digest(source_path.read_bytes()) != _digest(source_bytes):
        raise RuntimeError(
            "signed sizing Clay source changed during compact intent materialization"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(rewritten, encoding="utf-8", newline="\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    source_sha = _digest(output_path.read_bytes())
    return CompactIntentClayArtifact(
        path=output_path,
        source_sha256=source_sha,
        authoring_sha256=authoring_sha,
        candidate_sha256=candidate.digest,
    )


__all__ = [
    "CompactIntentClayArtifact",
    "CompactIntentMaterializationError",
    "OptimizedClayArtifact",
    "UnresolvedCompactIntentError",
    "canonicalize_sizing_value",
    "compact_intent_search_requested",
    "declared_compact_candidate",
    "materialize_compact_intent_clay",
    "materialize_optimized_clay",
    "materialize_sizing_candidate_clay",
    "require_resolved_compact_intent",
]
