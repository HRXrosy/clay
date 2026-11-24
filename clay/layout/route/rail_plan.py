"""Exact two-stage placement plan for declared edge power rails.

This module owns only the pre-route rail-placement artifact.  It consumes a
closed scene of integer-nanometre facts, materializes each declaration's exact
internal-edge candidate, and selects all declarations simultaneously.  It
deliberately does not draw geometry and none of its certificates claim LVS
connectivity.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
from typing import Optional

from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
from clay.layout.route.resource_world import pdk_provenance_digest


BBoxNm = tuple[int, int, int, int]
PointNm = tuple[int, int]
LayerKey = tuple[int, int]

MAX_RAIL_DEMANDS = 64
MAX_RAIL_PAIR_CONTRACTS = MAX_RAIL_DEMANDS * (MAX_RAIL_DEMANDS - 1) // 2
MAX_RAIL_SCENE_FACTS = 4_096
# The production OTA's two declared supply trunks compile to 8,406 exact
# candidates after relational-event closure.  Keep a hard, power-of-two bound,
# but leave enough headroom for that proved small-cell instance instead of
# rejecting it at the former 8,192 implementation limit.
MAX_RAIL_CANDIDATES = 16_384
MAX_RAIL_CONFLICTS = 65_536
MAX_RAIL_ENUMERATION_WORK = 2_000_000
MAX_RAIL_STRING_LENGTH = 1_024
MAX_RAIL_LAYER_N = 32
# One global budget shared by three exact objective proofs and one canonical
# fixed-search witness.  Callers may request a smaller/larger bounded budget.
DEFAULT_RAIL_PLACEMENT_DETERMINISTIC_TIME = 60.0
MAX_RAIL_DETERMINISTIC_TIME = 60.0
MAX_COORD_ABS_NM = 10_000_000
MAX_ACTIVE_OVERLAP_NM2 = (2 * MAX_COORD_ABS_NM) ** 2

_SAT_PROOF_SCOPE = (
    "Exact replay of the bounded rail-placement candidate pool and its "
    "lexicographic CP-SAT optimum; this is a placement proof, not an LVS claim."
)
_UNSAT_PROOF_SCOPE = (
    "Sufficient CP-SAT assumption core for this bounded rail-placement "
    "candidate pool; it is not a minimal IIS and makes no LVS claim."
)
_UNKNOWN_PROOF_SCOPE = (
    "No placement proof was produced; UNKNOWN carries no partial plan or "
    "infeasibility certificate and makes no LVS claim."
)
_RAIL_REPLAY_ISSUER_AUTHORITY = object()
_RAIL_AXIS_RECEIPT_ISSUER_AUTHORITY = object()
_RAIL_AXIS_ISSUER_AUTHORITY = object()
_RAIL_REPLAY_CAPABILITIES: dict[int, tuple[object, tuple[object, ...]]] = {}
_RAIL_AXIS_RECEIPT_CAPABILITIES: dict[
    int, tuple[object, tuple[object, ...]]
] = {}
_RAIL_AXIS_CAPABILITIES: dict[int, tuple[object, tuple[object, ...]]] = {}


class RailPlacementError(ValueError):
    """Base class for invalid or stale rail-placement artifacts."""


class RailPlacementLimitError(RailPlacementError):
    """A bounded rail-placement input or derived artifact exceeded its cap."""


@dataclass(frozen=True)
class _ActiveUnionStripeProfile:
    """Exact active-union area integral along one rail cross axis."""

    cross_events_nm: tuple[int, ...]
    prefix_areas_nm2: tuple[int, ...]
    run_lengths_nm: tuple[int, ...]


@dataclass(frozen=True)
class _ConductorForbiddenStartIndex:
    """Merged inclusive rail-start intervals forbidden by existing metal."""

    lower_bounds_nm: tuple[int, ...]
    upper_bounds_nm: tuple[int, ...]


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _bounded_iterable(
    values: object,
    *,
    limit: int,
    label: str,
) -> tuple[object, ...]:
    if type(limit) is not int or limit < 0:
        raise TypeError(f"{label} limit must be an exact non-negative integer")
    try:
        count = len(values)  # type: ignore[arg-type]
    except TypeError:
        count = None
    if count is not None and count > limit:
        raise RailPlacementLimitError(f"{label} count {count} exceeds {limit}")
    try:
        result = tuple(islice(iter(values), limit + 1))  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be iterable") from exc
    if len(result) > limit:
        raise RailPlacementLimitError(f"{label} count exceeds {limit}")
    return result


def _exact_tuple(values: object, *, limit: int, label: str) -> tuple[object, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    return _bounded_iterable(values, limit=limit, label=label)


def _charge_enumeration_work(current: int, amount: int, label: str) -> int:
    updated = current + amount
    if updated > MAX_RAIL_ENUMERATION_WORK:
        raise RailPlacementLimitError(
            f"rail enumeration work budget {MAX_RAIL_ENUMERATION_WORK} exceeded "
            f"while {label}"
        )
    return updated


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise RailPlacementError(f"{label} must be a non-empty exact string")
    if len(value) > MAX_RAIL_STRING_LENGTH:
        raise RailPlacementLimitError(
            f"{label} length {len(value)} exceeds {MAX_RAIL_STRING_LENGTH}"
        )
    return value


def _digest(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RailPlacementError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _bbox(value: object, label: str) -> BBoxNm:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{label} must be an exact four-integer tuple")
    if not all(_is_exact_int(item) for item in value):
        raise RailPlacementError(f"{label} must contain exact integer nanometres")
    x0, y0, x1, y1 = value
    if not all(-MAX_COORD_ABS_NM <= item <= MAX_COORD_ABS_NM for item in value):
        raise RailPlacementLimitError(
            f"{label} coordinate magnitude exceeds {MAX_COORD_ABS_NM} nm"
        )
    if x0 >= x1 or y0 >= y1:
        raise RailPlacementError(f"{label} must have positive area")
    return value


def _point(value: object, label: str) -> PointNm:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact two-integer tuple")
    if not all(_is_exact_int(item) for item in value):
        raise RailPlacementError(f"{label} must contain exact integer nanometres")
    if not all(-MAX_COORD_ABS_NM <= item <= MAX_COORD_ABS_NM for item in value):
        raise RailPlacementLimitError(
            f"{label} coordinate magnitude exceeds {MAX_COORD_ABS_NM} nm"
        )
    return value


def _layer_key(value: object, label: str) -> LayerKey:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact two-integer tuple")
    if not all(_is_exact_int(item) and 0 <= item <= 65_535 for item in value):
        raise RailPlacementError(
            f"{label} must contain bounded non-negative exact integers"
        )
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _axis_declaration_payload(
    *,
    actual_axis: str,
    axis_source: str,
    declaration_ref: str,
    source_receipt_digest: Optional[str],
) -> dict[str, object]:
    return {
        "actual_axis": actual_axis,
        "axis_source": axis_source,
        "declaration_ref": declaration_ref,
        "schema": "clay.rail-axis-declaration.v2",
        "source_receipt_digest": source_receipt_digest,
    }


def _axis_declaration_digest(
    *,
    actual_axis: str,
    axis_source: str,
    declaration_ref: str,
    source_receipt_digest: Optional[str] = None,
) -> str:
    return _sha256(
        _axis_declaration_payload(
            actual_axis=actual_axis,
            axis_source=axis_source,
            declaration_ref=declaration_ref,
            source_receipt_digest=source_receipt_digest,
        )
    )


def _axis_relation(actual_axis: str, current_preferred_axis: Optional[str]) -> str:
    if current_preferred_axis is None:
        return "undeclared"
    return "preferred" if actual_axis == current_preferred_axis else "nonpreferred"


def _canonical_parsed_rail_ast_value(value: object) -> object:
    if is_dataclass(value):
        return {
            item.name: _canonical_parsed_rail_ast_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonical_parsed_rail_ast_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_parsed_rail_ast_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"rail source AST contains non-canonical value {value!r}")


def _canonical_parsed_rail_declaration(declaration: object) -> dict[str, object]:
    payload = _canonical_parsed_rail_ast_value(declaration)
    if not isinstance(payload, dict):
        raise TypeError("parsed rail declaration must lower to a canonical object")
    span = payload.get("source_span")
    if isinstance(span, dict):
        payload["source_span"] = {
            key: span[key]
            for key in ("line", "column", "end_line", "end_column")
            if key in span
        }
    payload["schema"] = "clay.parsed-layout-intent-declaration.v1"
    return payload


def _read_authoritative_rail_source(
    *,
    source_locator: str,
    cell_name: str,
) -> tuple[str, str, str, tuple[str, ...]]:
    """Reread and canonicalize the exact parsed source behind a receipt."""

    _nonempty_string(source_locator, "rail source locator")
    _nonempty_string(cell_name, "rail source/AST cell name")
    try:
        locator = Path(source_locator).resolve(strict=True)
        source_bytes_before = locator.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise RailPlacementError(
            "rail source locator cannot be reread during verification"
        ) from exc
    try:
        from clay.ir import CellDef, ParsedLayoutIntentDecl
        from clay.parser import parse_clay_file

        reparsed = parse_clay_file(locator)
    except Exception as exc:
        raise RailPlacementError(
            "rail source locator cannot be parsed as authoritative source"
        ) from exc
    try:
        source_bytes_after = locator.read_bytes()
    except OSError as exc:
        raise RailPlacementError(
            "rail source locator cannot be reread after authoritative parse"
        ) from exc
    if source_bytes_before != source_bytes_after:
        raise RailPlacementError("rail source bytes changed during authoritative parse")
    matches = tuple(
        item
        for item in reparsed.cell_defs
        if type(item) is CellDef
        and str(getattr(item, "local_name", "") or "") == cell_name
    )
    if len(matches) != 1:
        raise RailPlacementError(
            "rail source bytes do not resolve one authoritative cell"
        )
    declarations = tuple(getattr(matches[0], "layout_intents", ()) or ())
    if not declarations or not all(
        type(item) is ParsedLayoutIntentDecl for item in declarations
    ):
        raise RailPlacementError(
            "authoritative rail source has no exact parsed layout-intent AST"
        )
    declaration_json = tuple(
        _canonical_json(_canonical_parsed_rail_declaration(item))
        for item in declarations
    )
    source_bytes_digest = hashlib.sha256(source_bytes_after).hexdigest()
    source_payload = {
        "canonical_source_bytes_digest": source_bytes_digest,
        "cell_name": cell_name,
        "layout_intents": [json.loads(item) for item in declaration_json],
        "schema": "clay.parsed-layout-intent-source.v2",
    }
    return (
        str(locator),
        source_bytes_digest,
        _canonical_json(source_payload),
        declaration_json,
    )


@dataclass(frozen=True, init=False)
class RailAxisDeclarationReceipt:
    """Module-issued proof that one declaration is a member of parsed source AST."""

    cell_name: str
    declaration_ref: str
    source_locator: str
    source_bytes_digest: str
    canonical_source_ast_json: str
    canonical_declaration_ast_json: str
    source_ast_digest: str
    declaration_ast_digest: str
    _issuer_authority: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "RailAxisDeclarationReceipt is a module-issued source/AST capability"
        )

    def __post_init__(self) -> None:
        _verify_rail_axis_declaration_receipt(self)

    def sha256(self) -> str:
        return _sha256(_axis_source_receipt_payload(self))


def _issue_rail_axis_declaration_receipt(
    *,
    cell_name: str,
    declaration_ref: str,
    source_locator: str,
    source_bytes_digest: str,
    canonical_source_ast_json: str,
    canonical_declaration_ast_json: str,
) -> RailAxisDeclarationReceipt:
    (
        authoritative_locator,
        authoritative_source_bytes_digest,
        authoritative_source_ast_json,
        authoritative_declarations,
    ) = _read_authoritative_rail_source(
        source_locator=source_locator,
        cell_name=cell_name,
    )
    if source_bytes_digest != authoritative_source_bytes_digest:
        raise RailPlacementError(
            "rail source-bytes digest differs from authoritative source"
        )
    if canonical_source_ast_json != authoritative_source_ast_json:
        raise RailPlacementError(
            "caller rail source AST differs from authoritative parsed source"
        )
    if canonical_declaration_ast_json not in authoritative_declarations:
        raise RailPlacementError(
            "caller rail declaration is absent from authoritative parsed source"
        )
    receipt = object.__new__(RailAxisDeclarationReceipt)
    values = {
        "cell_name": cell_name,
        "declaration_ref": declaration_ref,
        "source_locator": authoritative_locator,
        "source_bytes_digest": authoritative_source_bytes_digest,
        "canonical_source_ast_json": authoritative_source_ast_json,
        "canonical_declaration_ast_json": canonical_declaration_ast_json,
        "source_ast_digest": hashlib.sha256(
            authoritative_source_ast_json.encode("utf-8")
        ).hexdigest(),
        "declaration_ast_digest": hashlib.sha256(
            canonical_declaration_ast_json.encode("utf-8")
        ).hexdigest(),
        "_issuer_authority": _RAIL_AXIS_RECEIPT_ISSUER_AUTHORITY,
    }
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    _RAIL_AXIS_RECEIPT_CAPABILITIES[id(receipt)] = (
        receipt,
        _rail_axis_receipt_state(receipt),
    )
    try:
        receipt.__post_init__()
    except Exception:
        _RAIL_AXIS_RECEIPT_CAPABILITIES.pop(id(receipt), None)
        raise
    return receipt


def _rail_axis_receipt_state(
    receipt: RailAxisDeclarationReceipt,
) -> tuple[object, ...]:
    return (
        receipt.cell_name,
        receipt.declaration_ref,
        receipt.source_locator,
        receipt.source_bytes_digest,
        receipt.canonical_source_ast_json,
        receipt.canonical_declaration_ast_json,
        receipt.source_ast_digest,
        receipt.declaration_ast_digest,
        receipt._issuer_authority,
    )


def _verify_rail_axis_declaration_receipt(
    receipt: RailAxisDeclarationReceipt,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(receipt) is not RailAxisDeclarationReceipt:
        raise TypeError("explicit rail axis requires a typed source/AST receipt")
    if (
        getattr(receipt, "_issuer_authority", None)
        is not _RAIL_AXIS_RECEIPT_ISSUER_AUTHORITY
    ):
        raise TypeError("rail source/AST receipt must be module-issued")
    registered = _RAIL_AXIS_RECEIPT_CAPABILITIES.get(id(receipt))
    if registered is None or registered[0] is not receipt:
        raise TypeError(
            "rail source/AST receipt must be the exact module-issued object; "
            "copied capabilities are invalid"
        )
    if registered[1] != _rail_axis_receipt_state(receipt):
        raise TypeError("rail source/AST receipt was mutated after issuance")
    _nonempty_string(receipt.cell_name, "rail source/AST cell name")
    _nonempty_string(receipt.declaration_ref, "rail source/AST declaration ref")
    _nonempty_string(receipt.source_locator, "rail source locator")
    _digest(receipt.source_bytes_digest, "rail source-bytes digest")
    _digest(receipt.source_ast_digest, "rail source AST digest")
    _digest(receipt.declaration_ast_digest, "rail declaration AST digest")
    try:
        source = json.loads(receipt.canonical_source_ast_json)
        declaration = json.loads(receipt.canonical_declaration_ast_json)
    except (TypeError, ValueError) as exc:
        raise RailPlacementError("rail source/AST receipt is not canonical JSON") from exc
    if not isinstance(source, dict) or not isinstance(declaration, dict):
        raise RailPlacementError("rail source/AST receipt payload must be objects")
    if _canonical_json(source) != receipt.canonical_source_ast_json:
        raise RailPlacementError("rail source AST bytes are not canonical")
    if _canonical_json(declaration) != receipt.canonical_declaration_ast_json:
        raise RailPlacementError("rail declaration AST bytes are not canonical")
    if hashlib.sha256(receipt.canonical_source_ast_json.encode("utf-8")).hexdigest() != receipt.source_ast_digest:
        raise RailPlacementError("rail source AST digest does not replay")
    if hashlib.sha256(receipt.canonical_declaration_ast_json.encode("utf-8")).hexdigest() != receipt.declaration_ast_digest:
        raise RailPlacementError("rail declaration AST digest does not replay")
    if source.get("schema") != "clay.parsed-layout-intent-source.v2":
        raise RailPlacementError("rail source AST schema is unsupported")
    if source.get("cell_name") != receipt.cell_name:
        raise RailPlacementError("rail source AST cell binding does not replay")
    source_bytes_digest = source.get("canonical_source_bytes_digest")
    _digest(source_bytes_digest, "rail canonical source-bytes digest")
    if source_bytes_digest != receipt.source_bytes_digest:
        raise RailPlacementError("rail source AST names a different source digest")
    declarations = source.get("layout_intents")
    if not isinstance(declarations, list) or declaration not in declarations:
        raise RailPlacementError(
            "rail declaration is absent from the authoritative source AST"
        )
    (
        authoritative_locator,
        current_source_digest,
        authoritative_source_ast_json,
        authoritative_declarations,
    ) = _read_authoritative_rail_source(
        source_locator=receipt.source_locator,
        cell_name=receipt.cell_name,
    )
    if authoritative_locator != receipt.source_locator:
        raise RailPlacementError("rail source locator is not canonical")
    if current_source_digest != receipt.source_bytes_digest:
        raise RailPlacementError(
            "rail source bytes changed after authority issuance"
        )
    if authoritative_source_ast_json != receipt.canonical_source_ast_json:
        raise RailPlacementError(
            "rail source AST differs from current authoritative parse"
        )
    if receipt.canonical_declaration_ast_json not in authoritative_declarations:
        raise RailPlacementError(
            "rail declaration is absent from current authoritative parse"
        )
    return source, declaration


@dataclass(frozen=True, init=False)
class RailAxisAuthority:
    """Typed authority for the actual axis of one declared edge rail.

    ``actual_axis`` is the geometry contract.  ``current_preferred_axis`` is
    replayed adapter metadata, not an authorization by itself.  Only an exact
    ``explicit_intent`` declaration may therefore carry a ``nonpreferred``
    relation; a ``pdk_default`` axis must reproduce the current PDK decision.
    """

    actual_axis: str
    axis_source: str
    current_preferred_axis: Optional[str]
    relation: str
    declaration_ref: str
    declaration_digest: str
    source_receipt: Optional[RailAxisDeclarationReceipt]
    _issuer_authority: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("RailAxisAuthority is a module-issued capability")

    def __post_init__(self) -> None:
        _nonempty_string(self.actual_axis, "rail actual axis")
        _nonempty_string(self.axis_source, "rail axis source")
        _nonempty_string(self.relation, "rail axis relation")
        _nonempty_string(self.declaration_ref, "rail axis declaration ref")
        _digest(self.declaration_digest, "rail axis declaration digest")
        if self.actual_axis not in ("H", "V"):
            raise RailPlacementError("rail actual axis must be exact H or V")
        if self.axis_source not in ("explicit_intent", "pdk_default"):
            raise RailPlacementError(
                "rail axis source must be explicit_intent or pdk_default"
            )
        if self.current_preferred_axis is not None and (
            type(self.current_preferred_axis) is not str
            or self.current_preferred_axis not in ("H", "V")
        ):
            raise RailPlacementError(
                "rail current preferred axis must be exact H, V, or None"
            )
        expected_relation = _axis_relation(
            self.actual_axis, self.current_preferred_axis
        )
        if self.relation != expected_relation:
            raise RailPlacementError(
                "rail axis relation does not match actual/current preferred axes"
            )
        if (
            self.axis_source == "pdk_default"
            and self.current_preferred_axis is not None
            and self.actual_axis != self.current_preferred_axis
        ):
            raise RailPlacementError(
                "PDK default axis authority cannot declare a nonpreferred rail"
            )
        if getattr(self, "_issuer_authority", None) is not _RAIL_AXIS_ISSUER_AUTHORITY:
            raise TypeError("rail axis authority must be module-issued")
        if self.axis_source == "explicit_intent":
            _verify_rail_axis_declaration_receipt(self.source_receipt)
            if self.source_receipt.declaration_ref != self.declaration_ref:
                raise RailPlacementError(
                    "rail source/AST receipt declaration ref mismatch"
                )
        elif self.source_receipt is not None:
            raise RailPlacementError(
                "PDK-default rail authority cannot carry a source/AST receipt"
            )
        expected_digest = _axis_declaration_digest(
            actual_axis=self.actual_axis,
            axis_source=self.axis_source,
            declaration_ref=self.declaration_ref,
            source_receipt_digest=(
                None if self.source_receipt is None else self.source_receipt.sha256()
            ),
        )
        if self.declaration_digest != expected_digest:
            raise RailPlacementError(
                "rail axis declaration digest does not replay"
            )
        registered = _RAIL_AXIS_CAPABILITIES.get(id(self))
        if registered is None or registered[0] is not self:
            raise TypeError(
                "rail axis authority must be the exact module-issued object; "
                "copied capabilities are invalid"
            )
        if registered[1] != _rail_axis_authority_state(self):
            raise TypeError("rail axis authority was mutated after issuance")


def make_rail_axis_authority(
    *,
    actual_axis: str,
    axis_source: str,
    current_preferred_axis: Optional[str],
    declaration_ref: str,
) -> RailAxisAuthority:
    """Issue PDK-default authority; explicit intent requires parsed AST receipt."""

    if axis_source == "explicit_intent":
        raise TypeError(
            "explicit rail axis authority requires a route-cell module-issued "
            "source/AST receipt"
        )
    return _issue_rail_axis_authority(
        actual_axis=actual_axis,
        axis_source=axis_source,
        current_preferred_axis=current_preferred_axis,
        declaration_ref=declaration_ref,
        source_receipt=None,
    )


def _issue_rail_axis_authority(
    *,
    actual_axis: str,
    axis_source: str,
    current_preferred_axis: Optional[str],
    declaration_ref: str,
    source_receipt: Optional[RailAxisDeclarationReceipt],
) -> RailAxisAuthority:
    """Private issuer consumed by route-cell after exact parsed-AST matching."""

    relation = _axis_relation(actual_axis, current_preferred_axis)
    authority = object.__new__(RailAxisAuthority)
    receipt_digest = None if source_receipt is None else source_receipt.sha256()
    values = {
        "actual_axis": actual_axis,
        "axis_source": axis_source,
        "current_preferred_axis": current_preferred_axis,
        "relation": relation,
        "declaration_ref": declaration_ref,
        "declaration_digest": _axis_declaration_digest(
            actual_axis=actual_axis,
            axis_source=axis_source,
            declaration_ref=declaration_ref,
            source_receipt_digest=receipt_digest,
        ),
        "source_receipt": source_receipt,
        "_issuer_authority": _RAIL_AXIS_ISSUER_AUTHORITY,
    }
    for name, value in values.items():
        object.__setattr__(authority, name, value)
    _RAIL_AXIS_CAPABILITIES[id(authority)] = (
        authority,
        _rail_axis_authority_state(authority),
    )
    try:
        authority.__post_init__()
    except Exception:
        _RAIL_AXIS_CAPABILITIES.pop(id(authority), None)
        raise
    return authority


def _rail_axis_authority_state(
    authority: RailAxisAuthority,
) -> tuple[object, ...]:
    return (
        authority.actual_axis,
        authority.axis_source,
        authority.current_preferred_axis,
        authority.relation,
        authority.declaration_ref,
        authority.declaration_digest,
        authority.source_receipt,
        authority._issuer_authority,
    )


@dataclass(frozen=True)
class EdgeRailDemand:
    """One normalized declaration for an edge-oriented power rail."""

    demand_id: str
    declaration_ref: str
    net: str
    layer_n: int
    layer_key: LayerKey
    axis_authority: RailAxisAuthority
    edge: str
    width_nm: int
    width_source: str
    offset_nm: int = 0

    def __post_init__(self) -> None:
        _nonempty_string(self.demand_id, "rail demand_id")
        _nonempty_string(self.declaration_ref, "rail declaration_ref")
        _nonempty_string(self.net, "rail net")
        _layer_key(self.layer_key, "rail layer_key")
        _nonempty_string(self.width_source, "rail width_source")
        _nonempty_string(self.edge, "rail edge")
        if type(self.axis_authority) is not RailAxisAuthority:
            raise RailPlacementError(
                "rail demand requires exact RailAxisAuthority"
            )
        if self.axis_authority.declaration_ref != self.declaration_ref:
            raise RailPlacementError(
                "rail demand declaration ref differs from axis authority"
            )
        if not _is_exact_int(self.layer_n) or not 1 <= self.layer_n <= MAX_RAIL_LAYER_N:
            raise RailPlacementError(
                f"rail layer_n must be an exact integer in [1,{MAX_RAIL_LAYER_N}]"
            )
        allowed_edges = (
            ("top", "bottom")
            if self.actual_axis == "H"
            else (
                "left",
                "right",
            )
        )
        if self.edge not in allowed_edges:
            raise RailPlacementError(
                f"rail edge {self.edge!r} is incompatible with "
                f"actual_axis {self.actual_axis!r}"
            )
        if not _is_exact_int(self.width_nm) or self.width_nm <= 0:
            raise RailPlacementError(
                "rail width_nm must be a positive exact integer nanometre value"
            )
        if not _is_exact_int(self.offset_nm) or self.offset_nm < 0:
            raise RailPlacementError("rail offset_nm must be a nonnegative integer")
        if self.width_nm > 2 * MAX_COORD_ABS_NM:
            raise RailPlacementLimitError("rail width_nm exceeds coordinate bounds")

    @property
    def actual_axis(self) -> str:
        return self.axis_authority.actual_axis

    @property
    def preferred_axis(self) -> str:
        """Compatibility projection: geometry has always consumed this as actual."""

        return self.actual_axis


@dataclass(frozen=True)
class RailProjectionFact:
    """One exact projection supplied by placement or existing artwork."""

    bbox_nm: BBoxNm
    layer_key: LayerKey
    owner_net: str | None
    kind: str

    def __post_init__(self) -> None:
        _bbox(self.bbox_nm, "rail projection bbox_nm")
        _layer_key(self.layer_key, "rail projection layer_key")
        if self.owner_net is not None and type(self.owner_net) is not str:
            raise RailPlacementError(
                "rail projection owner_net must be an exact string or explicit None"
            )
        if self.owner_net is not None and len(self.owner_net) > MAX_RAIL_STRING_LENGTH:
            raise RailPlacementLimitError("rail projection owner_net is too long")
        _nonempty_string(self.kind, "rail projection kind")


@dataclass(frozen=True)
class RailTerminalFact:
    """One exact electrical access fact supplied by topology lowering."""

    net: str
    xy_nm: PointNm
    layer_key: LayerKey
    provenance: str

    def __post_init__(self) -> None:
        _nonempty_string(self.net, "rail terminal net")
        _point(self.xy_nm, "rail terminal xy_nm")
        _layer_key(self.layer_key, "rail terminal layer_key")
        _nonempty_string(self.provenance, "rail terminal provenance")


@dataclass(frozen=True)
class RailPlacementScene:
    """Closed, content-addressed input scene for candidate enumeration."""

    placement_provenance_digest: str
    pdk_provenance_digest: str
    declaration_provenance_digest: str
    cell_bbox_nm: BBoxNm
    active_projection_facts: tuple[RailProjectionFact, ...] = ()
    conductor_facts: tuple[RailProjectionFact, ...] = ()
    terminal_facts: tuple[RailTerminalFact, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.placement_provenance_digest, "placement provenance")
        _digest(self.pdk_provenance_digest, "PDK provenance")
        _digest(self.declaration_provenance_digest, "declaration provenance")
        _bbox(self.cell_bbox_nm, "rail scene cell_bbox_nm")
        active = _exact_tuple(
            self.active_projection_facts,
            limit=MAX_RAIL_SCENE_FACTS,
            label="active projection facts",
        )
        conductors = _exact_tuple(
            self.conductor_facts,
            limit=MAX_RAIL_SCENE_FACTS,
            label="conductor facts",
        )
        terminals = _exact_tuple(
            self.terminal_facts,
            limit=MAX_RAIL_SCENE_FACTS,
            label="rail terminal facts",
        )
        if not all(type(item) is RailProjectionFact for item in active):
            raise RailPlacementError("active projection facts contain invalid values")
        if not all(type(item) is RailProjectionFact for item in conductors):
            raise RailPlacementError("conductor facts contain invalid values")
        if not all(type(item) is RailTerminalFact for item in terminals):
            raise RailPlacementError("rail terminal facts contain invalid values")
        if any(fact.owner_net == "" for fact in conductors):
            raise RailPlacementError("conductor facts require a non-empty owner_net or explicit None")
        object.__setattr__(
            self, "active_projection_facts", tuple(sorted(active, key=_fact_sort_key))
        )
        object.__setattr__(
            self, "conductor_facts", tuple(sorted(conductors, key=_fact_sort_key))
        )
        object.__setattr__(
            self,
            "terminal_facts",
            tuple(
                sorted(
                    terminals,
                    key=lambda item: (
                        item.net,
                        item.xy_nm,
                        item.layer_key,
                        item.provenance,
                    ),
                )
            ),
        )

    def canonical_json(self) -> str:
        return _canonical_json(_scene_payload(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RailPlacementCandidate:
    """One fully measured rail rectangle at an exact scene event coordinate."""

    candidate_id: str
    demand_id: str
    bbox_nm: BBoxNm
    access_xy_nm: PointNm
    placement_kind: str
    active_overlap_nm2: int
    exterior_extension_nm: int
    edge_distance_nm: int
    declared_edge_eligible: bool

    def __post_init__(self) -> None:
        _nonempty_string(self.candidate_id, "rail candidate_id")
        _nonempty_string(self.demand_id, "rail candidate demand_id")
        bbox = _bbox(self.bbox_nm, "rail candidate bbox_nm")
        point = _point(self.access_xy_nm, "rail candidate access_xy_nm")
        if not (bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]):
            raise RailPlacementError("rail candidate access point must lie on its bbox")
        _nonempty_string(self.placement_kind, "rail candidate placement_kind")
        if self.placement_kind not in ("interior", "over_cell", "exterior"):
            raise RailPlacementError("unknown rail candidate placement_kind")
        if type(self.declared_edge_eligible) is not bool:
            raise RailPlacementError(
                "rail candidate declared_edge_eligible must be an exact bool"
            )
        for name, upper in (
            ("active_overlap_nm2", MAX_ACTIVE_OVERLAP_NM2),
            ("exterior_extension_nm", 2 * MAX_COORD_ABS_NM),
            ("edge_distance_nm", 2 * MAX_COORD_ABS_NM),
        ):
            value = getattr(self, name)
            if not _is_exact_int(value) or not 0 <= value <= upper:
                raise RailPlacementError(
                    f"rail candidate {name} must be a bounded non-negative exact integer"
                )


@dataclass(frozen=True)
class RailCandidateConflict:
    """A hard incompatibility between two candidates from different demands."""

    left_candidate_id: str
    right_candidate_id: str
    reason: str

    def __post_init__(self) -> None:
        _nonempty_string(self.left_candidate_id, "left rail candidate id")
        _nonempty_string(self.right_candidate_id, "right rail candidate id")
        _nonempty_string(self.reason, "rail conflict reason")
        if self.left_candidate_id >= self.right_candidate_id:
            raise RailPlacementError(
                "rail conflict candidate ids must be distinct and canonical"
            )


@dataclass(frozen=True)
class RailDemandPairSpacingContract:
    """Compact exact spacing rule for one same-layer demand pair."""

    left_demand_id: str
    right_demand_id: str
    layer_key: LayerKey
    preferred_axis: str
    relation: str
    minimum_gap_nm: int

    def __post_init__(self) -> None:
        _nonempty_string(self.left_demand_id, "left rail demand id")
        _nonempty_string(self.right_demand_id, "right rail demand id")
        if self.left_demand_id >= self.right_demand_id:
            raise RailPlacementError(
                "rail spacing-contract demand ids must be distinct and canonical"
            )
        _layer_key(self.layer_key, "rail spacing-contract layer_key")
        if type(self.preferred_axis) is not str or self.preferred_axis not in (
            "H",
            "V",
        ):
            raise RailPlacementError(
                "rail spacing-contract preferred_axis must be exact H or V"
            )
        if type(self.relation) is not str or self.relation not in (
            "different_net",
            "same_net",
        ):
            raise RailPlacementError("unknown rail spacing-contract relation")
        if (
            not _is_exact_int(self.minimum_gap_nm)
            or not 0 <= self.minimum_gap_nm <= 2 * MAX_COORD_ABS_NM
        ):
            raise RailPlacementError(
                "rail spacing-contract minimum_gap_nm must be bounded and non-negative"
            )
        if self.relation == "different_net" and self.minimum_gap_nm < 1:
            raise RailPlacementError(
                "different-net rail spacing contract requires at least 1 nm"
            )


@dataclass(frozen=True)
class RailPlacementCandidatePool:
    """Canonical simultaneous-choice problem bound to one exact input scene."""

    placement_provenance_digest: str
    pdk_provenance_digest: str
    declaration_provenance_digest: str
    rail_placement_scene_hash: str
    demands: tuple[EdgeRailDemand, ...]
    candidates: tuple[RailPlacementCandidate, ...]
    conflicts: tuple[RailCandidateConflict, ...] = ()
    spacing_contracts: tuple[RailDemandPairSpacingContract, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.placement_provenance_digest, "pool placement provenance")
        _digest(self.pdk_provenance_digest, "pool PDK provenance")
        _digest(self.declaration_provenance_digest, "pool declaration provenance")
        _digest(self.rail_placement_scene_hash, "pool rail scene hash")
        demands = _exact_tuple(
            self.demands, limit=MAX_RAIL_DEMANDS, label="rail pool demands"
        )
        candidates = _exact_tuple(
            self.candidates, limit=MAX_RAIL_CANDIDATES, label="rail pool candidates"
        )
        conflicts = _exact_tuple(
            self.conflicts, limit=MAX_RAIL_CONFLICTS, label="rail pool conflicts"
        )
        spacing_contracts = _exact_tuple(
            self.spacing_contracts,
            limit=MAX_RAIL_PAIR_CONTRACTS,
            label="rail pool spacing contracts",
        )
        if not all(type(item) is EdgeRailDemand for item in demands):
            raise RailPlacementError("rail pool demands contain invalid values")
        if not all(type(item) is RailPlacementCandidate for item in candidates):
            raise RailPlacementError("rail pool candidates contain invalid values")
        if not all(type(item) is RailCandidateConflict for item in conflicts):
            raise RailPlacementError("rail pool conflicts contain invalid values")
        if not all(
            type(item) is RailDemandPairSpacingContract for item in spacing_contracts
        ):
            raise RailPlacementError(
                "rail pool spacing contracts contain invalid values"
            )
        ordered_demands = tuple(sorted(demands, key=lambda item: item.demand_id))
        demand_ids = tuple(item.demand_id for item in ordered_demands)
        if len(set(demand_ids)) != len(demand_ids):
            raise RailPlacementError("rail pool contains duplicate demand ids")
        declaration_refs = tuple(item.declaration_ref for item in ordered_demands)
        if len(set(declaration_refs)) != len(declaration_refs):
            raise RailPlacementError("rail pool contains duplicate declaration refs")
        ordered_candidates = tuple(
            sorted(candidates, key=lambda item: item.candidate_id)
        )
        candidate_ids = tuple(item.candidate_id for item in ordered_candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise RailPlacementError("rail pool contains duplicate candidate ids")
        demand_id_set = set(demand_ids)
        if any(item.demand_id not in demand_id_set for item in ordered_candidates):
            raise RailPlacementError("rail candidate references an unknown demand")
        candidates_by_demand: dict[str, list[RailPlacementCandidate]] = {
            demand_id: [] for demand_id in demand_ids
        }
        for candidate in ordered_candidates:
            candidates_by_demand[candidate.demand_id].append(candidate)
        candidate_id_set = set(candidate_ids)
        ordered_conflicts = tuple(
            sorted(
                conflicts,
                key=lambda item: (
                    item.left_candidate_id,
                    item.right_candidate_id,
                    item.reason,
                ),
            )
        )
        conflict_pairs: set[tuple[str, str]] = set()
        for conflict in ordered_conflicts:
            if (
                conflict.left_candidate_id not in candidate_id_set
                or conflict.right_candidate_id not in candidate_id_set
            ):
                raise RailPlacementError(
                    "rail conflict references an unknown candidate"
                )
            pair = (conflict.left_candidate_id, conflict.right_candidate_id)
            if pair in conflict_pairs:
                raise RailPlacementError("rail pool contains duplicate conflict pairs")
            conflict_pairs.add(pair)
        demands_by_id = {item.demand_id: item for item in ordered_demands}
        ordered_contracts = tuple(
            sorted(
                spacing_contracts,
                key=lambda item: (item.left_demand_id, item.right_demand_id),
            )
        )
        contract_pairs: set[tuple[str, str]] = set()
        for contract in ordered_contracts:
            pair = (contract.left_demand_id, contract.right_demand_id)
            if pair in contract_pairs:
                raise RailPlacementError(
                    "rail pool contains duplicate demand-pair spacing contracts"
                )
            contract_pairs.add(pair)
            left = demands_by_id.get(contract.left_demand_id)
            right = demands_by_id.get(contract.right_demand_id)
            if left is None or right is None:
                raise RailPlacementError(
                    "rail spacing contract references an unknown demand"
                )
            if (
                left.layer_key != right.layer_key
                or contract.layer_key != left.layer_key
                or left.preferred_axis != right.preferred_axis
                or contract.preferred_axis != left.preferred_axis
            ):
                raise RailPlacementError(
                    "rail spacing contract does not match demand layer and axis"
                )
            expected_relation = "same_net" if left.net == right.net else "different_net"
            if contract.relation != expected_relation:
                raise RailPlacementError(
                    "rail spacing-contract relation does not match demand nets"
                )
            left_candidates = candidates_by_demand[contract.left_demand_id]
            right_candidates = candidates_by_demand[contract.right_demand_id]
            if left_candidates and right_candidates:
                run_intervals = {
                    (
                        (candidate.bbox_nm[0], candidate.bbox_nm[2])
                        if contract.preferred_axis == "H"
                        else (candidate.bbox_nm[1], candidate.bbox_nm[3])
                    )
                    for candidate in (*left_candidates, *right_candidates)
                }
                if len(run_intervals) != 1:
                    raise RailPlacementError(
                        "rail spacing-contract candidates must share one exact "
                        "common run-axis interval"
                    )
        object.__setattr__(self, "demands", ordered_demands)
        object.__setattr__(self, "candidates", ordered_candidates)
        object.__setattr__(self, "conflicts", ordered_conflicts)
        object.__setattr__(self, "spacing_contracts", ordered_contracts)

    def candidates_for(self, demand_id: str) -> tuple[RailPlacementCandidate, ...]:
        return tuple(item for item in self.candidates if item.demand_id == demand_id)

    def canonical_json(self) -> str:
        return _canonical_json(_pool_payload(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SelectedRailPlacement:
    """A complete candidate snapshot, not merely an unbound candidate id."""

    demand_id: str
    candidate_id: str
    bbox_nm: BBoxNm
    access_xy_nm: PointNm
    placement_kind: str
    active_overlap_nm2: int
    exterior_extension_nm: int
    edge_distance_nm: int
    declared_edge_eligible: bool
    axis_authority: RailAxisAuthority

    def __post_init__(self) -> None:
        # Reuse candidate validation so a selected snapshot cannot weaken types.
        RailPlacementCandidate(
            candidate_id=self.candidate_id,
            demand_id=self.demand_id,
            bbox_nm=self.bbox_nm,
            access_xy_nm=self.access_xy_nm,
            placement_kind=self.placement_kind,
            active_overlap_nm2=self.active_overlap_nm2,
            exterior_extension_nm=self.exterior_extension_nm,
            edge_distance_nm=self.edge_distance_nm,
            declared_edge_eligible=self.declared_edge_eligible,
        )
        if type(self.axis_authority) is not RailAxisAuthority:
            raise RailPlacementError(
                "selected rail requires exact RailAxisAuthority"
            )

    @classmethod
    def from_candidate(
        cls,
        candidate: RailPlacementCandidate,
        axis_authority: RailAxisAuthority,
    ) -> "SelectedRailPlacement":
        if type(candidate) is not RailPlacementCandidate:
            raise TypeError("candidate must be RailPlacementCandidate")
        if type(axis_authority) is not RailAxisAuthority:
            raise TypeError("axis_authority must be RailAxisAuthority")
        return cls(
            demand_id=candidate.demand_id,
            candidate_id=candidate.candidate_id,
            bbox_nm=candidate.bbox_nm,
            access_xy_nm=candidate.access_xy_nm,
            placement_kind=candidate.placement_kind,
            active_overlap_nm2=candidate.active_overlap_nm2,
            exterior_extension_nm=candidate.exterior_extension_nm,
            edge_distance_nm=candidate.edge_distance_nm,
            declared_edge_eligible=candidate.declared_edge_eligible,
            axis_authority=axis_authority,
        )


@dataclass(frozen=True)
class RailPlacementPlan:
    """Content-addressed SAT artifact for one candidate pool."""

    candidate_pool_hash: str
    selected: tuple[SelectedRailPlacement, ...]
    objective_values: tuple[int, int, int, int]
    proof_scope: str = _SAT_PROOF_SCOPE

    def __post_init__(self) -> None:
        _digest(self.candidate_pool_hash, "rail plan candidate pool hash")
        selected = _exact_tuple(
            self.selected, limit=MAX_RAIL_DEMANDS, label="selected rail placements"
        )
        if not all(type(item) is SelectedRailPlacement for item in selected):
            raise RailPlacementError("selected rail placements contain invalid values")
        demand_ids = tuple(item.demand_id for item in selected)
        if len(set(demand_ids)) != len(demand_ids):
            raise RailPlacementError("rail plan selects a demand more than once")
        if type(self.objective_values) is not tuple or len(self.objective_values) != 4:
            raise TypeError(
                "rail plan objective_values must be an exact four-int tuple"
            )
        if not all(
            _is_exact_int(item) and 0 <= item <= 2**63 - 1
            for item in self.objective_values
        ):
            raise RailPlacementError(
                "rail plan objective values must be non-negative exact integers"
            )
        _nonempty_string(self.proof_scope, "rail plan proof_scope")
        object.__setattr__(
            self, "selected", tuple(sorted(selected, key=lambda item: item.demand_id))
        )

    def canonical_json(self) -> str:
        return _canonical_json(_plan_payload(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class RailPlacementOptimalityReplay:
    """Receipt for one completed independent optimum replay.

    The receipt is an internal transport artifact.  It lets downstream canonical
    consumers bind to the exact pool and plan that were already replayed without
    silently turning the public deep verifier into a hash-only check.
    """

    candidate_pool_hash: str
    rail_plan_hash: str
    proof_scope: str
    axis_authorities: tuple[RailAxisAuthority, ...]
    _issuer_authority: object = field(repr=False, compare=False, init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("RailPlacementOptimalityReplay is a module-issued capability")


def _issue_rail_placement_optimality_replay(
    *,
    candidate_pool_hash: str,
    rail_plan_hash: str,
    proof_scope: str,
    axis_authorities: tuple[RailAxisAuthority, ...],
) -> RailPlacementOptimalityReplay:
    """Issue one process-local capability after the caller completes deep replay."""

    _digest(candidate_pool_hash, "rail replay receipt pool hash")
    _digest(rail_plan_hash, "rail replay receipt plan hash")
    _nonempty_string(proof_scope, "rail replay receipt proof_scope")
    values = _exact_tuple(
        axis_authorities,
        limit=MAX_RAIL_DEMANDS,
        label="rail replay axis authorities",
    )
    if not all(type(item) is RailAxisAuthority for item in values):
        raise RailPlacementError(
            "rail replay axis authorities must be exact RailAxisAuthority values"
        )
    receipt = object.__new__(RailPlacementOptimalityReplay)
    object.__setattr__(receipt, "candidate_pool_hash", candidate_pool_hash)
    object.__setattr__(receipt, "rail_plan_hash", rail_plan_hash)
    object.__setattr__(receipt, "proof_scope", proof_scope)
    object.__setattr__(receipt, "axis_authorities", values)
    object.__setattr__(receipt, "_issuer_authority", _RAIL_REPLAY_ISSUER_AUTHORITY)
    _RAIL_REPLAY_CAPABILITIES[id(receipt)] = (
        receipt,
        _rail_replay_state(receipt),
    )
    return receipt


def _rail_replay_state(
    receipt: RailPlacementOptimalityReplay,
) -> tuple[object, ...]:
    return (
        receipt.candidate_pool_hash,
        receipt.rail_plan_hash,
        receipt.proof_scope,
        receipt.axis_authorities,
        receipt._issuer_authority,
    )


@dataclass(frozen=True)
class RailPlacementInfeasibilityCertificate:
    """Replayable sufficient assumption core for the bounded CP model."""

    candidate_pool_hash: str
    assumption_core_ids: tuple[str, ...]
    proof_scope: str = _UNSAT_PROOF_SCOPE

    def __post_init__(self) -> None:
        _digest(self.candidate_pool_hash, "rail infeasibility pool hash")
        core = _exact_tuple(
            self.assumption_core_ids,
            limit=MAX_RAIL_DEMANDS,
            label="rail infeasibility assumption core",
        )
        if not core or not all(type(item) is str and item for item in core):
            raise RailPlacementError("rail infeasibility core must contain ids")
        for item in core:
            _nonempty_string(item, "rail infeasibility core id")
        if len(set(core)) != len(core):
            raise RailPlacementError("rail infeasibility core ids must be unique")
        _nonempty_string(self.proof_scope, "rail infeasibility proof_scope")
        object.__setattr__(self, "assumption_core_ids", tuple(sorted(core)))

    def canonical_json(self) -> str:
        return _canonical_json(
            {
                "assumption_core_ids": list(self.assumption_core_ids),
                "candidate_pool_hash": self.candidate_pool_hash,
                "proof_scope": self.proof_scope,
                "schema": "clay.rail-placement-infeasibility.v1",
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RailPlacementSelectionResult:
    """Total SAT/INFEASIBLE/UNKNOWN result with no partial UNKNOWN artifact."""

    status: str
    solver_status: str
    candidate_pool_hash: str
    plan: Optional[RailPlacementPlan] = None
    certificate: Optional[RailPlacementInfeasibilityCertificate] = None
    proof_scope: str = ""

    def __post_init__(self) -> None:
        _nonempty_string(self.status, "rail result status")
        if self.status not in ("SAT", "INFEASIBLE", "UNKNOWN"):
            raise RailPlacementError("unknown rail placement result status")
        _nonempty_string(self.solver_status, "rail solver_status")
        _digest(self.candidate_pool_hash, "rail result candidate pool hash")
        if self.status == "SAT":
            if type(self.plan) is not RailPlacementPlan or self.certificate is not None:
                raise RailPlacementError("SAT requires exactly one plan")
            if self.plan.candidate_pool_hash != self.candidate_pool_hash:
                raise RailPlacementError("SAT plan and result pool hashes differ")
            expected_scope = self.plan.proof_scope
        elif self.status == "INFEASIBLE":
            if (
                self.plan is not None
                or type(self.certificate) is not RailPlacementInfeasibilityCertificate
            ):
                raise RailPlacementError("INFEASIBLE requires exactly one certificate")
            if self.certificate.candidate_pool_hash != self.candidate_pool_hash:
                raise RailPlacementError(
                    "infeasibility certificate and result pool hashes differ"
                )
            expected_scope = self.certificate.proof_scope
        else:
            if self.plan is not None or self.certificate is not None:
                raise RailPlacementError("UNKNOWN may not carry a partial artifact")
            expected_scope = _UNKNOWN_PROOF_SCOPE
        if self.proof_scope and self.proof_scope != expected_scope:
            raise RailPlacementError(
                "rail result proof_scope does not match its status"
            )
        object.__setattr__(self, "proof_scope", expected_scope)

    def canonical_json(self) -> str:
        return _canonical_json(
            {
                "candidate_pool_hash": self.candidate_pool_hash,
                "certificate_hash": (
                    None if self.certificate is None else self.certificate.sha256()
                ),
                "plan_hash": None if self.plan is None else self.plan.sha256(),
                "proof_scope": self.proof_scope,
                "schema": "clay.rail-placement-result.v1",
                "solver_status": self.solver_status,
                "status": self.status,
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MaterializedRailFact:
    """One exact rail emitted from a selected placement candidate."""

    content_id: str
    demand_id: str
    net: str
    layer_key: LayerKey
    bbox_nm: BBoxNm
    access_xy_nm: PointNm
    axis_authority: RailAxisAuthority

    def __post_init__(self) -> None:
        _nonempty_string(self.content_id, "materialized rail content_id")
        _nonempty_string(self.demand_id, "materialized rail demand_id")
        _nonempty_string(self.net, "materialized rail net")
        _layer_key(self.layer_key, "materialized rail layer_key")
        if type(self.axis_authority) is not RailAxisAuthority:
            raise RailPlacementError(
                "materialized rail requires exact RailAxisAuthority"
            )
        bbox = _bbox(self.bbox_nm, "materialized rail bbox_nm")
        point = _point(self.access_xy_nm, "materialized rail access_xy_nm")
        if not (bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]):
            raise RailPlacementError("materialized rail access must lie on its bbox")


@dataclass(frozen=True)
class MaterializedRailPlanCertificate:
    """Replay binding from selected rails to the exact final routing world.

    This certificate proves identity of emitted rail artwork and access pins.  It
    deliberately makes no LVS-connectivity claim.
    """

    candidate_pool_hash: str
    rail_plan_hash: str
    final_routing_world_hash: str
    materialized_facts: tuple[MaterializedRailFact, ...]
    proof_scope: str = (
        "Exact selected rail bbox/access replay against final emitted geometry, "
        "pins, routing-world rail facts, and owned rail components; not LVS."
    )

    def __post_init__(self) -> None:
        _digest(self.candidate_pool_hash, "materialized rail pool hash")
        _digest(self.rail_plan_hash, "materialized rail plan hash")
        _digest(self.final_routing_world_hash, "materialized rail world hash")
        facts = _exact_tuple(
            self.materialized_facts,
            limit=MAX_RAIL_DEMANDS,
            label="materialized rail facts",
        )
        if not all(type(item) is MaterializedRailFact for item in facts):
            raise RailPlacementError("materialized rail facts contain invalid values")
        if len({item.content_id for item in facts}) != len(facts):
            raise RailPlacementError("materialized rail content ids must be unique")
        if len({item.demand_id for item in facts}) != len(facts):
            raise RailPlacementError("materialized rail demand ids must be unique")
        _nonempty_string(self.proof_scope, "materialized rail proof_scope")
        object.__setattr__(
            self, "materialized_facts", tuple(sorted(facts, key=lambda x: x.demand_id))
        )

    def canonical_json(self) -> str:
        return _canonical_json(
            {
                "candidate_pool_hash": self.candidate_pool_hash,
                "final_routing_world_hash": self.final_routing_world_hash,
                "materialized_facts": [
                    {
                        "access_xy_nm": list(item.access_xy_nm),
                        "axis_authority": _axis_authority_payload(
                            item.axis_authority
                        ),
                        "bbox_nm": list(item.bbox_nm),
                        "content_id": item.content_id,
                        "demand_id": item.demand_id,
                        "layer_key": list(item.layer_key),
                        "net": item.net,
                    }
                    for item in self.materialized_facts
                ],
                "proof_scope": self.proof_scope,
                "rail_plan_hash": self.rail_plan_hash,
                "schema": "clay.materialized-rail-plan-certificate.v3",
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def compile_edge_rail_demands(declarations: object) -> tuple[EdgeRailDemand, ...]:
    """Validate and canonicalize already-normalized declaration facts.

    Width is mandatory in :class:`EdgeRailDemand`; this compiler has no numeric
    default and therefore cannot silently synthesize an 8-um rail.
    """

    values = _bounded_iterable(
        declarations, limit=MAX_RAIL_DEMANDS, label="edge rail declarations"
    )
    if not all(type(item) is EdgeRailDemand for item in values):
        raise RailPlacementError(
            "edge rail declarations must contain exact EdgeRailDemand records"
        )
    ordered = tuple(sorted(values, key=lambda item: item.demand_id))
    demand_ids = tuple(item.demand_id for item in ordered)
    declaration_refs = tuple(item.declaration_ref for item in ordered)
    if len(set(demand_ids)) != len(demand_ids):
        raise RailPlacementError("edge rail declarations contain duplicate demand ids")
    if len(set(declaration_refs)) != len(declaration_refs):
        raise RailPlacementError("edge rail declarations contain duplicate refs")
    return ordered


def enumerate_rail_placement_candidates(
    scene: RailPlacementScene,
    demands: object,
    tech: object,
) -> RailPlacementCandidatePool:
    """Enumerate exact-width rail rectangles from scene boundary events only."""

    if type(scene) is not RailPlacementScene:
        raise TypeError("scene must be RailPlacementScene")
    normalized = compile_edge_rail_demands(demands)
    try:
        current_pdk_digest = pdk_provenance_digest(tech)
    except Exception as exc:
        raise RailPlacementError("cannot compute current PDK provenance") from exc
    if current_pdk_digest != scene.pdk_provenance_digest:
        raise RailPlacementError("scene PDK provenance does not match current tech")

    try:
        grid_nm = pdk_manufacturing_grid_nm(tech)
    except Exception as exc:
        raise RailPlacementError("cannot resolve PDK manufacturing grid") from exc
    if not _is_exact_int(grid_nm) or not 1 <= grid_nm <= MAX_COORD_ABS_NM:
        raise RailPlacementError(
            "PDK manufacturing grid must be a bounded positive exact integer"
        )

    min_spaces: dict[int, int] = {}
    for demand in normalized:
        _verify_demand_against_tech(demand, tech)
        min_spaces[demand.layer_n] = _required_metal_rule(
            tech, demand.layer_n, "min_space"
        )
        if demand.width_nm % grid_nm:
            raise RailPlacementError(
                f"rail width {demand.width_nm} is off the {grid_nm}-nm "
                "manufacturing grid"
            )
        run_bounds = (
            (scene.cell_bbox_nm[0], scene.cell_bbox_nm[2])
            if demand.preferred_axis == "H"
            else (scene.cell_bbox_nm[1], scene.cell_bbox_nm[3])
        )
        if any(value % grid_nm for value in run_bounds):
            raise RailPlacementError(
                "rail scene run-axis cell bounds are off manufacturing grid"
            )

    scene_hash = scene.sha256()
    candidates: list[RailPlacementCandidate] = []
    enumeration_work = 0

    access_origins: dict[str, PointNm] = {}
    for demand in normalized:
        enumeration_work = _charge_enumeration_work(
            enumeration_work,
            len(scene.terminal_facts),
            "collecting topology terminal facts",
        )
        topology_facts = tuple(
            fact for fact in scene.terminal_facts if fact.net == demand.net
        )
        if not topology_facts:
            continue
        access_origins[demand.demand_id] = (
            _nearest_grid_nm(
                _median_nm(tuple(fact.xy_nm[0] for fact in topology_facts)),
                grid_nm,
            ),
            _nearest_grid_nm(
                _median_nm(tuple(fact.xy_nm[1] for fact in topology_facts)),
                grid_nm,
            ),
        )

    active_union_profiles: dict[str, _ActiveUnionStripeProfile] = {}
    for axis in sorted(
        {
            demand.preferred_axis
            for demand in normalized
            if demand.demand_id in access_origins
        }
    ):
        profile, preprocessing_work = _build_active_union_stripe_profile(
            scene.active_projection_facts,
            scene.cell_bbox_nm,
            axis,
        )
        enumeration_work = _charge_enumeration_work(
            enumeration_work,
            preprocessing_work,
            f"building exact {axis}-rail active-union profile",
        )
        active_union_profiles[axis] = profile

    conductor_start_indices: dict[str, _ConductorForbiddenStartIndex] = {}
    for demand in normalized:
        if demand.demand_id not in access_origins:
            continue
        index, preprocessing_work = _build_conductor_forbidden_start_index(
            demand,
            scene.conductor_facts,
            scene.cell_bbox_nm,
            min_spaces[demand.layer_n],
        )
        enumeration_work = _charge_enumeration_work(
            enumeration_work,
            preprocessing_work,
            "indexing exact existing-conductor forbidden starts",
        )
        conductor_start_indices[demand.demand_id] = index

    for demand in normalized:
        access_origin = access_origins.get(demand.demand_id)
        if access_origin is None:
            continue
        cross_low, cross_high = (
            (scene.cell_bbox_nm[1], scene.cell_bbox_nm[3])
            if demand.preferred_axis == "H"
            else (scene.cell_bbox_nm[0], scene.cell_bbox_nm[2])
        )
        exact_start = (
            cross_low + demand.offset_nm
            if demand.edge in {"bottom", "left"}
            else cross_high - demand.width_nm - demand.offset_nm
        )
        if exact_start % grid_nm:
            raise RailPlacementError(
                "declared internal edge rail is off the manufacturing grid"
            )
        # EdgeRailDemand is a hard in-floorplan boundary obligation. The CP
        # selector has always discarded every non-edge candidate, so retaining
        # thousands of interior/exterior breakpoints only consumed memory and
        # could hit MAX_RAIL_CANDIDATES before the one selectable candidate was
        # reached. Enumerate exactly the candidate the selector is authorized to
        # choose; conductor/active checks below remain unchanged and fail closed.
        starts = (exact_start,)
        for start in starts:
            bbox = _candidate_bbox(scene.cell_bbox_nm, demand, start)
            if any(abs(value) > MAX_COORD_ABS_NM for value in bbox):
                continue
            enumeration_work = _charge_enumeration_work(
                enumeration_work,
                _conductor_start_query_work(conductor_start_indices[demand.demand_id]),
                "querying existing-conductor forbidden starts",
            )
            if _conductor_start_is_forbidden(
                conductor_start_indices[demand.demand_id],
                start,
            ):
                continue
            enumeration_work = _charge_enumeration_work(
                enumeration_work,
                _active_union_query_work(active_union_profiles[demand.preferred_axis]),
                "querying exact active-union profile",
            )
            active_overlap = _active_union_stripe_overlap(
                active_union_profiles[demand.preferred_axis],
                bbox,
                demand.preferred_axis,
            )
            exterior = _exterior_extension(
                bbox, scene.cell_bbox_nm, demand.preferred_axis
            )
            declared_edge_eligible = _bbox_is_on_declared_internal_edge(
                bbox, scene.cell_bbox_nm, demand.edge, demand.offset_nm
            )
            if not declared_edge_eligible:
                continue
            edge_distance = _edge_distance(bbox, scene.cell_bbox_nm, demand.edge)
            access = (
                min(max(access_origin[0], bbox[0]), bbox[2]),
                min(max(access_origin[1], bbox[1]), bbox[3]),
            )
            placement_kind = (
                "exterior"
                if exterior
                else ("over_cell" if active_overlap else "interior")
            )
            candidate_payload = {
                "access_xy_nm": list(access),
                "bbox_nm": list(bbox),
                "demand": _demand_payload(demand),
                "rail_placement_scene_hash": scene_hash,
            }
            candidate = RailPlacementCandidate(
                candidate_id=f"rail:{demand.demand_id}:{_sha256(candidate_payload)[:24]}",
                demand_id=demand.demand_id,
                bbox_nm=bbox,
                access_xy_nm=access,
                placement_kind=placement_kind,
                active_overlap_nm2=active_overlap,
                exterior_extension_nm=exterior,
                edge_distance_nm=edge_distance,
                declared_edge_eligible=declared_edge_eligible,
            )
            candidates.append(candidate)
            if len(candidates) > MAX_RAIL_CANDIDATES:
                raise RailPlacementLimitError(
                    f"rail candidate count exceeds {MAX_RAIL_CANDIDATES}"
                )

    candidates_tuple = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    spacing_contracts: list[RailDemandPairSpacingContract] = []
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            enumeration_work = _charge_enumeration_work(
                enumeration_work, 1, "compiling demand-pair spacing contracts"
            )
            if left.layer_key != right.layer_key:
                continue
            if left.preferred_axis != right.preferred_axis:
                raise RailPlacementError(
                    "same-layer rail demands have inconsistent preferred axes"
                )
            rule_spacing = max(min_spaces[left.layer_n], min_spaces[right.layer_n])
            same_net = left.net == right.net
            spacing_contracts.append(
                RailDemandPairSpacingContract(
                    left_demand_id=left.demand_id,
                    right_demand_id=right.demand_id,
                    layer_key=left.layer_key,
                    preferred_axis=left.preferred_axis,
                    relation="same_net" if same_net else "different_net",
                    minimum_gap_nm=(
                        rule_spacing if same_net else max(rule_spacing, grid_nm)
                    ),
                )
            )
    return RailPlacementCandidatePool(
        placement_provenance_digest=scene.placement_provenance_digest,
        pdk_provenance_digest=scene.pdk_provenance_digest,
        declaration_provenance_digest=scene.declaration_provenance_digest,
        rail_placement_scene_hash=scene_hash,
        demands=normalized,
        candidates=candidates_tuple,
        conflicts=(),
        spacing_contracts=tuple(spacing_contracts),
    )


def verify_rail_placement_candidate_pool(
    scene: RailPlacementScene,
    demands: object,
    tech: object,
    pool: RailPlacementCandidatePool,
) -> None:
    """Freshly re-enumerate a pool from its exact scene and PDK inputs."""

    if type(pool) is not RailPlacementCandidatePool:
        raise TypeError("pool must be RailPlacementCandidatePool")
    replayed = enumerate_rail_placement_candidates(scene, demands, tech)
    if replayed.canonical_json() != pool.canonical_json():
        raise RailPlacementError(
            "rail candidate pool does not match fresh scene replay"
        )


def _load_cp_model():
    """Narrow import seam; an unavailable solver is UNKNOWN, never fallback."""

    from ortools.sat.python import cp_model

    return cp_model


@dataclass(frozen=True)
class _RailDemandCpChoice:
    """One bounded canonical candidate-index choice in the CP model."""

    candidate_index_var: object
    candidates: tuple[RailPlacementCandidate, ...]
    active_overlap_var: object
    exterior_extension_var: object
    edge_distance_var: object


def _element_var(
    model: object,
    index_var: object,
    values: tuple[int, ...],
    *,
    name: str,
):
    """Bind one exact integer attribute to a canonical candidate array."""

    sentinel_values = (*values, 0)
    result = model.NewIntVar(
        min(sentinel_values),
        max(sentinel_values),
        name,
    )
    model.AddElement(index_var, sentinel_values, result)
    return result


def _checked_int64_sum(values: tuple[int, ...], *, label: str) -> int:
    """Return a non-negative sum only when CP-SAT can represent it exactly."""

    total = 0
    for value in values:
        if not _is_exact_int(value) or value < 0:
            raise RailPlacementError(
                f"{label} terms must be non-negative exact integers"
            )
        total += value
        if total > 2**63 - 1:
            raise RailPlacementLimitError(f"{label} exceeds the CP-SAT int64 bound")
    return total


def _build_cp_model(
    pool: RailPlacementCandidatePool,
    cp_model: object,
    *,
    assumed_core_ids: Optional[tuple[str, ...]] = None,
):
    model = cp_model.CpModel()
    enabled = (
        {f"required_demand:{demand.demand_id}" for demand in pool.demands}
        if assumed_core_ids is None
        else set(assumed_core_ids)
    )
    assumption_index_to_id: dict[int, str] = {}
    assumption_vars: dict[str, object] = {}
    choices: dict[str, _RailDemandCpChoice] = {}
    cross_coordinate_vars: dict[str, tuple[object, object]] = {}
    for demand in pool.demands:
        options = tuple(
            candidate
            for candidate in pool.candidates_for(demand.demand_id)
            if candidate.declared_edge_eligible
        )
        sentinel_index = len(options)
        candidate_index = model.NewIntVar(
            0,
            sentinel_index,
            f"rail-candidate-index::{demand.demand_id}",
        )
        assumption_id = f"required_demand:{demand.demand_id}"
        assumption = model.NewBoolVar(f"assumption::{assumption_id}")
        assumption_vars[demand.demand_id] = assumption
        model.Add(candidate_index < sentinel_index).OnlyEnforceIf(assumption)
        model.Add(candidate_index == sentinel_index).OnlyEnforceIf(assumption.Not())
        assumption_index_to_id[assumption.Index()] = assumption_id
        if assumption_id in enabled:
            model.AddAssumption(assumption)
        else:
            model.Add(assumption == 0)
        low_values = tuple(
            (
                candidate.bbox_nm[1]
                if demand.preferred_axis == "H"
                else candidate.bbox_nm[0]
            )
            for candidate in options
        )
        high_values = tuple(
            (
                candidate.bbox_nm[3]
                if demand.preferred_axis == "H"
                else candidate.bbox_nm[2]
            )
            for candidate in options
        )
        low_var = _element_var(
            model,
            candidate_index,
            low_values,
            name=f"rail-cross-low::{demand.demand_id}",
        )
        high_var = _element_var(
            model,
            candidate_index,
            high_values,
            name=f"rail-cross-high::{demand.demand_id}",
        )
        cross_coordinate_vars[demand.demand_id] = (low_var, high_var)
        choices[demand.demand_id] = _RailDemandCpChoice(
            candidate_index_var=candidate_index,
            candidates=options,
            active_overlap_var=_element_var(
                model,
                candidate_index,
                tuple(item.active_overlap_nm2 for item in options),
                name=f"rail-active-overlap::{demand.demand_id}",
            ),
            exterior_extension_var=_element_var(
                model,
                candidate_index,
                tuple(item.exterior_extension_nm for item in options),
                name=f"rail-exterior-extension::{demand.demand_id}",
            ),
            edge_distance_var=_element_var(
                model,
                candidate_index,
                tuple(item.edge_distance_nm for item in options),
                name=f"rail-edge-distance::{demand.demand_id}",
            ),
        )

    for attribute in (
        "active_overlap_nm2",
        "exterior_extension_nm",
        "edge_distance_nm",
    ):
        _checked_int64_sum(
            tuple(
                max(
                    (getattr(item, attribute) for item in choice.candidates),
                    default=0,
                )
                for choice in choices.values()
            ),
            label=f"rail {attribute} objective upper bound",
        )

    candidate_locations = {
        candidate.candidate_id: (demand_id, candidate_index)
        for demand_id, choice in choices.items()
        for candidate_index, candidate in enumerate(choice.candidates)
    }
    forbidden_by_demand_pair: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for conflict in pool.conflicts:
        left_location = candidate_locations.get(conflict.left_candidate_id)
        right_location = candidate_locations.get(conflict.right_candidate_id)
        if left_location is None or right_location is None:
            continue
        left_demand, left_index = left_location
        right_demand, right_index = right_location
        if left_demand == right_demand:
            # One index variable can never select two distinct candidates.
            continue
        if left_demand < right_demand:
            demand_pair = (left_demand, right_demand)
            forbidden = (left_index, right_index)
        else:
            demand_pair = (right_demand, left_demand)
            forbidden = (right_index, left_index)
        forbidden_by_demand_pair.setdefault(demand_pair, set()).add(forbidden)
    for (left_demand, right_demand), forbidden in sorted(
        forbidden_by_demand_pair.items()
    ):
        model.AddForbiddenAssignments(
            [
                choices[left_demand].candidate_index_var,
                choices[right_demand].candidate_index_var,
            ],
            sorted(forbidden),
        )
    for contract in pool.spacing_contracts:
        left_assumption = assumption_vars[contract.left_demand_id]
        right_assumption = assumption_vars[contract.right_demand_id]
        both_present = model.NewBoolVar(
            "rail-pair-present::"
            f"{contract.left_demand_id}::{contract.right_demand_id}"
        )
        model.Add(both_present <= left_assumption)
        model.Add(both_present <= right_assumption)
        model.Add(both_present >= left_assumption + right_assumption - 1)
        left_low, left_high = cross_coordinate_vars[contract.left_demand_id]
        right_low, right_high = cross_coordinate_vars[contract.right_demand_id]
        left_before = model.NewBoolVar(
            "rail-left-before::"
            f"{contract.left_demand_id}::{contract.right_demand_id}"
        )
        right_before = model.NewBoolVar(
            "rail-right-before::"
            f"{contract.left_demand_id}::{contract.right_demand_id}"
        )
        if contract.relation == "different_net":
            model.Add(left_before + right_before == both_present)
        else:
            touching = model.NewBoolVar(
                "rail-touching::"
                f"{contract.left_demand_id}::{contract.right_demand_id}"
            )
            model.Add(left_before + right_before + touching == both_present)
            model.Add(left_low <= right_high).OnlyEnforceIf(touching)
            model.Add(right_low <= left_high).OnlyEnforceIf(touching)
        model.Add(left_high + contract.minimum_gap_nm <= right_low).OnlyEnforceIf(
            left_before
        )
        model.Add(right_high + contract.minimum_gap_nm <= left_low).OnlyEnforceIf(
            right_before
        )
    return model, choices, assumption_index_to_id


def _new_solver(cp_model: object, budget: float, *, fixed_search: bool = False):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    # Large canonical Element arrays are already the compact formulation.
    # Expanding them during presolve can exhaust the whole stage budget before
    # search starts; keep the exact native constraints instead.
    solver.parameters.cp_model_presolve = False
    solver.parameters.max_deterministic_time = budget
    solver.parameters.max_time_in_seconds = budget
    if fixed_search:
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
    return solver


def _solve_optimal(
    pool: RailPlacementCandidatePool,
    cp_model: object,
    budget: float,
):
    model, choices, assumption_indices = _build_cp_model(pool, cp_model)
    # The public budget is global to this lexicographic solve, rather than being
    # silently multiplied by the number of objective tiers and declarations.
    solve_count = 4
    stage_budget = budget / solve_count
    objective_specs = (
        (
            "active_overlap_nm2",
            sum(choice.active_overlap_var for choice in choices.values()),
            "maximize",
        ),
        (
            "exterior_extension_nm",
            sum(choice.exterior_extension_var for choice in choices.values()),
            "minimize",
        ),
        (
            "edge_distance_nm",
            sum(choice.edge_distance_var for choice in choices.values()),
            "minimize",
        ),
    )
    objective_values: list[int] = []
    for _, expression, direction in objective_specs:
        if direction == "maximize":
            model.Maximize(expression)
        else:
            model.Minimize(expression)
        solver = _new_solver(cp_model, stage_budget)
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return status, solver, assumption_indices, None, None
        optimum = int(solver.Value(expression))
        objective_values.append(optimum)
        model.Add(expression == optimum)

    model.ClearObjective()
    if choices:
        # Candidate arrays and demands are canonical.  One fixed-search phase
        # therefore chooses the minimum feasible index for each demand in
        # demand-id order after all three exact objective optima are fixed.
        model.AddDecisionStrategy(
            [choice.candidate_index_var for choice in choices.values()],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )
    last_solver = _new_solver(cp_model, stage_budget, fixed_search=True)
    last_status = last_solver.Solve(model)
    if last_status != cp_model.OPTIMAL:
        return last_status, last_solver, assumption_indices, None, None
    selected_ids = tuple(
        choice.candidates[
            int(last_solver.Value(choice.candidate_index_var))
        ].candidate_id
        for choice in choices.values()
    )
    tie_total = _objectives_for_selection(pool, set(selected_ids))[3]
    return (
        last_status,
        last_solver,
        assumption_indices,
        selected_ids,
        tuple(objective_values + [tie_total]),
    )


def select_rail_placement_plan(
    pool: RailPlacementCandidatePool,
    *,
    max_deterministic_time: float = DEFAULT_RAIL_PLACEMENT_DETERMINISTIC_TIME,
) -> RailPlacementSelectionResult:
    """Select one candidate per declaration with exact lexicographic objectives."""

    if type(pool) is not RailPlacementCandidatePool:
        raise TypeError("pool must be RailPlacementCandidatePool")
    budget = _deterministic_budget(max_deterministic_time)
    pool_hash = pool.sha256()
    if budget == 0:
        return RailPlacementSelectionResult(
            status="UNKNOWN",
            solver_status="BUDGET_EXHAUSTED",
            candidate_pool_hash=pool_hash,
        )
    try:
        cp_model = _load_cp_model()
    except (ImportError, OSError):
        return RailPlacementSelectionResult(
            status="UNKNOWN",
            solver_status="SOLVER_UNAVAILABLE",
            candidate_pool_hash=pool_hash,
        )
    status, solver, assumption_indices, selected_ids, objective_values = _solve_optimal(
        pool, cp_model, budget
    )
    status_name = solver.StatusName(status)
    if status == cp_model.OPTIMAL:
        by_id = {candidate.candidate_id: candidate for candidate in pool.candidates}
        demands_by_id = {demand.demand_id: demand for demand in pool.demands}
        plan = RailPlacementPlan(
            candidate_pool_hash=pool_hash,
            selected=tuple(
                SelectedRailPlacement.from_candidate(
                    by_id[candidate_id],
                    demands_by_id[by_id[candidate_id].demand_id].axis_authority,
                )
                for candidate_id in selected_ids
            ),
            objective_values=objective_values,
            proof_scope=_SAT_PROOF_SCOPE,
        )
        _verify_rail_placement_plan_structure(pool, plan)
        return RailPlacementSelectionResult(
            status="SAT",
            solver_status=status_name,
            candidate_pool_hash=pool_hash,
            plan=plan,
        )
    if status == cp_model.INFEASIBLE:
        core_ids = tuple(
            sorted(
                {
                    assumption_indices[index]
                    for index in solver.SufficientAssumptionsForInfeasibility()
                    if index in assumption_indices
                }
            )
        )
        if core_ids:
            certificate = RailPlacementInfeasibilityCertificate(
                candidate_pool_hash=pool_hash,
                assumption_core_ids=core_ids,
                proof_scope=_UNSAT_PROOF_SCOPE,
            )
            verify_rail_placement_infeasibility(
                pool,
                certificate,
                max_deterministic_time=max_deterministic_time,
            )
            return RailPlacementSelectionResult(
                status="INFEASIBLE",
                solver_status=status_name,
                candidate_pool_hash=pool_hash,
                certificate=certificate,
            )
        return RailPlacementSelectionResult(
            status="UNKNOWN",
            solver_status="INFEASIBLE_WITHOUT_ASSUMPTION_CORE",
            candidate_pool_hash=pool_hash,
        )
    return RailPlacementSelectionResult(
        status="UNKNOWN",
        solver_status=status_name,
        candidate_pool_hash=pool_hash,
    )


def _verify_rail_placement_plan_structure(
    pool: RailPlacementCandidatePool,
    plan: RailPlacementPlan,
) -> tuple[str, ...]:
    if type(pool) is not RailPlacementCandidatePool:
        raise TypeError("pool must be RailPlacementCandidatePool")
    if type(plan) is not RailPlacementPlan:
        raise TypeError("plan must be RailPlacementPlan")
    if plan.candidate_pool_hash != pool.sha256():
        raise RailPlacementError("rail plan candidate pool hash mismatch")
    if plan.proof_scope != _SAT_PROOF_SCOPE:
        raise RailPlacementError("rail plan proof scope mismatch")
    candidates_by_id = {item.candidate_id: item for item in pool.candidates}
    demands_by_id = {item.demand_id: item for item in pool.demands}
    selected_ids: list[str] = []
    for selected in plan.selected:
        candidate = candidates_by_id.get(selected.candidate_id)
        if candidate is None or candidate.demand_id != selected.demand_id:
            raise RailPlacementError("rail plan selects an unknown candidate")
        if selected != SelectedRailPlacement.from_candidate(
            candidate, demands_by_id[selected.demand_id].axis_authority
        ):
            raise RailPlacementError("selected rail candidate snapshot mismatch")
        if not candidate.declared_edge_eligible:
            raise RailPlacementError(
                "rail plan selects a candidate outside its declared internal edge contract"
            )
        selected_ids.append(selected.candidate_id)
    if {item.demand_id for item in plan.selected} != {
        demand.demand_id for demand in pool.demands
    }:
        raise RailPlacementError("rail plan does not select every demand exactly once")
    selected_set = set(selected_ids)
    for conflict in pool.conflicts:
        if {
            conflict.left_candidate_id,
            conflict.right_candidate_id,
        }.issubset(selected_set):
            raise RailPlacementError("rail plan selects a hard candidate conflict")
    selected_by_demand = {item.demand_id: item for item in plan.selected}
    for contract in pool.spacing_contracts:
        if _violates_demand_pair_spacing_contract(
            selected_by_demand[contract.left_demand_id].bbox_nm,
            selected_by_demand[contract.right_demand_id].bbox_nm,
            contract,
        ):
            raise RailPlacementError(
                "rail plan violates a demand-pair spacing contract"
            )
    expected_objectives = _objectives_for_selection(pool, selected_set)
    if plan.objective_values != expected_objectives:
        raise RailPlacementError("rail plan objective values do not replay")
    return tuple(selected_ids)


def verify_rail_placement_plan(
    pool: RailPlacementCandidatePool,
    plan: RailPlacementPlan,
    *,
    tech: object,
    max_deterministic_time: float = DEFAULT_RAIL_PLACEMENT_DETERMINISTIC_TIME,
) -> None:
    """Independently replay a SAT artifact and its canonical optimum."""

    _verify_pool_against_current_tech(pool, tech)
    selected_ids = _verify_rail_placement_plan_structure(pool, plan)
    budget = _deterministic_budget(max_deterministic_time)
    if budget == 0:
        raise RailPlacementError("rail plan replay budget must be positive")
    try:
        cp_model = _load_cp_model()
    except (ImportError, OSError) as exc:
        raise RailPlacementError("cannot replay rail plan without CP-SAT") from exc
    status, _, _, optimum_ids, optimum_values = _solve_optimal(pool, cp_model, budget)
    if status != cp_model.OPTIMAL:
        raise RailPlacementError("rail plan optimum replay did not finish OPTIMAL")
    if tuple(sorted(selected_ids)) != tuple(sorted(optimum_ids)):
        raise RailPlacementError("rail plan is not the canonical optimum")
    if plan.objective_values != optimum_values:
        raise RailPlacementError("rail plan objective optimum mismatch")


def certify_rail_placement_plan_replay(
    pool: RailPlacementCandidatePool,
    plan: RailPlacementPlan,
    *,
    tech: object,
    max_deterministic_time: float = DEFAULT_RAIL_PLACEMENT_DETERMINISTIC_TIME,
) -> RailPlacementOptimalityReplay:
    """Deeply replay one SAT plan and return its exact downstream receipt."""

    verify_rail_placement_plan(
        pool,
        plan,
        tech=tech,
        max_deterministic_time=max_deterministic_time,
    )
    return _issue_rail_placement_optimality_replay(
        candidate_pool_hash=pool.sha256(),
        rail_plan_hash=plan.sha256(),
        proof_scope=plan.proof_scope,
        axis_authorities=tuple(
            demand.axis_authority for demand in pool.demands
        ),
    )


def _verify_rail_placement_optimality_replay(
    pool: RailPlacementCandidatePool,
    plan: RailPlacementPlan,
    receipt: RailPlacementOptimalityReplay,
    *,
    tech: object,
) -> None:
    """Verify receipt structure and bindings without running CP-SAT again."""

    if type(receipt) is not RailPlacementOptimalityReplay:
        raise TypeError("receipt must be RailPlacementOptimalityReplay")
    if getattr(receipt, "_issuer_authority", None) is not _RAIL_REPLAY_ISSUER_AUTHORITY:
        raise TypeError("receipt must be a module-issued replay capability")
    _verify_pool_against_current_tech(pool, tech)
    _verify_rail_placement_plan_structure(pool, plan)
    if receipt.candidate_pool_hash != pool.sha256():
        raise RailPlacementError("rail replay receipt pool hash mismatch")
    if receipt.rail_plan_hash != plan.sha256():
        raise RailPlacementError("rail replay receipt plan hash mismatch")
    if (
        receipt.proof_scope != plan.proof_scope
        or receipt.proof_scope != _SAT_PROOF_SCOPE
    ):
        raise RailPlacementError("rail replay receipt proof scope mismatch")
    expected_authorities = tuple(
        demand.axis_authority for demand in pool.demands
    )
    if receipt.axis_authorities != expected_authorities:
        raise RailPlacementError("rail replay receipt axis authority mismatch")
    registered = _RAIL_REPLAY_CAPABILITIES.get(id(receipt))
    if registered is None or registered[0] is not receipt:
        raise TypeError(
            "rail replay must be the exact module-issued object; copied replay "
            "capabilities are invalid"
        )
    if registered[1] != _rail_replay_state(receipt):
        raise TypeError("rail replay capability was mutated after issuance")


def verify_rail_placement_infeasibility(
    pool: RailPlacementCandidatePool,
    certificate: RailPlacementInfeasibilityCertificate,
    *,
    max_deterministic_time: float = DEFAULT_RAIL_PLACEMENT_DETERMINISTIC_TIME,
) -> None:
    """Freshly replay a claimed sufficient assumption core."""

    if type(pool) is not RailPlacementCandidatePool:
        raise TypeError("pool must be RailPlacementCandidatePool")
    if type(certificate) is not RailPlacementInfeasibilityCertificate:
        raise TypeError("certificate must be RailPlacementInfeasibilityCertificate")
    if certificate.candidate_pool_hash != pool.sha256():
        raise RailPlacementError("rail infeasibility candidate pool hash mismatch")
    if certificate.proof_scope != _UNSAT_PROOF_SCOPE:
        raise RailPlacementError("rail infeasibility proof scope mismatch")
    budget = _deterministic_budget(max_deterministic_time)
    if budget == 0:
        raise RailPlacementError("rail infeasibility replay budget must be positive")
    valid_ids = {f"required_demand:{demand.demand_id}" for demand in pool.demands}
    core_ids = set(certificate.assumption_core_ids)
    if not core_ids or not core_ids.issubset(valid_ids):
        raise RailPlacementError("rail infeasibility core contains unknown ids")
    try:
        cp_model = _load_cp_model()
    except (ImportError, OSError) as exc:
        raise RailPlacementError(
            "cannot replay rail infeasibility without CP-SAT"
        ) from exc
    model, _, _ = _build_cp_model(
        pool,
        cp_model,
        assumed_core_ids=certificate.assumption_core_ids,
    )
    solver = _new_solver(cp_model, budget)
    status = solver.Solve(model)
    if status != cp_model.INFEASIBLE:
        raise RailPlacementError(
            f"rail infeasibility core replay was {solver.StatusName(status)}, not INFEASIBLE"
        )


def _deterministic_budget(value: object) -> float:
    if type(value) not in (int, float):
        raise RailPlacementError("max_deterministic_time must be a finite number")
    budget = float(value)
    if budget < 0 or not math.isfinite(budget) or budget > MAX_RAIL_DETERMINISTIC_TIME:
        raise RailPlacementError(
            "max_deterministic_time must be finite, non-negative, and no "
            f"greater than {MAX_RAIL_DETERMINISTIC_TIME:g}"
        )
    return budget


def _objectives_for_selection(
    pool: RailPlacementCandidatePool,
    selected_ids: set[str],
) -> tuple[int, int, int, int]:
    selected = tuple(
        candidate
        for candidate in pool.candidates
        if candidate.candidate_id in selected_ids
    )
    tie = 0
    for demand in pool.demands:
        options = tuple(
            sorted(
                pool.candidates_for(demand.demand_id),
                key=lambda item: item.candidate_id,
            )
        )
        chosen = tuple(item for item in options if item.candidate_id in selected_ids)
        if len(chosen) != 1:
            raise RailPlacementError(
                "objective replay requires one candidate per demand"
            )
        tie += options.index(chosen[0])
    return (
        sum(item.active_overlap_nm2 for item in selected),
        sum(item.exterior_extension_nm for item in selected),
        sum(item.edge_distance_nm for item in selected),
        tie,
    )


def _verify_demand_against_tech(demand: EdgeRailDemand, tech: object) -> None:
    try:
        layer_key = tech.metal_key(demand.layer_n)
        axis = tech.metal_preferred_axis_declared(demand.layer_n)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RailPlacementError(
            f"PDK cannot resolve declared rail layer {demand.layer_n}"
        ) from exc
    try:
        exact_layer_key = tuple(layer_key)
    except TypeError as exc:
        raise RailPlacementError("PDK returned an invalid rail layer key") from exc
    if exact_layer_key != demand.layer_key:
        raise RailPlacementError("rail demand layer key does not match current PDK")
    if axis is not None and (
        type(axis) is not str or axis not in ("H", "V")
    ):
        raise RailPlacementError(
            "rail current preferred axis must be exact H, V, or undeclared"
        )
    authority = demand.axis_authority
    if type(authority) is not RailAxisAuthority:
        raise RailPlacementError("rail demand axis authority is not typed")
    if getattr(authority, "_issuer_authority", None) is not _RAIL_AXIS_ISSUER_AUTHORITY:
        raise TypeError("rail demand axis authority must be module-issued")
    if authority.declaration_ref != demand.declaration_ref:
        raise RailPlacementError(
            "rail axis authority declaration ref does not match demand"
        )
    if authority.current_preferred_axis != axis:
        raise RailPlacementError(
            "rail axis authority current preferred axis differs from current PDK"
        )
    expected_relation = _axis_relation(authority.actual_axis, axis)
    if authority.relation != expected_relation:
        raise RailPlacementError(
            "rail axis authority relation differs from current PDK"
        )
    if authority.axis_source == "pdk_default":
        if authority.source_receipt is not None:
            raise RailPlacementError(
                "PDK-default rail authority cannot carry a source/AST receipt"
            )
        try:
            default_axis = tech.metal_preferred_axis(demand.layer_n)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RailPlacementError(
                "PDK default axis authority cannot be recomputed"
            ) from exc
        if default_axis != authority.actual_axis:
            raise RailPlacementError(
                "PDK default axis authority does not match current PDK default"
            )
    elif authority.axis_source == "explicit_intent":
        _source, declaration = _verify_rail_axis_declaration_receipt(
            authority.source_receipt
        )
        if authority.source_receipt.declaration_ref != demand.declaration_ref:
            raise RailPlacementError(
                "rail source/AST receipt declaration ref does not match demand"
            )
        if declaration.get("kind") != "trunk":
            raise RailPlacementError(
                "rail source/AST receipt does not name a trunk declaration"
            )
        attributes = declaration.get("attributes")
        if not isinstance(attributes, dict):
            raise RailPlacementError("rail trunk AST attributes are unavailable")
        if str(attributes.get("net") or "") != demand.net:
            raise RailPlacementError("rail trunk AST net does not match demand")
        if str(attributes.get("orient") or "").strip().upper() != authority.actual_axis:
            raise RailPlacementError("rail trunk AST orientation does not match authority")
        if str(attributes.get("edge") or "").strip().lower() != demand.edge:
            raise RailPlacementError("rail trunk AST edge does not match demand")
        layer_token = str(attributes.get("layer") or "").strip().casefold()
        layer_suffix = layer_token
        for prefix in ("metal", "met", "m"):
            if layer_token.startswith(prefix):
                layer_suffix = layer_token[len(prefix) :]
                break
        if not layer_suffix.isdigit() or int(layer_suffix) != demand.layer_n:
            raise RailPlacementError("rail trunk AST layer does not match demand")
        declared_width = attributes.get("width")
        if declared_width not in (None, ""):
            try:
                raw_width_nm = round(float(declared_width) * 1000.0)
            except (TypeError, ValueError) as exc:
                raise RailPlacementError(
                    "rail trunk AST width is not numeric"
                ) from exc
            current_min_width = _required_metal_rule(
                tech, demand.layer_n, "min_width"
            )
            if demand.width_nm != max(raw_width_nm, current_min_width):
                raise RailPlacementError(
                    "rail trunk AST width does not match demand"
                )
    else:
        raise RailPlacementError(
            "rail axis source must be explicit_intent or pdk_default"
        )
    expected_digest = _axis_declaration_digest(
        actual_axis=authority.actual_axis,
        axis_source=authority.axis_source,
        declaration_ref=authority.declaration_ref,
        source_receipt_digest=(
            None if authority.source_receipt is None else authority.source_receipt.sha256()
        ),
    )
    if authority.declaration_digest != expected_digest:
        raise RailPlacementError(
            "rail axis declaration digest does not replay"
        )
    registered_authority = _RAIL_AXIS_CAPABILITIES.get(id(authority))
    if registered_authority is None or registered_authority[0] is not authority:
        raise TypeError(
            "rail axis authority must be the exact module-issued object; copied "
            "capabilities are invalid"
        )
    if registered_authority[1] != _rail_axis_authority_state(authority):
        raise TypeError("rail axis authority was mutated after issuance")
    minimum_width = _required_metal_rule(tech, demand.layer_n, "min_width")
    if demand.width_nm < minimum_width:
        raise RailPlacementError(
            f"rail width {demand.width_nm} is below PDK minimum {minimum_width}"
        )


def _current_spacing_contracts(
    demands: tuple[EdgeRailDemand, ...], tech: object
) -> tuple[RailDemandPairSpacingContract, ...]:
    try:
        grid_nm = pdk_manufacturing_grid_nm(tech)
    except Exception as exc:
        raise RailPlacementError("cannot resolve current PDK manufacturing grid") from exc
    if not _is_exact_int(grid_nm) or not 1 <= grid_nm <= MAX_COORD_ABS_NM:
        raise RailPlacementError(
            "current PDK manufacturing grid must be a bounded positive integer"
        )
    min_spaces = {
        demand.layer_n: _required_metal_rule(tech, demand.layer_n, "min_space")
        for demand in demands
    }
    contracts: list[RailDemandPairSpacingContract] = []
    for left_index, left in enumerate(demands):
        for right in demands[left_index + 1 :]:
            if left.layer_key != right.layer_key:
                continue
            if left.actual_axis != right.actual_axis:
                raise RailPlacementError(
                    "same-layer rail demands have inconsistent current axes"
                )
            rule_spacing = max(min_spaces[left.layer_n], min_spaces[right.layer_n])
            same_net = left.net == right.net
            contracts.append(
                RailDemandPairSpacingContract(
                    left_demand_id=left.demand_id,
                    right_demand_id=right.demand_id,
                    layer_key=left.layer_key,
                    preferred_axis=left.actual_axis,
                    relation="same_net" if same_net else "different_net",
                    minimum_gap_nm=(
                        rule_spacing if same_net else max(rule_spacing, grid_nm)
                    ),
                )
            )
    return tuple(contracts)


def _verify_pool_against_current_tech(
    pool: RailPlacementCandidatePool, tech: object
) -> None:
    """Rebind a stored pool to current PDK identity and every spacing rule."""

    if type(pool) is not RailPlacementCandidatePool:
        raise TypeError("pool must be RailPlacementCandidatePool")
    try:
        current_digest = pdk_provenance_digest(tech)
    except Exception as exc:
        raise RailPlacementError("cannot recompute current PDK provenance") from exc
    if current_digest != pool.pdk_provenance_digest:
        raise RailPlacementError(
            "pool PDK provenance does not match current tech"
        )
    demands = compile_edge_rail_demands(pool.demands)
    for demand in demands:
        _verify_demand_against_tech(demand, tech)
    expected_spacing = _current_spacing_contracts(demands, tech)
    if expected_spacing != pool.spacing_contracts:
        raise RailPlacementError(
            "rail pool spacing contracts do not match current PDK min_space"
        )


def _required_metal_rule(tech: object, layer_n: int, method_name: str) -> int:
    method = getattr(tech, method_name, None)
    if not callable(method):
        raise RailPlacementError(f"PDK lacks required {method_name} rule lookup")
    value = None
    for name in (f"metal{layer_n}", f"met{layer_n}", f"m{layer_n}"):
        try:
            value = method(name)
        except (KeyError, TypeError, ValueError):
            value = None
        if value is not None:
            break
    if not _is_exact_int(value) or value < 0 or value > 2 * MAX_COORD_ABS_NM:
        raise RailPlacementError(
            f"PDK {method_name} for metal{layer_n} must be a bounded exact "
            "non-negative integer"
        )
    return value


def _candidate_bbox(cell: BBoxNm, demand: EdgeRailDemand, start: int) -> BBoxNm:
    if demand.preferred_axis == "H":
        return (cell[0], start, cell[2], start + demand.width_nm)
    return (start, cell[1], start + demand.width_nm, cell[3])


def _bbox_is_on_declared_internal_edge(
    bbox: BBoxNm,
    cell: BBoxNm,
    edge: str,
    offset_nm: int = 0,
) -> bool:
    """Return whether a full-run rail occupies its exact internal edge band.

    The floorplan rectangle is a hard boundary and ``edge`` is a hard placement
    declaration, not a distance preference. A legal rail is exactly the declared
    inward offset from its boundary, spans the orthogonal dimension, and stays inside.
    """

    if edge == "top":
        return (
            bbox[0] == cell[0]
            and bbox[2] == cell[2]
            and cell[1] <= bbox[1] < bbox[3] <= cell[3]
            and bbox[3] == cell[3] - offset_nm
        )
    if edge == "bottom":
        return (
            bbox[0] == cell[0]
            and bbox[2] == cell[2]
            and cell[1] <= bbox[1] < bbox[3] <= cell[3]
            and bbox[1] == cell[1] + offset_nm
        )
    if edge == "right":
        return (
            bbox[1] == cell[1]
            and bbox[3] == cell[3]
            and cell[0] <= bbox[0] < bbox[2] <= cell[2]
            and bbox[2] == cell[2] - offset_nm
        )
    if edge == "left":
        return (
            bbox[1] == cell[1]
            and bbox[3] == cell[3]
            and cell[0] <= bbox[0] < bbox[2] <= cell[2]
            and bbox[0] == cell[0] + offset_nm
        )
    raise RailPlacementError(f"unsupported rail edge {edge!r}")


def _median_nm(values: tuple[int, ...]) -> int:
    if not values:
        raise RailPlacementError("topology median requires at least one value")
    ordered = tuple(sorted(values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _floor_grid_nm(value: int, grid_nm: int) -> int:
    return (value // grid_nm) * grid_nm


def _ceil_grid_nm(value: int, grid_nm: int) -> int:
    return -((-value) // grid_nm) * grid_nm


def _nearest_grid_nm(value: int, grid_nm: int) -> int:
    low = _floor_grid_nm(value, grid_nm)
    high = _ceil_grid_nm(value, grid_nm)
    return low if value - low <= high - value else high


def _sort_work_upper_bound(count: int) -> int:
    if count < 2:
        return count
    return 2 * count * count.bit_length()


def _build_active_union_stripe_profile(
    facts: tuple[RailProjectionFact, ...],
    cell: BBoxNm,
    axis: str,
) -> tuple[_ActiveUnionStripeProfile, int]:
    """Sweep active rectangles once into an exact cross-axis area integral.

    Every H rail spans the cell's full x interval and every V rail spans its
    full y interval.  Between consecutive active cross-axis events, the union
    of covered run-axis intervals is constant.  A coordinate-compressed cover
    tree therefore produces an exact area density for every slab; prefix areas
    answer each later rail-stripe query without revisiting active facts.
    """

    if axis not in ("H", "V"):
        raise RailPlacementError("active-union profile axis must be H or V")
    run_cell_low, run_cell_high = (
        (cell[0], cell[2]) if axis == "H" else (cell[1], cell[3])
    )
    clipped: list[tuple[int, int, int, int]] = []
    work = len(facts)
    for fact in facts:
        cross_low, cross_high, run_low, run_high = (
            (
                fact.bbox_nm[1],
                fact.bbox_nm[3],
                fact.bbox_nm[0],
                fact.bbox_nm[2],
            )
            if axis == "H"
            else (
                fact.bbox_nm[0],
                fact.bbox_nm[2],
                fact.bbox_nm[1],
                fact.bbox_nm[3],
            )
        )
        run_low = max(run_low, run_cell_low)
        run_high = min(run_high, run_cell_high)
        if run_low < run_high:
            clipped.append((cross_low, cross_high, run_low, run_high))
    if not clipped:
        return _ActiveUnionStripeProfile((), (0,), ()), work

    run_events = tuple(
        sorted(
            {
                coordinate
                for _, _, run_low, run_high in clipped
                for coordinate in (run_low, run_high)
            }
        )
    )
    cross_events = tuple(
        sorted(
            {
                coordinate
                for cross_low, cross_high, _, _ in clipped
                for coordinate in (cross_low, cross_high)
            }
        )
    )
    work += _sort_work_upper_bound(len(run_events))
    work += _sort_work_upper_bound(len(cross_events))
    run_indices = {coordinate: index for index, coordinate in enumerate(run_events)}
    updates: dict[int, list[tuple[int, int, int]]] = {
        coordinate: [] for coordinate in cross_events
    }
    for cross_low, cross_high, run_low, run_high in clipped:
        low_index = run_indices[run_low]
        high_index = run_indices[run_high] - 1
        updates[cross_low].append((low_index, high_index, 1))
        updates[cross_high].append((low_index, high_index, -1))
    work += 4 * len(clipped) + len(cross_events)

    segment_count = len(run_events) - 1
    cover_counts = [0] * (4 * segment_count)
    covered_lengths = [0] * (4 * segment_count)

    def update_cover(
        node: int,
        left: int,
        right: int,
        query_left: int,
        query_right: int,
        delta: int,
    ) -> None:
        nonlocal work
        work += 1
        if query_left <= left and right <= query_right:
            cover_counts[node] += delta
            if cover_counts[node] < 0:
                raise RailPlacementError(
                    "active-union sweep produced a negative cover count"
                )
        else:
            middle = (left + right) // 2
            if query_left <= middle:
                update_cover(
                    node * 2,
                    left,
                    middle,
                    query_left,
                    query_right,
                    delta,
                )
            if middle < query_right:
                update_cover(
                    node * 2 + 1,
                    middle + 1,
                    right,
                    query_left,
                    query_right,
                    delta,
                )
        if cover_counts[node] > 0:
            covered_lengths[node] = run_events[right + 1] - run_events[left]
        elif left == right:
            covered_lengths[node] = 0
        else:
            covered_lengths[node] = (
                covered_lengths[node * 2] + covered_lengths[node * 2 + 1]
            )

    prefix_areas = [0]
    slab_run_lengths: list[int] = []
    for event_index, coordinate in enumerate(cross_events):
        # End events remove rectangles active in the preceding slab; start
        # events then add rectangles active in the following slab.
        for delta in (-1, 1):
            for low_index, high_index, update_delta in updates[coordinate]:
                if update_delta == delta:
                    update_cover(
                        1,
                        0,
                        segment_count - 1,
                        low_index,
                        high_index,
                        delta,
                    )
        if event_index + 1 == len(cross_events):
            continue
        next_coordinate = cross_events[event_index + 1]
        run_length = covered_lengths[1]
        slab_run_lengths.append(run_length)
        prefix_areas.append(
            prefix_areas[-1] + run_length * (next_coordinate - coordinate)
        )
    return (
        _ActiveUnionStripeProfile(
            cross_events_nm=cross_events,
            prefix_areas_nm2=tuple(prefix_areas),
            run_lengths_nm=tuple(slab_run_lengths),
        ),
        work,
    )


def _active_union_query_work(profile: _ActiveUnionStripeProfile) -> int:
    # Two prefix queries, each bounded by one binary search plus arithmetic.
    return 2 * (max(1, len(profile.cross_events_nm).bit_length()) + 2)


def _active_union_area_below(
    profile: _ActiveUnionStripeProfile,
    coordinate: int,
) -> int:
    events = profile.cross_events_nm
    if not events or coordinate <= events[0]:
        return 0
    slab_index = bisect_right(events, coordinate) - 1
    if slab_index >= len(profile.run_lengths_nm):
        return profile.prefix_areas_nm2[-1]
    return profile.prefix_areas_nm2[slab_index] + profile.run_lengths_nm[slab_index] * (
        coordinate - events[slab_index]
    )


def _active_union_stripe_overlap(
    profile: _ActiveUnionStripeProfile,
    bbox: BBoxNm,
    axis: str,
) -> int:
    cross_low, cross_high = (bbox[1], bbox[3]) if axis == "H" else (bbox[0], bbox[2])
    return _active_union_area_below(profile, cross_high) - _active_union_area_below(
        profile, cross_low
    )


def _exterior_extension(bbox: BBoxNm, cell: BBoxNm, axis: str) -> int:
    if axis == "H":
        return max(cell[1] - bbox[1], 0) + max(bbox[3] - cell[3], 0)
    return max(cell[0] - bbox[0], 0) + max(bbox[2] - cell[2], 0)


def _edge_distance(bbox: BBoxNm, cell: BBoxNm, edge: str) -> int:
    boundary = {
        "left": cell[0],
        "bottom": cell[1],
        "right": cell[2],
        "top": cell[3],
    }[edge]
    low, high = (
        (bbox[0], bbox[2])
        if edge in ("left", "right")
        else (
            bbox[1],
            bbox[3],
        )
    )
    if low <= boundary <= high:
        return 0
    return min(abs(boundary - low), abs(boundary - high))


def _axis_gap(left_low: int, left_high: int, right_low: int, right_high: int) -> int:
    if left_high < right_low:
        return right_low - left_high
    if right_high < left_low:
        return left_low - right_high
    return 0


def _rects_within_spacing(left: BBoxNm, right: BBoxNm, spacing: int) -> bool:
    x_gap = _axis_gap(left[0], left[2], right[0], right[2])
    y_gap = _axis_gap(left[1], left[3], right[1], right[3])
    # Even a PDK-reported zero spacing cannot legalize geometrically touching
    # different-net conductors: that would be an electrical short.
    return (x_gap == 0 and y_gap == 0) or (x_gap < spacing and y_gap < spacing)


def _rects_touch_or_overlap(left: BBoxNm, right: BBoxNm) -> bool:
    return (
        _axis_gap(left[0], left[2], right[0], right[2]) == 0
        and _axis_gap(left[1], left[3], right[1], right[3]) == 0
    )


def _violates_demand_pair_spacing_contract(
    left_bbox: BBoxNm,
    right_bbox: BBoxNm,
    contract: RailDemandPairSpacingContract,
) -> bool:
    if not _rects_within_spacing(left_bbox, right_bbox, contract.minimum_gap_nm):
        return False
    return not (
        contract.relation == "same_net"
        and _rects_touch_or_overlap(left_bbox, right_bbox)
    )


def _build_conductor_forbidden_start_index(
    demand: EdgeRailDemand,
    facts: tuple[RailProjectionFact, ...],
    cell: BBoxNm,
    spacing: int,
) -> tuple[_ConductorForbiddenStartIndex, int]:
    """Compile exact existing-metal spacing into rail-start intervals.

    A rail occupies the full cell run axis, so a conductor's run-axis gap is
    constant for every candidate.  When that gap can trigger
    :func:`_rects_within_spacing`, the remaining cross-axis predicate is one
    inclusive integer interval.  Same-net facts touching the run interval
    subtract the cross-axis touch interval because such contact is legal.
    """

    if not _is_exact_int(spacing) or spacing < 0:
        raise RailPlacementError(
            "conductor forbidden-start spacing must be a non-negative integer"
        )
    run_cell_low, run_cell_high = (
        (cell[0], cell[2]) if demand.preferred_axis == "H" else (cell[1], cell[3])
    )
    intervals: list[tuple[int, int]] = []
    work = len(facts)

    def add_interval(low: int, high: int) -> None:
        if low <= high:
            intervals.append((low, high))

    for fact in facts:
        if fact.layer_key != demand.layer_key:
            continue
        fact_run_low, fact_run_high, fact_cross_low, fact_cross_high = (
            (
                fact.bbox_nm[0],
                fact.bbox_nm[2],
                fact.bbox_nm[1],
                fact.bbox_nm[3],
            )
            if demand.preferred_axis == "H"
            else (
                fact.bbox_nm[1],
                fact.bbox_nm[3],
                fact.bbox_nm[0],
                fact.bbox_nm[2],
            )
        )
        run_gap = _axis_gap(
            run_cell_low,
            run_cell_high,
            fact_run_low,
            fact_run_high,
        )
        run_touches = run_gap == 0
        if not run_touches and not (spacing > 0 and run_gap < spacing):
            continue

        if spacing == 0:
            forbidden_low = fact_cross_low - demand.width_nm
            forbidden_high = fact_cross_high
        else:
            forbidden_low = fact_cross_low - demand.width_nm - spacing + 1
            forbidden_high = fact_cross_high + spacing - 1

        if fact.owner_net == demand.net and run_touches:
            # Cross-axis touch/overlap is legal only because run-axis contact
            # makes the rectangles themselves touch.  Positive sub-spacing
            # gaps on either side remain forbidden.
            add_interval(
                forbidden_low,
                fact_cross_low - demand.width_nm - 1,
            )
            add_interval(fact_cross_high + 1, forbidden_high)
        else:
            add_interval(forbidden_low, forbidden_high)

    work += _sort_work_upper_bound(len(intervals)) + len(intervals)
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for low, high in intervals:
        if not merged or low > merged[-1][1] + 1:
            merged.append((low, high))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
    return (
        _ConductorForbiddenStartIndex(
            lower_bounds_nm=tuple(low for low, _ in merged),
            upper_bounds_nm=tuple(high for _, high in merged),
        ),
        work,
    )


def _conductor_start_query_work(index: _ConductorForbiddenStartIndex) -> int:
    return max(1, len(index.lower_bounds_nm).bit_length()) + 2


def _conductor_start_is_forbidden(
    index: _ConductorForbiddenStartIndex,
    start: int,
) -> bool:
    interval_index = bisect_right(index.lower_bounds_nm, start) - 1
    return interval_index >= 0 and start <= index.upper_bounds_nm[interval_index]


def _violates_existing_conductor_spacing(
    bbox: BBoxNm,
    demand: EdgeRailDemand,
    facts: tuple[RailProjectionFact, ...],
    spacing: int,
) -> bool:
    for fact in facts:
        if fact.layer_key != demand.layer_key:
            continue
        if not _rects_within_spacing(bbox, fact.bbox_nm, spacing):
            continue
        if fact.owner_net == demand.net and _rects_touch_or_overlap(bbox, fact.bbox_nm):
            continue
        return True
    return False


def verify_materialized_rail_plan_certificate(
    pool: RailPlacementCandidatePool,
    plan: RailPlacementPlan,
    certificate: MaterializedRailPlanCertificate,
    *,
    tech: object,
    geometry: object,
    placement_um: object,
    pins_by_net: object,
    routing_world: object,
) -> bool:
    """Replay selected rail artwork/pins and require owned ``rail`` atoms."""

    if type(certificate) is not MaterializedRailPlanCertificate:
        raise TypeError("certificate must be MaterializedRailPlanCertificate")
    replay_receipt = certify_rail_placement_plan_replay(
        pool, plan, tech=tech
    )
    return _verify_materialized_rail_plan_certificate_with_replay(
        pool,
        plan,
        replay_receipt,
        certificate,
        tech=tech,
        geometry=geometry,
        placement_um=placement_um,
        pins_by_net=pins_by_net,
        routing_world=routing_world,
    )


def _verify_materialized_rail_plan_certificate_with_replay(
    pool: RailPlacementCandidatePool,
    plan: RailPlacementPlan,
    replay_receipt: RailPlacementOptimalityReplay,
    certificate: MaterializedRailPlanCertificate,
    *,
    tech: object,
    geometry: object,
    placement_um: object,
    pins_by_net: object,
    routing_world: object,
) -> bool:
    """Verify emitted rail evidence using an already-audited optimum replay."""

    if type(certificate) is not MaterializedRailPlanCertificate:
        raise TypeError("certificate must be MaterializedRailPlanCertificate")
    _verify_rail_placement_optimality_replay(
        pool, plan, replay_receipt, tech=tech
    )
    if certificate.candidate_pool_hash != pool.sha256():
        raise RailPlacementError("materialized rail candidate-pool hash mismatch")
    if certificate.rail_plan_hash != plan.sha256():
        raise RailPlacementError("materialized rail plan hash mismatch")
    world_sha = getattr(routing_world, "sha256", None)
    if not callable(world_sha) or certificate.final_routing_world_hash != world_sha():
        raise RailPlacementError("materialized rail final-world hash mismatch")
    if not isinstance(geometry, dict) or not isinstance(placement_um, dict):
        raise TypeError("geometry and placement_um must be exact mappings")
    if not isinstance(pins_by_net, dict):
        raise TypeError("pins_by_net must be an exact mapping")

    demands = {item.demand_id: item for item in pool.demands}
    selected = {item.demand_id: item for item in plan.selected}
    facts = {item.demand_id: item for item in certificate.materialized_facts}
    if set(facts) != set(selected) or set(selected) != set(demands):
        raise RailPlacementError("materialized rail facts do not cover the plan")
    world_rails = set(tuple(item) for item in getattr(routing_world, "rails", ()))
    for demand_id in sorted(facts):
        fact = facts[demand_id]
        demand = demands[demand_id]
        choice = selected[demand_id]
        if (
            fact.net != demand.net
            or fact.layer_key != demand.layer_key
            or fact.axis_authority != demand.axis_authority
            or choice.axis_authority != demand.axis_authority
            or fact.bbox_nm != choice.bbox_nm
            or fact.access_xy_nm != choice.access_xy_nm
        ):
            raise RailPlacementError(
                "materialized rail fact axis authority or geometry differs from selection"
            )
        stamp = geometry.get(fact.content_id)
        if stamp is None or placement_um.get(fact.content_id) != (0.0, 0.0):
            raise RailPlacementError("materialized rail stamp/placement is absent")
        if str(getattr(stamp, "source", "")) != "power_rail":
            raise RailPlacementError("materialized rail stamp has wrong provenance")
        rings = (getattr(stamp, "polys_by_layer", {}) or {}).get(fact.layer_key, ())
        expected_ring = (
            (fact.bbox_nm[0] / 1000.0, fact.bbox_nm[1] / 1000.0),
            (fact.bbox_nm[2] / 1000.0, fact.bbox_nm[1] / 1000.0),
            (fact.bbox_nm[2] / 1000.0, fact.bbox_nm[3] / 1000.0),
            (fact.bbox_nm[0] / 1000.0, fact.bbox_nm[3] / 1000.0),
        )
        if len(tuple(rings)) != 1 or tuple(tuple(p) for p in rings[0]) != expected_ring:
            raise RailPlacementError("materialized rail geometry bbox mismatch")
        exact_pins = [
            pin
            for pin in pins_by_net.get(fact.net, ())
            if str(getattr(pin, "device", "")) == fact.content_id
            and str(getattr(pin, "terminal", "")) == "rail"
            and tuple(getattr(pin, "layer", ())) == fact.layer_key
            and tuple(round(float(v) * 1000) for v in getattr(pin, "xy_um", ()))
            == fact.access_xy_nm
        ]
        if len(exact_pins) != 1:
            raise RailPlacementError("materialized rail requires one exact access pin")
        world_fact = (
            fact.net,
            fact.layer_key[0],
            fact.layer_key[1],
            *fact.bbox_nm,
        )
        if world_fact not in world_rails:
            raise RailPlacementError("materialized rail is absent from final world")

    from clay.layout.route.resource_conductors import OwnedConductorComponentIndex

    index = OwnedConductorComponentIndex.build(routing_world)
    for fact in certificate.materialized_facts:
        if not any(
            component.net == fact.net
            and component.layer_key == fact.layer_key
            and "rail" in component.kinds
            and component.bbox_nm[0] <= fact.access_xy_nm[0] <= component.bbox_nm[2]
            and component.bbox_nm[1] <= fact.access_xy_nm[1] <= component.bbox_nm[3]
            for component in index.components
        ):
            raise RailPlacementError("materialized rail lacks an owned rail component")
    return True


def _fact_payload(fact: RailProjectionFact) -> dict[str, object]:
    return {
        "bbox_nm": list(fact.bbox_nm),
        "kind": fact.kind,
        "layer_key": list(fact.layer_key),
        "owner_net": fact.owner_net,
    }


def _fact_sort_key(fact: RailProjectionFact) -> str:
    return _canonical_json(_fact_payload(fact))


def _terminal_payload(fact: RailTerminalFact) -> dict[str, object]:
    return {
        "layer_key": list(fact.layer_key),
        "net": fact.net,
        "provenance": fact.provenance,
        "xy_nm": list(fact.xy_nm),
    }


def _scene_payload(scene: RailPlacementScene) -> dict[str, object]:
    return {
        "active_projection_facts": [
            _fact_payload(fact) for fact in scene.active_projection_facts
        ],
        "cell_bbox_nm": list(scene.cell_bbox_nm),
        "conductor_facts": [_fact_payload(fact) for fact in scene.conductor_facts],
        "declaration_provenance_digest": scene.declaration_provenance_digest,
        "pdk_provenance_digest": scene.pdk_provenance_digest,
        "placement_provenance_digest": scene.placement_provenance_digest,
        "terminal_facts": [_terminal_payload(fact) for fact in scene.terminal_facts],
        "schema": ("clay.rail-placement-scene.v2"
                   if any(fact.owner_net is None for fact in scene.conductor_facts)
                   else "clay.rail-placement-scene.v1"),
    }


def _axis_authority_payload(
    authority: RailAxisAuthority,
) -> dict[str, object]:
    return {
        "actual_axis": authority.actual_axis,
        "axis_source": authority.axis_source,
        "current_preferred_axis": authority.current_preferred_axis,
        "declaration_digest": authority.declaration_digest,
        "declaration_ref": authority.declaration_ref,
        "relation": authority.relation,
        "schema": "clay.rail-axis-authority.v2",
        "source_receipt": (
            None
            if authority.source_receipt is None
            else _axis_source_receipt_payload(authority.source_receipt)
        ),
    }


def _axis_source_receipt_payload(
    receipt: RailAxisDeclarationReceipt,
) -> dict[str, object]:
    return {
        "canonical_declaration_ast_json": receipt.canonical_declaration_ast_json,
        "canonical_source_ast_json": receipt.canonical_source_ast_json,
        "cell_name": receipt.cell_name,
        "declaration_ast_digest": receipt.declaration_ast_digest,
        "declaration_ref": receipt.declaration_ref,
        "schema": "clay.rail-axis-source-receipt.v2",
        "source_bytes_digest": receipt.source_bytes_digest,
        "source_locator_digest": hashlib.sha256(
            receipt.source_locator.encode("utf-8")
        ).hexdigest(),
        "source_ast_digest": receipt.source_ast_digest,
    }


def _demand_payload(demand: EdgeRailDemand) -> dict[str, object]:
    return {
        "axis_authority": _axis_authority_payload(demand.axis_authority),
        "declaration_ref": demand.declaration_ref,
        "demand_id": demand.demand_id,
        "edge": demand.edge,
        "layer_key": list(demand.layer_key),
        "layer_n": demand.layer_n,
        "net": demand.net,
        "width_nm": demand.width_nm,
        "width_source": demand.width_source,
        **({"offset_nm": demand.offset_nm} if demand.offset_nm else {}),
    }


def _candidate_payload(candidate: RailPlacementCandidate) -> dict[str, object]:
    return {
        "access_xy_nm": list(candidate.access_xy_nm),
        "active_overlap_nm2": candidate.active_overlap_nm2,
        "bbox_nm": list(candidate.bbox_nm),
        "candidate_id": candidate.candidate_id,
        "declared_edge_eligible": candidate.declared_edge_eligible,
        "demand_id": candidate.demand_id,
        "edge_distance_nm": candidate.edge_distance_nm,
        "exterior_extension_nm": candidate.exterior_extension_nm,
        "placement_kind": candidate.placement_kind,
    }


def _pool_payload(pool: RailPlacementCandidatePool) -> dict[str, object]:
    return {
        "candidates": [_candidate_payload(item) for item in pool.candidates],
        "conflicts": [
            {
                "left_candidate_id": item.left_candidate_id,
                "reason": item.reason,
                "right_candidate_id": item.right_candidate_id,
            }
            for item in pool.conflicts
        ],
        "declaration_provenance_digest": pool.declaration_provenance_digest,
        "demands": [_demand_payload(item) for item in pool.demands],
        "pdk_provenance_digest": pool.pdk_provenance_digest,
        "placement_provenance_digest": pool.placement_provenance_digest,
        "rail_placement_scene_hash": pool.rail_placement_scene_hash,
        "schema": "clay.rail-placement-candidate-pool.v5",
        "spacing_contracts": [
            {
                "layer_key": list(item.layer_key),
                "left_demand_id": item.left_demand_id,
                "minimum_gap_nm": item.minimum_gap_nm,
                "preferred_axis": item.preferred_axis,
                "relation": item.relation,
                "right_demand_id": item.right_demand_id,
            }
            for item in pool.spacing_contracts
        ],
    }


def _selected_payload(selected: SelectedRailPlacement) -> dict[str, object]:
    return {
        "access_xy_nm": list(selected.access_xy_nm),
        "active_overlap_nm2": selected.active_overlap_nm2,
        "axis_authority": _axis_authority_payload(selected.axis_authority),
        "bbox_nm": list(selected.bbox_nm),
        "candidate_id": selected.candidate_id,
        "declared_edge_eligible": selected.declared_edge_eligible,
        "demand_id": selected.demand_id,
        "edge_distance_nm": selected.edge_distance_nm,
        "exterior_extension_nm": selected.exterior_extension_nm,
        "placement_kind": selected.placement_kind,
    }


def _plan_payload(plan: RailPlacementPlan) -> dict[str, object]:
    return {
        "candidate_pool_hash": plan.candidate_pool_hash,
        "objective_values": list(plan.objective_values),
        "proof_scope": plan.proof_scope,
        "schema": "clay.rail-placement-plan.v3",
        "selected": [_selected_payload(item) for item in plan.selected],
    }


__all__ = [
    "DEFAULT_RAIL_PLACEMENT_DETERMINISTIC_TIME",
    "MAX_RAIL_DEMANDS",
    "MAX_RAIL_ENUMERATION_WORK",
    "EdgeRailDemand",
    "RailAxisAuthority",
    "RailAxisDeclarationReceipt",
    "MaterializedRailFact",
    "MaterializedRailPlanCertificate",
    "RailCandidateConflict",
    "RailDemandPairSpacingContract",
    "RailPlacementCandidate",
    "RailPlacementCandidatePool",
    "RailPlacementError",
    "RailPlacementInfeasibilityCertificate",
    "RailPlacementLimitError",
    "RailPlacementPlan",
    "RailPlacementScene",
    "RailPlacementSelectionResult",
    "RailProjectionFact",
    "RailTerminalFact",
    "SelectedRailPlacement",
    "compile_edge_rail_demands",
    "make_rail_axis_authority",
    "certify_rail_placement_plan_replay",
    "enumerate_rail_placement_candidates",
    "select_rail_placement_plan",
    "verify_rail_placement_candidate_pool",
    "verify_rail_placement_infeasibility",
    "verify_rail_placement_plan",
    "verify_materialized_rail_plan_certificate",
]
