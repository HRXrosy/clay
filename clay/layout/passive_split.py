"""Materialize topology-preserving passive split intent.

Phase 1 supports parallel capacitor banks because Clay's existing PDK-driven
MIM builder, pin extractor, emitter, and full-extraction LVS fold already share
the ``value-per-unit + M`` representation.  Unsupported split forms fail before
placement; they are never silently treated as an unsplit passive.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

from clay.layout.intent import layout_intent_from_cell_def


class PassiveSplitError(ValueError):
    """A declared passive split cannot be materialized exactly."""


@dataclass(frozen=True)
class PassiveSplitEvidence:
    status: str
    passive: str
    unit_count: int
    unit_value: float
    total_value: float
    expected_value: float
    reason: str = ""


def equivalent_passive_value(
    kind: str,
    unit_value: float,
    unit_count: int,
    connection: str,
) -> float:
    """Return the exact scalar equivalent of a uniform passive bank."""

    count = int(unit_count)
    if count < 1:
        raise PassiveSplitError("passive split unit count must be positive")
    value = float(unit_value)
    if not math.isfinite(value) or value <= 0.0:
        raise PassiveSplitError("passive split unit value must be positive and finite")
    passive_kind = str(kind).lower()
    mode = str(connection).lower()
    if mode == "parallel" and passive_kind.startswith("cap"):
        return value * count
    if mode == "parallel" and passive_kind.startswith("res"):
        return value / count
    if mode == "series" and passive_kind.startswith("cap"):
        return value / count
    if mode == "series" and passive_kind.startswith("res"):
        return value * count
    raise PassiveSplitError(
        f"unsupported passive equivalent kind={kind!r} connection={connection!r}"
    )


def _declared_splits(cell_def: Any) -> tuple[Any, ...]:
    intent = layout_intent_from_cell_def(cell_def)
    return tuple(sorted(intent.splits, key=lambda item: str(item.passive)))


def _base_total(kind: str, value: float, multiplicity: int) -> float:
    passive_kind = str(kind).lower()
    if passive_kind.startswith("cap"):
        return equivalent_passive_value(kind, value, multiplicity, "parallel")
    if passive_kind.startswith("res"):
        return equivalent_passive_value(kind, value, multiplicity, "parallel")
    raise PassiveSplitError(f"unsupported split device class {kind!r}")


def _split_values(
    *,
    passive: str,
    kind: str,
    value: float,
    multiplicity: int,
    count: int,
    connection: str,
) -> tuple[float, int, float]:
    if connection != "parallel" or not str(kind).lower().startswith("cap"):
        raise PassiveSplitError(
            f"@split({passive}, n={count}, match={connection}) is unsupported in "
            "phase 1; only parallel capacitor banks have a materializer and checker"
        )
    if type(count) is not int or count < 1:
        raise PassiveSplitError(f"@split({passive}) n must be a positive integer")
    total = _base_total(kind, float(value), int(multiplicity))
    unit = total / count
    replay = equivalent_passive_value(kind, unit, count, connection)
    if not math.isclose(replay, total, rel_tol=1e-12, abs_tol=1e-30):
        raise PassiveSplitError(
            f"@split({passive}) cannot prove equivalent value: {replay!r} != {total!r}"
        )
    return unit, count, total


def realize_passive_split_records(
    cell_def: Any,
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Return copied passive records with declared parallel banks materialized."""

    out: dict[str, dict[str, object]] = {}
    for name, record in records.items():
        copied = dict(record)
        if isinstance(copied.get("pins"), Mapping):
            copied["pins"] = dict(copied["pins"])  # type: ignore[arg-type]
        out[str(name)] = copied
    for split in _declared_splits(cell_def):
        name = str(split.passive)
        if name not in out:
            raise PassiveSplitError(
                f"@split references unknown passive {name!r}; available={sorted(out)}"
            )
        record = out[name]
        kind = str(record.get("kind", ""))
        value = float(record.get("value", 0.0) or 0.0)
        multiplicity = max(1, int(record.get("M", 1) or 1))
        connection = str(split.match).lower()
        unit, count, total = _split_values(
            passive=name,
            kind=kind,
            value=value,
            multiplicity=multiplicity,
            count=int(split.n),
            connection=connection,
        )
        record["value"] = unit
        record["M"] = count
        record["split_connection"] = connection
        record["split_total_value"] = total
    return out


def realize_passive_split_views(
    cell_def: Any,
    passives: list[Any],
) -> list[Any]:
    """Apply the same record transform to router passive protocol objects."""

    records = {
        str(passive.name): {
            "kind": str(passive.kind),
            "value": float(passive.value),
            "pins": dict(passive.nets),
            "M": int(getattr(passive, "M", 1) or 1),
        }
        for passive in passives
    }
    realized = realize_passive_split_records(cell_def, records)
    out: list[Any] = []
    for passive in passives:
        clone = copy.copy(passive)
        record = realized[str(passive.name)]
        clone.value = float(record["value"])
        clone.M = int(record["M"])
        out.append(clone)
    return out


def _declared_passive_total(cell_def: Any, name: str) -> float:
    matches = [
        device
        for device in (getattr(cell_def, "devices", ()) or ())
        if str(getattr(device, "instance_name", "")) == name
    ]
    if len(matches) != 1:
        raise PassiveSplitError(
            f"split checker requires exactly one declared passive {name!r}; "
            f"found {len(matches)}"
        )
    device = matches[0]
    sizing = getattr(device, "sizing", {}) or {}
    kind = str(getattr(device, "device_class", ""))
    value = float(sizing.get("C", sizing.get("R", sizing.get("value", 0.0))) or 0.0)
    multiplicity = max(1, int(round(float(sizing.get("M", 1) or 1))))
    return _base_total(kind, value, multiplicity)


def verify_parallel_cap_split(
    cell_def: Any,
    records: Mapping[str, Mapping[str, object]],
    *,
    passive: str | None = None,
) -> PassiveSplitEvidence:
    """Verify one declared split record against the source electrical value."""

    splits = [
        split
        for split in _declared_splits(cell_def)
        if passive is None or str(split.passive) == str(passive)
    ]
    if len(splits) != 1:
        raise PassiveSplitError(
            f"split checker requires exactly one declaration; found {len(splits)}"
        )
    split = splits[0]
    name = str(split.passive)
    if name not in records:
        raise PassiveSplitError(f"split checker has no realized record for {name!r}")
    record = records[name]
    kind = str(record.get("kind", ""))
    count = int(record.get("M", 0) or 0)
    unit = float(record.get("value", 0.0) or 0.0)
    total = equivalent_passive_value(kind, unit, count, str(split.match))
    expected = _declared_passive_total(cell_def, name)
    passed = count == int(split.n) and math.isclose(
        total, expected, rel_tol=1e-12, abs_tol=1e-30
    )
    return PassiveSplitEvidence(
        status="pass" if passed else "fail",
        passive=name,
        unit_count=count,
        unit_value=unit,
        total_value=total,
        expected_value=expected,
        reason="" if passed else "realized bank does not replay declared value/count",
    )


__all__ = [
    "PassiveSplitError",
    "PassiveSplitEvidence",
    "equivalent_passive_value",
    "realize_passive_split_records",
    "realize_passive_split_views",
    "verify_parallel_cap_split",
]
