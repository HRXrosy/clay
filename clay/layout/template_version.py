"""Cell template version metadata.

Attaches ``(template_name, semver, content_hash, generated_at_iso)`` to an
``AnnotatedCell`` at build time, enabling compatibility checks between
templates of different vintages.

The content hash is a SHA-256 over a deterministic dump of:

* device names (sorted),
* the symbolic layout sizing of every device (finger count, pitch
  multiplier, stripe count), and
* layout annotations attached to each device.

Two cells with identical structure produce the same hash regardless of
insertion order or annotation order; structural changes (renaming a
device, resizing fingers, adding/removing annotations) flip the hash.

Semver compatibility follows ``MAJOR.MINOR.PATCH``: same template name
and same MAJOR version are compatible; MAJOR mismatch is not.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional, Tuple


_HASH_PREFIX_LEN = 16


@dataclass(frozen=True)
class TemplateVersion:
    """Immutable version stamp for an emitted cell template."""

    template_name: str
    semver: str
    content_hash_hex: str
    generated_at_iso: str


def _serialize_value(value: Any) -> Any:
    """Map arbitrary annotation values to JSON-safe primitives.

    Handles ``Fraction`` (rendered as ``"num/den"``), enums (rendered as
    their ``.value``), dataclass-like objects (rendered via ``__dict__``),
    lists/tuples, and pass-through scalars. Falls back to ``repr`` for
    anything exotic so the hash stays deterministic.
    """

    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Enum: enums expose ``.value``; check before generic ``__dict__``.
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        return _serialize_value(value.value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if hasattr(value, "__dict__"):
        cls = type(value).__name__
        payload = {k: _serialize_value(v) for k, v in sorted(value.__dict__.items())}
        return {"__cls__": cls, **payload}
    return repr(value)


def _layout_signature(layout: Any) -> dict:
    finger_count = getattr(layout, "finger_count", None)
    stripe_count = getattr(layout, "stripe_count", None)
    pitch_mult = getattr(layout, "pitch_multiplier", None)
    return {
        "finger_count": finger_count,
        "stripe_count": stripe_count,
        "pitch_multiplier": _serialize_value(pitch_mult),
    }


def _cell_payload(cell: Any) -> dict:
    """Build the deterministic dict that will be JSON-encoded for hashing."""

    layouts = getattr(cell, "layouts", {}) or {}
    annotations = getattr(cell, "annotations", {}) or {}

    devices = sorted(set(layouts.keys()) | set(annotations.keys()))

    device_blocks = []
    for dev in devices:
        layout = layouts.get(dev)
        ann_list = annotations.get(dev, []) or []
        # Annotations are serialized as a *sorted* list of records so
        # that insertion order does not affect the hash.
        serialized_anns = [_serialize_value(a) for a in ann_list]
        serialized_anns.sort(key=lambda a: json.dumps(a, sort_keys=True))
        device_blocks.append(
            {
                "device": dev,
                "layout": _layout_signature(layout) if layout is not None else None,
                "annotations": serialized_anns,
            }
        )

    return {
        "cell_name": getattr(cell, "cell_name", ""),
        "devices": device_blocks,
    }


def compute_cell_content_hash(cell: Any) -> str:
    """SHA-256 over a deterministic dump of the cell's structural content.

    Returns the full 64-character hex digest; the first
    :data:`_HASH_PREFIX_LEN` (16) characters are what
    :func:`stamp_cell` stores on the ``TemplateVersion``.
    """

    payload = _cell_payload(cell)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stamp_cell(
    cell: Any,
    template_name: str,
    semver: str = "1.0.0",
    *,
    timestamp_iso: Optional[str] = None,
) -> TemplateVersion:
    """Compute the content hash and build a :class:`TemplateVersion`.

    Parameters
    ----------
    cell:
        An :class:`~clay.layout.annotations.AnnotatedCell` (or anything
        exposing ``layouts``/``annotations``/``cell_name``).
    template_name:
        Logical template identifier (e.g. ``"five_t_ota"``).
    semver:
        Template semantic version ``MAJOR.MINOR.PATCH``. Defaults to
        ``"1.0.0"``. Validated via :func:`parse_semver`.
    timestamp_iso:
        Optional override for the ``generated_at_iso`` field. When
        ``None``, the current UTC time is used.
    """

    # Validate semver eagerly so callers get a clean error.
    parse_semver(semver)

    full_hash = compute_cell_content_hash(cell)
    prefix = full_hash[:_HASH_PREFIX_LEN]

    if timestamp_iso is None:
        timestamp_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

    return TemplateVersion(
        template_name=template_name,
        semver=semver,
        content_hash_hex=prefix,
        generated_at_iso=timestamp_iso,
    )


def parse_semver(s: str) -> Tuple[int, int, int]:
    """Parse ``"MAJOR.MINOR.PATCH"`` into ``(int, int, int)``.

    Raises :class:`ValueError` if the string is not three dot-separated
    non-negative integers.
    """

    if not isinstance(s, str):
        raise ValueError(f"semver must be a string, got {type(s).__name__}")
    parts = s.split(".")
    if len(parts) != 3:
        raise ValueError(f"semver must have 3 parts MAJOR.MINOR.PATCH, got {s!r}")
    out = []
    for part in parts:
        if not part or not part.isdigit():
            raise ValueError(f"semver components must be non-negative integers, got {s!r}")
        out.append(int(part))
    return (out[0], out[1], out[2])


def is_compatible(a: TemplateVersion, b: TemplateVersion) -> bool:
    """Two stamps are compatible iff same template name + same MAJOR.

    Same MAJOR with different MINOR or PATCH is still compatible; a
    different MAJOR breaks compatibility. Different template names are
    never compatible.
    """

    if a.template_name != b.template_name:
        return False
    major_a, _, _ = parse_semver(a.semver)
    major_b, _, _ = parse_semver(b.semver)
    return major_a == major_b
