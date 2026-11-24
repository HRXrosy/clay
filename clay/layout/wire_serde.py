"""JSON serde for routed wire segments.

Round-trips lists of ``WireSegment`` objects produced by
``clay.layout.routing`` (Phase 8a) and the multi-bend / cross-layer
routers in ``clay.layout.routing_advanced`` (Phase 8b/c).

Wire format (one dict per segment):

    {
        "x_start_nm": int,
        "y_start_nm": int,
        "x_end_nm":   int,
        "y_end_nm":   int,
        "width_nm":   int,
        "layer_name": str | None,   # optional, only if attribute present
        "net_name":   str | None,   # optional, only if attribute present
    }

The actual ``WireSegment`` dataclass (frozen, eq=True) currently exposes
``start_x / start_y / end_x / end_y / width_nm`` only. Any future
``net``/``wire_name``/``layer_name`` attribute attached by upstream
routing code will be picked up via ``getattr`` and round-tripped
through the optional fields. Unknown JSON keys are silently ignored
on deserialize for forward compatibility.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .routing import WireSegment


# === Field name constants ===

_X_START = "x_start_nm"
_Y_START = "y_start_nm"
_X_END = "x_end_nm"
_Y_END = "y_end_nm"
_WIDTH = "width_nm"
_LAYER = "layer_name"
_NET = "net_name"


def _first_attr(obj: Any, *names: str) -> Optional[Any]:
    """Return first attribute that exists and is not None, else None."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def serialize_wire_segment(seg: Any) -> Dict[str, Any]:
    """Convert a ``WireSegment`` (or duck-typed equivalent) into a
    JSON-safe dict.

    Required keys: ``x_start_nm, y_start_nm, x_end_nm, y_end_nm,
    width_nm``. Optional keys ``layer_name`` and ``net_name`` are
    included only when the segment exposes a matching attribute
    (``layer_name``/``layer``, ``net_name``/``net``/``wire_name``).
    """
    out: Dict[str, Any] = {
        _X_START: int(seg.start_x),
        _Y_START: int(seg.start_y),
        _X_END: int(seg.end_x),
        _Y_END: int(seg.end_y),
        _WIDTH: int(seg.width_nm),
    }
    layer = _first_attr(seg, "layer_name", "layer")
    if layer is not None:
        out[_LAYER] = str(layer)
    net = _first_attr(seg, "net_name", "net", "wire_name")
    if net is not None:
        out[_NET] = str(net)
    return out


def deserialize_wire_segment(d: Dict[str, Any]) -> WireSegment:
    """Reconstruct a ``WireSegment`` from a serde dict.

    Required keys: ``x_start_nm, y_start_nm, x_end_nm, y_end_nm,
    width_nm``. Optional keys (``layer_name``, ``net_name``) and any
    additional unknown keys are silently ignored: the current
    ``WireSegment`` dataclass is frozen and does not store them, so
    attaching them would either fail or be lost on round-trip. The
    coordinate/width round-trip is preserved exactly.
    """
    try:
        return WireSegment(
            start_x=int(d[_X_START]),
            start_y=int(d[_Y_START]),
            end_x=int(d[_X_END]),
            end_y=int(d[_Y_END]),
            width_nm=int(d[_WIDTH]),
        )
    except KeyError as e:
        raise ValueError(
            f"deserialize_wire_segment: missing required key {e.args[0]!r}; "
            f"got keys {sorted(d.keys())}"
        ) from None


def serialize_wire_list(segments: List[Any]) -> str:
    """Serialize a list of ``WireSegment`` objects to a JSON string.

    Empty list yields ``"[]"``. The output is a JSON array of dicts
    in the format produced by :func:`serialize_wire_segment`.
    """
    return json.dumps([serialize_wire_segment(s) for s in segments])


def deserialize_wire_list(json_text: str) -> List[WireSegment]:
    """Inverse of :func:`serialize_wire_list`.

    Raises ``ValueError`` if the JSON does not parse to a list, or
    any element is not a dict with the required keys.
    """
    parsed = json.loads(json_text)
    if not isinstance(parsed, list):
        raise ValueError(
            f"deserialize_wire_list: expected top-level JSON array, "
            f"got {type(parsed).__name__}"
        )
    out: List[WireSegment] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(
                f"deserialize_wire_list: element [{i}] is not a dict "
                f"(got {type(item).__name__})"
            )
        out.append(deserialize_wire_segment(item))
    return out


def write_wires_json(segments: List[Any], output_path: str) -> None:
    """Write a list of wire segments to ``output_path`` as JSON text
    (UTF-8, indented for human readability)."""
    payload = [serialize_wire_segment(s) for s in segments]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_wires_json(path: str) -> List[WireSegment]:
    """Read a wire-list JSON file written by :func:`write_wires_json`."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return deserialize_wire_list(text)
