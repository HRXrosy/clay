"""JSON serde for Clay layout dataclasses (Phase 17 cross-tool interop).

Provides pure-Python serialization / deserialization of the layout-as-co-variable
data model — ``SymbolicLayout``, ``AnnotatedCell`` (with all 6 annotation kinds),
``ParameterizedLibrary``, and ``DeviceSizing`` — to and from JSON-compatible
``dict`` / ``str`` representations.

Design notes
------------
* All enum fields encode as their ``.value`` string (e.g. ``"random_vt"``,
  ``"nwell"``, ``"vertical"``, ``"abab"``).
* ``Fraction`` fields encode as ``"num/den"`` strings to preserve rationality
  across JSON's float-only number type (e.g. ``"3/2"`` round-trips exactly,
  ``1.5`` would not).
* The annotation dispatch table is keyed by a ``"kind"`` discriminator with
  six values: ``matching``, ``guard_ring``, ``well_isolated``, ``symmetric``,
  ``interdigitated``, ``adjacent``. Unknown discriminators raise ``ValueError``.
* ``DeviceSizing`` is a plain ``Dict[str, Tuple[float, float]]`` type alias
  (see ``backend.py``). It serializes as ``{device: [w_um, l_um]}`` and
  deserializes back to ``(w_um, l_um)`` tuples.
* No external deps — only ``json`` and ``fractions`` from stdlib.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, Dict, List

from .annotations import (
    Adjacent,
    AnnotatedCell,
    GuardRing,
    InterdigitationPattern,
    Interdigitated,
    LayoutAnnotation,
    Matching,
    MatchingKind,
    Symmetric,
    SymmetryAxis,
    WellIsolated,
    WellType,
)
from .backend import ParameterizedLibrary, LayerPolygon, NamedCell, Point, Polygon
from .symbolic_layout import SymbolicLayout


# --- Fraction helpers ---


def _fraction_to_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def _fraction_from_str(s: str) -> Fraction:
    # Fraction("3/2") works natively; using explicit split for clearer errors.
    if not isinstance(s, str):
        raise ValueError(f"expected str for Fraction, got {type(s).__name__}")
    if "/" not in s:
        raise ValueError(f"expected 'num/den' Fraction string, got {s!r}")
    num_s, den_s = s.split("/", 1)
    return Fraction(int(num_s), int(den_s))


# --- SymbolicLayout ---


def serialize_symbolic_layout(layout: SymbolicLayout) -> dict:
    return {
        "finger_count": layout.finger_count,
        "pitch_multiplier": _fraction_to_str(layout.pitch_multiplier),
        "stripe_count": layout.stripe_count,
    }


def deserialize_symbolic_layout(d: dict) -> SymbolicLayout:
    return SymbolicLayout(
        finger_count=int(d["finger_count"]),
        pitch_multiplier=_fraction_from_str(d["pitch_multiplier"]),
        stripe_count=int(d["stripe_count"]),
    )


# --- LayoutAnnotation dispatch ---


def _serialize_annotation(ann: LayoutAnnotation) -> dict:
    if isinstance(ann, Matching):
        return {"kind": "matching", "other": ann.other, "match_kind": ann.kind.value}
    if isinstance(ann, GuardRing):
        return {"kind": "guard_ring", "well_type": ann.well_type.value}
    if isinstance(ann, WellIsolated):
        return {"kind": "well_isolated", "from_device": ann.from_device}
    if isinstance(ann, Symmetric):
        return {
            "kind": "symmetric",
            "devices": list(ann.devices),
            "axis": ann.axis.value,
        }
    if isinstance(ann, Interdigitated):
        return {
            "kind": "interdigitated",
            "with_device": ann.with_device,
            "pattern": ann.pattern.value,
            "finger_pairs": ann.finger_pairs,
        }
    if isinstance(ann, Adjacent):
        return {
            "kind": "adjacent",
            "to": ann.to,
            "max_separation_multiplier": _fraction_to_str(
                ann.max_separation_multiplier
            ),
        }
    raise ValueError(
        f"unknown LayoutAnnotation subclass: {type(ann).__name__}"
    )


def _deserialize_annotation(d: dict) -> LayoutAnnotation:
    kind = d.get("kind")
    if kind == "matching":
        return Matching(other=d["other"], kind=MatchingKind(d["match_kind"]))
    if kind == "guard_ring":
        return GuardRing(well_type=WellType(d["well_type"]))
    if kind == "well_isolated":
        return WellIsolated(from_device=d["from_device"])
    if kind == "symmetric":
        return Symmetric(
            devices=list(d["devices"]),
            axis=SymmetryAxis(d["axis"]),
        )
    if kind == "interdigitated":
        return Interdigitated(
            with_device=d["with_device"],
            pattern=InterdigitationPattern(d["pattern"]),
            finger_pairs=int(d["finger_pairs"]),
        )
    if kind == "adjacent":
        return Adjacent(
            to=d["to"],
            max_separation_multiplier=_fraction_from_str(
                d["max_separation_multiplier"]
            ),
        )
    raise ValueError(f"unknown annotation kind: {kind!r}")


# --- AnnotatedCell ---


def serialize_annotated_cell(cell: AnnotatedCell) -> dict:
    """Serialize cell + all annotations to JSON-compatible dict."""
    return {
        "cell_name": cell.cell_name,
        "layouts": {
            dev: serialize_symbolic_layout(layout)
            for dev, layout in cell.layouts.items()
        },
        "annotations": {
            dev: [_serialize_annotation(a) for a in anns]
            for dev, anns in cell.annotations.items()
        },
    }


def deserialize_annotated_cell(d: dict) -> AnnotatedCell:
    """Reconstruct AnnotatedCell + all annotation dataclasses from dict."""
    cell = AnnotatedCell(cell_name=d["cell_name"])
    for dev, layout_d in d.get("layouts", {}).items():
        cell.layouts[dev] = deserialize_symbolic_layout(layout_d)
    for dev, anns in d.get("annotations", {}).items():
        cell.annotations[dev] = [_deserialize_annotation(a) for a in anns]
    return cell


# --- ParameterizedLibrary (Phase 3a GDS-shaped IR) ---


def _serialize_point(p: Point) -> List[int]:
    return [p.x, p.y]


def _deserialize_point(seq: List[int]) -> Point:
    return Point(x=int(seq[0]), y=int(seq[1]))


def _serialize_polygon(poly: Polygon) -> dict:
    return {"vertices": [_serialize_point(v) for v in poly.vertices]}


def _deserialize_polygon(d: dict) -> Polygon:
    return Polygon(vertices=[_deserialize_point(v) for v in d.get("vertices", [])])


def _serialize_layer_polygon(lp: LayerPolygon) -> dict:
    return {
        "layer": lp.layer,
        "datatype": lp.datatype,
        "polygon": _serialize_polygon(lp.polygon),
    }


def _deserialize_layer_polygon(d: dict) -> LayerPolygon:
    return LayerPolygon(
        layer=int(d["layer"]),
        datatype=int(d["datatype"]),
        polygon=_deserialize_polygon(d["polygon"]),
    )


def _serialize_named_cell(cell: NamedCell) -> dict:
    return {
        "name": cell.name,
        "polygons": [_serialize_layer_polygon(lp) for lp in cell.polygons],
    }


def _deserialize_named_cell(d: dict) -> NamedCell:
    return NamedCell(
        name=d["name"],
        polygons=[_deserialize_layer_polygon(lp) for lp in d.get("polygons", [])],
    )


def serialize_parameterized_library(lib: ParameterizedLibrary) -> dict:
    return {
        "name": lib.name,
        "unit_user": lib.unit_user,
        "unit_db_meters": lib.unit_db_meters,
        "cells": [_serialize_named_cell(c) for c in lib.cells],
    }


def deserialize_parameterized_library(d: dict) -> ParameterizedLibrary:
    return ParameterizedLibrary(
        name=d["name"],
        unit_user=float(d.get("unit_user", 0.001)),
        unit_db_meters=float(d.get("unit_db_meters", 1e-9)),
        cells=[_deserialize_named_cell(c) for c in d.get("cells", [])],
    )


# --- DeviceSizing ---
#
# DeviceSizing is a type alias: Dict[str, Tuple[float, float]] (W, L in um).
# We serialize as {device: [w_um, l_um]} and reconstruct tuples on the way out.


def serialize_device_sizing(sizing: Dict[str, Any]) -> dict:
    out: Dict[str, List[float]] = {}
    for dev, wl in sizing.items():
        w, l = wl
        out[dev] = [float(w), float(l)]
    return out


def deserialize_device_sizing(d: dict) -> Dict[str, Any]:
    return {dev: (float(wl[0]), float(wl[1])) for dev, wl in d.items()}


# --- JSON string wrappers ---


_DISPATCH = {
    "symbolic_layout": (serialize_symbolic_layout, deserialize_symbolic_layout, SymbolicLayout),
    "annotated_cell": (serialize_annotated_cell, deserialize_annotated_cell, AnnotatedCell),
    "parameterized_library": (
        serialize_parameterized_library,
        deserialize_parameterized_library,
        ParameterizedLibrary,
    ),
    "device_sizing": (serialize_device_sizing, deserialize_device_sizing, dict),
}


def _detect_kind(obj: Any) -> str:
    if isinstance(obj, SymbolicLayout):
        return "symbolic_layout"
    if isinstance(obj, AnnotatedCell):
        return "annotated_cell"
    if isinstance(obj, ParameterizedLibrary):
        return "parameterized_library"
    if isinstance(obj, dict):
        return "device_sizing"
    raise ValueError(f"cannot detect serde kind for {type(obj).__name__}")


def to_json(obj: Any, indent: int = None) -> str:
    """Serialize a supported layout dataclass (or DeviceSizing dict) to JSON string."""
    kind = _detect_kind(obj)
    serializer, _, _ = _DISPATCH[kind]
    return json.dumps(serializer(obj), indent=indent)


def from_json(s: str, kind: str) -> Any:
    """Deserialize a JSON string. ``kind`` selects the target type.

    Supported kinds: ``symbolic_layout``, ``annotated_cell``,
    ``parameterized_library``, ``device_sizing``.
    """
    if kind not in _DISPATCH:
        raise ValueError(
            f"unknown serde kind: {kind!r}; expected one of "
            f"{sorted(_DISPATCH.keys())}"
        )
    _, deserializer, _ = _DISPATCH[kind]
    return deserializer(json.loads(s))
