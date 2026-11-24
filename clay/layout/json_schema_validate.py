"""JSON schema validator for Phase 17 serialized layout JSON.

Validates the *shape* of ``dict`` payloads before they are handed to the
``json_serde`` deserializers. The goal is to fail fast with a precise
JSON path (e.g. ``"cells[0].polygons[1].layer"``) when an upstream tool
produces a malformed document, rather than letting a generic ``KeyError``
or ``TypeError`` bubble out of the deserializer.

This module is dependency-free pure Python (no ``jsonschema``). It mirrors
the four top-level shapes produced by ``json_serde``:

* ``SymbolicLayout``        -> ``validate_symbolic_layout_schema``
* ``AnnotatedCell``         -> ``validate_annotated_cell_schema``
* ``ParameterizedLibrary``  -> ``validate_parameterized_library_schema``
* ``DeviceSizing``          -> ``validate_device_sizing_schema``

Each validator returns a ``SchemaValidationResult``. Errors are *collected*
(not raised) so callers can surface every problem at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, List


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SchemaValidationError:
    """A single schema-validation problem, with a JSON path locator."""

    path: str
    message: str


@dataclass
class SchemaValidationResult:
    """Aggregate validation outcome — ``valid`` is ``False`` iff errors is non-empty."""

    valid: bool
    errors: List[SchemaValidationError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_dict(value: Any, path: str, errors: List[SchemaValidationError]) -> bool:
    if not isinstance(value, dict):
        errors.append(
            SchemaValidationError(
                path=path or "$",
                message=f"expected object/dict, got {_type_name(value)}",
            )
        )
        return False
    return True


def _check_required_keys(
    obj: dict,
    required: List[str],
    path: str,
    errors: List[SchemaValidationError],
) -> List[str]:
    """Return the subset of ``required`` that were present."""
    present: List[str] = []
    for key in required:
        if key not in obj:
            errors.append(
                SchemaValidationError(
                    path=f"{path}.{key}" if path else key,
                    message=f"missing required field {key!r}",
                )
            )
        else:
            present.append(key)
    return present


def _check_int(
    value: Any,
    path: str,
    errors: List[SchemaValidationError],
    *,
    minimum: int | None = None,
) -> bool:
    # bool is a subclass of int in Python — reject it explicitly so that
    # ``True``/``False`` cannot masquerade as a finger count.
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected int, got {_type_name(value)}",
            )
        )
        return False
    if minimum is not None and value < minimum:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected int >= {minimum}, got {value}",
            )
        )
        return False
    return True


def _check_str(
    value: Any, path: str, errors: List[SchemaValidationError]
) -> bool:
    if not isinstance(value, str):
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected str, got {_type_name(value)}",
            )
        )
        return False
    return True


def _check_fraction_str(
    value: Any, path: str, errors: List[SchemaValidationError]
) -> bool:
    if not _check_str(value, path, errors):
        return False
    if "/" not in value:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected 'num/den' Fraction string, got {value!r}",
            )
        )
        return False
    num_s, _, den_s = value.partition("/")
    try:
        Fraction(int(num_s), int(den_s))
    except (ValueError, ZeroDivisionError) as exc:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"invalid Fraction string {value!r}: {exc}",
            )
        )
        return False
    return True


def _check_number(
    value: Any, path: str, errors: List[SchemaValidationError]
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected number, got {_type_name(value)}",
            )
        )
        return False
    return True


def _check_list(
    value: Any, path: str, errors: List[SchemaValidationError]
) -> bool:
    if not isinstance(value, list):
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected list, got {_type_name(value)}",
            )
        )
        return False
    return True


# ---------------------------------------------------------------------------
# SymbolicLayout
# ---------------------------------------------------------------------------


def _validate_symbolic_layout(
    data: Any, path: str, errors: List[SchemaValidationError]
) -> None:
    if not _check_dict(data, path, errors):
        return
    present = _check_required_keys(
        data,
        ["finger_count", "pitch_multiplier", "stripe_count"],
        path,
        errors,
    )
    if "finger_count" in present:
        _check_int(
            data["finger_count"],
            f"{path}.finger_count" if path else "finger_count",
            errors,
            minimum=1,
        )
    if "pitch_multiplier" in present:
        _check_fraction_str(
            data["pitch_multiplier"],
            f"{path}.pitch_multiplier" if path else "pitch_multiplier",
            errors,
        )
    if "stripe_count" in present:
        _check_int(
            data["stripe_count"],
            f"{path}.stripe_count" if path else "stripe_count",
            errors,
            minimum=1,
        )


def validate_symbolic_layout_schema(data: dict) -> SchemaValidationResult:
    """Validate a serialized ``SymbolicLayout`` dict.

    Expected shape:
        ``{finger_count: int >= 1, pitch_multiplier: str (Fraction), stripe_count: int >= 1}``
    """
    errors: List[SchemaValidationError] = []
    _validate_symbolic_layout(data, "", errors)
    return SchemaValidationResult(valid=not errors, errors=errors)


# ---------------------------------------------------------------------------
# AnnotatedCell
# ---------------------------------------------------------------------------

# Annotation kind -> required keys (besides ``kind``)
_ANNOTATION_REQUIRED = {
    "matching": ["other", "match_kind"],
    "guard_ring": ["well_type"],
    "well_isolated": ["from_device"],
    "symmetric": ["devices", "axis"],
    "interdigitated": ["with_device", "pattern", "finger_pairs"],
    "adjacent": ["to", "max_separation_multiplier"],
}


def _validate_annotation(
    ann: Any, path: str, errors: List[SchemaValidationError]
) -> None:
    if not _check_dict(ann, path, errors):
        return
    if "kind" not in ann:
        errors.append(
            SchemaValidationError(
                path=f"{path}.kind",
                message="missing required field 'kind'",
            )
        )
        return
    kind = ann["kind"]
    if not isinstance(kind, str) or kind not in _ANNOTATION_REQUIRED:
        errors.append(
            SchemaValidationError(
                path=f"{path}.kind",
                message=(
                    f"unknown annotation kind {kind!r}; expected one of "
                    f"{sorted(_ANNOTATION_REQUIRED.keys())}"
                ),
            )
        )
        return
    required = _ANNOTATION_REQUIRED[kind]
    for key in required:
        if key not in ann:
            errors.append(
                SchemaValidationError(
                    path=f"{path}.{key}",
                    message=f"missing required field {key!r} for kind={kind!r}",
                )
            )
    # Light per-kind type checks (only on fields that survived the presence check).
    if kind == "interdigitated" and "finger_pairs" in ann:
        _check_int(ann["finger_pairs"], f"{path}.finger_pairs", errors, minimum=0)
    if kind == "adjacent" and "max_separation_multiplier" in ann:
        _check_fraction_str(
            ann["max_separation_multiplier"],
            f"{path}.max_separation_multiplier",
            errors,
        )
    if kind == "symmetric" and "devices" in ann:
        if _check_list(ann["devices"], f"{path}.devices", errors):
            for i, dev in enumerate(ann["devices"]):
                _check_str(dev, f"{path}.devices[{i}]", errors)


def validate_annotated_cell_schema(data: dict) -> SchemaValidationResult:
    """Validate a serialized ``AnnotatedCell`` dict.

    Expected shape:
        ``{cell_name: str, layouts: dict[str, SymbolicLayout], annotations: dict[str, list]}``
    """
    errors: List[SchemaValidationError] = []
    if not _check_dict(data, "", errors):
        return SchemaValidationResult(valid=False, errors=errors)

    present = _check_required_keys(
        data, ["cell_name", "layouts", "annotations"], "", errors
    )

    if "cell_name" in present:
        _check_str(data["cell_name"], "cell_name", errors)

    if "layouts" in present:
        if _check_dict(data["layouts"], "layouts", errors):
            for dev, layout_d in data["layouts"].items():
                _validate_symbolic_layout(
                    layout_d, f"layouts.{dev}", errors
                )

    if "annotations" in present:
        if _check_dict(data["annotations"], "annotations", errors):
            for dev, anns in data["annotations"].items():
                ann_path = f"annotations.{dev}"
                if _check_list(anns, ann_path, errors):
                    for i, ann in enumerate(anns):
                        _validate_annotation(ann, f"{ann_path}[{i}]", errors)

    return SchemaValidationResult(valid=not errors, errors=errors)


# ---------------------------------------------------------------------------
# ParameterizedLibrary
# ---------------------------------------------------------------------------


def _validate_point(
    value: Any, path: str, errors: List[SchemaValidationError]
) -> None:
    if not _check_list(value, path, errors):
        return
    if len(value) != 2:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected [x, y] of length 2, got length {len(value)}",
            )
        )
        return
    for i, comp in enumerate(value):
        if isinstance(comp, bool) or not isinstance(comp, int):
            errors.append(
                SchemaValidationError(
                    path=f"{path}[{i}]",
                    message=f"expected int, got {_type_name(comp)}",
                )
            )


def _validate_polygon(
    poly: Any, path: str, errors: List[SchemaValidationError]
) -> None:
    if not _check_dict(poly, path, errors):
        return
    if "vertices" not in poly:
        errors.append(
            SchemaValidationError(
                path=f"{path}.vertices",
                message="missing required field 'vertices'",
            )
        )
        return
    verts = poly["vertices"]
    if _check_list(verts, f"{path}.vertices", errors):
        for i, v in enumerate(verts):
            _validate_point(v, f"{path}.vertices[{i}]", errors)


def _validate_layer_polygon(
    lp: Any, path: str, errors: List[SchemaValidationError]
) -> None:
    if not _check_dict(lp, path, errors):
        return
    present = _check_required_keys(
        lp, ["layer", "datatype", "polygon"], path, errors
    )
    if "layer" in present:
        _check_int(lp["layer"], f"{path}.layer", errors)
    if "datatype" in present:
        _check_int(lp["datatype"], f"{path}.datatype", errors)
    if "polygon" in present:
        _validate_polygon(lp["polygon"], f"{path}.polygon", errors)


def _validate_named_cell(
    cell: Any, path: str, errors: List[SchemaValidationError]
) -> None:
    if not _check_dict(cell, path, errors):
        return
    present = _check_required_keys(cell, ["name", "polygons"], path, errors)
    if "name" in present:
        _check_str(cell["name"], f"{path}.name", errors)
    if "polygons" in present:
        polys = cell["polygons"]
        if _check_list(polys, f"{path}.polygons", errors):
            for i, lp in enumerate(polys):
                _validate_layer_polygon(lp, f"{path}.polygons[{i}]", errors)


def validate_parameterized_library_schema(data: dict) -> SchemaValidationResult:
    """Validate a serialized ``ParameterizedLibrary`` dict.

    Expected shape:
        ``{name: str, unit_user: number, unit_db_meters: number,
           cells: list of {name: str, polygons: list of {layer, datatype, polygon}}}``
    """
    errors: List[SchemaValidationError] = []
    if not _check_dict(data, "", errors):
        return SchemaValidationResult(valid=False, errors=errors)

    present = _check_required_keys(
        data,
        ["name", "unit_user", "unit_db_meters", "cells"],
        "",
        errors,
    )
    if "name" in present:
        _check_str(data["name"], "name", errors)
    if "unit_user" in present:
        _check_number(data["unit_user"], "unit_user", errors)
    if "unit_db_meters" in present:
        _check_number(data["unit_db_meters"], "unit_db_meters", errors)
    if "cells" in present:
        cells = data["cells"]
        if _check_list(cells, "cells", errors):
            for i, cell in enumerate(cells):
                _validate_named_cell(cell, f"cells[{i}]", errors)

    return SchemaValidationResult(valid=not errors, errors=errors)


# ---------------------------------------------------------------------------
# DeviceSizing
# ---------------------------------------------------------------------------


def validate_device_sizing_schema(data: dict) -> SchemaValidationResult:
    """Validate a serialized ``DeviceSizing`` dict.

    Expected shape:
        ``{device_name: [w_um, l_um]}`` — a dict whose values are length-2
        lists of numbers.
    """
    errors: List[SchemaValidationError] = []
    if not _check_dict(data, "", errors):
        return SchemaValidationResult(valid=False, errors=errors)
    for dev, wl in data.items():
        if not isinstance(dev, str):
            errors.append(
                SchemaValidationError(
                    path=f"{dev!r}",
                    message=f"expected str device key, got {_type_name(dev)}",
                )
            )
            continue
        path = dev
        if not _check_list(wl, path, errors):
            continue
        if len(wl) != 2:
            errors.append(
                SchemaValidationError(
                    path=path,
                    message=f"expected [w, l] of length 2, got length {len(wl)}",
                )
            )
            continue
        _check_number(wl[0], f"{path}[0]", errors)
        _check_number(wl[1], f"{path}[1]", errors)
    return SchemaValidationResult(valid=not errors, errors=errors)


__all__ = [
    "SchemaValidationError",
    "SchemaValidationResult",
    "validate_symbolic_layout_schema",
    "validate_annotated_cell_schema",
    "validate_parameterized_library_schema",
    "validate_device_sizing_schema",
]
