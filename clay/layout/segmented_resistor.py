"""PDK-bound straight-segment resistor materialization.

This module mirrors the coordinate-free topology of the installed external_pdk
``npolyf_u``/``ppolyf_u`` PCell.  It intentionally does not guess a segment count
from resistance: the caller must provide the typed source declaration.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional

from .backend import LayerPolygon, Polygon
from .device_cell import DeviceGeom, Terminal


_REQUIRED_RULE_KEYS = frozenset({
    "grid_nm",
    "segment_space_nm",
    "contact_size_nm",
    "cell_side_margin_nm",
    "cell_end_margin_nm",
    "poly_head_outer_margin_nm",
    "poly_head_inner_margin_nm",
    "contact_min_side_margin_nm",
    "contact_min_space_nm",
    "contact_outer_margin_nm",
    "m1_side_margin_nm",
    "m1_outer_margin_nm",
    "m1_head_height_nm",
    "res_marker_side_margin_nm",
    "implant_side_margin_nm",
    "pcell_layout_sha256",
    "pcell_layout_path",
    "pcell_ivpcell_sha256",
    "pcell_ivpcell_path",
    "pcell_p_layout_sha256",
    "pcell_p_layout_path",
    "pcell_p_ivpcell_sha256",
    "pcell_p_ivpcell_path",
    "pcell_manual_sha256",
    "pcell_manual_path",
    "foundry_drc_sha256",
    "foundry_drc_path",
    "provenance",
})
_INTEGER_RULE_KEYS = frozenset({
    "grid_nm",
    "segment_space_nm",
    "contact_size_nm",
    "cell_side_margin_nm",
    "cell_end_margin_nm",
    "poly_head_outer_margin_nm",
    "poly_head_inner_margin_nm",
    "contact_min_side_margin_nm",
    "contact_min_space_nm",
    "contact_outer_margin_nm",
    "m1_side_margin_nm",
    "m1_outer_margin_nm",
    "m1_head_height_nm",
    "res_marker_side_margin_nm",
    "implant_side_margin_nm",
})
_SHA_KEYS = frozenset({
    "pcell_layout_sha256",
    "pcell_ivpcell_sha256",
    "pcell_p_layout_sha256",
    "pcell_p_ivpcell_sha256",
    "pcell_manual_sha256",
    "foundry_drc_sha256",
})
_BOUND_FILES = (
    ("pcell_layout_path", "pcell_layout_sha256"),
    ("pcell_ivpcell_path", "pcell_ivpcell_sha256"),
    ("pcell_p_layout_path", "pcell_p_layout_sha256"),
    ("pcell_p_ivpcell_path", "pcell_p_ivpcell_sha256"),
    ("pcell_manual_path", "pcell_manual_sha256"),
    ("foundry_drc_path", "foundry_drc_sha256"),
)
_N_MODELS = frozenset({"npolyf_u"})
_P_MODELS = frozenset({"ppolyf_u"})
_SUPPORTED_MODELS = _N_MODELS | _P_MODELS
_MODELS_BY_PDK = {
    "gf180mcu": frozenset({"npolyf_u", "ppolyf_u"}),
}
_MAX_SEGMENTS = 4096


def _migrated_rules(tech) -> Mapping[str, object]:
    """Straight-unit topology using the target process's own layers and rules.

    This is not a claim of an OA-PCell clone. The open_pdk authority path below
    remains unchanged. external_pdk RPO.C.2/C.5 and external_pdk GR_2 require 220 nm body/contact
    separation; GF180 LRES.7 requires the same. Side margins exceed external_pdk's
    220 nm RPO.C.5 and GF180's 280 nm LRES.6. All contacts and poly/metal rules
    come from the resolved target technology, not the source process.
    """
    pdk = str(tech.pdk_id)
    if pdk not in {"gf180mcu", "sky130"}:
        raise ValueError(f"segmented resistor materializer is unsupported on PDK {pdk!r}")
    poly_name = "poly2" if pdk == "gf180mcu" else "poly"
    contact = int(tech.via_size("contact"))
    poly_enc = int(tech.enclosure("contact", poly_name) or 0)
    metal_enc = int(tech.enclosure("contact", "metal1") or 0)
    poly_enc = max(poly_enc, 100)
    metal_enc = max(metal_enc, 60)
    grid = 5
    up = lambda value: ((int(value) + grid - 1) // grid) * grid
    poly_enc, metal_enc = up(poly_enc), up(metal_enc)
    side_contact = max(poly_enc, metal_enc + grid)
    poly_outer = 300
    contact_outer = poly_outer + poly_enc
    m1_side = side_contact - metal_enc
    m1_height = up(max(contact + 2 * metal_enc, tech.min_width("metal1")))
    # GF's resistor recognition has a stricter spacing than ordinary MOS poly:
    # native n+/p+_poly_resistor.drc LRES.2/PRES.2 require 400 nm, not 240 nm.
    resistor_space = 400 if pdk == "gf180mcu" else 0
    return {
        "grid_nm": grid,
        "segment_space_nm": up(max(resistor_space, tech.min_space(poly_name),
                                    tech.min_space("metal1") - 2*m1_side)),
        "contact_size_nm": contact,
        "cell_side_margin_nm": 600,
        "cell_end_margin_nm": contact_outer + contact + 220,
        "poly_head_outer_margin_nm": poly_outer,
        "poly_head_inner_margin_nm": contact_outer + contact + poly_enc,
        "contact_min_side_margin_nm": side_contact,
        "contact_min_space_nm": 250,
        "contact_outer_margin_nm": contact_outer,
        "m1_side_margin_nm": m1_side,
        "m1_outer_margin_nm": contact_outer - metal_enc,
        "m1_head_height_nm": m1_height,
        "res_marker_side_margin_nm": 50,
        "implant_side_margin_nm": 100,
    }


def _rules(tech) -> Mapping[str, object]:
    return _migrated_rules(tech)


def _positive_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _snap_exact_um(value_um: float, *, grid_nm: int, label: str) -> int:
    raw_nm = value_um * 1000.0
    rounded = int(round(raw_nm))
    if not math.isclose(raw_nm, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} must resolve to an exact integer nm")
    if rounded <= 0 or rounded % grid_nm:
        raise ValueError(f"{label} must be positive and on the PDK grid")
    return rounded


def build_smic_segmented_resistor_r0(
    tech,
    value_ohms: float,
    *,
    count: int,
    connection: str,
    segment_width_um: float,
    segment_length_um: float,
    segment_resistance_ohms: float,
    value_quantum_ohms: float,
    model: str,
    segment_resistance_quantum_ohms: float = 0.0,
    nets: Optional[Dict[str, str]] = None,
) -> DeviceGeom:
    """Build one unrotated external_pdk straight-segment resistor PCell equivalent."""

    rules = _rules(tech)
    if type(count) is not int or count <= 0 or count > _MAX_SEGMENTS:
        raise ValueError(f"count must be an exact integer in 1..{_MAX_SEGMENTS}")
    if type(connection) is not str or connection not in {"series", "parallel"}:
        raise ValueError("connection must be exactly 'series' or 'parallel'")
    if type(model) is not str or model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"model must be one of {sorted(_SUPPORTED_MODELS)}, got {model!r}"
        )
    if model not in _MODELS_BY_PDK[str(tech.pdk_id)]:
        raise ValueError(f"resistor model {model!r} is not native to {tech.pdk_id}")
    value = _positive_finite(value_ohms, label="value_ohms")
    segment_resistance = _positive_finite(
        segment_resistance_ohms, label="segment_resistance_ohms"
    )
    value_quantum = _positive_finite(
        value_quantum_ohms, label="value_quantum_ohms"
    )
    if (
        isinstance(segment_resistance_quantum_ohms, bool)
        or not isinstance(segment_resistance_quantum_ohms, (int, float))
        or not math.isfinite(float(segment_resistance_quantum_ohms))
        or float(segment_resistance_quantum_ohms) < 0.0
    ):
        raise ValueError(
            "segment_resistance_quantum_ohms must be a non-negative finite number"
        )
    segment_quantum = float(segment_resistance_quantum_ohms)
    expected = (
        segment_resistance * count
        if connection == "series"
        else segment_resistance / count
    )
    interval_radius = value_quantum / 2.0 + (
        segment_quantum * count / 2.0
        if connection == "series"
        else segment_quantum / count / 2.0
    )
    if not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=interval_radius,
    ):
        raise ValueError(
            f"declared equivalent resistance {expected:.12g} does not match "
            f"value {value:.12g} within source quantization intervals "
            f"value={value_quantum:.12g}, segment={segment_quantum:.12g}"
        )

    grid = int(rules["grid_nm"])
    width = _snap_exact_um(
        _positive_finite(segment_width_um, label="segment_width_um"),
        grid_nm=grid,
        label="segment_width_um",
    )
    length = _snap_exact_um(
        _positive_finite(segment_length_um, label="segment_length_um"),
        grid_nm=grid,
        label="segment_length_um",
    )
    if str(tech.pdk_id) == "gf180mcu" and width < 800:
        raise ValueError("GF180 LRES.1 requires segmented resistor width >= 0.8 um")
    spacing = int(rules["segment_space_nm"])
    contact = int(rules["contact_size_nm"])
    cell_side = int(rules["cell_side_margin_nm"])
    cell_end = int(rules["cell_end_margin_nm"])
    poly_outer = int(rules["poly_head_outer_margin_nm"])
    poly_inner = int(rules["poly_head_inner_margin_nm"])
    contact_side = int(rules["contact_min_side_margin_nm"])
    contact_space = int(rules["contact_min_space_nm"])
    contact_outer = int(rules["contact_outer_margin_nm"])
    m1_side = int(rules["m1_side_margin_nm"])
    m1_outer = int(rules["m1_outer_margin_nm"])
    m1_height = int(rules["m1_head_height_nm"])
    marker_side = int(rules["res_marker_side_margin_nm"])
    implant_side = int(rules["implant_side_margin_nm"])
    pitch = width + spacing
    cell_width = 2 * cell_side + count * width + (count - 1) * spacing
    cell_height = length + 2 * cell_end

    if not (0 < poly_outer < poly_inner < cell_end):
        raise ValueError("segmented resistor poly end authority is inconsistent")
    if contact_outer + contact > poly_inner:
        raise ValueError("segmented resistor contact exceeds its poly head")
    if m1_outer > contact_outer or m1_outer + m1_height < contact_outer + contact:
        raise ValueError("segmented resistor M1 head does not enclose contacts")
    if marker_side >= cell_side or implant_side >= cell_side:
        raise ValueError("segmented resistor envelope side authority is inconsistent")
    contact_capacity = (
        width - 2 * contact_side + contact_space
    ) // (contact + contact_space)
    if contact_capacity < 1:
        raise ValueError("segment width cannot carry a proven head contact")
    if 2 * m1_side >= width:
        raise ValueError("segment width cannot carry the proven M1 head")

    poly = tech.build_layer("poly")
    contact_key = tech.contact_key()
    metal1 = tech.metal_key(1)
    sab = tech.marker_key("sab")
    res_mk = tech.marker_key("res_mk")
    implant_name = "nplus" if model in _N_MODELS else "pplus"
    implant = tech.build_layer(implant_name)
    keys = {
        "poly": poly,
        "contact": contact_key,
        "metal1": metal1,
        "sab": sab,
        "res_mk": res_mk,
        "implant": implant,
    }
    if any(
        type(key) is not tuple
        or len(key) != 2
        or any(type(value) is not int for value in key)
        for key in keys.values()
    ):
        raise ValueError(f"open_pdk segmented resistor layer authority incomplete: {keys}")

    polygons: list[LayerPolygon] = []

    def rect(key, box: tuple[int, int, int, int]) -> None:
        if (
            type(key) is not tuple
            or len(key) != 2
            or any(type(value) is not int for value in key)
        ):
            raise ValueError("segmented resistor layer key must be one pair")
        if any(type(value) is not int or value % grid for value in box):
            raise ValueError(f"segmented resistor box is off-grid: {box!r}")
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"segmented resistor box has non-positive extent: {box!r}")
        polygons.append(
            LayerPolygon(
                layer=int(key[0]),
                datatype=int(key[1]),
                polygon=Polygon.rectangle(x0, y0, x1, y1),
            )
        )

    head_pads: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []

    for index in range(count):
        x0 = cell_side + index * pitch
        x1 = x0 + width
        rect(poly, (x0, poly_outer, x1, poly_inner))
        rect(poly, (x0, poly_inner, x1, cell_height - poly_inner))
        rect(poly, (x0, cell_height - poly_inner, x1, cell_height - poly_outer))

        if contact_capacity == 1:
            cut_xs = (x0 + (width - contact) // 2,)
        else:
            residual = width - 2 * contact_side - contact_capacity * contact
            if residual < (contact_capacity - 1) * contact_space:
                raise ValueError("segmented resistor contact capacity is inconsistent")
            if residual % (contact_capacity - 1):
                raise ValueError("segmented resistor contact array is off-grid")
            distributed_space = residual // (contact_capacity - 1)
            cut_xs = tuple(
                x0 + contact_side + index * (contact + distributed_space)
                for index in range(contact_capacity)
            )
        for cut_x0 in cut_xs:
            rect(
                contact_key,
                (cut_x0, contact_outer, cut_x0 + contact, contact_outer + contact),
            )
            rect(
                contact_key,
                (
                    cut_x0,
                    cell_height - contact_outer - contact,
                    cut_x0 + contact,
                    cell_height - contact_outer,
                ),
            )
        bottom_pad = (
            x0 + m1_side,
            m1_outer,
            x1 - m1_side,
            m1_outer + m1_height,
        )
        top_pad = (
            x0 + m1_side,
            cell_height - m1_outer - m1_height,
            x1 - m1_side,
            cell_height - m1_outer,
        )
        rect(metal1, bottom_pad)
        rect(metal1, top_pad)
        head_pads.append((bottom_pad, top_pad))

    rect(sab, (0, cell_end, cell_width, cell_height - cell_end))
    rect(
        res_mk,
        (marker_side, cell_end, cell_width - marker_side, cell_height - cell_end),
    )
    implant_box = (
        implant_side,
        0,
        cell_width - implant_side,
        cell_height,
    )
    rect(implant, implant_box)

    if connection == "parallel":
        bottom_bus = (
            head_pads[0][0][0],
            head_pads[0][0][1],
            head_pads[-1][0][2],
            head_pads[0][0][3],
        )
        top_bus = (
            head_pads[0][1][0],
            head_pads[0][1][1],
            head_pads[-1][1][2],
            head_pads[0][1][3],
        )
        rect(metal1, bottom_bus)
        rect(metal1, top_bus)
        p_pad, n_pad = bottom_bus, top_bus
    else:
        for index in range(count - 1):
            side = 1 if index % 2 == 0 else 0
            left = head_pads[index][side]
            right = head_pads[index + 1][side]
            connector = (left[2], left[1], right[0], left[3])
            rect(metal1, connector)
        p_pad = head_pads[0][0]
        n_pad = head_pads[-1][0 if count % 2 == 0 else 1]

    nets = nets or {}
    return DeviceGeom(
        polygons=polygons,
        terminals={
            "p": Terminal(role="p", bbox=p_pad, net=str(nets.get("p", ""))),
            "n": Terminal(role="n", bbox=n_pad, net=str(nets.get("n", ""))),
        },
        kind="res",
        finger_count=count,
        gate_length_nm=length,
        channel_width_nm=width,
        comp_bbox=implant_box,
        orient="R0",
        source_rail_bbox=p_pad,
        drain_rail_bbox=n_pad,
    )


__all__ = ["build_smic_segmented_resistor_r0"]
