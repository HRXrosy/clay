"""Pure SKY130 native PCell input contract; execution is imported lazily."""
from decimal import Decimal
import sys

from clay.pdk.geometry import (GeometryInputError, GeometryUnsupportedError, NativeGeometryProvider,
                               NativePrimitiveObservationCapability)
from clay.pdk.native_spice import NativeSpiceFamily


class NativeSky130GeometryError(GeometryInputError):
    pass


class NativeSky130GeometryUnsupported(GeometryUnsupportedError, NativeSky130GeometryError):
    pass


SKY130_MOS_RULESET_NM = {
    "grid_nm": 5,
    "gate_min_l_nm": 150,
    "gate_min_w_nm": 420,
    "gate_to_diff_licon_nm": 60,
    "licon_size_nm": 170,
    "licon_space_nm": 170,
    "diff_enclose_licon_side_nm": 40,
    "diff_enclose_licon_end_nm": 60,
    "poly_enclose_licon_side_nm": 50,
    "poly_enclose_licon_end_nm": 80,
    "poly_space_nm": 210,
    "gate_poly_extension_nm": 130,
    "gate_to_poly_licon_nm": 55,
    "npc_enclose_poly_licon_nm": 100,
    "li1_width_nm": 170,
    "li1_space_nm": 170,
    "li1_enclose_licon_end_nm": 80,
    "li1_min_area_nm2": 56100,
    "mcon_size_nm": 170,
    "mcon_space_nm": 190,
    "m1_width_nm": 140,
    "m1_space_nm": 140,
    "m1_enclose_mcon_side_nm": 30,
    "m1_enclose_mcon_end_nm": 60,
    "m1_enclose_mcon_sym_nm": 45,
    "m1_min_area_nm2": 83000,
    "tap_min_width_nm": 170,
    "tap_enclose_licon_end_nm": 120,
    "diff_to_tap_space_nm": 270,
    "diff_to_opposite_tap_space_nm": 400,
    "sdm_enclose_diff_nm": 125,
    "sdm_enclose_tap_nm": 130,
    "sdm_space_nm": 380,
    "nwell_enclose_diff_nm": 180,
    "nwell_enclose_tap_nm": 180,
    "nwell_enclose_lvtn_nm": 180,
    "lvtn_enclose_gate_nm": 180,
    "lvtn_to_non_lvt_fet_nm": 360,
    "capm_min_width_nm": 1000,
    "capm_space_nm": 840,
    "m3_enclose_capm_nm": 140,
    "capm_to_unrelated_m3_nm": 1340,
    "capm_to_via2_nm": 140,
    "capm_stacked_m3_surround_nm": 1400,
}


def sky130_mos_ruleset_nm():
    """Return a copy of the canonical SKY130 FEOL/MEOL/CAPM ruleset in integer nm."""
    return dict(SKY130_MOS_RULESET_NM)


def snap_up_grid_nm(value_nm, grid_nm=5):
    """Snap a non-negative integer nanometre quantity up to the manufacturing grid."""
    _require(type(value_nm) is int and type(grid_nm) is int and grid_nm > 0,
             "grid snap requires positive integer nanometre arguments")
    return ((value_nm + grid_nm - 1) // grid_nm) * grid_nm


def derive_column_pitches_nm(l_gate_nm=150, *, rules=None):
    """Derive SKY130 MOS column pitches from foundry design rules (no hardcoded tables).

    Returns exact 5nm-grid pitches for:
      * ``li1_only_nm``: internal/supply S/D column contacted on ``licon + li1`` only
        (``L + 2 * gate_to_diff_licon + licon_size = 440 nm`` at ``L = 150 nm``);
      * ``sd_mcon_nm``: S/D column carrying an ``mcon + M1`` pad meeting ``met1.6``
        minimum area (``320 + 140 = 460 nm`` at ``L = 150 nm``);
      * ``poly_contacted_nm``: adjacent unmerged gate ``poly`` contact heads
        (``licon_size + 2 * poly_enclose_licon_side + poly_space = 480 nm``).
    """
    r = SKY130_MOS_RULESET_NM if rules is None else rules
    grid = int(r["grid_nm"])
    _require(type(l_gate_nm) is int and l_gate_nm >= int(r["gate_min_l_nm"]) and l_gate_nm % grid == 0,
             "SKY130 gate length must be on the 5nm grid and >= minimum gate length")
    li1_only = snap_up_grid_nm(
        l_gate_nm + 2 * int(r["gate_to_diff_licon_nm"]) + int(r["licon_size_nm"]),
        grid,
    )
    if (li1_only - l_gate_nm) % (2 * grid) != 0:
        li1_only += grid
    m1_pad_h = int(r["mcon_size_nm"]) + 2 * int(r["m1_enclose_mcon_sym_nm"])
    min_w_from_mcon = int(r["mcon_size_nm"]) + 2 * int(r["m1_enclose_mcon_end_nm"])
    min_w_from_area = snap_up_grid_nm(
        (int(r["m1_min_area_nm2"]) + m1_pad_h - 1) // m1_pad_h,
        grid,
    )
    m1_pad_w = max(min_w_from_mcon, min_w_from_area)
    sd_mcon = max(li1_only, snap_up_grid_nm(m1_pad_w + int(r["m1_space_nm"]), grid))
    if (sd_mcon - l_gate_nm) % (2 * grid) != 0:
        sd_mcon += grid
    poly_pad_w = int(r["licon_size_nm"]) + 2 * int(r["poly_enclose_licon_side_nm"])
    poly_contacted = max(sd_mcon, snap_up_grid_nm(poly_pad_w + int(r["poly_space_nm"]), grid))
    if (poly_contacted - l_gate_nm) % (2 * grid) != 0:
        poly_contacted += grid
    return {
        "grid_nm": grid,
        "l_gate_nm": l_gate_nm,
        "li1_only_nm": li1_only,
        "sd_mcon_nm": sd_mcon,
        "poly_contacted_nm": poly_contacted,
        "m1_pad_w_nm": m1_pad_w,
        "m1_pad_h_nm": m1_pad_h,
        "poly_pad_w_nm": poly_pad_w,
    }


def sd_licon_layout_nm(w_finger_nm, *, y_bottom_nm=0, rules=None):
    """Compute vertical S/D ``licon`` cut count and ``(y0, y1)`` spans for arbitrary ``W_finger``."""
    r = SKY130_MOS_RULESET_NM if rules is None else rules
    grid = int(r["grid_nm"])
    _require(type(w_finger_nm) is int and w_finger_nm >= int(r["gate_min_w_nm"]) and w_finger_nm % grid == 0,
             "SKY130 finger width must be on the 5nm grid and >= 420nm")
    cut = int(r["licon_size_nm"])
    space = int(r["licon_space_nm"])
    end_enc = int(r["diff_enclose_licon_end_nm"])
    pitch = cut + space
    usable = w_finger_nm - 2 * end_enc
    _require(usable >= cut, "SKY130 finger width cannot enclose one S/D licon cut")
    count = max(1, (usable + space) // pitch)
    span = count * cut + (count - 1) * space
    offset = ((w_finger_nm - span) // 2 // grid) * grid
    if offset < end_enc:
        offset = snap_up_grid_nm(end_enc, grid)
    cuts = tuple(
        (y_bottom_nm + offset + idx * pitch, y_bottom_nm + offset + idx * pitch + cut)
        for idx in range(count)
    )
    return count, cuts


def tap_rail_layout_nm(x_left_nm, x_right_nm, *, active_bottom_nm, active_top_nm, side, gap_nm=None, rules=None):
    """Derive a width-aligned horizontal ``tap + licon + li1`` substrate/well bar with 0 X overhang."""
    r = SKY130_MOS_RULESET_NM if rules is None else rules
    grid = int(r["grid_nm"])
    _require(side in ("top", "bottom"), "tap rail side must be 'top' or 'bottom'")
    _require(type(x_left_nm) is int and type(x_right_nm) is int and x_right_nm > x_left_nm,
             "tap rail X span must be a non-empty integer interval")
    width = x_right_nm - x_left_nm
    cut = int(r["licon_size_nm"])
    space = int(r["licon_space_nm"])
    end_enc = int(r["tap_enclose_licon_end_nm"])
    _require(width >= cut + 2 * end_enc, "active width is narrower than minimum horizontal tap licon enclosure")
    sep = int(r["diff_to_opposite_tap_space_nm"]) if gap_nm is None else int(gap_nm)
    _require(sep >= int(r["diff_to_tap_space_nm"]) and sep % grid == 0,
             "tap rail separation violates SKY130 diff/tap spacing")
    bar_h = cut + 2 * int(r["m1_enclose_mcon_sym_nm"])
    if side == "top":
        y0 = active_top_nm + sep
        y1 = y0 + bar_h
    else:
        y1 = active_bottom_nm - sep
        y0 = y1 - bar_h
    pitch = cut + space
    usable = width - 2 * end_enc
    count = max(1, (usable + space) // pitch)
    span = count * cut + (count - 1) * space
    x_offset = ((width - span) // 2 // grid) * grid
    if x_offset < end_enc:
        x_offset = snap_up_grid_nm(end_enc, grid)
    yc0 = y0 + ((bar_h - cut) // 2 // grid) * grid
    licon_boxes = tuple(
        (x_left_nm + x_offset + idx * pitch, yc0, x_left_nm + x_offset + idx * pitch + cut, yc0 + cut)
        for idx in range(count)
    )
    return {
        "tap_box_nm": (x_left_nm, y0, x_right_nm, y1),
        "li1_box_nm": (x_left_nm, y0, x_right_nm, y1),
        "licon_boxes_nm": licon_boxes,
        "licon_count": count,
    }


def mim_vertical_stack_m3_surround_nm(*, rules=None):
    """Minimum M3 bottom-plate enclosure around CAPM for concentric vertical passive stacks.

    Ensures ``M3`` simultaneously satisfies ``capm.3`` (``>= 140 nm`` enclosure) and
    keeps every external/unrelated ``M3`` outside ``capm.11`` (``>= 1340 nm`` halo)
    with one solid un-notched ``M3`` plate (``0`` ``capm.2b_a`` notch errors).
    """
    r = SKY130_MOS_RULESET_NM if rules is None else rules
    grid = int(r["grid_nm"])
    required = max(
        int(r["m3_enclose_capm_nm"]),
        int(r["capm_to_unrelated_m3_nm"]) + 12 * grid,
        int(r["capm_stacked_m3_surround_nm"]),
    )
    return snap_up_grid_nm(required, grid)


def stacked_mim_surround_selections(cell, *, specs=None):
    """Detect MIM capacitors concentrically stacked over lower-metal passives via ``@align``."""
    by_pair_edges = {}
    for decl in getattr(cell, "layout_intents", ()) or ():
        if getattr(decl, "kind", "") != "align":
            continue
        attrs = getattr(decl, "attributes", None) or {}
        nets = getattr(decl, "nets", ()) or ()
        first = str(attrs.get("first") or (nets[0] if len(nets) > 0 else ""))
        second = str(attrs.get("second") or (nets[1] if len(nets) > 1 else ""))
        edge = str(attrs.get("edge") or "")
        offset = int(attrs.get("offset_nm") or 0)
        if first and second and first != second and offset == 0 and edge in ("center_x", "center_y"):
            by_pair_edges.setdefault(frozenset((first, second)), set()).add(edge)
    concentric_pairs = tuple(pair for pair, edges in by_pair_edges.items()
                             if {"center_x", "center_y"} <= edges)
    if not concentric_pairs:
        return {}
    resolved = specs
    if resolved is None:
        resolved = {d.instance_name: native_geometry_spec(d) for d in getattr(cell, "devices", ()) or ()}
    surround = mim_vertical_stack_m3_surround_nm()
    out = {}
    for pair in concentric_pairs:
        a, b = tuple(sorted(pair))
        sa, sb = resolved.get(a), resolved.get(b)
        if sa is None or sb is None:
            continue
        if sa.get("family") == "mim" and sb.get("family") != "mim":
            out[a] = surround
        elif sb.get("family") == "mim" and sa.get("family") != "mim":
            out[b] = surround
    return out


def _require(ok, message, *, unsupported=False):
    if not ok:
        error = NativeSky130GeometryUnsupported if unsupported else NativeSky130GeometryError
        raise error(message)


def native_diffusion_contact_coverage(model, width_nm, length_nm, finger_count, profile):
    """Admit the two source-declared SKY130 profiles validated for the six-finger PMOS.

    Tokens are PDK PCell parameters, including their sign.  No contact masks
    are resized or synthesized by this contract.
    """
    if profile is None:
        return ()
    _require(type(profile) is tuple and profile in (("+20", "-20"), ("-20", "+20")),
             "unsupported SKY130 source/drain contact coverage", unsupported=True)
    _require((model, width_nm, length_nm, finger_count) ==
             ("sky130_fd_pr__pfet_01v8", 2000, 150, 6),
             "SKY130 contact coverage profile is unsupported for this native row", unsupported=True)
    return (("viasrc", profile[0]), ("viadrn", profile[1]))


def native_geometry_spec(device):
    """Consume the canonical units/multiplicity contract without physical I/O."""
    from clay.frontend.native_spice_import import validate_native_device

    validated = validate_native_device(device, pdk_id="sky130")
    contract = validated.contract
    numbers = dict(validated.numeric_parameters)
    # Preserve the exact existing decimal conversion and manufacturing-grid gate.
    dims_nm = {key: Decimal(str(numbers[key])) * Decimal(str(scale)) * Decimal(10**9)
               for key, scale in contract.length_scales}
    _require(set(dims_nm) == {"w", "l"}, "native PCell requires explicit W/L")
    nf = int(numbers.get("nf", 1))
    counts = {}
    for key in ("m", "mult"):
        value = numbers.get(key, 1)
        _require(float(value).is_integer() and 0 < value <= 4096,
                 f"{device.instance_name}: nonintegral/oversized physical {key} is unsupported",
                 unsupported=True)
        counts[key] = int(value)
    multiplicity = counts["m"] * counts["mult"]
    _require(nf * multiplicity <= 4096, "native PCell occurrence budget exceeded", unsupported=True)
    _require(all(value > 0 and value % 5 == 0 for value in dims_nm.values()),
             f"{device.instance_name}: W/L must lie on the native 5nm manufacturing grid")
    if contract.family is NativeSpiceFamily.MOS:
        _require(dims_nm["w"] % (nf * 5) == 0,
                 f"{device.instance_name}: total W/nf is off the native 5nm grid")
    else:
        _require(nf == 1 and counts["mult"] == 1,
                 "non-MOS native finger interpretation is unsupported", unsupported=True)
    if contract.family is NativeSpiceFamily.VARACTOR:
        _require(numbers.get("vm", 1) == 1,
                 "native vm != 1 needs a separate physical/statistical contract", unsupported=True)
        _require(not ({"wc", "lc"} & {key.casefold() for key, _ in device.native_parameters}),
                 "explicit varactor wc/lc overrides require a proved geometry binding", unsupported=True)
    if contract.family is NativeSpiceFamily.MIM:
        _require(numbers.get("mf", 1) == 1,
                 "native MIM mf != 1 needs statistical metadata preservation in PEX", unsupported=True)
    return {"model": contract.model, "family": contract.family.value,
            "w_nm": int(dims_nm["w"]), "l_nm": int(dims_nm["l"]), "nf": nf,
            "m": counts["m"], "mult": counts["mult"], "replicas": multiplicity,
            "occurrences": nf * multiplicity, "contract": contract.canonical_data()}


def single_tap_selections(cell):
    """Read source annotation objects without importing the physical layout package."""
    annotations_module = sys.modules.get('clay.layout.annotations')
    tap_type = getattr(annotations_module, 'NativeBodyTap', ())
    result = {}
    for name, annotations in (cell.__dict__.get('_layout_annotations', {}) or {}).items():
        chosen = [a for a in annotations if isinstance(a, tap_type)]
        _require(len(chosen) <= 1, 'duplicate native body-tap declaration')
        if chosen:
            _require(chosen[0].mode == 'single', 'unsupported native body-tap mode')
            result[name] = chosen[0].mode
    return result


def native_cell_geometry_specs(cell, *, validate_device=native_geometry_spec):
    """Resolve device and cell-level native geometry together for check and generation."""
    selections = single_tap_selections(cell)
    annotations_module = sys.modules.get('clay.layout.annotations')
    isolation_type = getattr(annotations_module, 'WellIsolated', ())
    for target, annotations in (cell.__dict__.get('_layout_annotations', {}) or {}).items():
        for annotation in annotations:
            if isinstance(annotation, isolation_type):
                _require(target not in selections and annotation.from_device not in selections,
                         'single native body tap does not implement well isolation')
    devices = {d.instance_name: d for d in cell.devices}
    _require(set(selections) <= set(devices), 'native body-tap target is not a source device')
    specs = {}
    for name, device in devices.items():
        spec = validate_device(device)
        if name in selections:
            key = (spec['model'], spec['w_nm'], spec['l_nm'], spec['nf'], spec['replicas'])
            pmos = ('sky130_fd_pr__pfet_01v8', 840, 150, 2, 1)
            nmos = ('sky130_fd_pr__nfet_01v8', 420, 150, 1, 1)
            _require(key in (pmos, nmos), 'single native body tap is unsupported for this geometry',
                     unsupported=True)
            if key == nmos:
                from clay.port_roles import declared_substrate_ports
                substrate = declared_substrate_ports(cell.port_roles)
                _require(len(substrate) == 1 and device.pins.get('b') == substrate[0],
                         'single native NMOS tap requires the uniquely declared global substrate')
            spec = {**spec, 'body_tap': selections[name]}
        specs[name] = spec
    for mim_name, surround_nm in stacked_mim_surround_selections(cell, specs=specs).items():
        specs[mim_name] = {**specs[mim_name], 'vertical_stack_m3_surround_nm': surround_nm}
    return specs


def _materialize(cell, **kwargs):
    from clay.layout.native_sky130 import materialize_native_sky130

    return materialize_native_sky130(cell, **kwargs)


def _observe_primitive(cell, **kwargs):
    from clay.layout.native_sky130 import materialize_native_sky130

    return materialize_native_sky130(cell, **kwargs)


NATIVE_GEOMETRY_PROVIDER = NativeGeometryProvider(
    "sky130.installed_magic_pcell", native_geometry_spec, _materialize,
    NativePrimitiveObservationCapability('clay.native-primitive-observation/v1', _observe_primitive),
)
