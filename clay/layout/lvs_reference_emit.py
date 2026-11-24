"""Reproducible LVS reference-netlist emitter for the gf180 two-stage OTA.

This module emits a SPICE reference subcircuit for ``build_ota_intent()`` whose
device cards are name/model-aligned to what a real KLayout LVS compare on the
routed gf180 OTA will accept. The netlist is generated DIRECTLY from the live
``ExtractedIntent`` (devices + passives + nets) -- it is never hand-typed, so it
stays in lockstep with the schematic source
(``refCircuit/twoStageAmp/.../OTA_CS_Q.src.net`` corresponded to ``layout.gds``,
geometry-only, leakage-free: it carries NO reference routed layout).

Two model-name conventions are supported, matching the two real LVS paths:

* ``"synth"`` (default) -- the SYNTHESIZED deck used by the gate
  (``clay.layout.klayout_deck_gen.klayout_lvs_from_pdk``): MOS models
  ``nfet_03v3``/``pfet_03v3``, MIM model ``mim_1p0fF``. That deck extracts ONLY
  MOS + MIM (no resistor extraction), and reads the reference through a
  ``Mos3Reader`` delegate that DROPS the MOS bulk node (3-terminal compare).
  For this convention the resistors are, by default, COLLAPSED to wires so the
  reference's surviving-device topology matches the extraction.

* ``"foundry"`` -- the foundry ``gf180mcu.lvs`` deck
  (``tools/klayout_lvs_signoff_v145/run.py::run_gf180_lvs``): MOS models
  ``nmos_3p3``/``pmos_3p3``, poly-resistor model ``ppolyf_u``, MIM model
  ``mim_<d>fF``. This deck DOES extract the poly resistors, so the resistors are
  emitted as real ``R`` cards.

Cell-name correspondence: KLayout pairs the reference ``.subckt`` with the GDS
top cell by NAME. The gate emits top cell ``ota_routed``, so the default
``subckt_name`` is ``ota_routed``.

MIM C-tolerance quirk (see
``tools/klayout_lvs_signoff_v145/gf180_mim_botplate_connect_fix.md``): the
foundry deck's SPICE reader force-overwrites every reference cap card's ``C`` to
a constant ``2e-16 F`` before parsing, and the deck pins the real capacitance via
the plate AREA ``A`` at zero tolerance instead. The cap's declared ``C`` value
therefore does NOT drive the compare on the foundry path; the corrected deck
widens the MIM ``C`` tolerance so the cap pairs on ``A``. We still emit the real
intent ``C`` value (informational / synth-path use) and append the ``mim_*``
model token on the card, which the foundry reader requires. We do NOT emit an
``A=`` on the reference card: the deck computes the schematic ``A`` from the
model and the layout ``A`` from geometry; emitting a wrong ``A`` would trip the
zero-tolerance ``A`` check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


# Default stable output location (alongside the other OTA review GDS artifacts).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(_REPO, "_review_gds", "ota_routed_ref.cir")

# Model-name tables for the two real deck paths.
_MODELS = {
    "synth": {
        "nmos": "nfet_03v3",
        "pmos": "pfet_03v3",
        "res": "ppolyf_u",          # only used if resistors are emitted
        "cap": "mim_1p0fF",
    },
    "foundry": {
        "nmos": "nmos_3p3",
        "pmos": "pmos_3p3",
        "res": "ppolyf_u",
        "cap": "mim_1p0fF",
    },
    # open_pdk: self-consistent class names (our deck extracts these, our reference
    # declares these). The OTA netlist is the twoStageAmp reference; the comparison is
    # our external_pdk P&R layout vs that netlist.
    # open_pdk: the synthesized KLayout deck assigns MOS device CLASSES from the
    # adapter's own subckt names (nmos_3p3 / pmos_3p3 -- the 1.8 V core devices in
    # _PDK_LVS["open_pdk"]["devices"]); KLayout pairs device classes BY NAME, so
    # the reference declares the SAME class tokens. R = the external_pdk poly-resistor
    # class the synth deck's ResLvsSpec registers (npolyf_u, the non-silicide N+
    # poly the passive facts use); C = the external_pdk MIM device class (mimcap, the LVS
    # deck's ``DEVICE C(MIM) ... mimcap`` name).
    # open_pdk core (non-RF) library uses flat .model cards rather than the
    # open_pdk *_ckt subcircuits. The passive device classes remain identical.
}


@dataclass
class RefNetlistSpec:
    """Knobs controlling reference-netlist emission."""

    convention: str = "synth"          # "synth" (gate path) | "foundry"
    subckt_name: str = "ota_routed"    # must equal the GDS top-cell name
    # Target PDK identifier (e.g. ``"gf180mcu"``). When set AND registered in
    # ``clay.pdk_dims``, the emitter clamps each MOS L (and W) UP to the PDK drawn
    # minimum before building the card -- the SAME clamp the layout device cell
    # applies (gf180 ``GF180_GATE_MIN_LENGTH_NM`` = 280 nm = pdk_dims.min_L). This
    # keeps the two sides of the compare in lockstep BY CONSTRUCTION when a source
    # device asks for a sub-min L (e.g. a 1.8 V n18/p18 device at L=0.18u migrated
    # to gf180 3.3 V, whose drawn min is 0.28u): the layout draws 0.28u, so the
    # reference must declare 0.28u too, or the zero-tolerance L compare FAILs.
    # ``None`` (or an unregistered PDK) -> no clamp (byte-stable legacy behaviour).
    pdk_id: Optional[str] = None
    # Top-cell port order. The synthesized-deck contract pairs by topology; the
    # ports just have to be a superset of the reference's externally-named nets.
    ports: Tuple[str, ...] = ("VIP", "VIN", "vout", "vdd!", "gnd!")
    # Emit resistors as real R cards (True) or collapse them to wires (False).
    # Default follows the convention: synth deck extracts no R -> collapse;
    # foundry deck extracts poly R -> emit.
    emit_resistors: Optional[bool] = None
    # Emit the MOS bulk node on each card (4-terminal). The synth deck's
    # Mos3Reader drops it either way; the foundry deck reads 4 nodes.
    emit_bulk: bool = True
    # DEVICE-FOLD form (the partition the KLayout compare actually pairs against).
    # KLayout runs ``netlist.simplify`` / ``combine_devices`` on the LAYOUT before
    # ``compare``: a transistor laid out as M abutted-diffusion fingers (or M
    # parallel devices on the same D/G/S nets) is folded into ONE device whose
    # ``W`` is the SUM of the finger widths, carrying NO ``M`` parameter (verified:
    # 4x W=21 fingers -> one W=84, no M). The reference must present the SAME folded
    # partition or per-device pairing fails on the fold axis.
    #
    # * ``fold_parallel=True`` (default): emit each schematic MOS as ONE card with
    #   ``W = W*M`` (the TOTAL channel width) and NO ``M=`` token -- byte-faithful to
    #   the post-``combine_devices`` layout card. This MIRRORS the physical merge.
    # * ``fold_parallel=False``: emit the literal schematic ``W`` plus ``M=<mult>``.
    #   This ALSO pairs on the gate path, because the SPICE reader pre-multiplies the
    #   card's W by M before ``combine_devices`` runs (W*M is identical either way) --
    #   kept for a reader/flow that prefers the explicit multiplier notation.
    fold_parallel: bool = True
    # Device-class bindings from the same extraction deck configuration.
    # Thick-oxide classes must be explicitly bound; never substitute thin MOS.
    mos_models: Optional[Mapping[str, str]] = None


def _fmt_um(value: float) -> str:
    """Format a micron value, trimming trailing zeros (e.g. ``84`` not ``84.0``)."""
    s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def resistor_collapse_alias(intent: Any) -> Dict[str, str]:
    """The net->canonical-net alias map that COLLAPSES every resistor to a wire.

    The synthesized LVS deck (``klayout_lvs_from_pdk``) extracts ONLY MOS + MIM; it
    does NOT extract resistors. A resistor laid out as a poly strip therefore looks,
    to that deck, like a plain wire that shorts its two terminal nets into ONE
    physical node. So the reference netlist collapses each resistor's two nets into a
    single canonical net, and the ROUTED GEOMETRY must do the same (route the two
    resistor-terminal nets onto one copper net) or the compare disagrees on topology.

    This is a NETLIST fact (which two nets each resistor bridges, from
    ``intent.passives``), not a layout fact -- it is leakage-free: no reference routed
    layout or ground-truth coordinate is read. The canonical survivor of each resistor
    class is the externally-named net when present (``vout`` / ``gnd!`` / ...), else the
    lexicographically smallest member, so externally-visible nets stay the survivor.

    Returns ``{net: canonical_net}`` for every net in a resistor class (a net not in
    any resistor class is absent from the map -> it maps to itself).
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in getattr(intent, "passives", []) or []:
        kind = p.kind if hasattr(p, "kind") else p["kind"]
        if kind != "res":
            continue
        nets = p.nets if hasattr(p, "nets") else p["nets"]
        union(nets["p"], nets["n"])

    groups: Dict[str, List[str]] = {}
    for nm in list(parent):
        groups.setdefault(find(nm), []).append(nm)
    alias: Dict[str, str] = {}
    for _root, members in groups.items():
        external = [m for m in members if not m.lower().startswith("net")]
        canon = sorted(external)[0] if external else sorted(members)[0]
        for m in members:
            alias[m] = canon
    return alias


def _pdk_drawn_minimums(
    pdk_id: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(min_L_um, min_W_um)`` drawn minimums for ``pdk_id``.

    Sourced from the single per-PDK process-constant table in
    :mod:`clay.pdk_dims` (the SAME table the sizing/layout side reads), converted
    from SI metres to microns. Returns ``(None, None)`` when ``pdk_id`` is falsy or
    not registered there -> the caller applies no clamp (legacy byte-stable form).
    This is data-driven: no hardcoded device names, no per-circuit branch -- the
    clamp floor comes entirely from the PDK's own minimum-geometry constants.
    """
    if not pdk_id:
        return None, None
    try:
        from clay.pdk_dims import PdkDims

        dims = PdkDims.from_pdk_name(pdk_id)
    except Exception:
        return None, None
    return float(dims.min_L) * 1e6, float(dims.min_W) * 1e6


def _mos_card(d: Dict[str, Any], model: str, emit_bulk: bool,
              fold_parallel: bool = True,
              min_L_um: Optional[float] = None,
              min_W_um: Optional[float] = None) -> str:
    n = d["nets"]
    drain, gate, source, body = n["drain"], n["gate"], n["source"], n["body"]
    nodes = f"{drain} {gate} {source}"
    if emit_bulk:
        nodes += f" {body}"
    # Card name keeps the schematic device name (e.g. MM14). W/L in microns.
    # Geometry params (AS/AD/PS/PD) are left to the deck -- the deck compares L/W;
    # area params are not part of the synthesized-deck MOS compare contract.
    #
    # DEVICE FOLD: KLayout folds the laid-out M parallel fingers into ONE device of
    # TOTAL width (W*M, no M parameter) via combine_devices before compare. To present
    # the SAME folded partition the reference is, by default, emitted pre-folded:
    # ONE card with W = W*M and no M= token (fold_parallel=True). The non-folded form
    # (literal W + M=<mult>) is equivalent for pairing because the SPICE reader
    # pre-multiplies W by M, but the pre-folded form is byte-faithful to the
    # post-combine layout card -- it mirrors the physical merge directly.
    mult = int(d["M"])
    # CLAMP the drawn dimensions UP to the PDK minimum, mirroring the layout device
    # cell (gf180_device_geom clamps L up to GF180_GATE_MIN_LENGTH_NM = 280 nm and W
    # up to GF180_CHANNEL_MIN_WIDTH_NM = 220 nm). A source device that asks for a
    # sub-min geometry (e.g. an L=0.18u 1.8 V device migrated to gf180 3.3 V) is
    # DRAWN at the legal minimum, so the reference card must DECLARE that same
    # minimum or the deck's near-zero-tolerance L/W compare FAILs and stalls the
    # touching net cluster. The clamp floor is data-driven (clay.pdk_dims); when no
    # PDK minimum is known (min_*_um is None) the raw intent value is emitted
    # verbatim -- byte-stable legacy behaviour.
    l_draw = float(d["L"])
    if min_L_um is not None:
        l_draw = max(l_draw, min_L_um)
    w_per = float(d["W"])
    if min_W_um is not None:
        w_per = max(w_per, min_W_um)
    if fold_parallel:
        w_total = w_per * mult
        parts = [
            f"M{d['name']}", nodes, model,
            f"L={_fmt_um(l_draw)}u", f"W={_fmt_um(w_total)}u",
        ]
    else:
        parts = [
            f"M{d['name']}", nodes, model,
            f"L={_fmt_um(l_draw)}u", f"W={_fmt_um(w_per)}u",
        ]
        if mult != 1:
            parts.append(f"M={mult}")
    return " ".join(parts)


def _res_card(p: Dict[str, Any], model: str) -> str:
    n = p["nets"]
    # ppolyf_u poly resistor; value in ohms. Model token appended (foundry
    # reader treats it as the device-class selector).
    return f"R{p['name']} {n['p']} {n['n']} {p['value']:.6g} {model}"


def _cap_card(p: Dict[str, Any], model: str) -> str:
    n = p["nets"]
    # MIM cap; C in farads. The foundry reader force-rewrites C to 2e-16 and
    # pins the cap on the plate AREA at zero tolerance (see the MIM-botplate fix
    # doc); we still emit the true intent C value -- it is informational on the
    # foundry path and load-bearing on the synth path -- plus the mim_* token.
    return f"C{p['name']} {n['p']} {n['n']} {p['value']:.6g} {model}"


def _bjt_binding(pdk_id: Optional[str], kind: str):
    """Resolve the foundry BJT reference convention from adapter DATA."""
    if not pdk_id:
        raise ValueError("BJT reference emission requires RefNetlistSpec.pdk_id")
    from clay.pdk import get_pdk, load_all

    load_all()
    binding = (get_pdk(pdk_id).bjt_bindings or {}).get(str(kind).lower())
    if binding is None:
        raise ValueError(f"{pdk_id}: no BJT binding for {kind!r}")
    return binding


def _bjt_card(device: Dict[str, Any], binding: Any) -> str:
    nets = device["nets"]
    area = float(device.get("area", 1.0) or 1.0)
    mult = float(device.get("M", 1.0) or 1.0)
    units = area * mult
    unit_count = int(round(units))
    if units <= 0.0 or abs(units - unit_count) > 1e-9:
        raise ValueError(
            "BJT reference requested a non-integral fixed-cell bank"
        )
    name = str(device["name"])
    if name[:1].upper() == "Q":
        name = name[1:]
    model = str(binding.lvs_model)
    multiplier = str(binding.lvs_parallel_parameter)
    return (
        f"Q{name} {nets['c']} {nets['b']} {nets['e']} {model} "
        f"{multiplier}={unit_count}"
    )


def build_reference_netlist(intent: Any, spec: Optional[RefNetlistSpec] = None) -> str:
    """Return the reference SPICE text for ``intent`` under ``spec``.

    ``intent`` is an :class:`~clay.layout.extract.types.ExtractedIntent`
    (e.g. from ``build_ota_intent()``). The text is generated purely from
    ``intent.devices`` / ``intent.passives`` -- no reference layout is read.
    """
    spec = spec or RefNetlistSpec()
    if spec.convention not in _MODELS:
        raise ValueError(f"unknown convention {spec.convention!r}; "
                         f"expected one of {sorted(_MODELS)}")
    models = dict(_MODELS[spec.convention])
    if spec.mos_models is not None:
        for key in ("nmos", "pmos", "nmos3v", "pmos3v"):
            if key in spec.mos_models:
                value = spec.mos_models[key]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid MOS model binding for {key!r}")
                models[key] = value

    # PDK drawn minimums (um) used to clamp sub-min MOS L/W to what the layout
    # device cell actually draws (data-driven from clay.pdk_dims; (None, None) when
    # the spec names no PDK or an unregistered one -> no clamp).
    min_L_um, min_W_um = _pdk_drawn_minimums(spec.pdk_id)

    emit_r = spec.emit_resistors
    if emit_r is None:
        # synth deck extracts no resistors -> collapse to wires by default;
        # foundry deck extracts poly resistors -> emit by default.
        emit_r = spec.convention == "foundry"

    lines: List[str] = []
    lines.append(
        f"* Reference netlist for the gf180 two-stage OTA "
        f"(convention={spec.convention})."
    )
    lines.append(
        "* Generated from build_ota_intent() devices/passives -- "
        "leakage-free (no reference routed layout)."
    )
    lines.append(
        f"* MOS={models['nmos']}/{models['pmos']}  R={models['res']}  "
        f"C={models['cap']}"
    )
    lines.append(
        "* device fold: "
        + ("parallel fingers folded -> W=W*M, no M= "
           "(mirrors KLayout combine_devices)"
           if spec.fold_parallel else
           "literal W + M= multiplier (reader pre-multiplies W*M)")
    )

    # Resistor collapse: when resistors are dropped (synth path), their two nets
    # become electrically the SAME node. The alias map is computed by the shared,
    # leakage-free :func:`resistor_collapse_alias` so the ROUTED GEOMETRY (the gate's
    # ``_materialize``) and this reference netlist collapse the SAME nets -- one source
    # of truth, no drift between the two sides of the compare.
    alias: Dict[str, str] = {}
    if not emit_r:
        alias = resistor_collapse_alias(intent)
        if alias:
            collapsed = ", ".join(
                f"{k}->{v}" for k, v in sorted(alias.items()) if k != v
            )
            lines.append(f"* resistors collapsed to wires: {collapsed}")

    def a(net: str) -> str:
        return alias.get(net, net)

    lines.append(f".subckt {spec.subckt_name} {' '.join(spec.ports)}")

    # MOSFETs
    for d in intent.devices:
        dd = {
            "name": d.name if hasattr(d, "name") else d["name"],
            "kind": d.kind if hasattr(d, "kind") else d["kind"],
            "W": d.W if hasattr(d, "W") else d["W"],
            "L": d.L if hasattr(d, "L") else d["L"],
            "M": d.M if hasattr(d, "M") else d["M"],
            "nets": dict(d.nets if hasattr(d, "nets") else d["nets"]),
        }
        # apply resistor-collapse alias to MOS terminals too
        dd["nets"] = {k: a(v) for k, v in dd["nets"].items()}
        from clay.layout.annotate_to_solver import kind_polarity_oxide

        polarity, kind_oxide = kind_polarity_oxide(dd["kind"])
        if polarity is None:
            raise ValueError(f"unknown MOS kind {dd['kind']!r} for {dd['name']}")
        explicit_oxide = (getattr(d, "oxide", None) if hasattr(d, "kind")
                          else d.get("oxide"))
        if explicit_oxide not in (None, "", "thin", "thick"):
            raise ValueError(f"unknown MOS oxide {explicit_oxide!r} for {dd['name']}")
        thick = kind_oxide == "thick" or explicit_oxide == "thick"
        model_key = polarity + ("3v" if thick else "")
        if model_key not in models:
            raise ValueError(f"missing MOS model binding for {model_key!r}")
        model = models[model_key]
        lines.append(_mos_card(dd, model, spec.emit_bulk,
                               fold_parallel=spec.fold_parallel,
                               min_L_um=min_L_um, min_W_um=min_W_um))

    # Passives
    for p in intent.passives:
        pp = {
            "name": p.name if hasattr(p, "name") else p["name"],
            "kind": p.kind if hasattr(p, "kind") else p["kind"],
            "value": p.value if hasattr(p, "value") else p["value"],
            "nets": {k: a(v) for k, v in (
                p.nets if hasattr(p, "nets") else p["nets"]).items()},
        }
        if pp["kind"] == "res":
            if emit_r:
                resistor_model = models["res"]
                if spec.pdk_id:
                    from clay.layout.route.lvs_gate import _resolved_lvs_config

                    recognition = _resolved_lvs_config(
                        spec.pdk_id, full_extraction=True)["res"]
                    if recognition is None:
                        raise ValueError(f"{spec.pdk_id}: no resistor recognition recipe")
                    source_model = (getattr(p, "model", "") if hasattr(p, "kind")
                                    else p.get("model", ""))
                    resistor_model = recognition.resolve_source_model(
                        str(source_model or "")).model
                lines.append(_res_card(pp, resistor_model))
            # else: collapsed to wire (handled via alias), no card emitted
        elif pp["kind"] == "cap":
            lines.append(_cap_card(pp, models["cap"]))

    # Adapter-bound three-terminal BJTs. Their C/B/E order is a hard language +
    # physical-cell contract; model identity comes from PDK data, never a circuit.
    for device in getattr(intent, "bjts", []) or []:
        dd = {
            "name": device.name if hasattr(device, "name") else device["name"],
            "kind": device.kind if hasattr(device, "kind") else device["kind"],
            "area": (
                device.area if hasattr(device, "area") else device.get("area", 1.0)
            ),
            "M": device.M if hasattr(device, "M") else device.get("M", 1.0),
            "nets": {
                key: a(value)
                for key, value in dict(
                    device.nets if hasattr(device, "nets") else device["nets"]
                ).items()
            },
        }
        lines.append(_bjt_card(dd, _bjt_binding(spec.pdk_id, dd["kind"])))

    lines.append(".ends")
    lines.append("")
    return "\n".join(lines)


def emit_reference_netlist(
    out_path: str = DEFAULT_OUT,
    *,
    spec: Optional[RefNetlistSpec] = None,
) -> str:
    """Build ``build_ota_intent()`` and write its reference netlist to ``out_path``.

    Returns the absolute path written.
    """
    from clay.layout.extract.migrate import build_ota_intent

    intent = build_ota_intent()
    text = build_reference_netlist(intent, spec)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--convention", default="synth", choices=sorted(_MODELS))
    ap.add_argument("--subckt", default="ota_routed")
    ap.add_argument("--emit-resistors", dest="emit_resistors",
                    action="store_true", default=None)
    ap.add_argument("--no-bulk", dest="emit_bulk", action="store_false",
                    default=True)
    ap.add_argument("--no-fold", dest="fold_parallel", action="store_false",
                    default=True,
                    help="emit literal W + M= instead of pre-folded W*M (default folds)")
    args = ap.parse_args()
    spec = RefNetlistSpec(
        convention=args.convention,
        subckt_name=args.subckt,
        emit_resistors=args.emit_resistors,
        emit_bulk=args.emit_bulk,
        fold_parallel=args.fold_parallel,
    )
    path = emit_reference_netlist(args.out, spec=spec)
    print(f"[lvs-ref] wrote {path}  (convention={args.convention}, "
          f"subckt={args.subckt})")
    with open(path, encoding="utf-8") as fh:
        print(fh.read())
