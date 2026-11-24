"""
Built-in template: `two_stage_miller_pmos_explicit_bias` (v0.0.26 / G2).

Variant of `two_stage_miller_pmos` that consumes a declarative
`bias_tree { ... }` block from the .clay source instead of hardcoding
the bias chain in Python. The signal-path stages (PMOS diff pair, NMOS
mirror load, NMOS common-source second stage, Miller compensation cap)
are identical to the canonical PMOS Miller topology — only the bias
generation is now user-authored.

Surface contract (the bias_tree the LLM writes):

    bias_tree {
      Iref:      external = 10uA;
      pbias:     pmos_diode from Iref;
      nbias:     nmos_diode from pbias;
      pbias_cas: pmos_mirror of pbias by 4;
    }

Semantic interpretation of each BiasNode kind:

  * external      : in-cell current source between vdd and the named
                    node (drains the node at `value` amperes).
  * pmos_diode    : PMOS, drain=gate=<name>, source=vdd. Supplies
                    current from vdd. Sets <name>'s rail to vdd - |Vgs_p|.
  * nmos_diode    : NMOS, drain=gate=<name>, source=vss. Sinks current
                    to vss. For the NMOS diode to actually conduct, an
                    auxiliary PMOS mirror (gate=<src>) is added that
                    pulls current from <src>'s rail to feed the diode.
  * pmos_mirror   : PMOS mirror, gate=<src>, drain=<name>, source=vdd,
                    width multiplier M=<by_M>. Drain is exposed as a
                    bias node downstream consumers can tap.
  * nmos_mirror   : Symmetric NMOS mirror, drain=<name>, gate=<src>,
                    source=vss.

The hardwired Mtail (diff pair tail PMOS mirror) and Mlcs (second-stage
load PMOS mirror) still gate from `pbias` — the variant therefore
REQUIRES the bias_tree to declare a `pbias` node (`pmos_diode`-kind so
its rail is bounded to vdd - |Vgs_p|). The CLI surfaces a clear error
if `pbias` is absent.

This file is sibling to `two_stage_miller_pmos.py`; the existing
template is untouched so every pre-G2 cell that compiles against
`two_stage_miller_pmos` keeps working unmodified.
"""

from __future__ import annotations

from typing import Optional

from ..ir import (
    AcAnalysis, BiasNode, Cell, Device, Measure, OpAnalysis, Source, Spec,
    Testbench,
)


def _mosfet(name: str, kind: str, d: str, g: str, s: str, b: str,
            W: float, L: float, M: int = 1) -> Device:
    return Device(
        instance_name=name,
        device_class=kind,
        pdk_variant=None,
        pins={"d": d, "g": g, "s": s, "b": b},
        sizing={"W": W, "L": L, "M": float(M)},
    )


def _cap(name: str, p: str, n: str, value: float) -> Device:
    return Device(
        instance_name=name,
        device_class="cap",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def _leak_res(name: str, p: str, n: str, value: float = 1e9) -> Device:
    """High-impedance resistor used to give an otherwise-floating
    mirror drain a DC path to vss. 1 GOhm sinks ~picoamps at typical
    bias voltages — negligible against the µA-scale tail/load currents
    in the signal path."""
    return Device(
        instance_name=name,
        device_class="res",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def _materialize_bias_chain(
    bias_tree: list[BiasNode],
    W_bp: float,
    L_bp: float,
) -> tuple[list[Device], float, set[str]]:
    """Translate a list of BiasNode IRs into actual Device objects.

    Returns:
        devices: the bias chain devices (current source + transistors)
        I_ref_value: the external current value (for downstream sizing)
        produced_nodes: set of node names this chain establishes

    The mapping rule per node kind is documented at the top of this
    file. Auxiliary mirrors generated for `nmos_diode from X` and
    `pmos_diode from <non-external>` are named `Maux_<dst>` and
    automatically wired. The polarity of the `external` current
    source is selected based on what kind of device consumes its
    output: a PMOS diode rail must be sunk to vss; an NMOS diode rail
    must be sourced from vdd.
    """
    devices: list[Device] = []
    produced: set[str] = set()
    consumed_drain: set[str] = set()
    I_ref_value: float | None = None

    # Map src-name → BiasNode so we can introspect references.
    by_name: dict[str, BiasNode] = {n.name: n for n in bias_tree}

    # Precompute "what node consumes each external" and "what kind"
    # so we can wire the current source to the consumer's bias rail
    # and pick the polarity correctly. The external bias node's name
    # is a *label* for the current source — the source physically
    # connects between a rail (vdd or vss) and the FIRST downstream
    # device's bias node, not to a circuit node literally named
    # `Iref`. Multiple downstream consumers of the same external are
    # supported but the source still attaches to the first one (other
    # consumers must mirror off it via diode/mirror chains).
    downstream_of: dict[str, list[BiasNode]] = {}
    for m in bias_tree:
        if m.src is not None:
            downstream_of.setdefault(m.src, []).append(m)

    for n in bias_tree:
        produced.add(n.name)
        if n.kind == "external":
            # An external WITHOUT a consumer is degenerate — error out.
            downstream = downstream_of.get(n.name, [])
            if not downstream:
                raise ValueError(
                    f"two_stage_miller_pmos_explicit_bias: external "
                    f"bias node {n.name!r} has no consumer. At least "
                    f"one downstream node must declare "
                    f"`<name>: pmos_diode from {n.name}` (or "
                    f"`nmos_diode from {n.name}`)."
                )
            # Attach the current source to the first PMOS-diode or
            # NMOS-diode consumer. Polarity:
            #   PMOS diode  → drain the consumer's node to vss
            #   NMOS diode  → supply current from vdd to the consumer
            consumer = downstream[0]
            if consumer.kind == "pmos_diode":
                pins = {"p": consumer.name, "n": "vss"}
            elif consumer.kind == "nmos_diode":
                pins = {"p": "vdd", "n": consumer.name}
            else:
                raise ValueError(
                    f"two_stage_miller_pmos_explicit_bias: external "
                    f"{n.name!r} feeds {consumer.name!r} of kind "
                    f"{consumer.kind!r}; only pmos_diode/nmos_diode "
                    f"are supported as direct downstream of an "
                    f"`external` source in this MVP."
                )
            devices.append(Device(
                instance_name=f"I_{n.name}",
                device_class="cur_src",
                pdk_variant=None,
                pins=pins,
                sizing={"value": float(n.value or 0.0)},
            ))
            if I_ref_value is None:
                I_ref_value = float(n.value or 0.0)
        elif n.kind == "pmos_diode":
            # PMOS, drain=gate=<name>, source=vdd. Current path: from
            # vdd through this device to <n.name>. The named node
            # settles at vdd - |Vgs_p|.
            devices.append(_mosfet(
                f"Mbp_{n.name}", "pmos",
                d=n.name, g=n.name, s="vdd", b="vdd",
                W=W_bp, L=L_bp,
            ))
            # Track that we expect <src> to supply the sink current —
            # but the actual sink (Iref or NMOS mirror) is the upstream
            # device, so no aux mirror is needed when `from Iref`.
            # If the src is another diode/mirror, an auxiliary NMOS
            # mirror sinks the current from <name> to vss. That mirror
            # is gated by <src>'s rail (i.e., the NMOS-mirror form
            # requires an NMOS that mirrors the upstream NMOS — which
            # only exists if the chain already produced one). To keep
            # the MVP semantics simple, we ONLY support pmos_diode
            # whose src is `external`; chained pmos diodes (rare in
            # practice) would need an explicit mirror declaration.
            src_kind = by_name[n.src].kind if (n.src and n.src in by_name) else None
            if src_kind not in ("external", None):
                raise ValueError(
                    f"two_stage_miller_pmos_explicit_bias: bias node "
                    f"{n.name!r} is a pmos_diode driven by {n.src!r} of "
                    f"kind {src_kind!r}. Only `external` sources are "
                    f"supported as the direct upstream for a pmos_diode "
                    f"in this MVP (declare an intermediate mirror)."
                )
        elif n.kind == "nmos_diode":
            # NMOS, drain=gate=<name>, source=vss. Sinks current to vss.
            # For the diode to conduct we add an auxiliary PMOS mirror
            # of <src> that pulls current from vdd to <name>'s drain.
            # This is the bridge between a PMOS-controlled rail and an
            # NMOS diode rail.
            devices.append(_mosfet(
                f"Mbn_{n.name}", "nmos",
                d=n.name, g=n.name, s="vss", b="vss",
                W=W_bp, L=L_bp,
            ))
            src_kind = by_name[n.src].kind if (n.src and n.src in by_name) else None
            if src_kind in ("pmos_diode", "pmos_mirror"):
                # PMOS auxiliary mirror, drain=<name>, gate=<src>,
                # source=vdd. Implicit M=1.
                devices.append(_mosfet(
                    f"Maux_{n.name}", "pmos",
                    d=n.name, g=n.src, s="vdd", b="vdd",
                    W=W_bp, L=L_bp,
                ))
                consumed_drain.add(n.src)
            elif src_kind in ("nmos_diode", "nmos_mirror"):
                # NMOS diode driven by NMOS source is rare — degenerate
                # case. We don't support it in the MVP; let the user
                # declare an intermediate PMOS mirror.
                raise ValueError(
                    f"two_stage_miller_pmos_explicit_bias: bias node "
                    f"{n.name!r} is an nmos_diode driven by NMOS src "
                    f"{n.src!r}. Bridge it through a pmos_mirror first."
                )
            else:
                # external src: just an Iref source from vdd to nbias.
                # That's how the baseline works (Iref → NMOS diode at
                # nbias). Already covered if upstream produced the
                # current source.
                pass
        elif n.kind == "pmos_mirror":
            # PMOS, drain=<name>, gate=<src>, source=vdd, M=<by_M>.
            devices.append(_mosfet(
                f"M_{n.name}", "pmos",
                d=n.name, g=n.src or n.name, s="vdd", b="vdd",
                W=W_bp, L=L_bp, M=n.by_M or 1,
            ))
            if n.src:
                consumed_drain.add(n.src)
        elif n.kind == "nmos_mirror":
            # NMOS, drain=<name>, gate=<src>, source=vss, M=<by_M>.
            devices.append(_mosfet(
                f"M_{n.name}", "nmos",
                d=n.name, g=n.src or n.name, s="vss", b="vss",
                W=W_bp, L=L_bp, M=n.by_M or 1,
            ))
            if n.src:
                consumed_drain.add(n.src)
        else:
            raise ValueError(
                f"two_stage_miller_pmos_explicit_bias: unknown bias "
                f"kind {n.kind!r} for node {n.name!r}."
            )

    if I_ref_value is None:
        # No external current source declared — bias chain has no seed.
        raise ValueError(
            f"two_stage_miller_pmos_explicit_bias: bias_tree must "
            f"declare an `external` source (e.g. "
            f"`Iref: external = 10uA;`) to seed the bias chain."
        )
    return devices, I_ref_value, produced


def build_two_stage_miller_pmos_explicit_bias(
    cell_name: str,
    spec: Spec,
    *,
    bias_tree: list[BiasNode] | None = None,
    # ─── Bias / mirror sizing (used for ALL bias-chain devices) ──
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    # Mtail and Mlcs multipliers — these gate from pbias just like
    # the baseline. The user's bias_tree pbias_cas node (if any) is
    # built as an additional mirror but does not replace Mtail.
    N_tail: int = 4,
    K_cs_load: int = 16,
    # ─── Per-device sizing: physical (W, L) form ────────────────
    W_diff: float = 20e-6, L_diff: float = 0.5e-6,
    W_load1: float = 4e-6, L_load1: float = 1e-6,
    W_cs: float = 40e-6,   L_cs: float = 0.5e-6,
    # ─── Per-device sizing: intent (gm/Id) form ─────────────────
    gm_id_diff: Optional[float] = None,
    gm_id_load1: Optional[float] = None,
    gm_id_cs: Optional[float] = None,
    Cc: float = 1.5e-12,

    # ─── R121 C: qualitative DRC annotation surface ────────
    # Additive metadata. Default False keeps every pre-R121
    # caller byte-for-byte identical. When True, three
    # qualitative hints attach to the cell:
    #   MATCHING                on (M1, M2)   — PMOS diff pair
    #   CURRENT_MIRROR_MATCHING on (M3, M4)   — NMOS diode mirror
    #   GUARD_RING              on (Mtail,)   — PMOS tail (nwell)
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the 2-stage Miller PMOS-input opamp whose
    bias chain is *declared* in the .clay source via a bias_tree block.

    The signal path (diff pair, mirror load, output stage, Miller cap)
    is identical to `build_two_stage_miller_pmos`. The bias chain is
    materialized from `bias_tree`. The variant REQUIRES bias_tree to
    contain (at least) a `pbias` node of kind `pmos_diode` because
    Mtail and Mlcs gate from `pbias`.
    """
    if not bias_tree:
        raise ValueError(
            "two_stage_miller_pmos_explicit_bias requires a non-empty "
            "bias_tree. Declare a bias_tree { ... } block in your "
            ".clay source."
        )

    # Lazy import: closed-form sizing solver.
    from ..sizing import closed_form_W

    # Materialize the user-declared bias chain.
    bias_devices, I_ref, produced = _materialize_bias_chain(
        bias_tree, W_bp=W_bp, L_bp=L_bp,
    )
    if "pbias" not in produced:
        raise ValueError(
            "two_stage_miller_pmos_explicit_bias: bias_tree must "
            "declare a `pbias` node (the PMOS rail that gates Mtail "
            "and Mlcs). Found: " + str(sorted(produced))
        )

    # Compute per-stage operating currents from the bias plan. We use
    # the user's declared I_ref together with the variant's tail and
    # output multipliers — matching the baseline's polarity convention.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs = K_cs_load * I_ref

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff, I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1, I_d=I_load1, L=L_load1).W
    if gm_id_cs is not None:
        W_cs = closed_form_W("nmos", gm_id=gm_id_cs, I_d=I_cs, L=L_cs).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = list(bias_devices)

    # First-stage tail current source: PMOS mirror of pbias, ratio N.
    devices.append(_mosfet(
        "Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=N_tail,
    ))

    # PMOS differential pair.
    devices.append(_mosfet(
        "M1", "pmos", d="d1", g="inp", s="tail", b="vdd",
        W=W_diff, L=L_diff,
    ))
    devices.append(_mosfet(
        "M2", "pmos", d="d2", g="inn", s="tail", b="vdd",
        W=W_diff, L=L_diff,
    ))

    # NMOS current-mirror load (diode + replica).
    devices.append(_mosfet(
        "M3", "nmos", d="d1", g="d1", s="vss", b="vss",
        W=W_load1, L=L_load1,
    ))
    devices.append(_mosfet(
        "M4", "nmos", d="d2", g="d1", s="vss", b="vss",
        W=W_load1, L=L_load1,
    ))

    # Second stage: NMOS common-source amplifier.
    devices.append(_mosfet(
        "Mcs", "nmos", d="out", g="d2", s="vss", b="vss",
        W=W_cs, L=L_cs,
    ))

    # Second-stage PMOS current-source load (mirror of pbias).
    devices.append(_mosfet(
        "Mlcs", "pmos", d="out", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=K_cs_load,
    ))

    # Miller compensation cap from out to d2 (no nulling resistor).
    devices.append(_cap("Cc", p="out", n="d2", value=Cc))

    # Leak resistors for any declared mirror whose drain is not
    # otherwise tapped — gives ngspice a DC path to vss for the
    # otherwise-floating node. 1 GΩ → sub-pA leakage, negligible
    # against the µA-scale bias currents.
    referenced_nodes: set[str] = {"pbias", "vdd", "vss"}
    for n in bias_tree:
        if n.src is not None:
            referenced_nodes.add(n.src)
    for n in bias_tree:
        if n.kind in ("pmos_mirror", "nmos_mirror") and n.name not in referenced_nodes:
            devices.append(_leak_res(
                f"Rleak_{n.name}", p=n.name, n="vss",
            ))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R121 C: optional qualitative DRC annotations ─────────
    # Three annotations covering the canonical 2-stage Miller PMOS-
    # input topology. The bias_tree declaration mechanism is
    # orthogonal to the layout-DRC vocabulary — the signal-path
    # matched groups are identical to the hardcoded-bias sibling.
    # Default attach_drc=False keeps pre-R121 emit byte-identical.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": (
                    "PMOS diff pair — Vth mismatch maps directly to "
                    "input-referred offset for the 2-stage Miller "
                    "topology. The bias_tree's user-declared pbias rail "
                    "does not alter signal-path symmetry requirements; "
                    "common-centroid M1/M2 remains the gating layout "
                    "constraint"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "NMOS diode mirror (M3 diode, M4 mirror leg). "
                    "Mirror ratio error injects a static differential "
                    "current into d1 that the stage-2 NMOS CS amplifier "
                    "converts into a DC output offset — common-centroid "
                    "unit-cell array required for the 1:1 ratio"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) gated by the "
                    "user-declared pbias rail. Sinks the full stage-1 "
                    "bias current from vdd into tail. nwell guard ring "
                    "tied to vdd shields the high-Z d1 node from "
                    "supply-coupled substrate noise that would couple "
                    "through the bias_tree chain back into the diode-"
                    "connected pbias node"
                ),
                "well_type": "nwell",
            },
        ))

    return cell


# ─── Testbench builder for AC characterization ─────────────────────
# Identical to the baseline two_stage_miller_pmos testbench — the
# signal-path ports (inn, inp, out, vdd, vss) are unchanged.

def build_ac_testbench(cell: Cell, supply: float | None = None, vcm_ratio: float = 0.5,
                       f_start: float = 1.0, f_stop: float = 1e9,
                       points_per_dec: int = 20) -> Testbench:
    """Standard AC testbench for the explicit-bias PMOS Miller variant.
    Mirrors the baseline testbench shape so direct gain/GBW comparisons
    against `two_stage_miller_pmos` are apples-to-apples.

    `supply=None` auto-resolves from `cell.pdk_id`.
    """
    if supply is None:
        from ._supply import resolve_supply
        supply = resolve_supply(cell)
    sources: list[Source] = [
        Source("voltage_dc",  "VDD",  pins={"p": "vdd", "n": "0"}, params={"dc": supply}),
        Source("voltage_dc",  "VSS",  pins={"p": "vss", "n": "0"}, params={"dc": 0.0}),
        Source("voltage_dc",  "VCM",  pins={"p": "vcm", "n": "0"}, params={"dc": supply * vcm_ratio}),
        Source("voltage_ac",  "VSIG", pins={"p": "sig", "n": "0"}, params={"dc": 0.0, "ac": 1.0}),
    ]
    extras = [
        Device("Edp", "vcvs", None,
               pins={"op": "inp", "on": "vcm", "ip": "sig", "in": "0"},
               sizing={"gain": 0.5}),
        Device("Edn", "vcvs", None,
               pins={"op": "inn", "on": "vcm", "ip": "sig", "in": "0"},
               sizing={"gain": -0.5}),
        Device("CL", "cap", None,
               pins={"p": "out", "n": "vss"},
               sizing={"value": 2e-12}),
    ]
    measures = [
        Measure("gain_dc",      "ac", "vdb(out) @ freq=1"),
        Measure("gbw",          "ac", "freq when vdb(out)=0"),
        Measure("phase_at_gbw", "ac", "vp(out) when vdb(out)=0"),
    ]
    analyses: list = [
        OpAnalysis(),
        AcAnalysis(sweep_kind="dec", points_per_decade=points_per_dec,
                   f_start=f_start, f_stop=f_stop),
    ]
    return Testbench(
        cell=cell,
        sources=sources,
        extra_devices=extras,
        analyses=analyses,
        measures=measures,
        pdk_corner="tt",
        supply_volt=supply,
    )
