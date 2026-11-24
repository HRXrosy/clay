"""
Digital-primitive expansion (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
Each digital primitive in the .clay surface (`inv` / `nand2` / `dff` /
`dff_async_clear`) expands to N concrete `Device` IR entries at parse
time. The expanded devices share the W/L/M sizing supplied on the
primitive's body. Auto-named local nets are prefixed by the primitive's
local handle so multiple primitives in one cell don't collide on shared
default names (`mid`, `clk_n`, `m_out`, etc.).

Device-naming convention:
  inv NAME   → <NAME>_pmos, <NAME>_nmos
  nand2 NAME → <NAME>_pmos_a, <NAME>_pmos_b, <NAME>_nmos_a, <NAME>_nmos_b
  dff NAME   → <NAME>_<role>_<polarity> where:
               <role> ∈ clk_inv1, clk_inv2,
                        m_tg_in, m_inv, m_fb_inv, m_tg_fb,
                        s_tg_in, s_inv
               <polarity> ∈ p (PMOS), n (NMOS)

R77 A Rust delegation surface preserved byte-exact — the
`_try_rust_digital_decl_delegation` hook routes through
`semantic_digital_decl::expand_digital_decl_from_fields` when
`CLAY_USE_RUST_PARSER=1` is set. Tests pin the equivalence:
  * `tests/test_rust_digital_decl_expansion_v101.py`

R153 (BL-153C): the `dff` / `dff_async_clear` expanders gained OPT-IN
per-MOS-role sizing (`dff_dynamic` / `keeper_scale` / `s_fb_inv_*`)
to fix the R103-diagnosed slave-latch race (uniform `s_inv` / `s_fb_inv`
lock `q` HIGH after the first capture; real ngspice 7/12 edges). At that
point the fix was opt-in only: the DEFAULT expansion still emitted the
racing uniform-sizing topology, so `device DFF : dff {...}` in a `.clay`
file was still non-functional out of the box.

R185 (BL-185, release-blocker close): the fix is now the DEFAULT, applied
as per-MOS-role sizing that keeps the 18-MOS / 24-MOS topology and every
instance name intact. A bare `dff` / `dff_async_clear` keyword expands so
that the slave FORWARD inverter `s_inv` is stronger than the slave
FEEDBACK inverter `s_fb_inv` (the keeper):

  * `s_inv` width is scaled up by `_DEFAULT_FWD_W_SCALE` (4x), and
  * `s_fb_inv` channel length is scaled up by `_DEFAULT_KEEPER_L_SCALE`
    (8x, which weakens its drive ~8x at fixed W).

So when the slave input TG opens on the active clock edge, the forward
path always wins the `s_in` contention and `q` re-captures the new value
instead of locking. Real-ngspice proven 12/12 across 40 / 70 / 100 MHz
(sky130, tt, 1.8 V) — see `tests/test_dff_primitive_fix_v185.py`. The
device count (18 / 24) and all instance / internal-net names are
UNCHANGED, so the only test expectations that move are the ones that
asserted the OLD uniform `s_inv` == `s_fb_inv` sizing (a real fix, not a
masked regression).

The Rust mirror (`crates/clay_parser/src/semantic_digital_decl.rs`)
applies the identical per-role sizing, so the Python<->Rust
byte-equivalence parity is preserved.

A caller building cells programmatically can still override the keeper
geometry with `keeper_scale` / `s_fb_inv_*` (highest precedence), or drop
the keeper entirely with `dff_dynamic` (keeperless dynamic latch, 16 / 22
MOS). The `.clay` body grammar accepts only W / L / M / nf, so a `.clay`
source always gets the functional R185 per-role default and can never
re-select the racing uniform sizing.

See `_DFF_CONTROL_KEYS` and the `expand_dff_fixed` / `repair_dff_devices`
helpers below.
"""

from __future__ import annotations

from ..ir import Device


# ─── R185 (BL-185): default per-MOS-role sizing for the DFF fix ──────
#
# The R103 lockup is a drive-strength fight on `s_in` between the slave
# forward inverter `s_inv` and the slave feedback inverter `s_fb_inv`
# (the keeper). When the two are equal-sized and the keeper is ungated,
# the forward path cannot win and `q` locks after the first capture. The
# R185 default breaks the tie by making the forward path stronger than
# the keeper, in a way that keeps the 18-MOS / 24-MOS device count and
# all instance names intact:
#
#   * the slave forward inverter `s_inv` width is multiplied by
#     `_DEFAULT_FWD_W_SCALE` (a stronger forward path), and
#   * the slave feedback inverter `s_fb_inv` channel length is multiplied
#     by `_DEFAULT_KEEPER_L_SCALE` (a longer channel = weaker keeper at
#     fixed W; lengthening L rather than shrinking W keeps the geometry
#     above the sky130 device minimum so ngspice never aborts).
#
# Real-ngspice proven (sky130, tt, 1.8 V): 12/12 capture across the
# 40 / 70 / 100 MHz design band at these factors (FWD_W=4x, KEEP_L=8x).
# The keeper must be made clearly weaker than the forward path; weaker
# factor pairs (e.g. 2x/4x) sit right at the metastable margin and do not
# capture reliably. Above the band (150 MHz) the lengthened-L keeper
# starts to lose margin (the keeperless `dff_dynamic` opt-in is the
# robust choice for >100 MHz). Both the Python expander and the Rust
# mirror apply identical factors so the byte-equivalence parity holds.
_DEFAULT_FWD_W_SCALE: float = 4.0
_DEFAULT_KEEPER_L_SCALE: float = 8.0


# ─── R153 (BL-153C): per-MOS-role sizing control keys ────────────────
#
# R103 diagnosed (and R152 PROVED at the direct-IR level) that the
# committed master-slave `dff` / `dff_async_clear` expansion races: the
# slave forward inverter `s_inv` and the slave feedback inverter
# `s_fb_inv` carry IDENTICAL drive strength and the keeper has NO
# feedback transmission gate, so it permanently drives `s_in`. When the
# slave input TG opens on the active clock edge, the two equal-strength
# inverters fight over `s_in`, the forward path cannot win, and `q` locks
# HIGH after the first 0->1 capture. Under real ngspice (sky130, tt,
# 1.8 V) the committed primitive gets only 7 of 12 clock edges right.
#
# R185 (BL-185): the fix is now the DEFAULT. The bare expansion applies
# per-MOS-role sizing (`_fwd_slave_sizing` widens `s_inv`,
# `_keeper_sizing` lengthens `s_fb_inv`) so the forward path out-drives
# the keeper on `s_in`. The 18-MOS / 24-MOS device count and every
# instance name are UNCHANGED; only the slave `s_inv` width and
# `s_fb_inv` length move from the old uniform values. The control keys
# below remain for callers that want a different keeper geometry or a
# keeperless dynamic latch.
#
# Control keys (consumed here, NEVER written into a device's sizing IR):
#   * `dff_dynamic` (truthy) — drop the `s_fb_inv` keeper pair entirely.
#       Keeperless dynamic-storage slave latch: `s_in` holds via the
#       slave inverter's gate capacitance over the half clock cycle the
#       slave TG is off. There is then NO device fighting the forward
#       path, so the R103 race is structurally impossible. Real-ngspice
#       proven 12/12 across 40 / 70 / 100 MHz (R153). This is the
#       recommended fix. `dff` -> 16 MOS; `dff_async_clear` -> 22 MOS.
#   * `keeper_scale` (0 < s <= 1) — keep the keeper but make it weaker
#       than the forward path. A value of 0 is treated as `dff_dynamic`.
#       Implemented by LENGTHENING the keeper channel (L_keeper = base_L
#       / scale) rather than shrinking W: shrinking W below the sky130
#       device minimum (0.42 um) makes ngspice abort, whereas a longer L
#       is always legal and monotonically weakens drive. Note that a
#       lengthened keeper still races at the fast band end (100 MHz) for
#       moderate scale; prefer `dff_dynamic` for robustness.
#   * `s_fb_inv_W`, `s_fb_inv_L`, `s_fb_inv_M` — explicit per-role
#       override of the keeper pair's W / L / M. Highest precedence; lets
#       a caller dial in an exact weak-keeper geometry.
#
# These keys are accepted only on the programmatic expander surface
# (callers building cells directly, and the `repair_dff_devices` /
# `build_fixed_dff_cell` helpers below). The `.clay` body grammar still
# validates against `_DEVICE_SIZE_KEYS` (W / L / M / nf) upstream in the
# transformer, so a `.clay` source carrying a control key is rejected
# there until that whitelist is extended — by design, so the Rust
# byte-equivalence parity (which only ever uses W / L / M / nf) is never
# perturbed.
_DFF_CONTROL_KEYS: frozenset[str] = frozenset({
    "dff_dynamic",
    "keeper_scale",
    "s_fb_inv_W",
    "s_fb_inv_L",
    "s_fb_inv_M",
})


def _split_dff_controls(
    sizing: dict[str, float],
) -> "tuple[dict[str, float], dict[str, float]]":
    """Split a sizing dict into (device_sizing, control_kwargs).

    `device_sizing` is the W / L / M / nf subset that every expanded MOS
    Device IR entry carries. `control_kwargs` is the per-MOS-role fix
    knobs (`dff_dynamic` / `keeper_scale` / `s_fb_inv_*`) consumed by the
    DFF expanders and NEVER leaked into a device's sizing map.

    When no control key is present the returned `control_kwargs` is empty
    and `device_sizing` is a shallow copy of the input — the expansion is
    then byte-for-byte identical to the pre-R153 behaviour.
    """
    device_sizing = {
        k: v for k, v in sizing.items() if k not in _DFF_CONTROL_KEYS
    }
    controls = {
        k: v for k, v in sizing.items() if k in _DFF_CONTROL_KEYS
    }
    return device_sizing, controls


def _keeper_sizing(
    base_sizing: dict[str, float],
    controls: dict[str, float],
) -> dict[str, float]:
    """Compute the slave-feedback (keeper) inverter sizing.

    Precedence (high to low):
      1. explicit `s_fb_inv_W` / `s_fb_inv_L` / `s_fb_inv_M` overrides,
      2. `keeper_scale` (lengthen L by 1/scale to weaken the keeper),
      3. R185 default: lengthen the keeper channel by
         `_DEFAULT_KEEPER_L_SCALE` so the keeper is weaker than the
         forward path (the slave-latch race fix).
    """
    s = dict(base_sizing)
    scale = controls.get("keeper_scale")
    explicit = (
        scale is not None
        or "s_fb_inv_W" in controls
        or "s_fb_inv_L" in controls
        or "s_fb_inv_M" in controls
    )
    if scale is not None and scale > 0.0:
        # Weaken via a longer channel: L_keeper = base_L / scale. A
        # longer channel monotonically reduces drive and is always a
        # legal geometry (unlike shrinking W below the PDK minimum).
        s["L"] = float(base_sizing.get("L", 0.5e-6)) / float(scale)
    elif not explicit:
        # R185 default per-MOS-role fix: no caller override, so weaken the
        # keeper by lengthening its channel relative to the forward path.
        s["L"] = float(base_sizing.get("L", 0.5e-6)) * _DEFAULT_KEEPER_L_SCALE
    if "s_fb_inv_W" in controls:
        s["W"] = float(controls["s_fb_inv_W"])
    if "s_fb_inv_L" in controls:
        s["L"] = float(controls["s_fb_inv_L"])
    if "s_fb_inv_M" in controls:
        s["M"] = float(controls["s_fb_inv_M"])
    return s


def _fwd_slave_sizing(
    base_sizing: dict[str, float],
    controls: dict[str, float],
) -> dict[str, float]:
    """Compute the slave-forward inverter (`s_inv`) sizing.

    R185 default: scale the forward inverter width up by
    `_DEFAULT_FWD_W_SCALE` so the forward path out-drives the keeper on
    `s_in`. Any explicit keeper override (`keeper_scale` / `s_fb_inv_*`)
    means the caller is dialing in their own per-role geometry, so the
    forward path is left at the base width and the caller owns the
    relative strength.
    """
    s = dict(base_sizing)
    explicit_keeper = (
        controls.get("keeper_scale") is not None
        or "s_fb_inv_W" in controls
        or "s_fb_inv_L" in controls
        or "s_fb_inv_M" in controls
    )
    if not explicit_keeper:
        s["W"] = float(base_sizing.get("W", 1e-6)) * _DEFAULT_FWD_W_SCALE
    return s


def _dff_is_dynamic(controls: dict[str, float]) -> bool:
    """True iff the keeper should be dropped (keeperless dynamic latch).

    Triggered by a truthy `dff_dynamic` flag OR a `keeper_scale` of 0
    (a vanishing keeper is, in the limit, no keeper at all).
    """
    if controls.get("dff_dynamic"):
        return True
    scale = controls.get("keeper_scale")
    return scale is not None and float(scale) == 0.0


def _try_rust_digital_decl_delegation(
    *,
    name: str,
    kind: str,
    pins: dict[str, str],
    sizing: dict[str, float],
) -> "list[Device] | None":
    """v0.1.00 R77 A: opt-in Rust delegation for digital_decl expansion.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt the digital-primitive expansion through the Rust
    semantic transformer
    (`semantic_digital_decl::expand_digital_decl_from_fields`).
    Returns a `list[Device]` on success or `None` on any failure
    (env var unset, extension missing, validation rejection, unknown
    kind, etc.). Python callers MUST treat `None` as "fall back to
    the canonical Python `_expand_inv` / `_expand_nand2` / `_expand_dff`
    / `_expand_dff_async_clear`".

    The Rust path is BYTE-EQUIVALENT to the Python expansion: same
    instance names (`<NAME>_pmos` / `<NAME>_nmos` for inv;
    `<NAME>_pmos_a/b` + `<NAME>_nmos_a/b` + `<NAME>_mid` for nand2;
    18-MOS clock-inverter chain + master/slave latches for dff;
    24-MOS DFF + clear-inverter + 4-NMOS pulldown for dff_async_clear),
    same pin dict keys (`d`, `g`, `s`, `b`), same device_class, same
    sizing dict (including the auto-filled `W=1u`, `L=0.5u`, `M=1.0`).
    The bit-equivalence contract is enforced by the
    `tests/test_rust_digital_decl_expansion_v101.py` suite.

    The hasattr check on `try_rust_expand_digital_decl_solo` guards
    against a bridge module that pre-dates this delegation surface —
    older deployments missing the function silently fall back to
    Python.
    """
    try:
        from .._rust_bridge import try_rust_expand_digital_decl_solo
    except (ImportError, AttributeError):
        return None
    try:
        rust_devs = try_rust_expand_digital_decl_solo(
            name,
            kind,
            dict(pins),
            dict(sizing),
        )
    except Exception:
        return None
    if rust_devs is None:
        return None
    # Translate `PyParsedDevice` instances to canonical `Device` dataclasses.
    # The pin map and sizing map are coerced to plain Python dicts so
    # downstream consumers (sizing pass, simulator drivers, IR-bridge)
    # see the exact same shape as the Python expansion's output.
    out: list[Device] = []
    for d in rust_devs:
        out.append(Device(
            instance_name=d.instance_name,
            device_class=d.device_class,
            pdk_variant=d.pdk_variant,
            pins=dict(d.pins),
            sizing={k: float(v) for k, v in dict(d.sizing).items()},
        ))
    return out


def _mos_device(
    name: str, *, kind: str,
    d: str, g: str, s: str, b: str,
    sizing: dict[str, float],
) -> Device:
    """One-line constructor for a single MOS Device IR entry.

    `kind` is `"pmos"` or `"nmos"`. We deep-copy the sizing dict so
    each transistor keeps its own copy (later sizing-pass tweaks
    on one device don't leak across siblings).
    """
    return Device(
        instance_name=name,
        device_class=kind,
        pdk_variant=None,
        pins={"d": d, "g": g, "s": s, "b": b},
        sizing=dict(sizing),
    )


def _expand_inv(
    name: str,
    pins: dict[str, str],
    sizing: dict[str, float],
) -> list[Device]:
    """`inv NAME { in; out; vdd; vss; W; L; }` → 1 PMOS + 1 NMOS.

    Pull-up PMOS: gate = in, drain = out, source = vdd, body = vdd
    Pull-down NMOS: gate = in, drain = out, source = vss, body = vss
    """
    in_net  = pins["in"]
    out_net = pins["out"]
    vdd     = pins["vdd"]
    vss     = pins["vss"]
    return [
        _mos_device(
            f"{name}_pmos", kind="pmos",
            d=out_net, g=in_net, s=vdd, b=vdd, sizing=sizing,
        ),
        _mos_device(
            f"{name}_nmos", kind="nmos",
            d=out_net, g=in_net, s=vss, b=vss, sizing=sizing,
        ),
    ]


def _expand_nand2(
    name: str,
    pins: dict[str, str],
    sizing: dict[str, float],
) -> list[Device]:
    """`nand2 NAME { a; b; out; vdd; vss; }` → 2 PMOS + 2 NMOS.

    Pull-up: two PMOS in parallel
      PMOS_a: gate = a, drain = out, source = vdd, body = vdd
      PMOS_b: gate = b, drain = out, source = vdd, body = vdd
    Pull-down: two NMOS in series
      NMOS_a: gate = a, drain = out,         source = <NAME>_mid, body = vss
      NMOS_b: gate = b, drain = <NAME>_mid,  source = vss,        body = vss
    """
    a_net   = pins["a"]
    b_net   = pins["b"]
    out_net = pins["out"]
    vdd     = pins["vdd"]
    vss     = pins["vss"]
    mid_net = f"{name}_mid"
    return [
        _mos_device(
            f"{name}_pmos_a", kind="pmos",
            d=out_net, g=a_net, s=vdd, b=vdd, sizing=sizing,
        ),
        _mos_device(
            f"{name}_pmos_b", kind="pmos",
            d=out_net, g=b_net, s=vdd, b=vdd, sizing=sizing,
        ),
        _mos_device(
            f"{name}_nmos_a", kind="nmos",
            d=out_net, g=a_net, s=mid_net, b=vss, sizing=sizing,
        ),
        _mos_device(
            f"{name}_nmos_b", kind="nmos",
            d=mid_net, g=b_net, s=vss, b=vss, sizing=sizing,
        ),
    ]


def _expand_dff(
    name: str,
    pins: dict[str, str],
    sizing: dict[str, float],
) -> list[Device]:
    """`dff NAME { d; clk; q; vdd; vss; }` → 18 MOS functional DFF.

    Positive-edge-triggered master-slave D flip-flop with transmission-
    gate front-ends and a feedback inverter on the master latch. The
    layout below mirrors the classic Rabaey CMOS Digital text §7.5.2,
    with the R185 fix: the slave forward inverter `s_inv` is sized
    stronger than the slave feedback inverter `s_fb_inv` (the keeper) so
    the forward path wins the `s_in` contention and `q` re-captures on
    every active edge instead of locking (the R103 race). Device count
    (18) and all instance names are unchanged.

    Internal-net naming (all prefixed by `<NAME>_`):
      clk_n    — inverted clock (clk → clk_n via the first clock inverter)
      clk_buf  — buffered clock (clk_n → clk_buf, equals clk in steady state)
      m_in     — master latch storage node (D-side after the input TG)
      m_out    — master inverter output (and slave-TG input)
      m_fb     — master feedback-inverter output (drives feedback TG)
      s_in     — slave latch storage node (after the slave-input TG)

    The flip-flop captures D on the RISING edge of clk:
      When clk=0:  master TG_in passes D to m_in; slave TG_in is OFF
                   so q holds the previous value.
      When clk=1:  master TG_in is OFF (master holds via m_fb → TG_fb);
                   slave TG_in passes m_out to s_in; q updates.

    Body ties: PMOS body to vdd, NMOS body to vss (matches every other
    expansion in this module).
    """
    # R153/R185: peel off the per-MOS-role fix knobs. With no control key
    # present `controls` is empty and the R185 default per-role sizing is
    # applied: the slave forward inverter `s_inv` is widened
    # (`_fwd_slave_sizing`) and the keeper `s_fb_inv` is lengthened
    # (`_keeper_sizing`) so the forward path out-drives the keeper on
    # `s_in`. Device count (18) is unchanged. With `dff_dynamic` the
    # keeper pair is omitted (16 MOS, keeperless dynamic slave); with
    # `keeper_scale` / `s_fb_inv_*` the caller sizes the keeper itself.
    sizing, controls = _split_dff_controls(sizing)
    dynamic = _dff_is_dynamic(controls)
    keeper_sizing = _keeper_sizing(sizing, controls)
    fwd_slave_sizing = _fwd_slave_sizing(sizing, controls)

    d_net   = pins["d"]
    clk_net = pins["clk"]
    q_net   = pins["q"]
    vdd     = pins["vdd"]
    vss     = pins["vss"]

    # Internal nets — prefix every name with `<NAME>_` so multiple
    # DFFs in one cell don't share `m_in` / `clk_n` accidentally.
    clk_n   = f"{name}_clk_n"
    clk_buf = f"{name}_clk_buf"
    m_in    = f"{name}_m_in"
    m_out   = f"{name}_m_out"
    m_fb    = f"{name}_m_fb"
    s_in    = f"{name}_s_in"

    devices: list[Device] = []

    # ── Clock-inversion chain (2 inverters, 4 MOS) ────────────
    # Inverter 1: clk → clk_n
    devices.append(_mos_device(
        f"{name}_clk_inv1_p", kind="pmos",
        d=clk_n, g=clk_net, s=vdd, b=vdd, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_clk_inv1_n", kind="nmos",
        d=clk_n, g=clk_net, s=vss, b=vss, sizing=sizing,
    ))
    # Inverter 2: clk_n → clk_buf (a buffered copy of clk)
    devices.append(_mos_device(
        f"{name}_clk_inv2_p", kind="pmos",
        d=clk_buf, g=clk_n, s=vdd, b=vdd, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_clk_inv2_n", kind="nmos",
        d=clk_buf, g=clk_n, s=vss, b=vss, sizing=sizing,
    ))

    # ── Master latch (8 MOS) ──────────────────────────────────
    # Input transmission gate (TG_in): passes D when clk=0
    #   NMOS: gate = clk_n (HIGH when clk=0 → NMOS ON)
    #   PMOS: gate = clk_buf (LOW when clk=0 → PMOS ON)
    devices.append(_mos_device(
        f"{name}_m_tg_in_n", kind="nmos",
        d=d_net, g=clk_n, s=m_in, b=vss, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_m_tg_in_p", kind="pmos",
        d=d_net, g=clk_buf, s=m_in, b=vdd, sizing=sizing,
    ))
    # Master forward inverter: m_in → m_out
    devices.append(_mos_device(
        f"{name}_m_inv_p", kind="pmos",
        d=m_out, g=m_in, s=vdd, b=vdd, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_m_inv_n", kind="nmos",
        d=m_out, g=m_in, s=vss, b=vss, sizing=sizing,
    ))
    # Master feedback inverter: m_out → m_fb
    devices.append(_mos_device(
        f"{name}_m_fb_inv_p", kind="pmos",
        d=m_fb, g=m_out, s=vdd, b=vdd, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_m_fb_inv_n", kind="nmos",
        d=m_fb, g=m_out, s=vss, b=vss, sizing=sizing,
    ))
    # Feedback transmission gate (TG_fb): closes loop when clk=1
    #   NMOS: gate = clk_buf (HIGH when clk=1 → NMOS ON)
    #   PMOS: gate = clk_n (LOW when clk=1 → PMOS ON)
    devices.append(_mos_device(
        f"{name}_m_tg_fb_n", kind="nmos",
        d=m_fb, g=clk_buf, s=m_in, b=vss, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_m_tg_fb_p", kind="pmos",
        d=m_fb, g=clk_n, s=m_in, b=vdd, sizing=sizing,
    ))

    # ── Slave latch (6 MOS) ───────────────────────────────────
    # Slave input transmission gate (TG_in): passes m_out when clk=1
    #   NMOS: gate = clk_buf (HIGH when clk=1 → NMOS ON)
    #   PMOS: gate = clk_n (LOW when clk=1 → PMOS ON)
    devices.append(_mos_device(
        f"{name}_s_tg_in_n", kind="nmos",
        d=m_out, g=clk_buf, s=s_in, b=vss, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_s_tg_in_p", kind="pmos",
        d=m_out, g=clk_n, s=s_in, b=vdd, sizing=sizing,
    ))
    # Slave forward inverter: s_in → q. R185: sized stronger than the
    # keeper (`fwd_slave_sizing`) so it wins the `s_in` contention.
    devices.append(_mos_device(
        f"{name}_s_inv_p", kind="pmos",
        d=q_net, g=s_in, s=vdd, b=vdd, sizing=fwd_slave_sizing,
    ))
    devices.append(_mos_device(
        f"{name}_s_inv_n", kind="nmos",
        d=q_net, g=s_in, s=vss, b=vss, sizing=fwd_slave_sizing,
    ))
    # Slave feedback inverter: q → s_in (the "keeper"). This pair was the
    # R103 race source: at UNIFORM sizing it fought the forward path over
    # `s_in` whenever the slave input TG opened, locking `q` after the
    # first capture.
    #
    #   * default (R185, no control key): keep the pair but weaken it
    #     (`keeper_sizing` lengthens its channel by `_DEFAULT_KEEPER_L_SCALE`)
    #     while the slave forward inverter is widened. The forward path
    #     then out-drives the keeper and `q` re-captures every edge. 18
    #     MOS, count + names unchanged. Real-ngspice proven 12/12 across
    #     40 / 70 / 100 MHz.
    #   * `keeper_scale` / `s_fb_inv_*`: caller dials in the keeper
    #     geometry explicitly (forward path left at base width).
    #   * `dff_dynamic`: omit the keeper entirely (16 MOS, keeperless
    #     dynamic-storage slave — race structurally impossible).
    if not dynamic:
        devices.append(_mos_device(
            f"{name}_s_fb_inv_p", kind="pmos",
            d=s_in, g=q_net, s=vdd, b=vdd, sizing=keeper_sizing,
        ))
        devices.append(_mos_device(
            f"{name}_s_fb_inv_n", kind="nmos",
            d=s_in, g=q_net, s=vss, b=vss, sizing=keeper_sizing,
        ))

    return devices


def _expand_dff_async_clear(
    name: str,
    pins: dict[str, str],
    sizing: dict[str, float],
) -> list[Device]:
    """`dff_async_clear NAME { d; clk; q; clr; vdd; vss; }` → 24 MOS.

    Same master-slave DFF as `_expand_dff` (18 MOS), with an
    additional clear-inverter (2 MOS) producing the internal
    `<NAME>_clr_n` net and a 4-NMOS pull-down array (4 MOS) on the
    latch storage nodes.

    The base-DFF expansion is invoked first so all 18 of its devices
    keep their original instance names + internal-net names; the
    async-clear devices are appended with the `<NAME>_clr_inv_*` /
    `<NAME>_mclr_*` prefix convention so they don't collide with the
    base DFF's device-name space.

    Internal nets introduced by the async-clear extension:
      clr_n        — inverted clear, drives the pull-down NMOS gates

    Pull-down NMOS naming (4 devices):
      <NAME>_mclr_m_out  : drain=<NAME>_m_out, gate=<NAME>_clr_n
      <NAME>_mclr_m_fb   : drain=<NAME>_m_fb,  gate=<NAME>_clr_n
      <NAME>_mclr_q      : drain=q,            gate=<NAME>_clr_n
      <NAME>_mclr_s_in   : drain=<NAME>_s_in,  gate=<NAME>_clr_n

    When clr = LOW:
      clr_n = HIGH → all 4 pull-down NMOS conduct → m_out, m_fb, q,
      s_in all clamped to vss. The forward inverters fight (their
      PMOS pulls up; the pull-down NMOS pulls down) but with the
      pull-down NMOS sized comparably to the inverter NMOS, the
      pull-down wins for a w/ ratio ≥ 1. ngspice transient converges
      to all four storage nodes ≈ 0 V within a few hundred ps.

    When clr returns HIGH (released):
      clr_n = LOW → all 4 pull-down NMOS turn off. The latch resumes
      normal operation from the cleared state (q = 0; on the next
      rising clk, q = D regardless of the pre-clear history).
    """
    # The base DFF honours the R153 per-MOS-role fix knobs (dynamic /
    # keeper_scale / s_fb_inv_*) carried in `sizing`. Strip those control
    # keys here too so the clear-inverter + pull-down array devices added
    # below never inherit a bogus `dff_dynamic` / `keeper_scale` entry in
    # their own sizing IR. With no control key present, `clean_sizing`
    # equals the input and the async-clear extension is byte-identical to
    # the pre-R153 expansion.
    devices = _expand_dff(name, pins, sizing)
    clean_sizing, _controls = _split_dff_controls(sizing)
    sizing = clean_sizing

    # Internal clr_n net — driven by a single CMOS inverter that takes
    # the external `clr` input and emits its complement.
    clr_net = pins["clr"]
    vdd     = pins["vdd"]
    vss     = pins["vss"]
    clr_n   = f"{name}_clr_n"
    m_out   = f"{name}_m_out"
    m_fb    = f"{name}_m_fb"
    s_in    = f"{name}_s_in"
    q_net   = pins["q"]

    # ── Clear inverter (2 MOS) ────────────────────────────────
    # clr → clr_n. PMOS pulls clr_n up to vdd when clr is LOW;
    # NMOS pulls clr_n down to vss when clr is HIGH. Standard CMOS
    # inverter.
    devices.append(_mos_device(
        f"{name}_clr_inv_p", kind="pmos",
        d=clr_n, g=clr_net, s=vdd, b=vdd, sizing=sizing,
    ))
    devices.append(_mos_device(
        f"{name}_clr_inv_n", kind="nmos",
        d=clr_n, g=clr_net, s=vss, b=vss, sizing=sizing,
    ))

    # ── Pull-down NMOS array (4 MOS) ──────────────────────────
    # One NMOS per latch storage node. Gate = clr_n; conducts when
    # clr is LOW (clear asserted). Source = vss; pulls the storage
    # node to ground.
    for label, target_net in (
        ("m_out", m_out),
        ("m_fb",  m_fb),
        ("q",     q_net),
        ("s_in",  s_in),
    ):
        devices.append(_mos_device(
            f"{name}_mclr_{label}", kind="nmos",
            d=target_net, g=clr_n, s=vss, b=vss, sizing=sizing,
        ))

    return devices


# ─── R153 (BL-153C): public DFF-fix helpers ──────────────────────────
#
# The expanders above honour the per-MOS-role fix knobs when they are
# present in the `sizing` dict. The `.clay` body grammar validates size
# fields against `_DEVICE_SIZE_KEYS` (W / L / M / nf) BEFORE the expander
# runs, so a `.clay` source cannot itself carry a control key today. The
# two helpers below give callers that ALREADY hold a parsed / compiled
# `.clay` cell (composers, divider builders, tests) a supported way to
# carry the proven fix into the expanded primitive without changing the
# default `.clay` expansion or perturbing the Rust byte-equivalence
# parity. They are pure functions over Clay IR.


def expand_dff_fixed(
    name: str,
    pins: dict[str, str],
    sizing: dict[str, float],
    *,
    topology: str = "dynamic",
    keeper_scale: "float | None" = None,
    s_fb_inv_W: "float | None" = None,
    s_fb_inv_L: "float | None" = None,
    s_fb_inv_M: "float | None" = None,
    async_clear: bool = False,
) -> list[Device]:
    """Expand a corrected (non-racing) DFF directly.

    Thin keyword wrapper over `_expand_dff` / `_expand_dff_async_clear`
    that injects the R153 per-MOS-role control keys for the caller.

    Args:
        name / pins / sizing: same as the bare expanders. For
            `async_clear=True`, `pins` must include `clr`.
        topology: ``"dynamic"`` (default — drop the keeper, 16 MOS / 22
            MOS; real-ngspice proven 12/12) or ``"static_keeper"`` (keep
            a weak gated-by-sizing keeper, 18 MOS / 24 MOS).
        keeper_scale / s_fb_inv_W / s_fb_inv_L / s_fb_inv_M: weak-keeper
            knobs, honoured only for ``topology="static_keeper"``.

    Returns the expanded `list[Device]` with NO control key leaked into
    any device's sizing map.
    """
    merged = dict(sizing)
    if topology == "dynamic":
        merged["dff_dynamic"] = 1.0
    elif topology == "static_keeper":
        if keeper_scale is not None:
            merged["keeper_scale"] = float(keeper_scale)
        if s_fb_inv_W is not None:
            merged["s_fb_inv_W"] = float(s_fb_inv_W)
        if s_fb_inv_L is not None:
            merged["s_fb_inv_L"] = float(s_fb_inv_L)
        if s_fb_inv_M is not None:
            merged["s_fb_inv_M"] = float(s_fb_inv_M)
    else:
        raise ValueError(
            f"unknown DFF topology {topology!r}; expected 'dynamic' or "
            f"'static_keeper'"
        )
    if async_clear:
        return _expand_dff_async_clear(name, pins, merged)
    return _expand_dff(name, pins, merged)


def repair_dff_devices(
    devices: list[Device],
    *,
    mode: str = "dynamic",
    keeper_scale: "float | None" = None,
    s_fb_inv_L: "float | None" = None,
) -> list[Device]:
    """Repair already-expanded DFF device lists in place of the racing
    keeper.

    Operates on a flat `list[Device]` that came out of a `.clay` parse +
    compile (or any other source). It locates every slave feedback
    inverter pair — devices whose `instance_name` ends in `_s_fb_inv_p`
    or `_s_fb_inv_n` — and either:

      * ``mode="dynamic"`` (default): removes the keeper pair, yielding
        the keeperless dynamic-storage slave latch that R153 proved
        toggles 12/12 under real ngspice. This is how the committed
        `.clay` `dff` / `dff_async_clear` keyword's expansion is carried
        to a correct, non-racing topology from a real parsed cell.
      * ``mode="static_keeper"``: keeps the keeper pair but weakens it
        (longer channel via `keeper_scale`, or an explicit `s_fb_inv_L`).

    Returns a NEW device list; the input list is not mutated. Non-DFF
    devices pass through untouched, so a cell mixing DFFs with analog
    devices / inverters is handled correctly.
    """
    if mode not in ("dynamic", "static_keeper"):
        raise ValueError(
            f"unknown repair mode {mode!r}; expected 'dynamic' or "
            f"'static_keeper'"
        )
    out: list[Device] = []
    for dev in devices:
        is_keeper = (
            dev.instance_name.endswith("_s_fb_inv_p")
            or dev.instance_name.endswith("_s_fb_inv_n")
        )
        if not is_keeper:
            out.append(dev)
            continue
        if mode == "dynamic":
            # Drop the keeper entirely.
            continue
        # static_keeper: weaken in place (new Device so input is intact).
        new_sizing = dict(dev.sizing)
        if keeper_scale is not None and keeper_scale > 0.0:
            new_sizing["L"] = (
                float(dev.sizing.get("L", 0.5e-6)) / float(keeper_scale)
            )
        if s_fb_inv_L is not None:
            new_sizing["L"] = float(s_fb_inv_L)
        out.append(Device(
            instance_name=dev.instance_name,
            device_class=dev.device_class,
            pdk_variant=dev.pdk_variant,
            pins=dict(dev.pins),
            sizing=new_sizing,
        ))
    return out
