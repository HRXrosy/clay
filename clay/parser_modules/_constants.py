"""
Parse-time constants + small pure helpers (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B). Every
constant + helper here matches its pre-split definition byte-exactly — the
sole change is the new module location. The legacy `clay.parser` module
re-exports each name through a thin shim so existing imports
(`from clay.parser import _VALID_ROLE_KINDS, ...`) keep resolving.

Contents:
  * Pin-name canonicalization tables (MOS long→short, passive set)
  * Device / digital-primitive / role / functional-role validation
    sets and per-kind metadata.
  * Unit conversion table (`_UNITS`) + `_resolve_unit_multiplier`.
  * `_extract_derived_refs` — regex scan for `<setup>.<measure>` refs
    inside a derived-measure expression string.
  * `_validate_combinator_stage_refs` — recursive `stage_ref` range guard.
"""

from __future__ import annotations

from typing import Any


# v0.0.28 (device primitives): canonical short-form pin keys used by
# `Device.pins` for MOS-family devices. The grammar accepts both these
# short forms AND their long-form aliases (drain, gate, source, body),
# canonicalizing to the short form at parse time so the IR has one shape.
_MOS_PIN_LONG_TO_SHORT: dict[str, str] = {
    "drain":  "d",
    "gate":   "g",
    "source": "s",
    "body":   "b",
    # Self-aliases — accept short form unchanged.
    "d": "d", "g": "g", "s": "s", "b": "b",
}

# Three-terminal bipolar schema. It remains disjoint from the MOS aliases so a
# collector on an nmos is rejected instead of being silently reinterpreted.
_BJT_PIN_LONG_TO_SHORT: dict[str, str] = {
    "collector": "c",
    "base": "b",
    "emitter": "e",
    "c": "c", "b": "b", "e": "e",
}
_BJT_DEVICE_CLASSES: frozenset[str] = frozenset({"pnp", "npn"})
_BJT_SIZE_KEYS: frozenset[str] = frozenset({"area", "M"})

# Same for two-terminal passives: capacitors and resistors use `p` / `n`
# (positive / negative). Surface form is identical to IR form — no aliases.
_PASSIVE_PIN_NAMES: frozenset[str] = frozenset({"p", "n"})

# v0.0.28 (device primitives): keys allowed in a `device { ... }` body's
# size-field list. `W`, `L` are physical width/length; `M` is the device
# multiplier (integer); `nf` is the finger count (also integer). Other
# keys are rejected at parse time so typos surface early.
_DEVICE_SIZE_KEYS: frozenset[str] = frozenset({"W", "L", "M", "nf"})

# Capacitor / resistor size-field keys. `value` is in F for cap, Ohm for
# res. `M` is the SPICE inline device multiplier (N parallel unit devices):
# a cap's effective capacitance is value*M, a resistor's effective
# resistance is value/M. `M` is optional and defaults to 1.
_PASSIVE_SIZE_KEYS: frozenset[str] = frozenset({"value", "M"})


# v0.0.35 (PLL gap C): digital-primitive required pins. Each digital
# primitive declares a fixed pin contract — every entry in the per-kind
# set MUST be assigned in the surface body and only the listed labels
# are accepted. Sizing keys reuse `_DEVICE_SIZE_KEYS` (W / L / M / nf).
#
# `inv`:    one CMOS inverter (1 PMOS + 1 NMOS sharing input gate +
#           output drain).
# `nand2`:  two-input CMOS NAND (2 PMOS in parallel pulling up, 2 NMOS
#           in series pulling down).
# `dff`:    positive-edge-triggered master-slave D flip-flop (~18 MOS
#           textbook implementation — transmission-gate master + slave
#           with feedback inverter, plus a clock-inversion chain that
#           emits both clk_n and clk_buf phases).
# `dff_async_clear` (v0.0.40): same DFF + active-LOW asynchronous clear
#           input. When `clr` is LOW, the four latch storage nodes
#           (m_out, m_fb, q, s_in) are force-pulled to vss by an NMOS
#           array gated on `clr_n` (= NOT clr). Expansion: 18-MOS base
#           DFF + 2-MOS clear inverter + 4-MOS pull-down array = 24 MOS.
_DIGITAL_PIN_LABELS: dict[str, frozenset[str]] = {
    "inv":             frozenset({"in", "out", "vdd", "vss"}),
    "nand2":           frozenset({"a", "b", "out", "vdd", "vss"}),
    "dff":             frozenset({"d", "clk", "q", "vdd", "vss"}),
    "dff_async_clear": frozenset({"d", "clk", "q", "clr", "vdd", "vss"}),
}


# v0.0.31 (G3 closeout): valid role-decl kinds and per-kind metadata.
#
# Each role kind expands into N concrete `Device` entries with a fixed
# device_class (`pmos` / `nmos`), a deterministic supply/body net
# (`vdd` for PMOS-family, `vss` for NMOS-family), and a per-kind set of
# auto-named or override-named pins. The expansion logic lives in
# `_ClayTransformer.role_decl` so role decls are flattened to devices at
# parse time — the compiler only ever sees the underlying Device list.
_VALID_ROLE_KINDS: frozenset[str] = frozenset({
    "pmos_pair", "nmos_pair",
    "pmos_mirror", "nmos_mirror",
    "pmos_cs", "nmos_cs",
})

# Size keys accepted in any role body. `W`, `L`, `M`, `nf` are the same
# device-level fields validated for `device { ... }` bodies. `gm_id` is
# the role-level intent — forwarded to every contained device's sizing
# dict so the sizing pass picks it up identically to a stage-template
# cell. Other keys raise a precise error so typos surface early.
_ROLE_SIZE_KEYS: frozenset[str] = frozenset({
    "W", "L", "M", "nf", "gm_id",
})

# Per-kind single-net pin override labels. Each label, when present in
# the role body, replaces the corresponding auto-named pin in the
# expansion.
_ROLE_PIN_LABELS: dict[str, frozenset[str]] = {
    "pmos_pair":   frozenset({"tail"}),
    "nmos_pair":   frozenset({"tail"}),
    "pmos_mirror": frozenset({"in", "out"}),
    "nmos_mirror": frozenset({"in", "out"}),
    "pmos_cs":     frozenset({"gate", "drain"}),
    "nmos_cs":     frozenset({"gate", "drain"}),
}

# Per-kind two-net pin override labels. `gates = (a, b);` and
# `drains = (a, b);` are only meaningful on pair roles.
_ROLE_PAIR_PIN_LABELS: dict[str, frozenset[str]] = {
    "pmos_pair": frozenset({"gates", "drains"}),
    "nmos_pair": frozenset({"gates", "drains"}),
    # mirrors / cs have no two-net override (the diode/mirror legs each
    # have at most one configurable terminal net)
    "pmos_mirror": frozenset(),
    "nmos_mirror": frozenset(),
    "pmos_cs":     frozenset(),
    "nmos_cs":     frozenset(),
}


# v0.0.62 (functional roles): valid functional-role keywords. Each
# keyword names an analog-function noun rather than a transistor-bundle
# noun. The lowering from functional role → underlying structural role
# or device primitive happens later in the compiler / lowering layer —
# this parser only validates the surface keyword and captures the
# attribute map. The grammar terminal `FUNCTIONAL_ROLE_KW` already
# restricts the surface to these strings; the set below is the
# defensive guard the transformer checks against.
#
# v0.0.90 (Round 66): extended from 6 to 12 keywords. Second batch adds
# six new behavioral-noun keywords: cascode_helper, pole_tracking_cap,
# dual_feedforward_pair, loop_filter_pole, current_mirror, diode_load.
# Each new keyword shares the same ParsedFunctionalRole IR node shape
# as the v0.0.86 first batch; only the attribute schema differs per
# keyword (described in `docs/superpowers/specs/clay-functional-roles-spec.html`).
#
# v0.0.92 (Round 68 Agent G): extended from 12 to 18 keywords. Third
# batch adds six behavioral nouns for CMFB / switched-cap / gain-boost /
# class-AB / dynamic-bias topologies: common_mode_feedback,
# sample_hold_switch, regulated_cascode, gain_boost_amp,
# output_class_AB, dynamic_bias. The lowering / real invariant check
# for the third batch is deferred to Round 69+ — Round 68 lands only
# the parser + Rust + placeholder check entries.
#
# v0.0.94 (Round 70 Agent C): extended from 18 to 23 keywords. Fourth
# batch adds five behavioral nouns for non-amp categories
# (LDO / Bandgap / PLL): bandgap_core, dropout_pass_device,
# ldo_feedback_divider, ptat_branch, pll_loop_filter. The lowering /
# real invariant check for the fourth batch is deferred to Round 71+ —
# Round 70 lands only the parser + Rust + placeholder check entries.
#
# v0.1.x: `functional_group` is a generic source-level grouping
# primitive. It carries device membership, labels, and layout/design
# intent for downstream tools such as the living schematic viewer
# without forcing every circuit-specific role token into this closed
# keyword set.
_VALID_FUNCTIONAL_ROLE_KINDS: frozenset[str] = frozenset({
    "functional_group",
    # First batch (v0.0.86)
    "input_transconductor",
    "output_buffer",
    "feedforward_path",
    "dominant_pole_cap",
    "compensation_zero",
    "tail_current_source",
    # Second batch (v0.0.90 Round 66)
    "cascode_helper",
    "pole_tracking_cap",
    "dual_feedforward_pair",
    "loop_filter_pole",
    "current_mirror",
    "diode_load",
    # Third batch (v0.0.92 Round 68 Agent G) — promoted to real checks
    # by Round 68 Agent H; spelling remains byte-exact.
    "common_mode_feedback",
    "sample_hold_switch",
    "regulated_cascode",
    "gain_boost_amp",
    "output_class_AB",
    "dynamic_bias",
    # Fourth batch (v0.0.94 Round 70 Agent C) — placeholder invariant
    # checks for non-amp categories (LDO / Bandgap / PLL).
    "bandgap_core",
    "dropout_pass_device",
    "ldo_feedback_divider",
    "ptat_branch",
    "pll_loop_filter",
})


# Valid PDK corner names accepted by the `corners { ... }` block.
# Mirrors `clay.pdk.sky130._VALID_CORNERS`. We don't import that symbol
# directly to keep the parser PDK-agnostic at import time and to avoid a
# circular import (parser is imported during clay.__init__ paths that
# can predate the PDK adapter being loaded). When a second PDK appears,
# this should move into a PDK-aware validation hook in the CLI layer.
_VALID_CORNERS: frozenset[str] = frozenset({"tt", "ss", "ff", "sf", "fs"})


# v0.0.22 (G10): valid port-role annotations accepted in
# `cell Foo(name: role, ...)` signatures. The role is informational
# metadata — `compile_cell_def` still checks port *names* only — but
# the parser rejects unknown roles so the LLM gets a clear error
# instead of silent semantic drift.
_VALID_PORT_ROLES: frozenset[str] = frozenset({
    "input", "output", "inout",
    "power_p", "power_n",
    "bias", "ground",
})


# v0.0.29 (G4 multi-setup verify): the keys allowed inside a
# `drive { ... }` block. The transformer accepts exactly ONE drive
# keyword per setup — mixing forms (`ac_diff = 1; ac_common = 1;`)
# is rejected at parse time as a surface mistake. Future drive keys
# (e.g. `ac_input_n`, `dc_offset`) get added here.
#
# v0.0.33 (LDO support): four new drive modes for non-amplifier
# circuits — DC sweep on Vin (line regulation), DC sweep on Iload
# (load regulation), transient step on Iload (load transient), and
# transient step on Vin (line transient). The corresponding rewrite
# logic lives in `clay.verifier.rewrite_testbench_for_drive_mode`
# and consumes an LDO-shaped testbench (see
# `clay.templates.ldo_testbench.build_ldo_testbench`). The opamp
# drive modes above remain untouched — they assume the differential
# AC rig produced by `build_ac_testbench`. Mixing an LDO drive mode
# against an opamp testbench (or vice-versa) raises ValueError in
# the rewriter with a precise error message.
_VALID_DRIVE_KEYS: frozenset[str] = frozenset({
    "ac_diff", "ac_common", "ac_supply_p", "ac_supply_n",
    "dc_vin_sweep", "dc_iload_sweep",
    "tran_load_step", "tran_line_step",
})


def _extract_derived_refs(expr_str: str) -> list[tuple[str, str]]:
    """Scan a derived-measure expression string for `<setup>.<measure>` refs.

    The verifier re-parses the same string at evaluation time; here we
    only need the references for parse-time validation against the
    declared setups. We use a small regex rather than a full
    re-parse — the expression is already validated by the grammar and
    consists of tokens joined by single spaces.
    """
    import re as _re
    refs: list[tuple[str, str]] = []
    for m in _re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\.([A-Za-z_][A-Za-z_0-9]*)\b", expr_str):
        refs.append((m.group(1), m.group(2)))
    return refs


# Unit conversion table (SI base where applicable; dB/dBm are passthrough)
_UNITS: dict[str, float] = {
    # Frequency
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9,
    # Voltage
    "V": 1.0, "mV": 1e-3, "uV": 1e-6,
    # Current
    "A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12,
    # Capacitance
    "F": 1.0, "pF": 1e-12, "fF": 1e-15, "nF": 1e-9, "uF": 1e-6,
    # Resistance
    "Ohm": 1.0, "kOhm": 1e3, "MOhm": 1e6,
    # Inductance
    "H": 1.0, "pH": 1e-12, "nH": 1e-9, "uH": 1e-6, "mH": 1e-3,
    # Time
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
    # Length
    "m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9,
    # Angle
    "deg": 1.0, "rad": 57.2957795,
    # Logarithmic
    "dB": 1.0, "dBm": 1.0,
    # v0.0.28 (device primitives): bare SPICE-style SI prefixes.
    # In `device { W = 20u; L = 0.5u; }` the bare `u` is canonical
    # SPICE shorthand for "micro" (10^-6). The unit-less number is
    # implicitly in whichever SI base the field expects (meters for
    # W/L, Farads for cap value, etc.). Bare `m` is intentionally
    # omitted — it conflicts with the existing "meters" entry and
    # the milli-prefix is rarely needed in transistor sizing.
    # Lowercase `f` is femto (10^-15); uppercase `F` is already
    # Farads in the table above and stays that way.
    "k": 1e3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def _resolve_unit_multiplier(unit: str) -> float:
    """Resolve a (possibly compound) unit suffix to its SI multiplier.

    Simple form (`V`, `kHz`, `pF`, ...): direct `_UNITS` lookup.

    Compound form `X/Y` (v0.0.23, G12 — e.g. `V/us`, `kHz/V`): both
    halves are looked up independently and the multiplier returned is
    `_UNITS[X] / _UNITS[Y]`. Example: `1V/us` parses as
    `1.0 * (1.0 / 1e-6) = 1e6` — the SI value of "1 V per microsecond"
    (slew rate). Triple-slash forms like `X/Y/Z` are rejected at this
    layer (the grammar's UNIT regex only matches at most one `/`, so
    they should never reach us; the explicit check below is a
    defensive guard).

    Unknown components on either side raise ValueError with a precise
    error message so the LLM (or human) can see exactly which half
    didn't resolve.
    """
    if "/" not in unit:
        mult = _UNITS.get(unit)
        if mult is None:
            raise ValueError(f"unknown unit suffix {unit!r}")
        return mult
    # Compound form: numerator / denominator.
    parts = unit.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"compound unit {unit!r} has too many slashes — only "
            f"`X/Y` form is supported (e.g. `V/us`, `kHz/V`)."
        )
    num, den = parts[0], parts[1]
    num_mult = _UNITS.get(num)
    if num_mult is None:
        raise ValueError(
            f"unknown numerator in compound unit {unit!r}: "
            f"{num!r} not in unit table"
        )
    den_mult = _UNITS.get(den)
    if den_mult is None:
        raise ValueError(
            f"unknown denominator in compound unit {unit!r}: "
            f"{den!r} not in unit table"
        )
    return num_mult / den_mult


# ─── v0.0.92 (Round 68): combinator AST validation helpers ────────
#
# The combinator block's `build = expr;` payload is an opaque dict-tree
# (see `clay.ir.ParsedCombinatorBlock` docstring for the schema). The
# transformer needs to walk that tree to validate `stages[N]` references
# against the declared stage list length. Live module-scope so the
# transformer's `combinator_block` method can call it directly without
# growing another class method (which would clutter the dispatch table).

def _validate_combinator_stage_refs(
    node: Any,
    *,
    stages_len: int,
    block_name: str,
) -> None:
    """Recursively walk `node` and reject any `stage_ref` with an
    out-of-range index. Bare `stage_name` references are accepted as-is
    here (compile-time resolution lives in `clay.compiler`).
    """
    if not isinstance(node, dict):
        # Defensive — every well-formed AST node is a dict; leaves
        # surface as `stage_ref` / `stage_name` dicts. A non-dict at
        # any position implies a parser bug.
        return
    kind = node.get("kind")
    if kind == "stage_ref":
        idx = node.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= stages_len:
            raise ValueError(
                f"combinator {block_name!r}: `stages[{idx}]` is out of "
                f"range — declared stage list has {stages_len} "
                f"entries (valid indices 0..{stages_len - 1})."
            )
        return
    if kind == "stage_name":
        return
    if kind == "cascade":
        _validate_combinator_stage_refs(
            node.get("left"), stages_len=stages_len, block_name=block_name,
        )
        _validate_combinator_stage_refs(
            node.get("right"), stages_len=stages_len, block_name=block_name,
        )
        return
    if kind == "feedforward":
        _validate_combinator_stage_refs(
            node.get("source"), stages_len=stages_len, block_name=block_name,
        )
        _validate_combinator_stage_refs(
            node.get("sink"), stages_len=stages_len, block_name=block_name,
        )
        return
    if kind == "compensate":
        _validate_combinator_stage_refs(
            node.get("inner"), stages_len=stages_len, block_name=block_name,
        )
        return
    if kind == "mirror_match":
        _validate_combinator_stage_refs(
            node.get("left"), stages_len=stages_len, block_name=block_name,
        )
        _validate_combinator_stage_refs(
            node.get("right"), stages_len=stages_len, block_name=block_name,
        )
        return
    if kind == "cascade_chain":
        for child in node.get("stages", []):
            _validate_combinator_stage_refs(
                child, stages_len=stages_len, block_name=block_name,
            )
        return
    if kind == "decouple":
        # v0.0.95 (Round 72 D): supply-rail decoupling combinator
        # wraps a single inner stage.
        _validate_combinator_stage_refs(
            node.get("inner"), stages_len=stages_len, block_name=block_name,
        )
        return
    if kind == "sample_and_hold":
        # v0.0.99 (Round 75 C): track-and-hold combinator wraps a
        # single inner stage's output net.
        _validate_combinator_stage_refs(
            node.get("inner"), stages_len=stages_len, block_name=block_name,
        )
        return
    # Unknown kind — defensive guard so the transformer surfaces a
    # precise error if the AST grows a new node type without
    # updating this walker.
    raise ValueError(
        f"combinator {block_name!r}: AST node has unknown kind "
        f"{kind!r}; valid kinds are 'stage_ref', 'stage_name', "
        f"'cascade', 'feedforward', 'compensate', 'mirror_match', "
        f"'cascade_chain', 'decouple', 'sample_and_hold'."
    )
