"""
Digital-primitive transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
`inv_decl`, `nand2_decl`, `dff_decl`, `dff_clear_decl`, `_digital_decl`.

Each digital primitive (inv / nand2 / dff / dff_async_clear) expands
to N concrete `Device` IR entries at parse time. The expanded devices
share the W/L/M sizing supplied on the primitive's body. Auto-named
local nets are prefixed by the primitive's local handle.

R77 A Rust delegation surface preserved byte-exact via
`digital_expand._try_rust_digital_decl_delegation`.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

from ..ir import Device
from ._constants import _DEVICE_SIZE_KEYS, _DIGITAL_PIN_LABELS
from .digital_expand import (
    _expand_dff, _expand_dff_async_clear, _expand_inv, _expand_nand2,
    _try_rust_digital_decl_delegation,
)


class _DigitalMixin:
    """Digital-primitive transformer methods (inv / nand2 / dff / dff_async_clear)."""

    # ─── v0.0.35 (PLL gap C): digital primitives ─────────────

    def inv_decl(self, items) -> list[Device]:
        """`inv NAME { in = ...; out = ...; vdd = ...; vss = ...; W = ...; L = ...; }`

        Expands a CMOS inverter to TWO concrete `Device` IR entries:
          * `<NAME>_pmos`: PMOS, gate=in, drain=out, source=vdd, body=vdd
          * `<NAME>_nmos`: NMOS, gate=in, drain=out, source=vss, body=vss

        Both transistors share the W/L/M sizing supplied in the body.
        Default sizing if not specified: W=1u / L=0.5u / M=1.
        """
        return self._digital_decl(items, kind="inv")

    def nand2_decl(self, items) -> list[Device]:
        """`nand2 NAME { a = ...; b = ...; out = ...; vdd = ...; vss = ...; }`

        Expands a 2-input CMOS NAND to FOUR concrete `Device` entries:
          * `<NAME>_pmos_a`: PMOS, gate=a, drain=out, source=vdd, body=vdd
          * `<NAME>_pmos_b`: PMOS, gate=b, drain=out, source=vdd, body=vdd
            (both PMOS in parallel — out is pulled HIGH unless both
            inputs are HIGH)
          * `<NAME>_nmos_a`: NMOS, gate=a, drain=out, source=<NAME>_mid,
                             body=vss
          * `<NAME>_nmos_b`: NMOS, gate=b, drain=<NAME>_mid, source=vss,
                             body=vss
            (NMOS in series — out is pulled LOW only when both inputs
            HIGH)
        """
        return self._digital_decl(items, kind="nand2")

    def dff_decl(self, items) -> list[Device]:
        """`dff NAME { d = ...; clk = ...; q = ...; vdd = ...; vss = ...; }`

        Expands a positive-edge-triggered master-slave D flip-flop to
        18 concrete `Device` entries. Topology (textbook
        transmission-gate flavor):
          * Clock-inversion chain (4 MOS): clk → <NAME>_clk_n → <NAME>_clk_buf
            so we have BOTH the inverted clock (`clk_n`) AND a buffered
            copy of clk (`clk_buf`) available for the transmission
            gates. Two phases are needed because each TG uses one NMOS
            (driven by the active-high phase) and one PMOS (driven by
            the active-low phase).
          * Master latch (8 MOS): input TG passes D when clk=0 (clk_n=1
            opens NMOS, clk_buf=0 opens PMOS), inverter produces
            <NAME>_m_out, feedback inverter produces <NAME>_m_fb, and a
            feedback TG closes the loop when clk=1.
          * Slave latch (6 MOS): input TG passes m_out when clk=1
            (clk_buf=1 opens NMOS, clk_n=0 opens PMOS), inverter
            produces q. (Note: the standard textbook DFF puts a
            feedback inverter on the slave too, but ngspice transient
            convergence is generally fine without it because Q drives
            an external load that holds the state. We keep the slave
            simple at 6 MOS to match the ~18 MOS budget in the surface
            spec, and reserve the headroom for the feedback inverter
            on the master where it actually matters for hold-time.)
        Total: 4 + 8 + 6 = 18 MOS.
        """
        return self._digital_decl(items, kind="dff")

    def dff_clear_decl(self, items) -> list[Device]:
        """`dff_async_clear NAME { d=...; clk=...; q=...; clr=...; vdd=...; vss=...; }`

        Expands a master-slave DFF with an active-LOW asynchronous
        clear input to 24 concrete `Device` entries (~22 in the v0.0.40
        ship target — 18 base DFF + 2 clear inverter + 4 pull-downs).

        Topology (textbook DFF + async-reset stub):
          * 18-MOS base DFF — identical to `dff` expansion (see
            `_expand_dff` for the master-slave / clock-inversion-chain
            details).
          * Clear inverter (2 MOS): clr → <NAME>_clr_n via an inverter
            so the pull-down NMOS array can be driven gate-high when
            clr is asserted LOW. PMOS source = vdd, NMOS source = vss.
          * Pull-down array (4 NMOS): one NMOS per latch storage node
            (m_out, m_fb, q, s_in). Drain = storage node, gate = clr_n,
            source = vss, body = vss. When clr_n is HIGH (clr asserted
            LOW = clear active), every storage node is force-pulled to
            vss, overpowering whatever the regular CMOS structure was
            trying to hold. Once clr returns HIGH (inactive), the NMOS
            array opens and the DFF resumes normal clock-driven
            operation from a known-zero state.

        Why these four nodes specifically:
          * m_out and m_fb form the master latch's static-storage pair
            (each is the output of one of the back-to-back inverters
            that latches the master). Forcing both to 0 puts the master
            into the "master sees D=0" state; on the next falling clk
            edge the slave will sample 0.
          * q is the external output AND the slave forward inverter's
            drain. Pulling q to vss directly forces the observable
            output to 0 even before the next clock edge — critical for
            a textbook PFD whose NAND2 reset path needs both DFFs to
            reset within a fraction of a reference cycle.
          * s_in is the slave latch's static-storage node (output of
            the slave feedback inverter). Forcing s_in to 0 means the
            slave forward inverter drives q = vdd... which would FIGHT
            the simultaneous direct pull-down on q. ngspice's transient
            convergence handles the contention (both legs are minimum-
            sized NMOS pull-downs vs CMOS inverters; the inverters
            lose). When clr returns HIGH, the slave feedback loop
            re-establishes q = NOT s_in correctly because s_in is
            already 0 → s_fb_inv pulls s_in toward vdd, but q is no
            longer being pulled down, so the slave inverter pair
            settles at (s_in=1, q=0) — exactly the cleared state.

        Wiring of pull-down NMOS Mclr_<node>:
            drain  = <node>     (m_out / m_fb / q / s_in)
            gate   = clr_n      (NMOS conducts when clr_n=HIGH, i.e.
                                 when clr=LOW = clear asserted)
            source = vss
            body   = vss
        """
        return self._digital_decl(items, kind="dff_async_clear")

    def _digital_decl(
        self, items, *, kind: str,
    ) -> list[Device]:
        """Shared body parser for inv / nand2 / dff declarations.

        items layout (after grammar transforms):
          [KW_token, CNAME(instance_name),
           device_body_field, device_body_field, ...]
        where each `device_body_field` is a tagged triple `("pin", key,
        net)` from `device_pin_assignment` or `("size", key, value)`
        from `device_size_field`.

        Validates pin assignments against the per-kind required set and
        size keys against `_DEVICE_SIZE_KEYS`. Returns the expanded
        device list (1 inverter → 2 MOS; 1 NAND2 → 4 MOS; 1 DFF →
        18 MOS).
        """
        instance_name = str(items[1])
        body_fields = items[2:]

        required_pins = _DIGITAL_PIN_LABELS[kind]
        pins: dict[str, str] = {}
        sizing: dict[str, float] = {}

        for entry in body_fields:
            tag, key, value = entry
            if tag == "pin":
                if key not in required_pins:
                    raise ValueError(
                        f"{kind} {instance_name!r}: unknown pin label "
                        f"{key!r}. Valid pins: {sorted(required_pins)}."
                    )
                if key in pins:
                    raise ValueError(
                        f"{kind} {instance_name!r}: pin {key!r} assigned "
                        f"more than once."
                    )
                pins[key] = value
            elif tag == "size":
                if key not in _DEVICE_SIZE_KEYS:
                    raise ValueError(
                        f"{kind} {instance_name!r}: unknown size key "
                        f"{key!r}. Valid keys: "
                        f"{sorted(_DEVICE_SIZE_KEYS)}."
                    )
                if key in sizing:
                    raise ValueError(
                        f"{kind} {instance_name!r}: size key {key!r} "
                        f"assigned more than once."
                    )
                if key in ("M", "nf"):
                    if not float(value).is_integer():
                        raise ValueError(
                            f"{kind} {instance_name!r}: {key} must be "
                            f"an integer, got {value!r}."
                        )
                    sizing[key] = float(int(value))
                else:
                    sizing[key] = float(value)
            else:
                raise ValueError(
                    f"{kind} {instance_name!r}: unknown body-field "
                    f"tag {tag!r}."
                )

        # Required-pin check — every pin in the contract MUST be
        # assigned. Missing a pin would leave a transistor terminal
        # floating after expansion, which produces opaque ngspice
        # errors at sim time.
        missing = required_pins - pins.keys()
        if missing:
            raise ValueError(
                f"{kind} {instance_name!r}: missing required pin "
                f"assignment(s): {sorted(missing)}. Digital primitive "
                f"{kind!r} needs all of {sorted(required_pins)}."
            )

        # Default sizing: digital primitives ship with conservative
        # defaults so the user can write `inv inv1 { in = a; out = b;
        # vdd = vdd; vss = vss; }` without spelling out W/L. The
        # sky130 minimum is L=0.15u; we pick L=0.5u for ngspice
        # convergence headroom and W=1u as a starting point.
        sizing.setdefault("W", 1e-6)
        sizing.setdefault("L", 0.5e-6)
        sizing.setdefault("M", 1.0)

        # v0.1.00 R77 A: Rust delegation hook. When
        # `CLAY_USE_RUST_PARSER=1` is set and the Rust extension is
        # loadable, attempt the digital-primitive expansion through
        # the Rust semantic transformer
        # (`semantic_digital_decl::expand_digital_decl_from_fields`).
        # Returns a `list[Device]` on success, or `None` on any
        # failure (env var unset, extension missing, validation
        # rejection, unknown kind). Bit-equivalent to the Python
        # expansion below — same instance names, same pin dict keys
        # (`d`, `g`, `s`, `b`), same device_class, same sizing dict
        # (including the post-default-fill `W=1u`, `L=0.5u`, `M=1.0`
        # for omitted keys). Falls back to Python on None.
        rust_result = _try_rust_digital_decl_delegation(
            name=instance_name,
            kind=kind,
            pins=pins,
            sizing=sizing,
        )
        if rust_result is not None:
            return rust_result

        # Dispatch to the per-kind expansion function. Each expander
        # returns a list of fully-formed Device IR entries with
        # auto-named local nets prefixed by `<instance_name>_`.
        if kind == "inv":
            return _expand_inv(instance_name, pins, sizing)
        if kind == "nand2":
            return _expand_nand2(instance_name, pins, sizing)
        if kind == "dff":
            return _expand_dff(instance_name, pins, sizing)
        if kind == "dff_async_clear":
            return _expand_dff_async_clear(instance_name, pins, sizing)
        raise ValueError(
            f"digital-primitive expansion: unknown kind {kind!r} "
            f"(internal bug)."
        )
