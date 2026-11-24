"""
Device-primitive transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
`device_decl`, `cap_decl`, `res_decl`, `ind_decl`, `varactor_decl`,
`_passive_decl`, `device_pin_assignment`, `device_size_field`.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

from ..ir import Device
from ._constants import (
    _BJT_DEVICE_CLASSES, _BJT_PIN_LONG_TO_SHORT, _BJT_SIZE_KEYS,
    _DEVICE_SIZE_KEYS, _MOS_PIN_LONG_TO_SHORT,
    _PASSIVE_PIN_NAMES, _PASSIVE_SIZE_KEYS,
)


class _DeviceMixin:
    """Device-primitive (MOS, cap, res, ind, varactor) transformer methods."""

    # ─── v0.0.28: device primitives ──────────────────────────

    def device_decl(self, items) -> Device:
        """`device M1 : pmos { drain = d1; gate = inp; ...; W = 20u; L = 0.5u; }`

        items layout (after grammar transforms):
          [DEVICE_KW token, CNAME(instance_name), CNAME(device_kind),
           device_body_field, device_body_field, ...]
        where each `device_body_field` is a (kind, key, value) triple
        from `device_pin_assignment` or `device_size_field`.

        Builds a `Device` IR entry with canonicalized pin keys and
        validated size fields. The device_class is the bare CNAME of
        the kind (`pmos`, `nmos`, etc.); we do NOT validate it against
        the PDK adapter at this layer because the parser is PDK-agnostic.
        The compiler / driver catches unknown device classes when the
        binding map is consulted.
        """
        # items[0] is the DEVICE_KW terminal (keyword "device"); skip it.
        instance_name = str(items[1])
        device_class = str(items[2])
        body_fields = items[3:]

        if device_class == "native":
            pins, parameters = {}, []
            source_model = None
            seen_parameters = set()
            for kind, key, value in body_fields:
                if kind == "pin" and key == "model":
                    if source_model is not None:
                        raise ValueError(f"native device {instance_name!r}: duplicate model")
                    source_model = str(value)
                elif kind == "pin":
                    role = str(key).casefold()
                    if role in pins:
                        raise ValueError(f"native device {instance_name!r}: duplicate pin {role!r}")
                    pins[role] = str(value)
                elif kind == "native_param":
                    if str(key).casefold() in seen_parameters:
                        raise ValueError(f"native device {instance_name!r}: duplicate parameter {key!r}")
                    seen_parameters.add(str(key).casefold())
                    parameters.append((str(key), str(value)))
                else:
                    raise ValueError(f"native device {instance_name!r}: use native_param for original model parameters")
            if source_model is None:
                raise ValueError(f"native device {instance_name!r}: explicit model is required")
            return Device(instance_name, "native", source_model, pins, {},
                          source_model=source_model, native_parameters=tuple(parameters))

        is_bjt = device_class.lower() in _BJT_DEVICE_CLASSES
        pin_aliases = _BJT_PIN_LONG_TO_SHORT if is_bjt else _MOS_PIN_LONG_TO_SHORT
        size_keys = _BJT_SIZE_KEYS if is_bjt else _DEVICE_SIZE_KEYS
        pins: dict[str, str] = {}
        sizing: dict[str, float] = {}
        source_model: str | None = None

        for entry in body_fields:
            kind, key, value = entry
            if kind == "pin":
                if is_bjt and key == "model":
                    if source_model is not None:
                        raise ValueError(
                            f"device {instance_name!r}: model assigned more than once."
                        )
                    source_model = str(value)
                    continue
                short = pin_aliases.get(key)
                if short is None:
                    valid = (
                        "collector/base/emitter (long form) or c/b/e (short form)"
                        if is_bjt
                        else "drain/gate/source/body (long form) or d/g/s/b (short form)"
                    )
                    raise ValueError(
                        f"device {instance_name!r}: unknown pin name "
                        f"{key!r}. Valid: {valid}."
                    )
                if short in pins:
                    raise ValueError(
                        f"device {instance_name!r}: pin {short!r} "
                        f"assigned more than once (long-form and "
                        f"short-form aliases of the same pin collide)."
                    )
                pins[short] = value
            elif kind == "size":
                if key not in size_keys:
                    raise ValueError(
                        f"device {instance_name!r}: unknown size key "
                        f"{key!r}. Valid keys: "
                        f"{sorted(size_keys)}."
                    )
                if key in sizing:
                    raise ValueError(
                        f"device {instance_name!r}: size key {key!r} "
                        f"assigned more than once."
                    )
                if key in ("M", "nf"):
                    # Integer-valued multipliers. Accept float that is
                    # actually whole (e.g. `M = 4` parses as 4.0).
                    if not float(value).is_integer():
                        raise ValueError(
                            f"device {instance_name!r}: {key} must be "
                            f"an integer, got {value!r}."
                        )
                    sizing[key] = float(int(value))
                else:
                    sizing[key] = float(value)
            else:
                # Defensive — grammar should make this unreachable.
                raise ValueError(
                    f"device {instance_name!r}: unknown body-field "
                    f"kind {kind!r}."
                )

        required_pins = {"c", "b", "e"} if is_bjt else {"d", "g", "s", "b"}
        missing = required_pins - pins.keys()
        if missing:
            family = "BJT" if is_bjt else "MOS"
            terminals = (
                "collector/base/emitter (or c/b/e)"
                if is_bjt
                else "drain/gate/source/body (or d/g/s/b)"
            )
            raise ValueError(
                f"device {instance_name!r}: missing required pin "
                f"assignment(s): {sorted(missing)}. {family} devices need "
                f"{terminals}."
            )
        if is_bjt:
            sizing.setdefault("area", 1.0)
        else:
            # W/L are physical MOS obligations. BJT cells use adapter-selected
            # emitter geometry plus the dimensionless area/multiplier contract.
            if "W" not in sizing:
                raise ValueError(
                    f"device {instance_name!r}: missing required size "
                    f"field 'W'."
                )
            if "L" not in sizing:
                raise ValueError(
                    f"device {instance_name!r}: missing required size "
                    f"field 'L'."
                )
        sizing.setdefault("M", 1.0)

        return Device(
            instance_name=instance_name,
            device_class=device_class,
            pdk_variant=None,        # driver picks from PDK bindings
            pins=pins,
            sizing=sizing,
            source_model=source_model,
        )

    def cap_decl(self, items) -> Device:
        """`cap C1 { p = out; n = vss; value = 1.5pF; }`

        Builds a `Device` with `device_class='cap'`. Pin names are
        always `p` / `n` (no long-form aliases). The size field is the
        single `value` key (Farads after unit resolution).
        """
        return self._passive_decl(items, device_class="cap")

    def res_decl(self, items) -> Device:
        """`res R1 { p = out; n = in; value = 1kOhm; }`

        Builds a `Device` with `device_class='res'`. Pin names are
        always `p` / `n`; the size field is `value` (Ohms after unit
        resolution).
        """
        return self._passive_decl(items, device_class="res")

    def ind_decl(self, items) -> Device:
        """`ind L_tank { p = vco_out; n = vss; value = 1nH; }`

        Builds a `Device` with `device_class='ind'`. Pin names are
        always `p` / `n`; the size field is `value` (Henries after
        unit resolution — `pH`, `nH`, `uH`, `mH` all parse via the
        standard `number` rule).
        """
        return self._passive_decl(items, device_class="ind")

    def varactor_decl(self, items) -> Device:
        """`varactor Cvar { p = vctrl; n = tank_node; W = 10u; L = 0.5u; }`

        Builds a `Device` with `device_class='varactor'`. Pin names
        are `p` (gate / control voltage) and `n` (body / drain-source
        tied net = the actual cap terminal as seen from the circuit).
        Sized like a MOSFET (W, L, optional M) because the SPICE emit
        pattern is a MOSCAP (PMOS with d=s=n, g=p). Effective cap is
        ~Cox*W*L when |Vgs|>Vth_p (on), ~0 when |Vgs|<Vth_p (off).
        """
        instance_name = str(items[1])
        body_fields = items[2:]
        pins: dict[str, str] = {}
        sizing: dict[str, float] = {}
        for entry in body_fields:
            kind, key, value = entry
            if kind == "pin":
                if key not in _PASSIVE_PIN_NAMES:
                    raise ValueError(
                        f"varactor {instance_name!r}: unknown pin name "
                        f"{key!r}. Valid: 'p' / 'n'."
                    )
                if key in pins:
                    raise ValueError(
                        f"varactor {instance_name!r}: pin {key!r} "
                        f"assigned more than once."
                    )
                pins[key] = value
            elif kind == "size":
                if key not in {"W", "L", "M"}:
                    raise ValueError(
                        f"varactor {instance_name!r}: unknown size key "
                        f"{key!r}. Valid: 'W', 'L', 'M'."
                    )
                if key in sizing:
                    raise ValueError(
                        f"varactor {instance_name!r}: size key {key!r} "
                        f"assigned more than once."
                    )
                sizing[key] = float(value)
            else:
                raise ValueError(
                    f"varactor {instance_name!r}: unknown body-field "
                    f"kind {kind!r}."
                )
        missing_pins = _PASSIVE_PIN_NAMES - pins.keys()
        if missing_pins:
            raise ValueError(
                f"varactor {instance_name!r}: missing required pin "
                f"assignment(s): {sorted(missing_pins)}."
            )
        if "W" not in sizing or "L" not in sizing:
            raise ValueError(
                f"varactor {instance_name!r}: missing required size "
                f"field(s): need both 'W' and 'L'."
            )
        sizing.setdefault("M", 1.0)
        return Device(
            instance_name=instance_name,
            device_class="varactor",
            pdk_variant=None,
            pins=pins,
            sizing=sizing,
        )

    def _passive_decl(self, items, *, device_class: str) -> Device:
        """Shared parser for cap_decl / res_decl bodies.

        items layout: [CAP_KW or RES_KW token, CNAME(instance_name),
        device_body_field, device_body_field, ...]
        """
        instance_name = str(items[1])
        body_fields = items[2:]
        pins: dict[str, str] = {}
        sizing: dict[str, float] = {}
        source_model: str | None = None
        for entry in body_fields:
            kind, key, value = entry
            if kind == "pin":
                if key == "model":
                    if source_model is not None:
                        raise ValueError(
                            f"{device_class} {instance_name!r}: model assigned "
                            "more than once."
                        )
                    source_model = str(value)
                    continue
                if key not in _PASSIVE_PIN_NAMES:
                    raise ValueError(
                        f"{device_class} {instance_name!r}: unknown pin "
                        f"name {key!r}. Valid: 'p' / 'n'."
                    )
                if key in pins:
                    raise ValueError(
                        f"{device_class} {instance_name!r}: pin {key!r} "
                        f"assigned more than once."
                    )
                pins[key] = value
            elif kind == "size":
                if key not in _PASSIVE_SIZE_KEYS:
                    raise ValueError(
                        f"{device_class} {instance_name!r}: unknown "
                        f"size key {key!r}. Valid: "
                        f"{sorted(_PASSIVE_SIZE_KEYS)}."
                    )
                if key in sizing:
                    raise ValueError(
                        f"{device_class} {instance_name!r}: size key "
                        f"{key!r} assigned more than once."
                    )
                sizing[key] = float(value)
            else:
                raise ValueError(
                    f"{device_class} {instance_name!r}: unknown body-"
                    f"field kind {kind!r}."
                )
        missing_pins = _PASSIVE_PIN_NAMES - pins.keys()
        if missing_pins:
            raise ValueError(
                f"{device_class} {instance_name!r}: missing required "
                f"pin assignment(s): {sorted(missing_pins)}."
            )
        if "value" not in sizing:
            raise ValueError(
                f"{device_class} {instance_name!r}: missing required "
                f"size field 'value'."
            )
        return Device(
            instance_name=instance_name,
            device_class=device_class,
            pdk_variant=None,
            pins=pins,
            sizing=sizing,
            source_model=source_model,
        )

    def device_pin_assignment(self, items) -> tuple[str, str, str]:
        """A single `<pin_name> = <net_name>;` line.

        Returns a tagged triple `("pin", pin_name_raw, net_name)`. The
        caller (`device_decl` / `_passive_decl`) canonicalizes the pin
        name to its short form and validates the net name is a legal
        CNAME (already enforced by the grammar).
        """
        key = str(items[0])
        net = str(items[1])
        return ("pin", key, net)

    def device_size_field(self, items) -> tuple[str, str, float]:
        """A single `<key> = <number>;` line inside a device body.

        Returns a tagged triple `("size", key, value)`. The caller
        validates the key against the allowed-keys set and checks
        integer typing for M / nf.
        """
        key = str(items[0])
        value = items[1]   # already resolved to SI float by `number` rule
        return ("size", key, value)

    def device_native_parameter(self, items) -> tuple[str, str, str]:
        return ("native_param", str(items[0]), str(items[1]))
