"""
Annotation-building factories (extracted from `_grammar.py`).

This module holds the self-contained, transformer-independent helpers that
turn parsed `@<kind>(<args>)` annotation surface into concrete
`clay.layout` annotation dataclass instances. Splitting them out of
`_grammar.py` keeps that module under the per-file LOC cap while preserving
behavior exactly — the lark transformer methods in `_grammar.py` import
`_build_layout_annotation` (and the `_KindLiteral` / `_KwArg` arg wrappers)
back from here.

These functions take only already-parsed argument values (bare str,
`_KindLiteral`, `_KwArg`, or float) and return annotation dataclasses; they
hold no reference to the lark transformer instance state, so the move is a
pure refactor with no parsing-behavior change.

IMPORTANT: this module must NOT import `_grammar` (that would create an
import cycle — `_grammar` imports from here).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re

from ..layout import (
    Adjacent,
    Bulk,
    EqualSize,
    Fix,
    GuardRing,
    InterdigitationPattern,
    Interdigitated,
    LayoutAnnotation,
    DenseContacts,
    NativeBodyTap,
    Fold,
    Matching,
    MatchingKind,
    Orient,
    Orientation,
    ResistorSegments,
    Symmetric,
    SymmetryAxis,
    Tap,
    WellIsolated,
    WellType,
)


class _KindLiteral(str):
    """Wrapper marking a value as a KIND_LITERAL terminal token.

    Subclasses `str` so the value compares equal to the same bare string
    (e.g. `_KindLiteral("nwell") == "nwell"` is True). The per-kind
    factory uses `isinstance(arg, _KindLiteral)` to distinguish a
    well-type / pattern / axis literal from a bare identifier — important
    for slots like `@well_isolated(<device_id>)` where a literal is NOT
    a valid substitute for a device-id.
    """
    __slots__ = ()


@dataclass
class _KwArg:
    """A parsed `key=value` keyword argument inside an annotation block.

    Carries the keyword name (`key`, e.g. `where`) and the resolved value
    (a bare str, a `_KindLiteral`, or a float — whatever the value
    alternative produced). The per-kind factory consumes these for kinds
    that accept named slots (currently `@tap(net, where=<loc>)`); kinds
    with only positional args reject any `_KwArg` they receive.
    """
    key: str
    value: object


# ─── annotation_kind → LayoutAnnotation factory ──────────────────────

# Closed set of annotation kinds. The grammar's `annotation_kind` rule
# already restricts the surface to these six keywords; we re-check here
# as a defensive guard against future grammar tweaks that add a spelling
# without updating the factory.
_VALID_ANNOTATION_KINDS: frozenset[str] = frozenset({
    "matching", "guard_ring", "well_isolated",
    "symmetric", "interdigitated", "adjacent",
    "bulk", "tap", "orient", "dense_contacts", "native_body_tap", "fold", "resistor_segments",
    "equal_size", "fix",
})


def _build_layout_annotation(*, kind: str, args: list) -> LayoutAnnotation:
    """Construct the right `LayoutAnnotation` subclass for `(kind, args)`.

    Each subclass has its own arg-arity contract (see
    `clay/layout/annotations.py`):
      * matching       — (other_device_id, [matching_kind_literal])
      * guard_ring     — (well_type_literal)
      * well_isolated  — (from_device_id)
      * symmetric      — (device_id+, [axis_literal])   (>=1 device)
      * interdigitated — (with_device_id,
                          [pattern_literal,
                          [finger_pairs_int]])
      * adjacent       — (to_device_id, [separation_multiplier_number])
      * bulk           — (body_net_id)
      * tap            — (tap_net_id, where=<location_id>)
      * equal_size     — (device_id, device_id+)   (>=2 devices)
      * fix            — (device_id, x=<um>, y=<um>)   (1 device + x/y kwargs)
    """
    if kind not in _VALID_ANNOTATION_KINDS:
        # Defensive — the grammar already restricts to the six kinds.
        raise ValueError(
            f"annotation_block: unknown kind {kind!r}. Valid: "
            f"{sorted(_VALID_ANNOTATION_KINDS)}."
        )
    if kind == "matching":
        return _build_matching(args)
    if kind == "guard_ring":
        return _build_guard_ring(args)
    if kind == "dense_contacts":
        return _build_dense_contacts(args)
    if kind == "native_body_tap":
        if len(args) != 1 or type(args[0]) is not str or args[0] != "single":
            raise ValueError("@native_body_tap: expected single")
        return NativeBodyTap(mode="single")
    if kind == "fold":
        return _build_fold(args)
    if kind == "resistor_segments":
        return _build_resistor_segments(args)
    if kind == "well_isolated":
        return _build_well_isolated(args)
    if kind == "symmetric":
        return _build_symmetric(args)
    if kind == "interdigitated":
        return _build_interdigitated(args)
    if kind == "adjacent":
        return _build_adjacent(args)
    if kind == "bulk":
        return _build_bulk(args)
    if kind == "tap":
        return _build_tap(args)
    if kind == "orient":
        return _build_orient(args)
    if kind == "equal_size":
        return _build_equal_size(args)
    if kind == "fix":
        return _build_fix(args)
    # Unreachable — _VALID_ANNOTATION_KINDS check above.
    raise ValueError(f"annotation_block: unhandled kind {kind!r}.")


_MATCHING_KIND_VALUES = frozenset({m.value for m in MatchingKind})
_WELL_TYPE_VALUES = frozenset({w.value for w in WellType})
_SYMMETRY_AXIS_VALUES = frozenset({a.value for a in SymmetryAxis})
_INTERDIGITATION_PATTERN_VALUES = frozenset(
    {p.value for p in InterdigitationPattern}
)
_ORIENTATION_VALUES = frozenset({o.value for o in Orientation})


def _build_matching(args: list) -> Matching:
    """`@matching(other_id [, kind])`. 1 or 2 args."""
    if not (1 <= len(args) <= 2):
        raise ValueError(
            f"@matching: expected 1 or 2 args (other_id [, kind]), "
            f"got {len(args)}: {args!r}"
        )
    other = args[0]
    if isinstance(other, _KindLiteral) or not isinstance(other, str):
        raise ValueError(
            f"@matching: first arg must be a device-id (bare "
            f"identifier), got {type(other).__name__}: {other!r}"
        )
    if len(args) == 2:
        kind_val = args[1]
        if not isinstance(kind_val, _KindLiteral) \
                or str(kind_val) not in _MATCHING_KIND_VALUES:
            raise ValueError(
                f"@matching: second arg must be a kind literal "
                f"(one of {sorted(_MATCHING_KIND_VALUES)}), got "
                f"{kind_val!r}"
            )
        return Matching(other=other, kind=MatchingKind(str(kind_val)))
    return Matching(other=other)


def _build_guard_ring(args: list) -> GuardRing:
    """`@guard_ring(well_type)`. Exactly 1 arg."""
    if len(args) != 1:
        raise ValueError(
            f"@guard_ring: expected 1 arg (well_type), got "
            f"{len(args)}: {args!r}"
        )
    well_val = args[0]
    if not isinstance(well_val, _KindLiteral) \
            or str(well_val) not in _WELL_TYPE_VALUES:
        raise ValueError(
            f"@guard_ring: arg must be a well-type literal (one of "
            f"{sorted(_WELL_TYPE_VALUES)}), got {well_val!r}"
        )
    return GuardRing(well_type=WellType(str(well_val)))


def _build_well_isolated(args: list) -> WellIsolated:
    """`@well_isolated(from_device_id)`. Exactly 1 arg."""
    if len(args) != 1:
        raise ValueError(
            f"@well_isolated: expected 1 arg (from_device), got "
            f"{len(args)}: {args!r}"
        )
    fd = args[0]
    if isinstance(fd, _KindLiteral):
        raise ValueError(
            f"@well_isolated: arg must be a device-id (bare identifier), "
            f"got KIND_LITERAL {fd!r}. Use a device-id like "
            f"`@well_isolated(M3)` instead."
        )
    if not isinstance(fd, str):
        raise ValueError(
            f"@well_isolated: arg must be a device-id, got "
            f"{type(fd).__name__}: {fd!r}"
        )
    return WellIsolated(from_device=fd)


def _build_symmetric(args: list) -> Symmetric:
    """`@symmetric(dev1 [, dev2 [, ...] ] [, axis])`.

    Accepts at least 1 arg. The LAST arg may be a SymmetryAxis literal
    (`vertical` / `horizontal`); all preceding args are device ids
    (CNAME). A bare `@symmetric(vertical)` is valid (axis only, no
    devices) per the round plan acceptance: `@symmetric(vertical)`.
    """
    if not args:
        raise ValueError(
            "@symmetric: expected at least 1 arg (device-id+ or axis), "
            "got 0"
        )
    # Detect trailing axis literal (must be _KindLiteral).
    axis: SymmetryAxis = SymmetryAxis.VERTICAL
    devices_args = list(args)
    last = args[-1]
    if isinstance(last, _KindLiteral) and str(last) in _SYMMETRY_AXIS_VALUES:
        axis = SymmetryAxis(str(last))
        devices_args = devices_args[:-1]
    # Remaining args must all be device-id idents (not literals).
    devices: list[str] = []
    for d in devices_args:
        if isinstance(d, _KindLiteral):
            if str(d) in _SYMMETRY_AXIS_VALUES:
                raise ValueError(
                    f"@symmetric: axis literal {d!r} must appear LAST. "
                    f"Got it before device-id."
                )
            raise ValueError(
                f"@symmetric: unexpected literal {d!r}; expected a "
                f"device-id identifier."
            )
        if not isinstance(d, str):
            raise ValueError(
                f"@symmetric: device-id args must be identifiers, "
                f"got {type(d).__name__}: {d!r}"
            )
        devices.append(d)
    return Symmetric(devices=devices, axis=axis)


def _build_equal_size(args: list) -> EqualSize:
    """`@equal_size(d1, d2, ...)`. At least 2 device-id idents.

    Every arg is a bare device/passive LABEL (CNAME) — no kind literals, no
    keyword args. The members are constrained to an IDENTICAL placed footprint
    (same width AND height) by the placer; a `check_equal_size` obligation
    verifies it on the real placement. Fewer than 2 labels is rejected: an
    equal-size group of one device is vacuous (a device always equals itself).
    """
    if len(args) < 2:
        raise ValueError(
            f"@equal_size: expected at least 2 device-ids "
            f"(@equal_size(d1, d2, ...)), got {len(args)}: {args!r}"
        )
    devices: list[str] = []
    for d in args:
        if isinstance(d, _KwArg):
            raise ValueError(
                f"@equal_size: takes positional device-ids only, not a keyword "
                f"arg ({d.key}=...)."
            )
        if isinstance(d, _KindLiteral):
            raise ValueError(
                f"@equal_size: every arg must be a device-id (bare identifier), "
                f"got KIND_LITERAL {d!r}."
            )
        if not isinstance(d, str):
            raise ValueError(
                f"@equal_size: device-id args must be identifiers, got "
                f"{type(d).__name__}: {d!r}"
            )
        devices.append(d)
    return EqualSize(devices=devices)


def _build_fix(args: list) -> Fix:
    """`@fix(device, x=<um>, y=<um>)`. One positional device-id + x/y kwargs.

    The first arg is the pinned device LABEL (a bare CNAME). The `x=` / `y=`
    keyword args carry the device CENTROID absolute position in microns (parsed as
    floats by the `annotation_arg_kwarg` value alternative). Both `x` and `y` are
    REQUIRED — an absolute pin needs both coordinates; a missing one is rejected
    (never defaulted to 0, which would silently mis-pin). The device must be a bare
    identifier (a KIND_LITERAL / number is not a valid device-id), mirroring
    `@equal_size`'s positional-device validation.
    """
    positionals: list = []
    coords: dict = {}
    for a in args:
        if isinstance(a, _KwArg):
            if a.key not in ("x", "y"):
                raise ValueError(
                    f"@fix: unknown keyword arg {a.key!r}; the only keyword slots "
                    f"are `x` and `y` (the device centroid in microns)."
                )
            if a.key in coords:
                raise ValueError(f"@fix: `{a.key}` given more than once.")
            val = a.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"@fix: `{a.key}` must be a number (microns), got "
                    f"{type(val).__name__}: {val!r}."
                )
            coords[a.key] = float(val)
        else:
            positionals.append(a)
    if len(positionals) != 1:
        raise ValueError(
            f"@fix: expected exactly 1 positional device-id "
            f"(@fix(device, x=<um>, y=<um>)), got {len(positionals)}: "
            f"{positionals!r}"
        )
    device = positionals[0]
    if isinstance(device, _KindLiteral):
        raise ValueError(
            f"@fix: device arg must be a device-id (bare identifier), got "
            f"KIND_LITERAL {device!r}."
        )
    if not isinstance(device, str):
        raise ValueError(
            f"@fix: device arg must be a device-id, got "
            f"{type(device).__name__}: {device!r}"
        )
    if "x" not in coords or "y" not in coords:
        missing = sorted({"x", "y"} - set(coords))
        raise ValueError(
            f"@fix: both `x` and `y` are required (the device centroid in "
            f"microns); missing {missing}."
        )
    return Fix(device=device, x=coords["x"], y=coords["y"])


def _build_interdigitated(args: list) -> Interdigitated:
    """`@interdigitated(with_device [, pattern [, finger_pairs]])`.

    1 to 3 args. `pattern` ∈ {abab, abba}; defaults to abab.
    `finger_pairs` is a positive integer; defaults to 1.

    A bare `@interdigitated(abab)` is accepted (one positional arg whose
    value is a pattern literal) so the round plan's
    `@interdigitated(abab)` example parses. In that case the
    `with_device` arg is treated as a sentinel ("__pattern_only__") —
    layout-side consumers must supply the device-id from context. The
    common cohort use case ALWAYS pairs a device-id with the pattern
    (`@interdigitated(M2, abab)`), so this fallback is conservative.
    """
    if not (1 <= len(args) <= 3):
        raise ValueError(
            f"@interdigitated: expected 1-3 args "
            f"(with_device [, pattern [, finger_pairs]]), got "
            f"{len(args)}: {args!r}"
        )
    pattern: InterdigitationPattern = InterdigitationPattern.ABAB
    finger_pairs: int = 1
    rest = list(args)
    # Single-arg pattern-only form: `@interdigitated(abab)`.
    if len(rest) == 1 and isinstance(rest[0], _KindLiteral) \
            and str(rest[0]) in _INTERDIGITATION_PATTERN_VALUES:
        pattern = InterdigitationPattern(str(rest[0]))
        return Interdigitated(
            with_device="__pattern_only__",
            pattern=pattern,
            finger_pairs=finger_pairs,
        )
    # Multi-arg form: first is the with_device ident.
    with_device = rest[0]
    if isinstance(with_device, _KindLiteral):
        raise ValueError(
            f"@interdigitated: first arg must be a device-id "
            f"(bare identifier, not a kind literal). Got "
            f"{with_device!r} — write `@interdigitated(<device_id>, "
            f"{with_device})` if you intended this as a pattern."
        )
    if not isinstance(with_device, str):
        raise ValueError(
            f"@interdigitated: first arg (with_device) must be a "
            f"device-id, got {type(with_device).__name__}: {with_device!r}"
        )
    if len(rest) >= 2:
        p = rest[1]
        if not isinstance(p, _KindLiteral) \
                or str(p) not in _INTERDIGITATION_PATTERN_VALUES:
            raise ValueError(
                f"@interdigitated: pattern arg must be a kind literal "
                f"(one of {sorted(_INTERDIGITATION_PATTERN_VALUES)}), "
                f"got {p!r}"
            )
        pattern = InterdigitationPattern(str(p))
    if len(rest) == 3:
        fp_val = rest[2]
        if isinstance(fp_val, bool):
            raise ValueError(
                f"@interdigitated: finger_pairs must be a positive int, "
                f"got bool {fp_val!r}"
            )
        if isinstance(fp_val, (int, float)):
            if not float(fp_val).is_integer() or fp_val <= 0:
                raise ValueError(
                    f"@interdigitated: finger_pairs must be a positive "
                    f"integer, got {fp_val!r}"
                )
            finger_pairs = int(fp_val)
        else:
            raise ValueError(
                f"@interdigitated: finger_pairs must be a positive "
                f"integer, got {type(fp_val).__name__}: {fp_val!r}"
            )
    return Interdigitated(
        with_device=with_device,
        pattern=pattern,
        finger_pairs=finger_pairs,
    )


def _build_orient(args: list) -> Orient:
    """``@orient(R*|MR*)``. Exactly 1 orthogonal-transform literal arg (a
    KIND_LITERAL, not a device-id -- like `@guard_ring(nwell)`)."""
    if len(args) != 1:
        raise ValueError(
            f"@orient: expected 1 orientation arg (one of "
            f"{sorted(_ORIENTATION_VALUES)}), got {len(args)}: {args!r}"
        )
    val = args[0]
    if not isinstance(val, _KindLiteral) or str(val) not in _ORIENTATION_VALUES:
        raise ValueError(
            f"@orient: arg must be an orientation literal (one of "
            f"{sorted(_ORIENTATION_VALUES)}), got {val!r}"
        )
    return Orient(orientation=Orientation(str(val)))


def _build_dense_contacts(args: list) -> DenseContacts:
    """`@dense_contacts`. A 0-arg FLAG: the annotated device is stamped with a full
    S/D contact-cut column (reference-parity contact density). Any argument is
    rejected (the flag names no target/kind -- the annotation TARGET already names
    the device)."""
    if args:
        raise ValueError(
            f"@dense_contacts: expected NO args (it is a flag on the annotated "
            f"device), got {len(args)}: {args!r}"
        )
    return DenseContacts()


def _build_fold(args: list) -> Fold:
    """`@fold` or `@fold(<h_max_um>)`. 0 or 1 arg.

    A device-level FLAG declaring the device is FOLDED into bounded-height
    multi-finger fingers (reference compact unit-transistor style; total gate
    width ``W_eff`` conserved, only per-finger height changes). With NO arg the
    module-default maximum finger height applies
    (:data:`clay.layout.extract.fold.DEFAULT_H_MAX_UM`, ~22um); a single POSITIVE
    numeric arg overrides it (``@fold(16)`` -> 16um bound). Any other arg (an
    identifier / literal, or a non-positive number) is rejected -- the flag names
    no device target (the annotation TARGET already names the device)."""
    if len(args) > 1:
        raise ValueError(
            f"@fold: expected 0 or 1 args (an optional h_max_um), got "
            f"{len(args)}: {args!r}"
        )
    if not args:
        return Fold()
    val = args[0]
    if isinstance(val, _KindLiteral) or not isinstance(val, (int, float)):
        raise ValueError(
            f"@fold: the optional arg must be a positive number (max finger "
            f"height in um), got {type(val).__name__}: {val!r}"
        )
    h = float(val)
    if not (h > 0.0):
        raise ValueError(
            f"@fold: h_max_um must be > 0 (got {h!r})"
        )
    return Fold(h_max_um=h)


_RESISTOR_SEGMENT_KEYS = frozenset({
    "pdk",
    "count",
    "connection",
    "width_um",
    "length_um",
    "segment_resistance_ohms",
    "segment_resistance_quantum_ohms",
    "value_quantum_ohms",
})
_REQUIRED_RESISTOR_SEGMENT_KEYS = _RESISTOR_SEGMENT_KEYS - {
    "segment_resistance_quantum_ohms"
}
_RESISTOR_CONNECTIONS = frozenset({"series", "parallel"})
_MAX_RESISTOR_SEGMENTS = 4096


def _build_resistor_segments(args: list) -> ResistorSegments:
    """Build one complete, keyword-only segmented-resistor declaration.

    The declaration mirrors the coordinate-free subset of the external_pdk resistor
    PCell interface.  Topology and geometry slots are required.  The optional
    segment-resistance quantum defaults to zero for legacy declarations whose
    total-value interval already proves equivalence.
    """

    values: dict[str, object] = {}
    for arg in args:
        if type(arg) is not _KwArg:
            raise ValueError(
                "@resistor_segments: every argument must be keyword=value"
            )
        if arg.key not in _RESISTOR_SEGMENT_KEYS:
            raise ValueError(
                f"@resistor_segments: unknown keyword {arg.key!r}; expected "
                f"{sorted(_RESISTOR_SEGMENT_KEYS)}"
            )
        if arg.key in values:
            raise ValueError(
                f"@resistor_segments: duplicate keyword {arg.key!r}"
            )
        values[arg.key] = arg.value
    missing = _REQUIRED_RESISTOR_SEGMENT_KEYS - set(values)
    if missing:
        raise ValueError(
            f"@resistor_segments: missing required keyword(s) {sorted(missing)}"
        )

    pdk = values["pdk"]
    if type(pdk) is not str or re.fullmatch(r"[a-z][a-z0-9_]*", pdk) is None:
        raise ValueError(
            f"@resistor_segments: pdk must be one canonical identifier, got {pdk!r}"
        )
    connection = values["connection"]
    if type(connection) is not str or connection not in _RESISTOR_CONNECTIONS:
        raise ValueError(
            "@resistor_segments: connection must be exactly 'series' or "
            f"'parallel', got {connection!r}"
        )

    count_raw = values["count"]
    if (
        isinstance(count_raw, bool)
        or not isinstance(count_raw, (int, float))
        or not math.isfinite(float(count_raw))
        or not float(count_raw).is_integer()
    ):
        raise ValueError(
            f"@resistor_segments: count must be a positive integer, got {count_raw!r}"
        )
    count = int(count_raw)
    if count <= 0 or count > _MAX_RESISTOR_SEGMENTS:
        raise ValueError(
            f"@resistor_segments: count must be in 1..{_MAX_RESISTOR_SEGMENTS}, "
            f"got {count}"
        )

    resolved: dict[str, float] = {}
    for key in (
        "width_um",
        "length_um",
        "segment_resistance_ohms",
        "value_quantum_ohms",
    ):
        raw = values[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"@resistor_segments: {key} must be a positive finite number, "
                f"got {raw!r}"
            )
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                f"@resistor_segments: {key} must be a positive finite number, "
                f"got {raw!r}"
            )
        resolved[key] = number
    raw_segment_quantum = values.get("segment_resistance_quantum_ohms", 0.0)
    if (
        isinstance(raw_segment_quantum, bool)
        or not isinstance(raw_segment_quantum, (int, float))
        or not math.isfinite(float(raw_segment_quantum))
        or float(raw_segment_quantum) < 0.0
    ):
        raise ValueError(
            "@resistor_segments: segment_resistance_quantum_ohms must be a "
            f"non-negative finite number, got {raw_segment_quantum!r}"
        )

    return ResistorSegments(
        pdk=pdk,
        count=count,
        connection=connection,
        width_um=resolved["width_um"],
        length_um=resolved["length_um"],
        segment_resistance_ohms=resolved["segment_resistance_ohms"],
        value_quantum_ohms=resolved["value_quantum_ohms"],
        segment_resistance_quantum_ohms=float(raw_segment_quantum),
    )


def _build_adjacent(args: list) -> Adjacent:
    """`@adjacent` or `@adjacent(to_device [, separation_multiplier])`.

    0, 1 or 2 args. With 0 args, both `to` and the multiplier default —
    `to = "__unspecified__"`, multiplier = Fraction(2, 1). With 1 arg,
    that's the to-device id. With 2 args, the second is a positive
    rational separation multiplier (passed as a float; stored as
    `Fraction.from_float(value).limit_denominator(1000)` so common
    `1.5` / `2.0` / `2.5` values get clean denominators).
    """
    if len(args) > 2:
        raise ValueError(
            f"@adjacent: expected 0-2 args (to [, max_sep]), got "
            f"{len(args)}: {args!r}"
        )
    if len(args) == 0:
        return Adjacent(to="__unspecified__")
    to = args[0]
    if isinstance(to, _KindLiteral):
        raise ValueError(
            f"@adjacent: first arg (to_device) must be a bare device-id, "
            f"got KIND_LITERAL {to!r}."
        )
    if not isinstance(to, str):
        raise ValueError(
            f"@adjacent: first arg (to_device) must be a device-id, "
            f"got {type(to).__name__}: {to!r}"
        )
    if len(args) == 1:
        return Adjacent(to=to)
    mult_val = args[1]
    if isinstance(mult_val, bool) or not isinstance(mult_val, (int, float)):
        raise ValueError(
            f"@adjacent: max_separation_multiplier must be a positive "
            f"number, got {type(mult_val).__name__}: {mult_val!r}"
        )
    if mult_val <= 0:
        raise ValueError(
            f"@adjacent: max_separation_multiplier must be > 0, got "
            f"{mult_val!r}"
        )
    # Use Fraction.from_float + limit_denominator to keep ratios clean
    # for common surface inputs (`1.5` -> 3/2, `2.0` -> 2/1).
    mult = Fraction(mult_val).limit_denominator(1000)
    return Adjacent(to=to, max_separation_multiplier=mult)


def _build_bulk(args: list) -> Bulk:
    """`@bulk(body_net)`. Exactly 1 positional net-id arg.

    The body net is a bare identifier (CNAME): a net name such as `vss`,
    `vdd`, `bulk`, or an explicit body node. A KIND_LITERAL (well-type /
    axis / pattern) is NOT a valid net-id; a keyword arg is not accepted.
    """
    if len(args) != 1:
        raise ValueError(
            f"@bulk: expected 1 arg (body_net), got {len(args)}: {args!r}"
        )
    net = args[0]
    if isinstance(net, _KwArg):
        raise ValueError(
            f"@bulk: takes a single positional net-id, not a keyword arg "
            f"({net.key}=...)."
        )
    if isinstance(net, _KindLiteral):
        raise ValueError(
            f"@bulk: arg must be a net-id (bare identifier), got "
            f"KIND_LITERAL {net!r}."
        )
    if not isinstance(net, str):
        raise ValueError(
            f"@bulk: arg must be a net-id, got {type(net).__name__}: {net!r}"
        )
    return Bulk(net=net)


def _build_tap(args: list) -> Tap:
    """`@tap(tap_net, where=<location>)`. 1 positional net-id + `where=`.

    The first arg is the tap net (a bare net-id such as `vss` / `vdd`).
    The `where=` keyword names the tap LOCATION intent (a bare identifier:
    `substrate`, `ring`, `abut`, `shared`, ...). `where=` defaults to
    `"substrate"` when omitted so `@tap(vss)` is a legal shorthand for a
    plain substrate tie.
    """
    positionals: list = []
    where: str | None = None
    for a in args:
        if isinstance(a, _KwArg):
            if a.key != "where":
                raise ValueError(
                    f"@tap: unknown keyword arg {a.key!r}; the only "
                    f"keyword slot is `where`."
                )
            if where is not None:
                raise ValueError("@tap: `where` given more than once.")
            wv = a.value
            if isinstance(wv, _KindLiteral) or not isinstance(wv, str):
                # A bare identifier is required for the location; a
                # KIND_LITERAL/number is not a valid location name.
                raise ValueError(
                    f"@tap: `where` must be a bare location identifier, "
                    f"got {wv!r}."
                )
            where = wv
        else:
            positionals.append(a)
    if len(positionals) != 1:
        raise ValueError(
            f"@tap: expected exactly 1 positional net-id (tap_net) "
            f"[+ where=<loc>], got {len(positionals)}: {positionals!r}"
        )
    net = positionals[0]
    if isinstance(net, _KindLiteral):
        raise ValueError(
            f"@tap: tap_net must be a net-id (bare identifier), got "
            f"KIND_LITERAL {net!r}."
        )
    if not isinstance(net, str):
        raise ValueError(
            f"@tap: tap_net must be a net-id, got "
            f"{type(net).__name__}: {net!r}"
        )
    return Tap(net=net, where=where if where is not None else "substrate")
