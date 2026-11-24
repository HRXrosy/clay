"""
Lark grammar bootstrap (v0.1.13 parser split).

Extracted from `clay/parser.py` in v0.1.13 (R90 Agent B). Holds the
cached `lark.Lark` parser instance and the path to `grammar.lark`. The
parser is initialized lazily (first call to `_get_parser`) so importing
`clay.parser` stays cheap.

The grammar file itself (`clay/grammar.lark`) is NOT touched by the
split — only this loader module moves.

v0.1.16 (R92 B BL-126): extended with `_AnnotationMixin` carrying the
lark transformer methods for the `annotate <CellName> { ... }` top-level
block. The mixin is folded into the `_ClayTransformer` MRO in
`_ast_builder.py`. Each annotation_block lowers to an actual
`clay.layout.LayoutAnnotation` subclass instance (Matching / GuardRing /
WellIsolated / Symmetric / Interdigitated / Adjacent) — the parser DOES
build the dataclass directly so the IR carries the canonical type.

Surface form:

    annotate amp {
      M1: @matching(M2);
      M1: @guard_ring(nwell);
      M2: @matching(M1);
      @cell: @symmetric(vertical);
      diff_pair: @adjacent(load_pair);
    }

The annotation map for each cell is stashed on the corresponding
`CellDef` via `__dict__["_layout_annotations"]` (a `dict[str, list[
LayoutAnnotation]]`) so downstream consumers can build an `AnnotatedCell`
without growing the CellDef dataclass shape. The wiring (annotate block →
CellDef) is done in `_ast_builder.py`'s `_ClayTransformer.transform`
override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import lark

from ..ir import (
    ParsedFlowDecl,
    ParsedLayoutIntentDecl,
    ParsedPolicyDecl,
    ParsedQualityDecl,
)
from ..layout import LayoutAnnotation
from ._annotation_factories import (
    _build_layout_annotation,
    _KindLiteral,
    _KwArg,
)


# The grammar lives next to the legacy `clay/parser.py`, so we walk one
# directory up from this module (which lives in `clay/parser_modules/`)
# to find the grammar file. The same parent-of-parent relationship the
# pre-split module-level constant captured.
GRAMMAR_FILE = Path(__file__).resolve().parent.parent / "grammar.lark"


_PARSER: lark.Lark | None = None


def _get_parser() -> lark.Lark:
    global _PARSER
    if _PARSER is None:
        grammar = GRAMMAR_FILE.read_text(encoding="utf-8")
        _PARSER = lark.Lark(
            grammar,
            start="start",
            parser="earley",
            maybe_placeholders=False,
            propagate_positions=True,
        )
    return _PARSER


# ─── v0.1.16 R92 B BL-126: annotate block IR + transformer ─────────────
#
# `AnnotateDecl` is a small structural shell collecting every annotation
# entry inside one `annotate <CellName> { ... }` block. We keep it
# off the public `clay.ir` surface so `clay.ir`'s 50+ dataclasses stay
# byte-exact; the AnnotateDecl flows from the transformer up to
# `_ClayTransformer.transform`, where the entries get folded onto each
# `CellDef.__dict__['_layout_annotations']`.

# Sentinel string used as the dict key for cell-level annotations
# (annotations whose target is `@cell`). Matches the convention the
# downstream `AnnotatedCell` consumer expects — see
# `clay/layout/annotations.py`. Bare device / role names use their
# CNAME directly.
_CELL_ANNOTATION_TARGET = "__cell__"


@dataclass
class AnnotationEntry:
    """One `target: @kind(args);` line inside an `annotate { }` block."""
    target: str                      # device id, role/group id, or "__cell__"
    annotation: LayoutAnnotation     # concrete subclass instance


@dataclass
class AnnotateDecl:
    """Parsed `annotate <CellName> { entry+ }` block.

    Carries the cell's local name and the ordered list of annotation
    entries. The wiring step in `_ClayTransformer.transform` matches
    `cell_name` against the file's `CellDef`s and stashes the
    `{target: [LayoutAnnotation, ...]}` map on each one. Unknown cell
    names trigger a ValueError at the wiring step (not here) so the
    error message can name the available cells in the file.
    """
    cell_name: str
    entries: list[AnnotationEntry] = field(default_factory=list)


@dataclass
class LayoutIntentDecl:
    """Parsed `layout_intent <CellName> { entry+ }` block."""
    cell_name: str
    entries: list[ParsedLayoutIntentDecl] = field(default_factory=list)


class _AnnotationMixin:
    """Lark transformer methods for the `annotate` top-level block.

    Folded into `_ClayTransformer` via the MRO in `_ast_builder.py`.
    Pure-additive — every existing transformer method (cell_def /
    device_decl / role_decl / topology_block / etc.) is byte-exact
    preserved per the R90 B split contract. The annotation surface lives
    OUTSIDE the cell body, in a sibling block, so none of the existing
    body grammar / transformer paths need changes.
    """

    # ─── Terminal-level methods ──────────────────────────────

    def annotate_decl(self, items) -> tuple[str, AnnotateDecl]:
        """`annotate <CellName> { entry+ }` block.

        items[0] is the ANNOTATE_KW terminal (skip).
        items[1] is the cell name (CNAME).
        items[2..] are AnnotationEntry instances from `annotation_entry`.
        """
        cell_name = str(items[1])
        entries: list[AnnotationEntry] = list(items[2:])
        return ("annotate_decl", AnnotateDecl(
            cell_name=cell_name, entries=entries,
        ))

    def annotation_entry(self, items) -> AnnotationEntry:
        """`<target> : <annotation_block> ;`

        items[0] is the resolved target string (from
        `annotation_target_id` or `annotation_target_cell`).
        items[1] is the `LayoutAnnotation` instance from
        `annotation_block`.
        """
        target = items[0]
        annotation = items[1]
        if not isinstance(target, str):
            raise ValueError(
                f"annotation_entry: target must be a string, got "
                f"{type(target).__name__}"
            )
        if not isinstance(annotation, LayoutAnnotation):
            raise ValueError(
                f"annotation_entry: expected LayoutAnnotation, got "
                f"{type(annotation).__name__}"
            )
        return AnnotationEntry(target=target, annotation=annotation)

    def annotation_target_cell(self, items) -> str:
        """`@cell` sentinel — annotation applies to whole cell."""
        return _CELL_ANNOTATION_TARGET

    def annotation_target_id(self, items) -> str:
        """Bare CNAME — device-id or role/group-id."""
        return str(items[0])

    # ─── annotation_block → LayoutAnnotation factory ─────────

    def annotation_block(self, items) -> LayoutAnnotation:
        """`@<kind>` or `@<kind>(<args>)`.

        items[0] is the `annotation_kind` string.
        items[1:] is optional — the args list (a list[str|float]).
        Builds the appropriate `LayoutAnnotation` subclass.
        """
        kind = items[0]
        args: list = items[1] if len(items) > 1 else []
        return _build_layout_annotation(kind=kind, args=args)

    def annotation_kind(self, items) -> str:
        """Returns the bare kind string ('matching' / 'guard_ring' / ...)."""
        # The token text matches the kind keyword's terminal name.
        return str(items[0])

    def annotation_args(self, items) -> list:
        """Returns the ordered list of resolved arg values."""
        return list(items)

    def annotation_arg_literal(self, items):
        """KIND_LITERAL terminal — wrap as `_KindLiteral` for the factory.

        Wrapping distinguishes a literal-typed arg (`nwell`, `vertical`,
        `abab`) from a bare-identifier arg (`M2`, `load_pair`) so the
        per-kind factory can validate context-dependent slot semantics
        (e.g. `@well_isolated(nwell)` must reject because well-type
        literals can't substitute for device-ids).
        """
        return _KindLiteral(str(items[0]))

    def annotation_arg_number(self, items) -> float:
        """SIGNED_NUMBER terminal — return as float."""
        return float(str(items[0]))

    def annotation_arg_ident(self, items) -> str:
        """Bare CNAME — return as plain str (distinct from `_KindLiteral`)."""
        return str(items[0])

    def annotation_arg_kwarg(self, items) -> _KwArg:
        """`key=value` keyword arg — wrap as `_KwArg(key, value)`.

        items[0] is the key CNAME (a raw lark Token); items[1] is the
        already-transformed value (str / `_KindLiteral` / float from the
        `annotation_kwarg_val` alternatives). The factory matches on
        `isinstance(arg, _KwArg)` to route a named slot.
        """
        return _KwArg(key=str(items[0]), value=items[1])


def _layout_intent_head_kwargs(items) -> tuple[str | None, dict[str, object]]:
    """Split a `@name(head, kw=val, ...)` arg list into ``(head, kwargs)``.

    Reused by the Stage-1 craft-intent transformer methods. Each keyword arg
    arrives as a 2-tuple ``(key, value)`` from ``layout_place_arg`` (values are
    already reduced to ``str`` there); the optional leading POSITIONAL argument
    arrives as a bare ``CNAME`` token (a ``str`` subclass, never a tuple).
    Returns the head string (or ``None`` for an all-keyword construct such as
    ``@band``) and a ``{key: value}`` dict. A repeated keyword lets the last
    occurrence win (deterministic).
    """
    head: str | None = None
    kwargs: dict[str, object] = {}
    for it in items:
        if isinstance(it, tuple):
            kwargs[str(it[0])] = it[1]
        else:
            head = str(it)
    return head, kwargs


def _require_consumed_layout_keywords(items, *, declaration, allowed, positional):
    """Reject shadowed, repeated, or unconsumed arguments before dict folding."""
    seen: set[str] = set()
    pairs = tuple(item for item in items if isinstance(item, tuple))
    for key, _value in pairs:
        folded = str(key).casefold()
        if folded == positional:
            raise ValueError(f"@{declaration} {positional} must be specified only by the positional argument")
        if folded in seen:
            raise ValueError(f"@{declaration} duplicate keyword: {folded}")
        seen.add(folded)
    for key, _value in pairs:
        if str(key) not in allowed:
            raise ValueError(f"@{declaration} unknown keyword: {key}")


class _LayoutIntentMixin:
    """Lark transformer methods for net-level `layout_intent` blocks."""

    def layout_intent_decl(self, items) -> tuple[str, LayoutIntentDecl]:
        cell_name = str(items[0])
        entries: list[ParsedLayoutIntentDecl] = list(items[1:])
        return ("layout_intent_decl", LayoutIntentDecl(
            cell_name=cell_name,
            entries=entries,
        ))

    @lark.v_args(meta=True)
    def layout_match_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="match", nets=(str(items[0]), str(items[1])),
            attributes={"scope": str(items[2])} if len(items) == 3 else {},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_symmetric_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="symmetric",
            nets=(str(items[0]), str(items[1])),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_shield_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        shield_net = str(items[2]) if len(items) > 2 else "gnd"
        return ParsedLayoutIntentDecl(
            kind="shield",
            aggressor=str(items[0]),
            victim=str(items[1]),
            shield_net=shield_net,
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_budget_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        attributes = {}
        for key, value in items[4:]:
            if key in attributes:
                raise ValueError(f"@budget repeats driver {key!r}")
            attributes[key] = value
        return ParsedLayoutIntentDecl(
            kind="budget",
            budget_kind=str(items[0]),
            nets=(str(items[1]), str(items[2])),
            rel_tol=float(str(items[3])),
            attributes=attributes,
            source_span=_source_span_from_meta(meta),
        )

    def layout_budget_aggressors(self, items) -> tuple[str, tuple[str, ...]]:
        return ("aggressors", tuple(items[0]))

    def layout_budget_clock_net(self, items) -> tuple[str, str]:
        return ("clock_net", str(items[0]))

    @lark.v_args(meta=True)
    def layout_mirror_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="mirror",
            nets=(str(items[0]), str(items[1])),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_objective_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="objective",
            target=str(items[0]),
            goal=str(items[1]),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_area_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="area",
            limit_kind=str(items[0]),
            limit_value=float(str(items[1])),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_wire_width_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@wire_width(<net>, <width>)` — declared per-net routing width.

        ``items[1]`` is the ``number`` subtree already reduced to an SI length in
        METRES (the shared ``number`` rule applies the ``UNIT`` multiplier, so
        ``2u`` -> ``2e-6``). Carry the width on the decl in NANOMETRES (the router's
        internal unit) under ``attributes['width_nm']``; ``nets`` carries the single
        target net (padded so the (net, net) shape matches the other decls). The
        lowering in ``clay.layout.intent`` reads ``width_nm`` into a
        ``LayoutIntent.wire_widths`` entry.
        """
        net = str(items[0])
        width_m = float(items[1])
        width_nm = int(round(width_m * 1e9))
        return ParsedLayoutIntentDecl(
            kind="wire_width",
            nets=(net, net),
            attributes={"width_nm": width_nm, "net": net},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_net_spacing_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        from clay.layout.net_spacing import parse_spacing_nm

        a, b = str(items[0]), str(items[1])
        return ParsedLayoutIntentDecl(
            kind="net_spacing", nets=(a, b),
            attributes={"minimum_nm": parse_spacing_nm(str(items[2]), str(items[3]))},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_match_length_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@match_length(<netA>, <netB>)` — matched-net wire-LENGTH balance.

        A CHECK-only declaration (mirrors `@wire_width`'s plumbing but adds no
        router behaviour): the two nets' ROUTED metal run lengths must be
        balanced so a differential / matched pair carries balanced wire
        parasitic. Carry the ordered `(netA, netB)` pair both on `nets` and on
        `attributes['nets']`; the lowering in `clay.layout.intent` reads it into
        a `LayoutIntent.match_length_pairs` entry, verified by a `match_length`
        geometric obligation.
        """
        a = str(items[0])
        b = str(items[1])
        return ParsedLayoutIntentDecl(
            kind="match_length",
            nets=(a, b),
            attributes={"nets": (a, b)},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_order_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@order(<A>, <B>, <rel>)` — declared RELATIVE ORDER between two devices.

        ``rel`` is one of {left_of, above}: A is left_of / above B in the realised
        placement. Carry the ordered ``(A, B)`` pair on ``nets`` and the relation on
        ``attributes['rel']``; the lowering in ``clay.layout.intent`` reads it into a
        ``LayoutIntent.order_constraints`` entry, consumed by the seqpair placer as a
        directed constraint-graph edge and verified by a ``check_order`` obligation.
        """
        a = str(items[0])
        b = str(items[1])
        rel = str(items[2])
        return ParsedLayoutIntentDecl(
            kind="order",
            nets=(a, b),
            attributes={"a": a, "b": b, "rel": rel},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_align_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """Align corresponding edges, optionally with a relative nm offset."""

        first = str(items[0])
        second = str(items[1])
        edge = str(items[2])
        attrs = {"first": first, "second": second, "edge": edge}
        if len(items) > 3:
            attrs["offset_nm"] = str(items[3])
        return ParsedLayoutIntentDecl(
            kind="align",
            nets=(first, second),
            attributes=attrs,
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_center_group_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        target, groups = str(items[0]), tuple(items[1])
        return ParsedLayoutIntentDecl(
            kind="center_group", nets=(target,),
            attributes={"target": target, "groups": groups},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_group_align_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        first, second, edge = map(str, items)
        return ParsedLayoutIntentDecl(kind="group_align", nets=(first, second),
            attributes={"first": first, "second": second, "edge": edge},
            source_span=_source_span_from_meta(meta))

    @lark.v_args(meta=True)
    def layout_group_separate_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        before, after, axis = map(str, items[:3])
        attrs = {"before": before, "after": after, "axis": axis}
        if len(items) == 4:
            attrs["gap_nm"] = str(items[3])
        return ParsedLayoutIntentDecl(kind="group_separate", nets=(before, after),
            attributes=attrs,
            source_span=_source_span_from_meta(meta))

    @lark.v_args(meta=True)
    def layout_channel_align_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        first, second = map(str, items)
        return ParsedLayoutIntentDecl(kind="channel_align", nets=(first, second),
            attributes={"first": first, "second": second},
            source_span=_source_span_from_meta(meta))

    @lark.v_args(meta=True)
    def layout_terminal_access_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """``@terminal_access(D, terminal=g, side=top)`` is device-local."""

        device = str(items[0])
        terminal = str(items[1])
        side = str(items[2])
        return ParsedLayoutIntentDecl(
            kind="terminal_access",
            nets=(device,),
            attributes={
                "device": device,
                "terminal": terminal,
                "side": side,
            },
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_abut_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@abut(<A>, <B>)` — declared ADJACENCY between two devices.

        A and B are placed edge-adjacent (the facing inter-device gap is the
        DRC-minimum, not the default routing channel). Carry the ``(A, B)`` pair on
        ``nets`` and on ``attributes['pair']``; the lowering in ``clay.layout.intent``
        reads it into a ``LayoutIntent.abut_pairs`` entry, consumed by the seqpair
        placer (tight spacing + adjacency preference) and verified by a ``check_abut``
        obligation.
        """
        a = str(items[0])
        b = str(items[1])
        return ParsedLayoutIntentDecl(
            kind="abut",
            nets=(a, b),
            attributes={"pair": (a, b)},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_guard_ring_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@guard_ring(group=(M1, M2, ...), well=<p|n>, tie=<net>, hug=tight,
        role=tap|isolation)` -- net/cluster level GUARD RING.

        Wrap the named DEVICE cluster (``group``) in ONE PDK-rule-driven tap ring.
        The FIRST two arguments are fixed-position (``group=`` the member id-list at
        ``items[0]``, ``well=`` the ring polarity CNAME at ``items[1]``); EVERY
        argument after ``well`` arrives as a generic ``(key, value)`` pair from
        ``layout_place_arg`` (the same uniform kwarg surface the craft-intent rules
        use), so the vocabulary extends without another grammar shape change:

          * ``tie``     -- the supply/body net the ring is strapped to (optional);
          * ``hug``     -- ``tight`` = hug the cluster at PDK-min tap spacing
                           (vs ``enclose=<spacing>``);
          * ``role``    -- ``tap`` (MOS well/substrate tap ring) or ``isolation``
                           (a small passive resistor/cap ring).

        Carry the member device labels on ``nets`` (so the net-binding checker
        validates them as DEVICE references) and ``well`` / ``tie`` / ``hug`` /
        ``role`` / ``enclose`` on ``attributes``; the lowering in
        ``clay.layout.intent`` reads ``members`` / ``well`` / ``tie`` into a
        ``LayoutIntent.guard_rings`` entry (``hug`` / ``role`` / ``enclose`` are
        carried as declared attributes for the materializer that consumes them),
        materialized by ``route_cell`` (``group_guard_rings``) and verified by the
        ``native.guard_ring_materialized`` checker. Empty -> route_cell unchanged
        (byte-parity). Device labels + nets are DATA (no engine branch)."""
        members: tuple[str, ...] = tuple(str(m) for m in items[0])  # id-list
        well = str(items[1])                                        # well CNAME
        kw: dict[str, str] = {}
        for it in items[2:]:  # each is a (key, value) from layout_place_arg
            kw[str(it[0])] = str(it[1])
        return ParsedLayoutIntentDecl(
            kind="guard_ring",
            nets=members,
            attributes={
                "members": members,
                "well": well,
                "tie": kw.get("tie", ""),
                "hug": kw.get("hug", ""),
                "role": kw.get("role", ""),
                "enclose": kw.get("enclose", ""),
                "pack": kw.get("pack", ""),
            },
            source_span=_source_span_from_meta(meta),
        )

    # ── Stage-1 first-class analog-layout CRAFT intent (parse + lower only) ──
    # Each method mirrors the defensive shape of `layout_order_intent` /
    # `layout_guard_ring_intent`: it splits the `@name(head, kw=val, ...)` args
    # into a positional head + a keyword dict and carries them on `attributes`
    # (NOT on `nets`, so the intent net-binding checker leaves these coordinate-
    # free organizational kinds alone at this parse-only stage). The lowering in
    # `clay.layout.intent` reads these attributes into empty-by-default list
    # fields. Names / styles / layers are DATA (no engine branch).

    @lark.v_args(meta=True)
    def layout_array_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@array(<group>, style=, rows=, order=)` -- shared-diffusion / discrete row array."""
        occurrence_args = [item for item in items
                           if isinstance(item, tuple) and str(item[0]) == "occurrences"]
        if len(occurrence_args) > 1:
            raise ValueError("array occurrences cannot be declared more than once")
        if occurrence_args and type(occurrence_args[0][1]) is not tuple:
            raise ValueError("array occurrences must be a parenthesized owner list")
        for key in ("source_contact_coverage", "drain_contact_coverage"):
            if sum(isinstance(item, tuple) and str(item[0]) == key for item in items) > 1:
                raise ValueError(f"array {key} cannot be declared more than once")
        if sum(isinstance(item, tuple) and str(item[0]) == "access_profile" for item in items) > 1:
            raise ValueError("array access_profile cannot be declared more than once")
        if sum(isinstance(item, tuple) and str(item[0]) == "generation_profile" for item in items) > 1:
            raise ValueError("array generation_profile cannot be declared more than once")
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="array",
            attributes={"group": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_common_centroid_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@common_centroid(<group>, rows=, orientation=)` -- rigid CC macro."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="common_centroid",
            attributes={"group": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_dummy_flank_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@dummy_flank(<target>, per_end=, tie=, share_diffusion=)` -- generative end-caps."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="dummy_flank",
            attributes={"target": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_well_island_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@well_island(<group>, body=)` -- force a group into its own well tied to `body`."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="well_island",
            attributes={"group": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_well_island_share_intent(
        self, meta, items
    ) -> ParsedLayoutIntentDecl:
        """Lower the closed shared-well declaration without coordinates."""

        return ParsedLayoutIntentDecl(
            kind="well_island_share",
            attributes={"group": str(items[0]), "body": str(items[1])},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_band_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@band(top=, bottom=, center=)` -- P-top / N-bottom / passive-center floorplan."""
        _head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="band",
            attributes={**kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_rail_affinity_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@rail_affinity(<group>, side=top|bottom)` -- pin a group to a supply band."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="rail_affinity",
            attributes={"group": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_trunk_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@trunk(<net>, layer=, orient=)` -- horizontal/vertical signal-trunk layer assignment."""
        _require_consumed_layout_keywords(items, declaration="trunk", positional="net",
                                          allowed={"layer", "orient", "width", "edge", "offset"})
        head, kw = _layout_intent_head_kwargs(items)
        if "width" in kw:
            try:
                width = float(kw["width"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("@trunk width must be a finite number") from exc
            if not math.isfinite(width) or width < 0:
                raise ValueError("@trunk width must be a finite nonnegative number")
        if "offset" in kw:
            try:
                offset = float(kw["offset"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("@trunk offset must be a finite number") from exc
            if not math.isfinite(offset):
                raise ValueError("@trunk offset must be a finite number")
        return ParsedLayoutIntentDecl(
            kind="trunk",
            attributes={"net": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_power_plane_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@power_plane(<net>, layer=M5|M6)` -- upper-metal power plane assignment."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="power_plane",
            attributes={"net": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_split_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@split(<passive>, n=, match=series|parallel)` -- matched split of Rz/Rs/Cc."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="split",
            attributes={"passive": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_passive_bank_partition_intent(
        self, meta, items
    ) -> ParsedLayoutIntentDecl:
        """Lower a closed, coordinate-free passive sub-bank declaration."""

        parent = str(items[0])
        bank = str(items[1])
        members = tuple(sorted(str(member) for member in items[2]))
        return ParsedLayoutIntentDecl(
            kind="passive_bank_partition",
            attributes={"parent": parent, "bank": bank, "members": members},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_row_bounded_cavity_intent(
        self, meta, items
    ) -> ParsedLayoutIntentDecl:
        """Lower one closed cross-owner generated-cavity declaration."""

        return ParsedLayoutIntentDecl(
            kind="row_bounded_cavity",
            attributes={
                "host": str(items[0]),
                "guests": tuple(str(item) for item in items[1]),
            },
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_anchor_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@anchor(<net|group>, edge=left|right|top|bottom)` -- edge-access placement.

        Pin the cluster carrying a net (or a named group) to a cell EDGE (the
        routing-owed twin of the diff-pair LEFT column). The head (``target``) is a
        net / group / device name carried on ``attributes`` (NOT on ``nets`` -- like
        the craft intents, so the net-binding checker leaves this coordinate-free
        placement kind alone); ``edge`` rides in the kwargs. Lowered in
        ``clay.layout.intent`` to a ``LayoutIntent.anchors`` entry, consumed by
        ``place_blocks_2d`` as a SOFT edge pull. Empty -> placer unchanged."""
        head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="anchor",
            attributes={"target": head, **kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_channel_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """`@channel(orient=vertical|horizontal, width=<um>)` -- reserve a routing
        channel (a corridor free of device geometry) for cross-band straps.

        Head-less (all-keyword, like ``@band``): ``orient`` + ``width`` (um) ride in
        the kwargs. Lowered in ``clay.layout.intent`` to a ``LayoutIntent.channels``
        entry, consumed by ``place_blocks_2d`` as the reserved column->lower-region
        separation. Empty -> placer unchanged (byte-parity)."""
        _head, kw = _layout_intent_head_kwargs(items)
        return ParsedLayoutIntentDecl(
            kind="channel",
            attributes={**kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_density_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """Lower a coordinate-free structural whitespace target."""

        kw: dict[str, object] = {}
        for item in items:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("density accepts keyword attributes only")
            key, value = str(item[0]), item[1]
            if key in kw:
                raise ValueError("density attributes must be unique")
            kw[key] = value
        return ParsedLayoutIntentDecl(
            kind="density",
            attributes={**kw},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_od_fill_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """Lower a coordinate-free functional active/OD occupancy target."""

        kw: dict[str, object] = {}
        for item in items:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("od_fill accepts keyword attributes only")
            key, value = str(item[0]), item[1]
            if key in kw:
                raise ValueError("od_fill attributes must be unique")
            kw[key] = value
        return ParsedLayoutIntentDecl(
            kind="od_fill",
            attributes={**kw},
            source_span=_source_span_from_meta(meta),
        )

    @staticmethod
    def _electrical_budget_attributes(kind: str, items) -> dict[str, object]:
        net = str(items[0])
        attributes: dict[str, object] = {"net": net}
        for item in items[1:]:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError(f"{kind} accepts one net followed by keywords")
            key, value = str(item[0]), item[1]
            if key in attributes:
                raise ValueError(f"{kind} attributes must be unique")
            attributes[key] = value
        return attributes

    @lark.v_args(meta=True)
    def layout_power_budget_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="power_budget",
            attributes=self._electrical_budget_attributes("power_budget", items),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_sensitive_net_budget_intent(
        self, meta, items
    ) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="sensitive_net_budget",
            attributes=self._electrical_budget_attributes(
                "sensitive_net_budget", items
            ),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_kelvin_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """``@kelvin(force, sense=net, target=device)``."""

        force_net, sense_net, target_device = map(str, items)
        return ParsedLayoutIntentDecl(
            kind="kelvin",
            nets=(force_net, sense_net),
            attributes={
                "force_net": force_net,
                "sense_net": sense_net,
                "target_device": target_device,
            },
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_channel_route_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        attrs = {}
        for key, value in items:
            if key in attrs:
                raise ValueError(f"duplicate channel_route argument: {key}")
            attrs[key] = value
        return ParsedLayoutIntentDecl(
            kind="channel_route", attributes=attrs,
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_channel_branch_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="channel_branch", nets=(str(items[0]),),
            attributes={"layer": str(items[1])},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_imported_layout_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="imported_layout", attributes={"manifest": str(items[0]), "sha256": str(items[1])},
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_route_tail_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        return ParsedLayoutIntentDecl(
            kind="route_tail", nets=(str(items[0]),),
            attributes={"resource": str(items[1]), "trim_nm": str(items[2])},
            source_span=_source_span_from_meta(meta),
        )

    def layout_place_id_list(self, items) -> tuple[str, ...]:
        return tuple(str(item) for item in items)

    def layout_place_arg(self, items) -> tuple[str, object]:
        value = items[1]
        if isinstance(value, lark.Token):
            value = str(value)
        return (str(items[0]), value)

    @lark.v_args(meta=True)
    def layout_place_group_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        _require_consumed_layout_keywords(items, declaration="place_group", positional="group",
            allowed={"devices", "rows", "row", "priority", "col", "pack_rows", "packing"})
        group = str(items[0])
        attrs = {str(key): value for key, value in items[1:]}
        row_min = row_max = None
        if "row" in attrs and "rows" in attrs:
            raise ValueError("@place_group cannot represent both row and rows")
        rows = attrs.get("rows")
        if rows is not None:
            if not isinstance(rows, str) or rows.count("..") != 1:
                raise ValueError("@place_group rows must be an ordered nonnegative range")
            left, right = rows.split("..")
            if not left.isdecimal() or not right.isdecimal():
                raise ValueError("@place_group rows must be an ordered nonnegative range")
            row_min, row_max = int(left), int(right)
            if row_min >= row_max:
                raise ValueError("@place_group rows must be a nondegenerate increasing range")
        # A single `row=N` pins this group to ONE floorplan row (row_min==row_max).
        # This is the 2D-floorplan form: `@place_group(stage1, devices=(a, b),
        # row=1, col=0)` places those devices in row 1, column 0. `col=N` (a single
        # horizontal slot within the row) is carried on `attributes` for the placer
        # to order columns left-to-right.
        single_row = attrs.get("row")
        if single_row is not None:
            try:
                row_min = row_max = int(str(single_row))
            except (TypeError, ValueError) as exc:
                raise ValueError("@place_group row must be a nonnegative integer") from exc
            if row_min < 0:
                raise ValueError("@place_group row must be a nonnegative integer")
        if "col" in attrs:
            try:
                column = int(str(attrs["col"]))
            except (TypeError, ValueError) as exc:
                raise ValueError("@place_group col must be a nonnegative integer") from exc
            if column < 0:
                raise ValueError("@place_group col must be a nonnegative integer")
        devices = attrs.get("devices", ())
        if not isinstance(devices, tuple):
            devices = (str(devices),) if devices else ()
        return ParsedLayoutIntentDecl(
            kind="place_group",
            group=group,
            devices=tuple(str(device) for device in devices),
            row_min=row_min,
            row_max=row_max,
            priority=str(attrs.get("priority", "")) or None,
            attributes=dict(attrs),
            source_span=_source_span_from_meta(meta),
        )

    @lark.v_args(meta=True)
    def layout_route_group_intent(self, meta, items) -> ParsedLayoutIntentDecl:
        """Coordinate-free single- or multi-place-group route declaration."""

        _require_consumed_layout_keywords(items, declaration="route_group", positional="group",
                                          allowed={"groups", "ports"})
        attrs = {str(key): value for key, value in items[1:]}

        return ParsedLayoutIntentDecl(
            kind="route_group",
            attributes={"group": str(items[0]), **attrs},
            source_span=_source_span_from_meta(meta),
        )


@dataclass
class FlowDecl:
    """Parsed `flow <CellName> { entry+ }` block (transformer-internal)."""
    cell_name: str
    decl: ParsedFlowDecl


@dataclass
class QualityDecl:
    """Parsed `quality <CellName> { entry+ }` block (transformer-internal)."""
    cell_name: str
    decl: ParsedQualityDecl


@dataclass
class PolicyDecl:
    """Parsed `policy NAME { entry+ }` block (transformer-internal).

    `name` is the policy name (the file-level registry key); `decl` is the
    public `ParsedPolicyDecl`. v2-lang: policies are file-level, not bound
    to a cell, so this container carries the policy name (not a cell name).
    """
    name: str
    decl: ParsedPolicyDecl


class _FlowQualityMixin:
    """Lark transformer methods for `flow` / `quality` top-level blocks.

    Mirrors `_LayoutIntentMixin`: each block is a top-level statement
    named by its cell. `flow_decl` / `quality_decl` return tagged tuples
    that `_TopMixin.start` binds onto `CellDef.flow_decl` /
    `CellDef.quality_decl` by cell name.
    """

    # ─── flow block ──────────────────────────────────────────
    @lark.v_args(meta=True)
    def flow_decl(self, meta, items) -> tuple[str, FlowDecl]:
        cell_name = str(items[0])
        pdk: str | None = None
        corners: tuple[str, ...] = ()
        steps: tuple[str, ...] = ()
        layout_module: str | None = None
        policy_ref: str | None = None
        for entry in items[1:]:
            if not isinstance(entry, tuple):
                continue
            tag, value = entry
            if tag == "flow_pdk":
                pdk = value
            elif tag == "flow_corners":
                corners = value
            elif tag == "flow_steps":
                steps = value
            elif tag == "flow_layout":
                layout_module = value
            elif tag == "flow_policy":
                policy_ref = value
        return ("flow_decl", FlowDecl(
            cell_name=cell_name,
            decl=ParsedFlowDecl(
                pdk=pdk,
                corners=corners,
                steps=steps,
                layout_module=layout_module,
                policy_ref=policy_ref,
                source_span=_source_span_from_meta(meta),
            ),
        ))

    def flow_pdk(self, items) -> tuple[str, str]:
        # items[0] is the already-unquoted `string` value
        return ("flow_pdk", str(items[0]))

    def flow_corners(self, items) -> tuple[str, tuple[str, ...]]:
        return ("flow_corners", tuple(str(item) for item in items))

    def flow_steps(self, items) -> tuple[str, tuple[str, ...]]:
        return ("flow_steps", tuple(str(item) for item in items))

    def flow_layout(self, items) -> tuple[str, str]:
        # items[0] is the transformed `dotted_name` (already a str)
        return ("flow_layout", str(items[0]))

    def flow_policy(self, items) -> tuple[str, str]:
        # items[0] is the CNAME policy name selected by this flow.
        return ("flow_policy", str(items[0]))

    # ─── policy block (v2-lang) ──────────────────────────────
    @lark.v_args(meta=True)
    def policy_decl(self, meta, items) -> tuple[str, PolicyDecl]:
        name = str(items[0])
        place_policy: dict[str, object] = {}
        route_policy: dict[str, object] = {}
        converge: dict[str, object] = {}
        learn = False
        for entry in items[1:]:
            if not isinstance(entry, tuple):
                continue
            tag, value = entry
            if tag == "pol_place":
                place_policy = value
            elif tag == "pol_route":
                route_policy = value
            elif tag == "pol_converge":
                converge = _coerce_converge(value)
            elif tag == "pol_learn":
                learn = value
        return ("policy_decl", PolicyDecl(
            name=name,
            decl=ParsedPolicyDecl(
                name=name,
                place_policy=place_policy,
                route_policy=route_policy,
                converge=converge,
                learn=learn,
                source_span=_source_span_from_meta(meta),
            ),
        ))

    def policy_kv(self, items) -> tuple[str, tuple[str, object]]:
        # items: [CNAME key, policy_val]. policy_val is a CNAME (str), a
        # SIGNED_NUMBER token (str), or a tuple of CNAMEs (pol_id_list).
        key = str(items[0])
        return ("policy_kv", (key, items[1]))

    def pol_id_list(self, items) -> tuple[str, ...]:
        return tuple(str(item) for item in items)

    def _fold_kvs(self, items) -> dict[str, object]:
        out: dict[str, object] = {}
        for entry in items:
            if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == "policy_kv":
                key, value = entry[1]
                if isinstance(value, tuple):
                    out[key] = value
                else:
                    out[key] = str(value)
        return out

    def pol_place(self, items) -> tuple[str, dict[str, object]]:
        return ("pol_place", self._fold_kvs(items))

    def pol_route(self, items) -> tuple[str, dict[str, object]]:
        return ("pol_route", self._fold_kvs(items))

    def pol_converge(self, items) -> tuple[str, dict[str, object]]:
        return ("pol_converge", self._fold_kvs(items))

    def pol_learn(self, items) -> tuple[str, bool]:
        # items[0] is a CNAME: `on` / `off` / `true` / `false`.
        flag = str(items[0]).strip().lower()
        return ("pol_learn", flag in ("on", "true", "yes", "1"))

    # ─── quality block ───────────────────────────────────────
    @lark.v_args(meta=True)
    def quality_decl(self, meta, items) -> tuple[str, QualityDecl]:
        cell_name = str(items[0])
        kpis: list[tuple[str, str, float, str | None]] = []
        consistencies: list[tuple[str, float]] = []
        requires: list[str] = []
        for entry in items[1:]:
            if not isinstance(entry, tuple):
                continue
            tag, value = entry
            if tag == "q_kpi":
                kpis.append(value)
            elif tag == "q_consistency":
                consistencies.append(value)
            elif tag == "q_require":
                requires.append(value)
        return ("quality_decl", QualityDecl(
            cell_name=cell_name,
            decl=ParsedQualityDecl(
                kpis=kpis,
                consistencies=consistencies,
                requires=requires,
                source_span=_source_span_from_meta(meta),
            ),
        ))

    def q_kpi(self, items) -> tuple[str, tuple[str, str, float, str | None]]:
        # items: [CNAME key, string op, SIGNED_NUMBER value, (string unit)?]
        key = str(items[0])
        op = str(items[1])
        value = float(str(items[2]))
        unit = str(items[3]) if len(items) > 3 else None
        return ("q_kpi", (key, op, value, unit))

    def q_consistency(self, items) -> tuple[str, tuple[str, float]]:
        # items: [CNAME key, SIGNED_NUMBER rel_tol]
        return ("q_consistency", (str(items[0]), float(str(items[1]))))

    def q_require(self, items) -> tuple[str, str]:
        return ("q_require", str(items[0]))


def _coerce_converge(kvs: dict[str, object]) -> dict[str, object]:
    """Coerce `@converge(...)` kv strings to numbers.

    `place_iters` / `route_iters` -> int; `tol` (and any other key) -> float
    when numeric, otherwise left as-is. Tuple values are left untouched.
    """
    out: dict[str, object] = {}
    for key, value in kvs.items():
        if isinstance(value, tuple):
            out[key] = value
            continue
        text = str(value)
        if key in ("place_iters", "route_iters"):
            try:
                out[key] = int(float(text))
            except (TypeError, ValueError):
                out[key] = text
            continue
        try:
            out[key] = float(text)
        except (TypeError, ValueError):
            out[key] = text
    return out


def _source_span_from_meta(meta) -> dict[str, object] | None:
    line = getattr(meta, "line", None)
    column = getattr(meta, "column", None)
    if line is None or column is None:
        return None
    return {
        "line": int(line),
        "column": int(column),
        "end_line": int(getattr(meta, "end_line", line) or line),
        "end_column": int(getattr(meta, "end_column", column) or column),
    }


# ─── annotation_kind → LayoutAnnotation factory (moved) ──────────────
#
# `_build_layout_annotation` and the per-kind builders (`_build_matching`
# / `_build_guard_ring` / `_build_well_isolated` / `_build_symmetric` /
# `_build_interdigitated` / `_build_orient` / `_build_adjacent` /
# `_build_bulk` / `_build_tap`) plus the `_KindLiteral` / `_KwArg` arg
# wrappers now live in `_annotation_factories.py` (imported at the top of
# this module). They are transformer-independent, so the move is a pure
# refactor with no parsing-behavior change. The transformer methods above
# (`annotation_block` / `annotation_arg_literal` / `annotation_arg_kwarg`)
# call them via the import.
