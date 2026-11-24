"""
Spec / combinator / leaf transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
combinator_* methods + spec_block / sizing_override_block / spec_field +
across_corners_clause / corner_arg / with_margin_clause + override_field /
op / type_id / dotted_name / qualified_name / simple_cell_ref / number.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

import lark

from ..ir import ParsedCombinatorBlock, Spec, SpecField
from ._constants import (
    _VALID_CORNERS, _resolve_unit_multiplier,
    _validate_combinator_stage_refs,
)


class _SpecMixin:
    """Spec, combinator, and leaf-rule transformer methods."""

    # ─── v0.0.92 (combinator-in-source, Round 68) ──────────────────

    def combinator_block(self, items) -> tuple[str, ParsedCombinatorBlock]:
        """items: [COMBINATOR_KW, CNAME_block_name,
                   stages_list, build_expr_ast]

        The grammar guarantees exactly two body fields in fixed order:
        `stages = [...]` first, `build = expr;` second. Both are
        required; an empty `stages = [];` is rejected at the parse
        level (the grammar requires ≥1 CNAME inside the brackets).

        Validation:
          * stage list is non-empty
          * stage list has no duplicate template names
          * build_expr references `stages[i]` only for valid i (in
            range [0, len(stages))). Bare names inside the build expr
            are also allowed — those resolve at compile time against
            the StageInterface registry.
        """
        _combinator_tok = items[0]   # noqa: F841
        block_name = str(items[1])
        stages_list: list[str] = items[2]
        build_expr = items[3]
        if not stages_list:
            raise ValueError(
                f"combinator {block_name!r}: `stages = []` is empty. "
                f"Declare at least one stage template name."
            )
        # Duplicate-name check — combinator stages are identified by
        # template_name, so duplicates would make `stages[i]` and
        # `stages[j]` ambiguous when both point at the same registered
        # interface. Allow it though (the algebra is total even with
        # duplicate references to the same template).
        # Validate `stage_ref` indices against the declared list length.
        _validate_combinator_stage_refs(
            build_expr,
            stages_len=len(stages_list),
            block_name=block_name,
        )
        block = ParsedCombinatorBlock(
            name=block_name,
            stages=list(stages_list),
            build_expr=build_expr,
        )
        return ("combinator_block", block)

    def combinator_stages_field(self, items) -> list[str]:
        """items: [CNAME_1, CNAME_2, ...] — one ident per declared stage."""
        return [str(t) for t in items]

    def combinator_build_field(self, items):
        """items: [build_expr_ast] — a single combinator expression."""
        return items[0]

    def combinator_expr(self, items):
        """items: [atom_1, CASCADE_OP, atom_2, CASCADE_OP, atom_3, ...]

        Left-associative cascade: every `>>` between two atoms folds
        into a `cascade` AST node. A bare atom (no `>>`) passes through.
        """
        if len(items) == 1:
            return items[0]
        # Items alternate: atom, CASCADE_OP, atom, CASCADE_OP, ...
        # The transformer sees the CASCADE_OP tokens as `Token` objects.
        result = items[0]
        i = 1
        while i < len(items):
            # items[i] is the CASCADE_OP; items[i+1] is the right atom.
            right = items[i + 1]
            result = {
                "kind": "cascade",
                "left": result,
                "right": right,
            }
            i += 2
        return result

    def combinator_paren(self, items):
        # Parenthesized expression — just unwrap.
        return items[0]

    def combinator_call(self, items):
        """items: [call_kw_token, args_list]

        The call_kw is a high-priority terminal; we read its text and
        dispatch on it. `cascade_chain` expects a single
        `combinator_chain_list` argument; the other three accept a
        positional arg list with optional kwargs.
        """
        kw = str(items[0])
        args = items[1]
        if kw == "cascade_chain":
            # `cascade_chain([s1, s2, ...])` — the arg list MUST be a
            # single chain_list payload.
            if not (isinstance(args, dict) and args.get("kind") == "chain_list"):
                raise ValueError(
                    f"combinator: `cascade_chain(...)` expects a single "
                    f"list argument like `cascade_chain([s1, s2, ...])`."
                )
            return {
                "kind": "cascade_chain",
                "stages": args["stages"],
            }
        # Positional + kwarg call shape.
        if not isinstance(args, dict) or args.get("kind") != "arg_list":
            raise ValueError(
                f"combinator: `{kw}(...)` argument list malformed."
            )
        positional = args["positional"]
        kwargs = args["kwargs"]
        if kw == "feedforward":
            if len(positional) != 2:
                raise ValueError(
                    f"combinator: `feedforward(source, sink, gain=...)` "
                    f"requires exactly 2 positional arguments; got "
                    f"{len(positional)}."
                )
            return {
                "kind": "feedforward",
                "source": positional[0],
                "sink": positional[1],
                "kwargs": dict(kwargs),
            }
        if kw == "compensate":
            if len(positional) != 1:
                raise ValueError(
                    f"combinator: `compensate(stage, miller=..., "
                    f"zero=..., target_pm_deg=...)` requires exactly "
                    f"1 positional argument; got {len(positional)}."
                )
            return {
                "kind": "compensate",
                "inner": positional[0],
                "kwargs": dict(kwargs),
            }
        if kw == "mirror_match":
            if len(positional) != 2:
                raise ValueError(
                    f"combinator: `mirror_match(a, b)` requires "
                    f"exactly 2 positional arguments; got "
                    f"{len(positional)}."
                )
            if kwargs:
                raise ValueError(
                    f"combinator: `mirror_match(...)` does not accept "
                    f"keyword arguments; got {sorted(kwargs)!r}."
                )
            return {
                "kind": "mirror_match",
                "left": positional[0],
                "right": positional[1],
            }
        if kw == "decouple":
            # v0.0.95 (Round 72 D): supply-rail decoupling combinator.
            # Surface: `decouple(stage, decoupling_cap=Cd,
            # isolation_device_W=W, target_psrr_db=PSRR)`. Exactly 1
            # positional (the stage to wrap); kwargs supply either
            # explicit sizing (cap + W) or the auto-sizing target.
            if len(positional) != 1:
                raise ValueError(
                    f"combinator: `decouple(stage, decoupling_cap=..., "
                    f"isolation_device_W=..., target_psrr_db=...)` "
                    f"requires exactly 1 positional argument; got "
                    f"{len(positional)}."
                )
            return {
                "kind": "decouple",
                "inner": positional[0],
                "kwargs": dict(kwargs),
            }
        if kw == "sample_and_hold":
            # v0.0.99 (Round 75 C): track-and-hold combinator. Surface:
            # `sample_and_hold(stage, hold_cap=Ch, sample_switch=Wsw,
            # droop_rate_target=Rd, sample_bandwidth_target=BW)`. Exactly
            # 1 positional (the stage whose output is sampled); kwargs
            # supply either explicit sizing (Ch + Wsw) or the auto-
            # sizing target (Rd, optionally with BW).
            if len(positional) != 1:
                raise ValueError(
                    f"combinator: `sample_and_hold(stage, hold_cap=..., "
                    f"sample_switch=..., droop_rate_target=..., "
                    f"sample_bandwidth_target=...)` requires exactly 1 "
                    f"positional argument; got {len(positional)}."
                )
            return {
                "kind": "sample_and_hold",
                "inner": positional[0],
                "kwargs": dict(kwargs),
            }
        raise ValueError(
            f"combinator: unknown call keyword {kw!r}. Valid: "
            f"'feedforward', 'compensate', 'mirror_match', "
            f"'cascade_chain', 'decouple', 'sample_and_hold'."
        )

    def combinator_call_args(self, items):
        """Aggregate positional + kwarg args from a call argument list.

        Each child is either a `combinator_kwarg` dict
        (`{kind: 'kwarg', name, value}`) or a build_expr AST node.
        Returns `{kind: 'arg_list', positional: [...], kwargs: {...}}`.

        Special case: a single `combinator_chain_list` child surfaces
        as `{kind: 'chain_list', stages: [...]}` instead — the
        `combinator_call` dispatcher disambiguates on the result kind.
        """
        # Detect the chain_list case first — it appears as a single
        # child whose AST kind is `chain_list`.
        if (
            len(items) == 1
            and isinstance(items[0], dict)
            and items[0].get("kind") == "chain_list"
        ):
            return items[0]
        positional = []
        kwargs: dict[str, float] = {}
        for child in items:
            if isinstance(child, dict) and child.get("kind") == "kwarg":
                name = child["name"]
                if name in kwargs:
                    raise ValueError(
                        f"combinator: keyword argument {name!r} "
                        f"specified more than once."
                    )
                kwargs[name] = child["value"]
            else:
                positional.append(child)
        return {
            "kind": "arg_list",
            "positional": positional,
            "kwargs": kwargs,
        }

    def combinator_kwarg(self, items) -> dict:
        """items: [CNAME, number_value]"""
        name = str(items[0])
        value = items[1]
        # Numbers come through as `float` via the `number` rule.
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"combinator: keyword argument {name!r} value must be "
                f"a number, got {type(value).__name__}"
            )
        return {"kind": "kwarg", "name": name, "value": float(value)}

    def combinator_chain_list(self, items) -> dict:
        """items: [expr_1, expr_2, ...] — at least one expression."""
        return {"kind": "chain_list", "stages": list(items)}

    def combinator_stage_ref(self, items) -> dict:
        """items: [SIGNED_NUMBER_token] — `stages[N]` reference."""
        raw = float(items[0])
        if not raw.is_integer() or raw < 0:
            raise ValueError(
                f"combinator: `stages[N]` index must be a non-negative "
                f"integer, got {items[0]!r}"
            )
        return {"kind": "stage_ref", "index": int(raw)}

    def combinator_name(self, items) -> dict:
        """items: [CNAME] — bare template name reference."""
        return {"kind": "stage_name", "name": str(items[0])}

    # Note: spec_block is shared with verify_block; in cell_def context
    # its return value is the Spec object. We wrap it in ('spec', Spec)
    # so the cell_def transformer can dispatch on kind. Achieved by
    # patching the existing spec_block to return ('spec', value) when
    # used in cell_def context — but that's invasive. Simpler: add a
    # post-process step that detects raw Spec in items.

    # Actually: rework — we need spec_block in cell_def to be tagged.
    # Simplest is to add a dedicated cell-context rule, but the grammar
    # reuses spec_block. Workaround: detect by type in cell_def above.
    # See the cell_def transformer modification below.

    def spec_block(self, items) -> Spec:
        return Spec(fields=list(items))

    def sizing_override_block(self, items) -> dict[str, float]:
        return dict(items)

    @lark.v_args(meta=True)
    def spec_field(self, meta, items) -> SpecField:
        """spec_field: CNAME ":" type_id op number assertion_modifier* ";"

        v0.0.22 (G7): when `type_id` is the keyword `integer`, the
        right-hand value MUST be an exact integer. The parser accepts
        any integer (positive, zero, negative) so patterns like
        `M_disabled : integer >= 0;` ("may be disabled") can be
        expressed. A semantic check `M ≥ 1` is the PDK / sizing
        layer's job — at parse time we only enforce the *type*
        (whole-number).

        Non-integer values (e.g. `4.5`) raise ValueError.

        v0.0.30 (G11 closeout): the trailing items may include zero or
        more `assertion_modifier` clauses produced by
        `across_corners_clause` / `with_margin_clause`. Each clause is
        tagged `("across_corners", [...])` or `("with_margin", float)`
        so the transformer can sort them without ordering assumptions.
        A given modifier kind may appear at most once per field — a
        repeated `with_margin(...)` / `across_corners(...)` on the same
        field is rejected.
        """
        # First four items are the canonical name / type / op / value
        # tuple; anything beyond is an assertion modifier.
        if len(items) < 4:
            raise ValueError(
                f"spec_field: malformed item list {items!r}"
            )
        name, type_id, op_s, value = items[0], items[1], items[2], items[3]
        modifiers = items[4:]
        dimension = str(type_id)
        if dimension == "integer":
            # Reject any non-whole-number value. `float.is_integer()`
            # handles `4.0`, `4`, `-3.0` correctly; `4.5` returns False.
            if not (isinstance(value, (int, float)) and float(value).is_integer()):
                raise ValueError(
                    f"spec field {str(name)!r}: dimension 'integer' "
                    f"requires an integer value, got {value!r}."
                )
            value = int(value)

        across_corners: list[str] | None = None
        margin_pct: float | None = None
        for mod in modifiers:
            # Each modifier is a tagged tuple from
            # `across_corners_clause` / `with_margin_clause`.
            if not (isinstance(mod, tuple) and mod):
                raise ValueError(
                    f"spec field {str(name)!r}: unexpected modifier "
                    f"{mod!r}."
                )
            kind = mod[0]
            if kind == "across_corners":
                if across_corners is not None:
                    raise ValueError(
                        f"spec field {str(name)!r}: `across_corners(...)`"
                        f" appears more than once. Declare it at most "
                        f"once per spec field."
                    )
                across_corners = list(mod[1])
            elif kind == "with_margin":
                if margin_pct is not None:
                    raise ValueError(
                        f"spec field {str(name)!r}: `with_margin(...)` "
                        f"appears more than once. Declare it at most "
                        f"once per spec field."
                    )
                margin_pct = float(mod[1])
            else:
                raise ValueError(
                    f"spec field {str(name)!r}: unknown assertion "
                    f"modifier kind {kind!r}."
                )
        return SpecField(
            name=str(name),
            dimension=dimension,
            op=op_s,
            value=value,
            across_corners=across_corners,
            margin_pct=margin_pct,
            source_span=_source_span_from_meta(meta),
        )

    # ─── v0.0.30 (G11 closeout): assertion modifier clauses ──────

    def across_corners_clause(self, items) -> tuple[str, list[str]]:
        """`across_corners('tt', 'ss', 'ff')` → tagged tuple.

        items: list of corner-name strings produced by `corner_arg`.
        Each entry must be one of the PDK's valid corner names; an
        empty list is rejected. Duplicate corner names are also
        rejected so a typo like `('ss', 'ss')` surfaces clearly.
        """
        names: list[str] = [str(s) for s in items]
        if not names:
            raise ValueError(
                "across_corners(...) clause is empty — list at least "
                "one corner (e.g. `across_corners('tt')`). For a spec "
                "that applies to every corner, omit the clause entirely."
            )
        seen: set[str] = set()
        for n in names:
            if n in seen:
                raise ValueError(
                    f"across_corners(...) clause contains duplicate "
                    f"corner {n!r}. Each corner may be listed at most once."
                )
            seen.add(n)
            if n not in _VALID_CORNERS:
                raise ValueError(
                    f"Unknown corner {n!r} in `across_corners(...)` "
                    f"clause. Valid corners: {sorted(_VALID_CORNERS)}."
                )
        return ("across_corners", names)

    def corner_arg(self, items) -> str:
        """A single corner argument inside `across_corners(...)`.

        The grammar rule is just `string`, so `items[0]` is already a
        Python string with the surrounding quotes stripped by the
        `string` transformer above.
        """
        return str(items[0])

    def with_margin_clause(self, items) -> tuple[str, float]:
        """`with_margin(N%)` → ('with_margin', N).

        items: [SIGNED_NUMBER_token]. The literal `%` is consumed by
        the grammar but does not surface as a child.

        Validation:
          * N must be ≥ 0 — negative margins are rejected outright
            (loosening a spec is not the intended use).
          * NaN / infinity rejected as a defensive guard.

        Zero margin is permitted (no-op tightening). Margins ≥ 100%
        are accepted: `with_margin(100%)` doubles the required value
        on a `>=` spec, which is unusual but well-defined.
        """
        import math as _math
        raw = float(items[0])
        if not _math.isfinite(raw):
            raise ValueError(
                f"with_margin(...): margin must be a real number, "
                f"got {raw!r}."
            )
        if raw < 0.0:
            raise ValueError(
                f"with_margin({raw}%): negative margins are not "
                f"permitted. Use a positive percentage (e.g. "
                f"`with_margin(20%)`) to tighten the spec, or omit "
                f"the clause to apply the spec as written."
            )
        return ("with_margin", raw)

    def override_field(self, items):
        return (str(items[0]), items[1])

    def op(self, items):
        return str(items[0])

    def type_id(self, items):
        return str(items[0])

    def dotted_name(self, items) -> str:
        return ".".join(str(x) for x in items)

    def qualified_name(self, items) -> str:
        # qualified_name: CNAME ("." CNAME)* "::" CNAME
        # The "::" terminal is consumed; we know the LAST token is the cell
        # name (after ::) and the prior tokens form the package path.
        parts = [str(x) for x in items]
        return ".".join(parts[:-1]) + "::" + parts[-1]

    def simple_cell_ref(self, items) -> str:
        """v0.0.34 (PLL gap A): bare-name instance target.

        `instance amp = LocalAmp { ... };` produces a single CNAME with
        NO package path. We return the bare name verbatim; the compiler's
        `_build_subcell` distinguishes bare vs qualified by checking for
        the `::` separator.
        """
        return str(items[0])

    def number(self, items) -> float:
        n = float(items[0])
        if len(items) > 1:
            unit = str(items[1])
            mult = _resolve_unit_multiplier(unit)
            n *= mult
        return n


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
