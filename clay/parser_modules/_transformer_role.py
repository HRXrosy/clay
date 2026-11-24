"""
Role-decl + functional-role + object-literal transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
role_size_field / role_pin_field / role_pair_pin_field / role_decl /
functional_* / object_literal / object_field.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

from typing import Any

import lark

from ..ir import ParsedFunctionalRole, RoleDecl
from ._constants import (
    _ROLE_PAIR_PIN_LABELS, _ROLE_PIN_LABELS, _ROLE_SIZE_KEYS,
    _VALID_FUNCTIONAL_ROLE_KINDS, _VALID_ROLE_KINDS,
)


class _RoleMixin:
    """Role-decl, functional-role, and object-literal transformer methods."""

    # ─── v0.0.31 (G3 closeout): lean role declarations ───────

    def role_size_field(self, items) -> tuple[str, str, float]:
        """One `<key> = <number>;` line inside a role body.

        Tagged `("size", key, value)` so the `role_decl` transformer
        can sort it alongside pin overrides. The list of valid keys
        is enforced in `role_decl` (so we get one path with one
        precise error message).
        """
        key = str(items[0])
        value = items[1]
        return ("size", key, value)

    def role_pin_field(self, items) -> tuple[str, str, str]:
        """One `<label> = <net>;` single-net override inside a role body.

        Tagged `("pin", label, net)`. The label set depends on the role
        kind (`tail` for pair, `in`/`out` for mirror, `gate`/`drain` for
        cs); `role_decl` validates against the per-kind table.
        """
        label = str(items[0])
        net = str(items[1])
        return ("pin", label, net)

    def role_pair_pin_field(self, items) -> tuple[str, str, tuple[str, str]]:
        """One `<label> = (<net_a>, <net_b>);` two-net override.

        Tagged `("pair_pin", label, (net_a, net_b))`. Only valid in
        pair roles for `gates` / `drains` — enforced in `role_decl`.
        """
        label = str(items[0])
        net_a = str(items[1])
        net_b = str(items[2])
        return ("pair_pin", label, (net_a, net_b))

    def role_decl(self, items) -> RoleDecl:
        """`<role_kw> NAME { ... }` — lean role declaration.

        items: [role_kw_token, CNAME(name), body_field, body_field, ...].

        Each body_field is a tagged tuple from `role_size_field` /
        `role_pin_field` / `role_pair_pin_field`. We validate:
          * the role kind is one of `_VALID_ROLE_KINDS` (defensive — the
            grammar terminal patterns already restrict the surface)
          * the body is non-empty AND contains at least one of W / L /
            gm_id (an empty role body or one with only pin overrides
            would size every contained device to zero)
          * each size key is in `_ROLE_SIZE_KEYS`; M / nf are integer-
            valued
          * each single-net pin label is in `_ROLE_PIN_LABELS[kind]`
          * each two-net pin label is in `_ROLE_PAIR_PIN_LABELS[kind]`
          * no key/label is assigned twice
        On success, returns a `RoleDecl` IR entry. The CellDef
        transformer (`cell_def`) then expands each role into concrete
        `Device` entries via `_expand_role_to_devices`.
        """
        kind = str(items[0])
        if kind not in _VALID_ROLE_KINDS:
            # Defensive — the grammar terminals (ROLE_PAIR_KW,
            # BIAS_MIRROR_KIND, ROLE_CS_KW) collectively only emit these
            # six strings. If a future grammar tweak adds another spelling
            # without updating the table, this catches it.
            raise ValueError(
                f"role decl: unknown role kind {kind!r}. Valid: "
                f"{sorted(_VALID_ROLE_KINDS)}."
            )
        name = str(items[1])
        body_fields = items[2:]
        if not body_fields:
            raise ValueError(
                f"role {kind} {name!r}: body is empty. At least one of "
                f"W, L, M, or gm_id must be specified — an empty role "
                f"body would size every contained device to zero."
            )

        sizing: dict[str, float] = {}
        pin_overrides: dict[str, str] = {}
        pair_overrides: dict[str, tuple[str, str]] = {}

        allowed_pin = _ROLE_PIN_LABELS[kind]
        allowed_pair = _ROLE_PAIR_PIN_LABELS[kind]

        for entry in body_fields:
            tag = entry[0]
            if tag == "size":
                _t, key, value = entry
                if key not in _ROLE_SIZE_KEYS:
                    raise ValueError(
                        f"role {kind} {name!r}: unknown size key "
                        f"{key!r}. Valid keys: "
                        f"{sorted(_ROLE_SIZE_KEYS)}."
                    )
                if key in sizing:
                    raise ValueError(
                        f"role {kind} {name!r}: size key {key!r} "
                        f"assigned more than once."
                    )
                if key in ("M", "nf"):
                    if not float(value).is_integer():
                        raise ValueError(
                            f"role {kind} {name!r}: {key} must be an "
                            f"integer, got {value!r}."
                        )
                    sizing[key] = float(int(value))
                else:
                    sizing[key] = float(value)
            elif tag == "pin":
                _t, label, net = entry
                if label not in allowed_pin:
                    # Distinguish "wrong-axis label" from "unknown label"
                    # so the LLM sees the right hint.
                    if label in allowed_pair:
                        raise ValueError(
                            f"role {kind} {name!r}: pin label {label!r} "
                            f"expects a two-net tuple — write "
                            f"`{label} = (net_a, net_b);` instead of "
                            f"`{label} = {net};`."
                        )
                    raise ValueError(
                        f"role {kind} {name!r}: unknown pin label "
                        f"{label!r}. Valid single-net labels for this "
                        f"kind: {sorted(allowed_pin) or '<none>'}. "
                        f"Valid two-net labels: "
                        f"{sorted(allowed_pair) or '<none>'}."
                    )
                if label in pin_overrides:
                    raise ValueError(
                        f"role {kind} {name!r}: pin label {label!r} "
                        f"assigned more than once."
                    )
                pin_overrides[label] = net
            elif tag == "pair_pin":
                _t, label, pair = entry
                if label not in allowed_pair:
                    if label in allowed_pin:
                        raise ValueError(
                            f"role {kind} {name!r}: pin label {label!r} "
                            f"expects a single net — write "
                            f"`{label} = <net>;` instead of a tuple."
                        )
                    raise ValueError(
                        f"role {kind} {name!r}: unknown two-net pin "
                        f"label {label!r}. Valid two-net labels for "
                        f"this kind: {sorted(allowed_pair) or '<none>'}. "
                        f"Valid single-net labels: "
                        f"{sorted(allowed_pin) or '<none>'}."
                    )
                if label in pair_overrides:
                    raise ValueError(
                        f"role {kind} {name!r}: pin label {label!r} "
                        f"assigned more than once."
                    )
                pair_overrides[label] = pair
            else:
                # Defensive — grammar should ensure this is unreachable.
                raise ValueError(
                    f"role {kind} {name!r}: unknown body field "
                    f"tag {tag!r}."
                )

        # At least one sizing source must be present so the expansion
        # produces well-sized devices. Either explicit W+L OR a gm_id
        # intent (the sizing pass will derive W from it).
        has_W = "W" in sizing
        has_L = "L" in sizing
        has_gm_id = "gm_id" in sizing
        if not has_gm_id:
            # No gm_id intent — require explicit W and L.
            if not (has_W and has_L):
                raise ValueError(
                    f"role {kind} {name!r}: must specify either "
                    f"`gm_id = ...;` (intent-based sizing) OR both "
                    f"`W = ...;` and `L = ...;` (explicit sizing). "
                    f"Got: {sorted(sizing)}."
                )
        # M defaults to 1 (matches device_decl convention)
        sizing.setdefault("M", 1.0)

        return RoleDecl(
            name=name,
            kind=kind,
            sizing=sizing,
            pin_overrides=pin_overrides,
            pair_overrides=pair_overrides,
        )

    # ─── v0.0.62: functional-role declaration parsing ────────

    def functional_attr_id_list(self, items) -> list[str]:
        """Parenthesized identifier list: `(a, b, c)`.

        Returns `list[str]` so the attribute-field rule can store it
        verbatim. Empty `()` is impossible at the grammar level (the
        rule requires at least one CNAME).
        """
        return [str(x) for x in items]

    def functional_attr_field(self, items) -> tuple[str, Any]:
        """One `<key> = <value>` attribute inside a functional-role body.

        Returns the (key, value) pair so `functional_role_decl` can fold
        it into the attributes dict. Value normalization:
          * `number`   → already a `float` (resolved by `number()`).
          * `string`   → already a `str` with quotes stripped.
          * bare CNAME → stored as `str` (lark token coerced via
                         `str()`).
          * id list    → already a `list[str]`.
        Bare CNAMEs vs quoted strings are indistinguishable in the
        resulting dataclass — downstream consumers infer the
        distinction from the attribute key.
        """
        key = str(items[0])
        raw = items[1]
        # CNAME shows up as a lark Token; normalize to plain str so the
        # IR carries no lark-internal types.
        if isinstance(raw, lark.Token):
            value: Any = str(raw)
        else:
            value = raw
        return (key, value)

    def functional_attr_list(self, items) -> list[tuple[str, Any]]:
        """Attribute list inside a functional-role body.

        Returns the raw list of (key, value) pairs in source order so
        `functional_role_decl` can run duplicate-key detection with a
        precise error message.
        """
        return list(items)

    @lark.v_args(meta=True)
    def functional_role_decl(self, meta, items) -> ParsedFunctionalRole:
        """`<FUNCTIONAL_ROLE_KW> NAME { <attrs>? }` — functional-role
        declaration.

        items: [FUNCTIONAL_ROLE_KW_token, CNAME(name), attr_list?].
        The grammar makes the attribute list optional so the parser
        can distinguish "missing attrs" from "non-empty attrs" with a
        precise error message instead of a generic parse failure.

        Validation:
          * the kind is one of `_VALID_FUNCTIONAL_ROLE_KINDS` (defensive
            guard — the grammar terminal already restricts the surface)
          * the body is non-empty (`name { }` is rejected so every
            functional role carries at least one attribute)
          * no attribute key is assigned twice within the same body

        On success, returns a `ParsedFunctionalRole` IR entry. The
        `cell_def` / `topology_block` transformers route the entry into
        `CellDef.functional_roles`. The compiler / lowering layer is
        responsible for translating each functional role into
        underlying structural roles or device primitives — the parser
        does NOT lower or expand here.
        """
        kind = str(items[0])
        if kind not in _VALID_FUNCTIONAL_ROLE_KINDS:
            # Defensive — the grammar terminal restricts the surface
            # to these six strings. A future grammar tweak that adds
            # another spelling without updating the table will surface
            # here with a precise error.
            raise ValueError(
                f"functional role decl: unknown kind {kind!r}. Valid: "
                f"{sorted(_VALID_FUNCTIONAL_ROLE_KINDS)}."
            )
        name = str(items[1])
        # `functional_attr_list?` collapses to either a list (when
        # present) or is absent entirely.
        raw_attrs: list[tuple[str, Any]] = (
            items[2] if len(items) > 2 else []
        )
        if not raw_attrs:
            raise ValueError(
                f"functional role {kind} {name!r}: body is empty. At "
                f"least one attribute (`key = value`) must be "
                f"specified — a bare `{kind} {name} {{ }}` carries no "
                f"functional intent."
            )

        attributes: dict[str, Any] = {}
        for key, value in raw_attrs:
            if key in attributes:
                raise ValueError(
                    f"functional role {kind} {name!r}: attribute "
                    f"{key!r} assigned more than once."
                )
            attributes[key] = value

        return ParsedFunctionalRole(
            kind=kind,
            name=name,
            attributes=attributes,
            source_span=_source_span_from_meta(meta),
        )

    # ─── v0.0.23 (G9): object literal parsing ────────────────

    def object_literal(self, items) -> dict[str, object]:
        """Flat map literal: `{ key1: 'v1', key2: 42, key3: bare, }`.

        Returns a plain `dict[str, object]`. Values may be str / float
        / int — the same scalar types the topology_value rule already
        accepts. No nesting in this version; if a value is itself an
        object_literal the grammar rejects it. Trailing comma and
        empty `{ }` are both legal.
        """
        return dict(items)

    def object_field(self, items) -> tuple[str, object]:
        """Single `key: value` pair inside an object literal."""
        key = str(items[0])
        val = items[1]
        # CNAME-as-value comes back as a Token; treat it as a bareword
        # string so the map<string, string> contract holds for the
        # common rationale use case.
        if isinstance(val, lark.Token):
            val = str(val)
        return (key, val)


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
