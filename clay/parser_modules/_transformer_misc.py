"""
Miscellaneous transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
cell_sizing_block + corners_block / corner_name + temps_block /
temp_value + bias_tree_block / bias_decl / bias_external / bias_diode /
bias_mirror.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

from ..ir import BiasNode
from ._constants import _VALID_CORNERS, _resolve_unit_multiplier


class _MiscMixin:
    """Miscellaneous transformer methods (corners, temps, bias_tree)."""

    def cell_sizing_block(self, items):
        # items: list of (key, value) tuples from override_field
        return ("sizing", dict(items))

    # ─── v0.0.21: corners_block parsing ──────────────────────

    def corners_block(self, items) -> tuple[str, list[str]]:
        """items: list of corner_name strings (already de-tokenized).

        Validates each corner name against the sky130 corner set. We
        validate at parse time (not at sim time) so the LLM gets a
        clear error early. Duplicates are deliberately preserved: if a
        user writes `corners { tt; tt; }`, we trust them — there's no
        semantic reason a CI run can't double-up the typical corner.
        """
        names: list[str] = list(items)
        if not names:
            # Grammar enforces at least one corner_name, so this is
            # mainly a defensive check. An empty corners block is a
            # surface-level mistake — surface a clear error.
            raise ValueError(
                "corners { ... } block is empty — declare at least one "
                "corner (e.g. `corners { tt; }`)."
            )
        for name in names:
            if name not in _VALID_CORNERS:
                raise ValueError(
                    f"Unknown corner {name!r} in `corners {{ ... }}` "
                    f"block. Valid corners: {sorted(_VALID_CORNERS)}."
                )
        return ("corners", names)

    def corner_name(self, items) -> str:
        return str(items[0])

    # ─── v0.0.24 (G11): temps_block parsing ──────────────────

    def temps_block(self, items) -> tuple[str, list[float]]:
        """items: list of float Celsius values (already de-tokenized).

        Validates ≥1 temp and that every value is finite. Duplicates
        are preserved (a sweep may legitimately re-run a temp). NaN /
        infinity is rejected — a non-finite "temperature" is a surface
        mistake, not a useful sim point.
        """
        values: list[float] = list(items)
        if not values:
            # Defensive: the grammar already requires ≥1 temp_value, so
            # this path should be unreachable. Mirrors corners_block.
            raise ValueError(
                "temps { ... } block is empty — declare at least one "
                "temperature (e.g. `temps { 27C; }`)."
            )
        import math
        for v in values:
            if not math.isfinite(v):
                raise ValueError(
                    f"temps block contains non-finite value {v!r} — "
                    f"every temperature must be a real number in °C."
                )
        return ("temps", values)

    def temp_value(self, items) -> float:
        """A single `<number>` or `<number>C` entry inside temps { ... }.

        items[0] is a SIGNED_NUMBER Token; the optional `"C"` literal
        is consumed by the rule and does not surface as a child node
        (literal strings without a name don't appear in the tree).
        Returns a plain float Celsius value. Negative and zero are
        both legal — `-40C`, `0`, `0C` all parse to the obvious value.
        """
        return float(items[0])

    # ─── v0.0.26 (G2): bias_tree_block parsing ───────────────

    def bias_tree_block(self, items) -> tuple[str, list[BiasNode]]:
        """items: list of BiasNode (one per `bias_decl`).

        Semantic checks at parse time:
          * empty list → reject (grammar already enforces ≥1 decl,
            this is defensive)
          * duplicate names → reject
          * unknown kind → already rejected by grammar (only the four
            kind keywords match); the transformer for kind is
            defensive on top
          * forward reference (a node references one declared later)
            → reject
          * cycle (would imply forward ref, so caught by the same
            check; explicit message helps users)
          * mirror multiplier ≤ 0 → reject
          * external value ≤ 0 → reject
        """
        nodes: list[BiasNode] = list(items)
        if not nodes:
            raise ValueError(
                "bias_tree { ... } block is empty — declare at least "
                "one bias node (e.g. `Iref: external = 10uA;`)."
            )
        # Duplicate-name check
        seen: set[str] = set()
        for n in nodes:
            if n.name in seen:
                raise ValueError(
                    f"bias_tree: duplicate node name {n.name!r}. "
                    f"Each bias node must have a unique LHS name."
                )
            seen.add(n.name)
        # Forward-reference check: every `src` must refer to a node
        # declared earlier in the block. This catches both forward refs
        # and cycles (a cycle requires at least one forward ref by
        # definition in a linear declaration order).
        declared_so_far: set[str] = set()
        for n in nodes:
            if n.src is not None:
                if n.src not in declared_so_far:
                    if n.src in seen:
                        # Declared later → forward reference
                        raise ValueError(
                            f"bias_tree: node {n.name!r} references "
                            f"{n.src!r} which is declared LATER. "
                            f"Declare upstream nodes first (e.g. Iref "
                            f"before the pmos_diode that drains it)."
                        )
                    else:
                        # Never declared → unknown name
                        raise ValueError(
                            f"bias_tree: node {n.name!r} references "
                            f"undeclared bias node {n.src!r}. "
                            f"Declared so far: {sorted(declared_so_far)}."
                        )
            # Self-reference is a 1-cycle
            if n.src == n.name:
                raise ValueError(
                    f"bias_tree: node {n.name!r} references itself."
                )
            declared_so_far.add(n.name)
        # Multiplier and value sanity checks
        for n in nodes:
            if n.kind in ("pmos_mirror", "nmos_mirror"):
                if n.by_M is None or n.by_M <= 0:
                    raise ValueError(
                        f"bias_tree: mirror {n.name!r} has "
                        f"non-positive multiplier {n.by_M!r}; mirror "
                        f"multipliers must be positive integers."
                    )
            elif n.kind == "external":
                if n.value is None or n.value <= 0:
                    raise ValueError(
                        f"bias_tree: external node {n.name!r} has "
                        f"non-positive value {n.value!r}; reference "
                        f"currents must be positive."
                    )
        return ("bias_tree", nodes)

    def bias_decl(self, items) -> BiasNode:
        """A single `<name>: <rhs>;` bias declaration.

        items[0] is the CNAME LHS; items[1] is the parsed RHS (one of
        the three bias_* alternatives), which returns a tuple
        identifying the kind and its fields.
        """
        name = str(items[0])
        rhs = items[1]
        kind = rhs[0]
        if kind == "external":
            return BiasNode(name=name, kind="external", value=rhs[1])
        if kind == "diode":
            return BiasNode(name=name, kind=rhs[1], src=rhs[2])
        if kind == "mirror":
            return BiasNode(name=name, kind=rhs[1], src=rhs[2], by_M=rhs[3])
        # Defensive — grammar should make this unreachable.
        raise ValueError(
            f"bias_tree: unknown rhs kind {kind!r} for node {name!r}."
        )

    def bias_external(self, items) -> tuple[str, float]:
        """`external = <SIGNED_NUMBER> [UNIT]`.

        Unit is REQUIRED — bias chains carry physical current and a
        bare number is too ambiguous to accept silently. We see the
        raw tokens here (not through the `number` rule) so we can
        tell `external = 10` apart from `external = 10uA`.
        """
        # items: [SIGNED_NUMBER_token]  or  [SIGNED_NUMBER_token, UNIT_token]
        if len(items) < 2:
            raise ValueError(
                "bias_tree: `external = <value>` requires a unit "
                "(e.g. `external = 10uA;`). Bare numbers are rejected "
                "because the physical dimension would be ambiguous."
            )
        raw = float(items[0])
        unit = str(items[1])
        mult = _resolve_unit_multiplier(unit)
        return ("external", raw * mult)

    def bias_diode(self, items) -> tuple[str, str, str]:
        """`pmos_diode|nmos_diode from <name>`.

        items: [BIAS_DIODE_KIND_token, CNAME_token]. The kind keyword
        is matched as a high-priority terminal (see grammar) so it
        survives intact as a child token here.
        """
        kind = str(items[0])
        if kind not in ("pmos_diode", "nmos_diode"):
            raise ValueError(
                f"bias_tree: unknown diode kind {kind!r}. "
                f"Valid: 'pmos_diode' or 'nmos_diode'."
            )
        src = str(items[1])
        return ("diode", kind, src)

    def bias_mirror(self, items) -> tuple[str, str, str, int]:
        """`pmos_mirror|nmos_mirror of <name> by <int>`.

        items: [BIAS_MIRROR_KIND_token, CNAME_token, SIGNED_NUMBER_token].
        """
        kind = str(items[0])
        if kind not in ("pmos_mirror", "nmos_mirror"):
            raise ValueError(
                f"bias_tree: unknown mirror kind {kind!r}. "
                f"Valid: 'pmos_mirror' or 'nmos_mirror'."
            )
        src = str(items[1])
        raw_M = float(items[2])
        if not raw_M.is_integer():
            raise ValueError(
                f"bias_tree: mirror multiplier must be an integer, "
                f"got {raw_M!r}."
            )
        return ("mirror", kind, src, int(raw_M))
