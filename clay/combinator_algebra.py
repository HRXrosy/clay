"""
Clay combinator algebra: formal equivalence via mini e-graph
(v0.1.04 / Round 80 Agent C / BL-087).

LLM-extrapolation audit Tier 0 §10 minimum four-piece kit, final item.
This module gives the LLM authoring loop a *type-level* signal for
combinator topology rewrites: two expressions that an LLM emits as
syntactically different but semantically identical (e.g. one with
``cascade(cascade(A, B), C)`` and another with
``cascade(A, cascade(B, C))``) collapse to the same equivalence class.

Why this matters for LLM extrapolation
======================================

Without combinator algebra equivalence, an RL training loop scoring
LLM-generated designs would penalise two emitted expressions that are
mathematically the same just because their parenthesisation differs.
That is *false friction*: the gradient signal teaches the LLM to favour
one syntactic shape over another even though both produce the same
underlying netlist after combinator lowering. The equivalence class
lets the training loop project the LLM output onto its canonical
representative before computing reward, so the gradient signal is on
the *circuit decision* rather than on the *expression formatting*.

Three citations frame this:

* Willsey et al., *egg: Fast and Extensible Equality Saturation*,
  POPL 2021 — the e-graph data structure that makes equivalence
  saturation practical.
* Zhang et al., *Better Together: Unifying Datalog and Equality
  Saturation* (egglog, PLDI 2023) — the rule-declaration surface that
  inspired the four rules we land here.
* Yang et al., *LLM-Guided Equality Saturation* (EGRAPHS 2025) —
  the specific argument for routing LLM output through an e-graph
  equivalence query.

Public API
----------

The module exposes three call surfaces, in order of typical use:

* :func:`is_combinator_equivalent` — Boolean equivalence query. Takes
  two :class:`CombinatorExpr` values and returns ``True`` iff the
  mini e-graph collapses them to the same canonical form. This is the
  primary entry the RL training loop calls.

* :func:`equivalence_class_canonical` — returns the canonical form
  s-expression string for a given combinator expression. Useful as a
  dictionary key when grouping a batch of LLM outputs into clusters.

* :func:`combinator_rule_metadata` — returns ``(rule_count, [rule_names])``
  describing the declared rewrite rules. The R80 acceptance gate asserts
  ``rule_count >= 4``.

Declared rules (mirrored from ``crates/clay_ir/src/combinator_egg.rs``)
----------------------------------------------------------------------

* **R1 — Associativity of cascade**
  ``cascade(cascade(A, B), C) ≡ cascade(A, cascade(B, C))``

* **R2 — Identity is the unit of cascade**
  ``cascade(A, identity) ≡ A``
  ``cascade(identity, A) ≡ A``

* **R3 — Compensate/MirrorMatch commute within a stage**
  ``compensate(mirror_match(A, B)) ≡ mirror_match(compensate(A), B)``
  Assumption: Miller-cap insertion and bias-rail mirroring are
  orthogonal lowering effects.

* **R4 — Decouple distributes over cascade**
  ``decouple(cascade(A, B)) ≡ cascade(decouple(A), decouple(B))``
  Assumption: decoupling networks do not bridge stage boundaries.

Honest-scope reminder
---------------------

This slice ships scaffolding + the four declared rules. Performance
and full e-graph saturation (vs. our union-find on canonical forms)
are deferred to R81+. The Python API surface is stable; the Rust
back-end can swap to the upstream ``egg`` crate without changing the
Python entry points.

No real LLM API call is made by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

try:
    import clay_ir as _clay_ir
except ImportError as exc:  # pragma: no cover - import-error path
    raise ImportError(
        "clay.combinator_algebra requires the clay_ir Rust extension. "
        "Run `maturin develop --release -m crates/clay_ir/Cargo.toml` "
        "to rebuild the PyO3 module."
    ) from exc


__all__ = [
    "CombinatorExpr",
    "Leaf",
    "Identity",
    "Cascade",
    "Feedforward",
    "Compensate",
    "MirrorMatch",
    "Decouple",
    "SampleAndHold",
    "expr_to_sexp",
    "parse_sexp",
    "is_combinator_equivalent",
    "equivalence_class_canonical",
    "combinator_rule_metadata",
]


# ─── Expression AST ────────────────────────────────────────────────
#
# The Python AST mirrors the Rust enum 1-to-1 so the s-expression
# round-trip is byte-stable. Each variant is a frozen dataclass
# (hashable + equatable) so the Python side can also group a batch of
# expressions into a dict keyed on canonical-form strings without
# round-tripping through Rust for every pair.


@dataclass(frozen=True)
class _ExprBase:
    """Marker base for the seven combinator expression variants."""

    pass


@dataclass(frozen=True)
class Leaf(_ExprBase):
    """A named leaf — a registered ``StageInterface`` template name.

    Example: ``Leaf("common_source_pmos_input")``.
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Leaf.name must be a non-empty string, got {self.name!r}"
            )
        for ch in self.name:
            if ch.isspace() or ch in "()":
                raise ValueError(
                    f"Leaf.name must not contain whitespace or parens, "
                    f"got {self.name!r}"
                )


@dataclass(frozen=True)
class _IdentitySentinel(_ExprBase):
    """The algebraic identity — unit of ``cascade``. Use the module-level
    ``Identity`` singleton; do not construct this dataclass directly."""

    pass


Identity = _IdentitySentinel()


@dataclass(frozen=True)
class Cascade(_ExprBase):
    """``cascade(left, right)`` — the ``>>`` operator. Sequential
    composition along the signal path."""

    left: "CombinatorExpr"
    right: "CombinatorExpr"


@dataclass(frozen=True)
class Feedforward(_ExprBase):
    """``feedforward(source, sink, gain=k)`` — cross-stage gm path.

    ``gain`` is carried as a string so canonical-form comparison is
    exact: a Rust f64 formatting drift (``"0.5"`` vs ``"0.50"``) would
    incorrectly classify those as different gains. Callers should
    normalise gain strings upstream if they want loose equality.
    """

    source: "CombinatorExpr"
    sink: "CombinatorExpr"
    gain: str

    def __post_init__(self) -> None:
        if not isinstance(self.gain, str):
            raise ValueError(
                f"Feedforward.gain must be a string, got {type(self.gain).__name__}"
            )


@dataclass(frozen=True)
class Compensate(_ExprBase):
    """``compensate(inner)`` — frequency-compensation wrap on one stage."""

    inner: "CombinatorExpr"


@dataclass(frozen=True)
class MirrorMatch(_ExprBase):
    """``mirror_match(a, b)`` — operating-point coupling across two
    stages. Carries both stages so the rule-3 commutativity check can
    push ``compensate`` into one side."""

    left: "CombinatorExpr"
    right: "CombinatorExpr"


@dataclass(frozen=True)
class Decouple(_ExprBase):
    """``decouple(inner)`` — supply-rail decoupling wrap."""

    inner: "CombinatorExpr"


@dataclass(frozen=True)
class SampleAndHold(_ExprBase):
    """``sample_and_hold(inner)`` — track-and-hold wrap."""

    inner: "CombinatorExpr"


CombinatorExpr = Union[
    Leaf,
    _IdentitySentinel,
    Cascade,
    Feedforward,
    Compensate,
    MirrorMatch,
    Decouple,
    SampleAndHold,
]
"""Type alias for the seven legal combinator AST variants.

A function accepting ``CombinatorExpr`` should validate the variant via
:func:`isinstance` since the union has runtime-distinct dataclasses but
no shared base method.
"""


# ─── S-expression serialisation ────────────────────────────────────


def expr_to_sexp(expr: CombinatorExpr) -> str:
    """Serialise a combinator expression to an s-expression string.

    The format matches the Rust ``parse_sexp`` reader byte-for-byte:

    ::

        identity
        (leaf <name>)
        (cascade <expr> <expr>)
        (feedforward <expr> <expr> <gain>)
        (compensate <expr>)
        (mirror_match <expr> <expr>)
        (decouple <expr>)
        (sample_and_hold <expr>)
    """
    if isinstance(expr, _IdentitySentinel):
        return "identity"
    if isinstance(expr, Leaf):
        return f"(leaf {expr.name})"
    if isinstance(expr, Cascade):
        return f"(cascade {expr_to_sexp(expr.left)} {expr_to_sexp(expr.right)})"
    if isinstance(expr, Feedforward):
        return (
            f"(feedforward {expr_to_sexp(expr.source)} "
            f"{expr_to_sexp(expr.sink)} {expr.gain})"
        )
    if isinstance(expr, Compensate):
        return f"(compensate {expr_to_sexp(expr.inner)})"
    if isinstance(expr, MirrorMatch):
        return f"(mirror_match {expr_to_sexp(expr.left)} {expr_to_sexp(expr.right)})"
    if isinstance(expr, Decouple):
        return f"(decouple {expr_to_sexp(expr.inner)})"
    if isinstance(expr, SampleAndHold):
        return f"(sample_and_hold {expr_to_sexp(expr.inner)})"
    raise TypeError(
        f"expr_to_sexp(): unrecognised expr type {type(expr).__name__}"
    )


# ─── S-expression parsing (Python-side mirror of Rust parse_sexp) ──
#
# We do NOT delegate the parser to Rust because the Python facade should
# accept either an AST object or an s-expression string in tests / LLM
# wiring code, and the symmetric Python parser keeps the round-trip
# fully observable from Python without a Rust call.


def parse_sexp(s: str) -> CombinatorExpr:
    """Parse an s-expression string back into the Python AST.

    Inverse of :func:`expr_to_sexp`. Raises ``ValueError`` on malformed
    input. The parser is whitespace-tolerant and rejects unknown heads
    or bad arities.
    """
    tokens = _tokenise(s)
    idx = 0
    expr, idx = _parse_one(tokens, idx)
    if idx != len(tokens):
        raise ValueError(
            f"trailing tokens after expression: {tokens[idx:]!r}"
        )
    return expr


def _tokenise(s: str) -> List[str]:
    """Split an s-expression source into a flat token stream.

    Whitespace separates tokens; parens are tokens themselves. Empty
    input yields the empty list (caller raises).
    """
    tokens: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        # Bare identifier — consume until whitespace or paren.
        j = i
        while j < n and not s[j].isspace() and s[j] not in "()":
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def _parse_one(tokens: List[str], idx: int) -> Tuple[CombinatorExpr, int]:
    if idx >= len(tokens):
        raise ValueError("unexpected end of input")
    tok = tokens[idx]
    if tok == "identity":
        return Identity, idx + 1
    if tok != "(":
        raise ValueError(f"unexpected token: {tok!r}")
    # Expect head token next.
    idx += 1
    if idx >= len(tokens):
        raise ValueError("unexpected end of input after '('")
    head = tokens[idx]
    idx += 1
    sub_exprs: List[CombinatorExpr] = []
    tail_tokens: List[str] = []
    while idx < len(tokens) and tokens[idx] != ")":
        if tokens[idx] == "(":
            sub, idx = _parse_one(tokens, idx)
            sub_exprs.append(sub)
        elif tokens[idx] == "identity":
            sub_exprs.append(Identity)
            idx += 1
        else:
            tail_tokens.append(tokens[idx])
            idx += 1
    if idx >= len(tokens):
        raise ValueError(f"missing closing paren for {head!r}")
    idx += 1  # consume ')'
    if head == "leaf":
        if sub_exprs or len(tail_tokens) != 1:
            raise ValueError(
                f"bad arity for 'leaf': got {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return Leaf(tail_tokens[0]), idx
    if head == "cascade":
        if len(sub_exprs) != 2 or tail_tokens:
            raise ValueError(
                f"bad arity for 'cascade': {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return Cascade(sub_exprs[0], sub_exprs[1]), idx
    if head == "feedforward":
        if len(sub_exprs) != 2 or len(tail_tokens) != 1:
            raise ValueError(
                f"bad arity for 'feedforward': {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return Feedforward(sub_exprs[0], sub_exprs[1], tail_tokens[0]), idx
    if head == "compensate":
        if len(sub_exprs) != 1 or tail_tokens:
            raise ValueError(
                f"bad arity for 'compensate': {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return Compensate(sub_exprs[0]), idx
    if head == "mirror_match":
        if len(sub_exprs) != 2 or tail_tokens:
            raise ValueError(
                f"bad arity for 'mirror_match': {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return MirrorMatch(sub_exprs[0], sub_exprs[1]), idx
    if head == "decouple":
        if len(sub_exprs) != 1 or tail_tokens:
            raise ValueError(
                f"bad arity for 'decouple': {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return Decouple(sub_exprs[0]), idx
    if head == "sample_and_hold":
        if len(sub_exprs) != 1 or tail_tokens:
            raise ValueError(
                f"bad arity for 'sample_and_hold': {len(sub_exprs)} subs / "
                f"{len(tail_tokens)} tokens"
            )
        return SampleAndHold(sub_exprs[0]), idx
    raise ValueError(f"unknown combinator head: {head!r}")


# ─── Public equivalence API ────────────────────────────────────────


def _coerce(expr: Union[CombinatorExpr, str]) -> CombinatorExpr:
    """Accept either an AST node or an s-expression string."""
    if isinstance(expr, str):
        return parse_sexp(expr)
    if isinstance(
        expr,
        (
            Leaf,
            _IdentitySentinel,
            Cascade,
            Feedforward,
            Compensate,
            MirrorMatch,
            Decouple,
            SampleAndHold,
        ),
    ):
        return expr
    raise TypeError(
        f"expected CombinatorExpr or s-expression string, got "
        f"{type(expr).__name__}"
    )


def is_combinator_equivalent(
    expr1: Union[CombinatorExpr, str],
    expr2: Union[CombinatorExpr, str],
) -> bool:
    """Return ``True`` iff two combinator expressions live in the same
    equivalence class under the four declared rewrite rules.

    Accepts either AST nodes or s-expression strings on either side.
    Delegates the canonical-form computation to the Rust mini e-graph
    via PyO3.
    """
    e1 = _coerce(expr1)
    e2 = _coerce(expr2)
    s1 = expr_to_sexp(e1)
    s2 = expr_to_sexp(e2)
    return bool(_clay_ir._combinator_is_equivalent(s1, s2))


def equivalence_class_canonical(
    expr: Union[CombinatorExpr, str],
) -> str:
    """Return the canonical s-expression representative of an
    expression's equivalence class.

    Two expressions are equivalent iff their canonical strings are
    byte-equal — :func:`is_combinator_equivalent` is the recommended
    Boolean query; this entry is for dict-keying a batch of LLM outputs
    into clusters.
    """
    e = _coerce(expr)
    s = expr_to_sexp(e)
    return str(_clay_ir._combinator_canonical_form(s))


def combinator_rule_metadata() -> Tuple[int, List[str]]:
    """Return ``(rule_count, [rule_names])`` describing the declared
    rewrite rules currently shipped in the Rust mini e-graph.

    The R80 acceptance gate asserts ``rule_count >= 4``. R81+ slices
    that add rules MUST bump the rule list in
    ``crates/clay_ir/src/combinator_egg.rs`` alongside this report.
    """
    count, names = _clay_ir._combinator_egg_rule_metadata()
    return int(count), list(names)


# ─── Composed-stage bridge (optional, for stage_combinators users) ─
#
# A user holding a ``ComposedStage`` from ``clay.stage_combinators``
# may want to lift it into the equivalence query without re-typing the
# expression by hand. This bridge handles the seven legal kinds and
# delegates leaf naming to ``template_name``.


def from_composed_stage(composed) -> CombinatorExpr:
    """Lift a ``clay.stage_combinators.ComposedStage`` value into a
    ``CombinatorExpr`` for equivalence-class querying.

    The mapping is:

    * ``kind == "leaf"`` → :class:`Leaf` keyed on ``template_name``.
    * ``kind == "cascade"`` → :class:`Cascade` of the two parts.
    * ``kind == "feedforward"`` → :class:`Feedforward` of the two parts;
      ``gain`` is read from ``meta['gain']`` (formatted with
      ``repr``).
    * ``kind == "compensate"`` → :class:`Compensate` of the single part.
    * ``kind == "mirror"`` → :class:`MirrorMatch` of the two parts.
    * ``kind == "decouple"`` → :class:`Decouple` of the single part.
    * ``kind == "sample_and_hold"`` → :class:`SampleAndHold` of the
      single part.

    Importing this function does NOT pull in ``clay.stage_combinators``
    until called — the lazy import keeps the equivalence query
    standalone for the LLM-evaluation loop that does not need the full
    combinator lowering stack.
    """
    # Local import keeps the dependency edge optional.
    from clay.stage_combinators import ComposedStage  # noqa: F401
    from clay.stage_interface import ComposedStage as _Iface  # noqa: F401

    kind = getattr(composed, "kind", None)
    if kind is None:
        raise TypeError(
            f"from_composed_stage(): expected ComposedStage, got "
            f"{type(composed).__name__}"
        )
    parts = getattr(composed, "parts", ())
    meta = getattr(composed, "meta", {}) or {}
    if kind == "leaf":
        return Leaf(composed.template_name)
    if kind == "cascade":
        if len(parts) != 2:
            raise ValueError(
                f"cascade ComposedStage must have 2 parts, got {len(parts)}"
            )
        return Cascade(from_composed_stage(parts[0]), from_composed_stage(parts[1]))
    if kind == "feedforward":
        if len(parts) != 2:
            raise ValueError(
                f"feedforward ComposedStage must have 2 parts, "
                f"got {len(parts)}"
            )
        gain = meta.get("gain", 1.0)
        return Feedforward(
            from_composed_stage(parts[0]),
            from_composed_stage(parts[1]),
            repr(float(gain)),
        )
    if kind == "compensate":
        if len(parts) != 1:
            raise ValueError(
                f"compensate ComposedStage must have 1 part, "
                f"got {len(parts)}"
            )
        return Compensate(from_composed_stage(parts[0]))
    if kind == "mirror":
        if len(parts) != 2:
            raise ValueError(
                f"mirror ComposedStage must have 2 parts, got {len(parts)}"
            )
        return MirrorMatch(
            from_composed_stage(parts[0]),
            from_composed_stage(parts[1]),
        )
    if kind == "decouple":
        if len(parts) != 1:
            raise ValueError(
                f"decouple ComposedStage must have 1 part, got {len(parts)}"
            )
        return Decouple(from_composed_stage(parts[0]))
    if kind == "sample_and_hold":
        if len(parts) != 1:
            raise ValueError(
                f"sample_and_hold ComposedStage must have 1 part, "
                f"got {len(parts)}"
            )
        return SampleAndHold(from_composed_stage(parts[0]))
    raise ValueError(f"unrecognised ComposedStage.kind: {kind!r}")
