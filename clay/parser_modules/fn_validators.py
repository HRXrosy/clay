"""
Function-body parse-time validators (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B). Two
pieces of well-formedness we enforce at parse time so eval-time errors
are rare and predictable:

  1. EVERY EXECUTION PATH MUST RETURN. A body that drops off the
     end of its statement list — including all branches of an
     if/else — is rejected. An `if` without an `else` clause whose
     body returns is OK only if the body's TRAILING statements
     include a return at the same level after the if (the if then
     becomes a guard / sentinel).

  2. NO REDECLARATION IN THE SAME LEXICAL SCOPE. A `let` whose name
     collides with a function argument or with an earlier `let` in
     the same scope is rejected. Nested scopes (inside if/else)
     shadow freely.

R79 B Rust delegation surface preserved byte-exact — the hasattr /
ImportError guard pattern matches the pre-split version verbatim.
"""

from __future__ import annotations


def _validate_fn_body_terminates(body, *, fn_name: str) -> None:
    """Reject bodies whose execution can fall off the end without
    hitting a `return`. See module docstring for rules.

    v0.1.02 R79 B: when `CLAY_USE_RUST_PARSER=1` is set, the walk runs
    on the Rust side (`semantic_fn_eval::validate_fn_body_terminates`).
    The Rust path is bit-equivalent on accept-vs-reject; on rejection
    the Python caller still raises the canonical ValueError message
    (Rust returns a short status code only). On any None return (Rust
    path unavailable / malformed body) Python falls back to the local
    `_body_always_returns` walk verbatim.
    """
    # v0.1.02 R79 B: Rust delegation hook. Returns "ok" / "unterminated"
    # / None — None means fall back to Python.
    try:
        from .._rust_bridge import try_rust_validate_fn_body_terminates
    except (ImportError, AttributeError):
        try_rust_validate_fn_body_terminates = None  # type: ignore[assignment]
    if try_rust_validate_fn_body_terminates is not None:
        status = try_rust_validate_fn_body_terminates(body, fn_name)
        if status == "ok":
            return
        if status == "unterminated":
            raise ValueError(
                f"fn {fn_name!r}: not every execution path ends in a "
                f"`return EXPR;`. Add a final `return` after the if/else, "
                f"or give every branch its own `return`."
            )
        # status is None → Rust path unavailable or out-of-subset shape;
        # fall through to the Python walk below.
    if not _body_always_returns(body):
        raise ValueError(
            f"fn {fn_name!r}: not every execution path ends in a "
            f"`return EXPR;`. Add a final `return` after the if/else, "
            f"or give every branch its own `return`."
        )


def _body_always_returns(body) -> bool:
    """True if executing this body always reaches a `return`."""
    if not isinstance(body, dict) or body.get("kind") != "body":
        return False
    stmts = body.get("stmts", [])
    for stmt in stmts:
        kind = stmt.get("kind") if isinstance(stmt, dict) else None
        if kind == "return":
            return True
        if kind == "if":
            then_returns = _body_always_returns(stmt["then"])
            else_body = stmt["else"]
            else_returns = (
                else_body is not None and _body_always_returns(else_body)
            )
            if then_returns and else_returns:
                return True
            # If the if doesn't fully cover, fall through to the next
            # statement (which may itself be a return).
            continue
        # `let` is a no-op for termination analysis; skip and continue.
    return False


def _validate_fn_body_lets(
    body, *, fn_name: str, outer_names: set[str],
) -> None:
    """Reject lets that shadow function args or earlier lets in the
    same scope. Walks nested if/else bodies with fresh scopes.

    v0.1.02 R79 B: when `CLAY_USE_RUST_PARSER=1` is set, the walk runs
    on the Rust side (`semantic_fn_eval::validate_fn_body_lets`). The
    Rust path is bit-equivalent in terms of which (name, scope) is
    flagged on rejection; the Python caller raises the canonical
    ValueError message. On any None return Python falls back to the
    local walk verbatim.
    """
    # v0.1.02 R79 B: Rust delegation hook.
    try:
        from .._rust_bridge import try_rust_validate_fn_body_lets
    except (ImportError, AttributeError):
        try_rust_validate_fn_body_lets = None  # type: ignore[assignment]
    if try_rust_validate_fn_body_lets is not None:
        # Sort outer_names so the Rust → Python error message has a
        # stable scope listing (Python `set` iteration order is
        # insertion-stable but tests may compare against `sorted(scope)`
        # output verbatim).
        outer_list = sorted(outer_names)
        result = try_rust_validate_fn_body_lets(body, fn_name, outer_list)
        if result is not None:
            status, name, scope_list = result
            if status == "ok":
                return
            if status == "redecl":
                raise ValueError(
                    f"fn {fn_name!r}: let {name!r} re-declares an "
                    f"identifier already bound in the same scope "
                    f"({scope_list}). `let` is immutable / "
                    f"single-assignment in v0.0.37 — pick a new name."
                )
            # Other status → fall through to Python walk.
    scope = set(outer_names)
    if not isinstance(body, dict) or body.get("kind") != "body":
        return
    for stmt in body.get("stmts", []):
        kind = stmt.get("kind") if isinstance(stmt, dict) else None
        if kind == "let":
            name = stmt["name"]
            if name in scope:
                raise ValueError(
                    f"fn {fn_name!r}: let {name!r} re-declares an "
                    f"identifier already bound in the same scope "
                    f"({sorted(scope)}). `let` is immutable / "
                    f"single-assignment in v0.0.37 — pick a new name."
                )
            scope.add(name)
            continue
        if kind == "if":
            _validate_fn_body_lets(
                stmt["then"], fn_name=fn_name, outer_names=scope,
            )
            else_body = stmt["else"]
            if else_body is not None:
                _validate_fn_body_lets(
                    else_body, fn_name=fn_name, outer_names=scope,
                )
            continue
        # `return` has no scoping side-effect; skip.
