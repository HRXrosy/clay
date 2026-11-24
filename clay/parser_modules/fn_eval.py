"""
Function-expression evaluator + sizing-RHS resolver (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B). The
Python evaluator is the canonical source-of-truth for sizing-field RHS
resolution; the Rust delegation hook in `_try_rust_fn_eval_delegation`
is byte-equivalent for the in-subset AST shapes.

R78 C / R79 B Rust delegation surface preserved byte-exact — the
hasattr / ImportError guard pattern matches the pre-split version
verbatim. Tests pin the equivalence:
  * `tests/test_rust_fn_eval_expansion_v102.py`
  * `tests/test_rust_fn_multi_stmt_eval_v110.py`
"""

from __future__ import annotations

from ..ir import CellAlias, CellDef, FunctionDef, VerifyJob


def _lift_expr(node):
    """Lift a bare float / int into a `num` AST node.

    Any other shape (already an AST dict) is returned verbatim. Used
    at every transformer boundary that emits an expression value —
    keeps the eval-time walker free of `isinstance` checks on the
    leaf type.
    """
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return {"kind": "num", "value": float(node)}
    return node


def _simplify_body_for_ir(body):
    """Unwrap a single-`return EXPR;` body into a bare expression AST.

    Backward compatibility: prior to v0.0.37 Round 18, a function body
    was just an expression AST (the body of the only `return EXPR;`
    statement). The Round 18 extension introduces `let` / `if` / multi-
    statement bodies, which need a wrapping `{"kind": "body", ...}`
    dict to carry the statement list. When a body has the minimal
    shape (exactly one `return` statement, no preceding lets or
    branches), we strip the wrapper and store the inner expression
    directly so downstream consumers (and existing tests / tooling)
    see the same flat AST they did before.

    v0.1.16 R92 E: opt-in Rust delegation when `CLAY_USE_RUST_PARSER=1`
    is set. The Rust port lives in
    `crates/clay_parser/src/semantic_fn_eval.rs::simplify_body_for_ir`.
    On any failure (env var unset, extension missing, out-of-subset
    body shape) the Rust path returns None and Python falls through to
    the canonical local walk below — byte-equivalence preserved.
    """
    # v0.1.16 R92 E: Rust delegation hook. Returns the simplified body
    # dict, the original body dict, or None. On None, fall back to the
    # canonical Python walk below.
    rust_result = _try_rust_simplify_body_for_ir(body)
    if rust_result is not None:
        return rust_result
    if not isinstance(body, dict) or body.get("kind") != "body":
        return body
    stmts = body.get("stmts", [])
    if len(stmts) == 1 and isinstance(stmts[0], dict) and stmts[0].get("kind") == "return":
        return stmts[0]["expr"]
    return body


def _try_rust_simplify_body_for_ir(body):
    """v0.1.16 R92 E: opt-in Rust delegation for `_simplify_body_for_ir`.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt to run the unwrap decision through the Rust
    walker. Returns:
      * The unwrapped expression dict (single-`return` case).
      * The original body dict (no unwrap).
      * `None` — Rust path unavailable / out-of-subset shape; caller
        falls back to the canonical Python walk.

    The Rust path is BIT-EQUIVALENT to the local walk in terms of
    unwrap-vs-keep decision. The Python walk remains the canonical
    source-of-truth.
    """
    try:
        from .._rust_bridge import try_rust_simplify_body_for_ir
    except (ImportError, AttributeError):
        return None
    try:
        return try_rust_simplify_body_for_ir(body)
    except Exception:
        return None


def _bin_node(lhs, op: str, rhs):
    """Build a binary AST node, lifting leaf floats as needed."""
    return {
        "kind": "bin",
        "op": op,
        "lhs": _lift_expr(lhs),
        "rhs": _lift_expr(rhs),
    }


# v0.0.37: expression evaluator. Evaluates a small AST against a
# `{argname: float}` environment and a function-name → FunctionDef
# table. Raises ValueError with precise messages on:
#   * unknown function name
#   * unknown variable reference
#   * arity mismatch
#   * division by zero
#   * non-finite result
class _ExprEvalError(ValueError):
    """Raised when an expression cannot be evaluated."""


def _eval_expr(
    node,
    *,
    env: dict[str, float],
    functions: dict[str, FunctionDef],
    call_stack: tuple[str, ...] = (),
) -> float:
    """Walk a v0.0.37 expression AST and return its float value.

    `env`        — variable bindings for `{kind: ref}` lookups.
    `functions`  — function table for `{kind: call}` resolution.
    `call_stack` — chain of function names currently being evaluated,
                   used to detect recursion (forbidden in this MVP).
    """
    # Defensive: a bare float can land here if a caller forgot to
    # lift. Handle gracefully.
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node)
    if not isinstance(node, dict) or "kind" not in node:
        raise _ExprEvalError(
            f"internal: malformed expression node {node!r}"
        )
    kind = node["kind"]
    if kind == "num":
        return float(node["value"])
    if kind == "ref":
        name = node["name"]
        if name not in env:
            raise _ExprEvalError(
                f"unknown identifier {name!r} in expression. Known "
                f"variables: {sorted(env)}. (Bareword references in "
                f"sizing/spec expressions must be function arguments — "
                f"top-level sizing values cannot reference variables.)"
            )
        return float(env[name])
    if kind == "unary":
        op = node["op"]
        # `!` is short-circuit only in the sense that it just inverts
        # its operand's truthiness — there is no second operand to
        # short-circuit over. We DO need to evaluate the operand
        # before classifying it as truthy / falsy.
        v = _eval_expr(node["arg"], env=env, functions=functions, call_stack=call_stack)
        if op == "-":
            return -v
        if op == "!":
            # Truthy classification: anything non-zero (and finite) is
            # truthy; 0.0 (and -0.0) are falsy. Mirror Python's
            # `not <float>` semantics so the user can mix `!` with
            # logical-and / logical-or without surprise.
            return 0.0 if v else 1.0
        return v
    if kind == "bin":
        op = node["op"]
        # Short-circuit logical operators: evaluate LHS first, then
        # decide whether RHS evaluation is needed. The grammar lifts
        # `&&` / `||` above arithmetic so by the time we see a `bin`
        # node with op `&&` / `||` the operands are themselves any
        # valid expression — including a nested `bin` we'd want to
        # SKIP evaluating to avoid a division-by-zero in the unused
        # branch (e.g. `b != 0 && a / b > 1`).
        if op == "&&":
            lhs = _eval_expr(node["lhs"], env=env, functions=functions, call_stack=call_stack)
            if not lhs:
                return 0.0
            rhs = _eval_expr(node["rhs"], env=env, functions=functions, call_stack=call_stack)
            return 1.0 if rhs else 0.0
        if op == "||":
            lhs = _eval_expr(node["lhs"], env=env, functions=functions, call_stack=call_stack)
            if lhs:
                return 1.0
            rhs = _eval_expr(node["rhs"], env=env, functions=functions, call_stack=call_stack)
            return 1.0 if rhs else 0.0
        # All other binary ops evaluate both sides up front.
        lhs = _eval_expr(node["lhs"], env=env, functions=functions, call_stack=call_stack)
        rhs = _eval_expr(node["rhs"], env=env, functions=functions, call_stack=call_stack)
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            if rhs == 0.0:
                raise _ExprEvalError(
                    f"division by zero in expression (lhs={lhs!r})."
                )
            return lhs / rhs
        if op == "**":
            try:
                return lhs ** rhs
            except (OverflowError, ZeroDivisionError) as exc:
                raise _ExprEvalError(
                    f"`{lhs} ** {rhs}` failed: {exc}"
                ) from exc
        # Comparison operators — return 1.0 (true) / 0.0 (false) as
        # plain Python floats so downstream consumers (sizing values)
        # never see a `bool`.
        if op == "<":
            return 1.0 if lhs < rhs else 0.0
        if op == "<=":
            return 1.0 if lhs <= rhs else 0.0
        if op == ">":
            return 1.0 if lhs > rhs else 0.0
        if op == ">=":
            return 1.0 if lhs >= rhs else 0.0
        if op == "==":
            return 1.0 if lhs == rhs else 0.0
        if op == "!=":
            return 1.0 if lhs != rhs else 0.0
        raise _ExprEvalError(f"unknown binary operator {op!r}")
    if kind == "call":
        name = node["name"]
        if name not in functions:
            raise _ExprEvalError(
                f"unknown function {name!r}. Declared functions so far: "
                f"{sorted(functions)}. (Functions must be defined BEFORE "
                f"they are called in v0.0.37 — forward references are "
                f"not supported in this MVP.)"
            )
        if name in call_stack:
            raise _ExprEvalError(
                f"recursive call to {name!r} (call stack: "
                f"{' -> '.join(call_stack)} -> {name}). Recursive "
                f"functions are not supported in v0.0.37."
            )
        fn = functions[name]
        raw_args = node["args"]
        if len(raw_args) != len(fn.args):
            raise _ExprEvalError(
                f"function {name!r} called with {len(raw_args)} "
                f"argument(s), but expects {len(fn.args)} "
                f"({fn.args!r})."
            )
        # Evaluate arguments in caller's env, then bind to callee's
        # formal parameter names for the body evaluation.
        arg_values = [
            _eval_expr(a, env=env, functions=functions, call_stack=call_stack)
            for a in raw_args
        ]
        callee_env = dict(zip(fn.args, arg_values))
        sub_stack = call_stack + (name,)
        # v0.0.37 Round 18: function body may be a bare expression AST
        # (Round 17 single-return MVP, preserved for backward compat)
        # OR a `{"kind": "body", "stmts": [...]}` dict (multi-statement
        # body with `let` / `if`). Dispatch on the outer shape.
        body = fn.body_expr
        if isinstance(body, dict) and body.get("kind") == "body":
            return _eval_body(
                body, env=callee_env,
                functions=functions, call_stack=sub_stack,
            )
        return _eval_expr(
            body, env=callee_env,
            functions=functions, call_stack=sub_stack,
        )
    raise _ExprEvalError(f"unknown expression node kind {kind!r}")


# v0.0.37 Round 18: function-body (statement list) evaluator.
#
# A function body is a `{"kind": "body", "stmts": [...]}` dict whose
# stmts list contains `let`, `if`, and `return` entries in source
# order. We walk the list, threading let-bindings through subsequent
# statements, and stop at the first `return` reached (returning its
# expr value). An `if` whose condition is truthy descends into its
# `then` body; otherwise it descends into the `else` body (if any).
# A body that falls off the end without hitting a `return` is a
# parse-time error (caught by `_validate_fn_body_terminates`) — by
# eval time every reachable body is guaranteed to terminate.
#
# Scoping: `let` introduces a new binding in the current (and
# subsequent) statements. Each `if`/`else` body opens a new lexical
# scope so a let inside an if doesn't leak to siblings. We model this
# by passing a COPY of the env into each nested body so mutations
# (new let bindings) stay scoped.

class _ReturnSignal(Exception):
    """Internal control-flow signal carrying a return value upward.

    Not user-visible — caught by `_eval_body` to unwind nested
    `if` / `let` frames when a `return` fires inside them.
    """
    __slots__ = ("value",)

    def __init__(self, value: float):
        super().__init__()
        self.value = value


def _eval_body(
    body,
    *,
    env: dict[str, float],
    functions: dict[str, FunctionDef],
    call_stack: tuple[str, ...] = (),
) -> float:
    """Evaluate a function body and return the value of the executed `return`.

    The body must be a `{"kind": "body", "stmts": [...]}` dict
    produced by the `fn_body` transformer. Every reachable execution
    path must terminate in a `return` (validated at parse time by
    `_validate_fn_body_terminates`); falling off the end here raises
    an internal error.
    """
    if not isinstance(body, dict) or body.get("kind") != "body":
        raise _ExprEvalError(
            f"internal: expected fn_body dict, got {body!r}"
        )
    try:
        _exec_stmts(body["stmts"], env=dict(env),
                    functions=functions, call_stack=call_stack)
    except _ReturnSignal as ret:
        return ret.value
    # If we get here, the body didn't return — parse-time validation
    # should have caught this. Raise a clear internal error.
    raise _ExprEvalError(
        "internal: fn body completed without hitting a `return` "
        "statement — parse-time validation should have caught this."
    )


def _exec_stmts(
    stmts,
    *,
    env: dict[str, float],
    functions: dict[str, FunctionDef],
    call_stack: tuple[str, ...],
) -> None:
    """Execute a list of statements in `env` (mutating it for `let`).

    Raises `_ReturnSignal` when a `return` is reached; the caller
    (`_eval_body`) catches it. Nested `if` bodies recurse here with
    a copied env (new scope) so their lets don't leak.
    """
    for stmt in stmts:
        kind = stmt.get("kind") if isinstance(stmt, dict) else None
        if kind == "let":
            name = stmt["name"]
            if name in env:
                # Defensive: parse-time validation should have caught
                # same-scope re-declaration already. The cross-scope
                # case (shadowing a function argument) is rejected
                # by `_validate_fn_body_lets`, so this should be
                # unreachable.
                raise _ExprEvalError(
                    f"let {name!r}: name already bound in this scope "
                    f"({sorted(env)})."
                )
            value = _eval_expr(
                stmt["expr"], env=env,
                functions=functions, call_stack=call_stack,
            )
            env[name] = value
            continue
        if kind == "return":
            value = _eval_expr(
                stmt["expr"], env=env,
                functions=functions, call_stack=call_stack,
            )
            raise _ReturnSignal(value)
        if kind == "if":
            cond = _eval_expr(
                stmt["cond"], env=env,
                functions=functions, call_stack=call_stack,
            )
            if cond:
                _exec_stmts(
                    stmt["then"]["stmts"], env=dict(env),
                    functions=functions, call_stack=call_stack,
                )
                # If the then-branch returned, the _ReturnSignal
                # propagated past us. If it didn't, fall through to
                # the next sibling statement (which parse-time
                # validation guarantees is itself a returning path).
                continue
            else_body = stmt["else"]
            if else_body is not None:
                _exec_stmts(
                    else_body["stmts"], env=dict(env),
                    functions=functions, call_stack=call_stack,
                )
                continue
            # No else clause — fall through to the next sibling.
            continue
        raise _ExprEvalError(
            f"unknown statement kind {kind!r} in fn body."
        )


def _try_rust_fn_eval_delegation(
    raw,
    functions: dict[str, FunctionDef],
) -> "float | None":
    """v0.1.01 R78 C / v0.1.02 R79 B: opt-in Rust delegation for
    fn-expression evaluation.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt to evaluate the sizing-field RHS through the Rust
    semantic transformer (`semantic_fn_eval::eval_sizing_expr`).
    Returns a `float` on success or `None` on any failure (env var
    unset, extension missing, out-of-subset AST node, unknown function,
    arity mismatch, division by zero, recursion, non-finite). Python
    callers MUST treat `None` as "fall back to the canonical Python
    `_eval_expr` path for the canonical error message and full-subset
    coverage".

    The Rust path is BIT-EQUIVALENT to `_eval_expr` for the in-subset
    AST shapes: same numeric result for `num` / `ref` (with arg-bound
    env) / `unary` / `bin` (arith + cmp + logical with short-circuit
    semantics) / `call` (with arity check, recursion rejection,
    division-by-zero guard).

    v0.1.02 R79 B: multi-statement fn bodies (Round 18 `{kind: body,
    stmts: [...]}` with let / if / return) are NOW IN SCOPE on the Rust
    side as well — `coerce_fn_body` in `lib.rs` accepts both bare-expr
    and body shapes; `eval_body` / `exec_stmts` walk the statement list
    with the same scoping rules as Python `_eval_body` / `_exec_stmts`.
    A call into a multi-stmt fn evaluates entirely in Rust now;
    out-of-subset only triggers on malformed AST shapes.

    Tuple shape passed across the FFI: each fn becomes
    `(name, params, body_ast)` where `body_ast` is the same dict tree
    the Python parser built — Rust walks it via `coerce_fn_body` in
    `lib.rs` (which dispatches to either `coerce_expr_node` or
    `coerce_body_node` depending on the top-level `kind`).

    The hasattr check on `try_rust_eval_fn_expr` guards against a
    bridge module that pre-dates this delegation surface — older
    deployments missing the function silently fall back to Python.
    """
    try:
        from .._rust_bridge import try_rust_eval_fn_expr
    except (ImportError, AttributeError):
        return None
    # Pack the function table into the FFI-friendly tuple shape.
    # FunctionDef.body_expr may be a bare-expression AST (Round 17
    # shape — in scope) or a {"kind": "body", ...} dict (Round 18
    # shape — out of scope, dropped on the Rust side). We pass both
    # shapes across; the Rust coercer filters.
    try:
        fn_table = [
            (fn.name, list(fn.args), fn.body_expr)
            for fn in functions.values()
        ]
    except Exception:
        return None
    try:
        return try_rust_eval_fn_expr(raw, fn_table)
    except Exception:
        return None


def _try_rust_resolve_expr_for_sizing_delegation(
    raw,
    functions: dict[str, FunctionDef],
):
    """v0.1.18 R95 F: opt-in Rust delegation for one sizing-field RHS.

    Lifts the bare-numeric fast path + the eval walk + the non-finite
    guard into Rust so the Python coordinator becomes a thin "use this
    value or raise" layer. Returns:
      * `("ok", value)` — Rust resolved successfully. Caller uses the
        finite float verbatim.
      * `("non_finite", value)` — eval ran but produced NaN/±inf. The
        caller raises the canonical ValueError naming `field_name`.
      * `None` — Rust path disabled / extension missing / out-of-subset
        AST / in-subset eval error. The caller falls through to the
        Python `_eval_expr` walk for the canonical error message.

    The hasattr check on `try_rust_resolve_expr_for_sizing` guards
    against a bridge module that pre-dates this delegation surface —
    older deployments silently fall back.
    """
    try:
        from .._rust_bridge import try_rust_resolve_expr_for_sizing
    except (ImportError, AttributeError):
        return None
    try:
        fn_table = [
            (fn.name, list(fn.args), fn.body_expr)
            for fn in functions.values()
        ]
    except Exception:
        return None
    try:
        return try_rust_resolve_expr_for_sizing(raw, fn_table)
    except Exception:
        return None


def _resolve_expr_for_sizing(
    raw,
    *,
    field_name: str,
    functions: dict[str, FunctionDef],
) -> float:
    """Resolve an `override_field` RHS into a concrete float.

    The override_field transformer returns the RHS verbatim —
    either a bare float (legacy `Cc = 1.5pF;` shape) or an expression
    AST dict (new v0.0.37 shape). Either way we resolve to a float
    against the current function table.

    Top-level sizing values have NO variable environment — only the
    function table. A bare `CNAME` reference inside a sizing value
    is therefore an error (which `_eval_expr` raises with a clear
    message).

    v0.1.01 R78 C: when `CLAY_USE_RUST_PARSER=1` is set, the
    evaluation delegates to the Rust semantic transformer
    (`crates/clay_parser/src/semantic_fn_eval.rs::eval_sizing_expr`).
    The Rust output is byte-equivalent to the Python path on the in-
    subset AST shapes; any None return (out-of-subset AST, validation
    rejection, extension missing, env var unset) falls back to the
    Python `_eval_expr` walk for the canonical error message and
    out-of-subset coverage.

    v0.1.18 R95 F: the bare-numeric fast path + non-finite guard are
    now ALSO available as a single Rust FFI hop via
    `_try_rust_resolve_expr_for_sizing_delegation`. The R78 C surface
    is preserved as a defensive second-tier fallback (some test
    fixtures monkey-patch the Rust eval but not the resolve entry).
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    # v0.1.18 R95 F: combined resolve delegation (bare-numeric + eval +
    # non-finite guard in one FFI hop). On `("ok", v)` we return v
    # directly; on `("non_finite", v)` we raise the canonical error; on
    # None we fall through to the R78 C eval delegation + Python walk.
    rust_resolve = _try_rust_resolve_expr_for_sizing_delegation(raw, functions)
    if rust_resolve is not None:
        status = rust_resolve[0]
        if status == "ok":
            return rust_resolve[1]
        if status == "non_finite":
            value = rust_resolve[1]
            raise ValueError(
                f"sizing field {field_name!r}: expression evaluated to "
                f"non-finite value {value!r}."
            )
    # v0.1.01 R78 C: eval-only delegation (no fast-path, no error-shape
    # awareness). Returns None when the Rust path is disabled OR
    # rejects the input; falls through to Python in both cases. The
    # Python evaluator below is the canonical source-of-truth — Rust
    # only takes over when it can produce the same byte-for-byte float.
    rust_result = _try_rust_fn_eval_delegation(raw, functions)
    if rust_result is not None:
        # Re-apply the non-finite guard so the error path is identical
        # to the Python branch. The Rust evaluator already rejects
        # non-finite results internally (returns None), so this guard
        # is defensive in practice.
        import math as _math
        if not _math.isfinite(rust_result):
            raise ValueError(
                f"sizing field {field_name!r}: expression evaluated to "
                f"non-finite value {rust_result!r}."
            )
        return rust_result
    try:
        result = _eval_expr(raw, env={}, functions=functions)
    except _ExprEvalError as exc:
        raise ValueError(
            f"sizing field {field_name!r}: {exc}"
        ) from exc
    import math as _math
    if not _math.isfinite(result):
        raise ValueError(
            f"sizing field {field_name!r}: expression evaluated to "
            f"non-finite value {result!r}."
        )
    return result


def _try_rust_resolve_sizing_dict_delegation(
    sizing: dict[str, object],
    functions: dict[str, FunctionDef],
):
    """v0.1.18 R95 F: opt-in Rust delegation for an entire sizing dict.

    Walks the dict in one FFI hop. The function table is coerced once
    for the whole dict (vs once per key on the R78 C single-value
    path). Returns:
      * `("ok", resolved_dict)` — every key resolved.
      * `("non_finite", offending_key, offending_value)` — one key
        produced NaN/±inf; the caller raises the canonical ValueError.
      * `None` — Rust path disabled / out-of-subset / in-subset eval
        error somewhere. The caller falls back to the per-key Python
        walk so the canonical error message names the right field.
    """
    try:
        from .._rust_bridge import try_rust_resolve_sizing_dict
    except (ImportError, AttributeError):
        return None
    try:
        pairs = list(sizing.items())
    except Exception:
        return None
    try:
        fn_table = [
            (fn.name, list(fn.args), fn.body_expr)
            for fn in functions.values()
        ]
    except Exception:
        return None
    try:
        return try_rust_resolve_sizing_dict(pairs, fn_table)
    except Exception:
        return None


def _resolve_sizing_dict(
    sizing: dict[str, object],
    *,
    functions: dict[str, FunctionDef],
    context: str,
) -> dict[str, float]:
    """Resolve every value in a sizing-style dict to a concrete float.

    `context` is a short label used in error messages (e.g.
    `"cell MyAmp"` or `"verify MyCell"`) so the user can tell which
    sizing block contained the bad expression.

    The input dict's iteration order is preserved on the output.

    v0.1.18 R95 F: opt-in batch Rust delegation amortizes the function-
    table FFI coercion across the dict. When the Rust path produces
    `("ok", resolved_dict)` we return it verbatim; on `("non_finite",
    key, value)` we raise the canonical ValueError directly; on None
    we fall through to the per-key Python walk.
    """
    rust_batch = _try_rust_resolve_sizing_dict_delegation(sizing, functions)
    if rust_batch is not None:
        status = rust_batch[0]
        if status == "ok":
            return rust_batch[1]
        if status == "non_finite":
            _, key, value = rust_batch
            raise ValueError(
                f"{context}: sizing field {key!r}: expression evaluated to "
                f"non-finite value {value!r}."
            )
    out: dict[str, float] = {}
    for key, raw in sizing.items():
        try:
            out[key] = _resolve_expr_for_sizing(
                raw, field_name=key, functions=functions,
            )
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
    return out


def _resolve_cell_def_sizing(
    cd: CellDef,
    *,
    functions: dict[str, FunctionDef],
) -> None:
    """Finalize a CellDef's sizing_kwargs against the function table."""
    cd.sizing_kwargs = _resolve_sizing_dict(
        cd.sizing_kwargs,
        functions=functions,
        context=f"cell {cd.local_name!r}",
    )


def _resolve_verify_job_overrides(
    vj: VerifyJob,
    *,
    functions: dict[str, FunctionDef],
) -> None:
    """Finalize a VerifyJob's sizing_overrides against the function table."""
    vj.sizing_overrides = _resolve_sizing_dict(
        vj.sizing_overrides,
        functions=functions,
        context=f"verify {vj.cell_path!r}",
    )


def _resolve_cell_alias_overrides(
    ca: CellAlias,
    *,
    functions: dict[str, FunctionDef],
) -> None:
    """Finalize a CellAlias's sizing_overrides against the function table."""
    ca.sizing_overrides = _resolve_sizing_dict(
        ca.sizing_overrides,
        functions=functions,
        context=f"cell {ca.local_name!r}",
    )
