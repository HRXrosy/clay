"""
Expression-AST + fn_def transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds the
v0.0.37 expression-AST rules and Round-18 statement-list rules:
  * expr_add_bin / expr_mul_bin / expr_pow_bin / expr_unary_op /
    expr_paren / expr_ref / fn_call / fn_call_args / fn_arg_list
  * fn_def
  * fn_body / let_stmt / return_stmt / if_stmt / else_block / else_if
  * expr_cmp_bin / expr_and_bin / expr_or_bin / expr_not_op

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

from ..ir import FunctionDef
from .fn_eval import _bin_node, _lift_expr, _simplify_body_for_ir
from .fn_validators import _validate_fn_body_lets, _validate_fn_body_terminates


class _ExprMixin:
    """Expression-AST + fn-def + fn-body statement transformer methods."""

    # ─── v0.0.37 (user functions): expression AST + fn_def ──────────

    # Expression nodes are small dicts to keep IR-free and Pickleable.
    # Shapes:
    #   {"kind": "num",  "value": float}
    #   {"kind": "ref",  "name": str}                  # arg / variable ref
    #   {"kind": "bin",  "op": str, "lhs": node, "rhs": node}
    #   {"kind": "unary","op": str, "arg": node}
    #   {"kind": "call", "name": str, "args": [node, ...]}
    #
    # Lark's `?` prefix on `expr`, `expr_add`, `expr_mul`, `expr_unary`,
    # `expr_pow`, `expr_atom` means single-child Trees collapse —
    # `5` reaches us as the bare float, not wrapped in any expr-node.
    # Only the `_bin` / `_unary_op` / `expr_ref` / `fn_call` / `expr_paren`
    # alternatives invoke a transformer method.

    def expr_add_bin(self, items):
        return _bin_node(items[0], str(items[1]), items[2])

    def expr_mul_bin(self, items):
        return _bin_node(items[0], str(items[1]), items[2])

    def expr_pow_bin(self, items):
        return _bin_node(items[0], "**", items[2])

    def expr_unary_op(self, items):
        # items: [ADD_OP_token, operand]
        op = str(items[0])
        # `+x` is the identity — emit the operand verbatim so the AST
        # stays minimal.
        if op == "+":
            return items[1]
        return {"kind": "unary", "op": op, "arg": items[1]}

    def expr_paren(self, items):
        # The grammar rule is `"(" expr ")" -> expr_paren`; the single
        # child IS the expression. Re-emit it directly — parentheses
        # are syntactic, not semantic, beyond their grouping role
        # already encoded in the parse tree.
        return items[0]

    def expr_ref(self, items):
        # A bare CNAME inside an expression — interpreted as a variable
        # reference (function argument). The eval-time environment
        # resolves it.
        return {"kind": "ref", "name": str(items[0])}

    def fn_call(self, items):
        # items: [CNAME(name), fn_call_args?]
        name = str(items[0])
        args = items[1] if len(items) > 1 else []
        return {"kind": "call", "name": name, "args": list(args)}

    def fn_call_args(self, items):
        return list(items)

    def fn_arg_list(self, items):
        return [str(x) for x in items]

    def fn_def(self, items):
        # Grammar (v0.0.37 Round 18):
        #   fn_def: FN_KW CNAME "(" fn_arg_list? ")" "{" fn_body "}"
        # The FN_KW token is kept as a child so the surface keyword
        # text survives — we just discard it here.
        # `items`: [FN_KW_tok, CNAME, fn_arg_list?, fn_body_dict]
        # The fn_body transformer returns a {"kind": "body", "stmts": [...]}
        # dict.
        #
        # Backward-compat shape: for a single-`return EXPR;` body (the
        # Round 17 MVP shape) we unwrap and store `body_expr` AS the
        # expression AST (no `body` wrapper). The evaluator dispatches
        # on `kind` so both shapes work uniformly. Multi-statement
        # bodies (let / if / multi-return) store the full `body` dict.
        _fn_tok = items[0]   # noqa: F841 — kept for clarity
        name = str(items[1])
        # The grammar `fn_arg_list?` collapses to either a list (when
        # present, via `fn_arg_list` rule) or is absent entirely.
        if isinstance(items[2], list):
            args = items[2]
            body = items[3]
        else:
            args = []
            body = items[2]
        # Duplicate-arg check.
        seen_args: set[str] = set()
        for a in args:
            if a in seen_args:
                raise ValueError(
                    f"fn {name!r}: duplicate parameter name {a!r}."
                )
            seen_args.add(a)
        # Body-level validation: every execution path must terminate in
        # a `return` (an `if` without an `else` clause that doesn't have
        # a trailing top-level return is rejected at parse time).
        _validate_fn_body_terminates(body, fn_name=name)
        # Body-level validation: detect duplicate `let` declarations in
        # the same lexical scope. This is a parse-time error; nested
        # scopes (inside if/else blocks) get their own scope so shadowing
        # across branches is fine. Re-declaration in the SAME scope is
        # the only thing rejected here.
        _validate_fn_body_lets(body, fn_name=name, outer_names=set(args))
        # Unwrap the simple `return EXPR;` body shape so backward-compat
        # consumers (and the Round 17 test suite) see the same flat
        # expression AST they did before.
        stored_body = _simplify_body_for_ir(body)
        return ("fn_def", FunctionDef(
            name=name, args=args, body_expr=stored_body,
        ))

    # ─── v0.0.37 Round 18: fn body statement list ──────────────────

    def fn_body(self, items):
        """fn_body: fn_body_stmt+

        Returns a `{"kind": "body", "stmts": [...]}` dict. Each stmt
        is one of:
          * `{"kind": "let",    "name": str,  "expr": AST}`
          * `{"kind": "if",     "cond": AST,  "then": body, "else": body|None}`
          * `{"kind": "return", "expr": AST}`
        The `body` shape on `if`'s then/else mirrors the top-level
        `fn_body` shape so the evaluator can recurse uniformly.
        """
        return {"kind": "body", "stmts": list(items)}

    def let_stmt(self, items):
        # items: [LET_KW_tok, CNAME, expr]
        # The LET_KW token is consumed positionally.
        _let_tok = items[0]   # noqa: F841 — kept for clarity
        name = str(items[1])
        expr = _lift_expr(items[2])
        return {"kind": "let", "name": name, "expr": expr}

    def return_stmt(self, items):
        # items: [RETURN_KW_tok, expr]
        _ret_tok = items[0]   # noqa: F841
        expr = _lift_expr(items[1])
        return {"kind": "return", "expr": expr}

    def if_stmt(self, items):
        # items: [IF_KW_tok, cond_expr, then_body, else_clause?]
        # The else_clause may be absent, an `else_block` (body), or an
        # `else_if` (another if_stmt dict). We unify the two else
        # forms into either a body dict or a "synthetic body" wrapping
        # a single nested if_stmt — both are valid `fn_body` shapes.
        _if_tok = items[0]   # noqa: F841
        cond = _lift_expr(items[1])
        then_body = items[2]
        else_body = items[3] if len(items) > 3 else None
        return {
            "kind": "if",
            "cond": cond,
            "then": then_body,
            "else": else_body,
        }

    def else_block(self, items):
        # items: [ELSE_KW_tok, fn_body]
        _else_tok = items[0]   # noqa: F841
        return items[1]

    def else_if(self, items):
        # items: [ELSE_KW_tok, if_stmt_dict]
        # `else if (...) { ... }` is sugar for an else-body containing
        # a single nested if_stmt. We wrap it in a body shape so the
        # evaluator's recursive walker doesn't need a special case.
        _else_tok = items[0]   # noqa: F841
        inner_if = items[1]
        return {"kind": "body", "stmts": [inner_if]}

    # ─── v0.0.37 Round 18: comparison + logical operator AST emit ──

    def expr_cmp_bin(self, items):
        # items: [lhs, CMP_OP_token, rhs]
        return _bin_node(items[0], str(items[1]), items[2])

    def expr_and_bin(self, items):
        return _bin_node(items[0], "&&", items[2])

    def expr_or_bin(self, items):
        return _bin_node(items[0], "||", items[2])

    def expr_not_op(self, items):
        # items: [NOT_OP_token, operand]
        return {"kind": "unary", "op": "!", "arg": _lift_expr(items[1])}
