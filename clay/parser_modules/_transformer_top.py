"""
Top-level transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds the
methods that build the file-level IR shell + the small leaf-decl
methods (package_decl, pdk_decl, string, import_*, cell_alias_def,
verify_ref).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

import lark

from ..ir import (
    CellAlias, CellDef, ClayFile, FunctionDef, VerifyJob,
)
from .fn_eval import (
    _resolve_cell_alias_overrides,
    _resolve_cell_def_sizing,
    _resolve_verify_job_overrides,
)


class _TopMixin:
    """Top-level + simple-declaration transformer methods."""

    def start(self, items) -> ClayFile:
        """Build the ClayFile IR shell.

        v0.0.28: verify_ref resolution no longer happens here — it
        moved to `_resolve_verify_refs()`, called by
        `parse_clay_source` / `parse_clay_file` AFTER any file-path
        imports have been merged. This way `verify SharedCell;` can
        resolve to a cell brought in from another .clay file. The
        transformer stashes the raw verify_ref names and pending file
        imports on the ClayFile via private attributes; the late
        resolution step reads them back.

        v0.0.37 (user functions): we walk `items` in source order
        and maintain a function table. Function definitions
        (`fn NAME(...) { return EXPR; }`) accumulate into the table.
        When we see a cell_def / verify_block / cell_alias, we
        resolve any unresolved expression nodes in their sizing
        kwargs / overrides against the function table built up to
        THIS point. That enforces the "functions must be defined
        BEFORE first use" rule naturally — if a sizing expression
        calls a function that hasn't appeared yet, the lookup fails.
        Pure-literal sizing values (no fn_call inside) parse and
        resolve identically to pre-v0.0.37, preserving full
        backward compat.
        """
        package_name = None
        pdk_id = None
        imports: list[str] = []
        # v0.0.28: file-path imports collected separately from
        # Python-module imports. Each entry is (raw_path, alias|None).
        file_imports: list[tuple[str, str | None]] = []
        verify_jobs: list[VerifyJob] = []
        cell_aliases: list[CellAlias] = []
        cell_defs: list[CellDef] = []
        functions_in_order: list[FunctionDef] = []
        functions: dict[str, FunctionDef] = {}
        verify_refs: list[object] = []      # local name references + spans
        layout_intent_decls: list[object] = []
        flow_decls: list[object] = []
        quality_decls: list[object] = []
        policy_decls: list[object] = []
        for kind, value in items:
            if kind == "package":
                package_name = value
            elif kind == "pdk":
                pdk_id = value
            elif kind == "import":
                imports.append(value)
            elif kind == "import_file":
                file_imports.append(value)
            elif kind == "fn_def":
                # v0.0.37: register a file-scope function. Duplicate
                # names are rejected (no shadowing in MVP).
                fn = value
                if fn.name in functions:
                    raise ValueError(
                        f"fn {fn.name!r}: function with this name "
                        f"already declared earlier in the file. "
                        f"Duplicate names are not allowed."
                    )
                functions_in_order.append(fn)
                functions[fn.name] = fn
            elif kind == "verify":
                # v0.0.37: resolve any expression nodes in the
                # VerifyJob's sizing_overrides against the function
                # table built so far.
                _resolve_verify_job_overrides(value, functions=functions)
                verify_jobs.append(value)
            elif kind == "cell_alias":
                _resolve_cell_alias_overrides(value, functions=functions)
                cell_aliases.append(value)
            elif kind == "cell_def":
                _resolve_cell_def_sizing(value, functions=functions)
                cell_defs.append(value)
            elif kind == "verify_ref":
                verify_refs.append(value)
            elif kind == "layout_intent_decl":
                layout_intent_decls.append(value)
            elif kind == "flow_decl":
                flow_decls.append(value)
            elif kind == "quality_decl":
                quality_decls.append(value)
            elif kind == "policy_decl":
                policy_decls.append(value)
        # v2-lang: register file-level `policy NAME { ... }` blocks into the
        # ClayFile registry keyed by policy name. Policies are NOT per-cell;
        # a cell selects one via `flow { policy = NAME; }`. Duplicate names
        # are rejected (no shadowing).
        policies: dict[str, object] = {}
        for decl in policy_decls:
            pname = getattr(decl, "name", None)
            if pname in policies:
                raise ValueError(
                    f"policy {pname!r}: a policy with this name was already "
                    f"declared earlier in the file. Duplicate names are not allowed."
                )
            policies[pname] = getattr(decl, "decl", None)
        cf = ClayFile(
            package_name=package_name,
            pdk_id=pdk_id,
            imports=imports,
            verify_jobs=verify_jobs,
            cell_aliases=cell_aliases,
            cell_defs=cell_defs,
            functions=functions_in_order,
            policies=policies,
        )
        if layout_intent_decls:
            _attach_layout_intent_decls(cell_defs, layout_intent_decls)
        if flow_decls:
            _attach_flow_decls(cell_defs, flow_decls)
        if quality_decls:
            _attach_quality_decls(cell_defs, quality_decls)
        # Stash pending data on the ClayFile (off the public dataclass
        # surface so downstream consumers see the same IR shape they
        # always have). `_resolve_verify_refs` reads them back after
        # any file-imports have been merged.
        cf.__dict__["_pending_verify_refs"] = verify_refs
        cf.__dict__["_pending_file_imports"] = file_imports
        return cf

    def package_decl(self, items):
        return ("package", items[0])

    def pdk_decl(self, items):
        return ("pdk", items[0])

    def string(self, items):
        s = str(items[0])
        if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
            return s[1:-1]
        return s

    # v0.0.28: grammar split `import_decl` into two named alternatives
    # so file-path imports and Python-module imports can be
    # distinguished at the transformer level — they need different
    # downstream handling. Both forms accept an optional `as ALIAS`,
    # but file-imports use it as a namespace prefix while module
    # imports keep it ignored (matches pre-v0.0.28 behavior — the
    # `as` clause was parsed but unused for module imports too).
    def import_module_decl(self, items):
        # items: [dotted_name_str, optional CNAME alias]. The alias is
        # accepted by the grammar for surface-syntax parity with
        # file-imports but does not affect resolution today.
        return ("import", items[0])

    def import_file_decl(self, items):
        # items: [string_path, optional CNAME alias].
        raw_path = items[0]
        alias: str | None = str(items[1]) if len(items) > 1 else None
        return ("import_file", (raw_path, alias))

    def cell_alias_def(self, items):
        # items: [CNAME, qualified_name, spec_block, sizing_override_block?]
        local_name = str(items[0])
        parent_path = items[1]
        spec_obj = items[2]
        overrides = items[3] if len(items) > 3 else {}
        return ("cell_alias", CellAlias(
            local_name=local_name,
            parent_path=parent_path,
            spec=spec_obj,
            sizing_overrides=overrides,
        ))

    @lark.v_args(meta=True)
    def verify_ref(self, meta, items):
        return ("verify_ref", {
            "name": str(items[0]),
            "source_span": _source_span_from_meta(meta),
        })


def _attach_layout_intent_decls(
    cell_defs: list[CellDef],
    layout_intent_decls: list[object],
) -> None:
    cells_by_name = {cd.local_name: cd for cd in cell_defs}
    for decl in layout_intent_decls:
        cell_name = getattr(decl, "cell_name", None)
        cd = cells_by_name.get(cell_name)
        if cd is None:
            raise ValueError(
                f"layout_intent block targets unknown cell {cell_name!r}. "
                f"Cells defined in this file: {sorted(cells_by_name)}."
            )
        cd.layout_intents.extend(list(getattr(decl, "entries", []) or []))


def _attach_flow_decls(
    cell_defs: list[CellDef],
    flow_decls: list[object],
) -> None:
    cells_by_name = {cd.local_name: cd for cd in cell_defs}
    for decl in flow_decls:
        cell_name = getattr(decl, "cell_name", None)
        cd = cells_by_name.get(cell_name)
        if cd is None:
            raise ValueError(
                f"flow block targets unknown cell {cell_name!r}. "
                f"Cells defined in this file: {sorted(cells_by_name)}."
            )
        cd.flow_decl = getattr(decl, "decl", None)


def _attach_quality_decls(
    cell_defs: list[CellDef],
    quality_decls: list[object],
) -> None:
    cells_by_name = {cd.local_name: cd for cd in cell_defs}
    for decl in quality_decls:
        cell_name = getattr(decl, "cell_name", None)
        cd = cells_by_name.get(cell_name)
        if cd is None:
            raise ValueError(
                f"quality block targets unknown cell {cell_name!r}. "
                f"Cells defined in this file: {sorted(cells_by_name)}."
            )
        cd.quality_decl = getattr(decl, "decl", None)


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
