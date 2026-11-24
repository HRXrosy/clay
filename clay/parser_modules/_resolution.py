"""
Post-parse resolution helpers (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
These helpers run AFTER `_ClayTransformer` has built the IR shell and
AFTER any file-path imports have been merged.

  * `_attach_clay_file_to_cell_defs` — back-edge stash so
    `compile_cell_def` can resolve bare-name `instance` targets.
  * `_resolve_verify_refs` — resolves `verify NAME;` references
    against the parsed cell tables (CellAlias + CellDef).
"""

from __future__ import annotations

from ..ir import ClayFile, VerifyJob


def _attach_clay_file_to_cell_defs(cf: ClayFile) -> None:
    """v0.0.34 (PLL gap A): stash a back-reference from each CellDef to
    its enclosing ClayFile.

    This lets `compile_cell_def` resolve `instance NAME = BareName { ... };`
    by looking up `BareName` in the file's cell_defs / cell_aliases tables
    WITHOUT changing `compile_cell_def`'s public signature (callers like
    `clay.cli` don't need to pass the file explicitly). The stash is
    deliberately private — public IR consumers don't see this back-edge
    on the dataclass surface.

    Called AFTER `_merge_imported_cells` has folded any
    `import './lib.clay';` cell_defs / cell_aliases into `cf`, so the
    resolution scope of a bare-name `instance` includes both
    locally-defined AND imported cells.
    """
    for cd in cf.cell_defs:
        cd.__dict__["_clay_file"] = cf


def _resolve_verify_refs(cf: ClayFile) -> None:
    """Resolve `verify NAME;` references against the parsed cell tables.

    Moved out of `_ClayTransformer.start()` in v0.0.28 so the file-import
    layer can merge cells from imported .clay files before resolution
    runs (i.e. a `verify SharedCell;` can target a cell brought in
    from another file). The transformer stashes the raw verify_ref
    names on `cf._pending_verify_refs`; this function reads them back
    and appends one VerifyJob each.

    The sentinel `__celldef__::<local_name>` cell_path tells the CLI to
    route through `compile_cell_def` (instead of `module.resolve`).
    """
    pending: list[object] = cf.__dict__.pop("_pending_verify_refs", [])
    if not pending:
        return
    alias_by_name = {a.local_name: a for a in cf.cell_aliases}
    celldef_by_name = {c.local_name: c for c in cf.cell_defs}
    for item in pending:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            source_span = item.get("source_span")
        else:
            name = str(item)
            source_span = None
        if name in alias_by_name:
            a = alias_by_name[name]
            cf.verify_jobs.append(VerifyJob(
                cell_path=a.parent_path,
                spec=a.spec,
                sizing_overrides=dict(a.sizing_overrides),
                source_span=source_span,
            ))
        elif name in celldef_by_name:
            cd = celldef_by_name[name]
            vj = VerifyJob(
                cell_path=f"__celldef__::{cd.local_name}",
                spec=cd.spec,
                sizing_overrides=dict(cd.sizing_kwargs),
                source_span=source_span,
            )
            corners = cd.__dict__.get("_corners")
            if corners is not None:
                vj.corners = corners
            temps = cd.__dict__.get("_temps")
            if temps is not None:
                vj.temps = temps
            cf.verify_jobs.append(vj)
        else:
            raise ValueError(
                f"`verify {name};` references undeclared cell "
                f"(known: aliases={sorted(alias_by_name)}, "
                f"defs={sorted(celldef_by_name)})"
            )
