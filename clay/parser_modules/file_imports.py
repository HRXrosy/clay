"""
File-level .clay import resolution (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).

`import './lib.clay';` and `import '../shared/util.clay' as ns;`
pull cell_alias + cell_def declarations from another .clay file
into the importing file. Implemented as a recursive parse step in
`parse_clay_file`, with cycle detection via an "in-flight" stack.

Path resolution is RELATIVE TO THE IMPORTING FILE (not CWD), and
walls off escapes via an allow-list:
  * Only `./` / `../` relative paths accepted at the surface
  * Absolute paths (`/etc/passwd`, `C:\\...`) rejected
  * After resolution, the target must lie INSIDE the importing
    file's repo root (first ancestor with a `.git/` dir) — `../`
    segments that escape upward are blocked

Cells are merged into the importing file's IR. With `as <ns>`,
every cell from the imported file is renamed `ns::CellName`. With
no alias, cells appear under their local names — collisions are
rejected (we never silently shadow).
"""

from __future__ import annotations

from pathlib import Path

from ..ir import CellAlias, CellDef, ClayFile


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the first ancestor containing `.git/`.

    Returns that directory. If no `.git/` is ever found (e.g. a
    standalone tree), returns the topmost directory walked — this
    keeps the allow-list defensive without breaking on non-git trees.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    seen_root = cur
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:        # filesystem root
            return seen_root
        cur = cur.parent


def _resolve_file_import(
    raw_path: str,
    importing_file: Path,
) -> Path:
    """Resolve a `import 'PATH';` directive to an absolute file path.

    Validation:
      * `raw_path` must start with `./` or `../` (POSIX-style)
      * absolute paths rejected
      * resolved target must stay within the importing file's repo
        root (computed by walking up to the first `.git/` dir)

    Returns the absolute resolved Path. Raises ValueError with a
    precise message on any violation.
    """
    if not raw_path:
        raise ValueError(
            "file import: empty path string. Expected "
            "`import './foo.clay';` or `import '../bar.clay';`."
        )
    # Normalize separator for cross-platform pattern check.
    norm = raw_path.replace("\\", "/")
    # Allow-list: must be a clear relative path. Absolute paths
    # (POSIX `/etc/passwd`, Windows `C:\...`) and bare names like
    # `foo.clay` are rejected. The `./` or `../` prefix forces the
    # author to acknowledge they're reaching out of the current file.
    if not (norm.startswith("./") or norm.startswith("../")):
        raise ValueError(
            f"file import: path {raw_path!r} must start with `./` "
            f"or `../`. Absolute paths and bare filenames are "
            f"rejected by the allow-list."
        )
    importing_dir = importing_file.resolve().parent
    target = (importing_dir / raw_path).resolve()
    # Repo-root guard: target must live under the same repo root as
    # the importing file. Prevents `../../../../etc/passwd` escapes.
    repo_root = _find_repo_root(importing_file)
    try:
        target.relative_to(repo_root)
    except ValueError:
        raise ValueError(
            f"file import: resolved path {target!s} escapes the "
            f"repository root {repo_root!s}. Cross-repo imports "
            f"are blocked by the allow-list."
        )
    return target


def _merge_imported_cells(
    importing: ClayFile,
    imported: ClayFile,
    namespace: str | None,
    importing_path: Path,
    imported_path: Path,
) -> None:
    """Merge `imported`'s cell_aliases + cell_defs into `importing`.

    With `namespace=None`, each imported cell appears under its local
    name. Name collisions with cells declared in the importing file
    (or with cells brought in from previous imports) raise ValueError
    — we never silently shadow.

    With `namespace="foo"`, imported cells are renamed `foo::CellName`
    in the importing file's IR. Verify-refs in the importing file
    look these up by the namespaced name.

    Imported cells' VerifyJobs are NOT merged — running another file
    means running ITS verify jobs only. Other files just contribute
    library symbols.
    """
    # Build a "what does the importing file already know" set so we
    # can detect collisions.
    existing_alias = {a.local_name for a in importing.cell_aliases}
    existing_def = {c.local_name for c in importing.cell_defs}
    existing = existing_alias | existing_def

    for a in imported.cell_aliases:
        renamed = f"{namespace}::{a.local_name}" if namespace else a.local_name
        if renamed in existing:
            raise ValueError(
                f"file import: cell {renamed!r} from {imported_path!s} "
                f"collides with a cell declared in {importing_path!s}. "
                f"Use `import '...' as <ns>;` to namespace the import, "
                f"or rename one of the cells."
            )
        existing.add(renamed)
        importing.cell_aliases.append(CellAlias(
            local_name=renamed,
            parent_path=a.parent_path,
            spec=a.spec,
            sizing_overrides=dict(a.sizing_overrides),
        ))

    for c in imported.cell_defs:
        renamed = f"{namespace}::{c.local_name}" if namespace else c.local_name
        if renamed in existing:
            raise ValueError(
                f"file import: cell {renamed!r} from {imported_path!s} "
                f"collides with a cell declared in {importing_path!s}. "
                f"Use `import '...' as <ns>;` to namespace the import, "
                f"or rename one of the cells."
            )
        existing.add(renamed)
        # Build a renamed copy. CellDef is mutable; we copy mutable
        # subfields so future mutations of the source file's CellDef
        # don't leak through. The `devices` / `instances` / `role_decls`
        # / `functional_roles` axes are included so file-imported
        # primitive / composed / role-declared cells survive the
        # namespace rename intact.
        new_cd = CellDef(
            local_name=renamed,
            ports=list(c.ports),
            topology=dict(c.topology),
            spec=c.spec,
            sizing_kwargs=dict(c.sizing_kwargs),
            port_roles=dict(c.port_roles),
            bias_tree=list(c.bias_tree),
            instances=list(c.instances),
            devices=list(c.devices),
            role_decls=list(c.role_decls),
            functional_roles=list(c.functional_roles),
            layout_intents=list(c.layout_intents),
        )
        # Propagate any private stash attributes from the source
        # CellDef (e.g. `_corners`, `_temps`, `_instances`,
        # `_devices` from parallel agents' work). They become part
        # of the merged cell's behavior, the same as if it had been
        # declared locally.
        for stash_key, stash_val in c.__dict__.items():
            if stash_key.startswith("_") and stash_key not in new_cd.__dict__:
                new_cd.__dict__[stash_key] = stash_val
        importing.cell_defs.append(new_cd)
