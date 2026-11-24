"""
Python facade for the Rust-backed Clay parser (Phase 1 strangler-fig).

v0.0.63 shipped the workspace scaffolding (crates/clay_parser, crates/clay_ir)
with an empty grammar — every call returned None and fell back to lark.

v0.0.64 Phase 1 wires up a real pest grammar covering Clay's minimal subset:

  * `package NAME ;`
  * `pdk 'STRING' ;`
  * `import dotted.name ;` / `import dotted.name as Alias ;`
  * `cell NAME(port, ...) { topology { ... } spec { ... } [sizing { ... }] }`
    with topology supporting `style`, `rationale`, and `instance` only
  * `verify CELL ;`
  * `// line comments`

Out-of-scope features (device_decl / cap_decl / res_decl / role_decl /
digital primitives / fn_def / corners / temps / bias_tree / verify_block
with body / multi-setup / spec assertion modifiers / sizing expressions /
file-import deep parsing) parse to an unsupported state in the Rust path;
`try_parse_rust` returns None and the caller falls back to lark.

The hard contract: existing 1874 pytest cases stay 100% green through the
transition. That means:

  1. With `CLAY_USE_RUST_PARSER` unset (default), this module is a no-op
     and Python lark is the only parser path used.
  2. With `CLAY_USE_RUST_PARSER=1`, calls go through `try_parse_rust`
     first; on `None` return (out-of-scope syntax or parse error) callers
     MUST fall back to lark transparently.
  3. The bridge import never fails; if `clay_parser` (Rust extension)
     isn't built, the bridge silently disables itself.

v0.0.64 returns an opaque `ParsedClayHandle` (not a ClayFile), exposing
just enough getters (`package_name`, `pdk_id`, `imports`, `cell_names`,
`verify_jobs`) for round-trip tests to validate structural equivalence
with lark. The handle does NOT replace ClayFile — `clay/parser.py`
continues to build the full ClayFile IR via lark. The Rust handle is
the v0.0.64 stepping stone before Phase 2 grows the hypergraph builder
into clay_ir.

The Rust extension is built via `maturin develop -m crates/clay_parser`
(installed manually for now; CI integration is a v0.0.66+ item).
"""

from __future__ import annotations

import os
from typing import Any


_HAS_RUST_EXTENSION = False
try:
    import clay_parser as _rust  # type: ignore[import-not-found]
    _HAS_RUST_EXTENSION = True
except ImportError:
    _rust = None


def rust_parser_enabled() -> bool:
    """True iff the user opted into the Rust parser AND the extension
    actually loaded. v0.0.64: this is the only gate; callers MUST also
    handle the case where `try_parse_rust` returns None (out-of-scope
    syntax or parser error)."""
    if not _HAS_RUST_EXTENSION:
        return False
    flag = os.environ.get("CLAY_USE_RUST_PARSER", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def try_parse_rust(text: str, source_path: str | None = None) -> Any:
    """Attempt to parse `text` through the Rust pest grammar.

    Returns an opaque `ParsedClayHandle` on success, or `None` if the
    Rust path is disabled, the input is out-of-scope for the v0.0.64
    minimal grammar, or any parser-level error occurs. Callers MUST
    treat `None` as "fall back to lark transparently".

    v0.0.68: the optional `source_path` argument lets the resolver
    flatten `import './lib.clay';` directives in the source against
    the importing file's directory. Without `source_path`, file-imports
    in the source cause a None return (preserved v0.0.64-v0.0.67
    contract — same behavior as `parse_clay_source` raising ValueError
    on file-imports without a file context).
    """
    if not rust_parser_enabled():
        return None
    try:
        if source_path is not None:
            return _rust.parse_clay_source_rust(text, source_path)
        return _rust.parse_clay_source_rust(text)
    except Exception:
        # Any unexpected pyo3 boundary error — fail closed so the
        # caller falls back to lark rather than surfacing a Rust
        # exception to user code.
        return None


def try_parse_file_rust(path: str) -> Any:
    """Attempt to parse `path` through the Rust pest grammar.

    Mirrors `try_parse_rust` but takes a file path instead of source
    text. The Rust resolver in `crates/clay_parser/src/import_resolve.rs`
    walks any `import './...';` directives, parses each imported file
    recursively, merges the cells / aliases / fns into the importing
    file's ParsedFile, and returns a single flattened ParsedClayHandle.

    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The file is missing.
      * Any parser-level error occurs (grammar reject, walker reject,
        cycle, allow-list violation, name collision, etc.). Callers
        MUST treat `None` as "fall back to lark transparently".

    v0.0.68: introduced.
    """
    if not rust_parser_enabled():
        return None
    try:
        return _rust.parse_clay_file_rust(path)
    except Exception:
        return None


def probe_msg() -> str | None:
    """Diagnostic helper for tests; returns the Rust probe string or None
    if the extension isn't loaded."""
    if not _HAS_RUST_EXTENSION:
        return None
    return _rust.rust_parser_probe("hello")


# ─── v0.0.65 Layer 1 IR bridge re-export ───────────────────────
#
# `_ir_bridge.cell_to_hypergraph(cell)` translates a Python
# `clay.ir.Cell` into a Rust `clay_ir.CellGraphHandle`. Re-exported
# here so callers have a single facade (`clay._rust_bridge`) for
# both the parser opt-in (`try_parse_rust`) and the IR construction
# step. See `clay/_ir_bridge.py` for the full contract; that module
# is import-safe even when `clay_ir` is not built (returns None on
# every call).

from ._ir_bridge import (  # noqa: E402  (re-export after module-level imports)
    cell_to_hypergraph,
    has_rust_ir,
    parsed_to_hypergraph,
)


# ─── v0.0.81 Phase 1.5 strangler: role_decl expansion ─────────
#
# Rust port of the v0.0.31 G3 role_decl → Device-primitive expansion
# (`clay/parser.py::_expand_roles_to_devices`). When opted-in via
# `CLAY_USE_RUST_PARSER=1`, callers can run the role expansion in Rust
# and get back a bit-equivalent device list. The Python lark path
# (`_ClayTransformer.role_decl` + `_expand_role_to_devices`) stays as
# the source-of-truth anchor; the Rust path is the strangler-fig
# slice that closes one of the three Python-owned semantic
# transformations on the paper-blocking path.


def try_rust_expand_role_decls(
    text: str,
    cell_name: str,
) -> Any:
    """Attempt to expand the named cell's role_decls through the Rust
    semantic transformer.

    Returns a Python `list[PyParsedDevice]` on success (each entry has
    `instance_name`, `device_class`, `pdk_variant`, `pins`, `sizing`
    getters — same shape as `ParsedClayHandle.cell_devices`).
    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The pest grammar rejects the source.
      * The named cell doesn't exist on the parsed handle.
      * The role-decl body is out-of-subset (unknown size key, missing
        sizing source, pair role on a 1-port cell with no explicit
        `gates`, etc.) — in this case the Rust path returns None so the
        caller falls back to the lark transformer for the canonical
        error message.

    Strangler-fig contract: callers MUST treat `None` as "fall back to
    lark and re-run the expansion via the Python path". The existing
    `parse_clay_source` continues to be the only path that runs at
    default settings — this helper is opt-in via the env var.
    """
    if not rust_parser_enabled():
        return None
    try:
        handle = _rust.parse_clay_source_rust(text)
    except Exception:
        return None
    if handle is None:
        return None
    try:
        return handle.expand_cell_role_decls(cell_name)
    except Exception:
        # The Rust side raised on a validation failure (unknown key,
        # pair role on 1-port cell, etc.). Fall back to lark.
        return None


# ─── v0.0.82 Phase 1.5 strangler: cell_alias expansion ─────────
#
# Rust port of the v0.0.17 cell_alias surface
# (`clay/parser.py::_ClayTransformer.cell_alias_def` plus
# `_resolve_cell_alias_overrides`). The v0.0.65 honest-scope note
# documents this as the ONE construct where the Rust surface
# intentionally diverged from `_lark_surface` — Rust treated cell_alias
# as a first-class cell_name in `cell_names()` while lark kept aliases
# in a separate `cell_aliases` list. v0.0.82 RECONCILES that divergence
# by adding a full `expand_cell_aliases()` PyO3 method that returns
# bit-equivalent `PyParsedCellAlias` records (local_name + parent_path
# + spec_fields + sizing_overrides). The v0.0.65 round-trip surface
# (alias-as-cell_name) is preserved for backward compat with the
# direct-assertion tests in `tests/test_rust_parser_round_trip.py`.


def try_rust_expand_cell_aliases(text: str) -> Any:
    """Attempt to expand every `cell ALIAS = pkg::Parent { ... };` in
    `text` through the Rust semantic transformer.

    Returns a Python `list[PyParsedCellAlias]` on success. Each entry
    exposes:
      * `local_name: str`            — the alias's local name
      * `parent_path: str`           — `'pkg.path::CellName'`
      * `spec_fields: list[PyParsedSpecField]`
            each field exposes `name`, `dimension`, `op`, `value`,
            `across_corners`, `margin_pct`
      * `sizing_overrides: dict[str, float]`

    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The pest grammar rejects the source.
      * At least one alias body is out-of-subset (expression-RHS
        sizing_override, duplicate spec modifier, malformed body) — the
        Rust expansion returns None so the caller falls back to lark
        for the canonical expansion (which evaluates expressions
        against the function table).

    Strangler-fig contract: callers MUST treat `None` as "fall back to
    lark and re-run the expansion via the Python path". The existing
    `parse_clay_source` continues to be the only path that runs at
    default settings — this helper is opt-in via the env var.
    """
    if not rust_parser_enabled():
        return None
    try:
        handle = _rust.parse_clay_source_rust(text)
    except Exception:
        return None
    if handle is None:
        return None
    try:
        return handle.expand_cell_aliases()
    except Exception:
        return None


# ─── v0.0.83 Phase 1.5 strangler: digital_decl expansion ───────
#
# Rust port of the v0.0.35 / v0.0.40 digital-primitive expansion
# (`clay/parser.py::_ClayTransformer._digital_decl` + the per-kind
# helpers `_expand_inv`, `_expand_nand2`, `_expand_dff`,
# `_expand_dff_async_clear`). v0.0.83 CLOSES Phase 1.5 — the last of
# three Python-owned semantic transformations migrated to Rust under
# `CLAY_USE_RUST_PARSER=1`:
#   v0.0.81: role_decls
#   v0.0.82: cell_alias
#   v0.0.83: digital_decls (this slice)
# After this lands, all three transformations live on Rust as
# bit-equivalent Python ports; the Python `_ClayTransformer` paths stay
# in place as the source-of-truth anchor for diff-testing.


def try_rust_expand_digital_decls(
    text: str,
    cell_name: str,
) -> Any:
    """Attempt to expand the named cell's digital_decls (inv / nand2 /
    dff / dff_async_clear) through the Rust semantic transformer.

    Returns a Python `list[PyParsedDevice]` on success (each entry has
    `instance_name`, `device_class`, `pdk_variant`, `pins`, `sizing`
    getters — same shape as `ParsedClayHandle.cell_devices`).
    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The pest grammar rejects the source.
      * The named cell doesn't exist on the parsed handle.
      * Any digital_decl body is out-of-subset (unknown pin label,
        missing required pin, unknown size key, non-integer M / nf) —
        in this case the Rust path raises a `ValueError`, which the
        bridge catches and returns None so the caller falls back to the
        lark transformer for the canonical error message.

    Strangler-fig contract: callers MUST treat `None` as "fall back to
    lark and re-run the expansion via the Python path". The existing
    `parse_clay_source` continues to be the only path that runs at
    default settings — this helper is opt-in via the env var.

    Expansion convention (mirrors Python verbatim):
      inv NAME             →  2 MOS (<NAME>_pmos, <NAME>_nmos)
      nand2 NAME           →  4 MOS (<NAME>_pmos_a, _pmos_b, _nmos_a, _nmos_b)
      dff NAME             → 18 MOS (clock-inv chain + master + slave latches)
      dff_async_clear NAME → 24 MOS (18 DFF + 2 clear-inv + 4 pull-down NMOS)
    """
    if not rust_parser_enabled():
        return None
    try:
        handle = _rust.parse_clay_source_rust(text)
    except Exception:
        return None
    if handle is None:
        return None
    try:
        return handle.expand_cell_digital_decls(cell_name)
    except Exception:
        # The Rust side raised on a validation failure (unknown pin
        # label, missing required pin, etc.). Fall back to lark.
        return None


# ─── v0.0.86 Phase 1.5 slice 4: functional_role expansion ──────
#
# Rust port of the v0.0.62 functional-role surface
# (`clay/parser.py::_ClayTransformer.functional_role_decl` + the per-
# attribute helpers `functional_attr_field`, `functional_attr_list`,
# `functional_attr_id_list`). Six valid keywords:
#   * `input_transconductor`
#   * `output_buffer`
#   * `feedforward_path`
#   * `dominant_pole_cap`
#   * `compensation_zero`
#   * `tail_current_source`
#
# Where `role_decl` (v0.0.81) and `digital_decl` (v0.0.83) expand to
# concrete `Device` IR entries, `functional_role` is a behavioral-noun
# layer with NO transistor lowering at parse time. The compiler /
# lowering layer downstream is responsible for translating each
# functional role into underlying structural roles or device
# primitives. The parser only captures the surface intent (a kind +
# name + attribute map).


def try_rust_expand_functional_roles(
    text: str,
    cell_name: str,
):
    """Attempt to expand the named cell's functional_role_decls
    (input_transconductor / output_buffer / feedforward_path /
    dominant_pole_cap / compensation_zero / tail_current_source) through
    the Rust semantic transformer.

    Returns a Python `list[PyParsedFunctionalRole]` on success. Each
    entry exposes:
      * `kind: str`          — one of the six valid keywords
      * `name: str`          — local handle (`Mdp`, `Cc`, `Pff1`)
      * `attributes: dict[str, float | str | list[str]]`
                              — attribute key → value, where the value
                                is a Python-native type collapsed from
                                the Rust `FunctionalRoleAttrValue` enum.
                                Bare CNAMEs and quoted strings both
                                collapse to `str`; numbers to `float`
                                (SI base units); parenthesized lists to
                                `list[str]`.

    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The pest grammar rejects the source.
      * The named cell doesn't exist on the parsed handle.
      * Any functional_role_decl body is out-of-subset (empty body,
        duplicate attribute key, unknown kind, duplicate role name) —
        in this case the Rust path raises a `ValueError`, which the
        bridge catches and returns None so the caller falls back to the
        lark transformer for the canonical error message.

    Strangler-fig contract: callers MUST treat `None` as "fall back to
    lark and re-run the expansion via the Python path". The existing
    `parse_clay_source` continues to be the only path that runs at
    default settings — this helper is opt-in via the env var.

    Surface form (from the v0.0.62 spec):

        cell DiffPair(inn, inp, out, vdd, vss) {
          topology {
            input_transconductor Mdp {
              gm_target = 2mA/V,
              intent = 'differential_drive',
              polarity = pmos
            }
          }
        }

    The Rust parser produces a list with one entry: kind =
    `'input_transconductor'`, name = `'Mdp'`, attributes = `{
    'gm_target': 2e-3, 'intent': 'differential_drive', 'polarity': 'pmos' }`.
    """
    if not rust_parser_enabled():
        return None
    try:
        handle = _rust.parse_clay_source_rust(text)
    except Exception:
        return None
    if handle is None:
        return None
    try:
        return handle.expand_cell_functional_roles(cell_name)
    except Exception:
        # The Rust side raised on a validation failure (empty body,
        # duplicate key, unknown kind, duplicate role name). Fall back
        # to lark for the canonical error message.
        return None


# ─── v0.0.92 Phase 1.5 slice 5: combinator_block expansion ─────
#
# Rust port of the v0.0.92 combinator-in-source surface
# (`clay/parser.py::_ClayTransformer.combinator_block` + per-shape
# helpers `combinator_expr`, `combinator_call`, `combinator_kwarg`,
# `combinator_chain_list`, `combinator_stage_ref`, `combinator_name`,
# plus the module-scope walker `_validate_combinator_stage_refs`).
#
# v0.0.92 introduces the `combinator { ... }` block as a cell body
# part. The block declares a stage-template list and a build
# expression composing the stages via the combinator algebra (`>>`
# cascade, `feedforward(...)`, `compensate(...)`, `mirror_match(...)`,
# `cascade_chain([...])`). The expression tree is captured at parse
# time and lowered at compile time by
# `clay.compiler._compile_combinator_cell`, which dispatches to the
# Python combinator API (`clay.stage_combinators.*`).


def try_rust_expand_combinator_blocks(
    text: str,
    cell_name: str,
):
    """Attempt to expand the named cell's combinator_blocks through the
    Rust semantic transformer.

    Returns a Python `list[PyParsedCombinatorBlock]` on success. Each
    entry exposes:
      * `name: str`          — block local handle
      * `stages: list[str]`  — declared stage template names
      * `build_expr: dict`   — AST node tree (kind + children) matching
                                the lark transformer's output exactly:
                                  * `{'kind': 'stage_ref', 'index': N}`
                                  * `{'kind': 'stage_name', 'name': 'foo'}`
                                  * `{'kind': 'cascade', 'left': AST, 'right': AST}`
                                  * `{'kind': 'feedforward', 'source': AST, 'sink': AST, 'kwargs': {...}}`
                                  * `{'kind': 'compensate', 'inner': AST, 'kwargs': {...}}`
                                  * `{'kind': 'mirror_match', 'left': AST, 'right': AST}`
                                  * `{'kind': 'cascade_chain', 'stages': [AST, ...]}`

    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The pest grammar rejects the source.
      * The named cell doesn't exist on the parsed handle.
      * Any combinator_block fails validation (empty stages, out-of-range
        stage_ref, empty cascade_chain) — in this case the Rust path
        raises a `ValueError`, which the bridge catches and returns None
        so the caller falls back to the lark transformer for the
        canonical error message.

    Strangler-fig contract: callers MUST treat `None` as "fall back to
    lark and re-run the expansion via the Python path".
    """
    if not rust_parser_enabled():
        return None
    try:
        handle = _rust.parse_clay_source_rust(text)
    except Exception:
        return None
    if handle is None:
        return None
    try:
        return handle.expand_cell_combinator_blocks(cell_name)
    except Exception:
        return None


# ─── v0.0.93 Phase 1.5 slice 6: corner + temp expansion ────────
#
# Rust port of the v0.0.21 / v0.0.24 corner + temperature sweep surface
# (`clay/parser.py::_ClayTransformer.corners_block`,
# `temps_block`, `corner_name`, `temp_value`) plus the verify-driver
# cartesian loop (`clay/cli.py::509-510`). The block-level validation
# rules:
#
#   * `corners { ... }`: non-empty + each name ∈ {tt, ss, ff, sf, fs}
#     + duplicates preserved verbatim.
#   * `temps { ... }`: non-empty + every value finite + duplicates
#     preserved verbatim.
#
# The cartesian expansion is the (corner, temp) product the CLI walks
# at verify time. Row-major: corner outer, temp inner. Defaults apply
# when a list is omitted (`corners → ["tt"]`, `temps → [27.0]`),
# matching `VerifyJob.corners` / `VerifyJob.temps` defaults in
# `clay/ir.py::290-295`. After this slice lands, six Python-owned
# semantic transformations live on Rust as bit-equivalent ports under
# `CLAY_USE_RUST_PARSER=1`:
#
#   v0.0.81: role_decls
#   v0.0.82: cell_alias
#   v0.0.83: digital_decls
#   v0.0.86: functional_roles
#   v0.0.92: combinator_blocks
#   v0.0.93: corner + temp expansion (this slice)
#
# The Python `_ClayTransformer` paths stay in place as the source-of-
# truth anchor for diff-testing; this helper is opt-in via the env var.


def try_rust_expand_corner_temp(
    text: str,
    cell_name: str,
):
    """Attempt to expand the named cell's `corners { ... }` x
    `temps { ... }` cartesian sweep through the Rust semantic
    transformer.

    Returns a Python `list[tuple[str, float]]` on success — each entry
    is a `(corner_name, temp_celsius)` point in row-major order
    (corners outer loop, temps inner loop). Matches the verify-driver
    loop in `clay/cli.py::509-510` byte-for-byte.

    When the named cell omits a `corners { ... }` block the default
    `["tt"]` is applied; when the cell omits a `temps { ... }` block
    the default `[27.0]` is applied. These defaults mirror
    `VerifyJob.corners` / `VerifyJob.temps` in `clay/ir.py::290-295`,
    so a cell with no explicit sweep declarations expands to a single
    `("tt", 27.0)` point.

    Returns `None` if:
      * The Rust path is disabled (env var unset OR extension not built).
      * The pest grammar rejects the source.
      * The named cell doesn't exist on the parsed handle.
      * Validation fails (unknown corner, non-finite temp) — the Rust
        path raises a `ValueError`, which the bridge catches and returns
        None so the caller falls back to the lark transformer for the
        canonical error message.

    Strangler-fig contract: callers MUST treat `None` as "fall back to
    lark and re-run the expansion via the Python path". The existing
    `parse_clay_source` continues to be the only path that runs at
    default settings — this helper is opt-in via the env var.

    Validation rules mirror Python verbatim:
      corners { ... }: non-empty, each ∈ {tt, ss, ff, sf, fs},
                       duplicates preserved
      temps   { ... }: non-empty, every value finite (rejects NaN /
                       +inf / -inf), duplicates preserved
    """
    if not rust_parser_enabled():
        return None
    try:
        handle = _rust.parse_clay_source_rust(text)
    except Exception:
        return None
    if handle is None:
        return None
    try:
        return handle.expand_cell_corner_temp(cell_name)
    except Exception:
        # The Rust side raised on a validation failure (unknown corner,
        # non-finite temp, empty list). Fall back to lark for the
        # canonical error message.
        return None


# ─── v0.0.99 R76 A: pair-only delegation entry point ──────────
#
# Direct delegation interface for the Python `_ClayTransformer.role_decl`
# → `_expand_pair` path. Unlike `try_rust_expand_role_decls` (which
# requires re-parsing the source through pest first), this function
# takes the role's fields + the enclosing cell's port context directly
# as Python args. That lets `clay/parser.py::_expand_pair` hand off to
# the Rust expander WITHOUT round-tripping the source — a real
# end-to-end delegation channel, not just an audit hook.
#
# Returns a `list[PyParsedDevice]` on success, `None` on:
#   * The Rust path is disabled (env var unset OR extension not built).
#   * The role kind isn't `pmos_pair` / `nmos_pair`.
#   * The role validation fails (unknown size key, missing sizing
#     source, etc.) — the Python expansion path raises its canonical
#     error.
#   * The pair-needs-2-ports condition triggers — Python raises.
#
# Strangler-fig contract: callers MUST treat `None` as "fall back to
# Python `_expand_pair` and use its output". The Python path is the
# source-of-truth either way; the Rust path only takes over when it
# can produce the same byte-for-byte output.


def try_rust_expand_pair_role_solo(
    name: str,
    kind: str,
    sizing: dict,
    pin_overrides: dict,
    pair_overrides: dict,
    port_names: list,
    port_roles: dict,
    cell_name: str,
) -> Any:
    """Attempt to expand a single pair role_decl through the Rust
    semantic transformer.

    Returns a `list[PyParsedDevice]` with exactly 2 entries on success
    (M1, M2). Each entry exposes `instance_name`, `device_class`,
    `pdk_variant`, `pins`, `sizing` getters — same shape as
    `ParsedClayHandle.expand_cell_role_decls`.

    Args mirror `RoleDecl` (the Python lark IR class) + the enclosing
    cell's port context:
      * `name`            — role local handle (e.g. `'dp'`)
      * `kind`            — `'pmos_pair'` or `'nmos_pair'`
      * `sizing`          — `dict[str, float]` (W, L, M, nf, gm_id)
      * `pin_overrides`   — `dict[str, str]` (single-net labels: `tail`)
      * `pair_overrides`  — `dict[str, tuple[str, str]]` (two-net
                             labels: `gates`, `drains`)
      * `port_names`      — cell's port list (`['inn', 'inp', ...]`)
      * `port_roles`      — `dict[str, str | None]` (port → role tag,
                             e.g. `'input'` or `None`)
      * `cell_name`       — enclosing cell name (for error messages)

    Returns `None` if:
      * The Rust path is disabled.
      * `kind` isn't a pair kind.
      * Validation fails (Rust returns None on Err so Python falls back
        for the canonical error message).

    The Python caller MUST treat None as "use the Python expansion".
    The Python path is the byte-for-byte equivalence anchor.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    try:
        return _rust.try_rust_expand_pair_role_solo(
            name,
            kind,
            sizing,
            pin_overrides,
            pair_overrides,
            port_names,
            port_roles,
            cell_name,
        )
    except Exception:
        # Any unexpected error → fall back to Python.
        return None


# ─── v0.1.20 R97 E: pair-gate default-picker delegation ────────────
#
# Standalone Rust delegation for the gate-pick heuristic embedded in
# `clay/parser_modules/role_expand.py::_pick_default_pair_gates`. The
# function selects (gate_a, gate_b) for a pair role's M1/M2 from the
# enclosing cell's port signature, using the same three-tier heuristic
# as the Python source-of-truth:
#   1. Two `: input` annotated ports in declaration order.
#   2. Two ports matching a differential-input convention name pair
#      (`inn`/`inp`, `in_n`/`in_p`, `in_minus`/`in_plus`, `vinn`/`vinp`,
#      `vin_n`/`vin_p`).
#   3. The first two ports in the cell's signature.
#
# Unlike the R76 A pair-expansion channel (which runs the full
# expansion + sizing + device synthesis), this entry point returns
# JUST the `(gate_a, gate_b)` tuple. The Python caller can plug the
# result back into its own expansion path without going through the
# full Rust pair-expansion route.
#
# Strangler-fig contract: returns `tuple[str, str]` on success or
# `None` on:
#   * The Rust path is disabled (env var unset OR extension missing).
#   * Cell has fewer than 2 candidate ports (Python raises the
#     canonical ValueError after the None return).
#   * Any unexpected Rust-side failure.
# The Python caller MUST treat `None` as "use the Python heuristic".


def try_rust_pick_default_pair_gates(
    port_names: list,
    port_roles: dict,
    cell_name: str,
    role_name: str,
) -> Any:
    """Attempt to pick (gate_a, gate_b) defaults via the Rust path.

    Args mirror the Python `_pick_default_pair_gates` signature:
      * `port_names`  — cell's port list (declaration order preserved)
      * `port_roles`  — `dict[str, str | None]` (port → role tag,
                         e.g. `'input'` or `None`)
      * `cell_name`   — enclosing cell name (for error attribution)
      * `role_name`   — pair role's local handle (e.g. `'dp'`)

    Returns `(gate_a, gate_b)` on success, `None` on:
      * Rust path disabled.
      * Cell has < 2 candidate ports (Python raises the canonical
        ValueError after the None return).
      * Any unexpected boundary error.

    The Python caller MUST treat None as "use the Python heuristic".
    The Python path is the byte-for-byte equivalence anchor.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    try:
        return _rust.try_rust_pick_default_pair_gates(
            list(port_names),
            dict(port_roles),
            cell_name,
            role_name,
        )
    except Exception:
        # Any unexpected error → fall back to Python.
        return None


# ─── v0.1.21 R98 E: standalone role-kind classification bridges ────
#
# Twins of `try_rust_pick_default_pair_gates` (R97 E) for the two
# remaining pure helpers in `semantic_role_decl.rs`:
#   * `_is_pmos_kind(kind)` → bool
#   * `_supply_and_body(kind)` → (str, str)
#
# Both have been used internally by every role-expansion entry point
# since R76 A but were not callable directly from Python. R98 E exposes
# them so the Python source-of-truth (`role_expand._is_pmos_kind` /
# `role_expand._supply_and_body`) can delegate to Rust directly.
#
# The Python path is the byte-for-byte equivalence anchor; the Rust
# path returns the SAME bool / tuple the Python heuristic would
# compute. Failures (env var unset, extension missing) return None,
# falling back to Python.


def try_rust_is_pmos_kind(kind: str) -> Any:
    """Attempt to classify `kind` as PMOS-family via the Rust path.

    Returns `True` / `False` on success, `None` on:
      * Rust path disabled.
      * Extension missing or missing R98 E binding.
      * Any unexpected boundary error.

    The Python caller MUST treat None as "use the Python predicate".
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_is_pmos_kind"):
        return None
    try:
        return _rust.try_rust_is_pmos_kind(kind)
    except Exception:
        return None


def try_rust_supply_and_body(kind: str) -> Any:
    """Attempt to compute (supply_rail, body_rail) for `kind` via Rust.

    Returns `('vdd', 'vdd')` for PMOS-family kinds, `('vss', 'vss')`
    for everything else; `None` on:
      * Rust path disabled.
      * Extension missing or missing R98 E binding.
      * Any unexpected boundary error.

    The Python caller MUST treat None as "use the Python heuristic".
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_supply_and_body"):
        return None
    try:
        return _rust.try_rust_supply_and_body(kind)
    except Exception:
        return None


# ─── v0.1.17 R94 F: mirror + cs solo delegation entry points ───
#
# Twins of `try_rust_expand_pair_role_solo` (R76 A) for the remaining
# four lean-role kinds (`pmos_mirror` / `nmos_mirror` / `pmos_cs` /
# `nmos_cs`). The R76 A channel only covered the pair kinds; mirror and
# cs continued to flow through `_expand_mirror` / `_expand_cs` even
# when `CLAY_USE_RUST_PARSER=1` was set. R94 F closes the gap with
# byte-equal Rust ports, each fronted by a thin `try_rust_*` Python
# bridge that mirrors the existing R76 A pattern.
#
# Each entry point:
#   * Returns `list[PyParsedDevice]` on success (2 entries for mirror,
#     1 entry for cs).
#   * Returns `None` when the Rust path is disabled, the kind doesn't
#     match, or validation fails (so the Python caller falls back to
#     `_expand_mirror` / `_expand_cs` for the canonical error message).
#   * Strangler-fig: the Python expansion stays the byte-for-byte
#     equivalence anchor. The Rust path takes over ONLY when its
#     output matches the Python implementation bit-for-bit.


def try_rust_expand_mirror_role_solo(
    name: str,
    kind: str,
    sizing: dict,
    pin_overrides: dict,
    pair_overrides: dict,
    cell_name: str,
) -> Any:
    """Attempt to expand a single mirror role_decl through the Rust
    semantic transformer.

    Returns a `list[PyParsedDevice]` with exactly 2 entries on success
    (`<NAME>_M1` diode-tied + `<NAME>_M2` mirror leg). Each entry
    exposes `instance_name`, `device_class`, `pdk_variant`, `pins`,
    `sizing` getters — same shape as the pair channel.

    Args mirror the `RoleDecl` (Python lark IR class) shape, minus the
    port context that mirror roles don't need:
      * `name`            — role local handle (e.g. `'cm'`)
      * `kind`            — `'pmos_mirror'` or `'nmos_mirror'`
      * `sizing`          — `dict[str, float]` (W, L, M, nf, gm_id)
      * `pin_overrides`   — `dict[str, str]` (single-net labels: `in`, `out`)
      * `pair_overrides`  — must be empty (no two-net labels valid on
                             mirror roles; non-empty triggers validation
                             rejection)
      * `cell_name`       — enclosing cell name (for error attribution)

    Returns `None` if:
      * The Rust path is disabled.
      * `kind` isn't a mirror kind.
      * Validation fails (Rust returns None on Err so Python falls back
        for the canonical error message).

    The Python caller MUST treat None as "use the Python expansion".
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_expand_mirror_role_solo"):
        # Older pyd shipped before the R94 F binding landed; fall back
        # to the Python expander.
        return None
    try:
        return _rust.try_rust_expand_mirror_role_solo(
            name,
            kind,
            sizing,
            pin_overrides,
            pair_overrides,
            cell_name,
        )
    except Exception:
        # Any unexpected error → fall back to Python.
        return None


def try_rust_expand_cs_role_solo(
    name: str,
    kind: str,
    sizing: dict,
    pin_overrides: dict,
    pair_overrides: dict,
    cell_name: str,
) -> Any:
    """Attempt to expand a single cs role_decl through the Rust
    semantic transformer.

    Returns a `list[PyParsedDevice]` with exactly 1 entry on success
    (`<NAME>_M1`, the common-source MOS). Each entry exposes the same
    getters as the mirror channel.

    Args mirror the `RoleDecl` shape, minus the port context that cs
    roles don't need:
      * `name`            — role local handle (e.g. `'stage2'`)
      * `kind`            — `'pmos_cs'` or `'nmos_cs'`
      * `sizing`          — `dict[str, float]` (W, L, M, nf, gm_id)
      * `pin_overrides`   — `dict[str, str]` (single-net labels:
                             `gate`, `drain`)
      * `pair_overrides`  — must be empty (no two-net labels valid on
                             cs roles)
      * `cell_name`       — enclosing cell name

    Returns `None` if:
      * The Rust path is disabled.
      * `kind` isn't a cs kind.
      * Validation fails (Rust returns None on Err).

    The Python caller MUST treat None as "use the Python expansion".
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_expand_cs_role_solo"):
        return None
    try:
        return _rust.try_rust_expand_cs_role_solo(
            name,
            kind,
            sizing,
            pin_overrides,
            pair_overrides,
            cell_name,
        )
    except Exception:
        return None


def try_rust_expand_role_decl_solo(
    name: str,
    kind: str,
    sizing: dict,
    pin_overrides: dict,
    pair_overrides: dict,
    port_names: list,
    port_roles: dict,
    cell_name: str,
) -> Any:
    """Attempt to expand a single role_decl of ANY lean-role kind
    through the Rust semantic transformer.

    Routes to the appropriate per-kind solo entry point based on
    `kind`. Useful for the Python `_expand_role_to_devices` dispatcher
    that doesn't already know which kind it's looking at.

    Args:
      * `name` / `kind` / `sizing` / `pin_overrides` / `pair_overrides`
        — same `RoleDecl` field shape as the per-kind entries.
      * `port_names` / `port_roles` — only consumed by the pair branch
        (mirror / cs ignore them). Mirror / cs callers can pass empty
        list / dict.
      * `cell_name` — enclosing cell name.

    Returns `None` if the Rust path is disabled, the kind isn't one
    of the six valid lean-role kinds, or validation fails. Python
    callers MUST treat None as "use the Python expansion".
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_expand_role_decl_solo"):
        return None
    try:
        return _rust.try_rust_expand_role_decl_solo(
            name,
            kind,
            sizing,
            pin_overrides,
            pair_overrides,
            port_names,
            port_roles,
            cell_name,
        )
    except Exception:
        return None


# ─── v0.1.00 R77 A: digital-decl solo delegation entry point ───
#
# Twin of `try_rust_expand_pair_role_solo` (R76 A) for digital_decls.
# Direct delegation interface for the Python `_ClayTransformer.
# _digital_decl` path. Unlike `try_rust_expand_digital_decls` (which
# requires re-parsing the source through pest first and operates on a
# whole cell), this function takes the decl's fields directly as
# Python args. That lets `clay/parser.py::_digital_decl` hand off to
# the Rust expander WITHOUT round-tripping the source — a real
# end-to-end delegation channel for the digital-primitive surface,
# matching the role_decl pair channel R76 A established.
#
# Returns a `list[PyParsedDevice]` on success, `None` on:
#   * The Rust path is disabled (env var unset OR extension not built).
#   * The decl kind isn't one of `inv` / `nand2` / `dff` /
#     `dff_async_clear`.
#   * Decl validation fails (unknown pin label, missing required pin,
#     unknown size key, non-integer M / nf) — the Python expansion
#     path raises its canonical error.
#
# Strangler-fig contract: callers MUST treat `None` as "fall back to
# Python `_digital_decl` and use its output". The Python path is the
# source-of-truth either way; the Rust path only takes over when it
# can produce the same byte-for-byte output. Auto-naming convention
# is identical: `<NAME>_pmos` / `<NAME>_nmos` for inv, `<NAME>_pmos_a`
# / `<NAME>_nmos_b` for nand2, `<NAME>_clk_inv1_p` / ... for dff,
# and the additional `<NAME>_mclr_*` pulldowns for dff_async_clear.


def try_rust_expand_digital_decl_solo(
    name: str,
    kind: str,
    pins: dict,
    sizing: dict,
) -> Any:
    """Attempt to expand a single digital_decl through the Rust
    semantic transformer.

    Returns a `list[PyParsedDevice]` on success with the per-kind
    device count:
      * `inv` → 2 entries (`<NAME>_pmos`, `<NAME>_nmos`)
      * `nand2` → 4 entries (PMOS pair in parallel, NMOS in series
        through a shared `<NAME>_mid` net)
      * `dff` → 18 entries (clock-inv chain + master + slave latches)
      * `dff_async_clear` → 24 entries (18 DFF + 2 clear-inv + 4
        pull-down NMOS array)

    Each entry exposes `instance_name`, `device_class`, `pdk_variant`,
    `pins`, `sizing` getters — same shape as
    `ParsedClayHandle.expand_cell_digital_decls`.

    Args mirror what the Python `_digital_decl` has already built at
    delegation time (no `DigitalDecl` Python dataclass — `_digital_decl`
    directly returns `list[Device]` after building its local `pins` and
    `sizing` dicts):
      * `name`     — decl instance handle (e.g. `'inv1'`, `'nand2_0'`)
      * `kind`     — `'inv'` / `'nand2'` / `'dff'` / `'dff_async_clear'`
      * `pins`     — `dict[str, str]` (per-kind required-pin map: `in`,
                      `out`, `vdd`, `vss` for inv; `a`, `b`, `out`,
                      `vdd`, `vss` for nand2; etc.)
      * `sizing`   — `dict[str, float]` (W, L, M, nf) — must already
                      have ALL three defaults filled in (W=1u, L=0.5u,
                      M=1.0) by the Python caller. The Rust expander
                      also fills defaults idempotently so passing a
                      sizing dict that's missing W / L / M is safe.

    Returns `None` if:
      * The Rust path is disabled.
      * `kind` isn't one of the four valid kinds.
      * Validation fails (Rust returns None on Err so Python falls back
        for the canonical error message).

    The Python caller MUST treat None as "use the Python expansion".
    The Python path is the byte-for-byte equivalence anchor.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    try:
        return _rust.try_rust_expand_digital_decl_solo(
            name,
            kind,
            pins,
            sizing,
        )
    except Exception:
        # Any unexpected error → fall back to Python.
        return None


# ─── v0.1.01 R78 C: fn-expression eval delegation entry point ───
#
# Third slice of the Round 78 G4 dual-IR work. R76 A shipped the
# `expand_pair_role_solo` channel; R77 A shipped
# `expand_digital_decl_solo`; R78 C closes the third Phase 1.5 fn-eval
# slice. Direct delegation interface for the Python
# `_resolve_expr_for_sizing` path: when a sizing-field RHS is an
# expression AST (the v0.0.37 user-function surface), Python attempts
# the Rust evaluator first via this bridge; on None it falls back to
# its own `_eval_expr` walk.
#
# Returns a Python `float` on success, `None` on:
#   * The Rust path is disabled (env var unset OR extension not built).
#   * The expression AST contains an out-of-subset node kind. The
#     v0.1.01 honest scope ports: `num`, `ref`, `unary` (`-`/`!`),
#     `bin` (full arith + cmp + logical, with short-circuit `&&`/`||`),
#     `call` (with arity, recursion, div-by-zero, non-finite guards).
#     Multi-statement fn bodies (`{kind: body, stmts: [...]}` with let
#     / if / return) are OUT OF SCOPE — the Rust coercer drops them
#     from the function table and returns None when referenced.
#   * Validation failure inside the Rust path (unknown fn, arity
#     mismatch, division by zero, non-finite, recursion). Python then
#     raises the canonical error message.
#
# Strangler-fig contract: callers MUST treat `None` as "fall back to
# Python `_eval_expr` and use its output". The Python path is the
# byte-for-byte equivalence anchor — Rust only takes over when it can
# produce the same float.


def try_rust_eval_fn_expr(
    expr_ast: Any,
    functions: list,
) -> "float | None":
    """Attempt to evaluate one expression-AST tree through the Rust
    semantic transformer.

    Args:
      * `expr_ast` — the dict-tree the Python parser built for one
        sizing-field RHS. Shapes covered: `{kind: num, value: float}`,
        `{kind: ref, name: str}`, `{kind: unary, op: str, arg: node}`,
        `{kind: bin, op: str, lhs: node, rhs: node}`, `{kind: call,
        name: str, args: [node, ...]}`. Bare int/float (a literal
        sizing value not yet lifted by `_lift_expr`) is also accepted.
      * `functions` — list of `(name, params, body_ast)` tuples
        representing the file's function table. Each function's body
        must itself be one of the in-subset shapes; multi-statement
        bodies (Round 18 `{kind: body, stmts: [...]}`) are silently
        dropped from the Rust function table so the eval returns
        `UnknownFn` on call and the Python path takes over.

    Returns a `float` on successful eval, `None` on:
      * The Rust path is disabled.
      * The expression AST contains an out-of-subset node kind.
      * Validation failure inside the Rust eval (unknown fn, arity
        mismatch, division by zero, non-finite, recursion).

    The Python caller MUST treat `None` as "use the Python `_eval_expr`
    path". The Python path is the canonical source-of-truth — Rust
    only takes over when it can produce the same byte-for-byte float.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_eval_fn_expr"):
        # Older bridge module pre-dating this delegation surface —
        # silently fall back to Python.
        return None
    try:
        return _rust.try_rust_eval_fn_expr(expr_ast, functions)
    except Exception:
        # Any unexpected error → fall back to Python.
        return None


# ─── v0.1.02 R79 B: fn-body validator delegation entry points ───
#
# R78 C delegated the fn-body EVALUATION for bare-expression bodies.
# R79 B extends that to multi-statement bodies AND lifts the parse-time
# validators to Rust. With this slice the FULL Round 18 user-function
# surface (let / if / return + termination + let-shadowing checks)
# lives on the Rust side, behind `CLAY_USE_RUST_PARSER=1`. The Python
# `_validate_fn_body_terminates` / `_validate_fn_body_lets` paths stay
# in place as the source-of-truth anchor — Rust returns a short status
# code (`"ok"` / `"unterminated"` / `"redecl"`) and Python raises the
# canonical ValueError with the human-readable message.
#
# Strangler-fig contract: callers MUST treat `None` as "fall back to
# Python and run the canonical validator". The two functions below
# return `None` on:
#   * The Rust path is disabled (env var unset OR extension not built).
#   * The body_ast shape is unrecognized (malformed dict).
#   * Older bridge module pre-dating these surfaces (hasattr guard).


def try_rust_validate_fn_body_terminates(
    body_ast,
    fn_name: str,
) -> "str | None":
    """Attempt to validate fn-body termination through the Rust parse-
    time walker.

    Returns:
      * `"ok"`           — every execution path terminates in a `return`
      * `"unterminated"` — at least one path falls off the end. Python
                            caller raises the canonical ValueError.
      * `None`           — Rust path unavailable / out-of-subset shape.
                            Python caller MUST fall back to its own
                            `_validate_fn_body_terminates` walk.

    The Rust path is BIT-EQUIVALENT to `_validate_fn_body_terminates`
    in terms of accept-vs-reject behavior for every valid body shape.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_validate_fn_body_terminates"):
        return None
    try:
        return _rust.try_rust_validate_fn_body_terminates(body_ast, fn_name)
    except Exception:
        return None


def try_rust_validate_fn_body_lets(
    body_ast,
    fn_name: str,
    outer_names: list,
) -> "tuple[str, str, list] | None":
    """Attempt to validate fn-body let shadowing through the Rust
    parse-time walker.

    Returns a tuple `(status, name, scope)` where:
      * `status == "ok"`     — no shadowing or same-scope redecls.
      * `status == "redecl"` — `name` shadows an identifier already
                                bound in the scope. `scope` is the list
                                of names that were bound when the
                                offending let was seen. Python caller
                                raises the canonical ValueError.
      * `None`               — Rust path unavailable / malformed body.
                                Python caller MUST fall back.

    The Rust path is BIT-EQUIVALENT to `_validate_fn_body_lets` in
    terms of accept-vs-reject behavior + which (name, scope) pair is
    flagged on rejection.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_validate_fn_body_lets"):
        return None
    try:
        return _rust.try_rust_validate_fn_body_lets(
            body_ast, fn_name, list(outer_names),
        )
    except Exception:
        return None


# ─── v0.1.16 R92 E: BL-004 closure — body simplify + observability ──
#
# Three new bridges expose the helpers ported under R92 E:
#   * `try_rust_simplify_body_for_ir` — single-`return EXPR;` unwrap.
#     Python `_simplify_body_for_ir` in `clay/parser_modules/fn_eval.py`
#     is the canonical source-of-truth; the Rust path is opt-in.
#   * `try_rust_body_stats` — aggregate (num_stmts, num_returns,
#     max_if_depth, num_lets) in one FFI hop.
#   * `try_rust_collect_body_idents` — sorted (referenced_idents,
#     declared_let_names).
#
# All three returns are observability/transformation surfaces; they do
# NOT alter the byte-exact eval path. The Python `_simplify_body_for_ir`
# in `fn_eval.py` calls the bridge optionally; on `None` it falls back
# to its own walk.


def try_rust_simplify_body_for_ir(body_ast):
    """Attempt to simplify a fn-body AST through the Rust port of
    `_simplify_body_for_ir`.

    Returns:
      * `expr_dict`  — when the body was a single `return EXPR;`,
                        returns the unwrapped expression dict.
      * `body_dict`  — original input returned verbatim when the body
                        was multi-stmt / had branches.
      * `None`       — Rust path unavailable; Python caller MUST fall
                        back to its own `_simplify_body_for_ir`.

    The Rust path is BIT-EQUIVALENT to Python `_simplify_body_for_ir`
    in terms of unwrap-vs-keep decision.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_simplify_body_for_ir"):
        return None
    try:
        return _rust.try_rust_simplify_body_for_ir(body_ast)
    except Exception:
        return None


def try_rust_body_stats(body_ast):
    """Attempt to compute aggregate body statistics through the Rust
    walker.

    Returns a 4-tuple `(num_stmts, num_returns_total, max_if_depth,
    num_lets_total)` on success, or `None` if the Rust path is
    unavailable or the body shape is out-of-subset.

    Pure-observability surface — Python tooling that needs structural
    fingerprints of a fn body without running the evaluator can call
    this for a one-hop summary.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_body_stats"):
        return None
    try:
        return _rust.try_rust_body_stats(body_ast)
    except Exception:
        return None


def try_rust_collect_body_idents(body_ast):
    """Attempt to collect referenced + declared identifiers from a fn
    body through the Rust walker.

    Returns a tuple `(referenced_idents, declared_let_names)` where
    each is a sorted list of strings. Returns `None` if the Rust path
    is unavailable or the body shape is out-of-subset.

    Useful for free-variable analysis and dead-let detection at parse
    time — caller can diff the two lists to find unused let bindings.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_collect_body_idents"):
        return None
    try:
        return _rust.try_rust_collect_body_idents(body_ast)
    except Exception:
        return None


# ─── v0.1.18 R95 F: sizing-RHS resolve + batch dict resolve ──────
#
# Two bridge wrappers exposing the Rust port of `_resolve_expr_for_sizing`
# and `_resolve_sizing_dict`. The Python coordinators in
# `clay/parser_modules/fn_eval.py` call these BEFORE walking their own
# `_eval_expr`; on `None` they fall back to the canonical Python path.
#
# Returns from the Rust path are pre-shaped so the Python caller is a
# thin "raise or return" layer:
#   * `try_rust_resolve_expr_for_sizing` returns
#     `("ok", value)` | `("non_finite", value)` | `None`
#   * `try_rust_resolve_sizing_dict` returns
#     `("ok", {k: v, ...})` | `("non_finite", key, value)` | `None`


def try_rust_resolve_expr_for_sizing(raw, functions):
    """Attempt to resolve one sizing-field RHS through the Rust port of
    `_resolve_expr_for_sizing`.

    Args:
      * `raw` — either a bare int/float (legacy `Cc = 1.5pF;` shape) or
        an expression AST dict (post-`_lift_expr` shape).
      * `functions` — list of `(name, params, body_ast)` tuples
        representing the file's function table (same shape used by
        `try_rust_eval_fn_expr`).

    Returns:
      * `("ok", value)` — resolved successfully; `value` is the finite
        float the caller should use directly.
      * `("non_finite", value)` — eval produced NaN/±inf; Python caller
        MUST raise the canonical ValueError naming the field.
      * `None` — Rust path unavailable / out-of-subset / in-subset
        eval error. Python falls back to `_eval_expr` for the canonical
        error message.

    The Rust path is byte-equivalent to the Python `_eval_expr` path on
    the in-subset AST shapes (pinned by `test_rust_fn_eval_expansion`
    and the R95 F byte-equality tests).
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_resolve_expr_for_sizing"):
        return None
    try:
        result = _rust.try_rust_resolve_expr_for_sizing(raw, functions)
    except Exception:
        return None
    if result is None:
        return None
    success, value, error_kind = result
    if success:
        return ("ok", value)
    if error_kind == "non_finite":
        return ("non_finite", value)
    # Any other error_kind is treated as fall-back. Defensive — the Rust
    # path only emits "ok" / "non_finite" today, but keep the door open.
    return None


def try_rust_resolve_sizing_dict(pairs, functions):
    """Attempt to batch-resolve an entire sizing dict through the Rust
    port of `_resolve_sizing_dict`.

    Args:
      * `pairs` — list of `(key, raw)` tuples in source iteration order.
        Each `raw` is the same shape `try_rust_resolve_expr_for_sizing`
        accepts.
      * `functions` — list of `(name, params, body_ast)` tuples.

    Returns:
      * `("ok", resolved_dict)` — every key resolved; the dict preserves
        the input iteration order.
      * `("non_finite", offending_key, offending_value)` — one key
        produced NaN/±inf; Python caller raises the canonical ValueError
        naming that field.
      * `None` — Rust path unavailable / out-of-subset / in-subset eval
        error somewhere. Python caller MUST fall back to its own
        `_resolve_sizing_dict` walk (which produces the canonical error
        message via `_eval_expr`).

    Amortizes the function-table FFI coercion across the dict — for an
    N-key sizing block on a file with M fns, this is O(M) vs O(N × M)
    when each key delegates separately through `try_rust_eval_fn_expr`.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_resolve_sizing_dict"):
        return None
    try:
        result = _rust.try_rust_resolve_sizing_dict(pairs, functions)
    except Exception:
        return None
    if result is None:
        return None
    status, resolved_pairs, offending_key, offending_value = result
    if status == "ok":
        # Rebuild a Python dict in the order Rust returned it. The
        # bridge contract is "iteration-order preserved", same as the
        # Python `_resolve_sizing_dict` walk.
        return ("ok", dict(resolved_pairs))
    if status == "non_finite":
        return ("non_finite", offending_key, offending_value)
    # Any other status routes to fall-back. Defensive.
    return None


# ─── v0.2.73 R159 D: bias_tree { ... } block validation ──
#
# Rust port of the validation half of
# `clay/parser_modules/_transformer_misc.py::_MiscMixin.bias_tree_block`.
# The pure walker lives in `crates/clay_parser/src/semantic_bias_tree.rs`;
# this bridge marshals the already-parsed `BiasNode` list across the FFI
# boundary and returns a structured status tuple. The Python caller
# raises the canonical ValueError on any non-"ok" status so the surfaced
# message stays byte-exact with lark. On `None` (Rust path unavailable)
# the caller MUST fall back to its own Python validation loop.

def try_rust_validate_bias_tree(
    nodes,
) -> "tuple | None":
    """Attempt to validate a `bias_tree { ... }` block through the Rust
    parse-time walker.

    `nodes` is the ordered list of parsed `BiasNode` (or any object with
    `.name` / `.kind` / `.src` / `.value` / `.by_M` attributes). Returns
    a structured tuple `(status, node, src, declared, value, by_m)`:

      * `status == "ok"`                      — block is well-formed.
      * `status == "empty"`                   — zero nodes.
      * `status == "duplicate"`               — `node` name collides.
      * `status == "forward"`                 — `node.src` (== `src`) is
                                                declared LATER.
      * `status == "unknown"`                 — `node.src` (== `src`) is
                                                never declared; `declared`
                                                is the sorted list of
                                                names seen so far.
      * `status == "self"`                    — `node` references itself
                                                (unreachable on the linear
                                                path; kept for fidelity).
      * `status == "nonpositive_multiplier"`  — mirror `node` has a None /
                                                non-positive `by_m`.
      * `status == "nonpositive_value"`       — external `node` has a
                                                None / non-positive
                                                `value`.
      * `None`                                — Rust path unavailable;
                                                Python caller MUST fall
                                                back.

    The Rust path is BIT-EQUIVALENT to `bias_tree_block` in accept-vs-
    reject behavior AND in WHICH rule fires first for every block shape.
    """
    if not rust_parser_enabled():
        return None
    if _rust is None:
        return None
    if not hasattr(_rust, "try_rust_validate_bias_tree"):
        return None
    # Marshal the BiasNode list into plain tuples the FFI accepts. Any
    # node whose fields don't coerce cleanly (e.g. a non-int by_M) bails
    # the whole batch back to Python verbatim.
    try:
        payload = []
        for n in nodes:
            by_m = getattr(n, "by_M", None)
            by_m_int = None if by_m is None else int(by_m)
            value = getattr(n, "value", None)
            value_f = None if value is None else float(value)
            payload.append(
                (
                    str(n.name),
                    str(n.kind),
                    None if getattr(n, "src", None) is None else str(n.src),
                    value_f,
                    by_m_int,
                )
            )
    except Exception:
        return None
    try:
        return _rust.try_rust_validate_bias_tree(payload)
    except Exception:
        return None


__all__ = [
    "cell_to_hypergraph",
    "has_rust_ir",
    "parsed_to_hypergraph",
    "probe_msg",
    "rust_parser_enabled",
    "try_parse_file_rust",
    "try_parse_rust",
    "try_rust_body_stats",
    "try_rust_collect_body_idents",
    "try_rust_eval_fn_expr",
    "try_rust_expand_cell_aliases",
    "try_rust_expand_combinator_blocks",
    "try_rust_expand_corner_temp",
    "try_rust_expand_cs_role_solo",
    "try_rust_expand_digital_decl_solo",
    "try_rust_expand_digital_decls",
    "try_rust_expand_functional_roles",
    "try_rust_expand_mirror_role_solo",
    "try_rust_expand_pair_role_solo",
    "try_rust_expand_role_decl_solo",
    "try_rust_expand_role_decls",
    "try_rust_pick_default_pair_gates",
    "try_rust_resolve_expr_for_sizing",
    "try_rust_resolve_sizing_dict",
    "try_rust_simplify_body_for_ir",
    "try_rust_validate_bias_tree",
    "try_rust_validate_fn_body_lets",
    "try_rust_validate_fn_body_terminates",
]
