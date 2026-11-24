"""
clay.parser_modules — split implementation of the Clay parser (v0.1.13).

The parser slice was split out of the legacy single-file
`clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B). The public API
is byte-exact preserved: `from clay.parser import *` and every
`from clay.parser import <name>` path keeps working through the shim
in `clay/parser.py`, which re-exports this package.

Split layout (each file < 1000 LOC):
  * `_constants.py`           — pin / role / digital / functional /
                                corner / drive / unit tables and
                                small helpers (`_extract_derived_refs`,
                                `_resolve_unit_multiplier`,
                                `_validate_combinator_stage_refs`).
  * `_grammar.py`             — grammar bootstrap (`GRAMMAR_FILE`,
                                `_PARSER`, `_get_parser`).
  * `fn_eval.py`              — v0.0.37 function-expression
                                evaluator + sizing-RHS resolver +
                                R78 C / R79 B Rust delegation.
  * `fn_validators.py`        — parse-time fn-body validators
                                (termination + let-shadow checks)
                                + R79 B Rust delegation.
  * `role_expand.py`          — `_expand_role_to_devices` + pair /
                                mirror / cs helpers + R76 A Rust
                                delegation.
  * `digital_expand.py`       — `_expand_inv` / `_expand_nand2` /
                                `_expand_dff` / `_expand_dff_async_clear` +
                                `_mos_device` + R77 A Rust delegation.
  * `file_imports.py`         — file-import path resolver + cell
                                merger.
  * `_rust_delegation.py`     — v0.0.97 R74 D Rust-Python equivalence
                                audit hook.
  * `_resolution.py`          — `_attach_clay_file_to_cell_defs`,
                                `_resolve_verify_refs` post-parse
                                helpers.
  * `_transformer_top.py`     — `_TopMixin` (start + simple decls).
  * `_transformer_verify.py`  — `_VerifyMixin` (verify_block + setups).
  * `_transformer_cell.py`    — `_CellMixin` (cell_def + topology +
                                instance_decl).
  * `_transformer_device.py`  — `_DeviceMixin` (device / cap / res /
                                ind / varactor decls).
  * `_transformer_digital.py` — `_DigitalMixin` (inv / nand2 / dff
                                decls + `_digital_decl` body parser).
  * `_transformer_role.py`    — `_RoleMixin` (role_decl + functional
                                role decls + object_literal).
  * `_transformer_misc.py`    — `_MiscMixin` (sizing block + corners +
                                temps + bias_tree).
  * `_transformer_spec.py`    — `_SpecMixin` (combinator_* +
                                spec_block + spec_field + leaf rules).
  * `_transformer_expr.py`    — `_ExprMixin` (v0.0.37 expression AST +
                                fn_def + fn_body statements).
  * `_ast_builder.py`         — `_ClayTransformer` class assembled by
                                multiple inheritance from the mixins.
  * `_entry.py`               — `parse_clay_source` / `parse_clay_file`
                                entry points.

The `clay.parser` legacy single-file shim re-exports every symbol
listed in `__all__` PLUS the underscore-prefixed internals tests
historically reached into. See `clay/parser.py` for the shim.
"""

from __future__ import annotations

# ─── Public API surface ────────────────────────────────────────────
#
# Every re-export here is the same symbol the legacy single-file
# `clay/parser.py` exposed at module top-level. The order below mirrors
# the original section ordering for grep-friendly diffing.

# Constants + small helpers (used by tests / downstream tooling).
from ._constants import (
    _DEVICE_SIZE_KEYS,
    _DIGITAL_PIN_LABELS,
    _MOS_PIN_LONG_TO_SHORT,
    _PASSIVE_PIN_NAMES,
    _PASSIVE_SIZE_KEYS,
    _ROLE_PAIR_PIN_LABELS,
    _ROLE_PIN_LABELS,
    _ROLE_SIZE_KEYS,
    _UNITS,
    _VALID_CORNERS,
    _VALID_DRIVE_KEYS,
    _VALID_FUNCTIONAL_ROLE_KINDS,
    _VALID_PORT_ROLES,
    _VALID_ROLE_KINDS,
    _extract_derived_refs,
    _resolve_unit_multiplier,
    _validate_combinator_stage_refs,
)

# Grammar bootstrap.
from ._grammar import GRAMMAR_FILE, _get_parser

# Function-expression evaluator + sizing resolver (canonical Python
# path) plus the R78 C / R79 B Rust delegation hook.
from .fn_eval import (
    _ExprEvalError,
    _ReturnSignal,
    _bin_node,
    _eval_body,
    _eval_expr,
    _exec_stmts,
    _lift_expr,
    _resolve_cell_alias_overrides,
    _resolve_cell_def_sizing,
    _resolve_expr_for_sizing,
    _resolve_sizing_dict,
    _resolve_verify_job_overrides,
    _simplify_body_for_ir,
    _try_rust_fn_eval_delegation,
    _try_rust_resolve_expr_for_sizing_delegation,
    _try_rust_resolve_sizing_dict_delegation,
)

# Parse-time fn-body validators + R79 B Rust delegation.
from .fn_validators import (
    _body_always_returns,
    _validate_fn_body_lets,
    _validate_fn_body_terminates,
)

# Role-decl expansion + R76 A Rust delegation.
from .role_expand import (
    _expand_cs,
    _expand_mirror,
    _expand_pair,
    _expand_role_to_devices,
    _expand_roles_to_devices,
    _is_pmos_kind,
    _pick_default_pair_gates,
    _supply_and_body,
    _try_rust_pair_delegation,
)

# Digital-primitive expansion + R77 A Rust delegation.
from .digital_expand import (
    _expand_dff,
    _expand_dff_async_clear,
    _expand_inv,
    _expand_nand2,
    _mos_device,
    _try_rust_digital_decl_delegation,
)

# File-import resolver.
from .file_imports import (
    _find_repo_root,
    _merge_imported_cells,
    _resolve_file_import,
)

# Post-parse resolution helpers.
from ._resolution import (
    _attach_clay_file_to_cell_defs,
    _resolve_verify_refs,
)

# Rust audit hook.
from ._rust_delegation import _audit_rust_role_digital_delegation

# The transformer class (assembled from the per-topic mixins).
from ._ast_builder import _ClayTransformer

# Public parse entry points.
from ._entry import parse_clay_file, parse_clay_source


# `_PARSER` is the module-level cache used by `_get_parser`. Some
# legacy tests reach into `clay.parser._PARSER` to verify lazy init
# state, so we expose it from the split package too. NOTE: we import
# the SAME global cache here rather than rebinding to None at the
# parser_modules level — the canonical mutable cell lives in
# `_grammar._PARSER`. Tests that monkeypatch `clay.parser._PARSER`
# directly continue to hit that cell via the shim's re-export.
from ._grammar import _PARSER  # noqa: F401


__all__ = [
    # Public parse entry points
    "parse_clay_source",
    "parse_clay_file",
    # Constants frequently introspected by downstream tooling
    "GRAMMAR_FILE",
    "_VALID_ROLE_KINDS",
    "_VALID_FUNCTIONAL_ROLE_KINDS",
    "_VALID_CORNERS",
    "_VALID_PORT_ROLES",
    "_VALID_DRIVE_KEYS",
]
