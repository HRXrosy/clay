"""
Clay .clay text parser — backward-compatibility shim (v0.1.13).

The parser implementation was split out of this single file
(formerly 4467 LOC) into `clay/parser_modules/` in v0.1.13 (R90 Agent B).
This module is preserved as a re-export shim so every existing import
path continues to resolve:

  * `from clay.parser import *`
  * `from clay.parser import parse_clay_source, parse_clay_file`
  * `from clay.parser import _ClayTransformer, _eval_expr, ...`
  * `from clay.parser import _VALID_ROLE_KINDS, _UNITS, ...`

Every symbol the legacy single-file module exposed is re-exported
from `clay.parser_modules` here. The split-module layout is
documented in `clay/parser_modules/__init__.py`.

The shim preserves byte-exact behavior on every public + internal
surface:
  * R74 D Rust audit hook (`_audit_rust_role_digital_delegation`)
  * R76 A pair-role Rust delegation (`_try_rust_pair_delegation`)
  * R77 A digital-decl Rust delegation
    (`_try_rust_digital_decl_delegation`)
  * R78 C / R79 B fn-eval Rust delegation
    (`_try_rust_fn_eval_delegation`, `_validate_fn_body_*` rust paths)
  * Round-17 / Round-18 fn body evaluator + validator
  * R85 D + downstream sizing / role / digital / file-import behavior

See `clay/parser_modules/` for the actual implementation modules.
"""

from __future__ import annotations

# Re-export the full public + internal surface from the split package.
# The wildcard import gives us `__all__` (the public surface);
# explicitly re-importing the underscore-prefixed names below covers
# the internals that the legacy module exposed (tests reach into
# `_expand_pair`, `_try_rust_pair_delegation`, `_try_rust_fn_eval_delegation`,
# `_VALID_FUNCTIONAL_ROLE_KINDS`, `_find_repo_root`, etc.).
from .parser_modules import *  # noqa: F401, F403

# Module-level constants + small helpers (used by tests / downstream
# tooling — the legacy module exposed them as bare top-level names).
from .parser_modules._constants import (  # noqa: F401
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
from .parser_modules._grammar import (  # noqa: F401
    GRAMMAR_FILE,
    _PARSER,
    _get_parser,
)

# fn-expression evaluator + sizing resolver.
from .parser_modules.fn_eval import (  # noqa: F401
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
)

# fn-body validators.
from .parser_modules.fn_validators import (  # noqa: F401
    _body_always_returns,
    _validate_fn_body_lets,
    _validate_fn_body_terminates,
)

# Role-decl expansion. v0.1.18 R97 D mass-deletion Strip 2 removed the
# `_try_rust_cs_delegation` / `_try_rust_mirror_delegation` wrappers
# (unused after R95 D bypassed them with direct bridge dispatch); the
# R76 A `_try_rust_pair_delegation` channel stays on the strangler-fig
# pattern.
from .parser_modules.role_expand import (  # noqa: F401
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

# Digital-primitive expansion.
from .parser_modules.digital_expand import (  # noqa: F401
    _expand_dff,
    _expand_dff_async_clear,
    _expand_inv,
    _expand_nand2,
    _mos_device,
    _try_rust_digital_decl_delegation,
)

# File-import resolver.
from .parser_modules.file_imports import (  # noqa: F401
    _find_repo_root,
    _merge_imported_cells,
    _resolve_file_import,
)

# Post-parse resolution helpers + Rust audit hook.
from .parser_modules._resolution import (  # noqa: F401
    _attach_clay_file_to_cell_defs,
    _resolve_verify_refs,
)
from .parser_modules._rust_delegation import (  # noqa: F401
    _audit_rust_role_digital_delegation,
)

# Transformer class.
from .parser_modules._ast_builder import _ClayTransformer  # noqa: F401

# Public entry points.
from .parser_modules._entry import (  # noqa: F401
    parse_clay_file,
    parse_clay_source,
)
