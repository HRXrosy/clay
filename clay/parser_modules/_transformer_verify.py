"""
Verify-block transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
`verify_block` + setup_block / drive_block / drive_field /
derived_measure_decl / derived_expr* / cell_alias_def-adjacent
v0.0.29 multi-setup verify machinery.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

import lark

from ..ir import SetupSpec, Spec, VerifyJob
from ._constants import _VALID_DRIVE_KEYS, _extract_derived_refs


class _VerifyMixin:
    """Verify-block + multi-setup verify transformer methods."""

    @lark.v_args(meta=True)
    def verify_block(self, meta, items):
        """items: [qualified_name, body_part, body_part, ...].

        Body parts are: Spec (from spec_block), dict (from
        sizing_override_block), ('corners', list[str]) (from
        corners_block), ('temps', list[float]) (from temps_block,
        v0.0.24), SetupSpec (from setup_block, v0.0.29), or
        ('derived_measure', name, expr_str) (from
        derived_measure_decl, v0.0.29).

        v0.0.29 (G4): when ANY setup_block / derived_measure_decl is
        present the verify body is "multi-setup form" — the top-level
        `spec { ... }` block becomes OPTIONAL (because spec lives inside
        each setup), and `sizing_override` / `corners` / `temps` still
        apply to every setup. Without any setup_block, the
        single-setup form is preserved exactly as before (spec
        required, behavior identical to pre-v0.0.29).
        """
        cell_path = items[0]
        spec_obj: Spec | None = None
        overrides: dict[str, float] = {}
        corners: list[str] | None = None
        temps: list[float] | None = None
        setups: list[SetupSpec] = []
        derived_measures: dict[str, str] = {}
        for body in items[1:]:
            if isinstance(body, Spec):
                spec_obj = body
            elif isinstance(body, dict):
                overrides = body
            elif isinstance(body, SetupSpec):
                setups.append(body)
            elif isinstance(body, tuple) and body:
                if body[0] == "corners":
                    corners = body[1]
                elif body[0] == "temps":
                    temps = body[1]
                elif body[0] == "derived_measure":
                    # body shape: ("derived_measure", name, expr_str)
                    _tag, dm_name, dm_expr = body
                    if dm_name in derived_measures:
                        raise ValueError(
                            f"verify {cell_path}: derived measure "
                            f"{dm_name!r} declared more than once."
                        )
                    derived_measures[dm_name] = dm_expr
        is_multi_setup = bool(setups) or bool(derived_measures)
        if not is_multi_setup:
            # Single-setup form — preserve exact pre-v0.0.29 behavior.
            if spec_obj is None:
                raise ValueError(
                    f"verify {cell_path}: spec block is required"
                )
            job = VerifyJob(
                cell_path=cell_path,
                spec=spec_obj,
                sizing_overrides=overrides,
                source_span=_source_span_from_meta(meta),
            )
            if corners is not None:
                job.corners = corners
            if temps is not None:
                job.temps = temps
            return ("verify", job)
        # Multi-setup form: validate setup names + derived measure refs.
        if not setups:
            # A `measure ... :=` without ANY setup_block is meaningless.
            raise ValueError(
                f"verify {cell_path}: derived `measure ... :=` declared "
                f"but no `setup ... { '{' } ... { '}' }` block found. "
                f"Multi-setup verify requires at least one setup."
            )
        seen: set[str] = set()
        for s in setups:
            if s.name in seen:
                raise ValueError(
                    f"verify {cell_path}: duplicate setup name "
                    f"{s.name!r}. Setup handles must be unique within "
                    f"a verify block."
                )
            seen.add(s.name)
        # Resolve each derived measure's setup.measure references and
        # reject unknown ones at parse time. We need to know which
        # measures each setup will actually emit — for v0 we take the
        # union of {top-level spec field names} ∪ {per-setup spec
        # field names} ∪ a small built-in set ({gain_dc, gbw,
        # phase_at_gbw} — the historical defaults emitted by every
        # registered testbench builder). This catches the common
        # typo cases (`adm.gainn_dc`) without forcing the user to
        # redeclare every measure they intend to use.
        known_measures_by_setup: dict[str, set[str]] = {}
        builtin_measures = {"gain_dc", "gbw", "phase_at_gbw"}
        for s in setups:
            mset: set[str] = set(builtin_measures)
            if s.spec is not None:
                mset.update(f.name for f in s.spec.fields)
            known_measures_by_setup[s.name] = mset
        # Validate top-level spec — each field must be a derived
        # measure name (`cmrr`, `psrr`) so it resolves against the
        # cross-setup post-processed dict.
        if spec_obj is not None:
            for f in spec_obj.fields:
                if f.name not in derived_measures:
                    raise ValueError(
                        f"verify {cell_path}: top-level spec field "
                        f"{f.name!r} is not a declared derived measure. "
                        f"Either declare `measure {f.name} := ...;` at "
                        f"the verify top level, or move the assertion "
                        f"into a setup's spec block. Declared derived "
                        f"measures: {sorted(derived_measures)}."
                    )
        # Validate derived measure expressions.
        for dm_name, dm_expr in derived_measures.items():
            refs = _extract_derived_refs(dm_expr)
            for setup_name, measure_name in refs:
                if setup_name not in known_measures_by_setup:
                    raise ValueError(
                        f"verify {cell_path}: derived measure {dm_name!r} "
                        f"references setup {setup_name!r} which is not "
                        f"declared. Known setups: "
                        f"{sorted(known_measures_by_setup)}."
                    )
                if measure_name not in known_measures_by_setup[setup_name]:
                    raise ValueError(
                        f"verify {cell_path}: derived measure {dm_name!r} "
                        f"references {setup_name}.{measure_name} but "
                        f"{measure_name!r} is not a known measure in "
                        f"setup {setup_name!r}. Known measures: "
                        f"{sorted(known_measures_by_setup[setup_name])}."
                    )
        job = VerifyJob(
            cell_path=cell_path,
            # Top-level spec is OPTIONAL for multi-setup; if absent we
            # store an empty Spec so downstream code can keep assuming
            # `job.spec` is always a Spec instance.
            spec=spec_obj if spec_obj is not None else Spec(fields=[]),
            sizing_overrides=overrides,
            source_span=_source_span_from_meta(meta),
        )
        if corners is not None:
            job.corners = corners
        if temps is not None:
            job.temps = temps
        job.setups = setups
        job.derived_measures = derived_measures
        return ("verify", job)

    # ─── v0.0.29 (G4): setup_block / drive_block / derived measures ─

    def setup_block(self, items) -> SetupSpec:
        """A single `setup NAME { drive { ... } spec { ... } }` block.

        items: [CNAME(setup_name), body_part1, body_part2, ...]
        where each body_part is either a `("drive", key, value)` tuple
        (from drive_block) or a `Spec` object (from spec_block).
        """
        setup_name = str(items[0])
        drive_info: tuple[str, str, float] | None = None
        setup_spec: Spec | None = None
        for body in items[1:]:
            if isinstance(body, Spec):
                if setup_spec is not None:
                    raise ValueError(
                        f"setup {setup_name!r}: more than one `spec` "
                        f"block declared. Each setup has exactly one "
                        f"spec block (or none)."
                    )
                setup_spec = body
            elif isinstance(body, tuple) and body and body[0] == "drive":
                if drive_info is not None:
                    raise ValueError(
                        f"setup {setup_name!r}: more than one `drive` "
                        f"block declared. Each setup has exactly one "
                        f"drive block."
                    )
                drive_info = body
            else:
                raise ValueError(
                    f"setup {setup_name!r}: unexpected body part "
                    f"{body!r}."
                )
        if drive_info is None:
            raise ValueError(
                f"setup {setup_name!r}: `drive { '{' } ... { '}' }` block "
                f"is required. Declare exactly one of "
                f"{sorted(_VALID_DRIVE_KEYS)} inside it."
            )
        _tag, drive_key, drive_value = drive_info
        return SetupSpec(
            name=setup_name,
            drive_mode=drive_key,
            drive_value=drive_value,
            spec=setup_spec,
        )

    def drive_block(self, items) -> tuple[str, str, float]:
        """A `drive { key = value; ... }` block.

        items: list of `(key, value)` tuples from drive_field. Exactly
        ONE key must be present and that key must be in
        `_VALID_DRIVE_KEYS`. Mixing keys or leaving the block empty is
        a surface-level mistake — rejected here with a clear error.

        Returns a tagged triple `("drive", key, value)` so the setup
        transformer can pick it apart alongside the optional spec block.
        """
        if not items:
            raise ValueError(
                "drive { ... } block is empty — declare exactly one "
                f"of {sorted(_VALID_DRIVE_KEYS)}."
            )
        if len(items) > 1:
            keys = [k for k, _ in items]
            raise ValueError(
                f"drive { '{' } ... { '}' } block has multiple keys "
                f"({keys!r}). Declare exactly one drive mode per "
                f"setup. Future versions may allow combinations; v0 "
                f"requires a single keyword."
            )
        key, value = items[0]
        if key not in _VALID_DRIVE_KEYS:
            raise ValueError(
                f"drive { '{' } ... { '}' } block: unknown drive key "
                f"{key!r}. Valid keys: {sorted(_VALID_DRIVE_KEYS)}."
            )
        return ("drive", key, float(value))

    def drive_field(self, items) -> tuple[str, float]:
        """A single `<key> = <number>;` line inside a drive block."""
        key = str(items[0])
        value = items[1]   # already resolved by `number` rule
        return (key, float(value))

    def derived_measure_decl(self, items) -> tuple[str, str, str]:
        """A top-level `measure NAME := <expr>;` declaration.

        items: [CNAME(name), expr_string]. The expression is collected
        as a single space-joined string by the `derived_expr` rule. The
        verify_block transformer accumulates these into
        `VerifyJob.derived_measures` and validates their setup/measure
        references against the declared setups.
        """
        name = str(items[0])
        expr_str = items[1]
        return ("derived_measure", name, expr_str)

    def derived_expr(self, items) -> str:
        """Collapse the ordered token list into one expression string.

        Each item is either a string (from derived_expr_ref / paren / op)
        or a `lark.Token` (SIGNED_NUMBER, DERIVED_OP). We join with a
        single space so the downstream evaluator can re-tokenize without
        ambiguity.
        """
        parts: list[str] = []
        for it in items:
            if isinstance(it, lark.Token):
                parts.append(str(it))
            else:
                parts.append(str(it))
        return " ".join(parts)

    def derived_expr_ref(self, items) -> str:
        """`<setup>.<measure>` reference, kept verbatim with the dot."""
        return f"{items[0]}.{items[1]}"

    def derived_expr_paren(self, items) -> str:
        """Parenthesised sub-expression, restored with `( )` markers."""
        return f"( {items[0]} )"


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
