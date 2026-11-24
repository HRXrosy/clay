"""
`_ClayTransformer` — the lark Transformer that builds the ClayFile IR.

The legacy `clay/parser.py` (4467 LOC) defined this class as one
2561-LOC monolith. v0.1.13 R90 Agent B split it into nine small mixin
modules (one per topical cluster of transformer methods) so every
module stays under the 1000-LOC ceiling. The transformer class itself
is assembled here by multiple inheritance from those mixins on top of
`lark.Transformer`. Each rule method's body is byte-exact identical to
its pre-split implementation — only the module location changes.

Mixin order (inherited left-to-right per Python MRO):
  * _TopMixin           — start, package_decl, pdk_decl, string,
                          import_*, cell_alias_def, verify_ref
  * _VerifyMixin        — verify_block + setup / drive / derived_*
  * _CellMixin          — cell_def, port_list/port/port_role,
                          topology_block / topology_field, instance_decl,
                          port_map_field
  * _DeviceMixin        — device_decl, cap/res/ind/varactor_decl,
                          _passive_decl, device_pin_assignment,
                          device_size_field
  * _DigitalMixin       — inv_decl, nand2_decl, dff_decl,
                          dff_clear_decl, _digital_decl
  * _RoleMixin          — role_* + functional_* + object_*
  * _MiscMixin          — cell_sizing_block + corners/temps + bias_*
  * _SpecMixin          — combinator_* + spec_* + leaf rules (op,
                          type_id, dotted_name, qualified_name,
                          simple_cell_ref, number)
  * _ExprMixin          — expr_* + fn_* + let/return/if/else

v0.1.16 (R92 B BL-126): added `_AnnotationMixin` to the MRO. The new
mixin owns the `annotate <CellName> { ... }` top-level block + the six
annotation kinds (`@matching` / `@guard_ring` / `@well_isolated` /
`@symmetric` / `@interdigitated` / `@adjacent`). The `transform`
override at the bottom of this module flushes the parsed
`AnnotateDecl`s onto the matching `CellDef`s via
`__dict__['_layout_annotations']` so downstream consumers can build
an `AnnotatedCell` without growing the CellDef dataclass shape.
"""

from __future__ import annotations

import lark

from ._grammar import (
    AnnotateDecl,
    _AnnotationMixin,
    _CELL_ANNOTATION_TARGET,
    _FlowQualityMixin,
    _LayoutIntentMixin,
)
from ._transformer_cell import _CellMixin
from ._transformer_device import _DeviceMixin
from ._transformer_digital import _DigitalMixin
from ._transformer_expr import _ExprMixin
from ._transformer_misc import _MiscMixin
from ._transformer_role import _RoleMixin
from ._transformer_spec import _SpecMixin
from ._transformer_top import _TopMixin
from ._transformer_verify import _VerifyMixin


class _ClayTransformer(
    _TopMixin,
    _VerifyMixin,
    _CellMixin,
    _DeviceMixin,
    _DigitalMixin,
    _RoleMixin,
    _MiscMixin,
    _SpecMixin,
    _ExprMixin,
    _AnnotationMixin,
    _LayoutIntentMixin,
    _FlowQualityMixin,
    lark.Transformer,
):
    """Lark tree → ClayFile IR.

    Each `<rule>` method is called bottom-up with the rule's child items
    (already transformed). Returns the IR fragment for that rule.

    The dispatch table is built by lark from the union of methods
    contributed by every mixin. There is no method-name overlap across
    the mixins (verified by `tests/test_parser_split_v128.py`) so MRO
    resolution is unambiguous.

    v0.1.16 R92 B BL-126: `transform` is overridden to flush parsed
    annotate blocks onto the matching `CellDef`s. The base `start()`
    method (on `_TopMixin`) silently drops `("annotate_decl", ...)`
    tuples (the `if/elif` chain has no `else` branch); after the
    `ClayFile` is built, this override walks the lark tree once more
    looking for top-level `annotate_decl` children, collects them, and
    folds the annotation map onto each target `CellDef` via
    `__dict__['_layout_annotations']`.
    """

    def transform(self, tree):
        """Override `lark.Transformer.transform` to wire annotate blocks.

        Process:
          1. Walk the tree's top-level statements BEFORE transform to
             find any `annotate_decl` subtrees. Build them via
             `self.annotate_decl` so the AnnotateDecl IR shells
             materialize.
          2. Call the base `transform` to build the ClayFile.
          3. Walk the collected `AnnotateDecl`s and fold each entry's
             `target -> annotation` onto the matching CellDef's
             `__dict__['_layout_annotations']` dict (creating it if
             absent, list-append per target).

        Step 1 happens BEFORE the base transform because lark's
        `Transformer.transform` would consume the tree once and the
        `start` method discards unknown tag tuples. We instead inspect
        the tree's pre-transform children to collect `annotate_decl`
        nodes — but the safer / simpler trick is to install a custom
        `start` that captures them. We use a small post-pass on
        `cf._pending_annotate_decls` instead so the base `start`
        implementation does NOT have to grow a new branch.
        """
        # PHASE 1: build the ClayFile via the standard transform path.
        # The base `start()` is called bottom-up after every other rule
        # has been transformed. Annotate decls return
        # ("annotate_decl", AnnotateDecl(...)) tuples; `start()` will
        # silently ignore them (no `if kind == "annotate_decl"`
        # branch — its `if/elif` chain has no `else`).
        cf = super().transform(tree)

        # PHASE 2: extract the AnnotateDecls by re-walking the tree.
        # We can't capture them inside `start()` without modifying
        # `_TopMixin` (which would violate the R90 B byte-exact rule),
        # so we walk the lark tree directly here. The tree is the
        # ORIGINAL lark Tree (lark's Transformer mutates its
        # intermediate nodes but the root tree object is preserved).
        # Actually `Transformer.transform` returns the transformed root,
        # and the input tree is consumed — but we can use a side-channel:
        # the per-method transformers we authored (annotate_decl,
        # annotation_entry, ...) stash their results in a temporary
        # field on `self`. We collect them THERE.
        pending: list[AnnotateDecl] = getattr(
            self, "_pending_annotate_decls", []
        )
        # Reset for the next transform call (in case this transformer
        # instance is reused — lark uses one instance per parse).
        if hasattr(self, "_pending_annotate_decls"):
            try:
                delattr(self, "_pending_annotate_decls")
            except AttributeError:
                pass

        if not pending:
            return cf

        # PHASE 3: validate every annotate block targets a known cell
        # and fold the entries onto that cell's `_layout_annotations`.
        cells_by_name: dict[str, object] = {
            cd.local_name: cd for cd in cf.cell_defs
        }
        for adecl in pending:
            cd = cells_by_name.get(adecl.cell_name)
            if cd is None:
                raise ValueError(
                    f"annotate block targets unknown cell "
                    f"{adecl.cell_name!r}. Cells defined in this file: "
                    f"{sorted(cells_by_name)}."
                )
            ann_map = cd.__dict__.setdefault("_layout_annotations", {})
            for entry in adecl.entries:
                from clay.layout.annotations import NativeBodyTap
                if isinstance(entry.annotation, NativeBodyTap):
                    if entry.target not in {d.instance_name for d in cd.devices}:
                        raise ValueError(f"@native_body_tap: unknown device {entry.target!r}")
                    if any(isinstance(a, NativeBodyTap) for a in ann_map.get(entry.target, ())):
                        raise ValueError(f"@native_body_tap: duplicate declaration for {entry.target!r}")
                ann_map.setdefault(entry.target, []).append(entry.annotation)
        return cf

    def annotate_decl(self, items):
        """Override the mixin's annotate_decl to ALSO stash on self.

        The mixin defines `annotate_decl` to return a tuple. We extend
        the behavior to additionally append the AnnotateDecl to
        `self._pending_annotate_decls` so the post-pass in `transform`
        can fold the entries onto the CellDefs without re-walking the
        lark tree.

        The returned tuple is still emitted to `start()` (which ignores
        unknown kinds), preserving the byte-exact contract on
        `_TopMixin.start`.
        """
        result = _AnnotationMixin.annotate_decl(self, items)
        # result is ("annotate_decl", AnnotateDecl(...))
        _kind, adecl = result
        # Initialise the list on first use (one transform = one parse,
        # so the list is reset each call via `transform`'s delattr).
        pending = self.__dict__.setdefault("_pending_annotate_decls", [])
        pending.append(adecl)
        return result


__all__ = [
    "_ClayTransformer",
    "_CELL_ANNOTATION_TARGET",
]
