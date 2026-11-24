"""
Cell-def compiler — materialize a `CellDef` (from .clay text) into a
fully-built `Cell` IR by invoking the appropriate stage template.

Bridges the surface DSL to the Python-implemented circuit skeleton
library (stage_templates.py).

v0.0.28 (cell composition): in addition to the stage-template path, a
CellDef can compose itself from sub-cells via `instance` declarations
inside its topology block. The compiler resolves each sub-cell, builds
it via the existing module registry, then flattens the device lists
under one device list with net renaming applied (outer nets via the
port map; internal nets prefixed with the instance handle so siblings
don't collide on shared internal names like `tail`, `d1`, `pbias`).
"""

from __future__ import annotations

import copy
import inspect
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

from .ir import (
    BiasNode,
    Cell,
    CellDef,
    CellInstance,
    ClayFile,
    Device,
    ParsedFunctionalRole,
)
from .module import resolve as resolve_module
from .pdk import get_pdk
from .stage_templates import get_stage


class CellCompileError(ValueError):
    """Raised when a CellDef cannot be compiled (missing style, etc.)."""


# ─── v0.1.04 R80 B (BL-011): Rust IR compile-cell delegation hook ──
#
# When `CLAY_USE_RUST_IR_COMPILE=1` AND the `clay_ir` Rust extension is
# built AND the `try_rust_compile_cell_def` PyO3 surface exists, the
# primitive cell compile path (`cd.devices` non-empty, no style /
# instances / combinator) delegates to Rust first. On None return, the
# Python `_compile_primitive_cell` path runs verbatim — byte-exact
# behavior preserved.
#
# Honest scope: Rust handles ONLY the primitive path. Stage-template /
# composition / combinator stay in Python — those depend on the Python
# stage_templates registry, `module.resolve()` allow-list, and Python
# `stage_combinators` algebra that aren't ported this round.
#
# Strangler-fig contract: callers MUST treat None as "fall back to
# Python". The Python path is the byte-for-byte equivalence anchor.

_HAS_CLAY_IR_COMPILE = False
_HAS_CLAY_IR_STAGE_COMPILE = False
try:
    import clay_ir as _clay_ir_ext  # type: ignore[import-not-found]
    _HAS_CLAY_IR_COMPILE = hasattr(_clay_ir_ext, "try_rust_compile_cell_def")
    # v0.1.05 R81 B (BL-011 cont): stage_template path delegation surface.
    _HAS_CLAY_IR_STAGE_COMPILE = hasattr(
        _clay_ir_ext, "try_rust_compile_stage_template"
    )
except ImportError:
    _clay_ir_ext = None  # type: ignore[assignment]


# v0.1.05 R81 B: stage_template style names the Rust port handles. Loaded
# lazily at first call so test environments that monkey-patch the
# extension don't get a stale snapshot. The set is fixed once the
# extension is loaded — the Rust source defines the list.
_RUST_STAGE_TEMPLATE_NAMES: Optional[set[str]] = None


def _rust_stage_template_names() -> set[str]:
    """Lazy-init the supported stage-template name set."""
    global _RUST_STAGE_TEMPLATE_NAMES
    if _RUST_STAGE_TEMPLATE_NAMES is None:
        if _HAS_CLAY_IR_STAGE_COMPILE and _clay_ir_ext is not None:
            try:
                _RUST_STAGE_TEMPLATE_NAMES = set(
                    _clay_ir_ext._supported_stage_templates()
                )
            except Exception:
                _RUST_STAGE_TEMPLATE_NAMES = set()
        else:
            _RUST_STAGE_TEMPLATE_NAMES = set()
    return _RUST_STAGE_TEMPLATE_NAMES


def _rust_ir_compile_enabled() -> bool:
    """True iff the user opted into the Rust IR compile path AND the
    extension surface is available."""
    if not _HAS_CLAY_IR_COMPILE:
        return False
    flag = os.environ.get("CLAY_USE_RUST_IR_COMPILE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _try_rust_compile_primitive_cell(cd: CellDef) -> Optional[Cell]:
    """Attempt to compile a primitive cell through Rust.

    Returns a freshly-built `Cell` IR on success, or `None` when:
      * Rust path is disabled (env var unset OR extension missing).
      * The CellDef is out-of-subset (PDK-specific MOS kind, pdk_variant
        already set on a device, malformed bias tree, etc.). Rust path
        returns None and Python `_compile_primitive_cell` runs.
      * A structured Rust error fires that matches a Python-rejection
        case — we raise `CellCompileError` with byte-identical text so
        Python tests asserting on the error message keep passing.

    The caller is responsible for:
      * Calling this AFTER the dispatcher has confirmed the primitive
        path is selected (cd.devices non-empty, no style/instances/
        combinator).
      * Calling `_attach_functional_roles(cell, cd)` on the returned
        cell — the Rust path does not touch functional_roles or run
        the physical-invariant gate.
    """
    if not _rust_ir_compile_enabled():
        return None
    if _clay_ir_ext is None:
        return None
    # The PyO3 device tuple predates source-model provenance. Preserve it through
    # the Python path until that wire contract grows the field.
    if any(getattr(dev, "source_model", None) for dev in cd.devices):
        return None

    # Resolve PDK id the same way `_compile_primitive_cell` does.
    clay_file: Optional[ClayFile] = cd.__dict__.get("_clay_file")
    if clay_file is not None and clay_file.pdk_id:
        pdk_id = clay_file.pdk_id
    else:
        pdk_id = cd.__dict__.get("_clay_file_pdk_id", "sky130")

    # Coerce devices into the tuple shape PyO3 expects.
    devices_arg = []
    for dev in cd.devices:
        pins_list = list(dev.pins.items())
        sizing_list = [(k, float(v)) for k, v in dev.sizing.items()]
        devices_arg.append((
            dev.instance_name,
            dev.device_class,
            dev.pdk_variant,
            pins_list,
            sizing_list,
        ))

    # Coerce bias_tree into the tuple shape.
    bias_arg = []
    for n in cd.bias_tree:
        bias_arg.append((
            n.name,
            n.kind,
            n.src,
            None if n.value is None else float(n.value),
            None if n.by_M is None else int(n.by_M),
        ))

    try:
        result = _clay_ir_ext.try_rust_compile_cell_def(
            cd.local_name,
            list(cd.ports),
            devices_arg,
            bias_arg,
            pdk_id,
        )
    except Exception:
        # Defensive: any unexpected PyO3 boundary error falls back to
        # Python rather than surfacing a Rust exception.
        return None

    if result is None:
        return None

    # Tagged error tuple from Rust: ("__err__", kind, render).
    if (
        isinstance(result, tuple)
        and len(result) == 3
        and result[0] == "__err__"
    ):
        # The Python `_compile_primitive_cell` would raise CellCompileError
        # with the same text in these structural-rejection cases. Raise
        # it here so the caller short-circuits and the Python path is NOT
        # re-run (which would raise the same error again).
        raise CellCompileError(result[2])

    # Successful payload: (name, ports, devices, pdk_id, dangling_nets).
    if not (isinstance(result, tuple) and len(result) == 5):
        # Unexpected shape — fall back to Python defensively.
        return None
    name, ports, devices, ret_pdk_id, dangling_nets = result

    # Rebuild Device objects byte-exactly. The pin / sizing lists from
    # Rust come back in iteration order; Python `Device.pins` is a dict
    # whose insertion order doesn't affect equality. We re-insert into
    # a fresh dict so downstream code sees the same shape Python would
    # have produced (and the order matches Python's iteration over the
    # original bias_tree materializer / user device list).
    rebuilt: list[Device] = []
    for instance_name, device_class, pdk_variant, pins_list, sizing_list in devices:
        rebuilt.append(Device(
            instance_name=instance_name,
            device_class=device_class,
            pdk_variant=pdk_variant,
            pins=dict(pins_list),
            sizing=dict(sizing_list),
        ))

    cell = Cell(
        name=name,
        ports=list(ports),
        spec=cd.spec,
        devices=rebuilt,
        pdk_id=ret_pdk_id,
    )

    # Match Python's behavior: emit a UserWarning for dangling nets and
    # stash the list on cell.__dict__.
    if dangling_nets:
        warnings.warn(
            f"cell {cd.local_name!r}: nets touched by only one device "
            f"pin AND not in (ports, vdd/vss/0/gnd, bias_tree): "
            f"{list(dangling_nets)}. Likely typos.",
            stacklevel=3,
        )
        cell.__dict__["_dangling_nets"] = list(dangling_nets)

    return cell


def _try_rust_compile_stage_template(cd: CellDef, style: str) -> Optional[Cell]:
    """v0.1.05 R81 B (BL-011 cont): try the Rust stage_template path.

    The Rust port handles 5 of the ~35 stage_templates registered in
    `clay/stage_templates.py`:

      * `common_source_pmos_input`
      * `two_stage_miller_pmos`
      * `two_stage_miller_nmos`
      * `telescopic_cascode`  (sky130 1.8 V default path only)
      * `folded_cascode`      (sky130 1.8 V default path only)

    Returns:
      * A freshly-built `Cell` IR on byte-equal success.
      * `None` when:
          * Rust path disabled (env var unset OR extension missing).
          * Style not in the Rust subset.
          * sizing_kwargs contains a kwarg the Rust port doesn't know,
            or kwargs route into a high-supply branch.
          * Cell has a non-empty `bias_tree` (stage-template + bias_tree
            forwarding has fragile Python-side gating that the Rust
            subset doesn't replicate — Python handles).
          * Cell declares a non-empty `ports` list that mismatches the
            stage's port list (Python raises with the canonical
            sanity-check message — Rust subset skips).

    The caller MUST treat `None` as "fall back to Python's stage builder".
    """
    if not _rust_ir_compile_enabled():
        return None
    if _clay_ir_ext is None or not _HAS_CLAY_IR_STAGE_COMPILE:
        return None
    if style not in _rust_stage_template_names():
        return None
    # bias_tree forwarding is in Python's gate. Defer to Python so the
    # canonical "stage doesn't accept bias_tree" error message keeps
    # firing on the right line number.
    if cd.bias_tree:
        return None

    # Coerce sizing_kwargs into the (key, float) tuple list PyO3 wants.
    # All stage_template sizing kwargs are numeric (W / L / I_ref /
    # N_tail / gm_id_* / Cc / etc.) — non-float kwargs would have been
    # rejected at parse time. Defensive: any TypeError on cast → None.
    try:
        kwargs_arg = [(str(k), float(v)) for k, v in cd.sizing_kwargs.items()]
    except (TypeError, ValueError):
        return None

    try:
        result = _clay_ir_ext.try_rust_compile_stage_template(
            cd.local_name, style, kwargs_arg,
        )
    except Exception:
        # Defensive: any PyO3 boundary error → Python fallback.
        return None

    if result is None:
        return None

    # Tagged-fallback tuple from Rust: ("__fallback__",) → kwargs
    # out-of-subset, run Python.
    if (
        isinstance(result, tuple)
        and len(result) == 1
        and result[0] == "__fallback__"
    ):
        return None

    # Successful payload: (name, ports, devices, pdk_id).
    if not (isinstance(result, tuple) and len(result) == 4):
        return None
    name, ports, devices, ret_pdk_id = result

    # Rebuild Device objects byte-exactly. Same shape rule as the
    # primitive path — pins/sizing come back as ordered tuples; we
    # re-insert into dicts so downstream code sees the same shape
    # Python's stage builder produces.
    rebuilt: list[Device] = []
    for instance_name, device_class, pdk_variant, pins_list, sizing_list in devices:
        rebuilt.append(Device(
            instance_name=instance_name,
            device_class=device_class,
            pdk_variant=pdk_variant,
            pins=dict(pins_list),
            sizing=dict(sizing_list),
        ))

    cell = Cell(
        name=name,
        ports=list(ports),
        spec=cd.spec,
        devices=rebuilt,
        pdk_id=ret_pdk_id,
    )

    # Port-shape sanity check — mirror the post-builder check the
    # canonical stage-template path runs. If the user declared ports
    # AND they don't match the stage's port list, the same
    # CellCompileError must fire with byte-identical text.
    if cd.ports and set(cd.ports) != set(cell.ports):
        raise CellCompileError(
            f"cell {cd.local_name!r}: declared ports {cd.ports} do not "
            f"match the stage's port list {cell.ports}"
        )

    return cell


# v0.0.34 (PLL gap A): maximum recursion depth for nested composed-cell
# compilation. Each level pushes the outer cell's local_name onto a
# stack passed through `compile_cell_def` -> `_compile_composed_cell` ->
# `_build_subcell`. Hitting this cap signals either a malicious deeply
# nested .clay file or a missed cycle (the cycle detector would normally
# catch the latter; the cap is the safety net).
_MAX_COMPOSED_CELL_DEPTH: int = 16


def _attach_functional_roles(cell: Cell, cd: CellDef) -> Cell:
    """Attach parsed source intent metadata to the compiled Cell IR.

    The typed `cell.functional_roles` field is the preservation surface
    used by the proof-status layer. The legacy
    `cell.__dict__['_functional_roles']` stash is still written for one
    compatibility window so older tools keep working during migration.

    Returns the cell unchanged (the attach is in-place). Calling this
    on a CellDef with an empty `functional_roles` list is harmless —
    it stores the empty list, and downstream callers see "no
    functional roles" identically to the pre-v0.0.62 path.

    v0.1.01 (Round 78 §5.5): also stashes `cd.port_roles` and the
    bias-tree node names on the cell's `__dict__` so the physical-
    invariant runtime (`clay/physical_invariants.py`) can read them at
    elaboration time. Then dispatches the four physical invariants
    (NetDriver / PortPolarity / PowerReach / MosBodyWell) and raises
    `CellCompileError` on any fatal violation — promoting the runtime
    `PortConstraint` check (still alive on the Rust IR layer) to a
    compile-time type-level gate so SPICE never sees the broken cell.
    """
    functional_roles = _functional_roles_with_source_obligations(
        cell.name,
        getattr(cd, "functional_roles", []) or [],
    )
    cell.functional_roles = functional_roles
    cell.__dict__["_functional_roles"] = functional_roles
    layout_intents = list(getattr(cd, "layout_intents", []) or [])
    cell.layout_intents = layout_intents
    cell.__dict__["_layout_intents"] = layout_intents
    # Carry the file-level `annotate Cell { Mx: @matching(My) }` block (parsed
    # onto cd._layout_annotations) through to the compiled cell, so downstream
    # readers (layout intent obligations, the core elaborator) see the declared
    # matchings instead of silently dropping them.
    cell.__dict__["_layout_annotations"] = dict(
        getattr(cd, "_layout_annotations", {}) or {}
    )
    # v0.1.01: attach port_roles + bias_tree node names so the physical
    # invariant runtime can consult them. Both are stashed even when
    # empty — downstream readers default to "no metadata" identically.
    cell.__dict__["_port_roles"] = dict(
        getattr(cd, "port_roles", {}) or {}
    )
    cell.__dict__["_bias_nodes"] = {
        n.name for n in (getattr(cd, "bias_tree", []) or [])
    }
    source_obligation_ids = [
        field.source_obligation_id_for(cell.name, index)
        for index, field in enumerate(cell.spec.fields)
    ]
    cell.__dict__["_compiler_source_obligation_ids"] = source_obligation_ids
    cell.__dict__["_compiler_source_obligations"] = [
        {"obligation_id": obligation_id, "required": True}
        for obligation_id in source_obligation_ids
    ]
    # v0.1.01: run the elaboration-time physical-invariant gate. The
    # gate is opt-out via a per-cell escape hatch (cell.__dict__
    # ['_skip_physical_invariants'] = True) for tests that intentionally
    # construct degenerate cells; production paths leave the flag unset.
    if not cell.__dict__.get("_skip_physical_invariants", False):
        _gate_elaboration_with_physical_invariants(cell)
    return cell


def _functional_roles_with_source_obligations(
    cell_name: str,
    roles,
) -> list[ParsedFunctionalRole]:
    enriched = []
    for index, role in enumerate(roles):
        attributes = dict(getattr(role, "attributes", {}) or {})
        attributes.setdefault(
            "source_obligation_id",
            _functional_role_source_obligation_id(cell_name, index, role),
        )
        enriched.append(
            ParsedFunctionalRole(
                kind=role.kind,
                name=role.name,
                attributes=attributes,
            )
        )
    return enriched


def _functional_role_source_obligation_id(
    cell_name: str,
    index: int,
    role,
) -> str:
    return (
        f"role:{cell_name}:{index}:"
        f"{_source_obligation_atom(role.kind)}:"
        f"{_source_obligation_atom(role.name)}"
    )


def _source_obligation_atom(value: object) -> str:
    return str(value).replace(":", "_")


def _gate_elaboration_with_physical_invariants(cell: Cell) -> None:
    """v0.1.01 (Round 78 §5.5): elaboration-time gate that promotes the
    four physical invariants from runtime check to type-level rejection.

    A violation here means the cell is structurally broken in a way
    that SPICE would later catch as a Gmin / topology / convergence
    error — but at this stage the LLM sees a structured, actionable
    error in milliseconds instead of a second-class ngspice traceback.

    The function imports `physical_invariants` lazily to avoid a
    module-level cycle (the invariant runtime imports nothing from
    clay/compiler.py, but downstream callers may add cycles later).
    """
    from .physical_invariants import (
        PhysicalInvariantError, check_physical_invariants,
    )
    try:
        check_physical_invariants(cell, raise_on_violation=True)
    except PhysicalInvariantError as exc:
        # Re-frame as CellCompileError so the existing error category
        # for "compile-time cell rejection" stays homogeneous. The
        # original violation list is preserved on `.physical_violations`
        # for downstream tooling that wants structured access.
        err = CellCompileError(
            f"cell {cell.name!r}: physical invariant violated at "
            f"elaboration time — {len(exc.violations)} fatal "
            f"violation(s):\n{exc}"
        )
        err.physical_violations = list(exc.violations)  # type: ignore[attr-defined]
        raise err from exc


def _builder_accepts_kwarg(builder, kwarg_name: str) -> bool:
    """Return True iff `builder`'s signature has a `kwarg_name`
    parameter or accepts arbitrary **kwargs.

    Used to gate forwarding of optional kwargs (`bias_tree`) so
    stage templates that pre-date the kwarg keep working unchanged.
    """
    try:
        sig = inspect.signature(builder)
    except (TypeError, ValueError):
        # Builtin or otherwise unintrospectable — assume it does NOT
        # take the optional kwarg so we keep the strict legacy path.
        return False
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == kwarg_name:
            return True
    return False


def compile_cell_def(
    cd: CellDef,
    _compile_stack: Tuple[str, ...] = (),
) -> Cell:
    """Compile a CellDef → Cell IR.

    Three compilation paths:

    1. **Stage-template path** (pre-v0.0.28): topology declares a
       `style = '...'` field. The style is looked up in the stage
       template registry and invoked with the cell's spec + sizing
       kwargs. Bias-tree forwarding follows v0.0.26 (G2) rules.

    2. **Composition path** (v0.0.28): topology declares one or more
       `instance NAME = pkg::Cell { port = net; ... };` entries instead
       of a style. Each sub-cell is resolved through `module.resolve()`,
       built with its own default sizing (no kwargs forwarded — sub-cells
       are sized at their declaration site, not here), then its devices
       are folded into the outer cell with net renaming applied:
         * Outer-bound sub-cell ports → mapped to the outer net.
         * Internal sub-cell nets → prefixed with the instance handle
           (`amp_a.tail`, `amp_b.tail`) so sibling instances don't
           collide on shared internal names.

       v0.0.34 (PLL gap A): the composition path now ALSO accepts bare-
       name instance targets (`instance amp = LocalAmp { ... };`) and
       resolves them against cells defined in the SAME .clay file or
       pulled in via `import './lib.clay';`. The Python-registered
       package allow-list is the fallback, not the only resolution
       source. Cycle detection (`A -> B -> A`) and a recursion-depth
       cap (`_MAX_COMPOSED_CELL_DEPTH = 16`) guard against pathological
       graphs.

    3. **Primitive path** (v0.0.28 device primitives): topology declares
       one or more `device NAME : kind { ... }`, `cap NAME { ... }`, or
       `res NAME { ... }` entries instead of a style. The compiler
       builds the `Cell` IR directly from `cd.devices` + an optional
       `cd.bias_tree` — no stage builder, no sub-cell flatten. This is
       the path that closes the "still doesn't look like an independent
       language" gap: an LLM can describe a *new* topology fully in
       `.clay` text without writing a Python wrapper.

    The three paths are mutually exclusive in a given CellDef. Mixing
    any two raises CellCompileError. Declaring NONE is also an error.

    The returned Cell IR can then be wrapped in a Testbench and verified
    just like any cell built directly in Python.

    Args:
        cd: the CellDef to compile.
        _compile_stack: INTERNAL — chain of outer cell names currently
            being compiled, used for cycle detection on the composition
            path. External callers should NOT pass this; it's set by
            `_compile_composed_cell` when recursing into a sub-cell.
    """
    style = cd.topology.get("style")

    # v0.0.28: primitive (device) path takes precedence over the rest.
    # Mixing device declarations with stage style OR with instance
    # composition is invalid — drop one mode.
    if cd.devices:
        if style:
            raise CellCompileError(
                f"cell {cd.local_name!r}: topology declares BOTH "
                f"`style = {style!r}` and `device` declarations. "
                f"A cell may be built by a stage template OR from "
                f"explicit device primitives, not both. Drop one."
            )
        if cd.instances:
            # This case is also caught by `_compile_composed_cell` but
            # surface it from the top-level dispatcher for clarity.
            raise CellCompileError(
                f"cell {cd.local_name!r}: `device` and `instance` "
                f"declarations cannot coexist in the same cell body. "
                f"A cell is either a primitive (devices only) or a "
                f"composition (instances only)."
            )
        # v0.1.04 R80 B (BL-011): try the Rust primitive compile path
        # first when CLAY_USE_RUST_IR_COMPILE=1. On None return, the
        # Python path runs verbatim — byte-exact behavior preserved.
        # CellCompileError raised by the Rust delegation propagates to
        # the caller exactly like Python's path would have raised it.
        rust_cell = _try_rust_compile_primitive_cell(cd)
        if rust_cell is not None:
            return _attach_functional_roles(rust_cell, cd)
        return _attach_functional_roles(_compile_primitive_cell(cd), cd)

    # v0.0.28: dispatch on `instances` first. Composition takes
    # precedence over style. Mixing both is an error (the LLM should
    # choose one mode — a "use sub-cells AND override with a stage"
    # form would be ambiguous semantics).
    if cd.instances:
        if style:
            raise CellCompileError(
                f"cell {cd.local_name!r}: topology declares BOTH "
                f"`style = {style!r}` and `instance` declarations. "
                f"A cell may compose from sub-cells OR be built by a "
                f"stage template, not both. Drop one."
            )
        if cd.bias_tree:
            raise CellCompileError(
                f"cell {cd.local_name!r}: `bias_tree` block is only "
                f"valid with stage-template builds. Composed cells "
                f"inherit each sub-cell's bias chain from the sub-cell "
                f"itself."
            )
        if cd.sizing_kwargs:
            raise CellCompileError(
                f"cell {cd.local_name!r}: `sizing { '{...}' }` block is "
                f"only valid with stage-template builds. Composed "
                f"cells inherit sizing from each sub-cell's definition."
            )
        return _attach_functional_roles(
            _compile_composed_cell(cd, _compile_stack=_compile_stack), cd,
        )

    # v0.0.92 (combinator-in-source, Round 68): if the cell carries a
    # `combinator { ... }` block, lower it to a `ComposedStage` via the
    # Python combinator API and use the resulting builder. The block is
    # mutually exclusive with style / instances / devices — the
    # combinator algebra produces the cell's full device list on its
    # own. Sizing kwargs are forwarded into the lowered builder
    # verbatim, so callers can still tune e.g. `auto_bias=True` or
    # per-stage knobs via the cell's `sizing { ... }` block.
    if cd.combinator_block is not None:
        if style:
            raise CellCompileError(
                f"cell {cd.local_name!r}: cannot declare BOTH "
                f"`combinator { '{...}' }` and `topology.style = "
                f"{style!r}`. Pick one composition source."
            )
        cell = _compile_combinator_cell(cd)
        return _attach_functional_roles(cell, cd)

    # v0.0.62+ (G1 COMPILE-GAP): functional-group overlay cells. A
    # `design.clay` whose topology is expressed as
    # `functional_group {...}` role overlays (each listing device NAMES
    # via `devices = (...)`, with the actual W/L living in the bound
    # `*_gf180.cir` source netlist) compiles by lowering the SOURCE
    # netlist into the device list and validating that every overlay
    # device reference resolves against it. No style needed — the
    # netlist IS the topology. See `compile_design`.
    if _is_functional_group_overlay(cd):
        return compile_design(cd)

    if not style:
        raise CellCompileError(
            f"cell {cd.local_name!r}: topology block missing required field 'style'"
        )
    if not isinstance(style, str):
        raise CellCompileError(
            f"cell {cd.local_name!r}: topology.style must be a string, got "
            f"{type(style).__name__}"
        )

    # v0.1.05 R81 B (BL-011 cont): try the Rust stage-template path
    # first when CLAY_USE_RUST_IR_COMPILE=1 AND the style is in the
    # Rust subset. On None return, the Python path runs verbatim —
    # byte-exact behavior preserved. CellCompileError raised by the
    # Rust delegation (port-shape mismatch) propagates to the caller
    # exactly like Python's path would have raised it.
    #
    # The Rust path skips bias_tree forwarding, file-pdk_id propagation,
    # and TypeError/ValueError re-framing. When ANY of those Python-side
    # gates would alter behavior, the Rust path returns None so Python
    # handles the full surface. The byte-equal contract is: Rust output
    # exactly matches what Python's stage builder would emit for the
    # in-subset kwargs.
    rust_cell = _try_rust_compile_stage_template(cd, style)
    if rust_cell is not None:
        return _attach_functional_roles(rust_cell, cd)

    try:
        builder = get_stage(style)
    except KeyError as exc:
        raise CellCompileError(str(exc)) from exc

    # v0.0.26 (G2): bias_tree forwarding policy.
    extra_kwargs: dict = {}
    if cd.bias_tree:
        if not _builder_accepts_kwarg(builder, "bias_tree"):
            raise CellCompileError(
                f"cell {cd.local_name!r}: declared a `bias_tree` "
                f"block but stage style {style!r} does not consume "
                f"it (builder has no `bias_tree` parameter). "
                f"Use a stage that opts into explicit bias (e.g. "
                f"'two_stage_miller_pmos_explicit_bias') or remove "
                f"the bias_tree block."
            )
        extra_kwargs["bias_tree"] = cd.bias_tree

    # v0.0.60: propagate file-level `pdk 'X';` to stage builders that
    # accept a `pdk_id` kwarg. Without this, a .clay file declaring
    # `pdk 'gf180mcu';` would still build via the stage builder's
    # default `pdk_id="sky130"`, breaking downstream testbench
    # supply auto-resolution (resolve_supply reads cell.pdk_id). Only
    # forward when the builder consumes the kwarg AND the caller has
    # not already set it via the sizing block — keeps the kwarg
    # opt-in for builders that don't use it.
    clay_file = cd.__dict__.get("_clay_file")
    file_pdk_id = getattr(clay_file, "pdk_id", None) if clay_file else None
    if (
        file_pdk_id
        and _builder_accepts_kwarg(builder, "pdk_id")
        and "pdk_id" not in cd.sizing_kwargs
    ):
        extra_kwargs["pdk_id"] = file_pdk_id

    # Forward sizing kwargs verbatim. The builder must accept them; if
    # any are unknown, Python raises TypeError which we re-frame. The
    # builder may also raise ValueError (e.g. malformed bias_tree
    # contract) — re-frame those too as CellCompileError so the
    # cell-level error category stays consistent.
    try:
        cell = builder(
            cell_name=cd.local_name,
            spec=cd.spec,
            **cd.sizing_kwargs,
            **extra_kwargs,
        )
    except TypeError as exc:
        raise CellCompileError(
            f"cell {cd.local_name!r}: sizing kwarg rejected by stage "
            f"{style!r}: {exc}"
        ) from exc
    except ValueError as exc:
        raise CellCompileError(
            f"cell {cd.local_name!r}: stage {style!r} rejected the "
            f"build inputs: {exc}"
        ) from exc

    # Sanity-check the port list — if the user declared ports in .clay,
    # they must match the builder's port list.
    #
    # v0.0.22 (G10): only port *names* participate in this check.
    # Role annotations (`name: input`, `name: power_p`, etc.) are
    # metadata recorded in `cd.port_roles`; the compiler does not
    # cross-check them against builder semantics (yet — that is a
    # future hook for stage templates that publish expected roles).
    if cd.ports and set(cd.ports) != set(cell.ports):
        raise CellCompileError(
            f"cell {cd.local_name!r}: declared ports {cd.ports} do not "
            f"match the stage's port list {cell.ports}"
        )

    return _attach_functional_roles(cell, cd)


# ─── v0.0.62+ (G1): functional-group overlay compilation ────────
#
# A "design.clay" intent overlay declares its topology as a list of
# `functional_group { role=…, devices=(MN1, MN2), … }` blocks (parsed
# into `cd.functional_roles` of kind `functional_group`). Those blocks
# list device NAMES and roles — NOT sized device instances. The real
# W/L live in the circuit's `*_gf180.cir` source netlist, bound to the
# design via the case manifest's `metadata.source_spice` (and surfaced
# at flow time as `CircuitSpec.source_cir`).
#
# `compile_design` lowers that source netlist into the device list and
# validates that every functional_group device reference resolves
# against the compiled netlist. The compiled artifact = the real
# `.cir`-derived device netlist + the role bindings (attached via
# `_attach_functional_roles`, same as every other compile path).


def _is_functional_group_overlay(cd: CellDef) -> bool:
    """True iff `cd` is a functional-group intent overlay.

    Shape: no stage `style`, no primitive `device` list, no `instance`
    composition, no `combinator` block — but at least one parsed
    `functional_group` role carrying a `devices = (...)` list. This is
    the surface an LLM / designer writes when the sized netlist lives in
    a bound `.cir` and the `.clay` only declares the role overlay.
    """
    if cd.topology.get("style"):
        return False
    if cd.devices or cd.instances or cd.combinator_block is not None:
        return False
    for role in getattr(cd, "functional_roles", []) or []:
        if getattr(role, "kind", None) == "functional_group":
            if role.attributes.get("devices"):
                return True
    return False


def _normalize_device_name(name: str) -> str:
    """Canonical key for matching a `.cir` device against an overlay ref.

    PDK device subckt instances are written `xM1 …`, `xQ1 …` in the
    source netlist while the overlay refers to them as `M1`, `Q1`. We
    strip a single leading `x`/`X` instance-prefix (only when more name
    follows) and lowercase so the two forms compare equal. Plain devices
    (`M1`, `RRs`, `Cc`) are matched case-insensitively as-is.
    """
    key = name.strip()
    if len(key) > 1 and key[0] in ("x", "X"):
        key = key[1:]
    return key.lower()


# Source-netlist model token → primitive device_class. Prefix-based so
# both vpdk `nch`/`pch` and gf180 `nmos_3p3`/`pmos_3p3` lower correctly.
def _device_class_for_model(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    low = model.lower()
    if low.startswith("n"):
        return "nmos"
    if low.startswith("p"):
        return "pmos"
    return None


def _spice_device_to_ir(dev: Any) -> Optional[Device]:
    """Lower one parsed `SpiceDevice` into a `Device` IR entry.

    MOSFETs (kind 'M') and PDK MOS-subckt instances (kind 'X' whose
    model resolves to an nmos/pmos family) become sized `Device`s with
    d/g/s/b pins. Resistors / capacitors become `res` / `cap` primitives
    with their two pins. Other kinds (BJT/diode 'Q', sources, etc.) are
    represented as opaque devices so they still appear in the netlist
    and resolve against overlay references, but carry no MOS pin schema.
    Returns None for devices that carry no useful structure.
    """
    from .spice_analysis import SpiceDeviceKind

    name = dev.name
    params = {k: float(v) for k, v in (dev.params or {}).items()}
    nodes = list(dev.nodes or [])

    if dev.kind in (SpiceDeviceKind.MOSFET, SpiceDeviceKind.SUBCKT_INST):
        cls = _device_class_for_model(dev.model)
        if cls is not None and len(nodes) >= 4:
            return Device(
                instance_name=name,
                device_class=cls,
                pdk_variant=dev.model,
                pins={
                    "d": nodes[0],
                    "g": nodes[1],
                    "s": nodes[2],
                    "b": nodes[3],
                },
                sizing=params,
            )
        # X-instance that isn't a MOS family (e.g. a PNP device subckt
        # like `xQ1 … vpnp_…`): keep it as an opaque instance so it
        # still resolves against overlay device references.
        pins = {f"p{i}": n for i, n in enumerate(nodes)}
        return Device(
            instance_name=name,
            device_class=(dev.model or "subckt"),
            pdk_variant=dev.model,
            pins=pins,
            sizing=params,
        )

    if dev.kind == SpiceDeviceKind.RESISTOR and len(nodes) >= 2:
        sizing = dict(params)
        if dev.value is not None:
            sizing.setdefault("R", float(dev.value))
        return Device(
            instance_name=name,
            device_class="res",
            pdk_variant=None,
            pins={"p": nodes[0], "n": nodes[1]},
            sizing=sizing,
        )
    if dev.kind == SpiceDeviceKind.CAPACITOR and len(nodes) >= 2:
        sizing = dict(params)
        if dev.value is not None:
            sizing.setdefault("C", float(dev.value))
        return Device(
            instance_name=name,
            device_class="cap",
            pdk_variant=None,
            pins={"p": nodes[0], "n": nodes[1]},
            sizing=sizing,
        )
    # BJT / diode (Q) and any other 2+ node element: opaque device so
    # the overlay reference resolves. Sources (V/I) carry no structural
    # role in the overlay netlist and are dropped.
    if name and name[0:1].lower() == "q" and len(nodes) >= 2:
        return Device(
            instance_name=name,
            device_class=(dev.model or "bjt"),
            pdk_variant=dev.model,
            pins={f"p{i}": n for i, n in enumerate(nodes)},
            sizing=params,
        )
    return None


def _read_manifest_metadata(case_dir: Path) -> dict[str, Any]:
    """Best-effort read of a case manifest's ``metadata`` mapping."""
    for name in ("case.yaml", "case.yml", "case.json"):
        path = case_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                import json

                doc = json.loads(text)
            else:
                import yaml

                doc = yaml.safe_load(text)
        except Exception:
            return {}
        if isinstance(doc, dict) and isinstance(doc.get("metadata"), dict):
            return dict(doc["metadata"])
        return {}
    return {}


def _discover_source_cir(cd: CellDef) -> Optional[Path]:
    """Locate the bound source ``.cir`` for a functional-group overlay.

    Resolution: from the parsed cell's ``source_span['file']`` (the
    ``design.clay`` path), walk up to the enclosing case package, read
    its manifest ``metadata.source_spice``, and resolve it relative to
    the repo root. Returns ``None`` when no manifest / source_spice is
    found (caller raises a precise error).
    """
    span = getattr(cd, "source_span", None)
    clay_path_str = span.get("file") if isinstance(span, dict) else None
    if not clay_path_str:
        return None
    clay_path = Path(clay_path_str).resolve()

    # Walk up from the design.clay's directory looking for a manifest.
    for parent in [clay_path.parent, *clay_path.parents]:
        meta = _read_manifest_metadata(parent)
        source_spice = meta.get("source_spice") if meta else None
        if source_spice:
            spice_path = Path(str(source_spice))
            if spice_path.is_absolute():
                return spice_path.resolve() if spice_path.exists() else None
            # Repo-relative: find the repo root (a dir with circuits/
            # and clay/) above the case package.
            repo_root = _infer_repo_root(parent)
            resolved = (repo_root / spice_path).resolve()
            return resolved if resolved.exists() else None
        # Stop once we've climbed past a `cases` segment with no manifest
        # match — keep climbing for the repo root case anyway.
    return None


def _infer_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the repo root (dir with circuits/+clay/)."""
    for parent in [start, *start.parents]:
        if (parent / "circuits").is_dir() and (parent / "clay").is_dir():
            return parent
    return start


def compile_design(
    cd: CellDef,
    *,
    source_cir: "str | Path | None" = None,
) -> Cell:
    """Compile a functional-group intent-overlay cell into a device Cell.

    The overlay (`cd.functional_roles` of kind ``functional_group``)
    lists device NAMES + roles; the sized devices live in the bound
    source ``.cir``. This function:

      1. Resolves the source ``.cir`` (explicit ``source_cir`` arg, else
         auto-discovered from the case manifest's ``metadata.source_spice``
         via the parsed cell's ``source_span``).
      2. Parses it with :func:`clay.spice_analysis.parse_spice_netlist`
         and selects the ``.subckt`` matching ``cd.local_name``.
      3. Lowers every device line into a ``Device`` IR entry.
      4. Validates that EVERY functional_group ``devices = (...)`` name
         resolves against the compiled netlist (case-insensitive, with a
         leading ``x``/``X`` PDK-subckt prefix stripped). An unresolved
         reference is a :class:`CellCompileError`.
      5. Builds the ``Cell`` IR and attaches the role bindings.

    Non-overlay cells delegate to :func:`compile_cell_def` so this
    function is a safe single entry point for either form.
    """
    if not _is_functional_group_overlay(cd):
        return compile_cell_def(cd)

    cir_path: Optional[Path]
    if source_cir is not None:
        cir_path = Path(source_cir).resolve()
        if not cir_path.exists():
            raise CellCompileError(
                f"cell {cd.local_name!r}: source netlist not found: "
                f"{cir_path}"
            )
    else:
        cir_path = _discover_source_cir(cd)
        if cir_path is None:
            raise CellCompileError(
                f"cell {cd.local_name!r}: functional-group overlay has no "
                f"resolvable source netlist. Provide `source_cir=` or bind "
                f"one via the case manifest `metadata.source_spice`."
            )

    from .spice_analysis import parse_spice_netlist

    text = cir_path.read_text(encoding="utf-8-sig")
    netlist = parse_spice_netlist(text)

    # Select the subckt matching the overlay cell name. Fall back to the
    # sole subckt when there's exactly one and the name doesn't match
    # (some `.cir` files name the subckt differently from the .clay cell).
    subckt = None
    for sub in netlist.subckts:
        if sub.name == cd.local_name:
            subckt = sub
            break
    if subckt is None and len(netlist.subckts) == 1:
        subckt = netlist.subckts[0]
    if subckt is None:
        names = [s.name for s in netlist.subckts]
        raise CellCompileError(
            f"cell {cd.local_name!r}: no `.subckt {cd.local_name}` found in "
            f"source netlist {cir_path.name} (subckts: {names})."
        )

    devices: list[Device] = []
    for sdev in subckt.devices:
        ir_dev = _spice_device_to_ir(sdev)
        if ir_dev is not None:
            devices.append(ir_dev)

    # Index by normalized name for overlay reference resolution.
    by_norm: dict[str, Device] = {}
    for dev in devices:
        by_norm.setdefault(_normalize_device_name(dev.instance_name), dev)

    # Validate: every functional_group device reference must resolve.
    missing: list[tuple[str, str]] = []
    for role in getattr(cd, "functional_roles", []) or []:
        if getattr(role, "kind", None) != "functional_group":
            continue
        dev_refs = role.attributes.get("devices") or []
        if isinstance(dev_refs, str):
            dev_refs = [dev_refs]
        for ref in dev_refs:
            if _normalize_device_name(str(ref)) not in by_norm:
                missing.append((role.name, str(ref)))
    if missing:
        available = sorted(d.instance_name for d in devices)
        detail = ", ".join(f"{grp}:{ref}" for grp, ref in missing)
        raise CellCompileError(
            f"cell {cd.local_name!r}: functional_group device reference(s) "
            f"do not resolve against compiled netlist {cir_path.name}: "
            f"{detail}. Devices in netlist: {available}."
        )

    # Resolve PDK id from the enclosing ClayFile (gf180mcu for these).
    clay_file: Optional[ClayFile] = cd.__dict__.get("_clay_file")
    if clay_file is not None and clay_file.pdk_id:
        pdk_id = clay_file.pdk_id
    else:
        pdk_id = cd.__dict__.get("_clay_file_pdk_id", "sky130")

    ports = list(cd.ports) if cd.ports else list(subckt.ports)
    cell = Cell(
        name=cd.local_name,
        ports=ports,
        spec=cd.spec,
        devices=devices,
        pdk_id=pdk_id,
    )
    cell.__dict__["_source_cir"] = str(cir_path)
    # The netlist is real, silicon-validated SPICE — the physical-
    # invariant gate (an LLM-authoring guard) is not the right judge
    # here, and the overlay's role pins are netlist-derived, not the
    # canonical d/g/s/b the gate expects for every model. Skip it; the
    # resolution check above is the structural contract for this path.
    cell.__dict__["_skip_physical_invariants"] = True
    return _attach_functional_roles(cell, cd)


# ─── v0.0.28: composed cell compilation ─────────────────────────

# Separator used when prefixing internal sub-cell nets with the instance
# handle. Using `__` (two underscores) keeps the result a valid SPICE
# identifier (single dot `.` is hierarchy-significant in some
# simulators and rejected as a bare node name). The result of flatten
# looks like `amp_a__tail`, `amp_b__pbias`, etc.
_NET_PREFIX_SEP = "__"


def _resolve_in_file_subcell(
    target_name: str,
    clay_file: ClayFile,
) -> Optional["CellDef | object"]:
    """v0.0.34 (PLL gap A): look up a bare or namespaced cell name in
    the enclosing ClayFile's cell_defs and cell_aliases tables.

    Returns the matching CellDef OR CellAlias if found, or None if
    neither table contains `target_name`. The caller (`_build_subcell`)
    distinguishes the two cases:
      * CellDef → recurse into `compile_cell_def` to materialize the
        cell from its own topology block (which may itself be a
        composition, primitive cell, or stage-template build).
      * CellAlias → resolve the alias's parent_path through the Python
        module path, applying the alias's `sizing_overrides` as kwargs.

    `target_name` is matched verbatim. Imports via `import './foo.clay'
    as ns;` rename their cells to `ns::CellName` at merge time, so a
    qualified-style local reference (`instance x = ns::CellName { ... };`)
    targets the namespaced entry by passing its full `ns::CellName`
    string as the target.
    """
    for cd in clay_file.cell_defs:
        if cd.local_name == target_name:
            return cd
    for a in clay_file.cell_aliases:
        if a.local_name == target_name:
            return a
    return None


def _build_subcell(
    inst: CellInstance,
    *,
    clay_file: Optional[ClayFile] = None,
    compile_stack: Tuple[str, ...] = (),
) -> Cell:
    """Resolve `inst.cell_path` and invoke the sub-cell's builder.

    Resolution order:

    1. **In-file resolution** (v0.0.34, PLL gap A): if `inst.cell_path`
       has no `::` separator OR matches a namespaced cell brought in
       via `import 'foo.clay' as ns;`, look it up in the enclosing
       ClayFile's cell_defs and cell_aliases tables. CellDef matches
       recurse through `compile_cell_def` (pushing the outer name on
       `compile_stack` for cycle detection); CellAlias matches fall
       through to the Python-package path with the alias's
       parent_path + sizing_overrides applied.

    2. **Python-package resolution** (v0.0.28): the path is parsed as
       `pkg.path::CellName`. The package is loaded through
       `module.resolve()` (which enforces the allow-list against
       arbitrary Python imports — see `clay.module`). The cell name
       is then looked up in the package's exported cell table.

    Sub-cells are built with their builder default kwargs — composed
    cells inherit each sub-cell's pre-decided sizing rather than
    re-sizing at composition time. A future version may add a per-
    instance sizing override block; for v0.0.28 the contract is
    "sub-cells size themselves".

    Args:
        inst: the CellInstance to resolve and build.
        clay_file: optional enclosing ClayFile. Required for in-file
            resolution; absent (None) falls back to Python-package
            resolution only — preserves pre-v0.0.34 behavior for
            callers that don't have a ClayFile context.
        compile_stack: chain of outer cell names currently mid-
            compilation. Used for cycle detection on recursive
            composed-cell builds.
    """
    from .ir import CellAlias as _CellAlias
    from .ir import Spec as _Spec

    # ── Try in-file resolution first (v0.0.34) ───────────────
    # The bare-name form (`instance x = LocalCell { ... };`) has NO
    # `::` separator. Imported-and-namespaced cells (`import 'lib.clay'
    # as lib;`) appear under the renamed key `lib::CellName`. We
    # consult the ClayFile's tables for BOTH shapes before reaching
    # for `module.resolve`.
    if clay_file is not None:
        resolved = _resolve_in_file_subcell(inst.cell_path, clay_file)
        if isinstance(resolved, CellDef):
            # Cycle detection: refuse to recurse if the target is
            # already on the compile stack.
            if resolved.local_name in compile_stack:
                chain = " -> ".join(list(compile_stack) + [resolved.local_name])
                raise CellCompileError(
                    f"instance {inst.name!r}: composition cycle "
                    f"detected: {chain}. Cells cannot reference each "
                    f"other in a loop (direct or indirect)."
                )
            # Recursion-depth safety net.
            if len(compile_stack) >= _MAX_COMPOSED_CELL_DEPTH:
                raise CellCompileError(
                    f"instance {inst.name!r}: composed-cell recursion "
                    f"depth exceeded {_MAX_COMPOSED_CELL_DEPTH} levels. "
                    f"Stack: {' -> '.join(compile_stack)}."
                )
            try:
                return compile_cell_def(
                    resolved,
                    _compile_stack=compile_stack,
                )
            except CellCompileError:
                raise
            except Exception as exc:
                raise CellCompileError(
                    f"instance {inst.name!r}: in-file sub-cell "
                    f"{inst.cell_path!r} failed to compile: {exc}"
                ) from exc
        if isinstance(resolved, _CellAlias):
            # CellAlias resolution: route through the alias's parent_path
            # in the Python-module registry, but apply the alias's
            # sizing_overrides as kwargs to the builder. This is the same
            # contract as a top-level `verify <alias_name>;` job: the
            # parent stdlib cell's builder runs with the alias's overrides.
            pkg, sep, cell_name = a_parent_path_split(resolved.parent_path)
            if not sep:
                raise CellCompileError(
                    f"instance {inst.name!r}: cell alias "
                    f"{resolved.local_name!r} has malformed parent "
                    f"path {resolved.parent_path!r} (missing `::`)."
                )
            try:
                mod = resolve_module(pkg)
            except Exception as exc:
                raise CellCompileError(
                    f"instance {inst.name!r}: cell alias "
                    f"{resolved.local_name!r} parent path {pkg!r} "
                    f"failed to resolve: {exc}"
                ) from exc
            try:
                builder = mod.cell(cell_name)
            except Exception as exc:
                raise CellCompileError(
                    f"instance {inst.name!r}: cell alias "
                    f"{resolved.local_name!r}: package {pkg!r} has no "
                    f"cell {cell_name!r}: {exc}"
                ) from exc
            try:
                return builder(
                    spec=_Spec(fields=[]),
                    **resolved.sizing_overrides,
                )
            except Exception as exc:
                raise CellCompileError(
                    f"instance {inst.name!r}: builder for cell alias "
                    f"{resolved.local_name!r} (-> {resolved.parent_path!r}) "
                    f"raised during sub-cell build: {exc}"
                ) from exc
        # In-file lookup miss — fall through to Python-package path
        # for qualified names. Bare-name misses get a precise error
        # below.

    # ── Python-package resolution (v0.0.28) ──────────────────
    pkg, sep, cell_name = inst.cell_path.partition("::")
    if not sep:
        # Bare name with no in-file match → unresolved reference.
        # Produce a clear error listing what WAS visible so the LLM
        # can fix the typo (or add the missing import).
        known_defs: list[str] = []
        known_aliases: list[str] = []
        if clay_file is not None:
            known_defs = sorted(c.local_name for c in clay_file.cell_defs)
            known_aliases = sorted(a.local_name for a in clay_file.cell_aliases)
        raise CellCompileError(
            f"instance {inst.name!r}: cell {inst.cell_path!r} is not "
            f"defined in this .clay file or any imported .clay file, "
            f"and is not a fully-qualified `pkg::CellName` reference "
            f"into a Python-registered package. "
            f"Known in-file cell_defs: {known_defs}. "
            f"Known in-file cell_aliases: {known_aliases}. "
            f"Either define `cell {inst.cell_path}(...)` in this file, "
            f"import a .clay file that defines it, or use a fully-"
            f"qualified path (e.g. `clay.stdlib.opamps::CellName`)."
        )
    try:
        mod = resolve_module(pkg)
    except Exception as exc:
        raise CellCompileError(
            f"instance {inst.name!r}: failed to resolve package "
            f"{pkg!r} for cell_path {inst.cell_path!r}: {exc}"
        ) from exc
    try:
        builder = mod.cell(cell_name)
    except Exception as exc:
        raise CellCompileError(
            f"instance {inst.name!r}: package {pkg!r} has no cell "
            f"{cell_name!r}: {exc}"
        ) from exc
    # Use an empty Spec for sub-cells — composition is netlist-level,
    # not spec-level. The sub-cell's own `verify` job (if any) carries
    # the relevant spec.
    try:
        cell = builder(spec=_Spec(fields=[]))
    except Exception as exc:
        raise CellCompileError(
            f"instance {inst.name!r}: builder for {inst.cell_path!r} "
            f"raised during sub-cell build: {exc}"
        ) from exc
    return cell


def a_parent_path_split(parent_path: str) -> tuple[str, str, str]:
    """Split a `pkg.path::CellName` string into its three parts.

    Helper for in-file `CellAlias` resolution. Same semantics as
    `str.partition("::")` — preserved as a named function so the call
    sites read clearly.
    """
    return parent_path.partition("::")


def _rename_net(
    net: str,
    *,
    instance_name: str,
    port_map: dict[str, str],
    sub_cell_ports: set[str],
) -> str:
    """Translate a sub-cell-local net name into the composed cell's namespace.

    Rules:
      * If `net` is one of the sub-cell's ports (i.e. crosses the
        instance boundary) → use the outer net via `port_map[net]`.
      * Otherwise → prefix with `<instance_name>__` so sibling instances
        don't collide on shared internal names (`tail`, `d1`, `pbias`).
      * Reserved nets `'0'` and `'gnd'` are never renamed — they refer
        to the global SPICE ground rail and must remain shared.
    """
    if net in ("0", "gnd"):
        return net
    if net in sub_cell_ports:
        # Already validated upstream — every sub-cell port has a
        # matching port_map entry by the time we get here.
        return port_map[net]
    return f"{instance_name}{_NET_PREFIX_SEP}{net}"


def _compile_composed_cell(
    cd: CellDef,
    _compile_stack: Tuple[str, ...] = (),
) -> Cell:
    """Build a Cell IR by flattening multiple sub-cells per the
    `instance` declarations on `cd`.

    Algorithm:
      1. For each instance: resolve and build the sub-cell.
      2. Validate the port map: every sub-cell port MUST have a binding
         in `inst.port_map`; every binding MUST refer to a port on the
         sub-cell; every outer-net target MUST be one of the outer
         cell's ports OR a known global rail (`vdd`/`vss`/`0`/`gnd`).
      3. Rename device pins per `_rename_net` and prefix each device's
         instance_name with the instance handle so duplicates across
         sub-cells don't collide.
      4. Accumulate the renamed devices into the outer Cell IR.

    v0.0.34 (PLL gap A): for each instance, the resolution path is now
      a. In-file cell_defs / cell_aliases (`cd.__dict__['_clay_file']`)
      b. Python-package allow-list (pre-v0.0.34 behavior)
    See `_build_subcell` for the full resolution algorithm. `_compile_stack`
    tracks the chain of outer cells currently being compiled so cycle
    detection (`A -> B -> A`) can fire with a precise error chain.
    """
    if cd.devices:
        # Coexistence guard for Agent A's primitive-device path. The
        # surface allows `cell` bodies with EITHER explicit `device`
        # declarations OR `instance` declarations, but not both — mixing
        # them would muddle the "is this a primitive or a composition?"
        # contract.
        raise CellCompileError(
            f"cell {cd.local_name!r}: `device` and `instance` "
            f"declarations cannot coexist in the same cell body. "
            f"A cell is either a primitive (devices only) or a "
            f"composition (instances only)."
        )

    # v0.0.34 (PLL gap A): pull the enclosing ClayFile back off the
    # CellDef. The parser stashes this reference after all file imports
    # are merged; `compile_cell_def` was invoked at the top of the chain
    # without an explicit file argument, so this private back-edge keeps
    # the public signature backward-compatible.
    clay_file: Optional[ClayFile] = cd.__dict__.get("_clay_file")
    # Push the current cell's local_name onto the compile stack for the
    # duration of this composition. Sub-cell recursion via
    # `_build_subcell` consults the stack for cycle detection.
    sub_stack: Tuple[str, ...] = _compile_stack + (cd.local_name,)

    # The set of nets the user has named at the outer level — every
    # port_map RHS must be one of these (or a reserved global rail).
    outer_ports: set[str] = set(cd.ports)
    # Global rails the LLM never has to declare in the outer port
    # signature — the SPICE-level `0` / `gnd` reference is implicit.
    # `vdd` and `vss` are CONVENTIONAL but NOT magic: they only count
    # as legal targets if the outer cell actually declares them as
    # ports OR if the sub-cell port is one of those names AND
    # `port_map['vdd'] == 'vdd'` etc. (i.e. the LLM made a passthrough
    # binding). We treat the literal strings `'0'` and `'gnd'` as
    # always-legal targets because they refer to the universal SPICE
    # ground rail. Everything else must be a declared outer port.
    flat_devices: list[Device] = []

    # R119 A (M-DUAL-IR-UNIFY close): pre-resolve every sub-cell here,
    # then delegate the per-instance flatten+rename orchestrator to the
    # R118 A Rust port when the gate is open. The Rust orchestrator does
    # NOT recurse — `_build_subcell` is Python-owned and remains so
    # because it depends on the Python `module.resolve()` allow-list and
    # the in-file ClayFile resolver.
    sub_cells: list[Cell] = []
    for inst in cd.instances:
        sub_cells.append(
            _build_subcell(
                inst,
                clay_file=clay_file,
                compile_stack=sub_stack,
            )
        )

    if _rust_compose_orchestrator_enabled():
        rust_devices = _try_rust_compose_orchestrator(
            cd_local_name=cd.local_name,
            cd_ports=list(cd.ports),
            has_devices=False,  # already gated above
            instances=list(cd.instances),
            sub_cells=sub_cells,
        )
        if rust_devices is not None:
            return Cell(
                name=cd.local_name,
                ports=list(cd.ports),
                spec=cd.spec,
                devices=rust_devices,
                pdk_id="sky130",
            )

    # Python fallback path — byte-identical to v0.2.23- behavior.
    for inst, sub_cell in zip(cd.instances, sub_cells):
        sub_cell_ports = set(sub_cell.ports)

        # Validate port-map completeness — every sub-cell port must be
        # bound. Catches the "user forgot to map `vss`" class of bugs.
        missing = sub_cell_ports - set(inst.port_map.keys())
        if missing:
            raise CellCompileError(
                f"instance {inst.name!r} of {inst.cell_path!r}: the "
                f"port-map is missing bindings for sub-cell ports "
                f"{sorted(missing)}. Every sub-cell port must be "
                f"explicitly mapped to an outer net (no implicit / "
                f"positional fallback)."
            )
        # Validate port-map keys — every binding must name a real port
        # on the sub-cell. Catches typos like `inn = ...` when the
        # sub-cell calls it `in_`.
        extra = set(inst.port_map.keys()) - sub_cell_ports
        if extra:
            raise CellCompileError(
                f"instance {inst.name!r} of {inst.cell_path!r}: the "
                f"port-map references unknown sub-cell ports "
                f"{sorted(extra)}. Sub-cell ports are "
                f"{sorted(sub_cell_ports)}."
            )
        # Validate port-map values — every outer net target must be a
        # declared outer cell port OR a reserved global rail.
        for sub_port, outer_net in inst.port_map.items():
            if outer_net in ("0", "gnd"):
                continue
            if outer_net not in outer_ports:
                raise CellCompileError(
                    f"instance {inst.name!r}: port-map binding "
                    f"`{sub_port} = {outer_net}` references "
                    f"{outer_net!r} which is not a declared outer-cell "
                    f"port. Outer ports are {sorted(outer_ports)}."
                )

        # Flatten devices with net rename + device-name prefix. Use
        # deepcopy on each device so we don't mutate the sub-cell's
        # cached Cell IR (sub-cells may be referenced from multiple
        # `instance` slots and stage builders may cache results).
        for dev in sub_cell.devices:
            new_dev = copy.deepcopy(dev)
            new_dev.instance_name = (
                f"{inst.name}{_NET_PREFIX_SEP}{dev.instance_name}"
            )
            new_dev.pins = {
                pin_name: _rename_net(
                    net,
                    instance_name=inst.name,
                    port_map=inst.port_map,
                    sub_cell_ports=sub_cell_ports,
                )
                for pin_name, net in dev.pins.items()
            }
            flat_devices.append(new_dev)

    return Cell(
        name=cd.local_name,
        ports=list(cd.ports),
        spec=cd.spec,
        devices=flat_devices,
        # Outer cell's PDK matches the underlying sub-cells'; the
        # surface form doesn't currently let an outer composition
        # cross PDKs (which would be physically meaningless anyway).
        pdk_id="sky130",
    )


# ─── R119 A (M-DUAL-IR-UNIFY close): Rust compose orchestrator gate ──
#
# Env var `CLAY_RUST_COMPOSE_ENABLED` (default ON) controls whether the
# per-instance flatten+rename loop in `_compile_composed_cell` delegates
# to the R118 A Rust orchestrator. Set to `"0"` / `"false"` / `"off"` /
# `"no"` to force the Python fallback path.
#
# Honest scope (matches R118 A's bridge audit §4.2):
#   * `_build_subcell` resolution remains in Python (recurses into
#     `compile_cell_def` via `module.resolve()` allow-list).
#   * Outer `Cell(...)` IR construction stays in Python.
#   * Bias-tree materializer / functional-roles attachment stay in
#     Python (those run inside `compile_cell_def`, not here).
#
# Strangler-fig: when the Rust orchestrator returns None or the bridge
# is not importable, the Python fallback runs verbatim — zero behavioral
# change.


def _rust_compose_orchestrator_enabled() -> bool:
    """True iff the Rust compose orchestrator path should be attempted.

    **Default ON** (R120 main-session flip; R119 A originally shipped
    default-OFF due to 2 R109 D legacy tests pinning
    ``backed_by == 'python'``; R120 retired those legacy assertions in
    `test_ir_readonly_view_v207.py` to ``in {'python', 'rust'}``, then
    flipped default ON here).

    Set ``CLAY_RUST_COMPOSE_ENABLED=0`` (or ``"false"`` / ``"off"`` /
    ``"no"``) to force Python fallback path (debugging or environments
    without clay_ir extension).

    M-DUAL-IR-UNIFY status: **DONE** (R120 close). ``ReadOnlyClayIR.
    backed_by`` reports ``'rust'`` when the clay_ir extension is built
    and the env var is not explicitly disabled.
    """
    flag = os.environ.get("CLAY_RUST_COMPOSE_ENABLED", "").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    try:
        from ._ir_bridge_compile_composed_cell_orchestrator_rs import (
            orchestrator_rust_enabled,
        )
    except ImportError:
        return False
    return orchestrator_rust_enabled()


def _try_rust_compose_orchestrator(
    *,
    cd_local_name: str,
    cd_ports: list[str],
    has_devices: bool,
    instances: list,
    sub_cells: list,
) -> Optional[list[Device]]:
    """Attempt to compose flat devices via the Rust orchestrator.

    Returns the freshly-built flat device list on success, or ``None``
    when the Rust path should be skipped (bridge missing, env disabled,
    or unexpected exception). On a structured validation error from the
    Rust orchestrator (missing port, extra port, unknown outer-net,
    coexist gate), raises :class:`CellCompileError` with byte-identical
    text — matching the Python fallback path.
    """
    try:
        from ._ir_bridge_compile_composed_cell_orchestrator_rs import (
            compose_cell_rust,
        )
    except ImportError:
        return None

    try:
        flat_devs, err = compose_cell_rust(
            cd_local_name,
            list(cd_ports),
            has_devices,
            list(instances),
            list(sub_cells),
        )
    except Exception:
        # Defensive: any PyO3 boundary error falls back to Python rather
        # than surfacing a Rust exception. The Python path is the
        # byte-for-byte equivalence anchor.
        return None

    if err is not None:
        # Byte-identical CellCompileError text — Python fallback would
        # raise the same text. Don't re-run Python (it would just re-
        # raise the same error).
        raise CellCompileError(err)
    return flat_devs


# ─── v0.0.28: primitive cell compilation ────────────────────────

# Default sizing for transistors that the bias_tree materializer
# synthesizes (diodes, mirrors, auxiliary mirrors). These match the
# defaults used by `two_stage_miller_pmos_explicit_bias._materialize_bias_chain`
# so the visible bias-rail behavior is identical between the two paths.
_BIAS_CHAIN_DEFAULT_W: float = 4e-6
_BIAS_CHAIN_DEFAULT_L: float = 1e-6


def _materialize_bias_tree_devices(
    bias_tree: list[BiasNode],
    *,
    cell_name: str,
) -> list[Device]:
    """Translate a list of BiasNode IRs into actual Device objects.

    Mirror of the helper in
    `clay.templates.two_stage_miller_pmos_explicit_bias._materialize_bias_chain`,
    duplicated here so the primitive-cell build path doesn't have a
    dependency on a specific stage template module. Behavior is
    intentionally the same — same node kinds, same auxiliary-mirror
    rules, same `external`-current polarity choice — so an LLM can
    swap freely between stage-template + bias_tree and primitive
    devices + bias_tree.

    Args:
        bias_tree: parsed bias-node list (already validated for
            forward refs, cycles, units, positivity in the parser)
        cell_name: enclosing cell name, used only in error messages

    Returns:
        list of Device IR entries materializing the bias chain.
    """
    devices: list[Device] = []
    by_name: dict[str, BiasNode] = {n.name: n for n in bias_tree}

    downstream_of: dict[str, list[BiasNode]] = {}
    for m in bias_tree:
        if m.src is not None:
            downstream_of.setdefault(m.src, []).append(m)

    for n in bias_tree:
        if n.kind == "external":
            downstream = downstream_of.get(n.name, [])
            if not downstream:
                raise CellCompileError(
                    f"cell {cell_name!r}: bias_tree external node "
                    f"{n.name!r} has no consumer. At least one "
                    f"downstream node must declare `<name>: pmos_diode "
                    f"from {n.name}` (or `nmos_diode from {n.name}`)."
                )
            consumer = downstream[0]
            if consumer.kind == "pmos_diode":
                pins = {"p": consumer.name, "n": "vss"}
            elif consumer.kind == "nmos_diode":
                pins = {"p": "vdd", "n": consumer.name}
            else:
                raise CellCompileError(
                    f"cell {cell_name!r}: bias_tree external "
                    f"{n.name!r} feeds {consumer.name!r} of kind "
                    f"{consumer.kind!r}; only pmos_diode/nmos_diode "
                    f"are supported as direct downstream consumers "
                    f"in this MVP."
                )
            devices.append(Device(
                instance_name=f"I_{n.name}",
                device_class="cur_src",
                pdk_variant=None,
                pins=pins,
                sizing={"value": float(n.value or 0.0)},
            ))
        elif n.kind == "pmos_diode":
            devices.append(Device(
                instance_name=f"Mbp_{n.name}",
                device_class="pmos",
                pdk_variant=None,
                pins={"d": n.name, "g": n.name, "s": "vdd", "b": "vdd"},
                sizing={
                    "W": _BIAS_CHAIN_DEFAULT_W,
                    "L": _BIAS_CHAIN_DEFAULT_L,
                    "M": 1.0,
                },
            ))
        elif n.kind == "nmos_diode":
            devices.append(Device(
                instance_name=f"Mbn_{n.name}",
                device_class="nmos",
                pdk_variant=None,
                pins={"d": n.name, "g": n.name, "s": "vss", "b": "vss"},
                sizing={
                    "W": _BIAS_CHAIN_DEFAULT_W,
                    "L": _BIAS_CHAIN_DEFAULT_L,
                    "M": 1.0,
                },
            ))
            src_kind = by_name[n.src].kind if (n.src and n.src in by_name) else None
            if src_kind in ("pmos_diode", "pmos_mirror"):
                devices.append(Device(
                    instance_name=f"Maux_{n.name}",
                    device_class="pmos",
                    pdk_variant=None,
                    pins={"d": n.name, "g": n.src, "s": "vdd", "b": "vdd"},
                    sizing={
                        "W": _BIAS_CHAIN_DEFAULT_W,
                        "L": _BIAS_CHAIN_DEFAULT_L,
                        "M": 1.0,
                    },
                ))
        elif n.kind == "pmos_mirror":
            devices.append(Device(
                instance_name=f"M_{n.name}",
                device_class="pmos",
                pdk_variant=None,
                pins={"d": n.name, "g": n.src or n.name, "s": "vdd", "b": "vdd"},
                sizing={
                    "W": _BIAS_CHAIN_DEFAULT_W,
                    "L": _BIAS_CHAIN_DEFAULT_L,
                    "M": float(n.by_M or 1),
                },
            ))
        elif n.kind == "nmos_mirror":
            devices.append(Device(
                instance_name=f"M_{n.name}",
                device_class="nmos",
                pdk_variant=None,
                pins={"d": n.name, "g": n.src or n.name, "s": "vss", "b": "vss"},
                sizing={
                    "W": _BIAS_CHAIN_DEFAULT_W,
                    "L": _BIAS_CHAIN_DEFAULT_L,
                    "M": float(n.by_M or 1),
                },
            ))
        else:
            raise CellCompileError(
                f"cell {cell_name!r}: unknown bias_tree kind "
                f"{n.kind!r} for node {n.name!r}."
            )
    return devices


def _validate_primitive_device_class(
    dev: Device, *, pdk_id: str, cell_name: str,
) -> None:
    """Reject device kinds that the active PDK does not know about.

    Catches `device M1 : pomos { ... }` typos at compile time (the
    parser is PDK-agnostic by design, so this check has to happen
    later). The PDK adapter's `device_bindings` map defines the set of
    valid MOS-family kinds; passives (`cap`, `res`, `cur_src`, etc.)
    are universally supported by the driver layer and don't need a
    binding entry.
    """
    cls = dev.device_class
    if cls == "native":
        from clay.frontend.native_spice_import import validate_native_device
        try:
            validate_native_device(dev, pdk_id=pdk_id)
        except (TypeError, ValueError) as exc:
            raise CellCompileError(f"cell {cell_name!r}: {exc}") from exc
        return
    if cls in ("cap", "res", "ind", "cur_src", "vcvs", "pnp", "npn"):
        return
    try:
        pdk = get_pdk(pdk_id)
    except KeyError:
        # PDK not registered — let the driver complain later. The
        # primitive path stays usable for unit tests that don't have
        # a real PDK behind them.
        return
    if cls not in pdk.device_bindings:
        raise CellCompileError(
            f"cell {cell_name!r}: device {dev.instance_name!r} declares "
            f"unknown device kind {cls!r}. Valid kinds for PDK "
            f"{pdk_id!r}: {sorted(pdk.device_bindings)}, plus the "
            f"PDK-agnostic passives (cap, res, ind, cur_src, vcvs)."
        )


def _net_pin_counts(devices: list[Device]) -> dict[str, int]:
    """Return a `{net_name: how_many_device_pins_touch_it}` map.

    Used by the dangling-net warning path: a net that appears on only
    ONE device pin is potentially dangling (single-driver or single-
    consumer). Multi-pin nets are internal connectivity and harmless.
    Each pin counts once even if the same device touches the same net
    via multiple pins (rare but possible — e.g. a diode-connected
    transistor where d == g; we treat those as two pin-touches).
    """
    counts: dict[str, int] = {}
    for dev in devices:
        for net in dev.pins.values():
            counts[net] = counts.get(net, 0) + 1
    return counts


# Global rails treated as always-legal targets even if not declared as
# cell ports. SPICE node `0` is the universal ground reference.
_GLOBAL_RAILS: frozenset[str] = frozenset({"0", "gnd"})


def _compile_primitive_cell(cd: CellDef) -> Cell:
    """Build a Cell IR by collecting `cd.devices` + materializing
    `cd.bias_tree` — no stage builder, no sub-cell flatten.

    This is the path that lets an LLM describe a brand-new topology
    fully in `.clay` text. The compiler:
      1. Validates each device's class against the PDK adapter
         (unknown kinds → error).
      2. Materializes the bias_tree (if any) into additional Devices,
         re-using the same kind-rules as the
         `two_stage_miller_pmos_explicit_bias` stage so the visible
         bias-rail behavior is identical.
      3. Warns on dangling nets — pin targets that are neither a
         declared port nor a global rail nor a bias-tree node.
      4. Bundles everything into a `Cell` IR.

    PDK id is derived from the enclosing ClayFile's `pdk_decl` (the
    parser stashes a back-reference on the CellDef via
    `_attach_clay_file_to_cell_defs`, exposed as
    `cd.__dict__['_clay_file']`). If the file omits the declaration —
    or the CellDef was hand-constructed without going through the
    parser — the validator falls back to `'sky130'` so the pre-v0.0.46
    default behavior is preserved.
    """
    # v0.0.46: resolve the PDK id from the enclosing ClayFile rather
    # than hardcoding sky130. Without this lookup the gf180mcu cells
    # that use higher-Vth devices like `nmos_06v` / `pmos_06v` would
    # be rejected by `_validate_primitive_device_class` even though
    # the gf180mcu PDK adapter declares those bindings.
    clay_file: Optional[ClayFile] = cd.__dict__.get("_clay_file")
    if clay_file is not None and clay_file.pdk_id:
        pdk_id = clay_file.pdk_id
    else:
        pdk_id = cd.__dict__.get("_clay_file_pdk_id", "sky130")

    # Defensive guard: if the dispatcher routed here, devices must be
    # non-empty. (This function is only called from
    # `compile_cell_def`, which checks `cd.devices` first.)
    if not cd.devices:
        raise CellCompileError(
            f"cell {cd.local_name!r}: primitive-cell path invoked with "
            f"no `device` declarations — this is an internal bug."
        )

    # 1. Validate each declared device's class.
    for dev in cd.devices:
        _validate_primitive_device_class(
            dev, pdk_id=pdk_id, cell_name=cd.local_name,
        )

    # 2. Assemble the device list. Start with the user's explicit
    # devices, then append bias-tree materializations.
    all_devices: list[Device] = list(cd.devices)

    bias_tree_nodes: set[str] = set()
    if cd.bias_tree:
        bias_devs = _materialize_bias_tree_devices(
            cd.bias_tree, cell_name=cd.local_name,
        )
        all_devices.extend(bias_devs)
        bias_tree_nodes = {n.name for n in cd.bias_tree}

    # Final duplicate-name check across both lists. The parser already
    # rejects within-source duplicates; this guards against
    # bias-tree-generated names colliding with user devices.
    seen_names: set[str] = set()
    for dev in all_devices:
        if dev.instance_name in seen_names:
            raise CellCompileError(
                f"cell {cd.local_name!r}: device instance name "
                f"{dev.instance_name!r} is used more than once "
                f"(possibly a collision between an explicit `device` "
                f"and a name synthesized by the bias_tree)."
            )
        seen_names.add(dev.instance_name)

    # 3. Dangling-net warning. A pin target is potentially dangling
    # when it touches only one device pin AND is not a cell port,
    # a global rail, or a bias-tree node. Multi-pin internal nets
    # (`tail`, `d1`, etc.) are normal connectivity and ignored.
    pin_counts = _net_pin_counts(all_devices)
    legal_nets = set(cd.ports) | _GLOBAL_RAILS | bias_tree_nodes
    dangling = sorted(
        net for net, count in pin_counts.items()
        if count == 1 and net not in legal_nets
    )
    if dangling:
        # DANGLING-NET POLICY (resolved). The diagnostic still emits as a
        # soft `UserWarning` here so the LLM gets an immediate heads-up
        # and the compile path stays usable, but the warning is NO LONGER
        # the authoritative verdict: an accidental dangling net is a
        # first-class, VISIBLE structural obligation with a non-PASS
        # status. The proof-token layer reads these nets back via
        # `clay.physical_invariants.dangling_net_obligation_rows(cell)`
        # (driven off the `_dangling_nets` stash + the declared ports),
        # where each net becomes a REQUIRED `dangling_net` obligation
        # that blocks PASS and never folds into it. The exemption for an
        # intended-open net is to declare it as a cell port (or a global
        # rail / bias node) — that removes it from `dangling` above.
        # See docs/dangling_net_policy.html.
        import warnings
        warnings.warn(
            f"cell {cd.local_name!r}: nets touched by only one device "
            f"pin AND not in (ports, vdd/vss/0/gnd, bias_tree): "
            f"{dangling}. Likely typos.",
            stacklevel=2,
        )

    # Port-name sanity check — same shape rule as the stage-template
    # path. Empty ports list is rejected (a useful cell needs at least
    # one port). vdd/vss are conventional; if a device references them,
    # the cell should declare them too.
    if not cd.ports:
        raise CellCompileError(
            f"cell {cd.local_name!r}: primitive cells must declare at "
            f"least one port. An entirely floating cell isn't useful."
        )

    cell = Cell(
        name=cd.local_name,
        ports=list(cd.ports),
        spec=cd.spec,
        devices=all_devices,
        pdk_id=pdk_id,
    )
    # Stash the dangling-net list on the cell for test introspection.
    # Public API surface is unchanged.
    if dangling:
        cell.__dict__["_dangling_nets"] = dangling
    return cell


# ─── v0.0.92 (combinator-in-source, Round 68) ─────────────────────


def _compile_combinator_cell(cd: CellDef) -> Cell:
    """Lower a `combinator { ... }` block from a CellDef to a Cell IR.

    Steps:
      1. Resolve every `stage_ref` / `stage_name` AST leaf to a
         registered StageInterface via the entry-point registry.
      2. Walk the build expression bottom-up, applying combinator
         operators from `clay.stage_combinators.*` to produce a
         `ComposedStage`.
      3. Lower the ComposedStage to a builder via
         `lower_to_template(...)`.
      4. Call the builder with the CellDef's spec + sizing kwargs and
         return the resulting Cell IR. Forwarded sizing kwargs include
         `auto_bias=...` and per-stage knobs (`s1_W`, `s2_gm_id`, ...).

    Raises `CellCompileError` if:
      * Any stage reference fails to resolve (unknown template name).
      * The combinator algebra rejects the composition (type errors
        from `check_compose`, infinite-gain feedforward, etc.).
      * The builder rejects a sizing kwarg.
    """
    block = cd.combinator_block
    if block is None:
        raise CellCompileError(
            f"cell {cd.local_name!r}: _compile_combinator_cell called "
            f"with no combinator_block — internal dispatch error."
        )
    # Defer the heavy combinator import until we need it. This keeps
    # the existing compile paths (stage / composition / primitive)
    # free of any combinator-runtime cost.
    from .stage_combinators import (
        cascade as _cascade,
        cascade_chain as _cascade_chain,
        compensate as _compensate,
        decouple as _decouple,
        feedforward as _feedforward,
        mirror_match as _mirror_match,
        sample_and_hold as _sample_and_hold,
        lower_to_template as _lower_to_template,
    )
    from .stage_interface import get_stage_interface, has_stage_interface

    def _resolve_stage_token(token: str) -> object:
        """Look up a stage template_name in the registry and return
        the StageInterface. Raises CellCompileError on miss."""
        if not has_stage_interface(token):
            raise CellCompileError(
                f"cell {cd.local_name!r} combinator {block.name!r}: "
                f"stage template {token!r} is not registered. "
                f"Available templates: "
                f"see clay.stage_interface.get_stage_interface."
            )
        return get_stage_interface(token)

    def _eval(node: Any) -> Any:
        """Recursively evaluate the build_expr AST against the
        declared stages list, returning a ComposedStage (or a
        StageInterface for leaf cases)."""
        if not isinstance(node, dict):
            raise CellCompileError(
                f"cell {cd.local_name!r} combinator {block.name!r}: "
                f"malformed build expression — expected a dict node, "
                f"got {type(node).__name__}"
            )
        kind = node.get("kind")
        if kind == "stage_ref":
            idx = node["index"]
            template_name = block.stages[idx]
            return _resolve_stage_token(template_name)
        if kind == "stage_name":
            return _resolve_stage_token(node["name"])
        if kind == "cascade":
            left = _eval(node["left"])
            right = _eval(node["right"])
            try:
                return _cascade(left, right)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"cascade composition failed — {exc}"
                ) from exc
        if kind == "feedforward":
            source = _eval(node["source"])
            sink = _eval(node["sink"])
            kwargs = node.get("kwargs") or {}
            try:
                return _feedforward(source, sink, **kwargs)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"feedforward composition failed — {exc}"
                ) from exc
        if kind == "compensate":
            inner = _eval(node["inner"])
            kwargs = node.get("kwargs") or {}
            try:
                return _compensate(inner, **kwargs)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"compensate composition failed — {exc}"
                ) from exc
        if kind == "mirror_match":
            left = _eval(node["left"])
            right = _eval(node["right"])
            try:
                return _mirror_match(left, right)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"mirror_match composition failed — {exc}"
                ) from exc
        if kind == "cascade_chain":
            stage_exprs = [_eval(s) for s in node["stages"]]
            try:
                return _cascade_chain(stage_exprs)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"cascade_chain composition failed — {exc}"
                ) from exc
        if kind == "decouple":
            # v0.0.95 (Round 72 D): supply-rail decoupling combinator.
            inner = _eval(node["inner"])
            kwargs = node.get("kwargs") or {}
            try:
                return _decouple(inner, **kwargs)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"decouple composition failed — {exc}"
                ) from exc
        if kind == "sample_and_hold":
            # v0.0.99 (Round 75 C): track-and-hold combinator.
            inner = _eval(node["inner"])
            kwargs = node.get("kwargs") or {}
            try:
                return _sample_and_hold(inner, **kwargs)
            except Exception as exc:
                raise CellCompileError(
                    f"cell {cd.local_name!r} combinator {block.name!r}: "
                    f"sample_and_hold composition failed — {exc}"
                ) from exc
        raise CellCompileError(
            f"cell {cd.local_name!r} combinator {block.name!r}: "
            f"AST node has unknown kind {kind!r}."
        )

    composed = _eval(block.build_expr)
    builder = _lower_to_template(composed)

    try:
        cell = builder(
            cd.local_name,
            cd.spec,
            **cd.sizing_kwargs,
        )
    except TypeError as exc:
        raise CellCompileError(
            f"cell {cd.local_name!r} combinator {block.name!r}: "
            f"sizing kwarg rejected by lowered combinator builder: {exc}"
        ) from exc
    except ValueError as exc:
        raise CellCompileError(
            f"cell {cd.local_name!r} combinator {block.name!r}: "
            f"lowered combinator builder rejected build inputs: {exc}"
        ) from exc
    # Sanity-check port shape — only the names matter (roles are
    # informational, mirroring the stage-template path).
    if cd.ports and set(cd.ports) != set(cell.ports):
        raise CellCompileError(
            f"cell {cd.local_name!r} combinator {block.name!r}: "
            f"declared ports {cd.ports} do not match the composed "
            f"cell's port list {cell.ports}."
        )
    return cell
