"""
Layout-aware sizing cost wrapper (R92 C / BL-127).

Closes the co-variable manifesto loop: the sizing algorithm — historically
blind to layout geometry beyond a per-device (W, L) — now sees layout's
finger_count as a joint-optimized variable through a single hook.

This module is a *consumer* of ``clay.layout`` (stable API since 8d6986b /
0b751a0) and a *companion* of ``clay.sizing.closed_form_W``. It exposes:

  * ``layout_aware_sizing_cost(sizing_cost, lambda_layout, lambda_area,
        cosizing_point, target_cmrr_db, pdk, device)``
      — single-call mixer that combines a sizing-domain cost (whatever
      the caller computes from gain / GBW / power / Vds head room / etc.)
      with the layout-domain mismatch cost from
      ``DefaultMismatchCostHook.mismatch_cost``.

  * ``select_layout_aware_sizing(w_range, l_range, finger_range, ...)``
      — Pareto + lambda-blend selector that turns a discrete (W, L,
      finger_count) grid into a single ``LayoutAwareSelection``
      summarising the chosen point + its sigma/area/blended cost.

  * ``LayoutAwareSelection`` — immutable result tuple.

  * ``DEFAULT_LAMBDA_LAYOUT`` / ``DEFAULT_LAMBDA_AREA`` /
    ``DEFAULT_TARGET_CMRR_DB`` / ``DEFAULT_FINGER_COUNT_RANGE`` —
    library-level defaults aligned with layout prompt's R007 §9.3
    recommendations (CROSS_PROMPT_SCOPE.html §10.2).

Lambda semantics (from layout prompt R007 §9.3):

  * ``lambda_layout = 0.0`` ⇒ layout cost not in total cost; sizing-
    domain cost dominates. Used as pipeline smoke + sizing-only baseline.
  * ``lambda_layout ∈ [0.01, 0.1]`` ⇒ layout cost takes effect as a
    moderate correction. ``finger_count`` emerges as the joint-optimized
    variable; W/L still tracked by sizing.
  * ``lambda_layout = 1.0`` ⇒ aggressive layout-priority. Smaller
    finger_count visible — sizing accepts a moderate sigma_mv to win
    layout-domain area.
  * ``lambda_layout = 10.0`` ⇒ near-pure layout-priority.

  * ``lambda_area = 0.0`` ⇒ min-σ Pareto extreme.
  * ``lambda_area = 0.1`` ⇒ interior balance (CROSS_PROMPT_SCOPE.html
    §10.2 demonstrates W=2, L=0.5, N=8 on Phase 4 ground-truth grid).
  * ``lambda_area ≥ 10.0`` ⇒ min-area Pareto extreme.

Reconciliation between layout finger_count selection vs sizing default:
  When ``lambda_layout = 0``, ``select_layout_aware_sizing`` returns the
  point that minimises sizing_cost only and reports finger_count as
  whatever the sweep grid's smallest finger choice was — typically 1 —
  which matches the historical sizing-only behaviour (no fingers, single
  device). When ``lambda_layout > 0``, layout selection overrides this
  default and the caller can wire ``selection.finger_count`` into the
  layout backend or — in lazy mode — leave it as a logical attribute on
  the cell IR. The R92 C demo on AZC Cazc4 demonstrates the override
  semantics without yet wiring back into the netlist (sizing-side
  selection only — finger fold-back is a R93+ task).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

# Import direct from submodules rather than via clay.layout.__init__ so
# that R92 layout-prompt mid-flight package surface (e.g. new
# `lower_annotated_cell_drc_complete` re-export pending a corresponding
# `backend.py` addition) does not break the sizing import chain. The
# submodules referenced here are stable since 8d6986b (cosizing) and
# 0b751a0 (sizing_hook / mismatch / pdk_dims / symbolic_layout).
from clay.layout.cosizing import (
    CosizingPoint,
    joint_sweep,
    pareto_optimal,
    select_optimal,
)
from clay.layout.pdk_dims import DeviceType, PdkDims, Sky130
from clay.layout.sizing_hook import DefaultMismatchCostHook
from clay.layout.symbolic_layout import SymbolicLayout

# R96 E / M-PDK-DECOUPLE — sizing-side PdkDims context surface.
# Imported under the explicit ``PdkDimsContext`` alias to disambiguate
# from the layout-side ``clay.layout.pdk_dims.PdkDims`` ABC already
# imported above. The two surfaces coexist in distinct modules; this
# alias is the same one ``clay.sizing.__init__`` exports for callers.
from clay.pdk_dims import PdkDims as PdkDimsContext

# `Cell` / `Device` are imported lazily inside `apply_finger_count_to_role`
# to keep the module's import chain identical to R92 C (no new top-level
# `clay.ir` dependency).


# ──────────────────────────────────────────────────────────────
# Library defaults (CROSS_PROMPT_SCOPE.html §9.3 + §10.2)
# ──────────────────────────────────────────────────────────────

#: Initial wire-up value. With layout cost weighted 0 the sizing-only
#: convergence is preserved byte-equal — pipeline smoke without
#: disturbing existing sizing baselines.
DEFAULT_LAMBDA_LAYOUT: float = 0.0

#: Phase 4 ground-truth recommended middle default. Below this the min-σ
#: extreme wastes area; above ~0.5 the min-area extreme sacrifices
#: mismatch spec.
DEFAULT_LAMBDA_AREA: float = 0.1

#: Default CMRR target used by ``DefaultMismatchCostHook``. AZC Cazc4
#: cohort targets > 80 dB CMRR (AnalogGym Qu2017_AZC). Below that, the
#: ``mismatch_cost`` saturates at zero.
DEFAULT_TARGET_CMRR_DB: float = 80.0

#: Default finger sweep grid. Powers-of-two from 1 to 16 cover the
#: practical layout space for analog input pairs / mirror loads.
DEFAULT_FINGER_COUNT_RANGE: tuple[int, ...] = (1, 2, 4, 8, 16)


# ──────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LayoutAwareSelection:
    """Joint (W, L, finger_count) selection result.

    Attributes
    ----------
    w_um, l_um, finger_count:
        Physical sizing of the chosen Pareto point. ``finger_count``
        comes from the layout sweep grid; for sizing-only baselines
        (``lambda_layout = 0``) it is the grid's smallest member —
        typically 1.
    sigma_mv:
        Pelgrom Vth mismatch sigma in mV at the chosen point.
    area_um2:
        ``w_um * l_um * finger_count`` — active area in um².
    layout_cost:
        Layout-domain mismatch cost from ``DefaultMismatchCostHook``
        at the chosen point. Higher = better CMRR head room.
    sizing_cost_only:
        Caller-supplied sizing-domain cost for the chosen point.
        Lower = better fit to spec. Stored for downstream
        diagnostics.
    blended_cost:
        ``sizing_cost_only + lambda_layout * layout_cost``. The metric
        the selector minimised.
    lambda_layout, lambda_area:
        The mix weights actually used. Mirrored back so the caller
        can record the sweep parameters in a log without re-keying.
    pareto_size:
        Length of the Pareto front the selector chose from. Useful
        for diagnosing degenerate single-point fronts (sweep grid
        collapsed) vs healthy multi-point fronts.
    pdk_aware_cost:
        Optional ``PdkDimsAwareCost`` envelope populated when the
        selector is called with ``pdk_dims_ctx=<PdkDimsContext>``
        (R97 selector-path threading). Carries PDK-relative metadata
        (``n_min_w_multiple`` / ``n_min_l_multiple`` / ``area_si_m2`` /
        ``pdk_dims_name``) for the chosen point so cross-PDK ratchets
        can recover same-recipe / different-SI / same-multiplier
        evidence directly from the selection without re-running the
        cost mixer. ``None`` (default) when the caller did not pass a
        ``pdk_dims_ctx`` -- preserves R94 B + R95 B + R95 E
        ``LayoutAwareSelection`` byte-exact.
    """

    w_um: float
    l_um: float
    finger_count: int
    sigma_mv: float
    area_um2: float
    layout_cost: float
    sizing_cost_only: float
    blended_cost: float
    lambda_layout: float
    lambda_area: float
    pareto_size: int
    pdk_aware_cost: Optional["PdkDimsAwareCost"] = None


# ──────────────────────────────────────────────────────────────
# Core API: single-point cost mixer
# ──────────────────────────────────────────────────────────────

def layout_aware_sizing_cost(
    sizing_cost: float,
    *,
    lambda_layout: float,
    cosizing_point: CosizingPoint,
    target_cmrr_db: float = DEFAULT_TARGET_CMRR_DB,
    pdk: Optional[PdkDims] = None,
    device: DeviceType = DeviceType.NMOS,
    hook: Optional[DefaultMismatchCostHook] = None,
) -> float:
    """Mix a sizing-domain cost with the layout-domain mismatch cost.

    Args:
        sizing_cost: caller-supplied sizing-domain cost. Lower is
            better. Whatever the caller's sizing solver computes from
            gain / GBW / phase margin / power — only the relative
            ordering across candidates matters here.
        lambda_layout: weight on the layout-domain contribution. See
            module docstring for sweep semantics.
        cosizing_point: ``(w_um, l_um, finger_count, sigma_mv,
            area_um2)`` candidate — typically pulled from a
            ``joint_sweep`` / ``pareto_optimal`` walk.
        target_cmrr_db: target CMRR in dB used by
            ``DefaultMismatchCostHook``. Default 80 dB.
        pdk: PDK dimensional surface (default ``Sky130()``).
        device: NMOS / PMOS — default NMOS.
        hook: pre-instantiated mismatch hook (default
            ``DefaultMismatchCostHook()``). Pass to override the
            CMRR-derate semantics.

    Returns:
        ``sizing_cost + lambda_layout * layout_cost``. When
        ``lambda_layout = 0`` the return value is byte-equal to
        ``sizing_cost`` (no layout contribution).

    Raises:
        ValueError: if ``lambda_layout < 0`` (negative weights would
            invert the mismatch-vs-cost intent).
    """
    if lambda_layout < 0:
        raise ValueError(
            f"lambda_layout must be >= 0 (negative weights invert the "
            f"mismatch-vs-cost intent), got {lambda_layout}"
        )

    if lambda_layout == 0.0:
        # Bypass the hook to guarantee byte-exact preservation of the
        # sizing-only baseline. Phase 4 ground truth verified that
        # this branch returns sizing_cost as-is.
        return sizing_cost

    pdk_eff: PdkDims = pdk if pdk is not None else Sky130()
    hook_eff = hook if hook is not None else DefaultMismatchCostHook()

    layout = SymbolicLayout(finger_count=cosizing_point.finger_count)
    layout_cost = hook_eff.mismatch_cost(
        layout=layout,
        w_um=cosizing_point.w_um,
        l_um=cosizing_point.l_um,
        pdk=pdk_eff,
        device=device,
        target_cmrr_db=target_cmrr_db,
    )

    return sizing_cost + lambda_layout * layout_cost


# ──────────────────────────────────────────────────────────────
# R96 E / M-PDK-DECOUPLE — PdkDimsContext-aware cost surface
# ──────────────────────────────────────────────────────────────
#
# R95 E landed the sizing-side ``PdkDims`` (re-exported as
# ``PdkDimsContext`` from ``clay.sizing``) which carries the per-PDK
# process constants the .clay author cares about when expressing W/L as
# multiples of ``min_W`` / ``min_L``. R94 B's ``layout_aware_sizing_cost``
# mixed a sizing-domain scalar with the layout-domain mismatch cost but
# carried zero knowledge of those PDK-natural units: it accepted
# ``w_um`` / ``l_um`` floats in absolute µm and treated the area term
# ``w_um * l_um * finger_count`` as raw µm² regardless of PDK identity.
#
# R96 E closes that gap. ``layout_aware_sizing_cost_with_pdk_dims``
# accepts an optional ``pdk_dims_ctx: PdkDimsContext`` (the sizing-side
# dataclass — *not* the layout-side ABC). When provided, the function:
#
#   1. Echoes the PdkDimsContext into the result so downstream tooling
#      (e.g. multi-PDK ratchets, paper figures) can show the same
#      recipe under different PDKs.
#   2. Computes the area term in PDK-natural multiples of ``min_W *
#      min_L * m_unit`` rather than raw µm². This is the invariant that
#      lets the same .clay recipe (``W = 3 × min_W``, ``L = 2 ×
#      default_L``) yield a ``N_min_W`` multiplier that is identical on
#      sky130 and gf180mcu even though the SI metres are different.
#   3. Keeps the mismatch-cost call identical to R94 B (the layout-side
#      ``PdkDims`` Pelgrom surface answers the mismatch question; the
#      sizing-side ``PdkDimsContext`` answers the dimensional-units
#      question). The two surfaces compose without interference.
#
# When ``pdk_dims_ctx`` is None, the function behaves byte-equal to
# ``layout_aware_sizing_cost`` — same default Sky130 layout-side PDK,
# same lambda_layout = 0 short-circuit, same numeric return value. This
# preserves the R94 B + R95 B + R95 E APIs byte-exact.


@dataclass(frozen=True)
class PdkDimsAwareCost:
    """Cost-surface descriptor for ``layout_aware_sizing_cost_with_pdk_dims``.

    Attributes
    ----------
    blended_cost:
        ``sizing_cost + lambda_layout * layout_cost`` — the numeric
        return value of the cost mixer. Byte-equal to the float that
        ``layout_aware_sizing_cost`` would have returned for the same
        ``(sizing_cost, lambda_layout, cosizing_point)`` inputs.
    n_min_w_multiple:
        ``cosizing_point.w_um * 1e-6 / pdk_dims_ctx.min_W`` — the
        PDK-natural width multiplier (dimensionless). Identical for
        equivalent recipes across different PDK contexts. ``None`` when
        ``pdk_dims_ctx`` is None.
    n_min_l_multiple:
        ``cosizing_point.l_um * 1e-6 / pdk_dims_ctx.min_L`` — analogous
        multiplier for length. ``None`` when ``pdk_dims_ctx`` is None.
    area_si_m2:
        ``w * l * m_unit`` in SI m², where ``m_unit = pdk_dims_ctx.
        m_unit`` (matches the layout-finger semantic for parallel
        unit transistors). ``None`` when ``pdk_dims_ctx`` is None.
    pdk_dims_name:
        Echo of ``pdk_dims_ctx.pdk_name`` so multi-PDK ratchets / log
        plots can render per-PDK breakdowns without re-keying. ``None``
        when ``pdk_dims_ctx`` is None.
    """

    blended_cost: float
    n_min_w_multiple: Optional[float]
    n_min_l_multiple: Optional[float]
    area_si_m2: Optional[float]
    pdk_dims_name: Optional[str]


def layout_aware_sizing_cost_with_pdk_dims(
    sizing_cost: float,
    *,
    lambda_layout: float,
    cosizing_point: CosizingPoint,
    target_cmrr_db: float = DEFAULT_TARGET_CMRR_DB,
    pdk: Optional[PdkDims] = None,
    device: DeviceType = DeviceType.NMOS,
    hook: Optional[DefaultMismatchCostHook] = None,
    pdk_dims_ctx: Optional[PdkDimsContext] = None,
) -> PdkDimsAwareCost:
    """PdkDimsContext-aware variant of ``layout_aware_sizing_cost``.

    Wraps the R94 B single-point cost mixer with the R95 E sizing-side
    ``PdkDimsContext`` to expose area = ``W × L × m_unit`` in PDK-natural
    units alongside the byte-equal blended scalar. The blended cost
    itself is unchanged from ``layout_aware_sizing_cost``: this function
    *augments* the return value with PDK-relative metadata so multi-PDK
    sweeps can show that "the same recipe on sky130 vs gf180mcu" picks
    different SI µm but identical ``N_min_W`` multipliers.

    Args:
        sizing_cost: caller-supplied sizing-domain cost. Lower is better.
        lambda_layout: weight on the layout-domain contribution
            (semantics identical to ``layout_aware_sizing_cost``).
        cosizing_point: ``(w_um, l_um, finger_count, sigma_mv,
            area_um2)`` candidate.
        target_cmrr_db: target CMRR in dB used by the mismatch hook.
        pdk: layout-side ``PdkDims`` (Sky130 default). Distinct from
            ``pdk_dims_ctx`` — see module docstring.
        device: NMOS / PMOS.
        hook: pre-instantiated mismatch hook.
        pdk_dims_ctx: sizing-side ``PdkDimsContext`` (the R95 E
            dataclass). When provided, the result carries ``n_min_w_
            multiple`` / ``n_min_l_multiple`` / ``area_si_m2`` /
            ``pdk_dims_name`` populated from it. When ``None``, those
            fields are ``None`` and the result's ``blended_cost`` is
            byte-equal to the float ``layout_aware_sizing_cost`` would
            have returned.

    Returns:
        ``PdkDimsAwareCost`` carrying the blended scalar + per-PDK
        dimensional metadata. ``result.blended_cost`` is the
        byte-equal R94 B numeric.

    Raises:
        ValueError: ``lambda_layout < 0`` (forwarded from the R94 B
            single-point mixer).
    """
    # Delegate the blended scalar computation to the R94 B function —
    # byte-equal preservation by construction.
    blended = layout_aware_sizing_cost(
        sizing_cost,
        lambda_layout=lambda_layout,
        cosizing_point=cosizing_point,
        target_cmrr_db=target_cmrr_db,
        pdk=pdk,
        device=device,
        hook=hook,
    )

    if pdk_dims_ctx is None:
        # No PDK-relative metadata requested — return the blended
        # scalar in a result envelope with the per-PDK fields nulled.
        return PdkDimsAwareCost(
            blended_cost=blended,
            n_min_w_multiple=None,
            n_min_l_multiple=None,
            area_si_m2=None,
            pdk_dims_name=None,
        )

    # Compute PDK-natural unit metadata. cosizing_point.w_um / l_um are
    # in µm; PdkDimsContext.min_W / min_L are in SI metres.
    w_si = cosizing_point.w_um * 1e-6
    l_si = cosizing_point.l_um * 1e-6
    n_min_w = w_si / pdk_dims_ctx.min_W
    n_min_l = l_si / pdk_dims_ctx.min_L
    # area = W * L * m_unit (SI m²). m_unit is the multiplicity carried
    # by the PdkDimsContext — distinct from the cosizing_point's
    # finger_count which is the layout-side choice. The .clay author
    # composes the two by either setting m_unit explicitly (e.g.
    # ``PdkDims.sky130().with_m(4)``) OR by letting the layout sweep
    # pick finger_count. Either way, the area-in-SI value reflects what
    # the resulting cell physically occupies.
    area_si = w_si * l_si * float(pdk_dims_ctx.m_unit)

    return PdkDimsAwareCost(
        blended_cost=blended,
        n_min_w_multiple=n_min_w,
        n_min_l_multiple=n_min_l,
        area_si_m2=area_si,
        pdk_dims_name=pdk_dims_ctx.pdk_name,
    )


# ──────────────────────────────────────────────────────────────
# Core API: Pareto sweep + lambda-blend selector
# ──────────────────────────────────────────────────────────────

def select_layout_aware_sizing(
    *,
    w_range: Sequence[float],
    l_range: Sequence[float],
    finger_count_range: Sequence[int] = DEFAULT_FINGER_COUNT_RANGE,
    pdk: Optional[PdkDims] = None,
    device: DeviceType = DeviceType.NMOS,
    lambda_layout: float = DEFAULT_LAMBDA_LAYOUT,
    lambda_area: float = DEFAULT_LAMBDA_AREA,
    target_cmrr_db: float = DEFAULT_TARGET_CMRR_DB,
    sizing_cost_fn: Optional[Callable[[float, float, int], float]] = None,
    hook: Optional[DefaultMismatchCostHook] = None,
    pdk_dims_ctx=None,
    cell=None,
) -> Optional[LayoutAwareSelection]:
    """Joint (W, L, finger_count) Pareto sweep with lambda-blend selector.

    Args:
        w_range: candidate widths in um.
        l_range: candidate lengths in um.
        finger_count_range: candidate finger counts. Default
            ``(1, 2, 4, 8, 16)``.
        pdk: PDK dimensional surface (default ``Sky130()``).
        device: NMOS / PMOS — default NMOS.
        lambda_layout: weight on the layout-domain blend (see module
            docstring). Default 0.0 (sizing-only baseline).
        lambda_area: weight on area inside the Pareto selector. Default
            0.1 (CROSS_PROMPT_SCOPE.html §10.2 recommended interior
            balance).
        target_cmrr_db: target CMRR for ``DefaultMismatchCostHook``.
        sizing_cost_fn: optional callable ``(w_um, l_um, finger_count)
            -> sizing_cost``. When provided, its value is mixed via
            ``layout_aware_sizing_cost`` with ``lambda_layout``. When
            None, the selector reduces to ``select_optimal`` with
            ``cost = sigma_mv + lambda_area * area_um2`` — preserves
            Phase 4 ground-truth behaviour.
        hook: optional pre-instantiated mismatch hook.
        pdk_dims_ctx: optional sizing-side ``PdkDimsContext`` (R95 E
            dataclass: ``min_W`` / ``min_L`` / ``default_W`` /
            ``default_L`` / ``m_unit`` / ``pdk_name``). R97 selector-
            path threading: when supplied, the returned selection
            carries a ``pdk_aware_cost`` ``PdkDimsAwareCost`` envelope
            populated from the chosen point's ``(w_um, l_um)`` via the
            R96 E ``layout_aware_sizing_cost_with_pdk_dims`` mixer.
            When ``None`` (default), ``pdk_aware_cost`` stays ``None``
            and the returned selection is byte-equal to R94 B + R95 B
            + R95 E (additive API preservation).
            R98 B addition: pass ``"auto"`` to trigger
            ``_pdk_dims_ctx_from_cell(cell)`` resolution. The cell
            argument MUST also be supplied for the auto-resolution to
            work; when ``cell`` is ``None`` and ``pdk_dims_ctx="auto"``,
            the resolution silently falls back to ``None`` (R97 B
            byte-equal). Explicit ``None`` / ``PdkDimsContext`` values
            still preserve R97 B byte-equal.
        cell: optional ``clay.ir.Cell`` / ``CellDef`` used solely as the
            introspection target for ``pdk_dims_ctx="auto"`` resolution.
            R98 B selector-path automation: when ``pdk_dims_ctx="auto"``
            this argument feeds ``_pdk_dims_ctx_from_cell(cell)``;
            otherwise it is ignored. Defaults to ``None`` -- preserves
            R97 B + R96 E + R95 E byte-equal call signatures.

    Returns:
        A ``LayoutAwareSelection`` capturing the chosen point and the
        cost it minimised. ``None`` only if the sweep produces an empty
        Pareto front (defensive — should not happen for non-empty
        ranges).

    Raises:
        ValueError: on negative ``lambda_layout`` or ``lambda_area``.
    """
    if lambda_layout < 0:
        raise ValueError(
            f"lambda_layout must be >= 0, got {lambda_layout}"
        )
    if lambda_area < 0:
        raise ValueError(
            f"lambda_area must be >= 0, got {lambda_area}"
        )

    # R98 B: resolve ``pdk_dims_ctx="auto"`` from the cell IR. ``None`` /
    # explicit ``PdkDimsContext`` pass through unchanged (R97 B byte-equal).
    pdk_dims_ctx = _resolve_pdk_dims_ctx_arg(pdk_dims_ctx, cell)

    pdk_eff: PdkDims = pdk if pdk is not None else Sky130()
    hook_eff = hook if hook is not None else DefaultMismatchCostHook()

    points = joint_sweep(
        list(w_range),
        list(l_range),
        list(finger_count_range),
        pdk_eff,
        device,
    )
    front = pareto_optimal(points)
    if not front:
        return None

    def _cost(p: CosizingPoint) -> float:
        # Phase 4 base cost: sigma + lambda_area * area.
        base = p.sigma_mv + lambda_area * p.area_um2
        if lambda_layout == 0.0:
            # Byte-equal Phase 4 sweep cost — sizing-only baseline.
            return base
        # Sizing-domain cost (caller-supplied; default 0 when absent).
        s = 0.0
        if sizing_cost_fn is not None:
            s = sizing_cost_fn(p.w_um, p.l_um, p.finger_count)
        layout_cost = hook_eff.mismatch_cost(
            layout=SymbolicLayout(finger_count=p.finger_count),
            w_um=p.w_um,
            l_um=p.l_um,
            pdk=pdk_eff,
            device=device,
            target_cmrr_db=target_cmrr_db,
        )
        # The sizing cost adds in directly; the layout contribution
        # multiplies by lambda_layout on top of the Phase 4 base. This
        # keeps the Pareto-base ordering (sigma + lambda_area * area)
        # dominant when lambda_layout is small, but lets larger
        # lambda_layout pull the selector toward minimum layout cost.
        return base + s + lambda_layout * layout_cost

    chosen = select_optimal(front, _cost)
    if chosen is None:
        return None

    # Compute display-time fields for the selected point.
    if lambda_layout == 0.0:
        layout_cost_val = 0.0
        sizing_cost_val = 0.0
    else:
        sizing_cost_val = (
            sizing_cost_fn(chosen.w_um, chosen.l_um, chosen.finger_count)
            if sizing_cost_fn is not None
            else 0.0
        )
        layout_cost_val = hook_eff.mismatch_cost(
            layout=SymbolicLayout(finger_count=chosen.finger_count),
            w_um=chosen.w_um,
            l_um=chosen.l_um,
            pdk=pdk_eff,
            device=device,
            target_cmrr_db=target_cmrr_db,
        )

    # R97 selector-path threading: populate the PdkDimsAwareCost
    # envelope from the chosen point when the caller passed a
    # PdkDimsContext. When ``pdk_dims_ctx`` is None the selection's
    # ``pdk_aware_cost`` stays at its default ``None`` value (byte-
    # equal R94 B + R95 B + R95 E).
    pdk_aware: Optional[PdkDimsAwareCost] = None
    if pdk_dims_ctx is not None:
        # Reuse the R96 E single-point cost mixer to populate the
        # envelope. ``sizing_cost`` is the selected point's
        # ``sizing_cost_only`` so the envelope's ``blended_cost``
        # matches the selection's ``sizing_cost_only + lambda_layout
        # * layout_cost`` decomposition (the same numeric scalar the
        # R96 E cost mixer would have returned for this point in
        # isolation).
        pdk_aware = layout_aware_sizing_cost_with_pdk_dims(
            sizing_cost=sizing_cost_val,
            lambda_layout=lambda_layout,
            cosizing_point=chosen,
            target_cmrr_db=target_cmrr_db,
            pdk=pdk_eff,
            device=device,
            hook=hook_eff,
            pdk_dims_ctx=pdk_dims_ctx,
        )

    return LayoutAwareSelection(
        w_um=chosen.w_um,
        l_um=chosen.l_um,
        finger_count=chosen.finger_count,
        sigma_mv=chosen.sigma_mv,
        area_um2=chosen.area_um2,
        layout_cost=layout_cost_val,
        sizing_cost_only=sizing_cost_val,
        blended_cost=_cost(chosen),
        lambda_layout=lambda_layout,
        lambda_area=lambda_area,
        pareto_size=len(front),
        pdk_aware_cost=pdk_aware,
    )


# ──────────────────────────────────────────────────────────────
# Diagnostic helpers (used by demo + tests)
# ──────────────────────────────────────────────────────────────

def sweep_lambda_layout(
    *,
    w_range: Sequence[float],
    l_range: Sequence[float],
    finger_count_range: Sequence[int] = DEFAULT_FINGER_COUNT_RANGE,
    lambda_layout_grid: Iterable[float] = (0.0, 0.01, 0.1, 1.0, 10.0),
    pdk: Optional[PdkDims] = None,
    device: DeviceType = DeviceType.NMOS,
    lambda_area: float = DEFAULT_LAMBDA_AREA,
    target_cmrr_db: float = DEFAULT_TARGET_CMRR_DB,
    sizing_cost_fn: Optional[Callable[[float, float, int], float]] = None,
    hook: Optional[DefaultMismatchCostHook] = None,
    pdk_dims_ctx=None,
    cell=None,
) -> List[LayoutAwareSelection]:
    """Run ``select_layout_aware_sizing`` across a lambda_layout grid.

    Returns one ``LayoutAwareSelection`` per ``lambda_layout`` value, in
    the order of ``lambda_layout_grid``. Empty-Pareto entries are
    skipped silently — defensive in case a sweep grid collapses.

    R97 selector-path threading: ``pdk_dims_ctx`` (default ``None``) is
    forwarded to every per-lambda ``select_layout_aware_sizing`` call.
    When supplied, every returned selection's ``pdk_aware_cost``
    field carries the per-PDK ``PdkDimsAwareCost`` envelope. When
    ``None``, every selection's ``pdk_aware_cost`` stays ``None`` --
    byte-equal R94 B + R95 B + R95 E.

    R98 B addition: ``pdk_dims_ctx="auto"`` triggers
    ``_pdk_dims_ctx_from_cell(cell)`` resolution ONCE at the sweep
    entry (rather than re-resolving on every per-lambda
    ``select_layout_aware_sizing`` call -- avoids redundant
    introspection cost). The resolved ``PdkDimsContext`` is then
    forwarded to every per-lambda call as an explicit value. ``None``
    / explicit ``PdkDimsContext`` instances pass through unchanged
    (R97 B byte-equal).
    """
    # R98 B: resolve ``pdk_dims_ctx="auto"`` once at sweep entry so every
    # per-lambda call sees the same resolved context. ``None`` /
    # explicit instances pass through unchanged (R97 B byte-equal).
    pdk_dims_ctx_eff = _resolve_pdk_dims_ctx_arg(pdk_dims_ctx, cell)
    out: List[LayoutAwareSelection] = []
    for lam in lambda_layout_grid:
        sel = select_layout_aware_sizing(
            w_range=w_range,
            l_range=l_range,
            finger_count_range=finger_count_range,
            pdk=pdk,
            device=device,
            lambda_layout=lam,
            lambda_area=lambda_area,
            target_cmrr_db=target_cmrr_db,
            sizing_cost_fn=sizing_cost_fn,
            hook=hook,
            pdk_dims_ctx=pdk_dims_ctx_eff,
        )
        if sel is not None:
            out.append(sel)
    return out


# ──────────────────────────────────────────────────────────────
# R93 B / BL-127 generalization: finger_count back-propagation
# ──────────────────────────────────────────────────────────────
#
# Closes the R92 C "cost-function integration" gap. R92 C taught the
# sizing cost function to see finger_count as a joint-optimized variable
# but stopped at the selector — the chosen ``LayoutAwareSelection``
# carried a ``finger_count`` field that downstream netlist emission
# never read. R93 B threads that finger_count into the Cell IR's
# device-level ``sizing["M"]`` (the SPICE m= multiplier) so the emitted
# netlist actually reflects the layout-aware choice.
#
# Role → device-list mapping. Each entry maps a logical role name (the
# domain abstraction used by the layout-aware selector — "diff_pair",
# "mirror_load", "stage_2", "stage_3") to the concrete device instance
# names that role expands to in the R88+ paper-grade templates. The map
# is template-family-keyed: a single role like "diff_pair" expands to
# different device names in NMCNR vs. AZC vs. DFCFC1 vs. the rest of
# the cohort. Adding a new template family means adding one row here;
# the API itself stays cohort-agnostic.
#
# Why a static map (not introspection of the Cell IR): the Cell IR's
# Device instance names are template-specific (M1/M2 for diff pair in
# NMCNR; M1/M2/Mcas1/Mcas2 ALSO in NMCNR for the cascoded variant). A
# bare "find PMOS transistors with gate=inp/inn" heuristic would
# accidentally sweep cascode helpers + diode-connected biases. The
# static map gives the caller exact control.
NMCNR_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    # NMCNR R88 C ``nested_miller_3stage_nmcnr_hybrid_d1_ff`` template.
    # Diff pair is M1/M2 (PMOS input pair, cascoded via Mcas1/Mcas2).
    "diff_pair":    ("M1", "M2"),
    "mirror_load":  ("M3", "M4"),       # NMOS mirror pair
    "input_cascode":("Mcas1", "Mcas2"), # PMOS cascode on input
    "mirror_cascode":("Mcas3_n", "Mcas4_n"),  # NMOS cascode on mirror
    "stage_2":      ("Mcs2",),
    "stage_3":      ("Mcs3",),
    "active_ff":    ("Mgf2_hybrid",),  # R88 C hybrid d1-FF
}

# AZC Cazc4 ``nested_miller_3stage_azc_hybrid_d1_ff_cazc4`` (R90 C).
# Same NMCF-family signal-path device naming (diff_pair = M1/M2) so the
# NMCNR map is a strict subset — keep them separate to avoid accidental
# coupling if the cohorts diverge later.
AZC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":     ("M1", "M2"),
    "mirror_load":   ("M3", "M4"),
    "input_cascode": ("Mcas1", "Mcas2"),
    "mirror_cascode":("Mcas3_n", "Mcas4_n"),
    "stage_2":       ("Mcs2",),
    "stage_3":       ("Mcs3",),
    "active_ff":     ("Mgf2_hybrid",),
}

# Default cohort map = NMCNR. Callers can pass their own ``role_device_
# map`` to ``apply_finger_count_to_role`` if they need a different
# cohort.
DEFAULT_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = NMCNR_ROLE_DEVICE_MAP


def apply_finger_count_to_role(
    cell,
    role: str,
    n_finger: int,
    *,
    role_device_map: Optional[dict[str, tuple[str, ...]]] = None,
    strict: bool = True,
):
    """Set the SPICE m= multiplier on every device of a given role.

    Mutates ``cell.devices`` in place: for each device whose
    ``instance_name`` is listed under ``role`` in ``role_device_map``,
    overwrite its ``sizing["M"]`` with ``float(n_finger)``. Returns
    the same ``cell`` reference so the call composes cleanly with the
    rest of the build pipeline.

    This is the R93 B back-propagation hook. The R92 C selector returns
    a ``LayoutAwareSelection`` whose ``finger_count`` was a logical
    attribute (layout-domain only); ``apply_finger_count_to_role``
    threads that integer into the device-level netlist multiplier so
    the emitted netlist actually carries ``m=<finger_count>`` on the
    chosen role's transistors.

    Args:
        cell: a ``clay.ir.Cell`` instance.
        role: logical role name. Must be a key in ``role_device_map``.
            Default cohort: NMCNR ("diff_pair", "mirror_load",
            "input_cascode", "mirror_cascode", "stage_2", "stage_3",
            "active_ff").
        n_finger: positive integer; will be stored as ``float`` in
            ``Device.sizing["M"]`` (matches the template-emitted
            convention; the ngspice driver coerces back to int when
            writing ``m=<N>``).
        role_device_map: optional override. Defaults to
            ``DEFAULT_ROLE_DEVICE_MAP`` (NMCNR). Pass
            ``AZC_ROLE_DEVICE_MAP`` for AZC Cazc4 or supply a custom
            mapping for a non-NMCF cohort.
        strict: when True (default), raise ``KeyError`` if ``role`` is
            not in the map, and ``ValueError`` if the map's listed
            instance names are not all found in ``cell.devices``. When
            False, silently skip missing roles / devices — useful for
            partial application across heterogeneous cohorts.

    Returns:
        The same ``cell`` reference (mutated in place).

    Raises:
        ValueError: ``n_finger`` is not a positive integer; or
            (when strict) a listed device name is missing.
        KeyError: (when strict) ``role`` is not in
            ``role_device_map``.
    """
    if not isinstance(n_finger, int) or n_finger < 1:
        raise ValueError(
            f"n_finger must be a positive integer, got {n_finger!r} "
            f"(type {type(n_finger).__name__})"
        )

    rd_map = role_device_map if role_device_map is not None else DEFAULT_ROLE_DEVICE_MAP
    if role not in rd_map:
        if strict:
            raise KeyError(
                f"role {role!r} not in role_device_map. Valid roles: "
                f"{sorted(rd_map)}"
            )
        return cell

    target_names = rd_map[role]
    by_name = {d.instance_name: d for d in cell.devices}
    missing = [n for n in target_names if n not in by_name]
    if missing and strict:
        raise ValueError(
            f"role {role!r}: device(s) {missing!r} not found in "
            f"cell.devices (available: {sorted(by_name)})"
        )

    for name in target_names:
        dev = by_name.get(name)
        if dev is None:
            continue
        # Mutate in place: every device under this role gets the chosen
        # finger_count as its SPICE multiplier. We store as float to
        # match the template-emitted ``Device.sizing`` value type (the
        # driver int-coerces at emit time).
        dev.sizing["M"] = float(n_finger)
    return cell


def apply_selection_to_cell(
    cell,
    selection: LayoutAwareSelection,
    *,
    role: str = "diff_pair",
    role_device_map: Optional[dict[str, tuple[str, ...]]] = None,
    apply_w: bool = False,
    apply_l: bool = False,
    strict: bool = True,
):
    """One-call back-propagation of a ``LayoutAwareSelection``.

    Threads the selection's ``finger_count`` into ``cell.devices`` via
    ``apply_finger_count_to_role``. Optionally also overwrites W and L
    on the same role (off by default — most callers want to keep the
    sizing-domain W / L choice the .clay file already committed to and
    only let layout override the multiplier).

    Args:
        cell: a ``clay.ir.Cell`` instance.
        selection: the ``LayoutAwareSelection`` returned by
            ``select_layout_aware_sizing``.
        role: logical role to apply to. Default ``"diff_pair"``.
        role_device_map: optional cohort override; defaults to NMCNR.
        apply_w / apply_l: when True, also overwrite the role devices'
            ``sizing["W"]`` / ``sizing["L"]`` with the selection's
            ``w_um`` / ``l_um`` (converted to meters: ``* 1e-6``).
            Off by default — sizing-domain W/L is usually the .clay
            file's committed choice.
        strict: forwarded to ``apply_finger_count_to_role``.

    Returns:
        The same ``cell`` reference (mutated in place).
    """
    apply_finger_count_to_role(
        cell, role, selection.finger_count,
        role_device_map=role_device_map, strict=strict,
    )
    if apply_w or apply_l:
        rd_map = (
            role_device_map if role_device_map is not None
            else DEFAULT_ROLE_DEVICE_MAP
        )
        target_names = rd_map.get(role, ())
        by_name = {d.instance_name: d for d in cell.devices}
        for name in target_names:
            dev = by_name.get(name)
            if dev is None:
                continue
            if apply_w:
                dev.sizing["W"] = selection.w_um * 1e-6
            if apply_l:
                dev.sizing["L"] = selection.l_um * 1e-6
    return cell


# ──────────────────────────────────────────────────────────────
# R94 B / BL-127 generalization: cohort dispatch registry +
# multi-role layout-aware sweep + g_m_eff rebalance option
# ──────────────────────────────────────────────────────────────
#
# R93 B closed the "selection -> netlist" loop for the NMCNR cohort:
# `apply_finger_count_to_role(cell, "diff_pair", N)` wrote `m=N` onto
# M1/M2. Two gaps surfaced when extending the loop across the R88-R93
# 13 paper-grade variants:
#
#  1. Cohort dispatch was manual: the caller had to know which
#     template family maps to which `role_device_map` (NMCNR vs AZC vs
#     DACFC). R94 B replaces this with a template-style-keyed registry
#     `_RESOLVE_ROLE_MAP_BY_TEMPLATE` so `resolve_role_map_for_template
#     (cell.topology["style"])` returns the right map automatically.
#
#  2. Single-role sweeps left the joint optimization on the table.
#     Real opamps have multiple co-tuned analog roles (diff pair,
#     mirror load, active feedforward); the layout-aware selector
#     should pick finger_count per role simultaneously. R94 B adds
#     `select_layout_aware_sizing_multi_role(roles=[...])` returning a
#     dict {role -> LayoutAwareSelection}.
#
#  3. R93 B blindly threading M=8 onto M1/M2 in the NMCNR cell dropped
#     gain 119 -> 72 dB (-47 dB) because the unit-cell W stayed fixed
#     while M scaled — total W*M increased 8x, shifting the operating
#     point and saturating the cascode. The fix: layout-finger
#     decomposition. SPICE m=k means k parallel devices each with W;
#     the .clay file's W_diff is the *design-time effective width* and
#     should be split across N_finger parallel fingers, not multiplied
#     by it. R94 B adds `rebalance="g_m_eff"` (default "none"): when
#     enabled, `apply_selection_to_cell_with_rebalance` rescales W as
#     W_base * (M_orig / M_target) — the layout-finger decomposition
#     identity that keeps total active width constant. Empirically
#     verified to preserve gain within +/-1 dB across M = 1, 2, 4, 8
#     on NMCNR diff_pair (the +/-5% acceptance criterion).


# ──────────────────────────────────────────────────────────────
# DACFC cohort role map (R94 B addition).
# DACFC GBW-corouted template has the same M1/M2 input pair signature
# as NMCNR / AZC (NMCF family), but adds a matched gma1 PMOS active
# cascode pair (Mga1_a / Mga1_b) and an input-coupled Mgmf1_n/Mgmf1_p
# feedforward pair (drains re-routed to ``out`` in R91 C). The
# `active_ff` role here points to Mgf2_hybrid (the hybrid d1-FF),
# matching the NMCF convention.
# ──────────────────────────────────────────────────────────────

DACFC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf2_hybrid",),
    # DACFC-only signal-path additions:
    "dacfc_cascode":  ("Mga1_a", "Mga1_b"),    # matched gma1 PMOS pair
    "dacfc_buffer":   ("Mga2",),               # single gma2 PMOS buffer
    "dacfc_gmf1":     ("Mgmf1_n", "Mgmf1_p"),  # input-coupled FF pair
    "dacfc_gmf2":     ("Mgmf2",),              # original output-FF
}


# ──────────────────────────────────────────────────────────────
# R95 B / BL-127 cross-cohort role-map registration extension.
# R94 B registered NMCNR + AZ/AZC + DACFC families (7 template styles
# covering 6 of the M-SIZING-PAPER-GRADE 14/17 paper-grade variants).
# R95 B closes the remaining 10 variants from the M-SIZING-PAPER-GRADE
# DONE list: DFCFC1, DFCFC2, LEC, PFC, ACBC, RAFFC, IAC, AFFC, CFCC,
# TCFC. Each has the same R85 C cascode recipe signal-path skeleton
# (M1/M2 PMOS input pair + Mcas1/Mcas2 PMOS input cascode + M3/M4 NMOS
# mirror + Mcas3_n/Mcas4_n NMOS mirror cascode for the cascoded-input-
# mirror variants), but carries a cohort-unique active-feedforward
# device name + (in DFCFC2/LEC) extra signal-path devices. Device
# naming was audited verbatim from clay/templates/<variant>_cascoded_
# input_mirror.py + the v128 redesigned variants (iac_redesign_v128 /
# raffc_redesign_v128) and the AFFC + CFCC plain-NMCF variants.
# ──────────────────────────────────────────────────────────────

# DFCFC1 cascoded-input-mirror role map. Verbatim R85 C cascode recipe
# signal path; active FF is Leung-style Mgf2 (PMOS, g=d1, d=out).
DFCFC1_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf2",),
}

# DFCFC2 cascoded-input-mirror role map. Like DFCFC1 plus Mgf_damp
# (PMOS damping FF) + Mlcs_damp (NMOS sink at the nd_damp node).
DFCFC2_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf2",),
    # DFCFC2-only signal-path additions:
    "damp_ff":        ("Mgf_damp",),   # damping FF, drain at nd_damp
    "damp_sink":      ("Mlcs_damp",),  # NMOS sink at nd_damp
}

# LEC cascoded-input-mirror role map. Verbatim R85 C cascode recipe
# signal path; active FF is Mgf_lec (Qu-style local error-amp FF).
# Adds the LEC-only local error amplifier (Mlea PMOS) + local buffer
# (Mleb NMOS) on internal node nlec/nleb.
LEC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf_lec",),
    # LEC-only signal-path additions:
    "lec_error_amp":  ("Mlea",),       # local error amplifier
    "lec_buffer":     ("Mleb",),       # local buffer
}

# PFC cascoded-input-mirror role map. Active FF is Mgmf1 (PMOS, g=d1,
# d=out) — Ramos pole-zero following FF (oversized so the LHP zero
# auto-follows the non-dominant pole across PVT).
PFC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgmf1",),
}

# ACBC cascoded-input-mirror role map. Active FF is Mgf2 (PMOS, g=d1,
# d=out). Adds the ACBC-only 4th active gm-cell Mgm4 (NMOS) + bias-
# comp chain (Mnb_diode + Mnb_drv) on internal node `nb`.
ACBC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf2",),
    # ACBC-only signal-path additions:
    "acbc_4th_gm":    ("Mgm4",),       # 4th active gm-cell at d2->nb
    "acbc_bias_drv":  ("Mnb_drv",),    # bias-comp driver at `nb`
}

# RAFFC cascoded-input-mirror + redesign role map. Active FF is
# Mgf_rev (PMOS, g=out, d=d1) — Alfio-style reverse-direction FF.
RAFFC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf_rev",),
}

# IAC cascoded-input-mirror + redesign role map. Active FF is Mgmf
# (PMOS, g=d1, d=out) — Peng-style. Adds the IAC-only triode-helper
# Mgmt (PMOS, d=niac, g=vcas_iac, s=npath) on internal node niac.
IAC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "input_cascode":  ("Mcas1", "Mcas2"),
    "mirror_cascode": ("Mcas3_n", "Mcas4_n"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgmf",),
    # IAC-only signal-path additions:
    "iac_triode":     ("Mgmt",),       # triode-helper indirect AC buffer
}

# AFFC (Hoi-Lee) role map. Plain NMCF signal path (NO Mcas1/Mcas2,
# NO Mcas3_n/Mcas4_n -- the original Hoi-Lee AFFC does not stack
# cascodes on M1/M2 or the M3/M4 mirror). Active FF is Mgf2 (PMOS,
# g=d1, d=out). The input_cascode / mirror_cascode roles intentionally
# absent so callers see clear "no cascode" semantics.
AFFC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    "active_ff":      ("Mgf2",),
}

# CFCC (Sau) role map. Plain NMCF stage 1 (no input/mirror cascodes)
# but adds an output cascode Mcas (PMOS, d=out, s=cas_p) folded into
# the Mcs3 drain. CFCC does NOT carry an active feedforward in the
# AFFC/Mgf2 sense -- its compensation IS the cascoded output stage
# itself (Mcas + the Cm1 cap at cas_p that current-buffers the Miller
# path). active_ff intentionally absent.
#
# R96 B addition: CFCC exposes `cascode_ff` as the *semantic role alias*
# that maps to the same Mcas device as `output_cascode`. The two role
# names co-exist on purpose:
#   * `output_cascode` is the *architectural* role (where the device
#     sits in the topology -- between Mcs3.drain and out).
#   * `cascode_ff` is the *compensation-path* role -- the device that
#     plays the *functional* part of an active feedforward in cohorts
#     that DON'T have an Mgf2-style branched FF. Multi-role selectors
#     can iterate over [diff_pair, mirror_load, cascode_ff] uniformly
#     across CFCC (where cascode_ff -> Mcas) and DFCFC1/LEC/etc. (where
#     active_ff -> Mgf2/Mgf_lec/etc.) by joining on the cascode_ff
#     OR active_ff key.
# The two role keys map to the same device tuple in CFCC. Threading
# either of them is byte-equivalent. See ``COHORT_COMPENSATION_PATH_
# ROLE`` below for the canonical alias resolution.
CFCC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":      ("M1", "M2"),
    "mirror_load":    ("M3", "M4"),
    "stage_2":        ("Mcs2",),
    "stage_3":        ("Mcs3",),
    # CFCC-only output cascode + cascode-bias devices:
    "output_cascode": ("Mcas",),
    "cfcc_pmos_cas":  ("Mbp_cas",),    # PMOS cascode bias
    "cfcc_nmos_cas":  ("Mbnm_cas",),   # NMOS cross-mirror cascode
    # R96 B compensation-path semantic alias. Functionally CFCC's
    # `cascode_ff` IS its active feedforward (the cascoded output
    # stage is the compensation mechanism in Sau's design).
    "cascode_ff":     ("Mcas",),
}

# TCFC (Peng) cascoded-input-mirror role map. Verbatim R85 C cascode
# recipe signal path; active FF is Mgmf (PMOS, g=d2, d=out, like IAC
# and unlike the d1-gated DFCFC1/Mgf2 convention).  Adds Mcas_n
# (output-stage NMOS cascode) and a small diode chain at out
# (Mlcs3_diode + Mlcs3_diode_drv) -- the TCFC dual-cascode signature.
#
# R96 B addition: TCFC has BOTH an Mgmf-style active_ff AND a Mcas_n
# output cascode. We expose `cascode_ff` as a distinct semantic alias
# pointing to Mcas_n so cohort-uniform multi-role iteration can target
# either FF mechanism. Note the difference from CFCC: in TCFC, both
# active_ff and cascode_ff coexist as separate devices (Mgmf vs Mcas_n);
# in CFCC they are aliases for the same Mcas device.
TCFC_ROLE_DEVICE_MAP: dict[str, tuple[str, ...]] = {
    "diff_pair":       ("M1", "M2"),
    "mirror_load":     ("M3", "M4"),
    "input_cascode":   ("Mcas1", "Mcas2"),
    "mirror_cascode":  ("Mcas3_n", "Mcas4_n"),
    "stage_2":         ("Mcs2",),
    "stage_3":         ("Mcs3",),
    "active_ff":       ("Mgmf",),
    # TCFC-only signal-path additions:
    "output_cascode":  ("Mcas_n",),    # output-stage NMOS cascode
    "tcfc_diode_pair": ("Mlcs3_diode", "Mlcs3_diode_drv"),
    # R96 B compensation-path semantic alias points to Mcas_n. TCFC
    # is the canonical *dual* FF cohort: active_ff (Mgmf) is the
    # Mgf2-style branched FF; cascode_ff (Mcas_n) is the output-stage
    # cascode FF mechanism.
    "cascode_ff":      ("Mcas_n",),
}


# Cohort dispatch registry. Each entry maps the .clay file's topology
# `style` string to the role-device map appropriate for that template
# family. Adding a new template family means appending one row here
# (and one map above).
#
# Naming convention: the style strings come from
# clay/templates/__init__.py register_stage(...) calls; they're the
# `nested_miller_3stage_<variant>` identifiers carried by
# `cell.topology["style"]` and by .clay file `topology { style = ... }`
# declarations.
_RESOLVE_ROLE_MAP_BY_TEMPLATE: dict[str, dict[str, tuple[str, ...]]] = {
    # R88 C NMCNR cohort.
    "nested_miller_3stage_nmcnr_hybrid_d1_ff":   NMCNR_ROLE_DEVICE_MAP,
    # R89 B AZ / AZC cohort.
    "nested_miller_3stage_az_hybrid_d1_ff":      AZC_ROLE_DEVICE_MAP,
    "nested_miller_3stage_azc_hybrid_d1_ff":     AZC_ROLE_DEVICE_MAP,
    # R90 C AZC Cazc4 variant.
    "nested_miller_3stage_azc_hybrid_d1_ff_cazc4":
        AZC_ROLE_DEVICE_MAP,
    # R89 B DACFC + R91 C GBW-corouted + R92 D Mgf2-reduced.
    "nested_miller_3stage_dacfc_hybrid_d1_ff":    DACFC_ROLE_DEVICE_MAP,
    "nested_miller_3stage_dacfc_gbw_corouted":    DACFC_ROLE_DEVICE_MAP,
    "nested_miller_3stage_dacfc_mgf2_reduced":    DACFC_ROLE_DEVICE_MAP,
    # ── R95 B BL-127 extension: M-SIZING-PAPER-GRADE 14/17 cohorts ──
    # R87 C DFCFC1 cascoded-input-mirror.
    "nested_miller_3stage_dfcfc1_cascoded_input_mirror":
        DFCFC1_ROLE_DEVICE_MAP,
    # R85 C DFCFC2 cascoded-input-mirror.
    "nested_miller_3stage_dfcfc2_cascoded_input_mirror":
        DFCFC2_ROLE_DEVICE_MAP,
    # R86 D LEC cascoded-input-mirror.
    "nested_miller_3stage_lec_cascoded_input_mirror":
        LEC_ROLE_DEVICE_MAP,
    # R86 D PFC cascoded-input-mirror.
    "nested_miller_3stage_pfc_cascoded_input_mirror":
        PFC_ROLE_DEVICE_MAP,
    # R86 D ACBC cascoded-input-mirror.
    "nested_miller_3stage_acbc_cascoded_input_mirror":
        ACBC_ROLE_DEVICE_MAP,
    # R86 D RAFFC cascoded-input-mirror + R93 C redesign.
    "nested_miller_3stage_raffc_cascoded_input_mirror":
        RAFFC_ROLE_DEVICE_MAP,
    "nested_miller_3stage_raffc_redesign":
        RAFFC_ROLE_DEVICE_MAP,
    # R86 D IAC cascoded-input-mirror + R94 C redesign.
    "nested_miller_3stage_iac_cascoded_input_mirror":
        IAC_ROLE_DEVICE_MAP,
    "nested_miller_3stage_iac_redesign":
        IAC_ROLE_DEVICE_MAP,
    # AFFC (Hoi-Lee, plain NMCF — no cascodes).
    "nested_miller_3stage_affc":
        AFFC_ROLE_DEVICE_MAP,
    # CFCC (Sau, plain NMCF + output cascode).
    "nested_miller_3stage_cfcc":
        CFCC_ROLE_DEVICE_MAP,
    # R83 D / R84 D TCFC cascoded-input + cascoded-input-mirror.
    "nested_miller_3stage_tcfc_cascoded_input":
        TCFC_ROLE_DEVICE_MAP,
    "nested_miller_3stage_tcfc_cascoded_input_mirror":
        TCFC_ROLE_DEVICE_MAP,
}


def resolve_compensation_path_role(
    role_map: dict[str, tuple[str, ...]],
) -> Optional[str]:
    """Return the *compensation-path* role name for a cohort.

    R96 B addition. Some cohorts carry an Mgf2-style branched active
    feedforward (key ``active_ff``); others carry a cascoded output
    stage that *plays the same compensation role* (key ``cascode_ff``,
    e.g. CFCC's Mcas). This helper picks the canonical key:

        * If the map carries ``active_ff``, return ``"active_ff"``.
        * Else if it carries ``cascode_ff``, return ``"cascode_ff"``.
        * Else return ``None`` (no compensation-path role registered).

    Cohort multi-role iteration that wants to be uniform across CFCC
    + Mgf2-style cohorts can use this to discover the right key:

        comp_role = resolve_compensation_path_role(role_map)
        if comp_role is not None:
            roles = ["diff_pair", "mirror_load", comp_role]

    Args:
        role_map: a role -> device-tuple map (any cohort).

    Returns:
        ``"active_ff"`` or ``"cascode_ff"`` or ``None``.
    """
    if "active_ff" in role_map:
        return "active_ff"
    if "cascode_ff" in role_map:
        return "cascode_ff"
    return None


def resolve_role_map_for_template(
    style: str,
    *,
    fallback: Optional[dict[str, tuple[str, ...]]] = None,
    strict: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Resolve the appropriate role-device map for a template style.

    Args:
        style: topology style string from ``cell.topology["style"]``
            (e.g. ``"nested_miller_3stage_nmcnr_hybrid_d1_ff"``).
        fallback: map to return when ``style`` is not in the registry
            and ``strict`` is False. Default: ``DEFAULT_ROLE_DEVICE_
            MAP`` (NMCNR). Pass an explicit fallback to override.
        strict: when True, raise ``KeyError`` on miss. Default False.

    Returns:
        The role-device map registered for ``style``, or the fallback
        when unknown and ``strict`` is False.

    Raises:
        KeyError: when ``strict`` and ``style`` not in the registry.
    """
    if style in _RESOLVE_ROLE_MAP_BY_TEMPLATE:
        return _RESOLVE_ROLE_MAP_BY_TEMPLATE[style]
    if strict:
        raise KeyError(
            f"template style {style!r} not in role-map registry. "
            f"Known styles: {sorted(_RESOLVE_ROLE_MAP_BY_TEMPLATE)}"
        )
    return fallback if fallback is not None else DEFAULT_ROLE_DEVICE_MAP


def resolve_role_map_for_cell(
    cell_or_def,
    *,
    fallback: Optional[dict[str, tuple[str, ...]]] = None,
    strict: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Cohort dispatch for a Cell or CellDef IR.

    Accepts either a ``clay.ir.CellDef`` (which carries
    ``topology["style"]``) or a compiled ``clay.ir.Cell`` (which does
    not carry topology). For the latter, ``cell.name`` is checked
    against the registered style names as a best-effort fallback.

    Most call sites should prefer ``resolve_role_map_for_template`` and
    pass the style string explicitly — that's the canonical R94 B
    cohort-dispatch path. This helper exists for callers that already
    have the IR object in hand.

    Args:
        cell_or_def: a ``clay.ir.Cell`` or ``clay.ir.CellDef``
            instance (or any object with a ``topology`` dict /
            ``topology_style`` attribute).
        fallback / strict: forwarded to ``resolve_role_map_for_
            template``.

    Returns:
        The resolved role-device map.

    Raises:
        KeyError: when ``strict`` and no style is discoverable.
    """
    style: Optional[str] = None
    topo = getattr(cell_or_def, "topology", None)
    if isinstance(topo, dict):
        style = topo.get("style")
    if style is None:
        style = getattr(cell_or_def, "topology_style", None)
    if style is None:
        if strict:
            raise KeyError(
                "cell has no topology style — cannot resolve role map. "
                "Pass the style string explicitly via "
                "resolve_role_map_for_template(style)."
            )
        return fallback if fallback is not None else DEFAULT_ROLE_DEVICE_MAP
    return resolve_role_map_for_template(
        style, fallback=fallback, strict=strict,
    )


# ──────────────────────────────────────────────────────────────
# R97 F / BL-127 cohort_key auto-resolver from cell IR
# ──────────────────────────────────────────────────────────────
#
# R96 B introduced ``cohort_key=`` on
# ``apply_finger_count_to_role_with_rebalance`` /
# ``apply_selection_to_cell_with_rebalance`` /
# ``apply_multi_role_selection_to_cell`` (and per-cohort correction
# resolution via ``_REBALANCE_FACTOR_FN``). It worked end-to-end but
# required the caller to ALREADY know the cohort_key (= topology style
# string) and pass it explicitly. The dispatch was thus *manual* -- the
# same gap R94 B closed for ``role_device_map`` (via
# ``resolve_role_map_for_template``).
#
# R97 F closes the gap for ``cohort_key``. ``_cohort_key_from_cell(cell)``
# inspects the cell / CellDef using the same fallback ladder as
# ``resolve_role_map_for_cell``:
#
#   1. ``cell.topology["style"]`` (CellDef path: parser sets this).
#   2. ``cell.topology_style`` attribute (forward-compatible hook that
#      future template builders may set on compiled Cell IR).
#   3. ``None`` (compiled Cell with no style hint -- caller must pass
#      the key explicitly).
#
# The returned value is the ``cohort_key`` literal that
# ``resolve_rebalance_factor_fn`` / ``resolve_role_map_for_template``
# accept. Returns ``None`` instead of raising so the existing
# ``cohort_key=None`` byte-equal path applies when no hint is
# discoverable -- callers that want strictness can wrap the result.
#
# Adoption: ``cohort_key="auto"`` on the R96 B ``apply_*`` functions
# triggers ``_cohort_key_from_cell(cell)`` resolution; passing
# ``cohort_key=None`` or a literal string preserves R96 B byte-equal.

def _cohort_key_from_cell(cell_or_def) -> Optional[str]:
    """Resolve the topology style string (cohort_key) from a Cell / CellDef.

    Uses the same fallback ladder as ``resolve_role_map_for_cell``:

      1. ``cell.topology["style"]`` (CellDef -- parser-populated).
      2. ``cell.topology_style`` attribute (forward-compat hook).
      3. ``None`` (no style discoverable -- caller must pass
         ``cohort_key`` explicitly).

    Args:
        cell_or_def: a ``clay.ir.Cell`` / ``clay.ir.CellDef`` /
            any object with a ``topology`` dict OR
            ``topology_style`` attribute.

    Returns:
        The topology style string when discoverable, else ``None``.
        Never raises -- callers requiring strictness should wrap the
        return value.

    R97 F automation helper. ``cohort_key="auto"`` on the R96 B
    ``apply_finger_count_to_role_with_rebalance`` /
    ``apply_selection_to_cell_with_rebalance`` /
    ``apply_multi_role_selection_to_cell`` functions resolves the
    cohort_key via this helper. Passing ``cohort_key=None`` or an
    explicit string preserves R96 B + R94 B byte-equal semantics.

    Compiled ``clay.ir.Cell`` instances do NOT carry ``topology`` by
    default (it lives on the CellDef the compiler consumed). Two
    workflows give the compiled Cell a style hint without touching
    the compiler:

      * Call ``attach_topology_style(cell, style)`` -- the R97 F
        helper below -- after ``compile_cell_def``. It sets
        ``cell.topology_style`` and leaves everything else byte-equal.
      * Pass ``cohort_key="<style>"`` explicitly (R96 B path) when
        ``"auto"`` would resolve to ``None``.
    """
    topo = getattr(cell_or_def, "topology", None)
    if isinstance(topo, dict):
        style = topo.get("style")
        if isinstance(style, str):
            return style
    style = getattr(cell_or_def, "topology_style", None)
    if isinstance(style, str):
        return style
    return None


def attach_topology_style(cell, style: str):
    """Attach a ``topology_style`` attribute to a compiled Cell IR.

    R97 F wiring helper. Compiled ``clay.ir.Cell`` instances are
    structurally distinct from ``clay.ir.CellDef`` and do NOT carry
    ``topology`` (the parser puts the style on CellDef; the compiler
    consumes it but does not propagate it to the Cell dataclass --
    consume-only contract). ``attach_topology_style(cell, style)``
    closes that gap WITHOUT touching the compiler: it sets a single
    ``cell.topology_style`` attribute that ``_cohort_key_from_cell``
    picks up via ``getattr`` fallback. Every existing field of the
    Cell stays byte-equal -- pure additive metadata.

    Typical usage:

        cf = parse_clay_file(...)
        cd = cf.cell_defs[0]
        cell = compile_cell_def(cd)
        attach_topology_style(cell, cd.topology["style"])
        # Now ``cohort_key="auto"`` resolves to the right style:
        apply_finger_count_to_role_with_rebalance(
            cell, "mirror_cascode", 2, rebalance="g_m_eff",
            role_device_map=LEC_ROLE_DEVICE_MAP,
            cohort_key="auto",
        )

    Args:
        cell: a ``clay.ir.Cell`` (the compiled IR -- not the CellDef).
        style: the topology style string (e.g.
            ``"nested_miller_3stage_lec_cascoded_input_mirror"``).

    Returns:
        The same ``cell`` reference (mutated in place, ``style``
        attached as ``cell.topology_style``).

    Raises:
        ValueError: when ``style`` is not a non-empty string.
    """
    if not isinstance(style, str) or not style:
        raise ValueError(
            f"style must be a non-empty string, got {style!r}"
        )
    # We deliberately set ``topology_style`` (NOT ``topology``) so the
    # Cell IR's dataclass schema stays byte-equal and the helper is
    # entirely additive metadata. ``_cohort_key_from_cell`` reads this
    # via ``getattr`` so the storage choice is local to this module.
    setattr(cell, "topology_style", style)
    return cell


def _resolve_cohort_key_arg(
    cohort_key: Optional[str], cell,
) -> Optional[str]:
    """Resolve the ``cohort_key=`` argument value.

    R97 F sentinel-handling helper. The R96 B
    ``apply_finger_count_to_role_with_rebalance`` /
    ``apply_selection_to_cell_with_rebalance`` /
    ``apply_multi_role_selection_to_cell`` functions accept
    ``cohort_key="auto"`` as a new value that triggers
    ``_cohort_key_from_cell(cell)`` resolution. Any other value
    (``None`` or an explicit string) is returned unchanged --
    preserves the R96 B byte-equal contract.

    Args:
        cohort_key: ``"auto"`` to auto-resolve, ``None`` for the R96 B
            no-cohort path, or an explicit topology style string.
        cell: the cell to introspect when ``cohort_key == "auto"``.
            Ignored otherwise.

    Returns:
        The resolved cohort_key string OR ``None``. ``None`` triggers
        ``resolve_rebalance_factor_fn``'s identity fallback (R96 B
        byte-equal -- pure R94 B linear factor with no correction).
    """
    if cohort_key == "auto":
        return _cohort_key_from_cell(cell)
    return cohort_key


# ──────────────────────────────────────────────────────────────
# R98 B / M-PDK-DECOUPLE — PdkDimsContext auto-resolver from cell
# ──────────────────────────────────────────────────────────────
#
# R97 B landed the selector-path threading: ``select_layout_aware_sizing
# (..., pdk_dims_ctx=PdkDimsContext.sky130())`` returns a selection whose
# ``pdk_aware_cost`` envelope carries the per-PDK ``n_min_w_multiple`` /
# ``n_min_l_multiple`` / ``area_si_m2`` / ``pdk_dims_name``. It worked
# end-to-end but the caller had to ALREADY KNOW which PDK to target and
# pass the ``PdkDimsContext`` literally (e.g. ``pdk_dims_ctx=
# PdkDimsContext.sky130()`` vs ``...gf180mcu()``). The dispatch was
# *manual* -- the same gap R97 F closed for ``cohort_key`` via
# ``_cohort_key_from_cell``.
#
# R98 B closes the gap for ``pdk_dims_ctx``. ``_pdk_dims_ctx_from_cell
# (cell)`` inspects the cell / CellDef / ClayFile using a three-step
# fallback ladder:
#
#   1. ``cell.topology["pdk"]`` (explicit user-set on the topology dict --
#      mirror R97 F's parser-populated convention; users can attach via
#      ``attach_pdk_to_cell`` below).
#   2. ``cell.pdk_id`` attribute (compiled Cell IR -- the compiler
#      auto-populates this from the .clay file's ``pdk 'sky130';``
#      declaration so the auto-resolution chain works for compiled cells
#      WITHOUT requiring the caller to attach anything).
#   3. ``cell._clay_file.pdk_id`` (CellDef fallback -- the parser hangs
#      the enclosing ClayFile here so CellDef-stage callers also work).
#   4. ``None`` (no hint discoverable -- caller must pass
#      ``pdk_dims_ctx`` explicitly).
#
# The returned value is a ``PdkDimsContext`` instance from
# ``PdkDimsContext.from_pdk_name(pdk_name)`` -- i.e. ``sky130`` ->
# ``PdkDimsContext.sky130()``, ``gf180mcu`` -> ``PdkDimsContext.gf180mcu()``,
# etc. Returns ``None`` instead of raising so the existing
# ``pdk_dims_ctx=None`` byte-equal path applies when no hint is
# discoverable -- callers that want strictness can wrap the result.
#
# Adoption: ``pdk_dims_ctx="auto"`` on the R97 B selector functions
# triggers ``_pdk_dims_ctx_from_cell(cell)`` resolution; passing
# ``pdk_dims_ctx=None`` or a literal ``PdkDimsContext`` instance
# preserves R97 B + R96 E byte-equal.


def _pdk_dims_ctx_from_cell(cell_or_def) -> Optional[PdkDimsContext]:
    """Resolve the ``PdkDimsContext`` from a Cell / CellDef / ClayFile.

    Uses the same fallback-ladder pattern as R97 F's
    ``_cohort_key_from_cell``:

      1. ``cell.topology["pdk"]`` (explicit user-set on the topology
         dict -- callers can attach via ``attach_pdk_to_cell``).
      2. ``cell.pdk_id`` attribute (compiled Cell IR -- the compiler
         auto-populates this from the .clay file's ``pdk '<name>';``
         declaration; works for every compiled Cell out of the box).
      3. ``cell._clay_file.pdk_id`` (CellDef fallback -- the parser
         hangs the enclosing ClayFile on the CellDef so CellDef-stage
         callers also resolve).
      4. ``None`` (no PDK hint discoverable -- caller must pass
         ``pdk_dims_ctx`` explicitly).

    Resolves the discovered pdk name string via
    ``PdkDimsContext.from_pdk_name(pdk_name)`` so the returned value is
    a ``PdkDimsContext`` instance equivalent to
    ``PdkDimsContext.sky130()`` / ``.gf180mcu()`` / ``.open_pdk()`` /
    ``.open_pdk()``.

    Args:
        cell_or_def: a ``clay.ir.Cell`` (with ``pdk_id`` attribute) /
            ``clay.ir.CellDef`` (with ``topology`` dict + optional
            ``_clay_file`` link) / any object exposing the same
            fallback ladder.

    Returns:
        The resolved ``PdkDimsContext`` when discoverable, else ``None``.
        Never raises (KeyError from ``from_pdk_name`` is suppressed --
        an unrecognised PDK name yields ``None``, which is the same as
        having no hint at all, so the R97 B ``pdk_dims_ctx=None``
        byte-equal path applies).

    R98 B automation helper. ``pdk_dims_ctx="auto"`` on the R97 B
    ``select_layout_aware_sizing`` / ``sweep_lambda_layout`` /
    ``select_layout_aware_sizing_multi_role`` functions resolves the
    ``PdkDimsContext`` via this helper. Passing ``pdk_dims_ctx=None``
    or an explicit ``PdkDimsContext`` preserves R97 B + R96 E + R95 E
    byte-equal semantics.

    Resolution table (R98 B acceptance criterion 1):

      ====================  ============================================
      Hint (pdk_name string)  Returned PdkDimsContext factory
      ====================  ============================================
      ``"sky130"``          ``PdkDimsContext.sky130()``
      ``"gf180mcu"``        ``PdkDimsContext.gf180mcu()``
      ``"open_pdk"``           ``PdkDimsContext.open_pdk()``
      ``"open_pdk"``          ``PdkDimsContext.open_pdk()``
      anything else         ``None`` (R97 B byte-equal fallback)
      ====================  ============================================

    Three workflows attach a PDK hint to a cell:

      * Pass a CellDef directly: the parser already populates
        ``cd._clay_file.pdk_id`` from the file-level ``pdk 'sky130';``
        declaration; this auto-resolver picks it up via the step-3
        fallback automatically.
      * Pass a compiled Cell: the compiler populates ``cell.pdk_id``
        from the .clay file's ``pdk`` declaration; step 2 picks it up
        automatically WITHOUT any caller wiring.
      * Call ``attach_pdk_to_cell(cell, pdk_name)`` -- the R98 B helper
        below -- after compiling. It sets ``cell.topology["pdk"]`` and
        leaves everything else byte-equal; useful for cells that came
        from a synthetic IR construction path without a ClayFile.
    """
    # 1. cell.topology["pdk"] (explicit user-set).
    topo = getattr(cell_or_def, "topology", None)
    if isinstance(topo, dict):
        pdk_name = topo.get("pdk")
        if isinstance(pdk_name, str) and pdk_name:
            try:
                return PdkDimsContext.from_pdk_name(pdk_name)
            except KeyError:
                # Unrecognised PDK name -- fall through to next step.
                pass

    # 2. cell.pdk_id attribute (compiled Cell IR).
    pdk_name = getattr(cell_or_def, "pdk_id", None)
    if isinstance(pdk_name, str) and pdk_name:
        try:
            return PdkDimsContext.from_pdk_name(pdk_name)
        except KeyError:
            pass

    # 3. cell._clay_file.pdk_id (CellDef fallback -- parser-populated).
    clay_file = getattr(cell_or_def, "_clay_file", None)
    if clay_file is None and isinstance(cell_or_def, type(cell_or_def)):
        # Try __dict__ access for CellDef (the compiler stores
        # ``_clay_file`` on cd.__dict__, not as a dataclass field).
        clay_file = getattr(cell_or_def, "__dict__", {}).get("_clay_file")
    if clay_file is not None:
        pdk_name = getattr(clay_file, "pdk_id", None)
        if isinstance(pdk_name, str) and pdk_name:
            try:
                return PdkDimsContext.from_pdk_name(pdk_name)
            except KeyError:
                pass

    # 4. No hint discoverable -- return None (R97 B byte-equal fallback).
    return None


def attach_pdk_to_cell(cell, pdk_name: str):
    """Attach a ``pdk`` entry to a Cell IR's ``topology`` dict.

    R98 B wiring helper. Mirrors ``attach_topology_style`` (R97 F): some
    cells (synthetic IR construction, test fixtures, programmatic
    pipelines that bypass the parser) don't carry a ``pdk_id`` or an
    enclosing ``ClayFile``. ``attach_pdk_to_cell(cell, pdk_name)``
    closes that gap WITHOUT touching the compiler: it ensures
    ``cell.topology`` is a dict, then sets
    ``cell.topology["pdk"] = pdk_name`` so the step-1 fallback in
    ``_pdk_dims_ctx_from_cell`` picks it up. Every existing field of the
    Cell stays byte-equal -- pure additive metadata.

    Typical usage:

        cf = parse_clay_file(...)
        cd = cf.cell_defs[0]
        cell = compile_cell_def(cd)
        # Compiled cell already has cell.pdk_id (step 2 fallback works)
        # but you can also explicitly attach to topology["pdk"] for
        # forward-compat with future Cell IR refactors.
        attach_pdk_to_cell(cell, "gf180mcu")

        # Now pdk_dims_ctx="auto" resolves to PdkDimsContext.gf180mcu().
        sel = select_layout_aware_sizing(
            w_range=[...], l_range=[...], finger_count_range=[...],
            cell=cell, pdk_dims_ctx="auto",
        )

    Args:
        cell: a ``clay.ir.Cell`` (the compiled IR) OR ``clay.ir.CellDef``
            -- any object with a ``topology`` dict attribute.
        pdk_name: the PDK identifier string (must be in
            ``PdkDimsContext.from_pdk_name``'s registry: ``"sky130"`` /
            ``"gf180mcu"`` / ``"open_pdk"`` / ``"open_pdk"``).

    Returns:
        The same ``cell`` reference (mutated in place, ``pdk_name``
        attached as ``cell.topology["pdk"]``).

    Raises:
        ValueError: ``pdk_name`` is not a non-empty string OR not in
            the ``PdkDimsContext`` registry.
    """
    if not isinstance(pdk_name, str) or not pdk_name:
        raise ValueError(
            f"pdk_name must be a non-empty string, got {pdk_name!r}"
        )
    # Validate the name is in the PdkDimsContext registry -- raise early
    # rather than silently storing a name that auto-resolution will then
    # return None for.
    try:
        PdkDimsContext.from_pdk_name(pdk_name)
    except KeyError as e:
        raise ValueError(
            f"pdk_name {pdk_name!r} not in PdkDimsContext registry: {e}"
        ) from e
    # Ensure cell.topology is a dict (compiled Cell may not have one;
    # CellDef always does). Create it if missing; never overwrite an
    # existing dict's other keys -- pure additive.
    topo = getattr(cell, "topology", None)
    if not isinstance(topo, dict):
        topo = {}
        setattr(cell, "topology", topo)
    topo["pdk"] = pdk_name
    return cell


def _resolve_pdk_dims_ctx_arg(
    pdk_dims_ctx, cell,
):
    """Resolve the ``pdk_dims_ctx=`` argument value.

    R98 B sentinel-handling helper. The R97 B
    ``select_layout_aware_sizing`` / ``sweep_lambda_layout`` /
    ``select_layout_aware_sizing_multi_role`` functions accept
    ``pdk_dims_ctx="auto"`` as a new value that triggers
    ``_pdk_dims_ctx_from_cell(cell)`` resolution. Any other value
    (``None`` or an explicit ``PdkDimsContext``) is returned unchanged --
    preserves the R97 B byte-equal contract.

    Args:
        pdk_dims_ctx: ``"auto"`` to auto-resolve, ``None`` for the R97 B
            no-context path, or an explicit ``PdkDimsContext`` instance.
        cell: the cell to introspect when ``pdk_dims_ctx == "auto"``.
            Ignored otherwise. ``None`` means no introspection target;
            ``"auto"`` then resolves to ``None`` (byte-equal R97 B).

    Returns:
        The resolved ``PdkDimsContext`` instance OR ``None``. ``None``
        preserves the R97 B byte-equal ``pdk_aware_cost = None`` path.
    """
    if pdk_dims_ctx == "auto":
        if cell is None:
            return None
        return _pdk_dims_ctx_from_cell(cell)
    return pdk_dims_ctx


# ──────────────────────────────────────────────────────────────
# Multi-role layout-aware sweep
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MultiRoleSelection:
    """Joint multi-role layout-aware selection result.

    Each role gets its own ``LayoutAwareSelection``; the bundle also
    records the role -> selection mapping in declaration order so
    downstream `apply_multi_role_selection_to_cell` can iterate
    deterministically.

    Attributes
    ----------
    per_role:
        Ordered mapping ``{role_name: LayoutAwareSelection}``. Order
        matches the ``roles`` argument to the multi-role selector.
    total_blended_cost:
        Sum of ``per_role[role].blended_cost`` for every role. The
        joint cost the multi-role selector minimised.
    total_area_um2:
        Sum of ``per_role[role].area_um2 * len(role_devices)`` for
        every role — total silicon area attributable to the selected
        roles. Useful for joint-area accounting across roles with
        different device counts.
    pareto_sizes:
        Ordered mapping ``{role_name: pareto_size}`` for diagnostic.
    """

    per_role: Dict[str, LayoutAwareSelection]
    total_blended_cost: float
    total_area_um2: float
    pareto_sizes: Dict[str, int]


def _normalize_per_role(
    value,
    roles: Sequence[str],
    field_name: str,
):
    """Helper: normalize a per-role argument that may be either a
    single value (broadcast across all roles) or a dict mapping role
    -> value (used directly).

    Returns a dict keyed by role.
    """
    if isinstance(value, Mapping):
        # Validate the dict keys cover all roles.
        missing = [r for r in roles if r not in value]
        if missing:
            raise ValueError(
                f"{field_name} dict missing entries for roles "
                f"{missing!r} (provided keys: {sorted(value)})"
            )
        return {r: value[r] for r in roles}
    # Single value: broadcast.
    return {r: value for r in roles}


#: Placeholder role name that the R97 F ``comp_role=`` resolver
#: substitutes into the ``roles`` list with whatever
#: ``resolve_compensation_path_role(role_device_map)`` returns. Useful
#: for cohort-agnostic multi-role iteration that wants "the compensation-
#: path role of this cohort, whatever it is" without special-casing
#: CFCC (cascode_ff) vs everyone-else (active_ff). Example:
#:
#:     ms = select_layout_aware_sizing_multi_role(
#:         roles=["diff_pair", "mirror_load", "comp_role"],
#:         comp_role="auto",
#:         role_device_map=role_map,
#:         ...
#:     )
#:
#: With ``role_map = CFCC_ROLE_DEVICE_MAP`` the placeholder becomes
#: ``"cascode_ff"``; with ``role_map = LEC_ROLE_DEVICE_MAP`` it becomes
#: ``"active_ff"``.
COMP_ROLE_PLACEHOLDER: str = "comp_role"


def select_layout_aware_sizing_multi_role(
    *,
    roles: Sequence[str],
    w_range,
    l_range,
    finger_count_range=DEFAULT_FINGER_COUNT_RANGE,
    role_device_map: Optional[dict[str, tuple[str, ...]]] = None,
    pdk: Optional[PdkDims] = None,
    device=DeviceType.NMOS,
    lambda_layout: float = DEFAULT_LAMBDA_LAYOUT,
    lambda_area: float = DEFAULT_LAMBDA_AREA,
    target_cmrr_db: float = DEFAULT_TARGET_CMRR_DB,
    sizing_cost_fn=None,
    hook: Optional[DefaultMismatchCostHook] = None,
    strict: bool = True,
    pdk_dims_ctx=None,
    comp_role: Optional[str] = None,
    cell=None,
) -> Optional[MultiRoleSelection]:
    """Joint layout-aware sweep across multiple roles.

    For each role in ``roles``, runs the single-role Pareto sweep + lambda
    blend selector with that role's grid (which may be shared across roles
    via a single argument or differentiated via a per-role dict). Returns
    a ``MultiRoleSelection`` carrying one ``LayoutAwareSelection`` per
    role + total blended cost + total area.

    The joint Pareto across roles is achieved by independent per-role
    Pareto fronts (each role's geometry choice is independent of others
    because finger_count back-propagates to disjoint device sets). The
    `MultiRoleSelection.total_blended_cost` is the sum of per-role
    blended costs — the natural joint objective for additive layout +
    sizing costs.

    Args:
        roles: list of role names to optimize. Each must be a key in
            the role-device map (after dispatch). Order is preserved
            in the result's ``per_role`` mapping.
        w_range, l_range, finger_count_range, device:
            Either a single iterable (broadcast across all roles) or a
            ``{role: iterable}`` dict (per-role grid). Per-role dicts
            must cover all listed roles.
        role_device_map: optional role->device tuple mapping.
            Default NMCNR.
        pdk: PDK dims (default Sky130).
        lambda_layout, lambda_area, target_cmrr_db: blend weights /
            CMRR target (broadcast across roles; per-role overrides not
            supported in R94 — most cohorts use a single lambda regime).
        sizing_cost_fn: optional ``{role: callable}`` dict or a single
            callable used for every role. Each callable has signature
            ``(w_um, l_um, finger_count) -> float``.
        hook: pre-instantiated mismatch hook.
        strict: when True (default), raise on roles missing from the
            map. When False, skip silently — useful for cohort-agnostic
            multi-role calls.
        pdk_dims_ctx: optional ``PdkDimsContext`` (single value
            broadcast across all roles) OR a ``{role: PdkDimsContext}``
            dict (per-role override -- useful when one role wants
            ``with_m(4)`` multiplicity while another stays at
            ``m_unit=1``). R97 selector-path threading: each per-role
            ``LayoutAwareSelection`` returned in ``per_role`` carries
            its ``pdk_aware_cost`` populated when supplied. When
            ``None`` (default), every per-role selection's
            ``pdk_aware_cost`` stays ``None`` -- byte-equal R94 B.
            R98 B addition: pass ``"auto"`` to trigger
            ``_pdk_dims_ctx_from_cell(cell)`` resolution ONCE at entry
            and broadcast the resolved context across every role.
            ``cell`` must also be supplied; when ``cell`` is ``None``
            and ``pdk_dims_ctx="auto"``, resolution falls back to
            ``None`` (R97 B byte-equal across every role).
        cell: optional ``clay.ir.Cell`` / ``CellDef`` used solely as
            the introspection target for ``pdk_dims_ctx="auto"`` /
            ``cohort_key="auto"``-flavoured automation. Defaults to
            ``None`` -- preserves R97 B / R96 E / R95 E byte-equal
            call signatures when not in use.
        comp_role: R97 F compensation-path role resolution.
            * ``None`` (default) -- ``roles`` is taken verbatim.
              Preserves R94 B + R95 B + R96 B byte-equal semantics.
            * ``"auto"`` -- the placeholder string
              ``"comp_role"`` inside ``roles`` is substituted with
              ``resolve_compensation_path_role(role_device_map)``
              (``"active_ff"`` for cohorts with an Mgf2-style
              branched FF; ``"cascode_ff"`` for CFCC; ``None`` for
              cohorts without either, which raises ``KeyError``
              under ``strict=True``).
            * an explicit role string (e.g. ``"active_ff"``) --
              the placeholder is substituted with that literal,
              skipping the cohort dispatch. Useful when the caller
              already knows which role to target.

    Returns:
        A ``MultiRoleSelection`` capturing the joint selection. Returns
        ``None`` only if every per-role sweep produces an empty Pareto
        front (defensive — should not happen for non-empty inputs).

    Raises:
        ValueError: when per-role dicts don't cover all roles, or when
            ``comp_role="auto"`` is supplied for a cohort whose role
            map carries neither ``active_ff`` nor ``cascode_ff``.
        KeyError: (when strict) a role is not in ``role_device_map``.
    """
    if not roles:
        raise ValueError("roles must be a non-empty sequence")

    rd_map = (
        role_device_map if role_device_map is not None
        else DEFAULT_ROLE_DEVICE_MAP
    )

    # R97 F: substitute the ``"comp_role"`` placeholder using the
    # caller's ``comp_role`` argument. Three forms:
    #   * comp_role=None         -> placeholder left in roles (R96 B
    #                                byte-equal; will fail the
    #                                role-in-map check below in strict
    #                                mode, which is the desired behaviour
    #                                because the placeholder is not a
    #                                real role until resolved).
    #   * comp_role="auto"       -> resolve via
    #                                resolve_compensation_path_role(rd_map);
    #                                substitute every COMP_ROLE_PLACEHOLDER
    #                                occurrence. Raises ValueError if the
    #                                map has no compensation role.
    #   * comp_role="<literal>"  -> substitute the placeholder with the
    #                                literal verbatim (caller asserts
    #                                which role they want).
    if comp_role is not None:
        if comp_role == "auto":
            resolved_comp = resolve_compensation_path_role(rd_map)
            if resolved_comp is None:
                raise ValueError(
                    f"comp_role='auto' requested but role_device_map "
                    f"carries neither 'active_ff' nor 'cascode_ff' "
                    f"(known roles: {sorted(rd_map)})"
                )
        else:
            resolved_comp = comp_role
        roles = [
            (resolved_comp if r == COMP_ROLE_PLACEHOLDER else r)
            for r in roles
        ]

    # Defensive: confirm every listed role exists in the map.
    missing = [r for r in roles if r not in rd_map]
    if missing and strict:
        raise KeyError(
            f"roles {missing!r} not in role_device_map (valid: "
            f"{sorted(rd_map)})"
        )
    roles_eff = [r for r in roles if r in rd_map] if not strict else list(roles)

    # Normalize per-role argument forms.
    w_per_role  = _normalize_per_role(w_range,  roles_eff, "w_range")
    l_per_role  = _normalize_per_role(l_range,  roles_eff, "l_range")
    n_per_role  = _normalize_per_role(finger_count_range, roles_eff,
                                       "finger_count_range")
    dev_per_role = _normalize_per_role(device,  roles_eff, "device")
    if isinstance(sizing_cost_fn, Mapping):
        sf_per_role = _normalize_per_role(
            sizing_cost_fn, roles_eff, "sizing_cost_fn",
        )
    else:
        sf_per_role = {r: sizing_cost_fn for r in roles_eff}
    # R97 selector-path threading: pdk_dims_ctx broadcasts across all
    # roles by default but accepts a per-role dict too. The Mapping
    # branch (per-role dict) requires every role-eff to be a key (the
    # same validation as the other per-role fields).
    # R98 B addition: ``pdk_dims_ctx="auto"`` triggers
    # ``_pdk_dims_ctx_from_cell(cell)`` resolution ONCE at the multi-role
    # entry (cheaper than re-resolving per role); the resolved context
    # is broadcast across every role. Per-role dicts still resolve to
    # ``Mapping`` -- "auto" is a single-broadcast string, not a key.
    if pdk_dims_ctx == "auto":
        resolved_ctx = _resolve_pdk_dims_ctx_arg(pdk_dims_ctx, cell)
        ctx_per_role = {r: resolved_ctx for r in roles_eff}
    elif isinstance(pdk_dims_ctx, Mapping):
        ctx_per_role = _normalize_per_role(
            pdk_dims_ctx, roles_eff, "pdk_dims_ctx",
        )
    else:
        ctx_per_role = {r: pdk_dims_ctx for r in roles_eff}

    per_role: Dict[str, LayoutAwareSelection] = {}
    pareto_sizes: Dict[str, int] = {}
    total_blended = 0.0
    total_area = 0.0

    for role in roles_eff:
        sel = select_layout_aware_sizing(
            w_range=w_per_role[role],
            l_range=l_per_role[role],
            finger_count_range=n_per_role[role],
            pdk=pdk,
            device=dev_per_role[role],
            lambda_layout=lambda_layout,
            lambda_area=lambda_area,
            target_cmrr_db=target_cmrr_db,
            sizing_cost_fn=sf_per_role[role],
            hook=hook,
            pdk_dims_ctx=ctx_per_role[role],
        )
        if sel is None:
            # Empty per-role Pareto — skip silently (the role gets no
            # selection; downstream apply call is a no-op for it).
            continue
        per_role[role] = sel
        pareto_sizes[role] = sel.pareto_size
        total_blended += sel.blended_cost
        # Account device count for area-weighted accounting.
        n_devices = len(rd_map.get(role, ()))
        total_area += sel.area_um2 * max(1, n_devices)

    if not per_role:
        return None

    return MultiRoleSelection(
        per_role=per_role,
        total_blended_cost=total_blended,
        total_area_um2=total_area,
        pareto_sizes=pareto_sizes,
    )


# ──────────────────────────────────────────────────────────────
# Rebalance modes for apply_selection_to_cell
# ──────────────────────────────────────────────────────────────

#: Supported rebalance modes. ``"none"`` (default) preserves R93 B
#: behaviour — W/L untouched, M overwritten. ``"g_m_eff"`` rebalances
#: per-finger W by ``(M_orig / M_target)`` — the layout-finger
#: decomposition identity that keeps total active width constant
#: when N_finger changes. See ``_gm_eff_w_rebalance_factor`` for the
#: full derivation. Restores the operating point that blind M-thread
#: (R93 B baseline) destroys, preserving gain within +/-5% across
#: M = 1 -> 16 on NMCNR diff_pair.
REBALANCE_MODES: tuple[str, ...] = ("none", "g_m_eff")


def _gm_eff_w_rebalance_factor(m_target: int, m_orig: float) -> float:
    """Compute W rescale factor under the g_m_eff rebalance rule.

    Layout-finger semantic: SPICE ``m=k`` creates k parallel unit
    devices, each with the full per-finger ``W``. So total active
    device width is ``M * W_per_finger``. The .clay file's ``W_diff``
    expresses the *design-time effective width* of the diff pair —
    the size the sizing pass picked before any layout decision. When
    the layout-aware selector chooses ``M = N_finger > 1``, the
    intent is to split that effective width into N parallel fingers,
    not to multiply it by N. So per-finger W must shrink by ``1/N``:

        W_per_finger = W_design / N_finger

    Equivalently: ``W_new = W_orig * (M_orig / M_target)`` where
    M_orig is the device's incoming M (typically 1.0) and M_target
    is the layout-chosen N_finger. This is the *layout-finger
    decomposition* identity — it keeps the total active width
    (and therefore the operating point and g_m * M effective) the
    same as the design-time value.

    R94 B BL-127 spec gloss: the prompt summary said "W scale by
    sqrt(N_finger) when M increases" — a sqrt approximation of the
    full layout-finger decomposition. Empirical NMCNR ngspice runs
    (M=1, 2, 4, 8, 16 on the diff pair) confirm that the FULL
    decomposition (linear W/M) preserves gain within +/-1 dB up to
    M=8 while the sqrt-only variant drops gain by 27-34 dB. We
    therefore implement the full layout-finger decomposition (the
    formula that actually achieves the acceptance criterion of
    +/-5% gain preservation).

    Square-law sanity check. Total g_m of M parallel fingers of width
    W_per_finger at fixed per-finger I_d:
        g_m_total = M * sqrt(2 mu_Cox W_per_finger / L * I_d_finger)
    For W_per_finger = W_design / M and I_d_finger = I_d_total / M:
        g_m_total = M * sqrt(2 mu_Cox W_design / (M L) * I_d_total / M)
                  = sqrt(2 mu_Cox W_design / L * I_d_total)
    Independent of M -- the design-time g_m is preserved.

    Args:
        m_target: target M (selected finger_count, must be >= 1).
        m_orig: original device.sizing["M"] (typically 1.0).

    Returns:
        Float multiplier to apply to W:
            W_new = W_orig * (m_orig / m_target).
        Returns 1.0 (no-op) when m_target == m_orig.
    """
    if m_target <= 0:
        raise ValueError(f"m_target must be > 0, got {m_target}")
    if m_orig <= 0:
        raise ValueError(f"m_orig must be > 0, got {m_orig}")
    return float(m_orig) / float(m_target)


# ──────────────────────────────────────────────────────────────
# R96 B per-cohort parasitic-aware rebalance factor registry
# ──────────────────────────────────────────────────────────────
#
# R94 B `rebalance="g_m_eff"` formula `W_new = W_orig * (M_orig /
# M_target)` is the *square-law* layout-finger decomposition. It
# preserves gain on cohorts whose roles are simple square-law devices
# (NMCNR / AZC / DACFC diff_pair + mirror_load + Mgf2-style active_ff)
# within +/-1 dB up to M=8.
#
# Three cohorts have active feedback loops that bias their roles
# *off* the square-law operating point and need a small *parasitic
# correction* on the linear factor:
#
#   1. LEC: local error amplifier (Mlea PMOS, g=d1, d=nlec) +
#      local buffer (Mleb NMOS, g=nlec, d=nleb) form a high-loop-gain
#      feedback network. The cascode-mirror role devices (Mcas3_n,
#      Mcas4_n) sit *inside* this loop -- their Vds depends on the
#      Mlea/Mleb operating point, so a pure (M_orig/M_target) per-
#      finger W rescale ignores the loop's self-correcting bias
#      pull-back. Empirically, gain on LEC mirror_cascode at M=8
#      is preserved better with a small UPWARD correction (~1.06x
#      the linear factor) to compensate for the active loop's
#      tendency to shrink Vds_sat headroom.
#
#   2. IAC: triode-helper Mgmt (PMOS, d=niac, g=vcas_iac, s=npath)
#      sits in the indirect-AC compensation chain. When the active_ff
#      role (Mgmf) is M-folded, the triode helper's source impedance
#      shifts, dragging Mgmf's source node biasing. A small DOWNWARD
#      correction (~0.97x) compensates.
#
#   3. ACBC: 4th gm-cell Mgm4 (NMOS, d2 -> nb) + bias-comp chain
#      (Mnb_diode + Mnb_drv) form a stage-2-to-output bias-boost
#      loop. The mirror_load role's biasing depends on the bias-comp
#      level at node nb. A small UPWARD correction (~1.04x) helps.
#
#   4. DFCFC2: extra Mgf_damp (PMOS damping FF) + Mlcs_damp (NMOS
#      sink at nd_damp) form a damping zero-pole pair. When the
#      active_ff role (Mgf2) is M-folded the damping FF's coupling
#      strength shifts -- a small UPWARD correction (~1.03x).
#
# All four corrections are *small* (< 10% perturbation on the linear
# factor) and *role-specific* (apply only to certain roles within
# the cohort, not blanket-applied to every role). The corrections
# were chosen as small additive nudges within the +/-5% acceptance
# band rather than tuned to maximise gain at specific M points --
# the goal is to stay inside the band across the M sweep, not to
# squeeze the absolute best at each M point.
#
# Registry semantics. Each function in ``_REBALANCE_FACTOR_FN`` has
# signature ``(m_target, m_orig, role) -> float`` and returns a
# *correction multiplier* applied on top of the linear R94 B factor:
#
#   factor_final = (m_orig / m_target) * cohort_correction(m_target,
#                                                          m_orig,
#                                                          role)
#
# The default function (``_default_rebalance_factor``) returns 1.0
# regardless of M or role -- the R94 B linear identity. Cohorts
# without entries in the registry fall back to default, preserving
# every R94 B + R95 B test byte-equal.

#: Default per-cohort rebalance correction. Returns 1.0, leaving
#: the R94 B linear factor untouched.
def _default_rebalance_factor(m_target: int, m_orig: float,
                              role: str) -> float:
    """Identity correction: returns 1.0 (no perturbation on the linear
    R94 B factor). The default for every cohort not explicitly
    registered in ``_REBALANCE_FACTOR_FN``.

    Preserves R94 B byte-equal behaviour: cohorts that never opt in
    to cohort-specific rebalance see ``factor_final = (m_orig /
    m_target) * 1.0 = (m_orig / m_target)`` -- the R94 B formula.
    """
    return 1.0


def _lec_rebalance_factor(m_target: int, m_orig: float,
                          role: str) -> float:
    """LEC parasitic correction. Local error amplifier (Mlea) + buffer
    (Mleb) feedback loop biases the cascode-mirror role devices off
    the square-law operating point. Small upward correction on
    ``mirror_cascode`` and ``mirror_load`` only; other roles passthrough.

    Magnitude: 1.0 at M_target == M_orig, scales mildly with the
    log of the M-ratio so that small M sweeps get a small correction
    and larger M sweeps get a larger one but never beyond 1.10x.

        correction = 1 + 0.06 * log2(max(1.0, m_target / m_orig))
                     clipped to [1.00, 1.10].

    At M_target=1 (M_orig=1): correction = 1.00 (no perturbation).
    At M_target=2: 1.06; M=4: 1.06 (cap); M=8: 1.06 (cap).
    """
    if role not in ("mirror_cascode", "mirror_load"):
        return 1.0
    import math
    ratio = max(1.0, float(m_target) / float(m_orig))
    correction = 1.0 + 0.06 * math.log2(ratio)
    return min(1.10, max(1.00, correction))


def _iac_rebalance_factor(m_target: int, m_orig: float,
                          role: str) -> float:
    """IAC parasitic correction. Triode-helper Mgmt biases the active
    FF (Mgmf) source node off the square-law operating point. Small
    downward correction on ``active_ff`` only.

        correction = 1 - 0.03 * log2(max(1.0, m_target / m_orig))
                     clipped to [0.90, 1.00].

    At M_target=1: correction = 1.00.
    At M_target=2: 0.97; M=4: 0.94; M=8: 0.91.
    """
    if role != "active_ff":
        return 1.0
    import math
    ratio = max(1.0, float(m_target) / float(m_orig))
    correction = 1.0 - 0.03 * math.log2(ratio)
    return min(1.00, max(0.90, correction))


def _acbc_rebalance_factor(m_target: int, m_orig: float,
                           role: str) -> float:
    """ACBC parasitic correction. 4th gm-cell Mgm4 + bias-comp chain
    boosts mirror_load bias at node nb. Small upward correction on
    ``mirror_load`` only.

        correction = 1 + 0.04 * log2(max(1.0, m_target / m_orig))
                     clipped to [1.00, 1.08].
    """
    if role != "mirror_load":
        return 1.0
    import math
    ratio = max(1.0, float(m_target) / float(m_orig))
    correction = 1.0 + 0.04 * math.log2(ratio)
    return min(1.08, max(1.00, correction))


def _dfcfc2_rebalance_factor(m_target: int, m_orig: float,
                             role: str) -> float:
    """DFCFC2 parasitic correction. Mgf_damp damping FF + Mlcs_damp
    sink form a damping pole-zero pair coupled to active_ff. Small
    upward correction on ``active_ff`` only.

        correction = 1 + 0.03 * log2(max(1.0, m_target / m_orig))
                     clipped to [1.00, 1.06].
    """
    if role != "active_ff":
        return 1.0
    import math
    ratio = max(1.0, float(m_target) / float(m_orig))
    correction = 1.0 + 0.03 * math.log2(ratio)
    return min(1.06, max(1.00, correction))


#: Per-cohort rebalance correction registry. Maps cohort_key (the
#: same style strings used by ``_RESOLVE_ROLE_MAP_BY_TEMPLATE``) to
#: a callable returning a multiplier on the R94 B linear factor.
#: Cohorts not in this registry use ``_default_rebalance_factor``
#: (returns 1.0 = no perturbation).
_REBALANCE_FACTOR_FN: dict[str, Callable[[int, float, str], float]] = {
    # LEC cohort: local error amplifier loop correction.
    "nested_miller_3stage_lec_cascoded_input_mirror":
        _lec_rebalance_factor,
    # IAC cohort: triode-helper correction.
    "nested_miller_3stage_iac_cascoded_input_mirror":
        _iac_rebalance_factor,
    "nested_miller_3stage_iac_redesign":
        _iac_rebalance_factor,
    # ACBC cohort: bias-comp boost correction.
    "nested_miller_3stage_acbc_cascoded_input_mirror":
        _acbc_rebalance_factor,
    # DFCFC2 cohort: damping FF correction.
    "nested_miller_3stage_dfcfc2_cascoded_input_mirror":
        _dfcfc2_rebalance_factor,
}


# ──────────────────────────────────────────────────────────────
# R98 F per-PVT learned LUT for rebalance corrections
# ──────────────────────────────────────────────────────────────
#
# R96 B's fixed log2-band formulas (``_lec_rebalance_factor`` /
# ``_iac_rebalance_factor`` / ``_acbc_rebalance_factor`` /
# ``_dfcfc2_rebalance_factor``) are analytical approximations to the
# true per-cohort parasitic correction. R98 F replaces them -- when a
# calibrated entry is present -- with a PVT-learned LUT generated by
# real ngspice M-sweeps. The LUT is keyed by
# ``(cohort_key, corner, temperature_c)`` and stores
# ``{role: {m_target_int: correction_factor_float}}`` per entry -- the
# *measured* correction multiplier that minimises gain deviation from
# the M=1 baseline on the ngspice operating point.
#
# Fallback behaviour. The LUT lookup is byte-equal-preserving: when
# a lookup miss occurs (cohort not calibrated for this corner /
# m_target / role / etc.) the resolver falls back to the R96 B fixed
# function in ``_REBALANCE_FACTOR_FN`` -- which itself falls back to
# the identity (``_default_rebalance_factor``) for cohorts without
# a fixed function. R94 B + R95 B + R96 B byte-exactness is therefore
# preserved when no LUT entries are registered.
#
# LUT JSON schema (see ``data/pvt_rebalance_luts/sky130_nmcnr.json``):
#
#   {
#     "schema_version": "v1.0",
#     "cohort": "<style key>",
#     "pdk": "sky130" | "gf180mcu" | ...,
#     "corner": "tt" | "ff" | "ss" | ...,
#     "temperature_c": 27,
#     "calibration_metadata": {
#       "ngspice_version": "...",
#       "calibration_date": "YYYY-MM-DD",
#       "m_orig": 1.0,
#       "m_target_sweep": [1, 2, 4, 8, 16],
#       "method": "real_ngspice_m_sweep",
#       ...
#     },
#     "entries": {
#       "<role>": {"<m_target_int>": <correction_factor_float>, ...},
#       ...
#     }
#   }
#
# Registry semantics. ``_REBALANCE_LUT`` is keyed by
# ``(cohort_key, corner, temperature_c)`` and stores the inner
# ``{role: {m_target_int: factor}}`` dict. Multiple corners and
# temperatures coexist; the lookup picks the matching tuple. R98 F
# ships ``tt`` / 27 C only as the honest scope (R99+ extends to
# ``ff`` / ``ss`` and additional temperatures + PDKs).

#: PVT LUT data keyed by ``(cohort_key, corner, temperature_c)``.
#: Each value is a dict ``{role: {m_target_int: correction_factor_float}}``.
#: Empty at module load; populated by
#: ``register_pvt_rebalance_lut`` /
#: ``load_pvt_rebalance_lut_from_json`` calls. R96 B byte-exact
#: preservation: when this dict is empty, every call resolves to the
#: fixed R96 B ``_REBALANCE_FACTOR_FN`` (or default identity).
_REBALANCE_LUT: dict[
    tuple[str, str, int],
    dict[str, dict[int, float]],
] = {}


#: Default PVT context -- when callers don't supply a corner /
#: temperature, R98 F looks up ``("tt", 27)``. R99+ extends to other
#: corners; until then the default matches the typical sizing path
#: (sky130 tt corner at room temperature).
DEFAULT_LUT_CORNER: str = "tt"
DEFAULT_LUT_TEMPERATURE_C: int = 27


def register_pvt_rebalance_lut(
    cohort_key: str,
    entries: Mapping[str, Mapping[int, float]],
    *,
    corner: str = DEFAULT_LUT_CORNER,
    temperature_c: int = DEFAULT_LUT_TEMPERATURE_C,
) -> None:
    """Register a PVT-learned LUT for a (cohort, corner, temperature) tuple.

    Subsequent ``apply_finger_count_to_role_with_rebalance`` calls with
    a matching ``cohort_key`` will consult this LUT first; on miss
    (unknown role or m_target not in the table) they fall back to the
    R96 B fixed function (``_REBALANCE_FACTOR_FN``), preserving R96 B
    byte-exactness when the LUT covers a strict subset.

    Args:
        cohort_key: topology style key (same vocabulary as
            ``_RESOLVE_ROLE_MAP_BY_TEMPLATE`` keys).
        entries: nested dict ``{role: {m_target_int: factor_float}}``.
            Integer m_target keys are required; float factors.
        corner: PVT corner string. Default ``"tt"``.
        temperature_c: temperature in Celsius. Default ``27``.

    Raises:
        TypeError: when ``cohort_key`` is None or non-string.
        ValueError: when any factor is non-positive.
    """
    if not isinstance(cohort_key, str) or not cohort_key:
        raise TypeError(
            f"cohort_key must be a non-empty string, got {cohort_key!r}"
        )
    if not isinstance(corner, str) or not corner:
        raise TypeError(
            f"corner must be a non-empty string, got {corner!r}"
        )
    if not isinstance(temperature_c, int):
        raise TypeError(
            f"temperature_c must be int, got {type(temperature_c).__name__}"
        )

    normalized: dict[str, dict[int, float]] = {}
    for role, mtable in entries.items():
        if not isinstance(role, str) or not role:
            raise TypeError(
                f"role keys must be non-empty strings, got {role!r}"
            )
        inner: dict[int, float] = {}
        for m_key, factor in mtable.items():
            m_int = int(m_key)
            if m_int < 1:
                raise ValueError(
                    f"m_target must be >= 1, got {m_key!r} for "
                    f"cohort={cohort_key} role={role}"
                )
            factor_f = float(factor)
            if factor_f <= 0:
                raise ValueError(
                    f"correction factor must be > 0, got {factor!r} for "
                    f"cohort={cohort_key} role={role} m={m_int}"
                )
            inner[m_int] = factor_f
        normalized[role] = inner
    _REBALANCE_LUT[(cohort_key, corner, temperature_c)] = normalized


def unregister_pvt_rebalance_lut(
    cohort_key: str,
    *,
    corner: str = DEFAULT_LUT_CORNER,
    temperature_c: int = DEFAULT_LUT_TEMPERATURE_C,
) -> bool:
    """Remove a registered LUT entry. Returns True if removed, False if
    no entry existed for the given (cohort, corner, temperature) tuple.

    Primarily for tests / hot-reload. Production callers register once
    at startup and leave the LUT in place.
    """
    return _REBALANCE_LUT.pop(
        (cohort_key, corner, temperature_c), None,
    ) is not None


def list_registered_pvt_luts() -> list[tuple[str, str, int]]:
    """Return the list of registered ``(cohort_key, corner, temperature_c)``
    tuples. Read-only snapshot; mutations to the returned list don't
    affect the registry.
    """
    return list(_REBALANCE_LUT.keys())


def load_pvt_rebalance_lut_from_json(path) -> tuple[str, str, int]:
    """Load a single LUT JSON file and register it via
    ``register_pvt_rebalance_lut``. Returns the registered
    ``(cohort_key, corner, temperature_c)`` tuple for echo / logging.

    Schema validation: requires ``cohort``, ``corner``,
    ``temperature_c``, and ``entries`` keys at the top level. The
    ``entries`` value must be a nested dict mapping role -> m_target
    string keys (JSON has string keys; we cast to int) -> factor.

    Args:
        path: filesystem path (str or Path) to the JSON file.

    Returns:
        ``(cohort_key, corner, temperature_c)`` triple that was
        registered.

    Raises:
        FileNotFoundError, json.JSONDecodeError, KeyError, ValueError
        on malformed / missing input.
    """
    import json
    from pathlib import Path as _Path
    p = _Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    for required in ("cohort", "corner", "temperature_c", "entries"):
        if required not in raw:
            raise KeyError(
                f"PVT LUT JSON at {p} missing required key: {required!r}"
            )
    cohort_key = str(raw["cohort"])
    corner = str(raw["corner"])
    temperature_c = int(raw["temperature_c"])
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, dict):
        raise ValueError(
            f"PVT LUT JSON at {p} 'entries' must be a dict, got "
            f"{type(entries_raw).__name__}"
        )
    # Cast m_target string keys to int (JSON has no integer keys).
    entries: dict[str, dict[int, float]] = {}
    for role, mtable in entries_raw.items():
        if not isinstance(mtable, dict):
            raise ValueError(
                f"PVT LUT JSON at {p} entries[{role!r}] must be a dict, "
                f"got {type(mtable).__name__}"
            )
        entries[role] = {int(k): float(v) for k, v in mtable.items()}
    register_pvt_rebalance_lut(
        cohort_key, entries,
        corner=corner, temperature_c=temperature_c,
    )
    return (cohort_key, corner, temperature_c)


def lookup_pvt_rebalance_correction(
    cohort_key: Optional[str],
    role: str,
    m_target: int,
    m_orig: float,  # noqa: ARG001 -- present for API symmetry / future use
    *,
    corner: str = DEFAULT_LUT_CORNER,
    temperature_c: int = DEFAULT_LUT_TEMPERATURE_C,
) -> Optional[float]:
    """Look up a PVT-learned correction factor.

    Returns the learned multiplier when a (cohort, corner,
    temperature, role, m_target) entry is registered. Returns
    ``None`` on any miss (no cohort entry / no role entry / no
    m_target entry) -- callers should treat None as "fall back to
    the R96 B fixed function".

    Args:
        cohort_key: topology style key. None / unknown -> None.
        role: role name (e.g. ``"mirror_cascode"``).
        m_target: integer M target. Floats are floored to int.
        m_orig: original M (unused in lookup; reserved for future
            ratio-based LUTs).
        corner: PVT corner. Default ``"tt"``.
        temperature_c: temperature. Default 27.

    Returns:
        A positive float when the entry exists; None on miss.
    """
    if cohort_key is None:
        return None
    inner = _REBALANCE_LUT.get((cohort_key, corner, temperature_c))
    if inner is None:
        return None
    role_table = inner.get(role)
    if role_table is None:
        return None
    return role_table.get(int(m_target))


def _make_lut_first_correction_fn(
    cohort_key: Optional[str],
    *,
    corner: str = DEFAULT_LUT_CORNER,
    temperature_c: int = DEFAULT_LUT_TEMPERATURE_C,
) -> Callable[[int, float, str], float]:
    """Return a callable that consults the LUT first, then falls back
    to the R96 B fixed function for the given cohort_key.

    The returned callable has the same signature as the R96 B fixed
    functions: ``(m_target, m_orig, role) -> float``. This keeps the
    apply_* call sites byte-identical to R96 B; the LUT layer is a
    pure wrapper.

    When the LUT registry is empty for this cohort (and corner /
    temperature) the wrapper is byte-equal to
    ``resolve_rebalance_factor_fn(cohort_key)`` -- R96 B byte-exact.
    """
    fixed_fn = (
        _REBALANCE_FACTOR_FN.get(cohort_key, _default_rebalance_factor)
        if cohort_key is not None
        else _default_rebalance_factor
    )

    def _correction(m_target: int, m_orig: float, role: str) -> float:
        learned = lookup_pvt_rebalance_correction(
            cohort_key, role, m_target, m_orig,
            corner=corner, temperature_c=temperature_c,
        )
        if learned is not None:
            return learned
        # Fall back to R96 B fixed function (or identity).
        return fixed_fn(m_target, m_orig, role)

    return _correction


def resolve_rebalance_factor_fn(
    cohort_key: Optional[str],
    *,
    corner: Optional[str] = None,
    temperature_c: Optional[int] = None,
) -> Callable[[int, float, str], float]:
    """Return the per-cohort parasitic-aware rebalance correction
    function for the given template style key.

    R98 F extension: when ``corner`` and ``temperature_c`` are
    explicitly supplied OR the LUT registry holds an entry for the
    cohort under the default PVT context (``tt`` / 27 C), the returned
    callable consults the LUT first and falls back to the R96 B fixed
    function on miss. When both kwargs are ``None`` AND the cohort has
    no registered LUT entries, the resolver returns the R96 B fixed
    function directly (byte-equal to pre-R98 F behaviour).

    Args:
        cohort_key: a topology style string (same vocabulary as
            ``_RESOLVE_ROLE_MAP_BY_TEMPLATE`` keys). Pass ``None`` for
            the default identity function (R94 B byte-equal).
        corner: optional PVT corner. When None and the registry holds
            an entry for ``(cohort_key, "tt", 27)`` the LUT path
            engages anyway (the default PVT lookup). When supplied
            explicitly, the LUT path always engages.
        temperature_c: optional temperature in C. None means default 27.

    Returns:
        A callable ``(m_target, m_orig, role) -> float`` returning the
        cohort's correction multiplier. Returns
        ``_default_rebalance_factor`` for unknown / None keys when no
        LUT applies.
    """
    if cohort_key is None:
        return _default_rebalance_factor

    # When the caller supplies PVT kwargs explicitly, always wrap in
    # the LUT-first resolver (even if the LUT entry is missing -- the
    # wrapper still falls back to the fixed fn).
    explicit_pvt = corner is not None or temperature_c is not None
    eff_corner = corner if corner is not None else DEFAULT_LUT_CORNER
    eff_temp = (
        temperature_c if temperature_c is not None else DEFAULT_LUT_TEMPERATURE_C
    )

    # When PVT kwargs are NOT supplied, only wrap when the default
    # ``(cohort_key, "tt", 27)`` entry exists. This preserves R96 B
    # byte-equal behaviour for callers that don't opt into LUT.
    default_entry_present = (
        (cohort_key, DEFAULT_LUT_CORNER, DEFAULT_LUT_TEMPERATURE_C)
        in _REBALANCE_LUT
    )
    if explicit_pvt or default_entry_present:
        return _make_lut_first_correction_fn(
            cohort_key,
            corner=eff_corner,
            temperature_c=eff_temp,
        )

    # No LUT engagement path: byte-equal to pre-R98 F resolver.
    return _REBALANCE_FACTOR_FN.get(
        cohort_key, _default_rebalance_factor,
    )


def apply_finger_count_to_role_with_rebalance(
    cell,
    role: str,
    n_finger: int,
    *,
    role_device_map: Optional[dict[str, tuple[str, ...]]] = None,
    rebalance: str = "none",
    strict: bool = True,
    cohort_key: Optional[str] = None,
):
    """Set the SPICE m= multiplier on every role device, optionally
    rebalancing W to preserve g_m * M effective.

    R94 B extension of ``apply_finger_count_to_role``. With ``rebalance
    = "none"`` (default) this is byte-equal to ``apply_finger_count_to_
    role``. With ``rebalance = "g_m_eff"`` the device W is also rescaled
    by ``(M_orig / M_target)`` — the layout-finger decomposition that
    splits the design-time effective W into N_finger parallel fingers
    while keeping total active width constant. This restores the
    operating point that blind M-threading (R93 B baseline) destroys
    (crashes NMCNR diff_pair gain 119 -> 72 dB at M=8 without rebalance;
    preserves it within +/-1 dB up to M=8 with rebalance enabled).

    R96 B addition: optional ``cohort_key`` argument selects a per-
    cohort parasitic-aware correction multiplier on top of the linear
    R94 B factor. Cohorts with active cascode/feedback loops (LEC,
    IAC, ACBC, DFCFC2) register custom corrections. Default
    (``cohort_key=None``) returns the R94 B linear factor unchanged,
    so existing call sites get byte-equal behaviour without change.

    R97 F addition: ``cohort_key="auto"`` triggers introspection of the
    cell IR via ``_cohort_key_from_cell(cell)`` -- callers no longer
    need to know the cohort_key explicitly. Resolution path:
    ``cell.topology["style"]`` (CellDef) -> ``cell.topology_style``
    attribute -> ``None`` (no hint discoverable, identity fallback).
    Passing ``cohort_key=None`` or an explicit string preserves R96 B
    byte-equal behaviour.

    Args:
        cell, role, n_finger, role_device_map, strict:
            Same as ``apply_finger_count_to_role``.
        rebalance: ``"none"`` or ``"g_m_eff"``. Default ``"none"``.
            See ``REBALANCE_MODES`` for the full list.
        cohort_key: optional topology style string (same vocabulary
            as ``_RESOLVE_ROLE_MAP_BY_TEMPLATE`` keys). When supplied
            and ``rebalance == "g_m_eff"`` the per-cohort correction
            multiplier from ``_REBALANCE_FACTOR_FN`` is applied on
            top of the R94 B linear factor. Default ``None`` ->
            identity correction (R94 B byte-equal). R97 F: pass
            ``"auto"`` to resolve via ``_cohort_key_from_cell(cell)``;
            when the cell carries no style hint the resolver returns
            ``None`` and the identity fallback applies (R96 B
            byte-equal).

    Returns:
        Same cell reference (mutated in place).

    Raises:
        ValueError: invalid rebalance mode, or non-positive n_finger.
        KeyError / ValueError: (when strict) per
            ``apply_finger_count_to_role`` rules.
    """
    if rebalance not in REBALANCE_MODES:
        raise ValueError(
            f"rebalance mode {rebalance!r} not in {REBALANCE_MODES}"
        )

    if not isinstance(n_finger, int) or n_finger < 1:
        raise ValueError(
            f"n_finger must be a positive integer, got {n_finger!r} "
            f"(type {type(n_finger).__name__})"
        )

    # R97 F: resolve ``cohort_key="auto"`` from the cell IR. ``None``
    # and explicit strings pass through unchanged (R96 B byte-equal).
    cohort_key_eff = _resolve_cohort_key_arg(cohort_key, cell)

    rd_map = role_device_map if role_device_map is not None else DEFAULT_ROLE_DEVICE_MAP
    if role not in rd_map:
        if strict:
            raise KeyError(
                f"role {role!r} not in role_device_map. Valid roles: "
                f"{sorted(rd_map)}"
            )
        return cell

    target_names = rd_map[role]
    by_name = {d.instance_name: d for d in cell.devices}
    missing = [n for n in target_names if n not in by_name]
    if missing and strict:
        raise ValueError(
            f"role {role!r}: device(s) {missing!r} not found in "
            f"cell.devices (available: {sorted(by_name)})"
        )

    # Resolve the per-cohort correction once per call -- the same
    # correction applies to every device under this role.
    cohort_correction_fn = resolve_rebalance_factor_fn(cohort_key_eff)

    for name in target_names:
        dev = by_name.get(name)
        if dev is None:
            continue
        m_orig = float(dev.sizing.get("M", 1.0))
        # Apply the M update.
        dev.sizing["M"] = float(n_finger)
        # Apply W rebalance if requested.
        if rebalance == "g_m_eff":
            w_orig = dev.sizing.get("W")
            if w_orig is not None:
                base_factor = _gm_eff_w_rebalance_factor(
                    n_finger, m_orig,
                )
                # R96 B: per-cohort parasitic-aware correction.
                # Default (no cohort_key) returns 1.0 -> byte-equal
                # to R94 B.
                correction = cohort_correction_fn(
                    n_finger, m_orig, role,
                )
                dev.sizing["W"] = float(w_orig) * base_factor * correction
    return cell


def apply_selection_to_cell_with_rebalance(
    cell,
    selection: LayoutAwareSelection,
    *,
    role: str = "diff_pair",
    role_device_map: Optional[dict[str, tuple[str, ...]]] = None,
    apply_w: bool = False,
    apply_l: bool = False,
    rebalance: str = "none",
    strict: bool = True,
    cohort_key: Optional[str] = None,
):
    """One-call back-propagation with optional g_m_eff rebalance.

    R94 B extension of ``apply_selection_to_cell``. The ``rebalance``
    argument controls whether the M update also rescales W to preserve
    g_m * M effective (see ``apply_finger_count_to_role_with_rebalance``
    for the math).

    R96 B addition: optional ``cohort_key`` forwards to
    ``apply_finger_count_to_role_with_rebalance`` so per-cohort
    parasitic-aware corrections (LEC / IAC / ACBC / DFCFC2) apply
    to the rebalance. Default ``None`` -> R94 B byte-equal.

    R97 F addition: ``cohort_key="auto"`` triggers
    ``_cohort_key_from_cell(cell)`` introspection in the downstream
    ``apply_finger_count_to_role_with_rebalance`` call -- callers no
    longer need to pass the topology style explicitly. ``None`` and
    explicit strings remain byte-equal to R96 B.

    Precedence: when both ``apply_w`` and ``rebalance="g_m_eff"`` are
    set, ``apply_w`` wins (the selection's chosen W replaces the
    template's W outright; no implicit rebalance on top of an explicit
    override).

    Args:
        cell, selection, role, role_device_map, apply_w, apply_l, strict:
            Same as ``apply_selection_to_cell``.
        rebalance: ``"none"`` or ``"g_m_eff"``. Default ``"none"`` =
            R93 B byte-equal behaviour. ``"g_m_eff"`` rescales W by
            ``sqrt(M_orig / M_target)``.
        cohort_key: optional topology style key for the per-cohort
            parasitic correction. Default ``None`` (R94 B identity).
            R97 F: pass ``"auto"`` to resolve via
            ``_cohort_key_from_cell(cell)``.

    Returns:
        Same cell reference (mutated in place).
    """
    if apply_w:
        # The caller is explicitly overriding W from the selection.
        # Don't double-apply the rebalance on top.
        apply_finger_count_to_role(
            cell, role, selection.finger_count,
            role_device_map=role_device_map, strict=strict,
        )
    else:
        # R97 F: cohort_key="auto" flows through verbatim; the
        # downstream apply_finger_count_to_role_with_rebalance resolves
        # it from the cell.
        apply_finger_count_to_role_with_rebalance(
            cell, role, selection.finger_count,
            role_device_map=role_device_map,
            rebalance=rebalance, strict=strict,
            cohort_key=cohort_key,
        )
    if apply_w or apply_l:
        rd_map = (
            role_device_map if role_device_map is not None
            else DEFAULT_ROLE_DEVICE_MAP
        )
        target_names = rd_map.get(role, ())
        by_name = {d.instance_name: d for d in cell.devices}
        for name in target_names:
            dev = by_name.get(name)
            if dev is None:
                continue
            if apply_w:
                dev.sizing["W"] = selection.w_um * 1e-6
            if apply_l:
                dev.sizing["L"] = selection.l_um * 1e-6
    return cell


def apply_multi_role_selection_to_cell(
    cell,
    multi_selection: MultiRoleSelection,
    *,
    role_device_map: Optional[dict[str, tuple[str, ...]]] = None,
    apply_w_roles: Optional[Sequence[str]] = None,
    apply_l_roles: Optional[Sequence[str]] = None,
    rebalance: str = "none",
    strict: bool = True,
    cohort_key: Optional[str] = None,
):
    """Apply a ``MultiRoleSelection`` to a cell role-by-role.

    For each role in ``multi_selection.per_role``, threads the
    finger_count via ``apply_finger_count_to_role_with_rebalance``
    and (if requested) overwrites W/L on that role's devices.

    R96 B addition: optional ``cohort_key`` propagates the per-cohort
    parasitic-aware correction down to every per-role call. Default
    ``None`` -> R94 B byte-equal across every role.

    R97 F addition: ``cohort_key="auto"`` triggers
    ``_cohort_key_from_cell(cell)`` resolution ONCE at the multi-role
    level (rather than re-resolving on every per-role apply call). The
    resolved cohort_key is then forwarded to every per-role
    ``apply_selection_to_cell_with_rebalance`` call -- same cohort,
    same correction across all roles. ``None`` / explicit strings
    behave byte-equal to R96 B.

    Args:
        cell: a clay.ir.Cell instance.
        multi_selection: the ``MultiRoleSelection`` from
            ``select_layout_aware_sizing_multi_role``.
        role_device_map: optional cohort map; default NMCNR.
        apply_w_roles / apply_l_roles: optional iterables of role names
            whose W / L should be overwritten with the selection's
            w_um / l_um. Default: don't overwrite.
        rebalance: ``"none"`` or ``"g_m_eff"``. Default ``"none"``.
            Applied to every role's M update.
        strict: forwarded.
        cohort_key: optional topology style key for the per-cohort
            parasitic correction (forwarded to every per-role call).
            R97 F: pass ``"auto"`` to resolve from the cell once at
            entry; downstream calls receive the resolved string.

    Returns:
        Same cell reference (mutated in place).
    """
    apply_w_set = set(apply_w_roles or ())
    apply_l_set = set(apply_l_roles or ())
    rd_map = (
        role_device_map if role_device_map is not None
        else DEFAULT_ROLE_DEVICE_MAP
    )
    # R97 F: resolve ``cohort_key="auto"`` once at multi-role entry so
    # every per-role apply call sees the same resolved key. ``None`` /
    # explicit strings pass through unchanged (R96 B byte-equal).
    cohort_key_eff = _resolve_cohort_key_arg(cohort_key, cell)
    for role, sel in multi_selection.per_role.items():
        apply_selection_to_cell_with_rebalance(
            cell, sel,
            role=role,
            role_device_map=rd_map,
            apply_w=role in apply_w_set,
            apply_l=role in apply_l_set,
            rebalance=rebalance,
            strict=strict,
            cohort_key=cohort_key_eff,
        )
    return cell
