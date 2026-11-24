"""Inter-net lateral COUPLING capacitance for the geometric PEX estimate.

The per-net substrate model in ``pex_estimate_v164`` emits only cap-to-substrate
(area + perimeter-fringe to the bulk). It carries NO net-to-net coupling, so a
kickback / clk->regen obligation that reads the coupling cap between an aggressor
net and a victim pair sees 0 and reports the relationship as unmeasurable.

This module adds the dominant missing term: LATERAL (same-layer edge-to-edge)
coupling between two nets whose metal shapes run adjacent across a small gap. It
is a real PHYSICS-BASED GEOMETRIC estimate (parallel-plate between the facing
side-walls + side fringe), built from the actual shape geometry (the facing
parallel-run length and the real edge-to-edge spacing) and the real per-layer
conductor thickness / dielectric constant. It is NOT a field solve and NOT a
calibrated StarRC / foundry-ITF number: ``calibrated`` stays False on the report.

Model (per facing edge-pair, both shapes on the same metal layer)::

    C_lateral(aF) = eps * (T / s) * L * k_fringe

  * ``eps``      = eps0 * eps_r of the inter-metal dielectric (SiO2, k~4.0),
                   in aF/um  (eps0 = 8.8541878128 aF/um).
  * ``T``        = metal conductor THICKNESS (um) -- the real BEOL dim; the
                   side-wall plate height that faces the neighbour.
  * ``s``        = edge-to-edge SPACING (um) between the two shapes across the
                   gap (the real geometric separation; never below the layer's
                   min spacing, which floors the coupling at the densest legal
                   pitch instead of blowing up as s->0).
  * ``L``        = parallel-RUN length (um): the overlap of the two shapes along
                   the axis parallel to the facing edges. Two shapes that do not
                   overlap on that axis do not face each other -> L=0 -> C=0.
  * ``k_fringe`` = dimensionless multiplier (~2) for the two side-fringe field
                   paths that wrap above/below the pure parallel-plate edge term.

Coupling therefore scales as run-length / spacing -- the textbook lateral-cap
dependence. Nets that never run adjacent (no facing overlap, or a gap wider than
``max_gap_um``) couple ~0, honestly. Only pairs that actually run side-by-side
(the differential / clock nets in a matched layout) get a non-zero, and a
symmetric layout yields balanced (but non-vacuous) coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# SI conversion: 8.8541878128e-12 F/m * 1e-6 m/um * 1e18 aF/F.
from .pdk_constants import EPS0_aF_PER_UM
SIO2_K: float = 4.0
EPS_ILD_aF_PER_UM: float = EPS0_aF_PER_UM * SIO2_K  # ~35.4 aF/um

# gf180mcu metal1 (GF 0.18um Al 1P5M BEOL) real geometric dims:
#   * conductor thickness ~0.53 um (Al ~0.5 um; consistent with the Rsheet
#     ~90 mOhm/sq used by the substrate model).
#   * min edge-to-edge spacing 0.23 um -- from the COMMITTED gf180 DRC deck
#     (rule M1.2a: "min. metal1 spacing is 0.23um"). This floors ``s`` so the
#     1/s term saturates at the densest legal pitch, not at an unphysical s->0.
# These are documented process-typical dims (calibrated=False), NOT an ITF file.
GF180_METAL1_THICKNESS_UM: float = 0.53
GF180_METAL1_MIN_SPACE_UM: float = 0.23

# Side-fringe multiplier on the bare parallel-plate side-wall term. ~2 captures
# the two field paths that fringe over the top and under the bottom of the gap;
# an order-of-magnitude geometric factor, NOT a fitted/calibrated coefficient.
DEFAULT_K_FRINGE: float = 2.0

# Two shapes farther apart than this many MIN-spacings are treated as not
# adjacent. Lateral coupling falls as 1/s, so this is mostly a performance gate:
# pairs beyond it are already negligible. It is set to an intra-cell neighbour
# window (~30 tracks ~ 6.9 um at gf180 metal1) so genuine same-cell facing
# neighbours a few um apart are captured, while the 1/s term keeps distant pairs
# in the noise. NOT a tuned-to-pass coefficient: widening or narrowing it only
# adds/drops already-tiny far-pair terms; it never invents a coupling between
# two nets that do not run side by side (zero facing run -> zero, always).
DEFAULT_MAX_GAP_MULT: float = 30.0


# Default metal1->metal2 vertical plate separation (um) for the OVERLAP
# (metal-over-metal) coupling term. The gf180 default ~= the metal1 thickness
# (a one-thickness inter-layer gap); per-PDK callers override it with the real
# metal1/metal2 height delta from the PDK's interconnect-tech file.
GF180_METAL_VGAP_UM: float = GF180_METAL1_THICKNESS_UM


@dataclass(frozen=True)
class CouplingConstants:
    """Real per-layer geometric dims feeding the lateral + overlap coupling.

    Defaults are documented gf180mcu / GF 0.18um Al-BEOL process-typical values
    (``calibrated=False``); ``min_space_um`` is read from the committed DRC deck.
    Build a PER-PDK instance (real metal thickness / dielectric k / min space /
    metal1->metal2 gap sourced from that PDK's OWN tech file + deck) with
    :meth:`from_pdk`. ``vgap_um`` is the metal1->metal2 plate separation used by
    the vertical metal-over-metal overlap term.
    """

    eps_ild_aF_per_um: float = EPS_ILD_aF_PER_UM
    thickness_um: float = GF180_METAL1_THICKNESS_UM
    min_space_um: float = GF180_METAL1_MIN_SPACE_UM
    vgap_um: float = GF180_METAL_VGAP_UM
    k_fringe: float = DEFAULT_K_FRINGE
    max_gap_mult: float = DEFAULT_MAX_GAP_MULT
    source: str = (
        "gf180mcu / GF 0.18um Al 1P5M BEOL geometric coupling dims "
        "(metal1 T~0.53um, min space 0.23um from committed DRC rule M1.2a, "
        "SiO2 k~4.0); parallel-plate side-wall + side fringe + metal-over-metal "
        "overlap, NOT a field solve / NOT calibrated"
    )

    @classmethod
    def from_pdk(cls, pdk_id: str, *, root: str | None = None) -> "CouplingConstants":
        """Per-PDK coupling constants from the PDK's OWN process data.

        Sources metal1 thickness / inter-metal dielectric k / metal1->metal2 gap
        from the PDK's interconnect-tech file (Cadence ICT or external_lvs RCX
        procfile) and metal1 min spacing from its DRC deck, falling back to a
        documented node-scaled default only where the PDK is silent (see
        :func:`clay.layout.pex.pdk_constants.resolve_pdk_pex_constants`). Stays a
        PDK-SPECIFIC GEOMETRIC estimate (``calibrated=False``); NOT field-solver
        / StarRC. The ``k_fringe`` / ``max_gap_mult`` model factors are unchanged
        (they are model geometry, not per-PDK silicon).
        """
        from .pdk_constants import resolve_pdk_pex_constants

        c = resolve_pdk_pex_constants(pdk_id, root=root)
        real = ", ".join(c.real_fields) or "none"
        dflt = ", ".join(c.defaulted_fields) or "none"
        return cls(
            eps_ild_aF_per_um=c.eps_ild_aF_per_um,
            thickness_um=c.metal_thickness_um,
            min_space_um=c.min_space_um,
            vgap_um=c.metal_vgap_um,
            source=(
                f"{pdk_id} PDK-specific geometric coupling dims "
                f"(metal1 T={c.metal_thickness_um:g}um, ILD k={c.ild_k:g}, "
                f"min space {c.min_space_um:g}um, M1->M2 gap {c.metal_vgap_um:g}um); "
                f"real-from-PDK [{real}], defaulted [{dflt}]; parallel-plate "
                f"side-wall+fringe + metal-over-metal overlap, NOT field-solver / "
                f"NOT calibrated"
            ),
        )


Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1) in dbu


def _facing_run_and_gap_um(
    a: Box, b: Box, dbu_to_um: float
) -> Tuple[float, float]:
    """Parallel-facing run length and edge-to-edge gap (um) between two rects.

    Considers BOTH the vertical facing (a left/right of b -> gap in x, run in y)
    and the horizontal facing (a above/below b -> gap in y, run in x), and
    returns the facing with the LARGER run length (the dominant coupling face).
    Returns ``(run_um, gap_um)``; ``run_um == 0`` when the rects do not project
    onto each other on the relevant axis (they do not run side by side).

    A negative raw gap (the projected spans overlap, i.e. the shapes would touch
    on that axis) is clamped to 0 and handled by the caller's min-spacing floor;
    such pairs are normally the SAME net (union-find joined) and never reach here.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    best_run = 0.0
    best_gap = float("inf")

    # Vertical facing: shapes separated horizontally (gap in x), run along y.
    y_run = min(ay1, by1) - max(ay0, by0)
    if y_run > 0:
        if bx0 >= ax1:
            x_gap = bx0 - ax1
        elif ax0 >= bx1:
            x_gap = ax0 - bx1
        else:
            x_gap = 0  # x-spans overlap -> touching/abutting on x
        run_um = float(y_run) * dbu_to_um
        gap_um = float(x_gap) * dbu_to_um
        if run_um > best_run:
            best_run, best_gap = run_um, gap_um

    # Horizontal facing: shapes separated vertically (gap in y), run along x.
    x_run = min(ax1, bx1) - max(ax0, bx0)
    if x_run > 0:
        if by0 >= ay1:
            y_gap = by0 - ay1
        elif ay0 >= by1:
            y_gap = ay0 - by1
        else:
            y_gap = 0
        run_um = float(x_run) * dbu_to_um
        gap_um = float(y_gap) * dbu_to_um
        if run_um > best_run:
            best_run, best_gap = run_um, gap_um

    if best_run <= 0.0:
        return 0.0, float("inf")
    return best_run, best_gap


def lateral_cap_aF(
    run_um: float, gap_um: float, consts: CouplingConstants
) -> float:
    """Lateral coupling cap (aF) for one facing edge-pair.

    ``C = eps * (T / s) * L * k_fringe`` with ``s`` floored at the layer min
    spacing. Returns 0 for a zero-run (non-facing) pair.
    """
    if run_um <= 0.0:
        return 0.0
    s = max(gap_um, consts.min_space_um)
    return (
        consts.eps_ild_aF_per_um
        * (consts.thickness_um / s)
        * run_um
        * consts.k_fringe
    )


def _overlap_area_um2(a: Box, b: Box, dbu_to_um: float) -> float:
    """Area (um^2) of the axis-aligned intersection of two rects (0 if disjoint).

    This is the metal-over-metal overlap footprint -- the plate area of a
    vertical parallel-plate coupling between two shapes on adjacent routing
    layers. A touching-only intersection (zero-width overlap) has zero area.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    if ix <= 0 or iy <= 0:
        return 0.0
    return float(ix) * float(iy) * (dbu_to_um ** 2)


def overlap_cap_aF(area_um2: float, consts: CouplingConstants) -> float:
    """Vertical metal-over-metal parallel-plate cap (aF) for one overlap footprint.

    ``C = eps_ild * area / vgap`` -- the classic plate cap between two conductors
    on ADJACENT routing layers (e.g. an M1 riser passing under a foreign M2
    trunk), with the real inter-metal dielectric permittivity and the real
    metal1->metal2 vertical gap. Returns 0 for a zero overlap. A purely
    geometric estimate (no fringe-shielding / 3-D term); ``calibrated=False``.
    """
    if area_um2 <= 0.0 or consts.vgap_um <= 0.0:
        return 0.0
    return consts.eps_ild_aF_per_um * area_um2 / consts.vgap_um


# layers that are ADJACENT in the routing stack couple vertically (metal-over-
# metal). We treat any two DIFFERENT metal layers in the per-net layer map as a
# candidate overlap pair (the dominant term is the nearest pair; far pairs are
# attenuated by the much larger gap, but the model uses the configured vgap as a
# conservative single inter-layer separation). Same-layer pairs are handled by
# the lateral term, never here.
def overlap_caps_from_layer_nets(
    layer_nets: Dict[str, Dict[Tuple[int, int], Sequence[Box]]],
    dbu_to_um: float,
    consts: CouplingConstants = CouplingConstants(),
) -> List[Tuple[str, str, float]]:
    """Inter-net VERTICAL overlap caps as ``(net_a, net_b, cap_aF)`` triples.

    ``layer_nets`` maps each net to ``{(layer, datatype): [Box, ...]}`` -- the
    per-LAYER metal shapes of the net (so layer identity is preserved, unlike the
    flat ``bboxes`` partition). For every unordered pair of distinct nets, sums
    the metal-over-metal parallel-plate cap over every pair of their shapes that
    lie on DIFFERENT metal layers and overlap in plan view. Pairs with no
    cross-layer overlap contribute nothing (sparse output). Deterministic.
    """
    names = sorted(layer_nets.keys())
    out: List[Tuple[str, str, float]] = []
    for i in range(len(names)):
        na = names[i]
        la = layer_nets[na]
        if not la:
            continue
        for j in range(i + 1, len(names)):
            nb = names[j]
            lb = layer_nets[nb]
            if not lb:
                continue
            total = 0.0
            for lyr_a, boxes_a in la.items():
                for lyr_b, boxes_b in lb.items():
                    if lyr_a == lyr_b:
                        continue  # same layer -> lateral, not overlap
                    for sa in boxes_a:
                        for sb in boxes_b:
                            area = _overlap_area_um2(sa, sb, dbu_to_um)
                            if area > 0.0:
                                total += overlap_cap_aF(area, consts)
            if total > 0.0:
                out.append((na, nb, total))
    return out


def _merge_triples(
    *triple_lists: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float]]:
    """Sum coupling triples that share the same (sorted) net pair, deterministically."""
    acc: Dict[Tuple[str, str], float] = {}
    for triples in triple_lists:
        for a, b, c in triples:
            key = (a, b) if a <= b else (b, a)
            acc[key] = acc.get(key, 0.0) + c
    return [(a, b, c) for (a, b), c in sorted(acc.items()) if c > 0.0]


def coupling_caps_from_nets(
    nets: Dict[str, Dict[str, object]],
    dbu_to_um: float,
    consts: CouplingConstants = CouplingConstants(),
) -> List[Tuple[str, str, float]]:
    """Inter-net lateral coupling caps as ``(net_a, net_b, cap_aF)`` triples.

    ``nets`` is the partition map (``{net: {"bboxes": [Box, ...], ...}}``) the
    geometric estimator already builds from the REAL GDS connectivity. For every
    unordered pair of DISTINCT nets, sums the lateral coupling over every pair of
    their metal shapes that run adjacent within ``max_gap_mult`` min-spacings.
    Pairs with zero facing run (never side by side) contribute nothing and are
    omitted, so the returned list is sparse: only nets that actually run adjacent
    carry a coupling entry. Deterministic: net pairs are emitted in sorted order.

    When a net entry ALSO carries a ``"layer_bboxes"`` key (``{(layer, dt):
    [Box, ...]}``), the VERTICAL metal-over-metal overlap term is added on top of
    the lateral term (see :func:`overlap_caps_from_layer_nets`). The flat-map
    callers (no layer info) get the lateral term only, so this is fully backward
    compatible -- the public ``(net_a, net_b, cap_aF)`` triple shape is unchanged.
    """
    max_gap_um = consts.max_gap_mult * consts.min_space_um
    names = sorted(nets.keys())
    boxes_by_net: Dict[str, Sequence[Box]] = {
        n: tuple(nets[n].get("bboxes", ()))  # type: ignore[arg-type]
        for n in names
    }
    lateral: List[Tuple[str, str, float]] = []
    for i in range(len(names)):
        na = names[i]
        ba = boxes_by_net[na]
        if not ba:
            continue
        for j in range(i + 1, len(names)):
            nb = names[j]
            bb = boxes_by_net[nb]
            if not bb:
                continue
            total = 0.0
            for sa in ba:
                for sb in bb:
                    run_um, gap_um = _facing_run_and_gap_um(sa, sb, dbu_to_um)
                    if run_um <= 0.0 or gap_um > max_gap_um:
                        continue
                    total += lateral_cap_aF(run_um, gap_um, consts)
            if total > 0.0:
                lateral.append((na, nb, total))

    # vertical metal-over-metal overlap, only when the map carries per-layer shapes.
    layer_nets: Dict[str, Dict[Tuple[int, int], Sequence[Box]]] = {}
    for n in names:
        lb = nets[n].get("layer_bboxes")
        if lb:
            layer_nets[n] = lb  # type: ignore[assignment]
    if not layer_nets:
        return lateral
    overlap = overlap_caps_from_layer_nets(layer_nets, dbu_to_um, consts)
    return _merge_triples(lateral, overlap)


__all__ = [
    "CouplingConstants",
    "EPS_ILD_aF_PER_UM",
    "GF180_METAL1_THICKNESS_UM",
    "GF180_METAL1_MIN_SPACE_UM",
    "GF180_METAL_VGAP_UM",
    "lateral_cap_aF",
    "overlap_cap_aF",
    "coupling_caps_from_nets",
    "overlap_caps_from_layer_nets",
]
