"""R5: real-Magic parasitic balance on a symmetric route pair.

End-to-end proof of the user's third requirement: a SYMMETRIC layout yields
BALANCED parasitics, so the post-layout differential performance tracks the
pre-layout (ideal) one. We emit two metal routes to a real GDS, extract them with
the calibrated Magic backend (v0.3.24), and compare the two nets' parasitic
capacitance as a RELATIVE imbalance. A mirror-symmetric pair extracts to balanced
caps; a deliberately asymmetric pair does not — measured on REAL Magic, not a
model.

Gated on the Magic Docker image (like the other PEX e2e paths); raises if absent
rather than fabricating a number.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.pex import MagicPexBackend
from clay.layout.pex import magic_runner as mr
from clay.layout.pex.symmetry_balance import NetPairBalance

METAL1 = (34, 0)  # gf180mcu metal1


def mirror_rect_route(x0, y0, x1, y1, axis_x, *, layer=METAL1):
    """A metal rectangle and its exact x-mirror about ``axis_x`` (dbu).

    Returns ``(left_polys, right_polys)`` — two disjoint metal routes that Magic
    will extract as two separate nets, geometrically mirror-symmetric.
    """
    left = [LayerPolygon(layer=layer[0], datatype=layer[1],
                         polygon=Polygon.rectangle(x0, y0, x1, y1))]
    rx0, rx1 = 2 * axis_x - x1, 2 * axis_x - x0
    right = [LayerPolygon(layer=layer[0], datatype=layer[1],
                          polygon=Polygon.rectangle(rx0, y0, rx1, y1))]
    return left, right


GF180_METAL1_LABEL_KEY = (34, 10)  # gf180mcu metal1 text-label layer


def emit_labeled_routes_to_gds(
    named_polys, cell_name, gds_path, *, dbu=0.001
):
    """Emit metal routes WITH a per-net text label so Magic names the nets.

    ``named_polys`` maps a net name -> a sequence of metal1 LayerPolygon. Each net
    gets a text label (on the gf180 metal1 label layer 34/10) at its first poly's
    centroid, so ``ext2spice`` reports the net under that name (verified: Magic
    reads GDS text labels). Returns ``gds_path``.
    """
    import klayout.db as db

    ly = db.Layout()
    ly.dbu = dbu
    top = ly.create_cell(cell_name)
    m1 = ly.layer(METAL1[0], METAL1[1])
    lbl = ly.layer(GF180_METAL1_LABEL_KEY[0], GF180_METAL1_LABEL_KEY[1])
    for net, polys in named_polys.items():
        first = None
        for lp in polys:
            pts = [db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]
            top.shapes(m1).insert(db.Polygon(pts))
            if first is None:
                first = lp
        if first is not None:
            xs = [v.x for v in first.polygon.vertices]
            ys = [v.y for v in first.polygon.vertices]
            cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
            top.shapes(lbl).insert(db.Text(str(net), db.Trans(db.Vector(int(cx), int(cy)))))
    ly.write(str(gds_path))
    return str(gds_path)


@dataclass(frozen=True)
class PairPexBalance:
    """Real-Magic parasitic balance of a two-net route pair."""

    cap_a_aF: float
    cap_b_aF: float
    rel_imbalance: float
    rel_tol: float = 0.05
    backend: str = "magic"
    calibrated: bool = True
    field_solver: bool = False

    @property
    def balanced(self) -> bool:
        return self.rel_imbalance <= self.rel_tol


def extract_pair_balance(
    left_polys: Sequence[LayerPolygon],
    right_polys: Sequence[LayerPolygon],
    *,
    cell_name: str = "sym_pair",
    pdk: str = "gf180mcu",
    rel_tol: float = 0.05,
    workdir: Optional[str] = None,
) -> PairPexBalance:
    """Emit both routes to one GDS, extract with Magic, compare the two nets' caps.

    Requires the Magic Docker image. The two poly sets must be disjoint so Magic
    partitions them into two nets; their total caps are compared as a relative
    imbalance.
    """
    if not mr.magic_available():
        raise RuntimeError("Magic unavailable (no Docker image)")
    from clay.selfboot.magic_pex_aware import emit_polygons_to_gds

    d = workdir or tempfile.mkdtemp(prefix="sym_pex_")
    gds = os.path.join(d, f"{cell_name}.gds")
    emit_polygons_to_gds(list(left_polys) + list(right_polys), cell_name, gds,
                         only_layers={METAL1})
    rep = MagicPexBackend().extract(gds, cell_name, pdk)
    from .symmetry_balance import DEFAULT_GROUND_NETS
    ground = {g.lower() for g in DEFAULT_GROUND_NETS}
    # the two SIGNAL nets are the metal routes; exclude the substrate/ground node
    # (e.g. VSUBS) which carries the sum of both routes' substrate caps.
    signal = sorted(n for n in rep.nets() if n.lower() not in ground)
    if len(signal) < 2:
        raise RuntimeError(
            f"expected 2 signal nets from the route pair, got {len(signal)}: {signal} "
            f"(all nets: {sorted(rep.nets())})")
    ca = rep.net_total_cap_aF(signal[0])
    cb = rep.net_total_cap_aF(signal[1])
    denom = max(ca, cb, 1e-30)
    return PairPexBalance(
        cap_a_aF=ca, cap_b_aF=cb, rel_imbalance=abs(ca - cb) / denom,
        rel_tol=rel_tol, calibrated=rep.calibrated, field_solver=rep.field_solver,
    )


def extract_named_diff_balance(
    named_polys,
    diff_pair,
    *,
    cell_name: str = "named_diff",
    pdk: str = "gf180mcu",
    rel_tol: float = 0.05,
    workdir: Optional[str] = None,
):
    """Emit labeled routes (incl any aggressor), extract with Magic, and gate the
    named differential pair's parasitic balance.

    ``named_polys`` maps net name -> metal1 polys (e.g. {"clk": [...], "outn": [...],
    "outp": [...]}); ``diff_pair`` is the (net_a, net_b) to balance. Returns the
    :class:`ParasiticSymmetryVerdict`. Requires the Magic Docker image.
    """
    if not mr.magic_available():
        raise RuntimeError("Magic unavailable (no Docker image)")
    from clay.layout.symmetry_derive import SymmetryGroup
    from clay.layout.pex.symmetry_quality_gate import parasitic_symmetry_gate

    d = workdir or tempfile.mkdtemp(prefix="named_diff_")
    gds = os.path.join(d, f"{cell_name}.gds")
    emit_labeled_routes_to_gds(named_polys, cell_name, gds)
    rep = MagicPexBackend().extract(gds, cell_name, pdk)
    group = SymmetryGroup(pairs=(("A", "B"),), differential_nets=(tuple(diff_pair),),
                          source="explicit", kind="diff_pair")
    return parasitic_symmetry_gate(group, rep, rel_tol=rel_tol), rep
