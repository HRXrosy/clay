"""Render a GDS layout to a layer-coloured PNG for human eyeballing.

Self-contained: parses the GDS with klayout.db and draws every polygon with
matplotlib, coloured per layer. No KLayout GUI / Docker needed.

    python tools/render_gds.py <in.gds> <out.png> [title]

Layer colours default to the gf180mcu convention; unknown layers get a fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import klayout.db as db
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

# gf180mcu layer (layer,datatype) -> (label, color, alpha, zorder)
GF180 = {
    (21, 0): ("nwell", "#fff2b2", 0.45, 0),
    (22, 0): ("COMP/diff", "#2ca02c", 0.55, 1),
    (31, 0): ("pplus", "#f7b6d2", 0.30, 1),
    (32, 0): ("nplus", "#aec7e8", 0.30, 1),
    (30, 0): ("poly", "#d62728", 0.70, 2),
    (33, 0): ("contact", "#000000", 0.90, 3),
    (34, 0): ("metal1", "#1f77b4", 0.55, 4),
    (35, 0): ("via1", "#9467bd", 0.95, 5),
    (36, 0): ("metal2", "#ff7f0e", 0.50, 6),
    (38, 0): ("metal3", "#8c564b", 0.45, 7),
    (42, 0): ("m1.label", "#1f77b4", 0.0, 9),
    (42, 28): ("m1.text", "#000000", 0.0, 9),
}
_FALLBACK = ["#7f7f7f", "#17becf", "#bcbd22", "#e377c2", "#c5b0d5"]


def render(gds: Path, out: Path, title: str = "") -> None:
    ly = db.Layout()
    ly.read(str(gds))
    dbu = ly.dbu
    top = ly.top_cell()
    bb = top.bbox()
    w_um, h_um = bb.width() * dbu, bb.height() * dbu

    fig_w = max(4.0, min(20.0, w_um / 6.0))
    fig_h = max(4.0, min(26.0, h_um / 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    seen = {}
    fb = 0
    layers = sorted(ly.layer_indexes(),
                    key=lambda li: GF180.get((ly.get_info(li).layer, ly.get_info(li).datatype),
                                             ("", "", 0, 50))[3])
    for li in layers:
        info = ly.get_info(li)
        key = (info.layer, info.datatype)
        if key in GF180:
            name, color, alpha, z = GF180[key]
        else:
            name, color, alpha, z = (f"{info.layer}/{info.datatype}", _FALLBACK[fb % len(_FALLBACK)], 0.4, 50)
            fb += 1
        if alpha <= 0.0:
            continue
        reg = db.Region(top.begin_shapes_rec(li))
        n = 0
        for poly in reg.each():
            pts = [(p.x * dbu, p.y * dbu) for p in poly.each_point_hull()]
            if len(pts) >= 3:
                ax.add_patch(MplPoly(pts, closed=True, facecolor=color, edgecolor=color,
                                     alpha=alpha, linewidth=0.3, zorder=z))
                n += 1
        if n and name not in seen:
            seen[name] = color

    ax.set_xlim(bb.left * dbu - 1, bb.right * dbu + 1)
    ax.set_ylim(bb.bottom * dbu - 1, bb.top * dbu + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("um"); ax.set_ylabel("um")
    ax.set_title((title or top.name) + f"  ({w_um:.1f} x {h_um:.1f} um)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) for c in seen.values()]
    ax.legend(handles, list(seen.keys()), loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"rendered {gds.name} -> {out}  ({w_um:.1f} x {h_um:.1f} um, {len(seen)} layers)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    render(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
