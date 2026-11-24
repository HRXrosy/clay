"""FOLDING intent: the migratable DEVICE-GEOMETRY primitive.

The placement primitives (abut / align / order / symmetric) pin WHERE a device
sits; they say nothing about its SHAPE. A m=16 transistor laid as one row of 16
fingers is a wide strip; the same device folded into 2 rows of 8 is a compact
square. That FOLD choice -- (n_rows, fingers_per_row) -- is a real design intent:
it is what makes an analog cell compact and routable, and it is MIGRATABLE (a
small integer pair, not a coordinate -- the same fold applied in another PDK gives
that PDK's cell shape).

This module extracts the fold intent by choosing (n_rows, nf) so the built device
cell best matches the reference cell FOOTPRINT (the degradation-start read), and
realizes it as a tiled grid of single-row :func:`device_cell.build_device` cells.
The OUTPUT is the discrete fold + the target-PDK footprint -- no reference
coordinate is stored, so the fold transfers across PDKs.

HONEST GUARD: if the reference target footprint is PHYSICALLY too small to hold the
device's channel (e.g. a mis-corresponded bbox -- the OTA's MM21 reference bbox is
353 um^2 but m=16 w=21 l=2 needs >670 um^2 of gate alone), the search cannot match
it; ``implausible_target`` flags this so the caller does not shrink a device below
its real size. The build_device single-row size is then the physically correct one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_XY = Tuple[float, float]

_ROW_GAP_UM = 1.0
_MIN_W_FINGER_UM = 0.42


@dataclass(frozen=True)
class FoldChoice:
    """A device's fold intent + the realized footprint.

    * ``n_rows``      -- stacked rows of single-row cells (multi-row fold).
    * ``nf``          -- fingers per row.
    * ``w_finger_um`` -- per-finger channel width (total channel = n_rows*nf*w_finger).
    * ``footprint``   -- (w, h) um of the tiled cell in the TARGET PDK.
    * ``implausible_target`` -- True if the reference target was physically too small
      to match (a likely mis-corresponded bbox); the footprint is then the device's
      real minimum size, NOT the (impossible) reference one.
    """

    n_rows: int
    nf: int
    w_finger_um: float
    footprint: _XY
    implausible_target: bool = False


def fold_to_match(
    tech,
    kind: str,
    total_w_um: float,
    l_um: float,
    target_w_um: float,
    target_h_um: float,
    *,
    max_rows: int = 5,
    max_nf: int = 16,
    row_gap_um: float = _ROW_GAP_UM,
    _cache: Optional[dict] = None,
) -> FoldChoice:
    """Choose (n_rows, nf) so the tiled cell footprint best matches the target.

    Preserves the TOTAL channel width ``total_w_um = n_rows * nf * w_finger`` (the
    device is electrically unchanged). Searches n_rows in [1, max_rows] and nf in
    [1, max_nf]; the cell is ``n_rows`` stacked copies of a single-row
    ``build_device(w_finger, l, nf)`` cell. Returns the :class:`FoldChoice` with the
    smallest |dw| + |dh| to the target.
    """
    from clay.layout.device_cell import build_device

    cache = _cache if _cache is not None else {}

    def unit(wf: float, nf: int) -> _XY:
        key = (kind, round(wf, 2), round(l_um, 2), nf)
        if key not in cache:
            g = build_device(tech, kind, W=wf * 1e-6, L=l_um * 1e-6, nf=nf)
            e = g.extent()
            cache[key] = ((e[2] - e[0]) / 1000.0, (e[3] - e[1]) / 1000.0)
        return cache[key]

    best: Optional[Tuple[float, FoldChoice]] = None
    for n_rows in range(1, max_rows + 1):
        per_row_w = total_w_um / n_rows
        for nf in range(1, max_nf + 1):
            wf = per_row_w / nf
            if wf < _MIN_W_FINGER_UM:
                continue
            uw, uh = unit(wf, nf)
            cw = uw
            ch = n_rows * uh + (n_rows - 1) * row_gap_um
            err = abs(cw - target_w_um) + abs(ch - target_h_um)
            if best is None or err < best[0]:
                best = (err, FoldChoice(n_rows, nf, wf, (cw, ch)))

    assert best is not None
    fc = best[1]
    # implausible if even the closest cell is much larger than the target area
    # (the target cannot physically hold the channel -> likely a mis-corresponded bbox).
    tgt_area = max(target_w_um * target_h_um, 1e-9)
    implausible = (fc.footprint[0] * fc.footprint[1]) > 1.5 * tgt_area
    return FoldChoice(fc.n_rows, fc.nf, fc.w_finger_um, fc.footprint, implausible)


def folded_sizes(intent, tech, *, max_rows: int = 5) -> Tuple[Dict[str, _XY], Dict[str, FoldChoice]]:
    """Per-device folded footprints for ``intent`` realized in ``tech``.

    For each MOS: ``total_w = W * M``; the target footprint is the reference cell
    bbox; :func:`fold_to_match` picks the fold. Passives keep their reference bbox
    (cap / resistor geometry migration is a separate concern). Returns
    ``(sizes, folds)`` where ``sizes`` is ``name -> (w, h)`` and ``folds`` is the
    chosen :class:`FoldChoice` per MOS (its ``implausible_target`` flags a device
    whose reference bbox could not be matched).
    """
    bb = {n: b for n, b in intent.device_bboxes.items() if b}
    sizes: Dict[str, _XY] = {}
    folds: Dict[str, FoldChoice] = {}
    cache: dict = {}
    for d in intent.devices:
        if d.name not in bb:
            continue
        rw = bb[d.name][2] - bb[d.name][0]
        rh = bb[d.name][3] - bb[d.name][1]
        fc = fold_to_match(
            tech, d.kind, d.W * d.M, d.L, rw, rh, max_rows=max_rows, _cache=cache
        )
        folds[d.name] = fc
        sizes[d.name] = fc.footprint
    for p in intent.passives:
        if p.name in bb:
            sizes[p.name] = (bb[p.name][2] - bb[p.name][0], bb[p.name][3] - bb[p.name][1])
    return sizes, folds


__all__ = ["FoldChoice", "fold_to_match", "folded_sizes"]
