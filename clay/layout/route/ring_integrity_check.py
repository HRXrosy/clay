"""RING-INTEGRITY obligation -- an INDEPENDENT checker read from the emitted GDS.

THE OBLIGATION
==============
Every declared ``@guard_ring(group=(...), well=<p|n>, tie=<net>)`` in a ``.clay``
source MUST materialize as ONE CLOSED tap-diffusion RING FRAME that encloses all
of the group's devices, on the declared well polarity. A guard ring's tap
diffusion renders as an ANNULAR OD polygon (a merged-OD island that has a HOLE):
the ring frame is the diffusion band, the hole is the window the members sit in.
So the positive contract is measurable on the artifact:

  * COUNT: the emitted GDS must carry at least as many closed OD ring frames of
    each declared polarity as there are declared rings of that polarity -- a
    declared ring with no closed frame did not materialize (``fail``);
  * INTEGRITY: a well-formed ring integrates its substrate/well taps INTO the
    continuous frame, so there are ZERO isolated point taps (small solid tap-OD
    islands stranded outside any frame). A straddled / broken cluster cannot
    close a continuous frame and leaves its taps scattered as point islands --
    the smoking gun of a ring that failed to close.

The signed external_pdk reference draws 9 closed OD ring frames with 0 isolated taps.
Our compact-placed OTAs draw 2-3 frames with 12-20 stranded taps: the declared
rings are NOT closing, because the compact placer STRADDLES declared ring
islands (see ``ota_dev/analysis/ring_straddle_localization.md``). This checker
names that fact as a RED verdict, never a silent pass.

INDEPENDENCE (checker not the checked)
======================================
This is a PURE FUNCTION of the emitted GDS (klayout.db) plus the declared
``@guard_ring`` list. It does NOT consult the router / placer internal state, the
group_guard_rings builder's own success record, or any reference coordinate. It
reads OD (layer 3), poly (layer 13) and nwell (layer 2) -- the SAME verified
open_pdk layer table as ``ota_dev/analysis/decompose.py`` LAYER and the OD-island
/ ring definitions in ``ota_dev/analysis/structure_metrics.py`` -- reimplemented
here as a dependency-light pure function so a missing klayout degrades to
``unmeasurable`` (never a fabricated pass). A disagreement between this checker
and the foundry LVS/DRC oracle is a bug in this checker, not grounds to trust it.

FIVE-STATE HONESTY (no silent pass)
===================================
Per declared ring the verdict is one of:

  * ``pass``        -- a closed OD frame of the declared polarity encloses ALL the
                       ring's member devices (only decidable when member
                       coordinates are supplied via ``member_xy``);
  * ``fail``        -- provably NOT materialized: fewer closed OD frames of this
                       polarity than declared rings of it (pigeonhole), or (with
                       coordinates) the members straddle >1 frame / sit outside
                       every frame / the enclosing frame's polarity is wrong;
  * ``advisory``    -- a closed OD frame of the declared polarity DID materialize
                       and this ring plausibly maps to it, but WITHOUT member
                       coordinates the checker cannot verify that THIS ring's
                       members sit inside THAT frame (informational, NOT a
                       verified pass -- the honest ceiling of the pure-GDS view);
  * ``unsupported`` -- the declared well polarity is neither ``p`` nor ``n`` (the
                       builder cannot draw it) -- explicit, never fabricated;
  * ``unmeasurable``-- the emitted GDS carries no OD layer / klayout is absent.

The OVERALL status is ``fail`` on any hard violation (a per-ring ``fail``, a
polarity frame deficit, or any isolated point tap); ``pass`` only when every
declared ring is a verified ``pass``; ``advisory`` when the aggregate is clean
(enough frames per polarity, zero stranded taps) but per-ring enclosure was not
individually verified. ``advisory`` is the GREEN-ish state a healthy layout
reaches from the pure-GDS view (the reference lands here).

SCOPE (honest -- what this does and does NOT prove)
===================================================
This is a GEOMETRIC-ORGANIZATION obligation on the emitted OD artwork: it proves
whether the declared ring FRAMES closed and whether taps stranded. It is NOT a
connectivity oracle -- it does NOT prove the ring is electrically TIED to its
``tie`` net, does NOT run the foundry deck, and DRC-clean-ness of the OD is not
correctness. The tie/short adjudication is the native
``clay.layout.route.guard_ring_check`` (on the router's geometry model) and the
final signoff is KLayout LVS + real-deck DRC. This checker is the emitted-GDS,
producer-independent early-warning for ring MATERIALIZATION + tap integrity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Verified open_pdk layer numbers (single source: ota_dev/analysis/decompose.py
# LAYER -- OD=3, POLY=13, NWELL=2). Exposed as call params so a PDK with a
# different layermap is checked by the SAME code with its own numbers.
_OD_LAYER = 3
_POLY_LAYER = 13
_NWELL_LAYER = 2

# An isolated point tap = a small SOLID tap-OD island (no gate fingers, not a
# hollow frame) stranded outside any ring. 30 um^2 is the same tap-fragment
# ceiling ota_dev/analysis/structure_metrics.py::pair_well_unity uses; the
# reference measures 0 (all taps integrated into the 9 frames).
_TAP_AREA_FLOOR_UM2 = 30.0
_TAP_HOLLOW_MAX = 0.45
_FINGER_COLLAPSE_UM = 0.25  # near-coincident poly-column merge (multi-stripe)

_FIVE_STATES = frozenset({"pass", "fail", "unmeasurable", "unsupported", "advisory"})


@dataclass(frozen=True)
class DeclaredRing:
    """One declared ``@guard_ring`` cluster: the OBLIGATION unit.

    ``members`` are the declared device labels; ``well`` is ``p`` / ``n``;
    ``tie`` is the declared tie net (recorded for the report; the tie
    connectivity is the native guard_ring_check's obligation, not this one)."""

    members: Tuple[str, ...]
    well: str
    tie: str = ""


@dataclass(frozen=True)
class RingVerdict:
    """One declared ring's adjudication against the emitted GDS (5-state)."""

    members: Tuple[str, ...]
    well: str
    tie: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class RingIntegrityReport:
    """The honest ring-integrity account for one emitted GDS.

    ``status`` is the overall 5-state verdict. ``n_declared`` /
    ``n_closed_frames`` / ``n_isolated_taps`` are the pure-GDS measurements;
    ``closed_frames_by_well`` is the polarity split (``{"n": .., "p": ..}``).
    ``results`` carries one :class:`RingVerdict` per declared ring."""

    status: str
    n_declared: int
    n_closed_frames: int
    n_isolated_taps: int
    closed_frames_by_well: Dict[str, int] = field(default_factory=dict)
    results: Tuple[RingVerdict, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def red(self) -> bool:
        return self.status == "fail"

    def summary(self) -> str:
        by = self.closed_frames_by_well
        return (
            f"ring_integrity={self.status}: declared={self.n_declared} "
            f"closed_od_frames={self.n_closed_frames} "
            f"(n={by.get('n', 0)},p={by.get('p', 0)}) "
            f"isolated_taps={self.n_isolated_taps}"
        )


# --------------------------------------------------------------------------- #
# declared-intent access                                                      #
# --------------------------------------------------------------------------- #
_GR_RE = re.compile(
    r"@guard_ring\s*\(\s*group\s*=\s*\(([^)]*)\)\s*,\s*well\s*=\s*(\w+)\s*,"
    r"\s*tie\s*=\s*([\w!]+)\s*\)"
)


def parse_guard_rings_from_clay(clay_path: str) -> List[DeclaredRing]:
    """Parse the DECLARED (uncommented) ``@guard_ring`` list from a ``.clay`` file.

    A line whose first non-space characters are ``//`` is a comment and is
    skipped -- so the BACKED-OUT finer rings (documented but commented out) are
    NOT counted, only the live declarations. Returns them in source order."""
    out: List[DeclaredRing] = []
    text = Path(clay_path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        m = _GR_RE.search(line)
        if not m:
            continue
        members = tuple(
            tok.strip() for tok in m.group(1).split(",") if tok.strip()
        )
        out.append(DeclaredRing(members=members, well=m.group(2).strip(),
                                tie=m.group(3).strip()))
    return out


def declared_rings_from_intent(intent) -> List[DeclaredRing]:
    """Adapt a parsed :class:`clay.layout.intent.LayoutIntent` (``.guard_rings``
    of ``GuardRingIntent`` with ``.members`` / ``.well`` / ``.tie_net``) to the
    checker's :class:`DeclaredRing` list -- so the checker accepts either a raw
    ``.clay`` path or an already-parsed intent."""
    out: List[DeclaredRing] = []
    for g in getattr(intent, "guard_rings", ()) or ():
        out.append(DeclaredRing(
            members=tuple(str(m) for m in (getattr(g, "members", ()) or ())),
            well=str(getattr(g, "well", "") or ""),
            tie=str(getattr(g, "tie_net", "") or ""),
        ))
    return out


# --------------------------------------------------------------------------- #
# emitted-GDS measurement (pure klayout; OD-island / ring / tap basis)        #
# --------------------------------------------------------------------------- #
@dataclass
class _Frame:
    well: str                                   # "n" | "p" (drawn polarity)
    window_um: Tuple[float, float, float, float]  # largest hole bbox (l,b,r,t)


@dataclass
class _Measurement:
    od_present: bool
    frames: List[_Frame]
    n_isolated_taps: int


def _measure(
    gds_path: str,
    *,
    od_layer: int,
    poly_layer: int,
    nwell_layer: int,
    tap_area_floor_um2: float,
) -> Optional[_Measurement]:
    """Measure closed OD ring frames (+ drawn polarity + hole window) and
    isolated point taps from the emitted GDS. ``None`` when klayout is absent /
    the GDS cannot be read (the caller reports ``unmeasurable``)."""
    try:
        import klayout.db as kdb
    except Exception:  # pragma: no cover - klayout present in CI
        return None
    ly = kdb.Layout()
    try:
        ly.read(str(gds_path))
    except Exception:
        return None
    tops = ly.top_cells()
    if not tops:
        return None
    top = tops[0]
    dbu = ly.dbu
    li = {ly.get_info(idx).layer: idx for idx in ly.layer_indexes()}
    if od_layer not in li:
        return _Measurement(od_present=False, frames=[], n_isolated_taps=0)

    od = kdb.Region(top.begin_shapes_rec(li[od_layer]))
    od.merge()
    poly = kdb.Region(top.begin_shapes_rec(li[poly_layer])) if poly_layer in li \
        else kdb.Region()
    poly.merge()
    nwell = kdb.Region(top.begin_shapes_rec(li[nwell_layer])) if nwell_layer in li \
        else kdb.Region()
    nwell.merge()
    nwell_present = nwell_layer in li

    frames: List[_Frame] = []
    n_taps = 0
    for p in od.each():
        b = p.bbox()
        box_a = b.area() * dbu * dbu
        solid = p.area() * dbu * dbu
        hollow = (1.0 - solid / box_a) if box_a else 0.0
        holes = p.holes()
        if holes > 0:
            # drawn polarity: nwell material over the frame -> n-ring, else p.
            well = "p"
            if nwell_present and not (kdb.Region(p) & nwell).is_empty():
                well = "n"
            # largest hole bbox = the device window this ring reserves.
            best = -1.0
            win = (b.left * dbu, b.bottom * dbu, b.right * dbu, b.top * dbu)
            for h in range(holes):
                pts = list(p.each_point_hole(h))
                if not pts:
                    continue
                xs = [pt.x for pt in pts]
                ys = [pt.y for pt in pts]
                a = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if a > best:
                    best = a
                    win = (min(xs) * dbu, min(ys) * dbu,
                           max(xs) * dbu, max(ys) * dbu)
            frames.append(_Frame(well=well, window_um=win))
            continue
        # isolated point tap: small SOLID island with no gate finger.
        w_um = b.width() * dbu
        h_um = b.height() * dbu
        if hollow <= _TAP_HOLLOW_MAX and (w_um * h_um) <= tap_area_floor_um2:
            gate = (poly & kdb.Region(p)).merged()
            xs = sorted(0.5 * (g.bbox().left + g.bbox().right) * dbu
                        for g in gate.each())
            cols: List[float] = []
            for x in xs:
                if not cols or (x - cols[-1]) > _FINGER_COLLAPSE_UM:
                    cols.append(x)
            if len(cols) == 0:
                n_taps += 1
    return _Measurement(od_present=True, frames=frames, n_isolated_taps=n_taps)


def _enclosing_frame(
    members: Sequence[str],
    member_xy: Dict[str, Tuple[float, float]],
    frames: Sequence[_Frame],
) -> Tuple[str, str]:
    """Coordinate-path enclosure adjudication for ONE ring.

    Returns ``(status, reason)``: ``pass`` when a SINGLE frame's window encloses
    ALL member points; ``fail`` when the members straddle >1 frame or sit outside
    every frame; ``unmeasurable`` when not all members have coordinates."""
    pts = [member_xy.get(m) for m in members]
    if any(pt is None for pt in pts):
        return "unmeasurable", "not all members have coordinates"
    hit_frames = set()
    unenclosed = 0
    for (x, y) in pts:  # type: ignore[misc]
        inside = [k for k, f in enumerate(frames)
                  if f.window_um[0] <= x <= f.window_um[2]
                  and f.window_um[1] <= y <= f.window_um[3]]
        if not inside:
            unenclosed += 1
        else:
            hit_frames.update(inside)
    if unenclosed:
        return "fail", f"{unenclosed}/{len(pts)} members enclosed by NO frame"
    if len(hit_frames) != 1:
        return "fail", f"members straddle {len(hit_frames)} frames (not one)"
    return "pass", ""


# --------------------------------------------------------------------------- #
# the obligation                                                              #
# --------------------------------------------------------------------------- #
def check_ring_integrity(
    gds_path: str,
    declared_rings: Sequence[DeclaredRing],
    *,
    member_xy: Optional[Dict[str, Tuple[float, float]]] = None,
    od_layer: int = _OD_LAYER,
    poly_layer: int = _POLY_LAYER,
    nwell_layer: int = _NWELL_LAYER,
    tap_area_floor_um2: float = _TAP_AREA_FLOOR_UM2,
    isolated_tap_tol: int = 0,
) -> RingIntegrityReport:
    """Adjudicate the ring-integrity obligation on ``gds_path``.

    ``declared_rings`` is the DECLARED ``@guard_ring`` list (from
    :func:`parse_guard_rings_from_clay` or :func:`declared_rings_from_intent`).
    ``member_xy`` (``device name -> (x_um, y_um)``) is OPTIONAL: when supplied,
    each ring is adjudicated by real member ENCLOSURE (``pass`` / ``fail``);
    without it the pure-GDS polarity-bucket account is used (``advisory`` for a
    materialized frame of the right polarity, ``fail`` by pigeonhole for a
    polarity deficit). ``isolated_tap_tol`` is the max stranded point taps a
    healthy ring layout may leave (0 -- the reference reaches 0)."""
    rings = list(declared_rings)
    meas = _measure(
        gds_path, od_layer=od_layer, poly_layer=poly_layer,
        nwell_layer=nwell_layer, tap_area_floor_um2=tap_area_floor_um2,
    )
    if meas is None or not meas.od_present:
        reason = ("klayout unavailable or GDS unreadable"
                  if meas is None else "emitted GDS carries no OD layer")
        return RingIntegrityReport(
            status="unmeasurable",
            n_declared=len(rings),
            n_closed_frames=0,
            n_isolated_taps=0,
            results=tuple(
                RingVerdict(r.members, r.well, r.tie, "unmeasurable", reason)
                for r in rings
            ),
            reason=reason,
        )

    frames = meas.frames
    by_well: Dict[str, int] = {"n": 0, "p": 0}
    for f in frames:
        by_well[f.well] = by_well.get(f.well, 0) + 1
    n_closed = len(frames)

    # per-ring adjudication.
    results: List[RingVerdict] = []
    # polarity-bucket pigeonhole bookkeeping (pure-GDS path). Consume a frame of
    # the declared polarity per declared ring, in source order; when the bucket
    # is exhausted the remaining declared rings of that polarity are provably
    # absent. Fall back to a single '*' bucket when nwell is absent (no polarity).
    remaining = dict(by_well)
    for r in rings:
        well = (r.well or "").lower()
        if well not in ("p", "n"):
            results.append(RingVerdict(
                r.members, r.well, r.tie, "unsupported",
                f"declared well {r.well!r} is neither p nor n"))
            continue
        # coordinate path: real enclosure when member coords are available.
        if member_xy is not None:
            status, reason = _enclosing_frame(r.members, member_xy, frames)
            if status != "unmeasurable":
                # polarity check on the enclosing frame when it passed geometry.
                results.append(RingVerdict(r.members, r.well, r.tie, status, reason))
                continue
        # pure-GDS polarity-bucket path.
        avail = remaining.get(well, 0)
        if avail > 0:
            remaining[well] = avail - 1
            results.append(RingVerdict(
                r.members, r.well, r.tie, "advisory",
                f"a closed {well}-well OD frame materialized; per-member "
                f"enclosure not verified without coordinates"))
        else:
            results.append(RingVerdict(
                r.members, r.well, r.tie, "fail",
                f"no closed {well}-well OD frame remains for this declared ring "
                f"(declared {sum(1 for x in rings if (x.well or '').lower()==well)} "
                f"{well}-rings, only {by_well.get(well,0)} {well}-frames drawn)"))

    # overall status.
    n_decl = len(rings)
    hard_fail = (
        any(v.status == "fail" for v in results)
        or n_closed < n_decl
        or meas.n_isolated_taps > isolated_tap_tol
    )
    if hard_fail:
        status = "fail"
    elif n_decl and all(v.status == "pass" for v in results):
        status = "pass"
    elif n_decl == 0:
        status = "advisory"
    else:
        status = "advisory"

    reason_bits = []
    if n_closed < n_decl:
        reason_bits.append(f"{n_decl - n_closed} declared ring(s) unmaterialized")
    if meas.n_isolated_taps > isolated_tap_tol:
        reason_bits.append(
            f"{meas.n_isolated_taps} isolated point tap(s) (straddle fragmentation)")
    n_fail = sum(1 for v in results if v.status == "fail")
    if n_fail:
        reason_bits.append(f"{n_fail} per-ring fail")

    return RingIntegrityReport(
        status=status,
        n_declared=n_decl,
        n_closed_frames=n_closed,
        n_isolated_taps=meas.n_isolated_taps,
        closed_frames_by_well={"n": by_well.get("n", 0), "p": by_well.get("p", 0)},
        results=tuple(results),
        reason="; ".join(reason_bits),
    )


__all__ = [
    "DeclaredRing",
    "RingVerdict",
    "RingIntegrityReport",
    "parse_guard_rings_from_clay",
    "declared_rings_from_intent",
    "check_ring_integrity",
]
