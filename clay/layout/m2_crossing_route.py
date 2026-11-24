"""R163 (BL-163D) -- real M2-layer crossing routing for Clay (gf180mcu).

Why this lane exists
====================
The R162 Clay-vs-TED benchmark found Clay's router LAGS TED on routing
maturity: Clay routes on a single M1 grid (``a_star_router`` -- current_mirror /
diff_pair route on M1 only), so two nets that must cross each other CONFLICT on
M1 (they would short, or the router has to detour the whole channel). TED has a
maze / pattern multi-layer router (M1-M5 + via insertion + congestion-aware
ripup) so crossing nets coexist on different layers.

This module gives Clay the one structural capability that closes the largest
slice of that gap: a crossing net can hop onto **M2 (gf180mcu Metal2, 36/0)** over
an M1 net, jump the crossing with a real **via12 (Via1, 35/0)** stack at each
endpoint, and come back down to M1 -- so two crossing nets are DRC-legal on
separate layers instead of conflicting on one.

The crossing-conflict problem, made concrete
---------------------------------------------
Take an east-west M1 net A spanning a channel, and a north-south net B that must
get from below the channel to above it, crossing A. On ONE layer (both on M1)
the two rectangles overlap -> a metal short (one electrical net where there must
be two). The single-layer router cannot place both; it either detours B around
the whole of A (wirelength + congestion blow-up) or reports an unroutable
crossing.

The M2-offload fix (this module)
--------------------------------
Net B keeps its M1 stubs OUTSIDE A's footprint (one below A, one above), drops a
via12 stack on each M1 stub, and routes the SPAN that crosses A on M2. Because
M2 (36/0) and M1 (34/0) are different GDS layers, the M2 span over A's M1 is NOT
a short -- the layers do not interact except through a via, and there is no via
in the crossing region. Net A stays a single uninterrupted M1 wire. Two nets,
one crossing, zero conflict.

Real gf180mcu DRC rules this geometry must satisfy (verbatim from the deck:
``AnalogGym/PDK/gf180mcu_pdk/.../rules/klayout/drc/gf180mcu.drc``)::

    M1.1   min Metal1 width   0.23um (230 nm)            (deck line 2394)
    M1.2a  min Metal1 spacing 0.23um (230 nm)            (deck line 2400)
    M2.1   min Metal2 width   0.28um (280 nm)            (deck line 2422)
    M2.2a  min Metal2 spacing 0.28um (280 nm)            (deck line 2428)
    V1.1   Via1 size EXACTLY  0.26um (260 nm) sq          (deck line 2534)
    V1.2a  min Via1 spacing   0.26um (260 nm)            (deck line 2540)
    V1.3a  Metal1 must fully enclose Via1                 (deck line 2561)
    V1.3c  Metal1 EOL overlap of Via1  0.06um (60 nm)     (deck line 2571)
    V1.4a  Metal2 must enclose Via1 by >= 0.01um          (deck line 2598)
    V1.4b  Metal2 EOL overlap of Via1  0.06um (60 nm)     (deck line 2606)

So the via12 LANDING PAD on BOTH M1 and M2 must be >= 260 + 2*60 = 380 nm and
the via itself EXACTLY 260 nm. We use a 400 nm pad (70 nm enclosure each side,
> 60) which also clears the M1.3 min-area (0.1444um^2 = 380x380); 400x400 =
0.16um^2. The M2 span is 280 nm wide (>= M2.1) and via spacing / wire spacing
are kept >= the M2.2a 280 nm and V1.2a 260 nm minima by construction.

Honesty contract (CRITICAL)
---------------------------
* The geometry builders here are PURE (no fabricated DRC verdicts). The
  authoritative DRC verdict is the real ``gf180mcu.drc`` deck's own ``.lyrdb``
  violation count, read verbatim by the committed R139 harness.
* The congestion numbers are computed by the committed
  ``clay.layout.congestion.estimate_congestion`` on the actual emitted wires --
  the single-layer crossing count and the M2-offloaded crossing count are both
  read from that estimator, never asserted by hand.
* No claim that the geometry is DRC-clean is made unless the real deck ran and
  reported its count; absent the binary the result is honest PENDING.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backend import (
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)
from .congestion import estimate_congestion
from .gf180_layermap import GF180_VERIFIED_SOURCES

# Repo root: clay/layout/m2_crossing_route_v163.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "demo_out" / "m2_crossing_route_v163"
DEFAULT_KLAYOUT_BIN = Path(r"C:\Program Files (x86)\KLayout\klayout_app.exe")


# --- gf180mcu GDS layer keys (verified in gf180_layermap.GF180_VERIFIED_SOURCES)
def _key(name: str) -> Tuple[int, int]:
    layer, dt, _line, _txt = GF180_VERIFIED_SOURCES[name]
    return (layer, dt)


M1_KEY = _key("Metal1")  # (34, 0)
VIA1_KEY = _key("Via1")  # (35, 0)
M2_KEY = _key("Metal2")  # (36, 0)


# --- gf180mcu rule constants (nm; verbatim from gf180mcu.drc, cited above) ----
M1_MIN_WIDTH_NM = 230  # M1.1
M1_MIN_SPACE_NM = 230  # M1.2a
M2_MIN_WIDTH_NM = 280  # M2.1
M2_MIN_SPACE_NM = 280  # M2.2a
VIA1_SIZE_NM = 260  # V1.1 (EXACT min/max via1 size)
VIA1_MIN_SPACE_NM = 260  # V1.2a
VIA1_M1_ENCLOSURE_NM = 60  # V1.3c (EOL overlap)
VIA1_M2_ENCLOSURE_NM = 60  # V1.4b (EOL overlap)

# Landing pad side: via 260 + 2*70 = 400 (70 nm > the 60 nm V1.3c/V1.4b minimum,
# and 400x400 = 0.16um^2 > the M1.3 0.1444um^2 min-metal-area rule).
VIA12_PAD_NM = 400


# --- pure geometry helpers ----------------------------------------------------


def _rect(key: Tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> LayerPolygon:
    """An axis-aligned rectangle LayerPolygon on ``key`` (coords auto-ordered)."""
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    return LayerPolygon(
        layer=key[0],
        datatype=key[1],
        polygon=Polygon.rectangle(int(lo_x), int(lo_y), int(hi_x), int(hi_y)),
    )


def _bbox(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _hwire(key: Tuple[int, int], x0: int, x1: int, y_center: int, width_nm: int) -> LayerPolygon:
    """Horizontal wire centred on ``y_center`` from x0 to x1."""
    half = width_nm // 2
    return _rect(key, x0, y_center - half, x1, y_center + half)


def _vwire(key: Tuple[int, int], y0: int, y1: int, x_center: int, width_nm: int) -> LayerPolygon:
    """Vertical wire centred on ``x_center`` from y0 to y1."""
    half = width_nm // 2
    return _rect(key, x_center - half, y0, x_center + half, y1)


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Strict positive-area rectangle overlap (edge-touch does NOT count)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# --- via12 stack (M1 pad + Via1 + M2 pad), DRC-legal by construction ----------


def via12_stack(center: Point) -> List[LayerPolygon]:
    """A gf180mcu-legal M1+Via1+M2 stack centred at ``center``.

    Returns three polygons: an M1 landing pad (34/0), a Via1 (35/0) of EXACTLY
    260 nm (V1.1), and an M2 landing pad (36/0). Both pads are ``VIA12_PAD_NM``
    (400 nm) square, giving 70 nm enclosure each side -- comfortably past the
    60 nm V1.3c / V1.4b end-of-line overlap minima and the V1.3a / V1.4a full
    enclosure requirement.
    """
    half_via = VIA1_SIZE_NM // 2
    half_pad = VIA12_PAD_NM // 2
    cx, cy = center.x, center.y
    return [
        _rect(M1_KEY, cx - half_pad, cy - half_pad, cx + half_pad, cy + half_pad),
        _rect(VIA1_KEY, cx - half_via, cy - half_via, cx + half_via, cy + half_via),
        _rect(M2_KEY, cx - half_pad, cy - half_pad, cx + half_pad, cy + half_pad),
    ]


# --- routed-net records -------------------------------------------------------


@dataclass(frozen=True)
class RoutedNet:
    """A net's emitted geometry + which layers it landed on, for provenance."""

    name: str
    polygons: List[LayerPolygon] = field(default_factory=list)
    on_m2: bool = False
    via12_count: int = 0


# --- the crossing demo: single-layer (conflict) vs M2-offload (legal) ---------


@dataclass(frozen=True)
class CrossingDemo:
    """A two-net crossing case in both renderings.

    Attributes:
        single_layer: every wire on M1 -- net A and net B's crossing span SHORT.
        m2_offload: net A on M1, net B hops to M2 over the crossing with via12.
        net_a / net_b_single / net_b_m2: the per-net geometry for inspection.
        crossing_x / channel_y: where the two nets meet.
    """

    single_layer: List[LayerPolygon]
    m2_offload: List[LayerPolygon]
    net_a: RoutedNet
    net_b_single: RoutedNet
    net_b_m2: RoutedNet
    crossing_x: int
    channel_y: int


def route_crossing_on_m2(
    *,
    net_a_x0: int = 0,
    net_a_x1: int = 6000,
    channel_y: int = 3000,
    crossing_x: int = 3000,
    net_b_y0: int = 0,
    net_b_y1: int = 6000,
) -> CrossingDemo:
    """Build the crossing demo: net A horizontal (M1), net B vertical crossing it.

    Net A is one east-west M1 wire on the channel centre line ``channel_y``.
    Net B must get from ``net_b_y0`` (below A) to ``net_b_y1`` (above A) at
    ``crossing_x``, crossing A.

    Single-layer rendering: net B is ALSO on M1, so its vertical wire overlaps
    net A's horizontal wire -> a same-layer crossing (the estimator counts it,
    and physically it is a short).

    M2-offload rendering: net B's two M1 stubs stay clear of A (one ends below,
    one starts above, each leaving an M1.2a-legal gap to A), each carries a
    via12 stack, and the SPAN crossing A is an M2 wire. The M2 span over A's M1
    is two different GDS layers -> no crossing on a shared layer.
    """
    # ---- net A: one horizontal M1 wire across the channel --------------------
    net_a_poly = _hwire(M1_KEY, net_a_x0, net_a_x1, channel_y, M1_MIN_WIDTH_NM)
    net_a = RoutedNet(name="A", polygons=[net_a_poly], on_m2=False, via12_count=0)

    # ---- net B, single-layer (all M1): vertical wire straight through A ------
    nb_single_poly = _vwire(M1_KEY, net_b_y0, net_b_y1, crossing_x, M1_MIN_WIDTH_NM)
    net_b_single = RoutedNet(name="B", polygons=[nb_single_poly], on_m2=False, via12_count=0)

    # ---- net B, M2-offload ---------------------------------------------------
    # Keep the M1 stubs clear of net A by an M1.2a-legal gap on each side. A's
    # half-width is M1_MIN_WIDTH_NM//2; leave M1_MIN_SPACE_NM clearance + the
    # via pad half so the M1 pad never crowds A.
    a_half = M1_MIN_WIDTH_NM // 2
    pad_half = VIA12_PAD_NM // 2
    # Lower via12 sits below A: its pad TOP must clear A's bottom by >= M1.2a.
    lower_via_cy = channel_y - a_half - M1_MIN_SPACE_NM - pad_half - 100
    upper_via_cy = channel_y + a_half + M1_MIN_SPACE_NM + pad_half + 100

    lower_stack = via12_stack(Point(crossing_x, lower_via_cy))
    upper_stack = via12_stack(Point(crossing_x, upper_via_cy))

    # M1 stubs from the net ends up/down to the via pad edges (net B's M1 part).
    m1_stub_lo = _vwire(M1_KEY, net_b_y0, lower_via_cy, crossing_x, M1_MIN_WIDTH_NM)
    m1_stub_hi = _vwire(M1_KEY, upper_via_cy, net_b_y1, crossing_x, M1_MIN_WIDTH_NM)

    # The M2 span jumping OVER net A from the lower pad to the upper pad. M2.1
    # width >= 280; this is the wire that crosses A on the offload layer.
    m2_span = _vwire(M2_KEY, lower_via_cy, upper_via_cy, crossing_x, M2_MIN_WIDTH_NM)

    nb_m2_polys = [m1_stub_lo, m1_stub_hi, m2_span] + lower_stack + upper_stack
    net_b_m2 = RoutedNet(name="B", polygons=nb_m2_polys, on_m2=True, via12_count=2)

    single_layer = [net_a_poly, nb_single_poly]
    m2_offload = [net_a_poly] + nb_m2_polys

    return CrossingDemo(
        single_layer=single_layer,
        m2_offload=m2_offload,
        net_a=net_a,
        net_b_single=net_b_single,
        net_b_m2=net_b_m2,
        crossing_x=crossing_x,
        channel_y=channel_y,
    )


# --- congestion accounting (delegates to the committed estimator) -------------


@dataclass(frozen=True)
class CongestionDelta:
    """Crossing count before (single-layer) vs after (M2-offload).

    A "crossing" here is the committed ``estimate_congestion`` count: a pair of
    cross-axis wires that overlap ON THE SAME GDS (layer, datatype). Moving net
    B's crossing span to M2 removes the shared-layer overlap with net A, so the
    same-layer crossing count drops to 0.
    """

    crossings_single_layer: int
    crossings_m2_offload: int

    @property
    def reduced(self) -> bool:
        return self.crossings_m2_offload < self.crossings_single_layer

    @property
    def reduction(self) -> int:
        return self.crossings_single_layer - self.crossings_m2_offload


def crossing_congestion(demo: CrossingDemo, channel_pitch_nm: int = 1000) -> CongestionDelta:
    """Compute the same-layer crossing count before vs after the M2 offload."""
    before = estimate_congestion(demo.single_layer, channel_pitch_nm=channel_pitch_nm)
    after = estimate_congestion(demo.m2_offload, channel_pitch_nm=channel_pitch_nm)
    return CongestionDelta(
        crossings_single_layer=before.crossings,
        crossings_m2_offload=after.crossings,
    )


# --- a pure geometric sanity check (NOT a DRC verdict) ------------------------


@dataclass(frozen=True)
class GeometryAudit:
    """Pure structural checks on the M2-offload geometry (geometry, not a deck).

    These verify the geometry was BUILT to the rule minima; the authoritative
    DRC verdict still comes from the real deck. ``via_size_ok`` etc. are
    necessary-but-not-sufficient pre-checks so a malformed build is caught
    without a binary.
    """

    via_size_ok: bool
    m1_pad_encloses_via: bool
    m2_pad_encloses_via: bool
    m2_span_width_ok: bool
    m1_stub_clears_net_a: bool
    no_shared_layer_overlap: bool

    @property
    def all_ok(self) -> bool:
        return all(
            (
                self.via_size_ok,
                self.m1_pad_encloses_via,
                self.m2_pad_encloses_via,
                self.m2_span_width_ok,
                self.m1_stub_clears_net_a,
                self.no_shared_layer_overlap,
            )
        )


def audit_geometry(demo: CrossingDemo) -> GeometryAudit:
    """Run pure structural pre-checks against the gf180mcu rule minima."""
    polys = demo.m2_offload
    vias = [p for p in polys if (p.layer, p.datatype) == VIA1_KEY]
    m1s = [p for p in polys if (p.layer, p.datatype) == M1_KEY]
    m2s = [p for p in polys if (p.layer, p.datatype) == M2_KEY]

    # V1.1: every via is EXACTLY 260 nm square.
    via_size_ok = bool(vias) and all(
        (b[2] - b[0]) == VIA1_SIZE_NM and (b[3] - b[1]) == VIA1_SIZE_NM
        for b in (_bbox(v) for v in vias)
    )

    def _encl(pads: List[LayerPolygon], min_encl: int) -> bool:
        ok = True
        for v in vias:
            vb = _bbox(v)
            host = None
            for p in pads:
                pb = _bbox(p)
                if pb[0] <= vb[0] and pb[1] <= vb[1] and pb[2] >= vb[2] and pb[3] >= vb[3]:
                    host = pb
                    break
            if host is None:
                ok = False
                continue
            encl = min(vb[0] - host[0], vb[1] - host[1], host[2] - vb[2], host[3] - vb[3])
            if encl < min_encl:
                ok = False
        return ok

    m1_pad_encloses_via = _encl(m1s, VIA1_M1_ENCLOSURE_NM)
    m2_pad_encloses_via = _encl(m2s, VIA1_M2_ENCLOSURE_NM)

    # M2.1: the M2 span is >= 280 nm wide (its narrow dimension).
    m2_span_width_ok = all(
        min(b[2] - b[0], b[3] - b[1]) >= M2_MIN_WIDTH_NM for b in (_bbox(m) for m in m2s)
    )

    # M1.2a: net B's M1 stubs/pads leave >= 230 nm clearance from net A on M1.
    a_bb = _bbox(demo.net_a.polygons[0])
    m1_stub_clears_net_a = True
    for p in m1s:
        pb = _bbox(p)
        if pb == a_bb:
            continue  # net A itself
        if _rects_overlap(pb, a_bb):
            m1_stub_clears_net_a = False
            break
        # vertical gap (these are stacked along y at the same x band)
        gap = a_bb[1] - pb[3] if pb[3] <= a_bb[1] else pb[1] - a_bb[3]
        if 0 <= gap < M1_MIN_SPACE_NM:
            m1_stub_clears_net_a = False
            break

    # The offload invariant: no two wires overlap on the SAME (layer, datatype).
    no_shared_layer_overlap = crossing_congestion(demo).crossings_m2_offload == 0

    return GeometryAudit(
        via_size_ok=via_size_ok,
        m1_pad_encloses_via=m1_pad_encloses_via,
        m2_pad_encloses_via=m2_pad_encloses_via,
        m2_span_width_ok=m2_span_width_ok,
        m1_stub_clears_net_a=m1_stub_clears_net_a,
        no_shared_layer_overlap=no_shared_layer_overlap,
    )


# --- library build (for the real-deck DRC demo) -------------------------------


def build_m2_crossing_library(
    demo: Optional[CrossingDemo] = None,
    *,
    cell_name: str = "m2_crossing_v163",
) -> Tuple[ParameterizedLibrary, str]:
    """Wrap the M2-offload geometry in a single-top-cell library for GDS emit."""
    if demo is None:
        demo = route_crossing_on_m2()
    lib = ParameterizedLibrary(
        name="clay_m2_crossing_v163",
        cells=[NamedCell(name=cell_name, polygons=list(demo.m2_offload))],
    )
    return lib, cell_name


# --- availability predicates --------------------------------------------------


def klayout_binary_available(klayout_bin: Optional[Path] = None) -> bool:
    """True iff the KLayout binary exists at the expected path."""
    try:
        from tools.klayout_drc_signoff_v139.run import (
            klayout_binary_available as _k,
        )

        return bool(_k(klayout_bin))
    except Exception:  # noqa: BLE001
        k = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
        return k.exists()


def drc_deck_available() -> bool:
    """True iff the real ``gf180mcu.drc`` runset exists."""
    try:
        from tools.klayout_drc_signoff_v139.run import DEFAULT_DECK

        return DEFAULT_DECK.exists()
    except Exception:  # noqa: BLE001
        return False


def rust_gds_available() -> bool:
    """True iff the committed Rust binary-GDS emitter is callable."""
    try:
        from clay.layout.binary_gds_subprocess import cargo_binary_available

        return bool(cargo_binary_available())
    except Exception:  # noqa: BLE001
        return False


def real_drc_available(*, klayout_bin: Optional[Path] = None) -> bool:
    """True iff a REAL gf180mcu DRC signoff is possible end-to-end."""
    return (
        klayout_binary_available(klayout_bin)
        and drc_deck_available()
        and rust_gds_available()
    )


# --- end-to-end DRC demo on the M2-offload GDS --------------------------------


@dataclass
class M2CrossingDrcResult:
    """DRC verdict on the M2-offload crossing GDS (read VERBATIM from the deck).

    Plus the congestion delta (single-layer vs M2-offloaded crossings) and a
    list of the BEOL rules that fired (so a non-zero residual is honest).
    """

    ran_real_deck: bool = False
    total_violations: Optional[int] = None
    by_rule: Dict[str, int] = field(default_factory=dict)
    beol_violations: int = 0
    gds_path: Optional[str] = None
    gds_bytes: int = 0
    pending_reason: str = ""
    crossings_single_layer: int = 0
    crossings_m2_offload: int = 0
    geometry_audit_ok: bool = False

    # The BEOL (metal / via) rule prefixes the M2+via region can legitimately
    # produce; used to report the M2+via-region residual specifically.
    _BEOL_PREFIXES = ("M1.", "M2.", "M3.", "V1.", "V2.", "MET", "VIA")

    @property
    def is_clean(self) -> bool:
        return bool(self.ran_real_deck and self.total_violations == 0)

    @property
    def congestion_reduced(self) -> bool:
        return self.crossings_m2_offload < self.crossings_single_layer

    @property
    def verdict(self) -> str:
        if not self.ran_real_deck:
            return "PENDING" + (f" ({self.pending_reason})" if self.pending_reason else "")
        if self.total_violations == 0:
            return "DRC-CLEAN (0 violations on the M2+via12 crossing geometry)"
        head = ", ".join(f"{r}x{c}" for r, c in sorted(self.by_rule.items())[:8])
        return (
            f"DRC-VIOLATIONS ({self.total_violations} across {len(self.by_rule)} rules) "
            f"[{head}]"
        )

    def summary(self) -> str:
        cong = (
            f"crossings: single-layer={self.crossings_single_layer} -> "
            f"M2-offload={self.crossings_m2_offload}"
        )
        if not self.ran_real_deck:
            return f"R163 M2-crossing: real deck did NOT run | {cong}"
        return f"R163 M2-crossing: DRC={self.verdict} | {cong}"


def _emit_gds(library: ParameterizedLibrary, out: Path, cell_name: str) -> Optional[Path]:
    try:
        from clay.layout.binary_gds_subprocess import emit_binary_gds_via_subprocess

        gds = out / f"{cell_name}.gds"
        emit_binary_gds_via_subprocess(library, gds)
    except Exception:  # noqa: BLE001
        return None
    return gds if (gds.exists() and gds.stat().st_size > 0) else None


def run_m2_crossing_drc_demo(
    *,
    out_dir: Optional[Any] = None,
    run_drc: bool = True,
    klayout_bin: Optional[Path] = None,
    timeout_sec: float = 900.0,
) -> M2CrossingDrcResult:
    """Build the M2-offload crossing GDS, run the REAL gf180mcu.drc on it.

    Steps:
      1. Build the crossing demo + the M2-offload single-top-cell library.
      2. Compute the congestion delta (single-layer vs M2-offload) via the
         committed estimator, and run the pure geometry pre-check audit.
      3. Emit ONE binary GDS (committed Rust emitter).
      4. Run the REAL ``gf180mcu.drc`` deck on it; read the verbatim count and
         tally the BEOL (metal/via) subset specifically.

    Nothing is fabricated; an unavailable toolchain yields honest PENDING with
    the congestion delta + geometry-audit flag still populated (those are pure).
    """
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    demo = route_crossing_on_m2()
    cong = crossing_congestion(demo)
    audit = audit_geometry(demo)
    library, cell_name = build_m2_crossing_library(demo)

    result = M2CrossingDrcResult(
        crossings_single_layer=cong.crossings_single_layer,
        crossings_m2_offload=cong.crossings_m2_offload,
        geometry_audit_ok=audit.all_ok,
    )

    avail = real_drc_available(klayout_bin=klayout_bin)

    if not run_drc:
        result.pending_reason = "run_drc=False"
        return result
    if not avail:
        result.pending_reason = "real toolchain unavailable (deck / KLayout / Rust GDS)"
        return result

    gds = _emit_gds(library, out, cell_name)
    if gds is None:
        result.pending_reason = "Rust GDS emitter produced no GDS"
        return result
    result.gds_path = str(gds)
    result.gds_bytes = gds.stat().st_size

    try:
        from tools.klayout_drc_signoff_v139.run import run_gf180_drc

        report = run_gf180_drc(
            gds,
            report_path=out / f"{cell_name}.lyrdb",
            topcell=cell_name,
            klayout_bin=klayout_bin,
            run_mode="flat",
            timeout_sec=timeout_sec,
        )
        by_rule: Dict[str, int] = {}
        for v in getattr(report, "violations", []) or []:
            by_rule[getattr(v, "rule", "unknown")] = getattr(v, "count", 0)
        result.ran_real_deck = True
        result.total_violations = getattr(report, "total_violations", None)
        result.by_rule = by_rule
        result.beol_violations = sum(
            c
            for r, c in by_rule.items()
            if any(r.upper().startswith(p) for p in result._BEOL_PREFIXES)
        )
    except Exception as exc:  # noqa: BLE001 -- surface honestly
        result.pending_reason = (
            f"real DRC deck failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        )

    return result


__all__ = [
    "DEFAULT_KLAYOUT_BIN",
    "DEFAULT_OUT_DIR",
    "REPO_ROOT",
    "M1_KEY",
    "M2_KEY",
    "VIA1_KEY",
    "M1_MIN_WIDTH_NM",
    "M1_MIN_SPACE_NM",
    "M2_MIN_WIDTH_NM",
    "M2_MIN_SPACE_NM",
    "VIA1_SIZE_NM",
    "VIA1_MIN_SPACE_NM",
    "VIA12_PAD_NM",
    "CrossingDemo",
    "CongestionDelta",
    "GeometryAudit",
    "M2CrossingDrcResult",
    "RoutedNet",
    "audit_geometry",
    "build_m2_crossing_library",
    "crossing_congestion",
    "drc_deck_available",
    "klayout_binary_available",
    "real_drc_available",
    "route_crossing_on_m2",
    "run_m2_crossing_drc_demo",
    "rust_gds_available",
    "via12_stack",
]
