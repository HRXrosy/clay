"""E1 ELECTRICAL OBLIGATIONS -- topology-derived electrical-intent checkers.

Clay gated only LEGALITY (DRC/LVS) and GEOMETRIC fidelity. This module adds the
ELECTRICAL-intent obligations distilled in E1, each a NAMED 5-state checker
(pass/fail/unmeasurable/unsupported/advisory) whose verdict is produced from
(a) the design's TOPOLOGY + DEVICE ROLES and (b) the AS-BUILT route result /
EMITTED GDS -- never a reference coordinate.

The four electrical intents and where they come from (LEAKAGE LINE: all DERIVED
FROM CIRCUIT TOPOLOGY, never copied from the reference layout):

  * symmetry_deviation  (symmetry-rigid): a diff pair / current mirror must route
    with MIRROR-SYMMETRIC metal about its shared axis. Source = mirror_pairs /
    diff_pairs (topology). Oracle = the route's per-pair ``mirror_reports`` (cell-
    set deviation of net_b vs the reflection of net_a, measured on the emitted
    metal). RED when any pair's deviation_frac exceeds the band (offset/CMRR risk).

  * sensitive_over_active_coupling (sensitive-net keep-off-active / shield): a
    high-impedance gain / output node must not carry large foreign-metal coupling.
    Source = role inference (high-Z internal gain nodes + the output node) from
    topology. Oracle = the geometric PEX coupling caps on the sensitive net from
    the emitted GDS. RED when a sensitive net's foreign coupling exceeds the band.

  * parasitic_balance (matched-net R/C): the two nets of a matched/mirror pair
    must extract to BALANCED parasitic capacitance (else the differential pair's
    post-layout offset/CMRR degrades). Source = mirror_pairs/diff_pairs ->
    matched NET pairs (topology). Oracle = geometric PEX per-net cap, compared as
    a RELATIVE imbalance. RED when imbalance exceeds the band.

  * em_width (power / large-signal net EM): a power / large-signal net must route
    at >= its EM-required metal width. Source = net-role classification (power
    nets) from topology. Oracle = the route's per-rail measured trunk width
    class. RED when a power net routes below the power width class.

All bands are RELATIVE percentages (per the relative-consistency rule) or
PDK-rule-derived, never an absolute reference value. The PEX backend is
``get_backend('auto')`` -- the calibrated Magic ext2spice when its Docker image is
present, else the always-available PDK-grounded GEOMETRIC estimator
(calibrated=False, surfaced in every report.detail -- NEVER presented as a field-
solver number). The verdict honestly downgrades to ``unmeasurable`` when the
oracle cannot observe the relevant geometry, and to ``unsupported`` when a hook is
absent -- never a silent pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


FIVE_STATES = frozenset(
    {"pass", "fail", "unmeasurable", "unsupported", "advisory"}
)

# RELATIVE bands (percent fractions, relative-consistency rule -- never absolute).
SYMMETRY_DEVIATION_MAX = 0.05      # routed mirror cell-set deviation_frac <= 5%
PARASITIC_IMBALANCE_MAX = 0.10     # matched-net cap rel imbalance <= 10%
SENSITIVE_COUPLING_FRAC_MAX = 0.20  # foreign coupling / net total cap <= 20%

# PLACEMENT-geometry symmetry band (independent of the router's mirror_reports).
# A matched/mirror pair's two device origins must be (a) y-collinear (same row --
# mirror is a reflection about a VERTICAL axis, so |dy| ~ 0) and (b) every pair
# inside one mirror GROUP must share ONE axis (the per-pair midpoint x must agree
# across the group). Both are measured as a RELATIVE fraction of the placement
# span so a uniform translation/scale leaves them identical (leakage line).
SYMMETRY_AXIS_FRAC_MAX = 0.02      # per-group axis spread / cell span <= 2%
SYMMETRY_ROW_FRAC_MAX = 0.02       # |dy| of a mirror pair / cell span <= 2%
# AXIS-CENTERING band: the shared mirror axis of a differential mirror-system
# must sit near the ENVELOPE x-center (|shared_axis - envelope_center| / width).
# An off-center diff axis (e.g. the VIP/VIN axis parked at the cell's right edge)
# makes a routed mirror geometrically infeasible (net_b's reflection lands off the
# cell / off-grid) and breaks the differential pair's matched input loading. The
# band is RELATIVE to the envelope width (translation/scale-invariant -- a
# coordinate cannot leak through a ratio). 5% tolerates the discrete-pack grain.
SYMMETRY_AXIS_CENTER_FRAC_MAX = 0.05  # |shared axis - env center| / width <= 5%

# EM-width (Jmax-derived) band. The PDK carries no Imax table, so the EM minimum
# for a power / large-signal net is a Jmax-derived MULTIPLE of the signal metal
# min-width: a power net carries the whole stage bias / output current, which a
# 1x-min-width wire cannot sustain at the foundry Jmax. The multiple is the
# power-width class ratio (config-calibrated, NOT a reference coordinate); the
# MEASURED input is the net's ACTUAL routed metal width from the emitted geometry,
# never the router's self-assigned class label (checker independence).
EM_POWER_WIDTH_MULTIPLE = 4.0     # power net EM-min = 4x the signal metal min-width

# NOTE (RETIRED 2026-06-19): the cap-specific placement-proxy checkers
# (check_cap_over_nodes / check_cap_planar_edge and their CAP_OVER_NODES_* /
# CAP_PLANAR_* bands) were retired with the cap-mode placement bolt-ons. The
# UNIFIED all-element placement co-places the cap as a no-overlap tile, so cap
# floorplan QUALITY is now judged by the GENERIC geometric-fidelity metrics
# (fill / largest-contiguous-dead-region / aspect on the emitted GDS), not a
# per-cap proxy. The cap is no longer special.

# UPPER-METAL-PLANARITY bands: the routed UPPER metals (M3 and above -- M1/M2 are
# local interconnect and EXCLUDED) must sit MOSTLY OFF active diffusion (planar
# channel routing, not over-cell). over_active_frac(level) = routed metal area on
# the level over device OD / total routed metal area on the level. Two RELATIVE
# per-level ceilings, calibrated from the signed reference (M3 ~13% over active,
# M4/M5/M6 ~0%) -- the reference CALIBRATES the band, never enters the metric:
#   * M3 (the first upper trunk level) tolerates a fold of planar metal crossing
#     the cells it threads between (reference 13%) -> a generous 0.25 ceiling
#     excludes the over-cell route (measured 40%) while not demanding a perfect 0.
#   * M4 and above are the long planar trunks the reference keeps essentially OFF
#     active (~0%) -> a tight 0.15 ceiling excludes the over-cell route (measured
#     58%).
# M2 is INTENTIONALLY excluded (local-over-active ~66% is normal interconnect).
UPPER_METAL_M3_OVER_ACTIVE_MAX = 0.25
UPPER_METAL_HIGH_OVER_ACTIVE_MAX = 0.15


@dataclass(frozen=True)
class ElecVerdict:
    """One electrical obligation, 5-state, measured from topology + as-built."""

    name: str
    state: str                       # one of FIVE_STATES
    value: Optional[float]           # the measured scalar (None when unobservable)
    threshold: str                   # the band the value was compared against
    members: Tuple[str, ...]         # the topology-sourced nets/devices
    detail: str
    backend: str = ""                # PEX backend ('magic'/'geometric'/'') + calib

    def __post_init__(self):
        if self.state not in FIVE_STATES:
            raise AssertionError(f"non-5-state electrical verdict: {self.state!r}")


# --------------------------------------------------------------------------- #
# topology / role derivation (LEAKAGE-FREE: nets + roles only, no coordinate). #
# --------------------------------------------------------------------------- #
def _norm(n: str) -> str:
    return str(n).strip().rstrip("!").upper()


def matched_net_pairs(intent: Any) -> List[Tuple[str, str]]:
    """The NET pairs that must balance, derived from matched DEVICE pairs.

    For each mirror/diff DEVICE pair, the two devices' DIFFERING per-role nets
    (gate, drain, source) form a matched NET pair (a shared net -- e.g. a common
    tail source -- yields nothing to balance). Pure topology."""
    devs = {str(d.name): d for d in (getattr(intent, "devices", None) or [])}
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    dev_pairs: List[Tuple[str, str]] = []
    for p in (getattr(intent, "mirror_pairs", None) or []):
        if len(p) == 2:
            dev_pairs.append((str(p[0]), str(p[1])))
    for p in (getattr(intent, "diff_pairs", None) or []):
        if len(p) == 2:
            dev_pairs.append((str(p[0]), str(p[1])))
    for (da, db) in dev_pairs:
        a, b = devs.get(da), devs.get(db)
        if a is None or b is None:
            continue
        for role in ("gate", "drain", "source"):
            na = str((getattr(a, "nets", None) or {}).get(role, "") or "")
            nb = str((getattr(b, "nets", None) or {}).get(role, "") or "")
            if not na or not nb or na == nb:
                continue
            key = frozenset((na, nb))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((na, nb))
    return pairs


def sensitive_nets(intent: Any) -> List[str]:
    """High-impedance gain / output nets (sensitive to coupling), from topology.

    A net is sensitive when it is a high-impedance internal gain node or the
    output: heuristic from roles -- a net that is the DRAIN of a device but NOT a
    power/ground rail and NOT a gate-bias node is a gain/output node. Pure
    topology + role; no coordinate. The diff-pair input gate nets (VIP/VIN) are
    also sensitive (mismatch -> offset)."""
    rails = {_norm(r) for r in (getattr(intent, "rails", None) or [])}
    rails |= {"VDD", "VSS", "GND", "VCC"}
    drains: set = set()
    gates: set = set()
    sources: set = set()
    for d in (getattr(intent, "devices", None) or []):
        nets = getattr(d, "nets", None) or {}
        if nets.get("drain"):
            drains.add(_norm(nets["drain"]))
        if nets.get("gate"):
            gates.add(_norm(nets["gate"]))
        if nets.get("source"):
            sources.add(_norm(nets["source"]))
    # gain/output nodes: a drain net that is not a rail and not (only) a bias gate.
    out: set = set()
    for n in drains:
        if n in rails:
            continue
        out.add(n)
    # diff-pair input gate nets (sensitive to offset): the gates of the diff pair.
    devs = {str(d.name): d for d in (getattr(intent, "devices", None) or [])}
    for p in (getattr(intent, "diff_pairs", None) or []):
        if len(p) != 2:
            continue
        for dn in p:
            d = devs.get(str(dn))
            if d is None:
                continue
            g = str((getattr(d, "nets", None) or {}).get("gate", "") or "")
            if g and _norm(g) not in rails:
                out.add(_norm(g))
    return sorted(out)


def power_nets(intent: Any) -> List[str]:
    """Power / large-signal nets (EM-width obligation), from topology + role.

    Uses the same role classifier the router uses (classify_net_role) so the
    obligation's net set matches the rail rows the route reports."""
    try:
        from clay.layout.route.anchor_ota import classify_net_role
    except Exception:
        classify_net_role = None  # type: ignore
    nets: set = set()
    for d in (getattr(intent, "devices", None) or []):
        for _t, n in (getattr(d, "nets", None) or {}).items():
            if not n:
                continue
            if classify_net_role is not None:
                try:
                    if classify_net_role(str(n), intent) == "power":
                        nets.add(_norm(n))
                        continue
                except Exception:
                    pass
            if _norm(n) in {"VDD", "VSS", "GND", "VCC"}:
                nets.add(_norm(n))
    return sorted(nets)


# --------------------------------------------------------------------------- #
# the four named checkers (each reads as-built result + emitted GDS).          #
# --------------------------------------------------------------------------- #
def check_symmetry_deviation(
    route_result: Dict[str, Any], intent: Any,
    *, max_dev: float = SYMMETRY_DEVIATION_MAX,
) -> ElecVerdict:
    """Routed mirror-symmetry deviation oracle (symmetry-rigid intent).

    Reads the route's ``mirror_reports`` (the per-pair cell-set deviation of the
    emitted metal). PASS iff every reported pair is satisfied AND its
    deviation_frac <= max_dev. The obligation EXISTS iff topology declares a
    mirror/diff pair; if mirror_routing produced no report (no constraint), the
    obligation is unmeasurable (the route did not draw a symmetry to judge)."""
    pairs = [tuple(p) for p in (getattr(intent, "mirror_pairs", None) or [])
             if len(p) == 2]
    pairs += [tuple(p) for p in (getattr(intent, "diff_pairs", None) or [])
              if len(p) == 2]
    members = tuple(sorted({str(x) for pr in pairs for x in pr}))
    if not pairs:
        return ElecVerdict(
            "symmetry_deviation", "unsupported", None,
            f"deviation_frac <= {max_dev}", members,
            "topology declares no mirror/diff pair to keep symmetric")
    reports = list(route_result.get("mirror_reports", []) or [])
    if not reports:
        return ElecVerdict(
            "symmetry_deviation", "unmeasurable", None,
            f"deviation_frac <= {max_dev}", members,
            "no mirror_reports: mirror_routing produced no symmetry constraint "
            "to judge (route drew no symmetry to measure)")
    worst = max(float(r.get("deviation_frac") or 0.0) for r in reports)
    all_ok = all(bool(r.get("satisfied"))
                 and float(r.get("deviation_frac") or 0.0) <= max_dev
                 for r in reports)
    deviating = [f"{r.get('net_a')}/{r.get('net_b')}={float(r.get('deviation_frac') or 0.0):.3f}"
                 for r in reports if not bool(r.get("satisfied"))]
    return ElecVerdict(
        "symmetry_deviation", "pass" if all_ok else "fail", worst,
        f"deviation_frac <= {max_dev} (all pairs satisfied)", members,
        f"{len(reports)} mirror pair(s); worst deviation_frac={worst:.3f}; "
        f"deviating: {deviating}")


def _mirror_groups(intent: Any) -> List[Tuple[str, ...]]:
    """Connected components of mirror_pairs (one matched GROUP per component).

    A mirror group shares ONE symmetry axis (a current mirror's legs / a diff
    pair's halves reflect about a single line). Pure topology: union-find over
    ``mirror_pairs`` + ``diff_pairs``; no coordinate, so translation-invariant."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    edges = [tuple(p) for p in (getattr(intent, "mirror_pairs", None) or [])
             if len(p) == 2]
    edges += [tuple(p) for p in (getattr(intent, "diff_pairs", None) or [])
              if len(p) == 2]
    for a, b in edges:
        union(str(a), str(b))
    groups: Dict[str, List[str]] = {}
    for a, b in edges:
        for x in (str(a), str(b)):
            groups.setdefault(find(x), [])
            if x not in groups[find(x)]:
                groups[find(x)].append(x)
    return [tuple(sorted(v)) for v in groups.values() if len(v) >= 2]


def check_symmetry_placement(
    route_result: Dict[str, Any], intent: Any,
    *, axis_frac_max: float = SYMMETRY_AXIS_FRAC_MAX,
    row_frac_max: float = SYMMETRY_ROW_FRAC_MAX,
    axis_center_frac_max: float = SYMMETRY_AXIS_CENTER_FRAC_MAX,
) -> ElecVerdict:
    """Mirror-axis deviation measured INDEPENDENTLY from the canonical placement.

    Distinct from :func:`check_symmetry_deviation`, which echoes the router's own
    ``mirror_reports`` (native guidance -- the router signing its own work). This
    checker reads ONLY the canonical placement coordinates (the same
    ``_bundle.placement`` the emitted GDS is stamped from) and re-derives, per
    mirror GROUP (topology), whether the members place mirror-symmetric:

      * ROW collinearity: a mirror pair (A, B) reflects about a VERTICAL axis, so
        their origins must share a y (|y_A - y_B| ~ 0). Measured as |dy| / span.
      * AXIS agreement: every pair inside ONE mirror group must share ONE axis,
        so each pair midpoint x_mid = (x_A + x_B)/2 must agree across the group.
        Measured as (max x_mid - min x_mid) / span.

    Both are RELATIVE to the placement span (translation- AND scale-invariant --
    a coordinate cannot leak through a ratio of spans). PASS iff every group's
    axis spread <= axis_frac_max AND every pair's |dy| <= row_frac_max. This is
    the INDEPENDENT oracle (checker-independence rule: the router's native
    guidance does not sign off). Unmeasurable when no placement is available."""
    groups = _mirror_groups(intent)
    members = tuple(sorted({m for g in groups for m in g}))
    if not groups:
        return ElecVerdict(
            "symmetry_placement", "unsupported", None,
            f"axis spread <= {axis_frac_max}, |dy| <= {row_frac_max}", members,
            "topology declares no mirror group to keep symmetric")
    bundle = route_result.get("_bundle")
    placement = getattr(bundle, "placement", None) if bundle is not None else None
    if not placement:
        return ElecVerdict(
            "symmetry_placement", "unmeasurable", None,
            f"axis spread <= {axis_frac_max}, |dy| <= {row_frac_max}", members,
            "no canonical placement in route result to measure symmetry from")
    xs = [float(p[0]) for p in placement.values()]
    ys = [float(p[1]) for p in placement.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    env_lo, env_hi = min(xs), max(xs)
    env_width = max(env_hi - env_lo, 1e-9)
    env_center = (env_lo + env_hi) / 2.0

    worst_row = 0.0
    bad: List[str] = []
    pair_edges = [tuple(p) for p in (getattr(intent, "mirror_pairs", None) or [])
                  if len(p) == 2]
    pair_edges += [tuple(p) for p in (getattr(intent, "diff_pairs", None) or [])
                   if len(p) == 2]
    # group membership: a pair's mirror-SYSTEM is the connected mirror-group it
    # belongs to (union-find component). Every 2-member reflection pair carries a
    # well-defined single reflection axis (its midpoint x); the SYSTEM axis is the
    # axis those pairs must agree on. A >=3-member matched set is laid out as an
    # interdigitated row whose INTERNAL pairwise midpoints deliberately differ, so
    # it is reported ADVISORY (its common-centroid balance is a distinct intent).
    gmember = {}
    for gi, g in enumerate(groups):
        for m in g:
            gmember[m] = gi

    # AXIS = the per-pair midpoint x of every 2-member reflection pair across the
    # WHOLE cell (a diff pair / 2-leg mirror reflects about ONE vertical axis whose
    # x is the pair midpoint). The mirror-system obligation, made NON-VACUOUS:
    #   * AXIS AGREEMENT -- all such pairs must share ONE axis. Measured as the
    #     cross-pair spread (max_axis - min_axis) / env_width (the bug the old
    #     mid_frac=0.0 hid: it never compared pair midpoints, so any scatter
    #     passed). RELATIVE to the envelope width (translation/scale-invariant).
    #   * AXIS CENTERING -- the shared axis must sit near the envelope x-center.
    #     Measured as |shared_axis - env_center| / env_width. An off-center diff
    #     axis is the routed-mirror infeasibility root cause (net_b's reflection
    #     lands off-cell / off-grid).
    #   * ROW collinearity -- each pair's two origins share a y (|dy| ~ 0).
    axes: List[Tuple[str, float]] = []
    for (da, db) in pair_edges:
        gi = gmember.get(str(da))
        if gi is not None and len(groups[gi]) != 2:
            continue  # >=3-member matched set: advisory (interdigitated row)
        pa = placement.get(str(da))
        pb = placement.get(str(db))
        if pa is None or pb is None:
            continue
        dy_frac = abs(float(pa[1]) - float(pb[1])) / span
        worst_row = max(worst_row, dy_frac)
        if dy_frac > row_frac_max:
            bad.append(f"{da}/{db} |dy|={dy_frac:.3f}")
        axes.append((f"{da}/{db}", (float(pa[0]) + float(pb[0])) / 2.0))

    axis_xs = [ax for (_lbl, ax) in axes]
    if axis_xs:
        axis_spread = (max(axis_xs) - min(axis_xs)) / env_width
        shared_axis = sum(axis_xs) / len(axis_xs)
        axis_center_frac = abs(shared_axis - env_center) / env_width
    else:
        axis_spread = 0.0
        shared_axis = env_center
        axis_center_frac = 0.0

    if axis_spread > axis_frac_max:
        bad.append(f"axis-spread={axis_spread:.3f} (pairs disagree on the axis: "
                   + ", ".join(f"{lbl}@{ax:.1f}" for (lbl, ax) in axes) + ")")
    if axis_center_frac > axis_center_frac_max:
        bad.append(f"axis-center={axis_center_frac:.3f} "
                   f"(shared axis x={shared_axis:.1f} off env center "
                   f"x={env_center:.1f})")

    ok = (axis_spread <= axis_frac_max
          and axis_center_frac <= axis_center_frac_max
          and worst_row <= row_frac_max)
    n_two = sum(1 for g in groups if len(g) == 2)
    n_advisory = sum(1 for g in groups if len(g) != 2)
    return ElecVerdict(
        "symmetry_placement", "pass" if ok else "fail",
        max(axis_spread, axis_center_frac, worst_row),
        f"axis spread <= {axis_frac_max}, axis center <= {axis_center_frac_max}, "
        f"|dy| <= {row_frac_max} (measured from canonical placement, not router "
        "mirror_reports)",
        members,
        f"{n_two} two-member reflection pair(s) judged (+{n_advisory} >=3-member "
        f"matched set(s) advisory); axis-spread={axis_spread:.3f}, "
        f"axis-center={axis_center_frac:.3f}, worst |dy|={worst_row:.3f}; "
        f"violating: {bad or 'none'}")


def check_upper_metal_planarity(
    route_result: Dict[str, Any], intent: Any,
    *, m3_max: float = UPPER_METAL_M3_OVER_ACTIVE_MAX,
    high_max: float = UPPER_METAL_HIGH_OVER_ACTIVE_MAX,
    upper_min_level: int = 3,
    tech_id: str = "open_pdk",
) -> ElecVerdict:
    """upper-metal-not-over-active obligation: the routed UPPER metals (M3 and
    above) must sit MOSTLY OFF active diffusion -- planar channel routing, NOT a
    route that climbs straight over the device cluster (over-cell).

    The oracle reads ONLY the canonical bundle (the SAME placement/geometry/routed
    the LVS/DRC oracles ran on -- checker independence, no second layout path) and
    measures, per metal level >= ``upper_min_level``, the over-active fraction
    (routed metal area on the level over device OD / total routed metal area on the
    level). PASS iff every observed upper level is within its band: M3 <= ``m3_max``
    (the first trunk level tolerates a planar fold crossing the cells it threads
    between, reference ~13%), M4+ <= ``high_max`` (long planar trunks the reference
    keeps ~0% off active). M1/M2 are LOCAL interconnect -- excluded (M2 over-active
    ~66% is normal). RED on the over-cell cp route (M3 ~40% / M4 ~58%).
    Unmeasurable when no upper-metal route / no device OD is observable."""
    from clay.layout.extract.upper_metal_planarity import upper_metal_over_active

    bundle = route_result.get("_bundle")
    placement = getattr(bundle, "placement", None) if bundle is not None else None
    geometry = getattr(bundle, "geometry", None) if bundle is not None else None
    routed = getattr(bundle, "routed", None) if bundle is not None else None
    band = (f"M3 over_active <= {m3_max}, M{upper_min_level + 1}+ over_active "
            f"<= {high_max}")
    if not placement or not geometry or routed is None:
        return ElecVerdict(
            "upper_metal_planarity", "unmeasurable", None, band, (),
            "no canonical placement+geometry+routed in route result to measure "
            "upper-metal over-active from")
    try:
        from clay.layout.tech import LayoutTech
        tech = LayoutTech.resolve(tech_id)
    except Exception as e:  # noqa: BLE001
        return ElecVerdict(
            "upper_metal_planarity", "unmeasurable", None, band, (),
            f"tech {tech_id} unresolvable to map metal levels: {e}")
    per_level = upper_metal_over_active(
        intent, placement, geometry, routed, tech=tech)
    upper = {lvl: m for lvl, m in per_level.items() if lvl >= upper_min_level}
    if not upper:
        return ElecVerdict(
            "upper_metal_planarity", "unmeasurable", None, band,
            tuple(f"M{lvl}" for lvl in sorted(per_level)),
            "no routed metal on any upper level (M%d+) to judge planarity from"
            % upper_min_level)

    def _ceiling(lvl: int) -> float:
        return m3_max if lvl == upper_min_level else high_max

    bad = []
    worst = 0.0
    detail_parts = []
    for lvl in sorted(upper):
        m = upper[lvl]
        frac = m["over_active_frac"]
        ceil = _ceiling(lvl)
        worst = max(worst, frac)
        ok = frac <= ceil
        detail_parts.append(
            f"M{lvl}={frac:.3f}(<= {ceil}; {m['over_um2']:.0f}/"
            f"{m['total_um2']:.0f}um2)" + ("" if ok else " [FAIL]"))
        if not ok:
            bad.append(f"M{lvl}={frac:.3f}>{ceil}")
    members = tuple(f"M{lvl}" for lvl in sorted(upper))
    return ElecVerdict(
        "upper_metal_planarity", "pass" if not bad else "fail", worst, band,
        members,
        "; ".join(detail_parts) + (f"; violating: {bad}" if bad else
                                   "; all upper metals planar"))


def _pex_report(gds_path: str, tech_id: str, ground_net: str = "GND"):
    """Extract PEX from the emitted GDS via the auto backend (Magic if present,
    else PDK-grounded geometric). Returns (report, backend_tag) or (None, tag)."""
    try:
        from clay.layout.pex import get_backend, MagicPexBackend
        from clay.layout.pex import GeometricPexBackend
    except Exception as e:  # noqa: BLE001
        return None, f"pex import failed: {e}"
    magic = MagicPexBackend()
    if magic.available():
        try:
            rep = magic.extract(gds_path, "amp_" + tech_id, tech_id)
            return rep, "magic (calibrated 2.5D ext2spice)"
        except Exception as e:  # noqa: BLE001
            return None, f"magic extract failed: {e}"
    be = GeometricPexBackend(ground_net=ground_net, pdk_id=tech_id)
    try:
        rep = be.extract(gds_path, "amp_" + tech_id, tech_id)
        return rep, "geometric (PDK-grounded estimate, NOT calibrated)"
    except Exception as e:  # noqa: BLE001
        return None, f"geometric extract failed: {e}"


def check_parasitic_balance(
    gds_path: str, intent: Any, *, tech_id: str = "open_pdk",
    max_imbalance: float = PARASITIC_IMBALANCE_MAX,
    ground_net: str = "GND",
) -> ElecVerdict:
    """Matched-net parasitic-cap balance oracle (parasitic-balance intent).

    For each matched NET pair (topology), extract both nets' total parasitic cap
    from the emitted GDS and compare as a RELATIVE imbalance. PASS iff every
    observable pair's imbalance <= max_imbalance. Pairs whose nets are absent from
    the extraction are skipped (not silently passed -- if NONE is observable the
    verdict is unmeasurable)."""
    pairs = matched_net_pairs(intent)
    members = tuple(sorted({n for pr in pairs for n in pr}))
    if not pairs:
        return ElecVerdict(
            "parasitic_balance", "unsupported", None,
            f"rel imbalance <= {max_imbalance}", members,
            "topology declares no matched net pair to balance")
    rep, tag = _pex_report(gds_path, tech_id, ground_net)
    if rep is None:
        return ElecVerdict(
            "parasitic_balance", "unmeasurable", None,
            f"rel imbalance <= {max_imbalance}", members,
            f"PEX unavailable: {tag}", backend=tag)
    nets_present = {_norm(n): n for n in rep.nets()}
    worst = 0.0
    observed: List[str] = []
    n_obs = 0
    for (na, nb) in pairs:
        ra = nets_present.get(_norm(na))
        rb = nets_present.get(_norm(nb))
        if ra is None or rb is None:
            continue
        ca = rep.net_total_cap_aF(ra)
        cb = rep.net_total_cap_aF(rb)
        denom = max(ca, cb, 1e-30)
        imb = abs(ca - cb) / denom
        n_obs += 1
        worst = max(worst, imb)
        observed.append(f"{na}/{nb}={imb:.3f}")
    if n_obs == 0:
        return ElecVerdict(
            "parasitic_balance", "unmeasurable", None,
            f"rel imbalance <= {max_imbalance}", members,
            f"no matched net pair observable in extraction ({tag})", backend=tag)
    state = "pass" if worst <= max_imbalance else "fail"
    return ElecVerdict(
        "parasitic_balance", state, worst,
        f"rel imbalance <= {max_imbalance} over {n_obs} matched pairs", members,
        f"worst rel imbalance {worst:.3f}; per-pair: {observed}", backend=tag)


def check_sensitive_coupling(
    gds_path: str, intent: Any, *, tech_id: str = "open_pdk",
    max_frac: float = SENSITIVE_COUPLING_FRAC_MAX, ground_net: str = "GND",
) -> ElecVerdict:
    """Sensitive-net coupling oracle (sensitive-net keep-off-active / shield).

    For each topology-derived sensitive net, measure the fraction of its total
    parasitic cap that is COUPLING to other signal nets (foreign coupling), from
    the emitted GDS. PASS iff every observable sensitive net's coupling fraction
    <= max_frac (a shielded / kept-off net couples mostly to ground, not to
    signals). Unmeasurable when PEX is unavailable or no sensitive net is
    observed."""
    sens = sensitive_nets(intent)
    members = tuple(sens)
    if not sens:
        return ElecVerdict(
            "sensitive_over_active_coupling", "unsupported", None,
            f"coupling frac <= {max_frac}", members,
            "topology declares no sensitive (high-Z/output) net")
    rep, tag = _pex_report(gds_path, tech_id, ground_net)
    if rep is None:
        return ElecVerdict(
            "sensitive_over_active_coupling", "unmeasurable", None,
            f"coupling frac <= {max_frac}", members,
            f"PEX unavailable: {tag}", backend=tag)
    ground = {_norm(g) for g in (ground_net, "GND", "VSS", "VDD", "VCC", "0")}
    coupling = rep.coupling_caps([n for n in rep.nets()
                                  if _norm(n) in ground])
    nets_present = {_norm(n): n for n in rep.nets()}
    worst = 0.0
    observed: List[str] = []
    n_obs = 0
    for s in sens:
        r = nets_present.get(_norm(s))
        if r is None:
            continue
        total = rep.net_total_cap_aF(r)
        if total <= 0:
            continue
        coup = sum(c.cap_aF for c in coupling if r in (c.a, c.b))
        frac = coup / total
        n_obs += 1
        worst = max(worst, frac)
        observed.append(f"{s}={frac:.3f}")
    if n_obs == 0:
        return ElecVerdict(
            "sensitive_over_active_coupling", "unmeasurable", None,
            f"coupling frac <= {max_frac}", members,
            f"no sensitive net observable in extraction ({tag})", backend=tag)
    state = "pass" if worst <= max_frac else "fail"
    return ElecVerdict(
        "sensitive_over_active_coupling", state, worst,
        f"coupling frac <= {max_frac} over {n_obs} sensitive nets", members,
        f"worst signal-coupling fraction {worst:.3f}; per-net: {observed}",
        backend=tag)


def check_em_width(
    route_result: Dict[str, Any], intent: Any,
) -> ElecVerdict:
    """Power/large-signal EM-width oracle (em-width intent).

    Reads the route's per-rail measured trunk width class (``rail_rows``). PASS
    iff every power net with routed metal is at the 'power' width class
    (rails_power_class). Unmeasurable when the route reports no rail row."""
    pnets = power_nets(intent)
    members = tuple(pnets)
    rail_rows = list(route_result.get("rail_rows", []) or [])
    if not rail_rows:
        return ElecVerdict(
            "em_width", "unmeasurable", None,
            "every power net at power width class", members,
            "route reported no rail row to judge EM width")
    below = [f"{r['net']}={r['routed_width_um']}um({r['class']})"
             for r in rail_rows
             if r.get("routed_width_um", 0) > 0 and r.get("class") != "power"]
    worst_w = min((float(r["routed_width_um"]) for r in rail_rows
                   if r.get("routed_width_um", 0) > 0), default=0.0)
    ok = bool(route_result.get("rails_power_class")) and not below
    return ElecVerdict(
        "em_width", "pass" if ok else "fail", worst_w,
        "every power net at power width class", members,
        f"{len(rail_rows)} rail row(s); below power class: {below or 'none'}")


def _net_max_metal_width_um(route_result: Dict[str, Any], net: str
                            ) -> Optional[float]:
    """The MEASURED max metal-segment width (um) of a net from canonical wires.

    Reads ``_bundle.routed.nets[net].wires`` -- the same net-tagged metal the
    emitted GDS is stamped from (no second path). For each wire polygon the
    segment width is min(bbox_w, bbox_h) in um (a straight wire's width is its
    narrow side). Returns the max over the net's wires (the trunk width), or None
    when the net carries no wire. Pure measured geometry -- NOT the router's
    self-assigned ``design_widths_um`` / class label (checker independence)."""
    bundle = route_result.get("_bundle")
    routed = getattr(bundle, "routed", None) if bundle is not None else None
    nets = getattr(routed, "nets", None) if routed is not None else None
    if not nets:
        return None
    nr = None
    for k, v in nets.items():
        if _norm(k) == _norm(net):
            nr = v
            break
    if nr is None:
        return None
    wires = getattr(nr, "wires", None) or []
    best = None
    for w in wires:
        poly = getattr(w, "polygon", None)
        pts = list(getattr(poly, "vertices", []) or []) if poly is not None else []
        if not pts:
            continue
        xs = [float(getattr(p, "x", 0)) for p in pts]
        ys = [float(getattr(p, "y", 0)) for p in pts]
        wbox = max(xs) - min(xs)
        hbox = max(ys) - min(ys)
        seg_w = min(wbox, hbox)  # nm (DBU)
        if seg_w <= 0:
            continue
        best = seg_w if best is None else max(best, seg_w)
    return None if best is None else best / 1000.0


def check_em_width_measured(
    route_result: Dict[str, Any], intent: Any,
    *, tech_id: str = "open_pdk",
    power_multiple: float = EM_POWER_WIDTH_MULTIPLE,
) -> ElecVerdict:
    """Power/large-signal EM-width oracle, MEASURED vs a Jmax-derived minimum.

    Distinct from :func:`check_em_width`, which reads the router's self-assigned
    rail ``class`` label. This checker derives the EM minimum width from the PDK
    (Jmax-derived: a power net must route at >= ``power_multiple`` x the signal
    metal min-width, since it carries the whole stage bias/output current a
    1x-min wire cannot sustain at the foundry Jmax) and compares it against the
    net's ACTUAL measured trunk metal width from the canonical net-tagged wires.
    PASS iff every power net's measured width >= the EM minimum. The min-width is
    a PDK rule (never a reference coordinate); the multiple is config-calibrated.
    Unmeasurable when no power net carries measurable wire."""
    pnets = power_nets(intent)
    members = tuple(pnets)
    if not pnets:
        return ElecVerdict(
            "em_width_measured", "unsupported", None,
            f">= {power_multiple}x metal1 min-width", members,
            "topology declares no power / large-signal net")
    try:
        from clay.layout.tech import LayoutTech
        tech = LayoutTech.resolve(tech_id)
        min_w_nm = tech.min_width("metal1")
    except Exception as e:  # noqa: BLE001
        return ElecVerdict(
            "em_width_measured", "unmeasurable", None,
            f">= {power_multiple}x metal1 min-width", members,
            f"PDK metal1 min-width unavailable: {e}")
    if not min_w_nm:
        return ElecVerdict(
            "em_width_measured", "unmeasurable", None,
            f">= {power_multiple}x metal1 min-width", members,
            "PDK reports no metal1 min-width to derive the EM minimum from")
    em_min_um = power_multiple * (float(min_w_nm) / 1000.0)
    worst = None
    below: List[str] = []
    observed: List[str] = []
    n_obs = 0
    for net in pnets:
        w = _net_max_metal_width_um(route_result, net)
        if w is None:
            continue
        n_obs += 1
        worst = w if worst is None else min(worst, w)
        observed.append(f"{net}={w:.3f}um")
        if w + 1e-6 < em_min_um:
            below.append(f"{net}={w:.3f}um")
    if n_obs == 0:
        return ElecVerdict(
            "em_width_measured", "unmeasurable", None,
            f">= {em_min_um:.3f}um (EM-min)", members,
            "no power net carries measurable metal to check EM width")
    state = "pass" if not below else "fail"
    return ElecVerdict(
        "em_width_measured", state, worst,
        f">= {em_min_um:.3f}um = {power_multiple}x metal1 min-width "
        f"({min_w_nm}nm)", members,
        f"{n_obs} power net(s) measured; below EM-min: {below or 'none'}; "
        f"widths: {observed}")


def measure_electrical(
    route_result: Dict[str, Any], intent: Any,
    *, gds_path: Optional[str] = None, tech_id: str = "open_pdk",
) -> List[ElecVerdict]:
    """All four E1 electrical obligations over one as-built route + emitted GDS."""
    gds = gds_path or route_result.get("_emitted_gds")
    out = [
        check_symmetry_deviation(route_result, intent),
        check_symmetry_placement(route_result, intent),
        check_upper_metal_planarity(route_result, intent, tech_id=tech_id),
        check_em_width(route_result, intent),
        check_em_width_measured(route_result, intent, tech_id=tech_id),
    ]
    if gds:
        out.append(check_parasitic_balance(gds, intent, tech_id=tech_id))
        out.append(check_sensitive_coupling(gds, intent, tech_id=tech_id))
    else:
        out.append(ElecVerdict(
            "parasitic_balance", "unmeasurable", None, "", (),
            "no emitted GDS for PEX"))
        out.append(ElecVerdict(
            "sensitive_over_active_coupling", "unmeasurable", None, "", (),
            "no emitted GDS for PEX"))
    return out


__all__ = [
    "FIVE_STATES", "ElecVerdict",
    "matched_net_pairs", "sensitive_nets", "power_nets",
    "check_symmetry_deviation", "check_symmetry_placement",
    "check_upper_metal_planarity",
    "check_parasitic_balance", "check_sensitive_coupling",
    "check_em_width", "check_em_width_measured", "measure_electrical",
    "SYMMETRY_DEVIATION_MAX", "PARASITIC_IMBALANCE_MAX",
    "SENSITIVE_COUPLING_FRAC_MAX", "SYMMETRY_AXIS_FRAC_MAX",
    "SYMMETRY_ROW_FRAC_MAX", "EM_POWER_WIDTH_MULTIPLE",
    "CAP_OVER_NODES_OVERLAP_FRAC_MIN",
    "CAP_PLANAR_OVER_ACTIVE_FRAC_MAX", "CAP_PLANAR_CENTER_Y_LO",
    "CAP_PLANAR_CENTER_Y_HI", "CAP_PLANAR_EDGE_DIST_FRAC_MAX",
    "UPPER_METAL_M3_OVER_ACTIVE_MAX", "UPPER_METAL_HIGH_OVER_ACTIVE_MAX",
]
