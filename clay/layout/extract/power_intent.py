"""POWER-NET structural primitive for the leakage-free intent extractor.

WHAT THIS IS
============
A power rail (vdd / vss / gnd / vcc) is a STRUCTURAL fact a designer declares, not a
coordinate: "these devices tie to the same supply and their bodies/sources sit on a
shared rail, so they share a rail y". The other extractor halves
(``relgraph.build_constraint_graph`` + ``leakfree``) capture device-to-device abut /
align / symmetric adjacencies, but the supply rails -- which tie WHOLE ROWS of devices
to one y -- only ever surface as raw ``offset`` bridge edges (the leak). Stripping the
leak then disconnects those rows into separate floating components.

This module extracts the rail as a FIRST-CLASS, COORDINATE-FREE relation, derived from
the NETLIST ALONE (which devices share a supply net), and emits it as ``align:y``
:class:`RelEdge` chains that the leak-free solver already understands. Adding those
edges re-connects the rows that stripping the offset bridges had split apart -- the
power primitive REDUCES the floating-component count.

THE LEAKAGE DISCIPLINE (a hard scientific constraint)
=====================================================
The EXTRACTED relation must NEVER store a raw coordinate. We are allowed to READ
``intent.device_bboxes`` (the reference positions) ONLY to VALIDATE that a candidate
relation actually holds in the reference -- the degradation START. The OUTPUT is
coordinate-free:

* a rail group is identified purely from the netlist (a net that is the ``source`` tie
  of >= 2 devices, or a name matching a supply pattern);
* the reference y's are read ONLY to decide is_aligned (the centroid-y SPREAD scalar);
  the rail's absolute y is never stored on any emitted edge or returned object;
* every emitted :class:`RelEdge` is ``align:y`` with ``dx == dy == 0`` -- a placeholder,
  not a position. It says "these devices share a y" with NO y value.

A group whose reference centroids are NOT actually rail-aligned (e.g. a vdd net whose
devices are scattered over 56um of y because the supply fans out across several bands)
is NOT emitted as a rail -- validation gates emission, so we never fabricate a rail the
reference does not exhibit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from clay.layout.extract.relgraph import RelEdge

_Bbox = Tuple[float, float, float, float]

# Default rail-alignment tolerance (um): a group whose centroid-y spread (max - min)
# is within this is treated as sharing a rail. A relative-to-PDK quantity in spirit;
# the supplied default suits the external_pdk OTA (within-rail spread ~2.5um, cross-band
# fan-out ~56um -- a wide separation either way).
_RAIL_TOL_UM: float = 5.0

# Supply-net name patterns (case-insensitive, trailing '!' allowed). A net whose name
# matches is a power net regardless of device fan-out count.
_SUPPLY_NAMES = frozenset({"vdd", "vss", "gnd", "vcc", "vee", "vpwr", "vgnd"})


# --------------------------------------------------------------------------- #
# small geometry helpers (reference-read, validation only)                    #
# --------------------------------------------------------------------------- #


def _bboxes_of(intent: Any) -> Dict[str, _Bbox]:
    raw = getattr(intent, "device_bboxes", None)
    if raw is None and isinstance(intent, dict):
        raw = intent.get("device_bboxes")
    return dict(raw or {})


def _centroid_y(bb: _Bbox) -> float:
    return (float(bb[1]) + float(bb[3])) / 2.0


def _norm_net(name: Any) -> str:
    """Normalize a net name for supply matching: lowercase, strip a trailing '!'."""
    s = str(name).strip().lower()
    if s.endswith("!"):
        s = s[:-1]
    return s


def _is_supply_name(name: Any) -> bool:
    return _norm_net(name) in _SUPPLY_NAMES


# --------------------------------------------------------------------------- #
# 1. extract power nets from the NETLIST only (leakage-free)                   #
# --------------------------------------------------------------------------- #


def extract_power_nets(intent: Any) -> Dict[str, List[str]]:
    """Power nets -> sorted device names, derived from the NETLIST ALONE (no geometry).

    A net is a POWER net iff EITHER:
      * its name matches a supply pattern (vdd / vss / gnd / vcc / ... , case-
        insensitive, trailing '!' allowed), OR
      * it appears as the ``source`` role of >= 2 devices (a shared supply / tie node).

    For each such net we collect every device tied to it via its ``source`` OR ``body``
    role (the two roles that actually sit on the rail). Returns ``{net_name -> sorted
    device names}`` keyed by the ORIGINAL (un-normalized) net name; nets with fewer than
    2 tied devices are dropped (a single device is not a rail). Reads ONLY the netlist
    role nets -- NO coordinate is touched, so this is leakage-free by construction.
    """
    devices = getattr(intent, "devices", None) or []

    # First pass: count source-role occurrences per net so we can apply the
    # ">= 2 sources" structural rule without depending on the supply-name list.
    source_count: Dict[str, int] = {}
    for d in devices:
        nets = getattr(d, "nets", {}) or {}
        src = nets.get("source")
        if src is not None:
            source_count[str(src)] = source_count.get(str(src), 0) + 1

    # Second pass: for every net that qualifies (name match OR >=2 sources), collect
    # the devices tied to it through source or body.
    groups: Dict[str, set] = {}
    for d in devices:
        nets = getattr(d, "nets", {}) or {}
        name = getattr(d, "name", None)
        if name is None:
            continue
        for role in ("source", "body"):
            net = nets.get(role)
            if net is None:
                continue
            net = str(net)
            qualifies = _is_supply_name(net) or source_count.get(net, 0) >= 2
            if qualifies:
                groups.setdefault(net, set()).add(str(name))

    return {net: sorted(devs) for net, devs in groups.items() if len(devs) >= 2}


# --------------------------------------------------------------------------- #
# 2. validate rail alignment (degradation START read; scalar output only)     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RailValidation:
    """Result of checking whether a device group forms an aligned rail.

    ``is_aligned`` is ``spread_um <= tol_um``. ``spread_um`` is the centroid-y SPREAD
    (max - min) of the group in the reference (a single scalar -- NO per-device y is
    retained, so nothing here can replay a coordinate). ``axis`` names the shared axis
    ('y' for a horizontal rail). This object validates a relation; it is not a position.
    """

    is_aligned: bool
    spread_um: float
    axis: str = "y"


def validate_rail_alignment(
    intent: Any, devices: List[str], *, tol_um: float = _RAIL_TOL_UM
) -> RailValidation:
    """Validate whether ``devices`` share a horizontal rail in the reference.

    Reads ``intent.device_bboxes`` (the degradation-START reference read, the ONLY place
    a coordinate is touched) to compute the group's centroid-y SPREAD (max - min). The
    group ``is_aligned`` iff that spread is within ``tol_um``. We return ONLY the scalar
    spread + the boolean + the axis label -- no y value is stored, so this validation
    cannot leak a position into the extracted intent.

    A group with < 2 placed devices is trivially aligned (spread 0): there is no rail
    relation to violate.
    """
    bboxes = _bboxes_of(intent)
    cys = [_centroid_y(bboxes[n]) for n in devices if n in bboxes]
    if len(cys) < 2:
        return RailValidation(is_aligned=True, spread_um=0.0, axis="y")
    spread = max(cys) - min(cys)
    return RailValidation(is_aligned=spread <= tol_um, spread_um=spread, axis="y")


# --------------------------------------------------------------------------- #
# 3. validate band separation (nmos band vs pmos band; coordinate-free out)    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BandValidation:
    """Result of checking whether the nmos and pmos device bands are disjoint in y.

    ``separated`` is True iff the two kinds' centroid-y ranges do not overlap (overlap
    <= ``tol_um``). ``overlap_um`` is the (signed-to-zero) overlap extent -- a single
    scalar. ``lower_kind`` names which kind has the lower centroid-y band ('nmos' or
    'pmos'); it is INFORMATIONAL ONLY -- this module never hardcodes which kind belongs
    on top or bottom, it just reports what the reference shows. Coordinate-free output.
    """

    separated: bool
    overlap_um: float
    lower_kind: str = ""


def _kind_y_range(
    intent: Any, kind: str
) -> Optional[Tuple[float, float, float]]:
    """Return ``(y_lo, y_hi, y_mid)`` of the centroid-y range of ``kind`` devices, or
    None if the kind has no placed devices. Validation read only."""
    bboxes = _bboxes_of(intent)
    cys = [
        _centroid_y(bboxes[d.name])
        for d in (getattr(intent, "devices", None) or [])
        if getattr(d, "kind", None) == kind and d.name in bboxes
    ]
    if not cys:
        return None
    lo, hi = min(cys), max(cys)
    return (lo, hi, (lo + hi) / 2.0)


def validate_band_separation(
    intent: Any, *, tol_um: float = 0.0
) -> BandValidation:
    """Validate whether the nmos band and pmos band are disjoint in centroid-y.

    Reads ``intent.device_bboxes`` (validation START) to find each kind's centroid-y
    range, then computes the overlap of the two ranges. ``separated`` is True iff the
    overlap is <= ``tol_um`` (default 0 -> strictly disjoint). ``lower_kind`` reports
    which kind sits lower (smaller centroid-y MID) -- INFORMATIONAL, never used as a
    placement rule. Output is coordinate-free: a boolean, a scalar overlap, and a label.

    If either kind is absent, the bands cannot overlap -> ``separated`` True, overlap 0.
    """
    nr = _kind_y_range(intent, "nmos")
    pr = _kind_y_range(intent, "pmos")
    if nr is None or pr is None:
        lower = ""
        if nr is not None:
            lower = "nmos"
        elif pr is not None:
            lower = "pmos"
        return BandValidation(separated=True, overlap_um=0.0, lower_kind=lower)

    n_lo, n_hi, n_mid = nr
    p_lo, p_hi, p_mid = pr
    # overlap of [n_lo,n_hi] and [p_lo,p_hi]; negative -> a gap (no overlap).
    overlap = min(n_hi, p_hi) - max(n_lo, p_lo)
    overlap = max(overlap, 0.0)
    lower_kind = "nmos" if n_mid <= p_mid else "pmos"
    return BandValidation(
        separated=overlap <= tol_um, overlap_um=overlap, lower_kind=lower_kind
    )


# --------------------------------------------------------------------------- #
# 4. emit coordinate-free rail relations                                       #
# --------------------------------------------------------------------------- #


def _chain_align_y(devices: List[str]) -> List[RelEdge]:
    """Emit a coordinate-free ``align:y`` chain over ``devices`` (sorted, dx=dy=0).

    Chaining (consecutive pairs) is enough to put every device in ONE structural
    component sharing a y; a full clique is redundant. Every edge carries dx=dy=0 -- a
    placeholder, NOT a position -- so the relation says only "these share a y".
    """
    out: List[RelEdge] = []
    seq = sorted(devices)
    for i in range(len(seq) - 1):
        out.append(
            RelEdge(child=seq[i + 1], parent=seq[i], dx=0.0, dy=0.0, kind="align:y")
        )
    return out


def power_rail_relations(
    intent: Any, *, tol_um: float = _RAIL_TOL_UM
) -> List[RelEdge]:
    """Emit the coordinate-free ``align:y`` rail relations for ``intent``.

    For every power net (from :func:`extract_power_nets`) whose group VALIDATES as
    rail-aligned (:func:`validate_rail_alignment` is_aligned), emit an ``align:y`` chain
    over its devices. A group that FAILS validation (e.g. a vdd net whose devices fan
    out across 56um of y) is NOT emitted -- validation gates emission, so we never
    fabricate a rail the reference does not exhibit.

    Additionally, when :func:`validate_band_separation` reports the kinds are separated,
    emit an ``align:y`` chain WITHIN each kind's rail-aligned supply groups as well (the
    band structure reinforces that each supply row is one shared-y band). Every emitted
    edge is coordinate-free (dx == dy == 0). Returns the de-duplicated edge list.
    """
    edges: List[RelEdge] = []
    seen: set = set()

    def _add(es: List[RelEdge]) -> None:
        for e in es:
            key = (e.kind, frozenset((e.child, e.parent)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(e)

    power = extract_power_nets(intent)
    # deterministic order over net names
    aligned_groups: List[List[str]] = []
    for net in sorted(power):
        group = power[net]
        rv = validate_rail_alignment(intent, group, tol_um=tol_um)
        if rv.is_aligned:
            aligned_groups.append(group)
            _add(_chain_align_y(group))

    # band reinforcement: if nmos/pmos bands are separated, the already-validated rail
    # groups each sit in one band -- re-emit the within-group align:y chain (idempotent
    # via de-dup). This keeps each supply row a single shared-y component.
    bv = validate_band_separation(intent)
    if bv.separated:
        for group in aligned_groups:
            _add(_chain_align_y(group))

    return edges


# --------------------------------------------------------------------------- #
# 5. augment a stripped graph with the power-rail relations                    #
# --------------------------------------------------------------------------- #


def augment_graph(graph: Any, intent: Any, *, tol_um: float = _RAIL_TOL_UM) -> Any:
    """Return a NEW graph = the STRIPPED (coordinate-free) ``graph`` PLUS rail edges.

    Strips every raw-coordinate ``offset`` edge from ``graph`` (via
    ``leakfree.strip_coordinates``) and ADDs the :func:`power_rail_relations` edges.
    The input ``graph`` is NOT mutated -- a fresh graph of the same type is returned with
    a new edge list. Because the added edges are coordinate-free ``align:y`` relations,
    the result is still leakage-free (``leakfree.assert_no_coordinates`` passes), yet the
    rails re-connect device rows that stripping the offsets had split into floating
    components.
    """
    from clay.layout.extract import leakfree
    from clay.layout.extract.relgraph import ConstraintGraph

    stripped = leakfree.strip_coordinates(graph)
    rail_edges = power_rail_relations(intent, tol_um=tol_um)

    new_edges = list(stripped.edges) + list(rail_edges)
    return ConstraintGraph(
        root=stripped.root,
        edges=new_edges,
        sizes=dict(stripped.sizes),
        axis_x=stripped.axis_x,
        names=list(stripped.names),
    )


# --------------------------------------------------------------------------- #
# 6. human report (proves the primitive reduces floating)                      #
# --------------------------------------------------------------------------- #


def report_power(intent: Any) -> str:
    """Human-readable multi-line report of the power-net extraction + its effect.

    Lists each power net with its devices, the rail-alignment validation (is_aligned +
    spread), the band-separation result, and -- the proof -- how many disconnected
    components the GEOMETRY-ONLY stripped graph has versus the POWER-AUGMENTED graph
    (counted via ``leakfree.solve_leakfree`` on both). Fewer components after augment
    means the rail primitive re-connected device rows that the leak strip had split.
    """
    from clay.layout.extract import leakfree
    from clay.layout.extract.relgraph import build_constraint_graph

    lines: List[str] = []
    lines.append(f"[power] cell={getattr(intent, 'cell', '?')}")

    power = extract_power_nets(intent)
    if not power:
        lines.append("[power] no power nets detected")
    for net in sorted(power):
        group = power[net]
        rv = validate_rail_alignment(intent, group)
        lines.append(
            f"[power] net {net}: {group} -> is_aligned={rv.is_aligned} "
            f"spread={rv.spread_um:.3f}um axis={rv.axis}"
        )

    bv = validate_band_separation(intent)
    lines.append(
        f"[power] band separation: separated={bv.separated} "
        f"overlap={bv.overlap_um:.3f}um lower_kind={bv.lower_kind}"
    )

    # geometry-only vs power-augmented component counts.
    geo = leakfree.strip_coordinates(build_constraint_graph(intent))
    aug = augment_graph(build_constraint_graph(intent), intent)
    _gp, geo_rep = leakfree.solve_leakfree(geo)
    _ap, aug_rep = leakfree.solve_leakfree(aug)
    lines.append(
        f"[power] components geometry-only={len(geo_rep.components)} "
        f"-> power-augmented={len(aug_rep.components)} "
        f"(floating {geo_rep.n_floating} -> {aug_rep.n_floating})"
    )
    return "\n".join(lines)


__all__ = [
    "RailValidation",
    "BandValidation",
    "extract_power_nets",
    "validate_rail_alignment",
    "validate_band_separation",
    "power_rail_relations",
    "augment_graph",
    "report_power",
]
