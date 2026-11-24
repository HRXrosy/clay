"""Common-centroid INTERDIGITATED device cluster for a matched MOS pair.

The core analog-matching layout primitive: when two same-kind, same-W/L devices
A and B are declared ``@interdigitated(A, B)`` (or ``@matching`` + a
common-centroid intent), their gate fingers are INTERLEAVED into ONE shared
diffusion array with a coincident centroid, instead of being placed as two
separate side-by-side multi-finger devices. Interleaving the fingers
(ABBA / ABAB) makes the two devices share a single uniform COMP/poly/well region
whose two sub-device centroids coincide, so a linear (ABAB) or even-order (ABBA)
process gradient averages out across the pair -- the offset-cancellation matching
paradigm.

Correct-by-construction
=======================
The array is DRC-legal BY THE GEOMETRY IT EMITS (not by patching symptoms) and
its two sub-centroids COINCIDE EXACTLY for both even and odd finger counts:

  * Every dimension (finger pitch, COMP enclosure, contact size/spacing, M1
    width/spacing, poly end-cap, gate-contact poly enclosure, implant overlap,
    well sizing) is sourced from the SAME resolved PDK
    :class:`~clay.layout.device_cell.DeviceCellRules` the single-device builder
    uses, so the array is exactly as legal as one DRC-clean MOS device.
  * The S/D contacts sit at the channel centre with a source rail below and the
    drain rails above -- the SAME recipe as
    :func:`clay.layout.device_cell.build_device`. Source columns common to one
    shared-source rail; each member's drain columns common to that member's drain
    rail.
  * The GATE routing follows the single-device gate-bus + gate-contact recipe
    exactly. When the two members share a gate net (a mirror load whose gates are
    both the diode node) ONE shared bus over the top straps every finger, with one
    gate contact -- identical to a single device. When the gates DIFFER (a diff
    input pair: ``vip`` / ``vin``) member A's fingers strap UP to a TOP gate bus
    and member B's fingers strap DOWN to a BOTTOM gate bus (below the source rail,
    in the field band). The two buses are on OPPOSITE sides of the COMP so they
    never violate poly spacing, and each member's fingers only reach THEIR bus.
  * EVEN per-device finger count lays the array in ONE row as the centro-symmetric
    ``A B B A`` (ABBA) or ``A B A B | B A B A`` (ABAB), whose two sub-centroids
    coincide exactly. ODD per-device count CANNOT centroid-coincide in a single
    row (a parity obstruction: ``2*nf`` fingers split ``nf``/``nf`` can never have
    equal index-centroids), so the array is laid in TWO centro-symmetric ROWS
    (row0 = ``A B A``, row1 = ``B A B``) -- the classic 2-D common-centroid -- so
    the A-centroid and B-centroid coincide exactly in x.

Net assignment (the matching crux)
==================================
A matched pair shares its SOURCE net (both halves' sources tie to the same node:
the tail of a diff pair, vss/vdd of a mirror load) and each half drives its OWN
DRAIN net. Within each row the interleaved finger array has ``cols+1`` S/D
diffusion columns; even columns are the SHARED-SOURCE columns (the matched pair's
common source) and each odd (drain) column ties to the DRAIN net of the member
owning the finger to its LEFT (the channel that column drains). This is exactly
how a real common-centroid diff pair is drawn.

Result contract
===============
The returned :class:`DeviceGeom` duck-types a single device for the placer: it
exposes ``extent()`` / ``polygons`` / ``source_rail_bbox`` / ``drain_rail_bbox``
/ ``gate_contact_bbox`` / ``terminals`` / ``nwell_bbox`` so
:func:`clay.layout.generic_placement.place_generic` treats the whole interleaved
array as ONE placed cluster (placed at one position; it is already
centroid-symmetric, so it is never mirror-reflected by the pair machinery). The
``cluster`` field carries the per-device finger ownership, the interleave pattern
and the per-finger x columns for provenance / verification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .annotations import InterdigitationPattern
from .backend import LayerPolygon, Point, Polygon
from .common_centroid import CentroidPattern
from .device_cell import (
    Bbox,
    DeviceCellRules,
    DeviceGeom,
    LayerKey,
    Terminal,
    _LayerKeys,
    _resolve_layer_keys,
    _rules_from_tech,
)
from .native_owned_artwork import NativeOwnedRectangles, attach_native_owned_artwork, select_native_owners, native_owners_by_net


def _rect(layer_key: LayerKey, box: Bbox) -> LayerPolygon:
    x0, y0, x1, y1 = (int(v) for v in box)
    return LayerPolygon(
        layer=layer_key[0],
        datatype=layer_key[1],
        polygon=Polygon.rectangle(x0, y0, x1, y1),
    )


@dataclass(frozen=True)
class ClusterInfo:
    """Per-device finger ownership + interleave provenance for the array.

    ``pattern`` is the realised interleave (``ABAB`` / ``ABBA``); ``order`` is the
    finger sequence as ``(device, local_idx)`` left-to-right then row-by-row
    (``device`` is the A/B-labelled member name); ``finger_owner`` is the
    cluster-wide finger index -> owning device label; ``finger_col_x`` is the
    per-finger CENTRE x (same order as ``finger_owner``) of the realised geometry
    (repeated across rows for a 2-row array, so the A/B sub-centroids are measured
    on the true column positions); ``device_a`` / ``device_b`` are the two member
    labels; ``sd_column_nets`` is the per-column net assignment (one entry per S/D
    diffusion column, left-to-right, for ROW 0 -- the column net pattern is
    identical across rows); ``rows`` is the number of finger rows (1 for even nf,
    2 for the odd-nf 2-D common-centroid). ``dummy_fingers`` counts edge dummy
    fingers the pattern added for symmetry (0 here -- the supported arrangements
    are balanced by construction)."""

    device_a: str
    device_b: str
    pattern: str
    order: Tuple[Tuple[str, int], ...]
    finger_owner: Tuple[str, ...]
    sd_column_nets: Tuple[str, ...]
    finger_col_x: Tuple[float, ...] = ()
    rows: int = 1
    dummy_fingers: int = 0
    # Per DUMMY-EDGE drain column of a folded ``@dummy_flank(gate_off)`` end-cap:
    # ``(x_center_nm, band_cy_nm, drain_net, dummy_member_name)``. EMPTY when no
    # flank dummy is folded (byte-identical-off). These are the OUTER edge columns
    # (col 0 / col n_all) that the interior-only reprojection would otherwise skip;
    # they carry the dummy's own drain net (== an adjacent corner's signal drain by
    # centro-symmetry) and are attributed to the DUMMY name so member_pins / the
    # short oracle see them.
    flank_drain_cols: Tuple[Tuple[int, int, str, str], ...] = ()
    # Per-member GATE-contact bbox for a SPLIT-gate pair (member -> contact bbox);
    # member A straps UP to a top bus, member B DOWN to a bottom bus, so a diff-pair
    # member exposes its OWN gate landing rather than the single stored contact. EMPTY
    # for a shared-gate pair (both members use the one ``gate_contact_bbox``).
    gate_contacts_by_member: Dict[str, Bbox] = field(default_factory=dict)
    # Net -> externally routable M1 drain commoning rail after every private
    # interleaved-column hop/jog has been materialized inside the tile.
    drain_rails_by_net: Dict[str, Bbox] = field(default_factory=dict)

    @property
    def total_fingers(self) -> int:
        return len(self.finger_owner)

    def interleave_string(self) -> str:
        """Compact ``A B B A`` ownership string (single-letter per member)."""
        a_first = self.finger_owner[0] == self.device_a if self.finger_owner else True
        out: List[str] = []
        for owner in self.finger_owner:
            out.append("A" if owner == self.device_a else "B")
        _ = a_first
        return " ".join(out)


def _row_sequence(n_per: int, pattern: CentroidPattern) -> List[str]:
    """One centro-symmetric ROW of ``2*n_per`` fingers (``"A"``/``"B"`` labels).

    The row holds EXACTLY ``n_per`` ``A`` and ``n_per`` ``B`` fingers whose
    index-centroids coincide AND whose interleave is ELECTRICALLY VALID: same-device
    fingers are grouped in adjacent PAIRS so each pair shares a DRAIN column, with
    SHARED-SOURCE columns at every pair boundary (a different-device or pair-to-pair
    junction). A bare single-finger ``A B A B`` interleave is NOT a valid diff-pair
    topology (a single A finger between two B fingers has no A-drain column), so the
    interleave is built from 2-finger PAIRS:

      * the row is ``n_per`` worth of A-pairs + B-pairs (``n_per/2`` pairs each when
        ``n_per`` is even);
      * pairs are interleaved centro-symmetrically: the first half alternates
        ``A-pair B-pair ...`` and the second half is its REVERSE, so the pair
        sequence is point-symmetric about the centre -> the A and B finger
        centroids coincide exactly;
      * ``ABBA`` lays the pairs as ``AA BB BB AA`` (groups of four pairs), also
        cancelling the even-order gradient.

    ``n_per`` MUST be even here (the caller routes odd counts to the 2-row builder).
    Each entry is one finger label; a "pair" is two consecutive identical labels.
    """
    n_pairs = n_per // 2  # PAIRS per device (each pair = 2 same-device fingers)
    # the pair-level sequence has n_pairs A-pairs + n_pairs B-pairs (2*n_pairs total).
    if pattern is CentroidPattern.ABBA and n_pairs % 2 == 0:
        # ABBA at the PAIR level: A B B A groups of four pairs.
        pair_seq: List[str] = []
        for _ in range(n_pairs // 2):
            pair_seq += ["A", "B", "B", "A"]
    else:
        # centro-symmetric pair interleave: first half is n_pairs alternating
        # A,B,A,B...; second half is its REVERSE -> point-symmetric, coincident
        # centroid, with exactly n_pairs of each device.
        first = ["A" if i % 2 == 0 else "B" for i in range(n_pairs)]
        pair_seq = first + list(reversed(first))
    # expand each pair label into TWO adjacent same-device fingers (the pair shares
    # a single interior DRAIN column; pair boundaries are SHARED-SOURCE columns).
    seq: List[str] = []
    for p in pair_seq:
        seq += [p, p]
    return seq


def _two_row_pair_sequence(nf: int) -> List[str]:
    """ONE row of ``nf`` fingers for the 2-row layout, as same-device PAIRS.

    ``nf`` must be EVEN (``nf/2`` pairs); the row alternates A-pairs and B-pairs
    (``A A B B A A ...``) so each pair shares a drain and pair boundaries are shared
    sources. Row 1 (the caller's A/B swap of this row) places the opposite device in
    every column, so A and B occupy the SAME column set across the two rows -> their
    x-centroids coincide (the 2-D common centroid)."""
    n_pairs = nf // 2
    out: List[str] = []
    for i in range(n_pairs):
        lab = "A" if i % 2 == 0 else "B"
        out += [lab, lab]
    return out


def interleave_sequence(
    nf: int, pattern: CentroidPattern
) -> List[Tuple[str, int]]:
    """Balanced, PAIR-based interleave of two ``nf``-finger devices (ONE row).

    Returns the finger sequence as ``(device, local_idx)`` of length ``2*nf`` with
    EXACTLY ``nf`` ``A`` and ``nf`` ``B`` fingers grouped in same-device PAIRS (each
    pair shares a drain). For ``nf`` divisible by 4 the one-row pair interleave is
    centro-symmetric (the two sub-centroids COINCIDE); for other even ``nf`` the
    one-row form is balanced but not centroid-coincident (the BUILD lays it in TWO
    centro-symmetric rows -- :func:`build_interdigitated_cluster`). ``nf`` must be
    EVEN (an odd / single finger cannot be drain-paired)."""
    if nf < 2 or nf % 2 != 0:
        raise ValueError(
            f"nf must be EVEN and >= 2 for a drain-pairing interleave (got {nf})")
    if nf % 4 == 0:
        devs_seq = _row_sequence(nf, pattern)
    else:
        # nf % 4 != 0: one row of nf/2 same-device pairs each (balanced; the BUILD
        # uses two rows for centroid coincidence).
        a_pairs = nf // 2
        devs_seq = []
        for _ in range(a_pairs):
            devs_seq += ["A", "A"]
        for _ in range(a_pairs):
            devs_seq += ["B", "B"]
    seq: List[Tuple[str, int]] = []
    a_idx = b_idx = 0
    for d in devs_seq:
        if d == "A":
            seq.append(("A", a_idx)); a_idx += 1
        else:
            seq.append(("B", b_idx)); b_idx += 1
    return seq


def _centroid_pattern(pattern: InterdigitationPattern, n_per: int) -> CentroidPattern:
    """Resolve the realised interleave pattern for a per-row split of ``n_per``.

    ABBA needs an even per-row finger count; an ABBA request that cannot be
    realised (odd per-row count) degrades to the centro-symmetric ABAB
    (``CROSS``), which always yields a coincident centroid. The default request
    (no annotation) is ABAB."""
    want = CentroidPattern(pattern.value)
    if want is CentroidPattern.ABBA and n_per % 2 != 0:
        return CentroidPattern.CROSS
    return want


def is_interdigitatable(
    kind_a: Optional[str],
    kind_b: Optional[str],
    nf_a: int,
    nf_b: int,
) -> bool:
    """True iff two devices can form a common-centroid interleaved array.

    Requirements (each a HONEST rejection -> the pair stays two separate devices):

      * SAME kind (nmos/pmos);
      * EQUAL finger counts;
      * EVEN finger count >= 2. A VALID diff-pair interleave groups same-device
        fingers in PAIRS so each pair shares a drain column (a single interleaved
        finger has no own-device drain -- it would extract as the wrong topology), so
        the count must be EVEN. (A single finger each, or an ODD count, cannot be
        cleanly drain-paired in a common-centroid array, so it is rejected.)

    EVEN counts divisible by 4 lay out in ONE centro-symmetric row (an even number of
    pairs per device); other even counts (2, 6, ...) lay out in TWO centro-symmetric
    rows (the 2-D common-centroid). A mixed-kind / unequal-finger / odd pair is NOT
    interdigitated."""
    return (
        kind_a is not None
        and kind_a == kind_b
        and nf_a >= 2
        and nf_a == nf_b
        and nf_a % 2 == 0
    )


def build_interdigitated_cluster(
    tech,
    *,
    kind: str,
    W: float,
    L: float,
    nf: int,
    device_a: str,
    device_b: str,
    nets_a: Optional[Dict[str, str]] = None,
    nets_b: Optional[Dict[str, str]] = None,
    pattern: InterdigitationPattern = InterdigitationPattern.ABAB,
    force_rows: Optional[int] = None,
    flank_dummies: Optional[Sequence[Tuple[str, str, int]]] = None,
) -> DeviceGeom:
    """Build a common-centroid INTERLEAVED finger array for a matched pair.

    Both devices are the SAME ``kind`` with the SAME ``W`` / ``L`` and ``nf``
    fingers each; the returned :class:`DeviceGeom` is ONE array of ``2*nf``
    interleaved fingers (in ONE row for even ``nf``, in TWO centro-symmetric rows
    for odd ``nf``) over a single shared COMP (one COMP per row), implant, (PMOS)
    well, with the source/drain columns commoned per the SCHEMATIC nets of the
    device that owns each finger.

    Parameters
    ----------
    tech : LayoutTech
        The aggregated PDK technology surface.
    kind : {'nmos','pmos'}
    W, L : float
        Per-finger channel width / length in METRES (same for both halves).
    nf : int
        Per-device finger count (the array has ``2*nf`` fingers).
    device_a, device_b : str
        The two member labels (A is the FIRST finger in the interleave).
    nets_a, nets_b : dict, optional
        ``{'s','d','g','b' -> net}`` for each half. The two halves share their
        source net; each drives its own drain. Roles absent bind to ''.
    pattern : InterdigitationPattern
        ABAB (centro-symmetric, cancels linear gradient) or ABBA (also cancels
        even-order); ABBA degrades to ABAB when the per-row count is odd.
    force_rows : int, optional
        The row count is PARITY-driven by default (``None``): ``nf`` divisible by 4
        lays in ONE centro-symmetric row, every other even ``nf`` in TWO. Passing
        ``force_rows=2`` REQUESTS the 2-band 2-D common-centroid fold even for an
        ``nf`` divisible by 4 (the reference folds the even-nf input pair into two
        stacked bands: same area, a taller aspect for the floorplan). The 2-band fold
        is the SAME generic per-row pair interleave + A/B row swap used for the
        non-multiple-of-4 case, so no per-``nf`` special-casing. Any value other than
        ``2`` leaves the parity default untouched (legacy callers stay byte-identical).
    flank_dummies : sequence of resolved end-cap tuples, optional
        RESOLVED common-centroid end-cap dummies (validated upstream by
        :func:`clay.layout.common_centroid_array.build_common_centroid_array`). Two
        shapes:
          * DRAIN-EXPOSED ``(name, drain_net, per_row)``: ``per_row`` gate-off
            finger(s) at ONE matched end of EACH row (centro-symmetric: the end whose
            adjacent signal drain net == the dummy's ``drain_net``). Its gate ties
            DOWN to the row source rail; its OWN drain net lands on a dead-end OUTER
            edge column (== the matching corner's signal drain by construction, so the
            same-net rail merely extends -- never a foreign-net crossing).
          * FULLY-OFF RAIL ``(name, rail_net, per_end, True)``: an ALL-SOURCE end-cap
            folded at BOTH ends of EVERY row (``per_end`` finger(s) per end), every
            column on the source rail net, gate tied DOWN to the source rail -- the
            common-centroid analogue of ``build_shared_diffusion_array``'s rail
            end-cap.
        EMPTY / ``None`` (the default) is byte-identical to the un-folded tile.

    Returns
    -------
    DeviceGeom
        ONE interleaved common-centroid array, with a ``cluster`` :class:`ClusterInfo`
        attached (``geom.__dict__['cluster']``) carrying the finger ownership +
        realised pattern + per-finger x columns.
    """
    if kind not in ("nmos", "pmos"):
        raise ValueError("kind must be 'nmos' or 'pmos'")
    if nf < 2 or nf % 2 != 0:
        # a valid drain-pairing interleave needs an EVEN finger count >= 2 (see
        # is_interdigitatable); odd / single-finger pairs are not interdigitatable.
        raise ValueError(
            "nf must be EVEN and >= 2 per device for a drain-pairing interleave "
            f"(got {nf}); odd/single-finger pairs are not interdigitatable")

    nets_a = dict(nets_a or {})
    nets_b = dict(nets_b or {})
    source_a = str(nets_a.get("s", "") or "")
    source_b = str(nets_b.get("s", "") or "")
    if not source_a.strip() or not source_b.strip() or source_a != source_b:
        raise ValueError(
            "a common-centroid pair requires the same non-empty source net for "
            f"both members (got {source_a!r} and {source_b!r})")

    r = _rules_from_tech(tech)
    keys = _resolve_layer_keys(tech, kind)
    is_pmos = kind == "pmos"

    l_nm = max(int(round(float(L) * 1e9)), r.gate_min_length)
    w_nm = max(int(round(float(W) * 1e9)), r.channel_min_width)

    # The interleave is PAIR-based (same-device fingers grouped in adjacent pairs
    # sharing a drain -- the only ELECTRICALLY VALID diff-pair topology). A ONE-row
    # centro-symmetric pair interleave needs an EVEN number of PAIRS per device, i.e.
    # ``nf`` divisible by 4. When ``nf % 4 != 0`` (nf = 2, 6, ... or odd) one row
    # cannot centroid-coincide, so the array is laid in TWO centro-symmetric ROWS:
    # row0 is a pair-based interleave, row1 is its A/B SWAP, so each device occupies
    # the SAME column set across the two rows -> identical x-centroid (the 2-D common
    # centroid) and each row is independently a valid pair interleave.
    cp = _centroid_pattern(pattern, nf)
    # ``force_rows=2`` REQUESTS the 2-band 2-D common-centroid fold even for an
    # nf%4==0 pair (which would otherwise lay in ONE wide centro-symmetric row). The
    # 2-band fold is the SAME generic construction the nf%4!=0 case already uses --
    # each row is a same-device PAIR interleave and row1 is row0's A/B swap, so each
    # device occupies the SAME column set across the two rows (identical x-centroid)
    # and, for an even per-row split, the SAME finger count per row (identical
    # y-centroid) -> a true 2-D common centroid. No per-nf / per-circuit branch.
    force_two_band = (force_rows == 2)
    if nf % 4 == 0 and not force_two_band:
        n_rows = 1
        row_owner_labels = [_row_sequence(nf, cp)]
    else:
        n_rows = 2
        # each row carries nf fingers of each device as nf/2 ... but a single row of
        # nf+nf fingers paired needs nf even per row. For the 2-row split each row
        # holds nf fingers total laid as same-device PAIRS: row0 = pairs of A then B
        # interleaved, row1 = the A/B swap. Build one row's pair interleave of nf
        # fingers (nf/2 pairs, balanced) then swap for row1.
        row0 = _two_row_pair_sequence(nf)
        row1 = ["B" if c == "A" else "A" for c in row0]
        row_owner_labels = [row0, row1]

    owner_label = {"A": device_a, "B": device_b}

    # Real PDK same-layer spacing (PL.3a poly space, M1.2a metal1 space), resolved
    # from the SAME tech the device builder uses. These drive the bus / rail / strap
    # clearances so the interleave is DRC-legal by construction (not the rules
    # container's width-based proxies, which under-count the space rule).
    poly_space = _tech_space(tech, _poly_layer_name(tech), r.poly_min_width)
    m1_space = _tech_space(tech, "metal1", r.metal1_min_width)
    active_space = _exact_active_space(tech)
    m2_facts = _resolve_m2_facts(tech, r)

    rects, comp_bboxes, sd_centres_per_row, col_nets, src_rail_bbox, \
        drn_rail_bbox, gate_contact_bbox, gate_bus_bboxes, nwell_bbox, \
        finger_owner, finger_col_x, order, flank_drain_cols, n_dummy_fingers, \
        gate_contacts, drain_rails_by_net = \
        _build_cluster_geom(
            poly_space=poly_space,
            m1_space=m1_space,
            active_space=active_space,
            m2_facts=m2_facts,
            l_nm=l_nm,
            w_nm=w_nm,
            row_owner_labels=row_owner_labels,
            owner_label=owner_label,
            keys=keys,
            r=r,
            is_pmos=is_pmos,
            nets_a=nets_a,
            nets_b=nets_b,
            device_a=device_a,
            device_b=device_b,
            flank_dummies=list(flank_dummies or ()),
        )

    polys = [_rect(key, box) for (key, box) in rects]

    s_net = nets_a.get("s", nets_b.get("s", ""))
    g_net = nets_a.get("g", "")
    d_net = nets_a.get("d", "")
    # A DIODE-connected device A (gate net == drain net) has its gate and drain
    # commoned INSIDE the cluster (the M1 tie above). Expose only ONE terminal for
    # that shared node (the drain rail) and DROP the separate 'g' pin, so the BEOL
    # router does not bridge two coincident-net cluster terminals (which lands a
    # degenerate via). GENERIC: driven by net equality, never a device name.
    diode_tied = bool(g_net) and g_net == d_net
    terminals: Dict[str, Terminal] = {}
    if src_rail_bbox is not None:
        terminals["s"] = Terminal("s", src_rail_bbox, net=s_net)
    if drn_rail_bbox is not None:
        terminals["d"] = Terminal("d", drn_rail_bbox, net=d_net)
    if gate_contact_bbox is not None and not diode_tied:
        terminals["g"] = Terminal("g", gate_contact_bbox, net=g_net)

    # ``sd_column_nets`` carries EVERY row's S/D columns, concatenated row by row
    # (row0 first). Each row has ``2*nf_per_row + 1`` columns; the check splits by
    # ``rows`` and validates each row's column commoning. (For a 1-row array this is
    # exactly the row-0 column list, byte-identical to the legacy single-row form.)
    sd_column_nets = tuple(n for row in col_nets for n in row)

    cluster = ClusterInfo(
        device_a=device_a,
        device_b=device_b,
        pattern=cp.value,
        order=tuple(order),
        finger_owner=tuple(finger_owner),
        sd_column_nets=sd_column_nets,
        finger_col_x=tuple(finger_col_x),
        rows=n_rows,
        dummy_fingers=int(n_dummy_fingers),
        flank_drain_cols=tuple(flank_drain_cols),
        gate_contacts_by_member=dict(gate_contacts),
        drain_rails_by_net=dict(drain_rails_by_net),
    )

    # the array's overall COMP bbox (the union of the per-row COMPs) for the
    # placer's single-device surface.
    comp_union = (
        min(b[0] for b in comp_bboxes),
        min(b[1] for b in comp_bboxes),
        max(b[2] for b in comp_bboxes),
        max(b[3] for b in comp_bboxes),
    )

    geom = DeviceGeom(
        polygons=polys,
        terminals=terminals,
        kind=kind,
        finger_count=len(finger_owner),
        gate_length_nm=l_nm,
        channel_width_nm=w_nm,
        comp_bbox=comp_union,
        mirror=False,
        axis_x=None,
        nwell_bbox=nwell_bbox,
        source_rail_bbox=src_rail_bbox,
        drain_rail_bbox=drn_rail_bbox,
        gate_bus_bbox=gate_bus_bboxes[0] if gate_bus_bboxes else None,
        gate_contact_bbox=gate_contact_bbox,
    )
    geom.__dict__["cluster"] = cluster
    return attach_native_owned_artwork(geom, rects.binding(polys))


def _build_cluster_geom(
    *,
    poly_space: int,
    m1_space: int,
    active_space: Optional[int],
    m2_facts: "_M2Facts",
    l_nm: int,
    w_nm: int,
    row_owner_labels: Sequence[Sequence[str]],
    owner_label: Dict[str, str],
    keys: _LayerKeys,
    r: DeviceCellRules,
    is_pmos: bool,
    nets_a: Dict[str, str],
    nets_b: Dict[str, str],
    device_a: str,
    device_b: str,
    flank_dummies: Optional[Sequence[Tuple[str, str, int]]] = None,
):
    """Construct the interleaved-array geometry (integer coords only).

    Mirrors the single-device ``device_cell._build_device_geom`` finger / COMP /
    contact / rail / gate-bus / gate-contact recipe -- every dimension is the same
    PDK-resolved value -- over each ROW of interleaved fingers, with the S/D
    columns commoned per the OWNING device's net and the GATE routed per-member.

    ``flank_dummies`` (RESOLVED ``(name, drain_net, per_row)`` tuples) fold gate-off
    common-centroid END-CAPS into each row as ``k = per_row`` finger(s) at EACH end
    (the finger index space becomes ``[k dummy | signal | k dummy]`` per row, the
    same ``sig_lo`` / ``sig_hi`` indirection ``device_cell.build_shared_diffusion_
    array`` uses). EMPTY -> byte-identical to the un-folded interleave. See the
    module + :func:`build_interdigitated_cluster` docstrings.

    Returns the rect list + the key feature bboxes + the flat finger ownership /
    per-finger x / order the cluster provenance carries, PLUS the dummy-edge drain
    column provenance (``flank_drain_cols``) and the folded dummy finger count."""
    finger_space = r.finger_space
    endcap = r.poly_endcap
    csize = r.contact_size
    sd_clear = r.contact_to_gate
    comp_enc = r.comp_contact_enc
    m1_enc = r.metal1_contact_enc
    m1_cut_enc = max(m1_enc, r.metal1_line_end)
    m1_pad = _m1_contact_pad_side(r, csize, m1_cut_enc)
    m1_w = max(r.metal1_min_width, csize + 2 * m1_enc)
    bus_band = max(r.poly_min_width, 180)

    # The gate routing mode: if BOTH members drive the SAME gate net, ONE shared
    # gate bus straps every finger (identical to a single device). Otherwise the
    # two members have DIFFERENT gates, routed on OPPOSITE sides (A up, B down).
    g_a = nets_a.get("g", "")
    g_b = nets_b.get("g", "")
    shared_gate = bool(g_a) and g_a == g_b

    # The finger poly must reach the gate bus on the side(s) the bus lives. With a
    # shared gate the bus is ABOVE the COMP: every finger's top-endcap is the
    # field-poly-to-comp band + the bus band, exactly like the single device, and
    # the bottom endcap is the minimal poly end-cap. With SPLIT gates member A's
    # fingers reach a TOP bus (top endcap = bus band) and member B's fingers reach a
    # BOTTOM bus (bottom endcap = bus band); the OPPOSITE side of each finger is the
    # minimal end-cap, kept >= poly-space clear of the foreign bus so the two buses
    # never touch a foreign finger.
    # The minimal poly end-cap (the non-bus side of a finger). The same-side
    # endcap (``top_band``) must extend far enough that this device's bus band
    # (``bus_band`` tall) sits a full poly-space ABOVE the FOREIGN device's minimal
    # end-cap, so a shared/own bus that spans across foreign columns never violates
    # poly spacing to a foreign finger -- and the bus stays >= poly min width.
    min_cap = endcap
    # same-side endcap: reach past the foreign min-cap + poly-space + the bus band.
    top_band = max(r.field_poly_to_comp + bus_band, min_cap + poly_space + bus_band)

    n_rows = len(row_owner_labels)
    n_cols = len(row_owner_labels[0])

    # --- resolved flank dummies (common-centroid END-CAPS) -------------------
    # Two spec shapes (resolved upstream by build_common_centroid_array):
    #   * DRAIN-EXPOSED 3-tuple ``(name, drain_net, per_row)`` -> ``dummy_by_drain``
    #     maps the drain net to (name, per_row); the per-row assignment (which dummy
    #     lands at which END of which row) is derived from the row's signal end-owner
    #     drain -- the parity-free, name-free rule.
    #   * FULLY-OFF RAIL 4-tuple ``(name, rail_net, per_end, True)`` -> ``rail_dummy``
    #     = (name, per_end): an ALL-SOURCE end-cap that folds at BOTH ends of EVERY
    #     row, all columns on the source rail net (no distinct drain).
    # ``k`` = the uniform per-END finger count each dummy contributes (== per_row /
    # per_end); ``k == 0`` (no dummy) is byte-identical to the un-folded tile.
    dummy_by_drain: Dict[str, Tuple[str, int]] = {}
    rail_dummy: Optional[Tuple[str, int]] = None
    _per_rows = set()
    for spec in (flank_dummies or ()):
        d_name, d_key, d_per_row = str(spec[0]), str(spec[1]), int(spec[2])
        is_rail = len(spec) > 3 and bool(spec[3])
        if is_rail:
            rail_dummy = (d_name, d_per_row)
        else:
            dummy_by_drain[d_key] = (d_name, d_per_row)
        _per_rows.add(d_per_row)
    if len(_per_rows) > 1:
        raise ValueError(
            "common-centroid flank dummies must share ONE per-row finger count "
            f"(got {sorted(_per_rows)}); a mixed end-cap width is unsupported")
    k = _per_rows.pop() if _per_rows else 0
    dummy_tail_low = 0
    if k and not shared_gate and n_rows > 1:
        # The reflected member-B buses connect to their outside-COMP spine by
        # crossing the row's flank columns.  A flank dummy's gate tail descends
        # to the source-rail contact, so the compact below-COMP bus would cut
        # through that different-net tail.  Put the B bus a full POLY space
        # below the tail/contact.  Member-B finger poly already consumes
        # ``top_band`` as its lower end-cap, so this construction-derived growth
        # carries every B finger to the displaced bus without a coordinate
        # special case.
        source_rail_offset = m1_w // 2 + _source_rail_gap(r)
        dummy_tail_low = (
            source_rail_offset
            + csize // 2
            + r.poly_contact_enc
        )
        top_band = max(
            top_band,
            dummy_tail_low + poly_space + bus_band,
        )

    all_owners = (device_a, device_b) + tuple(dummy[0] for dummy in flank_dummies)
    rects = NativeOwnedRectangles(all_owners)

    # With exact active-spacing authority a two-row split-gate fold reflects the
    # gate-bus SIDES together with the finger ownership.  Only one gate net then
    # occupies the inter-row field, so its legal gap is gate-to-foreign-COMP (or
    # active spacing), not two opposing gate bands plus POLY spacing.  If a PDK is
    # silent on active spacing, retain the older opposing-bus construction and its
    # larger gap: missing authority can never manufacture a compaction.
    point_reflected_split = (
        not shared_gate
        and n_rows == 2
        and type(active_space) is int
        and active_space > 0
        and all(
            {device_a, device_b}.issubset(
                {owner_label[label] for label in row}
            )
            for row in row_owner_labels
        )
    )
    if point_reflected_split:
        # The centre contact is below the lower inward POLY bus, while the
        # upper row's source rail is above it.  With coincident rule-width
        # buses (row_gap = 2 * top_band - bus_band), their M1 edge clearance is
        # top_band + poly_contact_enc + csize/2 - m1_pad/2 - m1_w - rail_gap.
        # Reserve the actual area-qualified M1 pad, not only the contact cut.
        # Lowering this contact instead would violate contact-to-COMP spacing.
        top_band = max(
            top_band,
            m1_space + m1_pad // 2 + m1_w + _source_rail_gap(r)
            - r.poly_contact_enc - csize // 2,
        )
    row_pitch = w_nm + _row_gap(
        r,
        m1_w,
        top_band,
        poly_space,
        active_space=active_space,
        dummy_tail_low=dummy_tail_low,
        point_reflected_split=point_reflected_split,
    )
    # Finger index space runs [k DUMMY | n_cols SIGNAL | k DUMMY] per row (the same
    # sig_lo/sig_hi indirection build_shared_diffusion_array uses). ``finger_x``
    # (the SIGNAL slice) is what the gate-bus / source-rail routing sees, so their
    # signal handling is byte-identical to the un-folded tile. When k==0 this is the
    # legacy signal-only grid.
    n_all_cols = k + n_cols + k
    sig_lo = k
    sig_hi = k + n_cols
    finger_x_all: List[int] = [c * (l_nm + finger_space) for c in range(n_all_cols)]
    finger_x: List[int] = finger_x_all[sig_lo:sig_hi]
    last_gate_x_max = finger_x[-1] + l_nm

    comp_bboxes: List[Bbox] = []
    sd_centres_per_row: List[List[int]] = []
    # drain columns grouped by net, across all rows (x, y of each drain contact).
    drain_cols_xy_by_net: Dict[str, List[Tuple[int, int]]] = {}
    # per-device finger x columns (centre x), flat across rows (for centroid).
    finger_owner: List[str] = []
    finger_col_x: List[int] = []
    order: List[Tuple[str, int]] = []
    a_local = b_local = 0
    # per-device finger top/bottom y for gate strap routing: (device -> list of
    # (col_x0, finger_bottom_y, finger_top_y)).
    fingers_geom_of: Dict[str, List[Tuple[int, int, int]]] = {device_a: [], device_b: []}

    src_net = nets_a.get("s", nets_b.get("s", ""))
    drn_net = {device_a: nets_a.get("d", ""), device_b: nets_b.get("d", "")}

    col_nets_all_rows: List[List[str]] = []
    # dummy-edge drain columns (x, band_cy, drain_net, dummy_name) -- the OUTER edge
    # columns a signal-only reprojection would skip; empty when k==0.
    flank_drain_cols: List[Tuple[int, int, str, str]] = []
    # per-row dummy finger CENTRE x's (for the source-rail hull widen + gate tie).
    dummy_tie_xs_per_row: List[List[int]] = []
    n_dummy_fingers = 0
    finger_names_per_row = []

    for ri, labels in enumerate(row_owner_labels):
        y0 = ri * row_pitch
        y1 = y0 + w_nm
        cy = y0 + w_nm // 2

        owners_row = [owner_label[c] for c in labels]

        # --- per-row dummy assignment (centro-symmetric, name-free) ----------
        # The LEFT end abuts the row's leftmost SIGNAL finger; match the dummy whose
        # drain net == that device's drain. Likewise the RIGHT end. For the 2-row
        # A/B-swap fold this places each dummy at reflected ends across rows, so its
        # own centroid coincides with the pair centroid (2-D common centroid kept).
        left_name = right_name = ""
        left_drain = right_drain = ""
        if k:
            if rail_dummy is not None:
                # FULLY-OFF RAIL all-source end-cap: the SAME dummy folds at BOTH ends
                # of every row, all its columns on the source rail net (no distinct
                # drain -> the outer edge columns are ordinary source columns).
                left_name = right_name = rail_dummy[0]
                left_drain = right_drain = src_net
            else:
                if n_rows == 1:
                    # A centro-symmetric one-row signal palindrome has the same
                    # active owner at both outer ends.  Selecting flank owners
                    # from those ends would duplicate that owner's dummy and drop
                    # the other source-bound dummy.  Drain-exposed flanks are
                    # electrically closed source-to-source at their inner edge,
                    # so bind the canonical pair order to opposite ends instead.
                    left_owner = device_a
                    right_owner = device_b
                else:
                    left_owner = owners_row[0]
                    right_owner = owners_row[-1]
                left_dummy = dummy_by_drain.get(drn_net.get(left_owner, ""))
                right_dummy = dummy_by_drain.get(drn_net.get(right_owner, ""))
                if left_dummy is None or right_dummy is None:
                    raise ValueError(
                        "common-centroid flank dummy drain net does not match a row "
                        "end signal device drain; refusing (no drain-net-match end-cap)")
                left_name, left_drain = left_dummy[0], drn_net[left_owner]
                right_name, right_drain = right_dummy[0], drn_net[right_owner]

        # per-finger owner classification over ALL n_all_cols columns: the SIGNAL
        # device name for a signal finger, None (a dummy) otherwise; ``finger_name``
        # carries the OWNER name (dummy name or signal device) for provenance.
        finger_dev: List[Optional[str]] = []
        finger_name: List[str] = []
        for c in range(n_all_cols):
            if sig_lo <= c < sig_hi:
                dev = owners_row[c - sig_lo]
                finger_dev.append(dev)
                finger_name.append(dev)
            else:
                finger_dev.append(None)
                finger_name.append(left_name if c < sig_lo else right_name)
        finger_names_per_row.append(tuple(finger_name))

        # --- gate poly fingers (row height + per-finger endcaps) ------------
        # SIGNAL fingers reach their gate bus (per-device endcap). DUMMY fingers get
        # the MINIMAL end-cap on BOTH sides -- they reach NEITHER the top VIP bus nor
        # the bottom VIN bus (both foreign signal nets); their gate ties DOWN to the
        # row source rail in a later pass.
        for c in range(n_all_cols):
            x0 = finger_x_all[c]
            rects.select((finger_name[c],), 'gate_finger')
            if finger_dev[c] is None:
                rects.append((keys.poly, (x0, y0 - min_cap, x0 + l_nm, y1 + min_cap)))
            else:
                dev = finger_dev[c]
                top_ext, bot_ext = _finger_endcaps(
                    dev,
                    device_a,
                    device_b,
                    shared_gate,
                    top_band,
                    min_cap,
                    row_index=ri,
                    point_reflected_split=point_reflected_split,
                )
                rects.append((keys.poly, (x0, y0 - bot_ext, x0 + l_nm, y1 + top_ext)))

        # --- ONE shared COMP (active) over this row (dummies + signal) ------
        comp_x0 = finger_x_all[0] - sd_clear - csize - comp_enc
        comp_x1 = finger_x_all[-1] + l_nm + sd_clear + csize + comp_enc
        comp_bbox: Bbox = (comp_x0, y0, comp_x1, y1)
        comp_bboxes.append(comp_bbox)
        rects.select(finger_name, 'shared_diffusion_implant')
        rects.append((keys.active, comp_bbox))

        # --- gate implant: COMP bbox grown by the implant-gate overlap -------
        implant_overlap = r.implant_gate_overlap
        if keys.gate_implant is not None:
            rects.append(
                (
                    keys.gate_implant,
                    (
                        comp_x0 - implant_overlap,
                        y0 - implant_overlap,
                        comp_x1 + implant_overlap,
                        y1 + implant_overlap,
                    ),
                )
            )

        # --- S/D contact columns (n_all_cols+1) at the channel centre -------
        sd_centres: List[int] = [finger_x_all[0] - sd_clear - csize // 2]
        for c in range(n_all_cols - 1):
            gap_lo = finger_x_all[c] + l_nm
            gap_hi = finger_x_all[c + 1]
            sd_centres.append((gap_lo + gap_hi) // 2)
        sd_centres.append(finger_x_all[-1] + l_nm + sd_clear + csize // 2)
        sd_centres_per_row.append(sd_centres)

        # --- per-column net (SIGNAL-LOCAL parity, dummy-shift aware) ---------
        # The signal region owns columns k .. k+n_cols; each carries the SAME net the
        # un-folded tile assigns by its SIGNAL-LOCAL index i = j - k (i even = shared
        # SOURCE, i odd = the DRAIN of the device owning the finger to its left). This
        # is IDENTICAL to the old j%2 rule when k==0 (i == j), so it stays byte-
        # identical for every interleave pattern -- including runs longer than a pair,
        # where S/D must alternate WITHIN the run (a plain "same-device -> drain" rule
        # would wrongly mark the middle column of a 4-finger run a drain). Prepending k
        # dummy fingers merely shifts the signal index by k; the two OUTER edge columns
        # (col 0 / col n_all) carry the flank dummy's own drain, and any dummy-interior
        # column (only when per_row > 1) is a shared source.
        row_col_nets: List[str] = []
        for j in range(len(sd_centres)):
            if j == 0:
                net = left_drain if k else src_net
            elif j == n_all_cols:
                net = right_drain if k else src_net
            elif k <= j <= k + n_cols:
                i = j - k  # signal-local column index (== j when k==0)
                if i % 2 == 0:
                    net = src_net
                else:
                    owner = owners_row[i - 1] if (i - 1) < n_cols else owners_row[-1]
                    net = drn_net.get(owner, "")
            else:
                net = src_net  # a dummy-interior column (per_row > 1 end-cap)
            row_col_nets.append(net)
        col_nets_all_rows.append(row_col_nets)

        # --- register drain columns (signal pair interiors + dummy edges) ----
        # Same ``drain_cols_xy_by_net`` dict the signal drains use; a dummy EDGE
        # drain (net != source) also records provenance so the interior-only
        # reprojection does not drop it. Row-major, column-ascending order == the
        # legacy order when k==0 (only the odd interior columns are drains).
        for j in range(len(sd_centres)):
            net = row_col_nets[j]
            if net and net != src_net:
                drain_cols_xy_by_net.setdefault(net, []).append((sd_centres[j], cy))
                if j == 0 or j == n_all_cols:
                    dname = left_name if j == 0 else right_name
                    flank_drain_cols.append((sd_centres[j], cy, net, dname))

        # --- S/D contacts + M1 pads -----------------------------------------
        for column_index, cx in enumerate(sd_centres):
            rects.select(finger_name[max(0, column_index-1):min(n_all_cols, column_index+1)], 'sd_column')
            rects.append(_contact_rect(keys, csize, cx, cy))
            rects.append(_m1_pad_rect(keys, m1_pad, cx, cy))

        # --- record per-finger ownership / x / geom for gate routing --------
        # ALL fingers (dummies + signal) enter the cluster provenance so the
        # reprojection materialises the extra dummy fingers; only SIGNAL fingers feed
        # gate routing (the dummy gate ties to source, not a bus).
        row_tie_xs: List[int] = []
        for c in range(n_all_cols):
            finger_owner.append(finger_name[c])
            finger_col_x.append(finger_x_all[c] + l_nm // 2)
            if finger_dev[c] is None:
                order.append(("D", 0))
                n_dummy_fingers += 1
                row_tie_xs.append(finger_x_all[c] + l_nm // 2)
                continue
            dev = finger_dev[c]
            if dev == device_a:
                order.append(("A", a_local)); a_local += 1
            else:
                order.append(("B", b_local)); b_local += 1
            top_ext, bot_ext = _finger_endcaps(
                dev,
                device_a,
                device_b,
                shared_gate,
                top_band,
                min_cap,
                row_index=ri,
                point_reflected_split=point_reflected_split,
            )
            fingers_geom_of[dev].append((finger_x_all[c], y0 - bot_ext, y1 + top_ext))
        dummy_tie_xs_per_row.append(row_tie_xs)

    # ── per-row source rails (below each row's COMP), commoning sources ──────
    # SECOND PARITY SITE FIX: the source columns are taken from the EXPLICIT built
    # net map (columns whose net == source), NOT recomputed via j%2 -- after a dummy
    # is spliced j%2 would select DRAIN indices and build the rail over drain columns
    # (a catastrophic short). Because the structure rule == j%2 when k==0, the derived
    # source-x lists equal the old parity result -> byte-identical-off. The dummy
    # finger centre x's widen the rail hull (so the gate-tie contact lands on rail M1)
    # WITHOUT adding a source riser there.
    source_xs_per_row: List[List[int]] = []
    for ri in range(n_rows):
        row_nets = col_nets_all_rows[ri]
        row_cent = sd_centres_per_row[ri]
        source_xs_per_row.append(
            [row_cent[j] for j in range(len(row_cent)) if row_nets[j] == src_net])
    tie_xs = dummy_tie_xs_per_row if k else None
    rects.select(all_owners, 'shared_source_routing')
    src_rail_bbox, min_src_rail_y = _emit_source_rails(
        rects, keys, r, m1_w, n_rows, source_xs_per_row, row_pitch, w_nm,
        tie_xs_per_row=tie_xs,
    )

    # ── gate buses + gate contacts ──────────────────────────────────────────
    gate_bus_bboxes, gate_contact_bbox, gate_contacts = _emit_gate_routing(
        rects, keys, r, csize, m1_pad, bus_band, m1_w, m1_space, w_nm, l_nm,
        finger_x, poly_space, row_owner_labels, owner_label, device_a, device_b,
        shared_gate, n_rows, row_pitch, top_band, min_cap, min_src_rail_y,
        point_reflected_split,
        (
            min(bbox[0] for bbox in comp_bboxes),
            max(bbox[2] for bbox in comp_bboxes),
        ),
    )

    # ── flank-dummy gate tie: poly tail DOWN to the row source rail ─────────
    # After the source rails + gate buses exist, tie each dummy finger's gate to the
    # row source rail (poly tail + poly->M1 contact at the dummy finger centre, on
    # the widened rail M1). RAISES on a clearance failure (rail-to-drain, tie-to-bus,
    # finger_space<poly_space) so the resolver falls back and the dummy stays free --
    # never a silent short.
    if k:
        _emit_flank_dummy_gate_ties(
            rects, keys, r, csize, l_nm, m1_w, m1_pad, m1_space, sd_clear,
            poly_space, min_cap, row_pitch, n_rows, sig_lo, sig_hi, n_all_cols,
            finger_x_all, gate_bus_bboxes,
            rail_all_source=(rail_dummy is not None),
            finger_names_per_row=finger_names_per_row,
        )

    top_of_buses = (
        max(b[3] for b in gate_bus_bboxes) if gate_bus_bboxes
        else (n_rows - 1) * row_pitch + w_nm + top_band
    )
    # The topmost gate-contact M1 pad: the (side='up') gate contact's M1 pad rises
    # m1_pad//2 above its cut. The drain rails must clear it by the M1 min space.
    gate_m1_top = top_of_buses
    if gate_contact_bbox is not None:
        gate_m1_top = max(gate_m1_top, gate_contact_bbox[3] + m1_pad // 2)

    # ── drain rails (dA on M1, dB on M2 -- no interleaved-crossing short) ────
    # the union COMP x-span bounds any drain-rail extension a jogged isolation riser
    # needs, so the extended rail stays inside the tile footprint (COMP is enclosed by
    # the well / source-rail hull) -- the tile bbox is unchanged.
    comp_x_span = (min(b[0] for b in comp_bboxes), max(b[2] for b in comp_bboxes))
    drn_rail_bbox, drn_rails_by_net = _emit_drain_rails(
        rects, keys, r, m1_w, m1_space, gate_m1_top,
        m2_facts.m2_key, m2_facts.via1_key, m2_facts.via1_size,
        m2_facts.via1_enc_below, m2_facts.via1_enc_above,
        m2_facts.m2_w, m2_facts.m2_space,
        drn_net, device_a, device_b, drain_cols_xy_by_net, top_of_buses,
        comp_x_span=comp_x_span,
        owners_by_net=native_owners_by_net(*drn_net.items(),
            *((member, net) for _x, _y, net, member in flank_drain_cols)),
    )

    # ── DIODE tie: a device whose GATE net == its DRAIN net (a diode-connected
    # load / mirror reference) has its gate and drain on the SAME node. Common them
    # INSIDE the cluster with an M1 strap from the device's gate-contact M1 pad up to
    # its drain rail, so the array extracts the gate/drain as one net WITHOUT relying
    # on the BEOL router to bridge two coincident-net cluster terminals (which lands a
    # degenerate via). GENERIC: driven purely by net equality, never a device name.
    if gate_contact_bbox is not None and shared_gate:
        rects.select((device_a, device_b), 'shared_signal_gate_diode_tie')
        g_net = nets_a.get("g", "")
        if g_net and g_net == nets_a.get("d", "") and g_net in drn_rails_by_net:
            _tie_gate_to_drain(
                rects, keys, m1_w, gate_contact_bbox, drn_rails_by_net[g_net])
        elif g_net and g_net == nets_b.get("d", "") and g_net in drn_rails_by_net:
            _tie_gate_to_drain(
                rects, keys, m1_w, gate_contact_bbox, drn_rails_by_net[g_net])

    # ── PMOS shared Nwell enclosing the whole array (all rows) ──────────────
    nwell_bbox: Optional[Bbox] = None
    rects.select(all_owners, 'shared_well')
    if is_pmos and keys.nwell is not None:
        from .device_well_geometry import shared_nwell_bbox
        nwell_bbox = shared_nwell_bbox(comp_bboxes, r)
        rects.append((keys.nwell, nwell_bbox))

    return (
        rects,
        comp_bboxes,
        sd_centres_per_row,
        col_nets_all_rows,
        src_rail_bbox,
        drn_rail_bbox,
        gate_contact_bbox,
        gate_bus_bboxes,
        nwell_bbox,
        finger_owner,
        finger_col_x,
        order,
        flank_drain_cols,
        n_dummy_fingers,
        gate_contacts,
        drn_rails_by_net,
    )


# ── geometry helpers (all dims from the resolved PDK rules) ──────────────────


@dataclass(frozen=True)
class _M2Facts:
    """The metal2 / via1 GDS keys + dimensions for the second drain net's rail (the
    interleaved-pair's upper drain commoned on M2 to clear the M1 lower rail). All
    resolved from the SAME tech the device builder uses."""

    m2_key: LayerKey
    via1_key: LayerKey
    via1_size: int
    via1_enc_below: int
    via1_enc_above: int
    m2_w: int
    m2_space: int


def _resolve_m2_facts(tech, r: DeviceCellRules) -> _M2Facts:
    """Resolve metal2 + via1 facts from ``tech`` (keys, via size/enclosure, M2
    width/space). Via enclosure is a hard two-face declaration: a silent PDK is
    rejected instead of emitting an unverifiable landing."""
    m2_key = tuple(tech.metal_key(2))
    via1_key = tuple(tech.via_key(1))
    try:
        via1_size = int(tech.via_size("via1")) or r.contact_size
    except Exception:  # noqa: BLE001
        via1_size = r.contact_size
    try:
        via_rule = tech.via_rule("via1")
        via1_enc_below = int(via_rule.enclosure_below.value)
        via1_enc_above = int(via_rule.enclosure_above.value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "common-centroid M2 routing requires declared via1 enclosure on both faces"
        ) from exc
    m2_min_w = _width_or_node(tech, "metal2", via1_size + 2 * via1_enc_above)
    m2_w = max(m2_min_w, via1_size + 2 * via1_enc_above)
    m2_space = _tech_space(tech, "metal2", m2_min_w)
    return _M2Facts(
        m2_key=m2_key, via1_key=via1_key, via1_size=via1_size,
        via1_enc_below=via1_enc_below, via1_enc_above=via1_enc_above,
        m2_w=m2_w, m2_space=m2_space)


def _width_or_node(tech, name: str, fallback: int) -> int:
    try:
        v = tech.min_width(name)
    except Exception:  # noqa: BLE001
        v = None
    return int(v) if v else int(fallback)


def _poly_layer_name(tech) -> str:
    """The PDK rule-container poly layer name (gf180 'poly2', external_pdk 'poly')."""
    from .device_cell import _poly_name
    return _poly_name(tech)


def _exact_active_space(tech) -> Optional[int]:
    """Return an exact positive active spacing rule, or ``None`` when silent.

    Unlike :func:`_tech_space`, this optimization authority has no width-based
    fallback.  A missing rule keeps the older, larger row topology; it can never
    authorize a smaller gap.
    """
    from .device_cell import _active_name

    try:
        value = tech.min_space(_active_name(tech))
    except (KeyError, TypeError, ValueError):
        value = None
    return int(value) if type(value) is int and value > 0 else None


def _tech_space(tech, layer_name: str, fallback: int) -> int:
    """The PDK same-layer min space for ``layer_name`` (nm), or ``fallback``.

    Sourced from ``tech.min_space`` -- the REAL foundry space rule (gf180 poly2
    0.24um, metal1 0.23um) -- so the cluster's bus / rail / strap clearances are
    DRC-legal by construction. Falls back to the rules-resolved min width when the
    PDK does not expose an explicit space (a conservative lower bound)."""
    try:
        v = tech.min_space(layer_name)
    except Exception:  # noqa: BLE001 - missing space rule -> width-based fallback
        v = None
    return int(v) if v else int(fallback)


def _m1_contact_pad_side(
    r: DeviceCellRules,
    contact_size: int,
    cut_enclosure: int,
) -> int:
    """Smallest square M1 contact landing satisfying enclosure and min area."""
    min_area = max(0, int(r.metal1_min_area))
    area_side = math.isqrt(min_area)
    if area_side * area_side < min_area:
        area_side += 1
    side = max(int(contact_size) + 2 * int(cut_enclosure), 380, area_side)
    # Pads are centred on an integer-nm contact coordinate and emitted as
    # ``center +/- side//2``.  Keep the selected side even so that construction
    # cannot truncate an odd mathematical minimum by one nanometre.
    return side + side % 2


def _source_rail_gap(r: DeviceCellRules) -> int:
    """Generic source-rail field clearance shared by every CC consumer."""
    return r.field_poly_to_comp + 50


def _row_gap(
    r: DeviceCellRules,
    m1_w: int,
    top_band: int,
    poly_space: int,
    *,
    active_space: Optional[int] = None,
    dummy_tail_low: int = 0,
    point_reflected_split: bool = False,
) -> int:
    """Vertical gap between two finger rows.

    The lower row's gate band (member-A / shared bus reaches ``top_band`` above the
    row) sits in this gap, then >= poly-space clearance, then the upper row's source
    rail (``m1_w``) with field clearances. Rules-derived so the inter-row region is
    DRC-clean by construction."""
    if point_reflected_split:
        if type(active_space) is not int or active_space <= 0:
            raise ValueError(
                "point-reflected split-gate row gap requires exact positive "
                "active spacing authority"
            )
        # One bus net occupies the centre field, but the reflected foreign
        # finger in the opposite row still extends by ``poly_endcap``.  Keep the
        # reaching gate finger a full same-layer POLY space from that minimal
        # foreign endcap; also keep the active rows at their exact spacing.
        if type(dummy_tail_low) is not int or dummy_tail_low < 0:
            raise ValueError(
                "point-reflected split-gate row gap requires a non-negative "
                "dummy-tail witness"
            )
        return max(
            active_space,
            top_band + r.poly_endcap + poly_space,
            # Preserve one complete inward-bus overlap after any real
            # contact/source-metal clearance enlarges the gate bands.
            2 * top_band - max(r.poly_min_width, 180),
            (
                top_band + poly_space + dummy_tail_low
                if dummy_tail_low
                else 0
            ),
        )

    return max(
        (
            top_band                             # lower row top gate band
            + poly_space                         # clearance above the gate band
            + 2 * r.field_poly_to_comp + m1_w    # upper-row source rail + clears
            + 2 * r.field_poly_to_comp
        ),
        # A split-gate fold has a lower-row A bus extending up by
        # ``top_band`` and an upper-row B bus extending down by the same amount.
        # Keep those two different-net poly structures a full rule space apart.
        2 * top_band + poly_space,
    )


def _finger_endcaps(
    dev: str, device_a: str, device_b: str, shared_gate: bool,
    top_band: int, min_cap: int, *, row_index: int = 0,
    point_reflected_split: bool = False,
) -> Tuple[int, int]:
    """(top_extension, bottom_extension) of a finger's poly past the COMP.

    SHARED gate: every finger reaches the TOP bus (top = ``top_band``, bottom =
    minimal). SPLIT gates: member A reaches a TOP bus (top = ``top_band``, bottom
    minimal); member B reaches a BOTTOM bus (bottom = ``top_band``, top minimal).
    The minimal-cap side is kept clear of the foreign bus by construction (the bus
    is placed >= poly-space past the minimal cap)."""
    if point_reflected_split:
        if type(row_index) is not int or row_index not in (0, 1):
            raise ValueError(
                "point-reflected split-gate endcaps require row_index 0 or 1"
            )
        # Member A is outward (down in row 0, up in row 1); member B is inward.
        # The bus side therefore reflects together with the A/B row ownership.
        reaches_top = (dev == device_a) == (row_index == 1)
        return (top_band, min_cap) if reaches_top else (min_cap, top_band)
    if shared_gate or dev == device_a:
        return (top_band, min_cap)
    return (min_cap, top_band)


def _contact_rect(keys, csize, cx, ccy) -> Tuple[LayerKey, Bbox]:
    return (
        keys.contact,
        (cx - csize // 2, ccy - csize // 2, cx + csize // 2, ccy + csize // 2),
    )


def _m1_pad_rect(keys, m1_pad, cx, ccy) -> Tuple[LayerKey, Bbox]:
    half = m1_pad // 2
    return (keys.metal1, (cx - half, ccy - half, cx + half, ccy + half))


def _emit_source_rails(
    rects, keys, r, m1_w, n_rows, source_xs_per_row, row_pitch, w_nm,
    tie_xs_per_row=None,
) -> Tuple[Optional[Bbox], int]:
    """Emit ONE source rail per row below that row's COMP + per-source connectors.

    Each row's source columns common to a rail in the field band just below the
    row's COMP, with a connector from each source contact down to the rail -- the
    SAME construction as the single device. ``source_xs_per_row[ri]`` is the EXPLICIT
    list of source-column x's for row ``ri`` (derived from the built net map, so the
    rail is never mis-built over a drain column after a dummy splices the parity).
    ``tie_xs_per_row`` (optional) adds the dummy finger centre x's to the rail HULL
    (so a gate-tie contact lands on rail M1) WITHOUT a source riser there -- the rail
    stops at the dummy finger centre, short of the dummy's outer drain column.
    For a multi-row fold, one explicit source column common to every row also gets
    a vertical M1 spine between the row rails.  That turns the per-row local rails
    into the ONE declared source net; omitting this spine leaves every row above
    row 0 as an anonymous extracted source island even though the geometry has a
    single ``s`` terminal.

    Returns ``(row0_rail_bbox, min_rail_y)`` -- row 0's rail bbox (the device
    surface's single 's' terminal) and the LOWEST source-rail bottom y (so the
    split-gate bottom bus can drop clear of it)."""
    rail_gap = _source_rail_gap(r)
    m1_cut_enc = max(r.metal1_contact_enc, r.metal1_line_end)
    rail_end_half = max(
        m1_w // 2,
        r.contact_size // 2 + m1_cut_enc,
    )
    first_rail: Optional[Bbox] = None
    min_rail_y = 0
    rail_centres_y: List[int] = []
    for ri in range(n_rows):
        y0 = ri * row_pitch
        cy = y0 + w_nm // 2
        src_xs = list(source_xs_per_row[ri])
        hull_xs = list(src_xs)
        if tie_xs_per_row:
            hull_xs += list(tie_xs_per_row[ri])
        src_rail_cy = y0 - (m1_w // 2 + rail_gap)
        rail_centres_y.append(src_rail_cy)
        rail_bbox: Bbox = (
            min(hull_xs) - rail_end_half,
            src_rail_cy - m1_w // 2,
            max(hull_xs) + rail_end_half,
            src_rail_cy + m1_w // 2,
        )
        rects.append((keys.metal1, rail_bbox))
        for cx in src_xs:
            rects.append((keys.metal1, (
                cx - m1_w // 2, src_rail_cy, cx + m1_w // 2, cy,
            )))
        if ri == 0:
            first_rail = rail_bbox
        min_rail_y = min(min_rail_y, rail_bbox[1])

    if n_rows > 1:
        common_source_xs = set(source_xs_per_row[0])
        for row_xs in source_xs_per_row[1:]:
            common_source_xs.intersection_update(row_xs)
        if not common_source_xs:
            raise ValueError(
                "common-centroid source commoning requires at least one explicit "
                "source column shared by every row")
        spine_x = min(common_source_xs)
        rects.append((keys.metal1, (
            spine_x - m1_w // 2,
            min(rail_centres_y),
            spine_x + m1_w // 2,
            max(rail_centres_y),
        )))
    return first_rail, min_rail_y


def _bbox_overlap(a: Bbox, b: Bbox) -> bool:
    """True iff two integer bboxes overlap with POSITIVE area (edge-touch = clear)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _emit_flank_dummy_gate_ties(
    rects, keys, r, csize, l_nm, m1_w, m1_pad, m1_space, sd_clear, poly_space,
    min_cap, row_pitch, n_rows, sig_lo, sig_hi, n_all_cols, finger_x_all,
    gate_bus_bboxes, rail_all_source: bool = False, finger_names_per_row=None,
) -> None:
    """Tie each folded flank-dummy finger's gate DOWN to its row source rail.

    Mirrors ``device_cell.build_shared_diffusion_array``'s gate-tie recipe: a poly
    tail from the finger bottom down past the source-rail centre + a poly->M1 contact
    at the finger centre (which the widened rail M1 covers), so the dummy extracts
    with gate == source. RAISES ``ValueError`` on any clearance failure so the caller
    falls back to a free dummy rather than emit a short:

      * the widened source rail (reaching the dummy finger centre) must clear the
        dummy's OUTER drain M1 pad by the M1 min space (rail-to-drain) -- SKIPPED for
        a fully-off RAIL all-source end-cap (``rail_all_source``), whose outer column
        is the SOURCE net itself, so the rail legitimately spans it (no foreign drain
        pad to clear);
      * ``finger_space >= poly_space`` (the edge dummy sits a full finger pitch
        outside the interior signal fingers / buses);
      * neither the poly tail nor the contact may intersect any gate bus."""
    finger_space = r.finger_space
    poly_contact_enc = r.poly_contact_enc
    rail_gap = _source_rail_gap(r)
    tail_half = max(l_nm // 2, csize // 2 + poly_contact_enc)

    # rail-to-drain clearance: the rail edge reaches the finger centre (gx); the
    # dummy's outer drain column M1 pad centre is (l_nm//2 + sd_clear + csize//2)
    # beyond gx. Refuse if the rail M1 edge would come within M1 space of the pad.
    # A fully-off RAIL end-cap has NO foreign drain pad (its outer column is the
    # source rail net), so the check does not apply.
    rail_drain_gap = l_nm // 2 + sd_clear + csize // 2 - m1_w // 2 - m1_pad // 2
    if not rail_all_source and rail_drain_gap < m1_space:
        raise ValueError(
            "flank-dummy source-rail extension would violate M1 space to the dummy "
            f"drain pad (gap {rail_drain_gap} < m1_space {m1_space}); refusing fold")
    if finger_space < poly_space:
        raise ValueError(
            "flank-dummy end-cap needs finger_space >= poly_space "
            f"({finger_space} < {poly_space}); refusing fold")

    for ri in range(n_rows):
        y0 = ri * row_pitch
        src_rail_cy = y0 - (m1_w // 2 + rail_gap)
        finger_bot = y0 - min_cap
        for c in range(n_all_cols):
            if sig_lo <= c < sig_hi:
                continue  # a signal finger
            if finger_names_per_row is not None:
                select_native_owners(rects, (finger_names_per_row[ri][c],), 'dummy_gate_tie')
            gx = finger_x_all[c] + l_nm // 2
            tail = (
                gx - tail_half,
                src_rail_cy - csize // 2 - poly_contact_enc,
                gx + tail_half,
                finger_bot,
            )
            cut = (gx - csize // 2, src_rail_cy - csize // 2,
                   gx + csize // 2, src_rail_cy + csize // 2)
            for bb in gate_bus_bboxes:
                if _bbox_overlap(tail, bb) or _bbox_overlap(cut, bb):
                    raise ValueError(
                        "flank-dummy gate tie intersects a gate bus; refusing fold")
            rects.append((keys.poly, tail))
            rects.append((keys.contact, cut))


def _emit_gate_routing(
    rects, keys, r, csize, m1_pad, bus_band, m1_w, m1_space, w_nm, l_nm, finger_x,
    poly_space, row_owner_labels, owner_label, device_a, device_b, shared_gate,
    n_rows, row_pitch, top_band, min_cap, min_src_rail_y,
    point_reflected_split, comp_x_span,
) -> Tuple[List[Bbox], Optional[Bbox], Dict[str, Bbox]]:
    """Emit the gate bus(es) + gate contact(s), DRC-clean by the single-device
    recipe.

    For EACH row, the device's fingers in that row strap to a per-row bus placed in
    that row's gate band -- ABOVE the row for member A / a shared gate, BELOW the row
    for member B in the SPLIT-gate case -- so the two members' buses are on OPPOSITE
    sides and never violate poly spacing. A 2-row split-gate array's per-row buses
    are tied through rule-spaced field-poly spines OUTSIDE the COMP span (member A
    left, member B right), with a SINGLE gate contact on the outermost bus.
    Reflection swaps member columns between rows, so an in-array constant-X spine
    would cross the other member's fingers and create a third, unlabelled
    half-A/half-B gate. Shared-gate arrays retain the compact in-array spine because
    all crossed fingers intentionally share one gate net.

    The finger polys already carry the per-device endcap (member A / shared reach UP
    by ``top_band``, member B reaches DOWN by ``top_band``; the OPPOSITE side is the
    minimal end-cap ``min_cap``). Each bus OVERLAPS its members' reaching endcaps and
    is kept >= poly-space clear of the FOREIGN device's minimal end-cap (so a bus
    that spans across foreign columns is clear of those foreign fingers). The gate
    CONTACT is a poly pad RAISED past the OUTERMOST bus (the single-device
    construction) so the contact is fully poly-enclosed (CO.3) + clear of the COMP
    (CO.8) and never on field (CO.11)."""
    poly_contact_enc = r.poly_contact_enc

    # Per-device finger COLUMNS in each row (the column index list, per row). The
    # ``__all__`` pseudo-device collects EVERY column per row (the shared-gate case,
    # where one structure spans both members' fingers).
    cols_in_row: Dict[str, List[List[int]]] = {
        device_a: [], device_b: [], "__all__": []}
    for labels in row_owner_labels:
        per: Dict[str, List[int]] = {device_a: [], device_b: []}
        for c, lab in enumerate(labels):
            per[owner_label[lab]].append(c)
        for dev in (device_a, device_b):
            cols_in_row[dev].append(per[dev])
        cols_in_row["__all__"].append(list(range(len(labels))))

    gate_bus_bboxes: List[Bbox] = []

    def _row_top_bus(ri, cols) -> Bbox:
        """A bus ABOVE row ``ri`` (member A / shared)."""
        y1 = ri * row_pitch + w_nm
        by1 = y1 + top_band
        # >= poly-space above the FOREIGN device's minimal end-cap (at y1+min_cap).
        by0 = max(y1 + min_cap + poly_space, by1 - bus_band)
        bx0 = finger_x[cols[0]]
        bx1 = finger_x[cols[-1]] + l_nm
        rects.append((keys.poly, (bx0, by0, bx1, by1)))
        return (bx0, by0, bx1, by1)

    def _row_bot_bus(ri, cols) -> Bbox:
        """A bus BELOW row ``ri`` (member B, split-gate)."""
        y0 = ri * row_pitch
        by0 = y0 - top_band
        by1 = min(y0 - min_cap - poly_space, by0 + bus_band)
        bx0 = finger_x[cols[0]]
        bx1 = finger_x[cols[-1]] + l_nm
        rects.append((keys.poly, (bx0, by0, bx1, by1)))
        return (bx0, by0, bx1, by1)

    def _route_device(dev, side) -> Optional[Bbox]:
        """Route ``dev``'s gate across all rows on ``side`` ('up'/'down'); tie the
        per-row buses with one exact poly spine; place ONE gate contact on the
        outermost bus. Returns the contact bbox."""
        select_native_owners(rects, (device_a, device_b) if dev == '__all__' else (dev,), 'signal_gate_routing')
        row_buses: List[Bbox] = []
        for ri in range(n_rows):
            cols = cols_in_row[dev][ri]
            if not cols:
                continue
            box = _row_top_bus(ri, cols) if side == "up" else _row_bot_bus(ri, cols)
            row_buses.append(box)
            gate_bus_bboxes.append(box)
        if not row_buses:
            return None
        if shared_gate or n_rows == 1:
            # Compact historical construction: safe because each row uses the
            # same gate net (or there is no second reflected row to cross).
            spine_x0 = min(b[0] for b in row_buses)
            spine_x1 = spine_x0 + max(l_nm, r.poly_min_width)
        else:
            # Split gates use opposite outside-COMP lanes.  Horizontal row-bus
            # extensions live in their already rule-spaced field bands.
            spine_w = max(bus_band, r.poly_min_width)
            comp_left, comp_right = int(comp_x_span[0]), int(comp_x_span[1])
            if side == "up":
                spine_x1 = comp_left - poly_space
                spine_x0 = spine_x1 - spine_w
                for bx0, by0, _bx1, by1 in row_buses:
                    rects.append((keys.poly, (spine_x0, by0, bx0, by1)))
            else:
                spine_x0 = comp_right + poly_space
                spine_x1 = spine_x0 + spine_w
                for _bx0, by0, bx1, by1 in row_buses:
                    rects.append((keys.poly, (bx1, by0, spine_x1, by1)))
        sy0 = min(b[1] for b in row_buses)
        sy1 = max(b[3] for b in row_buses)
        if n_rows > 1:
            rects.append((keys.poly, (spine_x0, sy0, spine_x1, sy1)))
        # gate contact on the OUTERMOST bus (top-most for 'up', bottom-most for
        # 'down') so the contact pad is clear of the array.
        if side == "up":
            outer = max(row_buses, key=lambda b: b[3])
            _emit_gate_contact(rects, keys, r, csize, m1_pad, poly_contact_enc,
                               outer[0], l_nm, outer[1], outer[3], side="up")
        else:
            outer = min(row_buses, key=lambda b: b[1])
            # the contact's M1 pad must clear the SOURCE rail (the lowest rail) by
            # the M1 min space: drop the contact so its M1-pad TOP is m1_space below
            # the source rail bottom. The poly pad is extended down to keep the
            # contact poly-enclosed even at the lowered position.
            m1_pad_top_max = min_src_rail_y - m1_space
            _emit_gate_contact(rects, keys, r, csize, m1_pad, poly_contact_enc,
                               outer[0], l_nm, outer[1], outer[3], side="down",
                               m1_pad_top_max=m1_pad_top_max)
        return _last_gate_contact_box(rects, keys, csize)

    def _route_point_reflected_device(
        dev: str,
        row_sides: Tuple[str, str],
        *,
        spine_side: str,
        contact_side: str,
    ) -> Optional[Bbox]:
        """Route one member with its bus side reflected between the two rows.

        Outward-facing row buses stay on one outside-COMP POLY spine.  The two
        inward-facing buses share their exact overlap band in the inter-row field;
        they join there and place their contact inside that rule-proven field rather
        than expanding the rigid macro in x.
        """
        if type(row_sides) is not tuple or len(row_sides) != 2:
            raise ValueError("point-reflected gate sides must be an exact 2-tuple")
        if any(type(side) is not str or side not in ("up", "down") for side in row_sides):
            raise ValueError("point-reflected gate side must be 'up' or 'down'")
        if spine_side not in ("left", "right", "center") or contact_side not in (
            "up",
            "down",
        ):
            raise ValueError("invalid point-reflected gate routing side")

        select_native_owners(rects, (dev,), 'signal_gate_routing')
        row_buses: List[Tuple[str, Bbox]] = []
        for ri, side in enumerate(row_sides):
            cols = cols_in_row[dev][ri]
            if not cols:
                continue
            box = _row_top_bus(ri, cols) if side == "up" else _row_bot_bus(ri, cols)
            row_buses.append((side, box))
            gate_bus_bboxes.append(box)
        if len(row_buses) != 2:
            raise ValueError("point-reflected split gate must occupy both rows")

        if spine_side == "center":
            if row_sides != ("up", "down") or contact_side != "down":
                raise ValueError(
                    "centre-field gate join requires inward-facing row buses"
                )
            lower_bus = row_buses[0][1]
            upper_bus = row_buses[1][1]
            overlap_y0 = max(lower_bus[1], upper_bus[1])
            overlap_y1 = min(lower_bus[3], upper_bus[3])
            if overlap_y1 - overlap_y0 < max(bus_band, r.poly_min_width):
                raise ValueError(
                    "centre-field gate buses lack one rule-width overlap band"
                )
            bridge = (
                min(lower_bus[0], upper_bus[0]),
                overlap_y0,
                max(lower_bus[2], upper_bus[2]),
                overlap_y1,
            )
            rects.append((keys.poly, bridge))
            gate_bus_bboxes.append(bridge)
            _emit_gate_contact(
                rects,
                keys,
                r,
                csize,
                m1_pad,
                poly_contact_enc,
                lower_bus[0],
                l_nm,
                lower_bus[1],
                lower_bus[3],
                side="down",
            )
            return _last_gate_contact_box(rects, keys, csize)

        spine_w = max(bus_band, r.poly_min_width)
        comp_left, comp_right = int(comp_x_span[0]), int(comp_x_span[1])
        if spine_side == "left":
            spine_x1 = comp_left - poly_space
            spine_x0 = spine_x1 - spine_w
            for _side, (bx0, by0, _bx1, by1) in row_buses:
                rects.append((keys.poly, (spine_x0, by0, bx0, by1)))
        else:
            spine_x0 = comp_right + poly_space
            spine_x1 = spine_x0 + spine_w
            for _side, (_bx0, by0, bx1, by1) in row_buses:
                rects.append((keys.poly, (bx1, by0, spine_x1, by1)))

        outer_y = (
            (n_rows - 1) * row_pitch + w_nm + top_band
            if contact_side == "up"
            else -top_band
        )
        candidates = [
            box
            for side, box in row_buses
            if side == contact_side
            and (
                (contact_side == "up" and box[3] == outer_y)
                or (contact_side == "down" and box[1] == outer_y)
            )
        ]
        all_buses = [box for _side, box in row_buses]
        if candidates:
            contact_bus = candidates[0]
            contact_x0 = contact_bus[0]
            contact_l = l_nm
        else:
            # The inward-facing member has no outer row bus.  Add a POLY-only
            # landing on its existing outside spine.  It is outside the COMP x-span
            # and therefore cannot cross the other member's gate fingers.
            if contact_side == "up":
                stub = (
                    spine_x0,
                    outer_y - bus_band,
                    spine_x1,
                    outer_y,
                )
            else:
                stub = (
                    spine_x0,
                    outer_y,
                    spine_x1,
                    outer_y + bus_band,
                )
            rects.append((keys.poly, stub))
            gate_bus_bboxes.append(stub)
            all_buses.append(stub)
            contact_bus = stub
            contact_x0 = spine_x0
            contact_l = spine_x1 - spine_x0

        rects.append((
            keys.poly,
            (
                spine_x0,
                min(box[1] for box in all_buses),
                spine_x1,
                max(box[3] for box in all_buses),
            ),
        ))
        if contact_side == "up":
            _emit_gate_contact(
                rects,
                keys,
                r,
                csize,
                m1_pad,
                poly_contact_enc,
                contact_x0,
                contact_l,
                contact_bus[1],
                contact_bus[3],
                side="up",
            )
        else:
            _emit_gate_contact(
                rects,
                keys,
                r,
                csize,
                m1_pad,
                poly_contact_enc,
                contact_x0,
                contact_l,
                contact_bus[1],
                contact_bus[3],
                side="down",
                m1_pad_top_max=min_src_rail_y - m1_space,
            )
        return _last_gate_contact_box(rects, keys, csize)

    gate_contact_bbox: Optional[Bbox] = None
    # SPLIT-gate pairs have TWO gate contacts (member A up-bus, member B down-bus) on
    # OPPOSITE sides; ``gate_contacts`` maps each member to its OWN contact so a
    # diff-pair member (VIP vs VIN) exposes its own gate landing. SHARED-gate pairs
    # (a mirror: both gates the diode node) keep ONE contact -> empty map, both
    # members fall back to the single shared contact (byte-identical).
    gate_contacts: Dict[str, Bbox] = {}
    if shared_gate:
        # one shared gate: route BOTH members' fingers UP to per-row top buses, tied
        # by one spine + one contact. (Both members share the gate net, so a single
        # routed structure spans every finger.)
        gate_contact_bbox = _route_device("__all__", "up")
        # __all__ is handled by treating every column as one device's set:
    elif point_reflected_split:
        gate_contact_bbox = _route_point_reflected_device(
            device_a,
            ("down", "up"),
            spine_side="left",
            contact_side="up",
        )
        gate_contact_b = _route_point_reflected_device(
            device_b,
            ("up", "down"),
            spine_side="center",
            contact_side="down",
        )
        if gate_contact_bbox is not None:
            gate_contacts[device_a] = gate_contact_bbox
        if gate_contact_b is not None:
            gate_contacts[device_b] = gate_contact_b
    else:
        gate_contact_bbox = _route_device(device_a, "up")
        gate_contact_b = _route_device(device_b, "down")
        if gate_contact_bbox is not None:
            gate_contacts[device_a] = gate_contact_bbox
        if gate_contact_b is not None:
            gate_contacts[device_b] = gate_contact_b

    return gate_bus_bboxes, gate_contact_bbox, gate_contacts


def _emit_gate_contact(
    rects, keys, r, csize, m1_pad, poly_contact_enc, finger_x0, l_nm,
    by0, by1, *, side: str, m1_pad_top_max: Optional[int] = None,
):
    """A gate contact: a poly pad RAISED past the bus on ``side`` ('up'/'down') so
    the contact is fully poly-enclosed (CO.3) + clear of the COMP (CO.8/CO.11),
    plus the cut + its M1 pad. The single-device construction exactly.

    ``m1_pad_top_max`` (side='down' only): force the contact LOW enough that its M1
    pad TOP is <= this y (so the pad clears a rail above it by the caller's margin).
    The poly pad is extended to keep the contact poly-enclosed at the lowered y."""
    gate_cx = finger_x0 + l_nm // 2
    gpad_half = max(m1_pad // 2, csize // 2 + poly_contact_enc)
    if side == "up":
        gate_cy = by1 + poly_contact_enc + csize // 2
        pad_y0, pad_y1 = by0, gate_cy + csize // 2 + poly_contact_enc
    else:
        gate_cy = by0 - poly_contact_enc - csize // 2
        if m1_pad_top_max is not None:
            # lower the contact so its M1 pad TOP (gate_cy + m1_pad//2) <= the cap.
            gate_cy = min(gate_cy, m1_pad_top_max - m1_pad // 2)
        # the poly pad spans from the bus top (by1) down past the (lowered) contact.
        pad_y0, pad_y1 = gate_cy - csize // 2 - poly_contact_enc, by1
    rects.append((keys.poly, (gate_cx - gpad_half, pad_y0, gate_cx + gpad_half, pad_y1)))
    gc_box = (
        gate_cx - csize // 2,
        gate_cy - csize // 2,
        gate_cx + csize // 2,
        gate_cy + csize // 2,
    )
    rects.append((keys.contact, gc_box))
    rects.append(_m1_pad_rect(keys, m1_pad, gate_cx, gate_cy))


def _last_gate_contact_box(rects, keys, csize) -> Optional[Bbox]:
    """The contact bbox of the gate contact just emitted (the last contact rect on
    the contact key whose size is exactly csize)."""
    for key, box in reversed(rects):
        if key == keys.contact and (box[2] - box[0]) == csize and (box[3] - box[1]) == csize:
            return box
    return None


def _emit_drain_rails(
    rects, keys, r, m1_w, m1_space, gate_m1_top, m2_key, via1_key, via1_size,
    via1_enc_below, via1_enc_above, m2_w, m2_space, drn_net, device_a, device_b,
    drain_cols_xy_by_net, top_of_buses, comp_x_span=None, owners_by_net=None,
) -> Tuple[Optional[Bbox], Dict[str, Bbox]]:
    """Emit one commoning rail per drain net above the gate buses.

    With TWO interleaved drain nets a single metal layer cannot common both with
    straight connectors: the drains alternate in x, so a dB drain column sits inside
    the dA-rail x-hull and a straight dB connector would cross the dA M1 rail (a
    different-net short). Both drain rails are kept on M1 (stacked, >= M1 min space
    apart); where an UPPER-rail connector would cross a LOWER rail's M1 body it HOPS
    OVER on a LOCAL M2 jumper (via1 up below the rail, a short M2 segment over the
    rail band, via1 down above it), then continues on M1 to its rail. The jumper is
    the ONLY M2 the cluster emits and it is LOCAL (it never reaches the BEOL routing
    band), so it cannot collide with the router's M2. The interleaved drains are thus
    commoned correct-by-construction with no crossing short. ``device_a``'s rail (the
    lowest, all-M1) is the device surface's single 'd' terminal.
    Returns ``(dA_rail_bbox, {net: rail_bbox})``."""
    first_rail: Optional[Bbox] = None
    rails_by_net: Dict[str, Bbox] = {}
    base_y = max(
        top_of_buses + r.field_poly_to_comp + m1_w // 2 + 50,
        gate_m1_top + m1_space + m1_w // 2,
    )
    ordered_nets: List[str] = []
    for net in (drn_net.get(device_a, ""), drn_net.get(device_b, "")):
        if net and net not in ordered_nets:
            ordered_nets.append(net)

    rail_pitch = m1_w + m1_space + 50
    rail_cys = [base_y + ri * rail_pitch for ri in range(len(ordered_nets))]
    lower_rail_bands: List[Bbox] = []  # M1 rails already emitted (to hop over)

    h = via1_size // 2
    e_below = via1_size // 2 + via1_enc_below
    e_above = via1_size // 2 + via1_enc_above

    # A ROW-REFLECTED fold coincides a LOWER-row drain column in x with an UPPER-row
    # drain of the OTHER net (nsig row0 & ndio row1 both at x=2235): a straight M1
    # riser from the lower drain up to its rail would cross the upper foreign drain's
    # M1 pad AND the upper row's full-width source rail -> two distinct drains + the
    # source merged onto ONE M1 island (a real cross-net LVS short the same-layer OD
    # oracle misses). Route such a lower-row drain on a LOCAL M2 riser JOGGED laterally
    # to a clear column: its only M1 is the contact pad (bottom) and the rail landing
    # (top); the M2 crosses every foreign M1 (source rail, foreign drain) on a
    # DIFFERENT layer, so nothing merges. Detection + jog are purely geometric (a
    # foreign-net drain contact at the SAME x in a HIGHER row) -> byte-identical for a
    # 1-row array (no coincidence) and for the top-row drains (nothing foreign above).
    csize = r.contact_size
    m1_cut_enc = max(r.metal1_contact_enc, r.metal1_line_end)
    m1_pad = _m1_contact_pad_side(r, csize, m1_cut_enc)
    all_drain_contacts = [
        (x, y, n) for n, cs in drain_cols_xy_by_net.items() for (x, y) in cs]
    # a straight m1_w riser overlaps a foreign m1_pad within this centre-to-centre x.
    x_tol = (m1_w + m1_pad) // 2
    # the minimum lateral jog that clears the foreign drain pad by M1 min-space (plus
    # M2 pitch so the jogged riser also clears a foreign M2 hop at the shared column).
    jog_min = m1_pad // 2 + m1_w // 2 + m1_space
    jog_dx = jog_min + m2_space

    def _foreign_drain_above(cx, cyy, net) -> bool:
        return any(
            n2 != net and abs(x2 - cx) < x_tol and y2 > cyy
            for (x2, y2, n2) in all_drain_contacts)

    # the drain rail may EXTEND to reach a jogged riser column; it stays inside the
    # union COMP x-span (enclosed by the well / source-rail hull) so the tile footprint
    # is unchanged. Fall back to the drain columns' own hull when no COMP span given.
    if comp_x_span is not None:
        comp_lo, comp_hi = int(comp_x_span[0]), int(comp_x_span[1])
    else:
        _all_x = [x for cs in drain_cols_xy_by_net.values() for (x, _y) in cs]
        comp_lo, comp_hi = (min(_all_x), max(_all_x)) if _all_x else (0, 0)

    def _jog_target(cx, net) -> Optional[int]:
        """A clear column x to carry the local M2 riser for a reflected drain: offset
        from ``cx`` (toward the COMP centre first) far enough to clear the coincident
        foreign drain, clear of EVERY foreign drain column, and inside the COMP span so
        the (possibly extended) rail stays in the tile. Returns ``None`` when no such
        column fits -- the fold is too tight to isolate the reflected drains without
        growing the tile (caller raises, never a silent short)."""
        side_clearance = max(e_below, e_above)
        lo, hi = comp_lo + side_clearance, comp_hi - side_clearance
        foreign_xs = [x2 for (x2, _y2, n2) in all_drain_contacts if n2 != net]
        center = (comp_lo + comp_hi) // 2
        prefer = 1 if cx <= center else -1
        for mult in (1.0, 1.5, 2.0, 2.5, 3.0):
            d = int(jog_dx * mult)
            for sign in (prefer, -prefer):
                xj = cx + sign * d
                if (lo <= xj <= hi and abs(xj - cx) >= jog_min
                        and all(abs(xj - fx) >= jog_min for fx in foreign_xs)):
                    return xj
        # The finite offset samples are a fast path, not an infeasibility
        # proof.  Short-channel reflected rows can leave a legal interval
        # between foreign columns that none of those offsets visits.  Its
        # nearest feasible point must be a COMP boundary or an exclusion
        # boundary, so enumerate those exact integer-nanometre candidates.
        boundaries = {lo, hi, cx - jog_min, cx + jog_min}
        for fx in foreign_xs:
            boundaries.update((fx - jog_min, fx + jog_min))
        for xj in sorted(boundaries, key=lambda x: (abs(x - cx), (x - cx) * prefer < 0, x)):
            if (lo <= xj <= hi and abs(xj - cx) >= jog_min
                    and all(abs(xj - fx) >= jog_min for fx in foreign_xs)):
                return xj
        return None

    def _drain_riser_m2(cx, cyy, xj, drn_rail_cy):
        """Local M2 riser for a reflected lower-row drain: via1 UP on the drain's own
        M1 contact pad, M2 jog to the clear column ``xj`` then M2 up, via1 DOWN onto
        this net's rail. The landing is kept a full M1 space above any LOWER drain rail
        at ``xj`` (exactly ``_jumper_over``'s upper leg), so it never pokes above the
        tile -- footprint unchanged."""
        # lower drain rails already emitted that this column crosses (the UPPER net):
        # land the via1 a full M1 space ABOVE the highest one, then M1-strip up into
        # this net's own rail. The LOWER net has none -> land directly on its rail.
        lower_here = [b for b in lower_rail_bands if b[0] <= xj <= b[2]]
        land_y = (max(b[3] for b in lower_here) + m1_space + e_below) if lower_here \
            else drn_rail_cy
        lo_x, hi_x = (cx, xj) if cx <= xj else (xj, cx)
        # bottom via1 (metal1->metal2) on the existing drain M1 pad.
        rects.append((via1_key, (cx - h, cyy - h, cx + h, cyy + h)))
        rects.append((keys.metal1, (
            cx - e_below, cyy - e_below, cx + e_below, cyy + e_below)))
        # M2 jog across the contact band, then the M2 riser at the clear column.
        rects.append((m2_key, (lo_x - m2_w // 2, cyy - m2_w // 2,
                               hi_x + m2_w // 2, cyy + m2_w // 2)))
        rects.append((m2_key, (
            xj - m2_w // 2, cyy - m2_w // 2,
            xj + m2_w // 2, land_y + e_above)))
        # top via1 (metal2->metal1) + M1 strip up into the rail (same net -> merges).
        rects.append((via1_key, (xj - h, land_y - h, xj + h, land_y + h)))
        rects.append((keys.metal1, (
            xj - e_below, land_y - e_below,
            xj + e_below, land_y + e_below)))
        if land_y < drn_rail_cy:
            rects.append(
                (keys.metal1, (xj - m1_w // 2, land_y, xj + m1_w // 2, drn_rail_cy)))

    def _jumper_over(cx, y_from, y_to, band):
        """Vertical connector at ``cx`` from ``y_from`` to ``y_to``, HOPPING the M1
        ``band`` (x0,y0,x1,y1) on a local M2 jumper (via1 below, M2 over, via1
        above)."""
        bx0, by0, bx1, by1 = band
        # the via1's M1 landing half-height must clear the band by m1_space, so
        # the via1 centre sits e_below + m1_space beyond the band edge.
        v_lo_cy = by0 - m1_space - e_below
        v_hi_cy = by1 + m1_space + e_below
        # M1 legs (from source up into the lower via1 landing; from the upper via1
        # landing up to the rail).
        rects.append((keys.metal1, (cx - m1_w // 2, y_from, cx + m1_w // 2, v_lo_cy)))
        rects.append((keys.metal1, (cx - m1_w // 2, v_hi_cy, cx + m1_w // 2, y_to)))
        # via1 cuts + M1/M2 landings at each end.
        for vcy in (v_lo_cy, v_hi_cy):
            rects.append((via1_key, (cx - h, vcy - h, cx + h, vcy + h)))
            rects.append((keys.metal1, (
                cx - e_below, vcy - e_below,
                cx + e_below, vcy + e_below)))
        # the M2 segment spanning the band (overlaps both via1 M2 landings).
        rects.append((m2_key, (
            cx - m2_w // 2, v_lo_cy - e_above,
            cx + m2_w // 2, v_hi_cy + e_above)))

    for ri, net in enumerate(ordered_nets):
        owners = (owners_by_net or {}).get(net, tuple(member for member, value in drn_net.items() if value == net))
        select_native_owners(rects, owners, 'drain_net_routing')
        cols = drain_cols_xy_by_net.get(net, [])
        if not cols:
            continue
        drn_rail_cy = rail_cys[ri]
        drn_xs = [x for (x, _y) in cols]
        # pre-resolve the jog column of every reflected lower-row drain so the rail
        # hull can be widened to reach the jogged landings (the landing sits ON the
        # rail body). No coincidence -> empty -> byte-identical rail + connectors.
        iso: Dict[Tuple[int, int], int] = {}
        for (cx, cyy) in cols:
            if _foreign_drain_above(cx, cyy, net):
                xj = _jog_target(cx, net)
                if xj is None:
                    raise ValueError(
                        "common-centroid drain isolation infeasible: no clear jog "
                        f"column fits inside the COMP span for net {net!r} at "
                        f"x={cx} (min-space too tight to isolate the reflected drains "
                        "without growing the tile)")
                iso[(cx, cyy)] = xj
        hull_xs = drn_xs + list(iso.values())
        rbbox: Bbox = (
            min(hull_xs) - m1_w // 2, drn_rail_cy - m1_w // 2,
            max(hull_xs) + m1_w // 2, drn_rail_cy + m1_w // 2,
        )
        rects.append((keys.metal1, rbbox))
        for (cx, cyy) in cols:
            if (cx, cyy) in iso:
                # a reflected lower-row drain: isolate it on a jogged local M2 riser.
                _drain_riser_m2(cx, cyy, iso[(cx, cyy)], drn_rail_cy)
                continue
            crossed = [b for b in lower_rail_bands if b[0] <= cx <= b[2]]
            if not crossed:
                rects.append(
                    (keys.metal1, (cx - m1_w // 2, cyy, cx + m1_w // 2, drn_rail_cy)))
            else:
                _jumper_over(cx, cyy, drn_rail_cy,
                             max(crossed, key=lambda b: b[3]))
        rails_by_net[net] = rbbox
        lower_rail_bands.append(rbbox)
        if ri == 0:
            first_rail = rbbox
    return first_rail, rails_by_net


def _tie_gate_to_drain(rects, keys, m1_w, gate_contact_bbox, drain_rail_bbox):
    """Strap a diode-connected device's gate-contact M1 pad to its drain rail with an
    L-shaped M1 connector (same net), so the array extracts gate==drain internally
    -- no router bridge between two coincident-net cluster terminals. The L rises in
    the gate-contact column to the drain-rail level, then jogs horizontally to the
    rail. Both legs are the SAME net (gate==drain) so the merge is never a fabricated
    short."""
    gx0, gy0, gx1, gy1 = gate_contact_bbox
    gcx = (gx0 + gx1) // 2
    d_y_lo = min(drain_rail_bbox[1], drain_rail_bbox[3])
    d_y_hi = max(drain_rail_bbox[1], drain_rail_bbox[3])
    d_cy = (d_y_lo + d_y_hi) // 2
    # vertical leg in the gate-contact column from the pad up to the rail level.
    rects.append((keys.metal1, (gcx - m1_w // 2, min(gy0, d_y_lo),
                                gcx + m1_w // 2, max(gy1, d_y_hi))))
    # horizontal leg at the rail level from the gate column to the rail's near edge.
    rx0, rx1 = drain_rail_bbox[0], drain_rail_bbox[2]
    hx0 = min(gcx - m1_w // 2, rx0)
    hx1 = max(gcx + m1_w // 2, rx1) if gcx > rx1 else max(rx0, gcx + m1_w // 2)
    # clamp the horizontal leg to reach the nearest rail edge without overshoot.
    if gcx < rx0:
        hx0, hx1 = gcx - m1_w // 2, rx0 + m1_w
    elif gcx > rx1:
        hx0, hx1 = rx1 - m1_w, gcx + m1_w // 2
    else:
        hx0, hx1 = gcx - m1_w // 2, gcx + m1_w // 2  # already over the rail
    rects.append((keys.metal1, (hx0, d_cy - m1_w // 2, hx1, d_cy + m1_w // 2)))


__all__ = [
    "ClusterInfo",
    "build_interdigitated_cluster",
    "interleave_sequence",
    "is_interdigitatable",
]
