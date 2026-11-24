"""REFERENCE SPATIAL-INTENT extractor -- abstract relations, ZERO coordinate leak.

WHAT THIS IS
============
:mod:`clay.frontend.floorplan_intent` derives the LDO floorplan from SCHEMATIC
ROLES ONLY -- it never looks at how a reference layout actually ARRANGED the
device. As a result the migrated cell bears no relation to the reference style.

This module closes that gap for the SPATIAL half: it reads a reference layout's
device placements (the SREF instance positions grouped BY MASTER TYPE) and the
netlist device sizes, and derives the ABSTRACT ARRANGEMENT the reference realised
-- which functional block sits left / right / on top, which passive cluster is a
compact BANK, which device is the one wide PASS block -- as RELATIONS ONLY.

It is the spatial sibling of :func:`clay.layout.extract.geometric_intent.\
extract_geometric_intent` (the mature OTA per-device extractor). Where that one
runs at per-device granularity on a placement view, this one runs at FUNCTIONAL-
BLOCK (master-type) granularity on a reference GDS, because the reference SREF
instances are anonymous (one ``p33`` master placed 65 times) and carry no map to
individual netlist instance names -- the robust, leak-safe signal is the
per-master-type REGION centroid / extent, not a per-instance coordinate.

DISCIPLINE -- NO RAW COORDINATE ON THE RESULT
=============================================
Region centroids and extents ARE read from the reference, but they live only as
LOCAL variables inside the derivation. The returned :class:`ReferenceSpatialIntent`
carries ONLY abstract relations:

  * ``region_tier``   {region -> tier ordinal}  (0 = bottom, ascending = up)
  * ``region_column`` {region -> column ordinal} (0 = leftmost, ascending = right)
  * ``left_of``       [(a, b)]  region a is left of region b
  * ``above``         [(a, b)]  region a is above region b
  * ``adjacency``     [(a, b)]  region bounding boxes are edge-adjacent
  * ``banked``        {region}  a compact passive BANK (resistor / cap cluster)
  * ``wide_block``    {region}  holds the one wide PASS device (>> median W*M)
  * structure hints   master/body-domain rail affinity and explicit extraction gaps

Ordinals and pairwise relations only -- never an absolute x / y / centroid. The
leak audit (:func:`clay.frontend.clay_emit_compactness_check.\
assert_no_reference_coord_leak`) on the emitted design.clay therefore stays GREEN.

NO PER-CIRCUIT BRANCH: every rule keys off MASTER TYPE / device size, applied
uniformly. The same extractor runs for any reference GDS + netlist pair.
"""
from __future__ import annotations

import re
import statistics
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "REFERENCE_STRUCTURE_PRIMITIVES",
    "ReferenceSpatialIntent",
    "ReferenceStructureHint",
    "extract_reference_spatial_intent",
    "device_master_map",
    "reference_spatial_intent_from_data",
    "require_reference_structure_primitives",
]

# Device MASTER families a reference analog layout banks by. Each is a SCHEMATIC
# property (the PDK device master), not a position. Ordered longest-first is not
# needed (they are disjoint prefixes), but listed canonically for determinism.
_DEVICE_MASTERS: Tuple[str, ...] = (
    "p33",
    "n33",
    "p18",
    "n18",
    "npolyf_u",
    "ppolyf_u",
    "mim",
    "pnp",
    "npn",
)

# Passive master families -> a compact BANK (the resistor / cap cluster).
_PASSIVE_MASTERS: frozenset = frozenset(
    {"npolyf_u", "ppolyf_u", "mim", "resistor", "capacitor"}
)

REFERENCE_STRUCTURE_PRIMITIVES: Tuple[str, ...] = (
    "bank",
    "passive_bank_partition",
    "well_island_sharing",
    "finger_array_order",
    "density_whitespace_target",
    "od_fill_target",
    "channel",
    "rail_affinity",
    "power_sensitive_net_budget",
    "thermal_stress_neighborhood",
)
_REFERENCE_ONLY_BLOCKING_PRIMITIVES = frozenset({
    "passive_bank_partition",
    "finger_array_order",
    "density_whitespace_target",
    "od_fill_target",
    "channel",
    "power_sensitive_net_budget",
    "thermal_stress_neighborhood",
})
_HINT_ID_RE = re.compile(r"[a-z][a-z0-9_.]*\Z")
_MAX_HINTS = 256
_MAX_HINT_SUBJECTS = 64
_MAX_HINT_TEXT = 4_096


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class ReferenceStructureHint:
    """One coordinate-free structure observation or explicit extraction gap."""

    hint_id: str
    primitive: str
    subjects: Tuple[str, ...]
    status: Literal[
        "identified", "topology_declared", "unmeasurable", "unsupported"
    ]
    gap: str | None = None

    def __post_init__(self) -> None:
        self.assert_authoritative()

    def assert_authoritative(self) -> None:
        """Replay the exact hint schema after any hostile object mutation."""

        if (
            type(self.hint_id) is not str
            or _HINT_ID_RE.fullmatch(self.hint_id) is None
            or len(self.hint_id) > _MAX_HINT_TEXT
        ):
            raise ValueError("reference structure hint id is not canonical")
        if (
            type(self.primitive) is not str
            or self.primitive not in REFERENCE_STRUCTURE_PRIMITIVES
        ):
            raise ValueError("reference structure hint names an unmapped primitive")
        if (
            type(self.subjects) is not tuple
            or not self.subjects
            or len(self.subjects) > _MAX_HINT_SUBJECTS
            or any(
                type(item) is not str
                or not item
                or item != item.strip()
                or len(item) > _MAX_HINT_TEXT
                for item in self.subjects
            )
            or tuple(sorted(set(self.subjects))) != self.subjects
        ):
            raise ValueError(
                "reference structure hint subjects must be an exact canonical tuple"
            )
        if type(self.status) is not str or self.status not in {
            "identified",
            "topology_declared",
            "unmeasurable",
            "unsupported",
        }:
            raise ValueError("reference structure hint status is not canonical")
        if self.status in {"unmeasurable", "unsupported"}:
            if (
                type(self.gap) is not str
                or not self.gap
                or self.gap != self.gap.strip()
                or len(self.gap) > _MAX_HINT_TEXT
            ):
                raise ValueError("blocking reference structure hint requires a gap")
        elif self.gap is not None:
            raise ValueError("measurable reference structure hint cannot retain a gap")

    @property
    def blocking(self) -> bool:
        return self.status in {"unmeasurable", "unsupported"}

    def canonical_data(self) -> Dict[str, object]:
        self.assert_authoritative()
        return {
            "hint_id": self.hint_id,
            "primitive": self.primitive,
            "subjects": list(self.subjects),
            "status": self.status,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class ReferenceSpatialIntent:
    """The reference layout's ABSTRACT arrangement -- relations only, no coordinate.

    Every field is an ordinal or a pairwise relation keyed on REGION (master-type)
    name. Reconstructing a raw reference position from these is impossible: there is
    no length, no centroid, no axis -- only order and grouping.
    """

    regions: Tuple[str, ...]
    region_tier: Mapping[str, int]
    region_column: Mapping[str, int]
    left_of: Tuple[Tuple[str, str], ...] = ()
    above: Tuple[Tuple[str, str], ...] = ()
    adjacency: Tuple[Tuple[str, str], ...] = ()
    banked: frozenset = field(default_factory=frozenset)
    wide_block: frozenset = field(default_factory=frozenset)
    structure_hints: Tuple[ReferenceStructureHint, ...] = ()

    def __post_init__(self) -> None:
        labels = set(self.regions)
        if len(labels) != len(self.regions) or any(
            type(item) is not str or not item for item in self.regions
        ):
            raise ValueError("reference spatial regions must be unique names")
        if set(self.region_tier) != labels or set(self.region_column) != labels:
            raise ValueError("reference spatial ordinals must cover every region")
        if any(type(value) is not int or value < 0 for value in self.region_tier.values()):
            raise ValueError("reference spatial tiers must be nonnegative ordinals")
        tier_values = tuple(self.region_tier.values())
        if tier_values and set(tier_values) != set(range(max(tier_values) + 1)):
            raise ValueError("reference spatial tiers must be compact ordinals")
        columns = tuple(self.region_column.values())
        if any(type(value) is not int or value < 0 for value in columns):
            raise ValueError(
                "reference spatial columns must be nonnegative integer ordinals"
            )
        if sorted(columns) != list(range(len(self.regions))):
            raise ValueError("reference spatial columns must be a complete rank order")
        for relation_name in ("left_of", "above", "adjacency"):
            relations = getattr(self, relation_name)
            if any(
                type(pair) is not tuple
                or len(pair) != 2
                or pair[0] not in labels
                or pair[1] not in labels
                or pair[0] == pair[1]
                for pair in relations
            ):
                raise ValueError(
                    f"reference spatial {relation_name} contains an invalid relation"
                )
        if any(
            self.region_column[a] >= self.region_column[b]
            for a, b in self.left_of
        ):
            raise ValueError(
                "reference spatial left_of relation contradicts column ordinals"
            )
        if any(self.region_tier[a] <= self.region_tier[b] for a, b in self.above):
            raise ValueError(
                "reference spatial above relation contradicts tier ordinals"
            )
        if any(
            abs(self.region_tier[a] - self.region_tier[b]) > 1
            or abs(self.region_column[a] - self.region_column[b]) > 1
            for a, b in self.adjacency
        ):
            raise ValueError(
                "reference spatial adjacency contradicts region ordinals"
            )
        if not set(self.banked).issubset(labels) or not set(self.wide_block).issubset(labels):
            raise ValueError("reference spatial role sets must name known regions")
        if (
            type(self.structure_hints) is not tuple
            or len(self.structure_hints) > _MAX_HINTS
            or any(
                type(item) is not ReferenceStructureHint
                for item in self.structure_hints
            )
            or len({item.hint_id for item in self.structure_hints})
            != len(self.structure_hints)
        ):
            raise ValueError(
                "reference structure hints must be an exact bounded unique tuple"
            )
        for hint in self.structure_hints:
            hint.assert_authoritative()
        object.__setattr__(self, "regions", tuple(sorted(self.regions)))
        object.__setattr__(
            self,
            "region_tier",
            MappingProxyType(dict(sorted(self.region_tier.items()))),
        )
        object.__setattr__(
            self,
            "region_column",
            MappingProxyType(dict(sorted(self.region_column.items()))),
        )
        for relation_name in ("left_of", "above", "adjacency"):
            object.__setattr__(
                self,
                relation_name,
                tuple(sorted(set(getattr(self, relation_name)))),
            )
        object.__setattr__(self, "banked", frozenset(self.banked))
        object.__setattr__(self, "wide_block", frozenset(self.wide_block))
        object.__setattr__(
            self, "structure_hints", tuple(sorted(self.structure_hints))
        )

    def canonical_data(self) -> Dict[str, object]:
        """Return only coordinate-free craft, in a byte-stable field order."""

        return {
            "schema": "clay.reference-spatial-intent/v2",
            "regions": list(self.regions),
            "region_tier": {
                region: self.region_tier[region] for region in sorted(self.region_tier)
            },
            "region_column": {
                region: self.region_column[region]
                for region in sorted(self.region_column)
            },
            "left_of": [list(pair) for pair in sorted(self.left_of)],
            "above": [list(pair) for pair in sorted(self.above)],
            "adjacency": [list(pair) for pair in sorted(self.adjacency)],
            "banked": sorted(self.banked),
            "wide_block": sorted(self.wide_block),
            "structure_hints": [
                hint.canonical_data() for hint in self.structure_hints
            ],
        }

    @property
    def reference_scope_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def human_gap_measurements(self) -> Dict[str, object]:
        """Return the closed coordinate-free payload accepted by P2 observations."""

        return {
            "reference_available": True,
            "region_tier_ordinals": dict(self.region_tier),
            "region_column_ordinals": dict(self.region_column),
            "left_of_relations": [list(pair) for pair in self.left_of],
            "above_relations": [list(pair) for pair in self.above],
            "adjacency_relations": [list(pair) for pair in self.adjacency],
            "banked_regions": sorted(self.banked),
            "wide_block_regions": sorted(self.wide_block),
            "required_relation_count": (
                len(self.left_of) + len(self.above) + len(self.adjacency)
            ),
        }


def device_master_map(source_netlist: str) -> Dict[str, str]:
    """``device name -> PDK master`` from a SPICE/CDL source netlist.

    A MOS line ``M<name> d g s b <master> ...`` yields ``<master>``; a resistor /
    cap line yields its ``$.MODEL=<master>`` (or the bare passive master token).
    This is a SCHEMATIC property (the instantiated device master), the same token
    the reference SREF masters are grouped by -- so it maps every netlist device to
    its reference REGION without any geometry.
    """
    out: Dict[str, str] = {}
    for raw in source_netlist.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        toks = line.split()
        if not toks:
            continue
        head = toks[0]
        c0 = head[0].lower()
        if c0 == "m" and len(toks) >= 6:
            master = toks[5]
            if master in _DEVICE_MASTERS:
                out[head] = master
            elif master.casefold().startswith("p"):
                out[head] = "pmos"
            elif master.casefold().startswith("n"):
                out[head] = "nmos"
        elif c0 in ("r", "c"):
            m = re.search(r"\$\.MODEL\s*=\s*(\S+)", line)
            if m:
                master = m.group(1)
                out[head] = (
                    master
                    if master in _DEVICE_MASTERS
                    else ("resistor" if c0 == "r" else "capacitor")
                )
            else:
                for dt in _DEVICE_MASTERS:
                    if dt in _PASSIVE_MASTERS and re.search(rf"\b{dt}\b", line):
                        out[head] = dt
                        break
                else:
                    out[head] = "resistor" if c0 == "r" else "capacitor"
        elif c0 == "q" and len(toks) >= 5:
            model = toks[4].casefold()
            if model.startswith("pnp"):
                out[head] = "pnp"
            elif model.startswith("npn"):
                out[head] = "npn"
    return out


def _region_master_from_cell_name(name: str) -> str | None:
    """Return a coordinate-free device family for one GDS cell name.

    Foundry master names that already occur in the normalized topology retain
    their exact family.  Complete reference exports can instead use generated
    Cadence names (for example ``pmos3v_CDNS_*``); those are reduced to one of
    four semantic device classes without inspecting circuit identity or PDK.
    """

    for master in _DEVICE_MASTERS:
        if re.match(rf"^{master}(_|\$|\d|$)", name):
            return master
    folded = name.casefold()
    semantic_patterns = (
        (r"^pmos(?:_|\d|$)", "pmos"),
        (r"^nmos(?:_|\d|$)", "nmos"),
        (r"^(?:poly_res|resistor)(?:_|\d|$)", "resistor"),
        (r"^(?:mimcap|capacitor)(?:_|\d|$)", "capacitor"),
    )
    for pattern, family in semantic_patterns:
        if re.match(pattern, folded):
            return family
    return None


def _read_region_boxes(gds_path: str) -> Dict[str, Dict[str, float]]:
    """Per-master-type REGION centroid + extent from a reference GDS (LOCAL only).

    Groups every top-level SREF instance by which device master its cell name
    starts with, and accumulates the instance PLACEMENT POINTS (the SREF
    displacement) into a region centroid + bounding extent. The returned dict is
    INTERNAL to the derivation -- the public result never exposes these numbers.
    """
    import klayout.db as kdb  # optional dependency; only needed for extraction

    layout = kdb.Layout()
    layout.read(gds_path)
    dbu = layout.dbu
    tops = tuple(layout.top_cells())
    if len(tops) != 1:
        raise ValueError(
            f"reference spatial extraction requires one top cell; found {len(tops)}"
        )
    top = tops[0]

    pts: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def visit(cell, parent_trans) -> None:
        for inst in cell.each_inst():
            name = inst.cell.name
            transforms = tuple(inst.cell_inst.each_cplx_trans())
            for local_trans in transforms:
                composed = parent_trans * local_trans
                master = _region_master_from_cell_name(name)
                if master is not None:
                    disp = composed.disp
                    pts[master].append((disp.x * dbu, disp.y * dbu))
                else:
                    visit(inst.cell, composed)

    visit(top, kdb.ICplxTrans())

    regions: Dict[str, Dict[str, float]] = {}
    for master, pl in pts.items():
        xs = [p[0] for p in pl]
        ys = [p[1] for p in pl]
        regions[master] = {
            "cx": statistics.mean(xs),
            "cy": statistics.mean(ys),
            "x0": min(xs),
            "x1": max(xs),
            "y0": min(ys),
            "y1": max(ys),
            "n": float(len(pl)),
        }
    return regions


def _natural_break_tiers(regions: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    """Assign each region a TIER ordinal (0 = bottom) by NATURAL BREAKS in cy.

    Regions are sorted by centroid-y; a tier boundary is inserted at every gap that
    EXCEEDS ``max(2*median_gap, mean_gap + stdev_gap)`` -- a data-driven threshold
    that fires only on a gap that genuinely stands out from the typical spacing (in
    the reference LDO this isolates the resistor bank on top of the MOS+cap core).
    No fixed micron threshold, so the rule transfers to any reference. The returned
    map is ordinals only.
    """
    labels = sorted(regions, key=lambda r: regions[r]["cy"])
    cys = [regions[r]["cy"] for r in labels]
    if len(cys) <= 1:
        return {r: 0 for r in labels}
    gaps = [cys[i + 1] - cys[i] for i in range(len(cys) - 1)]
    median = statistics.median(gaps)
    mean = statistics.mean(gaps)
    sd = statistics.pstdev(gaps)
    thr = max(2.0 * median, mean + sd)
    tier: Dict[str, int] = {}
    t = 0
    for i, r in enumerate(labels):
        tier[r] = t
        if i < len(gaps) and gaps[i] > thr:
            t += 1
    return tier


def _columns(
    regions: Dict[str, Dict[str, float]], *, coordinate: str = "cx"
) -> Dict[str, int]:
    """Canonical COLUMN ordinal from one centroid axis, modulo axis reversal."""
    ordered = sorted(regions, key=lambda r: regions[r][coordinate])
    if tuple(ordered) > tuple(reversed(ordered)):
        ordered.reverse()
    return {r: i for i, r in enumerate(ordered)}


def _canonical_tier_direction(tiers: Dict[str, int]) -> Dict[str, int]:
    """Remove the global up/down sign while retaining tier grouping and order."""

    if not tiers:
        return tiers
    maximum = max(tiers.values())

    def groups(values: Dict[str, int]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(
            tuple(sorted(region for region, tier in values.items() if tier == index))
            for index in range(maximum + 1)
        )

    flipped = {region: maximum - tier for region, tier in tiers.items()}
    return flipped if groups(flipped) < groups(tiers) else tiers


def _canonical_axes(
    regions: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Choose a canonical abstract frame modulo quarter-turn rigid rotations.

    Both physical axes are tried as the column axis.  Each candidate has already
    discarded its sign, so selecting the lexicographically first REGION-role
    arrangement is independent of x/y coordinates and of the GDS top transform.
    """

    candidates: List[
        Tuple[
            Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]],
            Dict[str, int],
            Dict[str, int],
        ]
    ] = []
    for column_axis, tier_axis in (("cx", "cy"), ("cy", "cx")):
        columns = _columns(regions, coordinate=column_axis)
        projected = {
            region: {"cy": values[tier_axis]} for region, values in regions.items()
        }
        tiers = _canonical_tier_direction(_natural_break_tiers(projected))
        ordered_columns = tuple(sorted(columns, key=columns.__getitem__))
        maximum = max(tiers.values(), default=-1)
        tier_groups = tuple(
            tuple(sorted(region for region, tier in tiers.items() if tier == index))
            for index in range(maximum + 1)
        )
        candidates.append(((ordered_columns, tier_groups), tiers, columns))
    _, tiers, columns = min(candidates, key=lambda item: item[0])
    return tiers, columns


def _pairwise_orders(
    tiers: Dict[str, int],
    columns: Dict[str, int],
) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]:
    """``left_of`` + ``above`` DERIVED FROM the robust ordinals (column rank / tier),
    NOT raw centroids.

    Leak-safety: a bare ``cx_a < cx_b`` sign reads the centroids down to sub-micron, so a
    reference perturbation smaller than the real spacing could flip a relation -- that
    fine-position sensitivity IS a coordinate leak (proven by the jitter-invariance test:
    a small device whose y-centroid sits inside a full-height region's span had its
    ``above`` flip under a 1um jitter). The fix carries exactly the order information the
    already-leak-invariant ``region_column`` (centroid-x RANK) and ``region_tier``
    (natural-break tiers) encode, and nothing finer: ``left_of`` is the column-rank order;
    ``above`` is the TIER order -- regions in the SAME tier are co-height, so NO above/
    below claim between them. No centroid magnitude is read, so the relations inherit the
    ordinals' perturbation invariance.
    """
    labels = sorted(tiers)
    left_of: List[Tuple[str, str]] = []
    above: List[Tuple[str, str]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            if columns[a] < columns[b]:
                left_of.append((a, b))
            elif columns[b] < columns[a]:
                left_of.append((b, a))
            if tiers[a] > tiers[b]:
                above.append((a, b))
            elif tiers[b] > tiers[a]:
                above.append((b, a))
    return tuple(sorted(left_of)), tuple(sorted(above))


def _adjacency(
    tiers: Dict[str, int], columns: Dict[str, int]
) -> Tuple[Tuple[str, str], ...]:
    """Region pairs that are NEIGHBOURS in the leak-invariant ordinal grid: adjacent in
    tier (``|Δtier| <= 1``) AND adjacent in column rank (``|Δcolumn| <= 1``).

    Derived from the robust ordinals, NOT raw extents: a bounding-box overlap test reads
    the fine inter-region gap and flips at its tolerance boundary under a small reference
    jitter (proven a coordinate leak by the jitter-invariance test -- mim/n18 adjacency
    flipped under a 2um jitter). The ordinal-grid neighbour is the ``@abut`` signal --
    'these two blocks sit next to each other in the floorplan' -- with no fine-position
    dependence, so it inherits the ordinals' perturbation invariance.
    """
    labels = sorted(tiers)
    out: List[Tuple[str, str]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            if abs(tiers[a] - tiers[b]) <= 1 and abs(columns[a] - columns[b]) <= 1:
                out.append((a, b))
    return tuple(sorted(out))


def _wide_block_regions(
    regions: Sequence[str],
    region_max_wm: Dict[str, float],
) -> frozenset:
    """Regions holding a device whose drawn W*M is FAR above the median (the pass
    device). Keyed on schematic SIZE, confirmed by the reference's full-height span.
    """
    vals = [v for v in region_max_wm.values() if v > 0]
    if not vals:
        return frozenset()
    median = statistics.median(vals)
    return frozenset(
        r for r in regions if region_max_wm.get(r, 0.0) > 4.0 * median
    )


def _mos_master_body_counts(source_netlist: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for raw in source_netlist.splitlines():
        tokens = raw.strip().split()
        if len(tokens) < 6 or not tokens[0] or tokens[0][0].lower() != "m":
            continue
        master = tokens[5]
        if master not in _DEVICE_MASTERS:
            if master.casefold().startswith("p"):
                master = "pmos"
            elif master.casefold().startswith("n"):
                master = "nmos"
        counts[(master, tokens[4])] += 1
    return dict(counts)


def _mos_polarity(master: str) -> str | None:
    """Normalize a PCell/master family to its coordinate-free MOS polarity."""

    folded = str(master).casefold()
    if folded in {"p18", "p33", "pmos"}:
        return "p"
    if folded in {"n18", "n33", "nmos"}:
        return "n"
    return None


def _reference_structure_hints(
    regions: Mapping[str, Mapping[str, float]],
    *,
    banked: frozenset,
    wide_block: frozenset,
    source_netlist: str,
) -> Tuple[ReferenceStructureHint, ...]:
    """Emit structure-granularity hints without retaining reference coordinates."""

    labels = tuple(sorted(regions))
    if not labels:
        return ()
    hints: list[ReferenceStructureHint] = []
    for region in sorted(banked):
        hints.append(ReferenceStructureHint(
            hint_id=f"bank.{region}",
            primitive="bank",
            subjects=(region,),
            status="identified",
        ))
        hints.append(ReferenceStructureHint(
            hint_id=f"passive_bank_partition.{region}",
            primitive="passive_bank_partition",
            subjects=(region,),
            status="unmeasurable",
            gap=(
                "Anonymous reference instances identify a passive bank but do not "
                "identify a topology-owned sub-bank partition."
            ),
        ))

    mos_regions = tuple(
        region
        for region in labels
        if _mos_polarity(region) is not None
        and float(regions[region].get("n", 0.0)) >= 2.0
    )
    body_counts = _mos_master_body_counts(source_netlist)
    bodies_by_master: Dict[str, set[str]] = defaultdict(set)
    bodies_by_polarity: Dict[str, set[str]] = defaultdict(set)
    for (master, body), count in body_counts.items():
        polarity = _mos_polarity(master)
        if polarity is not None and count > 0 and body:
            bodies_by_master[master].add(body)
            bodies_by_polarity[polarity].add(body)
    for region in labels:
        polarity = _mos_polarity(region)
        if polarity is None:
            continue
        side = "top" if polarity == "p" else "bottom"
        bodies = tuple(sorted(
            bodies_by_master.get(region)
            or bodies_by_polarity.get(polarity, ())
        ))
        if bodies:
            hints.append(ReferenceStructureHint(
                hint_id=f"rail_affinity.{region}.{side}",
                primitive="rail_affinity",
                subjects=tuple(sorted((region, f"side:{side}", *(
                    f"body:{body}" for body in bodies
                )))),
                status="topology_declared",
            ))
        else:
            hints.append(ReferenceStructureHint(
                hint_id=f"rail_affinity.{region}.{side}",
                primitive="rail_affinity",
                subjects=tuple(sorted((region, f"side:{side}"))),
                status="unmeasurable",
                gap=(
                    "The reference GDS identifies a MOS PCell region, but the "
                    "source topology does not bind a same-polarity body domain."
                ),
            ))
    for region in mos_regions:
        hints.append(ReferenceStructureHint(
            hint_id=f"finger_array_order.{region}",
            primitive="finger_array_order",
            subjects=(region,),
            status="unmeasurable",
            gap=(
                "Anonymous reference MOS instances do not bind functional device "
                "identity to an exact finger sequence."
            ),
        ))
        shared_bodies = tuple(sorted(
            body
            for (master, body), count in body_counts.items()
            if master == region and count >= 2
        ))
        if shared_bodies:
            for ordinal, body in enumerate(shared_bodies):
                hints.append(ReferenceStructureHint(
                    hint_id=f"well_island_sharing.{region}.{ordinal:03d}",
                    primitive="well_island_sharing",
                    subjects=tuple(sorted((body, region))),
                    status="topology_declared",
                ))
        else:
            hints.append(ReferenceStructureHint(
                hint_id=f"well_island_sharing.{region}.unmeasurable",
                primitive="well_island_sharing",
                subjects=(region,),
                status="unmeasurable",
                gap=(
                    "Reference spatial instances do not expose a repeated body-domain "
                    "membership for this MOS family."
                ),
            ))

    hints.extend((
        ReferenceStructureHint(
            hint_id="density_whitespace_target.reference",
            primitive="density_whitespace_target",
            subjects=labels,
            status="unmeasurable",
            gap=(
                "Master-placement points do not provide complete active artwork for "
                "an exact density or whitespace target."
            ),
        ),
        ReferenceStructureHint(
            hint_id="od_fill_target.reference",
            primitive="od_fill_target",
            subjects=labels,
            status="unmeasurable",
            gap=(
                "Reference master-placement points do not identify complete, "
                "functionally owned OD artwork on an installed-PDK active layer."
            ),
        ),
        ReferenceStructureHint(
            hint_id="channel.reference",
            primitive="channel",
            subjects=labels,
            status="unmeasurable",
            gap=(
                "Ordinal region separation cannot prove an exact device-free routing "
                "corridor in the reference GDS."
            ),
        ),
        ReferenceStructureHint(
            hint_id="power_sensitive_net_budget.reference",
            primitive="power_sensitive_net_budget",
            subjects=labels,
            status="unmeasurable",
            gap=(
                "Reference topology and spatial GDS do not contain calibrated "
                "current, voltage-drop, resistance, or capacitance limits."
            ),
        ),
        ReferenceStructureHint(
            hint_id="thermal_stress_neighborhood.reference",
            primitive="thermal_stress_neighborhood",
            subjects=tuple(sorted(wide_block)) or labels,
            status="unmeasurable",
            gap=(
                "The reference contains no calibrated thermal or stress field from "
                "which to extract a neighborhood obligation."
            ),
        ),
    ))
    return tuple(sorted(hints))


def extract_reference_spatial_intent(
    gds_path: str,
    *,
    source_netlist: Optional[str] = None,
) -> ReferenceSpatialIntent:
    """Derive the reference layout's ABSTRACT arrangement from its GDS + netlist.

    Reads the per-master-type region placements from ``gds_path`` and (optionally)
    the per-device drawn sizes from ``source_netlist`` to flag the wide PASS block.
    Returns a :class:`ReferenceSpatialIntent` of relations ONLY -- no coordinate.

    Raises if the GDS cannot be read (klayout missing / bad path); the caller is
    expected to fall back to the role-only floorplan when extraction is unavailable.
    """
    regions = _read_region_boxes(gds_path)
    labels = tuple(sorted(regions))
    if not regions:
        return ReferenceSpatialIntent(
            regions=(), region_tier={}, region_column={}
        )

    tier, column = _canonical_axes(regions)
    left_of, above = _pairwise_orders(tier, column)
    adjacency = _adjacency(tier, column)

    banked = frozenset(r for r in labels if r in _PASSIVE_MASTERS)

    region_max_wm: Dict[str, float] = defaultdict(float)
    if source_netlist:
        masters = device_master_map(source_netlist)
        sizes = _device_wm(source_netlist)
        for name, master in masters.items():
            if master in regions:
                region_max_wm[master] = max(
                    region_max_wm[master], sizes.get(name, 0.0)
                )
    wide_block = _wide_block_regions(labels, region_max_wm)
    structure_hints = _reference_structure_hints(
        regions,
        banked=banked,
        wide_block=wide_block,
        source_netlist=source_netlist or "",
    )

    return ReferenceSpatialIntent(
        regions=labels,
        region_tier=tier,
        region_column=column,
        left_of=left_of,
        above=above,
        adjacency=adjacency,
        banked=banked,
        wide_block=wide_block,
        structure_hints=structure_hints,
    )


def reference_spatial_intent_from_data(data: Mapping[str, Any]) -> ReferenceSpatialIntent:
    """Strictly replay the canonical coordinate-free extraction artifact."""

    if type(data) is not dict:
        raise ValueError("reference spatial intent payload must be an exact object")
    expected_keys = {
        "schema",
        "regions",
        "region_tier",
        "region_column",
        "left_of",
        "above",
        "adjacency",
        "banked",
        "wide_block",
        "structure_hints",
    }
    if set(data) != expected_keys or data.get("schema") != "clay.reference-spatial-intent/v2":
        raise ValueError("reference spatial intent payload schema is not exact")
    for field_name in (
        "regions",
        "left_of",
        "above",
        "adjacency",
        "banked",
        "wide_block",
        "structure_hints",
    ):
        if type(data[field_name]) is not list:
            raise ValueError(f"reference spatial {field_name} must be an exact list")
    if type(data["region_tier"]) is not dict or type(data["region_column"]) is not dict:
        raise ValueError("reference spatial ordinal maps must be exact objects")

    hints = []
    for raw in data["structure_hints"]:
        if type(raw) is not dict or set(raw) != {
            "hint_id", "primitive", "subjects", "status", "gap"
        }:
            raise ValueError("reference structure hint payload keys are not exact")
        if type(raw["subjects"]) is not list:
            raise ValueError("reference structure hint subjects must be an exact list")
        hints.append(ReferenceStructureHint(
            hint_id=raw["hint_id"],
            primitive=raw["primitive"],
            subjects=tuple(raw["subjects"]),
            status=raw["status"],
            gap=raw["gap"],
        ))

    def relations(field_name: str) -> Tuple[Tuple[str, str], ...]:
        result = []
        for raw in data[field_name]:
            if type(raw) is not list or len(raw) != 2:
                raise ValueError(
                    f"reference spatial {field_name} relation is not exact"
                )
            result.append((raw[0], raw[1]))
        return tuple(result)

    replayed = ReferenceSpatialIntent(
        regions=tuple(data["regions"]),
        region_tier=dict(data["region_tier"]),
        region_column=dict(data["region_column"]),
        left_of=relations("left_of"),
        above=relations("above"),
        adjacency=relations("adjacency"),
        banked=frozenset(data["banked"]),
        wide_block=frozenset(data["wide_block"]),
        structure_hints=tuple(hints),
    )
    if replayed.canonical_data() != data:
        raise ValueError("reference spatial intent payload is not canonical")
    return replayed


def require_reference_structure_primitives(
    intent: ReferenceSpatialIntent,
    required_primitives: Tuple[str, ...] = REFERENCE_STRUCTURE_PRIMITIVES,
) -> None:
    """Fail closed when an identifiable structure class was silently dropped."""

    if type(intent) is not ReferenceSpatialIntent:
        raise TypeError("reference structure mapping requires ReferenceSpatialIntent")
    if (
        type(intent.structure_hints) is not tuple
        or not intent.structure_hints
        or len(intent.structure_hints) > _MAX_HINTS
        or any(type(item) is not ReferenceStructureHint for item in intent.structure_hints)
        or len({item.hint_id for item in intent.structure_hints})
        != len(intent.structure_hints)
    ):
        raise ValueError(
            "reference structure hints must be a non-empty exact bounded unique tuple"
        )
    for hint in intent.structure_hints:
        hint.assert_authoritative()
        if (
            hint.primitive in _REFERENCE_ONLY_BLOCKING_PRIMITIVES
            and not hint.blocking
        ):
            raise ValueError(
                f"reference-only {hint.primitive} hint must remain blocking"
            )
    if (
        type(required_primitives) is not tuple
        or not required_primitives
        or len(required_primitives) > len(REFERENCE_STRUCTURE_PRIMITIVES)
        or any(type(item) is not str for item in required_primitives)
        or len(set(required_primitives)) != len(required_primitives)
        or not set(required_primitives).issubset(REFERENCE_STRUCTURE_PRIMITIVES)
    ):
        raise ValueError("required reference structure primitives are not exact")
    present = {hint.primitive for hint in intent.structure_hints}
    missing = tuple(sorted(set(required_primitives) - present))
    if missing:
        raise ValueError(
            "reference structure hints are missing mapped primitives: "
            + ", ".join(missing)
        )
    from clay.frontend.primitive_registry import phase1_minimum_primitive_ids

    minimum = set(phase1_minimum_primitive_ids())
    unmapped = tuple(sorted(present - minimum))
    if unmapped:
        raise ValueError(
            "reference structure hints name primitives outside Phase 1: "
            + ", ".join(unmapped)
        )


def _device_wm(source_netlist: str) -> Dict[str, float]:
    """``MOS name -> drawn W * M`` (a schematic size, used to flag the pass block)."""
    out: Dict[str, float] = {}
    for raw in source_netlist.splitlines():
        line = raw.strip()
        toks = line.split()
        if not toks or toks[0][0].lower() != "m" or len(toks) < 6:
            continue
        w = m = 1.0
        for tok in toks:
            if tok.lower().startswith("w="):
                num = re.match(r"[-+]?\d*\.?\d+", tok[2:])
                if num:
                    w = float(num.group(0))
            elif tok.lower().startswith("m="):
                num = re.match(r"[-+]?\d*\.?\d+", tok[2:])
                if num:
                    m = float(num.group(0))
        out[toks[0]] = w * m
    return out
