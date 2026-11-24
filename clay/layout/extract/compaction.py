"""Intent-constraint-graph placement COMPACTION (outline minimization).

WHAT THIS IS
============
A post-solve pass that minimizes the placement OUTLINE by per-axis
constraint-graph compaction -- the classic longest-path form (equivalent to
the LP on the separation DAG) -- alternating x and y to a fixed point, with a
MINIMAL-PERTURBATION realization: longest paths give every coordinate a
[lower, upper] slack interval at the compacted extent, and each coordinate is
CLAMPED into its interval toward its input value. Only devices on binding
chains move; everything else stays put ("press, never rearrange").

THE FULL-2D FACE (mode='2d', 2026-06-12) -- the band "row owns the full
width" assumption RETIRED: a style row is a RELATIVE-ORDER constraint over
its members, never a full-width y strip. Two upgrades over the press face:
(1) an x-overlapping CROSS-row pair's y edge is directed by the ROW ORDER
(hard intent edge; a scrambled input is pressed back into order, never
legalized inverted), while x-disjoint cross-row pairs keep carrying NO y
edge -- the embedding freedom; (2) the deterministic :func:`embed_search`
stage relocates partner-rigid groups across row x-packs where that strictly
shrinks the pressed outline (measured on the test4 bench: the third row's
two members owned a 12.5 um full-width strip with ~920 um^2 of flanking
holes; the search re-slots groups and the press drops the stack --
whitespace 34.7% -> 16.0%, device outline 9131.7 -> 7095.7 um^2).

WHY THIS RECOVERS AREA THE BAND SOLVE LEAVES ON THE TABLE
=========================================================
``solve_hierarchical_nonoverlap`` stacks RIGID BANDS: row k+1's bottom sits a
channel above row k's bbox TOP -- the top of the row's TALLEST member, across
the full row width. Where the tall member and the row above do not share an
x-span, that rigid seam wastes the height difference (measured on the test4
style bench: the CC1 cap row holds the row above 3.4 um higher than the
members actually under it require). The compaction graph is x-overlap-aware:
two nodes constrain each other on an axis only where their perpendicular
spans (grown by the min device gap) actually interact.

THE INTENT EDGES ARE IN THE GRAPH -- preserved BY CONSTRUCTION
==============================================================
* ROW MEMBERSHIP / WITHIN-ROW ORDER / CROSS-ROW ORDER: every interacting
  pair's edge direction is its INPUT axis order -- compaction presses, never
  reorders (the relative-order intent survives verbatim).
* CHANNELS (S6): a cross-row edge's weight sums, per crossed row boundary,
  ``max(band_channel, channel_um_per_gap[k])`` -- a declared routing channel
  is a floor the press cannot squash (it may relocate, never narrow).
* RAIL RESERVATIONS (P2): a reservation row's height already carries its
  wall clearance, so reservation seams contribute 0 channel (FLUSH) and the
  corridor node itself rides the graph (it can never be overlapped).
* MATCHED / MIRROR / DIFF PARTNERS sharing a row: their exact relative
  offset is an EQUALITY edge (offset union-find) -- symmetric arms stay
  equidistant on both axes.

WHAT IS DELIBERATELY NOT PRESERVED (recorded, not silent): the per-row
shared BOTTOM line of the band solve. Pinning it would chain every row
member to the row's tallest member and forfeit exactly the slack this pass
exists to recover; members keep their row (order edges) but may sit at
different bottoms after the press. The non-compacted default keeps the P3
bottom-line contract untouched.

LEAKAGE
=======
Inputs are the solved placement, the caller's target sizes and the
DESIGN-derived :class:`CompactionSpec` (row partition from the sketch,
partner topology from the netlist intent). No reference coordinate is read;
the perturbation guard rides tests/test_compaction.py.

This module is import-clean (stdlib only) so both ``topo_sketch`` (spec
construction) and ``relgraph`` (solve-side application) can consume it
without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Tuple

_XY = Tuple[float, float]


def _iter_full2d_collision_pairs(nodes, owner_by_node, layer_by_node, compatible):
    """Enumerate precisely the interacting owner/layer pairs without global N^2.

    None remains a wildcard layer. Canonical node ordering makes the resulting
    pair obligations identical to the exhaustive construction after sorting.
    """
    grouped = {}
    for node in nodes:
        grouped.setdefault(owner_by_node[node], {}).setdefault(layer_by_node[node], []).append(node)
    owners = sorted(grouped)
    for first_owner, second_owner in combinations(owners, 2):
        if (first_owner, second_owner) in compatible:
            continue
        left, right = grouped[first_owner], grouped[second_owner]
        for layer, first_nodes in left.items():
            if layer is None:
                matching_groups = right.values()
            else:
                matching_groups = (right.get(layer, ()), right.get(None, ()))
            for second_nodes in matching_groups:
                for first in first_nodes:
                    for second in second_nodes:
                        yield (first, second) if first < second else (second, first)

# Tolerance for the perpendicular interaction test: two spans grown by half
# the min gap each side interact only if they STRICTLY overlap beyond this
# (so neighbours packed at exactly the min gap do NOT pick up a spurious
# same-axis press constraint from float noise).
_OVERLAP_TOL: float = 1e-6
# Coordinate change below this is "no movement" (fixed-point detection).
_MOVE_TOL: float = 1e-9


@dataclass(frozen=True)
class CompactionSpec:
    """The DESIGN-derived structure the compaction graph consumes.

    ``row_of``: name -> bottom-up index in the FULL emitted row stack (device
    rows AND rail pseudo rows -- the same stack the band solve recovers, so
    row gap k is band gap k for ``channel_um_per_gap``). ``rail_rows``: the
    indices of reservation pseudo rows (their seams press flush -- P2).
    ``partners``: declared matched/mirror/diff pairs that SHARE a row (their
    relative offset is pinned as an equality edge; cross-row partners keep
    order only -- pinning a cross-row offset would chain whole rows rigid).
    ``slack_um``: a conservative extra channel added on every non-reservation
    row boundary (the honest "back off if routing degrades" knob; 0 = pure
    press).

    ``mode`` (2026-06-12, the band "row owns the full width" retirement):

    * ``'press'`` -- the historical face: every interacting pair's y edge is
      directed by its INPUT coordinate order (the press never reorders, and
      it also never CORRECTS a scrambled input).
    * ``'2d'`` -- the row is a RELATIVE-ORDER constraint, nothing more: an
      x-overlapping CROSS-row pair's y edge is directed by the ROW ORDER
      (hard intent edge -- a scrambled input, e.g. a relax candidate that
      proposed an inversion, is pressed back into row order, never
      legalized upside down). x-disjoint cross-row pairs carry no y edge
      at all (the embedding freedom). Same-row pairs keep the input
      direction on both axes (the press still never reorders a row).
      ``'2d'`` also enables the solve-side :func:`embed_search` stage.

    CONSERVATIVE ROUTING MARGIN (the round's declared guardrail): NOT a
    spec field. A press/search-level margin was tried twice and refused on
    measurement -- widening the interaction gate forced channel-separated
    vertical neighbours apart horizontally (placement exploded to 47%
    whitespace), and widening only the binding edges contradicted the
    partner-pinned equality offsets (positive cycles on the production
    bench). The margin's correct seat is the CHAIN's existing spacing
    parameter faces (sketch ``dev_gap_um``, solve ``min_gap`` /
    ``band_channel_um``), which this module inherits single-source -- a
    margined chain produces margined partner offsets, so nothing
    contradicts.

    ``channel_xnets`` (S6 x-resolved, 2026-06-12): per row boundary k, the
    tuple of nets crossing that boundary, each net as the sorted tuple of
    its member device names -- the SAME derivation that produced the
    sketch's ``channel_demands`` (``topo_sketch._channel_crossing_nets``;
    one source, no second bookkeeping), NAMES ONLY (geometry is resolved
    from the placement under check, never a stored coordinate). Consumed by
    :func:`_channel_capacity_violations` to hold every crossing net's lane
    x segment against the mode='2d' embed trials. Empty (the default)
    disables the check -- exactly the no-S6-data path.

    ``keepout_names`` (2026-06-12): the devices whose placed footprint is a
    HARD routing keep-out (lowered from ``intent.no_route_under`` -- the
    same matching rule the router's keep-out lowering uses; names only).
    Everything that must pass or exit around such a footprint (boundary io
    stubs included) flows through its x FLANKS (footprint edge to the
    design extent); :func:`_keepout_flank_violations` holds those flanks at
    the boundary-channel class of the keepout's adjacent seams against the
    mode='2d' embed trials. Sub-bbox effects (pad-inset pin extents) are
    below this module's resolution -- the guard is a DECLARED conservative
    refusal, not an exact reachability proof. Empty (the default) is
    vacuous."""

    row_of: Mapping[str, int]
    rail_rows: frozenset
    partners: Tuple[Tuple[str, str], ...] = ()
    slack_um: float = 0.0
    mode: str = "press"
    channel_xnets: Tuple[Tuple[Tuple[str, ...], ...], ...] = ()
    keepout_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("press", "2d"):
            raise ValueError(
                f"CompactionSpec: unknown mode {self.mode!r} "
                "(expected 'press' or '2d')"
            )


@dataclass(frozen=True, order=True)
class Full2DSeparationEdge:
    """One directed edge-to-edge clearance in an orthogonal master DAG."""

    lower: str
    upper: str
    clearance_nm: int

    def __post_init__(self) -> None:
        if (
            type(self.lower) is not str
            or not self.lower
            or type(self.upper) is not str
            or not self.upper
            or self.lower == self.upper
        ):
            raise ValueError("full-2D separation edge endpoints are malformed")
        if type(self.clearance_nm) is not int or self.clearance_nm < 0:
            raise ValueError("full-2D separation clearance must be an exact nm value")

    def payload(self) -> Tuple[object, ...]:
        return (self.lower, self.upper, self.clearance_nm)


class Full2DAxisProvenanceError(ValueError):
    """A fixed full-2D axis has no declaration-backed provenance."""

    def __init__(self, edge_ids: Tuple[str, ...]) -> None:
        if (
            type(edge_ids) is not tuple
            or not edge_ids
            or edge_ids != tuple(sorted(set(edge_ids)))
            or any(type(edge_id) is not str or not edge_id for edge_id in edge_ids)
        ):
            raise ValueError("full-2D provenance error requires canonical edge IDs")
        self.edge_ids = edge_ids
        super().__init__(
            "full-2D fixed axes lack declared provenance: " + ", ".join(edge_ids)
        )


@dataclass(frozen=True, order=True)
class Full2DPairConstraint:
    """One typed order or physical-clearance obligation for an atom pair."""

    first: str
    second: str
    clearance_nm: int
    mode: str
    axis: str | None = None
    lower: str | None = None
    upper: str | None = None
    authority_kind: str = "none"
    source_constraint_ids: Tuple[str, ...] = ()
    seed_hint: Tuple[str, str, str, str] | Tuple[()] = ()

    def __post_init__(self) -> None:
        if (
            type(self.first) is not str
            or not self.first
            or type(self.second) is not str
            or not self.second
            or self.first >= self.second
        ):
            raise ValueError("full-2D pair endpoints must be canonical exact names")
        if type(self.clearance_nm) is not int or self.clearance_nm < 0:
            raise ValueError("full-2D pair clearance must be one exact nm value")
        if self.mode not in {"fixed_edge", "four_way_disjunction"}:
            raise ValueError("full-2D pair mode is malformed")
        if (
            type(self.source_constraint_ids) is not tuple
            or self.source_constraint_ids
            != tuple(sorted(set(self.source_constraint_ids)))
            or any(
                type(source_id) is not str or not source_id
                for source_id in self.source_constraint_ids
            )
        ):
            raise ValueError("full-2D pair source IDs must be canonical exact names")
        if self.seed_hint:
            if (
                type(self.seed_hint) is not tuple
                or len(self.seed_hint) != 4
                or set(self.seed_hint[:2]) != {self.first, self.second}
                or self.seed_hint[2] not in {"horizontal", "vertical"}
                or type(self.seed_hint[3]) is not str
                or not self.seed_hint[3]
            ):
                raise ValueError("full-2D pair seed hint is malformed")
        if self.mode == "fixed_edge":
            if (
                self.clearance_nm != 0
                or self.axis not in {"horizontal", "vertical"}
                or set((self.lower, self.upper)) != {self.first, self.second}
                or self.lower == self.upper
            ):
                raise ValueError("full-2D fixed pair direction is malformed")
        elif (
            any(value is not None for value in (self.axis, self.lower, self.upper))
            or self.authority_kind != "physical_nonoverlap"
            or self.source_constraint_ids != ("physical_nonoverlap",)
        ):
            raise ValueError(
                "full-2D live pair requires physical non-overlap provenance"
            )

    @property
    def edge_id(self) -> str:
        if self.mode != "fixed_edge":
            raise ValueError("full-2D live pair has no fixed edge ID")
        return f"fixed_axis:{self.lower}->{self.upper}:{self.axis}"

    def payload(self) -> Dict[str, object]:
        return {
            "first": self.first,
            "second": self.second,
            "clearance_nm": self.clearance_nm,
            "mode": self.mode,
            "axis": self.axis,
            "lower": self.lower,
            "upper": self.upper,
            "authority_kind": self.authority_kind,
            "source_constraint_ids": self.source_constraint_ids,
            "seed_hint": self.seed_hint,
        }


@dataclass(frozen=True)
class Full2DCompactionGraph:
    """Canonical H/V collision-component graph with rigid master ownership."""

    nodes: Tuple[str, ...]
    sizes_nm: Tuple[Tuple[str, Tuple[int, int]], ...]
    master_sizes_nm: Tuple[Tuple[str, Tuple[int, int]], ...]
    node_owner_offsets_nm: Tuple[Tuple[str, str, Tuple[int, int]], ...]
    node_layers: Tuple[Tuple[str, Tuple[int, int] | None], ...]
    pair_constraints: Tuple[Full2DPairConstraint, ...] = ()
    horizontal_edges: Tuple[Full2DSeparationEdge, ...] = ()
    vertical_edges: Tuple[Full2DSeparationEdge, ...] = ()
    x_equalities: Tuple[Tuple[str, str, int], ...] = ()
    y_equalities: Tuple[Tuple[str, str, int], ...] = ()
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = ()
    grid_nm: int = 1
    physical_pair_cover: tuple = ()
    physical_clearance_nm: int | None = None
    physical_atom_pair_count: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.nodes) is not tuple
            or self.nodes != tuple(sorted(set(self.nodes)))
            or any(type(name) is not str or not name for name in self.nodes)
        ):
            raise ValueError("full-2D graph nodes must be canonical exact names")
        if type(self.grid_nm) is not int or self.grid_nm <= 0:
            raise ValueError("full-2D graph grid must be one positive integer nm")
        if (
            type(self.sizes_nm) is not tuple
            or self.sizes_nm != tuple(sorted(self.sizes_nm))
            or tuple(name for name, _size in self.sizes_nm) != self.nodes
        ):
            raise ValueError("full-2D graph sizes must canonically cover its nodes")
        for name, size in self.sizes_nm:
            if (
                type(size) is not tuple
                or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
            ):
                raise ValueError(f"full-2D graph size is malformed for {name!r}")
        if (
            type(self.master_sizes_nm) is not tuple
            or self.master_sizes_nm != tuple(sorted(self.master_sizes_nm))
            or not self.master_sizes_nm
            or len({name for name, _size in self.master_sizes_nm})
            != len(self.master_sizes_nm)
        ):
            raise ValueError("full-2D graph master sizes must be canonical")
        for name, size in self.master_sizes_nm:
            if (
                type(name) is not str
                or not name
                or type(size) is not tuple
                or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
            ):
                raise ValueError(f"full-2D graph master size is malformed for {name!r}")
        masters = {name for name, _size in self.master_sizes_nm}
        if (
            type(self.overlap_compatible_pairs) is not tuple
            or self.overlap_compatible_pairs
            != tuple(sorted(set(self.overlap_compatible_pairs)))
        ):
            raise ValueError(
                "full-2D overlap-compatible pairs must be canonical and unique"
            )
        for pair in self.overlap_compatible_pairs:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                or pair[0] >= pair[1]
                or pair[0] not in masters
                or pair[1] not in masters
            ):
                raise ValueError(
                    "full-2D overlap-compatible pair is malformed or unowned"
                )
        if (
            type(self.node_owner_offsets_nm) is not tuple
            or self.node_owner_offsets_nm != tuple(sorted(self.node_owner_offsets_nm))
            or tuple(node for node, _owner, _offset in self.node_owner_offsets_nm)
            != self.nodes
        ):
            raise ValueError("full-2D graph node ownership must canonically cover nodes")
        for node, owner, offset in self.node_owner_offsets_nm:
            if (
                owner not in masters
                or type(offset) is not tuple
                or len(offset) != 2
                or any(type(value) is not int or value < 0 for value in offset)
                or any(value % self.grid_nm for value in offset)
            ):
                raise ValueError(
                    f"full-2D graph node ownership is malformed for {node!r}"
                )
        if (
            type(self.node_layers) is not tuple
            or self.node_layers != tuple(sorted(self.node_layers))
            or tuple(node for node, _layer in self.node_layers) != self.nodes
        ):
            raise ValueError("full-2D graph node layers must canonically cover nodes")
        for node, layer in self.node_layers:
            if layer is not None and (
                type(layer) is not tuple
                or len(layer) != 2
                or any(type(value) is not int or value < 0 for value in layer)
            ):
                raise ValueError(f"full-2D graph node layer is malformed for {node!r}")
        node_names = frozenset(self.nodes)
        if (
            type(self.pair_constraints) is not tuple
            or any(
                type(row) is not Full2DPairConstraint
                for row in self.pair_constraints
            )
            or self.pair_constraints != tuple(sorted(set(self.pair_constraints)))
            or any(
                row.first not in node_names or row.second not in node_names
                for row in self.pair_constraints
            )
        ):
            raise ValueError("full-2D pair constraints are not canonical and owned")
        unproven_fixed_edges = tuple(sorted(
            row.edge_id
            for row in self.pair_constraints
            if row.mode == "fixed_edge"
            and (
                row.authority_kind != "declared_hard"
                or not row.source_constraint_ids
            )
        ))
        if unproven_fixed_edges:
            raise Full2DAxisProvenanceError(unproven_fixed_edges)
        derived_horizontal = tuple(sorted(
            Full2DSeparationEdge(
                lower=str(row.lower),
                upper=str(row.upper),
                clearance_nm=row.clearance_nm,
            )
            for row in self.pair_constraints
            if row.mode == "fixed_edge" and row.axis == "horizontal"
        ))
        derived_vertical = tuple(sorted(
            Full2DSeparationEdge(
                lower=str(row.lower),
                upper=str(row.upper),
                clearance_nm=row.clearance_nm,
            )
            for row in self.pair_constraints
            if row.mode == "fixed_edge" and row.axis == "vertical"
        ))
        legacy_unproven = tuple(sorted((
            *(
                f"fixed_axis:{edge.lower}->{edge.upper}:horizontal"
                for edge in set(self.horizontal_edges) - set(derived_horizontal)
            ),
            *(
                f"fixed_axis:{edge.lower}->{edge.upper}:vertical"
                for edge in set(self.vertical_edges) - set(derived_vertical)
            ),
        )))
        if legacy_unproven:
            raise Full2DAxisProvenanceError(legacy_unproven)
        if self.horizontal_edges and self.horizontal_edges != derived_horizontal:
            raise ValueError("full-2D horizontal edges diverge from typed pairs")
        if self.vertical_edges and self.vertical_edges != derived_vertical:
            raise ValueError("full-2D vertical edges diverge from typed pairs")
        object.__setattr__(self, "horizontal_edges", derived_horizontal)
        object.__setattr__(self, "vertical_edges", derived_vertical)
        for axis, edges in (
            ("horizontal", self.horizontal_edges),
            ("vertical", self.vertical_edges),
        ):
            if (
                type(edges) is not tuple
                or any(type(edge) is not Full2DSeparationEdge for edge in edges)
                or edges != tuple(sorted(set(edges)))
                or any(
                    edge.lower not in self.nodes or edge.upper not in self.nodes
                    for edge in edges
                )
            ):
                raise ValueError(f"full-2D {axis} edges are not canonical and owned")
            self._assert_dag(edges, axis)
        for axis, equalities in (
            ("x", self.x_equalities),
            ("y", self.y_equalities),
        ):
            if type(equalities) is not tuple or equalities != tuple(
                sorted(set(equalities))
            ):
                raise ValueError(f"full-2D {axis} equalities must be canonical")
            for row in equalities:
                if (
                    type(row) is not tuple
                    or len(row) != 3
                    or type(row[0]) is not str
                    or type(row[1]) is not str
                    or row[0] >= row[1]
                    or row[0] not in self.nodes
                    or row[1] not in self.nodes
                    or type(row[2]) is not int
                    or row[2] % self.grid_nm
                ):
                    raise ValueError(
                        f"full-2D {axis} equality is malformed or off-grid"
                    )
        owner_by_node = {
            node: owner for node, owner, _offset in self.node_owner_offsets_nm
        }
        layer_by_node = dict(self.node_layers)
        compatible = set(self.overlap_compatible_pairs)
        if self.physical_clearance_nm is not None:
            if type(self.physical_clearance_nm) is not int or self.physical_clearance_nm < 0:
                raise ValueError("full-2D compressed clearance must be an exact nonnegative integer")
            if any(row.mode == "fixed_edge" for row in self.pair_constraints):
                raise ValueError("full-2D geometry cover cannot replace declared fixed-edge authority")
            from clay.layout.extract.full2d_geometry_cover import build_geometry_cover
            expected_cover, representatives, pair_count = build_geometry_cover(
                self.sizes_nm, self.node_owner_offsets_nm, self.node_layers,
                self.physical_clearance_nm, self.overlap_compatible_pairs)
            if (type(self.physical_pair_cover) is not tuple or self.physical_pair_cover != expected_cover
                    or type(self.physical_atom_pair_count) is not int or self.physical_atom_pair_count != pair_count):
                raise ValueError("full-2D compressed cover differs from complete original collision geometry")
            required_pairs = {frozenset(pair) for pair in representatives}
        else:
            if self.physical_pair_cover or self.physical_atom_pair_count:
                raise ValueError("full-2D compressed cover requires physical clearance authority")
            required_pairs = {
                frozenset(pair)
                for pair in _iter_full2d_collision_pairs(self.nodes, owner_by_node, layer_by_node, compatible)
            }
        physical_pairs = [
            frozenset((row.first, row.second)) for row in self.pair_constraints
            if row.mode == "four_way_disjunction"
        ]
        if (
            required_pairs != set(physical_pairs)
            or len(physical_pairs) != len(set(physical_pairs))
        ):
            raise ValueError(
                "full-2D graph must assign every interacting atom pair exactly "
                "one physical non-overlap disjunction"
            )
        fixed_pairs = [
            frozenset((row.first, row.second)) for row in self.pair_constraints
            if row.mode == "fixed_edge"
        ]
        if (
            not set(fixed_pairs).issubset(required_pairs)
            or len(fixed_pairs) != len(set(fixed_pairs))
        ):
            raise ValueError(
                "full-2D graph declared order edges must be unique interacting pairs"
            )

    @property
    def master_nodes(self) -> Tuple[str, ...]:
        return tuple(name for name, _size in self.master_sizes_nm)

    def _assert_dag(
        self,
        edges: Tuple[Full2DSeparationEdge, ...],
        axis: str,
    ) -> None:
        followers = {name: set() for name in self.nodes}
        incoming = {name: 0 for name in self.nodes}
        for edge in edges:
            if edge.upper not in followers[edge.lower]:
                followers[edge.lower].add(edge.upper)
                incoming[edge.upper] += 1
        ready = sorted(name for name in self.nodes if incoming[name] == 0)
        visited = 0
        while ready:
            name = ready.pop(0)
            visited += 1
            for follower in sorted(followers[name]):
                incoming[follower] -= 1
                if incoming[follower] == 0:
                    ready.append(follower)
                    ready.sort()
        if visited != len(self.nodes):
            raise ValueError(f"full-2D {axis} graph contains a cycle")

    @property
    def pair_coverage(self) -> Tuple[Tuple[str, str, str], ...]:
        return tuple(sorted(
            (
                row.first,
                row.second,
                str(row.axis) if row.mode == "fixed_edge" else row.mode,
            )
            for row in self.pair_constraints
        ))

    @property
    def layered_footprints_nm(
        self,
    ) -> Tuple[Tuple[str, Tuple[Tuple[int, int, int, int, int, int], ...]], ...]:
        """Reconstruct the exact signed master-local typed collision atoms."""

        owner_by_node = {
            node: owner for node, owner, _offset in self.node_owner_offsets_nm
        }
        offset_by_node = {
            node: offset for node, _owner, offset in self.node_owner_offsets_nm
        }
        size_by_node = dict(self.sizes_nm)
        atoms = {}
        for node, layer in self.node_layers:
            if layer is None:
                continue
            offset = offset_by_node[node]
            size = size_by_node[node]
            atoms.setdefault(owner_by_node[node], []).append((
                layer[0],
                layer[1],
                offset[0],
                offset[1],
                offset[0] + size[0],
                offset[1] + size[1],
            ))
        return tuple(
            (owner, tuple(sorted(rows))) for owner, rows in sorted(atoms.items())
        )

    def payload(self) -> Dict[str, object]:
        result = {
            "schema": "clay.full-2d-master-compaction-graph/v7",
            "nodes": self.nodes,
            "sizes_nm": self.sizes_nm,
            "master_sizes_nm": self.master_sizes_nm,
            "node_owner_offsets_nm": self.node_owner_offsets_nm,
            "node_layers": self.node_layers,
            "horizontal_edges": tuple(
                edge.payload() for edge in self.horizontal_edges
            ),
            "vertical_edges": tuple(edge.payload() for edge in self.vertical_edges),
            "pair_constraints": tuple(
                row.payload() for row in self.pair_constraints
            ),
            "x_equalities": self.x_equalities,
            "y_equalities": self.y_equalities,
            "overlap_compatible_pairs": self.overlap_compatible_pairs,
            "grid_nm": self.grid_nm,
        }
        if self.physical_clearance_nm is not None:
            result.update(schema="clay.full-2d-master-compaction-graph/v8",
                          physical_pair_cover=self.physical_pair_cover,
                          physical_clearance_nm=self.physical_clearance_nm,
                          physical_atom_pair_count=self.physical_atom_pair_count)
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("ascii")).hexdigest()

    def replay_diagnostic(
        self,
        boxes: Mapping[str, Tuple[int, int, int, int]],
    ) -> Tuple[Dict[str, object], ...]:
        """Name every graph row rejected by one otherwise read-only replay."""

        def failure(
            row_id: str,
            constraint_names: Tuple[str, ...],
            source_constraint_ids: Tuple[str, ...] = (),
        ) -> Dict[str, object]:
            return {
                "constraint_names": constraint_names,
                "row_id": row_id,
                "source_constraint_ids": source_constraint_ids,
            }

        try:
            canonical = _canonical_full2d_boxes(boxes)
        except (TypeError, ValueError):
            return (failure(
                "graph_input:canonical_boxes",
                ("canonical_full2d_boxes",),
            ),)
        if tuple(canonical) != self.master_nodes:
            return (failure(
                "graph_input:master_inventory",
                ("master_inventory",),
            ),)

        failures = []
        master_sizes = dict(self.master_sizes_nm)
        for name, box in canonical.items():
            if (box[2] - box[0], box[3] - box[1]) != master_sizes[name]:
                failures.append(failure(
                    f"master_size:{name}",
                    ("master_size",),
                ))
            if box[0] % self.grid_nm or box[1] % self.grid_nm:
                failures.append(failure(
                    f"manufacturing_grid:{name}",
                    ("manufacturing_grid",),
                ))
        if failures:
            return tuple(sorted(failures, key=lambda row: str(row["row_id"])))

        if self.physical_clearance_nm is not None:
            from clay.layout.extract.full2d_geometry_cover import cover_collisions
            failures.extend(failure(f"full_geometry_collision:{first}<->{second}",
                                    ("physical_nonoverlap",),("physical_nonoverlap",))
                            for first,second in cover_collisions(self.physical_pair_cover,canonical))

        sizes = dict(self.sizes_nm)
        node_boxes = {
            node: (
                canonical[owner][0] + offset[0],
                canonical[owner][1] + offset[1],
                canonical[owner][0] + offset[0] + sizes[node][0],
                canonical[owner][1] + offset[1] + sizes[node][1],
            )
            for node, owner, offset in self.node_owner_offsets_nm
        }
        for row in self.pair_constraints:
            first = node_boxes[row.first]
            second = node_boxes[row.second]
            gap = row.clearance_nm
            if row.mode == "fixed_edge":
                axis = 0 if row.axis == "horizontal" else 1
                lower = node_boxes[str(row.lower)]
                upper = node_boxes[str(row.upper)]
                if lower[axis + 2] + gap > upper[axis]:
                    failures.append(failure(
                        row.edge_id,
                        row.source_constraint_ids,
                        row.source_constraint_ids,
                    ))
            elif not (
                first[2] + gap <= second[0]
                or second[2] + gap <= first[0]
                or first[3] + gap <= second[1]
                or second[3] + gap <= first[1]
            ):
                failures.append(failure(
                    f"four_way_disjunction:{row.first}<->{row.second}",
                    ("physical_nonoverlap",),
                    row.source_constraint_ids,
                ))
        for axis, rows in (("x", self.x_equalities), ("y", self.y_equalities)):
            coordinate = 0 if axis == "x" else 1
            for first, second, delta in rows:
                if (
                    node_boxes[second][coordinate]
                    - node_boxes[first][coordinate]
                    != delta
                ):
                    failures.append(failure(
                        f"{axis}_equality:{first}->{second}:{delta}",
                        (f"{axis}_equality",),
                    ))
        return tuple(sorted(failures, key=lambda row: str(row["row_id"])))

    def replay(self, boxes: Mapping[str, Tuple[int, int, int, int]]) -> bool:
        try:
            canonical = _canonical_full2d_boxes(boxes)
        except (TypeError, ValueError):
            return False
        if tuple(canonical) != self.master_nodes:
            return False
        master_sizes = dict(self.master_sizes_nm)
        if any(
            (box[2] - box[0], box[3] - box[1]) != master_sizes[name]
            or box[0] % self.grid_nm
            or box[1] % self.grid_nm
            for name, box in canonical.items()
        ):
            return False
        if self.physical_clearance_nm is not None:
            from clay.layout.extract.full2d_geometry_cover import cover_collisions
            if next(cover_collisions(self.physical_pair_cover, canonical), None) is not None:
                return False
        sizes = dict(self.sizes_nm)
        node_boxes = {
            node: (
                canonical[owner][0] + offset[0],
                canonical[owner][1] + offset[1],
                canonical[owner][0] + offset[0] + sizes[node][0],
                canonical[owner][1] + offset[1] + sizes[node][1],
            )
            for node, owner, offset in self.node_owner_offsets_nm
        }
        for axis, edges in ((0, self.horizontal_edges), (1, self.vertical_edges)):
            for edge in edges:
                lower = node_boxes[edge.lower]
                upper = node_boxes[edge.upper]
                if lower[axis + 2] + edge.clearance_nm > upper[axis]:
                    return False
        for row in self.pair_constraints:
            if row.mode != "four_way_disjunction":
                continue
            first = node_boxes[row.first]
            second = node_boxes[row.second]
            gap = row.clearance_nm
            if not (
                first[2] + gap <= second[0]
                or second[2] + gap <= first[0]
                or first[3] + gap <= second[1]
                or second[3] + gap <= first[1]
            ):
                return False
        for axis, rows in ((0, self.x_equalities), (1, self.y_equalities)):
            if any(
                node_boxes[second][axis] - node_boxes[first][axis] != delta
                for first, second, delta in rows
            ):
                return False
        return True


def _canonical_full2d_boxes(
    boxes: Mapping[str, Tuple[int, int, int, int]],
) -> Dict[str, Tuple[int, int, int, int]]:
    if not isinstance(boxes, Mapping) or not boxes:
        raise TypeError("full-2D boxes require one non-empty mapping")
    result = {}
    for raw_name, raw_box in boxes.items():
        if type(raw_name) is not str or not raw_name:
            raise TypeError("full-2D box name must be one exact string")
        if (
            type(raw_box) is not tuple
            or len(raw_box) != 4
            or any(type(value) is not int for value in raw_box)
            or raw_box[2] <= raw_box[0]
            or raw_box[3] <= raw_box[1]
        ):
            raise ValueError(f"full-2D bbox is malformed for {raw_name!r}")
        result[raw_name] = raw_box
    return dict(sorted(result.items()))


def full2d_graph_from_seed(
    boxes: Mapping[str, Tuple[int, int, int, int]],
    *,
    minimum_clearance_nm: int,
    grid_nm: int,
    x_equalities: Tuple[Tuple[str, str, int], ...] = (),
    y_equalities: Tuple[Tuple[str, str, int], ...] = (),
    pair_axis_authority: Tuple[
        Tuple[str, str, str, str, str, Tuple[str, ...]], ...
    ] = (),
    pair_axis_preference: Tuple[Tuple[str, str, str], ...] = (),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    footprints_nm: Tuple[
        Tuple[str, Tuple[Tuple[int, int, int, int], ...]], ...
    ] = (),
    layered_footprints_nm: Tuple[
        Tuple[str, Tuple[Tuple[int, int, int, int, int, int], ...]], ...
    ] = (),
    compress_pairs: bool | None = None,
) -> Full2DCompactionGraph:
    """Build collision obligations without inventing order from a seed.

    A declaration-backed envelope separation may fix one directed axis.  Every
    other interacting pair remains a four-way disjunction; the seed contributes
    only a named search hint and never becomes private ordering authority.
    """

    canonical = _canonical_full2d_boxes(boxes)
    if type(minimum_clearance_nm) is not int or minimum_clearance_nm < 0:
        raise ValueError("full-2D minimum clearance must be an exact nm value")
    if type(grid_nm) is not int or grid_nm <= 0:
        raise ValueError("full-2D manufacturing grid must be positive")
    if any(box[0] % grid_nm or box[1] % grid_nm for box in canonical.values()):
        raise ValueError("full-2D seed origins are off the manufacturing grid")
    if type(footprints_nm) is not tuple:
        raise TypeError("full-2D footprints must be one exact tuple")
    footprint_by_owner = {}
    for row in footprints_nm:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or row[0] not in canonical
            or row[0] in footprint_by_owner
            or type(row[1]) is not tuple
            or not row[1]
        ):
            raise ValueError("full-2D footprint ownership is malformed")
        owner, rectangles = row
        width = canonical[owner][2] - canonical[owner][0]
        height = canonical[owner][3] - canonical[owner][1]
        if rectangles != tuple(sorted(set(rectangles))):
            raise ValueError("full-2D footprint rectangles must be canonical")
        for rectangle in rectangles:
            if (
                type(rectangle) is not tuple
                or len(rectangle) != 4
                or any(type(value) is not int for value in rectangle)
                or not (
                    0 <= rectangle[0] < rectangle[2] <= width
                    and 0 <= rectangle[1] < rectangle[3] <= height
                )
                or rectangle[0] % grid_nm
                or rectangle[1] % grid_nm
            ):
                raise ValueError("full-2D footprint rectangle is malformed or off-grid")
        footprint_by_owner[owner] = rectangles
    if tuple(owner for owner, _rectangles in footprints_nm) != tuple(
        sorted(footprint_by_owner)
    ):
        raise ValueError("full-2D footprints must be canonical and unique")
    if type(layered_footprints_nm) is not tuple:
        raise TypeError("full-2D layered footprints must be one exact tuple")
    layered_by_owner = {}
    for row in layered_footprints_nm:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or row[0] not in canonical
            or row[0] in layered_by_owner
            or type(row[1]) is not tuple
            or not row[1]
        ):
            raise ValueError("full-2D layered footprint ownership is malformed")
        owner, atoms = row
        width = canonical[owner][2] - canonical[owner][0]
        height = canonical[owner][3] - canonical[owner][1]
        if atoms != tuple(sorted(set(atoms))):
            raise ValueError("full-2D layered footprint atoms must be canonical")
        for atom in atoms:
            if (
                type(atom) is not tuple
                or len(atom) != 6
                or any(type(value) is not int for value in atom)
                or atom[0] < 0
                or atom[1] < 0
                or not (
                    0 <= atom[2] < atom[4] <= width
                    and 0 <= atom[3] < atom[5] <= height
                )
                or atom[2] % grid_nm
                or atom[3] % grid_nm
            ):
                raise ValueError(
                    "full-2D layered footprint atom is malformed or off-grid"
                )
        layered_by_owner[owner] = atoms
    if tuple(owner for owner, _atoms in layered_footprints_nm) != tuple(
        sorted(layered_by_owner)
    ):
        raise ValueError("full-2D layered footprints must be canonical and unique")

    atom_boxes = {}
    node_owner_offsets = []
    node_layers = []
    nodes_by_owner = {}
    for owner, master_box in canonical.items():
        width = master_box[2] - master_box[0]
        height = master_box[3] - master_box[1]
        layered = layered_by_owner.get(owner)
        rectangles = (
            tuple(atom[2:] for atom in layered)
            if layered is not None
            else footprint_by_owner.get(owner, ((0, 0, width, height),))
        )
        compound = (
            layered is not None
            or len(rectangles) != 1
            or rectangles[0] != (0, 0, width, height)
        )
        owner_nodes = []
        layer_multiplicity = {
            layer: sum(1 for atom in layered if atom[:2] == layer)
            for layer in {atom[:2] for atom in layered}
        } if layered is not None else {}
        layer_ordinals = {}
        for index, rectangle in enumerate(rectangles):
            layer = None if layered is None else layered[index][:2]
            layer_ordinal = layer_ordinals.get(layer, 0)
            if layer is not None:
                layer_ordinals[layer] = layer_ordinal + 1
            node = (
                owner
                if not compound
                else (
                    (
                        f"@full2d:{owner}:L{layer[0]}:{layer[1]}"
                        if layer_multiplicity[layer] == 1
                        else (
                            f"@full2d:{owner}:L{layer[0]}:{layer[1]}:"
                            f"{layer_ordinal:04d}"
                        )
                    )
                    if layer is not None
                    else f"@full2d:{owner}:{index:04d}"
                )
            )
            if node in atom_boxes or (compound and node in canonical):
                raise ValueError("full-2D footprint node name collides with a master")
            atom_boxes[node] = (
                master_box[0] + rectangle[0],
                master_box[1] + rectangle[1],
                master_box[0] + rectangle[2],
                master_box[1] + rectangle[3],
            )
            node_owner_offsets.append(
                (node, owner, (rectangle[0], rectangle[1]))
            )
            node_layers.append((node, layer))
            owner_nodes.append(node)
        nodes_by_owner[owner] = tuple(owner_nodes)
    atom_boxes = dict(sorted(atom_boxes.items()))
    node_owner_offsets = tuple(sorted(node_owner_offsets))
    node_layers = tuple(sorted(node_layers))
    owner_by_node = {
        node: owner for node, owner, _offset in node_owner_offsets
    }
    offset_by_node = {
        node: offset for node, _owner, offset in node_owner_offsets
    }
    layer_by_node = dict(node_layers)

    def canonical_node_equality(
        first: str, second: str, delta_nm: int
    ) -> Tuple[str, str, int]:
        if first > second:
            first, second, delta_nm = second, first, -delta_nm
        return first, second, int(delta_nm)

    def lower_master_equalities(
        rows: Tuple[Tuple[str, str, int], ...], axis: int
    ) -> Tuple[Tuple[str, str, int], ...]:
        if type(rows) is not tuple:
            raise TypeError("full-2D master equalities must be exact tuples")
        lowered = []
        for row in rows:
            if (
                type(row) is not tuple
                or len(row) != 3
                or type(row[0]) is not str
                or type(row[1]) is not str
                or row[0] >= row[1]
                or row[0] not in canonical
                or row[1] not in canonical
                or type(row[2]) is not int
                or row[2] % grid_nm
            ):
                raise ValueError("full-2D master equality is malformed or off-grid")
            first_node = nodes_by_owner[row[0]][0]
            second_node = nodes_by_owner[row[1]][0]
            lowered.append(canonical_node_equality(
                first_node,
                second_node,
                row[2]
                + offset_by_node[second_node][axis]
                - offset_by_node[first_node][axis],
            ))
        for owner in sorted(nodes_by_owner):
            first_node = nodes_by_owner[owner][0]
            for second_node in nodes_by_owner[owner][1:]:
                lowered.append(canonical_node_equality(
                    first_node,
                    second_node,
                    offset_by_node[second_node][axis]
                    - offset_by_node[first_node][axis],
                ))
        return tuple(sorted(set(lowered)))

    lowered_x_equalities = lower_master_equalities(x_equalities, 0)
    lowered_y_equalities = lower_master_equalities(y_equalities, 1)
    if type(pair_axis_authority) is not tuple:
        raise TypeError("full-2D pair axis authority must be one exact tuple")
    authority_rows = []
    for row in pair_axis_authority:
        if (
            type(row) is not tuple
            or len(row) != 6
            or type(row[0]) is not str
            or type(row[1]) is not str
            or row[0] >= row[1]
            or row[0] not in canonical
            or row[1] not in canonical
            or row[2] not in {"horizontal", "vertical"}
            or type(row[3]) is not str
            or type(row[4]) is not str
            or set(row[3:5]) != set(row[:2])
            or row[3] == row[4]
            or type(row[5]) is not tuple
            or not row[5]
            or row[5] != tuple(sorted(set(row[5])))
            or any(type(source_id) is not str or not source_id for source_id in row[5])
        ):
            raise ValueError("full-2D pair axis authority is malformed")
        authority_rows.append(row)
    if (
        tuple(authority_rows) != tuple(sorted(authority_rows))
        or len(authority_rows) != len(set(authority_rows))
        or len({row[:2] for row in authority_rows}) != len(authority_rows)
    ):
        raise ValueError("full-2D pair axis authority must be canonical and unique")
    authority_by_pair = {
        row[:2]: (row[2], row[3], row[4], row[5]) for row in authority_rows
    }
    if type(pair_axis_preference) is not tuple:
        raise TypeError("full-2D pair axis preference must be one exact tuple")
    preference_rows = []
    for row in pair_axis_preference:
        if (
            type(row) is not tuple
            or len(row) != 3
            or type(row[0]) is not str
            or type(row[1]) is not str
            or row[0] >= row[1]
            or row[0] not in canonical
            or row[1] not in canonical
            or row[2] not in {"horizontal", "vertical"}
        ):
            raise ValueError("full-2D pair axis preference is malformed")
        preference_rows.append(row)
    if (
        tuple(preference_rows) != tuple(sorted(preference_rows))
        or len(preference_rows) != len(set(preference_rows))
        or len({row[:2] for row in preference_rows}) != len(preference_rows)
    ):
        raise ValueError("full-2D pair axis preference must be canonical and unique")
    preference_by_pair = {row[:2]: row[2] for row in preference_rows}
    if type(overlap_compatible_pairs) is not tuple:
        raise TypeError("full-2D overlap-compatible pairs must be one exact tuple")
    compatible_rows = []
    for pair in overlap_compatible_pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or pair[0] >= pair[1]
            or pair[0] not in canonical
            or pair[1] not in canonical
        ):
            raise ValueError("full-2D overlap-compatible pair is malformed")
        compatible_rows.append(pair)
    compatible_pairs = tuple(compatible_rows)
    if (
        compatible_pairs != tuple(sorted(set(compatible_pairs)))
        or set(compatible_pairs).intersection(authority_by_pair)
    ):
        raise ValueError(
            "full-2D overlap-compatible pairs must be canonical, unique, and "
            "free of separation-axis authority"
        )
    compatible_set = set(compatible_pairs)
    physical_pair_cover = ()
    physical_atom_pair_count = 0
    compressed = not authority_by_pair and (len(atom_boxes) >= 500 if compress_pairs is None else compress_pairs)
    if type(compressed) is not bool:
        raise TypeError("full-2D compressed-pair selection must be boolean")
    if compressed:
        from clay.layout.extract.full2d_geometry_cover import build_geometry_cover, cover_collisions
        physical_pair_cover, interaction_pairs, physical_atom_pair_count = build_geometry_cover(
            tuple((node,(box[2]-box[0],box[3]-box[1])) for node,box in atom_boxes.items()),
            node_owner_offsets,node_layers,minimum_clearance_nm,compatible_pairs)
        bad_pair = next(cover_collisions(physical_pair_cover,canonical),None)
        if bad_pair is not None:
            raise ValueError(f"full-2D seed is not clearance-legal for pair {bad_pair[0]!r}, {bad_pair[1]!r}")
    else:
        interaction_pairs = _iter_full2d_collision_pairs(atom_boxes, owner_by_node, layer_by_node, compatible_set)
    pair_constraints = []
    for first, second in interaction_pairs:
        first_owner = owner_by_node[first]
        second_owner = owner_by_node[second]
        if first_owner == second_owner:
            continue
        if tuple(sorted((first_owner, second_owner))) in compatible_set:
            continue
        owner_pair = tuple(sorted((first_owner, second_owner)))
        authority = authority_by_pair.get(owner_pair)
        preference = preference_by_pair.get(owner_pair)
        if (
            layer_by_node[first] is not None
            and layer_by_node[second] is not None
            and layer_by_node[first] != layer_by_node[second]
        ):
            continue
        a = atom_boxes[first]
        b = atom_boxes[second]
        x_candidates = []
        if a[2] + minimum_clearance_nm <= b[0]:
            x_candidates.append((first, second))
        if b[2] + minimum_clearance_nm <= a[0]:
            x_candidates.append((second, first))
        y_candidates = []
        if a[3] + minimum_clearance_nm <= b[1]:
            y_candidates.append((first, second))
        if b[3] + minimum_clearance_nm <= a[1]:
            y_candidates.append((second, first))
        if authority is not None:
            axis, lower_owner, upper_owner, source_constraint_ids = authority
            lower = first if first_owner == lower_owner else second
            upper = second if second_owner == upper_owner else first
            pair_constraints.append(Full2DPairConstraint(
                first=min(first, second),
                second=max(first, second),
                clearance_nm=0,
                mode="fixed_edge",
                axis=axis,
                lower=lower,
                upper=upper,
                authority_kind="declared_hard",
                source_constraint_ids=source_constraint_ids,
            ))
        if not x_candidates and not y_candidates:
            if authority is None:
                raise ValueError(
                    "full-2D seed is not clearance-legal for pair "
                    f"{first_owner!r}, {second_owner!r}"
                )
            seed_hint = ()
        elif preference == "vertical" and y_candidates:
            lower, upper = y_candidates[0]
            hint_axis = "vertical"
            seed_hint = (
                lower,
                upper,
                hint_axis,
                f"implicit_seed_axis:{lower}->{upper}:{hint_axis}",
            )
        elif preference == "horizontal" and x_candidates:
            lower, upper = x_candidates[0]
            hint_axis = "horizontal"
            seed_hint = (
                lower,
                upper,
                hint_axis,
                f"implicit_seed_axis:{lower}->{upper}:{hint_axis}",
            )
        elif x_candidates:
            lower, upper = x_candidates[0]
            hint_axis = "horizontal"
            seed_hint = (
                lower,
                upper,
                hint_axis,
                f"implicit_seed_axis:{lower}->{upper}:{hint_axis}",
            )
        else:
            lower, upper = y_candidates[0]
            hint_axis = "vertical"
            seed_hint = (
                lower,
                upper,
                hint_axis,
                f"implicit_seed_axis:{lower}->{upper}:{hint_axis}",
            )
        pair_constraints.append(Full2DPairConstraint(
            first=min(first, second),
            second=max(first, second),
            clearance_nm=minimum_clearance_nm,
            mode="four_way_disjunction",
            authority_kind="physical_nonoverlap",
            source_constraint_ids=("physical_nonoverlap",),
            seed_hint=seed_hint,
        ))
    return Full2DCompactionGraph(
        nodes=tuple(atom_boxes),
        sizes_nm=tuple(
            (
                name,
                (box[2] - box[0], box[3] - box[1]),
            )
            for name, box in atom_boxes.items()
        ),
        master_sizes_nm=tuple(
            (name, (box[2] - box[0], box[3] - box[1]))
            for name, box in canonical.items()
        ),
        node_owner_offsets_nm=node_owner_offsets,
        node_layers=node_layers,
        pair_constraints=tuple(sorted(pair_constraints)),
        x_equalities=lowered_x_equalities,
        y_equalities=lowered_y_equalities,
        overlap_compatible_pairs=compatible_pairs,
        grid_nm=grid_nm,
        physical_pair_cover=physical_pair_cover,
        physical_clearance_nm=minimum_clearance_nm if compressed else None,
        physical_atom_pair_count=physical_atom_pair_count,
    )


def compact_full2d_master_placement(
    boxes: Mapping[str, Tuple[int, int, int, int]],
    graph: Full2DCompactionGraph,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Solve the fixed H/V graph on the manufacturing grid by longest paths."""

    canonical = _canonical_full2d_boxes(boxes)
    if type(graph) is not Full2DCompactionGraph:
        raise TypeError("full-2D compactor requires one exact graph")
    if any(
        row.mode == "four_way_disjunction" for row in graph.pair_constraints
    ):
        raise ValueError(
            "full-2D longest-path compactor cannot freeze live pair disjunctions"
        )
    if tuple(canonical) != graph.master_nodes:
        raise ValueError("full-2D compactor boxes differ from graph masters")
    if tuple(
        (name, (box[2] - box[0], box[3] - box[1]))
        for name, box in canonical.items()
    ) != graph.master_sizes_nm:
        raise ValueError("full-2D compactor boxes differ from graph master sizes")
    sizes = dict(graph.sizes_nm)

    def axis_coordinates(axis: int) -> Dict[str, int]:
        equalities = graph.x_equalities if axis == 0 else graph.y_equalities
        edges = graph.horizontal_edges if axis == 0 else graph.vertical_edges
        uf = _OffsetUF(list(graph.nodes))
        for first, second, delta in equalities:
            if not uf.union(first, second, float(delta)):
                raise ValueError("full-2D compactor has conflicting equality offsets")
        representatives = {name: uf.find(name) for name in graph.nodes}
        roots = sorted({root for root, _offset in representatives.values()})
        members = {
            root: tuple(
                name for name in graph.nodes if representatives[name][0] == root
            )
            for root in roots
        }
        local_offsets = {
            node: offset for node, _owner, offset in graph.node_owner_offsets_nm
        }
        base = {
            root: max(
                local_offsets[name][axis] - representatives[name][1]
                for name in members[root]
            )
            for root in roots
        }
        weighted = []
        for edge in edges:
            lower_root, lower_offset = representatives[edge.lower]
            upper_root, upper_offset = representatives[edge.upper]
            required = (
                lower_offset
                + sizes[edge.lower][axis]
                + edge.clearance_nm
                - upper_offset
            )
            if lower_root == upper_root:
                if required > 1e-9:
                    raise ValueError(
                        "full-2D equality contradicts a separation edge"
                    )
                continue
            weighted.append((lower_root, upper_root, required))
        for iteration in range(len(roots) + 1):
            changed = False
            for lower_root, upper_root, required in weighted:
                target = base[lower_root] + required
                target = (
                    int(target + graph.grid_nm - 1) // graph.grid_nm
                ) * graph.grid_nm
                if base[upper_root] < target:
                    base[upper_root] = float(target)
                    changed = True
            if not changed:
                break
            if iteration == len(roots):
                raise ValueError("full-2D compactor found a positive cycle")
        result = {}
        for name in graph.nodes:
            root, offset = representatives[name]
            value = base[root] + offset
            rounded = round(value)
            if abs(value - rounded) > 1e-9 or int(rounded) % graph.grid_nm:
                raise ValueError("full-2D compacted origin escaped the grid")
            result[name] = int(rounded)
        shift = min(
            result[name] - local_offsets[name][axis] for name in graph.nodes
        )
        if shift % graph.grid_nm:
            raise ValueError("full-2D compacted wall escaped the grid")
        return {name: value - shift for name, value in result.items()}

    x = axis_coordinates(0)
    y = axis_coordinates(1)
    owner_origins: Dict[str, set[Tuple[int, int]]] = {
        owner: set() for owner in graph.master_nodes
    }
    for node, owner, offset in graph.node_owner_offsets_nm:
        owner_origins[owner].add((x[node] - offset[0], y[node] - offset[1]))
    if any(len(origins) != 1 for origins in owner_origins.values()):
        raise ValueError("full-2D rigid component origins diverged")
    master_sizes = dict(graph.master_sizes_nm)
    rebuilt = {}
    for owner in graph.master_nodes:
        origin_x, origin_y = next(iter(owner_origins[owner]))
        width, height = master_sizes[owner]
        rebuilt[owner] = (
            origin_x,
            origin_y,
            origin_x + width,
            origin_y + height,
        )
    if not graph.replay(rebuilt):
        raise ValueError("full-2D compacted placement failed graph replay")
    return rebuilt


class _OffsetUF:
    """Union-find with an offset to the root: coord(n) = coord(root) + off[n]."""

    def __init__(self, names: List[str]) -> None:
        self.parent: Dict[str, str] = {n: n for n in names}
        self.off: Dict[str, float] = {n: 0.0 for n in names}

    def find(self, n: str) -> Tuple[str, float]:
        if self.parent[n] == n:
            return n, 0.0
        root, up = self.find(self.parent[n])
        self.parent[n] = root
        self.off[n] += up
        return root, self.off[n]

    def union(self, a: str, b: str, delta: float) -> bool:
        """Pin ``coord(b) - coord(a) = delta``. Returns False on a
        CONFLICTING existing pin (caller decides how to fail honestly)."""
        ra, oa = self.find(a)
        rb, ob = self.find(b)
        if ra == rb:
            return abs((ob - oa) - delta) <= 1e-6
        # deterministic: smaller root name stays canonical.
        if ra < rb:
            self.parent[rb] = ra
            self.off[rb] = oa + delta - ob
        else:
            self.parent[ra] = rb
            self.off[ra] = ob - delta - oa
        return True


def _axis_pass(
    pos: Dict[str, _XY],
    sizes: Mapping[str, _XY],
    spec: CompactionSpec,
    axis: int,
    min_gap: float,
    band_channel: float,
    channel_um_per_gap: Optional[Mapping[int, float]],
) -> bool:
    """One longest-path press + minimal-perturbation clamp on ``axis``
    (0 = x, 1 = y). Mutates ``pos``; returns True iff anything moved."""
    perp = 1 - axis
    names = sorted(pos)

    def half(n: str, ax: int) -> float:
        return float(sizes.get(n, (0.0, 0.0))[ax]) / 2.0

    def is_rail(n: str) -> bool:
        return spec.row_of[n] in spec.rail_rows

    # rails are full-width corridor strips: they ride the y graph (the
    # corridor must keep its seam relations) but are PINNED in x (a corridor
    # has no x freedom to press; its consumer re-reads its solved x).
    movable = [n for n in names if axis == 1 or not is_rail(n)]
    if len(movable) < 2:
        return False

    def boundary_channel(k: int) -> float:
        if k in spec.rail_rows or (k + 1) in spec.rail_rows:
            return 0.0  # P2: reservation seams are flush
        ch = float(band_channel)
        if channel_um_per_gap is not None:
            ch = max(ch, float(channel_um_per_gap.get(k, band_channel)))
        return ch + float(spec.slack_um)

    def required_sep(a: str, b: str) -> float:
        halves = half(a, axis) + half(b, axis)
        if axis == 0:
            return halves + float(min_gap)
        ra, rb = spec.row_of[a], spec.row_of[b]
        if ra == rb:
            return halves + float(min_gap)
        lo, hi = (ra, rb) if ra <= rb else (rb, ra)
        return halves + sum(boundary_channel(k) for k in range(lo, hi))

    def interact(a: str, b: str) -> bool:
        """Perpendicular spans, grown by min_gap/2 each, strictly overlap."""
        g = float(min_gap) / 2.0
        lo = max(pos[a][perp] - half(a, perp) - g,
                 pos[b][perp] - half(b, perp) - g)
        hi = min(pos[a][perp] + half(a, perp) + g,
                 pos[b][perp] + half(b, perp) + g)
        return (hi - lo) > _OVERLAP_TOL

    # EQUALITY groups: same-row partner pairs pin their input offset.
    uf = _OffsetUF(movable)
    movable_set = set(movable)
    for a, b in spec.partners:
        if a in movable_set and b in movable_set:
            delta = pos[b][axis] - pos[a][axis]
            if not uf.union(a, b, delta):
                raise ValueError(
                    f"compaction: conflicting equality intent for partner "
                    f"pair ({a}, {b}) on axis {axis}"
                )

    rep: Dict[str, Tuple[str, float]] = {n: uf.find(n) for n in movable}
    groups = sorted({r for r, _o in rep.values()})

    # separation edges between interacting pairs. Direction: INPUT order,
    # EXCEPT a cross-row pair on the y axis in mode='2d' -- there the ROW
    # ORDER is the hard intent edge (a scrambled input is pressed back into
    # row order, never legalized inverted). Same-row pairs and the x axis
    # keep the input direction (the press never reorders a row).
    order_by_row = (axis == 1 and spec.mode == "2d")
    # Equality merging makes the group graph a GENERAL difference-constraint
    # graph (a partner pair straddling a third node yields a benign cycle
    # group -> mid -> group whose total weight is non-positive exactly when
    # the pinned offset already accommodates the mid node), so the bounds
    # are solved by Bellman-Ford-style relaxation, not a topological order;
    # a POSITIVE cycle is the genuine inconsistency and raises.
    edge_w: Dict[Tuple[str, str], float] = {}
    for i in range(len(movable)):
        for j in range(i + 1, len(movable)):
            a, b = movable[i], movable[j]
            if not interact(a, b):
                continue
            # order by input coordinate on the axis (tie by name); in '2d'
            # a cross-row y pair is ordered by the ROW indices instead.
            if order_by_row and spec.row_of[a] != spec.row_of[b]:
                if spec.row_of[a] < spec.row_of[b]:
                    lo_n, hi_n = a, b
                else:
                    lo_n, hi_n = b, a
            elif (pos[a][axis], a) <= (pos[b][axis], b):
                lo_n, hi_n = a, b
            else:
                lo_n, hi_n = b, a
            sep = required_sep(lo_n, hi_n)
            g_lo, o_lo = rep[lo_n]
            g_hi, o_hi = rep[hi_n]
            if g_lo == g_hi:
                # same equality group: consistency check only, margin-free
                # (the declared partner offset outranks the margin).
                if (pos[hi_n][axis] - pos[lo_n][axis]) < sep - 1e-6:
                    raise ValueError(
                        f"compaction: equality intent contradicts the "
                        f"separation of ({lo_n}, {hi_n}) on axis {axis}"
                    )
                continue
            s = sep + o_lo - o_hi
            key = (g_lo, g_hi)
            if key not in edge_w or edge_w[key] < s:
                edge_w[key] = s
    edges = sorted(edge_w.items())  # deterministic relaxation order

    # group member lists (for wall bounds / extent).
    members: Dict[str, List[str]] = {g: [] for g in groups}
    for n in movable:
        members[rep[n][0]].append(n)

    # the wall: the input outline's low edge on this axis over the MOVABLE
    # nodes (on the x axis the pinned full-width rail strips must NOT drag
    # the wall outward -- that would press devices toward the rail edge, a
    # pure perturbation with zero area gain).
    wall = min(pos[n][axis] - half(n, axis) for n in movable)

    def _relax(vals: Dict[str, float], push_up: bool) -> None:
        """Longest-path relaxation to a fixed point (<= |V| rounds on a
        consistent system); a still-relaxing |V|+1-th round is a positive
        cycle -- conflicting intent, refused honestly."""
        for _round in range(len(groups) + 1):
            changed = False
            for (g, h), s in edges:
                if push_up:
                    if vals[h] < vals[g] + s - 1e-12:
                        vals[h] = vals[g] + s
                        changed = True
                else:
                    if vals[g] > vals[h] - s + 1e-12:
                        vals[g] = vals[h] - s
                        changed = True
            if not changed:
                return
        raise ValueError(
            "compaction: positive separation cycle through an equality "
            "group (conflicting intent) -- refusing to compact"
        )

    # LOWER bounds: longest path up from the wall.
    low: Dict[str, float] = {
        g: max(wall + half(m, axis) - rep[m][1] for m in members[g])
        for g in groups
    }
    _relax(low, push_up=True)

    # the compacted extent top (over movable nodes; pinned rails on the x
    # axis are outside the press and keep their own extent). Never beyond
    # the input top (the input is feasible, so the press only shrinks).
    top_star = max(low[g] + rep[m][1] + half(m, axis)
                   for g in groups for m in members[g])
    input_top = max(pos[m][axis] + half(m, axis) for m in movable)
    top_star = min(top_star, input_top)

    # UPPER bounds: reverse longest path down from the compacted top.
    up: Dict[str, float] = {
        g: min(top_star - half(m, axis) - rep[m][1] for m in members[g])
        for g in groups
    }
    _relax(up, push_up=False)

    # MINIMAL-PERTURBATION realization: clamp every group toward its input
    # coordinate inside [low, up], then push-up relaxation to feasibility
    # (push-ups stay <= up by construction: x_g <= up_g and up respects
    # every edge, so x_g + s <= up_h). Deterministic, monotone, terminates.
    final: Dict[str, float] = {
        g: min(max(pos[g][axis], low[g]), up[g]) for g in groups
    }
    _relax(final, push_up=True)

    moved = False
    for n in movable:
        g, off = rep[n]
        new = final[g] + off
        if abs(new - pos[n][axis]) > _MOVE_TOL:
            moved = True
        pos[n] = (new, pos[n][1]) if axis == 0 else (pos[n][0], new)
    return moved


def compact_placement(
    placement: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    spec: CompactionSpec,
    *,
    min_gap: float = 1.0,
    band_channel: float = 1.0,
    channel_um_per_gap: Optional[Mapping[int, float]] = None,
    max_rounds: int = 4,
) -> Dict[str, _XY]:
    """Compact a solved placement's outline -- x/y alternating constraint-graph
    press to a fixed point, intent edges preserved by construction (see the
    module docstring for the edge inventory).

    ``placement``: name -> centroid (um), the band solve's output (rail
    pseudo rows included). ``sizes``: name -> (w, h) um, the SAME sizes the
    solve placed (rail pseudo sizes included). ``spec``: the design-derived
    :class:`CompactionSpec`. ``min_gap`` / ``band_channel`` /
    ``channel_um_per_gap``: the EXACT values the band solve enforced (single
    source -- the caller passes its own parameters through, so the press can
    never undercut what the solve guaranteed).

    Returns a NEW placement dict; the input is not mutated. Every name in
    ``placement`` must be covered by ``spec.row_of`` (a missing name is a
    contract violation, raised -- never silently skipped).
    """
    out: Dict[str, _XY] = {str(n): (float(p[0]), float(p[1]))
                           for n, p in placement.items()}
    if len(out) < 2:
        return out
    missing = sorted(n for n in out if n not in spec.row_of)
    if missing:
        raise ValueError(
            f"compaction: placement names missing from the spec row "
            f"partition: {missing}"
        )
    for _ in range(max(1, int(max_rounds))):
        moved_y = _axis_pass(out, sizes, spec, 1, min_gap, band_channel,
                             channel_um_per_gap)
        moved_x = _axis_pass(out, sizes, spec, 0, min_gap, band_channel,
                             channel_um_per_gap)
        if not (moved_y or moved_x):
            break
    return out


# --------------------------------------------------------------------------- #
# EMBED SEARCH (mode='2d'): deterministic cross-row x-repack local search.     #
#                                                                              #
# WHY A SEARCH AT ALL                                                          #
# ===================                                                          #
# The press is x-overlap-aware, but it PRESSES -- it never moves a device in   #
# x. Measured on the test4 bench: the third style row's two members (14.2 um   #
# of devices) sat on the critical y chain (cap row -> third row -> top row),   #
# so the whole outline carried their strip even though both flanks were        #
# empty (~920 um^2 of holes). No y press can fix that; the members must       #
# RELOCATE in x to a column whose support is short -- a composite move:        #
# open a slot in another row's x-pack AND move the group there, then let the   #
# press drop everything. Single-coordinate gradient steps and single-group     #
# greedy x moves both fail (measured: zero accepted moves); the joint          #
# "insert into a row's x-pack" move is the smallest action that works.        #
#                                                                              #
# THE MOVE (generic -- rows/groups/slots come from the spec, never a name):    #
# for each partner-rigid GROUP g (partner closure of the declared pairs; a     #
# group moves as one rigid unit so the symmetric offset survives exactly),     #
# for each device row r (rails excluded -- a corridor hosts nothing), for      #
# each insertion slot in r's group sequence: re-pack r's groups + g jointly    #
# left-to-right at the row's current left edge (x only -- y is the press's     #
# job), press, and score by the pressed DEVICE outline area. The best          #
# strictly-improving move of the round is taken (ties -> first in the          #
# sorted enumeration); repeat to a fixed point or the round cap.               #
#                                                                              #
# WHAT IT PRESERVES BY CONSTRUCTION: row MEMBERSHIP (row_of never changes --   #
# g keeps every cross-row order edge where it x-overlaps), partner offsets     #
# (rigid moves + the press equality edges), channels and rail corridors        #
# (the press graph), legality (every trial is press output), and -- with S6    #
# channel data -- every crossing net's X-RESOLVED LANE: a trial whose pressed  #
# result sinks a device into a seam's x segment owed to crossing nets is       #
# refused (_channel_capacity_violations; the pairwise press edges cannot see   #
# this, because the sunken device is x-disjoint from the row that defines      #
# the lane). WHAT IT GIVES UP, honestly: the within-row x ORDER of the         #
# repacked rows (recorded in the returned notes -- the row weakens to a        #
# relative-order constraint, which is exactly the mode='2d' contract).         #
#                                                                              #
# DETERMINISM + TERMINATION: sorted enumeration, strict-improvement            #
# acceptance (>= _EMBED_MIN_GAIN_UM2), bounded rounds.                         #
# --------------------------------------------------------------------------- #

# minimum outline-area gain (um^2) for an embed move to be accepted -- floats
# below this are noise, and strict improvement is the termination argument.
_EMBED_MIN_GAIN_UM2: float = 1e-6


def _device_outline_area(pos: Mapping[str, _XY], sizes: Mapping[str, _XY],
                         spec: CompactionSpec) -> float:
    """Bounding-box area (um^2) over the DEVICE rows (rail pseudo rows are
    full-width fixtures; counting them would mask every device-side gain)."""
    xs0: List[float] = []; ys0: List[float] = []
    xs1: List[float] = []; ys1: List[float] = []
    for n, (cx, cy) in pos.items():
        if spec.row_of[n] in spec.rail_rows:
            continue
        w, h = float(sizes.get(n, (0.0, 0.0))[0]), \
            float(sizes.get(n, (0.0, 0.0))[1])
        xs0.append(cx - w / 2.0); xs1.append(cx + w / 2.0)
        ys0.append(cy - h / 2.0); ys1.append(cy + h / 2.0)
    if not xs0:
        return 0.0
    return (max(xs1) - min(xs0)) * (max(ys1) - min(ys0))


def _channel_capacity_violations(
    pos: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    spec: CompactionSpec,
    channel_um_per_gap: Optional[Mapping[int, float]],
) -> frozenset:
    """X-RESOLVED CHANNEL CAPACITY (the S6 seam account, resolved over x).

    The band stack accounts each row-boundary seam as ONE scalar: crossing-
    net count x track (``channel_um_per_gap[k]``). That accounting assumes
    the seam is a full-width strip; mode='2d' retires exactly that
    assumption, so a relocated/pressed device can consume the lane's x
    segment while the scalar account stays satisfied -- and a crossing net
    loses its corridor (measured: t4 channel-aware + no_route_under, VOUT
    open behind a row-3 member pressed into its last left-side corridor).

    The obligation, per non-rail boundary k with S6 data: each crossing net
    (``spec.channel_xnets[k]`` -- the SAME derivation that produced the
    sketch's ``channel_demands``, names only) lays its demand over its pin
    x interval (the union x span of its member devices in THIS placement).
    Over that interval the free y window between the rows -- below: the
    highest top among rows <= k x-overlapping the interval; above: the
    lowest bottom among rows >= k+1 x-overlapping it -- must fit the seam
    demand where the net runs: max over the interval of (number of crossing
    nets covering x) x track_k, with ``track_k = channel_um_per_gap[k] /
    len(channel_xnets[k])`` (the scalar account split over its own nets --
    no second source). The band stack satisfies this BY CONSTRUCTION (the
    seam is a full-width strip of at least the scalar account), so only a
    press/embed that sinks a device into a seam's x segment can violate it.

    Returns the violated ``(gap_index, net_position)`` pairs. No S6 data
    (``channel_um_per_gap`` falsy or ``channel_xnets`` empty) returns the
    empty set: the per-net demand in um is unmeasurable then, and the
    scalar band channel stays enforced pairwise by the press edges (the
    historical path, byte-identical)."""
    xnets = tuple(getattr(spec, "channel_xnets", ()) or ())
    if not xnets or not channel_um_per_gap:
        return frozenset()

    def bbox(n: str) -> Tuple[float, float, float, float]:
        cx, cy = pos[n]
        w, h = sizes.get(n, (0.0, 0.0))
        return (cx - float(w) / 2.0, cy - float(h) / 2.0,
                cx + float(w) / 2.0, cy + float(h) / 2.0)

    names = sorted(n for n in pos if n in spec.row_of)
    bad = set()
    for k, nets in enumerate(xnets):
        if not nets or k in spec.rail_rows or (k + 1) in spec.rail_rows:
            continue
        ch = float(channel_um_per_gap.get(k, 0.0) or 0.0)
        if not ch > 0.0:
            continue
        track = ch / float(len(nets))
        # pin x interval per crossing net, from the placement under check
        # (the candidate's own geometry -- never a stored coordinate).
        spans: List[Optional[Tuple[float, float]]] = []
        for members in nets:
            bbs = [bbox(m) for m in members if m in pos]
            spans.append((min(b[0] for b in bbs), max(b[2] for b in bbs))
                         if bbs else None)
        cuts = sorted({e for s in spans if s is not None for e in s})
        for i, iv in enumerate(spans):
            if iv is None:
                continue
            lo, hi = iv
            floor: Optional[float] = None
            ceil: Optional[float] = None
            for n in names:
                b = bbox(n)
                if min(hi, b[2]) - max(lo, b[0]) <= _OVERLAP_TOL:
                    continue
                if spec.row_of[n] <= k:
                    floor = b[3] if floor is None else max(floor, b[3])
                else:
                    ceil = b[1] if ceil is None else min(ceil, b[1])
            if floor is None or ceil is None:
                continue  # one side open to the outline: nothing to squeeze
            # the worst concurrent crossing demand over this net's interval.
            need = 1
            for a, b2 in zip(cuts, cuts[1:]):
                mid = (a + b2) / 2.0
                if not (lo - _OVERLAP_TOL) < mid < (hi + _OVERLAP_TOL):
                    continue
                cov = sum(
                    1 for s in spans if s is not None
                    and s[0] - _OVERLAP_TOL < mid < s[1] + _OVERLAP_TOL
                )
                need = max(need, cov)
            if (ceil - floor) < need * track - 1e-6:
                bad.add((k, i))
    return frozenset(bad)


def _keepout_flank_violations(
    pos: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    spec: CompactionSpec,
    band_channel: float,
    channel_um_per_gap: Optional[Mapping[int, float]],
) -> Dict[Tuple[str, str], float]:
    """KEEPOUT FLANK obligation (the @no_route_under corridor account).

    A hard keep-out footprint blocks the routing layers over its whole
    placed bbox; every net that must pass or EXIT around it (boundary io
    stubs included -- the t4 evidence: a width-shrinking embed left the cap
    1.74 um from the design extent and the VOUT boundary stub landed inside
    the keep-out, an unroutable pin) flows through the x FLANKS between the
    footprint and the design extent. The obligation: each flank of each
    ``spec.keepout_names`` device must be at least the boundary-channel
    class of the keepout's ADJACENT seams (``max(band_channel,
    channel_um_per_gap[k])`` over k in {row-1, row} -- the SAME values the
    press enforces, single source; no S6 data means the band-channel floor
    only: the guard strengthens with data, never invents a number).

    Returns ``{(name, 'left'|'right'): measured_flank_um}`` for the flanks
    below their requirement; empty when clear or when the spec carries no
    keepout names (vacuous -- the historical path). The consumer compares
    against a baseline (monotone non-worsening), so a design whose band
    stack is already tight is not deadlocked. Sub-bbox effects (pad-inset
    pin extents) are below bbox resolution: this is a declared conservative
    corridor floor, not a reachability proof."""
    keepouts = tuple(getattr(spec, "keepout_names", ()) or ())
    keepouts = tuple(n for n in keepouts if n in pos)
    if not keepouts:
        return {}

    def x_span(n: str) -> Tuple[float, float]:
        cx = pos[n][0]
        w = float(sizes.get(n, (0.0, 0.0))[0])
        return (cx - w / 2.0, cx + w / 2.0)

    dev = [n for n in sorted(pos)
           if spec.row_of[n] not in spec.rail_rows]
    if not dev:
        return {}
    ext_lo = min(x_span(n)[0] for n in dev)
    ext_hi = max(x_span(n)[1] for n in dev)
    n_gaps = max(spec.row_of[n] for n in spec.row_of)

    def required(name: str) -> float:
        r = spec.row_of[name]
        req = float(band_channel)
        for k in (r - 1, r):
            if not 0 <= k < n_gaps:
                continue
            if k in spec.rail_rows or (k + 1) in spec.rail_rows:
                continue
            ch = float(band_channel)
            if channel_um_per_gap is not None:
                ch = max(ch, float(channel_um_per_gap.get(k, band_channel)))
            req = max(req, ch)
        return req

    bad: Dict[Tuple[str, str], float] = {}
    for name in keepouts:
        lo, hi = x_span(name)
        req = required(name)
        left = lo - ext_lo
        right = ext_hi - hi
        if left < req - 1e-6:
            bad[(name, "left")] = left
        if right < req - 1e-6:
            bad[(name, "right")] = right
    return bad


def _partner_groups(spec: CompactionSpec) -> Dict[int, List[List[str]]]:
    """Per device row: the partner-closure groups (sorted, deterministic).
    Every name in a group shares the row (the spec's same-row partner
    contract); a group is the atomic unit of an embed move."""
    parent: Dict[str, str] = {n: n for n in spec.row_of}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in spec.partners:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    rows: Dict[int, Dict[str, List[str]]] = {}
    for n in sorted(spec.row_of):
        k = spec.row_of[n]
        if k in spec.rail_rows:
            continue
        rows.setdefault(k, {}).setdefault(find(n), []).append(n)
    return {k: [grp[r] for r in sorted(grp)] for k, grp in sorted(rows.items())}


def embed_search(
    placement: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    spec: CompactionSpec,
    *,
    min_gap: float = 1.0,
    band_channel: float = 1.0,
    channel_um_per_gap: Optional[Mapping[int, float]] = None,
    max_rounds: int = 8,
) -> Tuple[Dict[str, _XY], Tuple[str, ...]]:
    """The mode='2d' embed stage: deterministic cross-row x-repack local
    search over press-legalized trials (see the block comment above).

    Returns ``(placement, notes)`` -- a NEW placement dict plus one honest
    note per ACCEPTED move (empty notes = the input was already a fixed
    point). ``min_gap`` / ``band_channel`` / ``channel_um_per_gap`` are the
    EXACT solve parameters (single source, same contract as
    :func:`compact_placement`); the same ``channel_um_per_gap`` /
    ``band_channel`` also feed the two x-resolved ROUTING-ROOM refusals
    (monotone against the pressed baseline -- if it does not fit, it is
    not embedded): the X-RESOLVED CHANNEL CAPACITY check
    (:func:`_channel_capacity_violations`; a trial may not sink a device
    into a lane x segment owed to crossing nets) and the KEEPOUT FLANK
    check (:func:`_keepout_flank_violations`; a trial may not pin a hard
    @no_route_under footprint against the design extent below its
    boundary-channel class).

    A conservative routing margin rides the CHAIN's spacing parameters
    (``min_gap`` / ``band_channel`` / the caller's sketch ``dev_gap_um``),
    which this search inherits verbatim -- see :class:`CompactionSpec` on
    why a press-local margin was refused."""
    def press(p: Mapping[str, _XY]) -> Dict[str, _XY]:
        return compact_placement(
            p, sizes, spec, min_gap=min_gap, band_channel=band_channel,
            channel_um_per_gap=channel_um_per_gap,
        )

    best = press(placement)
    best_area = _device_outline_area(best, sizes, spec)
    groups_by_row = _partner_groups(spec)
    notes: List[str] = []

    # X-RESOLVED CHANNEL CAPACITY baseline (S6): the band-stacked press
    # input satisfies every crossing net's lane obligation by construction;
    # a trial may never ADD a violation beyond this baseline (monotone
    # non-worsening -- see :func:`_channel_capacity_violations`).
    base_capacity = _channel_capacity_violations(
        best, sizes, spec, channel_um_per_gap)
    # KEEPOUT FLANK baseline (@no_route_under corridors): a trial may not
    # squeeze a keep-out footprint's flank below its channel class unless
    # the baseline was no wider there (monotone non-worsening -- see
    # :func:`_keepout_flank_violations`).
    base_flank = _keepout_flank_violations(
        best, sizes, spec, band_channel, channel_um_per_gap)

    def flank_ok(p: Mapping[str, _XY]) -> bool:
        bad = _keepout_flank_violations(
            p, sizes, spec, band_channel, channel_um_per_gap)
        return all(
            k in base_flank and v >= base_flank[k] - 1e-6
            for k, v in bad.items()
        )

    # CORRIDOR HARD BOUND: a rail pseudo row is a THROUGH-GOING corridor --
    # its x span (pinned by the press) must cover the whole device extent,
    # or the trunk contract breaks where a device sticks out past the
    # corridor end. The graph holds this only by x-overlap, so an x-repack
    # that pushes a device beyond a rail span must be REFUSED (the round's
    # risk guardrail: if it does not fit, it is not embedded).
    rail_spans: List[Tuple[float, float]] = []
    for n in sorted(best):
        if spec.row_of[n] in spec.rail_rows:
            w = float(sizes.get(n, (0.0, 0.0))[0])
            rail_spans.append((best[n][0] - w / 2.0, best[n][0] + w / 2.0))

    def within_corridors(p: Mapping[str, _XY]) -> bool:
        if not rail_spans:
            return True
        x_lo = min(p[n][0] - float(sizes[n][0]) / 2.0 for n in p
                   if spec.row_of[n] not in spec.rail_rows)
        x_hi = max(p[n][0] + float(sizes[n][0]) / 2.0 for n in p
                   if spec.row_of[n] not in spec.rail_rows)
        return all(x_lo >= r_lo - 1e-9 and x_hi <= r_hi + 1e-9
                   for r_lo, r_hi in rail_spans)

    def span(names: List[str], p: Mapping[str, _XY]) -> Tuple[float, float]:
        lo = min(p[n][0] - float(sizes[n][0]) / 2.0 for n in names)
        hi = max(p[n][0] + float(sizes[n][0]) / 2.0 for n in names)
        return lo, hi

    for _round in range(max(1, int(max_rounds))):
        cand: Optional[Tuple[float, Dict[str, _XY], str]] = None
        for k, groups in groups_by_row.items():
            for g in groups:
                for kp, host_groups in groups_by_row.items():
                    host = [h for h in host_groups if h != g]
                    if kp == k and not host:
                        continue  # g alone in its row: nothing to reorder
                    # the host row's current left edge anchors the repack.
                    host_members = [n for h in host for n in h] or g
                    left = min(best[n][0] - float(sizes[n][0]) / 2.0
                               for n in host_members)
                    for slot in range(len(host) + 1):
                        seq = host[:slot] + [g] + host[slot:]
                        trial = dict(best)
                        x = left
                        for gr in seq:
                            g_lo, g_hi = span(gr, best)
                            dx = x - g_lo
                            for n in gr:
                                trial[n] = (best[n][0] + dx, best[n][1])
                            x += (g_hi - g_lo) + float(min_gap)
                        try:
                            pressed = press(trial)
                        except ValueError:
                            continue  # conflicting intent: move refused
                        if not within_corridors(pressed):
                            continue  # device would stick out past a corridor
                        if not _channel_capacity_violations(
                                pressed, sizes, spec, channel_um_per_gap,
                        ) <= base_capacity:
                            # x-resolved channel capacity: the lane's x
                            # segment is owed to its crossing nets -- if the
                            # move does not fit, it is not embedded (the
                            # press keeps the band-stacked fallback; never
                            # density over connectivity).
                            continue
                        if not flank_ok(pressed):
                            # keepout flank: the corridor between a hard
                            # keep-out footprint and the design extent is
                            # owed to the nets passing/exiting around it --
                            # same refusal posture (never density over
                            # connectivity).
                            continue
                        area = _device_outline_area(pressed, sizes, spec)
                        if area < best_area - _EMBED_MIN_GAIN_UM2 and (
                                cand is None or area < cand[0]):
                            cand = (
                                area, pressed,
                                f"embed: group ({', '.join(g)}) of row {k} "
                                f"re-slotted into row {kp}'s x-pack at slot "
                                f"{slot} (device outline "
                                f"{best_area:.1f} -> {area:.1f} um^2; row "
                                "membership and x-overlap order edges "
                                "unchanged)",
                            )
        if cand is None:
            break
        best_area, best = cand[0], cand[1]
        notes.append(cand[2])
    return best, tuple(notes)


__all__ = [
    "CompactionSpec",
    "Full2DAxisProvenanceError",
    "Full2DCompactionGraph",
    "Full2DPairConstraint",
    "Full2DSeparationEdge",
    "compact_full2d_master_placement",
    "compact_placement",
    "embed_search",
    "full2d_graph_from_seed",
]
