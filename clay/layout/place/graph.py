"""PlacementConstraintGraph -- the CLOSED placement input object (spec §4.3, Wave B2).

Placement had NO closed input: ``place_envelope_min`` took ~17 loose kwargs and
``annotate_to_solver.SolverConstraints`` carried a second, divergent shape. This is the
ONE frozen contract both forms build (R5: not a second path -- one graph, byte-identical
adapters):

* :meth:`from_envelope_kwargs` builds the graph from the exact ``place_envelope_min``
  kwargs; :meth:`to_envelope_kwargs` returns them, so
  ``place_envelope_min(**graph.to_envelope_kwargs())`` is byte-identical to the direct
  call (a parity test asserts round-trip equality AND identical PlacementResult).
* :meth:`from_solver_constraints` subsumes ``SolverConstraints`` into the SAME graph, so
  the CP and annotate entry forms share one closed contract and one ``kinds()``.

The typed constraints are :mod:`clay.layout.place.constraints` (RigidMacro / Symmetry /
RelativeOrder / Abutment / KeepTogether / RegionAffinity / EdgeAffinity) -- generic
physical objects, never a circuit role. The reference-coupled ``relgraph.ConstraintGraph``
(exact dx/dy) is a leakage BACK-graph and is deliberately NOT an input here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from clay.layout.place.constraints import (
    PLACEMENT_CONSTRAINT_KINDS,
    Abutment,
    EdgeAffinity,
    KeepTogether,
    RegionAffinity,
    RelativeOrder,
    RigidMacro,
    Symmetry,
    is_hard,
)
from clay.layout.synth.placement import Group

__all__ = ["PlacementConstraintGraph"]


@dataclass(frozen=True)
class PlacementConstraintGraph:
    """Closed placement input: rigid ``macros`` + generic ``constraints`` + the
    connectivity (``nets`` / weights / signal) + solver params. Frozen; deterministic
    ``kinds()``. Built by :meth:`from_envelope_kwargs` / :meth:`from_solver_constraints`
    and consumed via :meth:`to_envelope_kwargs`."""

    macros: Tuple[RigidMacro, ...] = ()
    constraints: Tuple[Any, ...] = ()
    nets: Tuple[Tuple[str, ...], ...] = ()
    net_weights: Tuple[float, ...] = ()
    net_signal: Tuple[bool, ...] = ()
    gap_um: float = 0.5
    aspect_target: Optional[float] = None
    wl_weight: int = 1
    axis_mode: str = "shared_hard"
    max_time_s: float = 20.0
    backend: str = "auto"

    # -- adapters ---------------------------------------------------------------
    @classmethod
    def from_envelope_kwargs(
        cls,
        groups: Sequence[Group],
        *,
        nets: Sequence[Sequence[str]] = (),
        net_weights: Sequence[float] = (),
        net_signal: Sequence[bool] = (),
        symmetry_pairs: Sequence[Tuple[str, str]] = (),
        axis_groups: Sequence[Tuple[str, float]] = (),
        axis_mode: str = "shared_hard",
        gap: float = 0.5,
        wl_weight: int = 1,
        aspect_target: Optional[float] = None,
        max_time_s: float = 20.0,
        backend: str = "auto",
        disjoint_clusters: Sequence[Sequence[str]] = (),
        island_pads: Sequence[float] = (),
        band_of_group: Optional[Dict[str, int]] = None,
        order_constraints: Sequence[Tuple[str, str, str]] = (),
        abut_pairs: Sequence[Tuple[str, str]] = (),
        rail_affinities: Sequence[Tuple[str, str]] = (),
    ) -> "PlacementConstraintGraph":
        macros = tuple(RigidMacro(id=g.key, w_um=g.w, h_um=g.h) for g in groups)
        cons: List[Any] = []
        # symmetry (one object captures pairs + axis_groups + mode) -- only when present
        if symmetry_pairs or axis_groups:
            cons.append(
                Symmetry(
                    pairs=tuple(tuple(p) for p in symmetry_pairs),
                    axis_offset_by_id=tuple(tuple(a) for a in axis_groups),
                    mode=axis_mode,
                )
            )
        cons += [RelativeOrder(a, b, rel) for (a, b, rel) in order_constraints]
        cons += [Abutment(a, b) for (a, b) in abut_pairs]
        _pads = list(island_pads or ())
        for i, cl in enumerate(disjoint_clusters or ()):
            pad = _pads[i] if i < len(_pads) else 0.0
            cons.append(KeepTogether(members=tuple(cl), pad_um=pad))
        for gid, band in sorted((band_of_group or {}).items()):
            cons.append(RegionAffinity(id=gid, region=int(band)))
        cons += [EdgeAffinity(id=g, edge=str(side)) for (g, side) in rail_affinities]
        return cls(
            macros=macros,
            constraints=tuple(cons),
            nets=tuple(tuple(n) for n in nets),
            net_weights=tuple(net_weights),
            net_signal=tuple(net_signal),
            gap_um=gap,
            aspect_target=aspect_target,
            wl_weight=wl_weight,
            axis_mode=axis_mode,
            max_time_s=max_time_s,
            backend=backend,
        )

    def to_envelope_kwargs(self) -> Dict[str, Any]:
        """Reconstruct the ``place_envelope_min`` call: ``groups`` + kwargs. Byte-
        identical to the inputs :meth:`from_envelope_kwargs` was built from (parity)."""
        groups = [Group(key=mm.id, w=mm.w_um, h=mm.h_um) for mm in self.macros]
        symmetry_pairs: Tuple = ()
        axis_groups: Tuple = ()
        order_constraints: List = []
        abut_pairs: List = []
        disjoint_clusters: List = []
        island_pads: List = []
        band_of_group: Dict[str, int] = {}
        rail_affinities: List = []
        for c in self.constraints:
            if isinstance(c, Symmetry):
                symmetry_pairs = c.pairs
                axis_groups = c.axis_offset_by_id
            elif isinstance(c, RelativeOrder):
                order_constraints.append((c.a, c.b, c.rel))
            elif isinstance(c, Abutment):
                abut_pairs.append((c.a, c.b))
            elif isinstance(c, KeepTogether):
                disjoint_clusters.append(tuple(c.members))
                island_pads.append(c.pad_um)
            elif isinstance(c, RegionAffinity):
                band_of_group[c.id] = c.region
            elif isinstance(c, EdgeAffinity):
                rail_affinities.append((c.id, c.edge))
        return {
            "groups": groups,
            "nets": [list(n) for n in self.nets],
            "net_weights": list(self.net_weights),
            "net_signal": list(self.net_signal),
            "symmetry_pairs": symmetry_pairs,
            "axis_groups": axis_groups,
            "axis_mode": self.axis_mode,
            "gap": self.gap_um,
            "wl_weight": self.wl_weight,
            "aspect_target": self.aspect_target,
            "max_time_s": self.max_time_s,
            "backend": self.backend,
            "disjoint_clusters": disjoint_clusters,
            "island_pads": island_pads,
            "band_of_group": band_of_group,
            "order_constraints": order_constraints,
            "abut_pairs": abut_pairs,
            "rail_affinities": rail_affinities,
        }

    @classmethod
    def from_solver_constraints(
        cls,
        sc: Any,
        sizes: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> "PlacementConstraintGraph":
        """Subsume an ``annotate_to_solver.SolverConstraints`` into the SAME closed
        graph (R5: two entry forms, one contract). ``sizes`` (label -> (w,h)) supplies
        the rigid macro footprints; absent, the macros are omitted (a constraint-only
        graph). Symmetry / order / abut / floorplan-rows map to the closed set."""
        macros: Tuple[RigidMacro, ...] = ()
        if sizes:
            macros = tuple(
                RigidMacro(id=k, w_um=w, h_um=h) for k, (w, h) in sizes.items()
            )
        cons: List[Any] = []
        sym = tuple(getattr(sc, "symmetry_pairs", ()) or ())
        if sym:
            cons.append(Symmetry(pairs=tuple(tuple(p) for p in sym)))
        cons += [
            RelativeOrder(a, b, rel)
            for (a, b, rel) in (getattr(sc, "order_constraints", ()) or ())
        ]
        cons += [Abutment(a, b) for (a, b) in (getattr(sc, "abut_pairs", ()) or ())]
        # @place_group ROWS -> region affinity (device -> row ordinal).
        rows = getattr(sc, "floorplan_rows", {}) or {}
        for label, rc in sorted(rows.items()):
            row = rc[0] if isinstance(rc, (tuple, list)) else rc
            if row is not None:
                cons.append(RegionAffinity(id=label, region=int(row)))
        return cls(macros=macros, constraints=tuple(cons))

    # -- queries ----------------------------------------------------------------
    def kinds(self) -> FrozenSet[str]:
        """The set of placement-constraint KINDS present (for the handshake vs a
        placement backend's ``satisfies()``). ``rigid_macro`` is included when the
        graph carries macros."""
        ks = {str(getattr(c, "kind", "?")) for c in self.constraints}
        if self.macros:
            ks.add("rigid_macro")
        return frozenset(ks)

    def hard_kinds(self) -> FrozenSet[str]:
        ks = {str(getattr(c, "kind", "?")) for c in self.constraints if is_hard(c)}
        if self.macros:
            ks.add("rigid_macro")
        return frozenset(ks)

    def __post_init__(self) -> None:
        bad = self.kinds() - (PLACEMENT_CONSTRAINT_KINDS | {"rigid_macro"})
        if bad:
            raise ValueError(
                f"PlacementConstraintGraph carries non-closed kind(s) {sorted(bad)} "
                f"(closed set = {sorted(PLACEMENT_CONSTRAINT_KINDS)})"
            )
