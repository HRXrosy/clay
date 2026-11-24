"""Deck-connected cross-net isolation oracle for router geometry.

This checker names a hard routing contract: two distinct logical nets must not
occupy one physical component under the PDK metal/via ``connect()`` stack.  The
component graph is extracted by KLayout ``LayoutToNetlist`` itself; native
same-layer overlap scans are intentionally not consulted because they cannot
see a bridge through a via stack.

The oracle is suitable for router bring-up and incremental legality checks.  It
is not a replacement for full-device LVS or the foundry DRC deck.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import klayout.db as kdb


LayerKey = Tuple[int, int]
RectKey = Tuple[int, int, int, int]
_ORACLE = "KLayout.LayoutToNetlist.connect"


def _layer_key(value: Sequence[int]) -> LayerKey:
    return (int(value[0]), int(value[1]))


@dataclass(frozen=True, order=True)
class DeckConnection:
    """One deck ``lower <-> cut <-> upper`` connectivity declaration."""

    cut: LayerKey
    lower: LayerKey
    upper: LayerKey


@dataclass(frozen=True, order=True)
class MimCarve:
    """One full-extraction MIM dielectric carve from the synthesized deck.

    ``top_marker`` is the conductive top plate, ``bottom`` is the raw routing
    metal under it, and ``riser`` is the cut to the upper cover metal.  The
    lower-to-riser connection must use the derived bottom-metal complement,
    never the raw bottom layer underneath the marker.
    """

    top_marker: LayerKey
    bottom: LayerKey
    riser: LayerKey
    oversize_um: float = 0.0
    # Native SKY130 via3 is split by capm itself. Other decks select an
    # interacting whole bottom component (or its explicit grown marker).
    marker_only: bool = False


@dataclass(frozen=True)
class DeckConnectStack:
    """PDK-data-driven conductor and cut connectivity used by the oracle."""

    conductors: Tuple[LayerKey, ...]
    connections: Tuple[DeckConnection, ...]
    mim_carves: Tuple[MimCarve, ...] = ()

    def __post_init__(self) -> None:
        conductors = tuple(_layer_key(layer) for layer in self.conductors)
        connections = tuple(
            DeckConnection(
                cut=_layer_key(connection.cut),
                lower=_layer_key(connection.lower),
                upper=_layer_key(connection.upper),
            )
            for connection in self.connections
        )
        mim_carves = tuple(
            MimCarve(
                top_marker=_layer_key(carve.top_marker),
                bottom=_layer_key(carve.bottom),
                riser=_layer_key(carve.riser),
                oversize_um=float(carve.oversize_um),
                marker_only=bool(carve.marker_only),
            )
            for carve in self.mim_carves
        )
        if not conductors:
            raise ValueError("deck connect stack has no conductor layers")
        if len(set(conductors)) != len(conductors):
            raise ValueError("deck connect stack contains duplicate conductor layers")
        declared = set(conductors)
        for connection in connections:
            if connection.lower not in declared or connection.upper not in declared:
                raise ValueError(
                    "deck connection references an undeclared conductor: "
                    f"{connection!r}"
                )
        carve_connections = [(carve.bottom, carve.riser) for carve in mim_carves]
        if len(set(carve_connections)) != len(carve_connections):
            raise ValueError(
                "duplicate MIM carve for one lower-metal/riser connection"
            )
        connection_pairs = {(c.lower, c.cut) for c in connections}
        for carve in mim_carves:
            if (carve.bottom, carve.riser) not in connection_pairs:
                raise ValueError(
                    "MIM carve does not match a lower-metal/riser deck connection: "
                    f"{carve!r}"
                )
            if carve.oversize_um < 0:
                raise ValueError(f"MIM carve has negative oversize: {carve!r}")
        object.__setattr__(self, "conductors", conductors)
        object.__setattr__(self, "connections", connections)
        object.__setattr__(self, "mim_carves", mim_carves)

    @classmethod
    def from_tech(
        cls,
        tech: Any,
        *,
        mim_option: Optional[str] = None,
        cap_option: Optional[str] = None,
        metal_level: Optional[str] = None,
        pex_backend: Optional[str] = None,
    ) -> "DeckConnectStack":
        """Resolve the complete contiguous full-extraction deck stack.

        Layer numbers and the stack ceiling come only from the exact synthesized
        full-extraction configuration.  ``LayoutTech`` is checked for parity but
        cannot add, remove, or truncate a deck-declared layer.  A gap anywhere
        in the declared metal/via ladder fails closed.
        """
        eff_mim = getattr(tech, "mim_option", None) if mim_option is None else mim_option
        eff_cap = getattr(tech, "cap_option", None) if cap_option is None else cap_option
        eff_lvl = getattr(tech, "metal_level", None) if metal_level is None else metal_level
        eff_pex = getattr(tech, "pex_backend", None) if pex_backend is None else pex_backend

        if str(tech.pdk_id) == "sky130":
            # Native Magic SKY130 extraction: via3 under capm contacts the top
            # plate, while via3 outside capm contacts raw M3. This PDK uses
            # native full-device verification, not a synthesized MOS deck.
            conductors = tuple((68 + index, 20) for index in range(5))
            connections = tuple(DeckConnection((68 + index, 44),
                conductors[index], conductors[index+1]) for index in range(4))
            for index, key in enumerate(conductors, 1):
                if tuple(tech.metal_key(index)) != key:
                    raise ValueError("native SKY130 metal mapping differs from its extraction stack")
            for index, connection in enumerate(connections, 1):
                if tuple(tech.via_key(index)) != connection.cut:
                    raise ValueError("native SKY130 via mapping differs from its extraction stack")
            return cls(conductors, connections,
                       (MimCarve((89, 44), (70, 20), (70, 44), marker_only=True),))
        # The exact full-extraction deck is the layer-stack authority.  The
        # LayoutTech mapping is only a parity check: it cannot add an upper
        # layer that the extracting deck does not declare.
        from clay.layout.route.lvs_gate import _resolved_lvs_config

        _mim_kw = {
            k: v
            for k, v in (
                ("mim_option", eff_mim),
                ("cap_option", eff_cap),
                ("metal_level", eff_lvl),
                ("pex_backend", eff_pex),
            )
            if v is not None and str(v).strip()
        }
        cfg = _resolved_lvs_config(
            str(tech.pdk_id),
            full_extraction=True,
            **_mim_kw,
        )
        layermap = cfg["layermap"]
        declared_levels = sorted(
            int(token[len("metal") :])
            for token in layermap
            if token.startswith("metal") and token[len("metal") :].isdigit()
        )
        if not declared_levels:
            raise ValueError("full-extraction deck declares no routing metals")
        expected_levels = list(range(1, declared_levels[-1] + 1))
        if declared_levels != expected_levels:
            raise ValueError(
                "full-extraction deck has a non-contiguous metal ladder: "
                f"declared={declared_levels}, expected={expected_levels}"
            )
        conductors = []
        for level in declared_levels:
            deck_layer = layermap[f"metal{level}"]
            deck_key = _layer_key(deck_layer)
            try:
                tech_key = _layer_key(tech.metal_key(level))
            except (KeyError, LookupError) as exc:
                raise ValueError(
                    f"deck declares metal{level}={deck_key}, but LayoutTech does not"
                ) from exc
            if tech_key != deck_key:
                raise ValueError(
                    f"metal{level} mapping differs between full-extraction deck "
                    f"{deck_key} and LayoutTech {tech_key}"
                )
            conductors.append(deck_key)
        connections = []
        for level in range(1, len(conductors)):
            deck_cut = layermap.get(f"via{level}")
            if deck_cut is None:
                raise ValueError(
                    f"full-extraction deck has no via{level} between its declared "
                    f"metal{level}/metal{level + 1}"
                )
            cut = _layer_key(deck_cut)
            try:
                tech_cut = _layer_key(tech.via_key(level))
            except (KeyError, LookupError) as exc:
                raise ValueError(
                    f"deck declares via{level}={cut}, but LayoutTech does not"
                ) from exc
            if tech_cut != cut:
                raise ValueError(
                    f"via{level} mapping differs between full-extraction deck "
                    f"{cut} and LayoutTech {tech_cut}"
                )
            connections.append(
                DeckConnection(
                    cut=cut,
                    lower=conductors[level - 1],
                    upper=conductors[level],
                )
            )
        mim = cfg.get("mim")
        connection_pairs = {(c.lower, c.cut) for c in connections}
        carves = []
        if mim is not None:
            for level in mim.resolved_levels():
                top = layermap.get(level.top_token)
                bottom = layermap.get(level.bottom_metal_token)
                riser = (
                    layermap.get(level.top_via_token)
                    if level.top_via_token is not None
                    else None
                )
                if top is None or bottom is None or riser is None:
                    continue
                bottom_key = _layer_key(bottom)
                riser_key = _layer_key(riser)
                if (bottom_key, riser_key) not in connection_pairs:
                    continue
                carves.append(
                    MimCarve(
                        top_marker=_layer_key(top),
                        bottom=bottom_key,
                        riser=riser_key,
                        oversize_um=float(mim.oversize_um),
                    )
                )
        return cls(tuple(conductors), tuple(connections), tuple(carves))


@dataclass(frozen=True, order=True)
class NetProbe:
    """A logical-net claim at an exact conductor-layer point (layout DBU)."""

    net: str
    layer: LayerKey
    x: int
    y: int

    def __post_init__(self) -> None:
        net = str(self.net)
        if not net:
            raise ValueError("net probe has an empty logical-net name")
        object.__setattr__(self, "net", net)
        object.__setattr__(self, "layer", _layer_key(self.layer))
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))


@dataclass(frozen=True)
class DeckLiteralRect:
    """One exact integer-DBU rectangle in an in-memory router geometry view.

    ``net`` is a logical ownership claim and is permitted only on a conductor
    layer.  ``None`` represents literal, unowned geometry such as a via cut or
    MIM marker: it participates in deck connectivity but creates no claim.
    """

    layer: LayerKey
    rect: RectKey
    net: str | None = None

    def __post_init__(self) -> None:
        layer = _layer_key(self.layer)
        if len(self.rect) != 4:
            raise ValueError("literal rectangle must have four DBU coordinates")
        if any(type(value) is not int for value in self.rect):
            raise TypeError("literal rectangle coordinates must be exact integers")
        rect = tuple(self.rect)
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError(f"literal rectangle must have positive area: {rect}")
        net = None if self.net is None else str(self.net)
        if net == "":
            raise ValueError("owned literal rectangle has an empty net name")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "rect", rect)
        object.__setattr__(self, "net", net)


@dataclass(frozen=True)
class DeckSpacingReport:
    """Clearance measured on extracted final conductor polygons, not route boxes."""
    net_a: str
    net_b: str
    required_nm: int | None
    status: str
    # One entry per failing layer: (layer key, edge-pair count, enclosing DBU box).
    violations: tuple = ()
    missing_nets: tuple = ()


@dataclass(frozen=True)
class DeckIsolationReport:
    """Canonical fail-closed result of one deck-connected extraction."""

    ok: bool
    merged_net_groups: Tuple[Tuple[str, ...], ...]
    unresolved_probes: Tuple[Tuple[str, LayerKey, int, int], ...]
    ambiguous_nets: Tuple[str, ...]
    oracle: str
    verdict_digest: str
    spacing_reports: Tuple[DeckSpacingReport, ...] = ()
    # Optional evidence of equivalence classes of *resolved* input probes.
    # This is deliberately outside the legacy verdict digest.
    resolved_probe_groups: tuple = ()
    probe_partition_included: bool = False

    @property
    def spacing_satisfied(self) -> bool:
        return all(report.status == "pass" for report in self.spacing_reports)

    # ``verdict_digest`` is only a canonical hash of the issue lists above. It
    # intentionally does NOT bind or prove the deck text, probe declarations,
    # or GDS bytes.  Exact GDS/deck SHA receipts belong to the external truth
    # oracle and must be recorded separately.

    @property
    def cross_net_isolated(self) -> bool:
        """Whether no extracted component is claimed by distinct logical nets."""

        return not self.merged_net_groups

    @property
    def probe_complete(self) -> bool:
        """Whether all ownership probes resolve to one component per logical net."""

        return not self.unresolved_probes and not self.ambiguous_nets


@dataclass(frozen=True)
class ProbeConnectivityComparison:
    """Cross-scene probe relationships; pending connectivity stays pending."""
    lost_probe_groups: tuple
    newly_resolved_isolated_groups: tuple


def compare_probe_connectivity(before: DeckIsolationReport, after: DeckIsolationReport) -> ProbeConnectivityComparison:
    """Preserve resolved equivalences without comparing unstable cluster IDs."""
    if not before.probe_partition_included or not after.probe_partition_included:
        raise ValueError('probe partitions were not requested on both scenes')
    before_groups = tuple(frozenset(group) for group in before.resolved_probe_groups)
    after_groups = tuple(frozenset(group) for group in after.resolved_probe_groups)
    index = {}
    for i, group in enumerate(after_groups):
        for probe in group:
            if probe in index:
                raise ValueError('probe appears in multiple extracted components')
            index[probe] = i
    lost = []
    for group in before_groups:
        ids = {index.get(probe) for probe in group}
        if None in ids or len(ids) != 1:
            lost.append(tuple(sorted(group)))
    prior = {probe for group in before_groups for probe in group}
    isolated = []
    for group in after_groups:
        for net in sorted({p[0] for p in group}):
            own = {p for p in group if p[0] == net}
            new = own - prior
            if new and not (own & prior):
                isolated.append((net, tuple(sorted(new))))
    return ProbeConnectivityComparison(tuple(sorted(lost)), tuple(sorted(isolated)))


def _layer_index(layout: Any, layer: LayerKey):
    index = layout.find_layer(int(layer[0]), int(layer[1]))
    if index is None or int(index) < 0:
        return None
    return index


def _canonical_report(
    *,
    merged_net_groups: Iterable[Iterable[str]],
    unresolved_probes: Iterable[Tuple[str, LayerKey, int, int]],
    ambiguous_nets: Iterable[str],
    spacing_reports: Sequence[DeckSpacingReport] = (),
    resolved_probe_groups: Iterable[Iterable[Tuple[str, LayerKey, int, int]]] = (),
    probe_partition_included: bool = False,
) -> DeckIsolationReport:
    canonical_groups = set()
    for group in merged_net_groups:
        names = tuple(sorted(set(str(net) for net in group)))
        if len(names) > 1:
            canonical_groups.add(names)
    groups = tuple(sorted(canonical_groups))
    unresolved = tuple(
        sorted(
            (
                str(net),
                _layer_key(layer),
                int(x),
                int(y),
            )
            for net, layer, x, y in unresolved_probes
        )
    )
    ambiguous = tuple(sorted(set(str(net) for net in ambiguous_nets)))
    payload = {
        "ambiguous_nets": list(ambiguous),
        "merged_net_groups": [list(group) for group in groups],
        "oracle": _ORACLE,
        "unresolved_probes": [
            [net, list(layer), x, y] for net, layer, x, y in unresolved
        ],
    }
    if spacing_reports:
        payload["net_spacings"] = [asdict(report) for report in spacing_reports]
    verdict_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DeckIsolationReport(
        ok=not (groups or unresolved or ambiguous)
        and all(report.status == "pass" for report in spacing_reports),
        merged_net_groups=groups,
        unresolved_probes=unresolved,
        ambiguous_nets=ambiguous,
        oracle=_ORACLE,
        verdict_digest=verdict_digest,
        spacing_reports=tuple(spacing_reports),
        resolved_probe_groups=tuple(sorted({tuple(sorted({
            (str(net), _layer_key(layer), int(x), int(y)) for net, layer, x, y in group
        })) for group in resolved_probe_groups if group})),
        probe_partition_included=probe_partition_included,
    )


def check_deck_net_isolation(
    *,
    layout: Any,
    cell: Any,
    stack: DeckConnectStack,
    probes: Sequence[NetProbe],
    spacing_constraints: Sequence[Any] = (),
    include_probe_partition: bool = False,
) -> DeckIsolationReport:
    """Extract deck connectivity and reject cross-net physical components.

    ``probes`` are explicit logical ownership claims.  A probe outside physical
    conductor geometry is unresolved.  Multiple probes for one logical net on
    different physical components are ambiguous.  Both conditions fail closed,
    as does any extracted component claimed by two or more logical nets.
    """

    ordered_probes = tuple(sorted(NetProbe(p.net, p.layer, p.x, p.y) for p in probes))
    if type(include_probe_partition) is not bool:
        raise TypeError('include_probe_partition must be an exact bool')
    if not ordered_probes:
        return _canonical_report(
            merged_net_groups=(),
            unresolved_probes=(("<NO_PROBES>", stack.conductors[0], 0, 0),),
            ambiguous_nets=(),
            probe_partition_included=include_probe_partition,
            spacing_reports=_check_extracted_spacing(
                None, {}, stack, {}, spacing_constraints,
                invalid_nets=set(), dbu_um=layout.dbu,
            ),
        )
    conductor_set = set(stack.conductors)
    unknown_layers = sorted({probe.layer for probe in ordered_probes} - conductor_set)
    if unknown_layers:
        raise ValueError(
            "net probes reference layers outside the deck conductor stack: "
            f"{unknown_layers}"
        )

    l2n = kdb.LayoutToNetlist(kdb.RecursiveShapeIterator(layout, cell, []))
    l2n.threads = 1
    handles: Dict[LayerKey, Any] = {}
    all_layers = set(stack.conductors) | {
        connection.cut for connection in stack.connections
    } | {carve.top_marker for carve in stack.mim_carves}
    for position, layer in enumerate(sorted(all_layers)):
        index = _layer_index(layout, layer)
        if index is not None:
            handles[layer] = l2n.make_polygon_layer(
                index, f"DECK_ISOLATION_{position}_{layer[0]}_{layer[1]}"
            )

    for layer in stack.conductors:
        handle = handles.get(layer)
        if handle is not None:
            l2n.connect(handle)
    carves_by_connection = {
        (carve.bottom, carve.riser): carve for carve in stack.mim_carves
    }
    for connection in stack.connections:
        cut = handles.get(connection.cut)
        lower = handles.get(connection.lower)
        upper = handles.get(connection.upper)
        if cut is None:
            continue
        carve = carves_by_connection.get((connection.lower, connection.cut))
        marker = handles.get(carve.top_marker) if carve is not None else None
        if lower is not None and marker is not None and carve is not None:
            # Mirror klayout_deck_gen's exact MIM derivation.  ENCLOSE specs
            # select the interacting bottom-metal figure.  The gf180 oversize
            # form intersects that figure with the grown top marker.
            interacting = lower.interacting(marker)
            if carve.marker_only:
                mim_virtual = marker & interacting
            elif carve.oversize_um > 0:
                oversize_dbu = int(round(carve.oversize_um / float(layout.dbu)))
                mim_virtual = marker.sized(oversize_dbu) & interacting
            else:
                mim_virtual = interacting
            metal_ncap = lower - mim_virtual
            suffix = f"{connection.lower[0]}_{connection.lower[1]}"
            l2n.register(mim_virtual, f"MIM_VIRTUAL_{suffix}")
            l2n.register(metal_ncap, f"METAL_NCAP_{suffix}")

            # Bottom plate remains on its raw source-metal net, while the riser
            # can land on the routable complement only.  The top marker rises
            # through that same cut to the upper cover.  Thus the dielectric is
            # absent from the connectivity graph rather than assigned a price.
            l2n.connect(lower, mim_virtual)
            l2n.connect(metal_ncap, mim_virtual)
            l2n.connect(metal_ncap, lower)
            l2n.connect(metal_ncap, cut)
            l2n.connect(marker, cut)
        elif lower is not None:
            l2n.connect(lower, cut)
        if upper is not None:
            l2n.connect(cut, upper)
    l2n.extract_netlist()

    unresolved = []
    by_net: Dict[str, set] = {}
    by_component: Dict[Tuple[str, int], set] = {}
    physical_nets = {}
    probes_by_component = {}
    for probe in ordered_probes:
        handle = handles.get(probe.layer)
        physical = (
            l2n.probe_net(handle, kdb.Point(probe.x, probe.y))
            if handle is not None
            else None
        )
        if physical is None:
            unresolved.append((probe.net, probe.layer, probe.x, probe.y))
            continue
        circuit_name = str(physical.circuit().name)
        component = (circuit_name, int(physical.cluster_id))
        by_net.setdefault(probe.net, set()).add(component)
        by_component.setdefault(component, set()).add(probe.net)
        if include_probe_partition:
            probes_by_component.setdefault(component, set()).add(
                (probe.net, probe.layer, probe.x, probe.y))
        physical_nets.setdefault(probe.net, physical)

    ambiguous = tuple(net for net, components in by_net.items() if len(components) != 1)
    merged = tuple(nets for nets in by_component.values() if len(nets) > 1)
    spacing_reports = _check_extracted_spacing(
        l2n, handles, stack, physical_nets, spacing_constraints,
        invalid_nets=set(ambiguous) | {row[0] for row in unresolved}
        | {net for group in merged for net in group},
        dbu_um=layout.dbu,
    )
    return _canonical_report(
        merged_net_groups=merged,
        unresolved_probes=unresolved,
        ambiguous_nets=ambiguous,
        spacing_reports=spacing_reports,
        resolved_probe_groups=probes_by_component.values(),
        probe_partition_included=include_probe_partition,
    )


def _check_extracted_spacing(l2n, handles, stack, physical_nets, constraints,
                             *, invalid_nets, dbu_um):
    """Reuse the physical component extraction; include connected device metal.

    The caller supplies the same net spelling as its probes. All polygons on a
    component participate, including final healing and snapping. Region checks
    use Euclidean distance and disable shielding to check every facing edge.
    """
    from fractions import Fraction
    from math import ceil
    from clay.layout.route.constraints import is_hard
    from clay.layout.route.spacing_intent import spacing_nm

    regions = {}
    reports = []
    for declaration in constraints:
        if not is_hard(declaration):
            continue
        a, b = declaration.net_a, declaration.net_b
        required = spacing_nm(declaration, (a, b))
        missing = tuple(net for net in (a, b)
                        if net not in physical_nets or net in invalid_nets)
        if required is None or missing:
            reports.append(DeckSpacingReport(a, b, required, "unmeasurable", missing_nets=missing))
            continue
        violations = []
        distance_dbu = ceil(Fraction(required) / (Fraction(str(dbu_um)) * 1000))
        for layer in stack.conductors:
            if layer not in handles:
                continue
            for net in (a, b):
                if (net, layer) not in regions:
                    regions[net, layer] = l2n.shapes_of_net(physical_nets[net], handles[layer], True)
            left, right = regions[a, layer], regions[b, layer]
            if distance_dbu <= 0 or left.is_empty() or right.is_empty():
                continue
            errors = left.separation_check(right, distance_dbu, metrics=kdb.Region.Euclidian,
                                           ignore_angle=None, shielded=False)
            if not errors.is_empty():
                box = errors.bbox()
                violations.append((layer, errors.count(), (box.left, box.bottom, box.right, box.top)))
        reports.append(DeckSpacingReport(a, b, required, "fail" if violations else "pass",
                                         violations=tuple(violations)))
    return tuple(reports)


def _literal_rect_sort_key(shape: DeckLiteralRect) -> Tuple[Any, ...]:
    return (
        shape.layer,
        "" if shape.net is None else shape.net,
        shape.rect,
    )


def _integer_interior(lo: int, hi: int) -> int:
    if hi - lo < 2:
        raise ValueError(
            "owned literal rectangle has no strictly interior integer-DBU point"
        )
    return lo + (hi - lo) // 2


def check_deck_literal_isolation(
    *,
    tech: Any,
    stack: DeckConnectStack,
    shapes: Sequence[DeckLiteralRect],
) -> DeckIsolationReport:
    """Check a canonical in-memory rectangle view without writing a GDS file.

    The supplied stack must equal the complete full-extraction stack resolved
    from the exact :class:`LayoutTech`.  Each owned rectangle receives its own
    strictly interior probe, ensuring that disconnected same-net branches are
    visible as ambiguity.  Unowned literal geometry participates only in the
    physical ``connect()`` graph.  Logical owners follow emit's uppercase
    convention; spellings that would collapse under it fail closed.
    """

    from clay.layout.tech import LayoutTech

    if type(tech) is not LayoutTech:
        raise TypeError("tech must be an exact LayoutTech")
    if type(stack) is not DeckConnectStack:
        raise TypeError("stack must be an exact DeckConnectStack")
    complete_stack = DeckConnectStack.from_tech(tech)
    if stack != complete_stack:
        raise ValueError(
            "stack must equal the complete full-extraction deck connect stack"
        )

    literal_shapes = tuple(
        DeckLiteralRect(shape.layer, shape.rect, shape.net) for shape in shapes
    )
    owner_spellings = tuple(
        shape.net for shape in literal_shapes if shape.net is not None
    )
    if owner_spellings:
        _uppercase_net_set(owner_spellings, source="literal rectangle owners")
        literal_shapes = tuple(
            DeckLiteralRect(
                shape.layer,
                shape.rect,
                None if shape.net is None else shape.net.upper(),
            )
            for shape in literal_shapes
        )
    ordered_shapes = tuple(sorted(literal_shapes, key=_literal_rect_sort_key))
    conductor_layers = set(stack.conductors)
    known_layers = conductor_layers | {
        connection.cut for connection in stack.connections
    } | {carve.top_marker for carve in stack.mim_carves}
    unknown_layers = sorted(
        {shape.layer for shape in ordered_shapes} - known_layers
    )
    if unknown_layers:
        raise ValueError(
            "literal rectangles reference layers outside the complete deck stack: "
            f"{unknown_layers}"
        )
    invalid_owned_layers = sorted(
        {
            shape.layer
            for shape in ordered_shapes
            if shape.net is not None and shape.layer not in conductor_layers
        }
    )
    if invalid_owned_layers:
        raise ValueError(
            "owned literal rectangles must be on deck conductor layers: "
            f"{invalid_owned_layers}"
        )

    layout = kdb.Layout()
    layout.dbu = float(tech.rules.dbu_nm) / 1000.0
    cell = layout.create_cell("DECK_LITERAL_ISOLATION")
    probes = []
    for shape in ordered_shapes:
        layer_index = layout.layer(*shape.layer)
        cell.shapes(layer_index).insert(kdb.Box(*shape.rect))
        if shape.net is not None:
            x1, y1, x2, y2 = shape.rect
            probes.append(
                NetProbe(
                    shape.net,
                    shape.layer,
                    _integer_interior(x1, x2),
                    _integer_interior(y1, y2),
                )
            )

    return check_deck_net_isolation(
        layout=layout,
        cell=cell,
        stack=stack,
        probes=tuple(probes),
    )


def _uppercase_net_set(names: Sequence[str], *, source: str) -> Tuple[str, ...]:
    """Apply emit's uppercase convention and reject information-losing names."""

    seen: Dict[str, str] = {}
    for value in names:
        raw = str(value)
        if not raw:
            raise ValueError(f"{source} contains an empty net name")
        canonical = raw.upper()
        previous = seen.get(canonical)
        if previous is not None and previous != raw:
            raise ValueError(
                f"{source} net-name case collision: {previous!r} vs {raw!r} "
                f"-> {canonical!r}"
            )
        seen[canonical] = raw
    if not seen:
        raise ValueError(f"{source} must contain the complete logical-net set")
    return tuple(sorted(seen))


def check_gds_label_isolation(
    *,
    gds_path: Any,
    tech: Any,
    expected_net_names: Sequence[str],
    top_cell_name: str | None = None,
) -> DeckIsolationReport:
    """Read a labelled GDS and check its routing-stack net isolation.

    Text on conductor drawing layers supplies only the logical-net probe name
    and location.  Physical connectivity is extracted from polygon geometry,
    so text cannot manufacture or hide a component.  ``expected_net_names`` is
    required and must be the complete logical-net set from the signature/design
    (never reconstructed from observed labels).  Missing or unexpected labels
    fail probe completeness.  Expected and observed names follow emit's
    uppercase convention; information-losing case collisions are rejected.
    Every observed label still participates in merge detection.  The input is
    loaded into a throwaway layout before flattening.
    """

    path = Path(gds_path)
    layout = kdb.Layout()
    layout.read(str(path))
    if top_cell_name is not None:
        cell = layout.cell(str(top_cell_name))
        if cell is None:
            stack = DeckConnectStack.from_tech(tech)
            return _canonical_report(
                merged_net_groups=(),
                unresolved_probes=(
                    (f"<MISSING_TOP:{top_cell_name}>", stack.conductors[0], 0, 0),
                ),
                ambiguous_nets=(),
            )
    else:
        tops = tuple(layout.top_cells())
        if len(tops) != 1:
            stack = DeckConnectStack.from_tech(tech)
            return _canonical_report(
                merged_net_groups=(),
                unresolved_probes=(
                    (f"<AMBIGUOUS_TOP:{len(tops)}>", stack.conductors[0], 0, 0),
                ),
                ambiguous_nets=(),
            )
        cell = tops[0]

    layout.flatten(cell.cell_index(), -1, True)
    stack = DeckConnectStack.from_tech(tech)
    expected = set(
        _uppercase_net_set(expected_net_names, source="expected_net_names")
    )
    probes = []
    found = set()
    observed_spellings: Dict[str, str] = {}
    for layer in stack.conductors:
        index = _layer_index(layout, layer)
        if index is None:
            continue
        for shape in cell.shapes(index).each():
            if not shape.is_text():
                continue
            text = shape.text
            raw_net = str(text.string)
            if not raw_net:
                continue
            net = raw_net.upper()
            previous = observed_spellings.get(net)
            if previous is not None and previous != raw_net:
                raise ValueError(
                    "observed GDS label net-name case collision: "
                    f"{previous!r} vs {raw_net!r} -> {net!r}"
                )
            observed_spellings[net] = raw_net
            point = text.trans.disp
            probes.append(NetProbe(net, layer, int(point.x), int(point.y)))
            found.add(net)

    report = check_deck_net_isolation(
        layout=layout,
        cell=cell,
        stack=stack,
        probes=tuple(probes),
    )
    if expected == found:
        return report
    incomplete = tuple(
        (f"<MISSING_LABEL:{net}>", stack.conductors[0], 0, 0)
        for net in sorted(expected - found)
    ) + tuple(
        (f"<UNEXPECTED_LABEL:{net}>", stack.conductors[0], 0, 0)
        for net in sorted(found - expected)
    )
    return _canonical_report(
        merged_net_groups=report.merged_net_groups,
        unresolved_probes=tuple(report.unresolved_probes) + incomplete,
        ambiguous_nets=report.ambiguous_nets,
    )


__all__ = [
    "DeckConnectStack",
    "DeckConnection",
    "DeckIsolationReport",
    "DeckLiteralRect",
    "MimCarve",
    "NetProbe",
    "check_deck_literal_isolation",
    "check_deck_net_isolation",
    "check_gds_label_isolation",
]
