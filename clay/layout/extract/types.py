"""Shared dataclasses for the consistent-intent extractor.

These types form the contract between the three extractor halves:
- netlist_intent.py  (devices + passives + topology pairs + GDS correspondence)
- geometric_intent.py (geometric fields: bands/orders/abuts/aligns/symmetry/...)
- emit_clay.py / roundtrip.py (consumers)

Field names are STABLE. Do not rename; sibling modules import by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedDevice:
    """A single MOS device parsed from the reference netlist.

    W, L are in micrometres (um). M is the multiplicity (SPICE ``m=``).
    nets maps the role keys ``drain`` / ``gate`` / ``source`` / ``body`` to
    netlist node names.
    """

    name: str
    kind: str  # 'nmos' | 'pmos'
    W: float
    L: float
    M: int
    nets: dict


@dataclass
class ExtractedPassive:
    """A resistor or capacitor parsed from the reference netlist.

    value is normalized to base SI units (ohms for 'res', farads for 'cap') and
    is the PER-UNIT value; ``M`` is the parallel multiplicity (SPICE ``m=``,
    default 1) -- the vendor's signed netlist instantiates its Miller cap as
    ``830.76f m=2`` (two parallel MIM units), and its LVS compares per-unit
    wt/lt with m, so collapsing m into value would lose the banked structure.
    ``model`` carries the netlist's ``$.MODEL=`` annotation verbatim ('' when
    absent) -- e.g. 'M5' names the CTM5/METAL5 MIM level, 'NR' the n-poly
    resistor flavour: DESIGN-sourced data a builder may consume.
    nets maps the two terminal keys ``p`` / ``n`` to netlist node names.
    """

    name: str
    kind: str  # 'cap' | 'res'
    value: float
    nets: dict
    M: int = 1
    model: str = ""


@dataclass
class ExtractedIntent:
    """The full extracted intent for one reference cell.

    The netlist half fills devices/passives + topology pairs (matched/diff/
    mirror) and, via correspond_to_gds, device_bboxes. The geometric half
    fills the remaining spatial fields (bands/orders/abuts/aligns/
    equal_size_groups/symmetry_axis/spacings/net_layers).
    """

    cell: str
    devices: list = field(default_factory=list)
    passives: list = field(default_factory=list)
    matched_pairs: list = field(default_factory=list)
    diff_pairs: list = field(default_factory=list)
    mirror_pairs: list = field(default_factory=list)
    device_bboxes: dict = field(default_factory=dict)  # name -> (x0,y0,x1,y1) um
    bands: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    abuts: list = field(default_factory=list)
    aligns: list = field(default_factory=list)
    equal_size_groups: list = field(default_factory=list)
    symmetry_axis: object = None
    spacings: list = field(default_factory=list)
    net_layers: dict = field(default_factory=dict)
    # ----------------------------------------------------------------------- #
    # NOT-YET-implemented layout primitives carried for the emitter roadmap.
    #
    # These four fields name analog-layout primitives that already exist in the
    # Python extract pipeline (dcsolve edge-align / fold_intent / power_intent /
    # the perturbation don't-care classifier) but have NO Clay grammar surface
    # yet. The emitter records them as structured ``// @EXTRACTED-LAYOUT:``
    # comment lines (parse-safe, greppable) so the constructs are NAMED + EMITTED
    # without touching the production parser. Default empty -> existing callers
    # are unchanged; see emit_clay.py and the 2026-06-07 primitive-additions spec.
    #
    #   edge_aligns  : [(a, b, edge)]  edge in {left,right,top,bottom}
    #                  -- the dcsolve CORE primitive: two devices share an EDGE
    #                  (size-derived offset), distinct from center @align.
    #   folds        : {device -> (n_rows, fingers_per_row)}
    #                  -- the fold_intent.FoldChoice multi-row device fold.
    #   rails        : [(net, [devices])]
    #                  -- a validated power-rail group (power_intent) aligning a
    #                  set of devices on a shared supply.
    #   free_devices : [device]
    #                  -- a device whose absolute position is a declared design
    #                  freedom (the don't-care label from the classifier).
    edge_aligns: list = field(default_factory=list)
    folds: dict = field(default_factory=dict)
    rails: list = field(default_factory=list)
    free_devices: list = field(default_factory=list)
    #   no_route_under : [{'cap': name-or-'*', 'forbidden_metals': [int levels],
    #                      'provenance': 'declared'|'style-extracted',
    #                      'evidence': {...}}]
    #                  -- the @no_route_under layout intent: routing on the
    #                  listed metal LEVELS must stay out of the matching MIM
    #                  cap footprints. DECLARABLE by the designer, or
    #                  style-extracted from a reference as a relative/boolean
    #                  fact (which levels the reference keeps clear -- never a
    #                  coordinate). Lowered by clay.layout.model into hard
    #                  keep-out conductors; obligation = zero foreign metal of
    #                  those levels inside the footprints on the routed artwork.
    no_route_under: list = field(default_factory=list)
    #   edge_anchor_devices : [{'device': name-or-'*'|kind 'cap',
    #                           'provenance': 'declared'}]
    #                  -- the @edge_anchor layout intent: a large passive (a MIM
    #                  cap) anchors FLUSH against ONE die edge so that side
    #                  carries NO active circuit and the die does NOT grow. A
    #                  DECLARED design prior, not a reference-extracted fact
    #                  (provenance is always 'declared'; no coordinate and no
    #                  edge ever enters the entry -- the anchor edge is SOLVED).
    #                  Lowered by clay.layout.extract.place_relax into an
    #                  EDGE-ANCHOR potential (the matched device is pulled so one
    #                  side goes flush to the nearest die wall WITHOUT pushing the
    #                  wall outward); obligation = the cap's open side carries
    #                  zero active area AND the outline stays within the canonical
    #                  area budget (clay.layout.extract.edge_anchor). Default
    #                  empty -> byte-identical placement (the relax energy term is
    #                  gated on a matching entry).
    edge_anchor_devices: list = field(default_factory=list)
    #   edge_anchor_io_edges : ['left'|'right'|'top'|'bottom', ...]
    #                  -- the set of die edges already CLAIMED by an IO-edge
    #                  intent (extract_io_edge_intent brings each labelled signal
    #                  net to a block edge: VIP/VIN enter left, VOUT exits right).
    #                  A CATEGORICAL fact (a side token, never a coordinate). The
    #                  @edge_anchor lowering reads it so a cap's SOLVED clear side
    #                  is chosen from the edges IO does NOT occupy (priority 1: the
    #                  cap must not turn its open side into a wall an IO terminal
    #                  needs). Default empty -> the anchor picks the most-room axis
    #                  with no IO exclusion (byte-identical to the prior path).
    edge_anchor_io_edges: list = field(default_factory=list)
    #   placement_relations : RelationNetwork | None
    #                  -- the P2-EXTRACTED zero-coordinate QSR relation network
    #                  (clay.layout.extract.placement_relations.RelationNetwork).
    #                  When attached, the P3 edge-anchor lowering DERIVES its
    #                  member set from this network's anchored device<->frame TPP
    #                  relations (a cap that hugs a die boundary side) instead of
    #                  requiring a declared @edge_anchor prior -- the extract->lower
    #                  loop closed (intent EXTRACTED via style-transfer, not
    #                  DECLARED). The network is zero-coordinate by construction
    #                  (placement_relations.py P1 guard): the derivation reads ONLY
    #                  relation symbols, never device_bboxes -- the leakage
    #                  boundary. Default None -> only the declared source is
    #                  consulted (byte-identical to the prior path).
    placement_relations: object = None
    #   guard_islands : [{'members': (name, ...), 'well': 'p'|'n', 'tie': net}]
    #                  -- the DECLARED @guard_ring islands carried ACROSS the
    #                  LayoutIntent -> ExtractedIntent boundary (LayoutIntent.
    #                  guard_rings is otherwise DROPPED here, so the placement
    #                  grouping never learns the declared island partition and
    #                  fuses same-(kind,body) singletons ACROSS declared rings --
    #                  the straddle proved RED by
    #                  clay.layout.route.ring_integrity_check). Each entry names
    #                  ONE declared ring's member cluster + its well polarity +
    #                  tie net. Consumed by
    #                  well_islands.compact_matched_groups (the compact grouping
    #                  REFINES a declared island, never straddles it -- the
    #                  step-3 fold key gains island membership) and by
    #                  realize_render.group_guard_rings (declared-island honoring
    #                  + passive-cluster ring materialization). Populate it at the
    #                  conversion site from LayoutIntent.guard_rings (each
    #                  GuardRingIntent -> {'members','well','tie'}). Default empty
    #                  -> island_of empty -> the fold key gains a constant None
    #                  and the partition is BYTE-IDENTICAL (a cell that declares
    #                  no @guard_ring is unchanged). Pure declared data, no
    #                  coordinate.
    guard_islands: list = field(default_factory=list)
    # RETIRED (U1-U3, 2026-06-19): the @cap_over_nodes / @cap_planar_edge intent
    # slots are GONE. Both were cap-specific placement bolt-ons (the cap parked
    # over-cell, or beside the cluster as a planar side block) that the UNIFIED
    # all-element CP placement (clay.layout.synth.placement.cp_group_placement_pos)
    # subsumes: a cap is now a first-class TILE co-placed in the one envelope-min
    # pack via the net-wirelength term -- no cap branch. Their checkers
    # (check_cap_over_nodes / check_cap_planar_edge), modules (extract/cap_over_nodes.py,
    # extract/cap_planar_edge.py), and the cap-center proxy verdict are deleted; the
    # GENERIC measure_dead_region / measure_fill mosaic-completeness obligation
    # replaces them. No replacement field is needed (the cap carries no special
    # placement intent).
