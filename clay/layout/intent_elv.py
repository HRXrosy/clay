"""Intent vocabulary E/L/V matrix: Expressible / Lowered / Verified status per term.

Dev-discipline rule 4: an intent term must not stop at the parser. This module is the
台账 -- every layout intent term (a field on :class:`clay.layout.intent.LayoutIntent` /
its ``@decorator``) records:

* **E** Expressible -- the syntax/field exists (always true here; these ARE fields);
* **L** Lowered -- it lowers to an obligation OR drives the placer/router (``lowered_by``);
* **V** Verified -- a checker decides it on the artifact (``verified_by``).

The matrix is EVIDENCE-BACKED: ``lowered_by`` / ``verified_by`` are dotted ``module:attr``
paths that must actually resolve, so a term cannot CLAIM verified without a real checker.
The status is the term's honest tier:

* ``PASS``        -- VERIFIED: a checker gates it (E+L+V);
* ``ADVISORY``    -- lowered/drives placement but never gates the proof token (E+L, no V);
* ``UNSUPPORTED`` -- expressible only / declared-but-unverified (no metric yet) (E, no V).

A completeness test asserts EVERY declarative LayoutIntent field has a row here, so a
newly-added field fails the gate until its E/L/V is filled in.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from clay.core_contract import VerificationStatus

__all__ = ["IntentTerm", "INTENT_ELV", "intent_elv_matrix", "declarative_intent_fields",
           "resolve"]

# LayoutIntent fields that are PROVENANCE/meta, not declarative intent terms (excluded
# from the completeness requirement).
_NON_TERM_FIELDS = {"cell", "profile_id", "source"}


@dataclass(frozen=True)
class IntentTerm:
    """One intent vocabulary term's E/L/V record."""

    name: str
    field: str                       # the LayoutIntent field it lives on
    syntax: str                      # the surface syntax (@decorator / role-derived)
    status: VerificationStatus       # PASS=verified, ADVISORY=lowered-only, UNSUPPORTED
    lowered_by: Optional[str] = None  # "module:attr" lowering / placer-router driver
    verified_by: Optional[str] = None  # "module:attr" checker (required if status==PASS)
    note: str = ""

    @property
    def expressible(self) -> bool:
        return True  # every term here is a real field -> always expressible

    @property
    def lowered(self) -> bool:
        return self.lowered_by is not None

    @property
    def verified(self) -> bool:
        return self.verified_by is not None and self.status == VerificationStatus.PASS


def resolve(dotted: str) -> Any:
    """Resolve a ``"module:attr"`` evidence pointer to the object (callable checker /
    lowering). Raises on a dangling pointer -- the matrix cannot cite evidence that does
    not exist."""
    mod_name, _, attr = dotted.partition(":")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


# The intent vocabulary台账. Each row's status is cross-checked against its evidence by the
# tests (a PASS row MUST resolve a real checker).
_OBLIG = "clay.layout.intent_obligations"
_GEOM = "clay.layout.geometric_intent_checks"
_PRIORITY = "clay.layout.spec.priority"
# Stage-1 CRAFT field evidence (Wave B/A5: each craft field's registered impl body).
_RR = "clay.layout.extract.realize_render"
_GRC = "clay.layout.route.guard_ring_check"
_DEV = "clay.layout.device_cell"
_CCA = "clay.layout.common_centroid_array"
_RAIL = "clay.layout.place.rail_affinity_lower"

INTENT_ELV: Dict[str, IntentTerm] = {
    "parasitic_budget": IntentTerm(
        "parasitic_budget", "budgets", "@match / @parasitic (role-derived)",
        VerificationStatus.PASS, lowered_by=f"{_OBLIG}:check_layout_intent",
        verified_by=f"{_OBLIG}:check_layout_intent",
        note="matched_cc_balance / parasitic_cap / clk_regen / kickback cap budgets, "
             "checked on real PEX.",
    ),
    "diff_pair_balance": IntentTerm(
        "diff_pair_balance", "diff_pairs", "role-derived (differential roles)",
        VerificationStatus.PASS, lowered_by=f"{_OBLIG}:check_layout_intent",
        verified_by=f"{_OBLIG}:check_layout_intent",
        note="each diff pair -> a matched_cc_balance budget.",
    ),
    "shield": IntentTerm(
        "shield", "shields", "@shield",
        VerificationStatus.PASS, lowered_by=f"{_OBLIG}:check_shield_intent",
        verified_by=f"{_OBLIG}:check_shield_intent",
    ),
    "lde_budget": IntentTerm(
        "lde_budget", "lde_budgets", "@equal_well_edge / @equal_od / @equal_od_spacing",
        VerificationStatus.PASS, lowered_by=f"{_OBLIG}:_check_lde_budgets",
        verified_by=f"{_OBLIG}:_check_lde_budgets",
        note="per-device LDE matching over device geometry.",
    ),
    "wire_width": IntentTerm(
        "wire_width", "wire_widths", "@wire_width(net, w)",
        VerificationStatus.PASS, lowered_by=f"{_OBLIG}:_check_wire_width_intents",
        verified_by=f"{_GEOM}:check_wire_width",
        note="drives the router AND a geometric obligation verifies realised width.",
    ),
    "match_length": IntentTerm(
        "match_length", "match_length_pairs", "@match_length(a, b)",
        VerificationStatus.PASS, lowered_by=f"{_OBLIG}:_check_match_length_intents",
        verified_by=f"{_GEOM}:check_match_length",
        note="hard router Length constraint plus final routed-length balance check.",
    ),
    "order": IntentTerm(
        "order", "order_constraints", "@order(a, b, rel)",
        VerificationStatus.PASS, lowered_by=f"{_GEOM}:check_order",
        verified_by=f"{_GEOM}:check_order",
        note="drives seqpair placer (H/V edge) by construction + check_order verifies.",
    ),
    "abut": IntentTerm(
        "abut", "abut_pairs", "@abut(a, b)",
        VerificationStatus.PASS, lowered_by=f"{_GEOM}:check_abut",
        verified_by=f"{_GEOM}:check_abut",
        note="drives placer (tight adjacency) + check_abut verifies facing gap.",
    ),
    "place_group": IntentTerm(
        "place_group", "place_groups", "@place_group",
        VerificationStatus.PASS, lowered_by=f"{_GEOM}:check_place_group_realized",
        verified_by=f"{_GEOM}:check_place_group_realized",
    ),
    "priority": IntentTerm(
        "priority", "priorities", "@priority(target, tier)",
        VerificationStatus.PASS, lowered_by=f"{_PRIORITY}:resolve_priority",
        verified_by=f"{_PRIORITY}:resolve_priority",
        note="intent-priority axis: pins an obligation's tier (LAW > CORRECTNESS > "
             "QUALITY > OBJECTIVE). LOWERED to the resolver (resolve_priority "
             "adjudicates conflicting obligations by tier); VERIFIED by the resolver "
             "emitting an explicit CONFLICT verdict for unresolved same-tier ties "
             "(no silent priority). Acyclicity of '>' is enforced at lowering.",
    ),
    "superiority": IntentTerm(
        "superiority", "superiority", "A > B (intra-tier defeasible superiority)",
        VerificationStatus.PASS, lowered_by=f"{_PRIORITY}:resolve_priority",
        verified_by=f"{_PRIORITY}:resolve_priority",
        note="intent-priority axis: a defeats b when they conflict at the SAME tier "
             "(resolver's declared-superiority branch). LOWERED to the resolver; "
             "VERIFIED by the resolver -- a declared superiority resolves the "
             "conflict, its absence emits a CONFLICT verdict. The '>' partial order "
             "is checked ACYCLIC at lowering (a cycle raises, not a silent drop).",
    ),
    "mirror_pair": IntentTerm(
        "mirror_pair", "mirror_pairs", "role-derived (current-mirror roles)",
        VerificationStatus.ADVISORY, lowered_by=None, verified_by=None,
        note="asymmetric by design -> NO cap budget; consumed by placement_symmetry as a "
             "placement hint, NOT lowered to a token-gating obligation (advisory).",
    ),
    "objective": IntentTerm(
        "objective", "objective", "@objective(target, goal)",
        VerificationStatus.ADVISORY, lowered_by=None, verified_by=None,
        note="placement directive consumed by the seqpair placer; never lowers to a "
             "gating obligation / gates the proof token (advisory).",
    ),
    "area_max": IntentTerm(
        "area_max", "area_max_um2", "@area(max)",
        VerificationStatus.ADVISORY, lowered_by=None, verified_by=None,
        note="placement directive consumed by the placer; never gates the proof token "
             "(advisory).",
    ),
    "resistance_budget": IntentTerm(
        "resistance_budget", "resistance_budgets", "@resistance",
        VerificationStatus.UNSUPPORTED, lowered_by=None, verified_by=None,
        note="declared but UNVERIFIED -- no PEX resistance metric yet. Escalation: a PEX "
             "res backend + a resistance obligation checker.",
    ),
    "dynamic_net": IntentTerm(
        "dynamic_net", "dynamic_nets", "@dynamic",
        VerificationStatus.ADVISORY, lowered_by=f"{_OBLIG}:check_layout_intent",
        verified_by=None,
        note="net classification INPUT; its effect is verified transitively via the "
             "budgets/shields it produces.",
    ),
    "sensitive_net": IntentTerm(
        "sensitive_net", "sensitive_nets", "@sensitive",
        VerificationStatus.ADVISORY, lowered_by=f"{_OBLIG}:check_layout_intent",
        verified_by=None,
        note="net classification INPUT; effect verified transitively via budgets/shields.",
    ),
    # ── Stage-1 CRAFT fields (Wave B/A5). Each has ONE registered IntentImplementation
    #    (clay.layout.intent_impls_craft) keyed on the field KIND; the ELV status MUST
    #    equal the registered impl's declared status (tests/test_intent_registry.py). No
    #    row fabricates a PASS -- every state is the HONEST spine reality.
    "guard_rings": IntentTerm(
        "guard_rings", "guard_rings", "@guard_ring(group, well, tie)",
        VerificationStatus.PASS, lowered_by=f"{_RR}:group_guard_rings",
        verified_by=f"{_GRC}:check_guard_ring_materialized",
        note="materialized on the spine (group_guard_rings) + tied ring-tap Pin; the "
             "independent guard_ring_check verifies drawn polarity + tap net.",
    ),
    "arrays": IntentTerm(
        "arrays", "arrays", "@array(group, style, rows, order)",
        VerificationStatus.ADVISORY, lowered_by=f"{_DEV}:build_shared_diffusion_array",
        verified_by=None,
        note="materialized as one shared-OD rigid tile (rigid_macro request); drives "
             "placement, no token-gating checker on the @array field itself (advisory).",
    ),
    "common_centroids": IntentTerm(
        "common_centroids", "common_centroids", "@common_centroid(group, rows)",
        VerificationStatus.ADVISORY, lowered_by=f"{_CCA}:build_common_centroid_array",
        verified_by=None,
        note="materialized as a 2-row point-reflected tile (symmetry request); resolver "
             "wiring to the spine pending -> advisory, not gating.",
    ),
    "edge_alignments": IntentTerm(
        "edge_alignments", "edge_alignments", "@align(a, b, edge)",
        VerificationStatus.ADVISORY,
        lowered_by=(
            "clay.layout.place_blocks_2d:"
            "_bundle_declared_edge_alignment_composites"
        ),
        verified_by=None,
        note="materialized as a coordinate-free rigid placement component using "
             "actual device edge offsets; strict structure similarity supplies the "
             "independent pre-route acceptance gate.",
    ),
    "group_center_alignments": IntentTerm(
        "group_center_alignments", "group_center_alignments", "@center_align(group, groups)",
        VerificationStatus.ADVISORY,
        lowered_by="clay.layout.place.placement_relations:_aggregate_hierarchical_macro_alignments_from_intent",
        verified_by=None,
        note="materialized and replayed as an alignment of physical group envelopes.",
    ),
    "group_edge_alignments": IntentTerm(
        "group_edge_alignments", "group_edge_alignments", "@group_align(first, second, edge)",
        VerificationStatus.ADVISORY,
        lowered_by="clay.layout.place.placement_relations:_hierarchical_macro_constraints_from_intent",
        verified_by=None, note="aligns and replays complete group envelope edges.",
    ),
    "group_separations": IntentTerm(
        "group_separations", "group_separations", "@group_separate(before, after, axis)",
        VerificationStatus.ADVISORY,
        lowered_by="clay.layout.place.placement_relations:_hierarchical_macro_constraints_from_intent",
        verified_by=None, note="separates and replays complete group envelopes.",
    ),
    "channel_alignments": IntentTerm(
        "channel_alignments", "channel_alignments", "@channel_align(first, second)",
        VerificationStatus.ADVISORY,
        lowered_by="clay.layout.place.materialized_placement",
        verified_by=None, note="binds x alignment of two source-owned single channels.",
    ),
    "terminal_accesses": IntentTerm(
        "terminal_accesses", "terminal_accesses",
        "@terminal_access(device, terminal, side)",
        VerificationStatus.UNSUPPORTED,
        lowered_by=None,
        verified_by=None,
        note="device-local access-side relation is parsed and hashed; fixed-master "
             "consumption is not promoted until its materializer and independent "
             "local-wrapper comparison are both wired.",
    ),
    "rail_affinities": IntentTerm(
        "rail_affinities", "rail_affinities", "@rail_affinity(group, side)",
        VerificationStatus.PASS, lowered_by=f"{_RAIL}:lower_rail_affinities",
        verified_by=(
            "clay.layout.completeness.gds_invariants:"
            "verify_rail_affinities_on_gds"
        ),
        note="lowers to a placement EDGE-AFFINITY pull (Wave B4): the group is drawn "
             "toward its declared supply-rail edge in place_envelope_min. The final "
             "gate independently claims source-owned exact-GDS artwork and measures "
             "the group against the structural midline.",
    ),
    "anchors": IntentTerm(
        "anchors", "anchors", "@anchor(target, edge)",
        VerificationStatus.PASS,
        lowered_by="clay.layout.flow_placement:materialize_declared_compact_override",
        verified_by="clay.layout.completeness.gds_invariants:verify_anchors_on_gds",
        note="edge-access placement pin materialized by the canonical structure-aware "
             "compact stage and independently checked from exact route/artwork "
             "ownership. Unsupported edge variants stop typed.",
    ),
    "channels": IntentTerm(
        "channels", "channels", "@channel(orient, width)",
        VerificationStatus.PASS,
        lowered_by="clay.layout.flow_placement:materialize_declared_compact_override",
        verified_by="clay.layout.completeness.gds_invariants:verify_channels_on_gds",
        note="vertical device-free routing corridor materialized beside the declared "
             "common-centroid owner and independently measured on exact GDS artwork. "
             "Unsupported orientation/structure variants stop typed rather than no-op.",
    ),
    "densities": IntentTerm(
        "densities", "densities", "@density(max_whitespace_fraction, scope=structural)",
        VerificationStatus.PASS,
        lowered_by="clay.layout.flow_placement:materialize_declared_compact_override",
        verified_by=(
            "clay.layout.completeness.gds_invariants:"
            "verify_density_whitespace_on_gds"
        ),
        note="normalized structural occupancy drives dense compaction and is "
             "independently measured from uniquely owned exact-GDS artwork.",
    ),
    "od_fills": IntentTerm(
        "od_fills", "od_fills", "@od_fill(min_fraction, scope=core)",
        VerificationStatus.PASS,
        lowered_by="clay.layout.flow_placement:materialize_declared_compact_override",
        verified_by=(
            "clay.layout.completeness.gds_invariants:"
            "verify_functional_od_fill_on_gds"
        ),
        note="source-owned functional active occupancy drives dense compaction and "
             "is independently rebuilt from canonical PCells, uniquely claimed "
             "from the exact emitted GDS, and measured in the final core envelope.",
    ),
    "bands": IntentTerm(
        "bands", "bands", "@band(top, bottom, center)",
        VerificationStatus.PASS,
        lowered_by="clay.layout.flow_placement:materialize_declared_compact_override",
        verified_by="clay.layout.completeness.gds_invariants:verify_bands_on_gds",
        note="rail-side polarity banding materialized by the canonical structure-aware "
             "compact stage and independently replayed from authoritative source "
             "membership onto exact emitted GDS.",
    ),
    "well_islands": IntentTerm(
        "well_islands", "well_islands", "@well_island(group, body)",
        VerificationStatus.UNSUPPORTED, lowered_by=None, verified_by=None,
        note="force a group into its own well -- parsed but not yet consumed on the "
             "spine -> reported UNSUPPORTED.",
    ),
    "well_island_shares": IntentTerm(
        "well_island_shares", "well_island_shares",
        "@well_island_share(group, body)",
        VerificationStatus.PASS,
        lowered_by=(
            "clay.layout.well_island_sharing:"
            "materialize_well_island_share_region"
        ),
        verified_by=(
            "clay.layout.completeness.gds_invariants:"
            "verify_well_island_sharing_on_gds"
        ),
        note="canonical group/body ownership compiles to one PDK well layer, is "
             "materialized as a connected emitted island, and is independently "
             "checked from exact-GDS connected components.",
    ),
    "dummy_flanks": IntentTerm(
        "dummy_flanks", "dummy_flanks", "@dummy_flank(target, per_end, tie)",
        VerificationStatus.UNSUPPORTED, lowered_by=None, verified_by=None,
        note="generative edge dummies -- parsed but not yet consumed on the spine -> "
             "reported UNSUPPORTED.",
    ),
    "splits": IntentTerm(
        "splits", "splits", "@split(passive, n, match)",
        VerificationStatus.PASS,
        lowered_by="clay.layout.passive_split:realize_passive_split_records",
        verified_by="clay.layout.passive_split:verify_parallel_cap_split",
        note="parallel capacitor split is materialized as a value-per-unit + M bank "
             "and checked by electrical-value replay; unsupported series forms fail "
             "the per-declaration capability handshake.",
    ),
    "trunks": IntentTerm(
        "trunks", "trunks", "@trunk(net, layer, orient)",
        VerificationStatus.UNSUPPORTED, lowered_by=None, verified_by=None,
        note="signal-trunk layer/orientation -- parsed but not consumed as a first-class "
             "field on the spine -> reported UNSUPPORTED.",
    ),
    "power_planes": IntentTerm(
        "power_planes", "power_planes", "@power_plane(net, layer)",
        VerificationStatus.UNSUPPORTED, lowered_by=None, verified_by=None,
        note="upper-metal power plane -- parsed but not consumed on the spine -> "
             "reported UNSUPPORTED.",
    ),
}


def declarative_intent_fields() -> Set[str]:
    """The set of LayoutIntent fields that are declarative intent TERMS (provenance/meta
    fields excluded). Used by the completeness gate."""
    import dataclasses

    from clay.layout.intent import LayoutIntent

    return {
        f.name for f in dataclasses.fields(LayoutIntent)
        if f.name not in _NON_TERM_FIELDS
    }


def intent_elv_matrix() -> Dict[str, Dict[str, Any]]:
    """The full E/L/V matrix as plain rows (term -> {expressible, lowered, verified,
    status, field, syntax, note})."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, t in INTENT_ELV.items():
        out[name] = {
            "field": t.field,
            "syntax": t.syntax,
            "expressible": t.expressible,
            "lowered": t.lowered,
            "verified": t.verified,
            "status": t.status.value,
            "note": t.note,
        }
    return out
