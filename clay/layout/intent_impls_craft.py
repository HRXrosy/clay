"""Registered intent implementations for the Stage-1 CRAFT fields (spec Wave B/A5).

Each declarative craft field on :class:`clay.layout.intent.LayoutIntent`
(``guard_rings`` / ``arrays`` / ``common_centroids`` / ``dummy_flanks`` /
``well_islands`` / ``bands`` / ``rail_affinities`` / ``trunks`` / ``power_planes`` /
``splits`` / ``anchors`` / ``channels``) gets exactly ONE registered
:class:`~clay.layout.intent_impl.IntentImplementation` -- a THIN ADAPTER over the
existing body (materializer / checker), keyed on the field KIND, never on a circuit
name. This turns the 12 ``test_intent_elv_matrix`` RED fields into HONEST states:

* ``guard_rings``                       -> PASS      (materialized on the spine +
                                                       independent checker)
* ``arrays`` / ``common_centroids``     -> ADVISORY  (a real materializer, no
                                                       token-gating checker on the field)
* ``rail_affinities``                   -> PASS      (lowers to a placement edge-affinity +
                                                       independent exact-GDS checker)
* ``bands``                              -> PASS      (materialized on the spine +
                                                       exact-GDS checker)
* ``anchors``                            -> PASS      (materialized on the spine +
                                                       exact-GDS checker)
* ``channels``                           -> ADVISORY  (its declaration activates the
                                                       canonical structure-aware placement
                                                       stage; policy only supplies targets)
* every other craft field              -> UNSUPPORTED (parsed on the spine but no
                                                       consumer yet -- a materializer gap,
                                                       named honestly, NOT silently passed)

The honesty is ADVERSARIALLY guarded (spec R4): an UNSUPPORTED-declared field's adapter
must NOT resolve a passing obligation, and the ELV台账 status must equal the adapter's
declared status (``tests/test_intent_registry.py``). No adapter fabricates a PASS to make
a completeness test go green.
"""
from __future__ import annotations

from typing import Any, FrozenSet, Optional, Sequence, Tuple

from clay.core_contract import SourceObligation, VerificationStatus
from clay.layout import intent as _intent
from clay.layout.intent_impl import (
    IntentImplementation,
    LowerResult,
    Provenance,
    StructureRequest,
)
from clay.layout.intent_registry import register
from clay.signoff_verdict import CheckerVerdict

__all__ = ["CRAFT_IMPLS", "register_craft_impls"]

_PASS = VerificationStatus.PASS
_ADV = VerificationStatus.ADVISORY
_UNSUP = VerificationStatus.UNSUPPORTED

# HARD/SOFT hardness (mirrors route.constraints) for the placement requests an adapter
# emits: a rigid macro / symmetry / keep-together is HARD; an edge/region pull is SOFT.
_HARD = "hard"
_SOFT = "soft"


def _labels(decl: Any) -> Tuple[str, ...]:
    """The TOPOLOGY labels a craft declaration references (members / group / net /
    target / passive) -- pure identifiers, NEVER a coordinate. Used to fill a
    :class:`StructureRequest.spec` without leaking geometry."""
    for attr in ("members",):
        v = getattr(decl, attr, None)
        if v:
            return tuple(str(x) for x in v)
    parts = []
    for attr in (
        "first", "second", "group", "target", "net", "passive", "top", "bottom",
        "center", "side", "edge",
    ):
        v = getattr(decl, attr, None)
        if v:
            parts.append(str(v))
    return tuple(parts)


class _CraftAdapter(IntentImplementation):
    """A thin, data-driven adapter over one craft field's existing body. All behaviour
    is parameterised by DATA (kind / status / the generic placement-request kinds it
    lowers to / evidence pointers); there is no per-field code path and no circuit
    branch. Dispatch is nominal on ``kind``."""

    def __init__(
        self,
        *,
        kind: str,
        decl_type: type,
        status: VerificationStatus,
        required: bool = False,
        place_kinds: Sequence[str] = (),
        hardness: str = _HARD,
        lowered_by: Optional[str] = None,
        verified_by: Optional[str] = None,
        provenance: Provenance = Provenance.TOPOLOGY,
        note: str = "",
    ) -> None:
        self.kind = kind
        self._decl_type = decl_type
        self.status = status
        self.required = required
        self._place_kinds: FrozenSet[str] = frozenset(place_kinds)
        self._hardness = hardness
        self.lowered_by = lowered_by
        self.verified_by = verified_by
        self._prov = provenance
        self.note = note

    # -- interface ---------------------------------------------------------------
    def applicable(self, decl: Any, pdk: Any = None) -> bool:
        # match by kind string OR by declaration dataclass type -- NEVER a circuit name.
        if isinstance(decl, str):
            return decl == self.kind
        if isinstance(decl, self._decl_type):
            return True
        return str(getattr(decl, "kind", "")) == self.kind

    def lower(self, decl: Any, ctx: Any = None) -> LowerResult:
        if isinstance(decl, str) or not isinstance(decl, self._decl_type):
            # a kind-string probe carries no declaration to lower.
            return LowerResult()
        spec = _labels(decl)
        reqs = tuple(
            StructureRequest(kind=k, spec=spec, hardness=self._hardness)
            for k in sorted(self._place_kinds)
        )
        return LowerResult(structure_requests=reqs)

    def obligations(self, decl: Any, ctx: Any = None) -> Sequence[SourceObligation]:
        # only a VERIFIED (PASS) field owes a required, checkable obligation; advisory /
        # unsupported fields owe none (they never gate the token) -- honest by tier.
        if self.status != _PASS:
            return ()
        return (
            SourceObligation(
                id=f"intent.{self.kind}",
                observation_id=f"{self.kind}.materialized",
                observation_kind="geometry",
                unit="bool",
                required=self.required,
            ),
        )

    def capabilities(self, decl: Any, backend: Any) -> CheckerVerdict:
        """PRE-handshake 5-state verdict: can ``backend`` satisfy the generic placement
        kinds this decl lowers to? A missing required kind -> UNSUPPORTED (reported,
        never silently dropped). Otherwise the adapter's declared tier."""
        supported = frozenset(getattr(backend, "satisfies", lambda: frozenset())())
        missing = sorted(self._place_kinds - supported)
        obl = f"intent.{self.kind}.capability"
        ev = {
            "kind": self.kind,
            "needs": sorted(self._place_kinds),
            "backend_satisfies": sorted(supported),
        }
        if missing:
            return CheckerVerdict(
                checker="intent_capability",
                obligation_id=obl,
                status=_UNSUP,
                evidence={**ev, "missing": missing},
                blocker_reason=(
                    f"backend does not satisfy placement kind(s) {missing} required by "
                    f"@{self.kind} -- reported UNSUPPORTED (never silently dropped)."
                ),
                required=self.required,
            )
        if self.status == _PASS:
            return CheckerVerdict(
                checker="intent_capability", obligation_id=obl, status=_PASS,
                evidence=ev, required=self.required,
            )
        if self.status == _ADV:
            return CheckerVerdict(
                checker="intent_capability", obligation_id=obl, status=_ADV,
                evidence=ev, required=self.required,
            )
        return CheckerVerdict(
            checker="intent_capability", obligation_id=obl, status=_UNSUP,
            evidence=ev,
            blocker_reason=(
                f"@{self.kind} is parsed but has no spine consumer yet "
                "(materializer gap) -- reported UNSUPPORTED, never silently dropped."
            ),
            required=self.required,
        )

    def provenance(self, decl: Any) -> Provenance:
        return self._prov


class _PassiveSplitAdapter(_CraftAdapter):
    """Verified phase-1 adapter for exact parallel capacitor banks."""

    def capabilities(self, decl: Any, backend: Any) -> CheckerVerdict:
        if isinstance(decl, _intent.SplitIntent):
            connection = str(decl.match).lower()
            if connection != "parallel":
                return CheckerVerdict(
                    checker="intent_capability",
                    obligation_id="intent.splits.capability",
                    status=_UNSUP,
                    evidence={
                        "kind": "splits",
                        "passive": str(decl.passive),
                        "connection": connection,
                    },
                    blocker_reason=(
                        f"@split match={connection} is unsupported: phase 1 "
                        "materializes and verifies parallel capacitor banks only."
                    ),
                    required=True,
                )
        return super().capabilities(decl, backend)


# --- the craft adapters (DATA, one per field) -------------------------------- #
_OBLIG = "clay.layout.intent_obligations"
_RR = "clay.layout.extract.realize_render"
_GRC = "clay.layout.route.guard_ring_check"
_DEV = "clay.layout.device_cell"
_CCA = "clay.layout.common_centroid_array"
_PLACE_LOWER = "clay.layout.place.rail_affinity_lower"
_FLOW_PLACE = "clay.layout.flow_placement"

CRAFT_IMPLS: Tuple[_CraftAdapter, ...] = (
    _CraftAdapter(
        kind="guard_rings", decl_type=_intent.GuardRingIntent,
        status=_PASS, required=True, place_kinds=("keep_together",), hardness=_HARD,
        lowered_by=f"{_RR}:group_guard_rings",
        verified_by=f"{_GRC}:check_guard_ring_materialized",
        note="materialized on the spine (group_guard_rings) + tied ring-tap Pin; the "
             "independent guard_ring_check verifies drawn polarity + tap net.",
    ),
    _CraftAdapter(
        kind="arrays", decl_type=_intent.ArrayIntent,
        status=_ADV, place_kinds=("rigid_macro",), hardness=_HARD,
        lowered_by=f"{_DEV}:build_shared_diffusion_array",
        note="materialized as one shared-OD rigid tile; drives placement, no "
             "token-gating checker on the @array field itself (advisory).",
    ),
    _CraftAdapter(
        kind="common_centroids", decl_type=_intent.CommonCentroidIntent,
        status=_ADV, place_kinds=("symmetry",), hardness=_HARD,
        lowered_by=f"{_CCA}:build_common_centroid_array",
        note="materialized as a 2-row point-reflected tile (symmetry request); "
             "resolver wiring to the spine pending -> advisory, not gating.",
    ),
    _CraftAdapter(
        kind="edge_alignments", decl_type=_intent.EdgeAlignmentIntent,
        status=_ADV, place_kinds=("alignment",), hardness=_HARD,
        lowered_by=(
            "clay.layout.place_blocks_2d:"
            "_bundle_declared_edge_alignment_composites"
        ),
        note="materialized from actual device bbox edges as one rigid component; "
             "strict structure similarity remains the independent pre-route gate.",
    ),
    _CraftAdapter(
        kind="group_center_alignments", decl_type=_intent.GroupCenterAlignment,
        status=_ADV, place_kinds=("alignment",), hardness=_HARD,
        lowered_by="clay.layout.place.placement_relations:_aggregate_hierarchical_macro_alignments_from_intent",
        note="aligns a place-group center with the union of named physical groups.",
    ),
    _CraftAdapter(
        kind="group_edge_alignments", decl_type=_intent.GroupEdgeAlignment,
        status=_ADV, place_kinds=("alignment",), hardness=_HARD,
        lowered_by="clay.layout.place.placement_relations:_hierarchical_macro_constraints_from_intent",
        note="aligns named edges of complete place-group envelopes.",
    ),
    _CraftAdapter(
        kind="group_separations", decl_type=_intent.GroupSeparation,
        status=_ADV, place_kinds=("order",), hardness=_HARD,
        lowered_by="clay.layout.place.placement_relations:_hierarchical_macro_constraints_from_intent",
        note="separates complete place-group envelopes in the specified direction.",
    ),
    _CraftAdapter(
        kind="channel_alignments", decl_type=_intent.ChannelAlignment,
        status=_ADV, place_kinds=("alignment",), hardness=_HARD,
        lowered_by="clay.layout.place.materialized_placement",
        note="aligns x centers of source-owned single physical channels.",
    ),
    _CraftAdapter(
        kind="terminal_accesses", decl_type=_intent.TerminalAccessIntent,
        status=_UNSUP,
        note="parsed and hashed as a device-local side relation; fixed-master "
             "materialization/verification is the remaining explicit gap.",
    ),
    _CraftAdapter(
        kind="rail_affinities", decl_type=_intent.RailAffinityIntent,
        status=_PASS, required=True,
        place_kinds=("edge_affinity",), hardness=_SOFT,
        lowered_by=f"{_PLACE_LOWER}:lower_rail_affinities",
        verified_by=(
            "clay.layout.completeness.gds_invariants:"
            "verify_rail_affinities_on_gds"
        ),
        note="lowers to a placement EDGE-AFFINITY pull (Wave B4): the group is pulled "
             "toward its declared supply-rail edge; exact emitted source-owned artwork "
             "is independently measured against the structural midline. Its capability "
             "is UNSUPPORTED on a backend that does not satisfy edge_affinity.",
    ),
    _CraftAdapter(
        kind="anchors", decl_type=_intent.AnchorIntent,
        status=_PASS, required=True,
        place_kinds=("edge_affinity",), hardness=_SOFT,
        lowered_by=f"{_FLOW_PLACE}:materialize_declared_compact_override",
        verified_by="clay.layout.completeness.gds_invariants:verify_anchors_on_gds",
        note="@anchor pins a group to a cell edge. The canonical structure-aware "
             "placement stage materializes it and exact emitted route/artwork ownership "
             "is independently checked. Unsupported edge variants stop typed.",
    ),
    _CraftAdapter(
        kind="channels", decl_type=_intent.ChannelIntent,
        status=_PASS, required=True, place_kinds=("channel",), hardness=_SOFT,
        lowered_by=f"{_FLOW_PLACE}:materialize_declared_compact_override",
        verified_by="clay.layout.completeness.gds_invariants:verify_channels_on_gds",
        note="@channel(vertical) reserves a device-free corridor beside the "
             "common-centroid left-column owner. The exact checker replays placement "
             "causality and claims the emitted block artwork from GDS. Unsupported "
             "orientation/structure variants stop typed, never no-op.",
    ),
    _CraftAdapter(
        kind="bands", decl_type=_intent.BandIntent, status=_PASS, required=True,
        lowered_by=f"{_FLOW_PLACE}:materialize_declared_compact_override",
        verified_by="clay.layout.completeness.gds_invariants:verify_bands_on_gds",
        note="@band rail-side polarity structure is materialized by the canonical "
             "structure-aware placement stage and independently replayed from the "
             "authoritative source inventory onto exact emitted GDS.",
    ),
    _CraftAdapter(
        kind="well_islands", decl_type=_intent.WellIslandIntent, status=_UNSUP,
        note="@well_island (force a group into its own well) is parsed but not yet "
             "consumed on the spine -> reported UNSUPPORTED.",
    ),
    _CraftAdapter(
        kind="well_island_shares",
        decl_type=_intent.WellIslandShareIntent,
        status=_PASS,
        required=True,
        lowered_by="clay.layout.well_island_sharing:materialize_well_island_share_region",
        verified_by=(
            "clay.layout.completeness.gds_invariants:"
            "verify_well_island_sharing_on_gds"
        ),
        note="@well_island_share compiles canonical group/body ownership into a "
             "PDK well-layer plan, materializes one connected emitted island, and "
             "is independently replayed from exact GDS connected components.",
    ),
    _CraftAdapter(
        kind="dummy_flanks", decl_type=_intent.DummyFlankIntent, status=_UNSUP,
        note="@dummy_flank (generative edge dummies) is parsed but not yet consumed on "
             "the spine -> reported UNSUPPORTED.",
    ),
    _PassiveSplitAdapter(
        kind="splits", decl_type=_intent.SplitIntent, status=_PASS, required=True,
        place_kinds=("rigid_macro",), hardness=_HARD,
        lowered_by="clay.layout.passive_split:realize_passive_split_records",
        verified_by="clay.layout.passive_split:verify_parallel_cap_split",
        note="@split parallel capacitor banks lower to the shared value-per-unit + M "
             "materializer and replay their declared total value; unsupported series "
             "forms fail the capability handshake.",
    ),
    _CraftAdapter(
        kind="trunks", decl_type=_intent.TrunkIntent, status=_UNSUP,
        note="@trunk (signal-trunk layer/orientation) is a routing-side craft intent "
             "parsed but not consumed as a first-class field on the spine -> reported "
             "UNSUPPORTED.",
    ),
    _CraftAdapter(
        kind="power_planes", decl_type=_intent.PowerPlaneIntent, status=_UNSUP,
        note="@power_plane (upper-metal power plane) is parsed but not consumed on the "
             "spine -> reported UNSUPPORTED.",
    ),
)


def register_craft_impls() -> None:
    """Register all craft adapters. Idempotent (re-registers the same kind)."""
    for impl in CRAFT_IMPLS:
        register(impl)


# register on import so the registry is populated wherever it is queried.
register_craft_impls()
