"""The CLOSED placement-side generic vocabulary + capability surface (spec Wave B / P1).

Mirrors :mod:`clay.layout.route` on the placement side: a closed set of generic
placement constraints (:mod:`.constraints`), a frozen constraint graph
(:mod:`.graph`), a queryable placement-engine capability (:mod:`.engine`), and the
rail-affinity edge-affinity lowering (:mod:`.rail_affinity_lower`). The placer consumes
these generic physical objects; it never learns a circuit role.

The deterministic rigid-block path uses one implementation owner per concern.
:mod:`.cp_model_builder` constructs and solves the native CP-SAT models;
:mod:`.placement_replay` independently checks their complete hard domain.
Spatial, hierarchical, and well/guard predicates live in their respective
``constraints_*`` modules. Source-to-master geometry and declaration projection
are owned by :mod:`.placement_geometry` and :mod:`.placement_relations`.
:mod:`.materialized_placement` orchestrates the actual geometry transition,
while :mod:`.placement_certificates` binds its final result.

:mod:`.full2d_compactor` preserves the native domain during full-2D refinement;
:mod:`.lp_compactor` compacts an already-selected axis topology with HiGHS.
The historical :mod:`clay.layout.place_blocks_2d` module re-exports these exact
objects and preserves instrumentation at their actual module globals.
"""
