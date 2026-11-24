"""Source-selected edits of proved resources in an imported native layout.

This engine has a separate input/output contract from ordinary routing. It
preserves the imported physical implementation and consumes only explicit
integer-nanometre route-tail declarations. The returned artifact carries the
geometric preservation proof; native device equivalence and foundry signoff
remain the caller's independent obligations.
"""
from __future__ import annotations

from dataclasses import replace

from clay.layout.imported_route_intent import canonical_imported_routes
from clay.layout.intent import LayoutIntent

from .engine import RouteEngine


class ImportedRouteEngine(RouteEngine):
    """Materialize declared edits without inventing a search or routing budget."""

    name = "imported"
    consumes_resource_plan = False
    consumes_imported_route_resources = True

    def __init__(self):
        # Deliberately no **kwargs: a source-selected reliability, via, or
        # search option unsupported by this engine must fail at construction.
        pass

    def configuration(self) -> dict:
        return {
            "schema": "clay.imported-route-engine/v1",
            "name": self.name,
            "constructor_parameters": {},
            "entrypoint": "route_imported",
            "operation": "subtract_declared_integer_nm_from_observed_rectangular_metal_tails",
            "resource_authority": "geometry observation bound to the complete imported manifest SHA-256",
            "internal_search": False,
            "normal_route_cell_result": False,
            "signoff": "NOT_ESTABLISHED",
        }

    def route(self, pins_by_net, obstacles, constraints, pdk, *,
              preroutes=None, resource_plan=None):
        from .imported_route_resources import ImportedRouteUnsupported

        raise ImportedRouteUnsupported(
            "imported engine requires route_imported with an authenticated native layout; "
            "its separate preservation proof cannot be returned as a normal RoutedCell"
        )

    def route_imported(self, observation, *, intent: LayoutIntent,
                       source_sha256, output_gds):
        """Apply every declared tail after checking exact source/net authority.

        The manifest loader authenticates the source-relative manifest path and
        its complete evidence before constructing the observation. This seam
        checks that the source declaration names that same manifest digest.
        It does not infer a path base or substitute an arbitrary new authority.
        """
        from .imported_route_resources import (
            ImportedRouteObservation,
            ImportedRouteUnsupported,
            materialize_imported_route_resources,
        )

        if type(intent) is not LayoutIntent:
            raise TypeError("imported routing requires an exact typed LayoutIntent")
        imported, tails = canonical_imported_routes(intent.imported_layout, intent.route_tails)
        if imported is None:
            raise ImportedRouteUnsupported("imported routing requires a source imported_layout declaration")
        if type(observation) is not ImportedRouteObservation:
            raise TypeError("imported routing requires an owned ImportedRouteObservation")
        observation.require_current()
        observed = observation.canonical_data()
        if observed["source_binding_sha256"] != imported.sha256:
            raise ImportedRouteUnsupported("imported source manifest SHA-256 differs from observed authority")
        resources = {resource.resource_id: resource for resource in observation.resources}
        if len(resources) != len(observation.resources):
            raise ImportedRouteUnsupported("imported observation contains duplicate resource identities")
        trims = {}
        for tail in tails:
            resource = resources.get(tail.resource)
            if resource is None:
                raise ImportedRouteUnsupported("source selects an absent or unsupported route resource")
            if resource.net != tail.net:
                raise ImportedRouteUnsupported(
                    f"source route resource belongs to net {resource.net!r}, not declared net {tail.net!r}"
                )
            trims[tail.resource] = tail.trim_nm
        artifact = materialize_imported_route_resources(
            observation, trims=trims, source_sha256=source_sha256, output_gds=output_gds
        )
        proof = dict(artifact.proof)
        proof.update(
            engine=self.configuration(),
            layout_intent_digest=intent.digest,
            source_imported_layout=imported.canonical_data(),
            declared_route_tails=[tail.canonical_data() for tail in tails],
            consumed_intent_scope="imported_layout_and_route_tails_only",
        )
        return replace(artifact, proof=proof)


__all__ = ["ImportedRouteEngine"]
