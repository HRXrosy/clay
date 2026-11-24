"""Independent exact-GDS closure for device-level ``@adjacent`` intent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

@dataclass(frozen=True)
class GdsAdjacentObligation:
    source: str
    target: str
    declaration_sha256: str
    gds_sha256: str | None
    maximum_separation_nm: int | None
    measured_separation_nm: int | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    reason: str


def _union_bbox(claims: Mapping[tuple[int, int], object]) -> tuple[int, int, int, int]:
    boxes = [record.bbox for records in claims.values() for record in records]
    if not boxes:
        raise ValueError("adjacent endpoint has no claimed exact-GDS artwork")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _edge_gap_nm(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> int | None:
    x_overlap = min(left[2], right[2]) - max(left[0], right[0])
    y_overlap = min(left[3], right[3]) - max(left[1], right[1])
    gaps = []
    if y_overlap > 0:
        gaps.append(max(right[0] - left[2], left[0] - right[2]))
    if x_overlap > 0:
        gaps.append(max(right[1] - left[3], left[1] - right[3]))
    return min(gaps) if gaps else None


def _canonical_passive_bank_scopes(
    source_cell: object,
    canonical_intent: object,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Replay source-owned passive-bank membership for adjacency envelopes."""

    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent

    graph = physical_macro_graph_from_intent(
        source_cell, canonical_intent, device_records={}
    )
    scopes: dict[str, tuple[str, ...]] = {}
    member_to_scope: dict[str, str] = {}
    for macro in graph.macros:
        if macro.kind != "passive_bank":
            continue
        scopes[macro.macro_id] = macro.members
        for member in macro.members:
            if member in member_to_scope:
                raise ValueError(
                    f"adjacent member {member!r} belongs to multiple passive banks"
                )
            member_to_scope[member] = macro.macro_id
    return scopes, member_to_scope


def verify_adjacencies_on_gds(
    gds_path: str | Path,
    *,
    top_cell: str,
    adjacent_intents: object,
    exact_artifact: object,
    source_cell: object,
    tech: object,
    emitted_device_geometry: Mapping[str, object] | None,
    emitted_device_placement_um: Mapping[str, object] | None,
    final_placement_certificate: object = None,
    placement_state_sha256: str | None = None,
) -> tuple[GdsAdjacentObligation, ...]:
    """Claim both endpoint footprints in the bound GDS and measure their edge gap."""

    from clay.layout.completeness.gds_invariants import (
        _certified_assembly_transformed_layers,
        _emitted_shape_witness,
        _replay_exact_final_placement_artwork,
        _unique_emitter_owner,
    )
    from clay.layout.intent import (
        layout_intent_from_cell_def,
        replay_adjacent_inventory,
    )
    from clay.layout.place_blocks_2d import _installed_adjacency_pitch_nm
    from clay.layout.route.exact_artifact_evidence import (
        ExactEmittedArtifact,
        _replayed_source_cell_authority,
    )

    declarations = replay_adjacent_inventory(adjacent_intents)
    if not declarations:
        raise ValueError("adjacent exact-GDS checker requires a non-empty denominator")

    def blocked(reason: str, digest: str | None = None):
        return tuple(
            GdsAdjacentObligation(
                source=item.source,
                target=item.target,
                declaration_sha256=item.sha256,
                gds_sha256=digest,
                maximum_separation_nm=None,
                measured_separation_nm=None,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    try:
        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("adjacent checker requires ExactEmittedArtifact")
        if not isinstance(emitted_device_geometry, Mapping) or not isinstance(
            emitted_device_placement_um, Mapping
        ):
            raise TypeError("adjacent checker requires emitter geometry authority")
        source_sha, inventory_sha, _classes, canonical_cell = (
            _replayed_source_cell_authority(source_cell)
        )
        if (
            source_sha != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("adjacent source inventory differs from emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_cell)
        canonical = replay_adjacent_inventory(canonical_intent.adjacencies)
        if declarations != canonical:
            raise ValueError("adjacent declarations differ from canonical source bytes")
        from clay.layout.flow_placement import declared_craft_compaction_requested

        has_final_placement = (
            final_placement_certificate is not None
            or placement_state_sha256 is not None
        )
        if declared_craft_compaction_requested(None, canonical_intent) and not (
            final_placement_certificate is not None
            and placement_state_sha256 is not None
        ):
            raise ValueError(
                "adjacent compact materialization lacks final placement authority"
            )
        if has_final_placement:
            (
                _replayed_intent,
                _member_classes,
                transformed_layers,
                witness,
            ) = _replay_exact_final_placement_artwork(
                gds_path=str(gds_path),
                top_cell=top_cell,
                exact_artifact=exact_artifact,
                source_cell=canonical_cell,
                final_placement_certificate=final_placement_certificate,
                placement_state_sha256=placement_state_sha256,
                emitted_device_geometry=emitted_device_geometry,
                emitted_device_placement_um=emitted_device_placement_um,
                tech=tech,
            )
        else:
            transformed_layers, witness = _certified_assembly_transformed_layers(
                exact_artifact,
                str(gds_path),
                emitted_device_geometry,
                emitted_device_placement_um,
                return_witness=True,
                top_cell=top_cell,
            )
        if witness is None:
            raise ValueError("adjacent exact-GDS witness is absent")

        owners: dict[str, str] = {}
        members = tuple(sorted(
            {name for item in declarations for name in (item.source, item.target)}
        ))
        for member in members:
            owner = _unique_emitter_owner(member, emitted_device_geometry)
            owners[member] = owner

        def combined_artwork(stamp_names: tuple[str, ...]):
            combined = {}
            for stamp_name in stamp_names:
                placement = emitted_device_placement_um.get(stamp_name)
                if placement is None:
                    raise ValueError(
                        f"adjacent stamp {stamp_name!r} has no emitter placement"
                    )
                artwork = _emitted_shape_witness(
                    emitted_device_geometry[stamp_name],
                    placement,
                    excluded_layers=transformed_layers,
                )
                for layer, shapes in artwork.items():
                    combined.setdefault(layer, []).extend(shapes)
            for shapes in combined.values():
                shapes.sort()
            if not combined:
                raise ValueError("adjacent structural envelope has no emitter artwork")
            return combined

        relation_scopes = {}
        expected_by_level: dict[str, dict[str, object]] = {}
        if has_final_placement:
            from clay.layout.route.route_cell import (
                build_final_placement_resource_evidence,
            )

            resource = build_final_placement_resource_evidence(
                dict(emitted_device_placement_um),
                dict(emitted_device_geometry),
                canonical_intent,
                tech,
                final_placement_certificate=final_placement_certificate,
            )
            index = resource.cluster_index
            if resource.status != "MEASURABLE" or index is None:
                raise ValueError("adjacent final placement resources are unmeasurable")
            passive_scopes, member_to_passive_scope = (
                _canonical_passive_bank_scopes(canonical_cell, canonical_intent)
            )
            # The final certificate's master partition is useful solver evidence,
            # but it is not a canonical source fact: a caller can construct a
            # different self-consistent partition of the same clusters.  Scope
            # adjacency only with the cluster index independently rebuilt from
            # canonical intent plus the exact emitted geometry.  The certificate
            # remains mandatory for placement/artwork replay above, never for
            # choosing the measurement envelope.
            endpoint_hierarchy = {}
            for member in members:
                cluster = index.stamp_for_member(member)
                if cluster is None:
                    raise ValueError(
                        f"adjacent member {member!r} lacks final-cluster ownership"
                    )
                endpoint_hierarchy[member] = (
                    member_to_passive_scope.get(member), cluster, owners[member]
                )

            def scope_stamps(level: str, scope: str) -> tuple[str, ...]:
                if level == "owner":
                    return (scope,)
                if level == "cluster":
                    clusters = (scope,)
                else:
                    raw_clusters = {
                        index.stamp_for_member(member)
                        for member in passive_scopes[scope]
                    }
                    if None in raw_clusters:
                        raise ValueError(
                            "adjacent passive-bank member lacks final-cluster ownership"
                        )
                    clusters = tuple(sorted(raw_clusters))
                if not clusters:
                    raise ValueError(
                        "adjacent passive bank has no final-cluster ownership"
                    )
                return tuple(sorted({
                    name
                    for cluster in clusters
                    for name in index.cluster(cluster).members
                    if name in emitted_device_geometry
                }))

            def add_scope(level: str, scope: str) -> str:
                key = f"{level}:{scope}"
                expected_by_level.setdefault(level, {})[key] = combined_artwork(
                    scope_stamps(level, scope)
                )
                return key

            for declaration in declarations:
                left = endpoint_hierarchy[declaration.source]
                right = endpoint_hierarchy[declaration.target]
                if left[0] != right[0]:
                    left_level = "macro" if left[0] is not None else "cluster"
                    right_level = "macro" if right[0] is not None else "cluster"
                    left_scope = left[0] if left[0] is not None else left[1]
                    right_scope = right[0] if right[0] is not None else right[1]
                elif left[1] != right[1]:
                    left_level = right_level = "cluster"
                    left_scope, right_scope = left[1], right[1]
                else:
                    left_level = right_level = "owner"
                    left_scope, right_scope = left[2], right[2]
                relation_scopes[declaration.sha256] = (
                    add_scope(left_level, left_scope),
                    add_scope(right_level, right_scope),
                )
        else:
            expected_by_level["owner"] = {
                f"owner:{owner}": combined_artwork((owner,))
                for owner in sorted(set(owners.values()))
            }
            for declaration in declarations:
                relation_scopes[declaration.sha256] = tuple(
                    f"owner:{owners[member]}"
                    for member in (declaration.source, declaration.target)
                )

        claims = {
            key: owner_claims
            for expected in expected_by_level.values()
            for key, owner_claims in witness.claim_polygon_multiset(
                expected, eps=5
            ).items()
        }
        pitch_nm = _installed_adjacency_pitch_nm(tech)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        digest = getattr(getattr(exact_artifact, "scope", None), "exact_gds_sha256", None)
        return blocked(
            f"exact adjacent artifact cannot be replayed: {type(exc).__name__}: {exc}",
            digest,
        )

    obligations = []
    for declaration in declarations:
        ratio = declaration.max_separation_multiplier
        maximum = (pitch_nm * ratio.numerator + ratio.denominator - 1) // ratio.denominator
        try:
            left_scope, right_scope = relation_scopes[declaration.sha256]
            if left_scope == right_scope:
                raise ValueError("adjacent endpoints share one indivisible emitter owner")
            measured = _edge_gap_nm(
                _union_bbox(claims[left_scope]),
                _union_bbox(claims[right_scope]),
            )
            passed = measured is not None and 0 <= measured <= maximum
            obligations.append(GdsAdjacentObligation(
                source=declaration.source,
                target=declaration.target,
                declaration_sha256=declaration.sha256,
                gds_sha256=witness.gds_sha256,
                maximum_separation_nm=maximum,
                measured_separation_nm=measured,
                status="pass" if passed else "fail",
                measured_on_gds=True,
                reason=(
                    f"exact GDS edge gap {measured}nm is within {maximum}nm"
                    if passed
                    else "exact GDS endpoints lack a shared edge projection or exceed "
                    f"the {maximum}nm maximum"
                ),
            ))
        except Exception as exc:  # noqa: BLE001 - one obligation fails closed
            obligations.append(GdsAdjacentObligation(
                source=declaration.source,
                target=declaration.target,
                declaration_sha256=declaration.sha256,
                gds_sha256=witness.gds_sha256,
                maximum_separation_nm=maximum,
                measured_separation_nm=None,
                status="unmeasurable",
                measured_on_gds=False,
                reason=f"adjacent relation cannot be measured: {type(exc).__name__}: {exc}",
            ))
    return tuple(obligations)


__all__ = ["GdsAdjacentObligation", "verify_adjacencies_on_gds"]
