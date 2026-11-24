"""One read-only readiness entry for repository and external case packages."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from clay.flow.circuit_definition import DesignInputError, artifact_path, file_sha256, prepare_design
from clay.flow.readiness_result import IntentReadiness, ReadinessCheck
from clay.flow.reference_readiness import check_reference, pdk_snapshot, require_reference_identity_unchanged

REQUEST_STAGES = ("source", "placement", "comparison", "preroute")


class IntentReadinessRejected(ValueError):
    def __init__(self, result: IntentReadiness):
        self.result = result
        codes = ", ".join(row.code for row in result.checks if row.status != "pass")
        super().__init__(f"intent readiness blocked: {codes}")


def require_intent_readiness(case_dir, *, project_root=None, until="comparison") -> IntentReadiness:
    result = check_intent_readiness(case_dir, project_root=project_root, until=until)
    if not result.ready:
        raise IntentReadinessRejected(result)
    return result


def _locations(prepared, declarations):
    return tuple(dict(getattr(d, "source_span", None) or {}, artifact=prepared.definition.source_artifact)
                 for d in declarations)


def _placement_capabilities(prepared, parsed, cell, intent):
    checks = []
    spec = prepared.flow_spec
    # Match LayoutStep: a device-bearing case with no custom layout module
    # uses the generic producer, as does an explicit generic selection.
    if (spec.layout_module or '').strip().lower() not in ('', 'generic'):
        checks.append(ReadinessCheck("placement.provider", "unsupported", "LAYOUT_PROVIDER_UNSUPPORTED",
                                     "public placement currently accepts the installed generic provider only"))
        return checks
    try:
        from clay.layout.flow_placement import materialize_declared_compact_override
        from clay.layout.generic_placement import place_generic
        available = callable(place_generic) and callable(materialize_declared_compact_override)
    except (ImportError, AttributeError):
        available = False
    if available:
        checks.append(ReadinessCheck("placement.provider", "pass", "PLACEMENT_PRODUCERS_PRESENT",
                                     "generic placement and final compaction callables are installed; execution is not certified"))
    else:
        checks.append(ReadinessCheck("placement.provider", "unsupported", "PLACEMENT_PRODUCER_UNAVAILABLE",
                                     "generic placement and final compaction require callable installed producers"))
    from clay.pdk import get_pdk
    try:
        from clay.layout.annotate_to_solver import kind_polarity_oxide
    except ImportError as exc:
        checks.append(ReadinessCheck("placement.devices", "unsupported", "PLACEMENT_PRODUCER_UNAVAILABLE", str(exc)))
        return checks
    try:
        adapter = get_pdk(prepared.definition.generated_pdk_id)
        for device in cell.devices:
            kind = device.device_class
            if kind == "native":
                # The shared PDK-owned support check validates these against
                # the same provider consumed by canonical LayoutStep.
                continue
            supported = kind_polarity_oxide(kind)[0] is not None or kind in ("res", "cap")
            if kind in ("pnp", "npn"):
                binding = (getattr(adapter, "bjt_bindings", {}) or {}).get(kind)
                supported = binding is not None and binding.geometry_provider == "fixed_gds"
                if supported:
                    # Match the existing materializer's installed asset root;
                    # this is byte availability, not GDS/terminal certification.
                    try:
                        path = artifact_path(Path(__file__).resolve().parents[2], binding.geometry_path)
                        supported = file_sha256(path) == binding.geometry_sha256
                    except (DesignInputError, OSError):
                        supported = False
            if not supported:
                checks.append(ReadinessCheck("placement.devices", "unsupported", "DEVICE_MATERIALIZER_UNAVAILABLE",
                                             f"no available physical materializer for device kind {kind!r}", _locations(prepared, (device,))))
    except KeyError:
        pass  # The PDK capability check below supplies the named rejection.
    from clay.flow.placement_capabilities import plan_placement_checks
    checks.extend(plan_placement_checks(prepared, parsed, cell, intent))
    for module in ("ortools", "klayout"):
        if importlib.util.find_spec(module) is None:
            checks.append(ReadinessCheck("placement.tools", "unsupported", "TOOL_UNAVAILABLE", f"required Python package unavailable: {module}"))
    try:
        snapshot = pdk_snapshot(prepared.definition.generated_pdk_id)
        checks.append(ReadinessCheck("placement.pdk", "pass", "PDK_AUTHORITY_VERIFIED", "installed generated PDK authority captured",
                                     detail={"snapshot_sha256": snapshot.snapshot_sha256,
                                             "installed_authority_sha256": snapshot.installed_authority_sha256}))
    except DesignInputError as exc:
        checks.append(ReadinessCheck("placement.pdk", "unsupported", exc.code, str(exc)))
    return checks


def check_intent_readiness(case_dir, *, project_root=None, until="comparison") -> IntentReadiness:
    """Report supported capabilities without placing, routing or issuing PASS.

Source scope verifies declared inputs and source-level bindings. Placement and
comparison scopes additionally require current physical/reference capabilities.
No case-supplied callable or PASS payload is accepted by this API.
"""
    if until not in REQUEST_STAGES:
        return IntentReadiness(str(until), (ReadinessCheck("request", "fail", "UNKNOWN_REQUEST_STAGE", f"until must be one of {REQUEST_STAGES}"),))
    try:
        prepared = prepare_design(case_dir, project_root=project_root)
    except Exception as exc:
        return IntentReadiness(until, (ReadinessCheck("input", "fail", getattr(exc, "code", "CASE_INPUT_INVALID"), str(exc),
                                                     (exc.location,) if isinstance(exc, DesignInputError) and exc.location else ()),))
    from clay.parser import parse_clay_source
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.intent_net_binding import check_intent_net_binding
    from clay.layout.route.exact_artifact_evidence import _source_member_inventory_sha256
    parsed = parse_clay_source(prepared.source_bytes.decode("utf-8-sig"))
    cell = next(c for c in parsed.cell_defs if c.local_name == prepared.definition.top_cell)
    checks = [ReadinessCheck("source.identity", "pass", "SOURCE_IDENTITY_VERIFIED", "selected source, variant and compiled flow are bound")]
    identity = {"source_file": {"domain": "file_bytes", "sha256": prepared.source_sha256},
                "inputs": [{"artifact": b.artifact, "domain": "file_bytes", "sha256": b.sha256} for b in prepared.inputs]}
    intent = None
    from clay.flow.physical_support import check_device_support
    support = check_device_support(
        cell, prepared.definition.generated_pdk_id, physical=until != "source",
        source_artifact=prepared.definition.source_artifact,
    )
    checks.extend(support.checks)
    if (not cell.devices or parsed.imports or cell.instances or cell.topology
            or cell.combinator_block is not None or cell.bias_tree):
        checks.append(ReadinessCheck("source.projection", "unsupported", "SOURCE_PROJECTION_UNSUPPORTED",
                                     "public preflight requires non-empty explicit primitive topology without style, composition or imports"))
    try:
        identity["source_inventory"] = {"domain": "clay_source_member_inventory/v1", "sha256": _source_member_inventory_sha256(cell)}
        binding = check_intent_net_binding(cell)
        if binding.status != "pass":
            checks.append(ReadinessCheck("source.intent", binding.status, "INTENT_BINDING_FAILED", binding.describe(), _locations(prepared, cell.layout_intents)))
        else:
            intent = layout_intent_from_cell_def(cell)
            checks.append(ReadinessCheck("source.intent", "pass", "INTENT_BINDING_VERIFIED", "source declarations bind to the explicit topology"))
    except Exception as exc:
        checks.append(ReadinessCheck("source.intent", "fail", "INTENT_LOWERING_FAILED", str(exc), _locations(prepared, cell.layout_intents)))
    reference_checks, reference_identity = check_reference(prepared, cell, replay=until != "source")
    checks.extend(reference_checks)
    identity.update(reference_identity)
    if until != "source":
        try:
            checks.extend(_placement_capabilities(prepared, parsed, cell, intent))
        except Exception as exc:
            checks.append(ReadinessCheck("placement.capabilities", "fail", "CAPABILITY_CHECK_FAILED", str(exc)))
        if prepared.definition.mode == "reference_reconstruction":
            checks.append(ReadinessCheck("placement.reference_generation", "unsupported", "REFERENCE_GENERATION_PROVIDER_UNAVAILABLE",
                                         "reference-conditioned generation has not been integrated into the public placement provider"))
    if until == "comparison":
        native_arrays = tuple(row for row in getattr(intent, "arrays", ())
                              if getattr(row, "style", "") == "native_diffusion")
        if native_arrays:
            try:
                from clay.layout.final_layout_intent import (
                    _verify_native_diffusion_finger_orders_on_gds,
                )
            except Exception as exc:
                checks.append(ReadinessCheck(
                    "comparison.native_diffusion", "fail", "NATIVE_DIFFUSION_FINAL_ORDER_UNAVAILABLE",
                    f"native diffusion final GDS channel-order verifier failed to load: {exc}",
                    _locations(prepared, cell.layout_intents),
                    detail={"groups": [row.group for row in native_arrays]},
                ))
            else:
                if not callable(_verify_native_diffusion_finger_orders_on_gds):
                    checks.append(ReadinessCheck(
                        "comparison.native_diffusion", "unsupported", "NATIVE_DIFFUSION_FINAL_ORDER_UNAVAILABLE",
                        "native diffusion final GDS channel-order verifier is not callable",
                        _locations(prepared, cell.layout_intents),
                        detail={"groups": [row.group for row in native_arrays]},
                    ))
                else:
                    checks.append(ReadinessCheck(
                        "comparison.native_diffusion", "pass", "NATIVE_DIFFUSION_FINAL_ORDER_VERIFIED",
                        "native diffusion final GDS channel-order verifier is integrated",
                        _locations(prepared, cell.layout_intents),
                        detail={"groups": [row.group for row in native_arrays]},
                    ))
        checks.append(ReadinessCheck("comparison.provider", "unsupported", "COMPARISON_PRODUCER_UNAVAILABLE",
                                     "the public same-candidate strict comparison producer has not been integrated"))
    if until == "preroute":
        checks.extend(_preroute_gate_checks(prepared, parsed, cell, intent, checks))
    try:
        prepared.require_unchanged()
        require_reference_identity_unchanged(prepared, identity)
    except DesignInputError as exc:
        checks.append(ReadinessCheck("inputs.final", "fail", exc.code, str(exc), (exc.location,) if exc.location else ()))
    return IntentReadiness(until, tuple(checks), prepared.definition.to_dict(), identity)


def _preroute_gate_checks(prepared, parsed, cell, intent, prior_checks) -> list[ReadinessCheck]:
    """Run bounded placement + <5s T1/T2 pre-route gate for ``until='preroute'``."""
    import time

    from clay.layout.generic_placement import parse_cir_mos_records, place_generic
    from clay.layout.policy import resolve_policy
    from clay.layout.preroute_gate import run_preroute_gate

    t0 = time.monotonic()
    out: list[ReadinessCheck] = []
    if cell is None:
        return [
            ReadinessCheck(
                "preroute.gate",
                "fail",
                "PREROUTE_NO_TOP_CELL",
                "cannot run pre-route gate without a resolved top cell",
            )
        ]

    flow_spec = getattr(prepared, "flow_spec", None)
    pdk_id = str(
        getattr(flow_spec, "pdk", None)
        or getattr(prepared.definition, "generated_pdk_id", None)
        or getattr(cell, "target_pdk", None)
        or getattr(parsed, "pdk_id", None)
        or "gf180mcu"
    )
    flow_decl = getattr(cell, "flow_decl", None)
    policy_ref = getattr(flow_decl, "policy_ref", None)
    policy = (getattr(parsed, "policies", None) or {}).get(policy_ref)
    effective_policy = resolve_policy(policy, layout_intent=intent)

    source_cir = getattr(flow_spec, "source_cir", None)
    device_records = None
    if source_cir is not None and not getattr(cell, "devices", ()):
        try:
            device_records = parse_cir_mos_records(source_cir)
        except Exception:
            device_records = None

    prior_failed = any(
        getattr(row, "status", "pass") != "pass"
        and getattr(row, "name", "") != "placement.pdk"
        for row in prior_checks
    )
    placement = None
    if not prior_failed:
        try:
            placement = place_generic(
                cell,
                intent,
                pdk_id,
                source_cir=source_cir,
                time_limit_s=2.0,
            )
        except Exception as exc:
            out.append(
                ReadinessCheck(
                    "preroute.placement",
                    "fail",
                    "PREROUTE_PLACEMENT_FAILED",
                    f"fast pre-route placement failed: {exc}",
                    detail={"exception_type": type(exc).__name__},
                )
            )
            placement = None

    remaining_budget = max(1.5, 4.8 - (time.monotonic() - t0))
    gate = run_preroute_gate(
        placement,
        cell,
        intent,
        pdk_id,
        device_records=device_records,
        effective_policy=effective_policy,
        source_artifact=prepared.definition.source_artifact,
        budget_sec=remaining_budget,
    )

    t1_blockers = [b.to_dict() for b in gate.blockers if b.tier == "T1"]
    t2_blockers = [b.to_dict() for b in gate.blockers if b.tier == "T2"]
    out.append(
        ReadinessCheck(
            "preroute.t1_source_lvs",
            "pass" if gate.t1_passed else "fail",
            "PREROUTE_T1_VERIFIED" if gate.t1_passed else "PREROUTE_T1_BLOCKED",
            (
                "T1 source/PDK/LVS pre-route gate passed"
                if gate.t1_passed
                else "; ".join(b["reason"] for b in t1_blockers)
            ),
            detail={"blockers": t1_blockers, "metrics": gate.metrics},
        )
    )
    if placement is not None:
        out.append(
            ReadinessCheck(
                "preroute.t2_routability",
                "pass" if gate.t2_passed else "fail",
                "PREROUTE_T2_VERIFIED" if gate.t2_passed else "PREROUTE_T2_BLOCKED",
                (
                    f"T2 pin-escape/via/corridor gate passed ({gate.elapsed_ms:.1f}ms)"
                    if gate.t2_passed
                    else gate.summary
                ),
                detail={
                    "recommended_expand_nm": gate.recommended_expand_nm,
                    "blockers": t2_blockers,
                    "elapsed_ms": round(gate.elapsed_ms, 3),
                    "metrics": gate.metrics,
                },
            )
        )
    return out
