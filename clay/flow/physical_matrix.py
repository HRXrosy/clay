"""Circuit-neutral physical multi-PDK evidence and atomic ledger publication."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_STEPS = ("compile", "layout", "drc", "lvs", "pex", "intent", "sign")


def render_cross_ledger_summary(kpis: Sequence[object]) -> str:
    """Render the physical/electrical projection without changing row scope."""

    if not tuple(kpis):
        return "physical signed; electrical unmeasurable"
    return "physical signed; electrical measurements available"


def bind_physical_search_provenance(
    row: dict[str, Any],
    *,
    circuit_id: str,
    topology_id: str,
    normalized_connectivity_sha256: str,
    candidate_origin: str,
    search: Any,
    full_exact_count: int,
) -> dict[str, Any]:
    """Attach the bounded search facts that selected one physical candidate."""

    if type(circuit_id) is not str or not circuit_id.strip():
        raise ValueError("physical candidate circuit_id is empty")
    reported_circuit = row.get("circuit_id")
    if reported_circuit is not None and reported_circuit != circuit_id:
        raise ValueError("physical candidate circuit_id differs from runner authority")
    if not topology_id.strip():
        raise ValueError("physical candidate topology_id is empty")
    connectivity = normalized_connectivity_sha256.casefold()
    if len(connectivity) != 64 or any(ch not in "0123456789abcdef" for ch in connectivity):
        raise ValueError("physical candidate connectivity digest is invalid")
    if candidate_origin not in {"authoring", "derived"}:
        raise ValueError("physical candidate origin is invalid")
    evaluated = int(search.cheap_evaluated_count)
    domain_count = int(search.domain_total_combinations)
    budget = int(search.domain_budget) + int(search.refinement_budget)
    exact_count = int(full_exact_count)
    if min(evaluated, domain_count, budget, exact_count) < 0:
        raise ValueError("physical candidate search counts are negative")
    if evaluated > domain_count or evaluated > budget or exact_count > evaluated:
        raise ValueError("physical candidate search counts exceed their bounds")
    exhausted = search.domain_exhausted is True
    row.update(
        {
            "circuit_id": circuit_id,
            "topology_id": topology_id,
            "normalized_connectivity_sha256": connectivity,
            "candidate_origin": candidate_origin,
            "search_status": "domain_exhausted" if exhausted else "best_observed",
            "search_receipt": {
                "evaluated_count": evaluated,
                "domain_count": domain_count,
                "full_exact_count": exact_count,
                "budget": budget,
                "stop_reason": (
                    "domain_exhausted" if exhausted else "first_physical_signoff"
                ),
                "lower_bound": None,
                "optimality_gap": None,
            },
        }
    )
    return row


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_artifact(step: Any, key: str, *, pdk: str) -> Path:
    path = Path(str(step.artifacts.get(key, ""))).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{pdk} {step.name} missing {key}: {path}")
    return path


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return path


def _load_json_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def gds_geometry_metrics(
    path: str | Path, *, top_cell: str | None = None
) -> dict[str, Any]:
    """Measure flattened footprint facts used for reference-quality comparison."""

    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(path))
    cell = layout.cell(top_cell) if top_cell else None
    if top_cell and cell is None:
        raise ValueError(f"GDS is missing required top cell {top_cell!r}: {path}")
    if cell is None:
        tops = layout.top_cells()
        cell = tops[0] if tops else None
    if cell is None:
        raise ValueError(f"GDS has no top cell: {path}")
    bbox = cell.bbox()
    width_um = bbox.width() * layout.dbu
    height_um = bbox.height() * layout.dbu
    if width_um <= 0.0 or height_um <= 0.0:
        raise ValueError(f"GDS top cell has empty geometry: {path}")
    return {
        "width_um": width_um,
        "height_um": height_um,
        "bbox_area_um2": width_um * height_um,
        "aspect_ratio": max(width_um, height_um) / min(width_um, height_um),
    }


def summarize_physical_pdk_flow(
    spec: Any,
    flow_result: Any,
    *,
    reference_gds: str | Path | None = None,
    reference_top_cell: str | None = None,
) -> dict[str, Any]:
    """Normalize one canonical physical flow without inventing electrical truth."""

    from clay.flow.pdk_matrix import _full_extraction_body_evidence

    pdk = str(spec.pdk).casefold()
    by_step = dict(flow_result.by_step)
    statuses = {
        name: str(by_step[name].status) if name in by_step else "missing"
        for name in PHYSICAL_STEPS
    }
    failed = [name for name, status in statuses.items() if status != "pass"]
    if str(flow_result.verdict) != "pass" or failed:
        raise ValueError(f"{pdk} physical flow is not signed; failed steps={failed}")

    layout_step = by_step["layout"]
    drc_step = by_step["drc"]
    lvs_step = by_step["lvs"]
    pex_step = by_step["pex"]
    intent_step = by_step["intent"]
    sign_step = by_step["sign"]
    gds = _required_artifact(layout_step, "layout_gds", pdk=pdk)
    gds_sha = _sha256(gds)
    if str(layout_step.data.get("gds_sha256", "")) != gds_sha:
        raise ValueError(f"{pdk} layout GDS digest mismatch")
    exact_receipt_path = _required_artifact(
        layout_step, "exact_artifact_receipt", pdk=pdk
    )
    exact_receipt = _load_json_artifact(
        exact_receipt_path, label=f"{pdk} exact-artifact receipt"
    )
    exact_scope = dict(exact_receipt.get("scope") or {})
    exact_counts = (
        exact_scope.get("heal_shape_count"),
        exact_scope.get("upper_fill_shape_count"),
        exact_scope.get("post_assembly_fill_shape_count"),
        exact_scope.get("snapped_shape_count"),
    )
    if (
        exact_receipt.get("status") != "pass"
        or exact_receipt.get("exact_gds_sha256") != gds_sha
        or exact_scope.get("exhaustive") is not True
        or exact_scope.get("allocated_union_matches_router") is not True
        or exact_counts != (0, 0, 0, 0)
    ):
        raise ValueError(f"{pdk} exact-artifact gate failed")
    graph_data = layout_step.data.get("physical_macro_graph")
    graph_sha = str(layout_step.data.get("physical_macro_graph_sha256", ""))
    if not isinstance(graph_data, dict) or len(graph_sha) != 64:
        raise ValueError(f"{pdk} physical macro graph evidence is missing")
    if exact_receipt.get("physical_macro_graph_sha256") != graph_sha:
        raise ValueError(f"{pdk} exact artifact has a stale macro graph binding")
    final_intent = layout_step.data.get("final_layout_intent")
    if not isinstance(final_intent, dict):
        raise ValueError(f"{pdk} final layout intent evidence is missing")
    route_containment = dict(final_intent.get("route_containment") or {})
    if (
        final_intent.get("status") != "pass"
        or final_intent.get("gds_sha256") != gds_sha
        or route_containment.get("status") != "pass"
    ):
        raise ValueError(f"{pdk} route-containment gate failed")
    evidence_root = Path(flow_result.work_dir).resolve()
    graph_path = _write_json_artifact(
        evidence_root / "physical_macro_graph.json", graph_data
    )
    containment_path = _write_json_artifact(
        evidence_root / "final_layout_intent.json", final_intent
    )
    drc_report = _required_artifact(drc_step, "drc_report", pdk=pdk)
    violations = int(drc_step.data.get("n_violations", -1))
    if violations != 0:
        raise ValueError(f"{pdk} DRC violations={violations}")
    lvs_report = _required_artifact(lvs_step, "lvs_report", pdk=pdk)
    lvs_extracted = _required_artifact(lvs_step, "lvs_extracted", pdk=pdk)
    if lvs_step.data.get("clean") is not True:
        raise ValueError(f"{pdk} full-extraction LVS mismatch")
    design = Path(spec.case_dir).resolve() / "source" / "design.clay"
    body = _full_extraction_body_evidence(spec, lvs_report, design_clay=design)
    if body.get("status") != "pass" or body.get("measured") is not True:
        raise ValueError(f"{pdk} full-extraction body gate failed")
    pex = _required_artifact(pex_step, "pex_spice", pdk=pdk)
    from clay.pdk import get_pdk

    decks = get_pdk(pdk).klayout_decks()
    drc_deck = Path(str(decks.get("drc", ""))).resolve()
    lvs_deck = Path(str(decks.get("lvs", ""))).resolve()
    if not drc_deck.is_file() or not lvs_deck.is_file():
        raise ValueError(f"{pdk} signoff deck evidence is missing")
    generated_metrics = gds_geometry_metrics(gds, top_cell=spec.gds_cell_name)
    run_summary = Path(flow_result.work_dir).resolve() / "run_summary.json"
    if not run_summary.is_file():
        raise ValueError(f"{pdk} missing run summary: {run_summary}")
    return {
        "pdk": pdk,
        "topology_id": str(getattr(spec, "topology_id", "")),
        # Task-specific reference intake may replace this graph digest with its
        # normalized port/net/device connectivity digest before publication.
        "normalized_connectivity_sha256": graph_sha,
        "candidate_origin": "authoring",
        "search_status": "not_run",
        "search_receipt": {
            "evaluated_count": 0,
            "domain_count": 0,
            "full_exact_count": 0,
            "budget": 0,
            "stop_reason": "not_run",
            "lower_bound": None,
            "optimality_gap": None,
        },
        "verdict": "pass",
        "required_steps": statuses,
        "intent_source": str(design),
        "intent_source_sha256": _sha256(design),
        "candidate": {"path": str(design), "sha256": _sha256(design)},
        "physical_macro_graph": {
            "path": str(graph_path),
            "sha256": _sha256(graph_path),
            "logical_sha256": graph_sha,
        },
        "gds": {"path": str(gds), "sha256": gds_sha, "metrics": generated_metrics},
        "decks": {
            "drc": {
                "path": str(drc_deck),
                "sha256": _sha256(drc_deck),
                "logical_origin": f"{pdk}.klayout.drc",
                "version": "content-addressed-v1",
                "coverage": "partial",
                "foundry_signoff": False,
            },
            "lvs": {
                "path": str(lvs_deck),
                "sha256": _sha256(lvs_deck),
                "logical_origin": f"{pdk}.klayout.lvs",
                "version": "content-addressed-v1",
                "coverage": "partial",
                "foundry_signoff": False,
            },
        },
        "drc": {
            "status": "pass",
            "violations": 0,
            "report": str(drc_report),
            "report_sha256": _sha256(drc_report),
        },
        "lvs": {
            "status": "pass",
            "matched": True,
            "full_extraction": True,
            "report": str(lvs_report),
            "report_sha256": _sha256(lvs_report),
            "extracted": str(lvs_extracted),
            "extracted_sha256": _sha256(lvs_extracted),
        },
        "body": body,
        "pex": {"status": "pass", "netlist": str(pex), "sha256": _sha256(pex)},
        "exact_artifact": {
            "status": "pass",
            "path": str(exact_receipt_path),
            "sha256": _sha256(exact_receipt_path),
            "gds_sha256": gds_sha,
            "route_manifest_sha256": exact_receipt.get("route_manifest_sha256"),
        },
        "containment": {
            "status": "pass",
            "path": str(containment_path),
            "sha256": _sha256(containment_path),
        },
        "intent": {
            "status": "pass",
            **dict(intent_step.data),
            "signed": bool(intent_step.data.get("signed")),
        },
        "sign": {"status": "pass", "verdict": sign_step.data.get("verdict")},
        "run_summary": str(run_summary),
        "run_summary_sha256": _sha256(run_summary),
    }


def _commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _source_state_sha256(intent_locator: Mapping[str, Any]) -> str:
    """Hash the canonical, content-addressed source binding carried by the ledger."""

    state = {
        "schema": "clay.physical-source-state/v1",
        "intent": dict(intent_locator),
    }
    canonical = json.dumps(
        state,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def publish_physical_pdk_ledger(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    intent_path: str | Path,
    pdk_ids: Sequence[str],
    portfolio_circuit_id: str | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Package and atomically publish legality-only v2 physical rows."""

    from clay.core_contract import VerificationStatus
    from clay.flow.portfolio_signoff import (
        BaselineComparisonReceipt,
        DeckScopeReceipt,
        PHYSICAL_CLAIM_SCOPE,
        PortfolioSignoffRow,
        SearchReceipt,
        baseline_comparison_receipt_from_mapping,
        validate_portfolio_row_artifacts,
    )
    from clay.flow.signoff_artifacts import package_artifact

    expected = tuple(str(pdk).casefold() for pdk in pdk_ids)
    normalized = [dict(row) for row in rows]
    if tuple(str(row.get("pdk", "")).casefold() for row in normalized) != expected:
        raise ValueError(f"physical ledger rows must be ordered as {expected}")
    for pdk, row in zip(expected, normalized):
        statuses = dict(row.get("required_steps") or {})
        failed_steps = [name for name in PHYSICAL_STEPS if statuses.get(name) != "pass"]
        gates = {
            "verdict": row.get("verdict") == "pass",
            "drc": row.get("drc", {}).get("violations") == 0,
            "lvs": (
                row.get("lvs", {}).get("matched") is True
                and row.get("lvs", {}).get("full_extraction") is True
            ),
            "body": (
                row.get("body", {}).get("status") == "pass"
                and row.get("body", {}).get("measured") is True
            ),
            "pex": row.get("pex", {}).get("status") == "pass",
            "exact_artifact": row.get("exact_artifact", {}).get("status") == "pass",
            "containment": row.get("containment", {}).get("status") == "pass",
            "intent": row.get("intent", {}).get("signed") is True,
        }
        failed_gates = [name for name, passed in gates.items() if not passed]
        if failed_steps or failed_gates:
            raise ValueError(
                f"{pdk} physical row is unsigned; steps={failed_steps}, gates={failed_gates}"
            )
        gds = Path(str(row.get("gds", {}).get("path", "")))
        pex = Path(str(row.get("pex", {}).get("netlist", "")))
        if not gds.is_file() or _sha256(gds) != row.get("gds", {}).get("sha256"):
            raise ValueError(f"{pdk} GDS artifact is missing or hash-mismatched")
        if not pex.is_file():
            raise ValueError(f"{pdk} PEX artifact is missing")
    intent = Path(intent_path).resolve()
    root = Path(repo_root)
    if not root.is_absolute():
        raise ValueError("physical ledger repo_root must be an explicit absolute path")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    commit = _commit(root)
    source_locator = package_artifact(root, intent, logical_id="matrix_intent")
    physical_rows: list[PortfolioSignoffRow] = []

    def declared_artifact(
        payload: Mapping[str, Any],
        *,
        logical_id: str,
        path_key: str,
        digest_key: str,
    ):
        source = Path(str(payload.get(path_key, ""))).resolve()
        declared = str(payload.get(digest_key, "")).casefold()
        if not source.is_file() or _sha256(source) != declared:
            raise ValueError(
                f"{logical_id} artifact is missing or hash-mismatched: {source}"
            )
        return package_artifact(root, source, logical_id=logical_id)

    for pdk, raw in zip(expected, normalized):
        candidate = dict(raw.get("candidate") or {})
        graph = dict(raw.get("physical_macro_graph") or {})
        decks = dict(raw.get("decks") or {})
        drc_deck_data = dict(decks.get("drc") or {})
        lvs_deck_data = dict(decks.get("lvs") or {})
        gds = dict(raw.get("gds") or {})
        drc = dict(raw.get("drc") or {})
        lvs = dict(raw.get("lvs") or {})
        body = dict(raw.get("body") or {})
        pex = dict(raw.get("pex") or {})
        exact = dict(raw.get("exact_artifact") or {})
        containment = dict(raw.get("containment") or {})
        artifacts = [
            package_artifact(root, intent, logical_id="source"),
            declared_artifact(
                candidate,
                logical_id="candidate",
                path_key="path",
                digest_key="sha256",
            ),
            declared_artifact(
                graph,
                logical_id="physical_macro_graph",
                path_key="path",
                digest_key="sha256",
            ),
            declared_artifact(
                gds,
                logical_id="gds",
                path_key="path",
                digest_key="sha256",
            ),
            declared_artifact(
                drc_deck_data,
                logical_id="drc_deck",
                path_key="path",
                digest_key="sha256",
            ),
            declared_artifact(
                lvs_deck_data,
                logical_id="lvs_deck",
                path_key="path",
                digest_key="sha256",
            ),
            declared_artifact(
                drc,
                logical_id="drc_report",
                path_key="report",
                digest_key="report_sha256",
            ),
            declared_artifact(
                lvs,
                logical_id="lvs_report",
                path_key="report",
                digest_key="report_sha256",
            ),
            declared_artifact(
                pex,
                logical_id="pex",
                path_key="netlist",
                digest_key="sha256",
            ),
            declared_artifact(
                exact,
                logical_id="exact_artifact",
                path_key="path",
                digest_key="sha256",
            ),
            declared_artifact(
                containment,
                logical_id="containment",
                path_key="path",
                digest_key="sha256",
            ),
        ]
        from clay.flow.reference_convergence import (
            build_intent_measurement_receipt,
        )

        intent_measurement = build_intent_measurement_receipt(
            circuit_id=str(
                portfolio_circuit_id or raw.get("circuit_id") or "physical_matrix"
            ),
            topology_id=str(raw.get("topology_id", "")),
            pdk_id=pdk,
            exact_gds_sha256=str(gds.get("sha256", "")),
            intent_source=intent.read_text(encoding="utf-8"),
            intent_result=dict(raw.get("intent") or {}),
        )
        with tempfile.TemporaryDirectory(
            prefix=f"clay-{pdk}-intent-measurement-"
        ) as temporary_directory:
            receipt_path = Path(temporary_directory) / "intent_measurement_receipt.json"
            receipt_path.write_text(
                json.dumps(
                    intent_measurement.canonical_data(), indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            artifacts.append(
                package_artifact(
                    root,
                    receipt_path,
                    logical_id="intent_measurement_receipt",
                )
            )
        for optional_id, optional_payload, optional_path, optional_digest in (
            ("lvs_extracted", lvs, "extracted", "extracted_sha256"),
            (
                "run_summary",
                raw,
                "run_summary",
                "run_summary_sha256",
            ),
        ):
            if optional_payload.get(optional_path):
                artifacts.append(
                    declared_artifact(
                        optional_payload,
                        logical_id=optional_id,
                        path_key=optional_path,
                        digest_key=optional_digest,
                    )
                )
        try:
            search_receipt = SearchReceipt(**dict(raw.get("search_receipt") or {}))
        except TypeError as exc:
            raise ValueError(f"{pdk} search receipt is incomplete: {exc}") from exc
        raw_baseline_receipt = raw.get("baseline_comparison_receipt")
        if raw_baseline_receipt is None:
            baseline_receipt = None
        elif isinstance(raw_baseline_receipt, BaselineComparisonReceipt):
            baseline_receipt = raw_baseline_receipt
        elif isinstance(raw_baseline_receipt, Mapping):
            baseline_receipt = baseline_comparison_receipt_from_mapping(
                raw_baseline_receipt
            )
        else:
            raise ValueError(f"{pdk} baseline comparison receipt is invalid")
        drc_scope = DeckScopeReceipt(
            logical_origin=str(drc_deck_data.get("logical_origin", "")),
            version=str(drc_deck_data.get("version", "")),
            sha256=artifacts[4].sha256,
            coverage=str(drc_deck_data.get("coverage", "")),  # type: ignore[arg-type]
            foundry_signoff=drc_deck_data.get("foundry_signoff") is True,
        )
        lvs_scope = DeckScopeReceipt(
            logical_origin=str(lvs_deck_data.get("logical_origin", "")),
            version=str(lvs_deck_data.get("version", "")),
            sha256=artifacts[5].sha256,
            coverage=str(lvs_deck_data.get("coverage", "")),  # type: ignore[arg-type]
            foundry_signoff=lvs_deck_data.get("foundry_signoff") is True,
        )
        physical_row = PortfolioSignoffRow(
            circuit_id=str(
                portfolio_circuit_id or raw.get("circuit_id") or "physical_matrix"
            ),
            pdk_id=pdk,
            commit=commit,
            claim_scope=PHYSICAL_CLAIM_SCOPE,
            physical_status=VerificationStatus.PASS,
            physical_signed_off=True,
            topology_id=str(raw.get("topology_id", "")),
            normalized_connectivity_sha256=str(
                raw.get("normalized_connectivity_sha256", "")
            ).casefold(),
            candidate_origin=str(raw.get("candidate_origin", "")),  # type: ignore[arg-type]
            search_status=str(raw.get("search_status", "")),  # type: ignore[arg-type]
            search_receipt=search_receipt,
            drc_deck=drc_scope,
            lvs_deck=lvs_scope,
            artifacts=tuple(artifacts),
            drc_violations=int(drc.get("violations", -1)),
            lvs_matched=lvs.get("matched") is True,
            lvs_full_extraction=lvs.get("full_extraction") is True,
            body_measured=body.get("measured") is True,
            pex_status=str(pex.get("status", "")),
            exact_artifact_status=str(exact.get("status", "")),
            containment_status=str(containment.get("status", "")),
            promoted=True,
            baseline_comparison_receipt=baseline_receipt,
        )
        validate_portfolio_row_artifacts(
            physical_row,
            current_commit=commit,
            repo_root=root,
        )
        physical_rows.append(physical_row)

    canonical_rows = [row.canonical_data() for row in physical_rows]
    payload = {
        "schema": "clay.physical-pdk-matrix/v2",
        "claim_scope": "physical_artifact_signoff",
        "physical_status": "pass",
        "physical_signed_off": True,
        "commit": commit,
        "source_state_sha256": _source_state_sha256(
            source_locator.canonical_data()
        ),
        "intent": source_locator.canonical_data(),
        "summary": render_cross_ledger_summary(()),
        "signoffs": canonical_rows,
    }
    if portfolio_circuit_id is not None:
        payload["portfolio_rows"] = canonical_rows
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return payload


def _validate_strict_physical_ledger_content(
    payload: Mapping[str, Any],
    rows: Sequence[object],
    *,
    circuit_id: str,
    repo_root: str | Path,
    expected_source_sha256: str,
    expected_source_path: str,
    expected_topology_id: str,
    expected_normalized_connectivity_sha256: str,
    reference_manifest: object | None = None,
    three_reference_ledger: bool = False,
) -> None:
    """Bind typed top-level metadata to the physical-only canonical rows."""

    from clay.flow.portfolio_signoff import (
        ArtifactLocator,
        PortfolioSignoffError,
    )
    from clay.flow.signoff_artifacts import (
        SignoffArtifactError,
        verify_artifact_locator,
    )
    from clay.flow.reference_evidence import (
        FROZEN_TOPOLOGY_IDENTITIES,
        ReferenceManifest,
    )

    if type(three_reference_ledger) is not bool:
        raise PortfolioSignoffError(
            f"{circuit_id} three-reference authority flag is invalid"
        )
    if three_reference_ledger:
        frozen = FROZEN_TOPOLOGY_IDENTITIES.get(expected_topology_id)
        if frozen is None or frozen["three_reference_ledger"] is not True:
            raise PortfolioSignoffError(
                f"{circuit_id} topology is not an eligible three-reference identity"
            )
        if not isinstance(reference_manifest, ReferenceManifest):
            raise PortfolioSignoffError(
                f"{circuit_id} strict three-reference ledger requires a typed reference manifest"
            )
        if (
            reference_manifest.circuit_id != circuit_id
            or reference_manifest.topology_id != expected_topology_id
            or reference_manifest.normalized_connectivity_sha256
            != expected_normalized_connectivity_sha256
        ):
            raise PortfolioSignoffError(
                f"{circuit_id} typed reference manifest binding differs"
            )
    elif reference_manifest is not None and not isinstance(
        reference_manifest, ReferenceManifest
    ):
        raise PortfolioSignoffError(
            f"{circuit_id} reference manifest is not typed"
        )

    source_state = payload.get("source_state_sha256")
    if (
        type(source_state) is not str
        or len(source_state) != 64
        or any(character not in "0123456789abcdef" for character in source_state)
    ):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger source_state_sha256 is invalid"
        )

    expected_summary = render_cross_ledger_summary(())
    if type(payload.get("summary")) is not str or payload["summary"] != expected_summary:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger summary contradicts its "
            "physical-only, electrical-unmeasurable rows"
        )

    raw_intent = payload.get("intent")
    intent_fields = set(ArtifactLocator.__dataclass_fields__)
    if type(raw_intent) is not dict or set(raw_intent) != intent_fields:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger intent locator is non-canonical"
        )
    if (
        raw_intent.get("logical_id") != "matrix_intent"
        or type(raw_intent.get("path")) is not str
        or not raw_intent["path"]
        or type(raw_intent.get("sha256")) is not str
        or len(raw_intent["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in raw_intent["sha256"]
        )
        or raw_intent.get("artifact_type") != "file"
        or raw_intent.get("required") is not True
    ):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger intent locator is invalid"
        )
    intent_locator = ArtifactLocator(**raw_intent)
    try:
        verify_artifact_locator(repo_root, intent_locator)
    except SignoffArtifactError as exc:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger intent artifact is invalid: {exc}"
        ) from exc
    if source_state != _source_state_sha256(intent_locator.canonical_data()):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger source_state_sha256 differs from "
            "its verified intent binding"
        )
    if (
        intent_locator.sha256 != expected_source_sha256
        or intent_locator.path != expected_source_path
    ):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger source binding differs from its "
            "aggregate descriptor"
        )

    expected_source = intent_locator.canonical_data()
    expected_source["logical_id"] = "source"
    for row in rows:
        if row.topology_id != expected_topology_id:  # type: ignore[attr-defined]
            raise PortfolioSignoffError(
                f"{circuit_id} physical ledger topology binding differs from "
                "its aggregate descriptor"
            )
        if (
            row.normalized_connectivity_sha256  # type: ignore[attr-defined]
            != expected_normalized_connectivity_sha256
        ):
            raise PortfolioSignoffError(
                f"{circuit_id} physical ledger connectivity binding differs "
                "from its reference manifest"
            )
        source_locator = next(
            (
                locator
                for locator in row.artifacts  # type: ignore[attr-defined]
                if locator.logical_id == "source"
            ),
            None,
        )
        if source_locator is None or source_locator.canonical_data() != expected_source:
            raise PortfolioSignoffError(
                f"{circuit_id} physical ledger intent/source binding differs"
            )
        if reference_manifest is not None:
            from clay.flow.reference_evidence import (
                ReferenceEvidenceError,
                verify_clay_reference_topology,
            )

            candidate_locator = next(
                (
                    locator
                    for locator in row.artifacts  # type: ignore[attr-defined]
                    if locator.logical_id == "candidate"
                ),
                None,
            )
            if candidate_locator is None:
                raise PortfolioSignoffError(
                    f"{circuit_id} physical ledger candidate binding is missing"
                )
            try:
                for locator in (source_locator, candidate_locator):
                    verify_clay_reference_topology(
                        Path(repo_root).resolve() / Path(locator.path),
                        manifest=reference_manifest,  # type: ignore[arg-type]
                        pdk_id=row.pdk_id,  # type: ignore[attr-defined]
                        repo_root=repo_root,
                    )
            except ReferenceEvidenceError as exc:
                raise PortfolioSignoffError(
                    f"{circuit_id} physical ledger Clay/reference binding differs: {exc}"
                ) from exc


def load_strict_physical_pdk_ledger_rows(
    path: str | Path,
    *,
    current_commit: str,
    circuit_id: str,
    expected_source_sha256: str,
    expected_source_path: str,
    expected_topology_id: str,
    expected_normalized_connectivity_sha256: str,
    reference_manifest: object | None = None,
    three_reference_ledger: bool = False,
    repo_root: str | Path = REPO_ROOT,
    commit_validator: Callable[[str, str], bool] | None = None,
):
    """Load canonical v2 physical-matrix bytes and verify every artifact row."""

    from clay.core_contract import VerificationStatus
    from clay.flow.portfolio_signoff import (
        FOUR_PDK_IDS,
        PHYSICAL_CLAIM_SCOPE,
        PortfolioSignoffError,
        PortfolioSignoffRow,
        portfolio_row_from_mapping,
        validate_portfolio_row_artifacts,
    )

    ledger = Path(path)
    try:
        raw_bytes = ledger.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger is missing or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise PortfolioSignoffError(f"{circuit_id} physical ledger root is invalid")
    canonical = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if raw_bytes != canonical:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger bytes are non-canonical"
        )
    required_fields = {
        "schema",
        "claim_scope",
        "physical_status",
        "physical_signed_off",
        "commit",
        "source_state_sha256",
        "intent",
        "summary",
        "signoffs",
        "portfolio_rows",
    }
    actual_fields = set(payload)
    if actual_fields != required_fields:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger fields are non-canonical; "
            f"missing={sorted(required_fields - actual_fields)}, "
            f"unknown={sorted(actual_fields - required_fields)}"
        )
    if payload.get("schema") != "clay.physical-pdk-matrix/v2":
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger schema is unsupported"
        )
    if payload.get("claim_scope") != PHYSICAL_CLAIM_SCOPE:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger scope is unsupported"
        )
    try:
        physical_status = VerificationStatus(payload["physical_status"])
    except (TypeError, ValueError) as exc:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger physical_status is invalid"
        ) from exc
    if physical_status is not VerificationStatus.PASS:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger physical_status is not pass"
        )
    evidence_commit = payload.get("commit")
    if (
        payload.get("physical_signed_off") is not True
        or type(evidence_commit) is not str
        or len(evidence_commit) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in evidence_commit
        )
    ):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger is stale or not signed off"
        )
    if evidence_commit != current_commit and (
        commit_validator is None
        or not commit_validator(evidence_commit, current_commit)
    ):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger commit is not an unchanged "
            "authoritative ancestor"
        )
    raw_signoffs = payload["signoffs"]
    raw_rows = payload["portfolio_rows"]
    if type(raw_signoffs) is not list or raw_signoffs != raw_rows:
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger signoffs and portfolio_rows differ"
        )
    if not isinstance(raw_rows, list) or len(raw_rows) != len(FOUR_PDK_IDS):
        raise PortfolioSignoffError(
            f"{circuit_id} physical ledger does not contain four rows"
        )
    rows = []
    for expected_pdk, raw_row in zip(FOUR_PDK_IDS, raw_rows):
        row = portfolio_row_from_mapping(raw_row)
        if not isinstance(row, PortfolioSignoffRow):
            raise PortfolioSignoffError(
                f"{circuit_id} physical ledger contains a legacy row"
            )
        if (
            row.circuit_id != circuit_id
            or row.pdk_id != expected_pdk
            or row.canonical_data() != raw_row
        ):
            raise PortfolioSignoffError(
                f"{circuit_id} physical ledger row identity is non-canonical"
            )
        validate_portfolio_row_artifacts(
            row,
            current_commit=evidence_commit,
            repo_root=repo_root,
        )
        rows.append(row)
    _validate_strict_physical_ledger_content(
        payload,
        rows,
        circuit_id=circuit_id,
        repo_root=repo_root,
        expected_source_sha256=expected_source_sha256,
        expected_source_path=expected_source_path,
        expected_topology_id=expected_topology_id,
        expected_normalized_connectivity_sha256=(
            expected_normalized_connectivity_sha256
        ),
        reference_manifest=reference_manifest,
        three_reference_ledger=three_reference_ledger,
    )
    return tuple(rows)


__all__ = [
    "PHYSICAL_STEPS",
    "bind_physical_search_provenance",
    "gds_geometry_metrics",
    "load_strict_physical_pdk_ledger_rows",
    "publish_physical_pdk_ledger",
    "render_cross_ledger_summary",
    "summarize_physical_pdk_flow",
]
