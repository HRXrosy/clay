"""Map layout signoff results into proof-obligation trace rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def attach_signoff_result(cell: Any, result: Any) -> Any:
    """Attach one production signoff result using the CLI-readable contract."""
    existing = getattr(cell, "__dict__", {}).get("_signoff_results")
    if existing is None:
        cell.__dict__["_signoff_results"] = [result]
    elif isinstance(existing, list):
        existing.append(result)
    else:
        cell.__dict__["_signoff_results"] = [existing, result]
    return cell


def attach_signoff_results(cell: Any, results: Any) -> Any:
    """Attach multiple production signoff results using the CLI-readable contract."""
    for result in list(results or []):
        attach_signoff_result(cell, result)
    return cell


def run_and_attach_signoff_result(cell: Any, runner: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a production signoff callable and attach its result to ``cell``."""
    result = runner(*args, **kwargs)
    attach_signoff_result(cell, result)
    return result


def signoff_obligation_rows(cell_name: str, result: Any) -> list[dict]:
    """Expand a production signoff result into required proof-obligation rows."""
    if _has_any_attr(result, "results"):
        return [
            template_drc_obligation_row(cell_name, item)
            for item in _get(result, "results", [])
        ]
    erc_rows = _erc_obligation_rows(cell_name, result)
    flat_rows = _flat_signoff_obligation_rows(cell_name, result)
    if not (_has_any_attr(result, "drc") and _has_any_attr(result, "lvs")):
        if flat_rows:
            return flat_rows + erc_rows
        if erc_rows:
            return erc_rows
        raise TypeError(f"unsupported signoff result type: {type(result).__name__}")

    same_gds_required = hasattr(type(result), "same_gds") or _has_any_attr(result, "same_gds")
    drc_result = _get(result, "drc")
    lvs_result = _get(result, "lvs")
    if same_gds_required:
        drc_row = _coclean_drc_obligation_row(cell_name, drc_result)
        lvs_row = _coclean_lvs_obligation_row(cell_name, lvs_result)
        same_gds = bool(_get(result, "same_gds", False))
        co_clean_raw = _get(result, "co_clean", None)
        co_clean = (
            drc_row["status"] == "pass" and lvs_row["status"] == "pass" and same_gds
            if co_clean_raw is None
            else bool(co_clean_raw)
        )
    else:
        drc_row = _cell_drc_obligation_row(cell_name, drc_result)
        lvs_row = _cell_lvs_obligation_row(cell_name, lvs_result)
        co_clean_raw = _get(result, "dual_clean", None)
        co_clean = (
            drc_row["status"] == "pass" and lvs_row["status"] == "pass"
            if co_clean_raw is None
            else bool(co_clean_raw)
        )
        same_gds = None

    rows = [
        drc_row,
        lvs_row,
        _co_clean_obligation_row(
            cell_name,
            status="pass" if co_clean else "fail",
            same_gds_required=same_gds_required,
            same_gds=same_gds,
            co_clean=co_clean,
            drc_status=drc_row["status"],
            lvs_status=lvs_row["status"],
        ),
    ]
    return rows + erc_rows


def template_drc_obligation_row(cell_name: str, result: Any) -> dict:
    template_name = str(getattr(result, "template_name", "template"))
    skipped_reason = getattr(result, "skipped_reason", None)
    violation_count = int(getattr(result, "violation_count", 0) or 0)
    if skipped_reason:
        status = "unsupported"
    elif violation_count == 0 and bool(getattr(result, "clean", False)):
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="drc",
        check=f"template:{template_name}",
        status=status,
        coverage={"artifact_kind": "template_drc", "ran_real_tool": not bool(skipped_reason)},
        evidence={
            "clean": bool(getattr(result, "clean", False)),
            "violation_count": violation_count,
            "violations_by_rule": dict(getattr(result, "violations_by_rule", {}) or {}),
            "skipped_reason": skipped_reason,
        },
    )


def _cell_drc_obligation_row(cell_name: str, report: Any) -> dict:
    ran_real_deck = bool(getattr(report, "ran_real_deck", False))
    total_violations = getattr(report, "total_violations", None)
    if not ran_real_deck:
        status = "pending"
    elif total_violations == 0:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="drc",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "gds",
            "ran_real_tool": ran_real_deck,
            "gds_path": _path_str(getattr(report, "gds_path", None)),
        },
        evidence={
            "total_violations": total_violations,
            "counts_by_rule": dict(getattr(report, "by_rule", {}) or {}),
            "note": getattr(report, "note", ""),
        },
    )


def gf180_drc_obligation_row(cell_name: str, report: Any) -> dict:
    ran_real_deck = bool(getattr(report, "ran_real_deck", False))
    total_violations = int(getattr(report, "total_violations", 0) or 0)
    if not ran_real_deck:
        status = "pending"
    elif total_violations == 0:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="drc",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "lyrdb",
            "ran_real_tool": ran_real_deck,
            "gds_path": _path_str(getattr(report, "gds_path", None)),
            "report_path": _path_str(getattr(report, "lyrdb_path", None)),
        },
        evidence={
            "total_violations": total_violations,
            "rule_count": int(getattr(report, "rule_count", 0) or 0),
            "counts_by_rule": _call_or_empty(report, "counts_by_rule"),
            "returncode": getattr(report, "returncode", None),
        },
    )


def _cell_lvs_obligation_row(cell_name: str, report: Any) -> dict:
    ran_real_deck = bool(getattr(report, "ran_real_deck", False))
    matched = getattr(report, "matched", None)
    extracted_devices = getattr(report, "extracted_devices", None)
    schematic_devices = getattr(report, "schematic_devices", None)
    vacuous = bool(getattr(report, "vacuous", False))
    if not ran_real_deck:
        status = "pending"
    elif matched is True and not vacuous:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="lvs",
        check="gf180mcu",
        status=status,
        coverage={"artifact_kind": "lvsdb", "ran_real_tool": ran_real_deck},
        evidence={
            "matched": matched,
            "device_count": extracted_devices,
            "net_count": getattr(report, "extracted_nets", None),
            "schematic_device_count": schematic_devices,
            "schematic_net_count": getattr(report, "schematic_nets", None),
            "vacuous": vacuous,
            "note": getattr(report, "note", ""),
        },
    )


def gf180_lvs_obligation_row(
    cell_name: str,
    report: Any,
    *,
    schematic_device_count: int | None = None,
) -> dict:
    ran_real_deck = bool(getattr(report, "ran_real_deck", False))
    matched = getattr(report, "matched", None)
    mismatches = list(getattr(report, "mismatches", []) or [])
    device_count = getattr(report, "device_count", None)
    if not ran_real_deck:
        status = "pending"
    elif matched is not True or mismatches:
        status = "fail"
    elif not isinstance(device_count, int) or device_count <= 0:
        status = "fail"
    elif schematic_device_count is not None and device_count < schematic_device_count:
        status = "fail"
    else:
        status = "pass"
    return _row(
        cell_name=cell_name,
        obligation_kind="lvs",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "lvsdb",
            "ran_real_tool": ran_real_deck,
            "gds_path": _path_str(getattr(report, "gds_path", None)),
            "source_netlist_path": _path_str(getattr(report, "schematic_path", None)),
            "report_path": _path_str(getattr(report, "lvsdb_path", None)),
            "extracted_netlist_path": _path_str(
                getattr(report, "extracted_netlist_path", None)
            ),
        },
        evidence={
            "matched": matched,
            "device_count": device_count,
            "net_count": getattr(report, "net_count", None),
            "schematic_device_count": schematic_device_count,
            "mismatch_count": len(mismatches),
            "returncode": getattr(report, "returncode", None),
        },
    )


def _flat_drc_obligation_row(cell_name: str, result: Any) -> dict:
    ran = bool(_first_present(result, "drc_ran", "ran_real_deck", default=False))
    total_violations = _first_present(
        result,
        "drc_total_violations",
        "total_violations",
    )
    if not ran:
        status = "pending"
    elif total_violations == 0:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="drc",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "gds",
            "ran_real_tool": ran,
            "gds_path": _path_str(_get(result, "gds_path", None)),
        },
        evidence={
            "total_violations": total_violations,
            "counts_by_rule": dict(
                _first_present(result, "drc_by_rule", "by_rule", default={}) or {}
            ),
            "pending_reason": _first_present(
                result,
                "drc_pending_reason",
                "pending_reason",
                default="",
            ),
        },
    )


def _flat_lvs_obligation_row(cell_name: str, result: Any) -> dict:
    ran = bool(_first_present(result, "lvs_ran", "ran_real_lvs", "ran", default=False))
    matched = _lvs_matched(result)
    vacuous = bool(
        _first_present(
            result,
            "lvs_vacuous",
            "vacuous_extraction",
            "vacuous",
            default=False,
        )
    )
    if not ran:
        status = "pending"
    elif matched is True and not vacuous:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="lvs",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "lvsdb",
            "ran_real_tool": ran,
            "gds_path": _path_str(
                _first_present(result, "gds_path", "lvs_gds_path")
            ),
            "source_netlist_path": _path_str(
                _first_present(result, "source_netlist_path", "schematic_path")
            ),
        },
        evidence={
            "matched": matched,
            "device_count": _first_present(
                result,
                "lvs_device_count",
                "device_count",
                "device_count_layout",
                "extracted_device_count",
            ),
            "net_count": _first_present(
                result,
                "lvs_net_count",
                "net_count",
                "net_count_layout",
                "extracted_net_count",
            ),
            "schematic_device_count": _first_present(
                result,
                "schematic_device_count",
                "device_count_source",
            ),
            "schematic_net_count": _first_present(
                result,
                "schematic_net_count",
                "net_count_source",
            ),
            "vacuous": vacuous,
            "mismatch_count": int(_get(result, "mismatch_count", 0) or 0),
            "counts_by_category": dict(_get(result, "by_category", {}) or {}),
            "pending_reason": _first_present(
                result,
                "lvs_pending_reason",
                "pending_reason",
                "note",
                default="",
            ),
        },
    )


def erc_no_shorts_obligation_row(cell_name: str, result: Any) -> dict:
    ran = _erc_no_shorts_ran(result)
    short_count = _first_present(
        result,
        "union_find_short_count",
        "short_count",
    )
    distinct = _first_present(
        result,
        "union_find_distinct",
        "distinct",
        "no_shorts",
    )
    if not ran or short_count is None:
        status = "pending"
    elif int(short_count) == 0 and distinct is not False:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="erc_no_shorts",
        check="union_find",
        status=status,
        coverage={
            "artifact_kind": "layout_connectivity",
            "ran_real_tool": ran,
        },
        evidence={
            "short_count": short_count,
            "distinct": distinct,
            "merged_pairs": list(
                _first_present(
                    result,
                    "union_find_merged_pairs",
                    "merged_pairs",
                    default=[],
                )
                or []
            ),
        },
    )


def erc_power_connectivity_obligation_row(cell_name: str, result: Any) -> dict:
    ran = bool(_get(result, "power_conn_ran", False))
    passed = _get(result, "power_conn_passed", None)
    if not ran:
        status = "pending"
    elif passed is True:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="erc_power_connectivity",
        check="power_physical_connectivity",
        status=status,
        coverage={
            "artifact_kind": "layout_connectivity",
            "ran_real_tool": ran,
        },
        evidence={
            "passed": passed,
            "per_net": dict(_get(result, "power_conn_per_net", {}) or {}),
            "detail": _get(result, "power_conn_detail", ""),
        },
    )


def _coclean_drc_obligation_row(cell_name: str, report: Any) -> dict:
    ran_real_deck = bool(getattr(report, "ran_real_deck", False))
    total_violations = getattr(report, "total_violations", None)
    if not ran_real_deck:
        status = "pending"
    elif total_violations == 0:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="drc",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "lyrdb",
            "ran_real_tool": ran_real_deck,
            "gds_path": _path_str(getattr(report, "gds_path", None)),
        },
        evidence={
            "total_violations": total_violations,
            "counts_by_rule": dict(getattr(report, "by_rule", {}) or {}),
            "pending_reason": getattr(report, "pending_reason", ""),
        },
    )


def _coclean_lvs_obligation_row(cell_name: str, report: Any) -> dict:
    ran_real_deck = bool(getattr(report, "ran_real_deck", False))
    matched = getattr(report, "matched", None)
    device_count = getattr(report, "device_count_layout", None)
    source_device_count = getattr(report, "device_count_source", None)
    vacuous = bool(getattr(report, "vacuous_extraction", False))
    if not ran_real_deck:
        status = "pending"
    elif matched is True and not vacuous:
        status = "pass"
    else:
        status = "fail"
    return _row(
        cell_name=cell_name,
        obligation_kind="lvs",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "lvsdb",
            "ran_real_tool": ran_real_deck,
            "gds_path": _path_str(getattr(report, "gds_path", None)),
            "source_netlist_path": _path_str(
                getattr(report, "source_netlist_path", None)
            ),
        },
        evidence={
            "matched": matched,
            "device_count": device_count,
            "net_count": getattr(report, "net_count_layout", None),
            "schematic_device_count": source_device_count,
            "schematic_net_count": getattr(report, "net_count_source", None),
            "vacuous": vacuous,
            "pending_reason": getattr(report, "pending_reason", ""),
        },
    )


def _co_clean_obligation_row(
    cell_name: str,
    *,
    status: str,
    same_gds_required: bool,
    same_gds: bool | None,
    co_clean: bool,
    drc_status: str,
    lvs_status: str,
) -> dict:
    return _row(
        cell_name=cell_name,
        obligation_kind="co_clean",
        check="gf180mcu",
        status=status,
        coverage={
            "artifact_kind": "aggregate_signoff",
            "ran_real_tool": drc_status != "pending" and lvs_status != "pending",
            "same_gds_required": same_gds_required,
        },
        evidence={
            "co_clean": co_clean,
            "dual_clean": co_clean,
            "same_gds": same_gds,
            "drc_status": drc_status,
            "lvs_status": lvs_status,
        },
    )


def _row(
    *,
    cell_name: str,
    obligation_kind: str,
    check: str,
    status: str,
    coverage: dict,
    evidence: dict,
) -> dict:
    obligation_id = f"obl:{cell_name}:signoff:{obligation_kind}:{check}"
    return {
        "obligation_id": obligation_id,
        "trace_id": f"tr:{cell_name}:signoff:{obligation_kind}:{check}",
        "cell": cell_name,
        "source_kind": "layout_signoff",
        "obligation_kind": obligation_kind,
        "required": True,
        "status": status,
        "source": {
            "cell": cell_name,
            "name": check,
            "field": obligation_kind,
            "ir_symbol": f"{cell_name}.signoff.{obligation_kind}.{check}",
            "source_ref": {
                "kind": "signoff",
                "path": f"cell[{cell_name}].signoff[{obligation_kind}][{check}]",
            },
        },
        "coverage": coverage,
        "evidence": evidence,
    }


def _path_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _call_or_empty(obj: Any, name: str) -> dict:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return {}
    value = fn()
    return dict(value or {})


def _flat_signoff_obligation_rows(cell_name: str, result: Any) -> list[dict]:
    has_drc = _has_any_attr(
        result,
        "drc_ran",
        "drc_total_violations",
        "ran_real_deck",
        "total_violations",
    )
    has_lvs = _looks_like_lvs_result(result)
    if not has_drc and not has_lvs:
        return []

    rows = []
    drc_row = None
    lvs_row = None
    if has_drc:
        drc_row = _flat_drc_obligation_row(cell_name, result)
        rows.append(drc_row)
    if has_lvs:
        lvs_row = _flat_lvs_obligation_row(cell_name, result)
        rows.append(lvs_row)
    if drc_row is not None and lvs_row is not None:
        co_clean_raw = _get(result, "co_clean", None)
        co_clean = (
            drc_row["status"] == "pass" and lvs_row["status"] == "pass"
            if co_clean_raw is None
            else bool(co_clean_raw)
        )
        rows.append(
            _co_clean_obligation_row(
                cell_name,
                status="pass" if co_clean else "fail",
                same_gds_required=False,
                same_gds=None,
                co_clean=co_clean,
                drc_status=drc_row["status"],
                lvs_status=lvs_row["status"],
            )
        )
    return rows


def _erc_obligation_rows(cell_name: str, result: Any) -> list[dict]:
    rows = []
    if _has_any_attr(
        result,
        "union_find_ran",
        "union_find_short_count",
        "union_find_distinct",
        "distinct",
        "short_count",
        "no_shorts",
    ):
        rows.append(erc_no_shorts_obligation_row(cell_name, result))
    if _has_any_attr(
        result,
        "power_conn_ran",
        "power_conn_passed",
        "power_conn_per_net",
    ):
        rows.append(erc_power_connectivity_obligation_row(cell_name, result))
    return rows


def _erc_no_shorts_ran(result: Any) -> bool:
    if _has_any_attr(result, "union_find_ran"):
        return bool(_get(result, "union_find_ran"))
    return _has_any_attr(result, "union_find_short_count", "short_count", "distinct")


def _has_any_attr(obj: Any, *names: str) -> bool:
    if isinstance(obj, dict):
        return any(name in obj for name in names)
    return any(hasattr(obj, name) for name in names)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _looks_like_lvs_result(obj: Any) -> bool:
    if _has_any_attr(obj, "lvs_ran", "lvs_matched", "ran_real_lvs"):
        return True
    if _has_any_attr(obj, "device_match", "net_match", "vacuous_extraction"):
        return True
    return _has_any_attr(obj, "ran", "matched") and _has_any_attr(
        obj,
        "device_count",
        "net_count",
        "schematic_net_count",
    )


def _lvs_matched(result: Any) -> bool | None:
    matched = _first_present(result, "lvs_matched", "matched")
    if matched is not None:
        return matched
    device_match = _get(result, "device_match", None)
    net_match = _get(result, "net_match", None)
    if device_match is None and net_match is None:
        return None
    return bool(device_match is True and net_match is True)


def _first_present(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if _has_any_attr(obj, name):
            return _get(obj, name)
    return default
