"""Layout regression baseline utility.

Stores golden :class:`ParameterizedLibrary` snapshots on disk and compares
new library output against those baselines. Built on top of:

* ``clay.layout.json_serde`` (Phase 17) for stable on-disk JSON form.
* ``clay.layout.lib_diff`` (Phase 30) for cell-/polygon-level diffs.

Typical usage:

    >>> save_baseline(lib, "baselines/", "diff_pair_4f")          # one-time
    >>> result = compare_to_baseline(new_lib, "baselines/", "diff_pair_4f")
    >>> assert result.passed, result.diff_summary

Or in batch:

    >>> results = regression_check_all({"diff_pair_4f": new_lib, ...},
    ...                                "baselines/")
    >>> bad = [r for r in results if not r.passed]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from .backend import ParameterizedLibrary
from .json_serde import (
    deserialize_parameterized_library,
    serialize_parameterized_library,
)
from .lib_diff import diff_libraries, summarize_diff


__all__ = [
    "RegressionResult",
    "save_baseline",
    "load_baseline",
    "compare_to_baseline",
    "regression_check_all",
]


@dataclass
class RegressionResult:
    """Outcome of comparing a single new library against its golden baseline.

    Attributes
    ----------
    template_name:
        Logical name identifying the baseline slot (e.g. ``"diff_pair_4f"``).
    passed:
        ``True`` iff the new library is structurally identical to the baseline
        (same cell set, same polygon multiset per cell).
    diff_summary:
        One-line human-readable summary. ``"identical"`` on pass; on fail this
        is the return value of :func:`clay.layout.lib_diff.summarize_diff`
        (e.g. ``"1 cell differs, 1 only in B"``).
    """

    template_name: str
    passed: bool
    diff_summary: str


def _baseline_path(baseline_dir: Union[str, Path], template_name: str) -> Path:
    return Path(baseline_dir) / f"{template_name}.json"


def save_baseline(
    lib: ParameterizedLibrary,
    baseline_dir: Union[str, Path],
    template_name: str,
) -> Path:
    """Serialize ``lib`` to ``<baseline_dir>/<template_name>.json``.

    The directory is created if missing. An existing baseline file at the
    same path is overwritten unconditionally — there is no append mode.

    Returns the absolute :class:`Path` written, so callers can log it.
    """
    out_dir = Path(baseline_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _baseline_path(out_dir, template_name)
    payload = serialize_parameterized_library(lib)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_baseline(
    baseline_dir: Union[str, Path],
    template_name: str,
) -> ParameterizedLibrary:
    """Load and deserialize the baseline JSON for ``template_name``.

    Raises :class:`FileNotFoundError` if no baseline exists for this template.
    """
    in_path = _baseline_path(baseline_dir, template_name)
    if not in_path.is_file():
        raise FileNotFoundError(
            f"no baseline file for template {template_name!r} at {in_path}"
        )
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return deserialize_parameterized_library(payload)


def compare_to_baseline(
    new_lib: ParameterizedLibrary,
    baseline_dir: Union[str, Path],
    template_name: str,
) -> RegressionResult:
    """Diff ``new_lib`` against the golden baseline on disk.

    Returns a :class:`RegressionResult`. Raises :class:`FileNotFoundError`
    if the baseline is missing (callers should treat that as "run
    :func:`save_baseline` first to bless the current output").
    """
    baseline_lib = load_baseline(baseline_dir, template_name)
    diff = diff_libraries(baseline_lib, new_lib)
    if diff.libraries_identical:
        return RegressionResult(
            template_name=template_name, passed=True, diff_summary="identical"
        )
    return RegressionResult(
        template_name=template_name,
        passed=False,
        diff_summary=summarize_diff(diff),
    )


def regression_check_all(
    new_libs: Dict[str, ParameterizedLibrary],
    baseline_dir: Union[str, Path],
) -> List[RegressionResult]:
    """Compare a batch of templates against their respective baselines.

    Iterates ``new_libs`` in insertion order and produces one
    :class:`RegressionResult` per template. A missing baseline is surfaced
    as a non-passing result whose ``diff_summary`` starts with
    ``"missing baseline: "`` — this keeps the batch from short-circuiting
    on the first un-blessed template.
    """
    results: List[RegressionResult] = []
    for template_name, new_lib in new_libs.items():
        try:
            results.append(compare_to_baseline(new_lib, baseline_dir, template_name))
        except FileNotFoundError as exc:
            results.append(
                RegressionResult(
                    template_name=template_name,
                    passed=False,
                    diff_summary=f"missing baseline: {exc}",
                )
            )
    return results
