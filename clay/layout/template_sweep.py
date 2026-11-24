"""Parametric sweep utility over cell template factory args.

Given a cell template factory (e.g. ``build_diff_pair``), a dict of fixed
keyword arguments, and a dict of sweep keyword arguments mapping names to
lists of values, ``sweep_template`` runs the Cartesian product of all
sweep values, invokes the factory once per combination, and collects the
resulting ``(AnnotatedCell, DeviceSizing)`` plus a validation flag into a
list of :class:`SweepResult`.

Intentional design notes:

* The factory is never modified or wrapped; we simply call it with the
  merged kwargs.
* ``validation_clean`` calls :func:`clay.layout.spatial_validate.validate`
  and reports ``len(errors) == 0``.
* Cartesian product order is the iteration order of ``sweep_kwargs``
  (insertion order in Python 3.7+) so callers can reason about which
  axis varies fastest (the last one).
* An empty ``sweep_kwargs`` mapping yields exactly one result: the
  factory called with ``fixed_kwargs`` only. This is the natural
  identity element of a Cartesian product.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .spatial_validate import validate


@dataclass
class SweepResult:
    """One row of a template sweep.

    Attributes:
        args: the merged kwargs passed to the factory (fixed + this combo).
        cell: the AnnotatedCell returned by the factory.
        sizing: the DeviceSizing returned by the factory.
        validation_clean: True iff ``validate(cell) == []``.
    """
    args: Dict[str, Any]
    cell: Any
    sizing: Any
    validation_clean: bool


def sweep_template(
    factory: Callable,
    fixed_kwargs: Dict[str, Any],
    sweep_kwargs: Dict[str, List[Any]],
) -> List[SweepResult]:
    """Cartesian product of sweep_kwargs values, merged with fixed_kwargs.

    For each combination the ``factory`` is called once with the merged
    keyword arguments. The factory is expected to return a 2-tuple
    ``(cell, sizing)`` matching the cell-template convention (see e.g.
    :func:`clay.layout.cell_templates.build_diff_pair`).

    Args:
        factory: callable returning ``(AnnotatedCell, DeviceSizing)``.
        fixed_kwargs: keyword arguments held constant across the sweep.
            Pass an empty dict to omit. Must not share keys with
            ``sweep_kwargs``.
        sweep_kwargs: mapping from kwarg name to list of values. The
            Cartesian product of these lists defines the sweep. An empty
            mapping yields exactly one result.

    Returns:
        List of :class:`SweepResult`, one per combination, in Cartesian
        product order (axes earlier in the dict vary slowest).

    Raises:
        ValueError: if a key appears in both ``fixed_kwargs`` and
            ``sweep_kwargs`` (ambiguous).
    """
    overlap = set(fixed_kwargs).intersection(sweep_kwargs)
    if overlap:
        raise ValueError(
            f"keys appear in both fixed_kwargs and sweep_kwargs: "
            f"{sorted(overlap)}"
        )

    sweep_names = list(sweep_kwargs.keys())
    sweep_values = [list(sweep_kwargs[k]) for k in sweep_names]

    # itertools.product of zero iterables yields a single empty tuple,
    # which is exactly the identity element we want.
    combos = itertools.product(*sweep_values) if sweep_values else [()]

    results: List[SweepResult] = []
    for combo in combos:
        combo_kwargs = dict(zip(sweep_names, combo))
        merged = {**fixed_kwargs, **combo_kwargs}
        cell, sizing = factory(**merged)
        clean = len(validate(cell)) == 0
        results.append(SweepResult(
            args=merged,
            cell=cell,
            sizing=sizing,
            validation_clean=clean,
        ))
    return results


def filter_clean(results: List[SweepResult]) -> List[SweepResult]:
    """Return only results with ``validation_clean == True``.

    Order is preserved.
    """
    return [r for r in results if r.validation_clean]


def summarize_sweep(results: List[SweepResult]) -> str:
    """One-line text summary of a sweep.

    Format: ``"N=<total> clean=<m> varied=[<param1>, <param2>, ...]"``

    The varied-parameter list is computed by inspecting each result's
    ``args`` and selecting keys whose value is not constant across all
    results. If ``results`` is empty, the varied list is empty too.
    """
    total = len(results)
    clean_count = sum(1 for r in results if r.validation_clean)

    varied: List[str] = []
    if results:
        all_keys: List[str] = []
        seen = set()
        for r in results:
            for k in r.args.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        for k in all_keys:
            values = [r.args.get(k) for r in results]
            # A param is "varied" if at least two results have distinct values.
            # Use repr to compare unhashables uniformly.
            distinct_reprs = {repr(v) for v in values}
            if len(distinct_reprs) > 1:
                varied.append(k)

    return f"N={total} clean={clean_count} varied={varied}"
