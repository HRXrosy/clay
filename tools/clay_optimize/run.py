"""
CLI entry: ``python -m tools.clay_optimize``.

Wires the harness to a registered Clay cell template and runs N
iterations against a target spec. The CLI is intentionally minimal —
no model API, no result aggregation, no Pareto-front; that's tSS-BO's
job once it lands in v0.0.68+. v0.0.67 ships the smoke-test surface.

Usage::

    python -m tools.clay_optimize \\
        --cell sky130_two_stage_miller_pmos \\
        --target gain_dc=50dB --target gbw=1MHz \\
        --max-iters 10 \\
        --optimizer stub-random --seed 42

Targets accept the form ``NAME=VALUE[UNIT]`` with the operator implied
as ``>=`` for ``gain_dc`` / ``gbw`` and ``<=`` for ``power`` /
``area``; pass ``NAME>=VALUE`` to set it explicitly. Units recognized:
``dB`` (passthrough — interpreted as ``vdb`` already), ``Hz`` /
``kHz`` / ``MHz`` / ``GHz``, ``V`` / ``mV`` / ``uV``, ``A`` / ``mA`` /
``uA`` / ``nA``, ``F`` / ``pF`` / ``fF``, ``s`` / ``ms`` / ``us`` /
``ns`` / ``ps``. Plain numbers are treated as SI base units.

Exit codes:
  * 0 — at least one iteration produced a feasible entry with reward
        ≥ 0 (target met on every axis).
  * 1 — no feasible iteration produced positive reward (search did
        not find a satisfying sizing).
  * 2 — CLI / config error (unknown cell, bad target spec, etc.).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .budget import BoBudgetTracker, BudgetExceededError
from .harness import (
    OptimizerHistory,
    OptimizerHistoryEntry,
    OptimizerLoop,
    Sizing,
    SizingBounds,
    SpecTarget,
)
from .stub_optimizers import build_stub
from .tssbo_adapter import OPT_IN_ENV as TSSBO_OPT_IN_ENV
from .tssbo_adapter import build_tssbo_adapter


# ─── Unit table for target parsing ───────────────────────────────


_UNIT_TABLE: dict[str, float] = {
    "": 1.0,
    "db": 1.0,
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
    "a": 1.0,
    "ma": 1e-3,
    "ua": 1e-6,
    "na": 1e-9,
    "f": 1.0,
    "pf": 1e-12,
    "ff": 1e-15,
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    # Power: case-insensitive lookup, so `mW` lowercases to `mw`.
    # SI base is the watt.
    "w": 1.0,
    "mw": 1e-3,
    "uw": 1e-6,
    "nw": 1e-9,
    "pw": 1e-12,
    # Angle (deg / rad) for phase margin etc.
    "deg": 1.0,
    "rad": 1.0,
}


_TARGET_RE = re.compile(
    r"""^
    (?P<name>[A-Za-z_][A-Za-z_0-9]*)
    \s*
    (?P<op>>=|<=|>|<|=)?
    \s*
    (?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s*
    (?P<unit>[A-Za-z]*)
    \s*$
    """,
    re.VERBOSE,
)


def parse_target(token: str) -> SpecTarget:
    """Parse one ``--target`` value into a ``SpecTarget``.

    Forms (operator defaults shown last):

      * ``name>=value[unit]`` — explicit operator.
      * ``name<=value[unit]`` — explicit operator.
      * ``name=value[unit]``  — equality (rare; mainly for ``=``).
      * ``name value[unit]``  — operator inferred by name conventions:
        - ``gain*`` / ``gbw`` / ``bw`` / ``pm`` → ``>=``
        - ``power*`` / ``area*`` / ``noise*`` → ``<=``
        - everything else → ``>=`` (the optimistic default)

    Raises ValueError on parse error.
    """
    m = _TARGET_RE.match(token)
    if m is None:
        raise ValueError(
            f"bad --target spec {token!r}; expected "
            f"'NAME[OP]VALUE[UNIT]' (e.g. 'gain_dc>=50dB' or 'gbw=1MHz')"
        )
    name = m.group("name")
    op = m.group("op") or _infer_op_from_name(name)
    raw_value = float(m.group("value"))
    unit = m.group("unit").lower()
    if unit not in _UNIT_TABLE:
        raise ValueError(
            f"unknown unit {unit!r} on target {token!r}; "
            f"recognized: {sorted(_UNIT_TABLE)}"
        )
    return SpecTarget(
        name=name,
        op=op,
        value=raw_value * _UNIT_TABLE[unit],
        units=unit,
    )


def _infer_op_from_name(name: str) -> str:
    lower = name.lower()
    if (
        lower.startswith("power")
        or lower.startswith("area")
        or lower.startswith("noise")
        or "current" in lower
    ):
        return "<="
    return ">="


# ─── Cell registry ───────────────────────────────────────────────


def _build_two_stage_miller_pmos(
    cell_name: str, targets: Sequence[SpecTarget],
):
    """Build a Cell + matching testbench + sensible default sizing
    bounds for the two_stage_miller_pmos template.

    Returns a triple of (cell_builder, testbench_builder, bounds):
      * cell_builder: ``Sizing -> Cell``
      * testbench_builder: ``Cell -> Testbench``
      * bounds: default ``SizingBounds`` for the three gm/Id knobs.

    The cell builder closes over the cell name and converts the
    optimizer's sizing dict directly into the template's kwargs.
    Unknown keys in the sizing dict are passed through (Python
    raises ``TypeError`` if the template doesn't accept them — the
    harness records that as a builder failure).
    """
    from clay.ir import Spec, SpecField
    from clay.templates.two_stage_miller_pmos import (
        build_ac_testbench, build_two_stage_miller_pmos,
    )

    # Materialize a Spec from the target list; SpecField uses the same
    # (name, op, value) triple so the conversion is mechanical.
    fields = [
        SpecField(
            name=t.name,
            # Pick a generic dimension tag; the verifier only uses
            # name/op/value at the assertion-check site.
            dimension="real",
            op=t.op,
            value=t.value,
        )
        for t in targets
    ]
    spec = Spec(fields=fields)

    def cell_builder(sizing: Sizing):
        return build_two_stage_miller_pmos(
            cell_name=cell_name,
            spec=spec,
            **sizing,
        )

    def testbench_builder(cell):
        return build_ac_testbench(cell)

    # Default bounds: gm/Id values in the textbook strong-inversion
    # band [6, 25]. tSS-BO will replace these with truncated subspace
    # priors derived from previous runs once v0.0.68+ lands.
    bounds = [
        SizingBounds(knob="gm_id_diff", low=6.0, high=25.0),
        SizingBounds(knob="gm_id_load1", low=6.0, high=25.0),
        SizingBounds(knob="gm_id_cs", low=6.0, high=25.0),
    ]
    return cell_builder, testbench_builder, bounds


_CELL_REGISTRY = {
    # sky130 alias for the built-in PMOS-input two-stage Miller opamp.
    # The optimizer harness needs no other cell to validate Phase 3
    # plumbing; v0.0.68+ adds more registered cells.
    "sky130_two_stage_miller_pmos": _build_two_stage_miller_pmos,
    "two_stage_miller_pmos": _build_two_stage_miller_pmos,
}


def list_cells() -> List[str]:
    return sorted(_CELL_REGISTRY)


# ─── Default simulator wrapper (real ngspice path) ───────────────


def _real_ngspice_simulator(tb, work_dir):
    """Run ``clay.verifier.verify`` and pack its output to the
    ``Simulator`` callable contract the harness expects.

    Returns a dict with two keys:
      * ``measures`` — ``dict[str, float]`` of the simulator's
        extracted scalars.
      * ``simulator_error`` — ``None`` on success, error string on
        driver-level failure.

    Driver-level exceptions (ngspice missing, PDK path bad) propagate
    to the harness's ``_run_one_iter`` except handler; the harness
    records ``simulator_error`` itself from there.
    """
    from clay.verifier import verify  # lazy; keep CLI light

    result = verify(tb, work_dir=work_dir)
    return {
        "measures": dict(result.measures),
        "simulator_error": result.simulator_error,
    }


# ─── Pretty-print history table ──────────────────────────────────


def _render_history_table(
    history: OptimizerHistory, targets: Sequence[SpecTarget]
) -> str:
    """Produce a one-line-per-iteration summary table.

    Columns: ``iter``, ``feasible``, ``reward``, per-target actual
    value (or ``-`` when missing), ``violations`` count, ``elapsed``.
    """
    lines: List[str] = []
    header_cols = ["iter", "feas", "reward"]
    for t in targets:
        header_cols.append(t.name)
    header_cols += ["#viol", "elapsed_s"]
    lines.append(" ".join(f"{c:>12s}" for c in header_cols))
    lines.append(" ".join(f"{'-' * 12:>12s}" for _ in header_cols))
    for e in history:
        row = [
            f"{e.iteration:>12d}",
            f"{('yes' if e.is_feasible else 'no'):>12s}",
        ]
        if e.reward != e.reward:  # NaN check
            row.append(f"{'nan':>12s}")
        else:
            row.append(f"{e.reward:>12.4f}")
        for t in targets:
            v = e.measures.get(t.name)
            if v is None:
                row.append(f"{'-':>12s}")
            else:
                row.append(f"{v:>12.4g}")
        row.append(f"{len(e.violations):>12d}")
        row.append(f"{e.elapsed_sec:>12.3f}")
        lines.append(" ".join(row))
    best = history.best_entry()
    if best is None:
        lines.append("(no iterations completed)")
    else:
        lines.append("")
        lines.append(
            f"best iter={best.iteration} feasible={best.is_feasible} "
            f"reward={best.reward!r} "
            f"sizing={dict(best.sizing)}"
        )
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="tools.clay_optimize",
        description=(
            "Clay optimizer harness CLI (v0.0.67 Phase 3 first slice). "
            "Run a stub optimizer against a Clay cell + target spec."
        ),
    )
    p.add_argument(
        "--cell", required=False, default=None,
        help=(
            "name of a registered cell template "
            f"(one of: {', '.join(list_cells())})"
        ),
    )
    p.add_argument(
        "--target", action="append", default=[], metavar="SPEC",
        help=(
            "target spec field, e.g. 'gain_dc=50dB' or 'gbw>=1MHz'. "
            "Repeat for multiple targets."
        ),
    )
    p.add_argument(
        "--max-iters", type=int, default=10,
        help="number of iterations to run (default: 10)",
    )
    p.add_argument(
        "--optimizer", default="stub-random",
        choices=["stub-random", "stub-grid", "stub-noop", "tssbo"],
        help=(
            "which optimizer to use. ``tssbo`` selects the real "
            "Truncated Subspace Sampling BO (DATE 2024 / "
            "github.com/PascalInn/tSS-BO); the four-layer opt-in "
            "(env var, this flag, package installed, budget cap) "
            "must clear or the dispatch refuses with exit code 2."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="PRNG seed for stub-random / grid scramble (default: 0)",
    )
    p.add_argument(
        "--max-bo-iters", type=int, default=100,
        help=(
            "per-run BO iteration cap (tssbo only; default: 100). "
            "The loop's --max-iters must be <= this budget cap, "
            "otherwise dispatch refuses before any BO traffic."
        ),
    )
    p.add_argument(
        "--max-wall-clock-sec", type=float, default=300.0,
        help=(
            "per-run BO wall-clock cap in seconds (tssbo only; "
            "default: 300.0)"
        ),
    )
    p.add_argument(
        "--max-ngspice-calls", type=int, default=100,
        help=(
            "per-run BO ngspice call cap (tssbo only; default: 100)"
        ),
    )
    p.add_argument(
        "--max-bo-proposals", type=int, default=500,
        help=(
            "per-run BO propose() call cap (tssbo only; default: 500)"
        ),
    )
    p.add_argument(
        "--work-dir", default="tmp_clay_optimize",
        help="root dir for ngspice intermediate files",
    )
    p.add_argument(
        "--list-cells", action="store_true",
        help="print the registered cell templates and exit",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "skip the real ngspice path; use a stub simulator that "
            "returns synthetic measures (smoke test of CLI wiring)."
        ),
    )
    args = p.parse_args(argv)

    if args.list_cells:
        for name in list_cells():
            print(name)
        return 0

    if not args.cell:
        sys.stderr.write("--cell is required (or pass --list-cells)\n")
        return 2

    if args.cell not in _CELL_REGISTRY:
        sys.stderr.write(
            f"unknown --cell {args.cell!r}; "
            f"known: {list_cells()}\n"
        )
        return 2

    if not args.target:
        sys.stderr.write("at least one --target is required\n")
        return 2

    # Parse targets.
    try:
        targets = [parse_target(t) for t in args.target]
    except ValueError as exc:
        sys.stderr.write(f"target parse error: {exc}\n")
        return 2

    cell_name = f"opt_{args.cell}"
    builder_fn = _CELL_REGISTRY[args.cell]
    cell_builder, testbench_builder, bounds = builder_fn(cell_name, targets)

    # Pick simulator: real ngspice (default) or a synthetic dry-run.
    if args.dry_run:
        def _stub_sim(tb, _work):  # noqa: ARG001
            # Synthetic measures that depend on the sizing inputs so
            # the optimizer sees gradient. For each target, return a
            # value half-way between 0 and the requested value — the
            # default reward is always negative, but the loop runs.
            return {
                "measures": {t.name: t.value * 0.5 for t in targets},
                "simulator_error": None,
            }
        simulator = _stub_sim
    else:
        simulator = _real_ngspice_simulator

    # ── Optimizer dispatch ──────────────────────────────────────────
    # Stubs route through `build_stub`; the real tSS-BO adapter has its
    # own four-layer opt-in gate. The CLI's responsibility here is the
    # second layer (the `--optimizer tssbo` flag); layers 1 (env var),
    # 3 (package present), 4 (budget cap) fire inside the adapter
    # constructor and surface as `RuntimeError` / `BudgetExceededError`.
    if args.optimizer == "tssbo":
        # Layer 1 pre-check: the adapter will refuse without the env
        # var, but the CLI fails earlier with a clearer message so the
        # user doesn't have to read the adapter exception chain.
        if os.environ.get(TSSBO_OPT_IN_ENV, "").strip().lower() not in (
            "1", "true", "yes", "on",
        ):
            sys.stderr.write(
                f"error: --optimizer tssbo requires "
                f"${TSSBO_OPT_IN_ENV}=1 in the environment; both "
                f"the CLI flag AND the env var must be set before "
                f"the real tSS-BO library is dispatched.\n"
            )
            return 2
        # Layer 4 pre-check: refuse if the loop's max_iters exceeds
        # the BO iter budget. The adapter would catch this on the
        # first iteration anyway; failing early keeps the error
        # message close to the configuration error.
        if args.max_iters > args.max_bo_iters:
            sys.stderr.write(
                f"error: --max-iters {args.max_iters} exceeds "
                f"--max-bo-iters {args.max_bo_iters}; raise the BO "
                f"budget cap or reduce the iteration count.\n"
            )
            return 2
        budget = BoBudgetTracker(
            max_bo_iters=args.max_bo_iters,
            max_wall_clock_sec=args.max_wall_clock_sec,
            max_ngspice_calls=args.max_ngspice_calls,
            max_bo_proposals=args.max_bo_proposals,
            label=f"tssbo:{args.cell}",
        )
        try:
            optimizer = build_tssbo_adapter(
                bounds=bounds,
                seed=args.seed,
                budget=budget,
            )
        except RuntimeError as exc:
            # Covers layer 3 (package not installed) and any other
            # adapter-construction-time refusal. Exit 2 is the CLI
            # config-error convention; the user must install the
            # package or pick a stub.
            sys.stderr.write(f"error: {exc}\n")
            return 2
        except BudgetExceededError as exc:
            sys.stderr.write(f"error: budget refused tssbo: {exc}\n")
            return 2
    else:
        optimizer = build_stub(
            args.optimizer,
            bounds=bounds,
            # For stub-noop, use the midpoint of each bound as the fixed
            # sizing; that lets --optimizer stub-noop --dry-run produce
            # a sane single-row table.
            fixed_sizing={b.knob: 0.5 * (b.low + b.high) for b in bounds},
            seed=args.seed,
        )

    loop = OptimizerLoop(
        cell_builder=cell_builder,
        testbench_builder=testbench_builder,
        simulator=simulator,
        targets=targets,
        work_dir=Path(args.work_dir),
    )
    try:
        history = loop.run(optimizer, max_iters=args.max_iters)
    except BudgetExceededError as exc:
        sys.stderr.write(f"error: BO budget exceeded mid-run: {exc}\n")
        return 2

    print(_render_history_table(history, targets))
    print()
    print(
        f"completed {len(history)} iters, "
        f"{len(history.feasible_entries())} feasible"
    )

    # Exit code policy: 0 iff at least one iteration's reward ≥ 0
    # (target met everywhere). Otherwise 1.
    best = history.best_entry()
    if best is not None and best.is_feasible and best.reward >= 0:
        return 0
    return 1


# Support `python -m tools.clay_optimize` (loads __main__.py below).
if __name__ == "__main__":
    raise SystemExit(main())
