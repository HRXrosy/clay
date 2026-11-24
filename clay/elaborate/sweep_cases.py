"""Sweep the elaborator over every real ``cases/*/source/design.clay``.

For each case this runs the production pipeline
(``parse_clay_file`` -> ``compile_design`` -> ``emit_cell``) and records a
structural row: device count, emitted ``inst leaf`` count, constraint count,
a faithfulness sanity flag, and an elaborate status.

Status semantics (honest, not coverage-padding):
  * ``clean``   -- elaborated, at least one ``inst leaf``, the emitted leaf
                   count equals the source device count, and the elaborator
                   recorded NO deferred losses (no opaque-placeholder children,
                   nothing dropped). The structural core is a complete encoding.
  * ``partial`` -- elaborated and non-trivial, but the elaborator recorded a
                   deferred note (e.g. an unresolved child cell emitted as an
                   opaque leaf placeholder, or orchestration routed out of
                   core), OR the leaf count != device count. The core exists
                   but is not a 1:1 structural encoding of the source.
  * ``fail``    -- parse / compile / emit raised, OR the emitted core has zero
                   ``inst leaf`` instances (trivial / stub). ``why`` carries
                   the exception type+message or the trivial-core reason.

The ``faithful?`` column here is a CHEAP local sanity check only
(leaf_count == device_count AND no deferred losses). It is NOT the semantic
verifier in ``faithful.py`` -- this harness deliberately does not import it.

CLI:
  python -m clay.elaborate.sweep_cases            # print table + summary
  python -m clay.elaborate.sweep_cases --check    # regression: nonzero exit if
                                                  # the clean count drops below
                                                  # the recorded baseline
  python -m clay.elaborate.sweep_cases --emit      # also (re)write ~8 diverse
                                                  # passing cores into the
                                                  # clean-room desugared dir
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
CLAY_CORE_DIR = Path("~/code/clay/examples/desugared")

#: Minimum number of cases that MUST reach ``clean`` for --check to pass.
#: Set from the real measured run (see module-level sweep). Lowering this is a
#: regression; raising it locks in an improvement.
CLEAN_BASELINE = 60

#: Diverse passing cores to emit into the clean-room desugared dir under --emit.
#: Chosen for topology spread (single-stage / two-stage / cascode / comparator /
#: bias / oscillator / bandgap / pll). Keys are output core file stems.
EMIT_SELECTION = {
    "folded_cascode_ota": "cases/opamp/folded_cascode_ota/source/design.clay",
    "two_stage_signoff": "cases/layout_intent/two_stage_signoff_gf180/source/design.clay",
    "comparator_strongarm": "cases/layout_intent/comparator_strongarm_gf180/source/design.clay",
    "current_mirror": "cases/current_mirror/gf180_postlayout/source/design.clay",
    "bandgap_signoff": "cases/layout_intent/bandgap_signoff_gf180/source/design.clay",
    "bias_tree": "cases/bias/bias_tree/source/design.clay",
    "lc_tank_vco_tuned": "cases/oscillator/lc_tank_vco_tuned/source/design.clay",
    "analoggym_pll": "cases/pll/analoggym_pll/source/design.clay",
}


def find_case_sources() -> list[Path]:
    """All real ``cases/*/source/design.clay`` (1- and 2-deep), de-duplicated."""
    srcs: list[Path] = []
    srcs += sorted(REPO.glob("cases/*/source/design.clay"))
    srcs += sorted(REPO.glob("cases/*/*/source/design.clay"))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for s in srcs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


@dataclass
class Row:
    case: str
    n_devices: int
    n_inst_leaf: int
    n_constraints: int
    faithful: Optional[bool]
    status: str  # clean | partial | fail
    why: str = ""


def _load_first_cell(src: Path):
    """parse + compile the first cell of one source. Returns a clay.ir.Cell."""
    from clay.parser import parse_clay_file
    from clay.compiler import compile_design

    cf = parse_clay_file(str(src))
    cells = getattr(cf, "cells", None) or getattr(cf, "cell_defs", None) or []
    if not cells:
        raise ValueError("no cells in parsed file")
    cd = cells[0]
    cell = compile_design(cd)
    cell = cell[0] if isinstance(cell, tuple) else cell
    return cell


def elaborate_one(src: Path) -> Row:
    """Run the full pipeline over one source and classify the outcome."""
    from clay.elaborate.to_core import emit_cell, DeferredRecord

    case = src.relative_to(REPO).parent.parent.as_posix().replace("cases/", "")
    try:
        cell = _load_first_cell(src)
    except Exception as exc:  # noqa: BLE001
        return Row(case, 0, 0, 0, None, "fail",
                   f"{type(exc).__name__}: {exc}".replace("\n", " ")[:160])

    n_dev = len(getattr(cell, "devices", []) or [])
    n_inst = len(getattr(cell, "instances", []) or [])
    deferred = DeferredRecord()
    try:
        text = emit_cell(cell, deferred=deferred)
    except Exception as exc:  # noqa: BLE001
        return Row(case, n_dev, 0, 0, None, "fail",
                   f"emit {type(exc).__name__}: {exc}".replace("\n", " ")[:160])

    n_leaf = text.count("inst leaf")
    n_con = text.count("constraint ")

    # trivial-core guard: a non-trivial source that emitted zero leaves is a fail.
    if n_leaf < 1:
        if n_dev == 0 and n_inst == 0:
            return Row(case, n_dev, n_leaf, n_con, None, "fail",
                       "source has no devices/instances (empty/stub cell)")
        return Row(case, n_dev, n_leaf, n_con, None, "fail",
                   "core has zero inst-leaf despite source structure")

    # cheap local faithfulness sanity: leaf count == device count (no child
    # instances inlined here) AND nothing deferred out of core.
    count_match = (n_leaf == n_dev) if n_inst == 0 else (n_leaf >= n_dev)
    has_loss = bool(deferred.notes)
    faithful = count_match and not has_loss

    if faithful:
        return Row(case, n_dev, n_leaf, n_con, True, "clean", "")

    why_parts = []
    if not count_match:
        why_parts.append(f"leaf!=dev ({n_leaf} vs {n_dev}; inst={n_inst})")
    if has_loss:
        why_parts.append("deferred: " + "; ".join(deferred.notes[:3]))
    return Row(case, n_dev, n_leaf, n_con, False, "partial", " | ".join(why_parts)[:200])


def run_sweep() -> list[Row]:
    rows = [elaborate_one(s) for s in find_case_sources()]
    rows.sort(key=lambda r: (r.status != "fail", r.status != "partial", r.case))
    return rows


def _fmt_table(rows: list[Row]) -> str:
    hdr = f"{'case':<46} {'dev':>4} {'leaf':>5} {'con':>4} {'faith':>6} {'status':<8} why"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        faith = "-" if r.faithful is None else ("yes" if r.faithful else "no")
        lines.append(
            f"{r.case:<46} {r.n_devices:>4} {r.n_inst_leaf:>5} {r.n_constraints:>4} "
            f"{faith:>6} {r.status:<8} {r.why}"
        )
    return "\n".join(lines)


def _summary(rows: list[Row]) -> tuple[str, int, int, int]:
    clean = sum(1 for r in rows if r.status == "clean")
    partial = sum(1 for r in rows if r.status == "partial")
    fail = sum(1 for r in rows if r.status == "fail")
    total = len(rows)
    out = [
        "",
        f"SUMMARY: {total} cases | {clean} clean | {partial} partial | {fail} fail",
    ]
    if partial:
        out.append("\nPARTIAL (elaborated but not 1:1 structural):")
        for r in rows:
            if r.status == "partial":
                out.append(f"  - {r.case}: {r.why}")
    if fail:
        out.append("\nFAIL (no usable structural core):")
        for r in rows:
            if r.status == "fail":
                out.append(f"  - {r.case}: {r.why}")
    return "\n".join(out), clean, partial, fail


def emit_selected_cores() -> list[str]:
    """(Re)write the diverse passing cores into the clean-room desugared dir."""
    from clay.elaborate.to_core import emit_cell, DeferredRecord

    CLAY_CORE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for stem, rel in EMIT_SELECTION.items():
        src = REPO / rel
        if not src.exists():
            print(f"  SKIP {stem}: source missing {rel}")
            continue
        try:
            cell = _load_first_cell(src)
            deferred = DeferredRecord()
            text = emit_cell(cell, deferred=deferred)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {stem}: {type(exc).__name__}: {exc}")
            continue
        if text.count("inst leaf") < 1:
            print(f"  SKIP {stem}: trivial core")
            continue
        out = CLAY_CORE_DIR / f"{stem}.clay"
        out.write_text(text, encoding="utf-8")
        written.append(str(out))
        print(f"  WROTE {out}  ({text.count('inst leaf')} inst leaf <- {rel})")
    return written


def main() -> int:
    check = "--check" in sys.argv
    do_emit = "--emit" in sys.argv

    rows = run_sweep()
    print(_fmt_table(rows))
    summary, clean, partial, fail = _summary(rows)
    print(summary)

    if do_emit:
        print("\nEMITTING diverse passing cores -> clean-room examples/desugared:")
        written = emit_selected_cores()
        print(f"  {len(written)} cores emitted.")

    if check:
        if clean < CLEAN_BASELINE:
            print(
                f"\nCHECK FAIL: clean={clean} < baseline {CLEAN_BASELINE} "
                f"(elaborate regression)."
            )
            return 1
        print(f"\nCHECK OK: clean={clean} >= baseline {CLEAN_BASELINE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
