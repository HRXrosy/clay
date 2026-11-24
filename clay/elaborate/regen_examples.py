"""Regenerate the clean-room desugared example cores FROM real EDALLM sources.

The desugared cores under <clean-room>/examples/desugared/ must be reproducible
from a checked-in EDALLM source, not hand-authored. This binds each emitted core
to a real cases/ design.clay so the desugaring stays regression-guarded.

Run:  python -m clay.elaborate.regen_examples [--check]
  (no args) regenerates the cores; --check fails if any source elaborates to an
  empty/trivial core (guards against a silent desugaring regression).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLAY_CORE_DIR = Path("~/code/clay/examples/desugared")

#: output core name -> real EDALLM source (diverse topologies).
SOURCES = {
    "folded_cascode_ota": "cases/opamp/folded_cascode_ota/source/design.clay",
    "two_stage_signoff": "cases/layout_intent/two_stage_signoff_gf180/source/design.clay",
    "comparator_strongarm": "cases/layout_intent/comparator_strongarm_gf180/source/design.clay",
    "current_mirror": "cases/current_mirror/gf180_postlayout/source/design.clay",
    "bandgap_signoff": "cases/layout_intent/bandgap_signoff_gf180/source/design.clay",
}


def elaborate_source(src_path: Path) -> tuple[str, int, int]:
    """Elaborate one real source to core text. Returns (core, n_leaf, n_constraint)."""
    from clay.parser import parse_clay_file
    from clay.compiler import compile_design
    from clay.elaborate.to_core import emit_cell, DeferredRecord

    cf = parse_clay_file(str(src_path))
    cells = getattr(cf, "cells", None) or getattr(cf, "cell_defs", None) or []
    cd = cells[0]
    cell = compile_design(cd)
    cell = cell[0] if isinstance(cell, tuple) else cell
    deferred = DeferredRecord()
    text = emit_cell(cell, deferred=deferred)
    n_leaf = text.count("inst leaf")
    n_con = text.count("constraint ")
    return text, n_leaf, n_con


def main() -> int:
    check = "--check" in sys.argv
    CLAY_CORE_DIR.mkdir(parents=True, exist_ok=True)
    bad = []
    for name, rel in SOURCES.items():
        src = REPO / rel
        if not src.exists():
            print(f"SKIP {name}: source missing {rel}")
            continue
        try:
            text, n_leaf, n_con = elaborate_source(src)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            bad.append(name)
            continue
        if n_leaf < 1:
            print(f"FAIL {name}: core has no leaf instances (trivial/stub)")
            bad.append(name)
            continue
        out = CLAY_CORE_DIR / f"{name}.clay"
        if not check:
            out.write_text(text, encoding="utf-8")
        print(f"{'CHECK' if check else 'WROTE'} {name}: {n_leaf} inst leaf, {n_con} constraint  <- {rel}")
    if bad:
        print(f"\n{len(bad)} source(s) failed to elaborate: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
