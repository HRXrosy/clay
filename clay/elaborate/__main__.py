"""CLI shim: desugar an EDALLM .clay design to clean-room core text.

Usage:
    python -m clay.elaborate <design.clay> [cell_name]

Parses the EDALLM source, compiles the selected CellDef to a clay.ir.Cell
(devices populated), and prints the core .clay text to stdout. Pipe into the
clean-room kernel:

    python -m clay.elaborate design.clay MyAmp | \
        (cd /c/Users/Administrator/code/clay && lake exe clay check /dev/stdin)

DEFERRED constructs (flow/quality/policy/verify/corners/temps/fn/topology.style)
are skipped and logged to stderr, never coerced into the structural emit.
"""

from __future__ import annotations

import sys

from .to_core import emit_cell, DeferredRecord


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        _eprint("usage: python -m clay.elaborate <design.clay> [cell_name]")
        return 2
    path = argv[1]
    want = argv[2] if len(argv) > 2 else None

    from clay.parser import parse_clay_file
    from clay.compiler import compile_design

    cf = parse_clay_file(path)
    cell_defs = list(getattr(cf, "cell_defs", []) or [])
    if not cell_defs:
        _eprint(f"no cell_defs found in {path}")
        return 1

    chosen = None
    if want is not None:
        for cd in cell_defs:
            nm = getattr(cd, "name", None) or getattr(cd, "local_name", None)
            if nm == want:
                chosen = cd
                break
        if chosen is None:
            names = [getattr(cd, "name", getattr(cd, "local_name", "?")) for cd in cell_defs]
            _eprint(f"cell '{want}' not found; available: {names}")
            return 1
    else:
        chosen = cell_defs[0]

    cell = compile_design(chosen)

    deferred = DeferredRecord()
    text = emit_cell(cell, deferred=deferred)
    sys.stdout.write(text)

    if deferred.notes:
        _eprint("# deferred (out of structural core):")
        for n in deferred.notes:
            _eprint(f"#   - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
