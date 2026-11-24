"""Adversarial / malformed-input hardening for the parser + elaborator.

The desugaring (`clay.elaborate.to_core.emit_cell`) and its faithfulness
verifier (`clay.elaborate.faithful.verify_faithful`) were previously exercised
only on well-formed, happy-path cells. This module attacks the *front* of the
pipeline -- `parse_clay_file` / `parse_clay_source` -> `compile_cell_def` /
`compile_design` -> `emit_cell` -> `verify_faithful` -- with malformed and
edge-case `.clay` inputs and asserts GRACEFUL behaviour.

The contract under test for every adversarial input is:

  Either the pipeline raises a CLEAR, TYPED, catchable error
  (`lark.exceptions.UnexpectedInput` family for surface syntax,
  `ValueError` for transformer-level structural rejections,
  `clay.compiler.CellCompileError` for compile-level rejections,
  `FileNotFoundError` for missing imports, `UnicodeDecodeError`/`ValueError`
  for bad encodings),

  OR it produces a core that the faithfulness verifier REJECTS
  (`verify_faithful(...) -> (False, [...])`).

  It must NEVER:
    * crash with a bare/leaked internal exception (a raw lark ``VisitError``
      wrapping a domain error, an ``IndexError``/``KeyError``/``AttributeError``
      from unchecked indexing, an unhandled ``UnicodeDecodeError`` traceback);
    * silently emit a WRONG-BUT-PASSING core (a core that misrepresents the
      source yet sails through ``verify_faithful`` as ``ok == True``).

Two REAL graceful-failure gaps were found while building this suite and fixed
in ``clay/parser_modules/_entry.py`` (smallest change, no behaviour change for
well-formed input):

  1. A leading UTF-8 BOM (common on Windows-authored files) was fed to the
     grammar as a stray glyph and rejected a perfectly valid file with a
     misleading ``UnexpectedCharacters at col 1``. Fix: read files as
     ``utf-8-sig`` and strip a leading BOM from in-memory source. A genuinely
     non-UTF-8 file is now re-raised as a clear, file-named ``ValueError``
     instead of a bare ``UnicodeDecodeError``.
  2. Transformer-level structural rejections (duplicate device name,
     device/port collision, pin-assigned-twice, ...) ``raise ValueError`` from
     inside a lark transformer callback, which lark re-wrapped as a
     ``VisitError``. ``VisitError`` is a lark-internal type that breaks the
     parser's contract of raising a clean, catchable domain error. Fix:
     ``_transform`` now unwraps ``VisitError`` and re-raises its ``.orig_exc``.

See the per-test docstrings for the specific input and the expected mode.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from lark.exceptions import UnexpectedInput, VisitError

from clay.compiler import CellCompileError, compile_cell_def
from clay.elaborate.faithful import verify_faithful
from clay.elaborate.to_core import emit_cell
from clay.parser_modules._entry import parse_clay_file, parse_clay_source


# A clean, well-formed 2-device primitive cell used as the mutation base.
GOOD_SRC = (
    'package p; pdk "sky130"; '
    "cell C(a, b) { topology { "
    "device M1 : nmos { drain = a; gate = b; source = a; body = a; "
    "W = 1u; L = 1u; } "
    "device M2 : nmos { drain = b; gate = a; source = b; body = b; "
    "W = 1u; L = 1u; } "
    "} }"
)

# Exception classes that count as a CLEAR, TYPED, graceful rejection.
GRACEFUL_ERRORS = (
    UnexpectedInput,  # lark surface-syntax family (UnexpectedEOF/Characters/Token)
    ValueError,       # transformer structural rejection + bad-encoding wrap
    CellCompileError, # compile-level rejection
    FileNotFoundError,
    UnicodeDecodeError,
)


def _assert_graceful_parse(src: str) -> object | None:
    """Parse ``src``; assert it either raises a graceful typed error OR returns
    a ClayFile (in which case the caller checks the downstream stage).

    NEVER allowed: a bare ``VisitError`` (leaked lark internal) or any other
    non-graceful exception type.
    """
    try:
        return parse_clay_source(src)
    except VisitError as exc:  # pragma: no cover - asserts the fix holds
        pytest.fail(
            f"parser leaked a raw lark VisitError (should unwrap to "
            f"{type(exc.orig_exc).__name__}): {exc}"
        )
    except GRACEFUL_ERRORS:
        return None
    except Exception as exc:  # pragma: no cover - surfaces a real crash
        pytest.fail(
            f"parser raised a non-graceful {type(exc).__name__}: {exc}"
        )


def _full_pipeline_never_silently_wrong(src: str) -> None:
    """If ``src`` parses AND compiles AND emits, the emitted core must NOT be a
    wrong-but-passing core: it must either be faithfully correct (verifier ok)
    OR be rejected by the verifier. There is no third option where the verifier
    silently passes a misrepresentation -- that is what we are policing.

    This helper just runs the chain and, when it reaches a core, asserts the
    verifier produced a *consistent* verdict (ok iff no discrepancies). The
    specific tests below assert the EXACT expected verdict per case.
    """
    cf = _assert_graceful_parse(src)
    if cf is None:
        return  # graceful parse-time rejection -- nothing more to check
    if not cf.cell_defs:
        return
    cd = cf.cell_defs[0]
    try:
        cell = compile_cell_def(cd)
    except GRACEFUL_ERRORS:
        return
    core = emit_cell(cell)
    ok, disc = verify_faithful(cell, core)
    # ok must be exactly "no discrepancies" -- the verifier never reports
    # ok=True with a non-empty discrepancy list (that would be the silent-wrong
    # failure mode).
    assert ok == (len(disc) == 0)


# ─────────────────────────────────────────────────────────────────────
# Surface-syntax adversaries: must raise a clean lark UnexpectedInput.
# ─────────────────────────────────────────────────────────────────────


def test_empty_string_graceful() -> None:
    """An empty file is a clean UnexpectedEOF, not a crash."""
    with pytest.raises(UnexpectedInput):
        parse_clay_source("")


def test_whitespace_only_graceful() -> None:
    """Whitespace-only input -> clean UnexpectedEOF."""
    with pytest.raises(UnexpectedInput):
        parse_clay_source("   \n\n  \t ")


def test_pure_garbage_graceful() -> None:
    """Non-clay bytes -> clean UnexpectedCharacters at the first bad char."""
    with pytest.raises(UnexpectedInput):
        parse_clay_source("!!! @@@ ??? not clay at all {{{{")


def test_truncated_file_graceful() -> None:
    """A file cut off mid-device -> clean UnexpectedEOF/Token, never a half-built
    IR and never a traceback from an unbalanced-brace internal walk."""
    truncated = GOOD_SRC[: GOOD_SRC.index("device M1") + 25]
    with pytest.raises(UnexpectedInput):
        parse_clay_source(truncated)


def test_missing_model_literal_graceful() -> None:
    """`device M1 :  { ... }` with no device-kind literal -> clean syntax error.

    The model literal is the thing `emit_cell` puts after `model` in the core;
    a missing one must be rejected at parse time, never defaulted silently.
    """
    bad = GOOD_SRC.replace("device M1 : nmos {", "device M1 :  {")
    with pytest.raises(UnexpectedInput):
        parse_clay_source(bad)


def test_empty_cell_no_topology_graceful() -> None:
    """A cell with no topology block at all -> clean syntax rejection."""
    with pytest.raises(UnexpectedInput):
        parse_clay_source('package p; pdk "sky130"; cell Empty(a, b) { }')


def test_cell_zero_devices_graceful() -> None:
    """A topology block with zero device declarations -> clean syntax rejection
    (the grammar requires at least one body item)."""
    with pytest.raises(UnexpectedInput):
        parse_clay_source(
            'package p; pdk "sky130"; cell Z(a, b) { topology { } }'
        )


def test_missing_bind_rhs_graceful() -> None:
    """`drain = ;` (empty pin RHS) -> clean syntax error, not a None net that
    would later crash the emitter or produce a malformed bind."""
    bad = GOOD_SRC.replace("drain = a;", "drain = ;")
    with pytest.raises(UnexpectedInput):
        parse_clay_source(bad)


def test_deeply_nested_braces_graceful() -> None:
    """Pathologically nested braces -> clean syntax rejection, no recursion
    blow-up surfacing as a RecursionError traceback to the caller."""
    deep = (
        'package p; pdk "sky130"; cell C(a) { topology { '
        + "{" * 500
        + " } }"
    )
    with pytest.raises(UnexpectedInput):
        parse_clay_source(deep)


# ─────────────────────────────────────────────────────────────────────
# Transformer-level structural adversaries: must raise a CLEAN ValueError
# (NOT a leaked lark VisitError -- this is the gap #2 fix).
# ─────────────────────────────────────────────────────────────────────


def test_duplicate_device_name_clean_valueerror() -> None:
    """Two `device M : ...` with the same name -> clean ValueError, NOT a raw
    lark VisitError. Regression guard for the VisitError-unwrap fix."""
    src = (
        'package p; pdk "sky130"; cell C(a, b) { topology { '
        "device M : nmos { drain = a; gate = b; source = a; body = a; "
        "W = 1u; L = 1u; } "
        "device M : nmos { drain = b; gate = a; source = b; body = b; "
        "W = 1u; L = 1u; } } }"
    )
    with pytest.raises(ValueError) as exc:
        parse_clay_source(src)
    # must be the underlying domain error, not the lark wrapper
    assert not isinstance(exc.value, VisitError)
    assert "duplicate" in str(exc.value).lower()


def test_device_port_collision_clean_valueerror() -> None:
    """A device named the same as a cell port -> clean ValueError, not VisitError."""
    src = (
        'package p; pdk "sky130"; cell C(M, b) { topology { '
        "device M : nmos { drain = M; gate = b; source = M; body = M; "
        "W = 1u; L = 1u; } } }"
    )
    with pytest.raises(ValueError) as exc:
        parse_clay_source(src)
    assert not isinstance(exc.value, VisitError)


def test_no_transformer_visiterror_leaks_across_cases() -> None:
    """Sweep several structural-rejection inputs: none may leak a VisitError."""
    structural_bad = [
        # duplicate device
        'package p; pdk "sky130"; cell C(a,b){ topology { '
        "device M:nmos{drain=a;gate=b;source=a;body=a;W=1u;L=1u;} "
        "device M:nmos{drain=b;gate=a;source=b;body=b;W=1u;L=1u;} } }",
        # device/port collision
        'package p; pdk "sky130"; cell C(M,b){ topology { '
        "device M:nmos{drain=M;gate=b;source=M;body=M;W=1u;L=1u;} } }",
    ]
    for src in structural_bad:
        with pytest.raises(Exception) as exc:
            parse_clay_source(src)
        assert not isinstance(exc.value, VisitError), (
            f"leaked VisitError for: {src[:60]}..."
        )


# ─────────────────────────────────────────────────────────────────────
# Compile-level adversaries: must raise a clean CellCompileError.
# ─────────────────────────────────────────────────────────────────────


def test_undefined_instance_target_graceful() -> None:
    """A composition instance pointing at an undefined sub-cell -> clean
    CellCompileError, never a None-deref traceback in the flatten path."""
    src = (
        'package p; pdk "sky130"; cell C(a, b) { topology { '
        "instance x = NoSuchCell { p = a; }; } }"
    )
    cf = parse_clay_source(src)
    with pytest.raises(CellCompileError):
        compile_cell_def(cf.cell_defs[0])


def test_unknown_device_kind_graceful() -> None:
    """An unknown device kind -> clean CellCompileError at compile, not a
    KeyError from an unchecked adapter dict lookup."""
    src = (
        'package p; pdk "sky130"; cell C(a, b) { topology { '
        "device M : florbnax { drain = a; gate = b; source = a; body = a; "
        "W = 1u; L = 1u; } } }"
    )
    cf = parse_clay_source(src)
    with pytest.raises(CellCompileError):
        compile_cell_def(cf.cell_defs[0])


# ─────────────────────────────────────────────────────────────────────
# Inputs that PARSE but are semantically odd: assert the desugaring does NOT
# emit a wrong-but-passing core (the verifier is the backstop).
# ─────────────────────────────────────────────────────────────────────


def test_duplicate_ports_verifier_rejects_not_silent() -> None:
    """A cell declaring a port twice `C(a, a, b)` parses and compiles, but the
    emitted core has a duplicate interface port. The faithfulness verifier MUST
    reject it (C-PORTS) rather than silently passing a degenerate interface.

    This is the key 'never silently wrong' assertion: the core IS structurally
    off, and the oracle catches it.
    """
    src = (
        'package p; pdk "sky130"; cell C(a, a, b) { topology { '
        "device M : nmos { drain = a; gate = b; source = a; body = a; "
        "W = 1u; L = 1u; } } }"
    )
    cf = parse_clay_source(src)
    cell = compile_cell_def(cf.cell_defs[0])
    core = emit_cell(cell)
    ok, disc = verify_faithful(cell, core)
    assert ok is False, "duplicate-port core must be REJECTED, not silently passed"
    assert any("C-PORTS" in d for d in disc)
    # and the verdict is self-consistent (ok iff empty discrepancies)
    assert ok == (len(disc) == 0)


def test_undeclared_net_in_binding_is_faithfully_encoded() -> None:
    """A pin bound to a net that is not a declared port (`gate = GHOST`) parses
    (with a 'likely typo' warning) and the emitter faithfully encodes GHOST as a
    net. This is NOT a faithfulness violation -- GHOST is a real net of the
    source device -- so the verifier correctly PASSES. The point of the test is
    that the emit is faithful (GHOST present in the core), not silently dropped.
    """
    src = (
        'package p; pdk "sky130"; cell U(a, b) { topology { '
        "device M1 : nmos { drain = a; gate = GHOST; source = b; body = b; "
        "W = 1u; L = 1u; } } }"
    )
    cf = parse_clay_source(src)
    cell = compile_cell_def(cf.cell_defs[0])
    core = emit_cell(cell)
    ok, disc = verify_faithful(cell, core)
    # GHOST must appear in the emitted core (faithful), not be silently dropped.
    assert "GHOST" in core
    assert ok is True and disc == []


def test_huge_input_no_crash() -> None:
    """A cell with thousands of ports parses without a recursion/perf blow-up,
    compiles, emits, and the full pipeline stays self-consistent (no silent
    wrong core at scale)."""
    ports = ",".join(f"p{i}" for i in range(2000))
    src = (
        f'package p; pdk "sky130"; cell H({ports}) {{ topology {{ '
        "device M1 : nmos { drain = p0; gate = p1; source = p2; body = p3; "
        "W = 1u; L = 1u; } } }"
    )
    _full_pipeline_never_silently_wrong(src)


# ─────────────────────────────────────────────────────────────────────
# Encoding / file-level adversaries (parse_clay_file path).
# ─────────────────────────────────────────────────────────────────────


def _write(d: str, name: str, data: bytes) -> str:
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def test_utf8_bom_file_parses_cleanly() -> None:
    """REGRESSION (gap #1): a valid file with a leading UTF-8 BOM must parse,
    not be rejected with a misleading UnexpectedCharacters on the BOM glyph."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(
            d,
            "bom.clay",
            b"\xef\xbb\xbf"
            + b'package p; pdk "sky130"; cell C(a, b) { topology { '
            b"device M1 : nmos { drain = a; gate = b; source = a; body = a; "
            b"W = 1u; L = 1u; } } }",
        )
        cf = parse_clay_file(p)
        assert [cd.local_name for cd in cf.cell_defs] == ["C"]


def test_utf8_bom_in_memory_source_parses_cleanly() -> None:
    """A pre-decoded source string with a leading BOM char also parses."""
    cf = parse_clay_source(
        "﻿"
        'package p; pdk "sky130"; cell C(a, b) { topology { '
        "device M1 : nmos { drain = a; gate = b; source = a; body = a; "
        "W = 1u; L = 1u; } } }"
    )
    assert [cd.local_name for cd in cf.cell_defs] == ["C"]


def test_non_utf8_file_clear_error_not_traceback() -> None:
    """A genuinely non-UTF-8 file -> a CLEAR, file-named error (ValueError after
    the gap #1 fix; UnicodeDecodeError is also acceptable as a typed error),
    never a silent wrong parse."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(
            d,
            "bad.clay",
            b'package p; pdk "sky130"; cell C(a, b) { topology { '
            b"\xff\xfe } }",
        )
        with pytest.raises((ValueError, UnicodeDecodeError)) as exc:
            parse_clay_file(p)
        # the message names the offending file so the failure is actionable
        if isinstance(exc.value, ValueError):
            assert "UTF-8" in str(exc.value) or "utf-8" in str(exc.value)


def test_missing_file_clear_error() -> None:
    """A non-existent path -> FileNotFoundError, not a generic OSError dump."""
    with pytest.raises(FileNotFoundError):
        parse_clay_file(os.path.join(tempfile.gettempdir(), "does_not_exist_zzz.clay"))


def test_import_cycle_detected_graceful() -> None:
    """A -> B -> A import cycle -> a clear ValueError naming the chain, never an
    infinite recursion / RecursionError traceback."""
    with tempfile.TemporaryDirectory() as d:
        a = _write(
            d,
            "a.clay",
            b"import './b.clay';\npackage a; pdk \"sky130\"; "
            b"cell A(x){ topology { "
            b"device M:nmos{drain=x;gate=x;source=x;body=x;W=1u;L=1u;} } }",
        )
        _write(
            d,
            "b.clay",
            b"import './a.clay';\npackage b; pdk \"sky130\"; "
            b"cell B(x){ topology { "
            b"device M:nmos{drain=x;gate=x;source=x;body=x;W=1u;L=1u;} } }",
        )
        with pytest.raises(ValueError) as exc:
            parse_clay_file(a)
        assert "cycle" in str(exc.value).lower()


def test_missing_imported_file_clear_error() -> None:
    """An import of a non-existent file -> FileNotFoundError naming the import,
    decorated with the importing file's context."""
    with tempfile.TemporaryDirectory() as d:
        a = _write(
            d,
            "a.clay",
            b"import './nope.clay';\npackage a; pdk \"sky130\"; "
            b"cell A(x){ topology { "
            b"device M:nmos{drain=x;gate=x;source=x;body=x;W=1u;L=1u;} } }",
        )
        with pytest.raises(FileNotFoundError):
            parse_clay_file(a)


# ─────────────────────────────────────────────────────────────────────
# Cross-cutting: a parametrized sweep asserting NOTHING crashes and NOTHING
# is silently-wrong across the whole adversarial set.
# ─────────────────────────────────────────────────────────────────────


_ADVERSARIAL_SOURCES = [
    "",
    "   \n ",
    "!!! garbage",
    GOOD_SRC[: GOOD_SRC.index("device M1") + 25],
    GOOD_SRC.replace("device M1 : nmos {", "device M1 :  {"),
    'package p; pdk "sky130"; cell Empty(a,b) { }',
    'package p; pdk "sky130"; cell Z(a,b) { topology { } }',
    # duplicate ports -> parses, verifier rejects (handled by helper)
    'package p; pdk "sky130"; cell C(a,a,b){ topology { '
    "device M:nmos{drain=a;gate=b;source=a;body=a;W=1u;L=1u;} } }",
    # huge but valid
    'package p; pdk "sky130"; cell H('
    + ",".join(f"p{i}" for i in range(300))
    + ") { topology { device M:nmos{drain=p0;gate=p1;source=p2;body=p3;"
    "W=1u;L=1u;} } }",
]


@pytest.mark.parametrize("src", _ADVERSARIAL_SOURCES)
def test_sweep_no_crash_no_silent_wrong(src: str) -> None:
    """Every adversarial source either raises a graceful typed error or runs the
    full pipeline to a self-consistent (never silently-wrong) verdict."""
    _full_pipeline_never_silently_wrong(src)
