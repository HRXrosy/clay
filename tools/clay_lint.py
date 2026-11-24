"""
Static analyzer for .clay files.

`tools/clay_lint.py` runs over a .clay file BEFORE ngspice is invoked
and surfaces common LLM authoring mistakes (missing units, dangling
nets, out-of-range sizing intent, etc.). The goal is to shorten the
LLM iteration loop: instead of "write -> compile -> simulate -> fail
with cryptic ngspice error", the linter catches a structural issue in
< 0.1 s with a precise file:line:col diagnostic.

Output format
-------------
Each finding is one line:

    file.clay:line:col: severity [rule-id] message

Severity is one of `error` / `warning` / `info`. `error` causes a
non-zero exit when `--severity error` (the default) is in effect.

Rule catalogue
--------------
The CHECKS list at the bottom of the file is the canonical registry.
Each entry is `(rule_id, callable)` where the callable takes
`(file_path, ir, source_text)` and returns a list of Finding objects.

Rule IDs are stable identifiers (E001+, W001+, I001+) — once a rule is
shipped, its ID does not change so users can suppress it predictably.

CLI
---
    python tools/clay_lint.py FILE.clay              # all findings, exit 1 on error
    python tools/clay_lint.py FILE.clay --quiet      # only print errors
    python tools/clay_lint.py FILE.clay --severity error    # default
    python tools/clay_lint.py FILE.clay --severity warning  # fail on warnings too

Exit codes:
  * 0 — no findings at or above the threshold severity
  * 1 — at least one finding at or above the threshold severity
  * 2 — internal failure (parse error, missing file, etc.)

Design notes
------------
The linter operates on the parsed IR (`ClayFile`) PLUS the original
source text. The IR loses positional information (lark strips it
through the Transformer), so to recover `line:col` we scan the source
for the relevant identifier with simple regex.

This is "best effort" — if the same name appears twice in the source
(e.g. once in a spec field, once in a sizing kwarg), the linter picks
the first occurrence in the affected block. That is good enough for an
authoring tool: the user reads the file once and sees the issue near
the right line.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# Make sure we can import clay/ when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clay.ir import (
    CellAlias,
    CellDef,
    ClayFile,
    Device,
    Spec,
    SpecField,
    VerifyJob,
)
from clay.parser import parse_clay_file, parse_clay_source


# ─── Finding data model ─────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One linter diagnostic.

    The position is 1-indexed (line and column both start at 1) to
    match the standard `file:line:col:` format used by gcc / clang /
    pylint / ruff / etc.
    """
    rule_id: str
    severity: str            # 'error' | 'warning' | 'info'
    line: int
    col: int
    message: str

    def format(self, file_path: str) -> str:
        return (
            f"{file_path}:{self.line}:{self.col}: "
            f"{self.severity} [{self.rule_id}] {self.message}"
        )


# Severity ordering for the `--severity` threshold flag. A finding
# fails the lint iff `_SEVERITY_RANK[finding.severity] >= _SEVERITY_RANK[threshold]`.
_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


# ─── Source-position helpers ────────────────────────────────────


def _line_col_of(offset: int, source: str) -> tuple[int, int]:
    """Convert a char offset in `source` to a 1-indexed (line, col) pair.

    Linear scan — fine for files in the hundreds of lines. If the
    file ever grows past tens of thousands of lines we can replace
    this with a prebuilt offset->line table built once per file.
    """
    if offset < 0:
        return (1, 1)
    line = 1
    last_nl = -1
    for i, ch in enumerate(source[:offset]):
        if ch == "\n":
            line += 1
            last_nl = i
    col = offset - last_nl
    return (line, col)


def _strip_line_comments(source: str) -> str:
    """Replace every `// ...` line-comment with same-length spaces.

    Preserves char offsets (so `line:col` math stays correct) while
    making the comment text invisible to identifier searches. Lark
    strips comments at the tokenizer layer, so this matches the
    parser's view of the source.
    """
    # Walk the source char-by-char, tracking whether we're inside a
    # `'...'` or `"..."` string literal. A `//` that appears inside a
    # string is part of the string, not a comment. Newlines end the
    # comment region.
    out = []
    i = 0
    in_str: str | None = None
    while i < len(source):
        ch = source[i]
        if in_str is None:
            if ch in ("'", '"'):
                in_str = ch
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < len(source) and source[i + 1] == "/":
                # Replace the rest of this line with spaces.
                j = source.find("\n", i)
                if j < 0:
                    j = len(source)
                out.append(" " * (j - i))
                i = j
                continue
            out.append(ch)
            i += 1
        else:
            out.append(ch)
            if ch == in_str:
                in_str = None
            i += 1
    return "".join(out)


def _find_ident_pos(
    source: str,
    ident: str,
    *,
    near_keyword: str | None = None,
    occurrence: int = 0,
) -> tuple[int, int]:
    """Locate the `occurrence`-th appearance of `ident` as a whole-word
    identifier in `source`. Returns (line, col); falls back to (1, 1)
    if not found.

    Comments are stripped before searching so a mention of the
    identifier inside a `//` line-comment does not steal the first hit
    from the real declaration.

    `near_keyword` narrows the search: if given, only matches that
    follow the first occurrence of that keyword are counted. Useful
    for spec field names (search after `spec`) vs sizing fields
    (after `sizing`).
    """
    code = _strip_line_comments(source)
    start = 0
    if near_keyword:
        kw_pos = code.find(near_keyword)
        if kw_pos >= 0:
            start = kw_pos
    pattern = re.compile(rf"\b{re.escape(ident)}\b")
    seen = 0
    for m in pattern.finditer(code, start):
        if seen == occurrence:
            return _line_col_of(m.start(), source)
        seen += 1
    return (1, 1)


def _find_keyword_pos(source: str, keyword: str) -> tuple[int, int]:
    """Locate the first whole-word `keyword` in `source`. (1,1) on miss."""
    m = re.search(rf"\b{re.escape(keyword)}\b", source)
    if not m:
        return (1, 1)
    return _line_col_of(m.start(), source)


# Regex matching a spec field line, used to recover the trailing unit
# token that the parser discards.
#
# Captures: (1) field name, (2) type id, (3) operator, (4) numeric value,
# (5) unit text (may be empty).
_SPEC_FIELD_RE = re.compile(
    r"\b(?P<name>[A-Za-z_][\w]*)\s*"
    r":\s*"
    r"(?P<dim>[A-Za-z_][\w]*)\s*"
    r"(?P<op>>=|<=|=|>|<)\s*"
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?P<unit>[A-Za-z][A-Za-z/]*)?"
    r"\s*;",
)


@dataclass(frozen=True)
class _SourceSpecField:
    """One spec-field occurrence recovered from the source text."""
    name: str
    dim: str
    op: str
    value_text: str
    unit_text: str            # may be ''
    offset: int               # char offset of the field's name


def _enumerate_source_spec_fields(source: str) -> list[_SourceSpecField]:
    """Walk every `NAME : DIM OP VAL[UNIT];` occurrence in the source.

    The linter uses this to recover the original unit text per spec
    field (the parsed IR loses it — values are stored as plain SI
    floats). Findings keyed on a specific field then carry exact
    position information rather than a "somewhere in the file" miss.

    Comments are stripped first — the spec-field regex is greedy enough
    that a commented-out spec example (e.g. `// gain_dc : real >= 25;`)
    would otherwise match.
    """
    code = _strip_line_comments(source)
    out: list[_SourceSpecField] = []
    for m in _SPEC_FIELD_RE.finditer(code):
        out.append(_SourceSpecField(
            name=m.group("name"),
            dim=m.group("dim"),
            op=m.group("op"),
            value_text=m.group("value"),
            unit_text=m.group("unit") or "",
            offset=m.start("name"),
        ))
    return out


# ─── Spec-field heuristics ──────────────────────────────────────


# Field-name -> expected-dimension heuristic. Keyed on the LOWERCASE
# field name's longest matching suffix/keyword. Used by rules E001
# (missing unit hint) and W010 (wrong dimension type).
#
# `expected_dim` is the canonical dimension; `expected_unit_alts` is
# the set of unit suffixes that mark the spec as "well-formed". The
# tuple `("db", {"dB"})` means "the field name carries 'db'/'gain'
# semantics, so we expect a `dB` suffix on the value (or, if the LLM
# wrote `real` as the type id, at least flag the missing unit)".
_FIELD_NAME_HINTS: list[tuple[str, str, frozenset[str]]] = [
    # Gain: dB unit expected. We match suffix `_db` and the standalone
    # "gain"-keyword fields (gain_dc, dc_gain) which are universally dB.
    ("gain_dc",   "db", frozenset({"dB"})),
    ("gain_db",   "db", frozenset({"dB"})),
    ("dc_gain",   "db", frozenset({"dB"})),
    ("loop_gain", "db", frozenset({"dB"})),
    ("gain",      "db", frozenset({"dB"})),

    # Phase margin: degrees.
    ("phase_margin", "angle", frozenset({"deg"})),
    ("pm",           "angle", frozenset({"deg"})),

    # Bandwidth / frequency.
    ("gbw",        "freq", frozenset({"Hz", "kHz", "MHz", "GHz"})),
    ("ugb",        "freq", frozenset({"Hz", "kHz", "MHz", "GHz"})),
    ("fc",         "freq", frozenset({"Hz", "kHz", "MHz", "GHz"})),
    ("bw",         "freq", frozenset({"Hz", "kHz", "MHz", "GHz"})),
    ("bandwidth",  "freq", frozenset({"Hz", "kHz", "MHz", "GHz"})),

    # Slew rate: V/us is the canonical form (compound unit).
    ("slew_rate",  "compound_vt", frozenset({"V/us", "V/s", "V/ms", "V/ns"})),
    ("slew",       "compound_vt", frozenset({"V/us", "V/s", "V/ms", "V/ns"})),

    # Voltage / supply.
    ("vdd",        "volt", frozenset({"V", "mV", "uV"})),
    ("vss",        "volt", frozenset({"V", "mV", "uV"})),
    ("voffset",    "volt", frozenset({"V", "mV", "uV"})),
    ("offset",     "volt", frozenset({"V", "mV", "uV"})),
    ("vinmin",     "volt", frozenset({"V", "mV", "uV"})),
    ("vinmax",     "volt", frozenset({"V", "mV", "uV"})),
    ("vswing",     "volt", frozenset({"V", "mV", "uV"})),

    # Current.
    ("i_bias",     "current", frozenset({"A", "mA", "uA", "nA", "pA"})),
    ("i_ref",      "current", frozenset({"A", "mA", "uA", "nA", "pA"})),
    ("idd",        "current", frozenset({"A", "mA", "uA", "nA", "pA"})),
    ("power",      "current", frozenset({"A", "mA", "uA", "nA", "pA"})),

    # Capacitance.
    ("cload",      "cap", frozenset({"F", "pF", "fF", "nF", "uF"})),
    ("cl",         "cap", frozenset({"F", "pF", "fF", "nF", "uF"})),
    ("cc",         "cap", frozenset({"F", "pF", "fF", "nF", "uF"})),

    # Noise: dBc/Hz or V/sqrt(Hz) — neither fully supported by the
    # parser yet, so we only flag the dimension type, not the unit.
    ("noise",      "real", frozenset()),
    ("snr",        "db",   frozenset({"dB"})),
    ("psrr",       "db",   frozenset({"dB"})),
    ("cmrr",       "db",   frozenset({"dB"})),
    ("thd",        "db",   frozenset({"dB"})),
]


def _field_name_hint(name: str) -> tuple[str, frozenset[str]] | None:
    """Return (expected_dimension, valid_units) for a field name if a
    heuristic applies; None otherwise.

    Matching is case-insensitive and uses the LONGEST matching prefix
    or suffix in the hint table. The lookup walks the table in order
    so more specific entries (e.g. `gain_dc`) take precedence over
    less specific ones (e.g. `gain`).
    """
    lower = name.lower()
    for keyword, dim, units in _FIELD_NAME_HINTS:
        if lower == keyword:
            return (dim, units)
    for keyword, dim, units in _FIELD_NAME_HINTS:
        if keyword in lower:
            return (dim, units)
    return None


# ─── Sizing-intent heuristics ───────────────────────────────────


# Sizing kwarg key -> (min, max) reasonable range. Sizing-intent rule
# checks each key's value falls within this range. `gm_id` is the
# canonical inversion-coefficient knob — values 3..30 cover the full
# strong-inversion-to-sub-threshold spectrum; outside that range the
# value is almost certainly a typo or a unit mistake.
_GM_ID_KEYS = frozenset({
    "gm_id_diff", "gm_id_load1", "gm_id_load2",
    "gm_id_cs", "gm_id_input", "gm_id_gain",
    "gm_id_cas", "gm_id_tail",
})
_GM_ID_MIN, _GM_ID_MAX = 3.0, 30.0


# Pairs of sizing kwarg keys where the LLM might supply both forms —
# the gm/Id form wins (per templates' priority order). The lint warns
# when both are present so the LLM sees that one is being silently
# ignored.
_SIZING_OVERRIDE_PAIRS: list[tuple[str, str]] = [
    ("W_diff",   "gm_id_diff"),
    ("W_load1",  "gm_id_load1"),
    ("W_load2",  "gm_id_load2"),
    ("W_cs",     "gm_id_cs"),
    ("W_gain",   "gm_id_input"),
    ("W_input",  "gm_id_input"),
]


# ─── Stage-template port lists ──────────────────────────────────


# Static map of stage-style -> expected port list. Mirrors the values
# returned by each builder in clay/templates/. Kept inline (rather
# than introspected at runtime) so the lint stays fast and side-effect
# free — calling get_stage(...) loads template modules and PDK
# adapters, which is overkill for a lint pass.
_STAGE_PORTS: dict[str, list[str]] = {
    "common_source_pmos_input": ["in", "out", "vdd", "vss"],
    "folded_cascode":           ["inn", "inp", "out", "vdd", "vss"],
    "fully_differential_pmos_input": [
        "inn", "inp", "out_p", "out_n", "vdd", "vss",
    ],
    "nested_miller_3stage_feedforward": [
        "inn", "inp", "out", "vdd", "vss",
    ],
    "telescopic_cascode":       ["inn", "inp", "out", "vdd", "vss"],
    "two_stage_miller_nmos":    ["inn", "inp", "out", "vdd", "vss"],
    "two_stage_miller_pmos":    ["inn", "inp", "out", "vdd", "vss"],
    "two_stage_miller_pmos_explicit_bias": [
        "inn", "inp", "out", "vdd", "vss",
    ],
}


# ─── Reserved keywords ──────────────────────────────────────────


# Tokens that have a special meaning in the Clay grammar. Using them
# as a CELL NAME parses (the grammar's `CNAME` matches any identifier
# unless the keyword is explicitly higher-priority), but the result is
# confusing: `cell verify { ... }` then `verify verify;` is awful.
_RESERVED_KEYWORDS: frozenset[str] = frozenset({
    "package", "pdk", "import", "as",
    "cell", "topology", "spec", "sizing", "sizing_override",
    "verify", "corners", "temps", "bias_tree",
    "instance", "device", "cap", "res",
    "integer", "real", "freq", "volt", "current", "time",
    "angle", "length", "db",
    "external", "from", "of", "by",
    "input", "output", "inout",
    "power_p", "power_n", "bias", "ground",
    "pmos", "nmos", "pmos_diode", "nmos_diode",
    "pmos_mirror", "nmos_mirror",
})


# ─── Rule implementations ───────────────────────────────────────


def rule_E001_missing_db_unit_on_gain(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """E001 — spec field has a name implying dB (gain_dc, gain_db, dc_gain,
    SNR, CMRR, PSRR, THD) but the value carries no `dB` suffix.

    This is a warning (not an error): the parser already accepts
    `gain_dc : real >= 25` and the value is stored as a plain 25.0.
    But the LLM almost certainly meant 25 dB; flagging the missing
    unit catches the case where the LLM later writes something like
    `gain : real >= 0.05` thinking the value is in linear amplitude.
    """
    findings: list[Finding] = []
    for src_field in _enumerate_source_spec_fields(source):
        hint = _field_name_hint(src_field.name)
        if hint is None:
            continue
        expected_dim, valid_units = hint
        if expected_dim != "db":
            continue
        if src_field.unit_text:
            continue
        line, col = _line_col_of(src_field.offset, source)
        findings.append(Finding(
            rule_id="W001",
            severity="warning",
            line=line,
            col=col,
            message=(
                f"spec field {src_field.name!r} looks like a dB quantity "
                f"but value {src_field.value_text!r} has no unit; expected "
                f"a 'dB' suffix (e.g. `{src_field.name} : real >= "
                f"{src_field.value_text}dB;`)"
            ),
        ))
    return findings


def rule_W010_wrong_dimension(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W010 — spec-field dimension type does not match the name's
    typical dimension.

    Example: `slew_rate : freq >= 1V/us;` — slew_rate is volt/time,
    not frequency. The `freq` dimension would make the value parse to
    a frequency multiplier, then ngspice would silently use a wrong
    target. We warn (not error) because edge cases exist (an LLM may
    deliberately use `real` as a catch-all dimension since the parser
    accepts any unit when dim == 'real').
    """
    findings: list[Finding] = []
    for src_field in _enumerate_source_spec_fields(source):
        hint = _field_name_hint(src_field.name)
        if hint is None:
            continue
        expected_dim, valid_units = hint
        # `real` is the catch-all dimension and is always permissible
        # — the parser uses the unit to do the actual SI conversion.
        if src_field.dim == "real":
            continue
        # `db` dimension keyword is informational only; the spec
        # carrying a dB quantity may legitimately use `real` OR `db`.
        if expected_dim == "db":
            continue
        # compound_vt slew-rate uses a compound unit; the only
        # acceptable dim is `real` (catch-all). Anything else is a
        # red flag.
        if expected_dim == "compound_vt":
            if src_field.dim != "real":
                line, col = _line_col_of(src_field.offset, source)
                findings.append(Finding(
                    rule_id="W010",
                    severity="warning",
                    line=line,
                    col=col,
                    message=(
                        f"spec field {src_field.name!r} has dimension "
                        f"{src_field.dim!r} but the field name suggests a "
                        f"volt/time quantity (e.g. slew rate); use "
                        f"`real` and a V/us suffix on the value"
                    ),
                ))
            continue
        # Generic dimension mismatch (freq != volt, etc.).
        if src_field.dim != expected_dim:
            line, col = _line_col_of(src_field.offset, source)
            findings.append(Finding(
                rule_id="W010",
                severity="warning",
                line=line,
                col=col,
                message=(
                    f"spec field {src_field.name!r} has dimension "
                    f"{src_field.dim!r} but the field name suggests "
                    f"{expected_dim!r}"
                ),
            ))
    return findings


def rule_W002_excessive_sizing_intent(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W002 — sizing intent value is outside a reasonable engineering range.

    Specifically: `gm_id_<x>` values should be in [3, 30]. Values like
    `gm_id_diff = 50` are mechanically accepted by the closed-form
    solver but produce an unphysical W → either ridiculously tiny or
    negative. Catching this before sim time saves a round-trip.
    """
    findings: list[Finding] = []
    for cd in ir.cell_defs:
        for key, value in cd.sizing_kwargs.items():
            if key in _GM_ID_KEYS:
                if value < _GM_ID_MIN or value > _GM_ID_MAX:
                    line, col = _find_ident_pos(
                        source, key, near_keyword="sizing",
                    )
                    findings.append(Finding(
                        rule_id="W002",
                        severity="warning",
                        line=line,
                        col=col,
                        message=(
                            f"sizing intent {key}={value:g} is outside "
                            f"reasonable range [{_GM_ID_MIN:g}, "
                            f"{_GM_ID_MAX:g}] (3=strong inversion, "
                            f"30=deep sub-Vth); double-check the value"
                        ),
                    ))
    for ca in ir.cell_aliases:
        for key, value in ca.sizing_overrides.items():
            if key in _GM_ID_KEYS:
                if value < _GM_ID_MIN or value > _GM_ID_MAX:
                    line, col = _find_ident_pos(
                        source, key, near_keyword="sizing_override",
                    )
                    findings.append(Finding(
                        rule_id="W002",
                        severity="warning",
                        line=line,
                        col=col,
                        message=(
                            f"sizing override {key}={value:g} is outside "
                            f"reasonable range [{_GM_ID_MIN:g}, "
                            f"{_GM_ID_MAX:g}]"
                        ),
                    ))
    for vj in ir.verify_jobs:
        for key, value in vj.sizing_overrides.items():
            if key in _GM_ID_KEYS:
                if value < _GM_ID_MIN or value > _GM_ID_MAX:
                    line, col = _find_ident_pos(
                        source, key, near_keyword="sizing_override",
                    )
                    findings.append(Finding(
                        rule_id="W002",
                        severity="warning",
                        line=line,
                        col=col,
                        message=(
                            f"sizing override {key}={value:g} is outside "
                            f"reasonable range [{_GM_ID_MIN:g}, "
                            f"{_GM_ID_MAX:g}]"
                        ),
                    ))
    return findings


def rule_W003_w_and_gmid_both_set(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W003 — both a physical W kwarg and its corresponding gm_id intent
    are present; the gm_id form wins, the W is silently ignored.

    Templates (`build_two_stage_miller_pmos`, `folded_cascode`, etc.)
    derive `W_diff` from `(gm_id_diff, I_d, L_diff)` whenever
    `gm_id_diff` is supplied; the explicit `W_diff` kwarg is then
    overwritten. The LLM may set both thinking the explicit W is a
    "fallback" — actually it never takes effect. Flag it.
    """
    findings: list[Finding] = []

    def _check_pairs(kwargs: dict, key_kw: str) -> None:
        for w_key, gm_key in _SIZING_OVERRIDE_PAIRS:
            if w_key in kwargs and gm_key in kwargs:
                line, col = _find_ident_pos(
                    source, w_key, near_keyword=key_kw,
                )
                findings.append(Finding(
                    rule_id="W003",
                    severity="warning",
                    line=line,
                    col=col,
                    message=(
                        f"both {w_key!r}={kwargs[w_key]:g} and "
                        f"{gm_key!r}={kwargs[gm_key]:g} are set; the "
                        f"gm/Id form wins and the explicit W is ignored"
                    ),
                ))

    for cd in ir.cell_defs:
        _check_pairs(cd.sizing_kwargs, "sizing")
    for ca in ir.cell_aliases:
        _check_pairs(ca.sizing_overrides, "sizing_override")
    for vj in ir.verify_jobs:
        _check_pairs(vj.sizing_overrides, "sizing_override")
    return findings


def rule_W004_empty_topology_block(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W004 — empty `topology { }` block on a cell_def.

    The parser allows an empty topology block, but downstream compile
    will reject it (missing style / no instances / no devices). Catch
    it at lint time so the LLM sees a precise message instead of an
    opaque "topology block missing required field 'style'".
    """
    findings: list[Finding] = []
    for cd in ir.cell_defs:
        if (not cd.topology
                and not cd.instances
                and not cd.devices):
            # Locate the cell's name in the source so the diagnostic
            # points at the cell, not at the first character of the
            # file.
            line, col = _find_ident_pos(source, cd.local_name)
            findings.append(Finding(
                rule_id="W004",
                severity="warning",
                line=line,
                col=col,
                message=(
                    f"cell {cd.local_name!r} has an empty `topology {{ }}` "
                    f"block; declare a `style = '...'`, one or more "
                    f"`instance` entries, or one or more `device` entries"
                ),
            ))
    return findings


def rule_I001_unused_cell(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """I001 — a cell_def is declared but never referenced by `verify`.

    Library files routinely declare cells without verifying them
    (the verifying file imports them), so this is an `info`-level
    finding, not a warning. It's still useful: a standalone
    .clay file with `cell MyAmp { ... }` and no `verify MyAmp;` is
    almost certainly an oversight.
    """
    findings: list[Finding] = []
    # Build the set of cell-paths referenced by verify jobs.
    verified_paths: set[str] = set()
    for vj in ir.verify_jobs:
        verified_paths.add(vj.cell_path)
    for cd in ir.cell_defs:
        local_path = f"__celldef__::{cd.local_name}"
        if local_path in verified_paths:
            continue
        # Also consider that a cell_def could be referenced through a
        # cell_alias whose parent_path matches its qualified form
        # (rare but possible across imported namespaces). For the
        # simple case here, just check verify paths.
        line, col = _find_ident_pos(source, cd.local_name)
        findings.append(Finding(
            rule_id="I001",
            severity="info",
            line=line,
            col=col,
            message=(
                f"cell {cd.local_name!r} is declared but never "
                f"`verify`d in this file"
            ),
        ))
    return findings


def rule_W005_port_mismatch_with_stage(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W005 — a cell_def declares `style = '...'` but its port list
    does not match the stage's expected port list.

    The compiler catches this too (`compile_cell_def`'s shape check),
    but the lint message includes line/col context AND the surface
    error happens in `python -m clay verify` only; running the linter
    first means the LLM sees the diagnostic before invoking ngspice.
    """
    findings: list[Finding] = []
    for cd in ir.cell_defs:
        style = cd.topology.get("style")
        if not isinstance(style, str):
            continue
        expected = _STAGE_PORTS.get(style)
        if expected is None:
            # Unknown style — compile_cell_def will reject it. The
            # lint doesn't try to guess what it should be.
            continue
        if set(cd.ports) != set(expected):
            line, col = _find_ident_pos(source, cd.local_name)
            findings.append(Finding(
                rule_id="W005",
                severity="warning",
                line=line,
                col=col,
                message=(
                    f"cell {cd.local_name!r} declares style "
                    f"{style!r} which expects ports "
                    f"{sorted(expected)}, but the cell declares "
                    f"{sorted(cd.ports)}"
                ),
            ))
    return findings


def rule_W006_dangling_net(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W006 — a net referenced by exactly 1 device pin in a primitive
    cell. Usually means a typo of a real net.

    Skipped for stage-template and composed cells (their internal
    nets are wired through Python, not authored in .clay).
    """
    findings: list[Finding] = []
    legal_rails = {"0", "gnd"}
    for cd in ir.cell_defs:
        if not cd.devices:
            continue
        net_counts: dict[str, int] = {}
        for dev in cd.devices:
            for net in dev.pins.values():
                net_counts[net] = net_counts.get(net, 0) + 1
        legal_nets = (
            set(cd.ports)
            | legal_rails
            | {n.name for n in cd.bias_tree}
        )
        for net, count in sorted(net_counts.items()):
            if count == 1 and net not in legal_nets:
                line, col = _find_ident_pos(source, net)
                findings.append(Finding(
                    rule_id="W006",
                    severity="warning",
                    line=line,
                    col=col,
                    message=(
                        f"net {net!r} is touched by only 1 device pin "
                        f"and is not a declared port / global rail / "
                        f"bias_tree node; possible typo"
                    ),
                ))
    return findings


def rule_E002_reserved_keyword_as_cell_name(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """E002 — cell or alias is named after a reserved keyword (verify,
    topology, spec, ...). The grammar accepts it but the resulting
    file is confusing and bug-prone (`verify verify;`).
    """
    findings: list[Finding] = []
    for cd in ir.cell_defs:
        if cd.local_name in _RESERVED_KEYWORDS:
            line, col = _find_ident_pos(source, cd.local_name)
            findings.append(Finding(
                rule_id="E002",
                severity="error",
                line=line,
                col=col,
                message=(
                    f"cell name {cd.local_name!r} is a reserved Clay "
                    f"keyword; rename it (e.g. {cd.local_name}Amp)"
                ),
            ))
    for ca in ir.cell_aliases:
        if ca.local_name in _RESERVED_KEYWORDS:
            line, col = _find_ident_pos(source, ca.local_name)
            findings.append(Finding(
                rule_id="E002",
                severity="error",
                line=line,
                col=col,
                message=(
                    f"alias name {ca.local_name!r} is a reserved "
                    f"Clay keyword; rename it"
                ),
            ))
    return findings


def rule_W007_pdk_missing(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W007 — file has no `pdk '...';` declaration.

    Most files need a PDK pin for sizing solvers to pick mu_Cox and
    other process constants. Missing it leads to a runtime error
    at compile time. Flag at lint.
    """
    findings: list[Finding] = []
    if ir.pdk_id is None and (ir.cell_defs or ir.verify_jobs):
        findings.append(Finding(
            rule_id="W007",
            severity="warning",
            line=1,
            col=1,
            message=(
                "file declares cells / verify jobs but no "
                "`pdk '...';` directive; sizing solver needs a PDK pin"
            ),
        ))
    return findings


def rule_W008_package_missing(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W008 — file has no `package NAME;` declaration.

    A package name is required for imports to resolve consistently
    across files. Missing it is technically legal but rare.
    """
    findings: list[Finding] = []
    if ir.package_name is None and (ir.cell_defs or ir.cell_aliases):
        findings.append(Finding(
            rule_id="W008",
            severity="warning",
            line=1,
            col=1,
            message=(
                "file has no `package NAME;` declaration; "
                "cross-file imports may not resolve as expected"
            ),
        ))
    return findings


def rule_W009_no_spec_fields(
    file_path: str, ir: ClayFile, source: str,
) -> list[Finding]:
    """W009 — a cell_def has an empty `spec { }` block.

    Without spec fields, `verify Cell;` runs but has nothing to
    assert against. The cell will always "pass" (vacuously). Almost
    certainly an authoring oversight.
    """
    findings: list[Finding] = []
    for cd in ir.cell_defs:
        if not cd.spec.fields:
            line, col = _find_ident_pos(source, cd.local_name)
            findings.append(Finding(
                rule_id="W009",
                severity="warning",
                line=line,
                col=col,
                message=(
                    f"cell {cd.local_name!r} has an empty `spec` "
                    f"block; verify will pass vacuously"
                ),
            ))
    return findings


# Rule registry. Order in the list determines order of execution and
# output grouping. New rules append; rule IDs are stable.
CHECKS: list[tuple[str, Callable[[str, ClayFile, str], list[Finding]]]] = [
    ("W001", rule_E001_missing_db_unit_on_gain),
    ("W002", rule_W002_excessive_sizing_intent),
    ("W003", rule_W003_w_and_gmid_both_set),
    ("W004", rule_W004_empty_topology_block),
    ("W005", rule_W005_port_mismatch_with_stage),
    ("W006", rule_W006_dangling_net),
    ("W007", rule_W007_pdk_missing),
    ("W008", rule_W008_package_missing),
    ("W009", rule_W009_no_spec_fields),
    ("W010", rule_W010_wrong_dimension),
    ("E002", rule_E002_reserved_keyword_as_cell_name),
    ("I001", rule_I001_unused_cell),
]


# ─── Driver ─────────────────────────────────────────────────────


def lint_file(path: Path) -> list[Finding]:
    """Parse `path` and run every registered check on it.

    Raises ValueError on parse error (caller catches and converts to
    exit 2). FileNotFoundError if the path doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"clay file not found: {path}")
    source = path.read_text(encoding="utf-8")
    ir = parse_clay_file(path)
    findings: list[Finding] = []
    for _rule_id, check in CHECKS:
        try:
            findings.extend(check(str(path), ir, source))
        except Exception as exc:
            # Defensive: if a rule itself crashes, surface it as an
            # info-level diagnostic rather than failing the whole lint.
            findings.append(Finding(
                rule_id="L999",
                severity="info",
                line=1,
                col=1,
                message=f"internal: rule {_rule_id} raised {exc!r}",
            ))
    # Sort findings by (line, col, rule_id) for deterministic output.
    findings.sort(key=lambda f: (f.line, f.col, f.rule_id))
    return findings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clay_lint",
        description=(
            "Static analyzer for .clay files. Catches common authoring "
            "mistakes before ngspice is invoked. Each finding is "
            "printed in `file:line:col: severity [rule-id] message` "
            "form. Exit 1 if any finding's severity is at or above "
            "the --severity threshold; exit 2 on internal failure."
        ),
    )
    p.add_argument("path", help="path to a .clay file")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress non-error findings on stdout (still affects exit code)",
    )
    p.add_argument(
        "--severity",
        choices=("info", "warning", "error"),
        default="error",
        help=(
            "minimum severity to fail the lint pass with exit 1; "
            "default 'error' means warnings/infos do not cause failure"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = Path(args.path)
    try:
        findings = lint_file(path)
    except FileNotFoundError as exc:
        print(f"clay_lint: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Parse error or anything else upstream → exit 2.
        print(f"clay_lint: failed to parse {path}: {exc}", file=sys.stderr)
        return 2

    threshold_rank = _SEVERITY_RANK[args.severity]
    fail = False
    for f in findings:
        rank = _SEVERITY_RANK[f.severity]
        if rank >= threshold_rank:
            fail = True
        if args.quiet:
            # In quiet mode only print findings at or above threshold.
            if rank < threshold_rank:
                continue
        print(f.format(str(path)))

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
