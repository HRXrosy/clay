"""
SPICE post-simulation analysis: subset parser + device OP table + topological
stack detection + KVL consistency check + inter-run delta tracking.

This module is a fresh implementation in EDALLM style, inspired by the
``backend/src/acp/netlist/spice_parser.py`` and ``backend/src/agent/
simulate_analysis.py`` architecture from the cirona-acp-v5 reference
codebase. No code is copied verbatim — only the architectural pattern
(subset parser, per-device OP record, generic stack extraction, KVL
self-consistency, delta state) was adopted. See
``docs/superpowers/specs/spice-analysis-cirona-inspired.html`` for the
design notes and attribution.

The module is intentionally decoupled from ``clay/_ir_bridge.py`` and the
existing simulator-neutral ``clay/ir.py`` data classes: it operates over
raw SPICE text (the same text the ngspice driver emits) and raw log text
(the same log the driver parses for ``.meas`` results). That keeps the
post-analysis usable for any simulator backend, not just ngspice — exactly
the spirit of the Driver abstraction documented in
``feedback_decouple_simulator_backends`` memory.

Pipeline:

   parse_spice_netlist(text)        →  SpiceNetlist
                                       (list of SpiceDevice + sub-circuits)
   extract_device_op_table(log)     →  DeviceOpTable
                                       (Vgs, Vds, Vdsat, Id, vth, margin
                                        per MOSFET seen in @m.<path>[*])
   detect_topological_stacks(nl)    →  list[TopologicalStack]
                                       (NMOS chains GND→…, PMOS chains
                                        VDD→…, traversed source↔drain)
   check_kvl_consistency(nl, op)    →  list[KVLViolation]
                                       (per-stack |residual| > tol or
                                        device in triode)
   compute_simulation_delta(p,c)    →  DeltaTable
                                       (per-device Vds/margin shifts
                                        prev→curr; flags overshoot,
                                        SAT→TRIODE regression)

No I/O, no global state, no logging side effects beyond what callers ask
for. Designed to be wrapped into the existing testbench post-processing
without modifying the driver itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence


# ─── Public surface ────────────────────────────────────────────

__all__ = [
    # Netlist parsing
    "SpiceDeviceKind",
    "SpiceDevice",
    "SpiceSubckt",
    "SpiceNetlist",
    "parse_spice_netlist",
    # Operating-point table
    "DeviceOpRecord",
    "DeviceOpTable",
    "extract_device_op_table",
    # Topology
    "TopologicalStack",
    "detect_topological_stacks",
    # KVL
    "KVLViolation",
    "check_kvl_consistency",
    # Delta tracking
    "DeltaEntry",
    "DeltaTable",
    "compute_simulation_delta",
    # Constants
    "VDD_NODE_ALIASES",
    "GND_NODE_ALIASES",
    "DEFAULT_KVL_TOLERANCE_V",
]


# ─── Module-level constants ────────────────────────────────────

# Standard SPICE rail conventions. Lowercased for case-insensitive match.
VDD_NODE_ALIASES: frozenset[str] = frozenset({
    "vdd", "vcc", "vdda", "vddd", "vpwr", "v_dd",
})
GND_NODE_ALIASES: frozenset[str] = frozenset({
    "vss", "gnd", "0", "vgnd", "agnd", "dgnd",
})

# KVL residual tolerance: |Σ|Vds| − (V_top − V_bot)| must be below this
# to call a stack KVL-consistent. 10 mV is generous enough to absorb
# numerical noise on a 0.6V budget cascode while still flagging a real
# missing-device error (typically > 100 mV residual).
DEFAULT_KVL_TOLERANCE_V: float = 0.010

# Vds-vs-Vdsat margin threshold below which we call the device MARGINAL.
# Triode means margin < 0; saturation means margin ≥ MARGINAL threshold.
# 50 mV matches the common "design with at least 50 mV Vov headroom" rule.
_MARGINAL_THRESHOLD_V: float = 0.050


# ─── SPICE numeric value parsing ───────────────────────────────

# Magnitude suffixes per the SPICE convention. 'MEG' must be checked
# before single 'M' (which means milli, not mega — historical ngspice).
# ASCII-only; case-insensitive.
_SPICE_SUFFIX_MAP: dict[str, float] = {
    "T": 1e12,
    "G": 1e9,
    "K": 1e3,
    "M": 1e-3,
    "U": 1e-6,
    "N": 1e-9,
    "P": 1e-12,
    "F": 1e-15,
}


def _parse_spice_value(raw: str) -> Optional[float]:
    """Parse a SPICE-style numeric literal into a float.

    Accepts plain floats ('1.0', '-3.5e-9'), SI suffix forms ('8u' →
    8e-6, '1.5MEG' → 1.5e6), and rejects garbage. Returns ``None`` if
    the input cannot be coerced — caller decides whether to skip or
    raise. We never raise here: the SPICE parser must be tolerant of
    weird user input so a single malformed line doesn't abort the
    whole analysis pass.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    upper = s.upper()
    if upper.endswith("MEG"):
        try:
            return float(s[:-3]) * 1e6
        except ValueError:
            return None
    last = upper[-1] if upper else ""
    if last in _SPICE_SUFFIX_MAP and len(upper) > 1:
        try:
            return float(s[:-1]) * _SPICE_SUFFIX_MAP[last]
        except ValueError:
            return None
    return None


# ─── Netlist data model ────────────────────────────────────────


class SpiceDeviceKind:
    """Device kind tags used by the post-analysis.

    Plain string constants instead of an Enum: simpler to compare, no
    import surface bloat for callers, and serialises cleanly into JSON
    for the delta-state file (cirona uses Enum; we deliberately don't
    so the delta JSON file stays human-readable and version-stable
    across minor refactors).
    """

    MOSFET = "M"
    RESISTOR = "R"
    CAPACITOR = "C"
    INDUCTOR = "L"
    VSOURCE = "V"
    ISOURCE = "I"
    SUBCKT_INST = "X"


# Map device-letter → kind tag. The set of recognised top-level device
# prefixes is intentionally small — that's the v0.1.00 subset for
# analog cells. Behavioural elements (E/G/H/F/B etc.) are silently
# ignored: they appear in some PDK macros but never carry KVL meaning
# on their own.
_DEVICE_LETTER_TO_KIND: dict[str, str] = {
    "M": SpiceDeviceKind.MOSFET,
    "R": SpiceDeviceKind.RESISTOR,
    "C": SpiceDeviceKind.CAPACITOR,
    "L": SpiceDeviceKind.INDUCTOR,
    "V": SpiceDeviceKind.VSOURCE,
    "I": SpiceDeviceKind.ISOURCE,
    "X": SpiceDeviceKind.SUBCKT_INST,
}


@dataclass
class SpiceDevice:
    """A single SPICE device line, post-parse.

    For MOSFETs (kind=='M'), ``nodes`` is ordered [drain, gate, source,
    body], ``model`` is the model token, and ``params`` holds W/L/M etc.

    For two-pin passive R/C/L/V/I, ``nodes`` is [pin0, pin1]; ``value``
    holds the parsed numeric value (resistance / capacitance / etc.)
    when present.

    For X-subckt instances, ``nodes`` holds the port-net list (variable
    length, since user subckts have arbitrary pin counts), and ``model``
    holds the cell name that ``X<name> n1 n2 … cellname`` resolves to.
    """

    name: str                     # full instance name including the letter, e.g. "Mn1"
    kind: str                     # one of SpiceDeviceKind.*
    nodes: list[str]              # connection nodes in canonical order
    model: Optional[str] = None   # MOSFET model OR subckt cell name
    params: dict[str, float] = field(default_factory=dict)
    value: Optional[float] = None # passive 2-pin value, if parseable
    line_no: int = 0              # 1-based source line (for error reports)

    # Convenience accessors ----------------------------------------

    @property
    def is_mosfet(self) -> bool:
        return self.kind == SpiceDeviceKind.MOSFET

    @property
    def short_name(self) -> str:
        """Strip the device-letter prefix for log-key matching.

        Ngspice prints device state as ``@m.<short_name>[vds]`` where
        the prefix M is dropped and the rest is lowercased. We match
        that convention here so ``DeviceOpTable`` lookups round-trip.
        """
        return self.name[1:] if self.name and self.name[0].upper() in _DEVICE_LETTER_TO_KIND else self.name

    @property
    def drain(self) -> Optional[str]:
        return self.nodes[0] if self.is_mosfet and len(self.nodes) >= 1 else None

    @property
    def gate(self) -> Optional[str]:
        return self.nodes[1] if self.is_mosfet and len(self.nodes) >= 2 else None

    @property
    def source(self) -> Optional[str]:
        return self.nodes[2] if self.is_mosfet and len(self.nodes) >= 3 else None

    @property
    def body(self) -> Optional[str]:
        return self.nodes[3] if self.is_mosfet and len(self.nodes) >= 4 else None


@dataclass
class SpiceSubckt:
    """A ``.subckt name ports… / .ends`` block."""

    name: str
    ports: list[str]
    devices: list[SpiceDevice] = field(default_factory=list)


@dataclass
class SpiceNetlist:
    """Parsed SPICE file. Top-level devices live in ``devices``;
    subckt bodies live in ``subckts[*].devices``.

    The parser preserves source order so device.line_no can be quoted
    in error messages. Errors during parse are accumulated in
    ``errors`` (the parser never raises — single malformed lines are
    skipped, not fatal).
    """

    title: str = ""
    devices: list[SpiceDevice] = field(default_factory=list)
    subckts: list[SpiceSubckt] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def all_mosfets(self, *, include_subckts: bool = True) -> list[SpiceDevice]:
        """All MOSFET devices at top level (and, optionally, inside subckts).

        This is the workhorse accessor used by the stack detector. For
        most analog cells, all the MOSFETs live inside one ``.subckt``
        (the DUT) and ``include_subckts=True`` is correct; pass False
        only when you want to inspect testbench-level extras.
        """
        out: list[SpiceDevice] = [d for d in self.devices if d.is_mosfet]
        if include_subckts:
            for sub in self.subckts:
                out.extend(d for d in sub.devices if d.is_mosfet)
        return out


# ─── Netlist tokenisation ──────────────────────────────────────

# Strip ngspice-style inline comments ('$' or ';' through end-of-line).
# Stays outside quoted strings (we don't expect those in v0.1.00 device
# lines, but the helper is conservative anyway).
_INLINE_COMMENT_RE = re.compile(r"\s*[\$;].*$")


def _strip_inline_comment(line: str) -> str:
    return _INLINE_COMMENT_RE.sub("", line).rstrip()


def _join_continuation_lines(text: str) -> list[tuple[int, str]]:
    """Glue SPICE '+' continuation lines onto the preceding statement.

    Returns a list of ``(original_line_no, joined_text)``. Comment lines
    (starting with '*') and blank lines are dropped. Inline comments are
    stripped per-line BEFORE the join. The ``original_line_no`` is the
    1-based line in the source file where the *first* token of the
    joined statement appeared — useful for error reports.

    Orphan continuation lines (a '+' line with no prior statement) are
    silently dropped: that's a malformed file, and we'd rather not
    crash the analysis.
    """
    out: list[tuple[int, str]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        stripped = _strip_inline_comment(raw)
        lead = stripped.lstrip()
        if not lead or lead.startswith("*"):
            continue
        if lead.startswith("+"):
            if out:
                prev_line_no, prev_text = out[-1]
                out[-1] = (prev_line_no, prev_text + " " + lead[1:].strip())
            continue
        out.append((idx, stripped))
    return out


def _tokenise(line: str) -> list[str]:
    """Whitespace-split, preserving quoted runs as single tokens.

    Matches the ngspice manual's tokeniser closely enough for v0.1.00
    analog cells. Both single and double quotes are honoured. Any
    unterminated quote at end-of-string is closed implicitly so we
    don't lose tokens.
    """
    tokens: list[str] = []
    buf: list[str] = []
    quote_char: Optional[str] = None
    for ch in line:
        if quote_char is not None:
            buf.append(ch)
            if ch == quote_char:
                # Exit quote mode but keep accumulating: a closing quote does
                # not end the token, so a following ``)`` or text (e.g.
                # ``SIN('1 2 3')``) stays part of the same token until the next
                # real boundary (whitespace at depth 0 / end of line).
                quote_char = None
            continue
        if ch in ('"', "'"):
            quote_char = ch
            buf.append(ch)
            continue
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _is_param_token(tok: str) -> bool:
    return "=" in tok and not tok.startswith("=")


def _extract_param_assignments(tokens: Iterable[str]) -> dict[str, float]:
    """Collect ``KEY=VALUE`` assignments where VALUE is a parseable number.

    Unknown / non-numeric assignments (e.g. ``mname=nfet_01v8``) are
    skipped silently — the caller is expected to pick those up from
    the model-token position when relevant (MOSFET model lookup).
    """
    out: dict[str, float] = {}
    for tok in tokens:
        if not _is_param_token(tok):
            continue
        key, _, val = tok.partition("=")
        key = key.strip().upper()
        if not key:
            continue
        parsed = _parse_spice_value(val)
        if parsed is None:
            continue
        out[key] = parsed
    return out


# ─── Per-device parsing ────────────────────────────────────────


def _parse_mosfet_line(name: str, rest: list[str], line_no: int) -> Optional[SpiceDevice]:
    """``M<name> D G S B model [W=… L=… M=…]`` → SpiceDevice.

    Tolerates the ngspice convention of allowing the model token to
    come after the four pin tokens, with params in any order. Returns
    None if there are fewer than 4 non-param tokens before the
    presumed model position (insufficient data → caller skips the
    line and records an error).
    """
    pins: list[str] = []
    model: Optional[str] = None
    for tok in rest:
        if _is_param_token(tok):
            break
        if len(pins) < 4:
            pins.append(tok)
        else:
            model = tok
            break
    if len(pins) < 4:
        return None
    params = _extract_param_assignments(rest)
    return SpiceDevice(
        name=name,
        kind=SpiceDeviceKind.MOSFET,
        nodes=pins,
        model=model,
        params=params,
        line_no=line_no,
    )


def _parse_two_pin_line(
    kind: str, name: str, rest: list[str], line_no: int
) -> Optional[SpiceDevice]:
    """R/C/L/V/I: ``<name> A B value [params]``."""
    pins = [t for t in rest[:2] if not _is_param_token(t)]
    if len(pins) < 2:
        return None
    # Try to interpret the third non-param token as a numeric value.
    value: Optional[float] = None
    if len(rest) >= 3:
        non_param = [t for t in rest[2:] if not _is_param_token(t)]
        if non_param:
            value = _parse_spice_value(non_param[0])
    params = _extract_param_assignments(rest)
    return SpiceDevice(
        name=name,
        kind=kind,
        nodes=pins,
        model=None,
        params=params,
        value=value,
        line_no=line_no,
    )


def _parse_subckt_inst_line(
    name: str, rest: list[str], line_no: int
) -> Optional[SpiceDevice]:
    """``X<name> n1 n2 … cellname [params]``.

    By SPICE convention, the last non-param token is the cell name;
    everything before it is the port-net list. We require at least
    two non-param tokens (one port + one cell name) — anything less
    is malformed.
    """
    non_param = [t for t in rest if not _is_param_token(t)]
    if len(non_param) < 2:
        return None
    cell_name = non_param[-1]
    nodes = non_param[:-1]
    params = _extract_param_assignments(rest)
    return SpiceDevice(
        name=name,
        kind=SpiceDeviceKind.SUBCKT_INST,
        nodes=nodes,
        model=cell_name,
        params=params,
        line_no=line_no,
    )


def _parse_device_line(line: str, line_no: int) -> Optional[SpiceDevice]:
    """Dispatch to the appropriate device parser based on the first letter."""
    tokens = _tokenise(line)
    if not tokens:
        return None
    name = tokens[0]
    letter = name[0].upper()
    if letter not in _DEVICE_LETTER_TO_KIND:
        return None
    kind = _DEVICE_LETTER_TO_KIND[letter]
    rest = tokens[1:]
    if kind == SpiceDeviceKind.MOSFET:
        return _parse_mosfet_line(name, rest, line_no)
    if kind == SpiceDeviceKind.SUBCKT_INST:
        return _parse_subckt_inst_line(name, rest, line_no)
    return _parse_two_pin_line(kind, name, rest, line_no)


# ─── Main parser ───────────────────────────────────────────────


def parse_spice_netlist(text: str) -> SpiceNetlist:
    """Parse a SPICE-format netlist string into a SpiceNetlist.

    The first non-comment, non-directive line is treated as the title,
    matching the historical SPICE convention. ``.subckt`` / ``.ends``
    blocks are honoured: devices declared inside a subckt body end up
    in the corresponding ``SpiceSubckt.devices`` list, not in the
    top-level ``SpiceNetlist.devices`` list. Other directives
    (``.option``, ``.include``, ``.model``, ``.param``, ``.control``,
    measurement lines, etc.) are silently skipped — they carry no
    structural information for the post-analysis pipeline.

    Never raises. On a structurally malformed device line, an error
    message is appended to ``netlist.errors`` and the line is dropped.
    """
    nl = SpiceNetlist()

    # Title detection. Two paths:
    #
    # 1. First raw line begins with '*' → it's a SPICE comment line;
    #    the rest is the title. The continuation-joiner would
    #    otherwise drop comment lines entirely, so we catch this
    #    case against the raw text BEFORE handing it through.
    # 2. First non-comment, non-directive, non-device statement is
    #    the title. "Clay cascode example" hits this branch: token 0
    #    is `Clay` which starts with C, but the line cannot parse as
    #    a valid capacitor (no numeric value, no recognised model),
    #    so we treat it as title text.
    raw_lines = text.splitlines() if text else []
    consumed_raw_lines = 0
    for raw in raw_lines:
        s = raw.strip()
        if not s:
            consumed_raw_lines += 1
            continue
        if s.startswith("*"):
            nl.title = s.lstrip("*").strip()
            consumed_raw_lines += 1
        break

    remaining_text = "\n".join(raw_lines[consumed_raw_lines:])
    statements = _join_continuation_lines(remaining_text)
    if not statements:
        return nl

    # Re-base statement line numbers to the original file's numbering.
    statements = [(line_no + consumed_raw_lines, txt) for line_no, txt in statements]

    # Path 2: probe the first statement. Accept as title only if it
    # neither starts with '.' nor parses as a recognised device.
    if not nl.title and statements:
        first_line_no, first_text = statements[0]
        first_stripped = first_text.lstrip()
        if first_stripped and not first_stripped.startswith("."):
            first_char = first_stripped[0].upper()
            looks_like_device = False
            if first_char in _DEVICE_LETTER_TO_KIND:
                probe = _parse_device_line(first_stripped, first_line_no)
                # A "probable device" needs either a recognised model
                # (for MOSFET / subckt-inst) or a parseable numeric
                # value (for 2-pin passives). Without those, the line
                # is almost certainly title prose.
                if probe is not None and (
                    probe.model is not None or probe.value is not None
                    or probe.params
                ):
                    looks_like_device = True
            if not looks_like_device:
                nl.title = first_stripped.strip()
                statements = statements[1:]

    active_subckt: Optional[SpiceSubckt] = None

    for line_no, stmt in statements:
        lead = stmt.lstrip()
        if not lead:
            continue
        if lead.startswith("."):
            tokens = _tokenise(lead)
            if not tokens:
                continue
            directive = tokens[0][1:].lower()
            if directive == "subckt":
                if len(tokens) >= 2:
                    sub_name = tokens[1]
                    sub_ports = [t for t in tokens[2:] if not _is_param_token(t)]
                    active_subckt = SpiceSubckt(name=sub_name, ports=sub_ports)
                    nl.subckts.append(active_subckt)
                continue
            if directive == "ends":
                active_subckt = None
                continue
            # Any other directive is post-analysis-irrelevant.
            continue

        first_char = lead[0].upper()
        if first_char not in _DEVICE_LETTER_TO_KIND:
            continue
        dev = _parse_device_line(lead, line_no)
        if dev is None:
            nl.errors.append(
                f"line {line_no}: could not parse device statement {lead!r}"
            )
            continue
        if active_subckt is not None:
            active_subckt.devices.append(dev)
        else:
            nl.devices.append(dev)

    return nl


# ─── Operating-point table ────────────────────────────────────


@dataclass
class DeviceOpRecord:
    """One MOSFET's DC operating point, as scraped from a SPICE log.

    All fields are optional because different testbenches print
    different subsets. The ``margin`` property is the saturation
    margin |Vds| − |Vdsat| (absolute values so PMOS and NMOS share a
    convention); ``status`` summarises into one of {SAT, MARGINAL,
    TRIODE, UNKNOWN}.
    """

    name: str                          # full path token, lowercased ('xota.mn1')
    short_name: str                    # last path component ('mn1')
    vgs: Optional[float] = None
    vds: Optional[float] = None
    vdsat: Optional[float] = None
    id_amp: Optional[float] = None
    vth: Optional[float] = None
    gm: Optional[float] = None

    @property
    def has_saturation_data(self) -> bool:
        return self.vds is not None and self.vdsat is not None

    @property
    def margin(self) -> Optional[float]:
        if not self.has_saturation_data:
            return None
        # abs() so PMOS (Vds negative, Vdsat negative) yields the
        # same sign convention as NMOS.
        return abs(self.vds) - abs(self.vdsat)  # type: ignore[arg-type]

    @property
    def status(self) -> str:
        m = self.margin
        if m is None:
            return "UNKNOWN"
        if m < 0.0:
            return "TRIODE"
        if m < _MARGINAL_THRESHOLD_V:
            return "MARGINAL"
        return "SAT"


@dataclass
class DeviceOpTable:
    """Container for per-device DC operating-point records.

    Keyed by the full lowercased path (e.g. ``xdut.xota.mn5_bottom``).
    The ``by_short`` accessor lets callers look up by short name when
    the netlist scope is unambiguous (one record per short name).
    """

    records: dict[str, DeviceOpRecord] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records.values())

    def by_short(self, short: str) -> Optional[DeviceOpRecord]:
        short_l = short.lower()
        for rec in self.records.values():
            if rec.short_name == short_l:
                return rec
        return None

    def triode_devices(self) -> list[DeviceOpRecord]:
        return [r for r in self.records.values() if r.status == "TRIODE"]

    def sat_devices(self) -> list[DeviceOpRecord]:
        return [r for r in self.records.values() if r.status == "SAT"]


# Ngspice prints device state as ``@m.<path>[param] = <value>``. The
# path can include sub-circuit prefixes separated by dots.
_OP_PARAM_LINE_RE = re.compile(
    r"@m\.([A-Za-z0-9_.]+)\[(\w+)\]\s*=\s*([-+]?[\d.eE+\-]+)",
)


def extract_device_op_table(log_text: str) -> DeviceOpTable:
    """Scan a SPICE log for per-MOSFET operating-point lines.

    Looks for ngspice's ``@m.<path>[param] = <value>`` print format,
    where ``param ∈ {vgs, vds, vdsat, id, vth, gm}``. Any other params
    in the same format (e.g. ``[cgg]``, ``[ron]``) are silently ignored
    — adding them later is a one-field addition to ``DeviceOpRecord``.

    Returns an empty table for empty / None input. Records build
    incrementally: a line ``@m.mn1[vds]=…`` creates a record if none
    exists, then subsequent ``@m.mn1[vgs]=…`` lines populate further
    fields. Unparseable numeric values are skipped per-field, not
    per-record.
    """
    table = DeviceOpTable()
    if not log_text:
        return table

    for match in _OP_PARAM_LINE_RE.finditer(log_text):
        path = match.group(1).lower()
        param = match.group(2).lower()
        raw_value = match.group(3)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        rec = table.records.get(path)
        if rec is None:
            short = path.split(".")[-1]
            rec = DeviceOpRecord(name=path, short_name=short)
            table.records[path] = rec
        if param == "vgs":
            rec.vgs = value
        elif param == "vds":
            rec.vds = value
        elif param == "vdsat":
            rec.vdsat = value
        elif param == "id":
            rec.id_amp = value
        elif param == "vth":
            rec.vth = value
        elif param == "gm":
            rec.gm = value

    return table


# ─── Topological stack detection ───────────────────────────────


@dataclass
class TopologicalStack:
    """A chain of MOSFETs stacked between two power rails (or anchors).

    The chain is listed top-to-bottom in ``device_names`` — index 0 is
    the device closest to ``top_node``, index -1 is the device closest
    to ``bottom_node``. For NMOS chains the source of device 0 is the
    drain of device 1, and so on; for PMOS chains, drain of device 0
    is source of device 1. ``kind`` ∈ {nmos, pmos}.
    """

    kind: str
    top_node: str
    bottom_node: str
    device_names: list[str] = field(default_factory=list)
    drains: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.device_names)


def _classify_mosfet_kind(dev: SpiceDevice) -> Optional[str]:
    """Return 'nmos' / 'pmos' from device model name, or None if unknown.

    Heuristic: look for 'nmos', 'nfet', or 'n_' as substrings → NMOS;
    'pmos', 'pfet', or 'p_' → PMOS. We deliberately keep this loose
    so it covers sky130, gf180mcu, open_pdk, open_pdk, and most generic
    PDK names without a hand-maintained list. Subckt-wrapped devices
    (sky130's xMn → sky130_fd_pr__nfet_01v8) pass through transparently.
    """
    model = (dev.model or "").lower()
    if not model:
        return None
    if "nmos" in model or "nfet" in model or "_n_" in model or model.startswith("n"):
        if "pmos" in model or "pfet" in model:
            # ambiguous ("pmos…nmos" hybrid name) — give up
            return None
        return "nmos"
    if "pmos" in model or "pfet" in model or "_p_" in model or model.startswith("p"):
        return "pmos"
    return None


def _norm_node(n: Optional[str]) -> str:
    return (n or "").strip().lower()


def detect_topological_stacks(netlist: SpiceNetlist) -> list[TopologicalStack]:
    """Walk the netlist's MOSFETs and extract VDD→GND vertical chains.

    Algorithm:

    For NMOS chains, anchor at every NMOS whose source is on a known
    GND alias. Walk up through drain → next.source; stop on the first
    node that hits a VDD alias, or when the next-source lookup forks
    (>1 unvisited NMOS hangs off the same drain). The visited set
    prevents revisiting devices across multiple anchors.

    Mirrors for PMOS: anchor at every PMOS whose source is on a VDD
    alias; walk down through drain → next.source until the next-source
    lookup hits a GND alias or forks.

    The detector handles two device-modeling conventions:

      1. Direct MOSFET lines (M<name>) — model name carries the
         NMOS/PMOS hint.
      2. Subckt-wrapped MOSFETs (X<name> … sky130_fd_pr__nfet_01v8) —
         we promote those to MOSFET-equivalent if the cell name
         carries the n/p hint AND the device has at least 4 pins.
         This is the sky130 convention used by EDALLM's ngspice
         driver (``xMn1 d g s b sky130_fd_pr__nfet_01v8 W=…``).

    Returns NMOS chains followed by PMOS chains. Chains of length 1
    (a single device anchored at a rail) are included — those still
    matter for KVL sanity even though they have nothing to stack
    against.
    """
    # Promote subckt-instances that look like MOSFET wrappers to
    # MOSFET-equivalent records. This is the sky130 case: the driver
    # emits ``xMn1 d g s b sky130_fd_pr__nfet_01v8 W=… L=…`` instead
    # of a bare M-line because the PDK MOSFET is itself a subckt.
    mosfets: list[SpiceDevice] = []
    for dev in netlist.all_mosfets():
        mosfets.append(dev)
    for sub in netlist.subckts:
        for dev in sub.devices:
            if dev.kind == SpiceDeviceKind.SUBCKT_INST and len(dev.nodes) >= 4:
                # Same classification heuristic as plain MOSFETs.
                model_hint = (dev.model or "").lower()
                if "nfet" in model_hint or "nmos" in model_hint or "pfet" in model_hint or "pmos" in model_hint:
                    promoted = SpiceDevice(
                        name=dev.name,
                        kind=SpiceDeviceKind.MOSFET,
                        nodes=dev.nodes[:4],
                        model=dev.model,
                        params=dict(dev.params),
                        line_no=dev.line_no,
                    )
                    mosfets.append(promoted)
    # Also check top-level X-instances (testbench wraps DUT this way).
    for dev in netlist.devices:
        if dev.kind == SpiceDeviceKind.SUBCKT_INST and len(dev.nodes) >= 4:
            model_hint = (dev.model or "").lower()
            if "nfet" in model_hint or "nmos" in model_hint or "pfet" in model_hint or "pmos" in model_hint:
                promoted = SpiceDevice(
                    name=dev.name,
                    kind=SpiceDeviceKind.MOSFET,
                    nodes=dev.nodes[:4],
                    model=dev.model,
                    params=dict(dev.params),
                    line_no=dev.line_no,
                )
                mosfets.append(promoted)

    nmos: list[SpiceDevice] = []
    pmos: list[SpiceDevice] = []
    for dev in mosfets:
        kind = _classify_mosfet_kind(dev)
        if kind == "nmos":
            nmos.append(dev)
        elif kind == "pmos":
            pmos.append(dev)

    stacks: list[TopologicalStack] = []
    stacks.extend(_build_chains(nmos, anchor_set=GND_NODE_ALIASES, top_set=VDD_NODE_ALIASES, kind="nmos"))
    stacks.extend(_build_chains(pmos, anchor_set=VDD_NODE_ALIASES, top_set=GND_NODE_ALIASES, kind="pmos"))
    return stacks


def _build_chains(
    devices: Sequence[SpiceDevice],
    *,
    anchor_set: frozenset[str],
    top_set: frozenset[str],
    kind: str,
) -> list[TopologicalStack]:
    """Generic chain builder. For NMOS we anchor at source∈GND and walk
    upward via drain→next.source. For PMOS we anchor at source∈VDD and
    walk downward via drain→next.source.

    The traversal is identical for both polarities because the SPICE
    pin-order convention puts source closer to the *fixed* rail in
    both cases — what differs is which rail counts as the anchor.
    """
    if not devices:
        return []

    # Build an inverted index: node → list of (device, role) where role
    # tells us whether that node is the device's drain or source.
    node_index: dict[str, list[tuple[SpiceDevice, str]]] = {}
    for dev in devices:
        node_index.setdefault(_norm_node(dev.drain), []).append((dev, "drain"))
        node_index.setdefault(_norm_node(dev.source), []).append((dev, "source"))

    results: list[TopologicalStack] = []
    visited: set[str] = set()

    for anchor in devices:
        if anchor.name in visited:
            continue
        anchor_src = _norm_node(anchor.source)
        if anchor_src not in anchor_set:
            continue
        # Build the chain from anchor upward (or downward, for PMOS).
        # We always traverse anchor.drain → look for next-device whose
        # SOURCE is at that node. By the time we reach a fork or a
        # top-rail hit, the chain is complete.
        chain: list[SpiceDevice] = [anchor]
        visited.add(anchor.name)
        cursor = _norm_node(anchor.drain)
        while True:
            candidates = [
                dev for dev, role in node_index.get(cursor, [])
                if role == "source" and dev.name not in visited
            ]
            if len(candidates) != 1:
                break
            nxt = candidates[0]
            chain.append(nxt)
            visited.add(nxt.name)
            cursor = _norm_node(nxt.drain)
            if cursor in top_set:
                break

        # Reverse so the chain is listed top-to-bottom.
        if kind == "nmos":
            chain_top_down = list(reversed(chain))
        else:
            chain_top_down = list(chain)  # PMOS chain already runs top→bottom

        results.append(TopologicalStack(
            kind=kind,
            top_node=_norm_node(chain_top_down[0].drain) if kind == "nmos" else _norm_node(chain_top_down[0].source),
            bottom_node=_norm_node(chain_top_down[-1].source) if kind == "nmos" else _norm_node(chain_top_down[-1].drain),
            device_names=[d.name for d in chain_top_down],
            drains=[_norm_node(d.drain) for d in chain_top_down],
            sources=[_norm_node(d.source) for d in chain_top_down],
        ))
    return results


# ─── KVL consistency ───────────────────────────────────────────


@dataclass
class KVLViolation:
    """One stack's KVL audit result.

    ``residual = expected − actual``; positive means the chain doesn't
    consume the full rail-to-rail budget (one or more devices is in
    triode and the missing Vds is dropped elsewhere — usually across
    a tail current source). ``ok`` is True iff |residual| <
    tolerance AND no device in the chain is in triode.

    ``triode_devices`` lists the short-names of any chain members
    whose status is TRIODE — that's the actionable list for a sizing
    engine, separate from the KVL residual itself.
    """

    stack_kind: str
    top_node: str
    bottom_node: str
    expected_vds_sum: float           # = |V(top) − V(bottom)|
    actual_vds_sum: float             # = Σ |Vds| over chain
    residual: float                   # = expected − actual
    ok: bool
    devices_seen: list[str] = field(default_factory=list)
    triode_devices: list[str] = field(default_factory=list)


def _short_to_op_lookup(table: DeviceOpTable) -> dict[str, DeviceOpRecord]:
    """Build a short-name → record map. When multiple records share a
    short name (rare but possible if two different subckt instances
    have a same-named device), the LAST one wins. That matches the
    cirona behaviour and keeps the analyser predictable.
    """
    out: dict[str, DeviceOpRecord] = {}
    for rec in table.records.values():
        out[rec.short_name] = rec
    return out


def check_kvl_consistency(
    netlist: SpiceNetlist,
    op_table: DeviceOpTable,
    *,
    rail_voltages: Optional[Mapping[str, float]] = None,
    tolerance: float = DEFAULT_KVL_TOLERANCE_V,
) -> list[KVLViolation]:
    """For each stack, verify Σ|Vds| equals the rail-to-rail budget.

    ``rail_voltages`` is an optional map of rail-alias → DC voltage
    (e.g. {'vdd': 1.8, 'vss': 0.0}). When provided, the audit can
    compute an absolute residual; when omitted, we fall back to the
    self-consistent mode where ``expected = actual`` and ok=True iff
    no device in the chain is in triode.

    Devices missing from the op_table (no [vds] line in the log) are
    silently dropped from the actual-sum — the residual will still
    flag the discrepancy if a node voltage is given. We deliberately
    don't error: a partially-instrumented testbench should still
    produce a useful audit, not a fatal exception.
    """
    if not netlist.devices and not netlist.subckts:
        return []
    if not op_table.records:
        return []

    stacks = detect_topological_stacks(netlist)
    if not stacks:
        return []

    rail_v = {k.lower(): float(v) for k, v in (rail_voltages or {}).items()}

    short_lookup = _short_to_op_lookup(op_table)
    results: list[KVLViolation] = []

    for stack in stacks:
        # Try to resolve the rail-to-rail budget from rail_voltages.
        v_top = rail_v.get(stack.top_node)
        v_bot = rail_v.get(stack.bottom_node)
        if v_top is None and stack.top_node in VDD_NODE_ALIASES:
            v_top = rail_v.get("vdd")
        if v_bot is None and stack.bottom_node in GND_NODE_ALIASES:
            v_bot = rail_v.get("vss")
            if v_bot is None:
                v_bot = 0.0
        have_budget = v_top is not None and v_bot is not None
        expected_budget = abs(v_top - v_bot) if have_budget else 0.0

        # Compute actual Σ|Vds| from op data.
        #
        # Ngspice prints `@m.<path>[vds]` where `<path>` is the full
        # dotted instance path (subckt prefix + instance name) with
        # the leading device letter folded into the `@m.` prefix.
        # That means a Spice device named `Mmain` shows up in the log
        # as `mmain` (the leading M is dropped, lowercase). Lookup
        # tries two forms: (1) the lowercased full name (matches
        # `Mmain` → `mmain`), (2) the name minus its first letter
        # (matches `Mmain` → `main`, kept for legacy `mname` style).
        actual_sum = 0.0
        seen: list[str] = []
        triodes: list[str] = []
        for dev_name in stack.device_names:
            full_lower = dev_name.lower()
            stripped = (dev_name[1:] if dev_name and dev_name[0].upper() in _DEVICE_LETTER_TO_KIND
                        else dev_name).lower()
            rec = short_lookup.get(full_lower) or short_lookup.get(stripped)
            if rec is None or not rec.has_saturation_data:
                continue
            actual_sum += abs(rec.vds)  # type: ignore[arg-type]
            seen.append(rec.short_name)
            if rec.status == "TRIODE":
                triodes.append(rec.short_name)

        if have_budget:
            residual = expected_budget - actual_sum
            kvl_consistent = abs(residual) < tolerance
            ok = kvl_consistent and not triodes
        else:
            # Self-consistent mode: KVL is tautologically zero, but
            # any triode device still counts as a violation.
            residual = 0.0
            expected_budget = actual_sum
            ok = not triodes

        results.append(KVLViolation(
            stack_kind=stack.kind,
            top_node=stack.top_node,
            bottom_node=stack.bottom_node,
            expected_vds_sum=expected_budget,
            actual_vds_sum=actual_sum,
            residual=residual,
            ok=ok,
            devices_seen=seen,
            triode_devices=triodes,
        ))

    return results


# ─── Inter-run delta tracking ──────────────────────────────────


@dataclass
class DeltaEntry:
    """One device's between-run change.

    ``kind`` ∈ {vds, margin, status}. For vds and margin entries,
    ``prev`` and ``curr`` are floats (volts); for status entries they
    are strings (the actual status tags). ``note`` captures special
    transitions, currently:

      * 'overshoot'        — |Δvds| ≥ overshoot_threshold (default 0.2V)
      * 'sat_to_triode'    — status went SAT → TRIODE
      * 'triode_to_sat'    — status went TRIODE → SAT (fix landed)
      * 'marginal_change'  — crossed the MARGINAL boundary
    """

    name: str
    kind: str
    prev: object              # float or str
    curr: object              # float or str
    delta: Optional[float] = None
    note: str = ""


@dataclass
class DeltaTable:
    """List of between-run deltas plus summary counters."""

    entries: list[DeltaEntry] = field(default_factory=list)
    overshoots: int = 0
    regressions: int = 0
    fixes: int = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


def compute_simulation_delta(
    prev_op: DeviceOpTable,
    curr_op: DeviceOpTable,
    *,
    vds_threshold: float = 0.020,
    margin_threshold: float = 0.020,
    overshoot_threshold: float = 0.200,
) -> DeltaTable:
    """Return entries for devices whose Vds, margin, or status changed.

    Joins prev and curr on the (lowercased) device path. Devices that
    appear in only one of the two tables are skipped silently —
    that's almost always a topology change between runs, and we'd
    rather not flood the delta report with phantom 'appeared' /
    'disappeared' entries.

    Thresholds default to 20 mV for both vds and margin; overshoot at
    200 mV matches the cirona heuristic for "large move, manual
    review before another sizing iteration."
    """
    out = DeltaTable()
    if prev_op is None or curr_op is None:
        return out
    if not prev_op.records or not curr_op.records:
        return out

    for path, curr in curr_op.records.items():
        prev = prev_op.records.get(path)
        if prev is None:
            continue

        # Status transition (computed before Vds delta so the note
        # surfaces even when |Δvds| < vds_threshold).
        prev_status = prev.status
        curr_status = curr.status
        if prev_status != curr_status and prev_status != "UNKNOWN" and curr_status != "UNKNOWN":
            transition_note = ""
            if prev_status == "SAT" and curr_status == "TRIODE":
                transition_note = "sat_to_triode"
                out.regressions += 1
            elif prev_status == "TRIODE" and curr_status == "SAT":
                transition_note = "triode_to_sat"
                out.fixes += 1
            else:
                transition_note = "marginal_change"
            out.entries.append(DeltaEntry(
                name=path,
                kind="status",
                prev=prev_status,
                curr=curr_status,
                note=transition_note,
            ))

        # Vds delta.
        if prev.vds is not None and curr.vds is not None:
            d_vds = curr.vds - prev.vds
            if abs(d_vds) >= vds_threshold:
                note = "overshoot" if abs(d_vds) >= overshoot_threshold else ""
                if note:
                    out.overshoots += 1
                out.entries.append(DeltaEntry(
                    name=path,
                    kind="vds",
                    prev=prev.vds,
                    curr=curr.vds,
                    delta=d_vds,
                    note=note,
                ))

        # Margin delta.
        p_margin = prev.margin
        c_margin = curr.margin
        if p_margin is not None and c_margin is not None:
            d_margin = c_margin - p_margin
            if abs(d_margin) >= margin_threshold:
                note = ""
                if p_margin >= 0 and c_margin < 0:
                    note = "margin_to_negative"
                elif p_margin < 0 and c_margin >= 0:
                    note = "margin_recovered"
                out.entries.append(DeltaEntry(
                    name=path,
                    kind="margin",
                    prev=p_margin,
                    curr=c_margin,
                    delta=d_margin,
                    note=note,
                ))

    return out
