"""Phase 16 SPICE netlist parser stub.

Parse SPICE-style MOS device lines into a ParsedDevice dataclass, and
convert a parsed netlist to ``clay.layout.backend.DeviceSizing``
(``Dict[str, Tuple[float, float]]``) for the inverse-pipeline path
``existing SPICE netlist -> layout-aware sizing input``.

Supported dialects (best-effort; not a full SPICE parser):

* ngspice / spice3: ``M1 d g s b nmos W=4u L=0.5u M=4``
* hspice (SI float): ``M1 d g s b nmos W=4e-6 L=0.5e-6 m=4``
* spectre (lowercase keys): ``m1 d g s b nmos w=4u l=0.5u m=4``

Both 4-node (d g s b) and 5-node (d g s b extra) MOS instances are
accepted; the *extra* token is treated as part of the node list and is
not inspected.

Units handled in width/length values:
* trailing ``u`` / ``U`` / ``um`` / ``UM`` -> micrometres (no scaling)
* trailing ``n`` / ``N`` / ``nm`` -> nanometres (-> micrometres /1000)
* trailing ``m`` / ``M`` / ``mm`` -> millimetres (-> micrometres *1000)
* bare float / scientific (``4e-6``) -> SI metres (-> micrometres *1e6)

Multiplier ``M=`` / ``m=`` defaults to 1 when absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class SpiceParseError(ValueError):
    """Raised when a SPICE device line cannot be parsed."""


@dataclass
class ParsedDevice:
    """A parsed MOS device instance from a SPICE netlist line.

    ``nodes`` holds the token list between the instance name and the model
    name. For a standard ``M`` element this is ``[drain, gate, source,
    body]`` (4 tokens) or ``[drain, gate, source, body, extra]`` (5
    tokens). Width / length are normalised to micrometres regardless of
    the source dialect's unit suffix. ``multiplier`` defaults to 1 when
    the line carries no ``M=`` / ``m=`` parameter.
    """

    name: str
    nodes: List[str] = field(default_factory=list)
    model: str = ""
    w_um: float = 0.0
    l_um: float = 0.0
    multiplier: int = 1


# --- internal helpers ----------------------------------------------------


_KNOWN_MOS_MODELS = {
    "nmos",
    "pmos",
    "nfet",
    "pfet",
    "nch",
    "pch",
    # sky130 / gf180 style suffixes commonly found in netlists
    "nmos_1p8",
    "pmos_1p8",
    "sky130_fd_pr__nfet_01v8",
    "sky130_fd_pr__pfet_01v8",
}


def _parse_length(token: str) -> float:
    """Parse a SPICE numeric length token and return micrometres.

    Accepts:
      ``4u`` / ``4U`` / ``4um`` -> 4.0 um
      ``500n`` / ``500nm``     -> 0.5 um
      ``2m`` / ``2mm``         -> 2000.0 um
      ``4e-6``                 -> 4.0 um  (bare SI metres)
      ``0.5``                  -> 0.5 um? -- treated as bare SI metres
                                  *only* when it looks like a normalised
                                  float (i.e. contains 'e' or is < 1e-3);
                                  otherwise assumed already in micrometres.
    """
    tok = token.strip()
    if not tok:
        raise SpiceParseError("empty length token")

    lowered = tok.lower()

    # Strip 2-letter unit suffixes first (um / nm / mm) so we don't
    # confuse the 1-letter suffix logic below.
    if lowered.endswith("um"):
        return float(lowered[:-2])
    if lowered.endswith("nm"):
        return float(lowered[:-2]) / 1000.0
    if lowered.endswith("mm"):
        return float(lowered[:-2]) * 1000.0

    # 1-letter SPICE engineering suffixes.
    suffix_map = {
        "u": 1.0,          # micro -> um (identity)
        "n": 1.0 / 1000.0,  # nano  -> um
        "p": 1.0 / 1_000_000.0,  # pico  -> um (rarely used for W/L)
        "f": 1.0 / 1_000_000_000.0,  # femto
    }
    last = lowered[-1]
    if last in suffix_map and len(lowered) > 1 and (lowered[-2].isdigit() or lowered[-2] == "."):
        return float(lowered[:-1]) * suffix_map[last]

    # Special: trailing 'm' as 1-letter suffix is ambiguous in SPICE
    # (mega vs milli depending on dialect). We disambiguate the common
    # W/L case: a bare 'm' suffix here is treated as milli (-> mm).
    # Callers with hspice 'meg' should use scientific notation.
    if last == "m" and len(lowered) > 1 and (lowered[-2].isdigit() or lowered[-2] == "."):
        return float(lowered[:-1]) * 1000.0

    # No recognised suffix: assume SI metres (hspice style) and convert
    # to micrometres. A user passing ``0.5`` here gets 5e5 um which is
    # absurd for a MOS device; we treat any bare value >= 1e-3 as already
    # micrometres to give a sane fallback for hand-written test inputs.
    try:
        v = float(lowered)
    except ValueError as exc:
        raise SpiceParseError(f"cannot parse length token {token!r}: {exc}") from exc
    if abs(v) < 1e-3:
        return v * 1e6  # SI metres -> micrometres
    return v


def _looks_like_assignment(token: str) -> bool:
    return "=" in token


def _split_assignment(token: str) -> Tuple[str, str]:
    key, _, value = token.partition("=")
    return key.strip().lower(), value.strip()


# --- public API ----------------------------------------------------------


def parse_spice_device_line(line: str) -> ParsedDevice:
    """Parse one SPICE-style MOS device line.

    The first token must begin with ``M`` / ``m`` (otherwise we raise
    ``SpiceParseError`` so the caller can skip non-MOS lines).

    Recognised parameters: ``W`` (width), ``L`` (length), ``M`` /
    ``multiplier`` (multiplier). All other ``key=value`` tokens are
    silently ignored.
    """
    stripped = line.strip()
    if not stripped:
        raise SpiceParseError("empty line")

    # Drop trailing comments (after ';' or '$'), which some dialects support.
    for sep in (";", "$"):
        if sep in stripped:
            stripped = stripped.split(sep, 1)[0].strip()
            if not stripped:
                raise SpiceParseError("empty line after comment strip")

    tokens = stripped.split()
    if not tokens:
        raise SpiceParseError("no tokens")

    name = tokens[0]
    if not name or name[0].lower() != "m":
        raise SpiceParseError(
            f"not a MOS instance line (first token must start with 'M'): {name!r}"
        )

    # Split tokens into ``positional`` (no '=') vs ``params`` (key=value).
    rest = tokens[1:]
    positional: List[str] = []
    params: Dict[str, str] = {}
    for tok in rest:
        if _looks_like_assignment(tok):
            key, val = _split_assignment(tok)
            params[key] = val
        else:
            positional.append(tok)

    if len(positional) < 5:
        raise SpiceParseError(
            f"MOS line {name!r} needs at least 4 nodes + 1 model token, "
            f"got {len(positional)} positional tokens"
        )

    # Last positional is the model name. Everything before it is nodes.
    # Accept both 4-node (d g s b) and 5-node (d g s b extra) forms.
    model = positional[-1]
    nodes = positional[:-1]
    if len(nodes) not in (4, 5):
        raise SpiceParseError(
            f"MOS line {name!r}: expected 4 or 5 nodes before model, got {len(nodes)}"
        )

    # Light-touch model whitelist: don't hard-fail on unknown models, but
    # the caller can use the well-known list to filter via raises.
    # We do raise when the model token looks like a number (a sign of a
    # malformed line where the model name was omitted).
    try:
        float(model)
    except ValueError:
        pass
    else:
        raise SpiceParseError(
            f"MOS line {name!r}: model token {model!r} looks numeric, expected a model name"
        )

    if "w" not in params or "l" not in params:
        raise SpiceParseError(
            f"MOS line {name!r}: missing W= or L= parameter "
            f"(got params={sorted(params.keys())})"
        )

    w_um = _parse_length(params["w"])
    l_um = _parse_length(params["l"])

    mult_raw = params.get("m") or params.get("multiplier") or "1"
    try:
        multiplier = int(float(mult_raw))
    except ValueError as exc:
        raise SpiceParseError(
            f"MOS line {name!r}: multiplier {mult_raw!r} is not an integer"
        ) from exc
    if multiplier < 1:
        raise SpiceParseError(
            f"MOS line {name!r}: multiplier must be >= 1, got {multiplier}"
        )

    return ParsedDevice(
        name=name,
        nodes=nodes,
        model=model,
        w_um=w_um,
        l_um=l_um,
        multiplier=multiplier,
    )


def _is_directive_or_comment(line: str) -> bool:
    """Return True for SPICE comments and structural directives we skip."""
    s = line.strip()
    if not s:
        return True
    if s[0] in "*;#":
        return True
    if s.startswith("."):
        # .subckt / .ends / .model / .param / .global / .include etc.
        return True
    return False


def parse_spice_netlist(text: str) -> Dict[str, ParsedDevice]:
    """Parse a multi-line SPICE netlist, skipping comments / directives.

    Returns a dict keyed by device instance name. Non-MOS instance lines
    (``R*``, ``C*``, ``V*``, ``I*``, ``X*``, ...) are silently skipped —
    this stub only handles the ``M`` element since that's the only one
    the inverse pipeline currently consumes.

    Lines are joined on ``+`` continuation markers (a leading ``+`` token
    appends to the previous logical line, per standard SPICE).
    """
    logical_lines: List[str] = []
    for raw in text.splitlines():
        s = raw.rstrip()
        if not s.strip():
            continue
        if s.lstrip().startswith("+") and logical_lines:
            # Continuation: append (stripping the leading '+').
            logical_lines[-1] = logical_lines[-1] + " " + s.lstrip()[1:].strip()
        else:
            logical_lines.append(s)

    out: Dict[str, ParsedDevice] = {}
    for line in logical_lines:
        if _is_directive_or_comment(line):
            continue
        tokens = line.strip().split()
        if not tokens:
            continue
        first_char = tokens[0][0].lower()
        if first_char != "m":
            # Skip non-MOS instances (R, C, V, I, X, etc.). This stub
            # only consumes M elements.
            continue
        try:
            dev = parse_spice_device_line(line)
        except SpiceParseError:
            # Be tolerant: malformed MOS lines are skipped rather than
            # aborting the whole netlist parse, so the inverse pipeline
            # can salvage partial sizings.
            continue
        out[dev.name] = dev
    return out


def to_device_sizing(parsed: Dict[str, ParsedDevice]) -> Dict[str, Tuple[float, float]]:
    """Convert a parsed netlist to ``clay.layout.backend.DeviceSizing``.

    The multiplier is dropped (the layout pipeline currently models
    M=N by emitting N fingers via ``SymbolicLayout.finger_count`` rather
    than scaling W). Returned dict shape:

        {device_name: (w_um, l_um), ...}

    matching the type alias ``DeviceSizing = Dict[str, Tuple[float, float]]``
    in ``clay/layout/backend.py``.
    """
    return {name: (dev.w_um, dev.l_um) for name, dev in parsed.items()}
