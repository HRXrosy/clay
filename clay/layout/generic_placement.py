"""Generic, intent-driven device PLACER for Clay core.
This module is the GENERIC front-end the router builds against. It takes a parsed
:class:`clay.ir.CellDef` plus its :class:`clay.layout.intent.LayoutIntent` and
COMPOSES the existing R185 solver pipeline
(:mod:`clay.layout.annotate_to_solver`) to place the cell's devices,
respecting the symmetry / matching / rail intent declared in the ``.clay``
source. It does NOT reimplement the MILP solver, the foldable-device geometry, or
the quality scorer -- it wires them together behind one stable interface and
extracts per-device terminal coordinates the router needs.
What it composes (verbatim, imported -- never edited here)
==========================================================
* :func:`annotate_to_solver.annotations_to_solver_constraints` -- translates
  the parsed ``@symmetric`` / ``@matching`` annotations into the solver's
  :class:`SolverConstraints` (the declared mirror pairs + axis + inferred rails).
* :func:`annotate_to_solver._build_devices_from_clay` -- builds one
  DRC-legal-by-construction gf180 :class:`FoldableDevice` per MOS, from the W / L /
  finger count carried on the cell.
* :func:`annotate_to_solver.solve_placement_from_constraints` -- the scipy
  HiGHS MILP that places devices about a single vertical axis with the declared
  symmetry pairs mirror-constrained and NMOS / PMOS rail-banded.
* :func:`annotate_to_solver.score_placement` -- runs the REAL R172
  symmetry / common-centroid / rail rules on the solved bboxes.
The only NEW logic here is GENERIC (one uniform rule for every circuit, no
per-circuit special-casing):
  1. building an :class:`AnnotatedCell` from a parsed ``CellDef`` (so the placer
     works on the four production ``design.clay`` cells whose devices are sized by
     an external generator, not inside the ``.clay`` text);
  2. a uniform device-class inference (device-name prefix + functional-group role)
     and a documented default-geometry fallback when the source carries no W / L;
  3. extracting the per-device terminal (x, y) for s / d / g (and bulk) from each
     placed foldable device's rail / contact bboxes, and the cell's rail geometry
     from the device extents -- the :class:`Placement` interface the router needs.
Honesty contract
----------------
* The placement is the REAL MILP solver's output; the symmetry / centroid scores
  are the REAL R172 measurements on the solved bboxes.
* When the parsed cell carries sized devices (``cd.devices``), their real W / L /
  nf drive the geometry. When it does not (the four production cells size their
  devices in an external Python generator, NOT in the ``.clay``), the placer falls
  back to a UNIFORM default geometry and reports ``used_default_geometry=True`` --
  it never fabricates per-circuit sizes. The symmetry score is unaffected by the
  absolute geometry (it is a relative residual), so the structural placement claim
  is honest either way.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from weakref import WeakKeyDictionary

from clay.ir import CellDef
from clay.layout.annotations import AnnotatedCell
from clay.layout.intent import LayoutIntent

from .annotate_to_solver import (
    PlacementQuality,
    PlacementSolution,
    SolverConstraints,
    _build_devices_from_clay,
    _dev_gap_from_tech,
    _heights,
    _half_widths,
    _pair_gap,
    _resolve_tech,
    _row_plan,
    annotations_to_solver_constraints,
    kind_polarity_oxide,
    score_placement,
    solve_placement_from_constraints,
)
from .annotate_to_solver import PlacedDevice as _SolverPlacedDevice
from .backend import _mirror_x, _translate
from .physical_macro_contracts import PhysicalMacroProjectionError
from .nonrow_macro_plan import RigidNonmosMacroProjectionError
from .place.row_replay_session import row_phase_call, row_replay_phase, row_replay_invocation, bind_current_row_materialization

Bbox = Tuple[int, int, int, int]
_MAX_ROW_MASTER_BLOCKS = 256


class _PhysicalMacroScopeToken:
    """Identity-only capability minted at canonical graph compilation."""

    __slots__ = ("__weakref__",)


_PHYSICAL_MACRO_SCOPE_REGISTRY: WeakKeyDictionary = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class _PhysicalMacroScopeAuthority:
    graph_sha256: str
    compiler_binding: Optional[Tuple[str, ...]] = None
    row_source_binding: Optional[Tuple[object, ...]] = None


def _mint_physical_macro_scope(graph_sha256: str) -> object:
    token = _PhysicalMacroScopeToken()
    _PHYSICAL_MACRO_SCOPE_REGISTRY[token] = _PhysicalMacroScopeAuthority(
        str(graph_sha256)
    )
    return token


def _scope_digest(value: object, field: str) -> str:
    if value is None:
        return ""
    digest = getattr(value, "sha256", None)
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise PhysicalMacroProjectionError(f"MACRO_SCOPE_{field}_INVALID")
    return digest


def _physical_macro_compiler_binding_values(
    *,
    row_macro_contracts: object,
    physical_macro_contracts: object,
    physical_macro_compile_inputs: object,
    rigid_nonmos_macro_plans: object,
    rigid_nonmos_macro_inputs: object,
    rigid_nonmos_macro_interfaces: object,
) -> Tuple[str, ...]:
    return (
        _scope_digest(row_macro_contracts, "ROW_CONTRACTS"),
        _scope_digest(
            physical_macro_contracts, "SINGLETON_CONTRACTS"
        ),
        _scope_digest(
            physical_macro_compile_inputs, "SINGLETON_INPUTS"
        ),
        _scope_digest(rigid_nonmos_macro_plans, "RIGID_PLANS"),
        _scope_digest(rigid_nonmos_macro_inputs, "RIGID_INPUTS"),
        _scope_digest(
            rigid_nonmos_macro_interfaces, "RIGID_INTERFACES"
        ),
    )


def _physical_macro_compiler_binding(placement: object) -> Tuple[str, ...]:
    return _physical_macro_compiler_binding_values(
        row_macro_contracts=getattr(placement, "row_macro_contracts", None),
        physical_macro_contracts=getattr(
            placement, "physical_macro_contracts", None
        ),
        physical_macro_compile_inputs=getattr(
            placement, "physical_macro_compile_inputs", None
        ),
        rigid_nonmos_macro_plans=getattr(
            placement, "rigid_nonmos_macro_plans", None
        ),
        rigid_nonmos_macro_inputs=getattr(
            placement, "rigid_nonmos_macro_inputs", None
        ),
        rigid_nonmos_macro_interfaces=getattr(
            placement, "rigid_nonmos_macro_interfaces", None
        ),
    )


def _seal_physical_macro_scope(
    token: object,
    graph_sha256: str,
    compiler_binding: Tuple[str, ...],
    *,
    row_program: object = None,
    row_materialization: object = None,
) -> None:
    authority = _PHYSICAL_MACRO_SCOPE_REGISTRY.get(token)
    if (
        type(token) is not _PhysicalMacroScopeToken
        or type(authority) is not _PhysicalMacroScopeAuthority
        or authority.graph_sha256 != graph_sha256
        or authority.compiler_binding is not None
        or type(compiler_binding) is not tuple
        or len(compiler_binding) != 6
    ):
        raise PhysicalMacroProjectionError("MACRO_SCOPE_TOKEN_MISMATCH")
    for digest in compiler_binding:
        if digest and (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise PhysicalMacroProjectionError("MACRO_SCOPE_BINDING_INVALID")
    row_source_binding = None
    projection = getattr(row_materialization, "source_projection", None)
    if projection is not None:
        from clay.layout.place.row_replay_session import require_current_row_materialization
        from clay.layout.place.row_source_projection import require_row_source_projection

        require_row_source_projection(projection, program=row_program)
        require_current_row_materialization(row_materialization, program=row_program)
        row_source_binding = (
            projection, row_program, row_materialization,
            projection.sha256, row_program.sha256, row_materialization.certificate.sha256,
        )
    _PHYSICAL_MACRO_SCOPE_REGISTRY[token] = _PhysicalMacroScopeAuthority(
        graph_sha256, compiler_binding, row_source_binding
    )


def _verify_physical_macro_scope(
    token: object,
    graph_sha256: str,
    *,
    compiler_binding: Optional[Tuple[str, ...]] = None,
) -> None:
    authority = _PHYSICAL_MACRO_SCOPE_REGISTRY.get(token)
    if (
        type(token) is not _PhysicalMacroScopeToken
        or type(authority) is not _PhysicalMacroScopeAuthority
        or authority.graph_sha256 != graph_sha256
        or (
            compiler_binding is not None
            and authority.compiler_binding != compiler_binding
        )
    ):
        raise PhysicalMacroProjectionError("MACRO_SCOPE_TOKEN_MISMATCH")


def verify_rigid_nonmos_compiler_authority(
    token: object,
    graph_sha256: str,
    *,
    plans: object,
    compile_inputs: object,
    interfaces: object,
) -> bool:
    """Verify aggregate compiler metadata against the pre-solver scope seal."""

    try:
        _verify_physical_macro_scope(token, graph_sha256)
        authority = _PHYSICAL_MACRO_SCOPE_REGISTRY.get(token)
        current = (
            _scope_digest(plans, "RIGID_PLANS"),
            _scope_digest(compile_inputs, "RIGID_INPUTS"),
            _scope_digest(interfaces, "RIGID_INTERFACES"),
        )
        if (
            type(authority) is not _PhysicalMacroScopeAuthority
            or authority.compiler_binding is None
            or authority.compiler_binding[3:] != current
        ):
            raise PhysicalMacroProjectionError("MACRO_SCOPE_TOKEN_MISMATCH")
    except PhysicalMacroProjectionError as exc:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_SCOPE_TOKEN_MISMATCH"
        ) from exc
    return True


def verify_row_source_compiler_authority(placement, canonical_source_cell, tech):
    """Replay a projected row source and retain its exact pre-solver materialization."""
    from clay.layout.place.row_replay_session import (current_row_source_program,
        require_current_row_materialization)
    from clay.layout.place.row_source_projection import (
        RowSourceProjectionError,
        require_row_source_projection,
    )

    if type(placement) is not Placement:
        raise RowSourceProjectionError("ROW_SOURCE_PLACEMENT_TYPE")
    token = placement.physical_macro_scope_token
    if type(token) is not _PhysicalMacroScopeToken:
        raise RowSourceProjectionError("ROW_SOURCE_SCOPE_MISSING")
    authority = _PHYSICAL_MACRO_SCOPE_REGISTRY.get(token)
    if type(authority) is not _PhysicalMacroScopeAuthority or authority.row_source_binding is None:
        raise RowSourceProjectionError("ROW_SOURCE_SCOPE_MISSING")
    materialization, program = placement.row_materialization, placement.row_program
    projection = getattr(materialization, "source_projection", None)
    binding = authority.row_source_binding
    if projection is not binding[0] or program is not binding[1] or materialization is not binding[2]:
        raise RowSourceProjectionError("ROW_SOURCE_COMPILER_IDENTITY")
    require_row_source_projection(projection, program=program)
    graph = placement.physical_macro_graph
    if (graph is None or graph.sha256 != authority.graph_sha256
            or _physical_macro_compiler_binding(placement) != authority.compiler_binding):
        raise RowSourceProjectionError("ROW_SOURCE_SCOPE_MISMATCH")
    if (projection.sha256, program.sha256, materialization.certificate.sha256) != binding[3:]:
        raise RowSourceProjectionError("ROW_SOURCE_COMPILER_CONTENT")
    replayed_program = current_row_source_program(projection, canonical_source_cell, tech,
        program=program, materialization=materialization)
    if replayed_program.to_canonical_bytes() != program.to_canonical_bytes():
        raise RowSourceProjectionError('ROW_SOURCE_COMPILER_REPLAY_MISMATCH')
    require_current_row_materialization(materialization, program=program, tech_id=tech.pdk_id)
    return replayed_program, materialization

# Uniform default geometry used ONLY when the parsed cell carries no W / L (the
# production cells size devices in an external generator). One value for every
# device of a kind -- never a per-circuit table.
_DEFAULT_W_UM = 4.0
_DEFAULT_L_UM = 0.5
_DEFAULT_NF = 2


from clay.layout.source_bjt_materialization import BjtMaterializationUnavailable
from clay.layout.nonmos_placement_inputs import collect_nonmos_placement_inputs, native_member_geometry


# === generic source-netlist (.cir) sizing reader =============================
#
# The four production cells size their devices in the bound SPICE netlist
# (``<circuit>_gf180.cir`` / source_spice), NOT in the ``.clay`` text. Every
# circuit's ``.cir`` carries the per-device W / L / m on each transistor card,
# so the REAL design geometry is recoverable GENERICALLY by parsing the .cir and
# matching each card to a placer device by NAME. No per-circuit table: the same
# card grammar (plain ``M<name> d g s b <model> W= L= m=`` and the subckt-wrapped
# ``x<name> d g s b <nmos_3p3|pmos_3p3> W= L= m=``) covers all four.

# MOS model tokens any gf180 / vpdk / external_pdk / external_pdk .cir names a transistor with
# (recognised by substring so nch/pch/nmos_3p3/pmos_3p3/nfet_03v3/n33/p18/... all
# lower the same way). The n33/p33 (3.3V) and n18/p18 (1.8V) families are the external_pdk
# core devices; n25/p25/n12/p12 cover the common external_pdk 2.5V/1.2V families.
_NMOS_MODEL_HINTS = ("nmos", "nch", "nfet", "n33", "n18", "n25", "n12")
_PMOS_MODEL_HINTS = ("pmos", "pch", "pfet", "p33", "p18", "p25", "p12")


def _cir_model_kind(model: str) -> Optional[str]:
    low = (model or "").lower()
    if any(h in low for h in _PMOS_MODEL_HINTS):
        return "pmos"
    if any(h in low for h in _NMOS_MODEL_HINTS):
        return "nmos"
    # Fallback: a schematic-library device model with no PDK substring hint but a
    # p/n prefix (the external_pdk auCdl 'PD' = p-device, 'ND' = n-device convention).
    # This mirrors the mature extract-path recogniser
    # (clay.layout.extract.netlist_intent._model_kind), keeping the frontend and
    # extract MOS parsers in parity so a real external_pdk reference netlist is not silently
    # dropped to a 0-MOS decomposition. Only reached for M/X cards with 4 nodes
    # (parse_cir_mos_records gate), so a passive R/C model never lands here.
    if low.startswith("p"):
        return "pmos"
    if low.startswith("n"):
        return "nmos"
    return None


def _cir_params(text: str) -> Dict[str, str]:
    """Collect every ``.param NAME = VALUE`` (comment-stripped) from a .cir.

    Handles MANY assignments per ``.param`` line (``.param Wn=10u Ln=0.5u``) and
    both ``k=v`` and ``k = v`` spacing -- real netlists pack several per line. The
    value is a brace expression ``{...}`` or a non-space token; capturing only the
    first assignment (the old behaviour) silently dropped the rest, which made
    later device cards unresolvable and skipped."""
    params: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("$", 1)[0].strip()
        if line[:6].lower() != ".param":
            continue
        for m in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{[^}]*\}|[^\s=]+)", line[6:]
        ):
            params[m.group(1)] = m.group(2)
    return params


def _cir_resolve(token: str, params: Dict[str, str], _depth: int = 0) -> str:
    """Resolve a W/L/m token through ``.param`` (one level of brace + name)."""
    t = token.strip()
    if t.startswith("{") and t.endswith("}"):
        t = t[1:-1].strip()
    if t in params and _depth < 8:
        return _cir_resolve(params[t], params, _depth + 1)
    return t


def _cir_num_um(token: str, params: Dict[str, str]) -> Optional[float]:
    """Parse a geometry token (``20`` / ``3.3u`` / ``{W_DIFF}``) to microns.

    A bare number is microns (the gf180 decks set ``.option scale=1.0u`` so an
    unsuffixed W/L is already in um); a ``u``/``um`` suffix is um; ``n`` is nm.
    Returns ``None`` when the token does not resolve to a number.
    """
    t = _cir_resolve(token, params)
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)", t)
    if not m:
        return None
    val = float(m.group(1))
    suf = m.group(2).lower()
    if suf in ("", "u", "um"):
        return val
    if suf in ("n", "nm"):
        return val * 1e-3
    if suf in ("p", "pm"):
        return val * 1e-6
    return val


def _cir_int(token: str, params: Dict[str, str], default: int = 1) -> int:
    v = _cir_num_um(token, params)
    try:
        return max(1, int(round(v))) if v is not None else default
    except (TypeError, ValueError):
        return default


def _cir_device_name(raw: str) -> str:
    """Normalise a card instance name to the placer device label.

    A transistor written as a subckt instance ``x<name> ... <model>`` (two_stage /
    bandgap) carries the placer label in ``<name>`` (the ``x`` wrapper is dropped):
    ``xMP1`` -> ``MP1``, matching the functional-group device name in the .clay. A
    plain ``M<name>`` card (fc_ota) keeps its name verbatim.
    """
    n = raw.strip()
    if n[:1].lower() == "x":
        n = n[1:]
    return n


def parse_cir_mos_records(source_cir) -> Dict[str, Dict[str, object]]:
    """Parse a SPICE ``.cir`` into ``device_label -> {kind, W, L, nf, pins}``.

    GENERIC (no per-circuit branch): every line whose first token starts with
    ``M`` or ``x`` and whose model token maps to a MOS device (nch / pch /
    nmos_3p3 / pmos_3p3 / ...) is read as a transistor card; its 4 nodes
    (``d g s b``), W / L (in metres), and the ``m`` finger multiplier are resolved
    through ``.param``. The device label is the normalised card name (``xMP1`` ->
    ``MP1``) so it matches the placer's functional-group device name. Cards whose
    model is not a MOS (resistor R / cap C / BJT vpnp / instance) are skipped --
    the placer places MOS only. Returns an empty map when the file is unreadable.
    """
    try:
        text = Path(source_cir).read_text(encoding="utf-8-sig")
    except (OSError, TypeError):
        return {}
    params = _cir_params(text)
    out: Dict[str, Dict[str, object]] = {}
    for raw in text.splitlines():
        line = raw.split("$", 1)[0].strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        head = parts[0]
        if head[:1].upper() not in ("M", "X"):
            continue
        # split nodes / model / params: params begin at the first key=value tok.
        kv_start = len(parts)
        for i, tok in enumerate(parts[1:], start=1):
            if "=" in tok:
                kv_start = i
                break
        head_tokens = parts[1:kv_start]
        if len(head_tokens) < 2:
            continue
        model = head_tokens[-1]
        nodes = head_tokens[:-1]
        kind = _cir_model_kind(model)
        if kind is None or len(nodes) < 4:
            continue  # not a MOS card (BJT / instance / passive) -> skip
        kv = {
            k.upper(): v
            for k, v in re.findall(
                r"\b(W|L|m)\s*=\s*(\{[^}]*\}|[A-Za-z0-9_.+\-]+)", " ".join(parts[kv_start:]),
                flags=re.IGNORECASE,
            )
        }
        w_um = _cir_num_um(kv.get("W", ""), params)
        l_um = _cir_num_um(kv.get("L", ""), params)
        if w_um is None or l_um is None:
            continue  # geometry not resolvable -> leave to the default fallback
        nf = _cir_int(kv.get("M", "1"), params, default=1)
        out[_cir_device_name(head)] = {
            "kind": kind,
            # Preserve the source device family independently of its coarse
            # polarity.  Mixed-voltage front-ends need n18/p18 versus n33/p33
            # to survive into Clay; the PDK adapter decides the target model.
            "model": model,
            "W": w_um * 1e-6,
            "L": l_um * 1e-6,
            "nf": nf,
            # gf180 / vpdk MOS node order on the card is d g s b.
            "pins": {"d": nodes[0], "g": nodes[1], "s": nodes[2], "b": nodes[3]},
        }
    return out


def parse_cir_passive_records(source_cir) -> Dict[str, Dict[str, object]]:
    """Parse a SPICE ``.cir`` into ``device_label -> {kind, value, pins}`` for the
    2-terminal PASSIVE devices (resistors + MIM capacitors).

    GENERIC (no per-circuit branch): every line whose first token starts with
    ``R`` and has >= 2 nodes + a value is a resistor card (value in ohms); every
    line starting with ``C`` and >= 2 nodes + a value is a capacitor card (value
    in farads). Both resolve the value through ``.param`` + the engineering-suffix
    table. The 2 nodes are the ``p`` / ``n`` pins. The resistor/cap placer + router
    consume these exactly as the MOS records feed the MOS placer. Returns an empty
    map when the file is unreadable."""
    try:
        text = Path(source_cir).read_text(encoding="utf-8-sig")
    except (OSError, TypeError):
        return {}
    params = _cir_params(text)
    out: Dict[str, Dict[str, object]] = {}
    for raw in text.splitlines():
        line = raw.split("$", 1)[0].strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        head = parts[0]
        first = head[:1].upper()
        if len(parts) < 4 or first not in ("R", "C"):
            continue
        n1, n2, value = parts[1], parts[2], parts[3]
        # The engineering-suffix table is shared (k/meg/u/p/f...); a resistor value
        # is ohms, a capacitor value is farads -- same parse, different unit.
        magnitude = _res_value_to_ohms(value, params)
        if magnitude is None:
            continue
        # The SPICE inline multiplier `m=N` means N parallel unit devices. Carry
        # it through (default 1) so the per-unit value scales to its effective
        # magnitude downstream (cap C = value*M, res R = value/M).
        mult = 1
        for tok in parts[4:]:
            k, _, v = tok.partition("=")
            if "=" in tok and k.strip().lower() == "m":
                try:
                    mult = max(1, int(round(float(_cir_resolve(v.strip(), params)))))
                except (TypeError, ValueError):
                    mult = 1
                break
        if first == "R":
            out[head] = {
                "kind": "res",
                "value": float(magnitude),
                "pins": {"p": n1, "n": n2},
                "M": mult,
            }
        else:  # C -- MIM capacitor card (value in farads)
            out[head] = {
                "kind": "cap",
                "value": float(magnitude),
                "pins": {"p": n1, "n": n2},
                "M": mult,
            }
    return out


_RES_SUFFIX = {
    "": 1.0, "t": 1e12, "g": 1e9, "meg": 1e6, "x": 1e6, "k": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}


def _res_value_to_ohms(token: str, params: Dict[str, str]) -> Optional[float]:
    """Resolve a resistor value token (``510k`` / ``{Rfb}`` / ``2.2meg``) to ohms.

    Engineering suffix per SPICE convention (``k`` = 1e3, ``meg`` = 1e6, ``m`` =
    1e-3). Distinct from ``_cir_num_um`` (which is for geometry tokens in um)."""
    t = _cir_resolve(token, params)
    m = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)", t.strip()
    )
    if not m:
        return None
    val = float(m.group(1))
    suf = m.group(2).lower()
    if suf in ("ohm", "ohms", "r"):
        suf = ""
    mult = _RES_SUFFIX.get(suf)
    if mult is None:
        # an unrecognised suffix (e.g. 'kohm'): strip a trailing 'ohm'/'r'
        if suf.endswith("ohm"):
            mult = _RES_SUFFIX.get(suf[:-3], None)
        elif suf.endswith("r"):
            mult = _RES_SUFFIX.get(suf[:-1], None)
    if mult is None:
        return None
    return val * mult


def parse_cir_device_sizing(source_cir) -> Dict[str, Dict[str, float]]:
    """``device_label -> {kind, W, L, nf}`` (SI / count) for every MOS card.

    Thin projection of :func:`parse_cir_mos_records` (drops the ``pins``) so the
    placer's sizing reader stays a simple sizing-only view.
    """
    records = parse_cir_mos_records(source_cir)
    return {
        lbl: {k: rec[k] for k in ("kind", "W", "L", "nf") if k in rec}
        for lbl, rec in records.items()
    }


# === the router-facing interface =============================================


@dataclass(frozen=True)
class PlacedDevice:
    """One placed device: its kind, absolute bbox, terminal pins, and the placed
    (translated / mirrored) layer polygons.

    ``terminals`` maps a pin name (``"s"`` / ``"d"`` / ``"g"`` and, for a PMOS,
    ``"b"``) to the (x, y) CENTRE of that terminal's rail / contact at the solved
    position -- the point the router attaches a wire to.

    ``term_layers`` (optional) maps a pin name to the GDS ``(layer, datatype)`` of
    the ROUTABLE metal the terminal's pad sits on -- the layer the router must START
    its riser from. For a MOS / poly resistor every terminal pad is on Metal1, so
    this map is empty (``{}``) and the router uses its default Metal1 entry (so MOS
    cells are byte-unchanged). A MIM capacitor's plate pads live on a HIGHER metal
    (gf180: both plate handles on Metal2): the router reads the entry layer here and
    reaches the terminal ON that metal, never dropping a Via1 onto it -- which on a
    MIM bottom plate would fire MIM.10 ("no Via1 touching MIM bottom plate Metal2").
    """

    label: str
    kind: str
    bbox: Bbox
    terminals: Dict[str, Tuple[int, int]]
    polys: list
    term_layers: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Empty for an independently materialized device.  A consumed RowProgram
    # member is a terminal-only view into this one stored macro owner; its own
    # ``polys`` must then be empty so emit cannot duplicate the rigid artwork.
    geometry_owner: str = ""
    # Fixed-cell banks expose every physical terminal access. Ordinary devices
    # leave this empty and continue to use the single ``terminals`` point.
    terminal_accesses: Dict[str, Tuple[Tuple[int, int], ...]] = field(default_factory=dict)
    # Orientation carried by the materialized DeviceGeom.  This is final
    # placement-certificate evidence; an empty value is unknown and must not be
    # replaced by a source/default guess in the intent checker.
    orientation: str = ""
    # Independent materialized-view witness for the canonical topology graph.
    # Every place_generic view is stamped once; later projections preserve it.
    physical_macro_graph_sha256: str = ""
    # Exact fixed-cell authority carried into compact/routing. Empty keeps the
    # historical parameter-based MOS rebuild for every ordinary device.
    fixed_pcell: Dict[str, object] = field(default_factory=dict)
    # Evaluated channel occurrences in centroid-relative um. Populated for a
    # singleton discrete MOS array (direct or group-bound) so the routing bridge can
    # preserve its actual unit grid instead of reconstructing from scalar M.
    physical_occurrences: Tuple[Tuple[float, float, str], ...] = ()
    # Resolved fixed-master occurrences in the same centroid-relative frame.
    # This stays separate because one evaluated master can contain multiple
    # electrical fingers while strict placement compares the master pattern.
    resolved_master_occurrences: Tuple[Tuple[float, float, str], ...] = ()
    well_tap_isolated: bool = False
    # Immutable native passive artwork/terminal dependency. Appended to preserve
    # existing positional construction. Never a serialized source certificate.
    materialized_passive_access: object = None
    bjt_unit_occurrences: object = None
    source_resistor_assembly: object = None


@dataclass(frozen=True, slots=True)
class RowMacroBlockBinding:
    """One compiled row entry bound to exactly one rigid final-master block."""

    entry_id: str
    entry_sha256: str
    materialization_sha256: str
    artwork_sha256: str
    owner: str
    consumed_devices: Tuple[str, ...]
    bbox_nm: Bbox

    def __post_init__(self) -> None:
        if type(self.entry_id) is not str or not self.entry_id:
            raise ValueError("entry_id must be a non-empty exact string")
        if (
            type(self.entry_sha256) is not str
            or len(self.entry_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.entry_sha256)
        ):
            raise ValueError("entry_sha256 must be one canonical sha256")
        for field_name in ("materialization_sha256", "artwork_sha256"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{field_name} must be one canonical sha256")
        if type(self.owner) is not str or not self.owner:
            raise ValueError("row macro owner must be a non-empty exact string")
        members = _bounded_row_master(
            self.consumed_devices, _MAX_ROW_MASTER_BLOCKS, "consumed_devices"
        )
        if (
            not members
            or any(type(item) is not str or not item for item in members)
            or len(set(members)) != len(members)
        ):
            raise ValueError("row macro consumed_devices must be unique exact strings")
        object.__setattr__(self, "consumed_devices", tuple(sorted(members)))
        _validate_row_master_bbox(self.bbox_nm, "row macro bbox")


@dataclass(frozen=True, slots=True)
class RowFinalPlacementCertificate:
    """Closed row-macro/density evidence for one canonical Placement snapshot.

    Consumed active and dummy devices occur only inside ``macro_bindings``.  They
    are forbidden in ``free_device_blocks`` so the final master cannot quietly
    place a row member twice (once in its rigid macro and once as a free tile).
    """

    schema: str
    program_sha256: str
    materialization_sha256: str
    macro_bindings: Tuple[RowMacroBlockBinding, ...]
    free_device_blocks: Tuple[Tuple[str, Bbox], ...]
    bbox_nm: Bbox
    occupied_area_nm2: int
    bbox_area_nm2: int
    void_area_nm2: int
    whitespace_area_nm2: int
    largest_empty_rectangle_nm2: int
    placed_artwork_sha256: str
    physical_macro_graph_sha256: str = ""
    row_macro_contracts_sha256: str = ""
    # Retained typed native geometry proof; canonical payload binds its digest.
    # Full artwork envelopes remain unchanged even when their layers interleave.
    native_geometry_binding: object = None
    # A separately captured complete non-MOS assembly changes collision views,
    # while the original free-member and density denominators remain intact.
    nonmos_assembly_binding: object = None
    # Two independent native row owners; the actual solver proof is replayed
    # at row seating before this certificate can be built.
    native_cross_row_cp_proof: object = None
    native_cross_row_final_proof: object = None
    native_same_row_final_proof: object = None
    native_row_cp_frame_proof: object = None
    native_row_final_transported: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "program_sha256",
            "materialization_sha256",
            "placed_artwork_sha256",
        ):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be one canonical sha256")
        digests = (
            self.physical_macro_graph_sha256,
            self.row_macro_contracts_sha256,
        )
        if bool(digests[0]) != bool(digests[1]):
            raise ValueError(
                "physical graph and row macro contract hashes must be bound together"
            )
        expected_schema = (
            "clay.row-final-placement-certificate.v9"
            if self.native_row_cp_frame_proof is not None else
            "clay.row-final-placement-certificate.v8"
            if self.native_same_row_final_proof is not None else
            "clay.row-final-placement-certificate.v7"
            if self.native_cross_row_final_proof is not None else
            "clay.row-final-placement-certificate.v6"
            if self.native_cross_row_cp_proof is not None else
            "clay.row-final-placement-certificate.v5"
            if self.nonmos_assembly_binding is not None else
            "clay.row-final-placement-certificate.v4"
            if self.native_geometry_binding is not None else
            "clay.row-final-placement-certificate.v3"
            if digests[0]
            else "clay.row-final-placement-certificate.v2"
        )
        if self.schema != expected_schema:
            raise ValueError("certificate schema and macro hashes are inconsistent")
        if self.native_geometry_binding is not None and not digests[0]:
            raise ValueError("native row geometry requires the physical macro contract hashes")
        if self.native_cross_row_cp_proof is not None:
            from clay.layout.place.cross_row_matching import NativeCrossRowCpProof
            if type(self.native_cross_row_cp_proof) is not NativeCrossRowCpProof:
                raise TypeError("row certificate requires exact native cross-row CP proof")
            self.native_cross_row_cp_proof.__post_init__()
        if self.native_cross_row_final_proof is not None:
            from clay.layout.place.cross_row_matching import NativeCrossRowFinalProof
            if (type(self.native_cross_row_final_proof) is not NativeCrossRowFinalProof
                    or self.native_cross_row_final_proof.cp_proof != self.native_cross_row_cp_proof):
                raise TypeError("row certificate final cross-row proof differs from CP")
            self.native_cross_row_final_proof.__post_init__()
        if self.native_same_row_final_proof is not None:
            from clay.layout.place.native_same_row_matching import NativeSameRowFinalProof
            if type(self.native_same_row_final_proof) is not NativeSameRowFinalProof:
                raise TypeError("row certificate requires exact native same-row proof")
            self.native_same_row_final_proof.__post_init__()
        if self.native_row_cp_frame_proof is not None:
            from clay.layout.place.native_row_cp_frames import NativeRowCpFrameProof
            if type(self.native_row_cp_frame_proof) is not NativeRowCpFrameProof:
                raise TypeError('row certificate requires exact native row CP frame proof')
            self.native_row_cp_frame_proof.__post_init__()
        if type(self.native_row_final_transported) is not bool or (self.native_row_final_transported and self.native_row_cp_frame_proof is None):
            raise ValueError('native row final transport lacks CP frame proof')
        for field_name, value in zip(
            ("physical_macro_graph_sha256", "row_macro_contracts_sha256"),
            digests,
        ):
            if value and (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be one canonical sha256")

        if type(self.macro_bindings) is not tuple:
            raise ValueError("macro_bindings must be an exact tuple")
        bindings = _bounded_row_master(
            self.macro_bindings, _MAX_ROW_MASTER_BLOCKS, "macro_bindings"
        )
        if any(type(item) is not RowMacroBlockBinding for item in bindings):
            raise ValueError(
                "macro_bindings must contain exact RowMacroBlockBinding values"
            )
        if bindings != tuple(sorted(bindings, key=lambda item: item.owner)):
            raise ValueError("macro_bindings must use canonical owner order")
        if len({item.owner for item in bindings}) != len(bindings):
            raise ValueError("one row macro owner may bind only one rigid block")
        if len({item.entry_sha256 for item in bindings}) != len(bindings):
            raise ValueError("one row macro hash may bind only one rigid block")
        if len({item.entry_id for item in bindings}) != len(bindings):
            raise ValueError("one row entry id may bind only one rigid block")
        materialization_hashes = {item.materialization_sha256 for item in bindings}
        expected_materialization = (
            next(iter(materialization_hashes)) if materialization_hashes else "0" * 64
        )
        if len(materialization_hashes) > 1:
            raise ValueError("all row blocks must bind one materialization certificate")
        if self.materialization_sha256 != expected_materialization:
            raise ValueError(
                "materialization_sha256 differs from the derived macro binding"
            )
        consumed = [name for item in bindings for name in item.consumed_devices]
        if len(consumed) != len(set(consumed)):
            raise ValueError("a row device may be consumed by only one rigid block")

        if type(self.free_device_blocks) is not tuple:
            raise ValueError("free_device_blocks must be an exact tuple")
        free = _bounded_row_master(
            self.free_device_blocks,
            _MAX_ROW_MASTER_BLOCKS,
            "free_device_blocks",
        )
        rebuilt_free: List[Tuple[str, Bbox]] = []
        for item in free:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("free device blocks must be exact (name, bbox) tuples")
            name, bbox = item
            if type(name) is not str or not name:
                raise ValueError("free block name must be a non-empty exact string")
            rebuilt_free.append((name, _validate_row_master_bbox(bbox, "free device bbox")))
        if free != tuple(sorted(rebuilt_free)):
            raise ValueError("free_device_blocks must use canonical order")
        if len({name for name, _bbox in rebuilt_free}) != len(rebuilt_free):
            raise ValueError("free device block ids must be unique")
        if len(bindings) + len(rebuilt_free) > _MAX_ROW_MASTER_BLOCKS:
            raise ValueError(
                f"final row-master blocks exceed the {_MAX_ROW_MASTER_BLOCKS}-item physical cap"
            )
        duplicate = tuple(sorted(set(consumed) & {name for name, _bbox in rebuilt_free}))
        if duplicate:
            raise ValueError(
                f"consumed row member appears as duplicate free block {duplicate}"
            )
        blocks = [
            (item.owner, item.bbox_nm) for item in bindings
        ] + rebuilt_free
        if len({name for name, _bbox in blocks}) != len(blocks):
            raise ValueError("macro and free block ids must be globally unique")
        collision_blocks = _nonmos_row_collision_blocks(
            bindings, tuple(rebuilt_free), self.nonmos_assembly_binding,
        )
        if self.native_geometry_binding is not None:
            from clay.layout.place.row_native_geometry import RowNativeGeometryBinding

            if type(self.native_geometry_binding) is not RowNativeGeometryBinding:
                raise TypeError("row certificate requires exact native geometry binding")
            self.native_geometry_binding.require_row_blocks(
                macro_bindings=bindings, free_device_blocks=tuple(rebuilt_free),
            )
        else:
            for index, (left_name, left_bbox) in enumerate(collision_blocks):
                for right_name, right_bbox in collision_blocks[index + 1:]:
                    if _positive_bbox_overlap(left_bbox, right_bbox):
                        raise ValueError(
                            f"row certificate blocks overlap: {left_name}:{right_name}"
                        )

        expected_bbox = _row_blocks_bbox(blocks)
        if self.bbox_nm != expected_bbox:
            raise ValueError("row certificate differs from its derived bbox")
        expected_bbox_area = _bbox_area_nm2(expected_bbox)
        expected_occupied = _rect_union_area_nm2(
            [bbox for _name, bbox in blocks]
        )
        expected_whitespace = expected_bbox_area - expected_occupied
        expected_largest = _largest_empty_rectangle_nm2(
            expected_bbox, [bbox for _name, bbox in blocks]
        )
        derived_scalars = {
            "occupied_area_nm2": expected_occupied,
            "bbox_area_nm2": expected_bbox_area,
            "void_area_nm2": expected_whitespace,
            "whitespace_area_nm2": expected_whitespace,
            "largest_empty_rectangle_nm2": expected_largest,
        }
        for field_name, expected in derived_scalars.items():
            value = getattr(self, field_name)
            if type(value) is not int or value != expected:
                raise ValueError(
                    f"row certificate differs from derived {field_name}"
                )

    @classmethod
    def build(
        cls,
        *,
        program_sha256: str,
        macro_bindings,
        free_device_blocks,
        placed_artwork=(),
        physical_macro_graph_sha256: str = "",
        row_macro_contracts_sha256: str = "",
        native_geometry_binding: object = None,
        nonmos_assembly_binding: object = None,
        native_cross_row_cp_proof: object = None,
        native_cross_row_final_proof: object = None,
        native_same_row_final_proof: object = None,
        native_row_cp_frame_proof: object = None,
        native_row_final_transported: bool = False,
    ) -> "RowFinalPlacementCertificate":
        if (
            type(program_sha256) is not str
            or len(program_sha256) != 64
            or any(c not in "0123456789abcdef" for c in program_sha256)
        ):
            raise ValueError("program_sha256 must be one canonical sha256")
        bindings = _bounded_row_master(
            macro_bindings, _MAX_ROW_MASTER_BLOCKS, "macro_bindings"
        )
        if any(type(item) is not RowMacroBlockBinding for item in bindings):
            raise ValueError("macro_bindings must contain exact RowMacroBlockBinding values")
        bindings = tuple(
            RowMacroBlockBinding(
                entry_id=item.entry_id,
                entry_sha256=item.entry_sha256,
                materialization_sha256=item.materialization_sha256,
                artwork_sha256=item.artwork_sha256,
                owner=item.owner,
                consumed_devices=item.consumed_devices,
                bbox_nm=item.bbox_nm,
            )
            for item in bindings
        )
        if len({item.owner for item in bindings}) != len(bindings):
            raise ValueError("one row macro owner may bind only one rigid block")
        if len({item.entry_sha256 for item in bindings}) != len(bindings):
            raise ValueError("one row macro hash may bind only one rigid block")
        if len({item.entry_id for item in bindings}) != len(bindings):
            raise ValueError("one row entry id may bind only one rigid block")
        materialization_hashes = {item.materialization_sha256 for item in bindings}
        if len(materialization_hashes) > 1:
            raise ValueError("all row blocks must bind one materialization certificate")
        consumed = [name for item in bindings for name in item.consumed_devices]
        if len(consumed) != len(set(consumed)):
            raise ValueError("a row device may be consumed by only one rigid block")
        free: List[Tuple[str, Bbox]] = []
        for raw in _bounded_row_master(
            free_device_blocks, _MAX_ROW_MASTER_BLOCKS, "free_device_blocks"
        ):
            if type(raw) is not tuple or len(raw) != 2:
                raise ValueError("free device blocks must be exact (name, bbox) tuples")
            name, bbox = raw
            if type(name) is not str or not name:
                raise ValueError("free block name must be a non-empty exact string")
            _validate_row_master_bbox(bbox, "free device bbox")
            free.append((name, bbox))
        if len({name for name, _bbox in free}) != len(free):
            raise ValueError("free device block ids must be unique")
        if len(bindings) + len(free) > _MAX_ROW_MASTER_BLOCKS:
            raise ValueError(
                f"final row-master blocks exceed the {_MAX_ROW_MASTER_BLOCKS}-item physical cap"
            )
        duplicate = tuple(sorted(set(consumed) & {name for name, _bbox in free}))
        if duplicate:
            raise ValueError(f"consumed row member appears as duplicate free block {duplicate}")
        blocks = [(item.owner, item.bbox_nm) for item in bindings] + free
        if len({name for name, _bbox in blocks}) != len(blocks):
            raise ValueError("macro and free block ids must be globally unique")
        collision_blocks = _nonmos_row_collision_blocks(
            bindings, tuple(sorted(free)), nonmos_assembly_binding,
        )
        if native_geometry_binding is None:
            for index, (left_name, left_bbox) in enumerate(collision_blocks):
                for right_name, right_bbox in collision_blocks[index + 1:]:
                    if _positive_bbox_overlap(left_bbox, right_bbox):
                        raise RowFinalPlacementResidualError((
                            f"ROW_BLOCK_OVERLAP:{left_name}:{right_name}",
                        ))
        bbox = _row_blocks_bbox(blocks)
        bbox_area = _bbox_area_nm2(bbox)
        occupied = _rect_union_area_nm2([box for _name, box in blocks])
        whitespace = bbox_area - occupied
        largest = _largest_empty_rectangle_nm2(bbox, [box for _name, box in blocks])
        artwork_digest = _placed_artwork_sha256(list(placed_artwork))
        macro_digests = (
            physical_macro_graph_sha256,
            row_macro_contracts_sha256,
        )
        if bool(macro_digests[0]) != bool(macro_digests[1]):
            raise ValueError(
                "physical graph and row macro contract hashes must be bound together"
            )
        for field_name, value in zip(
            ("physical_macro_graph_sha256", "row_macro_contracts_sha256"),
            macro_digests,
        ):
            if value and (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be one canonical sha256")
        return cls(
            schema=(
                "clay.row-final-placement-certificate.v9"
                if native_row_cp_frame_proof is not None else
                "clay.row-final-placement-certificate.v8"
                if native_same_row_final_proof is not None else
                "clay.row-final-placement-certificate.v7"
                if native_cross_row_final_proof is not None else
                "clay.row-final-placement-certificate.v6"
                if native_cross_row_cp_proof is not None else
                "clay.row-final-placement-certificate.v5"
                if nonmos_assembly_binding is not None else
                "clay.row-final-placement-certificate.v4"
                if native_geometry_binding is not None else
                "clay.row-final-placement-certificate.v3"
                if physical_macro_graph_sha256
                else "clay.row-final-placement-certificate.v2"
            ),
            program_sha256=program_sha256,
            materialization_sha256=(
                next(iter(materialization_hashes)) if materialization_hashes else "0" * 64
            ),
            macro_bindings=tuple(sorted(bindings, key=lambda item: item.owner)),
            free_device_blocks=tuple(sorted(free)),
            bbox_nm=bbox,
            occupied_area_nm2=occupied,
            bbox_area_nm2=bbox_area,
            void_area_nm2=whitespace,
            whitespace_area_nm2=whitespace,
            largest_empty_rectangle_nm2=largest,
            placed_artwork_sha256=artwork_digest,
            physical_macro_graph_sha256=physical_macro_graph_sha256,
            row_macro_contracts_sha256=row_macro_contracts_sha256,
            native_geometry_binding=native_geometry_binding,
            nonmos_assembly_binding=nonmos_assembly_binding,
            native_cross_row_cp_proof=native_cross_row_cp_proof,
            native_cross_row_final_proof=native_cross_row_final_proof,
            native_same_row_final_proof=native_same_row_final_proof,
            native_row_cp_frame_proof=native_row_cp_frame_proof,
            native_row_final_transported=native_row_final_transported,
        )

    def to_canonical_bytes(self) -> bytes:
        if (self.native_geometry_binding is not None or self.nonmos_assembly_binding is not None
                or self.native_cross_row_cp_proof is not None
                or self.native_cross_row_final_proof is not None
                or self.native_same_row_final_proof is not None
                or self.native_row_cp_frame_proof is not None
                or self.schema.endswith((".v4", ".v5", ".v6", ".v7", ".v8", ".v9"))):
            self.__post_init__()
        payload = {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "materialization_sha256": self.materialization_sha256,
            "macro_bindings": tuple(
                (
                    item.entry_id, item.entry_sha256, item.materialization_sha256,
                    item.artwork_sha256, item.owner, item.consumed_devices, item.bbox_nm,
                )
                for item in self.macro_bindings
            ),
            "free_device_blocks": self.free_device_blocks,
            "bbox_nm": self.bbox_nm,
            "occupied_area_nm2": self.occupied_area_nm2,
            "bbox_area_nm2": self.bbox_area_nm2,
            "void_area_nm2": self.void_area_nm2,
            "whitespace_area_nm2": self.whitespace_area_nm2,
            "largest_empty_rectangle_nm2": self.largest_empty_rectangle_nm2,
            "placed_artwork_sha256": self.placed_artwork_sha256,
        }
        if self.physical_macro_graph_sha256:
            payload["physical_macro_graph_sha256"] = (
                self.physical_macro_graph_sha256
            )
            payload["row_macro_contracts_sha256"] = (
                self.row_macro_contracts_sha256
            )
        if self.native_geometry_binding is not None:
            payload["native_geometry_binding_sha256"] = self.native_geometry_binding.sha256
        if self.nonmos_assembly_binding is not None:
            payload["nonmos_assembly_binding_sha256"] = self.nonmos_assembly_binding.sha256
        if self.native_cross_row_cp_proof is not None:
            payload["native_cross_row_cp_proof_sha256"] = self.native_cross_row_cp_proof.sha256
        if self.native_cross_row_final_proof is not None:
            payload["native_cross_row_final_proof_sha256"] = self.native_cross_row_final_proof.sha256
        if self.native_same_row_final_proof is not None:
            payload["native_same_row_final_proof_sha256"] = self.native_same_row_final_proof.sha256
        if self.native_row_cp_frame_proof is not None:
            payload["native_row_cp_frame_proof_sha256"] = self.native_row_cp_frame_proof.sha256
            payload["native_row_final_transported"] = self.native_row_final_transported
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()


class RowFinalPlacementResidualError(RuntimeError):
    """Typed stop for a declared row structure that cannot be fully realized."""

    def __init__(self, residuals: tuple) -> None:
        self.residuals = tuple(residuals)
        codes = tuple(
            item if type(item) is str else str(getattr(item, "code", "UNKNOWN"))
            for item in self.residuals
        )
        super().__init__(f"row final-master residuals: {codes}")


def _nonmos_row_collision_blocks(bindings, free, nonmos_binding):
    if nonmos_binding is not None:
        from clay.layout.nonmos_placement_geometry import NonmosAssemblyPlacementBinding

        if type(nonmos_binding) is not NonmosAssemblyPlacementBinding:
            raise TypeError("row certificate requires exact nonmos assembly binding")
        free = nonmos_binding.collision_blocks(free)
    blocks = [(item.owner, item.bbox_nm) for item in bindings] + list(free)
    if len({name for name, _bbox in blocks}) != len(blocks):
        raise ValueError("nonmos collision macro and row block ids must be unique")
    return blocks


def _validate_row_master_bbox(value: object, field_name: str) -> Bbox:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(v) is not int for v in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise ValueError(f"{field_name} must be a positive exact integer bbox")
    return value


def _bounded_row_master(values, limit: int, field_name: str) -> tuple:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be an iterable of records")
    try:
        result = tuple(islice(iter(values), limit + 1))
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable") from exc
    if len(result) > limit:
        raise ValueError(f"{field_name} exceeds its {limit}-item cap+1 limit")
    return result


def _bbox_area_nm2(bbox: Bbox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _positive_bbox_overlap(a: Bbox, b: Bbox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _row_blocks_bbox(blocks: List[Tuple[str, Bbox]]) -> Bbox:
    if not blocks:
        return (0, 0, 0, 0)
    return (
        min(box[0] for _name, box in blocks),
        min(box[1] for _name, box in blocks),
        max(box[2] for _name, box in blocks),
        max(box[3] for _name, box in blocks),
    )


def _rect_union_area_nm2(boxes: List[Bbox]) -> int:
    xs, ys, occupied = _compressed_occupancy(boxes)
    return sum(
        (xs[x + 1] - xs[x]) * (ys[y + 1] - ys[y])
        for y, row in enumerate(occupied)
        for x, value in enumerate(row)
        if value
    )


def _largest_empty_rectangle_nm2(core: Bbox, boxes: List[Bbox]) -> int:
    if _bbox_area_nm2(core) == 0:
        return 0
    xs, ys, occupied = _compressed_occupancy(boxes, core=core)
    if len(xs) < 2 or len(ys) < 2:
        return 0
    best = 0
    heights = [0] * (len(xs) - 1)
    # Weighted maximal-rectangle histogram.  Coordinate compression plus one
    # monotonic-stack scan per elementary y strip is O(K^2), K<=2*256+2.
    for y, row in enumerate(occupied):
        dy = ys[y + 1] - ys[y]
        for x, blocked in enumerate(row):
            heights[x] = 0 if blocked else heights[x] + dy
        stack: List[Tuple[int, int]] = []
        for x, height in enumerate((*heights, 0)):
            start = x
            while stack and stack[-1][1] > height:
                left, prior_height = stack.pop()
                best = max(best, prior_height * (xs[x] - xs[left]))
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return best


def _compressed_occupancy(
    boxes: List[Bbox], *, core: Optional[Bbox] = None
) -> Tuple[List[int], List[int], List[List[bool]]]:
    if not boxes and core is None:
        return [], [], []
    xs = sorted({*(x for box in boxes for x in (box[0], box[2])),
                 *((core[0], core[2]) if core is not None else ())})
    ys = sorted({*(y for box in boxes for y in (box[1], box[3])),
                 *((core[1], core[3]) if core is not None else ())})
    xi = {value: index for index, value in enumerate(xs)}
    yi = {value: index for index, value in enumerate(ys)}
    diff = [[0] * (len(xs) + 1) for _ in range(len(ys) + 1)]
    for x0, y0, x1, y1 in boxes:
        xa, xb, ya, yb = xi[x0], xi[x1], yi[y0], yi[y1]
        diff[ya][xa] += 1
        diff[ya][xb] -= 1
        diff[yb][xa] -= 1
        diff[yb][xb] += 1
    occupied = [[False] * max(0, len(xs) - 1) for _ in range(max(0, len(ys) - 1))]
    for y in range(len(ys) - 1):
        for x in range(len(xs) - 1):
            above = diff[y - 1][x] if y else 0
            left = diff[y][x - 1] if x else 0
            diagonal = diff[y - 1][x - 1] if y and x else 0
            diff[y][x] += above + left - diagonal
            occupied[y][x] = diff[y][x] > 0
    return xs, ys, occupied


def _placed_artwork_sha256(polys: list) -> str:
    records = []
    for poly in polys:
        records.append((
            int(poly.layer), int(poly.datatype),
            tuple((int(point.x), int(point.y)) for point in poly.polygon.vertices),
        ))
    records.sort()
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _polygon_multiset_key(poly: object) -> tuple:
    """Exact polygon identity used for occurrence-safe aggregate subtraction."""
    return (
        int(poly.layer),
        int(poly.datatype),
        tuple((int(point.x), int(point.y)) for point in poly.polygon.vertices),
    )


def _subtract_polygon_multiset(source, remove) -> List[object]:
    """Remove at most the requested count of each exact polygon occurrence."""
    remaining = Counter(_polygon_multiset_key(poly) for poly in remove)
    result: List[object] = []
    for poly in source:
        key = _polygon_multiset_key(poly)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            result.append(poly)
    return result


def _subtract_polygon_key_multiset(source, remove_keys) -> List[object]:
    """Remove exactly the provenance-counted polygon occurrences from ``source``."""
    remaining = Counter(tuple(key) for key in remove_keys)
    result: List[object] = []
    for poly in source:
        key = _polygon_multiset_key(poly)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            result.append(poly)
    return result


def _nwell_bridge_generation_enabled(tech: object) -> bool:
    """Return the same PDK gate used by the original shared-NWELL pass."""
    try:
        return bool(tech.line_end_ext("metal1")) and (
            str(getattr(tech, "pdk_id", "")).lower() != "gf180mcu"
        )
    except Exception:  # noqa: BLE001 - a silent PDK does not opt into bridges
        return False


def _generated_nwell_bridge_keys(placement: "Placement", tech: object) -> tuple:
    """Recover only prior generated bridge occurrences, never all NWELL artwork."""
    recorded = tuple(getattr(placement, "row_generated_nwell_bridge_keys", ()) or ())
    if recorded:
        return recorded
    if not _nwell_bridge_generation_enabled(tech):
        return ()
    try:
        nwell_key = tech.build_layer("nwell")
    except Exception:  # noqa: BLE001 - no NWELL layer means no generated bridge
        return ()
    # Before the first RowProgram projection, shared-well bridges were generated
    # from exactly these independently owned device polygons.  Recompute that
    # deterministic multiset so only those occurrences are replaced.  Subsequent
    # projections carry explicit keys on Placement and do not need inference.
    owned = [
        poly
        for pd in placement.devices.values()
        for poly in (pd.polys or ())
    ]
    return tuple(
        _polygon_multiset_key(poly)
        for poly in _shared_nwell_bridges(owned, tuple(nwell_key))
    )


def placement_auxiliary_polys(placement: "Placement") -> List[object]:
    """Return aggregate FEOL artwork not owned by a device or stored row macro.

    ``Placement.polys`` can contain shared-well bridges and other aggregate-only
    artwork in addition to the per-device polygon lists.  Set subtraction is
    incorrect here because coincident duplicate stamps are meaningful; subtract
    exact occurrences as a multiset and preserve the unmatched objects verbatim.
    """
    remove: List[object] = [
        poly
        for pd in placement.devices.values()
        for poly in (pd.polys or ())
    ]
    certificate = placement.row_final_certificate
    program = placement.row_program
    materialization = placement.row_materialization
    if certificate is not None and program is not None and materialization is not None:
        realization_by_entry = {
            item.entry_id: item for item in materialization.realizations
        }
        binding_by_entry = {
            item.entry_id: item for item in certificate.macro_bindings
        }
        for entry in program.entries:
            realization = realization_by_entry.get(entry.entry_id)
            binding = binding_by_entry.get(entry.entry_id)
            if realization is None or binding is None:
                continue
            lx0, ly0, _lx1, _ly1 = realization.local_bbox_nm
            dx = binding.bbox_nm[0] - lx0
            dy = binding.bbox_nm[1] - ly0
            remove.extend(_translate(
                list(getattr(realization.geometry, "polygons", ()) or ()), dx, dy
            ))
    return _subtract_polygon_multiset(list(placement.polys or ()), remove)


@dataclass(frozen=True)
class Placement:
    """The generic placement result the router consumes.

    ``devices`` maps each device label to its :class:`PlacedDevice`. ``axis_x`` is
    the solved vertical symmetry axis (the mirror line). ``geom`` carries the cell
    rail / channel geometry derived from the placed device extents. ``polys`` is
    every placed device polygon (the full cell artwork sans routing). ``quality``
    is the R172 :class:`PlacementQuality` from :func:`score_placement`.
    """

    devices: Dict[str, PlacedDevice]
    axis_x: int
    geom: Dict[str, int]
    polys: list
    quality: object
    # provenance (generic, honest)
    solver_status: str = ""
    solver_success: bool = False
    solve_seconds: float = 0.0
    constraints: Optional[SolverConstraints] = None
    used_default_geometry: bool = False
    note: str = ""
    row_program: object = None
    row_materialization: object = None
    row_final_certificate: Optional[RowFinalPlacementCertificate] = None
    # Exact emitted-stamp bboxes for devices that remain free of a stored row
    # macro.  The logical PlacedDevice envelope is intentionally retained for
    # solver/terminal consumers; final row density and collision evidence uses
    # this independent physical view when canonical route geometry supplies it.
    row_free_physical_bboxes: Dict[str, Bbox] = field(default_factory=dict)
    row_native_geometry_binding: object = None
    row_residuals: tuple = ()
    # Exact immutable occurrence keys for the shared-NWELL straps generated by
    # the previous row projection.  They let the next projection replace only
    # those derived straps while preserving every legitimate auxiliary NWELL.
    row_generated_nwell_bridge_keys: tuple = ()
    # Canonical P2 structure compiled from typed intent before the placement
    # solver sees footprints.  ``row_macro_contracts`` binds its row owners to
    # exact PDK geometry and access interfaces; both values survive every later
    # compact projection unchanged and are replayed with the final certificate.
    physical_macro_graph: object = None
    physical_macro_scope_token: object = None
    row_macro_contracts: object = None
    # Non-row physical macro interfaces are closed from exact PDK geometry
    # before solving.  Their final certificate is a translation-only replay of
    # those interfaces over the placed device views.
    physical_macro_contracts: object = None
    physical_macro_compile_inputs: object = None
    physical_macro_placement_certificate: object = None
    # Exact PDK-sized aggregate passive/BJT tilings compiled before the MOS
    # solver.  Later placement stages may translate a whole plan only.
    rigid_nonmos_macro_inputs: object = None
    rigid_nonmos_macro_plans: object = None
    rigid_nonmos_macro_interfaces: object = None
    rigid_nonmos_placement_certificate: object = None
    # A layered native placement proof is also required for non-row-only
    # assemblies. Do not hide it in row metadata when no row program exists.
    rigid_nonmos_native_geometry_binding: object = None
    # Direct non-MOS matching certificate.  The key is the canonical unordered
    # pair and the value is the local mirror axis established by the matched
    # macro materializer.  The final-GDS intent checker reuses this immutable
    # authority; it must never infer matching from a place-group token.
    matching_pair_axes: Dict[
        Tuple[str, str], Tuple[str, float]
    ] = field(default_factory=dict)
    # Conditional source-bound atomic CP proof for matching pairs whose two
    # physical endpoints belong to distinct native row owners. This is not a
    # final-placement certificate until row seating and later transforms replay.
    native_cross_row_cp_proof: object = None
    native_cross_row_final_proof: object = None
    native_same_row_final_proof: object = None
    native_row_cp_frame_proof: object = None
    native_row_final_transported: bool = False
    # Only members the non-MOS materializer genuinely could not realize belong
    # here.  Members filtered out of the MOS solver but subsequently placed by
    # the passive/fixed-cell path are deliberately removed from this set.
    dropped_nonmos_matched: Tuple[str, ...] = ()
    # Exact physical PCell footprints for matching pairs whose abstract
    # ``PlacedDevice`` view is intentionally a different routing/solver envelope.
    # The final placement certificate is the sole producer of this derived view.
    matching_pair_physical_bboxes: Dict[
        Tuple[str, str], Tuple[Bbox, Bbox]
    ] = field(default_factory=dict)
    # Source-bound ordered rows and solved common axes. The actual device masks
    # and terminal accesses must replay this evidence before routing consumes it.
    ordered_row_evidence: object = None
    # Native geometry is source/PDK bound before solving and remains the sole
    # artwork authority when routing consumes the translated placements.
    native_geometry: object = None

    @property
    def symmetry_score(self) -> float:
        return float(getattr(self.quality, "symmetry_score", 0.0))

    @property
    def n_placed(self) -> int:
        return len(self.devices)

    def device_centroids(self) -> Dict[str, Tuple[float, float]]:
        """``{label: (x, y)}`` device centroids (each placed-bbox centre).

        The native form the convergence criteria
        (:meth:`clay.layout.guide_intent.ConvergenceCriteria.check_placement`)
        reads mirror-symmetry / common-centroid obligations from, so the criteria
        can measure the REAL placed geometry directly off a :class:`Placement`
        without an external adapter.
        """
        out: Dict[str, Tuple[float, float]] = {}
        for label, pd in self.devices.items():
            x0, y0, x1, y1 = pd.bbox
            out[str(label)] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return out


def _has_row_structure(intent: object) -> bool:
    return bool(
        (getattr(intent, "arrays", ()) or ())
        or (getattr(intent, "common_centroids", ()) or ())
        or (getattr(intent, "discrete_mosaics", ()) or ())
    )


def _validate_native_row_source_cir(cell_def, source_cir):
    """Compare native circuit operands with the existing native PDK contract."""
    from clay.frontend.native_spice_import import (
        _read_subcircuits, emit_native_spice, import_native_spice,
    )
    from clay.pdk.sky130_native_geometry import native_cell_geometry_specs

    canonical_text = emit_native_spice(cell_def, pdk_id='sky130')
    canonical_definitions = _read_subcircuits(canonical_text)
    if len(canonical_definitions) != 1:
        raise ValueError('native row canonical source must be flat')
    top = next(iter(canonical_definitions.values())).name
    actual_text = Path(source_cir).read_text(encoding='utf-8-sig')
    canonical = import_native_spice(canonical_text, top_cell=top, pdk_id='sky130').cell_def
    actual = import_native_spice(actual_text, top_cell=top, pdk_id='sky130').cell_def
    expected_specs = native_cell_geometry_specs(canonical)
    actual_specs = native_cell_geometry_specs(actual)
    expected = {d.instance_name: (expected_specs[d.instance_name], d.pins) for d in canonical.devices}
    observed = {d.instance_name: (actual_specs[d.instance_name], d.pins) for d in actual.devices}
    if observed != expected:
        raise ValueError('native row circuit/source physical contract differs')


def _compile_row_master(
    cell_def,
    intent,
    tech,
    source_cir,
    row_geometry_overrides=None,
    row_source_projection=None,
):
    """Compile and invoke the rigid row builders once, before placement.

    An undeclared cell takes the literal no-op path.  A declared but unrealizable
    row is a typed stop; it never falls back to duplicate free device geometry.
    """
    if intent is None or not _has_row_structure(intent):
        if row_source_projection is not None:
            from clay.layout.place.row_source_projection import RowSourceProjectionError

            raise RowSourceProjectionError("ROW_SOURCE_PROGRAM_MISSING")
        return None, None
    from clay.layout.place.row_program import (
        compile_row_program,
        materialize_row_program,
    )

    if source_cir is not None and any(getattr(d, 'device_class', None) == 'native'
                                     for d in getattr(cell_def, 'devices', ())):
        _validate_native_row_source_cir(cell_def, source_cir)

    if row_source_projection is not None:
        from clay.layout.place.row_source_projection import (
            RowSourceProjectionError, require_actual_row_source_topology,
        )

        records = require_actual_row_source_topology(row_source_projection, source_cir)
        if (any(getattr(row, 'style', '') == 'native_diffusion' for row in intent.arrays)
                and tuple(getattr(intent, 'discrete_mosaics', ())) != row_source_projection.discrete_mosaics):
            raise RowSourceProjectionError('NATIVE_MOSAIC_INTENT_PROVENANCE_MISMATCH')
    else:
        records = parse_cir_mos_records(source_cir) if source_cir else None
    program = compile_row_program(
        cell_def, intent, tech, device_records=(records or None)
    )
    if program.residuals:
        raise RowFinalPlacementResidualError(program.residuals)
    if row_source_projection is not None:
        from clay.layout.place.row_source_projection import (
            replay_row_source_projection,
            require_row_source_projection,
        )

        replay_row_source_projection(row_source_projection, cell_def, tech)
        require_row_source_projection(row_source_projection, program=program)
    materialization = materialize_row_program(
        program,
        tech,
        geometry_overrides=row_geometry_overrides,
        source_projection=row_source_projection,
    )
    if materialization.residuals:
        raise RowFinalPlacementResidualError(materialization.residuals)
    if len(materialization.realizations) != len(program.entries):
        raise RowFinalPlacementResidualError(("ROW_REALIZATION_CARDINALITY",))
    return program, materialization


def _compile_physical_macro_structure(
    cell_def, intent, tech, source_cir, annotated, row_program, row_materialization
):
    """Compile one graph/geometry structure authority before placement.

    The graph is topology/declaration-only.  RowProgram then binds exact dummy
    consumption and the materializer closes the PDK-specific bbox, access, body,
    orientation, and route-demand interface.  No reference geometry is read.
    """

    if intent is None:
        from clay.layout.intent import layout_intent_from_declarations
        # Compile source ownership without adding default placement constraints.
        intent = layout_intent_from_declarations(cell_def.local_name, ())
    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent

    records = parse_cir_mos_records(source_cir) if source_cir else {}
    annotated_kinds = getattr(annotated, "_device_kinds", {}) or {}
    annotated_pins = getattr(annotated, "_device_pins", {}) or {}
    for name, kind in annotated_kinds.items():
        records.setdefault(
            str(name),
            {
                "kind": str(kind),
                "pins": dict(annotated_pins.get(name, {}) or {}),
            },
        )
    try:
        graph = physical_macro_graph_from_intent(
            cell_def, intent, device_records=records
        )
    except Exception as exc:  # noqa: BLE001 - typed compiler boundary
        from clay.layout.physical_macro_contracts import PhysicalMacroContractError

        raise PhysicalMacroContractError(
            f"physical macro graph compilation failed: {exc}"
        ) from exc
    if row_program is None and row_materialization is None:
        return graph, None
    if row_program is None or row_materialization is None:
        raise RowFinalPlacementResidualError(("ROW_MASTER_METADATA_INCOMPLETE",))
    from clay.layout.place.row_program import compile_row_macro_contracts

    contracts = compile_row_macro_contracts(
        graph, row_program, row_materialization, tech=tech
    )
    return graph, contracts


def _row_expected_finger_owners(entry) -> Tuple[str, ...]:
    from clay.layout.place.row_program_ownership import expected_row_finger_owners
    from clay.layout.place.row_program_primitives import RowProgramError

    try:
        return expected_row_finger_owners(entry)
    except RowProgramError as exc:
        raise RowFinalPlacementResidualError(("ROW_ATOM_OWNER_UNRESOLVED",)) from exc


def row_macro_member_offsets_nm(entry, realization) -> Dict[str, Tuple[int, int]]:
    """Project one stored rigid realization into member-centroid offsets.

    The materializer has already certified the full finger-owner sequence.  This
    projection consumes that same sequence and geometry; it never invokes an array
    resolver or builder a second time.
    """
    geometry = realization.geometry
    fingers = tuple(getattr(geometry, "fingers", ()) or ())
    owners = _row_expected_finger_owners(entry)
    if len(fingers) != len(owners):
        raise RowFinalPlacementResidualError(("ROW_FINGER_CARDINALITY_MISMATCH",))
    x0, y0, x1, y1 = realization.local_bbox_nm
    # This is also the route CellGeometry's exact recentering origin.  Keeping it
    # on the integer-nm lattice prevents a half-nm centre from being rounded twice
    # into a non-rigid emitted macro.
    ecx = (x0 + x1) // 2
    ecy = (y0 + y1) // 2
    spans: Dict[str, List[Tuple[int, int, int]]] = {}
    default_cy = int(round(float(getattr(geometry, "channel_width_nm", 0)) / 2.0))
    for finger, certified_owner in zip(fingers, owners):
        raw_owner = str(getattr(finger, "member", "") or "")
        owner = raw_owner or certified_owner
        if owner != certified_owner:
            raise RowFinalPlacementResidualError(("ROW_FINGER_OWNER_MISMATCH",))
        raw_cy = getattr(finger, "cy", None)
        cy = default_cy if raw_cy is None else int(raw_cy)
        spans.setdefault(owner, []).append((int(finger.x0), int(finger.x1), cy))
    missing = tuple(sorted(set(entry.consumed_devices) - set(spans)))
    if missing:
        raise RowFinalPlacementResidualError((f"ROW_MEMBER_OFFSET_MISSING:{missing}",))
    result: Dict[str, Tuple[int, int]] = {}
    for owner, values in spans.items():
        mcx = (min(item[0] for item in values) + max(item[1] for item in values)) / 2.0
        mcy = (min(item[2] for item in values) + max(item[2] for item in values)) / 2.0
        result[owner] = (int(round(mcx - ecx)), int(round(mcy - ecy)))
    return result


def _move_placed_device(pd: PlacedDevice, cx_nm: int, cy_nm: int) -> PlacedDevice:
    x0, y0, x1, y1 = pd.bbox
    dx = int(cx_nm - (x0 + x1) // 2)
    dy = int(cy_nm - (y0 + y1) // 2)
    return replace(
        pd,
        bbox=(x0 + dx, y0 + dy, x1 + dx, y1 + dy),
        terminals={name: (x + dx, y + dy) for name, (x, y) in pd.terminals.items()},
        polys=_translate(list(pd.polys or []), dx, dy),
    )


def _verify_row_program_member_inventory(
    placement: Placement, program: object
) -> None:
    """Bind every compiled MOS row member to one final placed-device view."""

    try:
        expected = {member.device for member in program.members}
    except Exception as exc:  # noqa: BLE001 - typed row authority boundary
        raise RowFinalPlacementResidualError((
            f"ROW_PROGRAM_MEMBER_INVENTORY_INVALID:{exc}",
        )) from exc
    missing = tuple(sorted(expected - set(placement.devices)))
    if missing:
        raise RowFinalPlacementResidualError((
            f"ROW_PLACED_MEMBER_INVENTORY_MISMATCH:{missing}",
        ))
    unexpected_mos = tuple(sorted(
        name
        for name, view in placement.devices.items()
        if name not in expected
        and type(getattr(view, "kind", None)) is str
        and (
            str(view.kind).strip().lower().startswith("nmos")
            or str(view.kind).strip().lower().startswith("pmos")
        )
    ))
    if unexpected_mos:
        raise RowFinalPlacementResidualError((
            f"ROW_PLACED_MEMBER_INVENTORY_MISMATCH:{unexpected_mos}",
        ))


def _project_stored_row_matching_axes(
    placement: Placement,
    devices: Mapping[str, PlacedDevice],
    program: object,
    *,
    row_certificate: object = None,
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Transport materializer axes through one stored-row rigid translation.

    A stored row may replace each solver envelope with a larger terminal view, so
    footprint-size equality with the pre-projection view is not a valid proof.
    The RowProgram supplies the stronger ownership fact: both members must belong
    to one exact entry, retain equal footprints within each view, and move by the
    same integer-nm centre delta.  The prior axis is then translated by that delta;
    no pair identity or coordinate is inferred from final geometry alone.
    """

    raw_axes = getattr(placement, "matching_pair_axes", {})
    if type(raw_axes) is not dict:
        raise TypeError("stored row matching-axis inventory must be an exact dict")
    raw_entries = getattr(program, "entries", ())
    if type(raw_entries) is not tuple:
        raise TypeError("stored row entry inventory must be an exact tuple")
    owner_by_member: Dict[str, int] = {}
    for owner_index, entry in enumerate(raw_entries):
        members = getattr(entry, "consumed_devices", ())
        if (
            type(members) is not tuple
            or any(type(member) is not str or not member for member in members)
            or len(set(members)) != len(members)
        ):
            raise TypeError("stored row member inventory must be an exact tuple")
        for member in members:
            if member in owner_by_member:
                raise ValueError("stored row member has multiple entry owners")
            owner_by_member[member] = owner_index

    def exact_bbox(member: str, view: Any, label: str) -> Bbox:
        bbox = getattr(view, "bbox", None)
        if (
            type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError(
                f"stored row matching member {member!r} lacks exact {label} bbox"
            )
        return bbox

    native_projections = {}
    native_binding = placement.row_native_geometry_binding
    if native_binding is not None:
        if type(placement.matching_pair_physical_bboxes) is not dict:
            raise TypeError("stored row native matching physical inventory must be an exact dict")
        # The caller has replayed this exact binding against all row artwork.
        # Enumerate signed pairs first so deleting an axis cannot skip replay.
        from clay.layout.place.final_placement_state import MatchingPairAxisProjection

        source_bboxes = dict(native_binding.certificate.materialized_source_member_bboxes)
        for row in native_binding.certificate.matching_pair_axis_projections:
            if type(row) is not MatchingPairAxisProjection:
                raise TypeError("stored row native matching projection has a foreign type")
            if not any(member in owner_by_member for member in row.pair):
                continue
            if (row.pair not in raw_axes
                    or row.pair not in placement.matching_pair_physical_bboxes):
                raise ValueError("stored row native matching inventory is incomplete")
            if tuple(source_bboxes.get(member) for member in row.pair) != row.final_member_bboxes_nm:
                raise ValueError("stored row native matching physical sources do not replay")
            native_projections[row.pair] = row

    result = project_rigid_nonmos_matching_axes(placement, devices)
    cross_row_axes = {}
    same_row_axes = {}
    same_row_logical_axes = {}
    same_row_proof = getattr(placement, "native_same_row_final_proof", None)
    if native_binding is not None:
        retained = getattr(native_binding.certificate, "native_same_row_final_proof", None)
        if retained != same_row_proof:
            raise ValueError("stored same-row matching proof differs from native certificate")
    if same_row_proof is not None:
        from clay.layout.place.native_same_row_matching import (
            native_same_row_inventory_from_projection, native_same_row_occurrence_axes,
            replay_native_same_row_final_proof,
        )
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        from clay.layout.tech import LayoutTech
        tech = LayoutTech.resolve(program.tech_id)
        inventory = native_same_row_inventory_from_projection(program, placement.row_materialization, tech)
        if inventory is None or row_certificate is None:
            raise ValueError("stored same-row matching lacks source or row certificate")
        if getattr(row_certificate, "native_same_row_final_proof", None) != same_row_proof:
            raise ValueError("stored same-row matching row certificate differs")
        source_centres = dict(same_row_proof.source_centres2_nm)
        logical_axes = dict(same_row_proof.source_logical_axes4_nm)
        source_axes = native_same_row_occurrence_axes(inventory, source_centres, logical_axes)
        bindings = {row.entry_id: row for row in row_certificate.macro_bindings}
        final_centres = {}
        for owner, members in inventory.row_source_members:
            binding = bindings.get(owner)
            if binding is None or binding.consumed_devices != members:
                raise ValueError("stored same-row matching final owner differs")
            box = binding.bbox_nm
            final_centres[owner] = (box[0] + box[2], box[1] + box[3])
        signed = replay_native_same_row_final_proof(
            inventory, same_row_proof, source_centres, source_axes, final_centres,
            pdk_manufacturing_grid_nm(tech), logical_axes=logical_axes)
        if not set(signed) <= set(raw_axes):
            raise ValueError("stored same-row matching axis inventory is incomplete")
        same_row_axes = {pair: (axis, float(coordinate / 4))
                         for pair, (axis, coordinate) in signed.items()}
        same_row_logical_axes = dict(same_row_proof.final_logical_axes4_nm)
    cross_row_proof = getattr(placement, "native_cross_row_cp_proof", None)
    if cross_row_proof is not None:
        from clay.layout.place.cross_row_matching import (
            native_cross_row_inventory_from_projection, replay_native_cross_row_cp_proof,
            verify_native_cross_row_geometry,
        )
        from clay.layout.tech import LayoutTech

        inventory = native_cross_row_inventory_from_projection(
            program, placement.row_materialization, LayoutTech.resolve(program.tech_id),
        )
        if inventory is None or row_certificate is None:
            raise ValueError("stored cross-row matching lacks source or row certificate")
        source_pairs = {pair.members for pair in inventory.pairs}
        if any(pair not in raw_axes for pair in source_pairs):
            raise ValueError("stored cross-row matching source axis inventory is incomplete")
        directions = {pair: raw_axes[pair][0] for pair in source_pairs}
        signed_centres2, cross_row_axes = replay_native_cross_row_cp_proof(
            inventory, cross_row_proof, directions, row_program=program,
            row_materialization=placement.row_materialization, tech=LayoutTech.resolve(program.tech_id),
        )
        entries_by_id = {entry.entry_id: entry for entry in raw_entries}
        for owner in cross_row_proof.binding.owners:
            entry = entries_by_id.get(owner.row_owner)
            if entry is None or owner.solver_owner != f"A_{entry.group}":
                raise ValueError("stored cross-row CP owner differs from source row")
        actual_bindings = {row.entry_id: row for row in row_certificate.macro_bindings}
        actual_centres2 = {}
        for owner in signed_centres2:
            binding = actual_bindings.get(owner)
            if binding is None or binding.owner != owner:
                raise ValueError("stored cross-row final owner is missing")
            bbox = binding.bbox_nm
            actual_centres2[owner] = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        final_proof = getattr(placement, "native_cross_row_final_proof", None)
        if final_proof is None:
            if actual_centres2 != signed_centres2:
                raise ValueError("stored cross-row final owner pose differs from CP proof")
            verify_native_cross_row_geometry(inventory, actual_centres2, cross_row_axes)
        else:
            from clay.layout.place.cross_row_matching import replay_native_cross_row_final_proof
            from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
            if final_proof.cp_proof != cross_row_proof:
                raise ValueError("stored cross-row final proof differs from CP")
            cross_row_axes = replay_native_cross_row_final_proof(
                inventory, final_proof, directions, actual_centres2,
                pdk_manufacturing_grid_nm(LayoutTech.resolve(program.tech_id)),
                row_program=program, row_materialization=placement.row_materialization,
                tech=LayoutTech.resolve(program.tech_id))
    for raw_pair, raw_axis in tuple(sorted(raw_axes.items())):
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or any(type(member) is not str or not member for member in raw_pair)
            or raw_pair[0] >= raw_pair[1]
            or type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or type(raw_axis[0]) is not str
            or raw_axis[0] not in {"x", "y"}
            or type(raw_axis[1]) not in {int, float}
            or not math.isfinite(float(raw_axis[1]))
        ):
            raise TypeError("stored row matching-axis declaration is malformed")
        member_owners = tuple(owner_by_member.get(member) for member in raw_pair)
        if member_owners == (None, None):
            continue
        if None in member_owners or member_owners[0] != member_owners[1]:
            if raw_pair in cross_row_axes:
                axis, coordinate4 = cross_row_axes[raw_pair]
                result[raw_pair] = (axis, coordinate4 / 4.0)
                continue
            raise ValueError(
                "stored row matching pair does not have one exact entry owner"
            )
        if any(member not in placement.devices or member not in devices for member in raw_pair):
            raise ValueError("stored row matching pair lacks device views")
        source_boxes = tuple(
            exact_bbox(member, placement.devices[member], "source")
            for member in raw_pair
        )
        final_boxes = tuple(
            exact_bbox(member, devices[member], "final") for member in raw_pair
        )
        axis_index = 0 if raw_axis[0] == "x" else 1
        perpendicular_index = 1 - axis_index
        source_coordinate4 = sum(
            bbox[axis_index] + bbox[axis_index + 2] for bbox in source_boxes
        )
        declared_coordinate4 = Decimal(str(raw_axis[1])) * Decimal(4)
        if raw_pair in same_row_axes:
            if raw_axis != same_row_axes[raw_pair]:
                raise ValueError("stored same-row physical axis differs from proof")
            logical_axis, logical_coordinate4 = same_row_logical_axes[raw_pair]
            if logical_axis != raw_axis[0]:
                raise ValueError("stored same-row logical axis direction differs")
            for boxes in (source_boxes, final_boxes):
                coordinate4 = sum(bbox[axis_index] + bbox[axis_index + 2] for bbox in boxes)
                if coordinate4 != logical_coordinate4:
                    raise ValueError("stored same-row logical axis does not replay device views")
            if any(
                source_box[axis] + source_box[axis + 2]
                != final_box[axis] + final_box[axis + 2]
                for source_box, final_box in zip(source_boxes, final_boxes)
                for axis in (0, 1)
            ):
                raise ValueError("stored row adds a second native matching translation")
            result[raw_pair] = raw_axis
            continue
        # A final native MOS projection carries a physical PCell axis, while
        # these device views retain logical row centres. Replay both signed
        # views; seating the stored artwork must add no second translation.
        native_projection = native_projections.get(raw_pair)
        if native_projection is not None:
            if (
                raw_axis[0] != native_projection.axis
                or declared_coordinate4 != native_projection.final_coordinate4_nm
                or placement.matching_pair_physical_bboxes.get(raw_pair)
                != native_projection.final_member_bboxes_nm
            ):
                raise ValueError("stored row native matching physical view does not replay")
            for boxes in (source_boxes, final_boxes):
                coordinate4 = sum(
                    bbox[axis_index] + bbox[axis_index + 2] for bbox in boxes
                )
                if (
                    coordinate4 != native_projection.final_logical_coordinate4_nm
                    or boxes[0][perpendicular_index] + boxes[0][perpendicular_index + 2]
                    != boxes[1][perpendicular_index] + boxes[1][perpendicular_index + 2]
                ):
                    raise ValueError("stored row native matching logical view does not replay")
            if any(
                source_box[axis] + source_box[axis + 2]
                != final_box[axis] + final_box[axis + 2]
                for source_box, final_box in zip(source_boxes, final_boxes)
                for axis in (0, 1)
            ):
                raise ValueError("stored row adds a second native matching translation")
            result[raw_pair] = raw_axis
            continue
        if (
            declared_coordinate4 != declared_coordinate4.to_integral_value()
            or source_coordinate4 != int(declared_coordinate4)
        ):
            raise ValueError(
                "stored row matching source axis does not replay source bboxes"
            )
        for label, boxes in (("source", source_boxes), ("final", final_boxes)):
            if (
                boxes[0][perpendicular_index]
                + boxes[0][perpendicular_index + 2]
                != boxes[1][perpendicular_index]
                + boxes[1][perpendicular_index + 2]
            ):
                raise ValueError(
                    f"stored row matching pair lacks {label} perpendicular alignment"
                )
        translations2 = tuple(
            (
                final_bbox[0] + final_bbox[2] - source_bbox[0] - source_bbox[2],
                final_bbox[1] + final_bbox[3] - source_bbox[1] - source_bbox[3],
            )
            for source_bbox, final_bbox in zip(source_boxes, final_boxes)
        )
        if (
            translations2[0] != translations2[1]
            or any(value % 2 for value in translations2[0])
        ):
            raise ValueError(
                "stored row matching pair does not replay one translation"
            )
        final_coordinate4 = source_coordinate4 + 2 * translations2[0][axis_index]
        actual_final_coordinate4 = sum(
            bbox[axis_index] + bbox[axis_index + 2] for bbox in final_boxes
        )
        if actual_final_coordinate4 != final_coordinate4:
            raise ValueError("stored row matching axis translation does not replay")
        result[raw_pair] = (raw_axis[0], final_coordinate4 / 4.0)
    return result


def project_stored_row_master(
    placement: Placement, *, allow_collision_residual: bool = False
) -> Placement:
    """Seat stored row macros as rigid tiles and issue the final row certificate."""
    program = placement.row_program
    materialization = placement.row_materialization
    if program is None and materialization is None:
        return placement
    if program is None or materialization is None:
        raise RowFinalPlacementResidualError(("ROW_MASTER_METADATA_INCOMPLETE",))
    _verify_row_program_member_inventory(placement, program)
    from clay.layout.place.row_replay_session import require_current_row_materialization

    try:
        require_current_row_materialization(
            materialization, program=program, tech_id=program.tech_id
        )
    except Exception as exc:
        raise RowFinalPlacementResidualError((
            f"ROW_MATERIALIZATION_REPLAY_MISMATCH:{exc}",
        )) from exc
    try:
        row_tech = _resolve_tech(program.tech_id)
    except Exception:  # noqa: BLE001 - materialization replay already binds tech
        row_tech = None
    physical_graph = placement.physical_macro_graph
    macro_contracts = placement.row_macro_contracts
    prior_certificate = placement.row_final_certificate
    if (
        physical_graph is None
        and macro_contracts is None
        and prior_certificate is not None
        and (
            prior_certificate.physical_macro_graph_sha256
            or prior_certificate.row_macro_contracts_sha256
        )
    ):
        raise RowFinalPlacementResidualError((
            "ROW_MACRO_STRUCTURE_METADATA_REMOVED",
        ))
    if (physical_graph is None) != (macro_contracts is None):
        raise RowFinalPlacementResidualError((
            "ROW_MACRO_STRUCTURE_METADATA_INCOMPLETE",
        ))
    if macro_contracts is not None:
        if row_tech is None:
            raise RowFinalPlacementResidualError((
                "ROW_MACRO_STRUCTURE_TECH_UNRESOLVED",
            ))
        from clay.layout.place.row_program import compile_row_macro_contracts

        try:
            replayed_contracts = compile_row_macro_contracts(
                physical_graph,
                program,
                materialization,
                tech=row_tech,
            )
        except Exception as exc:
            raise RowFinalPlacementResidualError((
                f"ROW_MACRO_STRUCTURE_REPLAY_MISMATCH:{exc}",
            )) from exc
        if (
            replayed_contracts.sha256 != macro_contracts.sha256
            or replayed_contracts.payload() != macro_contracts.payload()
        ):
            raise RowFinalPlacementResidualError((
                "ROW_MACRO_STRUCTURE_REPLAY_MISMATCH",
            ))
    contracts_by_id = (
        {item.macro_id: item for item in macro_contracts.macros}
        if macro_contracts is not None
        else {}
    )
    prior_bridge_keys = (
        _generated_nwell_bridge_keys(placement, row_tech)
        if row_tech is not None else ()
    )
    # Preserve aggregate-only FEOL as a multiset.  First remove every standalone
    # device occurrence.  Once the final macro locations are known below, remove
    # exactly one occurrence of each already-projected macro as well; coincident
    # surplus occurrences remain auxiliary artwork rather than collapsing as a set.
    aggregate_remainder = _subtract_polygon_multiset(
        list(placement.polys or ()),
        [poly for pd in placement.devices.values() for poly in (pd.polys or ())],
    )
    by_entry = {item.entry_id: item for item in materialization.realizations}
    devices = dict(placement.devices)
    bindings: List[RowMacroBlockBinding] = []
    macro_polys: List[object] = []
    macro_polys_by_owner: Dict[str, tuple] = {}
    consumed: set = set()
    for entry in program.entries:
        realization = by_entry.get(entry.entry_id)
        if realization is None:
            raise RowFinalPlacementResidualError(("ROW_REALIZATION_MISSING",))
        offsets = row_macro_member_offsets_nm(entry, realization)
        missing = tuple(sorted(set(entry.consumed_devices) - set(devices)))
        if missing:
            raise RowFinalPlacementResidualError((f"ROW_PLACED_MEMBER_MISSING:{missing}",))
        candidate_centres = []
        for name in entry.consumed_devices:
            box = devices[name].bbox
            ox, oy = offsets[name]
            candidate_centres.append(((box[0] + box[2]) / 2.0 - ox,
                                      (box[1] + box[3]) / 2.0 - oy))
        tcx = int(round(sum(item[0] for item in candidate_centres) / len(candidate_centres)))
        tcy = int(round(sum(item[1] for item in candidate_centres) / len(candidate_centres)))
        lx0, ly0, lx1, ly1 = realization.local_bbox_nm
        local_cx = (lx0 + lx1) // 2
        local_cy = (ly0 + ly1) // 2
        macro_dx, macro_dy = tcx - local_cx, tcy - local_cy
        contract = contracts_by_id.get(f"matched:{entry.group}")
        contract_terminals: Dict[str, Dict[str, list]] = {}
        if contract is not None:
            for terminal_contract in contract.terminal_accesses:
                by_role = contract_terminals.setdefault(
                    terminal_contract.member, {}
                )
                by_role.setdefault(terminal_contract.role, []).append(terminal_contract)
        placed_macro_polys = _translate(
            list(getattr(realization.geometry, "polygons", ()) or ()),
            macro_dx,
            macro_dy,
        )
        actual_macro_bbox = _polys_bbox(placed_macro_polys)
        expected_macro_bbox = (
            lx0 + macro_dx, ly0 + macro_dy, lx1 + macro_dx, ly1 + macro_dy,
        )
        if actual_macro_bbox != expected_macro_bbox:
            raise RowFinalPlacementResidualError((
                f"ROW_ARTWORK_BBOX_REPLAY:{entry.entry_id}",
            ))
        for name in entry.consumed_devices:
            ox, oy = offsets[name]
            view = _move_placed_device(devices[name], tcx + ox, tcy + oy)
            try:
                local_pins = realization.geometry.member_pins(name)
            except KeyError:
                # Rail end-cap dummies own no independent drain column.  They are
                # electrically discharged inside the macro and therefore expose no
                # external routing terminal; the geometry owner remains explicit.
                local_pins = {}
            terminals = {
                str(role): (int(point[0]) + macro_dx, int(point[1]) + macro_dy)
                for role, point in local_pins.items()
            }
            member_contracts = contract_terminals.get(name, {})
            if contract is not None:
                if any(len({item.layer for item in items}) != 1
                       for items in member_contracts.values()):
                    raise RowFinalPlacementResidualError((
                        f"ROW_MEMBER_CONTRACT_LAYER_MISMATCH:{name}",
                    ))
                term_layers = {
                    role: items[0].layer
                    for role, items in sorted(member_contracts.items())
                }
                if contract.terminal_components is None and contract.joint_inventory is None:
                    terminal_accesses = {
                        role: tuple((point[0] + macro_dx, point[1] + macro_dy)
                                    for point in items[0].accesses_nm)
                        for role, items in sorted(member_contracts.items())
                    }
                    if terminals != {role: points[0] for role, points in terminal_accesses.items()}:
                        raise RowFinalPlacementResidualError((
                            f"ROW_MEMBER_CONTRACT_ACCESS_MISMATCH:{name}",
                        ))
                else:
                    if set(terminals) != set(member_contracts):
                        raise RowFinalPlacementResidualError((
                            f"ROW_MEMBER_CONTRACT_ACCESS_MISMATCH:{name}",
                        ))
                    # This compatibility view contains OR alternatives for a
                    # logical role. Independent physical components remain AND
                    # obligations in the retained typed macro contract; route
                    # consumers project those components directly.
                    terminal_accesses = {role: (point,) for role, point in terminals.items()}
            else:
                terminal_accesses = view.terminal_accesses
                term_layers = view.term_layers
            member_cx, member_cy = tcx + ox, tcy + oy
            vx0, vy0, vx1, vy1 = view.bbox
            half_w = max(
                1, (vx1 - vx0 + 1) // 2,
                *(abs(point[0] - member_cx) for point in terminals.values()),
            )
            half_h = max(
                1, (vy1 - vy0 + 1) // 2,
                *(abs(point[1] - member_cy) for point in terminals.values()),
            )
            devices[name] = replace(
                view,
                bbox=(member_cx - half_w, member_cy - half_h,
                      member_cx + half_w, member_cy + half_h),
                terminals=terminals,
                term_layers=term_layers,
                terminal_accesses=terminal_accesses,
                polys=[],
                geometry_owner=entry.owner,
                orientation=entry.orientation,
            )
        macro_polys.extend(placed_macro_polys)
        macro_polys_by_owner[entry.owner] = tuple(placed_macro_polys)
        bindings.append(RowMacroBlockBinding(
            entry_id=entry.entry_id,
            entry_sha256=entry.sha256,
            materialization_sha256=materialization.certificate.sha256,
            artwork_sha256=_placed_artwork_sha256(placed_macro_polys),
            owner=entry.owner,
            consumed_devices=entry.consumed_devices,
            bbox_nm=actual_macro_bbox,
        ))
        consumed.update(entry.consumed_devices)
    free_names = tuple(sorted(name for name in devices if name not in consumed))
    raw_free_physical = placement.row_free_physical_bboxes
    if type(raw_free_physical) is not dict:
        raise RowFinalPlacementResidualError((
            "ROW_FREE_PHYSICAL_BBOX_AUTHORITY_MALFORMED",
        ))
    if raw_free_physical:
        if (
            any(type(name) is not str or not name for name in raw_free_physical)
            or set(raw_free_physical) != set(free_names)
        ):
            raise RowFinalPlacementResidualError((
                "ROW_FREE_PHYSICAL_BBOX_INVENTORY_MISMATCH",
            ))
        try:
            free = tuple(
                (
                    name,
                    _validate_row_master_bbox(
                        raw_free_physical[name], "free physical device bbox"
                    ),
                )
                for name in free_names
            )
        except (TypeError, ValueError) as exc:
            raise RowFinalPlacementResidualError((
                f"ROW_FREE_PHYSICAL_BBOX_AUTHORITY_MALFORMED:{exc}",
            )) from exc
    else:
        free = tuple((name, devices[name].bbox) for name in free_names)
    free_polys = [
        poly for name, pd in devices.items()
        if name not in consumed for poly in (pd.polys or [])
    ]
    auxiliary_polys = _subtract_polygon_multiset(aggregate_remainder, macro_polys)
    auxiliary_polys = _subtract_polygon_key_multiset(
        auxiliary_polys, prior_bridge_keys
    )
    # Shared-well bridges are aggregate artwork whose coordinates are derived
    # from the current placed nwell islands.  Keeping an old bridge byte-for-byte
    # after an atomic re-place strands it at the previous coordinates.  Preserve
    # every other auxiliary occurrence exactly, but regenerate this one known
    # aggregate class from the newly seated macro/free artwork.
    generated_bridges: List[object] = []
    if row_tech is not None and _nwell_bridge_generation_enabled(row_tech):
        try:
            nwell_key = row_tech.build_layer("nwell")
        except Exception:  # noqa: BLE001 - a PDK without nwell has no bridge class
            nwell_key = None
        if nwell_key is not None:
            generated_bridges = _shared_nwell_bridges(
                [*macro_polys, *free_polys, *auxiliary_polys],
                tuple(nwell_key),
            )
    all_polys = [
        *macro_polys, *free_polys, *auxiliary_polys, *generated_bridges,
    ]
    if placement.row_native_geometry_binding is not None:
        from clay.layout.place.row_native_geometry import RowNativeGeometryBinding

        if type(placement.row_native_geometry_binding) is not RowNativeGeometryBinding:
            raise TypeError("stored row projection requires exact native geometry binding")
        placement.row_native_geometry_binding.require_row_artwork(
            macro_bindings=tuple(bindings), macro_polys_by_owner=macro_polys_by_owner,
        )
    nonmos_binding = None
    if placement.rigid_nonmos_macro_plans is not None:
        from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanSet
        from clay.layout.source_bound_nonmos_plan import SourceBoundNonmosMacroPlan
        from clay.layout.nonmos_placement_geometry import capture_nonmos_assembly_placement

        if type(placement.rigid_nonmos_macro_plans) is not RigidNonmosMacroPlanSet:
            raise TypeError("stored row projection requires exact nonmos macro plans")
        plan_set = replace(placement.rigid_nonmos_macro_plans)
        native_plans = tuple(p for p in plan_set.plans if type(p) is SourceBoundNonmosMacroPlan)
        if native_plans:
            if (physical_graph is None or plan_set.graph_sha256 != physical_graph.sha256
                    or plan_set.tech_id != program.tech_id):
                raise ValueError("nonmos placement plans differ from the current row source scope")
            nonmos_binding = capture_nonmos_assembly_placement(native_plans, devices=devices)
    if prior_certificate is not None and prior_certificate.nonmos_assembly_binding is not None:
        from clay.layout.nonmos_placement_geometry import NonmosAssemblyPlacementBinding

        previous_nonmos = prior_certificate.nonmos_assembly_binding
        if type(previous_nonmos) is not NonmosAssemblyPlacementBinding:
            raise TypeError("stored row certificate requires exact nonmos assembly binding")
        previous_nonmos.validate()
        if nonmos_binding is None:
            raise RowFinalPlacementResidualError(("ROW_NONMOS_ASSEMBLY_BINDING_REMOVED",))
        if previous_nonmos.plan_sha256s != nonmos_binding.plan_sha256s:
            raise RowFinalPlacementResidualError(("ROW_NONMOS_ASSEMBLY_PLAN_IDENTITY_CHANGED",))
    if placement.native_row_cp_frame_proof is not None:
        from clay.layout.place.native_row_cp_frames import replay_native_row_cp_frame_proof
        from clay.layout.tech import LayoutTech
        signed_centres = replay_native_row_cp_frame_proof(
            placement.native_row_cp_frame_proof, program, materialization,
            LayoutTech.resolve(program.tech_id))
        by_id = {row.entry_id: row for row in bindings}
        final_transport = placement.native_row_final_transported
        for entry in program.entries:
            key = 'A_' + entry.group
            if key not in signed_centres:
                continue
            bound = by_id.get(entry.entry_id)
            if bound is None:
                raise ValueError('native row CP frame owner is missing from final row projection')
            box = bound.bbox_nm
            centre2 = (box[0] + box[2], box[1] + box[3])
            if not final_transport and centre2 != signed_centres[key]:
                raise ValueError('native row seated pose differs from CP frame proof')
            realization = by_entry[entry.entry_id]
            local = realization.local_bbox_nm
            from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
            grid = pdk_manufacturing_grid_nm(LayoutTech.resolve(program.tech_id))
            if any((box[i] - local[i]) % grid for i in (0,1)):
                raise ValueError('native row seated artwork is off the PDK manufacturing grid')
        if placement.native_same_row_final_proof is not None:
            source_centres = dict(placement.native_same_row_final_proof.source_centres2_nm)
            for entry in program.entries:
                if entry.entry_id in source_centres and source_centres[entry.entry_id] != signed_centres.get('A_'+entry.group):
                    raise ValueError('native same-row final proof differs from source CP frame')
    try:
        certificate = RowFinalPlacementCertificate.build(
            program_sha256=program.sha256,
            macro_bindings=tuple(bindings),
            free_device_blocks=free,
            placed_artwork=all_polys,
            physical_macro_graph_sha256=(
                physical_graph.sha256 if physical_graph is not None else ""
            ),
            row_macro_contracts_sha256=(
                macro_contracts.sha256 if macro_contracts is not None else ""
            ),
            native_geometry_binding=placement.row_native_geometry_binding,
            nonmos_assembly_binding=nonmos_binding,
            native_cross_row_cp_proof=placement.native_cross_row_cp_proof,
            native_cross_row_final_proof=placement.native_cross_row_final_proof,
            native_same_row_final_proof=placement.native_same_row_final_proof,
            native_row_cp_frame_proof=placement.native_row_cp_frame_proof,
            native_row_final_transported=placement.native_row_final_transported,
        )
        collision_residuals = ()
    except RowFinalPlacementResidualError as exc:
        if not allow_collision_residual:
            raise
        certificate = None
        collision_residuals = exc.residuals
    if devices:
        extent = _row_blocks_bbox(
            [(binding.owner, binding.bbox_nm) for binding in bindings] + list(free)
        )
        geom = _cell_geom(devices, extent)
    else:
        geom = placement.geom
    return replace(
        placement,
        devices=devices,
        polys=all_polys,
        geom=geom,
        row_final_certificate=certificate,
        row_residuals=collision_residuals,
        row_generated_nwell_bridge_keys=tuple(
            _polygon_multiset_key(poly) for poly in generated_bridges
        ),
        matching_pair_axes=_project_stored_row_matching_axes(
            placement, devices, program, row_certificate=certificate,
        ),
    )


def verify_stored_row_projection(placement: Placement) -> bool:
    """Deep-replay the stored realization against final artwork and certificate."""
    if placement.row_program is None and placement.row_materialization is None:
        if (placement.row_final_certificate is not None or placement.row_residuals
                or placement.row_native_geometry_binding is not None):
            raise RowFinalPlacementResidualError(("ROW_MASTER_METADATA_INCONSISTENT",))
        return True
    if placement.row_residuals:
        raise RowFinalPlacementResidualError(tuple(placement.row_residuals))
    if placement.row_final_certificate is None:
        raise RowFinalPlacementResidualError(("ROW_FINAL_CERTIFICATE_MISSING",))
    if (placement.row_native_geometry_binding
            is not placement.row_final_certificate.native_geometry_binding):
        raise RowFinalPlacementResidualError(("ROW_NATIVE_GEOMETRY_BINDING_MISMATCH",))
    _verify_row_program_member_inventory(placement, placement.row_program)
    if (
        placement.physical_macro_graph is None
        and placement.row_macro_contracts is None
    ):
        raise RowFinalPlacementResidualError((
            "ROW_MACRO_STRUCTURE_METADATA_REMOVED",
        ))
    if (
        placement.physical_macro_graph is None
        or placement.row_macro_contracts is None
    ):
        raise RowFinalPlacementResidualError((
            "ROW_MACRO_STRUCTURE_METADATA_INCOMPLETE",
        ))
    if (
        placement.row_final_certificate.placed_artwork_sha256
        != _placed_artwork_sha256(placement.polys)
    ):
        raise RowFinalPlacementResidualError(("ROW_PLACEMENT_ARTWORK_MISMATCH",))
    expected = project_stored_row_master(
        replace(placement, row_final_certificate=None, row_residuals=())
    )
    actual_certificate = placement.row_final_certificate
    expected_certificate = expected.row_final_certificate
    if (
        expected_certificate is None
        or actual_certificate.to_canonical_bytes()
        != expected_certificate.to_canonical_bytes()
    ):
        raise RowFinalPlacementResidualError(("ROW_CERTIFICATE_REPLAY_MISMATCH",))
    if _placed_artwork_sha256(placement.polys) != _placed_artwork_sha256(expected.polys):
        raise RowFinalPlacementResidualError(("ROW_PLACEMENT_ARTWORK_MISMATCH",))
    consumed = {
        name: binding.owner
        for binding in actual_certificate.macro_bindings
        for name in binding.consumed_devices
    }
    for name, owner in consumed.items():
        view = placement.devices.get(name)
        expected_view = expected.devices.get(name)
        if (
            view is None
            or expected_view is None
            or view.geometry_owner != owner
            or view.polys
            or view.bbox != expected_view.bbox
            or view.terminals != expected_view.terminals
            or view.orientation != expected_view.orientation
            or view.term_layers != expected_view.term_layers
            or view.terminal_accesses != expected_view.terminal_accesses
        ):
            raise RowFinalPlacementResidualError((
                f"ROW_MEMBER_VIEW_MISMATCH:{name}",
            ))
    return True


def attach_row_master(
    placement: Placement, program: object, materialization: object
) -> Placement:
    if program is None and materialization is None:
        return placement
    return project_stored_row_master(
        replace(placement, row_program=program, row_materialization=materialization),
    )


def _bind_physical_macro_graph_views(
    placement: Placement, graph_sha256: str
) -> Placement:
    """Stamp or verify the graph identity on every materialized device view."""

    if (
        type(graph_sha256) is not str
        or len(graph_sha256) != 64
        or any(char not in "0123456789abcdef" for char in graph_sha256)
    ):
        raise PhysicalMacroProjectionError("MACRO_GRAPH_VIEW_MISMATCH")
    _verify_physical_macro_scope(
        placement.physical_macro_scope_token, graph_sha256
    )
    devices = {}
    for name, view in placement.devices.items():
        marker = getattr(view, "physical_macro_graph_sha256", "")
        if marker and marker != graph_sha256:
            raise PhysicalMacroProjectionError(
                f"MACRO_GRAPH_VIEW_MISMATCH:{name}"
            )
        devices[name] = replace(
            view, physical_macro_graph_sha256=graph_sha256
        )
    return replace(placement, devices=devices)


def _verify_physical_macro_graph_views(
    placement: Placement, graph_sha256: str
) -> None:
    try:
        _verify_physical_macro_scope(
            placement.physical_macro_scope_token,
            graph_sha256,
            compiler_binding=_physical_macro_compiler_binding(placement),
        )
    except PhysicalMacroProjectionError as exc:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_SCOPE_TOKEN_MISMATCH"
        ) from exc
    for name, view in placement.devices.items():
        if getattr(view, "physical_macro_graph_sha256", "") != graph_sha256:
            raise RigidNonmosMacroProjectionError(
                f"RIGID_NONMOS_GRAPH_VIEW_MISMATCH:{name}"
            )


def attach_physical_macro_projection(placement: Placement) -> Placement:
    """Issue the exact placed binding for every compiled non-row macro."""

    graph = placement.physical_macro_graph
    contracts = placement.physical_macro_contracts
    compile_inputs = placement.physical_macro_compile_inputs
    prior = placement.physical_macro_placement_certificate
    if graph is None and contracts is None and compile_inputs is None:
        if prior is not None:
            raise PhysicalMacroProjectionError("MACRO_STRUCTURE_METADATA_REMOVED")
        return attach_rigid_nonmos_projection(placement)
    if graph is None or contracts is None or compile_inputs is None:
        raise PhysicalMacroProjectionError("MACRO_STRUCTURE_METADATA_INCOMPLETE")
    from clay.layout.physical_macro_contracts import (
        PhysicalMacroCompileInputs,
        PhysicalMacroContractSet,
        bind_physical_macro_placement,
    )
    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    if (
        type(graph) is not PhysicalMacroGraph
        or type(contracts) is not PhysicalMacroContractSet
        or type(compile_inputs) is not PhysicalMacroCompileInputs
    ):
        raise PhysicalMacroProjectionError("MACRO_STRUCTURE_METADATA_TYPE_MISMATCH")
    if contracts.graph_sha256 != graph.sha256:
        raise PhysicalMacroProjectionError("MACRO_GRAPH_CONTRACT_MISMATCH")
    placement = _bind_physical_macro_graph_views(placement, graph.sha256)
    certificate = bind_physical_macro_placement(
        graph,
        contracts,
        placement.devices,
        compile_inputs=compile_inputs,
    )
    return attach_rigid_nonmos_projection(replace(
        placement, physical_macro_placement_certificate=certificate
    ))


def verify_physical_macro_projection(placement: Placement) -> bool:
    """Replay one stored non-row macro certificate against current placement."""

    certificate = placement.physical_macro_placement_certificate
    if certificate is None:
        metadata = (
            placement.physical_macro_graph,
            placement.physical_macro_contracts,
            placement.physical_macro_compile_inputs,
        )
        if all(item is None for item in metadata):
            raise PhysicalMacroProjectionError("MACRO_STRUCTURE_METADATA_REMOVED")
        if any(item is None for item in metadata):
            raise PhysicalMacroProjectionError("MACRO_STRUCTURE_METADATA_INCOMPLETE")
        raise PhysicalMacroProjectionError("MACRO_PLACEMENT_CERTIFICATE_MISSING")
    expected = attach_physical_macro_projection(
        replace(placement, physical_macro_placement_certificate=None)
    ).physical_macro_placement_certificate
    if (
        type(certificate) is not type(expected)
        or certificate.sha256 != expected.sha256
        or certificate.payload() != expected.payload()
    ):
        raise PhysicalMacroProjectionError("MACRO_PLACEMENT_CERTIFICATE_MISMATCH")
    return True


# === generic device-class / sizing inference =================================

# Functional-group role keywords that imply a device kind, used only when the
# device-name prefix is ambiguous (a bare ``M<digit>`` with no N/P marker). One
# uniform table for every circuit.
_PMOS_ROLE_HINTS = (
    "pmos", "pfet", "p_load", "pload", "tail",  # PMOS tail in a pmos-input amp
    "pass_device", "load_mirror", "fold_source",
)
_NMOS_ROLE_HINTS = (
    "nmos", "nfet", "n_load", "core_nmos", "mirror_load",
    "input_pair", "cascode", "sink",
)

# Non-MOS device-class prefixes / functional roles to skip (the placer places
# MOS transistors; resistors / caps / BJTs are not foldable MOS devices).
_NON_MOS_PREFIXES = ("R", "C", "Q", "D", "L", "X", "V", "I")


def _kind_from_name(name: str) -> Optional[str]:
    """Infer ``"nmos"`` / ``"pmos"`` from a device-name prefix, or ``None``.

    Convention: ``MN*`` -> nmos, ``MP*`` -> pmos. A bare ``M*`` without an N/P
    marker is ambiguous (resolved by the role hint). A non-``M`` leading letter is
    a non-MOS device (resistor / cap / BJT / instance) and returns ``None`` so the
    placer skips it -- it places MOS only.
    """
    n = name.strip()
    if not n:
        return None
    up = n.upper()
    if up[0] != "M":
        return None
    # M_op_inp / MN1 / MP2 / Mbias ... look at the first alphabetic after 'M'.
    rest = up[1:].lstrip("_")
    if rest.startswith("N"):
        return "nmos"
    if rest.startswith("P"):
        return "pmos"
    return None  # ambiguous bare M<digit> / M<word>; resolve via role hint


def _kind_from_role(role: str) -> Optional[str]:
    """Infer a device kind from a functional-group role keyword, or ``None``."""
    r = (role or "").lower()
    for h in _PMOS_ROLE_HINTS:
        if h in r:
            return "pmos"
    for h in _NMOS_ROLE_HINTS:
        if h in r:
            return "nmos"
    return None


def _device_kinds_from_cell_def(cd: CellDef) -> Dict[str, str]:
    """Build ``label -> "nmos"|"pmos"`` for every MOS device in the parsed cell.

    Source preference (uniform, generic):
      1. ``cd.devices`` (the sized-device list) when populated -- the authoritative
         ``device_class`` straight from the materialised IR.
      2. otherwise the topology ``functional_group`` device lists: each device's
         kind is inferred from its NAME prefix (``MN``/``MP``), falling back to the
         group's ``role`` keyword for an ambiguous bare ``M`` name.

    Non-MOS devices (resistors / caps / BJTs / instances) are excluded.
    """
    kinds: Dict[str, str] = {}

    # 1. sized devices win. The KIND is normalized to its MOS POLARITY, so a 3.3V
    # thick-ox kind (pmos3v/nmos3v) is accepted and stored as pmos/nmos --
    # byte-identical for a plain pmos/nmos, and no longer DROPPED for a 3v kind
    # (the oxide fact is carried separately by ``annotated_cell_from_cell_def``).
    for dev in getattr(cd, "devices", []) or []:
        pol, _ox = kind_polarity_oxide(getattr(dev, "device_class", None))
        if pol is not None:
            kinds[dev.instance_name] = pol
    if kinds:
        return kinds

    # 2. topology functional groups.
    for fr in getattr(cd, "functional_roles", []) or []:
        attrs = getattr(fr, "attributes", {}) or {}
        role = str(attrs.get("role", attrs.get("group", "")))
        devices = attrs.get("devices", []) or []
        role_kind = _kind_from_role(role)
        for name in devices:
            name = str(name)
            k = _kind_from_name(name)
            if k is None and name.upper().startswith("M"):
                # bare M<...> name (a MOS with no N/P marker, e.g. M_op_inp in a
                # summer-OTA group): resolve via the group's role keyword; if that
                # is also inconclusive default to nmos so the MOS is still PLACED
                # rather than silently dropped (a dropped pair member would crash
                # the row-based solver). The default never mislabels a clearly
                # marked device -- only an otherwise-unclassifiable MOS.
                k = role_kind or "nmos"
            # a non-``M`` device (resistor R*, cap C*, BJT Q*, ...) is not a
            # foldable MOS: leave it out so it never enters the placer.
            if k in ("nmos", "pmos"):
                kinds[name] = k
    return kinds


def _device_sizing_from_cell_def(
    cd: CellDef,
    kinds: Dict[str, str],
    cir_sizing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Build ``label -> {W, L, nf}`` (in metres / count) for each MOS device.

    Returns ``(sizing, used_default)``. Source preference (uniform, generic):

      1. ``cd.devices`` real sizing (when the IR carries sized devices);
      2. otherwise the bound source ``.cir`` per-device W / L / m, matched to the
         placer label by name (``cir_sizing``) -- this closes the sizing gap for
         the four production cells whose devices are sized in the netlist, not the
         ``.clay`` (so the placed geometry is the design's REAL geometry);
      3. otherwise the uniform default geometry (``used_default=True``, reported on
         the Placement and never hidden).
    """
    sized: Dict[str, Dict[str, float]] = {}
    for dev in getattr(cd, "devices", []) or []:
        if dev.instance_name in kinds and dev.sizing:
            sized[dev.instance_name] = dict(dev.sizing)

    cir_sizing = cir_sizing or {}
    used_default = False
    sizing: Dict[str, Dict[str, float]] = {}
    for lbl in kinds:
        s = sized.get(lbl)
        if s and ("W" in s or "L" in s):
            sizing[lbl] = s
            continue
        cs = cir_sizing.get(lbl)
        if cs and ("W" in cs or "L" in cs):
            sizing[lbl] = {k: cs[k] for k in ("W", "L", "nf") if k in cs}
            continue
        used_default = True
        sizing[lbl] = {
            "W": _DEFAULT_W_UM * 1e-6,
            "L": _DEFAULT_L_UM * 1e-6,
            "nf": _DEFAULT_NF,
        }
    return sizing, used_default


def annotated_cell_from_cell_def(
    cd: CellDef, source_cir=None, *, native_geometry=None
) -> Tuple[AnnotatedCell, bool]:
    """Build an :class:`AnnotatedCell` (the R185 placer's input) from a parsed
    ``CellDef``, carrying the parsed ``annotate`` annotations + the generically
    inferred device kinds and sizing.

    Returns ``(annotated_cell, used_default_geometry)``. This is the generic
    bridge: the four production ``design.clay`` cells parse to a ``CellDef`` whose
    topology is functional-group based and whose devices are sized externally;
    this turns that into exactly the shape :func:`annotations_to_solver_constraints`
    + :func:`_build_devices_from_clay` already consume.

    When ``source_cir`` resolves, the REAL per-device W / L / m is read from it
    (generic .cir parse, matched by device name) so the placed geometry is the
    design's actual geometry; the uniform default is the fallback only when no
    .cir is resolvable / a device is unmatched.
    """
    if native_geometry is None:
        kinds = _device_kinds_from_cell_def(cd)
        cir_sizing = parse_cir_device_sizing(source_cir) if source_cir else {}
        sizing, used_default = _device_sizing_from_cell_def(cd, kinds, cir_sizing)
    else:
        native_geometry.require_current(cd)
        kinds = {name: geom.kind for name, geom in native_geometry.geometries.items()}
        sizing = {name: {"W": geom.native_spec["w_nm"] * 1e-9,
                         "L": geom.native_spec["l_nm"] * 1e-9,
                         "nf": geom.native_spec["nf"]}
                  for name, geom in native_geometry.geometries.items()}
        used_default = False

    cell = AnnotatedCell(cell_name=cd.local_name)
    # carry the parsed device-level annotations (@matching / @symmetric / ...).
    ann_map = cd.__dict__.get("_layout_annotations", {}) or {}
    for target, anns in ann_map.items():
        for ann in anns:
            cell.add_annotation(target, ann)

    cell.__dict__["_device_kinds"] = kinds
    # Per-device gate-oxide domain (thick 3.3V / thin 1.8V) from the declared KIND,
    # an additive side map threaded into build_device(oxide=) by
    # ``_build_devices_from_clay``; absent -> thin. Only MOS devices with a declared
    # class carry it (a name-inferred kind from a functional_group owes no oxide).
    device_oxide: Dict[str, str] = {}
    for dev in getattr(cd, "devices", []) or []:
        pol, ox = kind_polarity_oxide(getattr(dev, "device_class", None))
        if pol is not None and getattr(dev, "instance_name", None):
            device_oxide[dev.instance_name] = ox
    cell.__dict__["_device_oxide"] = device_oxide
    cell.__dict__["_device_sizing"] = sizing
    from clay.port_roles import declared_substrate_ports

    cell.__dict__["_ground_nets"] = declared_substrate_ports(
        getattr(cd, "port_roles", {}) or {}
    )
    # carry each device's source/body net so the device-cell builder can decide the
    # SHARED-NWELL well-tap policy (a PMOS whose source != body draws no source-tied
    # tap; its bulk is the contiguous shared well). label -> {'s': net, 'b': net}.
    pins_map: Dict[str, Dict[str, str]] = {}
    for dev in getattr(cd, "devices", []) or []:
        lbl = getattr(dev, "instance_name", None)
        pins = getattr(dev, "pins", None) or {}
        if lbl is None or not pins:
            continue
        s = pins.get("s") or pins.get("source")
        b = pins.get("b") or pins.get("body") or pins.get("bulk")
        # d / g are carried too so the common-centroid cluster builder can common
        # each interleaved finger's drain to the OWNING device's drain net (the
        # SHARED-NWELL well policy uses only s / b, unchanged).
        d = pins.get("d") or pins.get("drain")
        g = pins.get("g") or pins.get("gate")
        pins_map[lbl] = {"s": str(s) if s is not None else "",
                         "b": str(b) if b is not None else "",
                         "d": str(d) if d is not None else "",
                         "g": str(g) if g is not None else ""}
    cell.__dict__["_device_pins"] = pins_map
    return cell, used_default


# === intent -> solver constraints (compose + intent overlay) =================


def _diff_net_sides_from_intent(
    cell_def: CellDef,
    intent: Optional[LayoutIntent],
    kinds: Dict[str, str],
) -> Dict[str, int]:
    """Build ``device_label -> -1 | +1`` half-plane assignment from the declared
    DIFFERENTIAL-NET pairs (``intent.diff_pairs``) + all device terminal pins.

    A matched net may reach a device through source, drain, gate, or body.  Restricting
    this lowering to drains silently missed stacked/source-connected mirror islands,
    allowing the same net identity to reverse sides between blocks.  Every device
    touching exactly one side of a declared pair now inherits that side; a device
    touching both is intentionally unassigned.  Only the first diff pair seen claims
    a device, preserving deterministic multi-pair behaviour.

    This drives the solver's half-plane constraint generically from the net-level
    intent; it never invents a device symmetry PAIR (that stays annotation-driven),
    it only records which half-plane each diff-net device belongs to.
    """
    if intent is None:
        return {}
    from clay.layout.net_match_scope import placement_diff_pairs

    pairs = placement_diff_pairs(intent)
    if not pairs:
        return {}
    device_nets: Dict[str, Set[str]] = {}
    for dev in getattr(cell_def, "devices", []) or []:
        lbl = getattr(dev, "instance_name", None)
        if lbl is None or lbl not in kinds:
            continue
        pins = getattr(dev, "pins", None) or {}
        device_nets[str(lbl)] = {
            str(net) for net in pins.values() if net is not None and str(net)
        }
    sides: Dict[str, int] = {}
    for (na, nb) in pairs:
        left_net, right_net = str(na), str(nb)
        for lbl in sorted(device_nets):
            nets = device_nets[lbl]
            touches_left = left_net in nets
            touches_right = right_net in nets
            if touches_left == touches_right:
                continue
            sides.setdefault(lbl, -1 if touches_left else +1)
    return sides


def _orient_symmetry_pairs_to_diff_sides(
    pairs: Sequence[Tuple[str, str]],
    sides: Mapping[str, int],
) -> Tuple[Tuple[str, str], ...]:
    """Orient declared pair identities so the diff-net left member is first.

    Device matching remains declaration-driven; this only chooses which member of
    an already-declared pair occupies which half-plane.  Unknown or contradictory
    side data leaves the source order unchanged and is caught by later physical
    polarity checks if it makes exact net mirroring impossible.
    """
    oriented = []
    for first, second in pairs:
        first_side = sides.get(str(first))
        second_side = sides.get(str(second))
        reverse = (
            (first_side == 1 and second_side in (-1, None))
            or (first_side is None and second_side == -1)
        )
        oriented.append(
            (str(second), str(first)) if reverse else (str(first), str(second))
        )
    return tuple(oriented)


def _cell_declares_place_group(cell_def: CellDef) -> bool:
    """True iff the parsed cell's ``layout_intent`` block declares a
    ``@place_group`` floorplan. A pure structural predicate over the parsed decls
    (kind == "place_group") -- no circuit / device-name branch. Used to gate the
    intent=None floorplan auto-derive so every cell that declares no floorplan
    stays byte-identical."""
    for decl in getattr(cell_def, "layout_intents", []) or []:
        if getattr(decl, "kind", None) == "place_group":
            return True
    return False


def _cell_declares_guard_ring(cell_def: CellDef) -> bool:
    """True iff the parsed cell declares any ``@guard_ring`` intent. A pure
    structural predicate (kind == "guard_ring") -- no circuit / device-name branch.
    Gates the intent=None auto-derive so a ring-declaring cell reserves ring room
    even when the caller passes no explicit intent; a cell that declares no ring is
    byte-identical (the derive fires only on the declaration)."""
    for decl in getattr(cell_def, "layout_intents", []) or []:
        if getattr(decl, "kind", None) == "guard_ring":
            return True
    return False


def _cell_declares_row_structure(cell_def: CellDef) -> bool:
    """True iff the parsed .clay declares a row-calculus structure."""
    return any(
        getattr(decl, "kind", None) in ("array", "common_centroid")
        for decl in (getattr(cell_def, "layout_intents", []) or [])
    )


class _RingRoomDevice:
    """A transparent view of a built MOS device whose reserved ``extent()`` is
    GROWN by the guard-ring outward extension + half the ring-to-neighbour
    clearance, so the placement solver reserves room for a declared ring around
    the device. Every OTHER attribute / method delegates to the real device, so
    the DRAWN polygons / terminals are untouched -- only the SPACING the solver
    reserves changes (the same ``group_ring_padded_extents`` contract, applied to
    the pre-solve footprint). Used ONLY for declared-ring member devices, so an
    unringed cell keeps the real device and places byte-identically."""

    __slots__ = ("_d", "_pad")

    def __init__(self, d: object, pad_nm: int) -> None:
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_pad", int(pad_nm))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_d"), name)

    def extent(self):
        e = object.__getattribute__(self, "_d").extent()
        p = object.__getattribute__(self, "_pad")
        return (e[0] - p, e[1] - p, e[2] + p, e[3] + p)


def _ring_room_padded_devs(
    devs: Dict[str, object], intent: Optional[LayoutIntent], tech
) -> Dict[str, object]:
    """A SOLVE-only view of ``devs`` in which every DECLARED-ring member's reserved
    footprint is padded with ring room (outward extension for its well polarity +
    half the ring-to-neighbour clearance), so the solver spaces the ringed cluster
    clear of its neighbours. Non-member devices pass through unchanged. Returns
    ``devs`` itself (identity) when the intent declares no ring -- byte-identical.

    The pad uses the SAME PDK-rule sizes ``realize_render.group_ring_padded_extents``
    reserves (single source, no second path); the drawn geometry is never touched
    (the returned members are :class:`_RingRoomDevice` proxies that only grow
    ``extent()``)."""
    rings = tuple(getattr(intent, "guard_rings", ()) or []) if intent else ()
    if not rings:
        return devs
    from clay.layout.extract.realize_render import (
        _deep_nwell_extension_nm,
        _group_ring_clearance_nm,
        _ring_extension_nm,
    )

    half_clear = (_group_ring_clearance_nm(tech) + 1) // 2
    pad_for_well = {
        "p": _ring_extension_nm(tech, "p") + half_clear,
        "n": _ring_extension_nm(tech, "n") + half_clear,
    }
    member_well: Dict[str, str] = {}
    isolated = set(
        str(name)
        for name in (getattr(intent, "well_isolated_devices", ()) or ())
    )
    for gr in rings:
        well = str(getattr(gr, "well", "") or "")
        for m in getattr(gr, "members", ()) or ():
            member_well.setdefault(str(m), well)
    out: Dict[str, object] = {}
    for lbl, d in devs.items():
        well = member_well.get(lbl)
        if well in ("p", "n"):
            pad = (
                _deep_nwell_extension_nm(tech) + half_clear
                if lbl in isolated
                else pad_for_well[well]
            )
            out[lbl] = _RingRoomDevice(d, pad)
        else:
            out[lbl] = d
    return out


def _constraints_from_intent_and_annotations(
    annotated: AnnotatedCell,
    intent: Optional[LayoutIntent],
    cell_def: Optional[CellDef] = None,
    *,
    tech: object = None,
) -> SolverConstraints:
    """Derive the solver constraints, composing the R185 translator and (when the
    annotate block declares no device pairs) the device-level intent.

    The primary driver is the parsed ``annotate`` block (device-level
    ``@matching`` / ``@symmetric``), translated by
    :func:`annotations_to_solver_constraints`. A :class:`LayoutIntent` is the
    NET-level contract (diff pairs on nets, budgets) and is recorded for
    provenance; net pairs are not device pairs, so the intent is NOT used to invent
    device symmetry pairs the annotate block did not declare. This keeps the device
    placement driven by the device-level declaration and avoids fabricating a
    device pairing from a net name.

    An exact-row DEVICE-level ``@place_group`` carries row/column band ownership;
    it is not a common-centroid declaration and therefore never mints a centroid
    constraint.  The historical non-exact multi-row range remains a balance
    preference for parity with the legacy placer.  Explicit matching/common-
    centroid authorities remain on their own typed paths.

    The NET-level diff pairs ARE used (when ``cell_def`` is given) to derive a
    half-plane SIDE assignment (``diff_net_sides``) -- not a device pair, only which
    side of the axis each diff-net drive column sits on. This makes a declared diff
    pair mirror-separable without inventing a device symmetry pairing.
    """
    from dataclasses import replace

    constraints = annotations_to_solver_constraints(annotated)

    # net-level diff pairs -> half-plane side assignment (mirror-separable columns).
    if cell_def is not None:
        kinds = getattr(annotated, "_device_kinds", {}) or {}
        diff_sides = _diff_net_sides_from_intent(cell_def, intent, kinds)
        if diff_sides:
            constraints = replace(
                constraints,
                diff_net_sides=diff_sides,
                symmetry_pairs=_orient_symmetry_pairs_to_diff_sides(
                    constraints.symmetry_pairs, diff_sides
                ),
            )

    # ── @order / @abut: cell-level placement constraints carried on the intent ───
    # These DRIVE the seqpair placer (relative-order edges + tight adjacency
    # spacing). They are forwarded verbatim from the declared intent; empty tuples
    # leave the placer's behaviour unchanged (byte-parity). Folded in BEFORE the
    # early ``return`` so a cell that declares ONLY @order/@abut (no @place_group)
    # still carries them.
    order_constraints = tuple(getattr(intent, "order_constraints", ()) or ()) if intent is not None else ()
    abut_pairs = tuple(getattr(intent, "abut_pairs", ()) or ()) if intent is not None else ()
    if order_constraints or abut_pairs:
        constraints = replace(
            constraints,
            order_constraints=order_constraints,
            abut_pairs=abut_pairs,
        )

    place_groups = getattr(intent, "place_groups", None) if intent is not None else None
    if not place_groups:
        return constraints

    cc = list(constraints.common_centroid_groups)
    seen = {frozenset(group) for group in cc}
    # A declared 2D FLOORPLAN: any @place_group pinned to a single row (row_min ==
    # row_max) assigns its devices to that floorplan row (+ optional col). This is
    # what makes the placement band-free and intent-driven -- the placer builds its
    # rows from this map instead of the NMOS-below-PMOS device-class band.
    floorplan: Dict[str, tuple] = {}
    for pg in place_groups:
        group = tuple(getattr(pg, "devices", ()) or ())
        row_min = getattr(pg, "row_min", None)
        row_max = getattr(pg, "row_max", None)
        exact_row = row_min is not None and row_max is not None and row_min == row_max
        if (
            not exact_row
            and row_min is not None
            and row_max is not None
            and len(group) >= 2
            and frozenset(group) not in seen
        ):
            cc.append(group)
            seen.add(frozenset(group))
        if exact_row:
            col = getattr(pg, "col", None)
            for d in group:
                floorplan[d] = (int(row_min), col)
    row_gap_minima = {}
    channel_routes = tuple(getattr(intent, "channel_routes", ()) or ())
    if channel_routes:
        from clay.layout.intent import place_group_floorplan_row
        from clay.layout.place.ordered_mirror_rows import OrderedRowError
        from clay.layout.route.channel_geometry import planned_channel_height_nm

        if tech is None:
            raise OrderedRowError("ORDERED_ROW_CHANNEL_REQUIRES_RESOLVED_PDK")
        group_rows = {group.group: place_group_floorplan_row(group) for group in place_groups}
        for channel in channel_routes:
            lower, upper = (group_rows.get(name) for name in channel.between)
            if lower is None or upper is None or lower >= upper:
                raise OrderedRowError("ORDERED_ROW_CHANNEL_GROUP_ROWS_INVALID")
            key = (lower, upper)
            row_gap_minima[key] = max(row_gap_minima.get(key, 0),
                planned_channel_height_nm(channel, tech))
    return replace(
        constraints,
        common_centroid_groups=tuple(cc),
        floorplan_rows=floorplan,
        floorplan_groups=tuple(sorted(
            (str(pg.group), tuple(sorted(str(name) for name in pg.devices)))
            for pg in place_groups if getattr(pg, 'row_min', None) is not None
            and getattr(pg, 'row_min', None) == getattr(pg, 'row_max', None))),
        row_gap_minima_nm=tuple((lower, upper, gap)
            for (lower, upper), gap in sorted(row_gap_minima.items())),
    )


def _restrict_constraints_to_devices(
    constraints: SolverConstraints, devs: Dict[str, object]
) -> Tuple[SolverConstraints, Tuple[str, ...]]:
    """Drop symmetry-pair / on-axis members the placer has no MOS geometry for.

    The parsed ``annotate`` block can declare ``@matching`` on devices that are NOT
    foldable MOS -- a bandgap cell matches its summing RESISTORS (``R_sum1`` /
    ``R_sum2``) and its PNP devices (``Q1`` / ``Q8_0``). Those are real matched
    pairs, but they are not gf180 foldable MOS, so ``_build_devices_from_clay``
    (correctly) does not build them. The row-based solver, however, assumes every
    symmetry-pair member has built geometry (it looks up ``half_w[member]``); an
    unbuilt member crashes it with a ``KeyError``.

    This is a robustness GUARD in the generic placer (NOT a fix to the composed
    solver): it filters the constraints so only pairs whose BOTH members were built
    as MOS survive into the placement, and an on-axis device must also be built.
    The dropped members are returned so the Placement can report them honestly as
    declared-but-not-MOS-placed.
    """
    from dataclasses import replace

    have = set(devs)
    kept_pairs = tuple(
        (a, b) for (a, b) in constraints.symmetry_pairs if a in have and b in have
    )
    dropped = tuple(
        m
        for (a, b) in constraints.symmetry_pairs
        for m in (a, b)
        if m not in have
    )
    kept_axis = tuple(d for d in constraints.on_axis_devices if d in have)
    kept_cc = tuple(
        g for g in constraints.common_centroid_groups if all(m in have for m in g)
    )
    kept_sides = {
        lbl: s for lbl, s in constraints.diff_net_sides.items() if lbl in have
    }
    kept_floorplan = {
        lbl: rc for lbl, rc in constraints.floorplan_rows.items() if lbl in have
    }
    # @order / @abut: keep only constraints whose BOTH members were built as MOS
    # (the seqpair placer looks up half_w/heights per member; an unbuilt member would
    # reference a device the longest-path never placed). A dropped constraint is
    # honestly reported as unmeasurable by the completeness gate (no covering check).
    kept_order = tuple(
        (a, b, rel) for (a, b, rel) in getattr(constraints, "order_constraints", ())
        if a in have and b in have
    )
    kept_abut = tuple(
        (a, b) for (a, b) in getattr(constraints, "abut_pairs", ())
        if a in have and b in have
    )
    restricted = replace(
        constraints,
        symmetry_pairs=kept_pairs,
        on_axis_devices=kept_axis,
        common_centroid_groups=kept_cc,
        diff_net_sides=kept_sides,
        floorplan_rows=kept_floorplan,
        floorplan_groups=tuple((name, tuple(member for member in members if member in have))
            for name, members in constraints.floorplan_groups),
        order_constraints=kept_order,
        abut_pairs=kept_abut,
    )
    return restricted, dropped


# === terminal extraction ======================================================


def _center(b: Optional[Bbox]) -> Optional[Tuple[int, int]]:
    if b is None:
        return None
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def _polys_bbox(polys: list) -> Optional[Bbox]:
    """Bounding box over a list of placed :class:`LayerPolygon` (or ``None``)."""
    xs: List[int] = []
    ys: List[int] = []
    for lp in polys:
        for v in lp.polygon.vertices:
            xs.append(v.x)
            ys.append(v.y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _shared_nwell_bridges(all_polys: list, nwell_key: Tuple[int, int]) -> list:
    """Bridge the placed PMOS per-device nwell boxes into ONE contiguous region.

    The SHARED-NWELL analog layout (standard PMOS-in-common-well): the comparator's
    PMOS share one nwell tied to vdd, so KLayout must extract them as a SINGLE well.
    The placer emits each PMOS with its OWN nwell box; adjacently-placed PMOS leave
    sub-cell gaps (the dev-gap channel between stacked / side-by-side devices) that
    leave the per-device nwell boxes DISJOINT, so the extractor would see N floating
    wells instead of one vdd-tied well.

    This adds thin nwell STRAPS that fill the gap between any two nwell boxes that
    overlap on one axis and have a clear gap on the OTHER (a vertical strap over the
    x-overlap for a y-gap; a horizontal strap over the y-overlap for an x-gap). The
    straps only ever fill the region BETWEEN two PMOS nwell boxes -- never extend the
    well past the PMOS collective footprint toward the NMOS band (the placer already
    spaces the bands), so no nwell-over-ncomp / NW.x rule is introduced. Iterated
    (union-find) until every nwell box is in ONE connected component, or no further
    single-axis bridge exists (a residual is reported by the caller, never hidden).

    Returns the list of added nwell :class:`LayerPolygon` straps (empty when the
    boxes are already contiguous, or fewer than two boxes exist).
    """
    from .backend import LayerPolygon, Polygon

    boxes: list = []
    for lp in all_polys:
        if (lp.layer, lp.datatype) != nwell_key:
            continue
        xs = [v.x for v in lp.polygon.vertices]
        ys = [v.y for v in lp.polygon.vertices]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    if len(boxes) < 2:
        return []

    from .spatial.component_bridges import closest_gap_bridges
    bridges: list = []
    for x0, y0, x1, y1 in closest_gap_bridges(boxes):
        if x1 <= x0 or y1 <= y0:
            continue
        bridges.append(
            LayerPolygon(
                layer=nwell_key[0],
                datatype=nwell_key[1],
                polygon=Polygon.rectangle(int(x0), int(y0), int(x1), int(y1)),
            )
        )

    return bridges


def _placed_device_centroid_x(base_dev: object, dx: int) -> int:
    """The vertical centre-line x of a device AFTER it is translated by ``dx``.

    This is the device's own placed centroid line -- the axis a mirrored member is
    reflected about so its centroid (and bbox) stay exactly where the translate put
    them (keeping the placement coordinates + symmetry score unchanged) while the
    physical S/D columns swap. Derived from the base device extent."""
    e = base_dev.extent()
    return ((e[0] + e[2]) // 2) + dx


def _placed_bbox(
    base_dev: object, dx: int, dy: int, mirror: bool
) -> Optional[Bbox]:
    """The PLACED reserved-extent bbox of a device, or ``None`` when the device has
    no ``placement_extent`` override (the caller then falls back to the drawn-polys
    bbox -- byte-unchanged for gf180 + every existing device).

    A SHARED-NWELL PMOS that suppressed its well tap reserves the TAP-INCLUSIVE
    extent (``placement_extent``); this returns that box translated by ``(dx, dy)``
    and, for a mirrored member, reflected about the device's OWN placed centroid
    line (the same reflection :func:`_place_polys` applies), so the placed bbox the
    router channel geometry reads is byte-identical to the tap version."""
    pe = getattr(base_dev, "placement_extent", None)
    if pe is None:
        return None
    x0, y0, x1, y1 = pe
    box = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    if mirror:
        ref_x = _placed_device_centroid_x(base_dev, dx)
        box = (2 * ref_x - box[2], box[1], 2 * ref_x - box[0], box[3])
    return box


def _placed_terminals(
    base_dev: object,
    dx: int,
    dy: int,
    mirror: bool,
    axis_x: int,
    use_tap_terminal: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Extract the s / d / g (+ bulk) terminal CENTRES of a device AFTER it is
    placed at (dx, dy) and, for a mirrored member, REFLECTED so its S/D physical
    columns swap.

    The terminal centres come from the foldable device's rail / contact bboxes
    (the same geometry the GDS carries), shifted by the solver's translation
    ``(dx, dy)``. The solver positions the right-hand member of a pair at the
    mirror-image x about ``axis_x`` (``target_cx = axis + side*off``) via a
    TRANSLATE, which lands the device CENTROID on its symmetric side. P1 fix: for a
    mirrored member we ADDITIONALLY reflect each terminal bbox about the device's
    OWN placed centroid line, so which PHYSICAL diffusion column carries ``'s'`` vs
    ``'d'`` is SWAPPED -- exactly what the hand router realises with
    ``left`` + ``_mirror_x(left, axis)``. Reflecting about the device's own
    centroid (not the cell axis) keeps the centroid + bbox identical to the
    translate-only placement, so the placement COORDINATES and the symmetry score
    (==1.0) are unchanged; only the terminal-to-physical-column mapping flips. The
    earlier code skipped this reflection and so attached the .cir source/drain net
    to the WRONG physical column on every mirrored member.
    """
    ref_x = _placed_device_centroid_x(base_dev, dx) if mirror else None

    def _shift(b: Optional[Bbox]) -> Optional[Tuple[int, int]]:
        if b is None:
            return None
        tb = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        if ref_x is not None:
            tb = (2 * ref_x - tb[2], tb[1], 2 * ref_x - tb[0], tb[3])
        return _center(tb)

    terms: Dict[str, Tuple[int, int]] = {}
    s = _shift(getattr(base_dev, "source_rail_bbox", None))
    d = _shift(getattr(base_dev, "drain_rail_bbox", None))
    g = _shift(getattr(base_dev, "gate_contact_bbox", None))
    if s is not None:
        terms["s"] = s
    if d is not None:
        terms["d"] = d
    if g is not None:
        terms["g"] = g
    # PMOS bulk tie terminal -- keyed PURELY on the device's own geometry (does the
    # built device expose a drawn tap-pad bbox?), never on a PDK name / circuit / net.
    # Three device-geometry cases:
    #
    #   (A) device DREW a source-tied tap pad (``terminals['b'].bbox`` present) and we
    #       are NOT on the standalone-riser path (``use_tap_terminal`` False). The drawn
    #       tap discharges the bulk tie INSIDE the cell: the in-well tap M1 already rises
    #       to the source rail, so the well body reaches its bulk net without any
    #       external help. Exposing a router ``b`` terminal here is REDUNDANT -- the
    #       router would land a separate rail-tap via12 pad at the (else-resolved) Nwell
    #       centre, ~120nm off the device's real tap M1 pad, grazing but never merging
    #       it -> same-net M1 min-spacing notches (the 42 gf180 M1.2a). So a device that
    #       drew its own tap exposes NO router ``b`` terminal. This GENERALIZES the
    #       shared-nwell treatment (case C) to tap-DRAWING self-bulk devices: in BOTH
    #       the per-device bulk tie is already satisfied in-cell, so the router needs no
    #       riser. (Replaces the earlier net-label/PDK-name ``self_bulk_pmos`` suppress.)
    #
    #   (B) ``use_tap_terminal`` -- a device whose builder draws the in-well tap as a
    #       STANDALONE M1 pad NOT auto-tied to the source rail (the line-end PDKs). Here
    #       the bulk tie is NOT discharged in-cell, so the router MUST rise to the tap's
    #       own M1 pad to connect the well tie to bulk. The router ``b`` terminal is the
    #       drawn tap-pad bbox (the device's ``b`` terminal). Unregressed external-riser
    #       path; keeps working for line-end PDKs.
    #
    #   (C) device drew NO tap pad (``draw_well_tap=False`` shared-nwell, or a pin ``b``
    #       carrying no drawn bbox). A shared-nwell device has no ``b`` terminal in its
    #       DeviceGeom at all -> no router ``b``. A ``b`` pin with no drawn pad falls back
    #       to the Nwell-centre coordinate (last resort, no in-cell tap to merge to).
    dev_terms = getattr(base_dev, "terminals", None)
    bt = (
        dev_terms.get("b")
        if dev_terms is not None and hasattr(dev_terms, "get")
        else None
    )
    drawn_tap_bbox = getattr(bt, "bbox", None) if bt is not None else None
    b_box = None
    if bt is not None:
        if (use_tap_terminal or bool(getattr(base_dev, "well_tap_isolated", False))
                or str(getattr(base_dev, "kind", "")) == "nmos"):
            # (B) standalone-riser path: the router rises to the device's tap M1 pad.
            b_box = drawn_tap_bbox if drawn_tap_bbox is not None else \
                getattr(base_dev, "nwell_bbox", None)
        elif drawn_tap_bbox is not None:
            # (A) device drew a source-tied tap -> bulk already tied in-cell, suppress.
            b_box = None
        else:
            # (C) ``b`` pin with no drawn pad -> Nwell-centre fallback.
            b_box = getattr(base_dev, "nwell_bbox", None)
    if b_box is not None:
        b = _shift(b_box)
        if b is not None:
            terms["b"] = b
    return terms


def _placed_terminal_accesses(
    base_dev: object,
    dx: int,
    dy: int,
    mirror: bool,
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Translate every materializer-owned access with the placed device artwork."""

    raw_accesses = getattr(base_dev, "terminal_accesses", {}) or {
        role: (terminal,)
        for role, terminal in (getattr(base_dev, "terminals", {}) or {}).items()
    }
    if not raw_accesses:
        return {}
    ref_x = _placed_device_centroid_x(base_dev, dx) if mirror else None
    placed: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for role, accesses in sorted(raw_accesses.items()):
        points = []
        for terminal in tuple(accesses or ()):
            box = getattr(terminal, "bbox", None)
            if box is None:
                raise ValueError(f"terminal access {role!r} lacks a drawn bbox")
            translated = (
                box[0] + dx,
                box[1] + dy,
                box[2] + dx,
                box[3] + dy,
            )
            if ref_x is not None:
                translated = (
                    2 * ref_x - translated[2],
                    translated[1],
                    2 * ref_x - translated[0],
                    translated[3],
                )
            points.append(_center(translated))
        if points:
            placed[str(role)] = tuple(points)
    return placed


def _place_polys(base_dev: object, dx: int, dy: int, mirror: bool, axis_x: int) -> list:
    """Return the device polygons placed at the solved position.

    Composes the existing :func:`backend._translate` helper (the same primitive the
    GDS emitter uses). The solver's ``dx`` translates the device centroid to its
    symmetric x about the cell axis. P1 fix: a MIRRORED member is additionally
    reflected (``backend._mirror_x``) about its OWN placed centroid line, so its
    physical S/D columns swap to match the reflected terminal map. Reflecting about
    the device's own centroid keeps the placed bbox + centroid identical to the
    translate-only result, so the placement coordinates + symmetry score are
    unchanged; only the diffusion-column orientation flips (the connectivity fix)."""
    polys = list(getattr(base_dev, "polygons", []) or [])
    placed = _translate(polys, dx, dy)
    if mirror:
        ref_x = _placed_device_centroid_x(base_dev, dx)
        placed = _mirror_x(placed, ref_x)
    return placed


def _placed_discrete_mos_occurrences(
    base_dev: object,
    dx: int,
    dy: int,
    mirror: bool,
    placed_bbox: Bbox,
    tech: object,
) -> Tuple[Tuple[float, float, str], ...]:
    """Project evaluated unit channels into the placed stamp's local frame."""

    from clay.layout.extract.realize_render import _physical_mos_occurrences_nm

    rows = _physical_mos_occurrences_nm(base_dev, tech)
    ref_x = _placed_device_centroid_x(base_dev, dx) if mirror else None
    mirrored_orientation = {
        "R0": "MR180",
        "R90": "MR90",
        "R180": "MR0",
        "R270": "MR270",
    }
    cx = (placed_bbox[0] + placed_bbox[2]) / 2.0
    cy = (placed_bbox[1] + placed_bbox[3]) / 2.0
    placed = []
    for x, y, orientation in rows:
        px = int(x) + int(dx)
        py = int(y) + int(dy)
        if ref_x is not None:
            px = 2 * int(ref_x) - px
            orientation = mirrored_orientation[str(orientation)]
        placed.append(
            ((px - cx) / 1000.0, (py - cy) / 1000.0, str(orientation))
        )
    return tuple(sorted(placed))


def _placed_resolved_master_occurrences(
    base_dev: object,
    mirror: bool,
) -> Tuple[Tuple[float, float, str], ...]:
    """Preserve fixed-master centers in the final stamp's local frame."""

    rows = tuple(getattr(base_dev, "resolved_master_occurrences", ()) or ())
    # A resolved master can itself contain several transistor fingers. Preserve
    # its declared replicas before bridging discards the discrete-array object.
    # The replica frame is generated local metadata, never placement/reference XY.
    fixed = getattr(base_dev, "fixed_pcell", None)
    structure = getattr(base_dev, "discrete_array", None)
    if not rows and type(fixed) is dict and "unit_occurrences" in fixed and structure is not None:
        offsets = getattr(structure, "replica_offsets_nm", None)
        count = fixed["unit_occurrences"]
        extent = tuple(base_dev.extent())
        orientation = str(getattr(base_dev, "orient", ""))
        if (type(count) is not int or count <= 0 or type(offsets) is not tuple
                or len(offsets) != count or len(extent) != 4
                or any(type(value) is not int for value in extent)
                or extent[0] >= extent[2] or extent[1] >= extent[3]
                or any(type(point) is not tuple or len(point) != 2
                       or any(type(value) is not int for value in point) for point in offsets)
                or orientation != "R0"):
            raise ValueError("fixed resolved-master replicas lack an exact current local frame")
        cx, cy = (extent[0] + extent[2]) / 2.0, (extent[1] + extent[3]) / 2.0
        rows = tuple(((x - cx) / 1000.0, (y - cy) / 1000.0, orientation) for x, y in offsets)
    if not rows:
        return ()
    mirrored_orientation = {
        "R0": "MR180",
        "R90": "MR90",
        "R180": "MR0",
        "R270": "MR270",
        "MR0": "R180",
        "MR90": "R90",
        "MR180": "R0",
        "MR270": "R270",
    }
    if not mirror:
        return tuple(sorted(
            (float(x), float(y), str(orientation))
            for x, y, orientation in rows
        ))
    return tuple(sorted(
        (-float(x), float(y), mirrored_orientation[str(orientation)])
        for x, y, orientation in rows
    ))


# === floorplan-honoring deterministic realizer (MILP fallback) =================
#
# The composed single-axis MILP (``solve_placement_from_constraints``) models every
# UNPAIRED row member with its x-centre FIXED to the one global axis variable (it has
# no per-device x offset). Two unpaired devices in the same floorplan row therefore
# both sit at x == axis, so the in-row non-overlap constraint
# ``xcentre(b) - xcentre(a) >= half_w[a] + half_w[b] + gap`` reduces to ``0 >= gap`` --
# INFEASIBLE. A declared @place_group floorplan that puts several unpaired devices in
# one row (e.g. the two_stage_ref top/bottom bands carry 4-6 unpaired MOS per row)
# thus makes the MILP report ``success=False`` and place_generic would drop ALL the MOS
# (only the passives, placed on a separate path, survive). That is a REALIZATION gap:
# an engine that silently drops every declared MOS cannot satisfy the intent.
#
# This realizer is the generic fallback: it consumes the SAME ``_row_plan`` (declared
# @place_group floorplan rows, mirror pairs kept together) the MILP consumes, and packs
# each row LEFT-TO-RIGHT with a real per-device x cursor -- overlap-free by construction
# (the cursor only advances). Mirror pairs are emitted as two members straddling a row-
# local centre (left member, then right member marked ``mirror=True``) so the declared
# symmetry survives. It is GENERIC (driven only by the row plan + built geometry, no
# circuit-/net-name branch) and is used ONLY when the exact MILP fails -- a cell the MILP
# CAN realize (e.g. the proven gf180 comparator: one unpaired device per row) keeps the
# exact MILP solution byte-for-byte, so gf180 parity is untouched.


def _floorplan_pack_solution(
    constraints: SolverConstraints,
    devs: Dict[str, object],
    tech,
    *,
    base_status: str,
) -> PlacementSolution:
    """Deterministic overlap-free realizer honoring the declared floorplan rows.

    Builds a :class:`PlacementSolution` of the SAME shape the MILP returns (each
    :class:`PlacedDevice` carries ``dx`` / ``dy`` from the device's native
    ``extent()`` centre, ``mirror``, ``extent``, ``kind``, ``row``) so every
    downstream consumer in :func:`place_generic` is unchanged.

    Packing rule (generic):
      * rows come from :func:`_row_plan` (declared @place_group floorplan, bottom-to-top);
      * within a row, cells pack left-to-right at a cursor that advances by
        ``half_w[prev] + half_w[cur] + gap`` (the same inter-device gap the MILP uses,
        opposite-type aware), so adjacent bboxes never overlap;
      * a mirror pair places its left member then its right member; the right member is
        flagged ``mirror=True`` (the physical S/D columns swap, as for the MILP);
      * each row's y-centre stacks above the previous by half each row height + gap, so
        rows never overlap on y either.

    The global ``axis_x`` is the mean device-centre x (the cell's vertical centre line)
    -- a real symmetry axis for the matched pairs that DID land symmetric within a row.
    """
    dev_gap = _dev_gap_from_tech(tech)
    half_w = _half_widths(devs)
    heights = _heights(devs)
    rows = _row_plan(constraints, devs)
    kinds = constraints.device_kinds

    def _gap(a: str, b: str) -> int:
        return _pair_gap(tech, dev_gap, kinds.get(a, "nmos"), kinds.get(b, "nmos"))

    # row heights (max device height in the row) -> stacked y-centres bottom-to-top.
    row_cells: List[List[str]] = []
    for row in rows:
        cells: List[str] = []
        for kind, payload in row:
            if kind == "pair":
                a, b = payload  # type: ignore[misc]
                cells += [a, b]
            else:
                cells.append(payload)  # type: ignore[arg-type]
        row_cells.append(cells)
    row_height = [
        (max(heights[l] for l in cells) if cells else 0) for cells in row_cells
    ]
    y_centre: List[int] = []
    cy = 0
    for r in range(len(rows)):
        if r == 0:
            cy = row_height[0] // 2
        else:
            cy = (
                y_centre[r - 1]
                + row_height[r - 1] // 2
                + dev_gap
                + row_height[r] // 2
            )
        y_centre.append(cy)

    # x-centre assignment: pack each row left-to-right; a mirror pair's two members are
    # adjacent (left, right) with the right member mirrored.
    target_cx: Dict[str, int] = {}
    side: Dict[str, int] = {}      # -1 left member / +1 right member / 0 unpaired
    dev_row_idx: Dict[str, int] = {}
    for r, row in enumerate(rows):
        cursor: Optional[int] = None
        prev: Optional[str] = None
        for kind, payload in row:
            members = list(payload) if kind == "pair" else [payload]  # type: ignore[arg-type]
            for k, lbl in enumerate(members):
                hw = half_w[lbl]
                if cursor is None:
                    cx = hw
                else:
                    # advance the cursor by the previous half-width + gap + this
                    # half-width (centre-to-centre clearance) -> no bbox overlap.
                    cx = cursor + half_w[prev] + _gap(prev, lbl) + hw  # type: ignore[index]
                target_cx[lbl] = cx
                cursor = cx
                prev = lbl
                dev_row_idx[lbl] = r
                if kind == "pair":
                    side[lbl] = -1 if k == 0 else +1
                else:
                    side[lbl] = 0

    if target_cx:
        axis_x = int(round(sum(target_cx.values()) / len(target_cx)))
    else:
        axis_x = 0

    placed: Dict[str, _SolverPlacedDevice] = {}
    for lbl, d in devs.items():
        e = d.extent()
        cur_cx = (e[0] + e[2]) // 2
        cur_cy = (e[1] + e[3]) // 2
        dx = target_cx.get(lbl, 0) - cur_cx
        dy = y_centre[dev_row_idx.get(lbl, 0)] - cur_cy
        ext = (e[0] + dx, e[1] + dy, e[2] + dx, e[3] + dy)
        placed[lbl] = _SolverPlacedDevice(
            label=lbl,
            kind=kinds.get(lbl, "nmos"),
            dx=dx,
            dy=dy,
            extent=ext,
            mirror=(side.get(lbl, 0) == 1),
            row=dev_row_idx.get(lbl, 0),
        )

    if placed:
        full_extent = (
            min(p.extent[0] for p in placed.values()),
            min(p.extent[1] for p in placed.values()),
            max(p.extent[2] for p in placed.values()),
            max(p.extent[3] for p in placed.values()),
        )
    else:
        full_extent = (0, 0, 0, 0)

    return PlacementSolution(
        placed=placed,
        axis_x=axis_x,
        status=(
            f"floorplan_pack_fallback (exact MILP infeasible: {base_status})"
        ),
        success=bool(placed),
        solve_seconds=0.0,
        objective=float("nan"),
        n_vars=0,
        n_constraints=0,
        full_extent=full_extent,
        constraints=constraints,
        note=(
            "exact single-axis MILP could not realize the declared floorplan "
            "(multiple unpaired devices per row collapse to one axis x -> infeasible); "
            "realized all built MOS overlap-free via the generic floorplan-pack "
            "fallback (declared @place_group rows honored, mirror pairs kept)"
        ),
    )


# === @equal_size consume (shared contract with the seqpair engine) =============


def _equal_size_targets(
    equal_size_groups: Tuple,
    devs: Dict[str, object],
    passives: Optional[Dict[str, Dict[str, object]]],
    tech,
) -> Dict[str, Tuple[int, int]]:
    """Group-wise MAX footprint ``{label: (w, h)}`` for every ``@equal_size`` member.

    Identical rule to the seqpair engine's ``_equal_size_footprints`` so BOTH engines
    AGREE on the size_class invariant: each group's members are forced to the group-wise
    MAXIMUM width AND MAXIMUM height (the smaller members GROW to match, none ever
    shrinks below its natural geometry). A MOS member is measured from its built
    ``DeviceGeom`` extent; a passive member from the same ``_build_passive_geom`` the
    placer uses downstream. Returns only members whose target is LARGER than natural in
    some axis. Generic + circuit-agnostic; an empty ``equal_size_groups`` returns ``{}``
    (a no-op -> gf180 byte-parity for cells that declare no ``@equal_size``).
    """
    if not equal_size_groups:
        return {}
    passives = passives or {}

    def _natural(lbl: str) -> Optional[Tuple[int, int]]:
        d = devs.get(lbl)
        if d is not None:
            e = d.extent()
            return (e[2] - e[0], e[3] - e[1])
        rec = passives.get(lbl)
        if rec is not None:
            try:
                g = _build_passive_geom(rec, tech, str(rec.get("orient", "R0")))
            except Exception:  # noqa: BLE001 - unbuildable passive -> not sizeable
                return None
            e = g.extent()
            return (e[2] - e[0], e[3] - e[1])
        return None

    targets: Dict[str, Tuple[int, int]] = {}
    for group in equal_size_groups:
        members = [m for m in group if _natural(m) is not None]
        if len(members) < 2:
            continue  # a singleton (or all-unbuildable) group pins nothing
        nat = {m: _natural(m) for m in members}
        max_w = max(w for (w, _h) in nat.values())  # type: ignore[index]
        max_h = max(h for (_w, h) in nat.values())  # type: ignore[index]
        for m in members:
            targets[m] = (max_w, max_h)
    return targets


def _pad_bbox_to_footprint(bbox: Bbox, target: Tuple[int, int]) -> Bbox:
    """Grow ``bbox`` SYMMETRICALLY about its own centroid so its width / height each
    reach (at least) ``target`` -- never shrinking either axis. The device polygons are
    untouched; only the reported (reserved) bbox enlarges, centred on the device, so a
    mirror pair's two members stay equidistant about the axis. Same rule the seqpair
    engine applies, so the two engines reserve the IDENTICAL equal footprint."""
    x0, y0, x1, y1 = bbox
    tw, th = target
    w, h = x1 - x0, y1 - y0
    if tw > w:
        grow = tw - w
        x0 -= grow // 2
        x1 += grow - grow // 2
    if th > h:
        grow = th - h
        y0 -= grow // 2
        y1 += grow - grow // 2
    return (x0, y0, x1, y1)


def _legalise_x_overlaps(
    placed: Dict[str, "PlacedDevice"], grown: set
) -> Dict[str, "PlacedDevice"]:
    """Push GROWN device bboxes apart along x so no two overlap; return the new map.

    Only runs when ``@equal_size`` enlarged some footprints (``grown`` non-empty).
    Devices are swept left-to-right by current left edge; a device that still overlaps
    an already-placed device it shares y with is shifted RIGHT just enough to clear (a
    standard 1-D legalisation that only moves right, so it terminates). With ``grown``
    empty this is a no-op (byte-parity for every cell that declares no ``@equal_size``).
    The bbox shift only updates the REPORTED footprint (the equal-size reservation);
    the drawn polygons are not moved -- the bbox is what the invariant harness measures.
    """
    if not grown:
        return placed
    order_lbls = sorted(
        placed, key=lambda l: (placed[l].bbox[0], placed[l].bbox[1], l)
    )
    out = dict(placed)
    for i, lbl in enumerate(order_lbls):
        x0, y0, x1, y1 = out[lbl].bbox
        shift = 0
        for j in range(i):
            o = out[order_lbls[j]].bbox
            if not (y0 < o[3] and o[1] < y1):
                continue  # no y-overlap -> never collide on the x sweep
            if (x0 + shift) < o[2] and (x1 + shift) > o[0]:
                shift = max(shift, o[2] - x0)
        if shift:
            pd = out[lbl]
            out[lbl] = PlacedDevice(
                label=pd.label,
                kind=pd.kind,
                bbox=(x0 + shift, y0, x1 + shift, y1),
                terminals=pd.terminals,
                polys=pd.polys,
                term_layers=pd.term_layers,
                geometry_owner=pd.geometry_owner,
                terminal_accesses=pd.terminal_accesses,
                materialized_passive_access=pd.materialized_passive_access,
                bjt_unit_occurrences=pd.bjt_unit_occurrences,
                orientation=pd.orientation,
                fixed_pcell=pd.fixed_pcell,
                physical_occurrences=pd.physical_occurrences,
                well_tap_isolated=pd.well_tap_isolated,
            )
    return out


def _apply_equal_size(
    placed: Dict[str, "PlacedDevice"],
    constraints: SolverConstraints,
    devs: Dict[str, object],
    passives: Optional[Dict[str, Dict[str, object]]],
    tech,
    full_extent: Bbox,
) -> Tuple[Dict[str, "PlacedDevice"], Bbox]:
    """Realize the ``@equal_size`` contract on a finished ``placed`` map.

    Forces every declared group's members to the group-wise MAX footprint (same w AND
    h, grow-only, symmetric about each device's own centroid -- the IDENTICAL rule the
    seqpair engine applies, so both engines reserve the same equal footprint), then
    legalises any x-overlap the growth introduced and widens the returned full extent.

    Read off ``constraints.equal_size_groups`` (the agreed interface). Defaults to a
    no-op when empty, so a cell that declares no ``@equal_size`` is byte-unchanged
    (gf180 parity). Returns the (possibly updated) ``placed`` map + full extent.
    """
    equal_size_groups = tuple(getattr(constraints, "equal_size_groups", ()) or ())
    if not equal_size_groups:
        return placed, full_extent
    targets = _equal_size_targets(equal_size_groups, devs, passives, tech)
    if not targets:
        return placed, full_extent
    grown: set = set()
    for lbl, target in targets.items():
        pd = placed.get(lbl)
        if pd is None:
            continue
        new_bbox = _pad_bbox_to_footprint(pd.bbox, target)
        if new_bbox != pd.bbox:
            grown.add(lbl)
        placed[lbl] = PlacedDevice(
            label=pd.label,
            kind=pd.kind,
            bbox=new_bbox,
            terminals=pd.terminals,
            polys=pd.polys,
            term_layers=pd.term_layers,
            geometry_owner=pd.geometry_owner,
            terminal_accesses=pd.terminal_accesses,
            materialized_passive_access=pd.materialized_passive_access,
            bjt_unit_occurrences=pd.bjt_unit_occurrences,
            orientation=pd.orientation,
            fixed_pcell=pd.fixed_pcell,
            physical_occurrences=pd.physical_occurrences,
            well_tap_isolated=pd.well_tap_isolated,
        )
    placed = _legalise_x_overlaps(placed, grown)
    if grown and placed:
        full_extent = (
            min(full_extent[0], min(p.bbox[0] for p in placed.values())),
            min(full_extent[1], min(p.bbox[1] for p in placed.values())),
            max(full_extent[2], max(p.bbox[2] for p in placed.values())),
            max(full_extent[3], max(p.bbox[3] for p in placed.values())),
        )
    return placed, full_extent


# === @fix consume (ABSOLUTE pins -- shared contract with the seqpair engine) ====

# Microns -> manufacturing-grid dbu (nm). Placement coordinates are in nm (1 dbu = 1 nm
# for gf180 / sky130), so a @fix pin authored in microns scales by 1000. Kept in sync
# with the seqpair engine's ``_PIN_UM_TO_NM`` so BOTH engines map a pin to the SAME nm
# target (the anchor property's strict-consistency requires identical pin resolution).
_PIN_UM_TO_NM = 1000.0


def pin_targets_nm(
    pinned_positions: Optional[Dict[str, Tuple[float, float]]],
    labels,
) -> Dict[str, Tuple[int, int]]:
    """Resolve each ``@fix`` pin to an integer ``(x_nm, y_nm)`` centroid target.

    ``pinned_positions`` maps a device LABEL -> its declared centroid ``(x_um, y_um)``
    in microns; this scales to the nm grid and keeps only labels present in ``labels``
    (the built / placed set). Empty / None -> ``{}`` (the no-op guard, byte-parity).
    Shared by both engines so a pin resolves to the IDENTICAL nm target in each.
    """
    if not pinned_positions:
        return {}
    present = set(str(l) for l in labels)
    out: Dict[str, Tuple[int, int]] = {}
    for lbl, (x_um, y_um) in pinned_positions.items():
        if str(lbl) in present:
            out[str(lbl)] = (
                int(round(float(x_um) * _PIN_UM_TO_NM)),
                int(round(float(y_um) * _PIN_UM_TO_NM)),
            )
    return out


def _apply_pins(
    placed: Dict[str, "PlacedDevice"],
    pin_targets: Dict[str, Tuple[int, int]],
) -> Dict[str, "PlacedDevice"]:
    """Translate every pinned placed device so its CENTROID lands exactly on its pin.

    For each label in ``pin_targets`` present in ``placed``: compute the integer delta
    from the device's current bbox-centroid to the pin and TRANSLATE its bbox, polys, and
    terminals by that delta (the device's built geometry / size / orientation is preserved
    -- only its translation changes). Returns a NEW ``placed`` map (the input is not
    mutated). Empty ``pin_targets`` -> the input map is returned unchanged (byte-parity).

    This is the absolute-placement override: a pinned device is moved off the engine's
    computed position to its declared coordinate, OVERRIDING the engine. When EVERY device
    is pinned the placement is fully determined by the pins, so the engine has zero
    remaining freedom -> milp and seqpair realise the SAME placement (the anchor property).
    """
    if not pin_targets:
        return placed
    out: Dict[str, "PlacedDevice"] = dict(placed)
    for lbl, (px, py) in pin_targets.items():
        pd = out.get(lbl)
        if pd is None:
            continue
        x0, y0, x1, y1 = pd.bbox
        cur_cx = (x0 + x1) // 2
        cur_cy = (y0 + y1) // 2
        dx = px - cur_cx
        dy = py - cur_cy
        if dx == 0 and dy == 0:
            continue
        new_bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        new_polys = _translate(list(pd.polys or []), dx, dy)
        new_terms = {
            k: (cx + dx, cy + dy) for k, (cx, cy) in (pd.terminals or {}).items()
        }
        out[lbl] = PlacedDevice(
            label=pd.label,
            kind=pd.kind,
            bbox=new_bbox,
            terminals=new_terms,
            polys=new_polys,
            term_layers=pd.term_layers,
            geometry_owner=pd.geometry_owner,
            terminal_accesses={
                role: tuple((x + dx, y + dy) for x, y in accesses)
                for role, accesses in pd.terminal_accesses.items()
            },
            materialized_passive_access=pd.materialized_passive_access,
            bjt_unit_occurrences=pd.bjt_unit_occurrences,
            orientation=pd.orientation,
            fixed_pcell=pd.fixed_pcell,
            physical_occurrences=pd.physical_occurrences,
            well_tap_isolated=pd.well_tap_isolated,
        )
    return out


# === the generic placer ========================================================


def _resolve_cell_mim_config(
    cell_def=None,
    intent=None,
    tech=None,
    *,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """Return ``(mim_option, cap_option, metal_level, pex_backend)`` for GF180 or ``None``.

    Default (Option-A) returns ``None`` unless Option-B, ``pex_backend='magic'``,
    or an explicit/declared option is supplied so existing Option-A builds stay
    byte-for-byte unchanged.
    """
    pdk_id = str(getattr(tech, "pdk_id", "") or getattr(cell_def, "pdk", "") or "").strip().lower()
    if pdk_id and pdk_id != "gf180mcu":
        if mim_option is not None and str(mim_option).strip().upper() not in ("A", "OPT_A", "OPTION_A"):
            raise ValueError(f"{pdk_id}: mim_option={mim_option!r} is only defined for gf180mcu")
        return None
    eff_pex = (
        pex_backend
        or getattr(intent, "pex_backend", None)
        or getattr(cell_def, "pex_backend", None)
        or getattr(getattr(cell_def, "__dict__", None) or {}, "get", lambda *_: None)("pex_backend")
    )
    eff_mim = (
        mim_option
        or getattr(intent, "mim_option", None)
        or getattr(intent, "cap_option", None)
        or getattr(cell_def, "mim_option", None)
        or getattr(cell_def, "cap_option", None)
        or getattr(getattr(cell_def, "__dict__", None) or {}, "get", lambda *_: None)("_mim_option")
    )
    eff_cap = cap_option or getattr(intent, "mim_cap", None) or getattr(cell_def, "mim_cap", None)
    eff_lvl = (
        metal_level
        or getattr(intent, "metal_level", None)
        or getattr(cell_def, "metal_level", None)
    )
    if eff_mim is None and eff_pex is None and eff_cap is None and eff_lvl is None:
        return None
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    opt, cap_opt, m_lvl = resolve_gf180_mim_config(
        tech,
        mim_option=eff_mim,
        cap_option=eff_cap,
        metal_level=eff_lvl,
        pex_backend=eff_pex,
    )
    return opt, cap_opt, m_lvl, (str(eff_pex) if eff_pex is not None else None)


def place_generic(
    cell_def: CellDef, intent: Optional[LayoutIntent] = None, pdk: object = None, *,
    source_cir=None, grid_nm: int = 5, time_limit_s: float = 30.0, rel_tol: float = 0.05,
    tech=None, dense_contacts: bool = False,
    device_geometry_overrides: Optional[Dict[str, object]] = None,
    passive_geometry_overrides: Optional[Dict[str, object]] = None,
    row_geometry_overrides: Optional[Dict[str, object]] = None,
    row_source_projection: object = None, nonmos_assemblies: tuple = (),
    native_geometry: object = None,
    native_composite_row: object = None,
    mim_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    cap_option: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Placement:
    from clay.layout.annotations import NativeBodyTap
    if (intent is not None
            and any(row.style == 'native_diffusion' for row in intent.arrays)
            and native_composite_row is None):
        raise ValueError('native diffusion requires its source-bound composite row producer')
    body_taps = {
        target for target, annotations in
        (getattr(cell_def, '__dict__', {}).get('_layout_annotations', {}) or {}).items()
        if any(isinstance(annotation, NativeBodyTap) for annotation in annotations)
    }
    if body_taps:
        devices = {device.instance_name: device for device in cell_def.devices}
        if (any(target not in devices or devices[target].device_class != 'native'
                for target in body_taps) or native_geometry is None):
            raise ValueError('@native_body_tap requires native device geometry')
    if any(getattr(device, "device_class", None) == "native"
           for device in (getattr(cell_def, "devices", ()) or ())):
        from clay.layout.native_sky130 import NativeGeometryBundle
        if type(native_geometry) is not NativeGeometryBundle:
            raise ValueError("native device declarations require a supported native physical backend; generic placement cannot omit or rebuild them")
        native_geometry.require_current(cell_def)
        if any((device_geometry_overrides, passive_geometry_overrides,
                row_geometry_overrides, row_source_projection, nonmos_assemblies)):
            raise ValueError("native geometry cannot be replaced by portable geometry overrides")
        if native_composite_row is not None:
            from clay.layout.native_composite_row import NativeCompositeRow
            if type(native_composite_row) is not NativeCompositeRow:
                raise ValueError("native composite row requires its source-bound producer")
            native_composite_row.require_current(cell=cell_def, bundle=native_geometry)
            row_source_projection = native_composite_row.projection
            row_geometry_overrides = {row.owner: row.geometry for row in
                                      native_composite_row.materialization.realizations}
    elif native_geometry is not None:
        raise ValueError("native geometry requires a complete native source cell")
    elif native_composite_row is not None:
        raise ValueError("native composite row requires a complete native source cell")
    if (any(decl.kind in {"imported_layout", "route_tail"}
            for decl in (getattr(cell_def, "layout_intents", ()) or ()))
            or (intent is not None and (intent.imported_layout is not None or intent.route_tails))):
        raise ValueError("imported-layout declarations require the imported routing backend; ordinary placement cannot consume them")
    _tech = _resolve_tech(tech if tech is not None else pdk)
    with row_replay_invocation(source_cell=cell_def, tech=_tech):
        return _place_generic_impl(cell_def, intent, pdk, source_cir=source_cir,
            grid_nm=grid_nm, time_limit_s=time_limit_s, rel_tol=rel_tol, tech=_tech,
            dense_contacts=dense_contacts, device_geometry_overrides=device_geometry_overrides,
            passive_geometry_overrides=passive_geometry_overrides, row_geometry_overrides=row_geometry_overrides,
            row_source_projection=row_source_projection, nonmos_assemblies=nonmos_assemblies,
            native_geometry=native_geometry, native_composite_row=native_composite_row,
            mim_option=mim_option, metal_level=metal_level, cap_option=cap_option, pex_backend=pex_backend)


def _place_generic_impl(
    cell_def: CellDef,
    intent: Optional[LayoutIntent] = None,
    pdk: object = None,
    *,
    source_cir=None,
    grid_nm: int = 5,
    time_limit_s: float = 30.0,
    rel_tol: float = 0.05,
    tech=None,
    dense_contacts: bool = False,
    device_geometry_overrides: Optional[Dict[str, object]] = None,
    passive_geometry_overrides: Optional[Dict[str, object]] = None,
    row_geometry_overrides: Optional[Dict[str, object]] = None,
    row_source_projection: object = None,
    nonmos_assemblies: tuple = (),
    native_geometry: object = None,
    native_composite_row: object = None,
    mim_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    cap_option: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Placement:
    """Place a parsed ``CellDef`` according to its declared layout intent.

    Composition (no solver re-implemented here):

      1. build an :class:`AnnotatedCell` from ``cell_def`` (generic device-kind /
         sizing inference + the parsed annotations);
      2. ``annotations_to_solver_constraints`` -> :class:`SolverConstraints`
         (declared mirror pairs + axis + inferred rails);
      3. ``_build_devices_from_clay`` -> one gf180 :class:`FoldableDevice` per MOS;
      4. ``solve_placement_from_constraints`` -> the MILP placement;
      5. ``score_placement`` -> the REAL R172 symmetry / centroid / rail scores;
      6. extract per-device terminals + cell rail geometry into a
         :class:`Placement`.

    ``intent`` is accepted (the net-level contract) and recorded for provenance;
    device placement is driven by the DEVICE-level ``annotate`` declaration, so a
    net-level diff pair never fabricates a device symmetry pair.

    ``source_cir`` (the bound ``*_gf180.cir`` / source_spice) supplies the REAL
    per-device W / L / m so the placed geometry is the design's actual geometry;
    without it the placer falls back to the uniform default geometry.
    """
    # Device geometry + placer spacing flow through one LayoutTech (gf180mcu by
    # default -- parity + no caller breaks). ``tech`` (P3+) takes precedence; a
    # legacy ``pdk`` (PdkRules / id string) is resolved to its tech.
    _tech = _resolve_tech(tech if tech is not None else pdk)

    # SPATIAL OVERLAY auto-derive: when the caller passes NO intent but the parsed
    # cell DECLARES a @place_group floorplan in its layout_intent block, lower that
    # declared intent so the floorplan rows reach the solver (the parsed block IS
    # the source of truth -- see layout_intent_from_cell_def). This is gated on the
    # cell actually declaring a @place_group so every existing intent=None caller
    # whose cell declares no floorplan is byte-unchanged (no diff_pairs side change,
    # no new constraints): the gate is a pure structural predicate, no circuit branch.
    if intent is None and (
        _cell_declares_place_group(cell_def)
        or _cell_declares_guard_ring(cell_def)
        or _cell_declares_row_structure(cell_def)
    ):
        from clay.layout.intent import layout_intent_from_cell_def

        intent = layout_intent_from_cell_def(cell_def)

    # RG5 structural authority.  Compile and invoke each rigid row builder once,
    # before any placement solve.  Later re-place stages consume the stored result;
    # they are not permitted to resolve the declarations through a router helper.
    row_program, row_materialization = row_phase_call('generic_placement.row_materialization', _compile_row_master,
        cell_def,
        intent,
        _tech,
        source_cir,
        row_geometry_overrides=row_geometry_overrides,
        row_source_projection=row_source_projection,
    )
    bind_current_row_materialization(row_materialization, program=row_program, tech=_tech)
    annotated, used_default = annotated_cell_from_cell_def(
        cell_def, source_cir, native_geometry=native_geometry)
    from clay.layout.annotate_to_solver import (
        _orient_map_from_cell,
        resolve_discrete_mos_array_specs,
    )

    discrete_mos_array_specs = resolve_discrete_mos_array_specs(
        cell_def, intent, annotated
    ) if intent is not None else {}
    declared_orientation_targets = frozenset(_orient_map_from_cell(annotated))
    if native_geometry is not None:
        if _tech.pdk_id != "sky130":
            raise ValueError("native SKY130 geometry requires the SKY130 technology")
        if discrete_mos_array_specs or (row_program is not None and native_composite_row is None):
            raise ValueError("native geometry composition/orientation needs a native rigid transform contract")
    physical_macro_graph, row_macro_contracts = row_phase_call('generic_placement.row_contracts', _compile_physical_macro_structure,
        cell_def,
        intent,
        _tech,
        source_cir,
        annotated,
        row_program,
        row_materialization,
    )
    physical_macro_scope_token = (
        _mint_physical_macro_scope(physical_macro_graph.sha256)
        if physical_macro_graph is not None
        else None
    )
    constraints = _constraints_from_intent_and_annotations(annotated, intent, cell_def, tech=_tech)
    # ``_build_devices_from_clay`` FUSES each declared ``@interdigitated(A, B)``
    # matched pair into ONE common-centroid interleaved finger array (keyed under
    # member A; member B dropped). The cluster is centroid-symmetric, so the
    # constraint filter below drops the now-absent member B from the symmetry
    # pairs / floorplan automatically (both engines filter to built devices).
    # ``fold_strips=True``: place_generic has NO placement-level wide-packing of free
    # strip devices (unlike the seqpair placer), so a tall many-finger MOS (the LDO
    # pass FET) folds at the CELL level here, via the mature serpentine fold. The
    # seqpair / MILP frozen oracles call _build_devices_from_clay with the default
    # (fold_strips=False) and stay byte-identical.
    devs = dict(native_geometry.geometries) if native_geometry is not None else _build_devices_from_clay(annotated, _tech, fold_strips=True,
                                    dense_contacts=dense_contacts,
                                    discrete_array_specs=discrete_mos_array_specs,
                                    geometry_overrides=device_geometry_overrides,
                                    native_bulk_access=bool(
                                        constraints.floorplan_rows and
                                        (getattr(intent, "channel_routes", ())
                                         or getattr(intent, "route_groups", ()))))
    if native_geometry is not None and declared_orientation_targets:
        from clay.layout.native_sky130 import orient_native_device
        orientation_map = _orient_map_from_cell(annotated)
        if set(orientation_map) - set(devs):
            raise ValueError("native orientation names an unknown device")
        for label, orientation in orientation_map.items():
            devs[label] = orient_native_device(devs[label], orientation, _tech)
    interleaved = [
        d.__dict__["cluster"]
        for d in devs.values()
        if isinstance(getattr(d, "__dict__", None), dict) and "cluster" in d.__dict__
    ]

    # Collect every NON-MOS cell. Passives and adapter-bound fixed-cell BJTs share
    # one placement currency (real polygons + routable terminal access), but their
    # builders remain disjoint and fail closed.
    nonmos = {} if native_geometry is not None else collect_nonmos_placement_inputs(cell_def, source_cir, intent, _tech,
        passive_geometry_overrides, nonmos_assemblies, physical_macro_graph)
    mim_cfg = _resolve_cell_mim_config(
        cell_def,
        intent,
        _tech,
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    if mim_cfg is not None and nonmos:
        opt, cap_opt, m_lvl, eff_pex = mim_cfg
        for _lbl, _rec in nonmos.items():
            if isinstance(_rec, dict) and str(_rec.get("kind", "")) == "cap":
                _rec.setdefault("mim_option", opt)
                _rec.setdefault("cap_option", cap_opt)
                _rec.setdefault("metal_level", m_lvl)
                if eff_pex is not None:
                    _rec.setdefault("pex_backend", eff_pex)
    physical_macro_contracts = None
    physical_macro_compile_inputs = None
    rigid_nonmos_macro_plans = None
    rigid_nonmos_macro_inputs = None
    rigid_nonmos_macro_interfaces = None
    if physical_macro_graph is not None:
        from clay.layout.physical_macro_contracts import (
            compile_physical_macro_contract_snapshot,
        )

        (
            physical_macro_contracts,
            physical_macro_compile_inputs,
        ) = compile_physical_macro_contract_snapshot(
            physical_macro_graph,
            geometries=devs,
            device_pins=getattr(annotated, "_device_pins", {}) or {},
            tech=_tech,
        )
        (
            compiled_nonmos_plans,
            compiled_nonmos_inputs,
            compiled_nonmos_interfaces,
        ) = compile_rigid_nonmos_macro_plans(
            physical_macro_graph, nonmos, _tech, intent, native_assemblies=nonmos_assemblies
        )
        if compiled_nonmos_plans.plans:
            rigid_nonmos_macro_plans = compiled_nonmos_plans
            rigid_nonmos_macro_inputs = compiled_nonmos_inputs
            rigid_nonmos_macro_interfaces = compiled_nonmos_interfaces
        _seal_physical_macro_scope(
            physical_macro_scope_token,
            physical_macro_graph.sha256,
            _physical_macro_compiler_binding_values(
                row_macro_contracts=row_macro_contracts,
                physical_macro_contracts=physical_macro_contracts,
                physical_macro_compile_inputs=physical_macro_compile_inputs,
                rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
                rigid_nonmos_macro_inputs=rigid_nonmos_macro_inputs,
                rigid_nonmos_macro_interfaces=rigid_nonmos_macro_interfaces,
            ),
            row_program=row_program,
            row_materialization=row_materialization,
        )

    if not devs:
        if nonmos:
            # Pure non-MOS cell: place the real fixed/passive cells on one row.
            base = replace(
                _passive_only_placement(
                    cell_def,
                    nonmos,
                    _tech,
                    constraints,
                    used_default,
                    rigid_macro_plans=rigid_nonmos_macro_plans,
                ),
                physical_macro_graph=physical_macro_graph,
                physical_macro_scope_token=physical_macro_scope_token,
                row_macro_contracts=row_macro_contracts,
                physical_macro_contracts=physical_macro_contracts,
                physical_macro_compile_inputs=physical_macro_compile_inputs,
                rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
                rigid_nonmos_macro_inputs=rigid_nonmos_macro_inputs,
                rigid_nonmos_macro_interfaces=rigid_nonmos_macro_interfaces,
            )
            result = attach_physical_macro_projection(
                attach_row_master(base, row_program, row_materialization)
            )
            verify_rigid_nonmos_macro_projection(result)
            return result
        # No MOS and no passive: return an honest empty placement rather than crash.
        return attach_physical_macro_projection(attach_row_master(Placement(
            devices={},
            axis_x=0,
            geom={},
            polys=[],
            quality=None,
            solver_status="no_mos_devices",
            solver_success=False,
            solve_seconds=0.0,
            constraints=constraints,
            used_default_geometry=used_default,
            physical_macro_graph=physical_macro_graph,
            physical_macro_scope_token=physical_macro_scope_token,
            row_macro_contracts=row_macro_contracts,
            physical_macro_contracts=physical_macro_contracts,
            physical_macro_compile_inputs=physical_macro_compile_inputs,
            rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
            rigid_nonmos_macro_inputs=rigid_nonmos_macro_inputs,
            rigid_nonmos_macro_interfaces=rigid_nonmos_macro_interfaces,
            note=(
                f"cell {cell_def.local_name!r}: no nmos/pmos devices inferred from "
                f"the parsed CellDef (topology has "
                f"{len(getattr(cell_def, 'functional_roles', []) or [])} functional "
                f"groups); nothing to place."
            ),
        ), row_program, row_materialization))

    # Capture the DECLARED passive floorplan (row/col) BEFORE the MOS-only restriction
    # below drops it (a passive is not a built MOS, so ``_restrict_constraints_to_devices``
    # filters its floorplan entry out). This lets ``_place_resistors`` honour the declared
    # 2-D floorplan -- a passive declaring its own COLUMN beyond the MOS grid banks into a
    # right-hand block; one declaring a reserved MIDDLE row lands between the MOS bands --
    # instead of always dropping to the legacy below-MOS row. Empty unless the cell declares
    # passive columns/rows, so every existing floorplan (MOS-only @place_group) is unchanged.
    _passive_labels = set(nonmos)
    # ``constraints.floorplan_rows`` intentionally contains only groups pinned
    # to an exact ROW because the MOS row solver cannot consume a column without
    # a row.  A passive bank, however, legitimately declares only ``col=N``: its
    # own-column materializer vertically packs that block beside the MOS core.
    # Read column DATA from the source intent itself before the MOS restriction,
    # so the accepted col-only syntax cannot silently disappear at this seam.
    _declared_col_of: Dict[str, int] = {}
    for _pg in tuple(getattr(intent, "place_groups", ()) or ()):
        _col = getattr(_pg, "col", None)
        if _col is None:
            continue
        for _label in tuple(getattr(_pg, "devices", ()) or ()):
            _declared_col_of[str(_label)] = int(_col)
    _passive_col_of = {
        lbl: col
        for lbl, col in _declared_col_of.items()
        if lbl in _passive_labels
    }
    _mos_decl_cols = [
        col
        for lbl, col in _declared_col_of.items()
        if lbl not in _passive_labels
    ]
    _mos_max_col = max(_mos_decl_cols) if _mos_decl_cols else None
    _nonmos_match_pairs = tuple(
        (str(left), str(right))
        for left, right in tuple(constraints.symmetry_pairs or ())
        if left in _passive_labels and right in _passive_labels
    )

    # When MOS devices declare explicit 2-D ``(row, col)`` floorplan columns and
    # the cell also carries dedicated non-MOS rows (e.g. wide segmented resistor
    # or MIM capacitor banks), pass the widest dedicated non-MOS row's physical
    # width as a floor to ``plan_geometry_bounded_subrows`` so the MOS solver
    # never wraps MOS rows into tall narrow multi-subrow stacks beneath a wider
    # passive/BJT bank.  Purely geometry-derived; inactive when no MOS ``col``
    # is declared (preserving byte-exact parity for every col-free baseline).
    _nonmos_floor_width_nm = 0
    _mos_has_explicit_cols = any(
        rc[1] is not None
        for lbl, rc in constraints.floorplan_rows.items()
        if lbl in devs
    )
    if _mos_has_explicit_cols and nonmos:
        _mos_rows_set = {
            rc[0]
            for lbl, rc in constraints.floorplan_rows.items()
            if lbl in devs
        }
        _nonmos_by_row: Dict[int, List[int]] = {}
        for _lbl, _rec in nonmos.items():
            _rc = constraints.floorplan_rows.get(_lbl)
            if _rc is None or _rc[0] in _mos_rows_set:
                continue
            _col = _declared_col_of.get(_lbl)
            if _col is not None and _mos_max_col is not None and _col > _mos_max_col:
                continue
            _g = _build_passive_geom(_rec, _tech, str(_rec.get("orient", "R0")))
            _ext = _g.extent()
            _nonmos_by_row.setdefault(int(_rc[0]), []).append(int(_ext[2] - _ext[0]))
        for _widths in _nonmos_by_row.values():
            if _widths:
                _row_span = sum(_widths) + _RES_ROW_GAP_NM * max(0, len(_widths) - 1)
                if _row_span > _nonmos_floor_width_nm:
                    _nonmos_floor_width_nm = _row_span

    # Guard: drop symmetry-pair members the placer has no MOS geometry for (a
    # bandgap matches resistors / PNPs that are not foldable MOS). The row-based
    # solver assumes every pair member has built geometry, so an unbuilt member
    # would crash it -- restrict the constraints to the built MOS set.
    constraints, dropped = _restrict_constraints_to_devices(constraints, devs)
    constraints = _demote_ambiguous_matching_constraints(constraints, devs)

    # RING ROOM (declared @guard_ring only): the solver reserves the ring's outward
    # extension + neighbour clearance around each ringed cluster member, so the ring
    # route_cell materialises later fits WITHOUT overlapping a neighbour's metal (the
    # cross-net short a post-placement ring insertion otherwise causes). SOLVE-only
    # view -- the drawn geometry / terminals are untouched. A cell that declares no
    # ring gets ``devs`` back unchanged (byte-identical placement).
    solve_devs = _ring_room_padded_devs(devs, intent, _tech)
    realized_mos_pair_atoms = _row_matching_pair_atoms(constraints, solve_devs)
    _ring_group_of: Optional[Dict[str, int]] = None
    if _mos_has_explicit_cols and intent is not None:
        _declared_rings = tuple(getattr(intent, "guard_rings", ()) or ())
        if _declared_rings:
            _rg_map: Dict[str, int] = {}
            for _ri, _gr in enumerate(_declared_rings):
                for _m in tuple(getattr(_gr, "members", ()) or ()):
                    _rg_map.setdefault(str(_m), _ri)
            if _rg_map:
                _ring_group_of = _rg_map

    sol: PlacementSolution = row_phase_call('generic_placement.initial_solve', solve_placement_from_constraints,
        constraints,
        solve_devs,
        tech=_tech,
        grid_nm=grid_nm,
        time_limit_s=time_limit_s,
        min_target_width_nm=_nonmos_floor_width_nm,
        ring_group_of=_ring_group_of,
    )

    # REALIZATION-GAP GUARD: the exact single-axis MILP fixes every UNPAIRED row member
    # to the one global axis x, so a declared floorplan with multiple unpaired devices in
    # one row is INFEASIBLE -- the MILP returns success=False and would drop ALL built MOS
    # (only passives, on a separate path, survive). When the MILP fails BUT devices were
    # built, realize the declared floorplan generically (overlap-free, mirror pairs kept)
    # so the placement carries every declared MOS. A cell the exact MILP CAN realize keeps
    # the exact solution byte-for-byte (gf180 parity), since this only triggers on failure.
    if not sol.success and devs:
        if constraints.floorplan_rows:
            from clay.layout.place.hierarchical_macro_placement import (
                NoFeasibleHierarchicalPlacement,
            )

            raise NoFeasibleHierarchicalPlacement(sol.status)
        sol = _floorplan_pack_solution(
            constraints, solve_devs, _tech, base_status=sol.status
        )

    quality: PlacementQuality = score_placement(sol, devs, rel_tol=rel_tol)

    # SHARED-NWELL bulk-pin policy (line-end PDK only): a PMOS whose tap rises to its
    # SOURCE rail ties its bulk to its source net. When source == bulk (the self-bulk
    # PMOS -- tail / output, source=bulk=vdd) the bulk is ALREADY carried by the
    # source terminal, so a SEPARATE 'b' riser is redundant: the router would land a
    # vdd via pad next to the FEOL tap, wedged against a different-net rail (a sub-rule
    # same-net notch that cannot be bridged) -- so the 'b' pin is suppressed and the
    # bulk reaches vdd through tap -> source rail -> (router) vdd. gf180 (no line-end
    # rule) is unaffected: it keeps its 'b' pin and stays byte-identical. The pins come
    # from the parsed cell (label -> {'s','b'}); unknown pins keep the 'b' pin.
    pins_map = getattr(annotated, "_device_pins", {}) or {}
    line_end_pdk = False
    try:
        line_end_pdk = (
            bool(_tech.line_end_ext("metal1"))
            and getattr(_tech, "pdk_id", "").lower() != "gf180mcu"
        )
    except Exception:  # noqa: BLE001 - PDK silent on line-end -> keep 'b' pin
        line_end_pdk = False
    self_bulk_pmos: set = set()
    if line_end_pdk and native_geometry is None:
        for lbl, p in pins_map.items():
            s, b = p.get("s", ""), p.get("b", "")
            if (s and b and s == b
                    and getattr(devs.get(lbl), "kind", None) == "pmos"):
                self_bulk_pmos.add(lbl)

    placed: Dict[str, PlacedDevice] = {}
    all_polys: list = []
    # The canonical resolver already maps both direct targets and functional
    # group aliases to one logical MOS owner. Preserve every resolved unit grid.
    discrete_mos = set(discrete_mos_array_specs)
    fixed_master_mos = {
        name
        for name, geometry in devs.items()
        if type(getattr(geometry, "fixed_pcell", None)) is dict
        and type(geometry.fixed_pcell.get("unit_occurrences")) is int
        and geometry.fixed_pcell["unit_occurrences"] > 0
    }
    exact_placed_mos = (
        discrete_mos
        | {name for name in declared_orientation_targets
           if getattr(devs.get(name), "kind", None) in ("nmos", "pmos")}
        | fixed_master_mos
    )
    if sol.success:
        for lbl, pd in sol.placed.items():
            base = devs[lbl]
            # Matching owns the symmetric *location*.  An explicit @orient owns
            # the artwork transform, including an explicit R0, and therefore
            # suppresses the solver's historical inferred reflection.
            materialize_mirror = (
                pd.mirror and lbl not in declared_orientation_targets
            )
            terms = _placed_terminals(
                base, pd.dx, pd.dy, materialize_mirror, sol.axis_x
            )
            terminal_accesses = _placed_terminal_accesses(
                base, pd.dx, pd.dy, materialize_mirror
            )
            if native_geometry is not None:
                terms = {role: points[0] for role, points in terminal_accesses.items()}
            if lbl in self_bulk_pmos:
                terms.pop("b", None)
                terminal_accesses.pop("b", None)
            terminal_accesses = {
                role: points
                for role, points in terminal_accesses.items()
                if role in terms
            }
            polys = _place_polys(
                base, pd.dx, pd.dy, materialize_mirror, sol.axis_x
            )
            all_polys += polys
            # the bbox must come from the PLACED (possibly mirrored) polygons so it
            # is consistent with the extracted terminals -- the solver's
            # ``pd.extent`` is the pre-mirror translated extent and would be on the
            # wrong side of the axis for a reflected right-hand member.
            #
            # A SHARED-NWELL PMOS that suppressed its well tap reserves the
            # TAP-INCLUSIVE extent (``base.placement_extent``) so the placed bbox --
            # which feeds the cell channel / rail geometry the router builds on -- is
            # byte-identical to the tap version (placement-neutral routing), even
            # though the drawn polys are smaller. Every other device keeps the drawn
            # extent.
            bbox = _placed_bbox(base, pd.dx, pd.dy, materialize_mirror) or _polys_bbox(polys) \
                or pd.extent
            physical_occurrences = (
                _placed_discrete_mos_occurrences(
                    base,
                    pd.dx,
                    pd.dy,
                    materialize_mirror,
                    bbox,
                    _tech,
                )
                if lbl in exact_placed_mos
                else ()
            )
            resolved_master_occurrences = _placed_resolved_master_occurrences(
                base,
                materialize_mirror,
            )
            placed[lbl] = PlacedDevice(
                label=lbl,
                kind=pd.kind,
                bbox=bbox,
                terminals=terms,
                polys=polys,
                terminal_accesses=terminal_accesses,
                orientation=str(getattr(base, "orient", "") or ""),
                term_layers={
                    role: layer
                    for role, layer in (getattr(base, "terminal_layers", {}) or {}).items()
                    if role in terms
                },
                fixed_pcell=dict(getattr(base, "fixed_pcell", {}) or {}),
                physical_occurrences=physical_occurrences,
                resolved_master_occurrences=resolved_master_occurrences,
                well_tap_isolated=bool(
                    getattr(base, "well_tap_isolated", False)
                ),
            )

    # SHARED-NWELL: bridge the placed PMOS per-device nwell boxes into ONE contiguous
    # region so the extractor sees a SINGLE vdd-tied well (the analog PMOS-in-common-
    # well layout). GATED on the PDK declaring a metal line-end rule (the external_pdk /
    # commercial line-end path that emits device FEOL into the generic flow); gf180
    # has no line-end rule, so this is a no-op there and the frozen gf180 placement /
    # emitted geometry stays byte-identical. The straps are appended to the cell's
    # FEOL polys (emitted as device_polys by the flow). When a residual gap cannot be
    # single-axis bridged it is left honestly (the LVS / net gates would surface it).
    if placed and native_geometry is None:
        line_end_pdk = False
        try:
            line_end_pdk = (
                bool(_tech.line_end_ext("metal1"))
                and getattr(_tech, "pdk_id", "").lower() != "gf180mcu"
            )
        except Exception:  # noqa: BLE001 - PDK silent on line-end -> no bridge
            line_end_pdk = False
        if line_end_pdk:
            try:
                nwell_key = _tech.build_layer("nwell")
            except Exception:  # noqa: BLE001
                nwell_key = None
            if nwell_key is not None:
                bridges = _shared_nwell_bridges(all_polys, nwell_key)
                all_polys += bridges

    # Place the cell's passive resistors on a row below the MOS band (a non-MOS
    # path: resistors are not folded MOS). Their placed devices + FEOL polys merge
    # into the placement; the router connects their p/n terminals like any net.
    res_extent = sol.full_extent
    realized_nonmos_pair_atoms: Tuple[Tuple[str, str], ...] = ()
    if nonmos:
        # BANK GRID is opt-in via the DECLARED floorplan: only a cell that declares a
        # @place_group floorplan wraps its overflowing below-MOS passive set into a
        # compact 2-D bank grid. Every cell that declares no floorplan keeps the
        # historical single passive row byte-for-byte (no regression to the frozen
        # OTA / comparator passive layout). The gate is a pure structural predicate.
        bank_grid = _cell_declares_place_group(cell_def)
        # Honour the DECLARED passive floorplan columns: a cap/resistor whose declared
        # column is beyond the MOS grid banks into its own right-hand block (vertically
        # centred on the MOS band) rather than the legacy below-MOS row -- so a pair of
        # large MIM caps sit BESIDE the MOS core (a landscape cell) instead of stacking
        # below it (a tall sliver). ``col_of`` is empty unless the cell declares passive
        # columns, so every existing floorplan stays byte-identical.
        _mos_bboxes = [pd.bbox for pd in placed.values()]
        _mos_xext = (
            (min(b[0] for b in _mos_bboxes), max(b[2] for b in _mos_bboxes))
            if _mos_bboxes else None
        )
        (
            res_placed,
            res_polys,
            res_extent,
            realized_nonmos_pair_atoms,
        ) = _place_resistors(
            nonmos, _tech, sol.full_extent, bank_grid=bank_grid,
            col_of=(_passive_col_of or None),
            mos_max_col=_mos_max_col,
            mos_x_extent=_mos_xext,
            matched_pairs=(_nonmos_match_pairs or None),
            rigid_macro_plans=rigid_nonmos_macro_plans,
        )
        placed.update(res_placed)
        all_polys += res_polys

    # @equal_size (SHARED CONTRACT with the seqpair engine): force every declared group's
    # members to the group-wise MAX footprint, growing the smaller members to match. This
    # PINS the size_class invariant so the MILP path and the seqpair path AGREE on it.
    if native_geometry is not None and getattr(constraints, "equal_size_groups", ()):
        raise ValueError("native geometry cannot be resized by @equal_size")
    placed, res_extent = _apply_equal_size(
        placed, constraints, devs, nonmos, _tech, res_extent
    )
    # Freeze the matched-macro axis after the materializer/equal-size transform
    # but BEFORE independent @fix overrides.  A pin that moves only one member
    # must be measured against the axis the matched atom established; recomputing
    # the axis after that move would manufacture a tautological pass.
    matching_pair_axes = _materialized_matching_axes(
        placed,
        realized_mos_pair_atoms + realized_nonmos_pair_atoms,
    )

    # @fix ABSOLUTE pins (SHARED CONTRACT with the seqpair engine): translate every
    # pinned device so its CENTROID lands exactly on its declared coordinate, OVERRIDING
    # the MILP's computed position. Applied AFTER equal_size so the final footprint is
    # what gets centred on the pin. Empty ``pinned_positions`` -> no-op (gf180 byte-parity).
    pin_targets = pin_targets_nm(
        getattr(constraints, "pinned_positions", {}) or {}, placed.keys()
    )
    if pin_targets:
        placed = _apply_pins(placed, pin_targets)
        all_polys = [p for pd in placed.values() for p in (pd.polys or [])]
        if placed:
            res_extent = (
                min(p.bbox[0] for p in placed.values()),
                min(p.bbox[1] for p in placed.values()),
                max(p.bbox[2] for p in placed.values()),
                max(p.bbox[3] for p in placed.values()),
            )

    dropped_nonmos_matched = tuple(sorted({
        member
        for pair in _canonical_matching_pairs(
            _nonmos_match_pairs, _passive_labels
        )
        for member in pair
        if member not in placed
    }))
    geom = _cell_geom(placed, res_extent, tech=_tech)

    placement = Placement(
        devices=placed,
        axis_x=sol.axis_x,
        geom=geom,
        polys=all_polys,
        quality=quality,
        solver_status=sol.status,
        solver_success=sol.success,
        solve_seconds=sol.solve_seconds,
        constraints=constraints,
        used_default_geometry=used_default,
        matching_pair_axes=matching_pair_axes,
        dropped_nonmos_matched=dropped_nonmos_matched,
        note=(
            sol.note
            + (
                f" | dropped {len(dropped_nonmos_matched)} non-MOS matched member(s) "
                f"{dropped_nonmos_matched}"
                if dropped_nonmos_matched
                else ""
            )
            + (
                " | interdigitated common-centroid clusters: "
                + "; ".join(
                    f"{ci.device_a}+{ci.device_b} "
                    f"[{ci.pattern.upper()} {ci.total_fingers}f: {ci.interleave_string()}]"
                    for ci in interleaved
                )
                if interleaved
                else ""
            )
        ),
        row_program=row_program,
        row_materialization=row_materialization,
        physical_macro_graph=physical_macro_graph,
        physical_macro_scope_token=physical_macro_scope_token,
        row_macro_contracts=row_macro_contracts,
        physical_macro_contracts=physical_macro_contracts,
        physical_macro_compile_inputs=physical_macro_compile_inputs,
        rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
        rigid_nonmos_macro_inputs=rigid_nonmos_macro_inputs,
        rigid_nonmos_macro_interfaces=rigid_nonmos_macro_interfaces,
        native_geometry=native_geometry,
    )
    if sol.ordered_row_program is not None:
        from clay.layout.place.ordered_mirror_rows import (
            OrderedRowError,
            attach_ordered_row_floorplan,
            bind_ordered_row_source,
            make_ordered_row_evidence,
            verify_ordered_row_placement,
        )

        if row_program is not None:
            # An independently stored shared-diffusion macro cannot be split
            # into free mirror halves. Retain its existing exact-once authority.
            raise OrderedRowError("ORDERED_ROW_STORED_MACRO_REQUIRES_ATOMIC_LOWERING")
        placement = replace(placement, ordered_row_evidence=make_ordered_row_evidence(
            sol.ordered_row_program, sol.ordered_axis_coordinates_nm, placed))
        verify_ordered_row_placement(placement)
        placement = attach_ordered_row_floorplan(placement, intent, _tech)
        placement = bind_ordered_row_source(placement, cell_def, source_cir)
    if row_program is None:
        result = attach_physical_macro_projection(placement)
        if result.physical_macro_graph is not None:
            verify_rigid_nonmos_macro_projection(result)
        return result

    # Structural rows are placement atoms, not decorations added after a
    # device-level solve.  Reuse the canonical routability-aware CP pack so all
    # row macros and all free devices participate in one AddNoOverlap2D master;
    # declaration-derived bands, order, guard islands, symmetry, and density
    # objectives remain in the same solve.  A short sub-budget keeps the canonical
    # placement probe bounded.  UNKNOWN/INFEASIBLE is a typed stop, never a return
    # to the overlapping free-member coordinates above.
    from clay.layout.envelope_place import cp_compact_replace_placement
    from clay.layout.synth.placement import atomic_placement_observation

    atomic_limit = max(0.25, min(5.0, float(time_limit_s)))
    projection = getattr(row_materialization, 'source_projection', None)
    with row_replay_phase('generic_placement.atomic_compaction',
            source_sha256=getattr(projection, 'source_sha256', None),
            source_inventory_sha256=getattr(projection, 'source_inventory_sha256', None),
            source_cell=getattr(cell_def, 'local_name', None),
            pdk_id=_tech.pdk_id, requested_atomic_limit=atomic_limit,
            row_entries=len(row_program.entries)):
        with atomic_placement_observation():
            final, atomic_result = cp_compact_replace_placement(
                placement, cell_def, intent, tech_id=_tech.pdk_id, max_time_s=atomic_limit,
            )
    if atomic_result is None or atomic_result.status not in ("OPTIMAL", "FEASIBLE"):
        status = "UNKNOWN" if atomic_result is None else str(
            getattr(atomic_result, "backend_status", None)
            or atomic_result.status
        )
        raise RowFinalPlacementResidualError((f"ROW_ATOMIC_PLACEMENT_{status}",))
    if final.row_residuals or final.row_final_certificate is None:
        raise RowFinalPlacementResidualError(
            tuple(final.row_residuals) or ("ROW_FINAL_CERTIFICATE_MISSING",)
        )
    row_phase_call('generic_placement.final_verification', verify_stored_row_projection, final)
    result = attach_physical_macro_projection(final)
    if result.physical_macro_graph is not None:
        verify_rigid_nonmos_macro_projection(result)
    return result


place_generic.__doc__ = _place_generic_impl.__doc__

# === passive (resistor) collection + placement ==============================
#
# Resistors are 2-terminal NON-MOS devices: they get their own placement path (no
# MILP folding, no symmetry banding) -- a row of resistor cells, each built by the
# PDK-tech-driven passive builder. The router then connects each resistor's p/n
# terminal net like any 2-terminal net.

# spacing between adjacent resistor cells in the passive row (a comfortable
# clearance above the LRES.2/LRES.4 resistor-to-resistor + resistor-to-poly space;
# the resistors are routed apart so a generous gap keeps the row DRC-clean).
_RES_ROW_GAP_NM = 2000
_RES_BAND_GAP_NM = 3000   # vertical gap from the MOS band down to the resistor row

# A below-MOS passive set of this many or more members whose single-row width
# overflows the MOS x-extent is wrapped into a compact 2-D bank grid (instead of a
# long thin strip). Below this count -- or when the row already fits -- the
# historical single left-to-right row is reproduced byte-for-byte.
_BANK_MIN_MEMBERS = 4


def _canonical_matching_pairs(
    matched_pairs: Sequence[Tuple[str, str]],
    members: Sequence[str] | Set[str],
) -> Tuple[Tuple[str, str], ...]:
    """Return stable, unique unordered pairs wholly contained in ``members``.

    Annotate lowering preserves both directed ``@matching`` declarations.  A
    physical matched macro is nevertheless one unordered atom, so reciprocal
    declarations must not create duplicate placement atoms or duplicate axes.
    """

    have = {str(member) for member in members}
    pairs = {
        tuple(sorted((str(left), str(right))))
        for left, right in matched_pairs
        if str(left) != str(right)
        and str(left) in have
        and str(right) in have
    }
    return tuple(sorted(pairs))


def _matching_atoms(
    labels: Sequence[str],
    matched_pairs: Sequence[Tuple[str, str]],
) -> Tuple[Tuple[str, ...], ...]:
    """Collapse disjoint matching pairs into stable placement atoms.

    The atom is emitted at the first member's position in the caller's ordering;
    its two members use canonical label order.  An overlapping declaration graph
    cannot be represented as disjoint two-member macros, so only the first stable
    edge becomes an atom; the remaining edge receives no axis certificate and is
    consequently unmeasurable rather than being silently claimed as realized.
    """

    ordered = tuple(str(label) for label in labels)
    pair_of: Dict[str, str] = {}
    for left, right in _canonical_matching_pairs(matched_pairs, ordered):
        if left in pair_of or right in pair_of:
            continue
        pair_of[left] = right
        pair_of[right] = left

    atoms: List[Tuple[str, ...]] = []
    consumed: Set[str] = set()
    for label in ordered:
        if label in consumed:
            continue
        peer = pair_of.get(label)
        if peer is None or peer in consumed:
            atom = (label,)
        else:
            atom = tuple(sorted((label, peer)))
        consumed.update(atom)
        atoms.append(atom)
    return tuple(atoms)


def _materialized_matching_axes(
    placed: Mapping[str, PlacedDevice],
    realized_pair_atoms: Sequence[Tuple[str, str]],
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Certify only explicitly returned local two-member materialization atoms.

    This function deliberately does not reconstruct atoms from the global placed
    label set.  The passive materializer owns the only honest knowledge of which
    pair shared one local bank/column packing operation.
    """

    axes: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for pair in _canonical_matching_pairs(realized_pair_atoms, placed.keys()):
        left, right = pair
        lb = placed[left].bbox
        rb = placed[right].bbox
        left_cx = (float(lb[0]) + float(lb[2])) / 2.0
        right_cx = (float(rb[0]) + float(rb[2])) / 2.0
        axes[pair] = ("x", (left_cx + right_cx) / 2.0)
    return axes


def _disjoint_matching_pairs(
    matched_pairs: Sequence[Tuple[str, str]],
    members: Sequence[str] | Set[str],
) -> Tuple[Tuple[str, str], ...]:
    """Return canonical pairs whose members occur on exactly one matching edge.

    A member shared by two declarations cannot belong to two independent rigid
    two-member atoms.  Withhold every edge in that ambiguous component instead
    of selecting one from declaration order or deriving a tautological axis from
    the final coordinates.
    """

    pairs = _canonical_matching_pairs(matched_pairs, members)
    degree = Counter(member for pair in pairs for member in pair)
    return tuple(
        pair
        for pair in pairs
        if degree[pair[0]] == 1 and degree[pair[1]] == 1
    )


def _demote_ambiguous_matching_constraints(
    constraints: SolverConstraints,
    members: Mapping[str, object] | Sequence[str] | Set[str],
) -> SolverConstraints:
    """Remove ambiguous pair authority while preserving every placement node.

    Pair orientation is retained for unambiguous edges (diff-side lowering may
    have chosen it deliberately).  A two-member common-centroid proxy generated
    from a removed matching edge is removed with that edge; unrelated declared
    common-centroid/place-group constraints remain intact.
    """

    labels = tuple(members)
    original = tuple(constraints.symmetry_pairs or ())
    kept_canonical = set(_disjoint_matching_pairs(original, labels))
    original_canonical = set(_canonical_matching_pairs(original, labels))
    kept_oriented: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for left, right in original:
        canonical = tuple(sorted((str(left), str(right))))
        if canonical not in kept_canonical or canonical in seen:
            continue
        kept_oriented.append((str(left), str(right)))
        seen.add(canonical)
    removed = original_canonical - kept_canonical
    kept_cc = tuple(
        group
        for group in tuple(constraints.common_centroid_groups or ())
        if not (
            len(group) == 2
            and tuple(sorted((str(group[0]), str(group[1])))) in removed
        )
    )
    return replace(
        constraints,
        symmetry_pairs=tuple(kept_oriented),
        common_centroid_groups=kept_cc,
    )


def _row_matching_pair_atoms(
    constraints: SolverConstraints,
    devs: Mapping[str, object],
) -> Tuple[Tuple[str, str], ...]:
    """Return only MOS pairs materialized as two-member row-plan atoms.

    The floorplan fallback may place several matching atoms in one row, each with
    its own local mirror axis.  Preserve the row plan's explicit atom boundary so
    downstream intent checks do not collapse those independent axes onto the
    cell-global ``axis_x``.  Overlapping or otherwise unselected matching edges
    are absent and therefore cannot acquire a certificate.
    """

    row_atoms = {
        tuple(sorted((str(payload[0]), str(payload[1]))))
        for row in _row_plan(constraints, dict(devs))
        for kind, payload in row
        if kind == "pair" and len(payload) == 2
    }
    disjoint = set(_disjoint_matching_pairs(
        tuple(constraints.symmetry_pairs or ()), tuple(devs)
    ))
    return tuple(sorted(
        pair
        for pair in row_atoms & disjoint
        if tuple(constraints.device_kinds.get(member) for member in pair)
        in {("nmos", "nmos"), ("pmos", "pmos")}
    ))


def _collect_resistors(cell_def, source_cir) -> Dict[str, Dict[str, object]]:
    """Collect ``label -> {value, pins, orient}`` for every resistor in the cell.

    Thin wrapper over :func:`_collect_passives` filtered to the ``res`` kind,
    kept so existing callers/tests are byte-unchanged."""
    return {
        lbl: {k: v for k, v in rec.items() if k != "kind"}
        for lbl, rec in _collect_passives(cell_def, source_cir).items()
        if rec.get("kind") == "res"
    }


def _collect_passives(cell_def, source_cir) -> Dict[str, Dict[str, object]]:
    """Collect ``label -> {kind, value, pins, orient}`` for every 2-terminal
    PASSIVE device (resistor + MIM capacitor) in the cell.

    Source preference (uniform, generic, no per-circuit branch):
      1. ``cd.devices`` with ``device_class in ('res','cap')`` (the materialised IR
         carries the value in ``sizing['R']`` / ``sizing['C']`` and the p/n pins);
      2. otherwise the bound source ``.cir`` ``R`` / ``C`` cards
         (:func:`parse_cir_passive_records`).
    A resistor value is ohms, a capacitor value is farads; both feed the same
    passive placement path (:func:`_place_passives`), which dispatches the geometry
    builder on ``kind``. The placer is therefore .cir-or-IR agnostic AND
    res-or-cap agnostic."""
    # per-device @orient annotation (R0 default -> byte-unchanged). A passive can
    # be declared @orient(R90); the passive builder realises it as a clean affine
    # rotation (LVS-invariant).
    orient_map = _resistor_orient_map(cell_def)
    _CLASS_VALUE_KEY = {"res": "R", "cap": "C"}
    out: Dict[str, Dict[str, object]] = {}
    for dev in getattr(cell_def, "devices", []) or []:
        dclass = getattr(dev, "device_class", None)
        if dclass not in ("res", "cap"):
            continue
        sizing = getattr(dev, "sizing", None) or {}
        value = sizing.get(_CLASS_VALUE_KEY[dclass], sizing.get("value"))
        pins = getattr(dev, "pins", None) or {}
        p = pins.get("p") or pins.get("plus")
        n = pins.get("n") or pins.get("minus")
        if value is None or p is None or n is None:
            continue
        # The SPICE inline multiplier `m=N` (N parallel unit devices) rides the
        # passive's `M` sizing field. Carry it onto the realization record so the
        # cap builds an m-unit bank (effective C = value*M) / the resistor builds
        # m parallel units (effective R = value/M). Absent / 1 -> single unit.
        m = max(1, int(round(float(sizing.get("M", 1) or 1))))
        out[dev.instance_name] = {
            "kind": dclass,
            "value": float(value),
            "pins": {"p": str(p), "n": str(n)},
            "orient": orient_map.get(dev.instance_name, "R0"),
            "M": m,
            "model": str(getattr(dev, "source_model", "") or ""),
        }
    if out:
        from clay.layout.passive_split import realize_passive_split_records

        return _consume_resistor_segment_intent(
            cell_def,
            _consume_passive_array_rows(
                cell_def,
                realize_passive_split_records(cell_def, out),
            ),
        )
    # fall back to the bound .cir R / C cards.
    if source_cir:
        recs = parse_cir_passive_records(source_cir)
        for lbl, rec in recs.items():
            if rec.get("kind") in ("res", "cap"):
                out[lbl] = {
                    "kind": rec["kind"],
                    "value": rec["value"],
                    "pins": dict(rec["pins"]),
                    "orient": orient_map.get(lbl, "R0"),
                    "M": max(1, int(rec.get("M", 1) or 1)),
                }
    from clay.layout.passive_split import realize_passive_split_records

    return _consume_resistor_segment_intent(
        cell_def,
        _consume_passive_array_rows(
            cell_def,
            realize_passive_split_records(cell_def, out),
        ),
    )


def _consume_resistor_segment_intent(
    cell_def,
    records: Dict[str, Dict[str, object]],
    *,
    pdk_id: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Attach exact ``@resistor_segments`` authority to passive records.

    This is the sole annotation-to-record lowering boundary.  It validates the
    existing parser schema without normalizing malformed collections, requires a
    resistor target and one unit PCell, and proves the declared series/parallel
    equivalent resistance.  The optional ``pdk_id`` is used by callers that
    already resolved a process; the canonical geometry consumer repeats that
    check unconditionally before drawing.
    """

    import math

    declaration_map = _resistor_segment_intent_map(cell_def)
    for target, declaration in declaration_map.items():
        record = records.get(target)
        if record is None:
            raise ValueError(
                f"@resistor_segments target {target!r} is not a materialized passive"
            )
        if record.get("kind") != "res":
            raise ValueError(
                f"@resistor_segments target {target!r} must be a resistor"
            )
        if pdk_id is not None and declaration.pdk != str(pdk_id):
            raise ValueError(
                f"@resistor_segments PDK {declaration.pdk!r} does not match "
                f"active PDK {str(pdk_id)!r}"
            )
        multiplicity = record.get("M", 1)
        if type(multiplicity) is not int or multiplicity != 1:
            raise ValueError(
                f"@resistor_segments target {target!r} requires exact M=1"
            )
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"@resistor_segments target {target!r} lacks a numeric resistance"
            )
        value = float(value)
        expected = (
            declaration.segment_resistance_ohms * declaration.count
            if declaration.connection == "series"
            else declaration.segment_resistance_ohms / declaration.count
        )
        quantum = declaration.value_quantum_ohms
        segment_quantum = declaration.segment_resistance_quantum_ohms
        interval_radius = quantum / 2.0 + (
            segment_quantum * declaration.count / 2.0
            if declaration.connection == "series"
            else segment_quantum / declaration.count / 2.0
        )
        if not (
            math.isfinite(value)
            and type(quantum) is float
            and math.isfinite(quantum)
            and quantum > 0.0
            and type(segment_quantum) is float
            and math.isfinite(segment_quantum)
            and segment_quantum >= 0.0
            and math.isclose(
                value,
                expected,
                rel_tol=0.0,
                abs_tol=interval_radius,
            )
        ):
            raise ValueError(
                f"@resistor_segments target {target!r} equivalent resistance "
                f"{expected:.12g} does not match schematic value {value:.12g} "
                "within the declared source quantization intervals "
                f"(value={quantum!r}, segment={segment_quantum!r})"
            )
        record["resistor_segments"] = declaration
    return records


def _resistor_segment_intent_map(cell_def) -> Dict[str, object]:
    """Return exact target-to-declaration bindings without schema coercion."""

    from clay.layout.annotations import ResistorSegments

    namespace = getattr(cell_def, "__dict__", {})
    if not isinstance(namespace, dict):
        raise TypeError("cell definition namespace must be one mapping")
    raw_map = (
        namespace["_layout_annotations"]
        if "_layout_annotations" in namespace
        else {}
    )
    if type(raw_map) is not dict:
        raise TypeError("layout annotation authority must be one canonical dict")

    out: Dict[str, object] = {}
    for raw_target, raw_declarations in raw_map.items():
        if (
            type(raw_target) is not str
            or not raw_target
            or raw_target != raw_target.strip()
        ):
            raise TypeError(
                "layout annotation target must be one exact non-empty string"
            )
        if type(raw_declarations) is not list:
            raise TypeError(
                f"layout annotation inventory for {raw_target!r} must be one "
                "canonical list"
            )
        for item in raw_declarations:
            if isinstance(item, ResistorSegments) and type(item) is not ResistorSegments:
                raise TypeError(
                    f"resistor {raw_target!r} declaration must be exact "
                    "ResistorSegments"
                )
        declarations = [
            item for item in raw_declarations if type(item) is ResistorSegments
        ]
        if not declarations:
            continue
        target = raw_target
        if target in out or len(declarations) != 1:
            raise ValueError(
                f"resistor {target!r} must have exactly one @resistor_segments "
                "declaration"
            )
        out[target] = declarations[0]
    return out


def _consume_passive_array_rows(
    cell_def,
    records: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Bind an exact passive ``@array`` declaration to its geometry record.

    MOS/common-centroid array declarations are intentionally ignored here.  A
    declaration whose group is an actual passive instance is a separate physical
    authority: it must describe one discrete ratio bank, name a capacitor, and
    tile its already-realized parallel multiplicity without a remainder.  The
    optional key is absent when no such declaration exists, preserving the legacy
    record and geometry path byte-for-byte for OTA and LDO.
    """
    from clay.layout.intent import layout_intent_from_cell_def

    intent = layout_intent_from_cell_def(cell_def)
    seen: set[str] = set()
    for declaration in intent.arrays:
        target = declaration.group
        if target not in records:
            if declaration.bank_interconnect is not None:
                raise ValueError(f"bank interconnect target {target} must be an actual passive capacitor")
            continue
        if target in seen:
            raise ValueError(f"duplicate passive array authority for {target}")
        seen.add(target)
        record = records[target]
        if record.get("kind") != "cap":
            raise ValueError(f"passive array target {target} must be a capacitor")
        if declaration.style != "discrete" or declaration.order != "ratio":
            raise ValueError(
                f"passive array target {target} requires style=discrete and order=ratio"
            )
        multiplicity = record.get("M")
        if type(multiplicity) is not int or multiplicity <= 1:
            raise ValueError(
                f"passive array target {target} requires an exact parallel multiplicity"
            )
        rows = declaration.rows
        if type(rows) is not int or rows <= 0 or rows > multiplicity:
            raise ValueError(f"passive array rows for {target} are out of range")
        if multiplicity % rows:
            raise ValueError(
                f"passive array rows for {target} must divide multiplicity {multiplicity}"
            )
        record["array_rows"] = rows
        if declaration.transforms:
            if len(declaration.transforms) != multiplicity:
                raise ValueError(
                    f"passive array transforms for {target} must have exactly "
                    f"multiplicity {multiplicity} members"
                )
            record["array_transforms"] = tuple(declaration.transforms)
        if declaration.align:
            record["array_align"] = declaration.align
        if declaration.bank_interconnect is not None:
            from clay.layout.bank_interconnect import replay_bank_interconnect_plan
            record["bank_interconnect_plan"] = replay_bank_interconnect_plan(declaration.bank_interconnect)
    return records


from clay.layout.source_bjt_materialization import _collect_bjts


def _resistor_orient_map(cell_def) -> Dict[str, str]:
    """``device_label -> orientation`` from each device's parsed ``@orient``
    annotation (carried on ``cell_def._layout_annotations``). Empty when none."""
    from clay.layout.annotations import Orient

    ann_map = getattr(cell_def, "__dict__", {}).get("_layout_annotations", {}) or {}
    out: Dict[str, str] = {}
    for target, anns in ann_map.items():
        for ann in anns or []:
            if isinstance(ann, Orient):
                out[target] = ann.orientation.value
    return out


def _validated_passive_geometry_override(
    label: str,
    rec: Dict[str, object],
    geometry: object,
    *, source_cell=None, tech=None,
):
    """Validate one exact fixed-passive geometry against its source record."""

    from clay.layout.passive_geometry_override import validate_passive_geometry_override
    return validate_passive_geometry_override(label, rec, geometry, source_cell=source_cell, tech=tech)


def _apply_passive_geometry_overrides(
    records: Dict[str, Dict[str, object]],
    overrides: Optional[Dict[str, object]],
    *, source_cell=None, tech=None,
) -> Dict[str, Dict[str, object]]:
    """Attach validated, source-keyed fixed geometry without mutating records."""

    if overrides is None:
        return records
    if type(overrides) is not dict:
        raise TypeError("passive geometry overrides must be one exact dict")
    unknown = tuple(sorted(set(overrides) - set(records)))
    if unknown:
        raise ValueError(f"unknown passive geometry override(s): {unknown!r}")
    out = {label: dict(record) for label, record in records.items()}
    for label, geometry in overrides.items():
        validated = _validated_passive_geometry_override(
            label,
            out[label],
            geometry,
            source_cell=source_cell, tech=tech,
        )
        from clay.layout.source_mim_materialization import SourceMimMaterialization, source_mim_record
        if type(geometry) is SourceMimMaterialization:
            out[label] = source_mim_record(out[label], geometry)
            continue
        out[label]["_geometry_override"] = validated
    return out


def _build_passive_geom(rec: Dict[str, object], tech, orient: str):
    """Build the :class:`DeviceGeom` for one passive record, dispatching on its
    ``kind`` (``res`` -> ``build_resistor``, ``cap`` -> ``build_capacitor``).

    GENERIC: the only thing that varies per device is which PDK-tech-driven passive
    builder is called; both return the same DeviceGeom contract (p/n terminals +
    FEOL/MIM polys), so the placement + routing path downstream is identical.

    The cap path dispatches through the per-PDK adapter
    :func:`passive_device_cell.build_capacitor_for_tech` (gf180 -> build_capacitor,
    open_pdk -> build_mim_capacitor_tsmc, external_pdk -> external_pdk MIM primitive) so the LDO's
    MIM cap builds on open_pdk / open_pdk instead of crashing on the bare
    gf180-only builder. ``m`` (parallel-unit bank count) and ``model`` (netlist
    ``$.MODEL`` annotation) carry off the passive record when present; this is the
    SAME mature per-PDK passive path anchor_ota uses."""
    if '_native_assembly' in rec:
        return native_member_geometry(rec, tech, orient)
    override = rec.get("_geometry_override")
    if override is not None:
        return _validated_passive_geometry_override(
            "<attached>",
            rec,
            override,
            tech=tech,
        )

    from .passive_device_cell import (
        build_capacitor_for_tech,
        build_resistor,
        build_segmented_resistor,
    )

    pins = rec["pins"]  # type: ignore[index]
    kind = str(rec.get("kind", "res"))
    if kind in ("pnp", "npn"):
        if str(orient or "R0").upper() != "R0":
            raise BjtMaterializationUnavailable(
                f"fixed-cell BJT orientation {orient!r} is not proven"
            )
        from clay.layout.bjt_device_cell import build_bjt_cell
        from clay.pdk import get_pdk, load_all

        load_all()
        return build_bjt_cell(
            get_pdk(str(getattr(tech, "pdk_id", ""))),
            kind,
            units=int(rec.get("units", 1) or 1),
            tap_declaration=rec.get("tap_declaration"),
            unit_pattern=rec.get("unit_pattern"),
        )
    m = max(1, int(rec.get("M", 1) or 1))  # type: ignore[arg-type]
    if kind == "cap":
        model = str(rec.get("model", "") or "")
        # `value` is the PER-UNIT capacitance; the m-unit parallel bank realizes
        # the effective C = value*M (the builder tiles m physical units).
        array_options = {}
        if "array_rows" in rec:
            array_options["bank_rows"] = rec["array_rows"]
        if "array_transforms" in rec:
            array_options["unit_transforms"] = rec["array_transforms"]
        if "array_align" in rec:
            array_options["bank_align"] = rec["array_align"]
        if "bank_interconnect_plan" in rec:
            array_options["bank_interconnect_plan"] = rec["bank_interconnect_plan"]
        if str(getattr(tech, "pdk_id", "")).lower() == "gf180mcu":
            if "mim_option" in rec and rec["mim_option"] is not None:
                array_options["mim_option"] = rec["mim_option"]
            if "metal_level" in rec and rec["metal_level"] is not None:
                array_options["metal_level"] = rec["metal_level"]
            if "cap_option" in rec and rec["cap_option"] is not None:
                array_options["cap_option"] = rec["cap_option"]
            if "pex_backend" in rec and rec["pex_backend"] is not None:
                array_options["pex_backend"] = rec["pex_backend"]
        return build_capacitor_for_tech(
            tech,
            float(rec["value"]),
            nets=pins,
            m=m,
            model=model,
            orient=orient,
            **array_options,
        )  # type: ignore[arg-type]
    # Resistor: `value` is the PER-UNIT resistance; m parallel units give an
    # effective R = value/M. `build_resistor` realizes a single body, so the
    # effective resistance is mapped to one body length (R = value/M). M == 1
    # leaves the body byte-unchanged.
    segment_intent = rec.get("resistor_segments")
    if segment_intent is not None:
        from clay.layout.annotations import ResistorSegments

        if type(segment_intent) is not ResistorSegments:
            raise TypeError("resistor_segments record must be exact ResistorSegments")
        if segment_intent.pdk != str(getattr(tech, "pdk_id", "")):
            raise ValueError(
                f"@resistor_segments PDK {segment_intent.pdk!r} does not match "
                f"active PDK {getattr(tech, 'pdk_id', '')!r}"
            )
        return build_segmented_resistor(
            tech,
            float(rec["value"]),
            count=segment_intent.count,
            connection=segment_intent.connection,
            segment_width_um=segment_intent.width_um,
            segment_length_um=segment_intent.length_um,
            segment_resistance_ohms=segment_intent.segment_resistance_ohms,
            value_quantum_ohms=segment_intent.value_quantum_ohms,
            segment_resistance_quantum_ohms=(
                segment_intent.segment_resistance_quantum_ohms
            ),
            model=str(rec.get("model", "") or ""),
            orient=orient,
            nets=pins,
        )
    return build_resistor(
        tech, float(rec["value"]) / m, orient=orient, nets=pins,
    )  # type: ignore[arg-type]


def _rigid_nonmos_member_relations_from_intent(graph, intent):
    """Project authored order rows owned wholly by one rigid non-MOS macro."""

    from clay.layout.nonrow_macro_plan import RigidNonmosMemberRelation

    structural = tuple(
        macro
        for macro in tuple(getattr(graph, "macros", ()) or ())
        if str(getattr(macro, "kind", "")) in {"passive_bank", "bjt_cluster"}
    )
    projected = []
    for raw_left, raw_right, raw_relation in tuple(
        getattr(intent, "order_constraints", ()) or ()
    ):
        left = str(raw_left)
        right = str(raw_right)
        relation = str(raw_relation)
        declaration_ref = f"@order({left},{right},{relation})"
        if relation == "above":
            left, right, relation = right, left, "below"
        elif relation not in {"left_of", "below"}:
            continue
        owners = tuple(
            macro
            for macro in structural
            if {left, right} <= set(getattr(macro, "members", ()))
        )
        if not owners:
            continue
        if len(owners) != 1:
            raise ValueError(
                "authored non-MOS order has ambiguous rigid ownership: "
                f"{declaration_ref}"
            )
        projected.append(RigidNonmosMemberRelation(
            left=left,
            right=right,
            relation=relation,
            declaration_ref=declaration_ref,
        ))
    return tuple(sorted(set(projected)))


def _rigid_nonmos_member_alignments_from_intent(graph, intent):
    """Project authored edge equalities owned wholly by one rigid non-MOS macro."""

    from clay.layout.nonrow_macro_plan import RigidNonmosMemberAlignment

    structural = tuple(
        macro
        for macro in tuple(getattr(graph, "macros", ()) or ())
        if str(getattr(macro, "kind", "")) in {"passive_bank", "bjt_cluster"}
    )
    projected = []
    for row in tuple(getattr(intent, "edge_alignments", ()) or ()):
        raw_first = str(getattr(row, "first", ""))
        raw_second = str(getattr(row, "second", ""))
        edge = str(getattr(row, "edge", ""))
        declaration_ref = f"@align({raw_first},{raw_second},edge={edge})"
        owners = tuple(
            macro
            for macro in structural
            if {raw_first, raw_second} <= set(getattr(macro, "members", ()))
        )
        if not owners:
            continue
        if getattr(row, "offset_nm", 0):
            raise ValueError("relative edge offsets within precompiled passive banks are unsupported")
        if len(owners) != 1:
            raise ValueError(
                "authored non-MOS alignment has ambiguous rigid ownership: "
                f"{declaration_ref}"
            )
        first, second = sorted((raw_first, raw_second))
        projected.append(RigidNonmosMemberAlignment(
            first=first,
            second=second,
            edge=edge,
            declaration_ref=declaration_ref,
        ))
    return tuple(sorted(set(projected)))


def compile_rigid_nonmos_macro_plans(graph, nonmos, tech, intent=None, *, native_assemblies=()):
    """Compile exact aggregate passive/BJT tilings before placement solve."""

    from clay.layout.nonrow_macro_plan import (
        RigidNonmosMacroCompileInputs,
        RigidNonmosMemberSize,
        compile_rigid_nonmos_macro_interfaces,
        compile_rigid_nonmos_macro_plans_from_inputs,
    )
    from clay.layout.physical_macro_contracts import PhysicalMacroTerminalAccess
    from clay.layout.nonrow_adjacency import (
        project_internal_adjacencies, compile_with_native_spacing,
    )

    aggregate_members = tuple(sorted({
        member
        for macro in graph.macros
        if macro.kind in ("passive_bank", "bjt_cluster")
        for member in macro.members
    }))
    geometries = {}
    for member in aggregate_members:
        record = nonmos.get(member)
        if record is None:
            from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanError

            raise RigidNonmosMacroPlanError(
                f"declared aggregate non-MOS member lacks geometry: {member}"
            )
        geometries[member] = _build_passive_geom(
            record, tech, str(record.get("orient", "R0"))
        )
    sizes = []
    for member in aggregate_members:
        x0, y0, x1, y1 = geometries[member].extent()
        sizes.append(RigidNonmosMemberSize(
            member=member,
            width_nm=int(x1) - int(x0),
            height_nm=int(y1) - int(y0),
        ))
    inputs = RigidNonmosMacroCompileInputs(
        graph_sha256=graph.sha256,
        tech_id=str(tech.pdk_id),
        gap_nm=_dev_gap_from_tech(tech),
        member_sizes=tuple(sizes),
        native_assemblies=native_assemblies,
        member_adjacencies=project_internal_adjacencies(graph, intent, tech),
        member_relations=(
            _rigid_nonmos_member_relations_from_intent(graph, intent)
            if intent is not None
            else ()
        ),
        member_alignments=(
            _rigid_nonmos_member_alignments_from_intent(graph, intent)
            if intent is not None
            else ()
        ),
    )
    plans, inputs = compile_with_native_spacing(graph, inputs, geometries, tech)
    interface_pins = {}
    canonical_terminal_accesses = {}
    for member in aggregate_members:
        record = nonmos[member]
        pins = {
            str(role): str(net)
            for role, net in (record.get("pins", {}) or {}).items()
            if str(net)
        }
        tap = record.get("tap_declaration")
        tap_net = str(getattr(tap, "net", "") or "")
        if tap_net:
            pins["tap"] = tap_net
            pins["body_tie"] = tap_net
        interface_pins[member] = pins
        if str(record.get("kind", "")).lower() == "cap":
            # Freeze the native materializer's actual accesses, including its
            # on-device risers and explicit bank wiring.  The legacy scalar
            # cap probe can choose a different upper-plate landing and cannot
            # authenticate the _PlacedPassiveStamp consumed by the router.
            native, _, _ = _build_resistor_placed(member, record, tech, 0, 0)
            canonical_terminal_accesses[member] = tuple(sorted(
                PhysicalMacroTerminalAccess(
                    member=member,
                    role=role,
                    net=pins[role],
                    layer=tuple(native.term_layers[role]),
                    accesses_nm=tuple(sorted(
                        native.terminal_accesses.get(role) or (point,)
                    )),
                )
                for role, point in native.terminals.items()
            ))
    interfaces, inputs = compile_rigid_nonmos_macro_interfaces(
        graph,
        plans,
        inputs,
        geometries=geometries,
        device_pins=interface_pins,
        canonical_terminal_accesses=canonical_terminal_accesses,
        tech=tech,
    )
    return plans, inputs, interfaces


def attach_rigid_nonmos_projection(placement: Placement) -> Placement:
    """Issue one final certificate for the current aggregate macro placement."""

    from clay.layout.nonrow_macro_plan import (
        RigidNonmosMacroCompileInputs,
        RigidNonmosMacroInterfaceSet,
        RigidNonmosMacroPlanSet,
        RigidNonmosPlacementCertificate,
    )
    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    graph = placement.physical_macro_graph
    plans = placement.rigid_nonmos_macro_plans
    inputs = placement.rigid_nonmos_macro_inputs
    interfaces = placement.rigid_nonmos_macro_interfaces
    prior = placement.rigid_nonmos_placement_certificate
    if graph is None:
        if any(item is not None for item in (plans, inputs, interfaces, prior)):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_STRUCTURE_METADATA_REMOVED"
            )
        return placement
    if type(graph) is not PhysicalMacroGraph:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_METADATA_INCOMPLETE"
        )
    _verify_physical_macro_graph_views(placement, graph.sha256)
    expected = tuple(
        macro for macro in graph.macros
        if macro.kind in ("passive_bank", "bjt_cluster")
    )
    if not expected:
        if any(item is not None for item in (plans, inputs, interfaces, prior)):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_METADATA_UNEXPECTED"
            )
        return placement
    if (
        type(plans) is not RigidNonmosMacroPlanSet
        or type(inputs) is not RigidNonmosMacroCompileInputs
        or type(interfaces) is not RigidNonmosMacroInterfaceSet
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_METADATA_INCOMPLETE"
        )
    certificate = RigidNonmosPlacementCertificate.build(
        plans, interfaces, placement.devices
    )
    return replace(
        placement, rigid_nonmos_placement_certificate=certificate
    )


def _rigid_nonmos_foreign_footprints(
    devices: Mapping[str, PlacedDevice],
    row_certificate: object,
    *,
    exact_device_bboxes: object = None,
) -> Tuple[Tuple[str, Bbox, frozenset[str]], ...]:
    """Return physical foreign footprints for aggregate-macro intrusion.

    A consumed row member has no standalone artwork: its ``PlacedDevice`` is a
    logical terminal view whose bbox can retain the wider pre-fold PCell extent.
    The row certificate instead binds the exact placed row-master artwork.  Use
    that one atomic bbox.  At the final-route seam, ordinary/free folded devices
    likewise consume their exact emitted-stamp bboxes instead of retaining wider
    logical PCell envelopes.  Without final-route authority the legacy logical
    views remain the conservative placement-time blockers.
    """

    consumed: Set[str] = set()
    footprints: List[Tuple[str, Bbox, frozenset[str]]] = []
    for binding in getattr(row_certificate, "macro_bindings", ()):
        members = frozenset(str(name) for name in binding.consumed_devices)
        consumed.update(members)
        footprints.append((str(binding.owner), binding.bbox_nm, members))
    routed_bboxes = None
    if exact_device_bboxes is not None:
        if type(exact_device_bboxes) is not dict:
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_FOREIGN_FOOTPRINT_AUTHORITY_MALFORMED"
            )
        routed_bboxes = {}
        for raw_name, raw_bbox in exact_device_bboxes.items():
            if (
                type(raw_name) is not str
                or not raw_name
                or type(raw_bbox) is not tuple
                or len(raw_bbox) != 4
                or any(type(value) is not int for value in raw_bbox)
                or raw_bbox[2] <= raw_bbox[0]
                or raw_bbox[3] <= raw_bbox[1]
            ):
                raise RigidNonmosMacroProjectionError(
                    "RIGID_NONMOS_FOREIGN_FOOTPRINT_AUTHORITY_MALFORMED"
                )
            routed_bboxes[raw_name] = raw_bbox
        missing = tuple(sorted(
            str(name)
            for name in devices
            if str(name) not in consumed and str(name) not in routed_bboxes
        ))
        if missing:
            raise RigidNonmosMacroProjectionError(
                f"RIGID_NONMOS_FOREIGN_FOOTPRINT_MISSING:{missing[0]}"
            )
    footprints.extend(
        (
            str(name),
            (
                view.bbox
                if routed_bboxes is None
                else routed_bboxes[str(name)]
            ),
            frozenset((str(name),)),
        )
        for name, view in devices.items()
        if str(name) not in consumed
    )
    return tuple(footprints)


def verify_rigid_nonmos_macro_projection(
    placement: Placement,
    *,
    exact_device_bboxes: object = None,
) -> bool:
    """Replay aggregate plans and prove no foreign physical footprint enters."""

    from clay.layout.nonrow_macro_plan import (
        RigidNonmosMacroCompileInputs,
        RigidNonmosMacroInterfaceSet,
        RigidNonmosMacroPlanSet,
        RigidNonmosPlacementCertificate,
        RigidNonmosMacroPlanError,
        compile_rigid_nonmos_macro_plans_from_inputs,
        replay_rigid_nonmos_macro_interfaces,
    )
    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    graph = placement.physical_macro_graph
    plans = placement.rigid_nonmos_macro_plans
    inputs = placement.rigid_nonmos_macro_inputs
    interfaces = placement.rigid_nonmos_macro_interfaces
    if graph is None:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_STRUCTURE_METADATA_REMOVED"
        )
    if type(graph) is not PhysicalMacroGraph:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_METADATA_INCOMPLETE"
        )
    _verify_physical_macro_graph_views(placement, graph.sha256)
    expected = tuple(
        macro for macro in getattr(graph, "macros", ())
        if getattr(macro, "kind", "") in ("passive_bank", "bjt_cluster")
    )
    if not expected:
        if (
            plans is not None
            or inputs is not None
            or interfaces is not None
            or placement.rigid_nonmos_placement_certificate is not None
        ):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_METADATA_UNEXPECTED"
            )
        return True
    if (
        type(plans) is not RigidNonmosMacroPlanSet
        or type(inputs) is not RigidNonmosMacroCompileInputs
        or type(interfaces) is not RigidNonmosMacroInterfaceSet
        or type(placement.rigid_nonmos_placement_certificate)
        is not RigidNonmosPlacementCertificate
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_METADATA_INCOMPLETE"
        )
    try:
        replayed = compile_rigid_nonmos_macro_plans_from_inputs(graph, inputs)
    except RigidNonmosMacroPlanError as exc:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_PLAN_REPLAY_MISMATCH"
        ) from exc
    if replayed.sha256 != plans.sha256 or replayed.payload() != plans.payload():
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_PLAN_REPLAY_MISMATCH"
        )
    try:
        replayed_interfaces = replay_rigid_nonmos_macro_interfaces(
            graph, plans, inputs
        )
    except RigidNonmosMacroPlanError as exc:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_INTERFACE_REPLAY_MISMATCH"
        ) from exc
    if (
        replayed_interfaces.sha256 != interfaces.sha256
        or replayed_interfaces.payload() != interfaces.payload()
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_INTERFACE_REPLAY_MISMATCH"
        )
    expected_certificate = RigidNonmosPlacementCertificate.build(
        plans, interfaces, placement.devices
    )
    certificate = placement.rigid_nonmos_placement_certificate
    owned = {member for plan in plans.plans for member in plan.members}
    devices = placement.devices
    try:
        verify_stored_row_projection(placement)
    except RowFinalPlacementResidualError as exc:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_ROW_PROJECTION_MISMATCH"
        ) from exc
    foreign_footprints = _rigid_nonmos_foreign_footprints(
        devices,
        placement.row_final_certificate,
        exact_device_bboxes=exact_device_bboxes,
    )
    native_geometry = placement.row_native_geometry_binding
    nonrow_native = placement.rigid_nonmos_native_geometry_binding
    if nonrow_native is not None:
        from clay.layout.place.row_native_geometry import RowNativeGeometryBinding

        if (type(nonrow_native) is not RowNativeGeometryBinding
                or native_geometry is not None or placement.row_program is not None):
            raise RigidNonmosMacroProjectionError("RIGID_NONMOS_NATIVE_GEOMETRY_BINDING_MISMATCH")
        native_geometry = nonrow_native
    if native_geometry is not None:
        # These plans declare rigid member geometry, not a whole-envelope
        # keepout. Replay the same native layer proof used by row projection.
        # Explicit flexible-envelope exclusion remains part of that certificate.
        # Bind the plan's logical member view as well as the row's independent
        # physical bbox view so a translated bank cannot borrow an old proof.
        native_geometry.require_source_bboxes(source_bboxes=tuple(
            (name, devices[name].bbox) for name in sorted(owned)
        ))
        if exact_device_bboxes is not None:
            free_names = (
                {name for name, _box in placement.row_final_certificate.free_device_blocks}
                if placement.row_final_certificate is not None else set(devices) - owned
            )
            native_geometry.require_source_bboxes(source_bboxes=tuple(
                (name, bbox) for name, bbox, _members in foreign_footprints
                if name in free_names
            ))
    else:
        for plan in plans.plans:
            macro_bbox = plan.placed_bbox(devices)
            for name, bbox, footprint_members in foreign_footprints:
                if footprint_members & set(plan.members):
                    continue
                if (
                    bbox[2] > macro_bbox[0]
                    and bbox[0] < macro_bbox[2]
                    and bbox[3] > macro_bbox[1]
                    and bbox[1] < macro_bbox[3]
                ):
                    raise RigidNonmosMacroProjectionError(
                        f"RIGID_NONMOS_FOREIGN_INTRUSION:{plan.macro_id}:{name}"
                    )
    if not owned <= set(devices):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_MEMBER_MISSING"
        )
    actual_axes = placement.matching_pair_axes
    if type(actual_axes) is not dict:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_MATCHING_AXIS_MISMATCH"
        )
    expected_axes = project_rigid_nonmos_matching_axes(placement, devices)
    for plan in plans.plans:
        for pair in plan.matching_pairs:
            if actual_axes.get(pair) != expected_axes[pair]:
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MATCHING_AXIS_MISMATCH:{plan.macro_id}:{pair}"
                )
    if (
        certificate.sha256 != expected_certificate.sha256
        or certificate.payload() != expected_certificate.payload()
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_PLACEMENT_CERTIFICATE_MISMATCH"
        )
    return True


def project_rigid_nonmos_matching_axes(
    placement: Placement,
    devices: Optional[Mapping[str, PlacedDevice]] = None,
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Translate certified pair axes only across proven rigid member motion.

    Every materialized pair already owns a local axis.  When both member views
    preserve their footprint and undergo the identical translation, translate
    that existing certificate by the same vector.  Independent member motion
    keeps the old axis so the downstream geometry check fails rather than
    manufacturing a tautological axis from the final centroids.

    Rigid non-MOS macro plans retain their stronger whole-macro replay below.
    """

    views = placement.devices if devices is None else devices
    result = dict(getattr(placement, "matching_pair_axes", {}) or {})
    original_views = dict(getattr(placement, "devices", {}) or {})
    for pair, (axis, coordinate) in tuple(sorted(result.items())):
        if axis not in {"x", "y"} or len(pair) != 2:
            continue
        try:
            original = [original_views[name].bbox for name in pair]
            projected = [views[name].bbox for name in pair]
        except (KeyError, AttributeError, TypeError):
            continue
        original_sizes = [
            (int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
            for box in original
        ]
        projected_sizes = [
            (int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
            for box in projected
        ]
        translations = [
            (
                int(after[0]) - int(before[0]),
                int(after[1]) - int(before[1]),
            )
            for before, after in zip(original, projected)
        ]
        if original_sizes != projected_sizes or translations[0] != translations[1]:
            continue
        delta = translations[0][0 if axis == "x" else 1]
        result[pair] = (axis, float(coordinate) + float(delta))

    plans = getattr(placement, "rigid_nonmos_macro_plans", None)
    for plan in getattr(plans, "plans", ()):
        for pair in plan.matching_pairs:
            try:
                left = views[pair[0]].bbox
                right = views[pair[1]].bbox
            except (KeyError, AttributeError, TypeError) as exc:
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MEMBER_MISSING:{pair}"
                ) from exc
            left_x = (float(left[0]) + float(left[2])) / 2.0
            right_x = (float(right[0]) + float(right[2])) / 2.0
            result[pair] = ("x", (left_x + right_x) / 2.0)
    return result


def _build_resistor_placed(
    label: str, rec: Dict[str, object], tech, x0: int, y0: int
) -> Tuple[PlacedDevice, list, Bbox]:
    """Build one passive cell (resistor or MIM cap) and translate it so its drawn
    extent's lower-left lands at ``(x0, y0)``. Returns
    ``(PlacedDevice, placed_polys, placed_extent)``.

    The device's p/n metal contact pads become the placed terminals the router
    connects; the FEOL/MIM polys are translated into the same coordinate frame the
    router's terminals live in. ``kind`` carries through to the PlacedDevice."""
    orient = str(rec.get("orient", "R0"))  # type: ignore[union-attr]
    kind = str(rec.get("kind", "res"))
    geom = _build_passive_geom(rec, tech, orient)
    from clay.layout.source_mim_materialization import SourceMimMaterialization
    source_mim = rec.get("_geometry_override")
    source_mim = source_mim if type(source_mim) is SourceMimMaterialization else None
    if source_mim is not None and source_mim.source_member[0] != label:
        raise ValueError("source MIM placed logical owner differs")
    native_layers = dict(source_mim.terminal_layers) if source_mim is not None else getattr(geom, "terminal_layers", {})
    native_accesses = getattr(geom, "terminal_accesses", {})
    if source_mim is not None:
        from clay.layout.device_cell import Terminal
        native_accesses = {role: tuple(Terminal(role, box, rec["pins"][role]) for box in boxes)
                           for role, boxes in source_mim.terminal_accesses}
    ex = geom.extent()
    dx = int(x0) - ex[0]
    dy = int(y0) - ex[1]
    placed_polys = _translate(list(geom.polygons), dx, dy)

    def _shift_center(b: Optional[Bbox]) -> Optional[Tuple[int, int]]:
        if b is None:
            return None
        tb = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        return ((tb[0] + tb[2]) // 2, (tb[1] + tb[3]) // 2)

    terms: Dict[str, Tuple[int, int]] = {}
    term_layers: Dict[str, Tuple[int, int]] = {}
    terminal_accesses: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    # the routable-metal ENTRY layer of each terminal pad (the layer the router must
    # start its riser from). GENERIC: read from the placed geometry -- the HIGHEST
    # routing-metal layer covering the terminal centre. A poly resistor's pads are on
    # Metal1 -> entry == M1 -> the router default (byte-unchanged). A MIM cap's plate
    # handles are on Metal2 -> entry == M2 -> the router reaches them ON M2 with no
    # Via1 (which on the MIM bottom plate would fire MIM.10).
    metal_keys: list = []
    for lvl in (1, 2, 3, 4, 5):
        try:
            mk = tech.metal_key(lvl)
        except Exception:  # noqa: BLE001
            mk = None
        metal_keys.append(tuple(mk) if mk is not None else None)
    terminal_roles = (
        tuple(role for role in ("c", "b", "e", "tap") if role in geom.terminals)
        if kind in ("pnp", "npn")
        else ("p", "n")
    )
    for role in terminal_roles:
        t = geom.terminals.get(role)
        c = _shift_center(t.bbox if t is not None else None)
        if c is not None:
            terms[role] = c
            entry = (native_layers or {}).get(role)
            if entry is None:
                entry = _terminal_entry_layer(t.bbox, geom.polygons, metal_keys)
            if entry is not None:
                term_layers[role] = tuple(entry)
        accesses = tuple(
            point
            for terminal in (native_accesses or {}).get(role, ())
            if (point := _shift_center(terminal.bbox)) is not None
        )
        if accesses:
            terminal_accesses[role] = accesses

    placed_extent = (ex[0] + dx, ex[1] + dy, ex[2] + dx, ex[3] + dy)
    passive_access = None
    source_resistor = getattr(geom, 'source_resistor_assembly', None)
    if kind in ("cap", "res") and source_resistor is None:
        from clay.layout.passive_terminal_binding import _capture_materialized_passive_access
        passive_access = _capture_materialized_passive_access(
            geom, record=rec, terminal_layers=term_layers, tech=tech,
            terminal_accesses=native_accesses,
            mim_site_ledger=source_mim.site_ledger if source_mim is not None else ())
    pd = PlacedDevice(
        label=label, kind=kind, bbox=placed_extent, terminals=terms,
        polys=placed_polys, term_layers=term_layers,
        terminal_accesses=terminal_accesses,
        materialized_passive_access=passive_access,
        source_resistor_assembly=source_resistor,
        bjt_unit_occurrences=getattr(geom, "unit_occurrences", None),
        orientation=str(getattr(geom, "orient", orient) or ""),
        fixed_pcell=dict(getattr(geom, "fixed_pcell", {}) or {}),
    )
    return pd, placed_polys, placed_extent


def _terminal_entry_layer(term_bbox, geom_polys, metal_keys):
    """The HIGHEST routing-metal ``(layer, datatype)`` whose polygon covers the
    centre of ``term_bbox`` (the layer the router must start a riser from).

    ``metal_keys`` is the ordered ``[M1, M2, M3, ...]`` key list (lowest -> highest).
    A terminal pad on Metal1 (the MOS / poly-resistor case) returns the M1 key; a MIM
    cap plate handle on Metal2 returns the M2 key. Returns ``None`` when no routing
    metal covers the terminal (the caller then keeps the M1 default)."""
    if term_bbox is None:
        return None
    cx = (term_bbox[0] + term_bbox[2]) // 2
    cy = (term_bbox[1] + term_bbox[3]) // 2
    key_set = {k for k in metal_keys if k is not None}
    covering: set = set()
    for lp in geom_polys:
        k = (lp.layer, lp.datatype)
        if k not in key_set:
            continue
        vs = lp.polygon.vertices
        if not vs:
            continue
        x0 = min(v.x for v in vs); x1 = max(v.x for v in vs)
        y0 = min(v.y for v in vs); y1 = max(v.y for v in vs)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            covering.add(k)
    # highest metal level among the covering routing metals.
    for k in reversed(metal_keys):
        if k is not None and k in covering:
            return k
    return None


def _place_resistors(
    resistors: Dict[str, Dict[str, object]], tech, mos_extent: Bbox,
    *,
    col_of: Optional[Dict[str, int]] = None,
    mos_max_col: Optional[int] = None,
    column_channel: Optional[int] = None,
    mid_band: Optional[Tuple[float, float]] = None,
    mid_rows: Optional[set] = None,
    passive_rows: Optional[Dict[str, int]] = None,
    mos_x_extent: Optional[Tuple[int, int]] = None,
    mos_col_centroid: Optional[Dict[int, float]] = None,
    bank_grid: bool = False,
    matched_pairs: Optional[Tuple[Tuple[str, str], ...]] = None,
    rigid_macro_plans=None,
) -> Tuple[
    Dict[str, PlacedDevice],
    list,
    Bbox,
    Tuple[Tuple[str, str], ...],
]:
    """Place every passive (resistor + MIM cap), honoring a DECLARED floorplan.

    THREE placement regimes, chosen PER PASSIVE off the declared ``@place_group``:

    * MIDDLE TIER (3-tier reference paradigm): a passive whose declared ROW is the
      reserved MIDDLE band between two MOS bands (``passive_rows[label] in mid_rows``)
      is laid out SIDE-BY-SIDE, ordered by its declared ``col``, INSIDE the reserved
      y-window ``mid_band = (y_lo, y_hi)`` -- vertically centred in the band, spanning
      the MOS x-extent. This realises the reference's middle tier: the two EQUAL caps
      side-by-side (col 0, col 1) plus the small poly resistors beside them, BETWEEN
      the PMOS top + NMOS bottom bands. Overlap-free: the reservation already opened a
      y-window taller than the passives, and the side-by-side x-cursor never overlaps.
    * OWN RIGHT-HAND COLUMN: a passive whose declared column (``col_of[label]``) is a
      NEW column BEYOND the MOS columns (``col > mos_max_col``) is laid out as a BLOCK
      in its OWN x-region to the RIGHT of the MOS grid, vertically centred on the MOS
      band -- so a large MIM plate gets its own footprint CLEAR of the MOS channel.
    * BELOW-MOS ROW (legacy / unconstrained): a passive with NO declared row/column
      keeps the historical single row BELOW the MOS band (``_RES_BAND_GAP_NM``
      clearance), laid left-to-right with ``_RES_ROW_GAP_NM`` spacing.

    GENERIC + circuit-agnostic: the regime is selected purely from the declared
    row/column data + the device bbox -- no per-circuit / net-name / device-label
    branch. Returns the placed devices, their combined polys, the UNION extent
    (MOS + passives), and the exact local two-member matching atoms actually
    packed together. The fourth return is placement-certificate authority; a
    declared pair split across middle/different-column regimes is absent."""
    placed: Dict[str, PlacedDevice] = {}
    polys: list = []
    col_of = col_of or {}
    mid_rows = mid_rows or set()
    passive_rows = passive_rows or {}
    mos_col_centroid = mos_col_centroid or {}
    matched_pairs = matched_pairs or ()
    plans = tuple(getattr(rigid_macro_plans, "plans", ()) or ())
    rigid_members = {
        member for plan in plans for member in plan.members
    }
    if not rigid_members <= set(resistors):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_MEMBER_MISSING"
        )
    realized_pair_atoms: set[Tuple[str, str]] = set()
    mx0, my0, mx1, my1 = mos_extent
    mos_cy = (my0 + my1) // 2
    chan = int(column_channel) if column_channel is not None else _RES_ROW_GAP_NM
    union = mos_extent

    # Partition: MIDDLE-tier passives (declared middle row) -> side-by-side middle
    # band; passives that declare their OWN column beyond the MOS grid -> right-hand
    # block region; everything else -> the legacy below-MOS row.
    middle: List[str] = []
    own_col: Dict[str, int] = {}
    below_row: List[str] = []
    for label in resistors:
        if label in rigid_members:
            continue
        if (
            mid_band is not None
            and passive_rows.get(label) is not None
            and passive_rows[label] in mid_rows
        ):
            middle.append(label)
            continue
        c = col_of.get(label)
        if c is not None and (mos_max_col is None or c > mos_max_col):
            own_col[label] = int(c)
        else:
            below_row.append(label)

    # ── MIDDLE tier: side-by-side, ordered by declared col, in the reserved window ──
    if middle:
        y_lo, y_hi = mid_band  # reserved y-window between the two MOS bands
        # order the middle passives left-to-right by declared col (then label).
        middle.sort(key=lambda l: (col_of.get(l, 1 << 30), l))
        band_cy = (y_lo + y_hi) / 2.0
        # TARGET x-CENTROID per middle passive, honouring the DECLARED global column
        # ORDER (the place-group check compares centroids). A passive whose declared
        # col matches a MOS column is centred on that COLUMN'S CENTROID (so a wide cap
        # keeps the column band order even though its plate spans neighbours -- it is
        # in a different TIER, so no bbox overlap). A passive whose col is BEYOND the
        # MOS columns extrapolates one column-pitch right of the last MOS centroid.
        # The targets are then made strictly INCREASING by sorted column + the passive
        # half-widths so adjacent middle passives never x-overlap each other.
        geoms = {
            lbl: _build_passive_geom(resistors[lbl], tech,
                                     str(resistors[lbl].get("orient", "R0")))
            for lbl in middle
        }
        half = {lbl: (geoms[lbl].extent()[2] - geoms[lbl].extent()[0]) / 2.0
                for lbl in middle}
        # column pitch for extrapolating cols beyond the MOS grid: the mean MOS
        # column spacing (or the channel when only one MOS column exists).
        mos_cols_sorted = sorted(mos_col_centroid)
        if len(mos_cols_sorted) >= 2:
            pitch = (mos_col_centroid[mos_cols_sorted[-1]]
                     - mos_col_centroid[mos_cols_sorted[0]]) / (len(mos_cols_sorted) - 1)
        else:
            pitch = float(chan)
        last_mos_c = mos_cols_sorted[-1] if mos_cols_sorted else 0
        last_mos_cx = (mos_col_centroid[last_mos_c]
                       if mos_cols_sorted else (mx0 + mx1) / 2.0)

        def _target_cx(lbl: str) -> float:
            c = col_of.get(lbl)
            if c is not None and c in mos_col_centroid:
                return mos_col_centroid[c]
            if c is not None and mos_cols_sorted and c > last_mos_c:
                return last_mos_cx + pitch * (c - last_mos_c)
            # no usable column hint -> fall to the MOS-extent left edge progression.
            return (mx0 if mos_x_extent is None else mos_x_extent[0]) + half[lbl]

        targets = {lbl: _target_cx(lbl) for lbl in middle}
        # enforce strictly increasing, non-overlapping centres in column order.
        prev_right = None
        for lbl in middle:
            cx = targets[lbl]
            if prev_right is not None and cx - half[lbl] < prev_right + chan:
                cx = prev_right + chan + half[lbl]
            gh = geoms[lbl].extent()[3] - geoms[lbl].extent()[1]
            x0 = int(round(cx - half[lbl]))
            y0 = int(round(band_cy - gh / 2.0))
            pd, ppolys, pext = _build_resistor_placed(label=lbl, rec=resistors[lbl],
                                                      tech=tech, x0=x0, y0=y0)
            placed[lbl] = pd
            polys += ppolys
            prev_right = pext[2]
            union = (
                min(union[0], pext[0]), min(union[1], pext[1]),
                max(union[2], pext[2]), max(union[3], pext[3]),
            )

    # ── right-hand declared-column blocks (each passive in its own x-region) ─────────
    # Group by declared column; lay columns left-to-right starting one channel right of
    # the MOS grid. Within a column, stack devices bottom-to-top, centred on the column.
    if own_col:
        cursor_x = mx1 + chan
        for c in sorted(set(own_col.values())):
            members = sorted(lbl for lbl, cc in own_col.items() if cc == c)
            geoms = {
                lbl: _build_passive_geom(
                    resistors[lbl], tech, str(resistors[lbl].get("orient", "R0")))
                for lbl in members
            }
            widths = {
                lbl: geoms[lbl].extent()[2] - geoms[lbl].extent()[0]
                for lbl in members
            }
            heights = {
                lbl: geoms[lbl].extent()[3] - geoms[lbl].extent()[1]
                for lbl in members
            }
            # A declared matched pair in ONE passive column is one horizontal
            # bank row, not two vertically unrelated cells.  This preserves the
            # local mirror environment and is also the compact construction the
            # column declaration asks for.  Unpaired members stay one-cell rows.
            atoms = list(_matching_atoms(members, matched_pairs))
            atom_width = {
                atom: (
                    sum(widths[lbl] for lbl in atom)
                    + chan * (len(atom) - 1)
                )
                for atom in atoms
            }
            atom_height = {
                atom: max(heights[lbl] for lbl in atom)
                for atom in atoms
            }
            col_w = max(atom_width.values())
            col_h = sum(atom_height.values()) + chan * (len(atoms) - 1)
            # stack bottom-to-top, vertically centred on the MOS band.
            cur_y = mos_cy - col_h // 2
            for atom in atoms:
                ah = atom_height[atom]
                atom_x = cursor_x + (col_w - atom_width[atom]) // 2
                x_cursor = atom_x
                atom_top = cur_y + ah
                for lbl in atom:
                    gh = heights[lbl]
                    y0 = cur_y + (ah - gh) // 2
                    pd, ppolys, pext = _build_resistor_placed(
                        lbl, resistors[lbl], tech, x_cursor, y0)
                    placed[lbl] = pd
                    polys += ppolys
                    x_cursor = pext[2] + chan
                    union = (
                        min(union[0], pext[0]), min(union[1], pext[1]),
                        max(union[2], pext[2]), max(union[3], pext[3]),
                    )
                if len(atom) == 2:
                    realized_pair_atoms.add(tuple(sorted((atom[0], atom[1]))))
                cur_y = atom_top + chan
            cursor_x += col_w + chan

    # ── precompiled aggregate macros: one rigid tile per declaration ────────────────
    # These local member layouts were closed before the MOS solve.  The legacy
    # passive placer chooses only one macro translation here; it cannot repack or
    # interleave individual members with free passives.
    rigid_top = my0 - _RES_BAND_GAP_NM
    for plan in plans:
        macro_x = mx0
        macro_y = rigid_top - plan.local_bbox_nm[3]
        for member_plan in plan.member_plans:
            lx0, ly0, _lx1, _ly1 = member_plan.local_bbox_nm
            label = member_plan.member
            pd, ppolys, pext = _build_resistor_placed(
                label,
                resistors[label],
                tech,
                macro_x + lx0,
                macro_y + ly0,
            )
            placed[label] = pd
            polys += ppolys
            union = (
                min(union[0], pext[0]), min(union[1], pext[1]),
                max(union[2], pext[2]), max(union[3], pext[3]),
            )
        realized_pair_atoms.update(plan.matching_pairs)
        rigid_top = macro_y - _RES_ROW_GAP_NM

    # ── below-MOS passive region (no declared column) ───────────────────────────────
    # A passive BANK (>= _BANK_MIN_MEMBERS passives whose single-row width would
    # OVERFLOW the MOS x-extent) is laid as a compact 2-D GRID below the MOS band,
    # wrapping into ``ncol`` columns so the bank stays roughly as wide as the MOS core
    # instead of a long thin strip spanning many times its width. A small set (or a
    # set that already fits) keeps the historical single left-to-right row exactly
    # (byte-identical: the grid path only triggers on the overflow predicate). The
    # grid is the same compact-cluster the reference's folded resistor bank uses; it
    # is overlap-free (each cell advances the cursor by its own footprint + gap).
    if below_row:
        labels = sorted(below_row)
        geoms = {
            lbl: _build_passive_geom(resistors[lbl], tech,
                                     str(resistors[lbl].get("orient", "R0")))
            for lbl in labels
        }
        widths = {lbl: geoms[lbl].extent()[2] - geoms[lbl].extent()[0] for lbl in labels}
        heights = {lbl: geoms[lbl].extent()[3] - geoms[lbl].extent()[1] for lbl in labels}
        atoms = _matching_atoms(labels, matched_pairs)
        atom_width = {
            atom: (
                sum(widths[label] for label in atom)
                + _RES_ROW_GAP_NM * (len(atom) - 1)
            )
            for atom in atoms
        }
        atom_height = {
            atom: max(heights[label] for label in atom)
            for atom in atoms
        }
        single_row_w = (
            sum(atom_width.values())
            + _RES_ROW_GAP_NM * (len(atoms) - 1)
        )
        mos_w = max(1, mx1 - mx0)

        # Number of columns: one physical row unless this is a wide bank that
        # overflows the MOS extent.  For an overflowing bank, evaluate every bounded
        # contiguous grid shape against the EXACT combined bbox already materialized
        # above.  The former ``widest cell * ncol`` proxy rejected legal two-column
        # shapes when the wide cell shared a row with a narrow one, creating a much
        # taller bank.  Exact row widths/heights preserve atom order and never split
        # a matching atom while avoiding that false resource bound.
        if bank_grid and len(labels) >= _BANK_MIN_MEMBERS and single_row_w > mos_w:
            from clay.layout.place.hierarchical_macro_placement import (
                plan_geometry_bounded_grid,
            )

            grid_plan = plan_geometry_bounded_grid(
                tuple((atom_width[atom], atom_height[atom]) for atom in atoms),
                gap_nm=_RES_ROW_GAP_NM,
                fixed_bbox_nm=union,
                anchor_x_nm=mx0,
                top_y_nm=rigid_top,
            )
            ncol = grid_plan.columns
        else:
            ncol = len(atoms)

        row_top = rigid_top
        # lay row-by-row, top-to-bottom; each grid row's height is its tallest cell.
        cur_top = row_top
        for r0 in range(0, len(atoms), ncol):
            row_atoms = atoms[r0:r0 + ncol]
            row_h = max(atom_height[atom] for atom in row_atoms)
            cursor_x = mx0
            for atom in row_atoms:
                atom_cy = cur_top - atom_height[atom] / 2.0
                for label in atom:
                    # Members of one matching atom share a row centre even when
                    # their footprints differ; the direct footprint checker will
                    # still report a measurable failure for the size mismatch.
                    y0 = int(round(atom_cy - heights[label] / 2.0))
                    pd, ppolys, pext = _build_resistor_placed(
                        label, resistors[label], tech, cursor_x, y0)
                    placed[label] = pd
                    polys += ppolys
                    cursor_x = pext[2] + _RES_ROW_GAP_NM
                    union = (
                        min(union[0], pext[0]), min(union[1], pext[1]),
                        max(union[2], pext[2]), max(union[3], pext[3]),
                    )
                if len(atom) == 2:
                    realized_pair_atoms.add(tuple(sorted((atom[0], atom[1]))))
            cur_top = cur_top - row_h - _RES_ROW_GAP_NM
    return placed, polys, union, tuple(sorted(realized_pair_atoms))


def _passive_only_placement(
    cell_def, resistors: Dict[str, Dict[str, object]], tech,
    constraints: SolverConstraints, used_default: bool,
    *, rigid_macro_plans=None,
) -> Placement:
    """Placement for a PURE PASSIVE cell (resistors / MIM caps, no MOS).

    The passives are laid out on one row starting at the origin; the cell geometry
    + rails are derived from their union extent, so the router can strap rail nets
    and route signal nets between the device terminals exactly as for a MOS cell.
    """
    # seed the row at the origin (no MOS band to sit under).
    seed = (0, 0, 0, 0)
    matching_pairs = _canonical_matching_pairs(
        tuple(getattr(constraints, "symmetry_pairs", ()) or ()),
        resistors.keys(),
    )
    placed, polys, union, realized_pair_atoms = _place_resistors(
        resistors,
        tech,
        seed,
        matched_pairs=matching_pairs,
        rigid_macro_plans=rigid_macro_plans,
    )
    # @equal_size also applies to a pure-passive cell (two equal MIM caps, etc.) -- the
    # passives are the only devices here, so the shared contract is realized over them.
    placed, union = _apply_equal_size(placed, constraints, {}, resistors, tech, union)
    geom = _cell_geom(placed, union, tech=tech)
    return Placement(
        devices=placed,
        axis_x=(union[0] + union[2]) // 2,
        geom=geom,
        polys=polys,
        quality=None,
        solver_status="passive_only",
        solver_success=True,
        solve_seconds=0.0,
        constraints=constraints,
        used_default_geometry=used_default,
        matching_pair_axes=_materialized_matching_axes(
            placed, realized_pair_atoms
        ),
        dropped_nonmos_matched=tuple(sorted({
            member
            for pair in matching_pairs
            for member in pair
            if member not in placed
        })),
        note=(
            f"cell {cell_def.local_name!r}: passive-only placement, "
            f"{len(placed)} passive device(s) on one row."
        ),
    )


def _cell_geom(placed: Dict[str, PlacedDevice], full_extent: Bbox, *, tech=None) -> Dict[str, int]:
    """Derive the cell rail / channel geometry from the placed device extents.

    ``vss_y`` / ``vdd_y`` are thin-strip rail lines just outside the lowest NMOS /
    highest PMOS band (the same convention :func:`score_placement` synthesises);
    ``channel_lo`` / ``channel_hi`` bound the active device band; ``span`` is the
    cell x-extent. When a band is absent (a single-rail cell) its rail falls back to
    the band edge so the geometry stays well-defined.
    """
    if not placed:
        return {}
    nmos_y0 = [p.bbox[1] for p in placed.values() if p.kind == "nmos"]
    nmos_y1 = [p.bbox[3] for p in placed.values() if p.kind == "nmos"]
    pmos_y0 = [p.bbox[1] for p in placed.values() if p.kind == "pmos"]
    pmos_y1 = [p.bbox[3] for p in placed.values() if p.kind == "pmos"]
    lo = full_extent[1]
    hi = full_extent[3]
    strip = 800
    # Are the NMOS / PMOS cleanly band-separated (every NMOS below every PMOS)?
    # The historical std-cell convention puts vss just under the NMOS band and vdd
    # just over the PMOS band. But a BAND-FREE 2D floorplan (intent-driven) can
    # interleave NMOS + PMOS rows, in which case that rule yields a vss ABOVE vdd
    # (broken rails). When the bands are NOT cleanly separated, derive the rails
    # from the cell EXTENT instead -- vss is a strap under the whole cell, vdd a
    # strap over it -- so the rails track the ACTUAL placement, not a fixed
    # vss-bottom/vdd-top device-class rule.
    banded = bool(nmos_y1) and bool(pmos_y0) and max(nmos_y1) <= min(pmos_y0)
    if banded:
        vss_y = min(nmos_y0) - strip
        vdd_y = max(pmos_y1) + strip
    else:
        vss_y = lo - strip
        vdd_y = hi + strip

    # MIM keep-out for a bottom / top MIM cap. The vss rail is drawn one ``strip``
    # (800nm) below the cell bottom and the vdd rail one strip above the top; each
    # rail's drops land M2 via-pads at that y. A MIM cap's bottom plate is Metal2,
    # and MIM.1 forbids ANY other Metal2 within 1.2um of it -- so when a MIM cap is
    # the bottom-most (or top-most) device, the 800nm strip puts the rail's M2
    # via-pad inside the MIM.1 keep-out (an MIM.1 violation between the cap plate and
    # the rail via-pad). Push the rail OUT to clear the MIM bottom-plate-to-Metal2
    # spacing whenever a cap touches that cell edge. Generic: keyed on a cap device's
    # extent reaching the cell edge + the PDK MIM keep-out, no per-cell branch; a cell
    # with no MIM cap (or a non-gf180 PDK -> keep-out 0) is byte-unchanged.
    try:
        from .passive_device_cell import mim_keepout_nm
        from .annotate_to_solver import _resolve_tech as _rt
        _mim_ko = int(mim_keepout_nm(tech if tech is not None else _rt(None)))
    except Exception:  # noqa: BLE001 - no MIM stack / unknown PDK -> no extra clearance
        _mim_ko = 0
    if _mim_ko > strip:
        cap_y0 = [p.bbox[1] for p in placed.values() if getattr(p, "kind", "") == "cap"]
        cap_y1 = [p.bbox[3] for p in placed.values() if getattr(p, "kind", "") == "cap"]
        # a cap whose bottom edge reaches the cell bottom -> widen the vss strip.
        if cap_y0 and min(cap_y0) <= lo:
            vss_y = min(vss_y, lo - _mim_ko)
        # a cap whose top edge reaches the cell top -> widen the vdd strip.
        if cap_y1 and max(cap_y1) >= hi:
            vdd_y = max(vdd_y, hi + _mim_ko)
    return {
        "vss_y": int(vss_y),
        "vdd_y": int(vdd_y),
        "channel_lo": int(lo),
        "channel_hi": int(hi),
        "span": int(full_extent[2] - full_extent[0]),
        "x_min": int(full_extent[0]),
        "x_max": int(full_extent[2]),
        "y_min": int(full_extent[1]),
        "y_max": int(full_extent[3]),
    }


__all__ = [
    "PlacedDevice",
    "Placement",
    "place_generic",
    "verify_row_source_compiler_authority",
    "annotated_cell_from_cell_def",
    "parse_cir_device_sizing",
    "parse_cir_mos_records",
    "parse_cir_passive_records",
]
