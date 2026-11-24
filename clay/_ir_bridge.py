"""
Python ↔ Rust IR bridge (v0.0.65 Phase 1, v0.0.66 Layer 2).

Translates Clay sources into a Rust-owned port-constrained hypergraph
(`clay_ir.CellGraphHandle`). The Rust side owns the graph state;
this module is the one-shot constructor. Two construction paths:

  * **Layer 1** — `cell_to_hypergraph(cell)`. Consumes a fully-compiled
    `clay.ir.Cell` and emits the matching Rust hypergraph by calling
    `CellGraphHandle.add_*_node` / `connect` through the PyO3 extension.
    The Python-side `compile_cell_def(...)` output round-trips through
    it cleanly. v0.0.65 shipped this surface.

  * **Layer 2 (v0.0.66)** — `parsed_to_hypergraph(handle, cell_name)`.
    Consumes a `clay_parser.ParsedClayHandle` directly and bypasses
    Python's `clay.ir.Cell` IR. Lifted onto the Rust side now that
    v0.0.66 surfaces deep per-cell getters (`cell_ports`,
    `cell_devices`, `cell_passives`, `cell_instances`, …) on the
    handle. Mirrors Layer 1's bucketing algorithm; coverage gaps
    relative to Layer 1 are documented in the function docstring
    (role_decls / digital_decls are not expanded — the expansion
    logic lives in `clay/parser.py::_ClayTransformer` and porting
    it to Rust is out-of-scope for v0.0.66).

Design constraints honored:

  * **No mutation of `clay/ir.py`.** Python-side IR stays the source of
    truth for Layer 1; the bridge is read-only over `Cell` objects.
  * **No Rust-side changes.** Layer 1 uses only the v0.0.64 PyO3
    surface (`add_external_port`, `add_mos_node`, `add_cap_node`,
    `add_res_node`, `add_ind_node`, `add_varactor_node`,
    `add_cell_instance_node`, `connect`, `set_pdk_id`).
  * **Hard-error when extension missing (v0.1.07 R84 C).** Top-level
    `_HAS_CLAY_IR = False` when `clay_ir` import fails; bridge entry
    points (`cell_to_hypergraph`, `parsed_to_hypergraph`,
    `compute_gradients`, `compute_dc_gain`, `compute_gbw`, `compute_pm`,
    `gradient_on_functional_target`, the 10 ported testbench builders)
    raise `RuntimeError` via `_require_clay_ir()` rather than returning
    None / silently emitting a Python-only netlist. The v0.0.65 graceful
    `return None` contract proved load-bearing for callers that
    immediately wrap the bridge call in `try/except Exception` (see
    `clay/invariants.py::augment_violation_with_gradient`,
    `clay/distance_to_sat.py::make_adjoint_fn_from_bridge`) — flipping
    to a hard error surfaces "extension missing" loudly through those
    soft-fail catches without changing the user-visible behaviour.
  * **No port-constraint declaration at this level.** v0.0.65 emits
    plain hyperedges with no constraints — the layer that decides
    "vdd must be PowerOnly" lives in Phase 2 semantic analysis,
    not in the bridge. Tests still pass `validate_port_constraints()`
    (empty constraint list trivially well-formed).

Device-class coverage:

  * `nmos` / `pmos` → `add_mos_node(..., is_nmos=...)`, 4 ports (d, g, s, b).
  * `cap` / `res` / `ind` → `add_{cap,res,ind}_node(...)`, 2 ports (p, n).
  * `varactor` → `add_varactor_node(...)`, 3 ports (p, n, b). The
    Python-side Device only carries `p` and `n`; the Rust `b` port is
    left unconnected (a valid hypergraph state — no edge touches it).
  * `cur_src` → `add_cell_instance_node(...,
    cell_path='clay.ir::CurrentSource', port_specs=[('p',...),('n',...)])`
    until v0.0.66 surfaces a dedicated `add_source_node` PyO3 method.
    The Rust node lands with `kind='cell'` and
    `variant='clay.ir::CurrentSource'`; sizing carries the source value.
  * `vcvs` → similar to `cur_src`: lands as a Cell instance with
    `variant='clay.ir::VCVS'`. (Not commonly seen on bare cells —
    typically a Testbench `extra_device`. Included for completeness
    so the bridge doesn't crash if a cell happens to carry one.)

Hyperedge construction:

  The Python `Device.pins: dict[str, str]` maps pin labels to net
  names. After walking every device, we bucket by net name and emit
  one hyperedge per net containing every (node_id, port_id) endpoint
  on that net. The order of endpoints within an edge follows iteration
  over devices in declaration order, which keeps round-trip-comparable
  output deterministic.

  Nets that touch external cell ports (i.e. one of `cell.ports`) get a
  net_name matching the port name. Internal nets stay as their Python
  string identifier (e.g. `tail`, `d1`, `nbias`).
"""

from __future__ import annotations

import math
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from .ir import Cell, Device, ParsedFunctionalRole


# ─── Extension import guard ─────────────────────────────────────

_HAS_CLAY_IR = False
try:
    import clay_ir as _clay_ir  # type: ignore[import-not-found]

    _HAS_CLAY_IR = True
except ImportError:
    _clay_ir = None  # type: ignore[assignment]


# v0.1.07 R84 C: hard-error message when the Rust extension is missing.
# Used by `_require_clay_ir()` below; centralised so every bridge entry
# point surfaces the same diagnostic.
_RUST_EXT_REQUIRED_MSG = (
    "clay_ir Rust extension required for this bridge entry point; "
    "build with `maturin develop` from the `crates/clay_ir/` directory. "
    "v0.1.07 R84 C flipped the v0.0.65 graceful-fallback "
    "(`return None` / silent Python emission) to a hard error so "
    "extensionless environments fail loudly instead of silently "
    "computing degraded answers."
)


def _require_clay_ir() -> None:
    """Raise `RuntimeError` when `clay_ir` is not importable.

    v0.1.07 R84 C replaced the v0.0.65 graceful `if not _HAS_CLAY_IR:
    return None` fallback at every bridge entry point with this hard
    error. Callers (e.g. `clay.invariants.augment_violation_with_gradient`,
    `clay.distance_to_sat.make_adjoint_fn_from_bridge`) wrap the bridge
    call in `try/except Exception` so the hard error propagates cleanly
    as a `"_gradient_error": ...` entry without crashing the LLM RL loop.

    v0.1.10 R86 C — Tier B.1.a/b mass deletion: the v0.0.87 and v0.0.92
    inject helpers (`_emit_functional_role_op_lines`,
    `_emit_v92_functional_role_lines`) are now Rust-canonical, so the
    non-empty-roles path also hard-errors on a missing extension instead
    of falling through to a Python implementation.
    """
    if not _HAS_CLAY_IR:
        raise RuntimeError(_RUST_EXT_REQUIRED_MSG)


def has_rust_ir() -> bool:
    """True iff the `clay_ir` PyO3 extension is importable."""
    return _HAS_CLAY_IR


# ─── Port-role inference ────────────────────────────────────────
#
# `Cell.ports` is a flat `list[str]` of port names. The Python IR
# carries optional role metadata on CellDef.port_roles but Cell itself
# drops it during compilation. We infer the (role, kind) tuple for
# external ports by name heuristic — matches what `clay/ir.py::Cell`
# semantically wants but doesn't formalize.

_POWER_NAMES = {"vdd", "vdd!", "vcc", "vpp"}
_GROUND_NAMES = {"vss", "vss!", "gnd", "0"}


def _infer_external_port_role(port_name: str) -> tuple[str, str]:
    """Return (role_str, kind_str) for an external port based on its
    name. Used because `Cell.ports` is a bare `list[str]` — the role
    metadata is dropped during `compile_cell_def`.

    Conventions:
      * `vdd` / `vcc` / `vpp` → ('power', 'power')
      * `vss` / `gnd` / `0` → ('ground', 'ground')
      * everything else → ('generic', 'analog') — neither input nor
        output is reliably inferable from the bare name. The Rust IR
        accepts 'generic' as a fallback role; analog as kind keeps
        a port-constraint validation pass-through.
    """
    lo = port_name.lower()
    if lo in _POWER_NAMES:
        return ("power", "power")
    if lo in _GROUND_NAMES:
        return ("ground", "ground")
    return ("generic", "analog")


# ─── Device → Rust node factories ───────────────────────────────


# v0.0.94 Round 70 Agent A: Cap-node sizing key normalization.
#
# Python `Device.sizing` uses `"value"` (the legacy single-Farad key the
# parser + ngspice driver write/read — see `clay/parser.py` rule
# `cap C1 { ... value = 1.5pF; }` and `clay/drivers/ngspice.py:
# C = dev.sizing["value"]`). The Rust adjoint AC stamp reads `"Cc"`
# first and falls back to `"C"` (see `crates/clay_ir/src/adjoint.rs::
# build_mna_system` and v0.0.93 test
# `test_pm_cc_on_miller_cell_is_finite_nonzero` for the contract).
#
# Before v0.0.94, the bridge passed the raw sizing dict through to
# `add_cap_node`, so `value`-keyed dicts from the Python compile path
# landed in the Rust hypergraph with no `Cc` entry — the AC stamp's
# `node.sizing.get("Cc").or_else("C")` returned 0.0 and the Miller cap
# contributed nothing to the C matrix. Adjoint ∂PM/∂Cc on registered
# templates (e.g. `two_stage_miller_pmos`) consequently returned 0.0,
# even though the same gradient on a hand-built handle (where the test
# fixture wrote `{"Cc": cc}` directly via `add_cap_node`) was finite
# and FD-agreeing.
#
# This helper synchronises all three known keys (`value`, `Cc`, `C`) so
# every reader (Python ngspice driver via `"value"`, Rust AC stamp via
# `"Cc"`, future readers via `"C"`) sees the same Farad number. Callers
# that already wrote multiple keys win — we never overwrite an
# explicitly-set entry, only fill the missing ones from the first
# present source in priority order (`Cc` > `C` > `value`).
#
# Non-cap nodes are not touched by this helper — only `_add_*` paths
# whose `cls == "cap"` invoke it. Resistor / inductor / varactor
# sizings retain their existing keys.

_CAP_SIZING_KEYS: tuple[str, ...] = ("Cc", "C", "value")

# v0.0.94 Round 71 Agent D — MOS sizing key canonicalization.
#
# Why this lives next to `_CAP_SIZING_KEYS`:
#   The Rust `add_mos_node` factory takes `HashMap<String, f64>` as the
#   sizing payload — opaque on the wire. Downstream Rust readers
#   (`crates/clay_ir/src/dc_op.rs`, `semantic.rs`, `spice_export.rs`,
#   `adjoint.rs`) look up *literal* string keys `"W"`, `"L"`, `"M"` with
#   no case-folding fallback. A Python template that wrote `{"w": ...}`
#   (lowercase) would land an unreachable knob — the Rust side would see
#   the default 0/1, and every downstream gm / go / Cgs computation
#   would silently use 0 width.
#
# What about Cgs / Cgd / Csb / Cdb?
#   Audited the entire Rust crate (`crates/clay_ir/src/*.rs`): no read
#   site touches a `"Cgs"` / `"Cgd"` / `"Csb"` / `"Cdb"` key on the MOS
#   sizing dict. Parasitic gate-source / gate-drain caps are *derived*
#   inside `adjoint.rs::capacitances_for_device(W, L, gm, v_ov)` from
#   the canonical `W`, `L`, and the device OP — not consumed from the
#   sizing dict. So a Python bridge that wrote a `"Cgs"` key would have
#   that key silently dropped on the Rust side; the AC stamp would
#   still use the (2/3)·W·L·Cox + overlap closed form.
#
#   This means there is no current Python→Rust MOS-cap mismatch to
#   normalize — the symmetric situation to R70 A's `_normalize_cap_sizing`
#   simply does not exist for MOS parasitic caps at this Rust ABI
#   version. If a future round (R72+) extends Rust to read `Cgs` /
#   `Cgd` as first-class sizing knobs (so a device can carry an
#   *explicit* parasitic cap override that bypasses the closed-form),
#   the normalizer below is the single point to extend with the same
#   value / Cgs / Cgd mirroring pattern `_normalize_cap_sizing` uses
#   for the Cap node.
#
# What this helper does today:
#   Case-canonicalize the three MOS sizing knobs Rust *does* consume:
#   `W`, `L`, `M`. If the caller passed `{"w": 1e-6, "L": 0.5e-6}` (mixed
#   case), the lowercase entry gets promoted to canonical `W` so every
#   downstream Rust read site picks it up. Explicit canonical entries
#   are NEVER overwritten by a lowercase variant (Cc-helper contract:
#   additive, no value-drift).
_MOS_SIZING_KNOBS: tuple[str, ...] = ("W", "L", "M")


def _normalize_mos_sizing(sizing: Optional[dict]) -> Optional[dict]:
    """Return a copy of `sizing` with the MOS knobs `W` / `L` / `M`
    promoted to their canonical-case keys.

    Why bother:
      Rust `dc_op.rs::compute_mosfet_op`, `semantic.rs::compute_dc_gain`,
      `spice_export.rs::emit_mosfet_line`, and the `adjoint.rs`
      perturbation helpers all read `node.sizing.get("W")` etc. with no
      case-folding fallback. A lowercase `"w"` key would land an
      unreachable knob (default 0 width on Rust → gm = 0 → every
      sensitivity vanishes). The PyO3 `single_param_sensitivity_complex`
      registry separately accepts `"w" | "W"` via `ParamKind::
      from_kind_str` for the *perturbation* string, but that registry
      indirection only protects the gradient-knob lookup; the underlying
      sizing dict read still uses literal uppercase.

    Behaviour matrix:
      * `sizing = None` or `sizing = {}` → returned unchanged
        (`None` / empty dict).
      * `{"W": 1e-6, "L": 0.5e-6, "M": 2.0}` (canonical) → returned
        unchanged (modulo identity — a defensive `dict(sizing)` copy is
        made only when a lowercase key was actually found).
      * `{"w": 1e-6}` → promoted to `{"W": 1e-6, "w": 1e-6}`. Both keys
        survive; the lowercase entry is kept so any Python read site
        that historically used lowercase still finds its value.
      * `{"W": 1e-6, "w": 2e-6}` → returned unchanged. Explicit
        canonical key wins; the lowercase mismatch is preserved as-is
        (additive contract — same as `_normalize_cap_sizing`).

    Parasitic cap (`Cgs` / `Cgd` / `Csb` / `Cdb`) keys: intentionally
    NOT touched. Rust does not consume them off the MOS sizing dict
    today (see module-level comment block above).
    """
    if not sizing:
        return sizing
    # Quick-out: are any lowercase variants present? If not, the dict is
    # already canonical and we hand back the input as-is (no copy cost
    # on the hot path, which dominates large registered cells).
    has_lowercase = any(k.lower() in {"w", "l", "m"} and k != k.upper() for k in sizing)
    if not has_lowercase:
        return sizing
    out = dict(sizing)
    for knob in _MOS_SIZING_KNOBS:
        lower = knob.lower()
        if knob not in out and lower in out:
            # Lowercase variant present, canonical missing — mirror.
            out[knob] = out[lower]
        # If both are present, leave both. Explicit canonical wins;
        # the lowercase variant is preserved (no silent drop) so any
        # Python read site that depends on it keeps working.
    return out


def _normalize_cap_sizing(sizing: Optional[dict]) -> Optional[dict]:
    """Return a copy of `sizing` with `Cc` / `C` / `value` all populated
    when at least one of them is present.

    Resolution priority (highest first):
      1. `Cc` — what the Rust AC stamp reads first.
      2. `C`  — the secondary key the Rust stamp falls back to.
      3. `value` — the legacy Python convention; the parser writes it
                   and the ngspice Python driver reads it.

    If `sizing` is None or empty, returned unchanged. If none of the
    three keys are present, returned unchanged (no Farad value to
    normalize — e.g. a cap node carrying only PDK-variant kwargs).

    The function NEVER overwrites an explicitly-set entry: if the
    caller passed `{"value": 1e-12, "Cc": 2e-12}` (mismatched on
    purpose for a future variant), both numbers survive untouched.
    Only the genuinely-missing keys are filled. This keeps the
    helper additive — no behaviour drift on cells that already use
    the new convention.
    """
    if not sizing:
        return sizing
    # Quick-out when none of the cap keys are present (e.g. a future
    # cap node with PDK kwargs only — no Farad number to mirror).
    if not any(k in sizing for k in _CAP_SIZING_KEYS):
        return sizing
    out = dict(sizing)
    # Pick the source value: first key present in priority order.
    src_value: Optional[float] = None
    for key in _CAP_SIZING_KEYS:
        if key in out:
            src_value = out[key]
            break
    if src_value is None:
        return out
    # Fill any missing key with the source value. Explicit entries are
    # preserved (the `if key not in out` guard).
    for key in _CAP_SIZING_KEYS:
        if key not in out:
            out[key] = src_value
    return out


def _add_device_node(handle, dev: Device) -> Optional[int]:
    """Emit one Rust HyperNode for a Python Device. Returns the new
    node id, or None if the device class is not currently mappable
    (the caller skips the device and emits a warning at the
    cell-walk level).

    See module docstring for the per-device-class mapping rules.
    """
    cls = dev.device_class
    sizing = dict(dev.sizing) if dev.sizing else None

    if cls == "nmos":
        # v0.0.94 R71 D: canonical-case W/L/M (idempotent on already-
        # canonical dicts). Rust does NOT read Cgs / Cgd from the MOS
        # sizing dict at this ABI version — see `_normalize_mos_sizing`
        # docstring for the audit.
        return handle.add_mos_node(
            dev.instance_name, True, dev.pdk_variant,
            _normalize_mos_sizing(sizing),
        )
    if cls == "pmos":
        return handle.add_mos_node(
            dev.instance_name, False, dev.pdk_variant,
            _normalize_mos_sizing(sizing),
        )
    if cls == "cap":
        # v0.0.94: synchronise `value` (Python convention) ↔ `Cc` / `C`
        # (Rust AC stamp keys) so the Miller cap value flows through
        # to the C matrix on registered template cells. See the
        # `_normalize_cap_sizing` docstring for the contract.
        return handle.add_cap_node(
            dev.instance_name, dev.pdk_variant, _normalize_cap_sizing(sizing)
        )
    if cls == "res":
        return handle.add_res_node(
            dev.instance_name, dev.pdk_variant, sizing
        )
    if cls == "ind":
        return handle.add_ind_node(
            dev.instance_name, dev.pdk_variant, sizing
        )
    if cls == "varactor":
        return handle.add_varactor_node(
            dev.instance_name, dev.pdk_variant, sizing
        )
    if cls == "cur_src":
        # Until v0.0.66 surfaces a dedicated `add_source_node` PyO3
        # method, route through `add_cell_instance_node` so the node
        # exists in the hypergraph. Two ports (p, n) carry the same
        # endpoints the Python `Source.pins` would.
        return handle.add_cell_instance_node(
            dev.instance_name,
            "clay.ir::CurrentSource",
            [("p", "plus", "analog"), ("n", "minus", "analog")],
            sizing,
        )
    if cls == "vcvs":
        # Rare on Cell.devices (usually a Testbench.extra_device), but
        # supported for completeness so the bridge never bails on a
        # legal Cell. VCVS has 4 ports in SPICE (out+, out-, in+, in-);
        # the Python IR uses `p` / `n` / `inp` / `inn` keys.
        return handle.add_cell_instance_node(
            dev.instance_name,
            "clay.ir::VCVS",
            [
                ("p", "plus", "analog"),
                ("n", "minus", "analog"),
                ("inp", "input", "analog"),
                ("inn", "input", "analog"),
            ],
            sizing,
        )
    return None


# ─── Top-level Layer 1 entry point ──────────────────────────────


def cell_to_hypergraph(cell: Cell):
    """Translate a Python `clay.ir.Cell` to a Rust `CellGraphHandle`.

    Returns the populated handle on success, or `None` when the
    `clay_ir` PyO3 extension is not importable (graceful fallback so
    callers can keep their Python-only code path active in CI without
    the Rust toolchain).

    Algorithm:

      1. Construct an empty `CellGraphHandle` with `cell.name`.
      2. Propagate `cell.pdk_id` via `set_pdk_id`.
      3. For each `cell.ports` name, emit an external port with role
         inferred from the name (`_infer_external_port_role`).
      4. For each `cell.devices` entry, emit the matching Rust node
         and remember the (instance_name, pin_name → port_id) map.
      5. Bucket every device's pin → net binding into a
         `{net_name: [(node_id, port_id), ...]}` map; then emit one
         hyperedge per non-empty bucket with no constraints.
      6. Return the handle.

    Devices whose `device_class` is not in the bridge's mappable set
    (see module docstring) are silently dropped at this layer — the
    caller can detect coverage gaps by comparing
    `handle.node_count()` to `len(cell.devices)`. A future round will
    surface a structured diagnostic.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). Soft-fail callers wrap the
    bridge call in `try/except Exception` so the user-visible behaviour
    is unchanged for them; envs with no extension now fail loudly
    instead of silently degrading.
    """
    _require_clay_ir()

    handle = _clay_ir.CellGraphHandle.new_empty(cell.name)
    if cell.pdk_id:
        handle.set_pdk_id(cell.pdk_id)

    # External ports: walk `cell.ports` in declaration order.
    for port_name in cell.ports:
        role, kind = _infer_external_port_role(port_name)
        handle.add_external_port(port_name, role, kind)

    # Devices: emit one Rust node each. Build a parallel index of
    # `(node_id, pin_name → port_id)` so the connection step can look
    # up endpoints without re-walking the graph.
    device_nodes: list[tuple[int, dict[str, str]]] = []
    for dev in cell.devices:
        nid = _add_device_node(handle, dev)
        if nid is None:
            # Unmappable device class — skip. Document gap; do not
            # block the rest of the cell from translating.
            device_nodes.append((-1, dev.pins))
            continue
        device_nodes.append((nid, dev.pins))

    # Bucket pin endpoints by net name. The Python pin map is
    # `{pin_label: net_name}`; we invert it per device and append the
    # resolved `(node_id, port_id)` tuple to each net's bucket.
    nets: dict[str, list[tuple[int, int]]] = {}
    for (nid, pins) in device_nodes:
        if nid == -1:
            continue
        for pin_label, net in pins.items():
            try:
                pid = handle.node_port_id_by_name(nid, pin_label)
            except KeyError:
                # Pin label not on the Rust-side node (e.g. a future
                # device-class extension Python knows about that the
                # Rust factory doesn't yet). Skip the binding so the
                # rest of the cell still wires up; the net's other
                # endpoints proceed as normal.
                continue
            nets.setdefault(net, []).append((nid, pid))

    # Emit one hyperedge per net. No port constraints declared at
    # this layer — Phase 2 semantic analysis is the home for
    # "vdd must be PowerOnly" sweeps.
    for net_name, endpoints in nets.items():
        if not endpoints:
            continue
        handle.connect(net_name, endpoints, None)

    return handle


# ─── Layer 2: parsed handle → hypergraph (v0.0.66) ─────────────


def _add_parsed_device_node(handle, dev):
    """Same per-device-class dispatch as Layer 1's `_add_device_node`,
    but consumes a `clay_parser.ParsedDevice` pyclass instead of a
    Python `clay.ir.Device` dataclass. Pin keys arriving here are
    already MOS-canonicalized by the Rust walker
    (`canonicalize_mos_pin` in `crates/clay_parser/src/lib.rs`).

    Returns the new Rust node id, or None when the device class is
    unmappable (caller skips and the rest of the cell still wires up).
    """
    cls = dev.device_class
    # `sizing` arrives as a dict[str, float]; pass through directly.
    sizing = dict(dev.sizing) if dev.sizing else None
    variant = dev.pdk_variant

    if cls == "nmos":
        # v0.0.94 R71 D: idempotent W/L/M canonicalization (matches
        # Layer 1 path). Parsed devices typically already use canonical
        # case (Rust walker emits "W" / "L" / "M") so this is a no-op
        # on the hot path; safety net for hand-built ParsedDevice
        # fixtures and any future grammar extension.
        return handle.add_mos_node(
            dev.instance_name, True, variant, _normalize_mos_sizing(sizing)
        )
    if cls == "pmos":
        return handle.add_mos_node(
            dev.instance_name, False, variant, _normalize_mos_sizing(sizing)
        )
    if cls == "cap":
        # v0.0.94: see Layer 1 `_add_device_node` cap branch.
        return handle.add_cap_node(
            dev.instance_name, variant, _normalize_cap_sizing(sizing)
        )
    if cls == "res":
        return handle.add_res_node(dev.instance_name, variant, sizing)
    if cls == "ind":
        return handle.add_ind_node(dev.instance_name, variant, sizing)
    if cls == "varactor":
        return handle.add_varactor_node(dev.instance_name, variant, sizing)
    return None


def _add_parsed_passive_node(handle, p):
    """Dispatch for `clay_parser.ParsedPassive` (cap / res / ind /
    varactor). Pin keys (`p` / `n` and `b` on varactor) are not
    canonicalized — the Rust walker passes them through unchanged."""
    cls = p.device_class
    sizing = dict(p.sizing) if p.sizing else None
    if cls == "cap":
        # v0.0.94: see Layer 1 `_add_device_node` cap branch.
        return handle.add_cap_node(
            p.instance_name, None, _normalize_cap_sizing(sizing)
        )
    if cls == "res":
        return handle.add_res_node(p.instance_name, None, sizing)
    if cls == "ind":
        return handle.add_ind_node(p.instance_name, None, sizing)
    if cls == "varactor":
        return handle.add_varactor_node(p.instance_name, None, sizing)
    return None


def parsed_to_hypergraph(parsed_handle, cell_name: str):
    """Translate a `clay_parser.ParsedClayHandle` directly to a Rust
    `clay_ir.CellGraphHandle` for the named cell.

    Returns the populated handle on success, or `None` when:
      * the `clay_ir` PyO3 extension is not importable (graceful
        fallback — same contract as Layer 1), or
      * the parsed handle doesn't contain a cell with that name.

    Algorithm (mirrors Layer 1 with the data source swapped):

      1. Construct an empty `CellGraphHandle` named after `cell_name`.
      2. Propagate `parsed_handle.pdk_id()` via `set_pdk_id` (file-
         scope `pdk 'X';` lives on the parsed file, not the cell).
      3. For each port in `parsed_handle.cell_ports(cell_name)`, emit
         an external port with role inferred from the name.
      4. For each device in `parsed_handle.cell_devices(cell_name)`,
         emit the matching Rust node and remember the
         `(node_id, pin_name → port_id)` index for the connection step.
      5. For each passive in `parsed_handle.cell_passives(cell_name)`,
         likewise (with its own node-add factory).
      6. For each instance in `parsed_handle.cell_instances(cell_name)`,
         emit a `cell_instance` node with `port_specs` derived from
         the `port_map` keys (sub-cell port names).
      7. Bucket every (device|passive|instance)-port → net binding into
         a `{net_name: [(node_id, port_id), ...]}` map; emit one
         hyperedge per non-empty bucket (no port constraints, matching
         Layer 1's contract).

    Coverage gaps relative to Layer 1 — for v0.0.66, the bridge
    does NOT expand:

      * `role_decls` (`pmos_pair` / `nmos_mirror` / `pmos_cs` and
        their nmos siblings). The Python parser expands these to
        concrete MOS Device entries inside `Cell.devices`; the
        Rust grammar surfaces them as `ParsedRoleDecl` items but
        the Rust walker doesn't run the expansion. Cells whose
        topology is role-only translate with zero devices on Layer 2.
      * `digital_decls` (`inv` / `nand2` / `dff` / `dff_async_clear`).
        The Python parser expands each to 2-24 concrete MOS devices;
        Layer 2 emits no nodes for them at this stripe. Cells whose
        topology is digital-only translate with zero devices.

    Both gaps will close when role/digital expansion lands on the
    Rust side. Until then, the bridge is best-suited to cells whose
    topology block contains explicit `device M : pmos { ... }` /
    `cap` / `res` / `ind` / `varactor` / `instance` declarations.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). `parsed_handle=None` still
    returns `None` because that is an argument-shape signal, not an
    extension-availability one.
    """
    _require_clay_ir()
    if parsed_handle is None:
        return None

    # Verify the cell exists on the parsed handle before touching the
    # Rust side. `cell_ports` returns None for unknown cells per the
    # v0.0.66 PyO3 contract; we use that as our existence probe.
    ports = parsed_handle.cell_ports(cell_name)
    if ports is None:
        return None

    handle = _clay_ir.CellGraphHandle.new_empty(cell_name)
    pdk = parsed_handle.pdk_id()
    if pdk:
        handle.set_pdk_id(pdk)

    # External ports.
    for port_name in ports:
        role, kind = _infer_external_port_role(port_name)
        handle.add_external_port(port_name, role, kind)

    # Track (node_id, pin_label → net_name) per device/passive/instance
    # so the connection pass can resolve each endpoint to a port_id
    # without re-walking the Rust graph.
    node_pin_bindings: list[tuple[int, dict[str, str]]] = []

    # Devices (MOS family).
    devices = parsed_handle.cell_devices(cell_name) or []
    for dev in devices:
        nid = _add_parsed_device_node(handle, dev)
        if nid is None:
            # Unmappable device class — track with sentinel -1 so the
            # bucketing pass can skip without altering the index.
            node_pin_bindings.append((-1, dict(dev.pins)))
            continue
        node_pin_bindings.append((nid, dict(dev.pins)))

    # Passives.
    passives = parsed_handle.cell_passives(cell_name) or []
    for p in passives:
        nid = _add_parsed_passive_node(handle, p)
        if nid is None:
            node_pin_bindings.append((-1, dict(p.pins)))
            continue
        node_pin_bindings.append((nid, dict(p.pins)))

    # Sub-cell instances. The Rust IR's `add_cell_instance_node` takes
    # a `port_specs` list of (name, role_str, kind_str) tuples. We
    # infer each sub-cell port's role from its NAME (same heuristic
    # used for external cell ports) because the parsed handle doesn't
    # carry per-instance port-role metadata.
    instances = parsed_handle.cell_instances(cell_name) or []
    for inst in instances:
        # Sort port labels so the port id assignment is deterministic
        # across runs (HashMap iteration order is not specified).
        port_labels = sorted(inst.port_map.keys())
        port_specs = []
        for label in port_labels:
            role, kind = _infer_external_port_role(label)
            port_specs.append((label, role, kind))
        nid = handle.add_cell_instance_node(
            inst.instance_name,
            inst.cell_path,
            port_specs,
            None,  # parsed handle doesn't carry sub-cell sizing kwargs
        )
        node_pin_bindings.append((nid, dict(inst.port_map)))

    # Bucket pins by net name and emit one hyperedge per net.
    nets: dict[str, list[tuple[int, int]]] = {}
    for nid, pins in node_pin_bindings:
        if nid == -1:
            continue
        for pin_label, net in pins.items():
            try:
                pid = handle.node_port_id_by_name(nid, pin_label)
            except KeyError:
                # Same defensive skip as Layer 1: pin label not on
                # the Rust-side node. Continue with the other endpoints.
                continue
            nets.setdefault(net, []).append((nid, pid))

    for net_name, endpoints in nets.items():
        if not endpoints:
            continue
        handle.connect(net_name, endpoints, None)

    return handle


# ─── v0.0.69 Phase 3 third slice: Adjoint Network gradient bridge ──


def compute_gradients(
    cell_or_handle,
    in_node: str,
    out_node: str,
    sizing_params,
    *,
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    metric: str = "gain_dc",
) -> Optional[dict[str, float]]:
    """Compute the small-signal gradient of a chosen metric w.r.t. each device's W.

    Returns a dict mapping `device_instance_name` to ∂(metric)/∂W,
    or None when the `clay_ir` PyO3 extension is not importable.

    Accepts either a `clay.ir.Cell` (the bridge runs `cell_to_hypergraph`
    on the fly) or a pre-built `CellGraphHandle`.

    ``sizing_params`` is an iterable of one of:
      * a string `instance_name` (e.g. "M1") — kind defaults to "W"
      * a tuple `(instance_name, kind)` — `kind ∈ {"W"}` today

    ``metric``:
      * ``"gain_dc"`` — ∂(V_out/V_in at DC)/∂W (v0.0.69 default).
      * ``"gbw"``     — ∂(unity-gain frequency, Hz)/∂W (v0.0.70).
      * ``"pm"``      — ∂(phase margin, degrees)/∂W (v0.0.70).

    Devices whose `instance_name` is not present in the cell get
    gradient 0.0 (caller can detect missing knobs by zip-checking
    against their input list).

    Honest scope:
      * Strong-inversion square-law. Sub-threshold devices report
        gradient 0.0 (gm = 0 at the OP).
      * "Local-OP" gradient — the DC OP is held fixed under the
        perturbation. An outer fixpoint that re-solves the OP per
        perturbation is left for v0.0.71+.
      * For ``gbw``/``pm`` the cell must have non-zero capacitive
        entries (any active MOS with Cox > 0) and a DC gain > 1.
        Cells that don't cross unity gain in the 1 Hz – 1 THz sweep
        window report gradient 0.0.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). The cell→handle conversion
    inherits that contract from `cell_to_hypergraph`; an explicitly
    `None` handle still short-circuits to `None` because that is a
    no-cell signal, not an extension-availability one.
    """
    _require_clay_ir()

    handle = cell_or_handle
    if isinstance(cell_or_handle, Cell):
        handle = cell_to_hypergraph(cell_or_handle)
        if handle is None:
            return None
    if handle is None:
        return None

    metric = (metric or "gain_dc").lower()
    if metric not in ("gain_dc", "gbw", "pm"):
        raise ValueError(
            f"metric must be one of 'gain_dc' | 'gbw' | 'pm', got {metric!r}"
        )

    # Normalise sizing_params to a list of (instance_name, kind).
    # v0.0.78 admits "Vbias" alongside W (the cur_src `value` knob);
    # the bridge keeps the v0.0.69 silent-coercion semantics for any
    # other unknown string so older callers don't crash.
    _ALLOWED_KINDS = ("W", "Vbias")
    norm_params: list[tuple[str, str]] = []
    for entry in sizing_params:
        if isinstance(entry, str):
            norm_params.append((entry, "W"))
        elif isinstance(entry, tuple) and len(entry) == 2:
            inst, kind = entry
            if kind not in _ALLOWED_KINDS:
                # Coerce unknown kinds to W so callers don't have to
                # branch on the Rust extension's coverage today.
                kind = "W"
            norm_params.append((str(inst), str(kind)))
        else:
            raise TypeError(
                f"sizing_params entry must be str or (str, str), "
                f"got {type(entry).__name__}: {entry!r}"
            )

    # Map instance_name → node_id by walking the handle. The
    # CellGraphHandle uses a monotonic id counter (graph.alloc_node_id),
    # so iterating 0..node_count is safe.
    name_to_nid: dict[str, int] = {}
    n_nodes = handle.node_count()
    for nid in range(n_nodes):
        try:
            inst_name = handle.node_instance_name(nid)
        except Exception:  # noqa: BLE001
            continue
        name_to_nid[inst_name] = nid

    device_ids: list[int] = []
    for inst, _kind in norm_params:
        nid = name_to_nid.get(inst)
        if nid is None:
            # Use a sentinel so the result list length stays aligned;
            # the Rust side returns 0.0 for unknown ids.
            device_ids.append(2**63 - 1)
        else:
            device_ids.append(nid)

    # v0.0.78: when ANY knob is Vbias-kind we route through the
    # cur_src-aware OP-resolve path so the cur_src `value` gradient is
    # non-zero. Mixed knob lists (W + Vbias) work on the same call —
    # the Vbias-aware path is byte-identical to the v0.0.75 path for
    # W/L/M/Cc on cells without cur_src, and reduces to a no-op on
    # cells that don't actually have any (proven by
    # tests/test_adjoint_vbias_knob.py::test_vbias_aware_path_equals_v075_path_on_no_cur_src_cell).
    has_vbias = any(kind == "Vbias" for _inst, kind in norm_params)
    if has_vbias:
        knobs = [(nid, kind) for nid, (_inst, kind) in zip(device_ids, norm_params)]
        if metric == "gain_dc":
            triples = handle.adjoint_gradient_op_resolve_vbias_aware(
                in_node, out_node, knobs, supply_vdd, pdk_id, 0.05
            )
        elif metric == "gbw":
            triples = handle.adjoint_gradient_gbw_op_resolve_vbias_aware(
                in_node, out_node, knobs, supply_vdd, pdk_id, 0.05
            )
        else:  # metric == "pm"
            # No dedicated PM Vbias method today — fall back to the
            # AC-magnitude path at a probe frequency near the dominant
            # pole. Callers wanting strict PM should use the v0.0.75
            # PM gradient with W knobs; Vbias's PM contribution flows
            # via the OP shift, which the magnitude proxy captures.
            triples = handle.adjoint_gradient_op_resolve_at_freq_vbias_aware(
                in_node, out_node, 1e6, knobs, supply_vdd, pdk_id, 0.05
            )
        raw_pairs = [(nid, val) for nid, _kind, val in triples]
    elif metric == "gain_dc":
        raw_pairs = handle.adjoint_gradient_w(
            in_node, out_node, device_ids, supply_vdd, pdk_id
        )
    elif metric == "gbw":
        raw_pairs = handle.adjoint_gradient_gbw_w(
            in_node, out_node, device_ids, supply_vdd, pdk_id
        )
    else:  # metric == "pm"
        raw_pairs = handle.adjoint_gradient_pm_w(
            in_node, out_node, device_ids, supply_vdd, pdk_id
        )
    result: dict[str, float] = {}
    for (inst, _kind), (_nid, grad) in zip(norm_params, raw_pairs):
        # Duplicate instance names: last write wins (matches dict
        # semantics).
        result[inst] = float(grad)
    return result


def compute_gbw(
    cell_or_handle,
    in_node: str,
    out_node: str,
    *,
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
) -> Optional[float]:
    """Return the unity-gain frequency (Hz) of the cell's small-signal
    transfer function from ``in_node`` to ``out_node``.

    Returns ``float('nan')`` when the cell does not cross unity gain
    in the 1 Hz – 1 THz sweep window.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). An explicitly `None`
    handle still short-circuits to `None`.
    """
    _require_clay_ir()
    handle = cell_or_handle
    if isinstance(cell_or_handle, Cell):
        handle = cell_to_hypergraph(cell_or_handle)
        if handle is None:
            return None
    if handle is None:
        return None
    return float(handle.compute_gbw(in_node, out_node, supply_vdd, pdk_id))


def compute_pm(
    cell_or_handle,
    in_node: str,
    out_node: str,
    *,
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
) -> Optional[float]:
    """Return the phase margin (degrees) of the cell at its unity-gain
    frequency. Returns ``float('nan')`` when GBW does not exist.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). An explicitly `None`
    handle still short-circuits to `None`.
    """
    _require_clay_ir()
    handle = cell_or_handle
    if isinstance(cell_or_handle, Cell):
        handle = cell_to_hypergraph(cell_or_handle)
        if handle is None:
            return None
    if handle is None:
        return None
    return float(handle.compute_pm(in_node, out_node, supply_vdd, pdk_id))


def gradient_on_functional_target(
    cell_or_handle,
    role_name: str,
    target_attr: str,
    in_node: str,
    out_node: str,
    *,
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    step_rel: float = 0.05,
    use_fd: bool = False,
) -> Optional[dict]:
    """v0.0.88: compute ∂(DC gain)/∂(functional target value) for one
    functional-role declaration.

    `role_name` is the role declaration's local handle — e.g. `"Mgf2"`
    for an `input_transconductor Mgf2 { gm_target = 200uA/V }` block.
    The bridge resolves it to a device-level NodeId by walking the
    `CellGraphHandle.node_instance_name` map; if no instance matches,
    returns a dict with `success=False`.

    `target_attr` is one of `"gm_target"`, `"gain_target"`,
    `"pole_freq_target"`, `"iref"`. Pass-through attributes (`gain` on
    feedforward_path; `zero_freq_target` on compensation_zero) return
    `success=False` with `reason="passthrough"`.

    Returns a dict with the following keys (None on extension-missing):
      * `gradient`   : float — ∂(DC gain)/∂(target) at the current OP.
                       Multiply by `∂loss/∂gain_dc` to get the loss
                       gradient when the spec is a function of gain_dc.
      * `success`    : bool  — True iff the chain-rule + adjoint paths
                       both succeeded.
      * `role_name`  : str   — echoed back for caller debugging.
      * `target_attr`: str   — echoed back.
      * `node_id`    : int   — the resolved NodeId; -1 if unresolved.
      * `method`     : str   — `"adjoint"` (default) or `"fd"` when
                       `use_fd=True`.

    `use_fd=True` switches to the finite-difference reference path for
    test cross-checking; the two should agree in sign and approximate
    magnitude when the lowering chain rule is valid.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). Soft-fail callers
    (`clay.invariants.augment_violation_with_gradient`,
    `clay.distance_to_sat.make_adjoint_fn_from_bridge`) catch the
    exception and surface the same `_gradient_error` extras key the
    old `None` return path produced — user-visible behaviour
    unchanged through them, but extensionless envs fail loudly
    instead of silently emitting zeroes.
    """
    _require_clay_ir()
    handle = cell_or_handle
    if isinstance(cell_or_handle, Cell):
        handle = cell_to_hypergraph(cell_or_handle)
        if handle is None:
            return None
    if handle is None:
        return None

    # Resolve role_name → NodeId by walking the handle. We do this in
    # Python because the Rust side doesn't expose a name→id lookup
    # (and the bridge already does the same walk in `compute_gradients`).
    name_to_nid: dict[str, int] = {}
    n_nodes = handle.node_count()
    for nid in range(n_nodes):
        try:
            inst_name = handle.node_instance_name(nid)
        except Exception:  # noqa: BLE001
            continue
        name_to_nid[inst_name] = nid
    node_id = name_to_nid.get(role_name, -1)
    if node_id < 0:
        return {
            "gradient": 0.0,
            "success": False,
            "role_name": role_name,
            "target_attr": target_attr,
            "node_id": -1,
            "method": "fd" if use_fd else "adjoint",
            "reason": "role_name_not_found",
        }
    # Check whether the requested target_attr has a lowering. Same list
    # the Rust side ships in `adjoint_supported_functional_targets`.
    supported = handle.adjoint_supported_functional_targets()
    if target_attr not in supported:
        return {
            "gradient": 0.0,
            "success": False,
            "role_name": role_name,
            "target_attr": target_attr,
            "node_id": node_id,
            "method": "fd" if use_fd else "adjoint",
            "reason": "passthrough",
        }
    if use_fd:
        grad, success = handle.finite_difference_gradient_on_functional_target(
            in_node, out_node, node_id, target_attr, supply_vdd, pdk_id, step_rel
        )
    else:
        grad, success = handle.adjoint_gradient_on_functional_target(
            in_node, out_node, node_id, target_attr, supply_vdd, pdk_id, step_rel
        )
    return {
        "gradient": float(grad),
        "success": bool(success),
        "role_name": role_name,
        "target_attr": target_attr,
        "node_id": int(node_id),
        "method": "fd" if use_fd else "adjoint",
        "reason": "ok" if success else "adjoint_failed",
    }


def compute_dc_gain(
    cell_or_handle,
    in_node: str,
    out_node: str,
    *,
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
) -> Optional[float]:
    """Return the small-signal DC voltage gain V_out/V_in at the OP.
    Same input contract as :func:`compute_gradients`.

    v0.1.07 R84 C: raises `RuntimeError` when the `clay_ir` extension
    is not importable (was: `return None`). An explicitly `None`
    handle still short-circuits to `None`.
    """
    _require_clay_ir()
    handle = cell_or_handle
    if isinstance(cell_or_handle, Cell):
        handle = cell_to_hypergraph(cell_or_handle)
        if handle is None:
            return None
    if handle is None:
        # Caller passed `None` (e.g. a builder stub that hasn't produced
        # a graph yet). Degrade to `None` rather than crashing.
        return None
    return float(
        handle.compute_dc_gain(in_node, out_node, supply_vdd, pdk_id)
    )


# ─── v0.0.73: Run real ngspice on a CellGraphHandle ────────────────


# Module-level regex compiled once. The `meas` echo lines we emit look
# like `name=1.234e-5` (no spaces) or ngspice's own `name = 1.234e-5`
# (spaced); the regex captures both forms with one alternation.
_MEAS_LINE_RE = re.compile(
    r"^\s*(\w+)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)


def _discover_ngspice_con() -> Optional[str]:
    """Locate the ngspice silent-batch binary.

    Returns the path to `ngspice_con` (Windows) or `ngspice` (POSIX)
    when one is on PATH or under the known Windows install dirs. On
    Windows we strictly prefer `ngspice_con.exe`; the GUI `ngspice.exe`
    pops a window on FATAL errors and blocks subprocess.run (see
    project memory `ngspice_con_for_silent`). Returns None if no
    silent binary is found — the caller (test or env) skips the e2e
    path gracefully.
    """
    is_win = platform.system() == "Windows"
    candidates = []
    if is_win:
        # Windows: ngspice_con only — refuse to fall back to ngspice.exe
        # so a misconfigured host doesn't quietly hang on a GUI pop.
        candidates.extend(["ngspice_con", "ngspice_con.exe"])
    else:
        candidates.append("ngspice")
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    # Windows install convention fallback.
    if is_win:
        for hard in (
            r"C:\Spice64\bin\ngspice_con.exe",
            r"C:\Program Files\Spice64\bin\ngspice_con.exe",
        ):
            if Path(hard).exists():
                return hard
    return None


def _resolve_pdk_bindings(pdk_id: Optional[str]) -> dict[str, str]:
    """Look up the PDK adapter's device_bindings map.

    Returns an empty dict when the pdk_id is None, unknown, or its
    adapter cannot be imported. The Rust spice_export emitter falls
    back to the sky130 default model names in that case (so the
    netlist is at least parseable). Callers that need a hard error
    on missing PDK should validate upstream.
    """
    if not pdk_id:
        return {}
    try:
        from .pdk import get_pdk
        adapter = get_pdk(pdk_id)
    except (KeyError, ImportError):
        return {}
    return dict(adapter.device_bindings)


def _resolve_corner_file(pdk_id: Optional[str], corner: str) -> Optional[Path]:
    """Locate the ngspice corner include file for the given PDK.

    Returns a Path even when the underlying file is missing — the
    caller decides whether a missing PDK corner is fatal (e2e test)
    or skip-worthy (CI without the PDK). Returns None when the PDK
    is unknown or the adapter cannot resolve a path.
    """
    if not pdk_id:
        return None
    try:
        from .pdk import get_pdk
        adapter = get_pdk(pdk_id)
    except (KeyError, ImportError):
        return None
    try:
        path = adapter.corner_file(corner)
    except (ValueError, RuntimeError):
        # ValueError: unknown corner; RuntimeError: PDK root unset (gf180).
        # Either way the caller decides whether to skip.
        return None
    return path


def _supply_label(pdk_id: Optional[str], supply_v: float) -> float:
    """Return the supply voltage to use in the testbench.

    If the caller passed a float, use it. Else fall back to the PDK's
    nominal vdd (1.8 for sky130, 3.3 for gf180mcu 3p3). Used by tests
    that build a hypergraph without explicit supply intent.
    """
    if supply_v is not None and supply_v > 0:
        return float(supply_v)
    if pdk_id:
        try:
            from .pdk import get_pdk
            return float(get_pdk(pdk_id).vdd_nom)
        except (KeyError, ImportError):
            pass
    return 1.8


# ─── v0.0.87 functional-role OP-quantity inject (R86 C: Rust-canonical)
#
# Three functional-role keywords produce device-level OP-quantity probes:
# `input_transconductor` (gm), `output_buffer` (gm+id), `tail_current_source`
# (id, emitted both as `id_<name>` and `i_<name>` so either resolution
# attempt lands). Other three v0.0.87 keywords (`feedforward_path` /
# `dominant_pole_cap` / `compensation_zero`) pass through silently —
# they're verified via AC measures the existing testbenches produce.
# R85 D ported the emit logic to Rust byte-equal; R86 C stripped the
# Python copy. The remaining Python surface is the dispatch table
# (`_OP_INJECTION_KINDS`, `_op_quantities_for_role`) used by the
# parse-log widener and the v92 measure-name helper.

_OP_INJECTION_KINDS: frozenset[str] = frozenset({
    "input_transconductor",
    "output_buffer",
    "tail_current_source",
})


def _op_quantities_for_role(
    role: ParsedFunctionalRole,
) -> list[tuple[str, str]]:
    """Per-role OP quantities to extract from the testbench.

    Returns a list of `(measure_name, ngspice_quantity_key)` pairs. The
    measure_name is what the resulting `gm_<name>` / `id_<name>` /
    `i_<name>` key in the parsed measure dict will look like; the
    quantity_key is the ngspice OP-parameter inside `@m<dev>[...]`.

    Pass-through roles (feedforward_path / dominant_pole_cap /
    compensation_zero) return an empty list — they have no device-level
    OP quantity to probe.
    """
    if role.kind == "input_transconductor":
        # Differential pair / single-device input stage. The invariant
        # runtime's first resolution attempt is `gm_<name>`, so that's
        # the canonical key. Pair with the raw `id_<name>` probe so a
        # downstream sizing audit can recover gm/Id if it wants to.
        return [(f"gm_{role.name}", "gm"), (f"id_{role.name}", "id")]
    if role.kind == "output_buffer":
        # 2nd-stage CS amp. The invariant runtime checks `gain_target`
        # via the linear-AC path (existing testbench measure), but a
        # device-level gm/id audit is still useful for the LLM's
        # sizing-suggestion logic — emit both so the OP-side record is
        # complete.
        return [(f"gm_{role.name}", "gm"), (f"id_{role.name}", "id")]
    if role.kind == "tail_current_source":
        # Tail / bias current sources. The invariant runtime's
        # `_check_tail_current_source` tries `i_<name>` first, then
        # `iref_<name>`, then global `iref` / `itail`. Emit both
        # `id_<name>` (the canonical OP quantity) AND `i_<name>` (the
        # invariant runtime's preferred key) so the resolution lands
        # on the first attempt.
        return [(f"id_{role.name}", "id"), (f"i_{role.name}", "id")]
    return []


# v0.1.10 R86 C: unconditional Rust dispatch for the inject helpers.
#
# R85 D landed `clay_ir::functional_role_emit::*` byte-equal to the
# v0.0.87 + v0.0.92 Python implementations and gated dispatch behind
# `CLAY_USE_RUST_INJECT_HELPERS`. R86 C flips the gate off: the env
# variable is gone, the Python implementation bodies have been stripped,
# and `_emit_functional_role_op_lines` / `_emit_v92_functional_role_lines`
# unconditionally delegate to the Rust port. `_inject_helpers_rust_enabled`
# is preserved as a thin capability predicate (True iff the Rust
# extension is loaded AND carries the R85 D bindings) so the R85 D
# byte-equal test suite continues to call it. The strip is the final
# acceptance for R79 D §6.2 condition #1 (inject layer ported) and the
# Tier B.1.a/b mass deletion the round-085 plan calls out.


def _inject_helpers_rust_enabled() -> bool:
    """True iff the Rust extension is loaded AND carries the R85 D
    inject-helper bindings.

    v0.1.10 R86 C: callers no longer have a Python fallback to fall
    through to — `_emit_functional_role_op_lines` and
    `_emit_v92_functional_role_lines` unconditionally delegate to the
    Rust port. This predicate stays as a capability probe used by the
    R85 D byte-equal test suite and by `_require_clay_ir()`-paired
    callers that want to surface a clearer diagnostic before invoking
    the helpers themselves.
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_emit_functional_role_op_lines_rust")
        and hasattr(_clay_ir, "_emit_v92_functional_role_lines_rust")
    )


def _roles_to_rust_payload(
    roles: Sequence[ParsedFunctionalRole],
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Marshal a sequence of `ParsedFunctionalRole` into the
    `(kind, name, [(attr_key, attr_value_str)])` tuple shape the Rust
    inject-helper bindings accept. Only string-valued attributes are
    surfaced because the inject helpers don't consume numeric attrs;
    non-string attributes are silently dropped at the boundary.

    Mirrors the marshalling pattern used by `try_rust_check_functional_role`
    elsewhere in this module (R82 B / R83 B / R84 B), keeping the
    PyO3-boundary shape consistent across delegates.
    """
    out: list[tuple[str, str, list[tuple[str, str]]]] = []
    for role in roles:
        str_attrs: list[tuple[str, str]] = []
        for k, v in role.attributes.items():
            if isinstance(v, str):
                str_attrs.append((k, v))
        out.append((role.kind, role.name, str_attrs))
    return out


def _emit_functional_role_op_lines(
    roles: Optional[Sequence[ParsedFunctionalRole]],
    *,
    subckt_instance: str = "xdut",
) -> list[str]:
    """Return ngspice control-deck lines that probe per-role OP
    quantities.

    Emits two pieces per quantity:

      * A literal `.print op @m<name>[<q>]` line in the deck.
      * A deterministic `let gm_<name> = @m.<subckt>.x<name>[<q>]`
        followed by `echo gm_<name>=$&gm_<name>` for log parsing.

    Args:
      roles: parsed functional-role declarations attached to a Cell.
        Roles whose `kind` is not in `_OP_INJECTION_KINDS` are silently
        skipped (pass-through for `feedforward_path` / `dominant_pole_cap`
        / `compensation_zero`).
      subckt_instance: the DUT subckt's instance name inside the
        testbench (`xdut` in every v0.0.87 builder).

    Returns:
      Flat list of `.control`-block lines (empty when `roles` is None
      or every role is a pass-through kind).

    v0.1.10 R86 C: mass-deletion strip. The Python implementation body
    (R69 / v0.0.87) has been removed; this function unconditionally
    delegates to `clay_ir::functional_role_emit::
    emit_functional_role_op_lines`, the byte-equal Rust port landed by
    R85 D. The `CLAY_USE_RUST_INJECT_HELPERS` env flag is also gone —
    Rust dispatch is the only path. Byte-equality across the Python
    reference and Rust port was asserted by R85 D before strip; the
    R85 D test suite (`tests/test_rust_inject_helpers_v123.py`) now
    verifies the delegate alone. Raises `RuntimeError` when the Rust
    extension is missing.
    """
    if not roles:
        return []
    _require_clay_ir()
    return list(
        _clay_ir._emit_functional_role_op_lines_rust(  # type: ignore[union-attr]
            _roles_to_rust_payload(roles),
            subckt_instance=subckt_instance,
        )
    )


def _functional_role_measure_names(
    roles: Optional[Sequence[ParsedFunctionalRole]],
) -> tuple[str, ...]:
    """Return the tuple of measure-dict keys the OP-injection emits.

    Used by `simulate_hypergraph` to widen the `wanted` list passed to
    `_parse_measure_log` so the dynamically-emitted per-device measures
    actually land in the returned dict.
    """
    if not roles:
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for role in roles:
        if role.kind not in _OP_INJECTION_KINDS:
            continue
        for meas_name, _qty in _op_quantities_for_role(role):
            if meas_name in seen:
                continue
            seen.add(meas_name)
            names.append(meas_name)
    # v0.0.92 third-batch additions land via `_v92_measure_names`.
    for name in _v92_measure_names(roles):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


# ─── v0.0.92 third-batch keyword inject (R86 C: Rust-canonical) ─────
#
# Six functional-role keywords with ngspice-bound check functions
# (`common_mode_feedback`, `sample_hold_switch`, `regulated_cascode`,
# `gain_boost_amp`, `output_class_AB`, `dynamic_bias`). The per-keyword
# emit contract (placeholder vs derived let/echo lines, push/pull
# device fallback rules, tran-window vs degenerate constant routing)
# is now Rust-canonical in `crates/clay_ir/src/functional_role_emit.rs`
# — R85 D ported it byte-equal; R86 C stripped the Python copy. The
# remaining Python surface in this module is the dispatch-table helpers
# (`_v92_measure_names_for_role`, `_v92_measure_names`) used by the
# parse-log widener; they have no Rust co-impl yet because the widener
# itself stays Python-side (it's called by `simulate_hypergraph`'s
# orchestration loop which marshals through `_parse_measure_log`).

_V92_INJECTION_KINDS: frozenset[str] = frozenset({
    "common_mode_feedback",
    "sample_hold_switch",
    "regulated_cascode",
    "gain_boost_amp",
    "output_class_AB",
    "dynamic_bias",
})


def _v92_measure_names_for_role(
    role: ParsedFunctionalRole,
) -> list[str]:
    """Per-role measure keys the v0.0.92 inject helper echoes.

    Returns the canonical measure-dict keys (`cmrr_<name>`, `droop_<name>`,
    `ro_boost_<name>`, `gain_<name>`, `iq_<name>`, `i_<name>` plus
    derived helper keys `gm_<name>` / `ro_<name>` / `gain_dm_<name>` /
    `gain_cm_<name>` where applicable) so `_parse_measure_log` keeps
    them when widening the `wanted` list.
    """
    if role.kind == "common_mode_feedback":
        return [
            f"cmrr_{role.name}",
            f"gain_dm_{role.name}",
            f"gain_cm_{role.name}",
        ]
    if role.kind == "sample_hold_switch":
        return [f"droop_{role.name}"]
    if role.kind == "regulated_cascode":
        return [
            f"ro_boost_{role.name}",
            f"gm_{role.name}",
            f"ro_{role.name}",
        ]
    if role.kind == "gain_boost_amp":
        return [f"gain_{role.name}"]
    if role.kind == "output_class_AB":
        return [f"iq_{role.name}"]
    if role.kind == "dynamic_bias":
        return [f"i_{role.name}"]
    return []


def _v92_measure_names(
    roles: Optional[Sequence[ParsedFunctionalRole]],
) -> tuple[str, ...]:
    """Flat tuple of every measure-key any v0.0.92 inject helper would
    echo into the log. Mirrors `_functional_role_measure_names` for the
    third-batch keywords."""
    if not roles:
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for role in roles:
        if role.kind not in _V92_INJECTION_KINDS:
            continue
        for nm in _v92_measure_names_for_role(role):
            if nm in seen:
                continue
            seen.add(nm)
            names.append(nm)
    return tuple(names)


def _emit_v92_functional_role_lines(
    roles: Optional[Sequence[ParsedFunctionalRole]],
    *,
    subckt_instance: str = "xdut",
    has_tran_window: bool = False,
    has_gain_dc: bool = False,
) -> list[str]:
    """Emit per-keyword `.print op` + `let` + `echo` lines for every
    declared v0.0.92 third-batch functional role.

    Args:
      roles: parsed functional-role declarations on the cell. Roles
        whose `kind` is not in `_V92_INJECTION_KINDS` are silently
        skipped (the v0.0.87 helper handles those).
      subckt_instance: DUT subckt instance name (`xdut` in every
        builder).
      has_tran_window: True iff the testbench builder calling this
        helper has already emitted `v_max`, `v_min`, `t_max`, `t_min`
        scalars (sample_hold_switch path).
      has_gain_dc: True iff `gain_dc` is in the deck's let-namespace
        (common_mode_feedback / gain_boost_amp AC-proxy path).

    Returns:
      Flat list of `.control`-block lines. Empty when `roles` is None
      or every role is non-v92.

    v0.1.10 R86 C: mass-deletion strip. The Python implementation body
    (R69 / v0.0.92) has been removed; this function unconditionally
    delegates to `clay_ir::functional_role_emit::
    emit_v92_functional_role_lines`, the byte-equal Rust port landed by
    R85 D. The `CLAY_USE_RUST_INJECT_HELPERS` env flag is also gone —
    Rust dispatch is the only path. Byte-equality across the 6 v0.0.92
    keyword × 4 flag-combination matrix was asserted by R85 D before
    strip; the R85 D test suite verifies the delegate alone after R86 C.
    Raises `RuntimeError` when the Rust extension is missing.
    """
    if not roles:
        return []
    _require_clay_ir()
    return list(
        _clay_ir._emit_v92_functional_role_lines_rust(  # type: ignore[union-attr]
            _roles_to_rust_payload(roles),
            subckt_instance=subckt_instance,
            has_tran_window=has_tran_window,
            has_gain_dc=has_gain_dc,
        )
    )


def _has_rust_testbench_emit() -> bool:
    """True iff the v0.0.98 R75 B Rust delegates are available.

    The single-ended testbench builders below delegate to the Rust
    emitter when this function returns True AND `functional_roles` is
    None / empty. Non-empty roles still flow through the Python testbench
    shell, but R86 C's strip ensures the v0.0.87 / v0.0.92 inject lines
    inside that shell come from the Rust port — the Python builder is
    a thin orchestrator only.
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_build_ac_testbench_rust")
    )


def _has_rust_diff_testbench_emit() -> bool:
    """True iff the v0.1.05 R81 D Rust delegates for the 5-port
    differential testbench builders are available.

    The 5-port differential builders (`_build_ac_diff_testbench` /
    `_build_tran_diff_testbench`) delegate to the Rust emitter when
    this function returns True AND `functional_roles` is None / empty.
    Non-empty roles still fall back to the Python builder so the
    v0.0.87 / v0.0.92 inject layers stay live. The DC-sweep diff and
    fully-differential builders remain Python-only (R82+ third slice).
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_build_ac_diff_testbench_rust")
        and hasattr(_clay_ir, "_build_tran_diff_testbench_rust")
    )


def _has_rust_fully_diff_testbench_emit() -> bool:
    """True iff the v0.1.06 R82 C Rust delegates for the 6-port
    fully-differential testbench builders are available.

    The 6-port fully-differential AC + transient builders
    (`_build_ac_fully_diff_testbench` / `_build_tran_fully_diff_testbench`)
    delegate to the Rust emitter when this function returns True AND
    `functional_roles` is None / empty. Non-empty roles still fall back
    to the Python builder so the v0.0.87 / v0.0.92 inject layers stay
    live.
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_build_ac_fully_diff_testbench_rust")
        and hasattr(_clay_ir, "_build_tran_fully_diff_testbench_rust")
    )


def _has_rust_dc_sweep_fully_diff_testbench_emit() -> bool:
    """True iff the v0.1.07 R83 C Rust delegate for the 6-port FD
    DC-sweep testbench builder is available.

    Pair with `_build_dc_sweep_fully_diff_testbench` — delegates when
    True AND `functional_roles` is empty; non-empty still flows through
    pure Python so the v0.0.87 / v0.0.92 inject layers stay live.
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_build_dc_sweep_fully_diff_testbench_rust")
    )


def _has_rust_dc_sweep_diff_testbench_emit() -> bool:
    """True iff the v0.1.16 R93 D Rust delegate for the 5-port diff
    DC-sweep testbench builder is available.

    v0.1.18 R96 D: pure capability probe — `_build_dc_sweep_diff_testbench`
    now always delegates to the Rust port (the post-strip posture matches
    R88 B fully-diff DC sweep). Monkey-patching this predicate no longer
    forces a Python fallback path because the fallback is gone; the probe
    survives so older test fixtures that still call it don't break, and
    so the R85 D `_inject_helpers_rust_enabled()` pattern stays consistent
    across the inject-helper + testbench-emit cohort.
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_build_dc_sweep_diff_testbench_rust")
    )


def _has_rust_cmrr_cm_testbench_emit() -> bool:
    """True iff the v0.1.07 R83 C Rust delegate for the CM-drive leg
    of the 2-run CMRR sweep is available.

    Pair with `_build_ac_fully_diff_cmrr_cm_testbench` — delegates
    unconditionally (the Python builder has no `functional_roles`
    parameter, so there is no Python-only inject layer to preserve).
    """
    return (
        _HAS_CLAY_IR
        and _clay_ir is not None
        and hasattr(_clay_ir, "_build_ac_fully_diff_cmrr_cm_testbench_rust")
    )


def _functional_roles_are_empty(
    roles: Optional[Sequence[ParsedFunctionalRole]],
) -> bool:
    """True iff `roles` would cause the inject layers to emit zero lines.

    The Rust delegate can only safely take over when no per-role
    visibility / let / echo lines need to be injected. The simplest
    correct predicate is "the sequence is None or empty" — strictly
    fewer cases than "the sequence has zero matching kinds", but it
    keeps the wrapper trivially conservative (no kind enumeration on
    the hot path) and the byte-for-byte equivalence assertion strong.
    """
    return roles is None or len(roles) == 0


# ─── v0.0.74: single-ended + extended testbench builders ──────────
#
# v0.0.62 R-split: the `_build_*_testbench` builder cluster moved to the
# focused module `clay.ir_testbench`. They are re-exported here at the
# spot they were defined so every `from clay._ir_bridge import _build_*`
# import site (and the internal `simulate_hypergraph` /
# `simulate_fully_diff_cmrr_2run` callers below) keeps resolving them in
# this module's namespace unchanged. The builders reference the Rust-
# extension state (`_HAS_CLAY_IR` / `_clay_ir`) and the
# `_require_clay_ir` / `_roles_to_rust_payload` helpers through the
# `clay._ir_bridge` module object at call time, so monkeypatching
# `clay._ir_bridge._HAS_CLAY_IR` / `._clay_ir` in test fixtures still
# drives the hard-error / dispatch branches exactly as before.
from .ir_testbench import (  # noqa: E402  # re-export
    _build_ac_diff_testbench,
    _build_ac_fully_diff_cmrr_cm_testbench,
    _build_ac_fully_diff_testbench,
    _build_ac_pm_testbench,
    _build_ac_testbench,
    _build_dc_sweep_diff_testbench,
    _build_dc_sweep_diff_testbench_pure_python,
    _build_dc_sweep_fully_diff_testbench,
    _build_dc_sweep_testbench,
    _build_tran_diff_testbench,
    _build_tran_fully_diff_testbench,
    _build_tran_testbench,
)


# Supported testbench kinds and the measure names each one returns.
# Keeps `simulate_hypergraph` dispatch + `_parse_measure_log` wanted-list
# in lockstep — adding a new kind here is the single source of truth.
#
# The ``tran`` entry now carries the v0.0.74 slew measure set
# (``v_max`` / ``v_min`` / ``t_max`` / ``t_min`` / ``slew_rate``) plus
# the v0.0.78 ``settling_time`` step-response measure (seconds from
# input-pulse rise to last entrance into the ±0.5 % band around the
# output's late-window average). ``v_final`` is the late-window output
# average the band is computed from — surfaced for callers that want
# to sanity-check the settling-band reference. Adding measures here
# never removes existing keys (which would break sister-test
# assertions); it widens the dict the simulator returns.
_TESTBENCH_MEASURES: dict[str, tuple[str, ...]] = {
    # v0.0.81: the AC measure set now includes ``cmfb_phase_margin`` so
    # the fully-differential (6-port) testbench can surface the CMFB
    # loop's phase margin alongside the differential gain_dc / gbw.
    # On single-ended (4-port) and 5-port differential cells the key
    # lands as NaN via the standard ``_parse_measure_log`` passthrough —
    # those cells have no CMFB loop to probe. On 6-port fully-diff
    # cells with the default ``cmfb_ea_bw=inf`` (ideal EA) the key
    # still lands NaN because there is no unity-gain crossover to
    # measure; opt in to a finite ``cmfb_ea_bw`` to get a real number.
    #
    # v0.0.82: ``cmfb_cm_offset`` (V) added to AC, tran, and dc_sweep
    # measure sets. This is the quantitative CM-regulation error
    # ``(V(outp) + V(outn)) / 2 − vcm_target`` that finite EA gain
    # leaves on the output common mode. On the AC builder it lands at
    # DC (f_start). On tran it is the time-average over the settling
    # window. On dc_sweep it is the average over the supply sweep.
    # Single-ended / 5-port diff paths land NaN via the standard
    # ``_parse_measure_log`` passthrough — those cells have no CM
    # output to probe.
    "ac": ("gain_dc", "gbw", "cmfb_phase_margin", "cmfb_cm_offset"),
    "ac_pm": ("gain_dc", "gbw", "phase_deg", "pm"),
    "tran": (
        "v_max",
        "v_min",
        "t_max",
        "t_min",
        "slew_rate",
        "v_final",
        "settling_time",
        # v0.0.82: FD CM-regulation offset, time-averaged over the
        # settling window. NaN on single-ended / 5-port paths.
        "cmfb_cm_offset",
    ),
    # v0.0.79: dc_sweep extends from v0.0.74's single ``vout_at_mid``
    # measure (input-sweep midpoint, still emitted on the single-ended
    # path) to a unified PSRR + CMRR DC measure set across all three
    # port-shape families.  v0.0.80 layers worst-case PSRR / CMRR over
    # the supply and CM sweep windows on top of the linearised
    # endpoint-secant numbers — both sets are emitted so backward
    # compatibility is preserved.
    #
    #   * Input-sweep result (single-ended only — CMFB regulation on
    #     diff/FD paths makes the input-sweep bias derived from Vcm,
    #     not a swept input source):
    #       - ``vout_at_mid``
    #   * Supply-sweep PSRR (all three port-shapes):
    #       - ``v_out_at_vdd_min``  / ``v_out_at_vdd_max``
    #       - ``psrr_dc``  = 20*log10(|ΔV(out)/ΔV(vdd)|) (endpoint-secant)
    #     v0.0.80 worst-case additions (all three port-shapes):
    #       - ``v_out_window_max`` / ``v_out_window_min``
    #       - ``vdd_at_v_out_max`` / ``vdd_at_v_out_min``  (`max_at` /
    #         `min_at` Vdd values where the window extrema are reached)
    #       - ``psrr_dc_worst`` = 20*log10(|max-min over window|/Δvdd)
    #   * Common-mode-sweep CMRR (diff + fully-diff paths only):
    #       - ``v_out_at_vcm_min``  / ``v_out_at_vcm_max``
    #       - ``cmrr_dc``  = 20*log10(|ΔV(out)/ΔV(vcm)|) (endpoint-secant)
    #     v0.0.80 worst-case CMRR additions:
    #       - ``v_out_cm_window_max`` / ``v_out_cm_window_min``
    #       - ``vcm_at_v_out_max`` / ``vcm_at_v_out_min``
    #       - ``cmrr_dc_worst`` = 20*log10(|max-min over window|/Δvcm)
    #
    # Measures the active port-shape doesn't emit (e.g. vout_at_mid on
    # an FD cell, or cmrr_dc on a single-ended cell, or the CM
    # worst-case keys on a single-ended cell) land as NaN via
    # `_parse_measure_log`'s default — same NaN-passthrough contract
    # the other testbench kinds already rely on.
    "dc_sweep": (
        "vout_at_mid",
        "v_out_at_vdd_min",
        "v_out_at_vdd_max",
        "psrr_dc",
        "v_out_at_vcm_min",
        "v_out_at_vcm_max",
        "cmrr_dc",
        # v0.0.80 worst-case PSRR (supply sweep window).
        "v_out_window_max",
        "v_out_window_min",
        "vdd_at_v_out_max",
        "vdd_at_v_out_min",
        "psrr_dc_worst",
        # v0.0.80 worst-case CMRR (CM sweep window — diff/FD only).
        "v_out_cm_window_max",
        "v_out_cm_window_min",
        "vcm_at_v_out_max",
        "vcm_at_v_out_min",
        "cmrr_dc_worst",
        # v0.0.82: FD CM-regulation offset, averaged over the Vdd
        # sweep window (ideal CM regulation is supply-independent so
        # the average is a clean scalar). NaN on single-ended /
        # 5-port paths.
        "cmfb_cm_offset",
    ),
}


# ─── v0.0.75: port-shape auto-detection + differential testbench ───
#
# v0.0.74's `_build_ac_testbench` hard-codes the DUT instance line as
# `xDUT in out vdd vss <CellName>` — a 4-port single-ended subckt. Any
# cell whose external ports differ from `{in, out, vdd, vss}` blows up
# at ngspice load with "subckt port mismatch", which is exactly the
# bug that bit `TwoStageMillerPMOS` (ports `inn / inp / out / vdd / vss`).
#
# v0.0.75 closes the gap with a port-shape detector that inspects the
# Rust handle's external-port names (recovered from the deterministic
# `.subckt <name> <p1> <p2> ...` header that `export_spice_subckt`
# always emits) and a sibling differential testbench builder that drives
# `inn` / `inp` from a single AC source through a pair of VCVS gain
# blocks. Dispatch in `simulate_hypergraph` picks the right template
# from the detector's shape verdict.

# Canonical single-ended port set: every classic 4-port subckt the v0.0.74
# CS-amp tests build. The detector accepts any superset of this — i.e.
# 5-port single-ended cells with an explicit `vbias` pin still route to
# the single-ended builder; only when the set crosses into clearly-
# differential territory (both `inn` and `inp`) do we switch templates.
_SINGLE_ENDED_REQUIRED = frozenset({"in", "out", "vdd", "vss"})

# Canonical differential input port pair. Cells with both names present
# AND a single `out` port land on the 5-port differential template; cells
# with `outn`+`outp` on top of that are fully-differential (6-port) and
# explicitly deferred for v0.0.75 — the template needs a CMFB loop, which
# is a larger surface than this slice carries.
_DIFF_INPUT_PAIR = ("inn", "inp")


def _extract_external_port_names(handle) -> list[str]:
    """Recover the cell's external port names from a CellGraphHandle.

    The PyO3 surface exposes `external_port_count()` but not the names
    themselves (v0.0.75 honest scope: a `external_port_names()`
    accessor lives in the v0.0.76 Rust slice). Until then we lift the
    names out of the deterministic `.subckt <cell> <p1> <p2> ...`
    header that `export_spice_subckt(...)` always emits — same Rust
    code path the simulator wraps, so the answer is exactly what the
    netlist will see.

    Returns an empty list when the handle has no external ports (legal
    subckt shape, just unusual) or when the header line cannot be
    located (defensive — the export contract guarantees it's there).
    """
    # Empty bindings → the export still emits the .subckt header. The
    # device lines may reference bare device_class strings like `nmos`
    # if no binding maps that class to a model name, but that has no
    # bearing on the header line we're parsing.
    subckt_text = handle.export_spice_subckt({})
    for line in subckt_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(".subckt"):
            tokens = stripped.split()
            # `.subckt <name> <p1> <p2> ...`. The first two tokens are
            # the directive + cell name; the rest are ports.
            if len(tokens) <= 2:
                return []
            return tokens[2:]
    return []


def _detect_port_shape(handle) -> dict:
    """Classify the external port layout of a CellGraphHandle.

    Returns a dict with keys:

      * ``shape`` — one of:

          - ``"single_ended"`` — `{in, out, vdd, vss}` (or any 4+ port
            superset that still has a single `in`, e.g. an explicit
            `vbias` pin). Routes to v0.0.74's `_build_ac_testbench`.
          - ``"differential"`` — `{inn, inp, out, vdd, vss}` (5-port
            single-ended-output differential pair). Routes to v0.0.75's
            new `_build_ac_diff_testbench`.
          - ``"fully_differential"`` — `{inn, inp, outn, outp, vdd, vss}`
            (6-port). Routes to v0.0.77's `_build_ac_fully_diff_testbench`
            on `testbench_kind="ac"`. Other kinds (ac_pm / tran /
            dc_sweep) on an FD cell still raise NotImplementedError.
          - ``"unknown"`` — none of the above patterns matched. Could
            be a degenerate cell (no supply ports), an exotic shape
            (e.g. multi-input adder), or a malformed handle. The caller
            decides whether to error or fall back.

      * ``input_ports`` — list of port names the detector classified as
        signal inputs. For differential this is `["inn", "inp"]`; for
        single-ended this is `["in"]`. Ordering is canonical so callers
        can pass the list straight into a testbench builder without
        re-sorting.
      * ``output_ports`` — list of signal-output port names. Always
        `["out"]` today; the fully-differential branch returns
        `["outn", "outp"]` so callers can branch on len().
      * ``supply_ports`` — list of `{vdd, vss}` (or whichever subset
        the cell declares). Empty when no supply rails are declared
        (legal but unusual; the testbench then provides no .include).

    The detector is name-heuristic only — no role / kind inspection.
    A cell can in principle declare a port named `inn` that semantically
    serves as a current bias; this is rare in practice (the stdlib
    opamps + stage templates all follow the naming convention) and a
    future v0.0.76 slice will tighten the detector to consult the
    PortRole metadata once the PyO3 surface exposes it.
    """
    port_names = _extract_external_port_names(handle)
    port_set = set(port_names)

    # Detect supply rails first — same for every shape.
    supply_ports = [p for p in ("vdd", "vss") if p in port_set]

    # Fully-differential check (6-port: inn + inp + outn + outp).
    has_diff_in = all(p in port_set for p in _DIFF_INPUT_PAIR)
    has_diff_out = "outn" in port_set and "outp" in port_set
    if has_diff_in and has_diff_out:
        return {
            "shape": "fully_differential",
            "input_ports": list(_DIFF_INPUT_PAIR),
            "output_ports": ["outn", "outp"],
            "supply_ports": supply_ports,
        }

    # Single-ended-output differential (5-port: inn + inp + out).
    if has_diff_in and "out" in port_set:
        return {
            "shape": "differential",
            "input_ports": list(_DIFF_INPUT_PAIR),
            "output_ports": ["out"],
            "supply_ports": supply_ports,
        }

    # Single-ended (4+ port: in + out + vdd + vss [+ vbias/...]).
    if _SINGLE_ENDED_REQUIRED.issubset(port_set):
        return {
            "shape": "single_ended",
            "input_ports": ["in"],
            "output_ports": ["out"],
            "supply_ports": supply_ports,
        }

    return {
        "shape": "unknown",
        "input_ports": [],
        "output_ports": [],
        "supply_ports": supply_ports,
    }


# ─── v0.0.75: 5-port differential AC testbench ────────────────────
# `_build_ac_diff_testbench` re-exported from `clay.ir_testbench` (see
# the re-export block above `_TESTBENCH_MEASURES`).


# ─── v0.0.77: fully-differential AC testbench with CMFB loop ─────
#
# A 6-port fully-differential opamp `{inn, inp, outn, outp, vdd, vss}` only
# controls the **differential** mode of the output pair — the differential
# transconductance `Gm_d` runs between (inp-inn) and (outp-outn), but the
# **common-mode** output level (`(outp+outn)/2`) is set by whichever common-
# mode feedback (CMFB) loop the caller wires in. Without that loop the CM
# output drifts to either rail and the diff pair drops out of saturation;
# the .meas then collapses to NaN even when the differential pole/zero
# structure is fine.
#
# v0.0.77 closes that gap with the textbook resistive-sense + high-gain
# error-amp CMFB driver:
#
#   * Two 1 MΩ sense resistors average `outn` + `outp` into `vcm_sensed`.
#   * A VCVS `E_cmfb` compares `vcm_sensed` against `vcm_target` (the bias
#     rail at supply/2) and drives a high-gain `vcmfb` node. The DUT does
#     NOT have a dedicated CMFB input today, so the regulator's job here
#     is "if CM output drifts, pull it back via the sense-resistor
#     leakage path." For most stdlib FD opamps this is enough to land the
#     CM level near vdd/2 because the diff pair's source-degenerated
#     tail still has finite CM gain via the bias network.
#   * Differential drive identical to v0.0.75: Vcm + Vsig + E1/E2 split
#     produces ±0.5*Vsig at inp/inn, centred on vcm_target.
#   * `E_diff` synthesises a `diff_out` node = (outp - outn) so the
#     downstream gain_dc/gbw .meas can keep using `vdb(...)` on a single
#     node — same parser code path as the v0.0.74 single-ended template
#     and the v0.0.75 5-port differential template.
#
# Honest deferred scope (NOT in v0.0.77):
#   * CMFB stability (the 1Meg/E_cmfb loop is functional but not phase-
#     compensated — fine for DC operating point + AC magnitude, will not
#     match silicon CMFB pole/zero placement).
#   * Finite-gain error amp (E_cmfb gain is 1e6 = ideal). A real CMFB
#     EA has its own pole that interacts with the main amp.
#   * CM-mode .meas (cm_gain, psrr_cm, cmrr) — v0.0.77 only ships the
#     differential `gain_dc` / `gbw` measure set; CMRR-style metrics
#     need a sibling testbench that sweeps `vcm` instead of `sig`.
#
# Cells with a dedicated CMFB input port (e.g. `vcmfb`) could in
# principle route `E_cmfb`'s output straight in; the detector does NOT
# look for that port today, so this template assumes the DUT regulates
# its CM via the bias network only.

# `_build_ac_fully_diff_testbench` re-exported from `clay.ir_testbench`
# (see the re-export block above `_TESTBENCH_MEASURES`).


# ─── v0.0.94 Round 70 Agent A: 2-run CMRR for fully-diff opamps ────
#
# v0.0.93 Round 69 Agent A's `_emit_v92_functional_role_lines` lands
# `cmrr_<name>` for the `common_mode_feedback` keyword, but it uses a
# placeholder constant (80 dB) — the helper docstring acknowledges
# this and notes "real 2-run resolution is the follow-up". v0.0.94
# closes that gap: a sibling testbench that drives the cell in pure
# common-mode (inp = inn = Vin_cm) so the AC sweep's |Vout_cm / Vin_cm|
# is the common-mode gain. The diff-mode gain comes from the standard
# `_build_ac_fully_diff_testbench` run (already in the test-bench
# family). CMRR = 20·log10(gain_dm / gain_cm).
#
# Why this is a separate builder rather than a flag on the existing one:
#
#   * Existing builder's bit-exact contract — v0.0.77/.78/.79/.80/.81/.82
#     /.87/.92 tests check specific lines (E1/E2 ±0.5 split, Vcm node
#     at supply/2, Vsig=`dc 0 ac 1`, etc.). Adding a new mode would risk
#     drifting any of those assertions.
#   * Per-call dispatch — a caller that wants CMRR runs two simulations
#     (one diff, one CM) and computes the ratio in Python. Each leg
#     uses a single, focused testbench string. The orchestrator
#     `simulate_fully_diff_cmrr_2run` below does both legs in one call.
#
# Driving convention for the CM-drive leg:
#
#   * Vcm sets `vcm_target` at supply/2 plus a small AC perturbation
#     (default 1 mV — same `cmfb_probe_ac` shape that v0.0.81 already
#     plumbs for the EA phase-margin probe).
#   * E1/E2 are REMOVED. inp = inn = vcm_target directly (a wire each).
#     This is the canonical "tie both inputs together and bump the
#     common-mode level by Vin_cm" configuration the textbook CMRR
#     definition asks for.
#   * The CMFB regulator path is identical to the diff testbench so
#     the operating point lands in the same place — only the input
#     drive differs.
#   * Output measured: `(outp + outn) / 2` is the common-mode output;
#     ngspice computes this with `let v_outcm = (v(outp) + v(outn))/2`.
#     The CM gain at f_start (DC) is then |v_outcm / vcm_target|,
#     extracted with `meas ac` lines.

# `_build_ac_fully_diff_cmrr_cm_testbench` re-exported from
# `clay.ir_testbench` (see the re-export block above
# `_TESTBENCH_MEASURES`).


def simulate_fully_diff_cmrr_2run(
    handle,
    *,
    supply_v: float = 1.8,
    corner: str = "tt",
    cload: float = 1e-12,
    cmfb_ea_gain: float = 1e6,
    cmfb_ea_bw: float = float("inf"),
    cm_drive_ac: float = 1.0,
    work_dir: Optional[str] = None,
    ngspice_bin: Optional[str] = None,
    timeout_s: float = 60,
) -> dict[str, float]:
    """Run the 2-run CMRR sweep for a fully-differential opamp handle.

    Two ngspice invocations:

      1. Diff-mode drive (the standard `_build_ac_fully_diff_testbench`):
         measures |gain_dm| in dB at DC.
      2. CM-mode drive (the v0.0.94 `_build_ac_fully_diff_cmrr_cm_testbench`):
         measures |gain_cm| in dB at DC.

    CMRR = gain_dm (dB) − gain_cm (dB) — directly in dB because both
    measurements use the `vdb()` convention.

    Args:
      handle: a 6-port fully-differential `clay_ir.CellGraphHandle`.
              Raises `RuntimeError` for None, `ValueError` for non-FD
              cells (single-ended / 5-port diff don't have a CMRR via
              this construction — use `dc_sweep` CMRR for diff cells).

    Returns:
      A dict with keys:
        * `gain_dm`     — diff-mode |Vout_dm/Vin_dm| in dB
        * `gain_cm`     — common-mode |Vout_cm/Vin_cm| in dB
        * `cmrr`        — gain_dm − gain_cm in dB
        * `gain_dm_lin` — linear V/V version of gain_dm
        * `gain_cm_lin` — linear V/V version of gain_cm

      All NaN when either leg fails to converge (e.g. the cell drifts
      out of saturation under pure CM drive).
    """
    if handle is None:
        raise RuntimeError(
            "simulate_fully_diff_cmrr_2run: handle is None — call "
            "cell_to_hypergraph(...) or parsed_to_hypergraph(...) first."
        )
    shape_info = _detect_port_shape(handle)
    if shape_info["shape"] != "fully_differential":
        raise ValueError(
            f"simulate_fully_diff_cmrr_2run: handle has shape "
            f"{shape_info['shape']!r}, not fully_differential. CMRR via "
            f"2-run AC sweep is only defined for 6-port FD cells. "
            f"For 5-port diff use simulate_hypergraph(testbench_kind='dc_sweep')."
        )
    pdk_id = handle.pdk_id()
    bindings = _resolve_pdk_bindings(pdk_id)
    corner_file = _resolve_corner_file(pdk_id, corner)
    subckt_text = handle.export_spice_subckt(bindings)
    cell_name = handle.cell_name()
    supply = _supply_label(pdk_id, supply_v)

    # Diff leg — re-use the standard builder so its assertions stay
    # bit-exact across the entire test suite.
    diff_netlist = _build_ac_fully_diff_testbench(
        subckt_text,
        cell_name,
        corner_file=corner_file,
        supply_v=supply,
        cload=cload,
        cmfb_ea_gain=cmfb_ea_gain,
        cmfb_ea_bw=cmfb_ea_bw,
    )
    # CM leg — v0.0.94's pure-CM-drive sibling.
    cm_netlist = _build_ac_fully_diff_cmrr_cm_testbench(
        subckt_text,
        cell_name,
        corner_file=corner_file,
        supply_v=supply,
        cload=cload,
        cmfb_ea_gain=cmfb_ea_gain,
        cmfb_ea_bw=cmfb_ea_bw,
        cm_drive_ac=cm_drive_ac,
    )

    if ngspice_bin is None:
        ngspice_bin = _discover_ngspice_con()
    if ngspice_bin is None:
        raise RuntimeError(
            "simulate_fully_diff_cmrr_2run: ngspice binary not found. "
            "On Windows install ngspice_con.exe (the silent batch "
            "variant); on POSIX install ngspice."
        )

    cleanup_tempdir: Optional[tempfile.TemporaryDirectory] = None
    if work_dir is None:
        cleanup_tempdir = tempfile.TemporaryDirectory(prefix="clay_cmrr_2run_")
        work_path = Path(cleanup_tempdir.name)
    else:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

    def _run_leg(netlist_text: str, leg_label: str) -> dict[str, float]:
        netlist_path = work_path / f"{cell_name}_{leg_label}.cir"
        log_path = work_path / f"{cell_name}_{leg_label}.log"
        netlist_path.write_text(netlist_text, encoding="utf-8")
        try:
            subprocess.run(
                [str(ngspice_bin), "-b", "-o", str(log_path), str(netlist_path)],
                cwd=str(work_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"simulate_fully_diff_cmrr_2run: ngspice timed out on "
                f"the {leg_label} leg after {timeout_s} s; netlist at "
                f"{netlist_path}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"simulate_fully_diff_cmrr_2run: ngspice binary not "
                f"found at {ngspice_bin!r}."
            ) from exc
        if not log_path.exists():
            raise RuntimeError(
                f"simulate_fully_diff_cmrr_2run: ngspice produced no "
                f"log on the {leg_label} leg; netlist at {netlist_path}"
            )
        # Each leg has its own wanted-list.
        if leg_label == "diff":
            wanted = ("gain_dc",)
        else:
            wanted = ("gain_cm",)
        return _parse_measure_log(log_path, wanted)

    try:
        diff_result = _run_leg(diff_netlist, "diff")
        cm_result = _run_leg(cm_netlist, "cm")
    finally:
        if cleanup_tempdir is not None:
            try:
                cleanup_tempdir.cleanup()
            except (OSError, PermissionError):
                pass

    gain_dm_db = diff_result.get("gain_dc", float("nan"))
    gain_cm_db = cm_result.get("gain_cm", float("nan"))
    # CMRR = |gain_dm/gain_cm| in dB. Both legs use vdb() so the
    # dB-domain subtraction IS the CMRR. NaN-propagates naturally when
    # either leg failed to converge.
    if math.isfinite(gain_dm_db) and math.isfinite(gain_cm_db):
        cmrr_db = gain_dm_db - gain_cm_db
    else:
        cmrr_db = float("nan")
    if math.isfinite(gain_dm_db):
        gain_dm_lin = 10.0 ** (gain_dm_db / 20.0)
    else:
        gain_dm_lin = float("nan")
    if math.isfinite(gain_cm_db):
        gain_cm_lin = 10.0 ** (gain_cm_db / 20.0)
    else:
        gain_cm_lin = float("nan")
    return {
        "gain_dm": gain_dm_db,
        "gain_cm": gain_cm_db,
        "cmrr": cmrr_db,
        "gain_dm_lin": gain_dm_lin,
        "gain_cm_lin": gain_cm_lin,
    }


# ─── v0.0.78: differential + fully-differential transient builders ─
#
# The v0.0.74 `_build_tran_testbench` hard-codes a 4-port single-ended
# DUT line and a single-source VPULSE on `input_node`. A 5-port diff
# cell (`{inn, inp, out, vdd, vss}`) or a 6-port FD cell
# (`{inn, inp, outn, outp, vdd, vss}`) drops into ngspice with a port-
# count mismatch the moment the dispatch routes it that way — same
# failure mode that bit the v0.0.75 / v0.0.77 AC slices.
#
# v0.0.78 closes the gap by mirroring the AC builder family:
#
#   * `_build_tran_diff_testbench` — 5-port: Vcm + Vsig (now VPULSE,
#     not AC) + E1/E2 split into ±0.5 around vcm, measured on V(out).
#   * `_build_tran_fully_diff_testbench` — 6-port: same diff drive
#     plus the CMFB sense + E_cmfb regulator from v0.0.77, plus an
#     E_diff that synthesises `diff_out = outp - outn` so the same
#     v0.0.74 .meas blocks measure a single node.
#
# Driving convention: the pulse swing is the same ±50 mV that the
# single-ended template uses by default — small enough to keep the
# diff pair in saturation even at the swing extremes, large enough
# for ngspice's max/min .meas resolution to land.
#
# Settling-time / slew-rate measure set is identical to the single-
# ended builder so `_parse_measure_log` and `_TESTBENCH_MEASURES`
# stay unchanged. The diff builder measures on `V(out)` directly;
# the FD builder measures on the synthesised `V(diff_out)` so the
# .meas line shape is invariant across all three port-shape branches.

# `_build_tran_diff_testbench` + `_build_tran_fully_diff_testbench`
# re-exported from `clay.ir_testbench` (see the re-export block above
# `_TESTBENCH_MEASURES`).


# ─── v0.0.79: differential + fully-differential DC-sweep builders ──
#
# v0.0.74 shipped the single-ended `_build_dc_sweep_testbench`; v0.0.75
# / v0.0.77 / v0.0.78 left the diff and fully-diff `dc_sweep` paths
# raising `NotImplementedError`. v0.0.79 closes the gap with two new
# builders that mirror the v0.0.78 tran-builder family:
#
#   * `_build_dc_sweep_diff_testbench` — 5-port `{inn, inp, out, vdd,
#     vss}`. Two sequential DC analyses share the .control block:
#       1. Supply sweep (Vdd from supply*(1-frac) → supply*(1+frac)).
#          Both inputs sit at vcm; output's response to ΔVdd gives
#          `psrr_dc`.
#       2. Common-mode sweep (Vcm from supply*(0.5-cm_frac) → supply*
#          (0.5+cm_frac)). E1/E2 carry the bias to inp/inn together
#          (the diff drive component is zero); output's response to
#          ΔVcm gives `cmrr_dc`. CMRR is the supply rejection's diff-
#          input cousin — supply-noise rejection is to differential
#          amps what common-mode rejection is.
#
#   * `_build_dc_sweep_fully_diff_testbench` — 6-port `{inn, inp, outn,
#     outp, vdd, vss}`. Same two-analysis structure but measured on
#     the synthesised `diff_out = outp - outn` node. Reuses the v0.0.77
#     / v0.0.78 CMFB regulator (Rcm1/Rcm2 + E_cmfb) so the CM operating
#     point lands cleanly during both sub-sweeps.
#
# Honest deferred scope:
#
#   * The CM-sweep is a small CM excursion (±100 mV by default) around
#     vcm = supply/2. Cells with extreme CM input range (e.g. rail-to-
#     rail topologies) might benefit from a wider sweep; the knob is
#     exposed via `vcm_sweep_frac` for callers.
#   * PSRR-DC and CMRR-DC are signed-positive (magnitudes wrapped in
#     `abs(...)` before the log). A cell whose output tracks the
#     supply 1:1 lands `psrr_dc ≈ 0` dB; well-rejected supplies land
#     negative dB. Same convention on the single-ended builder, so
#     numbers are comparable across port shapes.
#   * The diff builder's input-sweep (`vout_at_mid`) is NOT carried
#     over — the diff cell's input bias is set by Vcm via the E1/E2
#     split, not by sweeping `Vinput` directly. `vout_at_mid` lands
#     as NaN on the diff and fully-diff dispatch paths.


# `_build_dc_sweep_diff_testbench`,
# `_build_dc_sweep_diff_testbench_pure_python`, and
# `_build_dc_sweep_fully_diff_testbench` re-exported from
# `clay.ir_testbench` (see the re-export block above
# `_TESTBENCH_MEASURES`).


def simulate_hypergraph(
    handle,
    *,
    testbench_kind: str = "ac",
    input_node: str = "in",
    output_node: str = "out",
    supply_v: float = 1.8,
    corner: str = "tt",
    cload: float = 1e-12,
    work_dir: Optional[Path] = None,
    ngspice_bin: Optional[str] = None,
    timeout_s: float = 60.0,
    cmfb_ea_gain: float = 1e6,
    cmfb_ea_bw: float = float("inf"),
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> dict[str, float]:
    """Run real ngspice on a CellGraphHandle and return measured values.

    Closes the v0.0.72 → v0.0.73 loop: the Rust hypergraph's
    `.export_spice_subckt(bindings)` emits a self-contained subckt,
    this function wraps it in a small testbench (AC sweep by default),
    spawns the silent ngspice binary, and parses the .meas log.

    Args:
      handle: a `clay_ir.CellGraphHandle`. Its `pdk_id()` selects the
        PDK adapter for `device_bindings` and the corner include file.
      testbench_kind: one of:

          * ``"ac"`` (default, v0.0.73) — AC sweep, returns
            ``gain_dc`` / ``gbw``.
          * ``"ac_pm"`` (v0.0.74) — AC sweep + phase-margin .meas,
            returns ``gain_dc`` / ``gbw`` / ``phase_deg`` / ``pm``.
          * ``"tran"`` (v0.0.74; v0.0.78 adds settling-time + diff +
            fully-diff dispatch) — transient pulse step response.
            Returns ``v_max`` / ``v_min`` / ``t_max`` / ``t_min`` /
            ``slew_rate`` (V/s) plus ``v_final`` (V, late-window
            output asymptote) and ``settling_time`` (s, time from
            input rise to last entrance into ±0.5 % band around
            ``v_final``). Port-shape detector routes single-ended,
            differential (5-port), and fully-differential (6-port)
            cells to their respective tran builders.
          * ``"dc_sweep"`` (v0.0.74; v0.0.79 adds PSRR + CMRR + diff
            + fully-diff dispatch) — DC operating-point sweep. Returns
            ``vout_at_mid`` (single-ended input sweep) plus the v0.0.79
            PSRR-DC + CMRR-DC measure set: ``v_out_at_vdd_min`` /
            ``v_out_at_vdd_max`` / ``psrr_dc`` from a supply sweep, and
            ``v_out_at_vcm_min`` / ``v_out_at_vcm_max`` / ``cmrr_dc``
            from a common-mode sweep (diff + fully-diff only).

        Unsupported kinds raise ``ValueError`` listing the supported
        options.
      input_node, output_node: net names on the subckt the testbench
        drives / measures. Default ("in", "out") matches the
        stage-template convention.
      supply_v: VDD voltage in volts. The input is biased at supply_v/2.
      corner: PDK corner name (tt/ss/ff/sf/fs). Default "tt".
      cload: load cap on the output node (F). Default 1 pF.
      work_dir: where to drop the netlist + log files. If None, a
        temp dir is created and cleaned up before return.
      ngspice_bin: explicit path to the silent ngspice binary. If None,
        falls back to PATH lookup (`_discover_ngspice_con`).
      timeout_s: subprocess timeout. Default 60 s.

    Returns:
      A dict mapping measure name → value. Keys depend on
      ``testbench_kind`` — see ``_TESTBENCH_MEASURES`` at the top of
      this module. Values are NaN when the .meas line failed to
      evaluate. Empty dict when the simulator could not be launched.

    Raises:
      RuntimeError: if `handle` is None, or for ngspice launch
        failures (binary missing, sim timeout, etc.) with a clear
        diagnostic so the test layer's `_ngspice_installed()` helper
        can branch on availability vs. real errors.
      ValueError: if `testbench_kind` is not in the supported set —
        a programmer error that should surface as type-shaped rather
        than runtime-shaped.
    """
    if handle is None:
        raise RuntimeError(
            "simulate_hypergraph: handle is None — call "
            "cell_to_hypergraph(...) or parsed_to_hypergraph(...) "
            "first."
        )
    if testbench_kind not in _TESTBENCH_MEASURES:
        supported = sorted(_TESTBENCH_MEASURES.keys())
        raise ValueError(
            f"simulate_hypergraph: testbench_kind={testbench_kind!r} "
            f"not supported. Supported kinds: {supported}."
        )

    # 1. Resolve the PDK adapter for device bindings + corner file.
    pdk_id = handle.pdk_id()
    bindings = _resolve_pdk_bindings(pdk_id)
    corner_file = _resolve_corner_file(pdk_id, corner)

    # 2. Get the Rust-side subckt text.
    subckt_text = handle.export_spice_subckt(bindings)

    # 3. Wrap in a testbench. Dispatch on testbench_kind selects the
    #    appropriate builder — all single-ended builders share the same
    #    (subckt_text, cell_name, corner_file, input_node,
    #    output_node, supply_v, cload) prefix, differing in their
    #    analysis-specific extras (kept on builder defaults).
    cell_name = handle.cell_name()
    supply = _supply_label(pdk_id, supply_v)

    # v0.0.75 port-shape auto-detection. The detector recovers the cell's
    # external port names from the deterministic `.subckt` header in the
    # Rust-emitted text, so we don't need to plumb the port list down
    # from the caller. Cells with `inn`/`inp`/`out`/`vdd`/`vss` get the
    # new differential AC builder; everything else falls back to v0.0.74's
    # single-ended builder (which still hard-codes `xDUT in out vdd vss`).
    shape_info = _detect_port_shape(handle)
    shape = shape_info["shape"]
    if shape == "fully_differential" and testbench_kind == "ac":
        # v0.0.77: 6-port `outn`/`outp` topology now flows through the
        # CMFB-loop AC builder. Driver template wraps the DUT with a
        # resistive sense + high-gain VCVS regulator and synthesises
        # `diff_out = outp - outn` so the standard {gain_dc, gbw} .meas
        # set keeps working unchanged. CM-mode metrics (cm_gain, cmrr)
        # are deferred — they need a sibling testbench that sweeps vcm.
        # v0.0.81: callers can opt into a finite-gain CMFB EA via
        # `cmfb_ea_gain` / `cmfb_ea_bw`; defaults preserve the v0.0.77
        # ideal-EA contract bit-exactly.
        netlist = _build_ac_fully_diff_testbench(
            subckt_text,
            cell_name,
            corner_file=corner_file,
            supply_v=supply,
            cload=cload,
            cmfb_ea_gain=cmfb_ea_gain,
            cmfb_ea_bw=cmfb_ea_bw,
            functional_roles=functional_roles,
        )
    elif shape == "fully_differential" and testbench_kind == "tran":
        # v0.0.78: 6-port FD transient builder. Same CMFB regulator
        # topology as the v0.0.77 AC fully-diff builder, with the AC
        # source swapped for a VPULSE so the .meas block can extract
        # slew_rate + settling_time on the synthesised diff_out node.
        # CM-mode settling is not measured here — would need a sibling
        # .meas on `(outp+outn)/2`.
        netlist = _build_tran_fully_diff_testbench(
            subckt_text,
            cell_name,
            corner_file=corner_file,
            supply_v=supply,
            cload=cload,
            cmfb_ea_gain=cmfb_ea_gain,
            cmfb_ea_bw=cmfb_ea_bw,
            functional_roles=functional_roles,
        )
    elif shape == "fully_differential" and testbench_kind == "dc_sweep":
        # v0.0.79: 6-port FD DC-sweep builder. Same CMFB regulator
        # topology as the v0.0.77 AC fully-diff + v0.0.78 tran fully-
        # diff builders, with the AC/VPULSE source swapped for DC 0 so
        # the cell sees a pure CM input during both PSRR + CMRR sub-
        # sweeps.
        netlist = _build_dc_sweep_fully_diff_testbench(
            subckt_text,
            cell_name,
            corner_file=corner_file,
            supply_v=supply,
            cload=cload,
            cmfb_ea_gain=cmfb_ea_gain,
            cmfb_ea_bw=cmfb_ea_bw,
            functional_roles=functional_roles,
        )
    elif shape == "fully_differential":
        # v0.0.77 / v0.0.78 / v0.0.79 ship the AC, tran, and DC-sweep
        # fully-differential testbenches. ac_pm on an FD cell still
        # has no CMFB-aware template today — that's the remaining gap,
        # deferred until a phase-compensated CMFB EA + CM-mode .meas
        # land. Surface a typed error rather than emit a netlist that
        # would silently deliver wrong numbers.
        raise NotImplementedError(
            f"simulate_hypergraph: cell {cell_name!r} is fully-differential "
            f"(6-port) but testbench_kind={testbench_kind!r} has no CMFB-"
            f"aware template today. v0.0.77 covers AC; v0.0.78 adds tran; "
            f"v0.0.79 adds dc_sweep; ac_pm fully-differential is deferred."
        )
    elif shape == "differential" and testbench_kind == "ac":
        # Differential cells need the VCVS-based AC injection template
        # (Vcm + Vsig + E1/E2 split). v0.0.75 ships the AC builder;
        # v0.0.78 adds the matching tran builder; ac_pm / dc_sweep
        # diff templates still hard-code the single-ended `xDUT in
        # out vdd vss` line and are deferred.
        netlist = _build_ac_diff_testbench(
            subckt_text,
            cell_name,
            corner_file=corner_file,
            supply_v=supply,
            cload=cload,
            functional_roles=functional_roles,
        )
    elif shape == "differential" and testbench_kind == "tran":
        # v0.0.78: 5-port differential transient builder. Same drive
        # convention as the AC diff template (Vcm + E1/E2 ±0.5 split)
        # but the AC source is swapped for VPULSE so the .meas block
        # captures slew_rate + settling_time on V(out).
        netlist = _build_tran_diff_testbench(
            subckt_text,
            cell_name,
            corner_file=corner_file,
            supply_v=supply,
            cload=cload,
            functional_roles=functional_roles,
        )
    elif shape == "differential" and testbench_kind == "dc_sweep":
        # v0.0.79: 5-port differential DC-sweep builder. Same Vcm +
        # E1/E2 ±0.5 split as the v0.0.75 AC diff + v0.0.78 tran diff
        # builders, with the AC/VPULSE source swapped for DC 0.
        # Sub-sweeps Vdd (PSRR) and Vcm (CMRR) sequentially.
        netlist = _build_dc_sweep_diff_testbench(
            subckt_text,
            cell_name,
            corner_file=corner_file,
            supply_v=supply,
            cload=cload,
            functional_roles=functional_roles,
        )
    elif shape == "differential":
        # Differential cell + non-AC / non-tran / non-dc_sweep testbench
        # kind. Today that's only `ac_pm`. Surface a typed error rather
        # than emit a netlist that ngspice would reject for port-count
        # mismatch on the hard-coded single-ended xDUT line.
        raise NotImplementedError(
            f"simulate_hypergraph: cell {cell_name!r} is differential "
            f"(5-port) but testbench_kind={testbench_kind!r} has no diff "
            f"template today. v0.0.75 covers AC; v0.0.78 adds tran; "
            f"v0.0.79 adds dc_sweep; ac_pm differential is deferred."
        )
    else:
        # Single-ended (or unknown — fall back to the v0.0.74 builder,
        # which will surface a port-mismatch error at ngspice load time
        # if the shape is genuinely incompatible). This preserves the
        # v0.0.74 22-test green status: every CS-amp test exercises a
        # canonical 4-port shape and lands on this branch unchanged.
        builder_kwargs = dict(
            corner_file=corner_file,
            input_node=input_node,
            output_node=output_node,
            supply_v=supply,
            cload=cload,
            functional_roles=functional_roles,
        )
        if testbench_kind == "ac":
            netlist = _build_ac_testbench(
                subckt_text, cell_name, **builder_kwargs
            )
        elif testbench_kind == "ac_pm":
            netlist = _build_ac_pm_testbench(
                subckt_text, cell_name, **builder_kwargs
            )
        elif testbench_kind == "tran":
            netlist = _build_tran_testbench(
                subckt_text, cell_name, **builder_kwargs
            )
        else:  # testbench_kind == "dc_sweep"
            netlist = _build_dc_sweep_testbench(
                subckt_text, cell_name, **builder_kwargs
            )

    # 4. Locate ngspice_con.
    if ngspice_bin is None:
        ngspice_bin = _discover_ngspice_con()
    if ngspice_bin is None:
        raise RuntimeError(
            "simulate_hypergraph: ngspice binary not found. On "
            "Windows install ngspice_con.exe (the silent batch "
            "variant); on POSIX install ngspice. See "
            "clay.config.ClayConfig.auto() for the resolution order."
        )

    # 5. Write the netlist + run ngspice + parse the log.
    cleanup_tempdir: Optional[tempfile.TemporaryDirectory] = None
    if work_dir is None:
        cleanup_tempdir = tempfile.TemporaryDirectory(prefix="clay_sim_")
        work_path = Path(cleanup_tempdir.name)
    else:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

    try:
        netlist_path = work_path / f"{cell_name}.cir"
        log_path = work_path / f"{cell_name}.log"
        netlist_path.write_text(netlist, encoding="utf-8")
        try:
            subprocess.run(
                [
                    str(ngspice_bin),
                    "-b",
                    "-o",
                    str(log_path),
                    str(netlist_path),
                ],
                cwd=str(work_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"simulate_hypergraph: ngspice timed out after "
                f"{timeout_s} s on cell {cell_name!r}; netlist at "
                f"{netlist_path}"
            ) from exc
        except FileNotFoundError as exc:
            # The explicit ngspice_bin path was provided but does not
            # exist on disk. Surface a clear error so callers can
            # branch on "no simulator" without parsing the OS error.
            raise RuntimeError(
                f"simulate_hypergraph: ngspice binary not found at "
                f"{ngspice_bin!r}. On Windows install ngspice_con.exe; "
                f"on POSIX install ngspice."
            ) from exc
        if not log_path.exists():
            raise RuntimeError(
                f"simulate_hypergraph: ngspice produced no log file "
                f"({log_path}); netlist at {netlist_path}"
            )
        # v0.0.87: dynamically widen the wanted-list with the per-role
        # OP-quantity measure names emitted by `_emit_functional_role_op_lines`.
        # The injected `echo gm_<name>=...` / `echo id_<name>=...` lines
        # land in the log; without including them in `wanted`,
        # `_parse_measure_log` would discard them as unrequested noise.
        wanted = (
            _TESTBENCH_MEASURES[testbench_kind]
            + _functional_role_measure_names(functional_roles)
        )
        return _parse_measure_log(log_path, wanted)
    finally:
        if cleanup_tempdir is not None:
            # On Windows, ngspice sometimes keeps a brief handle open
            # on the log file after returncode 0. Suppress cleanup
            # errors so a measurable result is never lost to a tmpdir
            # cleanup race.
            try:
                cleanup_tempdir.cleanup()
            except (OSError, PermissionError):
                pass


def _parse_measure_log_pure_python(
    log_path: Path, wanted: tuple[str, ...]
) -> dict[str, float]:
    """Pure-Python implementation of `_parse_measure_log`.

    Kept byte-for-byte identical to the v0.0.96 implementation as the
    fallback path. Used when:
      * The `clay_ir` PyO3 extension is unavailable (CI/test envs that
        haven't built Rust).
      * The Rust delegate is built but doesn't expose
        `_parse_measure_log_rust` (older v0.0.96- pyd shipped alongside
        a freshly-pulled clay/_ir_bridge.py — additive ABI: Python ships
        the Rust call, Rust falls back to Python on miss).

    Behaviour:
      First match wins for each name (ngspice prints the .meas result
      line BEFORE the echo, so we always get the ngspice value when
      it converged; the echo line is a safety net for older ngspice
      builds whose .meas blocks log differently). Missing names land
      in the output dict as NaN so the caller can distinguish "ngspice
      ran but .meas failed" (NaN) from "ngspice didn't run at all"
      (empty dict / RuntimeError).
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {n: float("nan") for n in wanted}
    for m in _MEAS_LINE_RE.finditer(text):
        name = m.group(1)
        if name in out and math.isnan(out[name]):
            try:
                out[name] = float(m.group(2))
            except ValueError:
                continue
    return out


def _parse_measure_log(
    log_path: Path, wanted: tuple[str, ...]
) -> dict[str, float]:
    """Parse the ngspice batch-mode log for ``name = value`` lines.

    Accepts both ngspice's own `name = 1.234e-5` block-formatted
    output and the `name=value` echo lines the testbench emits. The
    first match wins for each name (ngspice prints the .meas result
    line BEFORE the echo, so we always get the ngspice value when
    it converged; the echo line is a safety net for older ngspice
    builds whose .meas blocks log differently).

    Missing names land in the output dict as NaN so the caller can
    distinguish "ngspice ran but .meas failed" (NaN) from "ngspice
    didn't run at all" (empty dict / RuntimeError).

    v0.0.97 R74 C: delegates to the Rust regex parser
    `clay_ir._parse_measure_log_rust` when available. Output is byte-
    for-byte identical to the pure-Python implementation
    (`_parse_measure_log_pure_python`, kept as a fallback for envs
    without a fresh Rust extension). The Rust path is a hand-rolled
    line scanner — same regex semantics as Python `_MEAS_LINE_RE`,
    no new Cargo deps.
    """
    if _HAS_CLAY_IR and hasattr(_clay_ir, "_parse_measure_log_rust"):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # PyO3 expects a list, not a tuple — the binding signature is
        # `wanted: Vec<String>`. The Rust side returns a dict[str,float]
        # with NaN for missing names, mirroring the Python contract.
        return _clay_ir._parse_measure_log_rust(text, list(wanted))
    return _parse_measure_log_pure_python(log_path, wanted)
