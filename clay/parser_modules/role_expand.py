"""
Role-decl expansion (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
Each role declaration ("pmos_pair NAME { ... }" etc.) becomes N concrete
Device IR entries with auto-named local nets prefixed by the role's
local name (`<NAME>_tail`, `<NAME>_d1`, `<NAME>_d2`, etc.).

R76 A Rust delegation surface preserved byte-exact — the
`_try_rust_pair_delegation` wrapper still routes through
`semantic_role_decl::expand_pair_role_from_fields` when
`CLAY_USE_RUST_PARSER=1` is set (kept as a test-facing strangler-fig
surface; the production `_expand_pair` path no longer consults it).
Tests pin the wrapper-level equivalence:
  * `tests/test_rust_role_decl_expansion_v100.py`

v0.1.18 R95 D mass-deletion strip (mirror + cs):
  R94 F shipped the byte-equal Rust ports for mirror + cs role expansion
  (`semantic_role_decl::expand_mirror_role_from_fields` /
  `expand_cs_role_from_fields`) and wired solo entry points
  (`try_rust_expand_mirror_role_solo` / `try_rust_expand_cs_role_solo`).
  53 byte-equivalence + boundary tests landed green pre-strip.

  R95 D follows the R86 C precedent (`_ir_bridge.py` inject-helpers strip,
  v0.1.10): the Python implementation bodies for `_expand_mirror` and
  `_expand_cs` are removed, and both functions unconditionally delegate
  to the Rust port. The `CLAY_USE_RUST_PARSER` env var is gone from the
  mirror + cs paths — Rust dispatch is the only path. Raises RuntimeError
  when the Rust extension is missing.

v0.1.21 R98 D mass-deletion strip (pair):
  Closes the last un-stripped lean-role channel. The R76 A pair channel
  carried the Python expansion body as a fallback under the strangler-fig
  pattern; R76 A byte-equality has been pinned for two iterations now
  (24 R76 A tests + R94 F regression + R95 D + R97 D Strip 2). R98 D
  strips the `_expand_pair` Python body, dispatching unconditionally to
  `_rust.try_rust_expand_pair_role_solo` via the R95 D-style direct
  bridge call. The `CLAY_USE_RUST_PARSER` env-var gate is gone from this
  path — Rust dispatch is the only path. Raises RuntimeError when the
  Rust extension is missing.

  The R76 A `_try_rust_pair_delegation` wrapper is preserved for test
  compatibility (R76 A test corpus + R97 D Strip 2 `pair_wrapper_preserved`
  test pin against it). It is no longer consulted by `_expand_pair` in
  production — callers wanting env-var-gated semantics must call the
  wrapper directly.
"""

from __future__ import annotations

from ..ir import Device, RoleDecl
from ._constants import _VALID_ROLE_KINDS


# v0.1.18 R95 D: capability probe for the mirror + cs Rust delegate.
#
# R86 C precedent (`_ir_bridge._require_clay_ir`): after the Python body
# was stripped, callers can no longer fall through to a Python
# implementation. The capability check moves from "is env var set" (R94 F
# strangler-fig gating) to "is the extension loaded" (R86 C hard-error
# gating). The env var gate is gone from the mirror + cs paths — Rust
# is the only path.

_RUST_EXT_REQUIRED_MSG_ROLE_EXPAND = (
    "clay_parser Rust extension required. Build with "
    "`maturin develop -m crates/clay_parser/Cargo.toml --release`. "
    "v0.1.18 R95 D stripped the Python `_expand_mirror` and "
    "`_expand_cs` bodies, and v0.1.21 R98 D stripped the Python "
    "`_expand_pair` body — Rust dispatch is the only path for all "
    "three lean-role kinds."
)


def _require_clay_parser_for_role_strip() -> None:
    """Raise `RuntimeError` when the `clay_parser` Rust extension is not
    importable.

    v0.1.18 R95 D — Tier B mass-deletion: the R94 F mirror + cs solo
    delegation surface has byte-equal coverage on all known fixtures, so
    the Python `_expand_mirror` / `_expand_cs` bodies are now gone.
    Callers hit this probe at the top of each stripped function; the
    error surfaces cleanly to user code instead of falling through to a
    deleted Python implementation.

    v0.1.21 R98 D — extends the probe to cover the pair channel too: the
    R76 A `try_rust_expand_pair_role_solo` binding is now required since
    the `_expand_pair` Python body is stripped (same R95 D pattern as
    mirror + cs).

    Mirror of `_ir_bridge._require_clay_ir` (R84 C / R86 C pattern).
    """
    from .. import _rust_bridge

    if not _rust_bridge._HAS_RUST_EXTENSION or _rust_bridge._rust is None:
        raise RuntimeError(_RUST_EXT_REQUIRED_MSG_ROLE_EXPAND)
    # Defensive: the R94 F bindings might be missing on a stale .pyd that
    # pre-dates R94 F. Surface that explicitly rather than letting the
    # delegate call raise AttributeError.
    if not hasattr(_rust_bridge._rust, "try_rust_expand_mirror_role_solo"):
        raise RuntimeError(
            _RUST_EXT_REQUIRED_MSG_ROLE_EXPAND
            + " (Detected: extension loaded but missing R94 F "
            "`try_rust_expand_mirror_role_solo` binding — likely a stale "
            ".pyd. Rebuild via maturin.)"
        )
    if not hasattr(_rust_bridge._rust, "try_rust_expand_cs_role_solo"):
        raise RuntimeError(
            _RUST_EXT_REQUIRED_MSG_ROLE_EXPAND
            + " (Detected: extension loaded but missing R94 F "
            "`try_rust_expand_cs_role_solo` binding — likely a stale "
            ".pyd. Rebuild via maturin.)"
        )
    # v0.1.21 R98 D: the pair binding was added by R76 A, but defend
    # against a .pyd that pre-dates R76 A or is missing the symbol for
    # any reason. The strip needs this binding because `_expand_pair`
    # no longer has a Python fallback body.
    if not hasattr(_rust_bridge._rust, "try_rust_expand_pair_role_solo"):
        raise RuntimeError(
            _RUST_EXT_REQUIRED_MSG_ROLE_EXPAND
            + " (Detected: extension loaded but missing R76 A "
            "`try_rust_expand_pair_role_solo` binding — likely a stale "
            ".pyd. Rebuild via maturin.)"
        )


# v0.0.31 (G3 closeout): role-decl expansion.
#
# Each role declaration ("pmos_pair NAME { ... }" etc.) becomes N
# concrete Device IR entries with auto-named local nets prefixed by the
# role's local name (`<NAME>_tail`, `<NAME>_d1`, `<NAME>_d2`, etc.).
# Single-net pin overrides on the role replace the corresponding auto-
# named pin. Two-net pair overrides (`gates = (a, b);`, `drains = (a, b);`)
# replace pin-pairs on pair roles.
#
# The expansion table is small enough to keep inline — six kinds — and
# the auto-naming convention is consistent so future kinds slot in
# without changing surrounding code.


def _try_rust_pick_default_pair_gates(
    port_names: list[str],
    port_roles: dict[str, "str | None"],
    *,
    cell_name: str,
    role_name: str,
) -> "tuple[str, str] | None":
    """v0.1.20 R97 E: opt-in Rust delegation for the pair-gate
    default-picker.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt the gate-pick decision through the Rust semantic
    transformer (`semantic_role_decl::pick_default_pair_gates_public`).
    Returns `(gate_a, gate_b)` on success or `None` on any failure
    (env var unset, extension missing, validation rejection, < 2
    ports, etc.).

    Python callers MUST treat `None` as "fall back to the canonical
    Python heuristic". The Rust path is BYTE-EQUIVALENT to the Python
    `_pick_default_pair_gates` output: same input-role ordering, same
    convention-pair ordering, same first-two-ports fallback. The
    bit-equivalence contract is enforced by the
    `tests/test_rust_pick_default_pair_gates_v128.py` suite.

    The import guard on `try_rust_pick_default_pair_gates` keeps older
    bridge deployments (pre-R97 E) safe — missing function silently
    falls back to Python.
    """
    try:
        from .._rust_bridge import try_rust_pick_default_pair_gates
    except (ImportError, AttributeError):
        return None
    try:
        result = try_rust_pick_default_pair_gates(
            list(port_names),
            dict(port_roles),
            cell_name,
            role_name,
        )
    except Exception:
        return None
    if result is None:
        return None
    # Rust returns a (str, str) tuple — coerce to plain Python tuple
    # so callers see the exact shape produced by the Python heuristic.
    return (str(result[0]), str(result[1]))


def _pick_default_pair_gates(
    port_names: list[str],
    port_roles: dict[str, "str | None"],
    *,
    cell_name: str,
    role_name: str,
) -> tuple[str, str]:
    """Pick default gate nets for a pair role's M1 / M2.

    Heuristic, applied in order:
      1. Two ports explicitly tagged `: input` (the v0.0.22 port-role
         annotation). If exactly two such ports exist, use them in
         declaration order.
      2. Two ports whose names match the differential-input convention
         (`inn`/`inp`, `in_n`/`in_p`, `in_minus`/`in_plus`).
      3. The first two ports in the cell's signature.

    Rejects (raises ValueError) when the cell has fewer than 2 candidate
    ports — a `pmos_pair` declaration on a 1-port cell can't be
    auto-wired.

    v0.1.20 R97 E: opt-in Rust delegation. When the Rust path is
    enabled AND succeeds, it returns the SAME `(gate_a, gate_b)` tuple
    the Python heuristic below would compute (byte-equivalent). On
    None return — env var unset, extension missing, < 2 ports — the
    canonical Python heuristic below runs unchanged.
    """
    rust_result = _try_rust_pick_default_pair_gates(
        port_names, port_roles,
        cell_name=cell_name, role_name=role_name,
    )
    if rust_result is not None:
        return rust_result
    # ─── Canonical Python heuristic (preserved as fallback) ───────
    input_role_ports = [p for p in port_names if port_roles.get(p) == "input"]
    if len(input_role_ports) >= 2:
        return (input_role_ports[0], input_role_ports[1])
    # Convention-name pairs — preserve order so `inn` lands on M1.g.
    pair_conventions = [
        ("inn", "inp"),
        ("in_n", "in_p"),
        ("in_minus", "in_plus"),
        ("vinn", "vinp"),
        ("vin_n", "vin_p"),
    ]
    port_set = set(port_names)
    for a, b in pair_conventions:
        if a in port_set and b in port_set:
            return (a, b)
    # Fallback: first two ports.
    if len(port_names) < 2:
        raise ValueError(
            f"cell {cell_name!r}: role {role_name!r} (pair) needs at "
            f"least 2 candidate input ports, but the cell declares "
            f"only {len(port_names)} ports total ({port_names!r}). "
            f"Either add more ports to the cell signature or supply "
            f"`gates = (a, b);` explicitly in the role body."
        )
    return (port_names[0], port_names[1])


def _expand_roles_to_devices(
    role_decls: list[RoleDecl],
    *,
    port_names: list[str],
    port_roles: dict[str, "str | None"],
    cell_name: str,
) -> list[Device]:
    """Expand every RoleDecl in `role_decls` to concrete Device IR entries.

    Each role expands to a fixed device count per its kind. Auto-named
    local pins are prefixed with `<role_name>_`. Single-net and two-net
    pin overrides on the role replace the corresponding auto-named pin.

    Returns the flat list of expanded devices (one role's devices land
    contiguously, preserving declaration order across roles).

    Raised errors: ValueError on missing required pair-gate ports
    (see `_pick_default_pair_gates`).
    """
    out: list[Device] = []
    for r in role_decls:
        out.extend(_expand_role_to_devices(
            r,
            port_names=port_names,
            port_roles=port_roles,
            cell_name=cell_name,
        ))
    return out


def _expand_role_to_devices(
    r: RoleDecl,
    *,
    port_names: list[str],
    port_roles: dict[str, "str | None"],
    cell_name: str,
) -> list[Device]:
    """Expand ONE role decl. See `_expand_roles_to_devices` for details."""
    name = r.name
    kind = r.kind

    # Build the device-level sizing dict by stripping the role-level
    # intent (`gm_id` lives ONLY in sizing maps that the sizing pass
    # consults; it doesn't make sense as a Device.sizing key — leaves
    # it sitting on every expanded device's sizing so the sizing pass
    # / driver can detect "this device has a gm/Id intent" and react).
    base_sizing: dict[str, float] = {}
    for k, v in r.sizing.items():
        base_sizing[k] = v
    # `role_origin` is a non-numeric breadcrumb so the sizing inspector
    # can trace a device back to its role. Stored as a string-valued
    # sizing entry intentionally — the Device.sizing dict is `dict[str,
    # float]` per the dataclass annotation, but Python doesn't enforce
    # it, and downstream consumers iterating numerically already skip
    # non-numeric values gracefully.
    # NOTE: we use a separate non-numeric key prefix `_role_` to make
    # filtering easy. Currently unused by downstream code but visible
    # to any future inspector.
    # (kept off the sizing dict to avoid surprising any consumer that
    # iterates sizing for numeric-only operations)
    # Instead we tag via Device's instance_name only (`<role>_<suffix>`)
    # which is already unambiguous.

    if kind in ("pmos_pair", "nmos_pair"):
        return _expand_pair(
            r, base_sizing=base_sizing,
            port_names=port_names, port_roles=port_roles,
            cell_name=cell_name,
        )
    if kind in ("pmos_mirror", "nmos_mirror"):
        return _expand_mirror(r, base_sizing=base_sizing, cell_name=cell_name)
    if kind in ("pmos_cs", "nmos_cs"):
        return _expand_cs(r, base_sizing=base_sizing, cell_name=cell_name)
    raise ValueError(
        f"cell {cell_name!r}: role {name!r} has unknown kind {kind!r}. "
        f"Valid: {sorted(_VALID_ROLE_KINDS)}."
    )


def _try_rust_is_pmos_kind(kind: str) -> "bool | None":
    """v0.1.21 R98 E: opt-in Rust delegation for the PMOS-family
    classifier.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt the kind classification through the Rust
    semantic transformer (`semantic_role_decl::is_pmos_kind_public`).
    Returns `True` / `False` on success or `None` on any failure
    (env var unset, extension missing, etc.).

    Python callers MUST treat `None` as "fall back to the canonical
    Python predicate". The Rust path is BYTE-EQUIVALENT to the Python
    `_is_pmos_kind` output: same `kind.startswith("pmos_")` rule.

    The import guard on `try_rust_is_pmos_kind` keeps older bridge
    deployments (pre-R98 E) safe — missing function silently falls
    back to Python.
    """
    try:
        from .._rust_bridge import try_rust_is_pmos_kind
    except (ImportError, AttributeError):
        return None
    try:
        result = try_rust_is_pmos_kind(kind)
    except Exception:
        return None
    if result is None:
        return None
    return bool(result)


def _is_pmos_kind(kind: str) -> bool:
    """Return True iff `kind` is a PMOS-family role kind.

    v0.1.21 R98 E: opt-in Rust delegation. When the Rust path is
    enabled AND succeeds, it returns the SAME bool the Python
    predicate below would compute (byte-equivalent). On None return
    — env var unset, extension missing — the canonical Python branch
    below runs unchanged.
    """
    rust_result = _try_rust_is_pmos_kind(kind)
    if rust_result is not None:
        return rust_result
    return kind.startswith("pmos_")


def _try_rust_supply_and_body(kind: str) -> "tuple[str, str] | None":
    """v0.1.21 R98 E: opt-in Rust delegation for the supply / body rail
    picker.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt the rail assignment through the Rust semantic
    transformer (`semantic_role_decl::supply_and_body_public`).
    Returns `(supply, body)` on success or `None` on any failure.

    Python callers MUST treat `None` as "fall back to the canonical
    Python heuristic". The Rust path is BYTE-EQUIVALENT to the Python
    `_supply_and_body` output: same `('vdd', 'vdd')` vs `('vss', 'vss')`
    rail assignment.

    The import guard on `try_rust_supply_and_body` keeps older bridge
    deployments (pre-R98 E) safe — missing function silently falls
    back to Python.
    """
    try:
        from .._rust_bridge import try_rust_supply_and_body
    except (ImportError, AttributeError):
        return None
    try:
        result = try_rust_supply_and_body(kind)
    except Exception:
        return None
    if result is None:
        return None
    # Rust returns a (str, str) tuple — coerce to plain Python tuple
    # so callers see the exact shape produced by the Python heuristic.
    return (str(result[0]), str(result[1]))


def _supply_and_body(kind: str) -> tuple[str, str]:
    """Return `(default_supply_or_ground_net, body_net)` for a role kind.

    PMOS-family roles tie source to `vdd` and body to `vdd` by default.
    NMOS-family roles tie source to `vss` and body to `vss` by default.
    Mirror/cs roles use this as the source rail; pair roles use it for
    body (sources tie at the auto-named tail node).

    v0.1.21 R98 E: opt-in Rust delegation. When the Rust path is
    enabled AND succeeds, it returns the SAME `(supply, body)` tuple
    the Python heuristic below would compute (byte-equivalent). On
    None return — env var unset, extension missing — the canonical
    Python branch below runs unchanged.
    """
    rust_result = _try_rust_supply_and_body(kind)
    if rust_result is not None:
        return rust_result
    return ("vdd", "vdd") if _is_pmos_kind(kind) else ("vss", "vss")


def _try_rust_pair_delegation(
    r: RoleDecl,
    *,
    port_names: list[str],
    port_roles: dict[str, "str | None"],
    cell_name: str,
) -> "list[Device] | None":
    """v0.0.99 R76 A: opt-in Rust delegation for pair role expansion.

    When `CLAY_USE_RUST_PARSER=1` is set AND the Rust extension is
    loadable, attempt the pair expansion through the Rust semantic
    transformer (`semantic_role_decl::expand_pair_role_from_fields`).
    Returns a `list[Device]` on success or `None` on any failure
    (env var unset, extension missing, validation rejection, etc.).
    Python callers MUST treat `None` as "fall back to the canonical
    Python expansion".

    v0.1.21 R98 D — the production `_expand_pair` path no longer consults
    this wrapper (the Python body was stripped; Rust dispatch is the only
    production path). This wrapper survives as the R76 A test-facing
    strangler-fig API: the R76 A test corpus
    (`tests/test_rust_role_decl_expansion_v100.py`) plus the R97 D
    Strip 2 `pair_wrapper_preserved` test pin against its env-var-gated
    behavior. Direct callers seeking opt-in semantics keep working.

    The Rust path is BYTE-EQUIVALENT to the pre-strip Python `_expand_pair`
    output: same instance names (`<role>_M1`, `<role>_M2`), same pin dict
    keys (`d`, `g`, `s`, `b`), same device_class, same sizing dict
    (including the auto-filled `M=1.0`). The bit-equivalence contract is
    enforced by the `tests/test_rust_role_decl_expansion_v100.py` suite.

    The hasattr check on `try_rust_expand_pair_role_solo` guards against
    a bridge module that pre-dates this delegation surface — older
    deployments missing the function silently fall back to Python.
    """
    try:
        from .._rust_bridge import try_rust_expand_pair_role_solo
    except (ImportError, AttributeError):
        return None
    try:
        rust_devs = try_rust_expand_pair_role_solo(
            r.name,
            r.kind,
            dict(r.sizing),
            dict(r.pin_overrides),
            dict(r.pair_overrides),
            list(port_names),
            dict(port_roles),
            cell_name,
        )
    except Exception:
        return None
    if rust_devs is None:
        return None
    return _coerce_rust_devs(rust_devs)


def _coerce_rust_devs(rust_devs) -> list[Device]:
    """Translate `PyParsedDevice` instances to canonical `Device`
    dataclasses. The pin map and sizing map are coerced to plain Python
    dicts so downstream consumers (sizing pass, simulator drivers,
    IR-bridge) see the exact same shape as the Python expansion's
    output."""
    out: list[Device] = []
    for d in rust_devs:
        out.append(Device(
            instance_name=d.instance_name,
            device_class=d.device_class,
            pdk_variant=d.pdk_variant,
            pins=dict(d.pins),
            sizing={k: float(v) for k, v in dict(d.sizing).items()},
        ))
    return out


# v0.1.18 R97 D mass-deletion Strip 2: the R94 F wrappers
# `_try_rust_mirror_delegation` / `_try_rust_cs_delegation` were thin
# RoleDecl → bridge adapters that became unused once R95 D stripped the
# `_expand_mirror` / `_expand_cs` Python bodies and dispatched directly
# to `_rust.try_rust_expand_mirror_role_solo` /
# `_rust.try_rust_expand_cs_role_solo`. The wrappers carried an env-var
# gate (`CLAY_USE_RUST_PARSER`) that no longer matches the R95 D
# unconditional-dispatch contract, so they were removed entirely. The
# R94 F test corpus migrated its 26 callsites to local helpers (see
# `tests/test_rust_role_expand_mirror_cs_v128.py`) that wrap the same
# bridge entry directly.


def _expand_pair(
    r: RoleDecl,
    *,
    base_sizing: dict[str, float],
    port_names: list[str],
    port_roles: dict[str, "str | None"],
    cell_name: str,
) -> list[Device]:
    """`pmos_pair NAME { ... }` → 2 PMOS, sources tied at tail.

    Default wiring (overrideable via pin/pair fields on the role body):
      M1.d = `<NAME>_d1`     (override: drains = (a, b) → use a)
      M1.g = first input port  (override: gates = (a, b) → use a)
      M1.s = `<NAME>_tail`   (override: tail = NET)
      M1.b = vdd             (PMOS) / vss (NMOS)
      M2.d = `<NAME>_d2`     (override: drains = (a, b) → use b)
      M2.g = second input port (override: gates = (a, b) → use b)
      M2.s = `<NAME>_tail`   (override: tail = NET)
      M2.b = vdd             (PMOS) / vss (NMOS)

    v0.1.21 R98 D mass-deletion strip. The Python implementation body
    (R76 A strangler-fig: Rust delegation under `CLAY_USE_RUST_PARSER=1`
    + canonical Python fallback) has been removed; this function now
    unconditionally delegates to
    `semantic_role_decl::expand_pair_role_from_fields`, the byte-equal
    Rust port that R76 A wired through `expand_pair_role_solo`. The
    `CLAY_USE_RUST_PARSER` env-var gate is gone from this path — Rust
    dispatch is the only path. Byte-equality across the Python reference
    and Rust port was asserted by R76 A (24 byte-equivalence + boundary
    tests on `tests/test_rust_role_decl_expansion_v100.py`) before
    strip, then re-pinned across R94 F regression / R95 D / R97 D Strip 2.
    Raises `RuntimeError` when the Rust extension is missing.

    The R76 A `_try_rust_pair_delegation` wrapper survives the strip
    (env-var-gated, returning the same `list[Device]` it always did) so
    the R76 A test corpus can continue to assert wrapper-level semantics.
    It is no longer consulted by this production path — callers wanting
    env-var-gated semantics must call the wrapper directly.

    Mirror of the R86 C `_ir_bridge._emit_functional_role_op_lines`
    strip pattern (v0.1.10) and the R95 D `_expand_mirror` / `_expand_cs`
    strip (v0.1.18).
    """
    _require_clay_parser_for_role_strip()
    # v0.1.21 R98 D: direct bridge dispatch (no env-var gate). The R76 A
    # bridge wrapper `_try_rust_pair_delegation` is preserved for the
    # R76 A test corpus + R97 D Strip 2 `pair_wrapper_preserved` test,
    # but this production path goes straight to the PyO3 binding to
    # match the R95 D mirror+cs convention. `base_sizing` is the same
    # dict shape `_try_rust_pair_delegation` would marshal from
    # `r.sizing` upstream (the caller `_expand_role_to_devices` rebuilds
    # `base_sizing` from `r.sizing` byte-for-byte before invoking us).
    from .._rust_bridge import _rust  # type: ignore[attr-defined]

    rust_devs = _rust.try_rust_expand_pair_role_solo(
        r.name,
        r.kind,
        dict(base_sizing),
        dict(r.pin_overrides),
        dict(r.pair_overrides),
        list(port_names),
        dict(port_roles),
        cell_name,
    )
    if rust_devs is None:
        # R76 A Rust side returns None on validation rejection (e.g.
        # unknown sizing key, fewer than 2 candidate ports with no
        # `gates` override, non-pair kind). Preserve the canonical
        # pre-strip Python error surface so callers see the same
        # ValueError text the parser tests pin against — including the
        # `_pick_default_pair_gates` "fewer than 2 candidate input
        # ports" message R76 A pinned in
        # `test_pair_with_too_few_ports_raises_in_both_paths`.
        raise ValueError(
            f"cell {cell_name!r}: pair role {r.name!r} ({r.kind!r}) "
            f"could not be expanded — Rust validation rejected the role. "
            f"Common causes: cell has fewer than 2 candidate input ports "
            f"with no `gates = (a, b);` override; unknown sizing key; "
            f"non-pair kind dispatched to `_expand_pair`."
        )
    return _coerce_rust_devs(rust_devs)


def _expand_mirror(
    r: RoleDecl,
    *,
    base_sizing: dict[str, float],
    cell_name: str = "",
) -> list[Device]:
    """`pmos_mirror NAME { ... }` → 2 PMOS, gates tied; diode-leg M1.

    Default wiring:
      M1 (diode):  d = `<NAME>_in`, g = `<NAME>_in`,
                   s = vdd / vss, b = vdd / vss
      M2 (mirror): d = `<NAME>_out`, g = `<NAME>_in`,
                   s = vdd / vss, b = vdd / vss
    Overrides:
      `in  = NET;` replaces `<NAME>_in`
      `out = NET;` replaces `<NAME>_out`

    v0.1.18 R95 D mass-deletion strip. The Python implementation body
    (R90 B parser-split / earlier inline `clay/parser.py` v0.0.31) has
    been removed; this function unconditionally delegates to
    `semantic_role_decl::expand_mirror_role_from_fields`, the byte-equal
    Rust port landed by R94 F. The `CLAY_USE_RUST_PARSER` env-var gate
    is gone from this path — Rust dispatch is the only path.
    Byte-equality across the Python reference and Rust port was asserted
    by R94 F (53/53 across `tests/test_rust_role_expand_mirror_cs_v128.py`)
    before strip. Raises `RuntimeError` when the Rust extension is missing.

    Mirror of the R86 C `_ir_bridge._emit_functional_role_op_lines`
    strip pattern (v0.1.10).
    """
    _require_clay_parser_for_role_strip()
    # v0.1.17 R94 F: marshal base_sizing into the Rust call so the
    # downstream sizing pass sees the auto-filled M=1.0 default that
    # `build_base_sizing` produced upstream. v0.1.18 R97 D removed the
    # env-var-gated `_try_rust_mirror_delegation` wrapper entirely; this
    # path dispatches directly to the bridge function with no gate.
    from .._rust_bridge import _rust  # type: ignore[attr-defined]

    rust_devs = _rust.try_rust_expand_mirror_role_solo(
        r.name,
        r.kind,
        dict(base_sizing),
        dict(r.pin_overrides),
        dict(r.pair_overrides),
        cell_name,
    )
    if rust_devs is None:
        # R94 F Rust side returns None on validation rejection (e.g.
        # missing sizing source). Preserve the canonical pre-strip
        # Python error surface so callers see the same ValueError text
        # the parser tests pin against.
        raise ValueError(
            f"cell {cell_name!r}: mirror role {r.name!r} ({r.kind!r}) "
            f"could not be expanded — Rust validation rejected the role. "
            f"Verify sizing (W+L or gm_id+L), pin_overrides (only `in` "
            f"and `out` are valid for mirror), and pair_overrides "
            f"(mirror has no two-net labels; pair_overrides must be empty)."
        )
    return _coerce_rust_devs(rust_devs)


def _expand_cs(
    r: RoleDecl,
    *,
    base_sizing: dict[str, float],
    cell_name: str = "",
) -> list[Device]:
    """`pmos_cs NAME { ... }` → single PMOS common-source stage.

    Default wiring:
      M1.d = `<NAME>_drain` (override: drain = NET)
      M1.g = `<NAME>_gate`  (override: gate = NET)
      M1.s = vdd / vss
      M1.b = vdd / vss

    v0.1.18 R95 D mass-deletion strip. The Python implementation body
    has been removed; this function unconditionally delegates to
    `semantic_role_decl::expand_cs_role_from_fields`, the byte-equal
    Rust port landed by R94 F. The `CLAY_USE_RUST_PARSER` env-var gate
    is gone from this path — Rust dispatch is the only path. Raises
    `RuntimeError` when the Rust extension is missing.

    Mirror of the R86 C `_ir_bridge._emit_functional_role_op_lines`
    strip pattern (v0.1.10).
    """
    _require_clay_parser_for_role_strip()
    from .._rust_bridge import _rust  # type: ignore[attr-defined]

    rust_devs = _rust.try_rust_expand_cs_role_solo(
        r.name,
        r.kind,
        dict(base_sizing),
        dict(r.pin_overrides),
        dict(r.pair_overrides),
        cell_name,
    )
    if rust_devs is None:
        raise ValueError(
            f"cell {cell_name!r}: cs role {r.name!r} ({r.kind!r}) "
            f"could not be expanded — Rust validation rejected the role. "
            f"Verify sizing (W+L or gm_id+L), pin_overrides (only `gate` "
            f"and `drain` are valid for cs), and pair_overrides "
            f"(cs has no two-net labels; pair_overrides must be empty)."
        )
    return _coerce_rust_devs(rust_devs)
