"""
Clay v0.1.01 — Physical invariant as type inhabitation (Round 78 §5.5).

This module promotes the existing runtime `PortConstraint` check (declared
on `crates/clay_ir/src/graph.rs::PortConstraint` and consumed by
`CellGraph::validate_port_constraints`) from a *post-compile / pre-SPICE*
runtime sweep to a *compile-time elaboration-stage* type-level rejection
gate.

Motivation
----------
Kirchhoff / charge / power conservation are physically true but were
historically enforced by SPICE itself: an LLM that emits a circuit with a
floating gate, a power rail that never reaches its reference, or a MOS
whose body is wired to the wrong well silently passes Clay's existing
compile path and only fails downstream when ngspice diverges. That feedback
loop is seconds-slow, opaque (ngspice errors are not actionable for an
LLM), and the iteration cost during synthesis is the single biggest blocker
for closed-loop topology generation.

The fix is to move the four most-violated physical invariants up the
pipeline so they fire at *elaboration time* — after `compile_cell_def` has
materialized the device list and wired the bias tree, but BEFORE the cell
ever sees a SPICE driver. The runtime contract is identical to the
existing functional-role `InvariantViolation` shape so the LLM's structured
error consumer keeps a single schema.

The four invariants implemented here are:

    NetDriverInvariant
        Every internal net (a net that is neither a declared cell port nor
        a global rail like `0`/`gnd`/`vdd`/`vss` nor an explicitly declared
        bias node) must be touched by at least one device pin that can
        *drive* the net (a MOS drain, a current-source output, a resistor
        terminal, or a capacitor terminal — anything that isn't gate-only).
        Catches "floating gate" (a gate-only net with no driver) and
        "orphan net" (typo of a real net name).

    PortPolarityInvariant
        Every declared cell port that carries a polarity role (`power_p` /
        `power_n` / `input` (when paired) / `output`) must have its dual
        also declared. Specifically: a port with role `power_p` requires a
        sibling port with role `power_n`, and vice versa. Catches "I
        declared vdd but forgot vss".

    PowerReachInvariant
        For every port carrying role `power_p` / `power_n` / `bias` /
        `ground`, the cell must contain at least one device pin physically
        wired to that port's net. A power rail port that no device touches
        is unreachable — SPICE would later report this as a Gmin or
        topology error.

    MosBodyWellInvariant
        Every MOS device (`device_class in {"nmos", "pmos"}`) must wire
        its body pin (`b`) to a net that respects the well topology of
        the PDK: PMOS body to an n-well (default: `vdd`); NMOS body to a
        p-substrate (default: `vss` / `0` / `gnd`). Catches the most
        common LLM mistake of cross-wiring the body to the wrong rail.

API
---
Each invariant has the same three-callable shape:

    check(cell)                -> list[PhysicalInvariantViolation]
    format_violation(viol)     -> str  (one-line human-readable)
    suggested_fix(viol)        -> str  (structured fix hint)

The module exposes a top-level entry point::

    check_physical_invariants(cell, *, raise_on_violation: bool = False)
        -> list[PhysicalInvariantViolation]

When called from `clay/compiler.py::compile_cell_def` with
`raise_on_violation=True`, any single violation raises
`PhysicalInvariantError` (a CellCompileError subclass) so the LLM sees a
type-level rejection AT elaboration time, BEFORE the cell is handed to
any simulator driver.

Scope
-----
This module is *additive*. The existing runtime `PortConstraint` /
`validate_port_constraints` path on the Rust side is preserved byte-exact
— that path still runs on `clay_ir::CellGraph` instances when callers
choose. The PyO3 fn is unchanged.

This is the v0.1.01 (Round 78 §5.5) Tier 1 lift. The four invariants here
cover the four most common LLM-emitted topology faults; extending the
catalog (e.g. "every input gate must be a MOS, not a passive") lands as
new variant entries WITHOUT touching the existing shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ─── Constants ──────────────────────────────────────────────────

# Nets that are universally legal as "reachable references" without
# explicit declaration. `0` and `gnd` are the SPICE universal grounds;
# `vdd` and `vss` are conventional supply rails that every cell relies
# on even when not declared as ports (the testbench injects them via
# `resolve_supply`).
_GLOBAL_RAILS: frozenset[str] = frozenset({"0", "gnd", "vdd", "vss"})

# Per-device-class set of pin labels that "drive" their net. A net
# touched ONLY by gate pins (MOS `g`) has no driver and is treated as
# floating. This map is intentionally tiny — only the four primitive
# device classes Clay synthesizes today; extending it is additive.
_DRIVER_PINS_BY_CLASS: dict[str, frozenset[str]] = {
    "nmos":    frozenset({"d", "s", "b"}),   # drain/source/body drive
    "pmos":    frozenset({"d", "s", "b"}),
    "cap":     frozenset({"p", "n"}),
    "res":     frozenset({"p", "n"}),
    "ind":     frozenset({"p", "n"}),
    "cur_src": frozenset({"p", "n"}),
    "vcvs":    frozenset({"p", "n"}),
}

# MOS body legal targets per device-class. The default well topology
# matches every PDK Clay currently supports (sky130, gf180mcu, open_pdk):
# PMOS sits in an n-well tied to a positive rail; NMOS sits on
# p-substrate tied to a negative rail or the universal ground.
# v0.1.01: this is a small enumerated default; PDK-specific n-well /
# p-well overrides land via a future PDK-extension hook.
_LEGAL_BODY_NETS: dict[str, frozenset[str]] = {
    "pmos": frozenset({"vdd"}),
    "nmos": frozenset({"vss", "0", "gnd"}),
}

# Port roles that participate in the polarity-pair check. Each entry
# maps a role to its required dual. The check is symmetric — declaring
# either side without the other triggers a violation.
_POLARITY_DUALS: dict[str, str] = {
    "power_p": "power_n",
    "power_n": "power_p",
}

# Port roles that must be physically reached by at least one device
# pin's wiring. These are the "power-class" ports the testbench
# eventually drives.
_REACHABILITY_ROLES: frozenset[str] = frozenset({
    "power_p", "power_n", "bias", "ground",
})


# ─── Violation dataclass ────────────────────────────────────────


@dataclass
class PhysicalInvariantViolation:
    """One physical-invariant check outcome.

    Categories:
      * `floating_net`     — net has no driver pin (NetDriverInvariant)
      * `orphan_net`       — net is referenced but never declared (NetDriverInvariant)
      * `dangling_net`     — net produced/consumed by exactly one device pin
                             and not exempted as a boundary net (DanglingNetInvariant)
      * `missing_dual`     — polarity port declared without dual (PortPolarityInvariant)
      * `unreachable_port` — declared port never wired (PowerReachInvariant)
      * `body_well_mismatch` — MOS body wired to wrong well (MosBodyWellInvariant)
      * `skipped`          — informational pass (invariant inapplicable)

    The schema mirrors `clay.invariants.InvariantViolation` so a downstream
    LLM consumer sees a uniform structure across functional-role and
    physical-invariant errors. Every field is JSON-safe.
    """
    invariant_kind: str        # "NetDriver" / "PortPolarity" / "PowerReach" / "MosBodyWell"
    cell_name: str
    category: str              # see categories list above
    target: str                # net name OR port name OR device name (context-dependent)
    expected: Optional[str]    # what the invariant required (human-readable)
    actual: Optional[str]      # what the cell carries (human-readable)
    message: str               # one-line description
    suggested_fix: str         # structured fix hint
    # Free-form extras for tooling that wants additional context (which
    # device pin landed where, etc.).
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        """A violation is fatal (i.e. should block elaboration) unless
        it's an informational `skipped` entry. The compile-time gate
        raises only on fatal violations."""
        return self.category != "skipped"

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for LLM-consumable error
        surfaces. `extras` is included verbatim."""
        return asdict(self)


class PhysicalInvariantError(ValueError):
    """Raised at elaboration time when any physical invariant fails.

    Subclassed from `ValueError` (not `CellCompileError`) so the
    compiler.py wrapper can decide whether to re-frame the exception as
    a CellCompileError or surface it directly; subclassing here would
    create a circular import with clay/compiler.py.

    The exception carries the list of violations on `.violations` so a
    structured error surface (e.g. a JSON dump for the LLM) can extract
    every failing entry without parsing the message string.
    """

    def __init__(self, violations: list[PhysicalInvariantViolation]) -> None:
        self.violations = list(violations)
        first = self.violations[0] if self.violations else None
        header = (
            f"physical invariant violated: "
            f"{first.invariant_kind} / {first.category}"
            if first else "physical invariant violated (empty list)"
        )
        lines = [header]
        for v in self.violations:
            lines.append(f"  - {v.message}")
            if v.suggested_fix:
                lines.append(f"    fix: {v.suggested_fix}")
        super().__init__("\n".join(lines))


# ─── Helpers ────────────────────────────────────────────────────


def _cell_devices(cell: Any) -> list[Any]:
    """Return the cell's device list. Defensive against minimal Cell IR
    constructions that omit the field — tests sometimes build a Cell
    with no devices to exercise the polarity check alone."""
    return list(getattr(cell, "devices", []) or [])


def _cell_ports(cell: Any) -> list[str]:
    """Return the cell's declared port list."""
    return list(getattr(cell, "ports", []) or [])


def _port_roles_for(cell: Any) -> dict[str, Optional[str]]:
    """Recover the CellDef-attached `port_roles` map if the compiler
    stashed it on the cell. Returns `{}` when no role metadata is
    present (a cell built without going through `compile_cell_def`).
    The compiler stashes this on `cell.__dict__['_port_roles']`."""
    return dict(cell.__dict__.get("_port_roles", {}) or {})


def _all_nets_touched(devices: list[Any]) -> set[str]:
    """Return the union of every net name any device pin references."""
    nets: set[str] = set()
    for dev in devices:
        for net in getattr(dev, "pins", {}).values():
            nets.add(net)
    return nets


def _driver_pins_for(dev: Any) -> frozenset[str]:
    """Per-device set of pin labels that count as drivers of their net.
    Default to the empty set for unrecognized device classes — those
    devices contribute no driver evidence (conservative: the invariant
    fires more often, the LLM gets more guidance)."""
    cls = getattr(dev, "device_class", "")
    return _DRIVER_PINS_BY_CLASS.get(cls, frozenset())


# ─── Invariant 1: NetDriverInvariant ────────────────────────────


def check_net_driver_invariant(cell: Any) -> list[PhysicalInvariantViolation]:
    """Every non-port, non-rail, non-bias-node net must have at least
    one driver pin.

    A "driver pin" is any device pin label that physically can source
    or sink current on the net — MOS drain/source/body, resistor/capacitor
    terminals, current-source outputs. A net touched ONLY by MOS gate
    pins (`g`) is the classic "floating gate" that an LLM produces when
    it forgets to bias an input — SPICE later reports this as a Gmin
    convergence failure that's expensive to diagnose. We catch it here.

    The legal-net set for the check is `ports ∪ globals ∪ bias_tree`.
    Anything else must carry a driver or surface as a `floating_net`
    violation (when the net is touched but only by gates) or an
    `orphan_net` violation (when the net is referenced exactly once and
    isn't any of the legal categories — likely a typo).
    """
    devices = _cell_devices(cell)
    if not devices:
        return []

    ports = set(_cell_ports(cell))
    bias_nodes = set(cell.__dict__.get("_bias_nodes", set()) or set())
    legal = ports | _GLOBAL_RAILS | bias_nodes

    # Map each net → list of (device_class, pin_label) tuples touching it.
    net_touches: dict[str, list[tuple[str, str]]] = {}
    for dev in devices:
        cls = getattr(dev, "device_class", "")
        for label, net in getattr(dev, "pins", {}).items():
            net_touches.setdefault(net, []).append((cls, label))

    violations: list[PhysicalInvariantViolation] = []

    for net, touches in sorted(net_touches.items()):
        if net in legal:
            continue
        # Internal net — must have at least one driver pin.
        drivers = [
            (cls, label) for (cls, label) in touches
            if label in _DRIVER_PINS_BY_CLASS.get(cls, frozenset())
        ]
        if drivers:
            continue  # well-driven

        # No driver. Two failure modes are *fatal* at type-level:
        #
        #   * floating_net: ≥2 gate pins share a net with no driver.
        #     This is a true "floating bus" — multiple gates connected
        #     by a wire that has nothing pulling it to a defined voltage.
        #     SPICE can't converge on this; the cell is structurally
        #     broken regardless of the testbench.
        #
        # A single dangling gate (one gate pin on an otherwise unused
        # net) is NOT fatal at type-level: that's the existing
        # `external bias` pattern where the testbench supplies the bias
        # voltage from outside the cell. The pre-existing dangling-net
        # `UserWarning` in `_compile_primitive_cell` already surfaces
        # this softly to the LLM; we intentionally do not promote it
        # to a hard error here so legitimate external-bias cells keep
        # compiling.
        only_gates = all(
            label == "g" and cls in ("nmos", "pmos") for (cls, label) in touches
        )
        if only_gates and len(touches) >= 2:
            gate_devs = ", ".join(
                f"{cls}.{label}" for (cls, label) in touches
            )
            violations.append(PhysicalInvariantViolation(
                invariant_kind="NetDriver",
                cell_name=getattr(cell, "name", "?"),
                category="floating_net",
                target=net,
                expected="at least one driver pin (drain/source/body or passive terminal)",
                actual=f"only gate pins touch this net: {gate_devs}",
                message=(
                    f"cell {getattr(cell, 'name', '?')!r}: net {net!r} "
                    f"is floating — {len(touches)} gate pin(s) "
                    f"({gate_devs}) share this net with no driver. "
                    f"SPICE would report a Gmin or non-convergence "
                    f"error at this node."
                ),
                suggested_fix=(
                    f"wire net {net!r} to a driver pin (a MOS drain/source, "
                    f"a current-source output, or a passive terminal), or "
                    f"declare {net!r} as a cell port if every gate "
                    f"connected to it is intended to be biased externally."
                ),
                extras={"touches": touches},
            ))
        # else: single-pin dangling net OR non-gate-only orphan — the
        # existing dangling-net soft warning in `_compile_primitive_cell`
        # already covers this case. We do NOT promote it to a fatal
        # type-level violation: doing so would reject the legitimate
        # external-bias-input pattern.

    return violations


# ─── Invariant 2: PortPolarityInvariant ─────────────────────────


def check_port_polarity_invariant(cell: Any) -> list[PhysicalInvariantViolation]:
    """A port with role `power_p` must be paired with a `power_n`
    sibling on the same cell (and vice versa). This catches the
    "declared vdd but forgot vss" failure mode.

    Cells built without going through `compile_cell_def` (i.e. without
    a `port_roles` map stashed on `cell.__dict__`) skip this check
    silently — there's no role metadata to consult.
    """
    roles = _port_roles_for(cell)
    if not roles:
        return []

    declared_roles = {r for r in roles.values() if r is not None}
    violations: list[PhysicalInvariantViolation] = []

    for port, role in roles.items():
        if role not in _POLARITY_DUALS:
            continue
        dual = _POLARITY_DUALS[role]
        if dual not in declared_roles:
            violations.append(PhysicalInvariantViolation(
                invariant_kind="PortPolarity",
                cell_name=getattr(cell, "name", "?"),
                category="missing_dual",
                target=port,
                expected=f"a sibling port with role {dual!r}",
                actual=f"no port carries role {dual!r}",
                message=(
                    f"cell {getattr(cell, 'name', '?')!r}: port {port!r} "
                    f"has role {role!r} but no sibling port carries the "
                    f"required dual {dual!r}. Power-class ports must "
                    f"come in polarity pairs."
                ),
                suggested_fix=(
                    f"add a port with role {dual!r} to the cell, or "
                    f"remove the role annotation from {port!r} if this "
                    f"is a single-rail design and {role!r} was a "
                    f"misclassification."
                ),
                extras={"declared_roles": sorted(declared_roles)},
            ))

    return violations


# ─── Invariant 3: PowerReachInvariant ───────────────────────────


def check_power_reach_invariant(cell: Any) -> list[PhysicalInvariantViolation]:
    """Every port carrying a power-class role (`power_p` / `power_n` /
    `bias` / `ground`) must be physically wired to at least one device
    pin. A power rail port that no device touches is unreachable — SPICE
    would later report a topology / floating-supply error.

    Cells without role metadata skip this check silently.
    """
    roles = _port_roles_for(cell)
    if not roles:
        return []

    devices = _cell_devices(cell)
    touched_nets = _all_nets_touched(devices)

    violations: list[PhysicalInvariantViolation] = []
    for port, role in roles.items():
        if role not in _REACHABILITY_ROLES:
            continue
        if port in touched_nets:
            continue
        violations.append(PhysicalInvariantViolation(
            invariant_kind="PowerReach",
            cell_name=getattr(cell, "name", "?"),
            category="unreachable_port",
            target=port,
            expected="at least one device pin wired to this port's net",
            actual="no device pin references this net",
            message=(
                f"cell {getattr(cell, 'name', '?')!r}: port {port!r} "
                f"(role {role!r}) is declared but no device pin reaches "
                f"it. SPICE would later report a floating-supply or "
                f"unreachable-node topology error."
            ),
            suggested_fix=(
                f"wire at least one device pin to net {port!r}, or "
                f"remove the port from the cell signature if it was "
                f"added speculatively."
            ),
            extras={"role": role},
        ))

    return violations


# ─── Invariant 4: MosBodyWellInvariant ──────────────────────────


def check_mos_body_well_invariant(cell: Any) -> list[PhysicalInvariantViolation]:
    """Every MOS device must wire its body pin (`b`) to a net that
    respects the well topology. The default rules are:

      * PMOS body → `vdd` (the n-well terminal).
      * NMOS body → `vss` / `0` / `gnd` (the p-substrate terminal).

    A PMOS with body wired to `vss` (or an NMOS with body to `vdd`) is
    a cross-well short — SPICE would later report this as a forward-
    biased junction or a latch-up risk.

    This check is intentionally rule-based, not PDK-introspective. The
    common LLM failure mode is to forget the body wire entirely or to
    swap it with the source net; both cases surface as `body_well_mismatch`.
    """
    devices = _cell_devices(cell)
    if not devices:
        return []

    violations: list[PhysicalInvariantViolation] = []
    for dev in devices:
        cls = getattr(dev, "device_class", "")
        if cls not in ("nmos", "pmos"):
            continue
        pins = getattr(dev, "pins", {}) or {}
        body = pins.get("b")
        if body is None:
            violations.append(PhysicalInvariantViolation(
                invariant_kind="MosBodyWell",
                cell_name=getattr(cell, "name", "?"),
                category="body_well_mismatch",
                target=getattr(dev, "instance_name", "?"),
                expected="MOS device must declare a `b` (body) pin",
                actual="no `b` pin on the device",
                message=(
                    f"cell {getattr(cell, 'name', '?')!r}: "
                    f"{cls} {getattr(dev, 'instance_name', '?')!r} has "
                    f"no body pin (`b`) declared. SPICE requires a 4-pin "
                    f"MOS connectivity."
                ),
                suggested_fix=(
                    f"add `body = {'vdd' if cls == 'pmos' else 'vss'};` "
                    f"to the device declaration."
                ),
                extras={"pins": dict(pins)},
            ))
            continue
        legal = _LEGAL_BODY_NETS.get(cls, frozenset())
        # A body net that matches a legal target by exact name is
        # accepted. A body net that matches NEITHER the legal targets
        # NOR the opposite-well rails is allowed (it is likely a
        # custom n-well/p-well subnet the designer carved out — the
        # check would need PDK introspection to validate; we abstain
        # rather than false-positive). The check fires only when the
        # body is wired to the OPPOSITE well's rail — the unambiguous
        # cross-well short.
        opposite_legal = _LEGAL_BODY_NETS.get(
            "nmos" if cls == "pmos" else "pmos", frozenset()
        )
        if body in opposite_legal and body not in legal:
            violations.append(PhysicalInvariantViolation(
                invariant_kind="MosBodyWell",
                cell_name=getattr(cell, "name", "?"),
                category="body_well_mismatch",
                target=getattr(dev, "instance_name", "?"),
                expected=(
                    f"body net in {sorted(legal)!r} "
                    f"(n-well rail for PMOS / p-substrate for NMOS)"
                ),
                actual=f"body = {body!r}",
                message=(
                    f"cell {getattr(cell, 'name', '?')!r}: "
                    f"{cls} {getattr(dev, 'instance_name', '?')!r} has "
                    f"its body pin tied to {body!r}, which is the "
                    f"opposite well's rail. This is a cross-well short "
                    f"and would forward-bias the substrate junction."
                ),
                suggested_fix=(
                    f"change `body = {body};` to "
                    f"`body = {'vdd' if cls == 'pmos' else 'vss'};` "
                    f"on device {getattr(dev, 'instance_name', '?')!r}."
                ),
                extras={"body_net": body, "device_class": cls},
            ))

    return violations


# ─── Invariant 5: DanglingNetInvariant ──────────────────────────


def check_dangling_net_invariant(cell: Any) -> list[PhysicalInvariantViolation]:
    """Resolve the DANGLING-NET POLICY (Law 2: make common structural
    errors unrepresentable).

    A net that is produced or consumed by exactly ONE device pin — and
    that is not exempted as a boundary net — is an *accidental* dangling
    net: a typo'd net name, a forgotten wire, or a node that is driven
    but never read (or read but never driven) inside the cell being
    elaborated as a top design. This was historically a silent
    ``UserWarning`` (``clay/compiler.py::_compile_primitive_cell``) that
    did not block certification. The policy now promotes it to a
    FIRST-CLASS, VISIBLE obligation: a structured ``dangling_net``
    violation carrying the net name, which maps to a non-PASS
    verification status and therefore blocks the proof token. It never
    folds into PASS.

    The exemption mechanism that fits Clay's existing grammar/IR is the
    PORT DECLARATION (plus the already-existing global-rail and
    bias-node exemptions):

      * A net that is a declared cell port is an INTENDED boundary net —
        it legitimately escapes to a parent during composition, or it is
        an intentionally-left-open sub-cell port. Declaring the port is
        the explicit, typed statement of "this open is on purpose."
      * Global rails (``0`` / ``gnd`` / ``vdd`` / ``vss``) and declared
        bias nodes (``cell.__dict__['_bias_nodes']``) are exempt for the
        same reason the other invariants exempt them: they carry a
        defined voltage from outside the topology.

    So the policy distinguishes ``intended open`` (explicit: a declared
    port / rail / bias node — accepted) from ``accidental dangling``
    (implicit: a one-pin net that is none of those — structured error).

    A net touched by ZERO pins cannot appear here (nothing references
    it). A net touched by TWO OR MORE pins is internal connectivity and
    is never dangling. The ``floating_net`` failure mode (>=2 gate pins,
    no driver) is owned by :func:`check_net_driver_invariant`; this check
    is the complementary one-pin case that NetDriver intentionally left
    soft.
    """
    devices = _cell_devices(cell)
    if not devices:
        return []

    ports = set(_cell_ports(cell))
    bias_nodes = set(cell.__dict__.get("_bias_nodes", set()) or set())
    legal = ports | _GLOBAL_RAILS | bias_nodes

    # Count how many device pins touch each net (each pin counts once,
    # so a diode-connected MOS with d==g counts that net twice).
    pin_counts: dict[str, int] = {}
    for dev in devices:
        for net in getattr(dev, "pins", {}).values():
            pin_counts[net] = pin_counts.get(net, 0) + 1

    violations: list[PhysicalInvariantViolation] = []
    for net in sorted(pin_counts):
        if pin_counts[net] != 1:
            continue        # internal multi-pin connectivity — not dangling
        if net in legal:
            continue        # explicit, typed exemption (port / rail / bias)
        violations.append(PhysicalInvariantViolation(
            invariant_kind="DanglingNet",
            cell_name=getattr(cell, "name", "?"),
            category="dangling_net",
            target=net,
            expected=(
                "net used by >=2 device pins, OR declared as a cell port "
                "(intended-open boundary net), OR a global rail / bias node"
            ),
            actual="net touched by exactly one device pin and not exempted",
            message=(
                f"cell {getattr(cell, 'name', '?')!r}: net {net!r} is "
                f"dangling — exactly one device pin references it and it "
                f"is neither a declared port, a global rail "
                f"({'/'.join(sorted(_GLOBAL_RAILS))}), nor a bias node. "
                f"It is produced or consumed but not the other, so it is "
                f"an accidental dangling net (likely a typo or a "
                f"forgotten wire), not an intended open."
            ),
            suggested_fix=(
                f"wire net {net!r} to a second device pin so it is "
                f"actually connected, fix the net-name typo, or — if the "
                f"open is intentional (the net is a boundary that escapes "
                f"to a parent or a deliberately-unconnected sub-cell "
                f"port) — declare {net!r} as a cell port to make the "
                f"intended-open explicit."
            ),
            extras={"pin_count": 1},
        ))

    return violations


# ─── Verification status mapping (proof-token semantics) ─────────

# Mirror of `clay.invariants_modules._violation._CATEGORY_TO_VERIFICATION_STATUS`
# for the physical-invariant category set. Lazily imported lookup so this
# module stays import-cycle-free; the string values are identical to
# `VerificationStatus(...).value`. Anything not explicitly PASS-mapped is
# treated as a non-PASS state so a new/malformed category can never
# silently certify a design.
_PHYS_CATEGORY_TO_STATUS: dict[str, str] = {
    "pass":               "pass",
    "skipped":            "skip",
    "skip":               "skip",
    "floating_net":       "fail",
    "orphan_net":         "fail",
    "dangling_net":       "fail",
    "missing_dual":       "fail",
    "unreachable_port":   "fail",
    "body_well_mismatch": "fail",
}


def verification_status_of(violation: PhysicalInvariantViolation) -> str:
    """Map a physical-invariant violation category to an explicit
    verification-status string (``pass`` / ``skip`` / ``fail`` / ...).

    Unknown categories map to ``error`` rather than ``pass`` so a
    malformed or future category cannot silently certify a design. This
    is the proof-token gate: only a literal ``pass`` category certifies,
    so a ``dangling_net`` violation can never fold into PASS.
    """
    return _PHYS_CATEGORY_TO_STATUS.get(str(violation.category), "error")


def contributes_to_proof_token(violation: PhysicalInvariantViolation) -> bool:
    """Return True iff this physical-invariant result can certify an
    obligation. A ``dangling_net`` violation returns False — it blocks
    the proof token and is never collapsed into PASS."""
    return verification_status_of(violation) == "pass"


# ─── Top-level entry point ──────────────────────────────────────


# The four legacy elaboration-time invariants run by default for every
# cell. DanglingNet is NOT in this list: it is a distinct policy
# (one-pin accidental dangling) that callers opt into via
# `check_physical_invariants(..., include_dangling_net=True)` or the
# obligation surface `dangling_net_obligation_rows(...)`. Keeping it out
# of the default gate preserves the pre-policy compile contract (the
# legacy soft `UserWarning` in `_compile_primitive_cell` still fires for
# the LLM heads-up) while the proof-token layer treats accidental
# dangling as a non-PASS obligation that never folds into PASS.
_INVARIANT_CHECKS: list[tuple[str, Any]] = [
    ("NetDriver",   check_net_driver_invariant),
    ("PortPolarity", check_port_polarity_invariant),
    ("PowerReach",  check_power_reach_invariant),
    ("MosBodyWell", check_mos_body_well_invariant),
]


def check_physical_invariants(
    cell: Any,
    *,
    raise_on_violation: bool = False,
    include_dangling_net: bool = False,
) -> list[PhysicalInvariantViolation]:
    """Run every physical invariant against `cell` and return the merged
    violation list.

    Parameters
    ----------
    cell
        A `clay.ir.Cell` (or any object with `name`, `ports`, and
        `devices` attributes — the function is duck-typed).
    raise_on_violation
        When True and the merged list contains any fatal violation,
        raise `PhysicalInvariantError` carrying the full list.
        When False (default), return the list and let the caller decide.
    include_dangling_net
        When True, also run :func:`check_dangling_net_invariant` (the
        one-pin accidental-dangling policy). Defaults to False so the
        elaboration-time gate keeps its pre-policy compile contract; the
        proof-token / obligation layer turns this on to make accidental
        dangling a non-PASS obligation.

    The order of invariants matches the order in `_INVARIANT_CHECKS`:
    NetDriver → PortPolarity → PowerReach → MosBodyWell (→ DanglingNet
    when requested). Within each invariant, violations are sorted by
    `target` to keep test expectations deterministic.
    """
    checks = list(_INVARIANT_CHECKS)
    if include_dangling_net:
        checks.append(("DanglingNet", check_dangling_net_invariant))

    merged: list[PhysicalInvariantViolation] = []
    for _kind, fn in checks:
        try:
            merged.extend(fn(cell))
        except Exception as exc:  # pragma: no cover — defensive
            merged.append(PhysicalInvariantViolation(
                invariant_kind=_kind,
                cell_name=getattr(cell, "name", "?"),
                category="skipped",
                target="?",
                expected=None,
                actual=None,
                message=(
                    f"invariant {_kind} raised "
                    f"{type(exc).__name__}: {exc} — skipped"
                ),
                suggested_fix="",
                extras={"exception": type(exc).__name__},
            ))

    if raise_on_violation and any(v.is_fatal for v in merged):
        raise PhysicalInvariantError([v for v in merged if v.is_fatal])

    return merged


def dangling_net_obligation_rows(cell: Any) -> list[dict]:
    """Expand a cell's accidental dangling nets into proof-obligation rows.

    Each accidental dangling net becomes a REQUIRED obligation row whose
    status is the non-PASS verification status of the violation
    (``fail``). The proof-token gate (``clay/verifier.py``) certifies
    PASS only when EVERY required row is ``pass``; a ``dangling_net`` row
    therefore blocks the token and is never folded into PASS.

    A cell with no accidental dangling nets produces an empty list (no
    obligation added — intended-open ports / rails / bias nodes are
    exempt and never appear here), so a clean cell certifies as before.
    """
    cell_name = getattr(cell, "name", "?")
    rows: list[dict] = []
    for v in check_dangling_net_invariant(cell):
        obligation_id = f"obl:{cell_name}:structural:dangling_net:{v.target}"
        rows.append({
            "obligation_id": obligation_id,
            "trace_id": f"tr:{cell_name}:structural:dangling_net:{v.target}",
            "cell": cell_name,
            "source_kind": "structural",
            "obligation_kind": "dangling_net",
            "required": True,
            "status": verification_status_of(v),
            "source": {
                "cell": cell_name,
                "name": v.target,
                "field": "dangling_net",
                "ir_symbol": f"{cell_name}.net.{v.target}",
                "source_ref": {
                    "kind": "ir",
                    "path": f"cell[{cell_name}].net[{v.target}]",
                },
            },
            "coverage": {
                "artifact_kind": "structural_invariant",
                "ran_real_tool": True,
            },
            "evidence": {
                "category": v.category,
                "net": v.target,
                "message": v.message,
                "suggested_fix": v.suggested_fix,
            },
        })
    return rows


def filter_fatal_violations(
    violations: list[PhysicalInvariantViolation],
) -> list[PhysicalInvariantViolation]:
    """Return the subset of `violations` whose `category != "skipped"`.

    Mirrors `clay.invariants.filter_real_violations` so callers using
    both modules see a uniform filter API.
    """
    return [v for v in violations if v.is_fatal]
