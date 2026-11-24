"""Intent net-binding checker: every declared layout-intent reference must BIND.

A ``@symmetric(X, Y)`` / ``@match(X, Y)`` / ``@mirror(X, Y)`` / ``@shield(A -> V,
S)`` / ``@budget(kind, X, Y, tol)`` declaration names NETS; an ``@equal_*`` /
``@place_group`` / ``@abut`` declaration names DEVICES, while ``@order`` may
name either devices or declared functional groups. When a
declaration references a net (or device) that does NOT exist in the cell's
parsed topology, the downstream lowering silently turns it into a no-op: the
diff-pair / budget is built over a net that is never routed, never mirrored,
never extracted -- so the intent is SILENTLY DROPPED with no verdict. That
silent no-op is the deepest "never silent pass" violation: a cell can run with
broken intent and still report a clean DRC, because the broken half simply does
nothing.

This module makes the silent no-op a NAMED verdict. For a parsed cell whose
topology net/device set is FULLY DETERMINED from the parse (it declares explicit
``device`` entries and is not driven by a later synthesis step), every
net/device a declaration references MUST appear in that set; an unbound
reference yields a RED (``unsatisfiable``) verdict and -- when wired into the
lowering -- raises :class:`IntentNetBindingError` rather than dropping the
intent.

Five-state honesty (no silent pass):
  * ``pass``          -- every reference binds to a real net/device.
  * ``fail``          -- at least one reference names an absent net/device
                         (the silent-no-op bug); ``unbound`` lists them.
  * ``unmeasurable``  -- the cell's net/device set is NOT determinable from the
                         parse (it has no explicit ``device`` entries and is
                         built by a later synthesis path -- a stage ``style``
                         or sub-cell ``instances`` -- so the
                         internal nodes the declarations reference materialise
                         only after compilation). Binding cannot be honestly
                         decided here, so the verdict WITHHOLDS a pass rather
                         than fabricating one. This is NOT a silent pass: the
                         caller sees ``status == "unmeasurable"`` and the
                         lowering does not raise.

The discriminator is keyed PURELY on parsed structure (does the cell carry
explicit ``device`` pins?), never on the circuit name, the PDK, or a net label.
A 13-device explicit-topology comparator is checkable on every PDK; a
functional-role-synthesised OTA is unmeasurable on every PDK. No per-circuit
branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from clay.ir import CellDef, ParsedLayoutIntentDecl

# Declaration kinds whose ``nets`` / aggressor / victim / shield_net name NETS
# (must bind to the cell's net set). ``shield_net`` is allowed to be a supply
# rail that is also a real cell net; it is checked against the net set like any
# other net reference (a rail net IS a topology net here -- it appears on device
# bodies/sources).
_NET_KINDS = frozenset({"symmetric", "match", "mirror", "shield", "budget", "route_tail", "trunk"})

# Declaration kinds whose members name DEVICE labels (must bind to the device
# instance-name set), not nets: the per-device LDE matching budgets and the
# device-placement directives.
_DEVICE_KINDS = frozenset(
    {
        "equal_well_edge",
        "equal_od",
        "equal_od_spacing",
        "lde_budget",
        "place_group",
        "align",
        "terminal_access",
        "abut",
        "passive_bank_partition",
    }
)


@dataclass(frozen=True)
class UnboundReference:
    """One declared reference that binds to no net/device in the topology."""

    kind: str          # the declaration kind ("symmetric", "budget", ...)
    name: str          # the referenced net or device label that is absent
    domain: str        # "net", "device", or "device_or_group"


@dataclass(frozen=True)
class IntentBindingVerdict:
    """The result of checking that every declared reference binds to topology.

    ``status`` is one of ``pass`` / ``fail`` / ``unmeasurable`` (a strict subset
    of the project's 5-state vocabulary -- this checker never produces
    ``unsupported`` / ``advisory``). ``unbound`` lists every reference that
    failed to bind (empty unless ``status == "fail"``). ``net_universe`` /
    ``device_universe`` are the determined sets the references were checked
    against (empty when ``unmeasurable``).
    """

    cell: str
    status: str
    unbound: Tuple[UnboundReference, ...]
    net_universe: Tuple[str, ...]
    device_universe: Tuple[str, ...]
    reason: str = ""

    @property
    def is_red(self) -> bool:
        """True iff a declared reference is unbound (a silent-no-op bug)."""
        return self.status == "fail"

    @property
    def unmeasurable(self) -> bool:
        return self.status == "unmeasurable"

    def describe(self) -> str:
        if self.status == "pass":
            return f"intent net-binding PASS for {self.cell!r}: all references bind."
        if self.status == "unmeasurable":
            return (
                f"intent net-binding UNMEASURABLE for {self.cell!r}: {self.reason}"
            )
        pretty = ", ".join(f"{u.kind}:{u.name} ({u.domain})" for u in self.unbound)
        return (
            f"intent net-binding FAIL for {self.cell!r}: declared reference(s) "
            f"name no topology {'net' if all(u.domain=='net' for u in self.unbound) else 'net/device'}: "
            f"{pretty}. net universe = {sorted(self.net_universe)}; "
            f"device universe = {sorted(self.device_universe)}."
        )


class IntentNetBindingError(ValueError):
    """Raised at intent lowering when a declared reference binds to no topology.

    Carries the full :class:`IntentBindingVerdict` so a caller can inspect every
    unbound reference. This is the NAMED RED verdict that replaces the old silent
    no-op: a mis-named ``@symmetric`` / ``@budget`` net is an unsatisfiable
    intent, surfaced explicitly rather than dropped.
    """

    def __init__(self, verdict: IntentBindingVerdict) -> None:
        self.verdict = verdict
        super().__init__(verdict.describe())


def topology_net_universe(cd: CellDef) -> Optional[Set[str]]:
    """The set of nets a parsed cell's topology determines, or ``None``.

    Returns the union of the cell's declared ports and every net appearing on a
    parsed ``device`` pin, IFF that set is FULLY DETERMINED from the parse: the
    cell declares explicit ``device`` entries and is not built by a later
    stage-style or sub-cell synthesis path that would introduce internal nodes
    not visible here.

    Returns ``None`` (the universe is not determinable -> the binding check is
    UNMEASURABLE) when the cell has no explicit ``device`` entries, or when it is
    driven by a stage style / instances -- in those cases the
    nets the declarations reference materialise only after compilation, so the
    parse-time net set is not the real net set and a binding verdict would be
    fabricated.
    """
    devices = list(getattr(cd, "devices", []) or [])
    if not devices:
        return None
    # Synthesis-driven cells introduce internal nodes post-parse; their parse-
    # time net set is not authoritative. Functional-group metadata does not
    # weaken an otherwise explicit `device` topology.
    if getattr(cd, "instances", None):
        return None
    if (getattr(cd, "topology", None) or {}).get("style"):
        return None

    nets: Set[str] = set(str(p) for p in (getattr(cd, "ports", None) or []))
    for d in devices:
        pins = getattr(d, "pins", None) or {}
        for net in pins.values():
            if net is not None:
                nets.add(str(net))
    return nets


def topology_device_universe(cd: CellDef) -> Optional[Set[str]]:
    """The set of device instance labels a parsed cell determines, or ``None``.

    Like :func:`topology_net_universe`, this is determinable only for an
    explicit-``device`` cell; a synthesis-driven cell's device labels are
    assigned during compilation, so the device universe is ``None``
    (UNMEASURABLE) there.
    """
    devices = list(getattr(cd, "devices", []) or [])
    if not devices:
        return None
    if getattr(cd, "instances", None):
        return None
    if (getattr(cd, "topology", None) or {}).get("style"):
        return None
    labels: Set[str] = set()
    for d in devices:
        name = getattr(d, "instance_name", None)
        if name is not None:
            labels.add(str(name))
    return labels


def _net_refs(decl: ParsedLayoutIntentDecl) -> List[str]:
    """Every NET name a net-domain declaration references."""
    refs: List[str] = [str(n) for n in (getattr(decl, "nets", None) or ())]
    agg = getattr(decl, "aggressor", None)
    vic = getattr(decl, "victim", None)
    shield_net = getattr(decl, "shield_net", None)
    if decl.kind == "shield":
        # A shield names an aggressor net, a victim net, and a shield net.
        if agg is not None:
            refs.append(str(agg))
        if vic is not None:
            refs.append(str(vic))
        if shield_net is not None:
            refs.append(str(shield_net))
    elif decl.kind == "budget":
        from .intent import parasitic_budget_driver_attributes

        drivers = parasitic_budget_driver_attributes(decl)
        refs.extend(drivers.get("aggressors", ()))
        if "clock_net" in drivers:
            refs.append(drivers["clock_net"])
    elif decl.kind == "trunk":
        net = (getattr(decl, "attributes", None) or {}).get("net")
        if net:
            refs.append(str(net))
    return refs


def _device_refs(decl: ParsedLayoutIntentDecl) -> List[str]:
    """Every DEVICE label a device-domain declaration references."""
    if decl.kind == "place_group":
        return [str(d) for d in (getattr(decl, "devices", None) or ())]
    if decl.kind == "passive_bank_partition":
        attrs = getattr(decl, "attributes", None) or {}
        return [str(d) for d in (attrs.get("members") or ())]
    # equal_* / lde_budget / order / abut carry their endpoint labels on `nets`.
    return [str(n) for n in (getattr(decl, "nets", None) or ())]


def check_intent_net_binding(cd: CellDef) -> IntentBindingVerdict:
    """Check that every declared layout-intent reference binds to the topology.

    For a cell whose net/device universe is determined from the parse, RED
    (``fail``) iff any net-domain declaration references a net absent from the
    net universe, or any device-domain declaration references a device absent
    from the device universe. ``pass`` when every reference binds.
    ``unmeasurable`` when the universe is not determinable (synthesis-driven
    cell) -- never a silent pass.
    """
    cell = getattr(cd, "local_name", "")
    decls: Sequence[ParsedLayoutIntentDecl] = (
        getattr(cd, "layout_intents", None) or []
    )

    net_universe = topology_net_universe(cd)
    device_universe = topology_device_universe(cd)

    if not decls:
        # No declarations to bind -> trivially nothing unbound. A cell with no
        # layout intent is a legitimate pass (there is no silent-dropped intent).
        return IntentBindingVerdict(
            cell=cell,
            status="pass",
            unbound=(),
            net_universe=tuple(sorted(net_universe or ())),
            device_universe=tuple(sorted(device_universe or ())),
        )

    if net_universe is None or device_universe is None:
        return IntentBindingVerdict(
            cell=cell,
            status="unmeasurable",
            unbound=(),
            net_universe=(),
            device_universe=(),
            reason=(
                "the cell declares no explicit `device` topology (it is built "
                "by functional roles / a stage style / sub-cell instances), so "
                "the nets/devices its declarations reference materialise only "
                "after compilation -- binding cannot be decided from the parse."
            ),
        )

    unbound: List[UnboundReference] = []
    order_decls = tuple(decl for decl in decls if decl.kind == "order")
    if order_decls:
        from clay.frontend.craft_infer import group_members

        declared_place_groups = {
            str(getattr(decl, "group", "")).strip()
            for decl in decls
            if decl.kind == "place_group"
            and str(getattr(decl, "group", "")).strip()
        }
        order_endpoints = (
            set(device_universe)
            | set(group_members(cd))
            | declared_place_groups
        )
    else:
        order_endpoints = set(device_universe)
    device_pins = {
        str(getattr(device, "instance_name", "")): {
            str(net) for net in (getattr(device, "pins", None) or {}).values()
        }
        for device in (getattr(cd, "devices", None) or ())
    }
    for decl in decls:
        kind = decl.kind
        if kind == "channel_route":
            attrs = decl.attributes or {}
            for name in tuple(attrs.get("nets", ())) + tuple(attrs.get("centered_nets", ())):
                if name not in net_universe:
                    unbound.append(UnboundReference(kind, str(name), "net"))
            group_names = {item.group for item in decls if item.kind == "place_group"}
            for name in attrs.get("between", ()):
                if name not in group_names:
                    unbound.append(UnboundReference(kind, str(name), "placement_group"))
        elif kind == "channel_branch":
            for name in decl.nets:
                if name not in net_universe:
                    unbound.append(UnboundReference(kind, str(name), "net"))
        elif kind in _NET_KINDS:
            for name in _net_refs(decl):
                if name and name not in net_universe:
                    unbound.append(UnboundReference(kind, name, "net"))
        elif kind in _DEVICE_KINDS:
            for name in _device_refs(decl):
                if name and name not in device_universe:
                    unbound.append(UnboundReference(kind, name, "device"))
        elif kind == "order":
            for name in _device_refs(decl):
                if name and name not in order_endpoints:
                    unbound.append(UnboundReference(kind, name, "device_or_group"))
        elif kind == "guard_ring":
            # A guard ring references BOTH domains: the member group names DEVICES
            # (must bind to the device universe -- a mis-named member would leave
            # that device un-ringed with no verdict) and the optional `tie` names a
            # NET (must bind to the net universe -- a mis-named tie would strap the
            # ring to a phantom net). Both are checked so neither reference silently
            # no-ops. The members ride on `attributes['members']` (fall back to
            # `nets`); the tie on `attributes['tie']`.
            attrs = getattr(decl, "attributes", None) or {}
            members = attrs.get("members") or getattr(decl, "nets", None) or ()
            for name in members:
                name = str(name)
                if name and name not in device_universe:
                    unbound.append(UnboundReference(kind, name, "device"))
            tie = str(attrs.get("tie") or "")
            if tie and tie not in net_universe:
                unbound.append(UnboundReference(kind, tie, "net"))
        elif kind == "kelvin":
            attrs = getattr(decl, "attributes", None) or {}
            for name in (attrs.get("force_net"), attrs.get("sense_net")):
                name = str(name or "")
                if name and name not in net_universe:
                    unbound.append(UnboundReference(kind, name, "net"))
            target = str(attrs.get("target_device") or "")
            if target and target not in device_universe:
                unbound.append(UnboundReference(kind, target, "device"))
            elif target:
                required_nets = {
                    str(attrs.get("force_net") or ""),
                    str(attrs.get("sense_net") or ""),
                }
                if not required_nets.issubset(device_pins.get(target, set())):
                    unbound.append(UnboundReference(kind, target, "device"))
        # Kinds with no net/device binding requirement (objective, area,
        # priority, superiority) carry no topology reference -> nothing to bind.

    # Deduplicate while preserving first-seen order (a net restated across a
    # @symmetric + @budget reports once).
    seen: Set[Tuple[str, str, str]] = set()
    deduped: List[UnboundReference] = []
    for u in unbound:
        key = (u.kind, u.name, u.domain)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)

    status = "fail" if deduped else "pass"
    return IntentBindingVerdict(
        cell=cell,
        status=status,
        unbound=tuple(deduped),
        net_universe=tuple(sorted(net_universe)),
        device_universe=tuple(sorted(device_universe)),
    )


def assert_intent_net_binding(cd: CellDef) -> IntentBindingVerdict:
    """Run the binding check and RAISE on a RED verdict.

    Returns the verdict (for ``pass`` / ``unmeasurable``); raises
    :class:`IntentNetBindingError` on ``fail`` so the silent no-op at lowering
    becomes a hard, named verdict. The lowering calls this so a mis-named
    ``@symmetric`` / ``@budget`` net stops being silently dropped.
    """
    verdict = check_intent_net_binding(cd)
    if verdict.is_red:
        raise IntentNetBindingError(verdict)
    return verdict
