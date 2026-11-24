"""Semantic faithfulness verifier for the EDALLM -> clean-room core desugaring.

``to_core.emit_cell`` turns a ``clay.ir.Cell`` into clean-room core text. Today
the only post-hoc check that the emitted text actually *encodes* the source cell
is a device COUNT comparison (regen_examples guards ``n_leaf >= 1``). A count
match is a weak oracle: a leaf could be dropped and another duplicated, a net
could be renamed, a bind could point at the wrong net, all while the count holds.

``verify_faithful(cell, core_text)`` is the stronger oracle. It re-parses the
emitted core text structurally (NOT via the Lean kernel -- this is a pure-Python
shadow parse of the surface the emitter is contractually committed to) and checks
that it is a STRUCTURALLY FAITHFUL encoding of the source ``cell``:

  C-PORTS:  the core's top-level ``interface`` ports == the source cell ports,
            as a SET of (name, role) where role is compared modulo the documented
            7->5 EDALLM-role -> kernel-role projection (``to_core._role_kw`` /
            ``_ROLE_MAP``). i.e. the verifier applies the SAME projection to the
            source side, then asks for set equality. (A pure-leaf emit -- one with
            no devices/instances -- uses ``leaf interface`` instead of ``node``;
            both are accepted.)

  C-NETS:   the core's ``nets { ... }`` set == the source cell's net set, where the
            source net set = (sanitized ports) U (sanitized device-pin nets) U
            (sanitized child bind nets). This is the SAME union ``_collect_nets``
            builds, so a dropped or renamed net in the core is caught. (A pure-leaf
            core has no ``nets {}`` block; C-NETS is vacuous there and skipped.)

  C-BINDS:  every ``inst leaf ... model <m> bind { p = net, ... }`` in the core maps
            to EXACTLY ONE source ``Device``, matched by its full pin->net binding
            set (sanitized), and the model literal agrees with the device's
            ``class[/variant]``. A planted bind that drops a pin, renames a net, or
            points at the wrong net fails to match any device and is reported. Each
            source device must be consumed exactly once (bijection), so a duplicated
            or dropped leaf is caught even when the gross count is unchanged.

  C-COUNT:  number of ``inst leaf`` in the core == number of source devices. (The
            bijection in C-BINDS already implies this, but it is reported as its own
            discrepancy class for a precise count delta.)

Scope / honest non-goals:
  * Child (hierarchical) ``inst <node|leaf> ... bind`` instances are NOT verified
    pin-for-pin against a source sub-cell here -- the verifier only asserts the
    DEVICE leaves are faithful. Nested-node binds are tolerated (their RHS nets are
    still folded into the source net set for C-NETS). Verifying inlined sub-cells
    recursively is future work; we flag presence of nested nodes in ``info`` rather
    than silently passing them as faithful.
  * Constraints / intent text are advisory below C1 and are NOT checked here.
  * The emitter's own escaping/role-inference is the reference: the verifier reuses
    ``to_core``'s ``_sanitize_ident`` / ``_role_kw`` / ``_pin_role`` so it tests
    "did the emitter encode THIS cell", not "does the emitter agree with a second,
    independently-bugged reimplementation".

Returns ``(ok: bool, discrepancies: list[str])``. ``ok`` is True iff
``discrepancies`` is empty. Each discrepancy string is prefixed with its check id
(``C-PORTS`` / ``C-NETS`` / ``C-BINDS`` / ``C-COUNT``) for grep-ability.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from clay.elaborate.to_core import (
    _sanitize_ident,
    _role_kw,
    _pin_role,
    _port_roles_of,
)


# ─────────────────────────────────────────────────────────────────────
# Shadow parse of the emitted core surface.
#
# We do NOT need a full grammar: we extract just the three structural facts
# the four checks consume -- the top interface ports, the nets{} set, and the
# list of leaf-instance bind maps. The emitter's surface is regular enough
# (one inst-leaf per `inst leaf ... bind { ... }`, single-line binds) that a
# brace-balanced scan + targeted regex is exact for what `to_core` emits.
# ─────────────────────────────────────────────────────────────────────


_ROLE_KW = ("in", "out", "supply", "bias", "clock")
# A port line inside an interface block: `<role> <ident>` (trailing comma stripped).
_PORT_RE = re.compile(r"^\s*(in|out|supply|bias|clock)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*,?\s*$")
# A leaf instance head: `inst leaf interface { ... } model <lit> bind { ... }`.
# model literal is a bare ident or a double-quoted string.
_LEAF_RE = re.compile(
    r"inst\s+leaf\s+interface\s*\{(?P<iface>[^}]*)\}\s*model\s+"
    r"(?P<model>\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_./-]*)\s*"
    r"bind\s*\{(?P<bind>[^}]*)\}"
)
# One `p = net` pair inside a bind block.
_BIND_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*([A-Za-z_][A-Za-z0-9_.]*)")


def _strip_header_and_constraints(core_text: str) -> str:
    """Drop the `# desugared ...` comment header (irrelevant to structure)."""
    lines = [ln for ln in core_text.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines)


def _find_matching_brace(s: str, open_idx: int) -> int:
    """Index of the `}` matching the `{` at ``open_idx``. -1 if unbalanced."""
    depth = 0
    for i in range(open_idx, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _top_interface_ports(core_text: str) -> Optional[list[tuple[str, str]]]:
    """Extract the TOP-LEVEL cell interface as [(role, name), ...].

    The top cell is the first token group: ``node interface { ... }`` or
    ``leaf interface { ... }`` at the start of the (header-stripped) text. We
    take the FIRST ``interface {`` that follows a leading ``node``/``leaf`` token
    so we don't pick up an inner instance's interface.
    """
    body = _strip_header_and_constraints(core_text).lstrip()
    m = re.match(r"(node|leaf)\b", body)
    if not m:
        return None
    # locate the first `interface {` after the kind token
    iface_kw = body.find("interface", m.end())
    if iface_kw < 0:
        return None
    open_idx = body.find("{", iface_kw)
    if open_idx < 0:
        return None
    close_idx = _find_matching_brace(body, open_idx)
    if close_idx < 0:
        return None
    inner = body[open_idx + 1 : close_idx]
    ports: list[tuple[str, str]] = []
    # interface bodies emitted by to_core are one port per line; a pure-leaf
    # node uses `, ` joins on a single line too -- split on commas then lines.
    for chunk in inner.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pm = _PORT_RE.match(chunk) or re.match(
            r"^(in|out|supply|bias|clock)\s+([A-Za-z_][A-Za-z0-9_.]*)$", chunk
        )
        if pm:
            ports.append((pm.group(1), pm.group(2)))
    return ports


def _nets_set(core_text: str) -> Optional[set[str]]:
    """Extract the top composition's ``nets { ... }`` set. None if no nets block."""
    body = _strip_header_and_constraints(core_text)
    idx = body.find("nets")
    if idx < 0:
        return None
    open_idx = body.find("{", idx)
    if open_idx < 0:
        return None
    close_idx = _find_matching_brace(body, open_idx)
    if close_idx < 0:
        return None
    inner = body[open_idx + 1 : close_idx]
    return {
        t.strip()
        for t in inner.replace("\n", ",").split(",")
        if t.strip()
    }


def _leaf_binds(core_text: str) -> list[tuple[str, dict[str, str]]]:
    """All leaf instances as [(model_literal, {pin: net}), ...]."""
    body = _strip_header_and_constraints(core_text)
    out: list[tuple[str, dict[str, str]]] = []
    for m in _LEAF_RE.finditer(body):
        model = m.group("model")
        if model.startswith('"') and model.endswith('"'):
            model = model[1:-1]
        binds = {p: n for p, n in _BIND_PAIR_RE.findall(m.group("bind"))}
        out.append((model, binds))
    return out


def _count_nested_nodes(core_text: str) -> int:
    """Count `inst node` (inlined sub-cell) heads -- reported, not verified."""
    return len(re.findall(r"\binst\s+node\b", _strip_header_and_constraints(core_text)))


# ─────────────────────────────────────────────────────────────────────
# Source-side derivations (reuse the EMITTER's own helpers so we test the
# emitter against THIS cell, not against a second reimplementation).
# ─────────────────────────────────────────────────────────────────────


def _source_ports(cell: Any) -> set[tuple[str, str]]:
    """Source cell ports as {(kernel_role, sanitized_name)}."""
    roles = _port_roles_of(cell)
    out: set[tuple[str, str]] = set()
    for p in getattr(cell, "ports", []) or []:
        out.add((_role_kw(p, roles.get(p)), _sanitize_ident(p)))
    return out


def _source_device_binds(cell: Any) -> list[dict[str, str]]:
    """Each source Device as {sanitized_pin: sanitized_net}."""
    out: list[dict[str, str]] = []
    for dev in getattr(cell, "devices", []) or []:
        pins = dict(getattr(dev, "pins", {}) or {})
        out.append({_sanitize_ident(p): _sanitize_ident(n) for p, n in pins.items()})
    return out


def _source_device_model(dev: Any) -> str:
    """Reconstruct the model literal the emitter would emit for a device."""
    cls = str(getattr(dev, "device_class", "device"))
    variant = getattr(dev, "pdk_variant", None)
    return f"{cls}/{variant}" if variant else cls


def _source_net_set(cell: Any) -> set[str]:
    """Source net set = sanitized ports U device-pin nets U child bind nets.

    Mirrors ``to_core._collect_nets`` so a faithful emit's nets{} matches exactly.
    """
    nets: set[str] = set()
    for p in getattr(cell, "ports", []) or []:
        nets.add(_sanitize_ident(p))
    for dev in getattr(cell, "devices", []) or []:
        for net in (getattr(dev, "pins", {}) or {}).values():
            nets.add(_sanitize_ident(net))
    for inst in getattr(cell, "instances", []) or []:
        for outernet in (getattr(inst, "port_map", {}) or {}).values():
            nets.add(_sanitize_ident(outernet))
    return nets


# ─────────────────────────────────────────────────────────────────────
# The verifier.
# ─────────────────────────────────────────────────────────────────────


def verify_faithful(cell: Any, core_text: str) -> tuple[bool, list[str]]:
    """Check ``core_text`` is a structurally faithful encoding of ``cell``.

    Returns ``(ok, discrepancies)``. ``ok`` iff ``discrepancies == []``. Each
    discrepancy is prefixed with its check id. See module docstring for the
    contract of each check.
    """
    disc: list[str] = []

    # ---- C-PORTS ----------------------------------------------------------
    core_ports_list = _top_interface_ports(core_text)
    if core_ports_list is None:
        disc.append("C-PORTS: could not locate a top-level interface block in core text")
        core_ports: set[tuple[str, str]] = set()
    else:
        core_ports = set(core_ports_list)
        if len(core_ports_list) != len(core_ports):
            # duplicate port lines in the interface
            seen: set[tuple[str, str]] = set()
            dups = {x for x in core_ports_list if x in seen or seen.add(x)}  # type: ignore[func-returns-value]
            disc.append(f"C-PORTS: duplicate interface port(s) in core: {sorted(dups)}")
    src_ports = _source_ports(cell)
    if core_ports_list is not None and core_ports != src_ports:
        missing = src_ports - core_ports
        extra = core_ports - src_ports
        if missing:
            disc.append(f"C-PORTS: source ports missing from core (role,name): {sorted(missing)}")
        if extra:
            disc.append(f"C-PORTS: core has ports not in source (role,name): {sorted(extra)}")

    # ---- C-NETS -----------------------------------------------------------
    core_nets = _nets_set(core_text)
    src_nets = _source_net_set(cell)
    has_structure = bool(getattr(cell, "devices", []) or []) or bool(
        getattr(cell, "instances", []) or []
    )
    if core_nets is None:
        if has_structure:
            disc.append("C-NETS: core has structure (devices/instances) but no nets{} block")
        # else: pure-leaf emit -> no nets block expected; C-NETS vacuous.
    else:
        if core_nets != src_nets:
            missing = src_nets - core_nets
            extra = core_nets - src_nets
            if missing:
                disc.append(f"C-NETS: source nets missing from core nets{{}}: {sorted(missing)}")
            if extra:
                disc.append(f"C-NETS: core nets{{}} has nets not in source: {sorted(extra)}")

    # ---- C-BINDS + C-COUNT ------------------------------------------------
    core_leaves = _leaf_binds(core_text)
    src_devices = list(getattr(cell, "devices", []) or [])
    src_binds = _source_device_binds(cell)

    # C-COUNT (own discrepancy class)
    if len(core_leaves) != len(src_devices):
        disc.append(
            f"C-COUNT: inst-leaf count {len(core_leaves)} != source device count "
            f"{len(src_devices)}"
        )

    # C-BINDS: bijective match core-leaf <-> source-device on full pin->net map.
    unmatched_src_idx = set(range(len(src_devices)))
    for li, (model, binds) in enumerate(core_leaves):
        match_idx = None
        for si in unmatched_src_idx:
            if binds == src_binds[si]:
                match_idx = si
                break
        if match_idx is None:
            # Was there a source device with this exact bind map at all (already
            # consumed)? Distinguish "no such device" from "duplicate".
            any_same = any(binds == src_binds[si] for si in range(len(src_devices)))
            if any_same:
                disc.append(
                    f"C-BINDS: core leaf #{li} (model={model!r}) duplicates a source "
                    f"device binding already consumed: {binds}"
                )
            else:
                disc.append(
                    f"C-BINDS: core leaf #{li} (model={model!r}) has bind {binds} that "
                    f"matches NO source device"
                )
            continue
        unmatched_src_idx.discard(match_idx)
        # model-literal agreement (class[/variant]) on the matched device
        want_model = _source_device_model(src_devices[match_idx])
        if model != want_model:
            disc.append(
                f"C-BINDS: core leaf #{li} model {model!r} != source device "
                f"{getattr(src_devices[match_idx], 'instance_name', '?')!r} model "
                f"{want_model!r}"
            )
    for si in sorted(unmatched_src_idx):
        dev = src_devices[si]
        disc.append(
            f"C-BINDS: source device "
            f"{getattr(dev, 'instance_name', '?')!r} (bind {src_binds[si]}) has NO "
            f"matching core leaf"
        )

    return (len(disc) == 0, disc)


def faithful_info(cell: Any, core_text: str) -> dict[str, Any]:
    """Non-judgmental structural summary -- useful for the sweep harness/report.

    Reports counts and the nested-node presence the verifier does NOT pin-verify,
    so a caller can surface "passed-but-has-nested-nodes" honestly.
    """
    ok, disc = verify_faithful(cell, core_text)
    return {
        "ok": ok,
        "n_discrepancies": len(disc),
        "discrepancies": disc,
        "n_source_devices": len(getattr(cell, "devices", []) or []),
        "n_source_instances": len(getattr(cell, "instances", []) or []),
        "n_core_inst_leaf": len(_leaf_binds(core_text)),
        "n_core_inst_node": _count_nested_nodes(core_text),
        "n_source_ports": len(getattr(cell, "ports", []) or []),
    }
