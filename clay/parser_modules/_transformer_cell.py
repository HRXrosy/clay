"""
Cell + topology transformer methods (v0.1.13 parser split).

Mixin used by `_ClayTransformer` (see `_ast_builder.py`). Holds
`cell_def`, `port_list`, `port`, `port_role`, `topology_block`,
`topology_field`, `instance_decl`, and `port_map_field`.

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).
"""

from __future__ import annotations

import lark

from ..ir import (
    BiasNode, CellDef, CellInstance, Device, ParsedCombinatorBlock,
    ParsedFunctionalRole, RoleDecl, Spec,
)
from ._constants import _VALID_PORT_ROLES
from .role_expand import _expand_roles_to_devices


class _CellMixin:
    """Cell + topology transformer methods."""

    # ─── v0.0.19: in-source cell_def parsing ─────────────────

    @lark.v_args(meta=True)
    def cell_def(self, meta, items):
        """items: [CNAME, port_list, body_part1, body_part2, ...]
        Each body_part is one of topology_block (tagged tuple), spec_block
        (Spec object), cell_sizing_block (tagged tuple), corners_block
        (tagged tuple). spec_block is not tagged because it's shared
        with verify_block — detect by type.

        v0.0.21: corners declared inside cell_def attach to the
        eventual VerifyJob via the verify_ref resolution step in
        `start()`; CellDef itself doesn't get a corners field so its
        post-compile shape stays minimal.

        v0.0.22 (G10): port_list now yields a list of (name, role)
        tuples (role may be None). We split that into a plain
        `ports: list[str]` (for the existing compile_cell_def shape
        check) and a `port_roles: dict[str, str | None]` (new
        metadata channel). Bare port_lists produce all-None roles.

        v0.0.26 (G2): `bias_tree { ... }` body part attaches as
        `CellDef.bias_tree: list[BiasNode]`. Validation (no forward
        refs / cycles / unknown kinds) happens in the bias_tree_block
        transformer so any error surfaces at parse time.
        """
        local_name = str(items[0])
        # port_list returns list of (name, role) tuples
        port_entries: list[tuple[str, str | None]] = items[1]
        port_names = [name for name, _role in port_entries]
        port_roles_map = {name: role for name, role in port_entries}
        topology: dict = {}
        instances: list[CellInstance] = []
        devices: list[Device] = []
        role_decls: list[RoleDecl] = []
        functional_roles: list[ParsedFunctionalRole] = []
        spec_obj: Spec | None = None
        sizing_kw: dict[str, float] = {}
        corners: list[str] | None = None
        temps: list[float] | None = None
        bias_tree: list[BiasNode] = []
        combinator_block_obj: ParsedCombinatorBlock | None = None
        for body in items[2:]:
            # spec_block returns Spec directly (shared with verify_block)
            if isinstance(body, Spec):
                spec_obj = body
                continue
            # v0.0.28: topology_block returned a 4-tuple. v0.0.31 widens
            # it to a 5-tuple with role_decls on the tail. v0.0.62
            # widens it to a 6-tuple with functional_roles. Older
            # readers using indices 1..N still work; we guard each new
            # tail axis with `len(body) > N`.
            if isinstance(body, tuple) and body and body[0] == "topology":
                topology = body[1]
                instances = body[2]
                devices = body[3] if len(body) > 3 else []
                role_decls = body[4] if len(body) > 4 else []
                functional_roles = body[5] if len(body) > 5 else []
                continue
            kind, value = body
            if kind == "sizing":
                sizing_kw = value
            elif kind == "corners":
                corners = value
            elif kind == "temps":
                temps = value
            elif kind == "bias_tree":
                bias_tree = value
            elif kind == "combinator_block":
                if combinator_block_obj is not None:
                    raise ValueError(
                        f"cell {local_name!r}: more than one "
                        f"`combinator { '{...}' }` block in the same "
                        f"cell body. Combine them into a single block."
                    )
                combinator_block_obj = value
        if spec_obj is None:
            spec_obj = Spec(fields=[])

        # v0.0.31 (G3): expand each role decl into concrete Device
        # entries, using the cell's port list for default pin wiring.
        # We do this AFTER spec_obj resolution so per-role errors can
        # mention the enclosing cell name in their messages.
        port_name_set = set(port_names)
        if role_decls:
            # Pre-flight: role name vs cell port collision. A
            # `pmos_pair inn { ... }` while `inn` is also a port would
            # cause `inn_d1`, `inn_d2`, etc. prefixed nets that obscure
            # the port's role in the cell.
            for r in role_decls:
                if r.name in port_name_set:
                    raise ValueError(
                        f"cell {local_name!r}: role {r.name!r} collides "
                        f"with cell port {r.name!r}. Role handles must "
                        f"not overlap with the cell's port list."
                    )
            expanded_devices = _expand_roles_to_devices(
                role_decls, port_names=port_names,
                port_roles=port_roles_map, cell_name=local_name,
            )
            # Cross-check: an expanded device name must not collide with
            # any explicit `device M : pmos { ... }` already in the
            # devices list, nor with another role's expansion.
            existing_dev_names = {d.instance_name for d in devices}
            for nd in expanded_devices:
                if nd.instance_name in existing_dev_names:
                    raise ValueError(
                        f"cell {local_name!r}: role expansion produced "
                        f"device name {nd.instance_name!r} that "
                        f"collides with an explicit `device` "
                        f"declaration in the same cell. Rename one to "
                        f"avoid the conflict."
                    )
                existing_dev_names.add(nd.instance_name)
            devices = list(devices) + list(expanded_devices)

        # v0.0.28: self-instantiation guard — `cell A { instance a1 = pkg::A { ... } }`
        # is rejected. The local name (`A` here) versus the sub-cell
        # name after `::` must not match. This catches direct
        # recursion at parse time; deeper cycles (A -> B -> A) require
        # cross-cell resolution and are out of scope for v0.
        #
        # v0.0.34 (PLL gap A): the bare-name instance target form
        # (`instance a1 = A { ... }`) also needs to be caught here.
        # For bare names the cell_path has NO `::`, so `partition("::")`
        # yields (cell_path, "", "") — we compare against the LHS.
        for inst in instances:
            pkg_part, sep, sub_cell_name = inst.cell_path.partition("::")
            # Qualified form: `pkg::CellName` → check the RHS.
            # Bare form: `CellName` → partition yields (CellName, "", "")
            # → check the LHS instead.
            effective_name = sub_cell_name if sep else pkg_part
            if effective_name == local_name:
                raise ValueError(
                    f"cell {local_name!r}: instance {inst.name!r} "
                    f"references {inst.cell_path!r}, which is the "
                    f"enclosing cell itself. Recursive cell composition "
                    f"is not supported in this version."
                )
        # v0.0.28 (device primitives): cell-port-name vs device-name
        # collision check. Declaring `device out : pmos { ... }` while
        # `out` is also a port is a surface mistake — `out` would refer
        # to both the device handle and the net at netlist-emit time.
        # Catch this at parse time so the error reads cleanly.
        for dev in devices:
            if dev.instance_name in port_name_set:
                raise ValueError(
                    f"cell {local_name!r}: device {dev.instance_name!r} "
                    f"collides with cell port {dev.instance_name!r}. "
                    f"Device instance names must not overlap with the "
                    f"cell's port list."
                )
        cd = CellDef(
            local_name=local_name,
            ports=port_names,
            topology=topology,
            spec=spec_obj,
            sizing_kwargs=sizing_kw,
            port_roles=port_roles_map,
            bias_tree=bias_tree,
            instances=instances,
            devices=devices,
            role_decls=role_decls,
            functional_roles=functional_roles,
            combinator_block=combinator_block_obj,
            source_span=_source_span_from_meta(meta),
        )
        # Stash corners on the dataclass via attribute; `start()` reads
        # it back when wiring up the synthetic VerifyJob. Not part of
        # the public CellDef API (kept off __dataclass_fields__ on
        # purpose — see the design note in the docstring above).
        if corners is not None:
            cd.__dict__["_corners"] = corners
        # v0.0.24 (G11): same stashing mechanism for temps.
        if temps is not None:
            cd.__dict__["_temps"] = temps
        return ("cell_def", cd)

    def port_list(self, items):
        # v0.0.22 (G10): each item is a (name, role) tuple from `port`.
        return list(items)

    def port(self, items) -> tuple[str, str | None]:
        """A single port entry: bare `CNAME` or `CNAME : role`.

        Returns (name, role) where role is None when no annotation
        was supplied. Role validation against `_VALID_PORT_ROLES`
        already happened in the `port_role` transformer.
        """
        name = str(items[0])
        role: str | None = items[1] if len(items) > 1 else None
        return (name, role)

    def port_role(self, items) -> str:
        """Validate the role identifier against the fixed role set."""
        role = str(items[0])
        if role not in _VALID_PORT_ROLES:
            raise ValueError(
                f"Unknown port role {role!r}. Valid roles: "
                f"{sorted(_VALID_PORT_ROLES)}."
            )
        return role

    def topology_block(self, items):
        """items: a mixed list — each element is one of:
          * `(key, val)` tuple from `topology_field`  (legacy stage form)
          * `CellInstance` from `instance_decl`        (v0.0.28 composition)
          * `Device` from `device_decl` / `cap_decl` / `res_decl` /
            `ind_decl` (v0.0.28 device primitives + v0.0.34 inductor)
          * `RoleDecl` from `role_decl`                (v0.0.31 lean roles)
          * `list[Device]` from `inv_decl` / `nand2_decl` / `dff_decl`
            (v0.0.35 digital primitives — pre-expanded to N MOS at
            parse time)
          * `ParsedFunctionalRole` from `functional_role_decl`
            (v0.0.62 functional layer — behavioral-noun declarations)

        We sort them into sibling lists and return them all under one
        tagged tuple so the cell_def transformer can pick them apart.
        The 6-tuple shape `("topology", topology_dict, instances,
        devices, role_decls, functional_roles)` is backward-compatible
        with older 3/4/5-tuple readers: the cell_def transformer always
        looks at indices 1..5 (with `len(body) > N` guards on the tail
        axes).

        v0.0.28 (cell composition): introduces the instances axis.
        v0.0.28 (device primitives): introduces the devices axis.
        v0.0.31 (G3 closeout): introduces the role_decls axis. Roles
        are EXPANDED to concrete devices here at parse time (before
        the duplicate-name checks below) so the rest of the pipeline
        sees them as ordinary devices. The original RoleDecl entries
        survive on `role_decls` for tooling that wants the role-level
        intent (e.g. "this cell has a differential pair").
        v0.0.35 (PLL gap C): digital primitives (`inv` / `nand2` /
        `dff`) arrive here as PRE-EXPANDED `list[Device]` payloads from
        their transformer methods. We flatten them into the same
        `devices` axis as `device_decl`-emitted entries — the compiler
        only sees the flat Device list.
        v0.0.62 (functional roles): introduces the functional_roles
        axis. Unlike `role_decls`, functional roles are NOT expanded at
        parse time — the lowering layer (compiler / invariant pass)
        consumes them later. They survive on the CellDef as parsed
        surface intent.
        """
        topology_dict: dict[str, object] = {}
        instances: list[CellInstance] = []
        devices: list[Device] = []
        role_decls: list[RoleDecl] = []
        functional_roles: list[ParsedFunctionalRole] = []
        for it in items:
            if isinstance(it, CellInstance):
                instances.append(it)
            elif isinstance(it, Device):
                devices.append(it)
            elif isinstance(it, list):
                # v0.0.35: digital primitive expansion returns a
                # list[Device]. Flatten into the devices axis.
                for sub in it:
                    if not isinstance(sub, Device):
                        raise ValueError(
                            f"topology block: unexpected list element "
                            f"of type {type(sub).__name__} (expected "
                            f"Device)."
                        )
                    devices.append(sub)
            elif isinstance(it, RoleDecl):
                role_decls.append(it)
            elif isinstance(it, ParsedFunctionalRole):
                functional_roles.append(it)
            else:
                # (key, value) tuple from topology_field
                key, val = it
                topology_dict[key] = val

        # v0.0.31 (G3): duplicate-role-name check. Expansion to concrete
        # devices happens later in `cell_def` (where the cell's port list
        # is visible — the `pmos_pair`'s gate defaults pull from the
        # first two input-looking ports). We catch role-handle collisions
        # here so the error surfaces from inside the topology block
        # context.
        seen_role: set[str] = set()
        for r in role_decls:
            if r.name in seen_role:
                raise ValueError(
                    f"topology block: duplicate role name {r.name!r}. "
                    f"Each role declaration must have a unique local "
                    f"handle within the cell."
                )
            seen_role.add(r.name)

        # Duplicate-name check on instances — every `instance NAME = ...;`
        # must have a unique NAME within this topology block. Catching
        # this here gives a clear error before flatten time.
        seen_inst: set[str] = set()
        for inst in instances:
            if inst.name in seen_inst:
                raise ValueError(
                    f"topology block: duplicate instance name "
                    f"{inst.name!r}. Each `instance` declaration must "
                    f"have a unique local handle within the cell."
                )
            seen_inst.add(inst.name)
        # Duplicate-name check on devices — `device M1 ...; device M1 ...;`
        # is rejected at parse time so the LLM sees the error early
        # instead of producing a netlist with two `xM1` lines (one of
        # which ngspice will silently shadow). Role-expanded device
        # names follow the `<role_name>_<suffix>` convention so a user's
        # `device diff_pair_M1 : pmos { ... }` colliding with a role's
        # expansion of `pmos_pair diff_pair { ... }` will surface here.
        seen_dev: set[str] = set()
        for dev in devices:
            if dev.instance_name in seen_dev:
                raise ValueError(
                    f"topology block: duplicate device name "
                    f"{dev.instance_name!r}. Each `device`/`cap`/`res`/`ind` "
                    f"declaration (or role expansion) must produce a "
                    f"unique instance name within the cell."
                )
            seen_dev.add(dev.instance_name)
        # Device-vs-instance name collision check too — they share the
        # same namespace at netlist-emit time (xM1 vs xamp_a).
        for dev in devices:
            if dev.instance_name in seen_inst:
                raise ValueError(
                    f"topology block: device {dev.instance_name!r} and "
                    f"instance {dev.instance_name!r} share a name. "
                    f"Device, instance, and capacitor/resistor handles "
                    f"share one namespace."
                )
        # Role-name vs instance / device name collisions. The
        # role-expanded devices already carry `<role_name>_<suffix>`,
        # which can collide with explicit user `device` names only if
        # the user picked a name like `diff_pair_M1` and ALSO declared
        # `pmos_pair diff_pair { ... }`. That case is caught above.
        # We also want to reject `role NAME` colliding with a sibling
        # `instance NAME` because the role's tail-net would prefix
        # devices with `<NAME>_…` while the instance's netlist nodes
        # would prefix with `<NAME>__…` — too easy to confuse.
        for r in role_decls:
            if r.name in seen_inst:
                raise ValueError(
                    f"topology block: role {r.name!r} collides with an "
                    f"instance of the same name. Use distinct handles."
                )

        # v0.0.62 (functional roles): duplicate-name check across the
        # functional_roles axis. The functional layer is its own
        # namespace (NOT lowered to devices at parse time), so a
        # functional role with the same handle as a structural role,
        # instance, or device is allowed at the parser level —
        # downstream lowering decides how to reconcile names. We only
        # catch duplicates within the functional-role axis itself.
        seen_func_role: set[str] = set()
        for fr in functional_roles:
            if fr.name in seen_func_role:
                raise ValueError(
                    f"topology block: duplicate functional role name "
                    f"{fr.name!r}. Each functional role declaration "
                    f"must have a unique local handle within the cell."
                )
            seen_func_role.add(fr.name)
        return (
            "topology", topology_dict, instances, devices, role_decls,
            functional_roles,
        )

    def topology_field(self, items):
        key = str(items[0])
        val = items[1]
        # val came through topology_value; pass it through. CNAME tokens
        # are turned into plain strings; dict values from object_literal
        # are kept as-is so `topology['rationale']['input']` works.
        if isinstance(val, lark.Token):
            val = str(val)
        return (key, val)

    # ─── v0.0.28: cell composition ───────────────────────────

    def instance_decl(self, items) -> CellInstance:
        """`instance NAME = pkg.path::Cell { port = net; ... };`

        items: [CNAME(name), qualified_name_str, port_map_field*, ...]
        where each port_map_field item is a `(sub_port, outer_net)`
        tuple from `port_map_field`. We assemble the CellInstance and
        run surface-level validation here. Path-existence and
        port-match validation happens at compile time (parser stays
        offline / PDK-agnostic). The empty-map rejection IS surface
        level — no sub-cell with ports is useful without bindings.
        """
        name = str(items[0])
        cell_path = items[1]   # already 'pkg.path::CellName' string
        # Remaining items are (sub_port, outer_net) tuples
        pairs: list[tuple[str, str]] = list(items[2:])
        # Empty-map rejection: forbid `instance amp = pkg::Cell { };`
        # at parse time. Even if a sub-cell happens to have zero
        # ports, leaving the body blank is surface ambiguity (did the
        # author forget to type the bindings?). Reject loudly.
        if not pairs:
            raise ValueError(
                f"instance {name!r}: empty port-map `{{ }}` is not "
                f"permitted. Every sub-cell port must be explicitly "
                f"bound to an outer net (e.g. `inn = in_a;`)."
            )
        # Duplicate sub-port-name guard: `inn = a; inn = b;` is a
        # surface mistake — reject before silently overwriting.
        port_map: dict[str, str] = {}
        for sub_port, outer_net in pairs:
            if sub_port in port_map:
                raise ValueError(
                    f"instance {name!r}: sub-cell port {sub_port!r} "
                    f"is bound more than once in the port map. "
                    f"Each sub-cell port maps to exactly one outer net."
                )
            port_map[sub_port] = outer_net
        return CellInstance(name=name, cell_path=cell_path, port_map=port_map)

    def port_map_field(self, items) -> tuple[str, str]:
        """One `port = net;` binding inside an instance_decl body.

        Returns the raw (sub_port_name, outer_net_name) pair. Validation
        (port exists on sub-cell, net exists on outer cell) happens at
        compile time when both sides are resolvable.
        """
        sub_port = str(items[0])
        outer_net = str(items[1])
        return (sub_port, outer_net)


def _source_span_from_meta(meta) -> dict[str, object] | None:
    line = getattr(meta, "line", None)
    column = getattr(meta, "column", None)
    if line is None or column is None:
        return None
    return {
        "line": int(line),
        "column": int(column),
        "end_line": int(getattr(meta, "end_line", line) or line),
        "end_column": int(getattr(meta, "end_column", column) or column),
    }
