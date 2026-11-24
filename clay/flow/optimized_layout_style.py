"""Derive executable compact-layout declarations from source circuit DATA.

The derivation is deliberately name-agnostic: functional groups, placement
columns/lanes, matching patterns, body nets, device polarity, and port roles are
all read from the parsed Clay source.  The result is emitted back into the
optimized derivative as existing Clay syntax.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from clay.layout.annotations import Fold, Orient
from clay.layout.compact_intent import (
    CompactIntentCandidate,
    MacroPartitionChoice,
    derive_guard_ring_atoms,
)
from clay.layout.tech import LayoutTech


DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM = 10.0
# Existing OTA house style used by every signed-off reference PDK source.  This
# is source intent (a current-distribution rail), not a measured GDS coordinate.
DEFAULT_ANALOG_SUPPLY_TRUNK_WIDTH_UM = 8.0


@dataclass(frozen=True)
class OptimizedLayoutStyle:
    layout_lines: tuple[str, ...]
    annotation_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptimizedPlacementGroup:
    """One exact placement row consumed by optimized-style materialization."""

    name: str
    members: tuple[str, ...]
    row_min: int | None = None
    row_max: int | None = None
    priority: str | None = None
    col: int | None = None
    pack_rows: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name) is None
        ):
            raise ValueError("optimized placement group name is not canonical")
        if (
            type(self.members) is not tuple
            or not self.members
            or len(set(self.members)) != len(self.members)
            or any(
                type(member) is not str
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", member) is None
                for member in self.members
            )
        ):
            raise ValueError("optimized placement group members are not canonical")
        rows = (self.row_min, self.row_max)
        if (rows[0] is None) != (rows[1] is None):
            raise ValueError("optimized placement group row bounds are incomplete")
        if rows[0] is not None and (
            type(rows[0]) is not int
            or type(rows[1]) is not int
            or rows[0] < 0
            or rows[1] < rows[0]
        ):
            raise ValueError("optimized placement group row bounds are not canonical")
        if self.col is not None and (type(self.col) is not int or self.col < 0):
            raise ValueError("optimized placement group column is not canonical")
        if self.pack_rows is not None and (
            type(self.pack_rows) is not int
            or self.pack_rows < 1
            or self.pack_rows > len(self.members)
        ):
            raise ValueError(
                "optimized placement group pack_rows exceeds member capacity"
            )
        if self.priority is not None and (
            type(self.priority) is not str
            or not self.priority
            or self.priority != self.priority.strip()
        ):
            raise ValueError("optimized placement group priority is not canonical")


def macro_partition_place_group_name(choice: MacroPartitionChoice, index: int) -> str:
    """Return the stable existing-syntax group name for one selected sub-bank."""

    if type(index) is not int or not 0 <= index < len(choice.member_groups):
        raise ValueError("macro partition place-group index is out of range")
    leaf = re.sub(r"[^A-Za-z0-9_]+", "_", choice.macro_id.split(":")[-1]).strip("_")
    if not leaf or leaf[0].isdigit():
        leaf = "macro_" + leaf
    tag = hashlib.sha256(choice.macro_id.encode("utf-8")).hexdigest()[:8]
    return f"compact_{leaf}_{tag}_{index + 1:02d}"


def _materialize_passive_partition_rows(
    placement_rows: list[
        tuple[str, tuple[str, ...], int | None, int | None, str | None]
    ],
    choices: tuple[MacroPartitionChoice, ...],
) -> list[tuple[str, tuple[str, ...], int | None, int | None, str | None]]:
    """Replace source rows by graph-selected physical-macro place groups.

    Passive banks and well islands use the same exact partition algebra: every
    selected member is consumed once, authored row constraints must agree, and
    each selected group receives one stable generated owner.  The historical
    helper name is retained for compatibility with existing focused consumers.
    """

    rows = list(placement_rows)
    for choice in choices:
        owned = {member for group in choice.member_groups for member in group}
        covered: set[str] = set()
        constraints: set[tuple[int | None, int | None, str | None]] = set()
        retained: list[
            tuple[str, tuple[str, ...], int | None, int | None, str | None]
        ] = []
        occupied_names = {row[0] for row in rows}
        for name, members, row_min, row_max, priority in rows:
            member_set = set(members)
            overlap = member_set & owned
            if not overlap:
                retained.append((name, members, row_min, row_max, priority))
                continue
            covered.update(overlap)
            constraints.add((row_min, row_max, priority))
            remainder = tuple(member for member in members if member not in owned)
            if remainder:
                # A physical partition may take a resistor out of a mixed
                # functional branch. The remaining placement row no longer
                # denotes that complete functional group: reusing its name
                # contradicts the canonical macro graph's exact membership.
                tag = hashlib.sha256("\0".join((name, *remainder)).encode("utf-8")).hexdigest()[:12]
                remainder_name = f"compact_remainder_{tag}"
                if remainder_name in occupied_names:
                    raise ValueError(f"generated remainder place group {remainder_name!r} collides")
                occupied_names.add(remainder_name)
                retained.append((remainder_name, remainder, row_min, row_max, priority))
        if covered != owned:
            missing = tuple(sorted(owned - covered))
            raise ValueError(
                f"macro partition {choice.macro_id!r} was not consumed by "
                f"derived placement rows; missing={missing!r}"
            )
        if len(constraints) != 1:
            raise ValueError(
                f"macro partition {choice.macro_id!r} crosses incompatible "
                "authored row regions"
            )
        row_min, row_max, priority = next(iter(constraints))
        existing_names = {row[0] for row in retained}
        generated = []
        for index, members in enumerate(choice.member_groups):
            name = macro_partition_place_group_name(choice, index)
            if name in existing_names:
                raise ValueError(f"generated macro place group {name!r} collides")
            existing_names.add(name)
            generated.append((name, members, row_min, row_max, priority))
        rows = retained + generated
    return rows


def canonicalize_fold_compatible_sizing(
    cell_def: Any,
    optimization: Any,
    sizing: dict[str, float],
    *,
    fold_height_um: float = DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM,
    grid_nm: int = 1,
) -> dict[str, float]:
    """Snap width DOFs to the nearest exactly conserved foldable nanometres.

    tSS-BO proposes continuous widths, while the canonical row/device
    materializers draw uniform integer-nanometre fingers on the PDK
    manufacturing grid (``grid_nm``).  A promoted width is therefore not
    physical truth unless the fold count derived from the same source can
    conserve it exactly on-grid.  This function performs the tiny,
    deterministic quantization *before* pre/post simulation, so schematic,
    layout, PEX, and optimized Clay all consume one circuit revision.

    The rule is entirely source-derived: device bindings identify each width
    DOF, functional common-centroid groups identify even fold plans, and the
    existing optimized-style height supplies the bounded-finger target.  No
    circuit, device name, or PDK identifier participates.
    """

    height_nm = int(round(float(fold_height_um) * 1000.0))
    if height_nm <= 0 or not math.isclose(
        float(fold_height_um) * 1000.0,
        height_nm,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("optimized fold height must be exact positive nanometres")
    grid_step = max(1, int(grid_nm))

    devices = {
        str(device.instance_name): device
        for device in (getattr(cell_def, "devices", ()) or ())
    }
    common_centroid_members: set[str] = set()
    for role in getattr(cell_def, "functional_roles", ()) or ():
        attributes = dict(getattr(role, "attributes", {}) or {})
        if str(attributes.get("matching_pattern", "")).lower() != "common_centroid":
            continue
        common_centroid_members.update(
            str(name) for name in (attributes.get("devices", ()) or ())
        )

    variables = tuple(getattr(optimization, "variables", ()) or ())
    output = dict(sizing)
    for variable in variables:
        name = str(getattr(variable, "name", ""))
        if name not in output:
            continue
        bound_devices: list[Any] = []
        for binding in getattr(variable, "bindings", ()) or ():
            parts = str(binding).split(".")
            if len(parts) != 3 or parts[0] != "device" or parts[2].upper() != "W":
                continue
            device = devices.get(parts[1])
            if device is not None and _category(device) in {"pmos", "nmos"}:
                bound_devices.append(device)
        if not bound_devices:
            continue

        raw_nm = int(round(float(output[name]) * 1e9))
        low_nm = int(math.ceil(float(variable.low) * 1e9 - 1e-9))
        high_nm = int(math.floor(float(variable.high) * 1e9 + 1e-9))
        decl_step = getattr(variable, "step", None)
        step_nm: int | None = None
        if decl_step is not None:
            cand_step = int(round(float(decl_step) * 1e9))
            if cand_step > 0 and math.isclose(float(decl_step) * 1e9, cand_step, rel_tol=0.0, abs_tol=1e-6):
                step_nm = cand_step

        def exactly_foldable(width_nm: int, *, req_grid: int) -> bool:
            if step_nm is not None and width_nm % step_nm != 0:
                return False
            if width_nm % req_grid != 0:
                return False
            for device in bound_devices:
                multiplicity = max(
                    1,
                    int(round(float(device.sizing.get("M", 1.0) or 1.0))),
                )
                total_nm = width_nm * multiplicity
                common = str(device.instance_name) in common_centroid_members
                if common:
                    fingers = max(
                        2,
                        int(math.ceil(total_nm / height_nm)),
                        multiplicity,
                    )
                    if fingers % 2:
                        fingers += 1
                    needs_fold = multiplicity < 2 or multiplicity % 2 != 0
                    needs_fold = needs_fold or total_nm > 2 * height_nm
                else:
                    fingers = max(
                        multiplicity,
                        int(math.ceil(total_nm / height_nm)),
                    )
                    needs_fold = total_nm > 2 * height_nm
                if needs_fold and total_nm % (fingers * req_grid):
                    return False
            return True

        selected: int | None = None
        grid_trials = (grid_step, 1) if grid_step > 1 else (1,)
        for req_grid in grid_trials:
            max_offset = max(raw_nm - low_nm, high_nm - raw_nm)
            for offset in range(max_offset + 1):
                # Stable tie break prefers the smaller physical device.
                candidates = (
                    (raw_nm,) if offset == 0 else (raw_nm - offset, raw_nm + offset)
                )
                for candidate_nm in candidates:
                    if not low_nm <= candidate_nm <= high_nm:
                        continue
                    if exactly_foldable(candidate_nm, req_grid=req_grid):
                        selected = candidate_nm
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            raise ValueError(
                f"sizing variable {name!r} has no exactly fold-compatible "
                f"integer-nm value in [{low_nm}, {high_nm}]"
            )
        output[name] = float(selected) * 1e-9
    return output


def _category(device: Any) -> str:
    kind = str(device.device_class).lower()
    if "pmos" in kind or kind.startswith("pfet"):
        return "pmos"
    if "nmos" in kind or kind.startswith("nfet"):
        return "nmos"
    return "passive"


def _device_order(cell_def: Any) -> dict[str, int]:
    return {
        device.instance_name: index for index, device in enumerate(cell_def.devices)
    }


def _tuple_source(names: tuple[str, ...]) -> str:
    return "(" + ", ".join(names) + ")"


def _supply_ports(cell_def: Any) -> tuple[str | None, str | None]:
    top = bottom = None
    for port in cell_def.ports:
        name = str(port)
        role = str(cell_def.port_roles.get(name, "")).lower()
        if role == "power_p" and top is None:
            top = name
        if role in {"power_n", "ground"} and bottom is None:
            bottom = name
    return top, bottom


def _horizontal_supply_trunk_layer(cell_def: Any) -> str:
    """First PDK-declared horizontal routing metal above device-local M1."""
    flow_decl = getattr(cell_def, "flow_decl", None)
    pdk = str(getattr(flow_decl, "pdk", "") or "")
    if not pdk:
        raise ValueError("optimized supply trunks require the flow's declared PDK")
    tech = LayoutTech.resolve(pdk)
    for metal_index in range(2, 33):
        try:
            tech.metal_key(metal_index)
        except KeyError:
            break
        if tech.metal_prefers_horizontal(metal_index):
            return f"metal{metal_index}"
    raise ValueError(
        f"optimized supply trunks require a horizontal metal above M1 in {pdk}"
    )


def _length_key(device: Any) -> int:
    """Integer-nm channel length key for topology grouping."""
    return int(round(float(device.sizing.get("L", 0.0)) * 1e9))


def _passive_kind(device: Any) -> str:
    """Canonical passive class used by compact-style lowering."""
    kind = str(device.device_class).lower()
    if kind.startswith("res"):
        return "res"
    if kind.startswith("cap"):
        return "cap"
    return ""


def _guard_islands(
    devices: dict[str, Any],
    order: dict[str, int],
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Partition MOS once by physical well domain and declared body net.

    Matching and functional groups constrain device placement inside a well;
    they do not create new electrical well domains.  Splitting one supply-tied
    well for every common-centroid or role group adds ring perimeter without
    satisfying a new body obligation.  Conversely, unlike-polarity devices or
    distinct declared body nets must never share a guard island.
    """

    partitions: dict[tuple[str, str], list[str]] = {}
    for name in sorted(devices, key=order.__getitem__):
        device = devices[name]
        category = _category(device)
        if category not in {"pmos", "nmos"}:
            continue
        body = str(device.pins.get("b", ""))
        if not body:
            raise ValueError(
                f"optimized guard island requires declared body net for {name}"
            )
        partitions.setdefault((category, body), []).append(name)
    return [
        (category, body, tuple(members))
        for (category, body), members in partitions.items()
    ]


def derive_optimized_placement_groups(
    cell_def: Any,
    *,
    candidate: CompactIntentCandidate | None = None,
) -> tuple[OptimizedPlacementGroup, ...]:
    """Return the exact unordered placement groups used by style lowering.

    Structural group-changing axes such as passive-bank partitions are consumed
    here.  ``group_order`` is deliberately applied by the caller after this
    inventory exists, so reference receipts and validators see the same complete
    authority that the final materializer orders.
    """

    devices = {device.instance_name: device for device in cell_def.devices}
    roles = list(getattr(cell_def, "functional_roles", ()) or ())
    from clay.layout.intent import layout_intent_from_cell_def

    authored_intent = layout_intent_from_cell_def(cell_def)
    authored_groups = tuple(
        group
        for group in authored_intent.place_groups
        if group.row_min is not None and group.row_max is not None
    )
    # Functional roles refine a floorplan; they do not own permission to erase
    # the coarser authored row region.  Carry the unique containing authored
    # range onto every derived subgroup so materialization remains idempotent.
    placement_rows: list[
        tuple[str, tuple[str, ...], int | None, int | None, str | None]
    ] = []
    for role in roles:
        attrs = dict(role.attributes)
        # ``placement_intent`` predates executable layout lowering and is free
        # form authoring metadata.  Keep it out of the control plane.  A well
        # domain is excluded by its typed role, not by interpreting prose (or a
        # magic value) as an execution switch.
        if attrs.get("role") == "well_domain":
            continue
        members = tuple(
            str(name) for name in attrs.get("devices", ()) if str(name) in devices
        )
        if not members:
            continue
        partitions: dict[str, list[str]] = {}
        for name in members:
            partitions.setdefault(_category(devices[name]), []).append(name)
        for category in ("nmos", "passive", "pmos"):
            category_members = tuple(partitions.get(category, ()))
            if not category_members:
                continue
            by_container: dict[str, list[str]] = {}
            container_by_name = {group.group: group for group in authored_groups}
            for member in category_members:
                containers = [
                    group for group in authored_groups if member in group.devices
                ]
                if len(containers) > 1:
                    raise ValueError(
                        "optimized functional member has ambiguous authored "
                        f"floorplan containers: member={member!r}, "
                        f"containers={[group.group for group in containers]}"
                    )
                container_name = containers[0].group if containers else ""
                by_container.setdefault(container_name, []).append(member)
            for container_name, spatial_members in sorted(by_container.items()):
                group_name = str(role.name)
                if len(partitions) > 1:
                    group_name = f"{group_name}_{category}"
                if len(by_container) > 1:
                    # A derivative already carries stable role-prefixed authored
                    # containers.  Reuse those names instead of adding the role
                    # prefix on every replay.
                    if container_name and (
                        container_name == group_name
                        or container_name.startswith(group_name + "_")
                    ):
                        group_name = container_name
                    else:
                        group_name = (
                            f"{group_name}_{container_name}"
                            if container_name
                            else f"{group_name}_unconstrained"
                        )
                container = container_by_name.get(container_name)
                placement_rows.append(
                    (
                        group_name,
                        tuple(spatial_members),
                        None if container is None else int(container.row_min),
                        None if container is None else int(container.row_max),
                        None if container is None else container.priority,
                    )
                )

    macro_partitions = (
        ()
        if candidate is None
        else (
            *candidate.passive_bank_partitions,
            *candidate.well_island_partitions,
        )
    )
    if macro_partitions:
        placement_rows = _materialize_passive_partition_rows(
            placement_rows,
            macro_partitions,
        )
    raw_floorplan = () if candidate is None else candidate.group_floorplan
    floorplan_by_group = {
        group: (tier, column) for group, tier, column in raw_floorplan
    }
    row_names = {name for name, _members, _row_min, _row_max, _priority in placement_rows}
    if raw_floorplan and (
        len(floorplan_by_group) != len(raw_floorplan)
        or not set(floorplan_by_group) <= row_names
    ):
        raise ValueError(
            "compact candidate group_floorplan contains an unknown or duplicate "
            "derived placement group"
        )
    groups = tuple(
        OptimizedPlacementGroup(
            name,
            members,
            floorplan_by_group[name][0] if name in floorplan_by_group else row_min,
            floorplan_by_group[name][0] if name in floorplan_by_group else row_max,
            priority,
            floorplan_by_group[name][1] if name in floorplan_by_group else None,
        )
        for name, members, row_min, row_max, priority in placement_rows
    )
    if candidate is not None and candidate.nonmos_macro_rows:
        from clay.layout.compact_intent_domain import physical_macro_graph_for_candidate

        # On admission replay the derivative already carries ``pack_rows`` in
        # ordinary place-group syntax.  Ask the graph builder to compare that
        # materialized value with the candidate instead of reusing the now-
        # replaced global floorplan row as its authoring domain.
        materialized_replay = any(
            group.pack_rows is not None for group in authored_intent.place_groups
        )
        graph = physical_macro_graph_for_candidate(
            cell_def,
            candidate,
            allow_materialized_passive_arrays=materialized_replay,
        )
        macros = {macro.macro_id: macro for macro in graph.macros}
        rebound = list(groups)
        for choice in candidate.nonmos_macro_rows:
            macro = macros.get(choice.macro_id)
            if macro is None:
                raise ValueError(
                    f"non-MOS macro row choice is absent from effective graph: "
                    f"{choice.macro_id}"
                )
            owners = [
                index
                for index, group in enumerate(rebound)
                if set(group.members) == set(macro.members)
            ]
            if len(owners) != 1:
                raise ValueError(
                    "non-MOS macro row choice must resolve to one exact derived "
                    f"place group: {choice.macro_id}"
                )
            index = owners[0]
            rebound[index] = replace(rebound[index], pack_rows=choice.rows)
        groups = tuple(rebound)
    names = tuple(group.name for group in groups)
    if len(set(names)) != len(names):
        raise ValueError("optimized placement group names must be unique")
    return groups


def _ordered_optimized_placement_groups(
    groups: tuple[OptimizedPlacementGroup, ...],
    group_order: tuple[str, ...],
) -> tuple[OptimizedPlacementGroup, ...]:
    if type(groups) is not tuple or any(
        type(group) is not OptimizedPlacementGroup for group in groups
    ):
        raise TypeError("optimized placement groups must be an exact tuple")
    if type(group_order) is not tuple or any(
        type(name) is not str for name in group_order
    ):
        raise TypeError("optimized placement group order must be an exact tuple")
    if not group_order:
        return groups
    by_group = {name: index for index, name in enumerate(group_order)}
    actual_groups = {group.name for group in groups}
    unknown = sorted(set(by_group) - actual_groups)
    missing = sorted(actual_groups - set(by_group))
    if len(by_group) != len(group_order) or unknown or missing:
        raise ValueError(
            "compact candidate group_order must name every derived placement "
            f"group exactly once; unknown={unknown}, missing={missing}"
        )
    return tuple(sorted(groups, key=lambda group: by_group[group.name]))


def derive_optimized_layout_style(
    cell_def: Any,
    *,
    fold_height_um: float = DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM,
    candidate: CompactIntentCandidate | None = None,
) -> OptimizedLayoutStyle:
    """Lower functional-group DATA into deterministic existing Clay syntax."""
    if candidate is not None and candidate.mos_fold_height_um is not None:
        fold_height_um = float(candidate.mos_fold_height_um)
    if fold_height_um <= 0:
        raise ValueError("optimized layout fold height must be positive")
    devices = {device.instance_name: device for device in cell_def.devices}
    order = _device_order(cell_def)
    roles = list(getattr(cell_def, "functional_roles", ()) or ())
    from clay.layout.intent import layout_intent_from_cell_def

    authored_intent = layout_intent_from_cell_def(cell_def)
    authored_anchors = tuple(authored_intent.anchors)
    authored_rail_affinities = tuple(authored_intent.rail_affinities)
    guard_pack_owner_by_member: dict[str, int] = {}
    authored_ring_pack_by_member = {
        member: ring.pack
        for ring in authored_intent.guard_rings
        for member in ring.members
    }
    if candidate is not None:
        base_groups = [ring.members for ring in authored_intent.guard_rings]
        authored_members = {
            member for group in base_groups for member in group
        }
        base_groups.extend(
            residual
            for atom in derive_guard_ring_atoms(cell_def)
            if (
                residual := tuple(
                    member
                    for member in atom.members
                    if member not in authored_members
                )
            )
        )
        for owner, members in enumerate(base_groups):
            for member in members:
                if member in guard_pack_owner_by_member:
                    raise ValueError(
                        "guard-ring prepack ownership must be unique"
                    )
                guard_pack_owner_by_member[member] = owner
    placement_groups = derive_optimized_placement_groups(
        cell_def,
        candidate=candidate,
    )
    placement_groups = _ordered_optimized_placement_groups(
        placement_groups,
        () if candidate is None else candidate.group_order,
    )
    common_centroids: list[str] = []
    common_centroid_groups: list[tuple[str, ...]] = []
    common_centroid_members: set[str] = set()
    output_nets: list[str] = []
    bias_nets: list[str] = []
    signal_orders: list[tuple[str, str]] = []
    vertical_signal_path_resistors: set[str] = set()

    for role in roles:
        attrs = dict(role.attributes)
        members = tuple(
            str(name) for name in attrs.get("devices", ()) if str(name) in devices
        )
        if not members:
            continue
        if str(attrs.get("matching_pattern", "")).lower() == "common_centroid":
            common_centroids.append(str(role.name))
            common_centroid_groups.append(members)
            common_centroid_members.update(members)
        output_nets.extend(str(net) for net in attrs.get("output_nets", ()))
        bias_nets.extend(str(net) for net in attrs.get("bias_nets", ()))
        signal_path = tuple(attrs.get("signal_path", ()) or ())
        if signal_path and len(members) >= 2:
            signal_orders.extend(zip(members, members[1:]))
            passive_kinds = {_passive_kind(devices[name]) for name in members}
            if {"res", "cap"} <= passive_kinds:
                vertical_signal_path_resistors.update(
                    name for name in members if _passive_kind(devices[name]) == "res"
                )

    lines: list[str] = []
    top, bottom = _supply_ports(cell_def)
    horizontal_layer = (
        _horizontal_supply_trunk_layer(cell_def)
        if (top and bottom) or bias_nets
        else ""
    )
    exact_floorplan_authority = bool(
        candidate is not None and candidate.group_floorplan
    )
    if top and bottom:
        # A complete reference-macro floorplan is a stronger, explicit
        # coordinate-free row/column authority.  Do not layer the generic
        # polarity/passive three-band heuristic on top of it: a two-tier
        # reference grammar (passives below an active tier) cannot satisfy the
        # synthesized NMOS/passive/PMOS order by construction.  The exact
        # @place_group(row,col) inventory below remains the hard placement
        # authority; trunks remain electrical/routing obligations.
        if not exact_floorplan_authority:
            lines.append(f"@band(top={top}, bottom={bottom}, center=passives);")
        supply_width = format(DEFAULT_ANALOG_SUPPLY_TRUNK_WIDTH_UM, ".12g")
        lines.append(
            f"@trunk({top}, layer={horizontal_layer}, orient=H, "
            f"width={supply_width}, edge=top);"
        )
        lines.append(
            f"@trunk({bottom}, layer={horizontal_layer}, orient=H, "
            f"width={supply_width}, edge=bottom);"
        )
    for net in dict.fromkeys(bias_nets):
        if net not in {top, bottom}:
            lines.append(f"@trunk({net}, layer={horizontal_layer}, orient=H);")
    derived_group_members: dict[str, tuple[str, ...]] = {}
    generated_rail_affinities: list[tuple[str, str]] = []
    for placement_group in placement_groups:
        group_name = placement_group.name
        members = placement_group.members
        row_min = placement_group.row_min
        row_max = placement_group.row_max
        priority = placement_group.priority
        col = placement_group.col
        derived_group_members[group_name] = members
        arguments = [
            group_name,
            f"devices={_tuple_source(members)}",
        ]
        if row_min is not None and row_max is not None:
            arguments.append(
                f"row={row_min}" if row_min == row_max else f"rows={row_min}..{row_max}"
            )
        if col is not None:
            arguments.append(f"col={col}")
        if placement_group.pack_rows is not None:
            arguments.append(f"pack_rows={placement_group.pack_rows}")
        if priority:
            arguments.append(f"priority={priority}")
        lines.append("@place_group(" + ", ".join(arguments) + ");")
        if not exact_floorplan_authority:
            categories = {_category(devices[name]) for name in members}
            if categories == {"pmos"}:
                generated_rail_affinities.append((group_name, "top"))
            elif categories == {"nmos"}:
                generated_rail_affinities.append((group_name, "bottom"))
    rail_affinities = list(generated_rail_affinities)
    authored_group_members = {
        group.group: set(group.devices) for group in authored_intent.place_groups
    }
    if not exact_floorplan_authority:
        for affinity in authored_rail_affinities:
            target = str(affinity.group)
            side = str(affinity.side)
            target_members = authored_group_members.get(target)
            if target_members is None:
                if (target, side) not in rail_affinities:
                    rail_affinities.append((target, side))
                continue
            exact_groups = [name for name, members in derived_group_members.items()
                            if set(members) == target_members]
            if len(exact_groups) == 1:
                # A PNP/passive assembly remains one source envelope. Splitting
                # its affinity into per-device facts both changes the source
                # meaning and makes the live rigid-master consumer reject it.
                rebound = (exact_groups[0], side)
                if rebound not in rail_affinities:
                    rail_affinities.append(rebound)
                continue
            covered = {
                member
                for group_name, generated_side in generated_rail_affinities
                if generated_side == side
                for member in derived_group_members.get(group_name, ())
            }
            for name, members in derived_group_members.items():
                if members and set(members) <= target_members and set(members) - covered:
                    rebound = (name, side)
                    if rebound not in rail_affinities:
                        rail_affinities.append(rebound)
                    covered.update(members)
            for member in sorted(target_members - covered):
                if (member, side) not in rail_affinities:
                    rail_affinities.append((member, side))
    lines.extend(
        f"@rail_affinity({target}, side={side});" for target, side in rail_affinities
    )
    for group_name in dict.fromkeys(common_centroids):
        # The existing common-centroid constructor owns the complete multi-MOS
        # row program.  A discrete array declares singleton MOS replication and
        # would send this same group to a different, incompatible constructor.
        lines.append(f"@common_centroid({group_name}, rows=2);")
    if candidate is not None:
        common_centroid_set = set(common_centroids)
        for choice in candidate.passive_arrays:
            if choice.passive in common_centroid_set:
                raise ValueError(
                    "passive array target collides with a common-centroid array"
                )
            lines.append(
                f"@array({choice.passive}, style=discrete, "
                f"rows={choice.rows}, order=ratio);"
            )
    for left, right in signal_orders:
        lines.append(f"@order({left}, {right}, left_of);")
    if candidate is None:
        for category, body, unique in _guard_islands(devices, order):
            well = "n" if category == "pmos" else "p"
            lines.append(
                f"@guard_ring(group={_tuple_source(unique)}, well={well}, tie={body});"
            )
    else:
        for split in candidate.passive_splits:
            lines.append(
                f"@split({split.passive}, n={split.unit_count}, "
                f"match={split.connection});"
            )
        for ring in candidate.guard_ring_groups:
            try:
                owners = {
                    guard_pack_owner_by_member[member]
                    for member in ring.members
                }
            except KeyError as exc:
                raise ValueError(
                    "compact candidate guard ring contains a member outside "
                    "canonical guard-ring atoms"
                ) from exc
            authored_packs = {
                authored_ring_pack_by_member[member]
                for member in ring.members
                if member in authored_ring_pack_by_member
                and authored_ring_pack_by_member[member]
            }
            if len(authored_packs) > 1:
                raise ValueError(
                    "compact candidate guard ring merges incompatible pack contracts"
                )
            pack = (
                "hierarchical"
                if len(owners) > 1
                else next(iter(authored_packs), "")
            )
            pack_clause = f", pack={pack}" if pack else ""
            lines.append(
                f"@guard_ring(group={_tuple_source(ring.members)}, "
                f"well={ring.well}, tie={ring.tie_net}{pack_clause});"
            )

    # Authored edge access is source intent and normally survives the optimized
    # derivative just like authored row ranges.  A complete reference-macro
    # floorplan is the stronger authority for a canonical member/group whose
    # declared column is not on that edge: retaining both would emit a
    # mechanically infeasible pair of hard constraints.  Net anchors remain
    # routing obligations and are preserved because a net can reach an edge
    # without pinning every connected device there.
    floorplan_column_by_member: dict[str, int] = {}
    if exact_floorplan_authority:
        selected_floorplan_groups = {
            group for group, _tier, _column in candidate.group_floorplan
        }
        for group in placement_groups:
            if group.name not in selected_floorplan_groups:
                continue
            if type(group.col) is not int or group.col < 0:
                raise ValueError(
                    "exact floorplan placement group lacks a canonical column"
                )
            for member in group.members:
                prior = floorplan_column_by_member.get(member)
                if prior is not None and prior != group.col:
                    raise ValueError(
                        f"floorplan member {member!r} belongs to conflicting columns"
                    )
                floorplan_column_by_member[member] = group.col
    floorplan_columns = set(floorplan_column_by_member.values())

    def floorplan_edge_compatible(target: str, edge: str) -> bool:
        if not exact_floorplan_authority or edge not in {"left", "right"}:
            return True
        if target in devices:
            members = {target}
        elif target in derived_group_members:
            members = set(derived_group_members[target])
        elif target in authored_group_members:
            members = set(authored_group_members[target])
        else:
            return True
        columns = {
            floorplan_column_by_member[member]
            for member in members
            if member in floorplan_column_by_member
        }
        if not columns:
            return True
        if len(columns) != 1 or not floorplan_columns:
            return False
        required = min(floorplan_columns) if edge == "left" else max(floorplan_columns)
        return next(iter(columns)) == required

    # A selected compact candidate may override an eligible passive's edge;
    # orientation-only passives are as physical as split passives and therefore
    # consume the same side axis.
    candidate_anchor_by_target: dict[str, str] = {}
    if candidate is not None and candidate.passive_side in {"left", "right"}:
        eligible_passives = {split.passive for split in candidate.passive_splits} | {
            passive for passive, _orientation in candidate.passive_orientations
        }
        candidate_anchor_by_target.update(
            (target, candidate.passive_side) for target in sorted(eligible_passives)
        )
    anchors: list[tuple[str, str]] = []
    seen_anchor_targets: set[str] = set()
    for anchor in authored_anchors:
        target = str(anchor.target)
        edge = candidate_anchor_by_target.pop(target, str(anchor.edge))
        if (
            target
            and target not in seen_anchor_targets
            and floorplan_edge_compatible(target, edge)
        ):
            anchors.append((target, edge))
            seen_anchor_targets.add(target)
    for target, edge in sorted(candidate_anchor_by_target.items()):
        if (
            target not in seen_anchor_targets
            and floorplan_edge_compatible(target, edge)
        ):
            anchors.append((target, edge))
            seen_anchor_targets.add(target)
    for net in dict.fromkeys(output_nets):
        if net not in seen_anchor_targets:
            anchors.append((net, "right"))
            seen_anchor_targets.add(net)
    lines.extend(f"@anchor({target}, edge={edge});" for target, edge in anchors)

    existing_annotations = getattr(cell_def, "_layout_annotations", {}) or {}
    annotation_lines: list[str] = []
    if candidate is not None:
        annotation_lines.extend(
            f"{split.passive}: @orient({split.orientation});"
            for split in candidate.passive_splits
        )
        annotation_lines.extend(
            f"{passive}: @orient({orientation});"
            for passive, orientation in candidate.passive_orientations
        )
    explicit_passive_orientations = (
        {split.passive for split in candidate.passive_splits}
        | {passive for passive, _orientation in candidate.passive_orientations}
        if candidate is not None
        else set()
    )
    for device in cell_def.devices:
        existing = existing_annotations.get(device.instance_name, ())
        if (
            device.instance_name in vertical_signal_path_resistors
            and device.instance_name not in explicit_passive_orientations
            and not any(isinstance(annotation, Orient) for annotation in existing)
        ):
            annotation_lines.append(f"{device.instance_name}: @orient(R90);")
        if _category(device) not in {"pmos", "nmos"}:
            continue
        fold_profile_override = (
            candidate is not None and candidate.mos_fold_height_um is not None
        )
        if not fold_profile_override and any(
            isinstance(annotation, Fold) for annotation in existing
        ):
            continue
        width_m = float(device.sizing.get("W", 0.0))
        multiplicity = float(device.sizing.get("M", 1.0))
        effective_width_um = width_m * 1e6 * max(multiplicity, 1.0)
        device_fold_height = fold_height_um
        if device.instance_name in common_centroid_members:
            desired_fingers = max(
                2,
                int(math.ceil(effective_width_um / fold_height_um)),
                int(math.ceil(multiplicity)),
            )
            if desired_fingers % 2:
                desired_fingers += 1
            device_fold_height = effective_width_um / desired_fingers
        needs_fold = (
            effective_width_um > 2.0 * fold_height_um + 1e-12
            or device.instance_name in common_centroid_members
            and (multiplicity < 2 or int(multiplicity) % 2 != 0)
        )
        if needs_fold:
            annotation_lines.append(
                f"{device.instance_name}: @fold({format(device_fold_height, '.12g')});"
            )

    return OptimizedLayoutStyle(
        layout_lines=tuple(lines),
        annotation_lines=tuple(annotation_lines),
    )


__all__ = [
    "canonicalize_fold_compatible_sizing",
    "DEFAULT_ANALOG_SUPPLY_TRUNK_WIDTH_UM",
    "DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM",
    "OptimizedLayoutStyle",
    "derive_optimized_layout_style",
    "macro_partition_place_group_name",
]
