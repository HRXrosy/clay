"""Lower Clay functional role declarations into viewer-facing metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from clay.ir import CellDef, ParsedFunctionalRole
from clay.parser import parse_clay_file


STRUCTURAL_ROLE_ATTRS = (
    "structure_role",
    "placement_column",
    "placement_lane",
    "placement_anchor",
    "input_nets",
    "output_nets",
    "bias_nets",
    "signal_path",
    "placement_intent",
    "layout_style",
    "row_group",
    "row_min",
    "row_max",
    "area_priority",
)


@dataclass(frozen=True)
class FunctionalGroupRecord:
    """One authored functional group, before it is projected onto devices.

    Unlike :func:`functional_role_catalog_from_cell`, this representation keeps
    group identity and therefore permits a device to participate in more than
    one functional role.  Physical ownership is resolved later by the macro
    graph; a functional group is only a semantic label.
    """

    name: str
    role: str
    members: tuple[str, ...]
    structure_role: str = ""
    attributes: tuple[tuple[str, str], ...] = ()


def functional_group_records_from_cell(
    cell: CellDef,
) -> tuple[FunctionalGroupRecord, ...]:
    """Return canonical group-level records from ``cell.functional_roles``.

    No structure is inferred from names, device kinds, or circuit identity.
    ``structure_role`` is preserved only when it was explicitly authored.
    """

    records: list[FunctionalGroupRecord] = []
    seen_names: set[str] = set()
    for parsed in getattr(cell, "functional_roles", []) or []:
        attrs = dict(getattr(parsed, "attributes", {}) or {})
        members = _attr_list(attrs, "devices") or _attr_list(attrs, "members")
        if not members:
            continue
        name = str(getattr(parsed, "name", "") or "")
        if not name:
            raise ValueError("functional group name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate functional group {name!r}")
        seen_names.add(name)

        canonical_members = tuple(sorted(str(member) for member in members))
        if len(canonical_members) != len(set(canonical_members)):
            raise ValueError(f"functional group {name!r} contains duplicate members")
        role = _attr_text(attrs, "role") or name
        structure_role = _attr_text(attrs, "structure_role")
        normalized_attrs = tuple(
            sorted(
                (str(key), _canonical_attribute_text(value))
                for key, value in attrs.items()
                if key not in {"devices", "members"}
            )
        )
        records.append(
            FunctionalGroupRecord(
                name=name,
                role=role,
                members=canonical_members,
                structure_role=structure_role,
                attributes=normalized_attrs,
            )
        )
    return tuple(sorted(records, key=lambda record: record.name))


def functional_role_catalog_from_clay(
    clay_path: str | Path,
    cell_name: str | None = None,
) -> dict[str, dict[str, str]]:
    """Return device-id keyed role metadata from a Clay source file.

    The source of truth is `functional_group` declarations whose
    attributes include a `devices = (...)` list. Concrete role tokens are
    attribute values, so new circuit families do not require frontend or
    grammar changes.
    """
    cf = parse_clay_file(clay_path)
    cells = list(getattr(cf, "cell_defs", []) or [])
    if cell_name is not None:
        cells = [
            cell
            for cell in cells
            if getattr(cell, "local_name", getattr(cell, "name", None)) == cell_name
        ]
    if not cells:
        target = cell_name or "<first cell>"
        raise ValueError(f"no cell {target!r} found in {clay_path}")
    if len(cells) > 1 and cell_name is None:
        raise ValueError("cell_name is required when a Clay file has multiple cells")
    return functional_role_catalog_from_cell(cells[0])


def functional_role_catalog_from_cell(cell: CellDef) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for role in getattr(cell, "functional_roles", []) or []:
        for device, metadata in _role_entries(role).items():
            existing = catalog.get(device)
            if existing is not None and existing != metadata:
                raise ValueError(
                    f"device {device!r} has conflicting functional role metadata: "
                    f"{existing!r} vs {metadata!r}"
                )
            catalog[device] = metadata
    return catalog


def _role_entries(role: ParsedFunctionalRole) -> dict[str, dict[str, str]]:
    attrs = dict(getattr(role, "attributes", {}) or {})
    devices = _attr_list(attrs, "devices") or _attr_list(attrs, "members")
    if not devices:
        return {}

    role_token = _attr_text(attrs, "role") or str(getattr(role, "name", ""))
    if not role_token:
        role_token = str(getattr(role, "kind", "functional_group"))
    group_token = _attr_text(attrs, "group") or role_token
    role_label = _attr_text(attrs, "label") or _humanize(role_token)
    group_label = _attr_text(attrs, "group_label") or role_label
    color = _attr_text(attrs, "color")
    intent = _attr_text(attrs, "intent")
    source = f"{getattr(role, 'kind', 'functional_role')}:{getattr(role, 'name', role_token)}"

    metadata = {
        "role": role_token,
        "role_group": group_token,
        "role_label": role_label,
        "role_group_label": group_label,
        "intent": intent,
        "role_source": source,
    }
    for key in STRUCTURAL_ROLE_ATTRS:
        if key in attrs:
            metadata[key] = _attr_metadata_value(attrs[key])
    if color:
        metadata["role_color"] = color
    return {device: dict(metadata) for device in devices}


def _attr_text(attrs: Mapping[str, Any], key: str) -> str:
    value = attrs.get(key)
    if value is None:
        return ""
    if isinstance(value, list):
        raise ValueError(f"functional role attribute {key!r} must be scalar")
    return str(value)


def _attr_list(attrs: Mapping[str, Any], key: str) -> list[str]:
    value = attrs.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"functional role attribute {key!r} must be a list")
    return [str(item) for item in value]


def _attr_metadata_value(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _canonical_attribute_text(value: Any) -> str:
    """Serialize one parsed attribute without losing list boundaries."""

    if isinstance(value, (list, tuple)):
        return json.dumps(
            [str(item) for item in value],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _humanize(token: str) -> str:
    return token.replace("_", " ").strip().capitalize()
