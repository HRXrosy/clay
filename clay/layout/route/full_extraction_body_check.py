"""Declaration-to-artwork MOS body gate over KLayout full extraction.

The electrical observation in this module is deliberately narrow: four-terminal
MOS records read from the ``.lvsdb`` produced by KLayout's full-extraction deck.
Native route connectivity and three-terminal extraction are not substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


def _norm_net(value: object) -> str:
    return str(value or "").strip().casefold()


def _is_anonymous_net(value: object) -> bool:
    net = _norm_net(value)
    return not net or net == "none" or net.startswith("$")


def _mos_kind(value: object) -> Optional[str]:
    raw = str(value or "").strip().casefold()
    if raw in {"pmos", "pmos3v", "pch", "pch3", "pd"} or raw.startswith(
        ("pmos_", "pch_")
    ) or (
        raw.startswith("p")
        and len(raw) > 1
        and (raw[1].isdigit() or raw[1:].startswith(("fet", "mos", "ch")))
    ):
        return "pmos"
    if raw in {"nmos", "nmos3v", "nch", "nch3", "nd"} or raw.startswith(
        ("nmos_", "nch_")
    ) or (
        raw.startswith("n")
        and len(raw) > 1
        and (raw[1].isdigit() or raw[1:].startswith(("fet", "mos", "ch")))
    ):
        return "nmos"
    return None


@dataclass(frozen=True)
class DeclaredMosBody:
    name: str
    kind: str
    drain: str
    gate: str
    source: str
    body: str
    width_um: float
    length_um: float


@dataclass(frozen=True)
class ExtractedMosBody:
    name: str
    kind: str
    drain: str
    gate: str
    source: str
    body: str
    width_um: float
    length_um: float


def canonicalize_extracted_mos_nets(
    extracted: Sequence[ExtractedMosBody],
    layout_to_schematic: dict[str, str],
) -> Tuple[ExtractedMosBody, ...]:
    """Project full-extraction terminal tokens through the LVS cross-reference.

    KLayout commonly names layout-side nets ``$<id>`` even when the same LVSDB
    has associated them with a declared schematic net.  Body checking must use
    that association for all four terminals: otherwise anonymous D/G/S tokens
    become topology wildcards and a correctly tied B terminal is reported as an
    unrelated floating net.
    """

    canonical: dict[str, str] = {}
    for raw, schematic in layout_to_schematic.items():
        target = str(schematic or "").strip()
        key = _norm_net(raw)
        if not key or not target:
            continue
        aliases = {key, key.lstrip("$")}
        for alias in aliases:
            if alias:
                canonical[alias] = target

    def project(value: str) -> str:
        key = _norm_net(value)
        return canonical.get(key, canonical.get(key.lstrip("$"), value))

    return tuple(
        replace(
            device,
            drain=project(device.drain),
            gate=project(device.gate),
            source=project(device.source),
            body=project(device.body),
        )
        for device in extracted
    )


@dataclass(frozen=True)
class BodyTerminalRecord:
    extracted_device: str
    kind: str
    actual_body: str
    expected_body: Optional[str]
    declared_devices: Tuple[str, ...]
    correct: bool
    reason: str = ""


@dataclass(frozen=True)
class BodyMaterializationReport:
    records: Tuple[BodyTerminalRecord, ...]
    coverage_complete: bool = True
    declared: Tuple[DeclaredMosBody, ...] = ()

    @property
    def ambiguous_count(self) -> int:
        return sum(row.expected_body is None for row in self.records)

    @property
    def measured(self) -> bool:
        return (
            bool(self.records)
            and self.ambiguous_count == 0
            and self.coverage_complete
        )

    @property
    def passed(self) -> bool:
        return self.measured and all(row.correct for row in self.records)

    @property
    def mismatches(self) -> Tuple[BodyTerminalRecord, ...]:
        return tuple(
            row for row in self.records
            if row.expected_body is not None and not row.correct
        )

    def _counts(self, predicate) -> dict[str, int]:
        return {
            kind: sum(
                1 for row in self.records
                if row.kind == kind and predicate(row)
            )
            for kind in ("pmos", "nmos")
        }

    @property
    def total_by_kind(self) -> dict[str, int]:
        return self._counts(lambda _row: True)

    @property
    def correct_by_kind(self) -> dict[str, int]:
        return self._counts(lambda row: row.correct)

    @property
    def mismatch_by_kind(self) -> dict[str, int]:
        return self._counts(
            lambda row: row.expected_body is not None and not row.correct
        )

    def _declaration_counts(self, predicate) -> dict[str, int]:
        return {
            kind: sum(
                1
                for declaration in self.declared
                if _mos_kind(declaration.kind) == kind
                and predicate(
                    tuple(
                        row
                        for row in self.records
                        if declaration.name in row.declared_devices
                    )
                )
            )
            for kind in ("pmos", "nmos")
        }

    @property
    def declared_count_by_kind(self) -> dict[str, int]:
        """Logical Clay declarations, independent of extractor combining."""

        return self._declaration_counts(lambda _rows: True)

    @property
    def covered_declaration_count_by_kind(self) -> dict[str, int]:
        """Declarations associated with at least one measured four-terminal row."""

        return self._declaration_counts(
            lambda rows: bool(rows) and all(row.expected_body is not None for row in rows)
        )

    @property
    def correct_declaration_count_by_kind(self) -> dict[str, int]:
        """Declarations whose complete associated extraction has the declared B net."""

        return self._declaration_counts(
            lambda rows: bool(rows) and all(row.correct for row in rows)
        )


def _close(a: float, b: float, *, atol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= atol * max(1.0, abs(float(a)), abs(float(b)))


def _named_net_compatible(extracted: str, declared: str) -> bool:
    return _is_anonymous_net(extracted) or _norm_net(extracted) == _norm_net(declared)


def _topology_matches(extracted: ExtractedMosBody, declared: DeclaredMosBody) -> bool:
    if _mos_kind(extracted.kind) != _mos_kind(declared.kind):
        return False
    if not _close(extracted.length_um, declared.length_um):
        return False
    if not _named_net_compatible(extracted.gate, declared.gate):
        return False

    # MOS source/drain are symmetric for association.  Anonymous extracted nets
    # are open fragments and act only as unknowns; named terminals must agree.
    direct = (
        _named_net_compatible(extracted.source, declared.source)
        and _named_net_compatible(extracted.drain, declared.drain)
    )
    swapped = (
        _named_net_compatible(extracted.source, declared.drain)
        and _named_net_compatible(extracted.drain, declared.source)
    )
    return direct or swapped


def _declared_group_key(device: DeclaredMosBody) -> tuple[object, ...]:
    """Exact source topology/body group, with MOS source/drain symmetry."""

    return (
        _mos_kind(device.kind) or str(device.kind).casefold(),
        round(float(device.length_um), 9),
        _norm_net(device.gate),
        tuple(sorted((_norm_net(device.source), _norm_net(device.drain)))),
        _norm_net(device.body),
    )


def _declared_lvs_device_name(device: DeclaredMosBody) -> str:
    """KLayout display name of the flat MOS card emitted for a declaration.

    SPICE consumes the first ``M`` as the card class designator, so a Clay
    instance ``MM28`` is displayed by KLayout as ``M28`` (and ``MNM0`` as
    ``NM0``).  This mirrors the generic effective-reference writer without
    importing the flow layer into the route/body checker.
    """

    name = str(device.name or "")
    if name[:1].casefold() == "x":
        name = name[1:]
    if name[:1].casefold() != "m":
        name = "M" + name
    return name[1:]


def evaluate_body_materialization(
    extracted: Sequence[ExtractedMosBody],
    declared: Sequence[DeclaredMosBody],
    *,
    extracted_to_schematic: Mapping[str, str] | None = None,
) -> BodyMaterializationReport:
    """Compare every extracted MOS B terminal with its declared-body obligation.

    Device association deliberately excludes B.  Multiple topology candidates
    are acceptable only when every candidate declares the same body net; an
    ambiguity across different body nets is unmeasurable and never passes.
    """

    declared_groups: dict[tuple[object, ...], list[DeclaredMosBody]] = {}
    for dec in declared:
        declared_groups.setdefault(_declared_group_key(dec), []).append(dec)

    # A clean full-LVS database already carries the authoritative layout-device
    # -> schematic-device association.  Use it when supplied instead of
    # re-comparing W with a second, much tighter tolerance than the deck.  The
    # topology check below remains active and the B terminal is still compared
    # exactly; xref only proves which declaration group the extracted row covers.
    reference_name_groups: dict[str, set[tuple[object, ...]]] = {}
    for key, group in declared_groups.items():
        for dec in group:
            reference_name_groups.setdefault(
                _norm_net(_declared_lvs_device_name(dec)), set()
            ).add(key)
    xref_mode = extracted_to_schematic is not None
    device_xref = {
        _norm_net(layout_name): str(schematic_name)
        for layout_name, schematic_name in (extracted_to_schematic or {}).items()
    }

    rows = []
    assigned_group_by_row: list[Optional[tuple[object, ...]]] = []
    for ext in extracted:
        if xref_mode:
            schematic_name = device_xref.get(_norm_net(ext.name))
            mapped_keys = reference_name_groups.get(
                _norm_net(schematic_name), set()
            ) if schematic_name else set()
            if mapped_keys:
                candidate_keys = tuple(
                    key
                    for key in mapped_keys
                    if any(
                        _topology_matches(ext, dec)
                        for dec in declared_groups[key]
                    )
                )
            elif schematic_name:
                # KLayout combine_devices may associate one layout MOS with a
                # generated schematic topology name (for example
                # ``VDD_VDD_PMOS_3P3``) instead of any original instance.  The
                # xref still proves this is a compared device, but not which
                # declaration group it covers.  Recover that association only
                # under the strict aggregate contract: one D/G/S/L topology
                # group whose summed effective width equals the extracted row.
                # The normal uniqueness/body checks below remain mandatory.
                candidate_keys = tuple(
                    key
                    for key, group in declared_groups.items()
                    if any(_topology_matches(ext, dec) for dec in group)
                    and _close(
                        ext.width_um,
                        sum(float(dec.width_um) for dec in group),
                    )
                )
            else:
                candidate_keys = ()
        else:
            candidate_keys = tuple(
                key
                for key, group in declared_groups.items()
                if any(_topology_matches(ext, dec) for dec in group)
            )
        candidates = tuple(
            dec for key in candidate_keys for dec in declared_groups[key]
        )
        bodies = {
            _norm_net(dec.body): dec.body for dec in candidates if _norm_net(dec.body)
        }
        names = tuple(sorted(dec.name for dec in candidates))
        if not candidate_keys:
            rows.append(BodyTerminalRecord(
                extracted_device=ext.name,
                kind=_mos_kind(ext.kind) or str(ext.kind),
                actual_body=ext.body,
                expected_body=None,
                declared_devices=(),
                correct=False,
                reason=(
                    (
                        "full-LVS device cross-reference has no unique declared "
                        "MOS group with matching D/G/S/L topology"
                    ) if xref_mode else (
                        "no declared MOS group has matching D/G/S/L topology and "
                        "enough effective width for the extracted record"
                    )
                ),
            ))
            assigned_group_by_row.append(None)
            continue
        if len(candidate_keys) != 1 or len(bodies) != 1:
            body_names = tuple(sorted(str(v) for v in bodies.values()))
            rows.append(BodyTerminalRecord(
                extracted_device=ext.name,
                kind=_mos_kind(ext.kind) or str(ext.kind),
                actual_body=ext.body,
                expected_body=None,
                declared_devices=names,
                correct=False,
                reason=(
                    "ambiguous declared topology/body obligations for extracted "
                    "MOS: " + ", ".join(body_names or ("NONE",))
                ),
            ))
            assigned_group_by_row.append(None)
            continue
        assigned_key = candidate_keys[0]
        expected = next(iter(bodies.values()))
        correct = _norm_net(ext.body) == _norm_net(expected)
        rows.append(BodyTerminalRecord(
            extracted_device=ext.name,
            kind=_mos_kind(ext.kind) or str(ext.kind),
            actual_body=ext.body,
            expected_body=expected,
            declared_devices=names,
            correct=correct,
            reason="" if correct else (
                f"full-extraction B={ext.body or 'NONE'}; declared body={expected}"
            ),
        ))
        assigned_group_by_row.append(assigned_key)

    # KLayout may split one declared device into multiple records or combine
    # parallel declarations into one.  Association alone is insufficient: every
    # source group must be covered exactly once in aggregate effective width, so
    # a missing fragment or duplicated extraction cannot reuse the same capacity.
    coverage_complete = True
    for key, group in declared_groups.items():
        indices = [
            index for index, assigned in enumerate(assigned_group_by_row)
            if assigned == key
        ]
        declared_width = sum(float(dec.width_um) for dec in group)
        extracted_width = sum(float(extracted[index].width_um) for index in indices)
        if xref_mode:
            if len(indices) == 1:
                continue
            reason = (
                "full-LVS device cross-reference covers declared MOS group "
                f"{len(indices)} times (expected exactly one matched device)"
            )
        else:
            if _close(extracted_width, declared_width):
                continue
            reason = (
                f"aggregate extracted width {extracted_width:g}um does not cover "
                f"declared effective width {declared_width:g}um"
            )
        coverage_complete = False
        names = tuple(sorted(dec.name for dec in group))
        expected_body = group[0].body
        if indices:
            for index in indices:
                prior = rows[index].reason
                rows[index] = replace(
                    rows[index],
                    correct=False,
                    reason=f"{prior}; {reason}" if prior else reason,
                )
        else:
            rows.append(BodyTerminalRecord(
                extracted_device="<missing-extracted-group>",
                kind=_mos_kind(group[0].kind) or str(group[0].kind),
                actual_body="",
                expected_body=expected_body,
                declared_devices=names,
                correct=False,
                reason=reason,
            ))
            assigned_group_by_row.append(key)
    return BodyMaterializationReport(
        records=tuple(rows),
        coverage_complete=coverage_complete,
        declared=tuple(declared),
    )


def declared_mos_bodies(
    cell_def: Any,
    *,
    pdk_id: Optional[str] = None,
) -> Tuple[DeclaredMosBody, ...]:
    """Project Clay MOS declarations into target-drawn body obligations.

    The target PDK may legally materialize a source declaration at its minimum
    drawn W/L.  Reuse that same data-driven clamp here so the full-extraction
    xref associates the materialized MOS with its declaration before checking
    the four-terminal body net.  Omitting ``pdk_id`` preserves the raw SI-IR
    projection used by existing callers.
    """

    from clay.layout.lvs_reference_emit import _pdk_drawn_minimums

    min_length_um, min_width_um = _pdk_drawn_minimums(pdk_id)

    out = []
    for dev in getattr(cell_def, "devices", ()) or ():
        kind = _mos_kind(getattr(dev, "device_class", ""))
        if kind is None:
            continue
        pins = getattr(dev, "pins", None) or {}
        sizing = getattr(dev, "sizing", None) or {}
        width_m = float(sizing.get("W", 0.0) or 0.0)
        length_m = float(sizing.get("L", 0.0) or 0.0)
        multiplicity = float(
            sizing.get("M", sizing.get("m", sizing.get("nf", 1.0))) or 1.0
        )
        per_width_um = width_m * 1e6
        length_um = length_m * 1e6
        if min_width_um is not None:
            per_width_um = max(per_width_um, min_width_um)
        if min_length_um is not None:
            length_um = max(length_um, min_length_um)

        def pin(*roles: str) -> str:
            for role in roles:
                value = pins.get(role)
                if value is not None:
                    return str(value)
            return ""

        out.append(DeclaredMosBody(
            name=str(getattr(dev, "instance_name", "") or ""),
            kind=kind,
            drain=pin("d", "drain"),
            gate=pin("g", "gate"),
            source=pin("s", "source"),
            body=pin("b", "body", "bulk"),
            width_um=per_width_um * multiplicity,
            length_um=length_um,
        ))
    return tuple(out)


def _kname(value: object) -> str:
    if value is None:
        return ""
    for attr in ("name", "expanded_name"):
        candidate = getattr(value, attr, None)
        if candidate is None:
            continue
        try:
            candidate = candidate() if callable(candidate) else candidate
        except Exception:
            continue
        if candidate:
            return str(candidate)
    return ""


def _lvs_net_xref_mapping(database: Any) -> dict[str, str]:
    """Return layout-net -> schematic-net names from one full LVS database."""

    xref = database.xref()
    if xref is None:
        return {}
    mapping: dict[str, str] = {}
    for circuit_pair in xref.each_circuit_pair():
        for net_pair in xref.each_net_pair(circuit_pair):
            layout_name = _kname(net_pair.first())
            schematic_name = _kname(net_pair.second())
            if not layout_name or not schematic_name:
                continue
            for key in {layout_name, layout_name.lstrip("$")}:
                normalized = _norm_net(key)
                previous = mapping.get(normalized)
                if previous is not None and _norm_net(previous) != _norm_net(
                    schematic_name
                ):
                    raise ValueError(
                        "full-extraction LVS cross-reference maps layout net "
                        f"{layout_name!r} to multiple schematic nets: "
                        f"{previous!r}, {schematic_name!r}"
                    )
                mapping[normalized] = schematic_name
    return mapping


def _lvs_device_xref_mapping(database: Any) -> dict[str, str]:
    """Return matched layout-MOS -> schematic-MOS names from the LVSDB.

    Only device pairs the full comparer reports as ``Match`` or
    ``MatchWithWarning`` are evidence.  Missing/conflicting associations are
    rejected by the downstream coverage gate rather than filled by topology or
    width guesses.
    """

    xref = database.xref()
    if xref is None:
        return {}
    mapping: dict[str, str] = {}
    for circuit_pair in xref.each_circuit_pair():
        for device_pair in xref.each_device_pair(circuit_pair):
            layout_device = device_pair.first()
            schematic_device = device_pair.second()
            if layout_device is None or schematic_device is None:
                continue
            if str(device_pair.status()) not in {"Match", "MatchWithWarning"}:
                continue
            if _mos_kind(layout_device.device_class().name) is None:
                continue
            layout_name = _kname(layout_device)
            schematic_name = _kname(schematic_device)
            if not layout_name or not schematic_name:
                continue
            key = _norm_net(layout_name)
            previous = mapping.get(key)
            if previous is not None and _norm_net(previous) != _norm_net(
                schematic_name
            ):
                raise ValueError(
                    "full-extraction LVS cross-reference maps layout device "
                    f"{layout_name!r} to multiple schematic devices: "
                    f"{previous!r}, {schematic_name!r}"
                )
            mapping[key] = schematic_name
    return mapping


def _read_full_extraction_mos_bodies_and_xref(
    lvsdb_path: str | Path,
) -> tuple[Tuple[ExtractedMosBody, ...], dict[str, str]]:
    """Read four-terminal layout MOS records and their full-LVS device xref."""

    import klayout.db as kdb

    path = Path(lvsdb_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"full-extraction lvsdb is absent or empty: {path}")
    database = kdb.LayoutVsSchematic()
    database.read(str(path))
    net_xref = _lvs_net_xref_mapping(database)
    device_xref = _lvs_device_xref_mapping(database)
    netlist = database.netlist()
    if netlist is None:
        raise ValueError(f"full-extraction lvsdb has no layout netlist: {path}")
    circuits = tuple(netlist.each_circuit())
    if not circuits:
        raise ValueError(f"full-extraction lvsdb has no layout circuit: {path}")

    out = []
    for device in circuits[0].each_device():
        kind = _mos_kind(device.device_class().name)
        if kind is None:
            continue
        terminal_ids = {
            definition.name: definition.id()
            for definition in device.device_class().terminal_definitions()
        }
        required = {"S", "G", "D", "B"}
        if not required <= terminal_ids.keys():
            raise ValueError(
                f"MOS class {device.device_class().name!r} is not four-terminal; "
                "a three-terminal/native deck cannot satisfy the body gate"
            )

        def net(role: str) -> str:
            return _kname(device.net_for_terminal(terminal_ids[role]))

        try:
            width_um = float(device.parameter("W"))
            length_um = float(device.parameter("L"))
        except Exception as exc:
            raise ValueError(
                f"full-extraction MOS {_kname(device)!r} lacks W/L parameters"
            ) from exc
        out.append(ExtractedMosBody(
            name=_kname(device),
            kind=kind,
            drain=net("D"),
            gate=net("G"),
            source=net("S"),
            body=net("B"),
            width_um=width_um,
            length_um=length_um,
        ))
    if not out:
        raise ValueError(
            f"full-extraction lvsdb contains no four-terminal MOS records: {path}"
        )
    return canonicalize_extracted_mos_nets(out, net_xref), device_xref


def read_full_extraction_mos_bodies(
    lvsdb_path: str | Path,
) -> Tuple[ExtractedMosBody, ...]:
    """Read four-terminal layout MOS records from a KLayout full LVS database."""

    extracted, _device_xref = _read_full_extraction_mos_bodies_and_xref(
        lvsdb_path
    )
    return extracted


def check_full_extraction_body_materialization(
    lvsdb_path: str | Path,
    cell_def: Any,
    *,
    pdk_id: Optional[str] = None,
) -> BodyMaterializationReport:
    extracted, device_xref = _read_full_extraction_mos_bodies_and_xref(
        lvsdb_path
    )
    return evaluate_body_materialization(
        extracted,
        declared_mos_bodies(cell_def, pdk_id=pdk_id),
        extracted_to_schematic=device_xref,
    )
