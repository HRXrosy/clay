"""Bridged MOS/passive/BJT data models, execution-scope primitives, and leaf utilities for route_cell."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.extract.realize_render import resolve_device_fold_target
from clay.layout.source_bjt_materialization import BjtMaterializationUnavailable
from clay.layout.tech import LayoutTech


def _rc_attr(name: str, default: Any) -> Any:
    """Resolve module-level attributes via clay.layout.route.route_cell so monkeypatches apply."""
    mod = sys.modules.get("clay.layout.route.route_cell")
    return getattr(mod, name, default) if mod is not None else default


_NM_PER_UM = 1000.0
_XY = Tuple[float, float]

# HVH direction-discipline soft weight (base-step-cost units) emitted when the
# design DECLARES a routing direction discipline (any @trunk). A design-side style
# knob (how much an off-preferred-axis step costs), never a reference-measured
# value; 0 for a cell that declares no @trunk keeps the route byte-identical.
_HVH_DIRECTION_WEIGHT = 1.0

# CellDef device pin keys -> the role keys extract_pins / build_device_geometry use.
_PIN_KEY_TO_ROLE = {"s": "source", "d": "drain", "g": "gate", "b": "body"}
_ROLE_TO_TERM = {"source": "s", "drain": "d", "gate": "g", "body": "b"}


def _declared_ground_ports(cell_def: Any) -> Tuple[str, ...]:
    """Semantic substrate-port names carried by Clay intent, in source order."""

    from clay.port_roles import declared_substrate_ports

    return declared_substrate_ports(
        getattr(cell_def, "port_roles", {}) or {}
    )


class RouteExecutionScope(str, Enum):
    SIGNOFF = "signoff"
    BRINGUP_DIAGNOSTIC = "bringup_diagnostic"


@dataclass(frozen=True)
class _DiagnosticRouteCompilerProduct:
    route_program: Any
    source_unknown: Any
    family_count: int
    family_winner_count: int
    failed_family_refs: Tuple[str, ...]
    unrouted_nets: Tuple[str, ...]


# --------------------------------------------------------------------------- #
# the place_generic <-> route/ data-model bridge
# --------------------------------------------------------------------------- #
# Recognized MOS device classes -> (polarity, gate-oxide domain). A thick 3.3V kind
# (pmos3v/nmos3v/pch3/nch3/PD/ND) MUST resolve to its polarity here so it is neither
# dropped by the MOS-selection filter nor by downstream ``kind in ("nmos","pmos")``
# checks; the oxide domain rides separately to build_device(oxide=). Plain thin kinds
# are byte-identical.
_MOS_CLASSES = frozenset({
    "nmos", "pmos", "nmos3v", "pmos3v", "nch", "pch", "nch3", "pch3", "nd", "pd",
})
_MOS_THICK = frozenset({"nmos3v", "pmos3v", "nch3", "pch3", "nd", "pd"})


def _mos_kind_oxide(device_class: Any) -> Tuple[Optional[str], str]:
    """``(polarity | None, oxide)`` for a device_class. ``None`` kind = not a MOS."""
    raw = str(device_class or "").lower()
    if raw not in _MOS_CLASSES:
        return None, "thin"
    oxide = "thick" if raw in _MOS_THICK else "thin"
    return ("pmos" if raw.startswith("p") else "nmos"), oxide


class _BridgedDevice:
    """One MOS device in the shape ``extract_pins`` / ``build_device_geometry`` read.

    Built from a parsed ``CellDef`` device record (``instance_name`` + ``pins``
    d/g/s/b->net + ``sizing`` W/L in METRES, M + ``device_class`` nmos/pmos). The
    route subsystem wants ``.name/.kind/.W(um)/.L(um)/.M`` and a ``.nets`` map
    keyed by role (``source``/``drain``/``gate``/``body``)."""

    __slots__ = ("name", "kind", "oxide", "nets", "W", "L", "M")

    def __init__(self, cd_dev: Any) -> None:
        self.name = str(getattr(cd_dev, "instance_name", ""))
        # Normalize the declared device_class to polarity + carry oxide separately
        # (see _mos_kind_oxide) so a 3.3V thick kind is not dropped downstream.
        _pol, self.oxide = _mos_kind_oxide(getattr(cd_dev, "device_class", ""))
        self.kind = _pol or str(getattr(cd_dev, "device_class", "")).lower()
        sizing = getattr(cd_dev, "sizing", None) or {}
        # sizing W/L are METRES (the materialised IR currency); route wants MICRONS.
        self.W = float(sizing.get("W", 0.0)) * 1e6
        self.L = float(sizing.get("L", 0.0)) * 1e6
        self.M = max(1, int(round(float(sizing.get("M", 1) or 1))))
        pins = getattr(cd_dev, "pins", None) or {}
        self.nets = {
            role: str(pins[k])
            for k, role in _PIN_KEY_TO_ROLE.items()
            if k in pins and pins[k] is not None
        }


class _BridgedPassive:
    """One passive (resistor / MIM cap) in the shape ``extract_pins`` /
    ``cap_plate_conductors`` / ``emit_labelled_gds`` read.

    Built from a parsed ``CellDef`` device record whose ``device_class`` is
    ``res``/``cap``. The route subsystem's passive consumers want an
    ``ExtractedPassive``-shaped view: ``.name``, ``.kind`` (``cap``/``res``),
    ``.value`` (farads / ohms), ``.nets`` map keyed by the two terminal roles
    ``p``/``n``, ``.M`` (parallel-unit bank count), and ``.model`` (the netlist
    ``$.MODEL`` annotation -- e.g. the MIM level). M/model ride along from the
    parsed sizing when present (``1``/``''`` otherwise), exactly as the mature OTA
    passive path carries them -- never a reference coordinate."""

    __slots__ = (
        "name", "kind", "value", "nets", "M", "model", "orient",
        "resistor_segments",
    )

    def __init__(
        self,
        cd_dev: Any,
        *,
        orient: str = "R0",
        resistor_segments: Any = None,
    ) -> None:
        self.name = str(getattr(cd_dev, "instance_name", ""))
        self.kind = str(getattr(cd_dev, "device_class", "")).lower()
        sizing = getattr(cd_dev, "sizing", None) or {}
        # value lives under the class key (R/C) or a generic 'value'.
        self.value = float(
            sizing.get("R", sizing.get("C", sizing.get("value", 0.0))) or 0.0
        )
        pins = getattr(cd_dev, "pins", None) or {}
        p = pins.get("p") or pins.get("plus")
        n = pins.get("n") or pins.get("minus")
        self.nets = {"p": str(p) if p is not None else "",
                     "n": str(n) if n is not None else ""}
        self.M = max(1, int(round(float(sizing.get("M", sizing.get("m", 1)) or 1))))
        self.model = str(
            sizing.get("model", "")
            or getattr(cd_dev, "source_model", "")
            or getattr(cd_dev, "model", "")
            or ""
        )
        normalized_orient = str(orient or "R0").upper()
        if normalized_orient not in {
            "R0", "R90", "R180", "R270",
            "MR0", "MR90", "MR180", "MR270",
        }:
            raise ValueError(
                f"passive {self.name!r} has unsupported orientation "
                f"{normalized_orient!r}"
            )
        self.orient = normalized_orient
        self.resistor_segments = resistor_segments


class _BridgedBjt:
    """Three-terminal BJT topology view carried independently of MOS/passives."""

    __slots__ = (
        "name", "kind", "nets", "area", "M", "model", "tap_net", "tap_where"
    )

    def __init__(self, cd_dev: Any, tap_declaration: Any = None) -> None:
        self.name = str(getattr(cd_dev, "instance_name", ""))
        self.kind = str(getattr(cd_dev, "device_class", "")).lower()
        pins = getattr(cd_dev, "pins", None) or {}
        self.nets = {role: str(pins.get(role, "")) for role in ("c", "b", "e")}
        sizing = getattr(cd_dev, "sizing", None) or {}
        self.area = float(sizing.get("area", 1.0) or 1.0)
        self.M = float(sizing.get("M", 1.0) or 1.0)
        self.model = str(getattr(cd_dev, "source_model", "") or "")
        self.tap_net = str(getattr(tap_declaration, "net", "") or "")
        self.tap_where = str(getattr(tap_declaration, "where", "") or "")


class _PlacedStamp:
    """A ``polys_by_layer`` (centroid-relative um rings) view of a device the
    placer ALREADY drew.

    The placer (:func:`clay.layout.generic_placement.place_generic`) emits each
    passive's translated artwork into ``PlacedDevice.polys`` (absolute nm). The
    route subsystem's obstacle builder / short oracle / emit loop read a stamp's
    ``polys_by_layer`` in CENTROID-RELATIVE um. Re-centring the placer's own
    polygons (never a re-build) makes the route geometry the SAME artwork the
    placer drew -- the one-canonical-geometry rule the MOS fold authority
    established, applied to passives."""

    __slots__ = ("polys_by_layer", "name")

    def __init__(self, name: str, polys: Sequence[Any], cx_nm: float, cy_nm: float) -> None:
        self.name = str(name)
        by: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
        for lp in polys or []:
            key = (int(lp.layer), int(lp.datatype))
            ring = [((float(v.x) - cx_nm) / _NM_PER_UM, (float(v.y) - cy_nm) / _NM_PER_UM)
                    for v in lp.polygon.vertices]
            by.setdefault(key, []).append(ring)
        self.polys_by_layer = by


class _PlacedBjtStamp(_PlacedStamp):
    """Placed fixed-cell artwork plus exact centroid-relative C/B/E access."""

    __slots__ = (
        "terminal_offsets_um",
        "terminal_accesses_um",
        "terminal_layers",
        "native_terminal_nodes",
        "native_terminal_authority",
        "fixed_pcell",
        "well_tap_isolated",
        "physical_occurrences",
        "resolved_master_occurrences",
        "bjt_unit_occurrences",
    )

    def __init__(self, placed_device: Any, cx_nm: float, cy_nm: float) -> None:
        super().__init__(
            str(getattr(placed_device, "label", "")),
            getattr(placed_device, "polys", []) or [],
            cx_nm,
            cy_nm,
        )
        self.terminal_offsets_um = {
            str(role): (
                (float(point[0]) - cx_nm) / _NM_PER_UM,
                (float(point[1]) - cy_nm) / _NM_PER_UM,
            )
            for role, point in (getattr(placed_device, "terminals", {}) or {}).items()
        }
        self.terminal_accesses_um = {
            str(role): tuple(
                (
                    (float(point[0]) - cx_nm) / _NM_PER_UM,
                    (float(point[1]) - cy_nm) / _NM_PER_UM,
                )
                for point in points
            )
            for role, points in (
                getattr(placed_device, "terminal_accesses", {}) or {}
            ).items()
        }
        self.terminal_layers = {
            str(role): tuple(int(v) for v in layer)
            for role, layer in (getattr(placed_device, "term_layers", {}) or {}).items()
        }
        self.fixed_pcell = dict(getattr(placed_device, "fixed_pcell", {}) or {})
        self.well_tap_isolated = bool(
            getattr(placed_device, "well_tap_isolated", False)
        )
        physical_occurrences = tuple(
            getattr(placed_device, "physical_occurrences", ()) or ()
        )
        if physical_occurrences:
            self.physical_occurrences = physical_occurrences
        resolved_master_occurrences = tuple(
            getattr(placed_device, "resolved_master_occurrences", ()) or ()
        )
        if resolved_master_occurrences:
            self.resolved_master_occurrences = resolved_master_occurrences
        self.bjt_unit_occurrences = getattr(placed_device, "bjt_unit_occurrences", None)


class _PlacedPassiveStamp(_PlacedBjtStamp):
    """Actual passive artwork plus the same native terminal materialization."""

    __slots__ = ("materialized_passive_access", "passive_artwork_origin_um")


class _RecordDevice:
    """A parsed-``.cir`` MOS record presented in the shape :class:`_BridgedDevice`
    consumes.

    The production generic flow binds each cell's REAL net graph + sizing in the
    bound source ``.cir`` (``clay.layout.generic_placement.parse_cir_mos_records``
    -> ``name -> {kind, W, L, nf, pins}``), NOT in the parsed ``CellDef.devices``
    list (which a topology-only ``design.clay`` leaves empty). This light adapter
    rekeys ONE such record into the ``.instance_name/.device_class/.sizing/.pins``
    attributes :class:`_BridgedDevice` reads, so the SAME bridge serves both a cell
    whose devices are materialised in the ``.clay`` (the parsed-``CellDef`` path) and
    one whose devices live only in the netlist (the records path). The two are the
    SAME canonical source the placer + the router netlist already consume -- never a
    per-circuit branch. ``W``/``L`` ride in METRES (the ``.cir`` currency); ``nf`` is
    the finger/parallel multiplier (``M``)."""

    __slots__ = ("instance_name", "device_class", "sizing", "pins", "model")

    def __init__(self, name: str, rec: Dict[str, Any]) -> None:
        self.instance_name = str(name)
        self.device_class = str(rec.get("kind", ""))
        self.sizing = {
            "W": float(rec.get("W", 0.0) or 0.0),
            "L": float(rec.get("L", 0.0) or 0.0),
            "M": rec.get("nf", rec.get("M", 1)),
        }
        self.pins = dict(rec.get("pins", {}) or {})
        self.model = str(rec.get("model", "") or "")


class _RecordPassive:
    """A parsed-``.cir`` PASSIVE record (resistor / MIM cap) in the shape
    :class:`_BridgedPassive` consumes (the passive sibling of :class:`_RecordDevice`).

    Built from ``parse_cir_passive_records`` -> ``name -> {kind, value, pins, M}``.
    The value rides under the generic ``value`` key (ohms / farads); ``M`` is the
    parallel-unit multiplier."""

    __slots__ = ("instance_name", "device_class", "sizing", "pins", "model")

    def __init__(self, name: str, rec: Dict[str, Any]) -> None:
        self.instance_name = str(name)
        self.device_class = str(rec.get("kind", ""))
        self.sizing = {
            "value": float(rec.get("value", 0.0) or 0.0),
            "M": rec.get("M", rec.get("m", 1)),
            "model": str(rec.get("model", "") or ""),
        }
        self.pins = dict(rec.get("pins", {}) or {})
        self.model = str(rec.get("model", "") or "")


class _BridgedIntent:
    """Minimal ``ExtractedIntent``-shaped view: MOS ``.devices`` + ``.passives``.

    The MOS net graph plus the cell's resistors / MIM caps (each a
    :class:`_BridgedPassive`). ``no_route_under`` is None (no cap keep-out intent
    on this general path)."""

    __slots__ = (
        "devices",
        "passives",
        "bjts",
        "no_route_under",
        "well_isolated_devices",
        "substrate_net",
        "deep_nwell_net",
    )

    def __init__(
        self,
        devices: Sequence[_BridgedDevice],
        passives: Sequence[_BridgedPassive] = (),
        bjts: Sequence[_BridgedBjt] = (),
        well_isolated_devices: Sequence[str] = (),
        substrate_net: str = "",
        deep_nwell_net: str = "",
    ) -> None:
        self.devices = list(devices)
        self.passives: List[Any] = list(passives)
        self.bjts: List[Any] = list(bjts)
        self.no_route_under = None
        self.well_isolated_devices = tuple(str(name) for name in well_isolated_devices)
        self.substrate_net = str(substrate_net)
        self.deep_nwell_net = str(deep_nwell_net)


def _bridge_devices_from_records(
    device_records: Optional[Dict[str, Dict[str, Any]]],
    *,
    orient_by_name: Optional[Dict[str, str]] = None,
    resistor_segments_by_name: Optional[Dict[str, Any]] = None,
) -> Tuple[List["_BridgedDevice"], List["_BridgedPassive"]]:
    """Build ``(devices, passives)`` from a parsed-``.cir`` records map (the netlist
    binding the production flow already consumes), keyed by device name.

    The records are the SAME ``{name -> {kind, W/L/nf | value, pins}}`` map
    ``parse_cir_mos_records`` + ``parse_cir_passive_records`` produce (the router
    netlist source). MOS kinds (``nmos``/``pmos``) become bridged MOS devices; the
    2-terminal passives (``res``/``cap``) become bridged passives. GENERIC -- the
    only dispatch is on device ``kind`` (a PDK/electrical fact), never circuit name."""
    devices: List["_BridgedDevice"] = []
    passives: List["_BridgedPassive"] = []
    orientation = orient_by_name or {}
    segment_intent = resistor_segments_by_name or {}
    for name, rec in (device_records or {}).items():
        kind = str(rec.get("kind", "")).lower()
        if kind in ("nmos", "pmos"):
            devices.append(_BridgedDevice(_RecordDevice(name, rec)))
        elif kind in ("res", "cap"):
            passives.append(_BridgedPassive(
                _RecordPassive(name, rec),
                orient=orientation.get(name, "R0"),
                resistor_segments=segment_intent.get(name),
            ))
    return devices, passives


def bridge_devices_from_cell_def(
    cell_def: Any,
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List["_BridgedDevice"], List["_BridgedPassive"]]:
    """``(devices, passives)`` bridged from a parsed ``CellDef`` -- the SINGLE source
    for the cell_def -> bridged MOS/passive mapping.

    MOS (``nmos``/``pmos``) become bridged MOS devices, 2-terminal passives
    (``res``/``cap``) become bridged passives, taken from ``cell_def.devices`` when
    the cell materialises them; when that list is topology-only (empty) the binding
    falls back to the bound-``.cir`` ``device_records`` (the SAME netlist source the
    placer + router consume). The compaction placer bridge and the router bridge both
    call this so they read the IDENTICAL device universe -- no second derivation, no
    per-circuit branch (the only dispatch is on device ``kind``, a PDK/electrical
    fact)."""
    # Orientation is executable device intent.  The generic placer already reads
    # this exact annotation map; the route bridge must carry it onto the passive
    # view used by ``extract_pins`` so terminal probes are rebuilt in the same
    # coordinate frame as the placed/emitted stamp.
    from clay.layout.generic_placement import (
        _consume_resistor_segment_intent,
        _resistor_orient_map,
        _resistor_segment_intent_map,
    )

    orient_by_name = _resistor_orient_map(cell_def)
    resistor_segments_by_name = _resistor_segment_intent_map(cell_def)
    devices = [
        _BridgedDevice(d)
        for d in (getattr(cell_def, "devices", []) or [])
        if _mos_kind_oxide(getattr(d, "device_class", ""))[0] is not None
    ]
    passives = [
        _BridgedPassive(
            d,
            orient=orient_by_name.get(str(getattr(d, "instance_name", "")), "R0"),
            resistor_segments=resistor_segments_by_name.get(
                str(getattr(d, "instance_name", ""))
            ),
        )
        for d in (getattr(cell_def, "devices", []) or [])
        if str(getattr(d, "device_class", "")).lower() in ("res", "cap")
    ]
    if not devices and not passives:
        devices, passives = _bridge_devices_from_records(
            device_records,
            orient_by_name=orient_by_name,
            resistor_segments_by_name=resistor_segments_by_name,
        )
    validation_records = {
        passive.name: {
            "kind": passive.kind,
            "value": passive.value,
            "M": passive.M,
        }
        for passive in passives
    }
    if len(validation_records) != len(passives):
        raise ValueError("route passive bridge has duplicate instance identities")
    _consume_resistor_segment_intent(cell_def, validation_records)
    for passive in passives:
        if passive.resistor_segments != validation_records[passive.name].get(
            "resistor_segments"
        ):
            raise ValueError(
                f"route passive {passive.name!r} lost segmented-resistor authority"
            )
    from clay.layout.passive_split import realize_passive_split_views

    passives = realize_passive_split_views(cell_def, passives)
    return devices, passives


def bridge_bjts_from_cell_def(cell_def: Any) -> List[_BridgedBjt]:
    """Return all explicitly declared semantic PNP/NPN devices.

    BJT intent is required to pass through ``.clay``; unlike the historical MOS
    netlist fallback, an absent declaration is never reconstructed from a loose
    side file here.
    """
    from clay.layout.annotations import Tap

    annotations = getattr(cell_def, "__dict__", {}).get("_layout_annotations", {}) or {}
    out: List[_BridgedBjt] = []
    for device in (getattr(cell_def, "devices", []) or []):
        if str(getattr(device, "device_class", "")).lower() not in ("pnp", "npn"):
            continue
        declarations = [
            item
            for item in annotations.get(str(getattr(device, "instance_name", "")), ())
            if isinstance(item, Tap)
        ]
        if len(declarations) > 1:
            raise BjtMaterializationUnavailable(
                f"declared BJT {device.instance_name!r} has multiple @tap obligations"
            )
        out.append(_BridgedBjt(device, declarations[0] if declarations else None))
    return out



def _cell_bbox_um(
    placement_um: Dict[str, _XY], geometry: Dict[str, Any]
) -> Optional[Tuple[float, float, float, float]]:
    """Absolute (x0,y0,x1,y1) um bbox over every placed geometry stamp polygon."""
    xs: List[float] = []
    ys: List[float] = []
    for name, geom in (geometry or {}).items():
        pos = (placement_um or {}).get(name)
        if pos is None:
            continue
        cx, cy = float(pos[0]), float(pos[1])
        for _key, rings in (getattr(geom, "polys_by_layer", {}) or {}).items():
            for ring in rings:
                for (px, py) in ring:
                    xs.append(float(px) + cx)
                    ys.append(float(py) + cy)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class _BringupRailConstructionEvidence:
    """Stable named evidence for an exact bring-up rail construction stop."""

    demand_id: str
    net: str
    reason: str

    @property
    def refs(self) -> Tuple[str, str]:
        return f"net:{self.net}", f"demand:{self.demand_id}"

    def sha256(self) -> str:
        payload = {
            "demand_id": self.demand_id,
            "net": self.net,
            "reason": self.reason,
            "schema": "clay.bringup-rail-construction-stop.v1",
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()


class RoutePlanningStop(RuntimeError):
    """Typed pre-emission stop for a bounded canonical planning phase."""

    _MAX_DIAGNOSTIC_REFS = 8
    _MAX_DIAGNOSTIC_REF_CHARS = 160

    def __init__(
        self,
        *,
        phase: str,
        state: str,
        evidence: Any,
        status: VerificationStatus = VerificationStatus.UNKNOWN,
        feedback_context: Any = None,
        rail_observation: Any = None,
    ) -> None:
        self.phase = str(phase)
        self.state = str(state)
        self.status = status
        self.evidence = evidence
        if rail_observation is not None:
            from clay.layout.route.rail_diagnostics import RailPlacementStopObservation

            if type(rail_observation) is not RailPlacementStopObservation:
                raise TypeError("rail_observation must be RailPlacementStopObservation")
            rail_observation.require_stop(
                phase=self.phase, state=self.state, evidence=evidence
            )
        self.rail_observation = rail_observation
        if feedback_context is not None:
            from clay.layout.place.resource_feedback import (
                RoutePlanningFeedbackContext,
            )

            if not isinstance(feedback_context, RoutePlanningFeedbackContext):
                raise TypeError(
                    "feedback_context must be RoutePlanningFeedbackContext"
                )
            if not (
                self.phase == "RESOURCE_SELECTION"
                and self.state == "INFEASIBLE"
                and self.status is VerificationStatus.FAIL
            ):
                raise ValueError(
                    "feedback context is allowed only for exact RESOURCE_SELECTION "
                    "INFEASIBLE stops"
                )
            if self.evidence != feedback_context.conflict:
                raise ValueError(
                    "feedback context conflict differs from stop evidence"
                )
        self.feedback_context = feedback_context
        digest = getattr(evidence, "sha256", None)
        self.evidence_hash = str(digest()) if callable(digest) else hashlib.sha256(
            repr(evidence).encode("utf-8")
        ).hexdigest()
        raw_refs = getattr(evidence, "refs", ())
        self.evidence_refs = tuple(
            str(ref)[: self._MAX_DIAGNOSTIC_REF_CHARS]
            for ref in (
                raw_refs[: self._MAX_DIAGNOSTIC_REFS]
                if type(raw_refs) in (tuple, list)
                else ()
            )
        )
        refs_text = (
            "" if not self.evidence_refs else f", refs={self.evidence_refs!r}"
        )
        super().__init__(
            f"route planning stopped in {self.phase}: {self.state} "
            f"({self.status.value}, evidence={self.evidence_hash}{refs_text})"
        )

    def flow_error_data(self) -> Dict[str, Any]:
        """Return bounded, JSON-safe evidence for the generic flow ledger."""
        out: Dict[str, Any] = {
            "schema": "clay.route-planning-stop.v1",
            "phase": self.phase,
            "state": self.state,
            "status": self.status.value,
            "evidence_hash": self.evidence_hash,
            "evidence_refs": list(self.evidence_refs),
        }
        canonical = getattr(self.evidence, "canonical_json", None)
        if callable(canonical):
            try:
                out["route_stop"] = json.loads(canonical())
            except (TypeError, ValueError):
                out["route_stop"] = str(canonical())
        if self.rail_observation is not None:
            out["rail_placement_observation"] = self.rail_observation.payload()
        return out


def _metal_index_of(layer: str, tech: LayoutTech) -> Optional[int]:
    """Resolve a declared trunk layer token ('metal2' / 'M2' / '2') to a 1-based
    PDK metal index, or None. Role-name only -- no reference coordinate."""
    s = str(layer).strip().lower()
    for prefix in ("metal", "met", "m"):
        if s.startswith(prefix):
            tail = s[len(prefix):]
            if tail.isdigit():
                return int(tail)
    if s.isdigit():
        return int(s)
    return None
