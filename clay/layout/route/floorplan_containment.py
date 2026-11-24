"""Hard containment contract for route-owned emitted geometry."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Tuple


Rect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class RoutingEnvelope:
    """Exact rectangular floorplan boundary in integer nanometres."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if any(type(value) is not int for value in values):
            raise TypeError("routing envelope coordinates must be exact integer nm")
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("routing envelope must be a non-empty rectangle")

    @property
    def bbox_nm(self) -> Rect:
        return (self.x0, self.y0, self.x1, self.y1)

    def contains_bbox(self, bbox_nm: Rect) -> bool:
        x0, y0, x1, y1 = (int(value) for value in bbox_nm)
        return (
            self.x0 <= x0 <= x1 <= self.x1
            and self.y0 <= y0 <= y1 <= self.y1
        )

    @classmethod
    def coerce(cls, value: "RoutingEnvelope | Iterable[int]") -> "RoutingEnvelope":
        if isinstance(value, cls):
            return value
        values = tuple(value)
        if len(values) != 4:
            raise ValueError("routing envelope must contain exactly four coordinates")
        return cls(*values)


def _exact_rect(value: Iterable[int], *, name: str) -> Rect:
    values = tuple(value)
    if (
        len(values) != 4
        or any(type(item) is not int for item in values)
        or values[0] >= values[2]
        or values[1] >= values[3]
    ):
        raise ValueError(f"{name} must be a non-empty exact integer-nm rectangle")
    return values


@dataclass(frozen=True)
class RoutingEnvelopeCertificate:
    """Bind the routable rectangle to one final placement and internal rails.

    A compact placement certificate describes the device/passive/guard-ring core.
    Declared edge rails occupy bands inside that core. Their exact materialized
    rectangles are recorded, but may never enlarge the human-visible floorplan in
    which all route-owned geometry must remain.
    """

    placement_state_digest: str
    placement_core_bbox_nm: Rect
    rail_bboxes_nm: Tuple[Rect, ...]
    envelope_nm: Rect

    def __post_init__(self) -> None:
        digest = self.placement_state_digest
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.casefold())
        ):
            raise ValueError("routing envelope requires a SHA-256 placement-state digest")
        core = _exact_rect(
            self.placement_core_bbox_nm, name="placement core bbox"
        )
        rails = tuple(
            _exact_rect(value, name="materialized rail bbox")
            for value in self.rail_bboxes_nm
        )
        envelope = _exact_rect(self.envelope_nm, name="routing envelope")
        if any(
            not (
                core[0] <= rail[0] < rail[2] <= core[2]
                and core[1] <= rail[1] < rail[3] <= core[3]
            )
            for rail in rails
        ):
            raise ValueError("materialized edge rail lies outside the placement floorplan")
        if envelope != core:
            raise ValueError(
                "routing envelope must equal the exact placement floorplan"
            )

    @property
    def sha256(self) -> str:
        payload = {
            "envelope_nm": list(self.envelope_nm),
            "placement_core_bbox_nm": list(self.placement_core_bbox_nm),
            "placement_state_digest": self.placement_state_digest,
            "rail_bboxes_nm": [list(rect) for rect in self.rail_bboxes_nm],
            "schema": "clay.routing-envelope-certificate.v1",
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def certify_routing_envelope(
    final_placement_certificate: Any,
    *,
    rail_bboxes_nm: Iterable[Iterable[int]],
) -> RoutingEnvelopeCertificate:
    """Derive the only legal route domain from the certified placement core."""

    if final_placement_certificate is None:
        raise ValueError("routing envelope requires a final placement certificate")
    placement_digest = getattr(
        final_placement_certificate, "placement_state_digest", ""
    )
    if callable(placement_digest):
        placement_digest = placement_digest()
    core = _exact_rect(
        getattr(final_placement_certificate, "bbox_nm", ()),
        name="placement core bbox",
    )
    rails = tuple(
        _exact_rect(value, name="materialized rail bbox")
        for value in rail_bboxes_nm
    )
    return RoutingEnvelopeCertificate(
        placement_state_digest=str(placement_digest),
        placement_core_bbox_nm=core,
        rail_bboxes_nm=rails,
        envelope_nm=core,
    )


@dataclass(frozen=True)
class RouteContainmentViolation:
    net: str
    layer: Tuple[int, int]
    bbox_nm: Rect


@dataclass(frozen=True)
class RouteContainmentReport:
    envelope_nm: Rect
    violations: Tuple[RouteContainmentViolation, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class GdsContainmentViolation:
    kind: str
    layer: Tuple[int, int]
    bbox_nm: Rect
    shape_count: int
    area_nm2: int


@dataclass(frozen=True)
class GdsRouteContainmentReport:
    envelope_nm: Rect
    violations: Tuple[GdsContainmentViolation, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.violations


class RouteContainmentError(RuntimeError):
    """The actual route geometry escaped its declared floorplan envelope."""

    def __init__(self, report: RouteContainmentReport) -> None:
        self.report = report
        detail = ", ".join(
            f"{item.net}@{item.layer}:{item.bbox_nm}"
            for item in report.violations
        )
        super().__init__(
            "route-owned geometry leaves floorplan envelope "
            f"{report.envelope_nm}: {detail}"
        )


def _polygon_bbox_nm(layer_polygon: Any) -> Rect | None:
    vertices = tuple(
        getattr(getattr(layer_polygon, "polygon", None), "vertices", ()) or ()
    )
    if not vertices:
        return None
    xs = [int(point.x) for point in vertices]
    ys = [int(point.y) for point in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def check_routed_containment(
    routed: Any,
    envelope: RoutingEnvelope | Iterable[int],
) -> RouteContainmentReport:
    """Read every route polygon consumed by the normal GDS emit path."""

    resolved = RoutingEnvelope.coerce(envelope)
    violations = []
    for net, net_route in sorted(
        (getattr(routed, "nets", {}) or {}).items(), key=lambda item: str(item[0])
    ):
        for layer_polygon in getattr(net_route, "wires", ()) or ():
            bbox = _polygon_bbox_nm(layer_polygon)
            if bbox is None or resolved.contains_bbox(bbox):
                continue
            violations.append(
                RouteContainmentViolation(
                    net=str(net),
                    layer=(
                        int(getattr(layer_polygon, "layer")),
                        int(getattr(layer_polygon, "datatype")),
                    ),
                    bbox_nm=bbox,
                )
            )
    violations.sort(key=lambda item: (item.net, item.layer, item.bbox_nm))
    return RouteContainmentReport(resolved.bbox_nm, tuple(violations))


def require_routed_containment(
    routed: Any,
    envelope: RoutingEnvelope | Iterable[int],
) -> RouteContainmentReport:
    report = check_routed_containment(routed, envelope)
    if not report.clean:
        raise RouteContainmentError(report)
    return report


def check_gds_routing_containment(
    gds_path: Any,
    tech: Any,
    envelope: RoutingEnvelope | Iterable[int],
) -> GdsRouteContainmentReport:
    """Flatten a GDS and measure every PDK routing-metal/via layer against it."""

    import klayout.db as db

    resolved = RoutingEnvelope.coerce(envelope)
    layout = db.Layout()
    layout.read(str(gds_path))
    top = layout.top_cell()
    if top is None:
        raise ValueError(f"GDS has no top cell: {gds_path}")
    nm_per_dbu = float(layout.dbu) * 1000.0
    if not math.isfinite(nm_per_dbu) or nm_per_dbu <= 0.0:
        raise ValueError(f"GDS has invalid database unit: {layout.dbu!r}")

    # The GDS grid may be coarser than the integer-nm floorplan contract.  Project
    # the legal rectangle INWARD: nearest rounding would enlarge it and create a
    # fail-open strip of up to half a DBU on every edge.
    x0_dbu = int(math.ceil(resolved.x0 / nm_per_dbu))
    y0_dbu = int(math.ceil(resolved.y0 / nm_per_dbu))
    x1_dbu = int(math.floor(resolved.x1 / nm_per_dbu))
    y1_dbu = int(math.floor(resolved.y1 / nm_per_dbu))
    legal = (
        db.Region(db.Box(x0_dbu, y0_dbu, x1_dbu, y1_dbu))
        if x0_dbu < x1_dbu and y0_dbu < y1_dbu
        else db.Region()
    )
    layer_kinds = []
    for number in range(1, 33):
        try:
            key = tuple(int(value) for value in tech.metal_key(number))
        except (KeyError, TypeError, ValueError):
            continue
        layer_kinds.append((f"metal{number}", key))
    for lower in range(1, 32):
        try:
            key = tuple(int(value) for value in tech.via_key(lower))
        except KeyError:
            continue
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"PDK returned an invalid via{lower} routing layer"
            ) from exc
        if len(key) != 2:
            raise ValueError(f"PDK via{lower} routing layer must be a GDS pair")
        layer_kinds.append((f"via{lower}", key))

    violations = []
    for kind, key in sorted(set(layer_kinds)):
        layer_index = layout.find_layer(*key)
        if layer_index is None:
            continue
        outside = db.Region(top.begin_shapes_rec(layer_index)) - legal
        if outside.is_empty():
            continue
        bbox = outside.bbox()
        bbox_nm = tuple(
            int(round(value * nm_per_dbu))
            for value in (bbox.left, bbox.bottom, bbox.right, bbox.top)
        )
        violations.append(
            GdsContainmentViolation(
                kind=kind,
                layer=key,
                bbox_nm=bbox_nm,
                shape_count=int(outside.count()),
                area_nm2=int(round(outside.area() * nm_per_dbu * nm_per_dbu)),
            )
        )
    violations.sort(key=lambda item: (item.kind, item.layer, item.bbox_nm))
    return GdsRouteContainmentReport(resolved.bbox_nm, tuple(violations))
