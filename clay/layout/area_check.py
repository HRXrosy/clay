"""Fail-closed emitted-GDS envelope-area measurement for ``@area`` intent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class EmittedGdsAreaReport:
    status: str
    bbox_nm: Optional[tuple[int, int, int, int]]
    area_um2: Optional[float]
    max_um2: Optional[float]
    reason: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def check_emitted_gds_area(
    gds_path: Path,
    *,
    top_cell: str | None,
    max_um2: float | None,
) -> EmittedGdsAreaReport:
    """Measure the recursive top-cell bbox and judge a declared maximum area.

    The bbox comes from the emitted artifact after all device, ring, rail, via,
    route, and exact-emission geometry has been assembled.  A declared limit is
    fail-closed: unreadable/missing geometry returns ``unmeasurable``, never pass.
    With no limit the same measurement is retained as ``advisory`` provenance.
    """
    path = Path(gds_path)
    limit = None if max_um2 is None else float(max_um2)
    if limit is not None and limit <= 0:
        return EmittedGdsAreaReport(
            status="unmeasurable",
            bbox_nm=None,
            area_um2=None,
            max_um2=limit,
            reason="declared maximum area must be positive",
        )
    if not path.is_file():
        return EmittedGdsAreaReport(
            status="unmeasurable",
            bbox_nm=None,
            area_um2=None,
            max_um2=limit,
            reason=f"emitted GDS does not exist: {path}",
        )
    try:
        import klayout.db as db

        layout = db.Layout()
        layout.read(str(path))
        cell = layout.cell(str(top_cell)) if top_cell else layout.top_cell()
        if cell is None:
            return EmittedGdsAreaReport(
                status="unmeasurable",
                bbox_nm=None,
                area_um2=None,
                max_um2=limit,
                reason=f"emitted GDS has no requested top cell {top_cell!r}",
            )
        bbox = cell.bbox()
        if bbox.empty():
            return EmittedGdsAreaReport(
                status="unmeasurable",
                bbox_nm=None,
                area_um2=None,
                max_um2=limit,
                reason="emitted GDS top cell has no measurable geometry",
            )
        nm_per_dbu = float(layout.dbu) * 1000.0
        bbox_nm = tuple(
            int(round(float(value) * nm_per_dbu))
            for value in (bbox.left, bbox.bottom, bbox.right, bbox.top)
        )
        width_nm = bbox_nm[2] - bbox_nm[0]
        height_nm = bbox_nm[3] - bbox_nm[1]
        if width_nm <= 0 or height_nm <= 0:
            raise ValueError("emitted GDS bbox is non-positive")
        area_um2 = (width_nm * height_nm) / 1_000_000.0
    except Exception as exc:  # noqa: BLE001 - a declared gate must return a verdict
        return EmittedGdsAreaReport(
            status="unmeasurable",
            bbox_nm=None,
            area_um2=None,
            max_um2=limit,
            reason=f"cannot measure emitted GDS area: {type(exc).__name__}: {exc}",
        )

    if limit is None:
        return EmittedGdsAreaReport(
            status="advisory",
            bbox_nm=bbox_nm,
            area_um2=area_um2,
            max_um2=None,
            reason="emitted GDS envelope area measured; no maximum was declared",
        )
    if area_um2 > limit:
        return EmittedGdsAreaReport(
            status="fail",
            bbox_nm=bbox_nm,
            area_um2=area_um2,
            max_um2=limit,
            reason=f"emitted GDS envelope area {area_um2:g} um2 exceeds {limit:g} um2",
        )
    return EmittedGdsAreaReport(
        status="pass",
        bbox_nm=bbox_nm,
        area_um2=area_um2,
        max_um2=limit,
        reason=f"emitted GDS envelope area {area_um2:g} um2 is within {limit:g} um2",
    )


__all__ = ["EmittedGdsAreaReport", "check_emitted_gds_area"]
