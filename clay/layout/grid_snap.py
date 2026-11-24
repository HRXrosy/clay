"""Manufacturing-grid snapper for ParameterizedLibrary geometry.

Rounds every polygon vertex's (x, y) to a multiple of a manufacturing grid
(typically the PDK minimum grid, e.g. 5 nm for sky130 / gf180mcu in DB units).

The snapper is a pure functional transform: the input ``ParameterizedLibrary``
is not mutated; a deep-copied library is returned alongside a ``SnapStats``
record that callers can assert against (e.g. for DRC pre-checks).

Mode semantics (operates on signed integer DB-unit coordinates):

* ``"round"``  — banker-free half-up rounding so ``2`` snaps to ``0`` on a
  ``5`` nm grid but ``3`` snaps to ``5`` and ``-3`` snaps to ``-5``. This is
  the default and matches what foundries mean by "snap to grid".
* ``"floor"``  — truncate toward negative infinity. ``3`` and ``-2`` both
  snap down (to ``0`` and ``-5`` respectively).
* ``"ceil"``   — round toward positive infinity. ``-3`` snaps to ``0`` and
  ``2`` snaps to ``5``.

``grid_nm`` must be a positive ``int``. Passing ``grid_nm=1`` is a no-op for
already-integer coordinates (the Point/Polygon API stores ``int`` vertices),
which is the practical "do nothing" knob for callers that always want to
route through this helper.

R143 (BL-143A) — gf180mcu DRC-safe snap for grid alignment
----------------------------------------------------------
R142's combined gf180mcu flow nudged M1 landing pads / routes by non-grid
displacements (the M1.2a spacing fix and M1.3 min-area pad growth), leaving
Metal1 vertices off the manufacturing grid; the real gf180mcu KLayout deck then
reports them as ``metal1_OFFGRID`` (6 instances at R142). R143 adds a DRC-safe
snapper (:func:`snap_to_grid` / :func:`snap_library`) that drives every vertex
on-grid before GDS emit.

Snap direction is chosen empirically against the real deck. A pure OUTWARD snap
(floor below centroid / ceil above, :func:`snap_polygon_outward`) guarantees each
polygon's own width / area never shrink, but where R142 placed two M1 pads at
EXACTLY the 0.23 um min spacing on an off-grid coordinate it pushes their facing
edges apart in opposite directions, closing the 230 nm gap to 225 nm and adding a
NEW M1.2a violation (verified: outward => metal1_OFFGRID 6->0 but M1.2a 0->1). The
default NEAREST snap moves both edges of a uniformly-nudged gap the same way, so
the 230 nm gap is preserved and NO new width / area / spacing violation appears
(verified: nearest => metal1_OFFGRID 6->0, no new rule). So :func:`snap_to_grid` /
:func:`snap_library` use nearest by default and expose ``outward=True`` for the
opt-in per-polygon-growth path. The legacy ``snap_library_to_grid`` (round/floor/
ceil modes) is preserved unchanged.

Grid value + source.
    The gf180mcu manufacturing grid is **0.005 um = 5 nm = 5 DBU**. Source
    (verbatim, real foundry deck): the OFFGRID section of the gf180mcu KLayout
    DRC runset
    ``AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/klayout/drc/gf180mcu.drc``
    checks every drawn layer with ``ongrid(0.005)``, e.g. line 5071::

        metal1.ongrid(0.005).output("metal1_OFFGRID", "OFFGRID : OFFGRID vertex on metal1")

    Every OFFGRID rule in that deck (comp / poly2 / nplus / pplus / nwell /
    metal1..5 / metaltop / via1..5 / pad / contact / ...) uses the same
    ``ongrid(0.005)``, so 5 nm is the single gf180mcu manufacturing grid, not an
    M1-only value.

Database units.
    The Clay backend (``clay/layout/backend.py``) emits GDS with
    ``unit_user = 0.001`` (1 user unit = 0.001 um) and ``unit_db_meters = 1e-9``,
    so 1 DBU = 1 nm and every ``Point.x`` / ``Point.y`` integer is a nm count.
    0.005 um therefore = 5 DBU == :data:`GF180_GRID_NM`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .backend import (
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)


_VALID_MODES = ("round", "floor", "ceil")

# gf180mcu manufacturing grid: 0.005 um == 5 nm == 5 DBU (DBU = 1 nm here).
# Source: gf180mcu.drc OFFGRID section, every layer checked with ongrid(0.005)
# (metal1 at line 5071). See module docstring.
GF180_GRID_UM: float = 0.005
GF180_GRID_NM: int = 5


@dataclass
class SnapStats:
    """Counters describing a single ``snap_library_to_grid`` invocation.

    Attributes:
        polygons_processed: Total number of polygons walked (one per
            ``LayerPolygon`` across all ``NamedCell`` entries in the library).
        vertices_snapped: Count of vertices whose (x, y) actually moved —
            i.e. either the x- or y-coordinate (or both) changed value
            after rounding. A vertex already on grid contributes 0.
        max_displacement_nm: Largest single-vertex displacement in nm,
            measured as ``max(|dx|, |dy|)`` (Chebyshev / L∞ distance).
            Zero if no vertex moved.
    """

    polygons_processed: int
    vertices_snapped: int
    max_displacement_nm: int


def _snap_coord(value: int, grid_nm: int, mode: str) -> int:
    """Snap a single integer coordinate to a multiple of ``grid_nm``.

    Symmetric for negatives: rounds half-away-from-zero for ``"round"`` mode
    so e.g. ``-3`` -> ``-5`` and ``3`` -> ``5`` on a 5 nm grid.
    """
    if mode == "round":
        # Half-up rounding that is symmetric across zero (avoids Python 3's
        # banker's rounding which would snap 2.5 -> 2 instead of 3).
        if value >= 0:
            q, r = divmod(value, grid_nm)
            return (q + (1 if r * 2 >= grid_nm else 0)) * grid_nm
        else:
            q, r = divmod(-value, grid_nm)
            return -((q + (1 if r * 2 >= grid_nm else 0)) * grid_nm)
    if mode == "floor":
        # Python's // already floors toward negative infinity for ints.
        return (value // grid_nm) * grid_nm
    if mode == "ceil":
        return -((-value // grid_nm) * grid_nm)
    raise ValueError(
        f"unknown snap mode {mode!r}; expected one of {_VALID_MODES}"
    )


def snap_point_to_grid(p: Point, grid_nm: int, mode: str = "round") -> Point:
    """Return a new ``Point`` with both coordinates snapped to ``grid_nm``.

    The original point is not mutated. ``grid_nm`` must be a positive int.
    Raises ``ValueError`` if ``mode`` is not one of ``"round"``, ``"floor"``,
    ``"ceil"`` or if ``grid_nm`` is non-positive.
    """
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"unknown snap mode {mode!r}; expected one of {_VALID_MODES}"
        )
    return Point(
        x=_snap_coord(p.x, grid_nm, mode),
        y=_snap_coord(p.y, grid_nm, mode),
    )


def check_polygon_on_grid(polygon: Polygon, grid_nm: int) -> bool:
    """Return True iff every vertex of ``polygon`` has both x and y divisible
    by ``grid_nm``.

    An empty polygon (zero vertices) is considered on-grid (vacuously true).
    """
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    for v in polygon.vertices:
        if v.x % grid_nm != 0 or v.y % grid_nm != 0:
            return False
    return True


def snap_library_to_grid(
    lib: ParameterizedLibrary,
    grid_nm: int = 5,
    *,
    mode: str = "round",
) -> Tuple[ParameterizedLibrary, SnapStats]:
    """Snap every polygon vertex in ``lib`` to the manufacturing ``grid_nm``.

    Returns a tuple ``(new_lib, stats)``. ``new_lib`` is a freshly constructed
    ``ParameterizedLibrary`` (no aliasing of the input's cells, polygons, or
    points) so callers can mutate or persist either side independently.

    ``stats.max_displacement_nm`` uses Chebyshev (L∞) distance so a single
    integer compares cleanly against PDK-published "max allowed grid drift"
    constants.
    """
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"unknown snap mode {mode!r}; expected one of {_VALID_MODES}"
        )

    polygons_processed = 0
    vertices_snapped = 0
    max_displacement = 0

    new_cells = []
    for cell in lib.cells:
        new_polys = []
        for lp in cell.polygons:
            polygons_processed += 1
            new_verts = []
            for v in lp.polygon.vertices:
                nx = _snap_coord(v.x, grid_nm, mode)
                ny = _snap_coord(v.y, grid_nm, mode)
                if nx != v.x or ny != v.y:
                    vertices_snapped += 1
                    disp = max(abs(nx - v.x), abs(ny - v.y))
                    if disp > max_displacement:
                        max_displacement = disp
                new_verts.append(Point(x=nx, y=ny))
            new_polys.append(
                LayerPolygon(
                    layer=lp.layer,
                    datatype=lp.datatype,
                    polygon=Polygon(vertices=new_verts),
                )
            )
        new_cells.append(NamedCell(name=cell.name, polygons=new_polys))

    new_lib = ParameterizedLibrary(
        name=lib.name,
        unit_user=lib.unit_user,
        unit_db_meters=lib.unit_db_meters,
        cells=new_cells,
    )
    stats = SnapStats(
        polygons_processed=polygons_processed,
        vertices_snapped=vertices_snapped,
        max_displacement_nm=max_displacement,
    )
    return new_lib, stats


# --- R143 (BL-143A): gf180mcu DRC-safe grid snap -----------------------------
#
# The functions below are the BL-143A surface required to clear R142's six
# metal1_OFFGRID violations against the REAL gf180mcu.drc deck.
#
# Snap direction — empirically chosen against the real deck.
#   The obvious "snap outward" rule (floor below centroid, ceil at/above) grows
#   every polygon so its OWN width / area never shrink. But growing two M1 pads
#   that the R142 M1.2a fix had placed at EXACTLY the 0.23 um minimum spacing on
#   an off-grid coordinate pushes their facing edges in OPPOSITE directions, so
#   the gap closes from 230 nm to 225 nm and a NEW M1.2a min-spacing violation
#   appears. Verified on the real R142 5T-OTA flow: snap_polygon_outward clears
#   metal1_OFFGRID 6 -> 0 but adds M1.2a 0 -> 1.
#
#   NEAREST rounding instead moves both facing edges of a uniformly-nudged gap in
#   the SAME direction, so the 230 nm gap is preserved (230 -> 230) while the pad
#   still snaps on-grid and grows where it needs to. Verified on the same real
#   flow: nearest clears metal1_OFFGRID 6 -> 0 and introduces NO new width / area
#   / spacing violation (the only residual is DF.4c, an out-of-lane Nwell rule).
#   Its max displacement is also bounded to grid/2 (<=2 nm) vs outward's <4 nm.
#
# Therefore snap_to_grid / snap_library use NEAREST by default and expose an
# explicit ``outward=True`` for callers who specifically need per-polygon growth
# (e.g. a min-area-only post-pass with no min-spacing-critical neighbours).
# snap_polygon_outward remains available for that opt-in path.


def is_on_grid(value_nm: int, grid_nm: int = GF180_GRID_NM) -> bool:
    """True iff an integer-nm coordinate is exactly on the ``grid_nm`` grid."""
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    return value_nm % grid_nm == 0


def point_on_grid(p: Point, grid_nm: int = GF180_GRID_NM) -> bool:
    """True iff both coordinates of ``p`` are on the grid."""
    return is_on_grid(p.x, grid_nm) and is_on_grid(p.y, grid_nm)


def polygon_all_on_grid(polygon: Polygon, grid_nm: int = GF180_GRID_NM) -> bool:
    """True iff every vertex of ``polygon`` is on the grid (empty => True)."""
    return all(point_on_grid(v, grid_nm) for v in polygon.vertices)


def polygons_all_on_grid(
    polygons: Iterable[LayerPolygon], grid_nm: int = GF180_GRID_NM
) -> bool:
    """True iff every vertex of every :class:`LayerPolygon` is on the grid."""
    return all(polygon_all_on_grid(lp.polygon, grid_nm) for lp in polygons)


def offgrid_vertex_count(
    polygons: Iterable[LayerPolygon], grid_nm: int = GF180_GRID_NM
) -> int:
    """Count vertices (across all polygons) whose x or y is off the grid.

    A vertex with both coordinates off-grid counts once — mirroring what the
    real deck's ``ongrid(0.005)`` flags per vertex (used for the before/after
    metal1_OFFGRID tally).
    """
    n = 0
    for lp in polygons:
        for v in lp.polygon.vertices:
            if not point_on_grid(v, grid_nm):
                n += 1
    return n


def _floor_to_grid(value_nm: int, grid_nm: int) -> int:
    """Round ``value_nm`` DOWN to the nearest multiple of ``grid_nm``."""
    # Python // floors toward -inf, which is grid-floor for either sign.
    return (value_nm // grid_nm) * grid_nm


def _ceil_to_grid(value_nm: int, grid_nm: int) -> int:
    """Round ``value_nm`` UP to the nearest multiple of ``grid_nm``."""
    return -((-value_nm) // grid_nm) * grid_nm


def _nearest_to_grid(value_nm: int, grid_nm: int) -> int:
    """Round ``value_nm`` to the NEAREST grid multiple (ties go up)."""
    floor = _floor_to_grid(value_nm, grid_nm)
    rem = value_nm - floor
    return floor if rem * 2 < grid_nm else floor + grid_nm


def snap_polygon_outward(
    polygon: Polygon, grid_nm: int = GF180_GRID_NM
) -> Polygon:
    """Snap every vertex of ``polygon`` OUTWARD to the grid (DRC-safe).

    Each coordinate below the polygon's bbox centre is floored to the grid;
    each coordinate at/above the centre is ceiled. This expands the polygon (or
    leaves it unchanged) so width / area / spacing margins never shrink. An axis
    along which the bbox is degenerate (all-equal x or all-equal y) is snapped to
    the nearest grid line on that axis (no inward bias is possible there), so the
    shape stays well-formed. Returns a NEW Polygon; ``polygon`` is not mutated.
    """
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    verts = polygon.vertices
    if not verts:
        return Polygon(vertices=[])

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Doubled bbox midpoint keeps the inside/outside test exact in ints.
    cx2 = x_min + x_max
    cy2 = y_min + y_max
    x_degenerate = x_min == x_max
    y_degenerate = y_min == y_max

    new_verts: List[Point] = []
    for v in verts:
        if x_degenerate:
            nx = _nearest_to_grid(v.x, grid_nm)
        elif 2 * v.x < cx2:
            nx = _floor_to_grid(v.x, grid_nm)  # left of centre -> outward = down
        else:
            nx = _ceil_to_grid(v.x, grid_nm)   # right of/at centre -> outward = up
        if y_degenerate:
            ny = _nearest_to_grid(v.y, grid_nm)
        elif 2 * v.y < cy2:
            ny = _floor_to_grid(v.y, grid_nm)
        else:
            ny = _ceil_to_grid(v.y, grid_nm)
        new_verts.append(Point(x=nx, y=ny))

    return Polygon(vertices=new_verts)


def snap_polygon_nearest(
    polygon: Polygon, grid_nm: int = GF180_GRID_NM
) -> Polygon:
    """Snap every vertex of ``polygon`` to the NEAREST grid line (ties up).

    Max per-coordinate displacement is grid/2. When a polygon was produced by a
    uniform off-grid nudge (the R142 case), nearest moves both edges of a
    min-spacing gap the same way, preserving the gap. Returns a NEW Polygon.
    """
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    return Polygon(
        vertices=[
            Point(_nearest_to_grid(v.x, grid_nm), _nearest_to_grid(v.y, grid_nm))
            for v in polygon.vertices
        ]
    )


def snap_layer_polygon(
    lp: LayerPolygon, grid_nm: int = GF180_GRID_NM, *, outward: bool = False
) -> LayerPolygon:
    """Snap one :class:`LayerPolygon` to the grid (layer/datatype kept).

    ``outward=False`` (default) snaps to the nearest grid line (the DRC-safe
    default — see the section comment above); ``outward=True`` grows the polygon
    via :func:`snap_polygon_outward`.
    """
    snapper = snap_polygon_outward if outward else snap_polygon_nearest
    return LayerPolygon(
        layer=lp.layer,
        datatype=lp.datatype,
        polygon=snapper(lp.polygon, grid_nm),
    )


def snap_to_grid(
    polygons: Iterable[LayerPolygon],
    grid_nm: int = GF180_GRID_NM,
    *,
    outward: bool = False,
) -> List[LayerPolygon]:
    """Snap every vertex of every polygon to the manufacturing grid.

    BL-143A primary entry point. Returns a NEW list of snapped
    :class:`LayerPolygon` (inputs are not mutated). After the call
    ``polygons_all_on_grid(result, grid_nm)`` is True.

    Parameters
    ----------
    polygons:
        Iterable of :class:`~clay.layout.backend.LayerPolygon`.
    grid_nm:
        Grid step in nm/DBU (default :data:`GF180_GRID_NM` = 5 nm = 0.005 um).
    outward:
        When ``False`` (default) snap to the nearest grid line — the DRC-safe
        choice verified against the real gf180mcu deck (clears metal1_OFFGRID
        without re-introducing a min-spacing violation, because both edges of a
        uniformly-nudged gap move the same way). When ``True``, grow every
        polygon outward (per-polygon width/area never shrink, but a
        min-spacing-critical neighbour gap can close — opt in only when that is
        not a concern).
    """
    return [snap_layer_polygon(lp, grid_nm, outward=outward) for lp in polygons]


def snap_library(
    library: ParameterizedLibrary,
    grid_nm: int = GF180_GRID_NM,
    *,
    outward: bool = False,
) -> ParameterizedLibrary:
    """Snap every vertex of every cell in ``library`` to the grid.

    Mutates ``library`` in place (each cell's polygon list is replaced with its
    snapped polygons) and returns it for chaining. Apply this to the combined
    flow's library just before GDS emit so the real gf180mcu deck sees an
    on-grid layout and reports zero ``*_OFFGRID`` violations. ``outward`` is
    forwarded to :func:`snap_to_grid` (default nearest — DRC-safe; see above).

    Distinct from :func:`snap_library_to_grid`, which returns a fresh copy +
    stats and uses configurable round/floor/ceil modes.
    """
    if not isinstance(grid_nm, int) or grid_nm <= 0:
        raise ValueError(f"grid_nm must be a positive int, got {grid_nm!r}")
    for cell in library.cells:
        snapped = snap_to_grid(cell.polygons, grid_nm, outward=outward)
        cell.polygons.clear()
        cell.polygons.extend(snapped)
    return library


__all__ = [
    "GF180_GRID_NM",
    "GF180_GRID_UM",
    "SnapStats",
    "check_polygon_on_grid",
    "is_on_grid",
    "offgrid_vertex_count",
    "point_on_grid",
    "polygon_all_on_grid",
    "polygons_all_on_grid",
    "snap_layer_polygon",
    "snap_library",
    "snap_library_to_grid",
    "snap_point_to_grid",
    "snap_polygon_nearest",
    "snap_polygon_outward",
    "snap_to_grid",
]
