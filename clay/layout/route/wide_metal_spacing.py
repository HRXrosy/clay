"""Canonical merged-geometry wide-metal spacing and justified owned repairs.

Scalar width tables do not classify attached narrow features. This module only
evaluates bands carrying an explicit opening/halo contract from the PDK. Search
may index lossless rectangle decompositions of these regions; final checks use
the full merged regions and exposed edges, never a component bounding box.
"""
from dataclasses import dataclass
from typing import Any, Tuple

import klayout.db as kdb


@dataclass(frozen=True)
class WideSpacingCheck:
    kind: str
    spacing_nm: int
    errors: Any


def geometry_spacing_entries(tech, layer_name):
    query = getattr(tech, "width_spacing_entries", None)
    if not callable(query):
        return ()
    return tuple(band for band in query(layer_name) if band.opening_nm is not None)


def wide_metal_regions(region, tech, layer_name):
    """Return (band, wide core, attached-metal window, attached metal) tuples."""
    from .emit_gds import pdk_manufacturing_grid_nm

    entries = geometry_spacing_entries(tech, layer_name)
    if not entries:
        return ()
    grid = pdk_manufacturing_grid_nm(tech)
    if type(grid) is not int or grid <= 0:
        raise ValueError("wide-metal geometry lacks a positive manufacturing grid")
    drawn = region.merged()
    result = []
    for band in entries:
        opening, space = band.opening_nm, band.spacing.value
        if (type(opening) is not int or opening <= 0 or opening % grid
                or type(space) is not int or space <= 0 or space % grid
                or band.run_length_ge != 0):
            raise ValueError("unsupported canonical wide-metal geometry band")
        huge = drawn.sized(-opening).sized(opening)
        core = (huge.snapped(grid, grid) & drawn).merged()
        window, attached = kdb.Region(), kdb.Region()
        if band.attached_halo_nm is not None:
            halo = band.attached_halo_nm
            if type(halo) is not int or halo <= 0 or halo % grid:
                raise ValueError("wide-metal attached halo is off the manufacturing grid")
            window = (huge.sized(halo) - huge).merged()
            attached = (window & drawn).merged()
        result.append((band, core, window, attached))
    return tuple(result)


def wide_spacing_checks(region, tech, layer_name):
    """Exact exposed-edge and attached-halo obligations on literal metal."""
    drawn = region.merged()
    result = []
    for band, core, _window, attached in wide_metal_regions(drawn, tech, layer_name):
        space = int(band.spacing.value)
        # Literal translation of the pinned KLayout foundry rule, including its
        # treatment of narrow attachments on the same merged component.
        narrow_edges = drawn.edges() - core
        wide_edges = core.edges().outside_part(drawn)
        errors = wide_edges.separation_check(narrow_edges, space) + wide_edges.space_check(space)
        # Magic widespacing extends along contiguous drawn wide edges up to
        # `space` past the shrink/grow core into flush corner steps.
        if not wide_edges.is_empty():
            flush_candidates = (
                (drawn.edges().with_length(band.wide_ge, None, False).interacting(wide_edges) - wide_edges)
                .inside_part(core.sized(space))
                .with_length(1, space + 1, False)
            )
            if not flush_candidates.is_empty():
                for poly in drawn.each():
                    poly_reg = kdb.Region(poly)
                    poly_flush = flush_candidates & poly_reg.edges()
                    if not poly_flush.is_empty():
                        other_edges = (drawn - poly_reg).edges()
                        errors = errors + poly_flush.separation_check(
                            other_edges, space, False, kdb.Edges.Euclidian, 180
                        )
        result.append(WideSpacingCheck("wide_edges", space, errors))
        if band.attached_halo_nm is not None:
            # The separate Magic cifspacing rule checks actual metal within the
            # wide-core halo. A whole-component space check misses these notches.
            result.append(WideSpacingCheck("attached_halo", space, attached.space_check(space)))
    return tuple(result)


def exact_region_rectangles(region) -> Tuple[Tuple[int, int, int, int], ...]:
    """Lossless rectilinear decomposition suitable for an integer spatial index."""
    rectangles = []
    cover = kdb.Region()
    for shape in region.decompose_trapezoids():
        polygon = kdb.Polygon(shape.polygon)
        box = polygon.bbox()
        rectangle = kdb.Region(box)
        if not (kdb.Region(polygon) ^ rectangle).is_empty():
            raise ValueError("wide-metal routing index requires rectilinear drawn geometry")
        rectangles.append((box.left, box.bottom, box.right, box.top))
        cover += rectangle
    if not (cover.merged() ^ region.merged()).is_empty():
        raise ValueError("wide-metal rectangle index changed literal geometry")
    return tuple(sorted(rectangles))


def wide_spacing_additions(own, foreign, tech, layer_name):
    """Only real wide-rule gaps justify closing; foreign clearance stays exact.

    This shares the existing upper-metal closing kernel. Every addition is
    returned to the caller for ownership, grid, containment and manifest checks.
    Legal equality and ordinary narrow gaps outside a wide halo create no fill.
    """
    drawn = own.merged()
    additions = kdb.Region()
    checks = wide_spacing_checks(drawn, tech, layer_name)
    for check in checks:
        if check.errors.is_empty():
            continue
        space = check.spacing_nm
        half = (space + 1) // 2
        candidate = ((drawn.sized(half).sized(-half) - drawn)
                     - foreign.sized(space)).merged()
        # One connected closing component may extend far beyond the wide halo
        # along an otherwise legal narrow slot. Only the actual error gap may
        # justify wide-only metal, not that entire closing component.
        additions += candidate & check.errors.polygons(0)
    additions.merge()
    if additions.is_empty():
        return additions

    from .emit_gds import pdk_manufacturing_grid_nm
    from .metal_junction_legalization import _grid_aligned_cover, legalize_island_junctions

    grid = pdk_manufacturing_grid_nm(tech)
    width = int(tech.min_width(layer_name))
    base_space = int(tech.min_space(layer_name))
    max_space = max(check.spacing_nm for check in checks)
    blocked = foreign.sized(max_space)
    additions = (_grid_aligned_cover(additions, grid) - blocked - drawn).merged()
    after = (drawn + additions).merged()
    old_width = drawn.width_check(width).polygons(0)
    new_width = (after.width_check(width).polygons(0) - old_width).merged()
    old_space = drawn.space_check(base_space).polygons(0)
    new_space = (after.space_check(base_space).polygons(0) - old_space).merged()
    if not new_width.is_empty() or not new_space.is_empty():
        # A projected wide gap can leave a sub-width neck or a new ordinary
        # gap to nearby own metal. Transport both caused obligations into the
        # existing junction kernel. Freeze support to their initial markers
        # plus one minimum-width distance; unrelated slots remain forbidden.
        allowed = ((new_width + new_space).sized(width) + additions).merged()
        work_box = after.bbox().enlarged(width)
        forbidden = (blocked + (kdb.Region(work_box) - allowed)).merged()
        after, _ = legalize_island_junctions(
            after, forbidden, min_width_nm=width, min_space_nm=base_space,
            manufacturing_grid_nm=grid,
        )
        additions = (after - drawn).merged()
    # Clipping against foreign material must not trade the obligation for a new
    # narrow neck, ordinary gap, or another wide-rule violation.
    if not (after.width_check(width).polygons(0) - old_width).is_empty():
        return kdb.Region()
    if not (after.space_check(base_space).polygons(0)
            - old_space).is_empty():
        return kdb.Region()
    after_checks = wide_spacing_checks(after, tech, layer_name)
    for before_check, after_check in zip(checks, after_checks):
        if not (after_check.errors.polygons(0) - before_check.errors.polygons(0)).is_empty():
            return kdb.Region()
    return additions
