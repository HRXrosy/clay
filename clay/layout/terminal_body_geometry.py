"""Intrinsic terminal/body axes, independent of source or reference authority."""
from dataclasses import dataclass

import klayout.db as kdb


def _require(value, message):
    if not value:
        raise ValueError('terminal body geometry: ' + message)


def _box(region):
    box = region.bbox()
    return box.left, box.bottom, box.right, box.top


def opposing_terminal_axis(body, sides):
    """Return length axis, width and length from two opposing physical sides.

    A quarter turn exchanges x/y while preserving physical W/L. Neither source
    labels nor the relative magnitudes of the dimensions determine that axis.
    """
    _require(len(sides) == 2, 'requires two complete opposing terminal regions')
    sides = tuple(sorted(sides, key=_box))
    _require(body.size() == 1, 'channel/body is not one region')
    box = body.bbox()
    _require((body ^ kdb.Region(box)).is_empty(), 'channel/body is not a complete rectangle')
    horizontal = sides[0].bbox().right <= box.left and sides[1].bbox().left >= box.right
    vertical = sides[0].bbox().top <= box.bottom and sides[1].bbox().bottom >= box.top
    _require(horizontal != vertical, 'terminal regions do not define a unique channel/body axis')
    return ((1, 0), box.height(), box.width()) if horizontal else ((0, 1), box.width(), box.height())


def mos_channel_dimensions_nm(active, poly):
    """Measure ordered W/L from actual masks and the complete OD-minus-poly sides."""
    body = (active & poly).merged()
    sides = tuple(kdb.Region(shape) for shape in (active - poly).merged().each())
    _axis, width, length = opposing_terminal_axis(body, sides)
    return width, length


class SharedMosChannelUnsupported(ValueError):
    """The complete masks do not prove two unambiguous channel sides."""


@dataclass(frozen=True, slots=True)
class SharedMosChannelObservation:
    """Immutable geometric observation; it conveys no source or PDK authority.

    Sides are complete diffusion components in canonical geometric order, not
    source/drain labels. Each is an immutable PhysicalTerminalPolygonNM.
    """
    length_axis: tuple[int, int]
    width_nm: int
    length_nm: int
    diffusion_sides_nm: tuple


def _shared_mos_channel_geometry(active, poly, channel):
    """Measure one channel and its complete sides in possibly shared MOS masks.

    ``active`` and ``poly`` are the complete observed masks, and ``channel``
    must be one complete component of their intersection.  Neighboring gates
    remain in the poly mask.  This function uses complete active-minus-poly
    components and actual common edges; it never clips a component or chooses
    a side by distance.  Unsupported topology raises
    :class:`SharedMosChannelUnsupported` rather than guessing an axis.
    """
    def unsupported_unless(condition, message):
        if not condition:
            raise SharedMosChannelUnsupported('shared MOS channel geometry: ' + message)

    unsupported_unless(all(isinstance(region, kdb.Region) for region in (active, poly, channel)),
                       'active, poly and channel must be regions')
    active, poly, channel = active.merged(), poly.merged(), channel.merged()
    box = channel.bbox()
    unsupported_unless(channel.size() == 1 and box.width() > 0 and box.height() > 0
                       and (channel ^ kdb.Region(box)).is_empty(),
                       'channel must be one complete rectangle')
    crossings = (active & poly).merged()
    candidates = (kdb.Region(shape) for shape in crossings.each())
    unsupported_unless(sum((candidate ^ channel).is_empty() for candidate in candidates) == 1,
                       'channel is not one complete active/poly component')

    def edge(x1, y1, x2, y2):
        result = kdb.Edges()
        result.insert(kdb.Edge(x1, y1, x2, y2))
        return result

    faces = (
        edge(box.left, box.bottom, box.left, box.top),
        edge(box.right, box.bottom, box.right, box.top),
        edge(box.left, box.bottom, box.right, box.bottom),
        edge(box.left, box.top, box.right, box.top),
    )
    sides, side_faces = [], []
    for shape in (active - poly).merged().each():
        diffusion = kdb.Region(shape)
        if diffusion.interacting(channel).is_empty():
            continue
        common = (channel.edges() & diffusion.edges()).merged()
        matches = [index for index, face in enumerate(faces)
                   if (common - face).is_empty() and (face - common).is_empty()]
        unsupported_unless(len(matches) == 1,
                           'diffusion touches a corner, partial edge or multiple edges')
        sides.append(diffusion)
        side_faces.append(matches[0])
    unsupported_unless(len(sides) == 2 and set(side_faces) in ({0, 1}, {2, 3}),
                       'requires two distinct components on opposite complete edges')
    try:
        axis, width, length = opposing_terminal_axis(channel, tuple(sides))
    except ValueError as error:
        raise SharedMosChannelUnsupported('shared MOS channel geometry: '
                                          'complete diffusion components do not define an axis') from error
    return axis, width, length, tuple(sorted(sides, key=_box))


def observe_shared_mos_channel_nm(active, poly, channel):
    """Return immutable complete diffusion sides and the observed axis/W/L.

    Conversion to the existing terminal polygon type can reject shapes beyond
    its representational limits. The observation is geometry only; callers
    retain responsibility for input, source and PDK authority.
    """
    from clay.layout.occurrence_channels import _shape_keys
    from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
    axis, width, length, sides = _shared_mos_channel_geometry(active, poly, channel)
    try:
        complete_sides = []
        for side in sides:
            records = _shape_keys(side)
            if len(records) != 1:
                raise ValueError('diffusion side is not one complete component')
            complete_sides.append(PhysicalTerminalPolygonNM(*records[0]))
    except ValueError as error:
        raise SharedMosChannelUnsupported('shared MOS channel geometry: '
                                          'complete diffusion components cannot be represented') from error
    return SharedMosChannelObservation(axis, width, length, tuple(complete_sides))


def shared_mos_channel_dimensions_nm(active, poly, channel):
    """Return W/L using the same complete shared-channel geometry core."""
    _axis, width, length, _sides = _shared_mos_channel_geometry(active, poly, channel)
    return width, length
