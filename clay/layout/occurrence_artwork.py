"""Immutable copies of actual occurrence artwork, without source authority.

Records preserve every layer and polygon multiplicity at the replication seam.
Their ownership is a materializer fact, not an inference from channels or boxes.
"""
from dataclasses import dataclass


class SourceArtworkError(ValueError):
    """An occurrence source or its retained geometry is malformed."""


def polygon_records(polygons):
    return tuple(sorted((item.layer, item.datatype,
                         tuple((point.x, point.y) for point in item.polygon.vertices))
                        for item in polygons))


@dataclass(frozen=True, slots=True)
class OccurrenceArtworkSource:
    owner: str
    origin_nm: tuple[int, int]
    orientation: str
    polygon_records: tuple
    terminals: tuple

    def __post_init__(self):
        owner = self.owner
        origin = self.origin_nm
        orient = self.orientation
        if (
            type(owner) is not str
            or not owner
            or len(owner) > 4096
            or type(origin) is not tuple
            or len(origin) != 2
            or type(origin[0]) is not int
            or type(origin[1]) is not int
            or origin[0] < -1_000_000_000
            or origin[0] > 1_000_000_000
            or origin[1] < -1_000_000_000
            or origin[1] > 1_000_000_000
            or type(orient) is not str
            or orient not in {'R0', 'R90', 'R180', 'R270', 'MR0', 'MR90', 'MR180', 'MR270'}
        ):
            raise SourceArtworkError('fixed occurrence source frame is malformed')
        records = self.polygon_records
        if type(records) is not tuple or not (1 <= len(records) <= 262144):
            raise SourceArtworkError('fixed occurrence artwork inventory is malformed')
        total_vertices = 0
        for row in records:
            if type(row) is not tuple or len(row) != 3:
                raise SourceArtworkError('fixed occurrence artwork inventory is malformed')
            layer, datatype, vertices = row
            if type(vertices) is not tuple:
                raise SourceArtworkError('fixed occurrence artwork exceeds vertex bound')
            vlen = len(vertices)
            total_vertices += vlen
            if total_vertices > 1048576:
                raise SourceArtworkError('fixed occurrence artwork exceeds vertex bound')
            if (
                type(layer) is not int
                or layer < 0
                or layer > 65535
                or type(datatype) is not int
                or datatype < 0
                or datatype > 65535
                or vlen < 3
                or vlen > 4096
            ):
                raise SourceArtworkError('fixed occurrence artwork record is malformed')
            for vertex in vertices:
                if type(vertex) is not tuple or len(vertex) != 2:
                    raise SourceArtworkError('fixed occurrence artwork record is malformed')
                vx, vy = vertex
                if (
                    type(vx) is not int
                    or type(vy) is not int
                    or vx < -1_000_000_000
                    or vx > 1_000_000_000
                    or vy < -1_000_000_000
                    or vy > 1_000_000_000
                ):
                    raise SourceArtworkError('fixed occurrence artwork record is malformed')
        for idx in range(len(records) - 1):
            if records[idx] > records[idx + 1]:
                raise SourceArtworkError('fixed occurrence artwork is not canonical')
        terminals = self.terminals
        if type(terminals) is not tuple or not (1 <= len(terminals) <= 4096):
            raise SourceArtworkError('fixed occurrence terminals are malformed')
        for terminal in terminals:
            if type(terminal) is not tuple or len(terminal) != 3:
                raise SourceArtworkError('fixed occurrence terminal record is malformed')
            t0, t1, t2 = terminal
            if (
                type(t0) is not str
                or not t0
                or len(t0) > 4096
                or type(t1) is not str
                or not t1
                or len(t1) > 4096
                or type(t2) is not tuple
                or len(t2) != 4
            ):
                raise SourceArtworkError('fixed occurrence terminal record is malformed')
            b0, b1, b2, b3 = t2
            if (
                type(b0) is not int
                or type(b1) is not int
                or type(b2) is not int
                or type(b3) is not int
                or b0 < -1_000_000_000
                or b0 > 1_000_000_000
                or b1 < -1_000_000_000
                or b1 > 1_000_000_000
                or b2 < -1_000_000_000
                or b2 > 1_000_000_000
                or b3 < -1_000_000_000
                or b3 > 1_000_000_000
                or b0 >= b2
                or b1 >= b3
            ):
                raise SourceArtworkError('fixed occurrence terminal record is malformed')
        for idx in range(len(terminals) - 1):
            if terminals[idx] > terminals[idx + 1]:
                raise SourceArtworkError('fixed occurrence terminals are not canonical')

    @classmethod
    def capture(cls, *, owner, origin_nm, orientation, polygons, terminals):
        return cls(owner, origin_nm, orientation, polygon_records(polygons), tuple(sorted(terminals)))

    def polygons(self):
        from clay.layout.backend import LayerPolygon, Point, Polygon
        self.__post_init__()
        return tuple(LayerPolygon(layer=layer, datatype=datatype,
            polygon=Polygon(vertices=[Point(x, y) for x, y in vertices]))
            for layer, datatype, vertices in self.polygon_records)

    def terminal_boxes(self, role):
        self.__post_init__()
        return tuple(box for terminal_role, _net, box in self.terminals if terminal_role == role)


def occurrence_artwork_payload(sources):
    """Bind the complete per-owner partition after independent row replay."""
    if (type(sources) is not tuple or not 1 <= len(sources) <= 4096
            or any(type(source) is not OccurrenceArtworkSource for source in sources)):
        raise SourceArtworkError('owned artwork occurrence source inventory is incomplete')
    polygons = vertices = 0
    for source in sources:
        source.__post_init__()
        polygons += len(source.polygon_records)
        vertices += sum(len(record[2]) for record in source.polygon_records)
        if polygons > 262144 or vertices > 1048576:
            raise SourceArtworkError('owned artwork source complexity exceeds bounds')
    return tuple((source.owner, source.origin_nm, source.orientation, source.polygon_records, source.terminals)
                 for source in sources)


def rotate_occurrence_orientation(initial, rotation):
    """Left-compose a proper row rotation with the occurrence's D4 transform."""
    if (initial not in {'R0', 'R90', 'R180', 'R270', 'MR0', 'MR90', 'MR180', 'MR270'}
            or rotation not in {'R90', 'R180', 'R270'}):
        raise SourceArtworkError('occurrence artwork rotation is malformed')
    prefix = 'MR' if initial.startswith('M') else 'R'
    return prefix + str((int(initial[len(prefix):]) + int(rotation[1:])) % 360)
