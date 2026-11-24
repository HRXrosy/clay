"""Verify policy-bound router-owned via arrays on an emitted GDS."""
from collections import Counter


def verify_router_via_arrays(
    gds_path,
    routed,
    tech,
    expected_redundancy,
    *,
    top_cell=None,
    allow_single_escape: bool = False,
):
    """Replay each router-owned site against its emitted cuts and exact GDS.

    Device contacts are not router-owned sites. The in-memory route is checked
    separately from the reopened layout so a missing site inventory cannot be
    hidden by unrelated fixed device artwork.
    """
    import klayout.db as kdb

    if type(expected_redundancy) is not int or expected_redundancy < 1:
        raise ValueError("via array policy must be a positive integer")
    if type(allow_single_escape) is not bool:
        raise TypeError("allow_single_escape must be an explicit bool")
    if getattr(routed, "router_via_redundancy", None) != expected_redundancy:
        raise ValueError("router via evidence differs from selected source policy")
    sites = getattr(routed, "router_via_sites", None)
    if type(sites) is not tuple:
        raise ValueError("router via site inventory is missing")
    allowed_cut_counts = (
        (1, expected_redundancy ** 2)
        if allow_single_escape
        else (expected_redundancy ** 2,)
    )
    expected = Counter()
    for site in sites:
        if type(site) is not tuple or len(site) != 3:
            raise ValueError("router via site is malformed")
        net, key, cuts = site
        if type(net) is not str or not net or type(key) is not tuple or len(key) != 2:
            raise ValueError("router via ownership or layer is malformed")
        if type(cuts) is not tuple or len(cuts) not in allowed_cut_counts or len(set(cuts)) != len(cuts):
            raise ValueError("router via site lacks distinct declared cuts")
        for box in cuts:
            if type(box) is not tuple or len(box) != 4 or any(type(x) is not int for x in box):
                raise ValueError("router via cut is malformed")
            expected[(net, key, box)] += 1
    # Adjacent PDK via cut layers are obtained from the same selected tech.
    via_keys = set()
    for level in range(1, 10):
        try:
            key = tech.via_key(level)
        except (KeyError, LookupError, ValueError, AttributeError):
            break
        if key is None:
            break
        via_keys.add(tuple(key))
    if any(key not in via_keys for _, key, _ in expected):
        raise ValueError("router via site names a foreign PDK cut layer")
    actual = Counter()
    for net, route in routed.nets.items():
        for poly in route.wires:
            key = (int(poly.layer), int(poly.datatype))
            if key not in via_keys:
                continue
            points = tuple((int(v.x), int(v.y)) for v in poly.polygon.vertices)
            if len(points) != 4:
                raise ValueError("router via cut is not an exact rectangle")
            xs, ys = zip(*points)
            box = (min(xs), min(ys), max(xs), max(ys))
            if set(points) != {(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3])}:
                raise ValueError("router via cut polygon is not rectangular")
            actual[(str(net), key, box)] += 1
    if actual != expected:
        raise ValueError("router via site inventory differs from emitted route cuts")
    layout = kdb.Layout()
    layout.read(str(gds_path))
    if layout.dbu != .001:
        raise ValueError("router via GDS DBU differs")
    cell = layout.cell(top_cell) if top_cell else layout.top_cell()
    if cell is None:
        raise ValueError("router via GDS top cell missing")
    polygons = {}
    for key in {key for _, key, _ in expected}:
        region = kdb.Region(cell.begin_shapes_rec(layout.layer(*key))).merged()
        polygons[key] = {tuple((p.bbox().left,p.bbox().bottom,p.bbox().right,p.bbox().top))
                         for p in region.each() if p.is_box()}
    if any(box not in polygons[key] for _, key, box in expected):
        raise ValueError("emitted GDS is missing or changed a router-owned via cut")
    return {"sites": len(sites), "cut_claims": sum(expected.values()),
            "redundancy": expected_redundancy}
