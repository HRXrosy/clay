"""Canonical, same-layer metal clearance declarations, in integer nanometres."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING


@dataclass(frozen=True, order=True)
class NetSpacingIntent:
    net_a: str
    net_b: str
    minimum_nm: int

    def __post_init__(self):
        if any(type(n) is not str or not n or n.strip() != n
               for n in (self.net_a, self.net_b)):
            raise ValueError("net_spacing requires two non-empty net names")
        if self.net_a.casefold() == self.net_b.casefold():
            raise ValueError("net_spacing requires two distinct nets")
        if type(self.minimum_nm) is not int or not 0 <= self.minimum_nm <= 2**53:
            raise ValueError("net_spacing minimum must be non-negative integer nm")
        a, b = sorted((self.net_a, self.net_b))
        object.__setattr__(self, "net_a", a)
        object.__setattr__(self, "net_b", b)


def parse_spacing_nm(number: str, unit: str) -> int:
    """Read the original decimal token; never shrink or round via binary floats."""
    scales = {"nm": 1, "u": 1000, "um": 1000, "mm": 1_000_000, "m": 1_000_000_000}
    if unit not in scales:
        raise ValueError("net_spacing requires a length unit: nm, u, um, mm or m")
    value = Decimal(number) * scales[unit]
    if not value.is_finite() or value < 0 or value > 2**53:
        raise ValueError("net_spacing length must be finite, non-negative and representable")
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def canonical_net_spacings(declarations):
    """A repeated or reversed pair keeps the strongest declared minimum."""
    if type(declarations) is not tuple:
        raise TypeError("net_spacing inventory requires an exact tuple")
    pairs = {}
    for item in declarations:
        if type(item) is not NetSpacingIntent:
            raise TypeError("net_spacing inventory requires NetSpacingIntent entries")
        value = NetSpacingIntent(item.net_a, item.net_b, item.minimum_nm)
        pair = (value.net_a, value.net_b)
        pairs[pair] = max(pairs.get(pair, 0), value.minimum_nm)
    return tuple(NetSpacingIntent(a, b, distance) for (a, b), distance in sorted(pairs.items()))


def lower_net_spacings(declarations, present_nets):
    """Resolve source spelling to physical nets without dropping absent pairs."""
    from clay.layout.route.constraints import Spacing

    present = set(present_nets)
    def resolve(name):
        if name in present:
            return name
        matches = sorted(n for n in present if n.casefold() == name.casefold())
        if len(matches) != 1:
            raise ValueError(f"net_spacing net {name!r} is absent or ambiguous in final terminals")
        return matches[0]

    return [Spacing(resolve(c.net_a), resolve(c.net_b), c.minimum_nm / 1000)
            for c in canonical_net_spacings(declarations)]


def verify_exact_net_spacings(artifact, tech, declarations):
    """Reread the bound GDS and measure complete connected metal components."""
    import klayout.db as db
    from clay.layout.route.deck_isolation import DeckConnectStack, NetProbe, check_deck_net_isolation
    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

    if type(artifact) is not ExactEmittedArtifact or tech is None:
        raise ValueError("net_spacing requires an exact GDS artifact and its routing technology")
    artifact.validate()
    stack = DeckConnectStack.from_tech(tech)
    metals = set(stack.conductors)
    probes = []
    for net, layer, bbox in artifact.route_manifest.entries:
        if tuple(layer) in metals:
            x0, y0, x1, y1 = bbox
            probes.append(NetProbe(net, layer, (x0 + x1) // 2, (y0 + y1) // 2))
    constraints = lower_net_spacings(declarations, {p.net for p in probes})
    layout = db.Layout()
    layout.read(str(artifact.exact_gds_path))
    # Route manifest coordinates are nm, whereas GDS probes are in file DBU.
    scale = 0.001 / layout.dbu
    probes = tuple(NetProbe(p.net, p.layer, round(p.x * scale), round(p.y * scale))
                   for p in probes)
    result = check_deck_net_isolation(layout=layout, cell=layout.top_cell(), stack=stack,
                                      probes=probes, spacing_constraints=constraints)
    artifact.validate()
    return result.spacing_reports
