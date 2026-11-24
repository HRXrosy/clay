"""Immutable stage scope for declared net matching, independent of device pairs.

Source spans retain the actual parser operands. Semantic digests omit whitespace
locations; callers binding source bytes retain the complete declaration snapshot.
An omitted scope preserves the historical placement and routing behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _pair(value, *, canonical=False):
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("net match pair must be an exact two-tuple")
    if any(type(item) is not str or not item or item != item.strip() for item in value) or value[0] == value[1]:
        raise ValueError("net match pair needs two distinct nonempty nets")
    if canonical and value != tuple(sorted(value)):
        raise ValueError("net match scope pair must be canonical")
    return tuple(sorted(value))


@dataclass(frozen=True)
class NetMatchDeclaration:
    kind: str
    nets: tuple[str, str]
    source_span: tuple[tuple[str, int | str], ...] = field(default=(), compare=False)

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in {"match", "symmetric"}:
            raise ValueError("net match declaration kind is unsupported")
        _pair(self.nets)
        if type(self.source_span) is not tuple:
            raise TypeError("net match source span must be an exact tuple")
        if self.source_span:
            coordinates = ("column", "end_column", "end_line", "line")
            with_file = ("column", "end_column", "end_line", "file", "line")
            if (any(type(row) is not tuple or len(row) != 2 for row in self.source_span)
                    or tuple(row[0] for row in self.source_span) not in (coordinates, with_file)):
                raise ValueError("net match source span is malformed")
            span = dict(self.source_span)
            if any(type(span[key]) is not int or span[key] < 1 for key in coordinates):
                raise ValueError("net match source span coordinates are malformed")
            if "file" in span and (type(span["file"]) is not str or not span["file"]):
                raise ValueError("net match source span file must be a nonempty string")
            if (span["end_line"], span["end_column"]) < (span["line"], span["column"]):
                raise ValueError("net match source span is reversed")

    def canonical_data(self):
        self.__post_init__()
        return {"kind": self.kind, "nets": list(self.nets),
                "source_span": dict(self.source_span) if self.source_span else None}


@dataclass(frozen=True)
class NetMatchScope:
    nets: tuple[str, str]
    scope: str
    declarations: tuple[NetMatchDeclaration, ...]

    def __post_init__(self):
        _pair(self.nets, canonical=True)
        if type(self.scope) is not str or self.scope not in {"legacy", "route"}:
            raise ValueError("net match scope is unsupported")
        if type(self.declarations) is not tuple or not self.declarations:
            raise TypeError("net match scope needs an exact nonempty declaration tuple")
        for declaration in self.declarations:
            if type(declaration) is not NetMatchDeclaration:
                raise TypeError("net match scope contains a foreign declaration")
            declaration.__post_init__()
            if _pair(declaration.nets) != self.nets:
                raise ValueError("net match scope declaration pair differs")
            if self.scope == "route" and declaration.kind != "match":
                raise ValueError("only @match supports explicit route scope")

    def canonical_data(self):
        self.__post_init__()
        return {"nets": list(self.nets), "scope": self.scope,
                "declarations": [row.canonical_data() for row in self.declarations]}


def replay_match_scopes(scopes, diff_pairs=None):
    if type(scopes) is not tuple:
        raise TypeError("net match scope inventory must be an exact tuple")
    by_pair = {}
    for row in scopes:
        if type(row) is not NetMatchScope:
            raise TypeError("net match scope inventory contains a foreign record")
        row.__post_init__()
        if row.nets in by_pair:
            raise ValueError("net match scope inventory repeats a pair")
        by_pair[row.nets] = row
    if tuple(by_pair) != tuple(sorted(by_pair)):
        raise ValueError("net match scope inventory must be canonical")
    if scopes and diff_pairs is not None:
        if type(diff_pairs) is not tuple:
            raise TypeError("scoped diff pair inventory must be an exact tuple")
        if not set(by_pair) <= {_pair(pair) for pair in diff_pairs}:
            raise ValueError("net match scope pair is absent from diff pair inventory")
    return scopes


def match_scopes_from_declarations(declarations):
    by_pair = {}
    for declaration in declarations:
        if declaration.kind not in {"match", "symmetric"}:
            continue
        attributes = declaration.attributes
        if type(attributes) is not dict or set(attributes) - {"scope"}:
            raise ValueError("net match scope attributes are unsupported")
        scope = "legacy"
        if "scope" in attributes:
            if declaration.kind != "match" or attributes["scope"] != "route":
                raise ValueError("explicit @match scope must be route")
            scope = "route"
        pair = _pair(declaration.nets)
        span = declaration.source_span
        if span is not None and type(span) is not dict:
            raise TypeError("parsed net match source span must be a dict or None")
        source = NetMatchDeclaration(declaration.kind, declaration.nets,
                                     tuple(sorted(span.items())) if span else ())
        if pair in by_pair and by_pair[pair][0] != scope:
            raise ValueError(f"conflicting net match scopes for {pair!r}")
        by_pair.setdefault(pair, (scope, []))[1].append(source)
    return tuple(NetMatchScope(pair, scope, tuple(rows))
                 for pair, (scope, rows) in sorted(by_pair.items()))


def merge_match_scopes(first, second):
    by_pair = {row.nets: row for row in replay_match_scopes(first)}
    for row in replay_match_scopes(second):
        previous = by_pair.get(row.nets)
        if previous is not None:
            if previous.scope != row.scope:
                raise ValueError(f"conflicting net match scopes for {row.nets!r}")
            # Keep all actual declaration operands, including repeated source lines.
            row = NetMatchScope(row.nets, row.scope, previous.declarations + row.declarations)
        by_pair[row.nets] = row
    return tuple(by_pair[pair] for pair in sorted(by_pair))


def placement_diff_pairs(intent):
    pairs = getattr(intent, "diff_pairs", ()) or ()
    scopes = replay_match_scopes(getattr(intent, "match_scopes", ()), pairs)
    if not scopes:
        return tuple(pairs)
    route_only = {row.nets for row in scopes if row.scope == "route"}
    return tuple(pair for pair in pairs if _pair(pair) not in route_only)


def scope_digest_data(scopes):
    """No additional digest key for any historical, unscoped declaration."""
    scopes = replay_match_scopes(scopes)
    return [{"nets": list(row.nets), "scope": row.scope}
            for row in scopes if row.scope == "route"]


def placement_net_subset(full_pairs, placement_pairs):
    """Normalize only the omitted legacy default; validate explicit subsets."""
    if placement_pairs is None:
        return full_pairs
    if type(full_pairs) is not tuple or type(placement_pairs) is not tuple:
        raise TypeError("placement net pair inventory must be an exact tuple")
    full = tuple(_pair(pair) for pair in full_pairs)
    subset = tuple(_pair(pair, canonical=True) for pair in placement_pairs)
    if (len(set(subset)) != len(subset) or not set(subset) <= set(full)
            or subset != tuple(pair for pair in full if pair in set(subset))):
        raise ValueError("placement net pairs must be a unique canonical projection of full net pairs")
    return placement_pairs


def routing_match_length_pairs(intent, present_nets):
    """Replay immutable match-length declarations against actual route nets.

    Both placement certification and route lowering consume this declaration.
    Net pairs are symmetric; repeated declarations preserve the historical one
    hard Length constraint. This function returns data and performs no routing.
    """
    raw_pairs = getattr(intent, "match_length_pairs", ())
    if type(raw_pairs) is not tuple:
        raise TypeError("match_length_pairs must be an exact tuple")
    canonical = set()
    for pair in raw_pairs:
        if (type(pair) is not tuple or len(pair) != 2
                or type(pair[0]) is not str or type(pair[1]) is not str
                or not pair[0] or not pair[1] or pair[0] == pair[1]):
            raise ValueError("match_length declaration must contain two distinct net names")
        net_a, net_b = sorted((pair[0], pair[1]))
        absent = tuple(net for net in (net_a, net_b) if net not in present_nets)
        if absent:
            raise ValueError("match_length declaration names absent route net(s): " + ", ".join(absent))
        canonical.add((net_a, net_b))
    return tuple(sorted(canonical))
