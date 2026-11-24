"""Layer-aware electrical connectivity primitive.

THE ROOT FIX for the terminal/cap problem. A device terminal may sit on ANY routing metal,
and a device (a MIM cap) may have several DISTINCT electrical plates. The router/verifier
must therefore reason about electrical nodes, not "metal1 only" and not "any 2D overlap".

This module derives electrical NODES from raw artwork using a PDK-data-driven connect model:
same-layer touching shapes merge; two shapes on different layers join ONLY when a via of the
declared layer-pair bridges them. A pure 2D-overlap test (what an earlier hand-rolled trace
used) wrongly shorts stacked plates -- the gf180 MIM top/bottom plate overlap in 2D but are a
real capacitor, two nodes. This engine is layer-aware, so it does not make that error.

Leakage discipline: the connect model comes from the PDK ADAPTER's own metal/via stack
(``tech.metal_key`` / ``tech.via_key`` / ``tech.contact_key``), never from a reference layout.
It is generic: one engine for every device and PDK, no per-circuit / per-device branching.

Device-recognition overlays (e.g. a MIM cap's derived bottom-plate vs routing-metal split)
are added as DATA on the model per PDK (reverse-engineered from the foundry deck), not as
code branches here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import klayout.db as kdb

_Layer = Tuple[int, int]

# A derived-op operand is either a RAW layer key (layer, datatype) tuple, or a
# back-reference "$N" to the N-th prior op's result inside the SAME DerivedLayer.
_Operand = Union[_Layer, str]


@dataclass(frozen=True)
class DerivedLayer:
    """A synthetic layer computed from raw layers by klayout boolean ops.

    Device-recognition (e.g. the gf180 MIM bottom plate ``mim_virtual``) is
    expressed as DATA with this spec rather than as a code branch.  ``name`` is
    the synthetic ``(layer, datatype)`` key the result is stored under; ``ops``
    is an ordered list of operations, each a tuple ``(op, a[, b])``.

    Op vocabulary (small, well-defined):
      * ``("sized", a, amount_dbu)``           -- ``a.sized(amount)``  (bloat in dbu)
      * ``("and", a, b)``                      -- ``a & b``  (intersection)
      * ``("not", a, b)``                      -- ``a - b``  (subtraction)
      * ``("or",  a, b)``                      -- ``a | b``  (union)
      * ``("interacting", a, b)``              -- shapes of ``a`` that touch ``b``

    Each operand is either a raw layer key already present in the input shapes,
    a previously-materialised derived layer's ``name``, or a back-reference
    ``"$N"`` to the result of this DerivedLayer's N-th op (0-based).  The LAST
    op's result is the value stored under ``name``.
    """
    name: _Layer
    ops: Tuple[Tuple, ...] = ()

    def __post_init__(self):
        # normalise ops/name to hashable, frozen forms (allow list input at call site).
        object.__setattr__(self, "name", tuple(self.name))
        object.__setattr__(self, "ops", tuple(tuple(o) for o in self.ops))


@dataclass(frozen=True)
class ConnectModel:
    """Electrical connect model: routing metals + via layer-pairs.

    ``metals``: routing-metal ``(layer, datatype)`` keys (any order; low->high preferred).
    ``vias``: tuple of ``(via_key, lower_metal_key, upper_metal_key)`` -- a via on ``via_key``
    connects a shape on ``lower_metal_key`` to a shape on ``upper_metal_key`` where it
    overlaps BOTH (layer-aware).
    ``extra_connect``: optional extra ``(cut_key, layer_a, layer_b)`` triples for non-metal
    conductors (e.g. poly/diffusion via contact) or device-recognition layers.
    ``derived``: optional ``DerivedLayer`` specs.  Before connectivity is run, each derived
    layer is MATERIALISED from the input shapes by its boolean-op recipe and added to the
    shape set under its synthetic name; ``metals`` / ``vias`` / ``extra_connect`` may then
    reference those synthetic names.  This is how a MIM cap's bottom plate (``mim_virtual``)
    becomes a node DISTINCT from the routable metal (``metal2_ncap``) -- as DATA, not code.
    """
    metals: Tuple[_Layer, ...] = ()
    vias: Tuple[Tuple[_Layer, _Layer, _Layer], ...] = ()
    extra_connect: Tuple[Tuple[_Layer, _Layer, _Layer], ...] = ()
    derived: Tuple[DerivedLayer, ...] = ()
    # ``bridges``: pairs ``(layer_a, layer_b)`` that connect by ABUTMENT (where
    # the two regions touch / interact), with NO via -- the foundry-deck
    # ``connect(layer_a, layer_b)`` primitive.  Used to rejoin a derived-split
    # routing layer to its plate (the gf180 ``connect(metal2_ncap, mim_virtual)``
    # bridge that the open deck is missing).
    bridges: Tuple[Tuple[_Layer, _Layer], ...] = ()

    @classmethod
    def from_tech(cls, tech, *, max_metal: int = 8, max_via: int = 7) -> "ConnectModel":
        """Build the standard-stack connect model from a PDK adapter, generically.

        metal_key(1..N) are routing metals; via_key(n) connects metal n<->n+1; contact_key
        connects (diffusion/poly) up to metal1. No PDK names are hard-coded; whatever the
        adapter exposes is what the model carries.
        """
        metals: List[_Layer] = []
        for n in range(1, max_metal + 1):
            try:
                metals.append(tuple(tech.metal_key(n)))  # type: ignore[arg-type]
            except Exception:
                break
        vias: List[Tuple[_Layer, _Layer, _Layer]] = []
        for n in range(1, max_via + 1):
            if n >= len(metals):
                break
            try:
                vk = tuple(tech.via_key(n))  # type: ignore[arg-type]
            except Exception:
                continue
            vias.append((vk, metals[n - 1], metals[n]))  # via n: metal n <-> metal n+1
        extra: List[Tuple[_Layer, _Layer, _Layer]] = []
        # contact: (diff/poly) -> metal1. We model it as bridging metal1 to itself's lower
        # conductors only when those layers are supplied as shapes; harmless otherwise.
        return cls(metals=tuple(metals), vias=tuple(vias), extra_connect=tuple(extra))


def _regions_by_layer(shapes_by_layer: Dict[_Layer, Sequence], ) -> Dict[_Layer, kdb.Region]:
    out: Dict[_Layer, kdb.Region] = {}
    for key, polys in (shapes_by_layer or {}).items():
        r = kdb.Region()
        for p in polys or ():
            r.insert(p)
        r.merge()
        out[tuple(key)] = r
    return out


def _resolve_operand(
    operand: _Operand,
    regs: Dict[_Layer, kdb.Region],
    locals_: List[kdb.Region],
) -> kdb.Region:
    """Resolve a derived-op operand to a Region.

    ``"$N"`` -> the N-th prior local result; any other value -> a raw/derived
    layer key looked up in ``regs`` (empty Region if absent).
    """
    if isinstance(operand, str):
        if operand.startswith("$"):
            return locals_[int(operand[1:])]
        raise ValueError(f"unknown string operand {operand!r} (expected '$N')")
    return regs.get(tuple(operand), kdb.Region())


def materialise_derived(
    shapes_by_layer: Dict[_Layer, Sequence],
    derived: Sequence[DerivedLayer],
) -> Dict[_Layer, kdb.Region]:
    """Materialise each ``DerivedLayer`` into a Region, returned by synthetic name.

    Raw input shapes are merged into Regions first; then each derived layer's op
    list is evaluated in order (back-references ``$N`` point at earlier ops of the
    SAME layer; bare layer keys point at raw OR earlier-derived layers).  The
    last op's result is the derived layer's value.  This is the PDK-data-driven
    device-recognition step (no per-device code lives here -- only the op vocab).
    """
    regs: Dict[_Layer, kdb.Region] = _regions_by_layer(shapes_by_layer)
    for spec in derived or ():
        locals_: List[kdb.Region] = []
        for op in spec.ops:
            kind = op[0]
            if kind == "sized":
                a = _resolve_operand(op[1], regs, locals_)
                res = a.sized(int(op[2]))
            elif kind == "and":
                a = _resolve_operand(op[1], regs, locals_)
                b = _resolve_operand(op[2], regs, locals_)
                res = a & b
            elif kind == "not":
                a = _resolve_operand(op[1], regs, locals_)
                b = _resolve_operand(op[2], regs, locals_)
                res = a - b
            elif kind == "or":
                a = _resolve_operand(op[1], regs, locals_)
                b = _resolve_operand(op[2], regs, locals_)
                res = a | b
            elif kind == "interacting":
                a = _resolve_operand(op[1], regs, locals_)
                b = _resolve_operand(op[2], regs, locals_)
                res = a.interacting(b)
            else:
                raise ValueError(f"unknown derived-layer op {kind!r}")
            res = res.dup()
            res.merge()
            locals_.append(res)
        regs[tuple(spec.name)] = locals_[-1] if locals_ else kdb.Region()
    return regs


def extract_nodes(
    shapes_by_layer: Dict[_Layer, Sequence],
    model: ConnectModel,
) -> List[Dict[_Layer, kdb.Region]]:
    """Return the electrical NODES of ``shapes_by_layer`` under ``model``.

    Each node is a ``{layer_key: kdb.Region}`` mapping (the part of that node on each layer).
    Same-layer touching shapes are one island; islands on different layers merge only through
    a via of the matching layer-pair that overlaps both.
    """
    from clay.layout.connectivity_queries import FragmentQueries, region_box

    # Step 1: materialise derived layers (device-recognition) from raw shapes,
    # so the connect model can reference synthetic names (e.g. mim_virtual).
    regs = materialise_derived(shapes_by_layer, model.derived)
    # fragment = a single connected island on one layer.
    frags: List[Tuple[_Layer, kdb.Region]] = []
    via_layers = {tuple(v[0]) for v in model.vias} | {tuple(e[0]) for e in model.extra_connect}
    for key, reg in regs.items():
        # Only DECLARED metals are node-bearing conductors. Raw source layers
        # consumed by a derived recipe (e.g. raw Metal2 split into mim_virtual /
        # metal2_ncap) and non-conductor markers (cap_mk / mim_l_mk) must NOT seed
        # nodes. When ``metals`` is declared this is the authoritative filter; an
        # empty ``metals`` (degenerate models) falls back to "anything but a via".
        if model.metals:
            if key not in model.metals:
                continue
        # A via layer that is ALSO listed in metals stays a frag; a via-only layer
        # is a connector, handled below.
        elif key in via_layers and key not in model.metals:
            continue
        for poly in reg.each():
            frags.append((key, kdb.Region(poly)))

    queries = FragmentQueries(frags)
    parent = list(range(len(frags)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # same-layer islands are already separate frags (distinct nodes unless a via joins).
    # bridge via layer-pairs.
    for (via_key, lo, hi) in tuple(model.vias) + tuple(model.extra_connect):
        vreg = regs.get(tuple(via_key))
        if vreg is None or vreg.is_empty():
            continue
        lo = tuple(lo)
        hi = tuple(hi)
        for vp in vreg.each():
            vr = kdb.Region(vp)
            box = region_box(vr)
            lo_hits = [i for i in queries.candidates(lo, box)
                       if not (frags[i][1] & vr).is_empty()]
            hi_hits = [i for i in queries.candidates(hi, box)
                       if not (frags[i][1] & vr).is_empty()]
            for i in lo_hits:
                for j in hi_hits:
                    union(i, j)

    # abutment bridges: two layers connect where their fragments touch (no via).
    for (la, lb) in tuple(model.bridges):
        la = tuple(la)
        lb = tuple(lb)
        a_frags = [(i, fr) for i, (k, fr) in enumerate(frags) if k == la]
        for i, fa in a_frags:
            for j in queries.candidates(lb, region_box(fa), include_touching=True):
                fb = frags[j][1]
                # touching == non-empty interaction (shared edge or overlap).
                if not fa.interacting(fb).is_empty():
                    union(i, j)

    groups: Dict[int, Dict[_Layer, kdb.Region]] = {}
    for i, (k, fr) in enumerate(frags):
        root = find(i)
        node = groups.setdefault(root, {})
        if k in node:
            node[k] += fr
            node[k].merge()
        else:
            node[k] = fr.dup()
    return list(groups.values())


def node_index_at(
    nodes: List[Dict[_Layer, kdb.Region]],
    layer: _Layer,
    x: int,
    y: int,
) -> Optional[int]:
    """Index of the node whose ``layer`` region contains point (x, y) in dbu, else None."""
    pt = kdb.Region(kdb.Box(int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1))
    layer = tuple(layer)
    for idx, node in enumerate(nodes):
        reg = node.get(layer)
        if reg is not None and not (reg & pt).is_empty():
            return idx
    return None
