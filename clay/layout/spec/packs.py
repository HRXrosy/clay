"""Spec-pack schema + module-level registry (SP1).

Pure stdlib data structures and a deterministic registry. No coordinates, no
pipeline coupling. Emitters here are declared but only wired into the
zero-coordinate QSR relation network in SP2.

Hardness taxonomy <-> Clay's explicit 5-state verdict mapping
-------------------------------------------------------------
A rule's ``hardness`` declares how its named checker is allowed to verdict it.
The five declarable classes map onto Clay's 5 verdict states as follows:

    hardness           -> reachable verdict states
    ---------------------------------------------------------------
    "hard"             -> pass | fail               (binary obligation;
                          may also be "unmeasurable" if the oracle cannot
                          observe the geometry, but never silently)
    "quantified_soft"  -> pass | fail               (verdicts against a
                          numeric/categorical threshold band)
    "fidelity"         -> pass | fail               (a GEOMETRIC-QUALITY band
                          that GATES: a RED measurement is a blocking FAIL,
                          like DRC/LVS -- NOT an advisory. This is the F1 class
                          that fills the empty QUALITY tier so a legal-but-
                          geometrically-wrong layout cannot be reported green;
                          may degrade to "unmeasurable" when the geometry is
                          unobservable, never a silent pass)
    "qualitative_soft" -> advisory                  (reported, never gates;
                          may degrade to "unmeasurable")
    "unsupported"      -> unsupported               (capability absent in the
                          target PDK/backend; explicit, never silent pass)

The fifth Clay state, "unmeasurable", is reachable from any hardness when the
checker's oracle cannot observe the relevant geometry; it is never collapsed to
a silent pass. SP1 only stores the declarative hardness class; the checker
named by each rule is the independent oracle that issues the actual verdict in
later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


# The five declarable hardness classes (see module docstring for the mapping
# onto Clay's explicit 5-state verdict model). Anything outside this set is a
# contract violation, not a silently-ignored value. ``fidelity`` is the F1
# gating QUALITY-quality class: a RED measurement gates (blocking FAIL), so a
# legal-but-geometrically-wrong layout cannot be reported green.
VALID_HARDNESS = frozenset(
    {"hard", "quantified_soft", "fidelity", "qualitative_soft", "unsupported"}
)


@dataclass
class SpecRule:
    """One design rule: a taxonomy tuple + detector + emitter + named checker.

    Fields
    ------
    name : str
        Stable identifier for the rule within its pack.
    metric : str
        What is measured (e.g. "offset", "mismatch").
    effect : str
        The physical effect targeted (e.g. "process_gradient").
    object : str
        The circuit object the rule applies to (e.g. "diff_pair").
    hardness : str
        One of :data:`VALID_HARDNESS`; declares the verdict class.
    detect : Callable
        Topological detector ``detect(intent) -> matches``. No coordinates.
    emit : Callable
        Emitter ``emit(intent, group, net) -> None`` that (in SP2) populates
        the zero-coordinate QSR relation network. No coordinates.
    checker : str
        Name of the independent oracle that verdicts this rule by hardness.
    threshold : optional
        Categorical threshold descriptor (e.g. "class_A"). Must NOT carry a
        coordinate-scale numeric value; thresholds are categories, not microns.
    tier : optional
        PRIORITY axis (orthogonal to hardness). One of
        :data:`clay.layout.spec.priority.VALID_TIER`, or ``None`` to derive the
        tier from hardness/checker via the default table. Declaring it overrides
        the default and carries provenance. Backward compatible: defaults to
        ``None`` so existing rule constructions are unchanged.
    superior_to : tuple of str
        PRIORITY axis (within-tier defeasible superiority). The names of the
        peer rules this rule defeats when they conflict (a partial order, not a
        total order). Backward compatible: defaults to the empty tuple.
    """

    name: str
    metric: str
    effect: str
    object: str
    hardness: str
    detect: Callable
    emit: Callable
    checker: str
    threshold: Optional[object] = None
    # --- PRIORITY axis (optional; None / () => derived defaults; see priority.py)
    tier: Optional[str] = None
    superior_to: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.hardness not in VALID_HARDNESS:
            raise ValueError(
                f"invalid hardness {self.hardness!r}; expected one of "
                f"{sorted(VALID_HARDNESS)}"
            )
        if not callable(self.detect):
            raise ValueError("detect must be callable")
        if not callable(self.emit):
            raise ValueError("emit must be callable")
        if not isinstance(self.checker, str) or not self.checker:
            raise ValueError("checker must be a non-empty string")
        # PRIORITY axis validation (orthogonal to hardness). Import locally to
        # keep the priority taxonomy as the single source of truth without a
        # module-load cycle.
        if self.tier is not None:
            from clay.layout.spec.priority import VALID_TIER

            if self.tier not in VALID_TIER:
                raise ValueError(
                    f"invalid tier {self.tier!r}; expected one of "
                    f"{list(VALID_TIER)} or None"
                )
        if not isinstance(self.superior_to, tuple) or not all(
            isinstance(s, str) for s in self.superior_to
        ):
            raise ValueError("superior_to must be a tuple of str")


@dataclass
class SpecPack:
    """A named, loadable collection of :class:`SpecRule`.

    ``applies_to`` lists circuit-category tags this pack serves; ``"*"`` means
    the pack is universal (applies to every category).
    """

    name: str
    rules: list = field(default_factory=list)
    applies_to: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Module-level registry (deterministic; register is idempotent via overwrite). #
# --------------------------------------------------------------------------- #
_REGISTRY: dict = {}


def register_pack(pack: SpecPack) -> None:
    """Register (or deterministically overwrite) a pack by name.

    Same-name re-registration overwrites the prior entry, so registration is
    idempotent and order-stable for repeated calls with equivalent packs.
    """
    if not isinstance(pack, SpecPack):
        raise TypeError("register_pack expects a SpecPack")
    if not isinstance(pack.name, str) or not pack.name:
        raise ValueError("pack name must be a non-empty string")
    _REGISTRY[pack.name] = pack


def load_pack(name: str) -> SpecPack:
    """Return the registered pack ``name``; raise ``KeyError`` if unknown.

    Never silently returns a default -- an unknown pack is a hard error.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown spec pack: {name!r}")
    return _REGISTRY[name]


def list_packs() -> list:
    """Return registered pack names, sorted for deterministic output."""
    return sorted(_REGISTRY.keys())


def packs_for_category(category: str) -> list:
    """Return names of packs that apply to ``category``.

    A pack applies if its ``applies_to`` contains ``category`` or the universal
    tag ``"*"``. Output is sorted for determinism.
    """
    out = [
        name
        for name, pack in _REGISTRY.items()
        if category in pack.applies_to or "*" in pack.applies_to
    ]
    return sorted(out)


def pack_as_dict(name: str) -> dict:
    """Declarative serialization of a pack: symbols only, no callables.

    Emits the pack name, its ``applies_to`` tags, and each rule's categorical
    string fields (metric/effect/object/hardness/checker/threshold). Callables
    (detect/emit) are deliberately excluded so the view is a pure declarative
    surface with no coordinate-scale numerics -- the zero-coordinate guard.
    """
    pack = load_pack(name)
    return {
        "name": pack.name,
        "applies_to": list(pack.applies_to),
        "rules": [
            {
                "name": r.name,
                "metric": r.metric,
                "effect": r.effect,
                "object": r.object,
                "hardness": r.hardness,
                "checker": r.checker,
                "threshold": r.threshold,
            }
            for r in pack.rules
        ],
    }


# --------------------------------------------------------------------------- #
# SP2 -- apply loaded packs over an intent, emitting into the zero-coordinate  #
# QSR relation network and recording tagged, provenance-bearing obligations.  #
# --------------------------------------------------------------------------- #
def _network_devices(intent):
    """Design-sourced device names (no geometry) for node population.

    Pulls MOS devices and passives from the netlist intent only. Order is the
    declaration order then any device with a bbox correspondence, deduplicated
    deterministically. No coordinate is read -- only the symbolic name/kind so
    the populated nodes are translation-invariant by construction."""
    devs: list = []
    seen: set = set()
    for d in (getattr(intent, "devices", None) or []):
        nm = str(d.name)
        if nm not in seen:
            seen.add(nm)
            devs.append((nm, str(getattr(d, "kind", None) or "") or None))
    for p in (getattr(intent, "passives", None) or []):
        nm = str(p.name)
        if nm not in seen:
            seen.add(nm)
            devs.append((nm, str(getattr(p, "kind", None) or "") or None))
    return devs


def apply_packs(intent, pack_names):
    """Run the named packs' rules over ``intent`` -> a ``RelationNetwork``.

    For each rule (in pack order, then rule order) the rule's topological
    ``detect(intent)`` yields the matched groups (e.g. diff/mirror pairs from
    the DESIGN, never name matches, never coordinates). For each matched group
    the rule's ``emit(intent, group, net)`` populates the zero-coordinate QSR
    network with relations/hyperedges, and a tagged obligation is appended to
    ``net.obligations`` carrying the (metric, effect, object, hardness, checker,
    rule, pack) provenance.

    Determinism: packs are applied in the order given; within a pack, rules in
    declaration order; within a rule, the detector's emission order is fixed.
    An empty ``pack_names`` yields an inert network (no groups, no obligations).

    Zero-coordinate / translation-invariance: nodes are populated from DESIGN
    names only and emitters add categorical relations only; nothing reads
    ``intent.device_bboxes``. Shifting the reference bboxes cannot change the
    output.
    """
    from clay.layout.extract.placement_relations import RelationNetwork

    net = RelationNetwork()
    net.provenance = "spec-pack"
    # obligation ledger: each entry is a categorical, provenance-bearing dict
    # (no coordinate-scale numeric leaf).
    net.obligations = []

    # device nodes (design-sourced, no geometry) so emitted hyperedges name
    # real nodes; deterministic order.
    for nm, kind in _network_devices(intent):
        net.add_device(nm, kind=kind)

    for pack_name in pack_names:
        pack = load_pack(pack_name)
        for rule in pack.rules:
            for group in rule.detect(intent):
                rule.emit(intent, group, net)
                net.obligations.append(
                    {
                        "rule": rule.name,
                        "pack": pack.name,
                        "metric": rule.metric,
                        "effect": rule.effect,
                        "object": rule.object,
                        "hardness": rule.hardness,
                        "checker": rule.checker,
                        "members": list(group),
                    }
                )
    return net
