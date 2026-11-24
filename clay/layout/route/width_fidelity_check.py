"""POWER-RAIL WIDTH-FIDELITY obligation -- an INDEPENDENT checker read from emitted GDS.

THE OBLIGATION
==============
The per-net wire width the router is TOLD to draw -- a declared ``@wire_width``
override, else the role-derived value from
:func:`clay.layout.route.width_model.derive_width_intent` (power rails floored at
``WidthPolicy.power_min_um`` = 4.0um; local signal at the PDK ``min_width``) -- must
be FAITHFULLY MATERIALIZED on that net's routed geometry. A power rail whose emitted
metal is NARROWER than its computed/declared width is a materializer-fidelity break:
the intent was computed but never reached the geometry.

This is the POSITIVE contract complementary to
:mod:`clay.layout.route.metal_min_width_check`. That checker asserts a FLOOR the PDK
owes (every metal edge >= ``min_width``); THIS checker asserts a FLOOR the DESIGN INTENT
owes (a power rail >= its 4.0um EM/IR width). A layout can be metal-min-width-clean and
still fail HERE: a rail drawn at 1.16um passes ``M1.1`` (0.23um) yet violates the 4.0um
power-width intent. The two obligations name different facts and neither subsumes the other.

INDEPENDENCE (checker not the checked)
======================================
This checker does NOT consult the router's internal state, its ``Width`` constraints, or
its ``unsatisfied`` list. It reads only the EMITTED GDS (the canonical artifact every
oracle reads) plus the per-net TARGET widths computed upstream. It IDENTIFIES the VDD /
GND conductors the SAME netlist-free way the standing structural instrument does --
:func:`ota_dev.analysis.structure_metrics.power_structure` (n-tap = OD & NIMP & NWELL ->
VDD, p-tap = OD & PIMP & !NWELL -> GND; tap-seeded metal-stack connectivity; longest
qualified run for the strap width). Reusing that tap-seeding is deliberate: net identity
comes from foundry-layer topology, NOT from any label or reference measurement, and NOT
by re-inventing a second identification path. A divergence between this checker and the
foundry LVS is a bug in this checker, never grounds to trust it over LVS.

WHAT THIS PROVES (honest scope)
===============================
It proves the emitted power-rail WIDTH meets the computed/declared width floor. It does
NOT prove connectivity or correctness -- that is KLayout LVS's job (DRC width-clean and
LVS-correct are separate facts; a rail can be wide AND shorted). It measures the widest
qualified metal RUN on each tap-seeded rail component (``_max_strap_width``); a rail that
happens to be wide because it merges a device source/tap PLATE (not a drawn distribution
strap) still MEASURES wide -- this checker gates the width floor, not strap provenance
(the latter is reported ``advisory`` when a rail is anomalously over-wide vs its target).

FIVE-STATE VERDICT (never a silent pass)
========================================
Per rail and overall, one of:
  * ``pass``         emitted rail width >= target (minus measurement tolerance);
  * ``fail``         emitted rail width < target -- the materialization break (RED);
  * ``unmeasurable`` no tap-seeded power component / klayout or structure_metrics
                     unavailable / rail present but width could not be measured;
  * ``unsupported``  the rail was identified but NO target width was supplied for it
                     (no computed/declared obligation to check against);
  * ``advisory``     the target is at (or below) the PDK ``min_width`` floor -- no
                     SUPRA-min widening was actually requested, so the width obligation
                     is inert here and already covered by the metal-min-width checker.
Overall is ``fail`` if ANY rail fails; else ``pass`` only if >= 1 rail truly passed and
none failed; else the strongest remaining non-pass state. A silent pass is never emitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

# Supply-net aliases, split high/low. Net IDENTITY on the geometry is tap-seeded
# (VDD = n-tap, GND = p-tap) by structure_metrics; these alias groups only map the
# canonical rail LABEL to the TARGET-WIDTH net names the width model keyed. Names are
# DATA (matched by normalized string), never a per-circuit / per-PDK branch.
DEFAULT_VDD_ALIASES: FrozenSet[str] = frozenset({
    "vdd", "vcc", "vpwr", "vdda", "vddd", "avdd", "dvdd", "vddio",
    "vdd18", "vdd33", "vbat", "vpb",
})
DEFAULT_GND_ALIASES: FrozenSet[str] = frozenset({
    "gnd", "vss", "vsub", "vgnd", "vssa", "vssd", "avss", "dvss",
    "gnda", "gndd", "vnb",
})


def _norm(net: str) -> str:
    """Same net-name normalization the width model uses (strip trailing '!', lower)."""
    return str(net).strip().rstrip("!").lower()


@dataclass
class RailWidthVerdict:
    """The width-fidelity verdict for ONE tap-seeded power rail.

    ``rail`` is the tap-seeded label (``"VDD"`` / ``"GND"``); ``status`` is one of the
    five states; ``emitted_um`` is the widest qualified metal run measured on the rail
    component; ``target_um`` is the computed/declared width it must meet; ``matched_nets``
    are the target-map net names that resolved to this rail."""

    rail: str
    status: str
    emitted_um: Optional[float]
    target_um: Optional[float]
    matched_nets: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass
class WidthFidelityReport:
    """Result of :func:`check_power_rail_width` on one emitted GDS.

    ``status`` is the aggregate five-state verdict; ``rails`` is the per-rail detail;
    ``reason`` explains an aggregate ``unmeasurable``/``unsupported``."""

    status: str = "unmeasurable"
    rails: List[RailWidthVerdict] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> List[RailWidthVerdict]:
        return [r for r in self.rails if r.status == "fail"]

    def rail(self, name: str) -> Optional[RailWidthVerdict]:
        key = _norm(name)
        for r in self.rails:
            if _norm(r.rail) == key:
                return r
        return None


def _import_structure_metrics():
    """Import the standing structural instrument, adding repo root to sys.path if
    ota_dev is not yet importable (the checker never edits or re-implements it)."""
    try:
        from ota_dev.analysis import structure_metrics as SM  # type: ignore
        return SM
    except Exception:
        import pathlib
        import sys

        repo = pathlib.Path(__file__).resolve().parents[3]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from ota_dev.analysis import structure_metrics as SM  # type: ignore
            return SM
        except Exception:
            return None


def _resolve_target(
    rail_label: str,
    targets_norm: Dict[str, float],
    vdd_aliases: FrozenSet[str],
    gnd_aliases: FrozenSet[str],
) -> tuple:
    """(target_um, matched_net_names) for a tap-seeded rail label.

    Direct normalized-name match first (``"VDD"`` -> target net ``"vdd"``); else the
    rail's alias group, taking the MAX matched target (the physical rail must satisfy
    the widest of the supply nets it carries). ``(None, [])`` when nothing matches."""
    key = _norm(rail_label)
    if key in targets_norm:
        return targets_norm[key], [key]
    group = vdd_aliases if key in vdd_aliases or key == "vdd" else (
        gnd_aliases if key in gnd_aliases or key == "gnd" else frozenset())
    matched = {n: w for n, w in targets_norm.items() if n in group}
    if matched:
        return max(matched.values()), sorted(matched)
    return None, []


def _aggregate(statuses: Iterable[str]) -> str:
    s = list(statuses)
    if not s:
        return "unmeasurable"
    if "fail" in s:
        return "fail"
    if "pass" in s:
        return "pass"          # >= 1 real pass, 0 fails
    if "advisory" in s:
        return "advisory"
    if "unsupported" in s:
        return "unsupported"
    return "unmeasurable"


def check_power_rail_width(
    gds_path: str,
    target_widths_um: Dict[str, float],
    *,
    pdk_min_um: Optional[float] = None,
    tolerance_um: float = 0.05,
    advisory_over_factor: float = 0.0,
    vdd_aliases: FrozenSet[str] = DEFAULT_VDD_ALIASES,
    gnd_aliases: FrozenSet[str] = DEFAULT_GND_ALIASES,
    probe: Any = None,
) -> WidthFidelityReport:
    """Measure the power-rail width-fidelity obligation on ``gds_path``.

    ``target_widths_um`` is the ``{net_name: width_um}`` map computed upstream -- the
    output of :func:`power_target_widths` / :func:`derive_width_intent` (declared
    ``@wire_width`` overrides folded in, power rails at the 4.0um floor). The checker
    identifies each VDD / GND conductor from the emitted GDS via the tap-seeded
    :func:`ota_dev.analysis.structure_metrics.power_structure` and compares its widest
    qualified metal run to the target.

    ``pdk_min_um`` (optional): when a rail's target is <= this floor no supra-min
    widening was requested, so the rail is reported ``advisory`` (inert; covered by the
    metal-min-width checker) rather than pass/fail. ``advisory_over_factor`` (optional,
    default 0 = off): when > 0, a rail meeting its target but wider than
    ``advisory_over_factor * target`` is flagged ``advisory`` (anomalously over-wide --
    likely a merged device/tap plate, not a drawn strap) instead of ``pass``.

    Pure function of the emitted GDS + the supplied targets; independent of the router.
    """
    SM = _import_structure_metrics()
    if SM is None:
        return WidthFidelityReport(
            status="unmeasurable",
            reason="ota_dev.analysis.structure_metrics unavailable (klayout / import)",
        )

    if probe is None:
        try:
            # Netlist-independent: the strap identification is tap-seeded geometry, so
            # no companion LVS extraction is read (a pure emitted-GDS measurement).
            probe = SM.LayoutProbe(str(gds_path), auto_netlist=False)
        except Exception as exc:  # pragma: no cover - klayout always present in CI
            return WidthFidelityReport(
                status="unmeasurable", reason=f"load {gds_path}: {exc}")

    ps = SM.power_structure(probe)
    if ps.get("status") != "measured":
        return WidthFidelityReport(
            status="unmeasurable",
            reason=str(ps.get("reason", "no tap-seeded power component")),
        )

    per_rail: Dict[str, float] = dict(ps.get("strap_width_per_rail_um", {}) or {})
    targets_norm = {_norm(k): float(v) for k, v in (target_widths_um or {}).items()}

    rails: List[RailWidthVerdict] = []
    for rail_label in sorted(per_rail):
        emitted = per_rail.get(rail_label)
        target, matched = _resolve_target(
            rail_label, targets_norm, vdd_aliases, gnd_aliases)

        if target is None:
            rails.append(RailWidthVerdict(
                rail=rail_label, status="unsupported", emitted_um=emitted,
                target_um=None, matched_nets=[],
                reason="no computed/declared width supplied for this rail"))
            continue
        if pdk_min_um is not None and target <= float(pdk_min_um) + 1e-9:
            rails.append(RailWidthVerdict(
                rail=rail_label, status="advisory", emitted_um=emitted,
                target_um=target, matched_nets=matched,
                reason=(f"target {target:g}um <= PDK min {float(pdk_min_um):g}um: "
                        "no supra-min widening requested (covered by metal-min-width)")))
            continue
        if emitted is None:
            rails.append(RailWidthVerdict(
                rail=rail_label, status="unmeasurable", emitted_um=None,
                target_um=target, matched_nets=matched,
                reason="rail component present but strap width could not be measured"))
            continue

        if emitted + tolerance_um < target:
            rails.append(RailWidthVerdict(
                rail=rail_label, status="fail", emitted_um=emitted,
                target_um=target, matched_nets=matched,
                reason=(f"emitted {emitted:g}um < target {target:g}um "
                        f"(short by {target - emitted:g}um): width intent not materialized")))
            continue
        if advisory_over_factor > 0 and emitted > advisory_over_factor * target:
            rails.append(RailWidthVerdict(
                rail=rail_label, status="advisory", emitted_um=emitted,
                target_um=target, matched_nets=matched,
                reason=(f"emitted {emitted:g}um >> target {target:g}um "
                        f"(> {advisory_over_factor:g}x): over-wide, likely merged plate")))
            continue
        rails.append(RailWidthVerdict(
            rail=rail_label, status="pass", emitted_um=emitted,
            target_um=target, matched_nets=matched,
            reason=f"emitted {emitted:g}um >= target {target:g}um"))

    report = WidthFidelityReport(rails=rails)
    report.status = _aggregate(r.status for r in rails)
    if not rails:
        report.reason = "no VDD/GND rail identified on the emitted GDS"
    return report


def power_target_widths(
    intent: Any,
    tech: Any,
    *,
    layer: str = "metal1",
    policy: Any = None,
    power_bases: Any = None,
    declared: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """The ``{net_name: width_um}`` target map, from the SAME leakage-free width model
    the router lowers -- :func:`clay.layout.route.width_model.derive_width_intent`.

    A thin adapter that unwraps the :class:`~clay.layout.route.intent.Declared` values to
    plain microns so the checker compares against exactly the widths the router was told
    to draw (power rails at ``WidthPolicy.power_min_um``, declared ``@wire_width`` folded
    in). Passing this map to :func:`check_power_rail_width` closes the intent->evidence
    loop without the checker touching the router."""
    from clay.layout.route.width_model import (
        DEFAULT_POWER_BASES,
        WidthPolicy,
        derive_width_intent,
    )

    widths = derive_width_intent(
        intent, tech, layer=layer,
        policy=policy or WidthPolicy(),
        power_bases=power_bases if power_bases is not None else DEFAULT_POWER_BASES,
        declared=declared,
    )
    return {net: float(d.value) for net, d in widths.items()}


__all__ = [
    "RailWidthVerdict",
    "WidthFidelityReport",
    "check_power_rail_width",
    "power_target_widths",
    "DEFAULT_VDD_ALIASES",
    "DEFAULT_GND_ALIASES",
]
