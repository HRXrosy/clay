"""Three-layer layout-intent composition: PDK knowledge + category profile + overrides.

The v1 layout-intent contract had a single source: a cell-name -> ``LayoutIntent``
table (``derive_layout_intent("comparator")``). That made every comparator-shaped
cell look identical and forced the metric layer to hardcode comparator-specific
nets (the ``clk`` strobe, ``net0`` / ``net1`` inputs, the ``crossn`` / ``crossp``
regen pair). A REAL design pipeline needs to express that:

  * the SAME category profile (a dynamic comparator) can be instantiated with a
    different strobe net (``clk`` / ``clkp`` / ``strobe``), different input nets,
    different regen-pair labels;
  * a PDK can change the SHIELDING / parasitic-budget defaults without rewriting
    the category (e.g. a fast process tightens kickback tolerance);
  * a designer can OVERRIDE one number on one instance without ripping up the
    profile.

This module is the smallest surface that makes those three layers explicit:

  * :class:`LayoutIntentProfile` -- a category's default budgets + roles, keyed
    by a stable ``profile_id`` (``"dynamic_comparator_v1"``, ``"5t_ota_v1"``).
    Owns the metric drivers a category cares about: which net is the strobe,
    which nets are the dynamic aggressors, which pair is the regen.

  * :func:`compose_intent` -- ``profile + role_binding (instance-level net names)
    + overrides (per-instance budgets / shields)`` -> :class:`LayoutIntent`. The
    role binding maps the profile's abstract role labels (``strobe``,
    ``inputs``, ``regen``, ``output``) to the THIS-instance net names; the
    overrides replace any budget on a ``(kind, pair)`` key the profile already
    declared.

  * ``register_profile`` / ``get_profile`` -- module-level registry. Composed
    intents carry ``profile_id`` + ``source="category-profile"`` on the result so
    the signoff record stores the provenance.

PDK knowledge is the third axis: a ``pdk_defaults`` mapping on the profile
parameterises the default rel-tols by PDK id, so the same category can ship one
default kickback tol for gf180mcu and a tighter one for an aggressive node.
Designer overrides win over PDK defaults, which win over profile defaults.

Composition is REGISTRY-DRIVEN, not hardcoded: every call to
:func:`compose_intent` reaches into the SAME registry the per-cell deriver uses,
so a new category profile is a single ``register_profile`` call away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .intent import (
    LayoutIntent,
    ParasiticBudget,
    Shield,
    _dedupe_budgets_prefer_later,
    _dedupe_shields_prefer_later,
    _normalize_pairs,
)


@dataclass(frozen=True)
class BudgetSpec:
    """A category-level budget TEMPLATE keyed by abstract role labels.

    ``pair_roles`` names two roles (e.g. ``("regen_n", "regen_p")``) the role
    binding will substitute with concrete instance nets at compose time.
    ``aggressor_roles`` (for kickback) names every role whose net should be
    summed as an aggressor; ``clock_role`` (for clk_regen) names the role whose
    net IS the strobe. ``rel_tol`` is the default RELATIVE imbalance budget.

    Composition resolves each role to the binding's concrete net; an unbound
    role suppresses the budget (the category was "I COULD measure clk_regen on
    this cell, but the instance has no strobe role bound" -- safer to omit than
    to invent a default that hides the missing role).
    """
    kind: str
    pair_roles: Tuple[str, str]
    rel_tol: float
    aggressor_roles: Optional[Tuple[str, ...]] = None
    clock_role: Optional[str] = None
    victim_roles: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class ShieldSpec:
    """A category-level shield TEMPLATE (role labels, resolved at compose time)."""
    aggressor_role: str
    victim_role: str
    shield_net: str = "gnd"


@dataclass(frozen=True)
class LayoutIntentProfile:
    """A category profile: default budgets + shields + dynamic-net roles.

    A profile is the SECOND of the three composition layers (PDK / category /
    instance). It carries the budget / shield TEMPLATES in role-label form; the
    instance's ``role_binding`` resolves those roles to concrete net names.

      * ``profile_id`` -- the stable id stored on the composed intent.
      * ``dynamic_roles`` / ``sensitive_roles`` -- which roles classify as
        dynamic aggressors / offset-critical victims (resolved per-instance).
      * ``diff_pair_roles`` / ``mirror_pair_roles`` -- the role-name pairs that
        become diff_pairs / mirror_pairs on the composed intent.
      * ``budgets`` / ``shields`` -- the category's default budget / shield
        templates.
      * ``pdk_defaults`` -- optional ``pdk_id -> {kind: rel_tol}`` map; the
        looked-up tol overrides the profile default when a matching PDK is
        named at compose time.
    """
    profile_id: str
    dynamic_roles: Tuple[str, ...] = ()
    sensitive_roles: Tuple[str, ...] = ()
    diff_pair_roles: Tuple[Tuple[str, str], ...] = ()
    mirror_pair_roles: Tuple[Tuple[str, str], ...] = ()
    budgets: Tuple[BudgetSpec, ...] = ()
    shields: Tuple[ShieldSpec, ...] = ()
    pdk_defaults: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


_PROFILES: Dict[str, LayoutIntentProfile] = {}


def register_profile(profile: LayoutIntentProfile) -> None:
    """Register a category profile under its ``profile_id``.

    A re-registration with the same id replaces the previous entry (so a test
    can override the dynamic-comparator default for one test without leaking).
    """
    _PROFILES[profile.profile_id] = profile


def get_profile(profile_id: str) -> LayoutIntentProfile:
    """Look up a registered profile by id (raises ``KeyError`` if absent)."""
    return _PROFILES[profile_id]


def list_profiles() -> Tuple[str, ...]:
    """All currently-registered profile ids (sorted, stable)."""
    return tuple(sorted(_PROFILES))


def _resolve_role(role: str, binding: Mapping[str, str]) -> Optional[str]:
    """Look up a role label in the binding, returning ``None`` when unbound."""
    return binding.get(role)


def _resolve_role_tuple(
    roles: Optional[Sequence[str]], binding: Mapping[str, str]
) -> Optional[Tuple[str, ...]]:
    """Resolve every role; drop unbound ones; return None when input is None."""
    if roles is None:
        return None
    out: List[str] = []
    for r in roles:
        n = _resolve_role(r, binding)
        if n is not None:
            out.append(n)
    return tuple(out)


def compose_intent(
    profile_id: str,
    cell: str,
    role_binding: Mapping[str, str],
    *,
    overrides: Optional[Sequence[ParasiticBudget]] = None,
    shield_overrides: Optional[Sequence[Shield]] = None,
    pdk: Optional[str] = None,
) -> LayoutIntent:
    """Compose a profile + role binding + (optional) overrides into a LayoutIntent.

    Resolution order, lowest to highest precedence (each layer wins over the
    previous on the same ``(kind, pair)`` / ``(aggressor, victim)`` key):

      1. profile defaults (category-level templates resolved through
         ``role_binding``);
      2. PDK defaults (when ``pdk=`` is supplied and the profile names that PDK);
      3. designer overrides (``overrides`` / ``shield_overrides``).

    Roles bound to ``None`` (absent from ``role_binding``) silently suppress the
    budget / shield: a category profile may declare a clk_regen budget, but an
    instance that has no strobe role bound declares NO clk_regen budget rather
    than synthesising one against an invented net. The instance can still add a
    clk_regen budget through ``overrides`` if appropriate.

    Returns a :class:`LayoutIntent` tagged ``profile_id`` /
    ``source="category-profile"`` so a downstream record can recover provenance.
    """
    profile = get_profile(profile_id)
    pdk_tol = profile.pdk_defaults.get(pdk, {}) if pdk is not None else {}

    # ── Net classes ──────────────────────────────────────────────────────────
    dynamic: List[str] = []
    sensitive: List[str] = []
    for r in profile.dynamic_roles:
        n = _resolve_role(r, role_binding)
        if n is not None and n not in dynamic:
            dynamic.append(n)
    for r in profile.sensitive_roles:
        n = _resolve_role(r, role_binding)
        if n is not None and n not in sensitive:
            sensitive.append(n)

    # ── Diff / mirror pairs ──────────────────────────────────────────────────
    diff_pair_nets: List[Tuple[str, str]] = []
    for ra, rb in profile.diff_pair_roles:
        na, nb = _resolve_role(ra, role_binding), _resolve_role(rb, role_binding)
        if na is not None and nb is not None and na != nb:
            diff_pair_nets.append((na, nb))
    mirror_pair_nets: List[Tuple[str, str]] = []
    for ra, rb in profile.mirror_pair_roles:
        na, nb = _resolve_role(ra, role_binding), _resolve_role(rb, role_binding)
        if na is not None and nb is not None and na != nb:
            mirror_pair_nets.append((na, nb))

    # ── Profile budgets resolved through the binding ─────────────────────────
    profile_budgets: List[ParasiticBudget] = []
    # 1. one matched_cc_balance per resolved diff pair (the universal "keep the
    #    two halves matched" rule), at the PDK / profile tol.
    match_tol = float(pdk_tol.get("matched_cc_balance", _profile_match_tol(profile)))
    for pair in _normalize_pairs(diff_pair_nets):
        profile_budgets.append(
            ParasiticBudget(
                "matched_cc_balance", pair, match_tol,
                source="category-profile",
                profile_id=profile.profile_id,
            )
        )
    # 2. each explicit BudgetSpec, resolved through the binding.
    for spec in profile.budgets:
        ra, rb = spec.pair_roles
        na, nb = _resolve_role(ra, role_binding), _resolve_role(rb, role_binding)
        if na is None or nb is None or na == nb:
            continue
        tol = float(pdk_tol.get(spec.kind, spec.rel_tol))
        aggressors = _resolve_role_tuple(spec.aggressor_roles, role_binding)
        victims = _resolve_role_tuple(spec.victim_roles, role_binding)
        if spec.clock_role:
            clock_net = _resolve_role(spec.clock_role, role_binding)
            if clock_net is None:
                continue
        else:
            clock_net = None
        profile_budgets.append(
            ParasiticBudget(
                spec.kind, (na, nb), tol,
                aggressors=aggressors,
                victim_set=victims,
                clock_net=clock_net,
                source="category-profile",
                profile_id=profile.profile_id,
            )
        )

    # ── Shields resolved through the binding ─────────────────────────────────
    profile_shields: List[Shield] = []
    for sh in profile.shields:
        na = _resolve_role(sh.aggressor_role, role_binding)
        nb = _resolve_role(sh.victim_role, role_binding)
        if na is None or nb is None:
            continue
        profile_shields.append(Shield(na, nb, sh.shield_net))

    # ── Apply designer overrides last (highest precedence) ───────────────────
    all_budgets = profile_budgets + list(overrides or [])
    all_shields = profile_shields + list(shield_overrides or [])

    return LayoutIntent(
        cell=cell,
        dynamic_nets=tuple(dynamic),
        sensitive_nets=tuple(sensitive),
        diff_pairs=_normalize_pairs(diff_pair_nets),
        budgets=_dedupe_budgets_prefer_later(all_budgets),
        shields=_dedupe_shields_prefer_later(all_shields),
        mirror_pairs=_normalize_pairs(mirror_pair_nets),
        profile_id=profile.profile_id,
        source="category-profile",
    )


def _profile_match_tol(profile: LayoutIntentProfile) -> float:
    """Default matched_cc_balance tol declared on the profile, or 0.10."""
    for spec in profile.budgets:
        if spec.kind == "matched_cc_balance":
            return float(spec.rel_tol)
    return 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Built-in profiles (registered at import time)
# ─────────────────────────────────────────────────────────────────────────────

# Dynamic comparator: clocked rail-to-rail block.
#   strobe       -> the clock net (dynamic aggressor)
#   in_p / in_n  -> the differential inputs (sensitive victims, kickback pair)
#   regen_n /
#   regen_p      -> the regenerative latch pair (dynamic AND sensitive)
#   out_p / out_n-> the rail-to-rail output pair (dynamic)
#   inter_n /
#   inter_p      -> the pre-amp pair (sensitive)
DYNAMIC_COMPARATOR_V1 = LayoutIntentProfile(
    profile_id="dynamic_comparator_v1",
    dynamic_roles=("strobe", "regen_n", "regen_p", "out_p", "out_n"),
    sensitive_roles=("in_p", "in_n", "inter_n", "inter_p", "regen_n", "regen_p"),
    diff_pair_roles=(
        ("inter_n", "inter_p"),
        ("regen_n", "regen_p"),
        ("out_p", "out_n"),
        ("in_p", "in_n"),
    ),
    budgets=(
        BudgetSpec(
            kind="kickback",
            pair_roles=("in_p", "in_n"),
            rel_tol=0.15,
            aggressor_roles=("strobe", "regen_n", "regen_p", "out_p", "out_n"),
        ),
        BudgetSpec(
            kind="clk_regen",
            pair_roles=("regen_n", "regen_p"),
            rel_tol=0.10,
            clock_role="strobe",
        ),
        BudgetSpec(
            kind="matched_cc_balance",
            pair_roles=("regen_n", "regen_p"),
            rel_tol=0.10,
        ),
    ),
    shields=(
        ShieldSpec("strobe", "regen_n"),
        ShieldSpec("strobe", "regen_p"),
    ),
)

# 5T-OTA: continuous-time, mirror-loaded.
#   in_p / in_n -> differential inputs (sensitive)
#   out_p       -> diode/reference leg of the mirror load (NOT differential)
#   out_n       -> single-ended output (the mirrored leg)
FIVE_T_OTA_V1 = LayoutIntentProfile(
    profile_id="5t_ota_v1",
    sensitive_roles=("in_p", "in_n"),
    diff_pair_roles=(("in_p", "in_n"),),
    mirror_pair_roles=(("out_p", "out_n"),),
)


register_profile(DYNAMIC_COMPARATOR_V1)
register_profile(FIVE_T_OTA_V1)


__all__ = [
    "BudgetSpec",
    "ShieldSpec",
    "LayoutIntentProfile",
    "register_profile",
    "get_profile",
    "list_profiles",
    "compose_intent",
    "DYNAMIC_COMPARATOR_V1",
    "FIVE_T_OTA_V1",
]
