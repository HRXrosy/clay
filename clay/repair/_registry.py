"""
Repair-operator registry.

Maps a ``(functional role_kind, target_attr)`` pair to the legal repair
knob(s) that may close a violation on that pair, plus the repair *class*
(sizing / bias / compensation / topology_preserving / topology_changing /
layout_aware) per the theory doc's Repair Algebra table.

Seeded from the "Functional Role Semantic Signature" table in
paper/clay-tcad/clay_language_completeness_theory.html — the "Legal Repair
Knobs" column, intersected with the per-kind canonical `target_attr` from
:mod:`clay.invariants_modules._violation` (``_TARGET_ATTR_BY_KIND``).

A repair class of ``topology_changing`` is treated as UNSUPPORTED for now
(it would alter ports/role identity, which the current algebra does not
permit); the registry still records it so the runtime can report
REPAIR_UNSUPPORTED rather than NO_REPAIR_OPERATOR.

This module imports the violation dataclass read-only; it does not edit
any existing invariant module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Read-only import of the canonical role -> target attribute map so the
# registry stays a single source of truth with the invariant runtime.
from clay.invariants_modules._violation import _TARGET_ATTR_BY_KIND


class RepairClass(Enum):
    """Repair operator class taxonomy (theory doc, Repair Algebra table).

    ``topology_changing`` is registered but deferred — see
    :data:`_DEFERRED_CLASSES`.
    """

    SIZING = "sizing"
    BIAS = "bias"
    COMPENSATION = "compensation"
    TOPOLOGY_PRESERVING = "topology_preserving"
    TOPOLOGY_CHANGING = "topology_changing"
    LAYOUT_AWARE = "layout_aware"


# Repair classes whose preconditions the current algebra cannot honour
# without breaking the preserved-obligation contract (ports / role
# identity). A violation whose ONLY operator is in this set resolves to
# RepairStatus.REPAIR_UNSUPPORTED.
_DEFERRED_CLASSES: frozenset[RepairClass] = frozenset({
    RepairClass.TOPOLOGY_CHANGING,
})


@dataclass(frozen=True)
class RepairOperator:
    """One registered repair operator for a (role_kind, target_attr) pair.

    Fields:
      * `role_kind`     — functional role keyword (matches
                          ``_KNOWN_KINDS``).
      * `target_attr`   — canonical target attribute the operator closes.
      * `knobs`         — ordered tuple of legal tunable knob names. The
                          first knob is the *primary* knob the proposer
                          sizes a Newton step on; the rest are alternative
                          knobs the same operator may touch.
      * `repair_class`  — the operator's :class:`RepairClass`.
      * `monotone_sign` — +1 if increasing the primary knob raises the
                          measured quantity toward its target, -1 if
                          increasing the knob lowers it. Used by the
                          proposer to translate a "need more / need less"
                          signal into a knob-increase / knob-decrease move
                          when an explicit gradient is unavailable but the
                          local response is known to be monotone.
    """

    role_kind: str
    target_attr: str
    knobs: tuple[str, ...]
    repair_class: RepairClass
    monotone_sign: int = 1

    @property
    def primary_knob(self) -> str:
        return self.knobs[0]

    @property
    def is_supported(self) -> bool:
        """False iff this operator's class is currently deferred."""
        return self.repair_class not in _DEFERRED_CLASSES


# ─── Registry seed ────────────────────────────────────────────────
#
# Per the Role Semantic Signature table's "Legal Repair Knobs" column.
# The `target_attr` for each kind is pulled from `_TARGET_ATTR_BY_KIND`
# so the keys can never drift out of sync with the invariant runtime.
#
# monotone_sign convention: the primary knob's sign w.r.t. the canonical
# target measure. e.g. for input_transconductor, the primary knob is W
# and gm rises with W, so +1; for dominant_pole_cap the primary knob is
# Cc and the pole frequency FALLS as Cc rises, so -1.


def _attr(kind: str) -> str:
    """Canonical target attribute for a role kind (read-only lookup)."""
    return _TARGET_ATTR_BY_KIND[kind]


_OPERATOR_SEED: tuple[RepairOperator, ...] = (
    # input_transconductor -> {W, L, M, gm_id, tail_current, degeneration}
    RepairOperator(
        role_kind="input_transconductor",
        target_attr=_attr("input_transconductor"),  # gm_target
        knobs=("W", "L", "M", "gm_id", "tail_current", "degeneration"),
        repair_class=RepairClass.SIZING,
        monotone_sign=+1,  # gm rises with W
    ),
    # current_mirror / tail_current_source ->
    #   {mirror_ratio, device_size, bias_current, cascode}
    RepairOperator(
        role_kind="current_mirror",
        target_attr=_attr("current_mirror"),  # ref_current
        knobs=("mirror_ratio", "device_size", "bias_current", "cascode"),
        repair_class=RepairClass.BIAS,
        monotone_sign=+1,  # branch current rises with mirror ratio
    ),
    RepairOperator(
        role_kind="tail_current_source",
        target_attr=_attr("tail_current_source"),  # iref
        knobs=("mirror_ratio", "device_size", "bias_current", "cascode"),
        repair_class=RepairClass.BIAS,
        monotone_sign=+1,
    ),
    # dominant_pole_cap / compensation_zero ->
    #   {Cc, Rz, comp_node, stage_gm}
    RepairOperator(
        role_kind="dominant_pole_cap",
        target_attr=_attr("dominant_pole_cap"),  # pole_freq_target
        knobs=("Cc", "Rz", "comp_node", "stage_gm"),
        repair_class=RepairClass.COMPENSATION,
        monotone_sign=-1,  # dominant pole freq FALLS as Cc rises
    ),
    RepairOperator(
        role_kind="compensation_zero",
        target_attr=_attr("compensation_zero"),  # zero_freq_target
        knobs=("Rz", "Cc", "comp_node", "stage_gm"),
        repair_class=RepairClass.COMPENSATION,
        monotone_sign=-1,  # zero freq FALLS as Rz rises (z = 1/(Rz*Cc))
    ),
    # output_buffer / dropout_pass_device ->
    #   {output_width, bias, classAB_bias}
    RepairOperator(
        role_kind="output_buffer",
        target_attr=_attr("output_buffer"),  # gain_target
        knobs=("output_width", "bias", "classAB_bias"),
        repair_class=RepairClass.SIZING,
        monotone_sign=+1,
    ),
    RepairOperator(
        role_kind="dropout_pass_device",
        target_attr=_attr("dropout_pass_device"),  # dropout_voltage_target
        knobs=("output_width", "bias", "classAB_bias"),
        repair_class=RepairClass.SIZING,
        monotone_sign=-1,  # wider pass device LOWERS dropout
    ),
    # ldo_feedback_divider / bandgap_core / ptat_branch / pll_loop_filter
    #   -> {resistor_ratio, cap_ratio, trim, bias, loop_RC}
    RepairOperator(
        role_kind="ldo_feedback_divider",
        target_attr=_attr("ldo_feedback_divider"),  # divide_ratio_target
        knobs=("resistor_ratio", "cap_ratio", "trim", "bias", "loop_RC"),
        repair_class=RepairClass.SIZING,
        monotone_sign=+1,
    ),
    RepairOperator(
        role_kind="bandgap_core",
        target_attr=_attr("bandgap_core"),  # vref_target
        knobs=("resistor_ratio", "cap_ratio", "trim", "bias", "loop_RC"),
        repair_class=RepairClass.BIAS,
        monotone_sign=+1,
    ),
    RepairOperator(
        role_kind="ptat_branch",
        target_attr=_attr("ptat_branch"),  # temp_coefficient_target
        knobs=("resistor_ratio", "cap_ratio", "trim", "bias", "loop_RC"),
        repair_class=RepairClass.BIAS,
        monotone_sign=+1,
    ),
    RepairOperator(
        role_kind="pll_loop_filter",
        target_attr=_attr("pll_loop_filter"),  # loop_bandwidth_target
        knobs=("loop_RC", "resistor_ratio", "cap_ratio", "trim", "bias"),
        repair_class=RepairClass.COMPENSATION,
        monotone_sign=-1,  # larger loop RC product lowers loop bandwidth
    ),
)


@dataclass
class RepairRegistry:
    """Lookup container mapping (role_kind, target_attr) -> RepairOperator.

    A second, looser index keyed by ``role_kind`` alone lets the proposer
    recover an operator when a violation carries the role's canonical
    target_attr (the common case). Construct the default registry via
    :func:`default_registry`.
    """

    _by_pair: dict[tuple[str, str], RepairOperator] = field(default_factory=dict)
    _by_kind: dict[str, RepairOperator] = field(default_factory=dict)

    def register(self, op: RepairOperator) -> None:
        self._by_pair[(op.role_kind, op.target_attr)] = op
        # First registration wins for the kind-only index; every seeded
        # kind has exactly one canonical target_attr so there is no clash.
        self._by_kind.setdefault(op.role_kind, op)

    def lookup(
        self, role_kind: str, target_attr: Optional[str],
    ) -> Optional[RepairOperator]:
        """Return the operator for the pair, or None if unregistered.

        Falls back to the kind-only index when the exact
        ``(role_kind, target_attr)`` pair is not present but the role kind
        IS known — this covers a violation that names a non-canonical
        sub-attribute (e.g. ``gm_secondary`` on a dual-feedforward role)
        while still resolving to the kind's primary operator.
        """
        if target_attr is not None:
            hit = self._by_pair.get((role_kind, target_attr))
            if hit is not None:
                return hit
        return self._by_kind.get(role_kind)

    def has_kind(self, role_kind: str) -> bool:
        return role_kind in self._by_kind

    def kinds(self) -> frozenset[str]:
        return frozenset(self._by_kind.keys())

    def operators(self) -> tuple[RepairOperator, ...]:
        return tuple(self._by_pair.values())


def default_registry() -> RepairRegistry:
    """Build the registry seeded from the theory doc's signature table."""
    reg = RepairRegistry()
    for op in _OPERATOR_SEED:
        reg.register(op)
    return reg
