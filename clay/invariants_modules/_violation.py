"""
InvariantViolation dataclass + role-kind constants (v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 1-241. Carries:

  * `DEFAULT_TOLERANCE`            — ±20% Round-62 contract tolerance.
  * `_KNOWN_KINDS`                 — frozenset of supported role kinds.
  * `_TARGET_ATTR_BY_KIND`         — kind → canonical target attribute.
  * `InvariantViolation` dataclass — the structured-error surface every
                                     check function emits.

The shim `clay/invariants.py` re-exports every symbol here so the
byte-exact import contract (`from clay.invariants import
DEFAULT_TOLERANCE, InvariantViolation, _KNOWN_KINDS, _TARGET_ATTR_BY_KIND`)
keeps working.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# ─── Default tolerance band ────────────────────────────────────

DEFAULT_TOLERANCE: float = 0.20   # ±20% — matches the Round 62 contract


# The functional-role keywords the runtime knows how to check.
# Used by the dispatch table at the bottom of the file AND for surfacing
# a precise error if a CellDef carries an unknown role kind through to
# the runtime (defense in depth — the parser also gates this).
#
# v0.0.90 (Round 66): extended from 6 to 12 kinds. Second batch keywords
# (cascode_helper, pole_tracking_cap, dual_feedforward_pair,
# loop_filter_pole, current_mirror, diode_load) are registered here so
# they pass the "unknown kind" gate in `check_functional_invariants`.
# Their check functions are PLACEHOLDERS — they return an empty
# `skipped` informational violation rather than doing real comparisons.
# Round 67 will replace each placeholder with a real check.
_KNOWN_KINDS: frozenset[str] = frozenset({
    # First batch (v0.0.86)
    "input_transconductor",
    "output_buffer",
    "feedforward_path",
    "dominant_pole_cap",
    "compensation_zero",
    "tail_current_source",
    # Second batch (v0.0.90 Round 66) — promoted to real checks in v0.0.90 Round 67
    "cascode_helper",
    "pole_tracking_cap",
    "dual_feedforward_pair",
    "loop_filter_pole",
    "current_mirror",
    "diode_load",
    # Third batch (v0.0.92 Round 68 Agent G+H) — real checks.
    "common_mode_feedback",
    "sample_hold_switch",
    "regulated_cascode",
    "gain_boost_amp",
    "output_class_AB",
    "dynamic_bias",
    # Fourth batch (v0.0.94 Round 70 Agent C) — placeholder checks for
    # non-amp categories (LDO / Bandgap / PLL). Each of the five new
    # keywords is registered here so the runtime accepts the kind
    # through the "unknown kind" gate; the dispatch table at the bottom
    # of the file routes each to a placeholder that returns a
    # `category="skipped"` informational violation with
    # `extras["deferred"]=True`. Round 71+ will replace each placeholder
    # with a real measurement comparison once the Bandgap / LDO / PLL
    # testbench plumbing lands.
    "bandgap_core",
    "dropout_pass_device",
    "ldo_feedback_divider",
    "ptat_branch",
    "pll_loop_filter",
})


# Per-keyword default target attribute name. The check functions accept
# the role's `attributes` dict and look up the value by this key. A
# missing key gets reported as a `category="skipped"` violation so the
# LLM sees "this role declaration didn't carry the expected attribute"
# rather than a silent pass.
#
# v0.0.90 (Round 66): second batch keywords each declare their canonical
# target attribute name here so future Round 67 lookups have a single
# source of truth. The placeholder checks do NOT consume these yet —
# they return empty violation lists regardless of attribute presence.
_TARGET_ATTR_BY_KIND: dict[str, str] = {
    # First batch (v0.0.86)
    "input_transconductor": "gm_target",
    "output_buffer":        "gain_target",
    "feedforward_path":     "gain",
    "dominant_pole_cap":    "pole_freq_target",
    "compensation_zero":    "zero_freq_target",
    "tail_current_source":  "iref",
    # Second batch (v0.0.90 Round 66) — promoted to real checks Round 67
    "cascode_helper":        "ro_boost_target",
    "pole_tracking_cap":     "target_pole",
    "dual_feedforward_pair": "gm_primary",
    "loop_filter_pole":      "target_loop_bw",
    "current_mirror":        "ref_current",
    "diode_load":            "gm_target",
    # Third batch (v0.0.92 Round 68 Agent G+H) — real checks.
    "common_mode_feedback":  "cmrr_target",
    "sample_hold_switch":    "hold_droop_target",
    "regulated_cascode":     "ro_boost_target",
    "gain_boost_amp":        "boost_gain_target",
    "output_class_AB":       "iq_target",
    "dynamic_bias":          "iref_target",
    # Fourth batch (v0.0.94 Round 70 Agent C → Round 71 Agent B promoted
    # to real). Each canonical target attribute is the single source of
    # truth the real check function below consumes via
    # `role.attributes.get(_TARGET_ATTR_BY_KIND[role.kind])`.
    "bandgap_core":          "vref_target",
    "dropout_pass_device":   "dropout_voltage_target",
    "ldo_feedback_divider":  "divide_ratio_target",
    "ptat_branch":           "temp_coefficient_target",
    "pll_loop_filter":       "loop_bandwidth_target",
}


@dataclass
class InvariantViolation:
    """One functional-role invariant check outcome.

    Categories:
      * `mismatch`        — target declared, simulator returned a value,
                            relative error exceeds tolerance
      * `measure_missing` — simulator did not produce the expected
                            quantity (no NaN propagation; surfaces a
                            structured "I don't have data for this" hint)
      * `skipped`         — the role declaration was incomplete (no
                            target attribute) or the runtime can't yet
                            check this kind (informational pass)

    Every InvariantViolation is JSON-serializable via `asdict()` — the
    `to_json_dict` method returns a plain Python dict so an LLM can
    consume the structured error directly.

    v0.0.90 gradient-augmented retry fields (populated only when a
    mismatch entry is fed through
    :func:`augment_violation_with_gradient` — the v0.0.89 Adjoint
    `gradient_on_functional_target` API resolves `∂(DC gain)/∂(target)`
    at the current OP and the augmentation step turns the violation
    into a precise numerical retry hint):

      * `gradient_value`        — `∂(DC gain)/∂(target_attr)` at the OP.
                                  Non-None only when adjoint succeeded.
      * `suggested_target_value` — Newton-step suggestion for the next
                                  `target_attr` value. None when the
                                  gradient is unavailable / degenerate.
      * `suggested_target_unit`  — Canonical unit string for the target
                                  attribute (e.g. `"A/V"` for gm_target,
                                  `"V/V"` for gain_target, `"Hz"` for
                                  pole_freq_target, `"A"` for iref).
                                  None when the attribute has no
                                  canonical unit.
    """
    role_kind: str
    role_name: str
    target_attr: str
    target_value: Optional[float]
    actual_value: Optional[float]
    relative_error: Optional[float]
    tolerance: float
    category: str
    message: str
    suggested_adjustment: str = ""
    # Free-form `{key: value}` for tooling that wants extra context
    # (e.g. which measure was consulted, what fallback path ran).
    extras: dict[str, Any] = field(default_factory=dict)
    # v0.0.90 gradient-augmented retry signal. Populated lazily by
    # `augment_violation_with_gradient`; default None preserves the
    # v0.0.62 / v0.0.86 contract for callers that never call the
    # augmentation step.
    gradient_value: Optional[float] = None
    suggested_target_value: Optional[float] = None
    suggested_target_unit: Optional[str] = None
    source_obligation_id: Optional[str] = None

    @property
    def passed(self) -> bool:
        """A violation is `passed` only when it's a `skipped` informational
        entry. Mismatches and measure_missing entries always fail."""
        return self.category == "skipped"

    def to_json_dict(self) -> dict[str, Any]:
        """Return a plain Python dict suitable for `json.dumps`.

        The `extras` map is included verbatim. `target_value` /
        `actual_value` / `relative_error` may be None — JSON null is
        the canonical representation. The v0.0.90 gradient-augmented
        fields (`gradient_value`, `suggested_target_value`,
        `suggested_target_unit`) are also JSON-safe: each is either a
        finite float or None."""
        return asdict(self)


class VerificationStatus(Enum):
    """Certification status used by proof-token generation.

    This is stricter than the legacy ``InvariantViolation.passed``
    compatibility property. ``skipped`` remains an informational runtime
    category, but it is not a proof token.
    """

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    PENDING = "pending"
    UNSUPPORTED = "unsupported"
    NO_CONVERGENCE = "no_convergence"
    ERROR = "error"


_CATEGORY_TO_VERIFICATION_STATUS: dict[str, VerificationStatus] = {
    "pass": VerificationStatus.PASS,
    "skipped": VerificationStatus.SKIP,
    "skip": VerificationStatus.SKIP,
    "mismatch": VerificationStatus.FAIL,
    "fail": VerificationStatus.FAIL,
    "measure_missing": VerificationStatus.PENDING,
    "missing_measure": VerificationStatus.PENDING,
    "pending": VerificationStatus.PENDING,
    "unsupported": VerificationStatus.UNSUPPORTED,
    "no_convergence": VerificationStatus.NO_CONVERGENCE,
    "error": VerificationStatus.ERROR,
}


def category_to_verification_status(category: str) -> VerificationStatus:
    """Map legacy violation categories to explicit verification status.

    Unknown categories are treated as ERROR rather than PASS so new or
    malformed runtime states cannot silently certify a design.
    """
    return _CATEGORY_TO_VERIFICATION_STATUS.get(
        str(category),
        VerificationStatus.ERROR,
    )


def contributes_to_proof_token(item: InvariantViolation) -> bool:
    """Return True iff an invariant result can certify an obligation."""
    return category_to_verification_status(item.category) is VerificationStatus.PASS
