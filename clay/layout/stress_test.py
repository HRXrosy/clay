"""Stress-test harness for cell template factories.

Runs every cell template factory with edge-case parameters (minimum finger,
maximum finger, minimum w/l, maximum w/l) and validates the resulting
AnnotatedCell via spatial_validate.validate.

Used to catch regressions where a factory throws on boundary inputs or
produces a cell that fails spatial validation under stress sizing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .cell_templates import (
    build_diff_pair,
    build_current_mirror,
    build_cascoded_current_mirror,
    build_5t_ota,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
    build_bandgap_reference,
)
from .spatial_validate import validate


@dataclass
class StressTestResult:
    template_name: str
    case_label: str
    succeeded: bool
    validation_clean: bool
    error_message: Optional[str] = None


# Edge-case parameter values (used for case labels across all templates).
_MIN_FINGER = 1
_MAX_FINGER = 32
_MIN_W_UM = 0.5
_MIN_L_UM = 0.18
_MAX_W_UM = 16.0
_MAX_L_UM = 5.0

# 4 case labels per template; current_mirror swaps max_w_l for asymmetric to
# exercise its dual-finger-count parameter and assert validator catches the
# MatchingFingerCountMismatch path.
_DEFAULT_CASES = ("min_finger", "max_finger", "min_w_l", "max_w_l")
_CURRENT_MIRROR_CASES = ("min_finger", "max_finger", "min_w_l", "asymmetric")


def _case_kwargs(template_name: str, case_label: str, w_um: float, l_um: float) -> Dict:
    """Translate (template, case) into the kwargs that template factory accepts.

    Each template factory has slightly different parameter names — this is the
    one place we centralise the mapping so the rest of the harness stays generic.
    """
    if template_name == "diff_pair":
        if case_label == "min_finger":
            return {"name": "stress_dp", "w_um": w_um, "l_um": l_um, "finger_count": _MIN_FINGER}
        if case_label == "max_finger":
            return {"name": "stress_dp", "w_um": w_um, "l_um": l_um, "finger_count": _MAX_FINGER}
        if case_label == "min_w_l":
            return {"name": "stress_dp", "w_um": _MIN_W_UM, "l_um": _MIN_L_UM}
        if case_label == "max_w_l":
            return {"name": "stress_dp", "w_um": _MAX_W_UM, "l_um": _MAX_L_UM}

    if template_name == "current_mirror":
        if case_label == "min_finger":
            return {
                "name": "stress_cm",
                "w_um": w_um, "l_um": l_um,
                "finger_count_ref": _MIN_FINGER,
                "finger_count_out": _MIN_FINGER,
            }
        if case_label == "max_finger":
            return {
                "name": "stress_cm",
                "w_um": w_um, "l_um": l_um,
                "finger_count_ref": _MAX_FINGER,
                "finger_count_out": _MAX_FINGER,
            }
        if case_label == "min_w_l":
            return {"name": "stress_cm", "w_um": _MIN_W_UM, "l_um": _MIN_L_UM}
        if case_label == "asymmetric":
            # ref != out finger count — matching annotation will flag
            # MatchingFingerCountMismatch via spatial_validate.
            return {
                "name": "stress_cm",
                "w_um": w_um, "l_um": l_um,
                "finger_count_ref": 4,
                "finger_count_out": 8,
            }

    if template_name == "cascoded_mirror":
        common = {
            "name": "stress_ccm",
            "w_um_low": w_um, "l_um_low": l_um,
            "w_um_cas": w_um, "l_um_cas": l_um,
        }
        if case_label == "min_finger":
            return {**common, "finger_count_low": _MIN_FINGER, "finger_count_cas": _MIN_FINGER}
        if case_label == "max_finger":
            return {**common, "finger_count_low": _MAX_FINGER, "finger_count_cas": _MAX_FINGER}
        if case_label == "min_w_l":
            return {
                "name": "stress_ccm",
                "w_um_low": _MIN_W_UM, "l_um_low": _MIN_L_UM,
                "w_um_cas": _MIN_W_UM, "l_um_cas": _MIN_L_UM,
            }
        if case_label == "max_w_l":
            return {
                "name": "stress_ccm",
                "w_um_low": _MAX_W_UM, "l_um_low": _MAX_L_UM,
                "w_um_cas": _MAX_W_UM, "l_um_cas": _MAX_L_UM,
            }

    if template_name == "5t_ota":
        if case_label == "min_finger":
            return {
                "name": "stress_5t", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MIN_FINGER,
                "load_finger_count": _MIN_FINGER,
                "tail_finger_count": _MIN_FINGER,
            }
        if case_label == "max_finger":
            return {
                "name": "stress_5t", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MAX_FINGER,
                "load_finger_count": _MAX_FINGER,
                "tail_finger_count": _MAX_FINGER,
            }
        if case_label == "min_w_l":
            return {
                "name": "stress_5t", "input_w_um": _MIN_W_UM, "input_l_um": _MIN_L_UM,
                "load_w_um": _MIN_W_UM, "load_l_um": _MIN_L_UM,
                "tail_w_um": _MIN_W_UM, "tail_l_um": _MIN_L_UM,
            }
        if case_label == "max_w_l":
            return {
                "name": "stress_5t", "input_w_um": _MAX_W_UM, "input_l_um": _MAX_L_UM,
                "load_w_um": _MAX_W_UM, "load_l_um": _MAX_L_UM,
                "tail_w_um": _MAX_W_UM, "tail_l_um": _MAX_L_UM,
            }

    if template_name == "folded_cascode":
        if case_label == "min_finger":
            return {
                "name": "stress_fc", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MIN_FINGER,
                "n_cascode_finger_count": _MIN_FINGER,
                "n_bias_finger_count": _MIN_FINGER,
                "p_cascode_finger_count": _MIN_FINGER,
                "p_bias_finger_count": _MIN_FINGER,
                "tail_finger_count": _MIN_FINGER,
            }
        if case_label == "max_finger":
            return {
                "name": "stress_fc", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MAX_FINGER,
                "n_cascode_finger_count": _MAX_FINGER,
                "n_bias_finger_count": _MAX_FINGER,
                "p_cascode_finger_count": _MAX_FINGER,
                "p_bias_finger_count": _MAX_FINGER,
                "tail_finger_count": _MAX_FINGER,
            }
        if case_label == "min_w_l":
            return {
                "name": "stress_fc",
                "input_w_um": _MIN_W_UM, "input_l_um": _MIN_L_UM,
                "n_cascode_w_um": _MIN_W_UM, "n_cascode_l_um": _MIN_L_UM,
                "n_bias_w_um": _MIN_W_UM, "n_bias_l_um": _MIN_L_UM,
                "p_cascode_w_um": _MIN_W_UM, "p_cascode_l_um": _MIN_L_UM,
                "p_bias_w_um": _MIN_W_UM, "p_bias_l_um": _MIN_L_UM,
                "tail_w_um": _MIN_W_UM, "tail_l_um": _MIN_L_UM,
            }
        if case_label == "max_w_l":
            return {
                "name": "stress_fc",
                "input_w_um": _MAX_W_UM, "input_l_um": _MAX_L_UM,
                "n_cascode_w_um": _MAX_W_UM, "n_cascode_l_um": _MAX_L_UM,
                "n_bias_w_um": _MAX_W_UM, "n_bias_l_um": _MAX_L_UM,
                "p_cascode_w_um": _MAX_W_UM, "p_cascode_l_um": _MAX_L_UM,
                "p_bias_w_um": _MAX_W_UM, "p_bias_l_um": _MAX_L_UM,
                "tail_w_um": _MAX_W_UM, "tail_l_um": _MAX_L_UM,
            }

    if template_name == "telescopic_cascode":
        if case_label == "min_finger":
            return {
                "name": "stress_tc", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MIN_FINGER,
                "cascode_finger_count": _MIN_FINGER,
                "pmos_load_finger_count": _MIN_FINGER,
                "tail_finger_count": _MIN_FINGER,
            }
        if case_label == "max_finger":
            return {
                "name": "stress_tc", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MAX_FINGER,
                "cascode_finger_count": _MAX_FINGER,
                "pmos_load_finger_count": _MAX_FINGER,
                "tail_finger_count": _MAX_FINGER,
            }
        if case_label == "min_w_l":
            return {
                "name": "stress_tc",
                "input_w_um": _MIN_W_UM, "input_l_um": _MIN_L_UM,
                "cascode_w_um": _MIN_W_UM, "cascode_l_um": _MIN_L_UM,
                "pmos_load_w_um": _MIN_W_UM, "pmos_load_l_um": _MIN_L_UM,
                "tail_w_um": _MIN_W_UM, "tail_l_um": _MIN_L_UM,
            }
        if case_label == "max_w_l":
            return {
                "name": "stress_tc",
                "input_w_um": _MAX_W_UM, "input_l_um": _MAX_L_UM,
                "cascode_w_um": _MAX_W_UM, "cascode_l_um": _MAX_L_UM,
                "pmos_load_w_um": _MAX_W_UM, "pmos_load_l_um": _MAX_L_UM,
                "tail_w_um": _MAX_W_UM, "tail_l_um": _MAX_L_UM,
            }

    if template_name == "two_stage_miller":
        if case_label == "min_finger":
            return {
                "name": "stress_tsm", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MIN_FINGER,
                "load_finger_count": _MIN_FINGER,
                "tail_finger_count": _MIN_FINGER,
                "cs_finger_count": _MIN_FINGER,
                "cs_load_finger_count": _MIN_FINGER,
                "miller_cap_finger_count": _MIN_FINGER,
            }
        if case_label == "max_finger":
            return {
                "name": "stress_tsm", "input_w_um": w_um, "input_l_um": l_um,
                "input_finger_count": _MAX_FINGER,
                "load_finger_count": _MAX_FINGER,
                "tail_finger_count": _MAX_FINGER,
                "cs_finger_count": _MAX_FINGER,
                "cs_load_finger_count": _MAX_FINGER,
                "miller_cap_finger_count": _MAX_FINGER,
            }
        if case_label == "min_w_l":
            return {
                "name": "stress_tsm",
                "input_w_um": _MIN_W_UM, "input_l_um": _MIN_L_UM,
                "load_w_um": _MIN_W_UM, "load_l_um": _MIN_L_UM,
                "tail_w_um": _MIN_W_UM, "tail_l_um": _MIN_L_UM,
                "cs_w_um": _MIN_W_UM, "cs_l_um": _MIN_L_UM,
                "cs_load_w_um": _MIN_W_UM, "cs_load_l_um": _MIN_L_UM,
            }
        if case_label == "max_w_l":
            return {
                "name": "stress_tsm",
                "input_w_um": _MAX_W_UM, "input_l_um": _MAX_L_UM,
                "load_w_um": _MAX_W_UM, "load_l_um": _MAX_L_UM,
                "tail_w_um": _MAX_W_UM, "tail_l_um": _MAX_L_UM,
                "cs_w_um": _MAX_W_UM, "cs_l_um": _MAX_L_UM,
                "cs_load_w_um": _MAX_W_UM, "cs_load_l_um": _MAX_L_UM,
            }

    if template_name == "bandgap":
        if case_label == "min_finger":
            return {
                "name": "stress_bg",
                "q1_emitter_units": _MIN_FINGER,
                "q2_emitter_units": _MIN_FINGER,
                "resistor_r1_units": _MIN_FINGER,
                "resistor_r2_units": _MIN_FINGER,
                "ota_input_finger_count": _MIN_FINGER,
                "ota_load_finger_count": _MIN_FINGER,
                "ota_tail_finger_count": _MIN_FINGER,
            }
        if case_label == "max_finger":
            return {
                "name": "stress_bg",
                "q1_emitter_units": _MAX_FINGER,
                "q2_emitter_units": _MAX_FINGER,
                "resistor_r1_units": _MAX_FINGER,
                "resistor_r2_units": _MAX_FINGER,
                "ota_input_finger_count": _MAX_FINGER,
                "ota_load_finger_count": _MAX_FINGER,
                "ota_tail_finger_count": _MAX_FINGER,
            }
        if case_label == "min_w_l":
            # Force matched finger / segment counts so Matching annotations on
            # Q1/Q2 and R1/R2 don't spuriously fail validation; we are
            # exercising w/l boundaries, not finger ratios.
            return {
                "name": "stress_bg",
                "bjt_unit_emitter_um2": _MIN_W_UM, "bjt_unit_l_um": _MIN_L_UM,
                "resistor_unit_w_um": _MIN_W_UM, "resistor_unit_l_um": _MIN_L_UM,
                "ota_input_w_um": _MIN_W_UM, "ota_input_l_um": _MIN_L_UM,
                "ota_load_w_um": _MIN_W_UM, "ota_load_l_um": _MIN_L_UM,
                "ota_tail_w_um": _MIN_W_UM, "ota_tail_l_um": _MIN_L_UM,
                "q1_emitter_units": 1, "q2_emitter_units": 1,
                "resistor_r1_units": 4, "resistor_r2_units": 4,
            }
        if case_label == "max_w_l":
            return {
                "name": "stress_bg",
                "bjt_unit_emitter_um2": _MAX_W_UM, "bjt_unit_l_um": _MAX_L_UM,
                "resistor_unit_w_um": _MAX_W_UM, "resistor_unit_l_um": _MAX_L_UM,
                "ota_input_w_um": _MAX_W_UM, "ota_input_l_um": _MAX_L_UM,
                "ota_load_w_um": _MAX_W_UM, "ota_load_l_um": _MAX_L_UM,
                "ota_tail_w_um": _MAX_W_UM, "ota_tail_l_um": _MAX_L_UM,
                "q1_emitter_units": 1, "q2_emitter_units": 1,
                "resistor_r1_units": 4, "resistor_r2_units": 4,
            }

    # Unrecognised template / case combination — signal to caller.
    raise KeyError(f"No stress case mapping for template={template_name}, case={case_label}")


_TEMPLATE_FACTORIES: Dict[str, Callable] = {
    "diff_pair": build_diff_pair,
    "current_mirror": build_current_mirror,
    "cascoded_mirror": build_cascoded_current_mirror,
    "5t_ota": build_5t_ota,
    "folded_cascode": build_folded_cascode,
    "telescopic_cascode": build_telescopic_cascode,
    "two_stage_miller": build_two_stage_miller,
    "bandgap": build_bandgap_reference,
}


def _cases_for(template_name: str) -> Tuple[str, ...]:
    if template_name == "current_mirror":
        return _CURRENT_MIRROR_CASES
    return _DEFAULT_CASES


def run_stress_test_for(
    template_name: str,
    w_um: float = 4.0,
    l_um: float = 0.5,
) -> List[StressTestResult]:
    """Run 4 edge-case stress cases against a single cell template factory.

    Cases (default): ``min_finger`` (1), ``max_finger`` (32), ``min_w_l`` (0.5,
    0.18), ``max_w_l`` (16, 5). For ``current_mirror`` the 4th case is
    ``asymmetric`` instead — exercising the dual-finger-count parameter and
    confirming the spatial validator flags MatchingFingerCountMismatch.

    Args:
        template_name: registered template key (e.g. ``"diff_pair"``).
        w_um: baseline width passed to finger-count cases (min/max finger).
        l_um: baseline length passed to finger-count cases.

    Returns:
        Always 4 ``StressTestResult`` entries — one per case. An unknown
        ``template_name`` collapses to a single failed result keyed
        ``case_label="unknown_template"``.
    """
    factory = _TEMPLATE_FACTORIES.get(template_name)
    if factory is None:
        return [StressTestResult(
            template_name=template_name,
            case_label="unknown_template",
            succeeded=False,
            validation_clean=False,
            error_message=f"Unknown template: {template_name}",
        )]

    results: List[StressTestResult] = []
    for case_label in _cases_for(template_name):
        try:
            kwargs = _case_kwargs(template_name, case_label, w_um, l_um)
        except KeyError as ke:
            results.append(StressTestResult(
                template_name=template_name,
                case_label=case_label,
                succeeded=False,
                validation_clean=False,
                error_message=f"No case mapping: {ke}",
            ))
            continue

        try:
            cell, _sizing = factory(**kwargs)
        except Exception as exc:  # noqa: BLE001 — harness must catch any factory failure
            results.append(StressTestResult(
                template_name=template_name,
                case_label=case_label,
                succeeded=False,
                validation_clean=False,
                error_message=f"{type(exc).__name__}: {exc}",
            ))
            continue

        try:
            errors = validate(cell)
            validation_clean = len(errors) == 0
            results.append(StressTestResult(
                template_name=template_name,
                case_label=case_label,
                succeeded=True,
                validation_clean=validation_clean,
                error_message=None if validation_clean else (
                    f"{len(errors)} spatial error(s): "
                    + "; ".join(type(e).__name__ for e in errors[:3])
                ),
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(StressTestResult(
                template_name=template_name,
                case_label=case_label,
                succeeded=False,
                validation_clean=False,
                error_message=f"validate raised {type(exc).__name__}: {exc}",
            ))

    return results


def run_stress_test_all() -> List[StressTestResult]:
    """Run stress test against every registered cell template factory.

    Returns 4 results per template * 8 templates = 32 total ``StressTestResult``.
    """
    all_results: List[StressTestResult] = []
    for template_name in _TEMPLATE_FACTORIES:
        all_results.extend(run_stress_test_for(template_name))
    return all_results


def summarize_stress(results: List[StressTestResult]) -> str:
    """Return a one-line summary: ``N total / M succeeded / K validation clean``."""
    total = len(results)
    succeeded = sum(1 for r in results if r.succeeded)
    clean = sum(1 for r in results if r.validation_clean)
    return f"{total} total / {succeeded} succeeded / {clean} validation clean"
