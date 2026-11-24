"""Layout-side gm/Id lookup table baseline.

This module provides a minimal, dependency-free gm/Id sizing baseline for the
layout subsystem. The built-in table targets sky130 NMOS at L=0.5 um and
V_DS=0.9 V (typical corner) and is intended as a coarse, deterministic
starting point for early sizing passes — it is NOT a substitute for full
characterization sweeps. Callers can supply their own ``List[GmIdEntry]``
to specialize for other technologies, channel lengths, or bias points.

Design notes:
  * Entries are stored sorted by ``gm_over_id`` ascending (low gm/Id ↔
    strong inversion ↔ high overdrive ↔ high current density).
  * ``lookup_by_gm_id`` returns the nearest entry (no interpolation).
  * ``interpolate_id_per_w`` linearly interpolates current density between
    the two surrounding entries, with clamping at both extremes.
  * ``size_w_for_target_id`` inverts the relation W = I_D / J_D where
    J_D = id_per_w_a_per_m is interpolated at the requested gm/Id, and
    returns W in micrometers.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GmIdEntry:
    """A single operating-point sample in the gm/Id lookup table.

    Attributes:
        gm_over_id: Transconductance efficiency gm/I_D in 1/V.
        vgs_minus_vth_v: Overdrive voltage V_GS - V_TH in volts.
        id_per_w_a_per_m: Drain current density per unit width, in A/m
            (i.e. amperes per meter of device width).
    """

    gm_over_id: float
    vgs_minus_vth_v: float
    id_per_w_a_per_m: float


# Built-in baseline LUT for sky130 NMOS at L=0.5 um, V_DS=0.9 V (typical).
# Sorted by gm/Id ascending (strong → weak inversion).
SKY130_NMOS_LUT_L05: List[GmIdEntry] = [
    GmIdEntry(gm_over_id=5.0, vgs_minus_vth_v=0.40, id_per_w_a_per_m=80e-6),
    GmIdEntry(gm_over_id=8.0, vgs_minus_vth_v=0.25, id_per_w_a_per_m=40e-6),
    GmIdEntry(gm_over_id=12.0, vgs_minus_vth_v=0.16, id_per_w_a_per_m=20e-6),
    GmIdEntry(gm_over_id=18.0, vgs_minus_vth_v=0.10, id_per_w_a_per_m=8e-6),
    GmIdEntry(gm_over_id=25.0, vgs_minus_vth_v=0.06, id_per_w_a_per_m=2e-6),
]


def _validate_lut(lut: List[GmIdEntry]) -> None:
    if not lut:
        raise ValueError("gm/Id LUT must contain at least one entry")


def _sorted_lut(lut: List[GmIdEntry]) -> List[GmIdEntry]:
    return sorted(lut, key=lambda e: e.gm_over_id)


def lookup_by_gm_id(lut: List[GmIdEntry], gm_over_id_target: float) -> GmIdEntry:
    """Return the LUT entry whose gm/Id is closest to the target.

    Ties (equal distance) resolve to the lower-gm/Id entry, which corresponds
    to stronger inversion and higher current density — a slightly more
    conservative sizing choice.
    """
    _validate_lut(lut)
    best = lut[0]
    best_dist = abs(best.gm_over_id - gm_over_id_target)
    for entry in lut[1:]:
        d = abs(entry.gm_over_id - gm_over_id_target)
        if d < best_dist:
            best = entry
            best_dist = d
    return best


def interpolate_id_per_w(lut: List[GmIdEntry], gm_over_id_target: float) -> float:
    """Linearly interpolate current density J_D = I_D / W at the target gm/Id.

    Returns id_per_w_a_per_m (A/m). Targets below the LUT minimum clamp to
    the lowest-gm/Id entry; targets above the maximum clamp to the
    highest-gm/Id entry.
    """
    _validate_lut(lut)
    sorted_lut = _sorted_lut(lut)
    if gm_over_id_target <= sorted_lut[0].gm_over_id:
        return sorted_lut[0].id_per_w_a_per_m
    if gm_over_id_target >= sorted_lut[-1].gm_over_id:
        return sorted_lut[-1].id_per_w_a_per_m
    # Locate bracketing pair.
    for i in range(len(sorted_lut) - 1):
        lo = sorted_lut[i]
        hi = sorted_lut[i + 1]
        if lo.gm_over_id <= gm_over_id_target <= hi.gm_over_id:
            span = hi.gm_over_id - lo.gm_over_id
            if span == 0.0:
                return lo.id_per_w_a_per_m
            t = (gm_over_id_target - lo.gm_over_id) / span
            return lo.id_per_w_a_per_m + t * (hi.id_per_w_a_per_m - lo.id_per_w_a_per_m)
    # Should be unreachable given clamps above; fall back to last entry.
    return sorted_lut[-1].id_per_w_a_per_m


def size_w_for_target_id(
    lut: List[GmIdEntry],
    target_id_a: float,
    gm_over_id_target: float,
) -> float:
    """Return the device width W (in micrometers) needed to deliver
    ``target_id_a`` amperes at the requested gm/Id operating point.

    Computes W [m] = I_D / J_D where J_D is linearly interpolated from
    the LUT, then converts to micrometers.

    Raises:
        ValueError: if ``target_id_a`` is non-positive or the LUT is empty.
    """
    _validate_lut(lut)
    if target_id_a <= 0.0:
        raise ValueError(
            f"target_id_a must be positive (got {target_id_a}); "
            "negative or zero currents are not physically meaningful for sizing"
        )
    j_d = interpolate_id_per_w(lut, gm_over_id_target)
    if j_d <= 0.0:
        raise ValueError(
            f"Interpolated current density is non-positive ({j_d}); check LUT entries"
        )
    w_m = target_id_a / j_d  # meters
    return w_m * 1e6  # micrometers
