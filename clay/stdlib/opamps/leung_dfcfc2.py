"""LeungDFCFC2 — 3-stage opamp with dual feedforward + damping path.

Leung 2003 v2 Damped Feed-Forward Compensation (DFCFC2). Same
signal path as the textbook NMCF
(`nested_miller_3stage_feedforward`) but the compensation network
adds three structural features:

  1. An active feedforward PMOS device `Mgf2` (PMOS, g=d1, d=out)
     — same structural role as HoiLee_AFFC's Mgf2 (primary FF path).
  2. A SECOND active feedforward PMOS `Mgf_damp` (PMOS, g=d1,
     d=nd_damp) — the dual-FF distinguishing device that drains
     into a new internal node `nd_damp` instead of out.
  3. An active sink `Mlcs_damp` (NMOS, d=nd_damp, g=nbias) providing
     the DC current path for Mgf_damp; its 1/gm loading damps the
     small-signal response at nd_damp.
  4. Re-routed compensation: Cm1 (out↔d1 skip-over) + Cm2_damp
     (nd_damp↔d1 damping cap). Cf is dropped.

Same 5-port surface as the NMCF baseline; the new devices and node
are internal. See `clay.templates.leung_dfcfc2` for the full topology
description and the Mgf2 / Mgf_damp / Cm2_damp kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.leung_dfcfc2 import build_nested_miller_3stage_dfcfc2
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("LeungDFCFC2", testbench=build_ac_testbench)
def LeungDFCFC2(spec: Spec, **kwargs) -> Cell:
    """Build a Leung DFCFC2 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    gm_id_diff / load1 / stage2 / stage3), the DFCFC2 cell accepts
    these new knobs:

      Cm2_damp: float = 0.5e-12   — damping Miller cap (nd_damp ↔ d1)
      Mgf2_scale: float = 1.0     — ratio of Mgf2 W to Mcs3 W
      W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing
      gm_id_gf2: gm/Id intent override for Mgf2
      Mgf_damp_scale: float = 0.5 — ratio of Mgf_damp W to Mcs3 W
      W_gf_damp, L_gf_damp: optional explicit overrides
      gm_id_gf_damp: gm/Id intent override for Mgf_damp

    Note: Cf kwarg from the NMCF baseline is NOT accepted here because
    the DFCFC2 topology does not contain Cf. Cm2 from NMCF is also
    NOT accepted — it is renamed Cm2_damp because the routing changes
    (nd_damp↔d1 instead of d2↔d1).
    """
    return build_nested_miller_3stage_dfcfc2(
        cell_name=kwargs.pop("cell_name", "LeungDFCFC2"),
        spec=spec,
        **kwargs,
    )
