"""Functional (digital / mixed-signal) cell builders verified by real ngspice.

This package hosts hand-rolled / parametric functional primitives that
are constructed directly as Clay IR ``Cell`` objects (no .clay surface
round-trip required) so they can be sized per-MOS-role. The first
inhabitant is ``dff_fixed_v152`` — a corrected D flip-flop that fixes
the R103 slave-feedback race-contention lockup in the committed
``dff`` / ``dff_async_clear`` primitive.
"""
