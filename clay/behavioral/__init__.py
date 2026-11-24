"""Behavioral (ideal / system-level) model library for Clay (Round 156).

This package hosts a small library of IDEAL, behavioral analog blocks
expressed entirely with ngspice controlled sources / ideal passives
(E-sources, B-sources, ideal R / C). Unlike everything under
``clay.functional`` (transistor-level cells that need a real PDK), these
blocks are PDK-INDEPENDENT: they simulate on real ngspice with no model
deck. That is exactly what makes them useful as system-level placeholders
in a top-down mixed-signal design flow -- a mixed-signal system can be
simulated / verified at the behavioral level *before* any transistor cell
exists, and the self-boot flywheel gets ideal blocks to stand in for
not-yet-synthesized stages.

The reference framework TED ships ~40 such ideal blocks (ideal opamp /
VCO / integrator / comparator / charge-pump, ...). This package serves as the
namespace for PDK-independent behavioral analog blocks.
"""

from __future__ import annotations
