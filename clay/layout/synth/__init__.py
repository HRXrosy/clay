"""Geometric-synthesis engine (longplan v3).

Evidence-driven layout synthesis: a deterministic CP-SAT envelope-min placement core
(validated in FIX1: 9 OTA groups -> 7760um2 envelope, OPTIMAL, beats reference 10704),
plus slack allocation and over-cell routing feasibility. The area objective is the
bounding-rectangle ENVELOPE (W*H), not fill-to-zero-slack: internal slack is a resource
for symmetry / routing, never compacted away.

See docs/superpowers/specs/clay-geometric-synthesis-engine-longplan-2026-06-17.html.
"""
from clay.layout.synth.placement import Group, PlacementResult, place_envelope_min

__all__ = ["Group", "PlacementResult", "place_envelope_min"]
