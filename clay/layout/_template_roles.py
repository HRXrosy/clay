"""Bridge: cell-template annotations -> layout-intent role bindings.

The Phase 6 cell templates (``clay.layout.cell_templates``) declare a cell as a
set of named device instances (``M_in_p``, ``M_load_n``, ``M_tail`` ...) carrying
``Matching`` / ``Symmetric`` / ``Adjacent`` spatial annotations. They were built
for the *placement* validator, so they record WHICH devices match and that a pair
is symmetric -- but they deliberately carry **no pin/net bindings** (a
``SymbolicLayout`` only knows ``finger_count`` / ``stripe_count``; there is no
gate/drain/source net anywhere in an ``AnnotatedCell``).

The layout-intent layer (:mod:`clay.layout.intent`), on the other hand, reasons
about NETS: which net is offset-critical, which differential nets must stay
parasitically matched. ``classify_nets_by_role`` consumes ROLE BINDINGS shaped
``{kind, gate?, output?, net?, diff_partner?, name}``. This module is the bridge
that turns a template's device+annotation structure into those bindings.

WHERE THE NETS COME FROM (and the honesty boundary)
---------------------------------------------------
A template stores no nets, so we cannot read gate/output nets off it. The only
net-bearing contract a template actually exposes is its **device-naming
convention**: the factories name devices ``M_in_p`` / ``M_in_n`` for the input
pair, ``M_load_p`` / ``M_load_n`` for the mirror load, ``M_tail`` for the tail,
etc. That convention is a deliberate, stable part of each factory's public API
(it is documented in every factory docstring and asserted by the placement
tests). We therefore synthesize the canonical schematic net names that the
convention implies -- the SAME canonical names the hand-mapped
``derive_diff_ota_intent`` already used (``inp`` / ``inn`` input gates,
``outp`` / ``outn`` differential outputs, ``tail`` source node).

This is NOT fabrication: it is reading the one net-level fact the template does
encode (its naming convention) and refusing to invent anything past it. When a
template's device names do not follow a convention we recognise, we emit no
bindings for those devices rather than guessing nets (see
``_classify_device`` returning ``None``). Such a template is reported as having
no clean role/net data instead of being given fictional pins.

MAPPING RULES (annotation/device-name -> role binding)
------------------------------------------------------
The mapping is driven by ``Matching`` pairs and the device-name suffix/stem:

  M1  A ``Matching(other=X)`` annotation makes the two devices a matched PAIR.
      Each half emits one binding that names the other as ``diff_partner`` so
      ``classify_nets_by_role`` re-discovers the differential pair from the
      bindings alone (rule R4 there). Bidirectional ``Matching`` (both halves
      annotate each other, as every factory emits) is collapsed to one pair.

  M2  The pair's KIND comes from the device-name stem:
        * ``M_in_*``                      -> ``input_transconductor``
        * ``M_load_*``                    -> ``current_mirror`` (mirror load)
        * ``M_ref_* / M_out_*``           -> ``current_mirror`` (ref/out legs)
        * ``M_*_cas* / M_n_cas* / M_p_cas*`` -> ``cascode``
      A device-name suffix ``_p`` / ``_n`` (or ``_l`` / ``_r``, or stem ``_ref``
      / ``_out``) tags the plus/minus half of the pair, fixing which synthesised
      net goes on which half.

  M2b Each matched pair also carries a ``match_kind`` tag distinguishing a
      DIFFERENTIAL pair from a single-ended-output MIRROR pair:
        * ``current_mirror`` role -> ``match_kind="mirror"`` -- the two legs are
          a current mirror: one node is the diode/reference leg, the other the
          mirrored output, so the two NODES are asymmetric BY DESIGN (a
          single-ended-output mirror load is not cap-balanced). Its matching is a
          DEVICE placement/sizing concern, handled by
          :mod:`clay.layout.placement_symmetry`, NOT a node cap budget.
        * every other matched role (``input_transconductor`` / ``cascode`` /
          regen / differential output) -> ``match_kind="differential"`` -- the
          two nodes are genuinely differential (symmetric roles) and must stay
          cap-balanced (a ``matched_cc_balance`` node budget).
      The classifier (:func:`clay.layout.intent.classify_nets_by_role`) routes
      ``differential`` net pairs to ``diff_pairs`` and ``mirror`` net pairs to a
      separate mirror-pair channel that carries no cap budget.

  M3  Per-kind net synthesis (the canonical convention names):
        * input pair  -> gate ``inp`` / ``inn`` (the offset-critical analog
          inputs; rule R1 in the classifier marks these sensitive), and the
          differential output drains ``outp`` / ``outn``.
        * mirror load -> the two drains are the differential output pair
          ``outp`` / ``outn`` (shared with the input drains: in a 5T-OTA the
          input drain and its mirror-load drain are the same node). The mirror
          gate is the diode-connected reference ``outn``.
        * tail        -> single ``net`` ``tail`` (a ``tail_current_source``;
          neither dynamic nor sensitive -- an on-axis common node).
        * cascode legs (folded/telescopic) -> internal differential nodes
          ``casp`` / ``casn``; no externally-driven input gate, so they are not
          marked sensitive.

  M4  A device with no ``Matching`` partner and a recognised solo stem (only
      ``M_tail`` today) emits a single-net binding. An unmatched device with an
      unrecognised stem emits nothing (honesty boundary).

The bindings this module emits feed straight into ``classify_nets_by_role`` ->
``derive_intent_from_roles``; nothing here decides net classes or budgets, so the
classification policy stays in one place.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell, Matching


# ── Net-name convention (canonical schematic names the factories imply) ───────
# These are the SAME canonical names the hand-mapped diff_ota deriver used, so a
# template-driven 5T-OTA reconciles byte-for-byte with the legacy hand mapping.
_INPUT_GATE_P = "inp"
_INPUT_GATE_N = "inn"
_OUT_P = "outp"
_OUT_N = "outn"
_CAS_P = "casp"
_CAS_N = "casn"
_TAIL_NET = "tail"


def _half_sign(device: str) -> Optional[str]:
    """Return ``"p"`` / ``"n"`` for the plus/minus half a device name encodes.

    Recognises the suffix conventions the factories use to tag the two halves of
    a matched pair: ``_p`` / ``_n`` (plus/minus), ``_l`` / ``_r`` (left/right),
    and the ``_ref`` / ``_out`` mirror legs (ref == "p" half, out == "n" half).
    Returns ``None`` when the name carries no recognised half-sign.
    """
    d = device.lower()
    if d.endswith("_p"):
        return "p"
    if d.endswith("_n"):
        return "n"
    if d.endswith("_l"):
        return "p"
    if d.endswith("_r"):
        return "n"
    # current-mirror ref/out legs (M_ref_low / M_out_low, M_ref / M_out ...)
    if "_ref" in d:
        return "p"
    if "_out" in d:
        return "n"
    return None


# Roles whose matched pair is a single-ended-output CURRENT MIRROR (asymmetric
# nodes by design) rather than a genuinely DIFFERENTIAL pair. A mirror pair's
# node matching is a DEVICE placement/sizing concern (placement_symmetry), not a
# matched-node cap budget.
_MIRROR_ROLES = frozenset({"current_mirror"})


def _match_kind_for(kind: str) -> str:
    """Return ``"mirror"`` for a current-mirror role, ``"differential"`` else.

    A ``current_mirror`` matched pair is a single-ended-output mirror (one node
    is the diode/reference leg, the other the mirrored output) -- the two NODES
    are asymmetric by design, so the pair is NOT a matched-node cap budget. Every
    other matched role (input pair, cascode legs, regen, differential output) is
    genuinely differential and its two nodes must stay cap-balanced.
    """
    return "mirror" if kind in _MIRROR_ROLES else "differential"


def _classify_device(device: str) -> Optional[Tuple[str, str]]:
    """Map a device name to ``(kind, group)`` or ``None`` if unrecognised.

    ``kind`` is the functional-role keyword ``classify_nets_by_role`` consumes;
    ``group`` is a coarse net-synthesis bucket (``"input"`` / ``"mirror"`` /
    ``"cascode"`` / ``"tail"``) that picks which canonical nets to attach.

    Returns ``None`` for a device whose name follows no convention we recognise
    -- the honesty boundary: we synthesise no nets for it rather than guessing.
    """
    d = device.lower()
    # Cascode legs first (they often also contain _p/_n which the input check
    # would otherwise swallow): "cas" anywhere in the stem.
    if "cas" in d:
        return ("cascode", "cascode")
    if d.startswith("m_in") or "_in_" in d or d == "m_in":
        return ("input_transconductor", "input")
    if d.startswith("m_load") or "_load" in d:
        return ("current_mirror", "mirror")
    if d.startswith("m_ref") or d.startswith("m_out") or "_ref" in d or "_out" in d:
        return ("current_mirror", "mirror")
    if d.startswith("m_tail") or d == "m_tail" or "tail" in d:
        return ("tail_current_source", "tail")
    return None


def _nets_for(group: str, sign: Optional[str]) -> Dict[str, Optional[str]]:
    """Synthesise the gate/output nets for one device of a matched pair.

    ``sign`` is ``"p"`` / ``"n"`` (the half this device is) or ``None``.
    Returns a dict with ``gate`` / ``output`` keys (either may be ``None`` when
    the convention does not pin that net for this group).
    """
    p = sign == "p"
    if group == "input":
        # Input pair: externally-driven analog gate + differential output drain.
        return {
            "gate": _INPUT_GATE_P if p else _INPUT_GATE_N,
            "output": _OUT_P if p else _OUT_N,
        }
    if group == "mirror":
        # Mirror load: both drains are the differential output pair; the gate is
        # the diode-connected reference node (outn) shared by both legs.
        return {
            "gate": _OUT_N,
            "output": _OUT_P if p else _OUT_N,
        }
    if group == "cascode":
        # Cascode legs: internal differential nodes, no input gate.
        return {
            "gate": None,
            "output": _CAS_P if p else _CAS_N,
        }
    return {"gate": None, "output": None}


def _matching_partner(cell: AnnotatedCell, device: str) -> Optional[str]:
    """Return the ``Matching`` partner of ``device``, or ``None`` if unmatched."""
    for ann in cell.annotations_for(device):
        if isinstance(ann, Matching):
            return ann.other
    return None


def template_role_bindings(template) -> List[Dict[str, object]]:
    """Derive layout-intent role bindings from a cell template.

    ``template`` is either an :class:`~clay.layout.annotations.AnnotatedCell` or
    the ``(AnnotatedCell, sizing)`` tuple the ``build_*`` factories return.

    Returns a list of role-binding dicts shaped
    ``{kind, gate?, output?, net?, diff_partner?, match_kind?, name}`` -- exactly
    what :func:`clay.layout.intent.classify_nets_by_role` consumes. Matched device
    pairs become two bindings naming each other as ``diff_partner`` and carrying a
    ``match_kind`` tag (``"differential"`` vs ``"mirror"``, see rule M2b); the
    gate/output nets are the canonical schematic nets the device-name convention
    implies (see module docstring for the full mapping + honesty boundary).

    Devices whose names follow no recognised convention contribute no bindings
    (we never invent nets for them); a template made entirely of such devices
    yields an empty list.
    """
    cell = template[0] if isinstance(template, tuple) else template
    if not isinstance(cell, AnnotatedCell):
        raise TypeError(
            f"expected an AnnotatedCell (or (AnnotatedCell, sizing) tuple), "
            f"got {type(cell).__name__}"
        )

    devices = list(cell.layouts.keys())  # deterministic: insertion order
    bindings: List[Dict[str, object]] = []
    paired_done = set()  # frozenset of the two device names, so each pair once

    for dev in devices:
        cls = _classify_device(dev)
        if cls is None:
            # Honesty boundary: unknown device convention -> no synthesised nets.
            continue
        kind, group = cls
        partner = _matching_partner(cell, dev)

        if partner is not None and cell.has_device(partner):
            # Matched pair: emit both halves once, naming each other.
            key = frozenset((dev, partner))
            if key in paired_done:
                continue
            paired_done.add(key)

            sign_a = _half_sign(dev)
            sign_b = _half_sign(partner)
            # If neither name carries a half-sign, fall back to declaration order
            # (first device = "p" half) so the pair still splits deterministically.
            if sign_a is None and sign_b is None:
                sign_a, sign_b = "p", "n"
            elif sign_a is None:
                sign_a = "n" if sign_b == "p" else "p"
            elif sign_b is None:
                sign_b = "n" if sign_a == "p" else "p"

            match_kind = _match_kind_for(kind)
            for d, s in ((dev, sign_a), (partner, sign_b)):
                nets = _nets_for(group, s)
                binding: Dict[str, object] = {
                    "kind": kind, "name": d,
                    "diff_partner": (partner if d == dev else dev),
                    "match_kind": match_kind,
                }
                if nets.get("gate"):
                    binding["gate"] = nets["gate"]
                if nets.get("output"):
                    binding["output"] = nets["output"]
                bindings.append(binding)
        else:
            # Unmatched device. Only emit when the stem carries a single defining
            # net the convention pins (the tail). Otherwise: honesty boundary.
            if group == "tail":
                bindings.append({"kind": kind, "name": dev, "net": _TAIL_NET})
            # else: an unmatched non-tail device (e.g. a lone bias mirror with no
            # Matching partner) has no convention-pinned net -> emit nothing.

    return bindings


__all__ = ["template_role_bindings", "_match_kind_for"]
