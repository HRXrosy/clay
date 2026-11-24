"""General matched-DEVICE pairing from the netlist's declared symmetry intent.

A matched device pair (the two transistors of a differential pair, or the two
load transistors of a current mirror) is what every placement-symmetry /
common-centroid obligation measures. The pairing MUST be derived from the
circuit's declared symmetry, never from a per-circuit table of device labels.

Derivation -- the differential-swap symmetry of the netlist
===========================================================
``intent.diff_pairs`` and ``intent.mirror_pairs`` are the NET pairs the designer
declared symmetric (each a ``(net_a, net_b)`` differential / mirror pair). Together
they define a net PERMUTATION ``sigma`` that swaps each declared pair and fixes
every other net (supply rails, a shared tail node, a clock, ...). Two distinct
devices ``L`` and ``R``
form a matched pair exactly when ``sigma`` maps ``L``'s terminal->net signature
onto ``R``'s: for every terminal kind ``tk``, ``sigma(net at L[tk]) == net at
R[tk]``. That is the differential-swap symmetry of the schematic -- the pair of
devices that exchange roles when the two halves of the differential circuit are
swapped. A device that sits ON the symmetry axis (the tail, a centre-tap) has a
``sigma``-invariant signature, maps to itself, and is correctly excluded.

This uses ONLY the declared intent pairs + the device->net membership. It reads
no device LABELS, no per-cell table, and no geometry, so it works identically on
a real cell and on a deliberately mis-placed synthetic stand-in (the membership
is geometry-independent), and it generalises to any differential / mirror circuit
without a per-circuit branch.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


def swap_permutation(intent) -> Dict[str, str]:
    """The net permutation ``sigma`` induced by the declared symmetric net pairs.

    Every ``(net_a, net_b)`` in ``intent.diff_pairs`` and ``intent.mirror_pairs``
    contributes ``sigma[net_a] = net_b`` and ``sigma[net_b] = net_a``; every other
    net is fixed (absent from the map, so callers treat a missing key as identity).
    """
    sigma: Dict[str, str] = {}
    pairs: List[Tuple[str, str]] = []
    pairs.extend(getattr(intent, "diff_pairs", ()) or ())
    pairs.extend(getattr(intent, "mirror_pairs", ()) or ())
    for (net_a, net_b) in pairs:
        sigma[net_a] = net_b
        sigma[net_b] = net_a
    return sigma


def _swap_signature(sig: Mapping[str, str], sigma: Mapping[str, str]) -> Dict[str, str]:
    """Apply ``sigma`` to every net in one device's ``{tk: net}`` signature."""
    return {tk: sigma.get(net, net) for tk, net in sig.items()}


def matched_device_pairs(
    dev_nets: Mapping[str, Mapping[str, str]],
    intent,
) -> List[Tuple[str, str]]:
    """Matched DEVICE pairs ``(L, R)`` derived from the declared symmetry intent.

    ``dev_nets`` is the device->net membership ``label -> {tk: net}`` (the same
    shape the placement checker already builds, from a cell's per-terminal tags or
    a device plan). For the net permutation ``sigma`` of the declared diff/mirror
    pairs, ``(L, R)`` is emitted when ``sigma`` carries ``L``'s signature exactly
    onto ``R``'s AND that signature is not ``sigma``-invariant (an on-axis device
    maps to itself and is skipped). Labels are scanned in sorted order and each is
    bound to its first matching partner, so the result is deterministic and each
    device appears in at most one pair. No device labels are special-cased.
    """
    sigma = swap_permutation(intent)
    if not sigma:
        return []
    labels = sorted(dev_nets)
    out: List[Tuple[str, str]] = []
    taken = set()
    for i, left in enumerate(labels):
        if left in taken:
            continue
        sig_left = dict(dev_nets[left])
        swapped = _swap_signature(sig_left, sigma)
        if swapped == sig_left:
            continue  # on-axis device: sigma-invariant, no distinct partner
        for right in labels[i + 1:]:
            if right in taken:
                continue
            if dict(dev_nets[right]) == swapped:
                out.append((left, right))
                taken.add(left)
                taken.add(right)
                break
    return out


__all__ = ["swap_permutation", "matched_device_pairs"]
