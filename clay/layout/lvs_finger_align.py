"""R146 (BL-146A) -- finger/device-count alignment for gf180mcu LVS.

The problem R145 C surfaced
===========================
The R145 LVS run on the DRC-clean 5T-OTA returned a *device-count* FAIL:

* the **layout** extraction (real ``gf180mcu.lvs`` deck, flat run) reported
  **10 single-finger** ``mos4`` devices -- one ``M$k`` card per physical finger
  (the extracted ``..._extracted.cir`` shows ``M$1..M$10``, 4 ``pmos_3p3`` +
  6 ``nmos_3p3``, no ``m=`` multiplier);
* the **schematic** R145 B (:mod:`clay.layout.lvs_netlist_export`) emits
  **5 logical** devices, each carrying ``m=N`` where ``N`` is the device's
  ``finger_count`` (so a 2-finger transistor is one card with ``m=2``).

5 logical cards vs 10 physical fingers -> LVS reports a device-count mismatch.
The two sides are electrically the SAME circuit; they only disagree on whether
each multi-finger transistor is reported as one ``m=N`` device or as ``N``
single-finger devices. This module aligns the *convention* so the device
counts match, WITHOUT changing connectivity, sizing, or the recognised
topology.

Two alignment paths (both real; one is selected per call)
=========================================================
**(A) deck-combine (preferred when running the real deck).**
The shipped ``gf180mcu.lvs`` deck supports parallel/folded-device combining via
its ``$combine`` runtime variable (deck lines 490-496: ``if $combine ->
COMBINE = true``; lines 1881-1883: ``if COMBINE -> netlist.combine_devices``).
``netlist.combine_devices`` is KLayout's built-in combiner that merges abutted
finger transistors back into one ``m=N`` device on the *layout* side. Passing
``-rd combine=true`` therefore collapses the 10 extracted fingers into 5 ``m=N``
devices that match the R145 B schematic's 5 cards. This module exposes that as
:func:`deck_combine_rd` -- the exact ``(key, value)`` ``-rd`` pair the R145 A
harness (:func:`tools.klayout_lvs_signoff_v145.run.build_klayout_lvs_command`)
accepts through its ``extra_rd`` argument. NOTE: setting ``$combine`` also flips
the deck's ``SIMPLIFY`` to false (deck lines 514-518); that only disables the
deck's auto-simplify of the *schematic* side, which is the correct behaviour
here because the schematic is already in its intended (combined) form.

**(B) schematic-split (offline-deterministic, no binary).**
When the real deck is run *without* ``$combine`` (the R145 C configuration that
produced the 10-finger extraction), the schematic must instead be expanded to
match the flat extraction: each ``m=N`` device becomes ``N`` single-finger
devices (``m=1``), reproducing exactly the ``M$1..M$N`` finger-by-finger device
list KLayout emits. This is what :func:`split_schematic_fingers` does, and it
is fully testable offline (no KLayout, no GDS) -- the device count is a pure
function of the schematic's ``finger_count`` annotations.

Both paths converge on a matching device count:

    5 logical (m=2/m=1) devices  ==deck-combine==>  layout: 5 combined devices
    layout: 10 single-finger     ==schematic-split=>  schematic: 10 devices

No fabricated LVS pass
======================
This module NEVER reports an LVS verdict and never claims the deck matched.
It only produces (a) a schematic netlist whose *device count* is reconciled
with a given/expected extraction count, and (b) a structural device-count
parity check. The authoritative match/don't-match verdict still comes from the
real deck's ``compare`` (R145 A). The R145 B exporter is imported READ-ONLY and
never edited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# R145 B exporter, imported READ-ONLY (we never mutate it; we wrap its output).
from .lvs_netlist_export import (
    LvsDevice,
    LvsSchematic,
    MosType,
    MOS_PIN_ORDER,
    build_lvs_schematic,
    export_lvs_netlist,
)
from .annotations import AnnotatedCell


class AlignStrategy(Enum):
    """How to reconcile schematic device count with layout extraction.

    ``DECK_COMBINE``
        Keep the schematic's ``m=N`` logical devices and ask the deck to
        combine the layout's single fingers (``-rd combine=true``). The
        schematic netlist is emitted unchanged; the alignment happens on the
        layout side at deck runtime.
    ``SCHEMATIC_SPLIT``
        Expand each schematic ``m=N`` device into ``N`` single-finger
        (``m=1``) devices so the schematic device count equals the flat
        extraction's finger count. No deck flag is needed.
    """

    DECK_COMBINE = "deck_combine"
    SCHEMATIC_SPLIT = "schematic_split"


# The deck's runtime-variable that enables ``netlist.combine_devices``
# (gf180mcu.lvs lines 490-496 / 1881-1883). Truthiness in the deck is "the
# variable is set at all" (``if $combine``), so the value is conventionally
# "true"; any non-empty string would also enable it, but "true" is the
# self-documenting form the deck's other booleans use.
DECK_COMBINE_RD_KEY = "combine"
DECK_COMBINE_RD_VALUE = "true"


def deck_combine_rd() -> Tuple[str, str]:
    """Return the ``-rd`` pair that enables layout-side device combining.

    Feed this into the R145 A harness's ``extra_rd`` argument
    (``build_klayout_lvs_command(..., extra_rd=[deck_combine_rd()])`` or
    ``run_gf180_lvs(...)`` once it forwards ``extra_rd``) to make the real deck
    run ``netlist.combine_devices`` and collapse the extracted single fingers
    into ``m=N`` devices that match the R145 B schematic.

    Returns
    -------
    (key, value)
        ``("combine", "true")`` -- the exact deck runtime variable.
    """
    return (DECK_COMBINE_RD_KEY, DECK_COMBINE_RD_VALUE)


# --- device-count helpers ----------------------------------------------------


def schematic_logical_device_count(sch: LvsSchematic) -> int:
    """Number of *logical* devices (one per transistor, counting ``m=N`` once)."""
    return sch.device_count()


def schematic_finger_device_count(sch: LvsSchematic) -> int:
    """Number of *physical fingers* across all devices (sum of ``m``).

    This is what a flat extraction (no ``combine_devices``) reports: one device
    card per finger. A device with ``nf<=1`` contributes 1; ``nf=N`` contributes
    ``N``.
    """
    total = 0
    for dev in sch.devices:
        total += max(1, int(dev.nf))
    return total


def expected_layout_device_count(
    sch: LvsSchematic, *, deck_combines: bool
) -> int:
    """Device count the layout extraction will report for ``sch``.

    Parameters
    ----------
    sch:
        The logical schematic (R145 B output) describing the cell.
    deck_combines:
        ``True`` iff the deck runs ``netlist.combine_devices`` (``$combine``).
        When it combines, fingers fold back into one ``m=N`` device, so the
        layout count equals the logical count. When it does not, the flat
        extraction reports one device per finger.
    """
    if deck_combines:
        return schematic_logical_device_count(sch)
    return schematic_finger_device_count(sch)


# --- schematic-split (path B) ------------------------------------------------


def split_schematic_fingers(sch: LvsSchematic) -> LvsSchematic:
    """Expand every ``m=N`` device into ``N`` single-finger (``m=1``) devices.

    The expanded schematic is electrically identical (same nets, same model,
    same W/L per finger) but its device count equals the flat extraction's
    finger count, so a no-``combine`` LVS run sees matching device counts.

    Each finger keeps the parent's nets/model/W/L and is named
    ``<instance>__f<k>`` (1-based). A device with ``nf<=1`` passes through with
    its name unchanged (no spurious suffix on single-finger devices).

    The original ``sch`` is not mutated; a new :class:`LvsSchematic` is
    returned.
    """
    split_devices: List[LvsDevice] = []
    for dev in sch.devices:
        nf = max(1, int(dev.nf))
        if nf == 1:
            split_devices.append(
                LvsDevice(
                    instance=dev.instance,
                    mos_type=dev.mos_type,
                    pins=dict(dev.pins),
                    w_um=dev.w_um,
                    l_um=dev.l_um,
                    nf=1,
                )
            )
            continue
        for k in range(1, nf + 1):
            split_devices.append(
                LvsDevice(
                    instance=f"{dev.instance}__f{k}",
                    mos_type=dev.mos_type,
                    pins=dict(dev.pins),
                    w_um=dev.w_um,
                    l_um=dev.l_um,
                    nf=1,
                )
            )
    return LvsSchematic(
        cell_name=sch.cell_name,
        ports=list(sch.ports),
        devices=split_devices,
    )


# --- alignment result --------------------------------------------------------


@dataclass
class FingerAlignResult:
    """Outcome of aligning a schematic's device count to a layout extraction."""

    strategy: AlignStrategy
    cell_name: str
    netlist: str
    schematic_device_count: int
    expected_layout_device_count: int
    # The ``-rd`` pair to pass to the LVS harness; empty for schematic-split.
    extra_rd: List[Tuple[str, str]] = field(default_factory=list)
    # If the caller supplied the real extracted count, the parity verdict.
    matched_extraction_count: Optional[bool] = None
    extraction_device_count: Optional[int] = None

    @property
    def device_count_aligned(self) -> bool:
        """True iff schematic device count == expected layout device count."""
        return self.schematic_device_count == self.expected_layout_device_count

    def summary(self) -> str:
        verdict = "ALIGNED" if self.device_count_aligned else "MISALIGNED"
        rd = (
            " ".join(f"-rd {k}={v}" for k, v in self.extra_rd)
            if self.extra_rd
            else "(none)"
        )
        return (
            f"LVS finger-align [{self.cell_name}] via {self.strategy.value}: "
            f"{verdict} (schematic={self.schematic_device_count} devices, "
            f"expected layout={self.expected_layout_device_count}; "
            f"deck flags: {rd})"
        )


# --- extracted-netlist device count (read the R145 C extraction) -------------


_DEVICE_PREFIXES = ("M", "D", "Q", "R", "C", "X")


def count_extracted_devices(source) -> int:
    """Count MOSFET/device cards in a KLayout-extracted SPICE netlist.

    Mirrors the structural counting the R145 A harness uses for the EXTRACTED
    side: one count per device card (a line whose first token starts with a
    device-class letter), folding ``+`` continuations and skipping comments and
    control (``.``) lines. ``source`` may be a path or the SPICE text itself.

    This lets a test/caller read the real R145 C ``..._extracted.cir`` and
    compare its count against the aligned schematic -- no count is invented.
    """
    import os

    if isinstance(source, (str, os.PathLike)) and os.path.exists(str(source)):
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = str(source)

    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith("+"):
            continue
        if line.startswith("."):  # .subckt / .ends / .model / .param
            continue
        if line[0].upper() in _DEVICE_PREFIXES:
            count += 1
    return count


# --- top-level alignment API -------------------------------------------------


def align_for_lvs(
    cell: AnnotatedCell,
    sizing: Optional[Dict[str, Tuple[float, float]]] = None,
    *,
    strategy: AlignStrategy = AlignStrategy.SCHEMATIC_SPLIT,
    bindings: Optional[Dict[str, str]] = None,
    default_w_um: float = 4.0,
    default_l_um: float = 0.5,
    extraction_device_count: Optional[int] = None,
    include_header: bool = True,
) -> FingerAlignResult:
    """Produce a schematic netlist whose device count matches the extraction.

    Builds the R145 B logical schematic for ``cell`` (read-only call into
    :mod:`clay.layout.lvs_netlist_export`), then aligns the device-count
    convention to the layout extraction using ``strategy``:

    * :attr:`AlignStrategy.SCHEMATIC_SPLIT` -- expand ``m=N`` devices into ``N``
      single-finger cards so the schematic count equals a no-``combine`` flat
      extraction. ``extra_rd`` is empty (no deck flag needed). This is the
      offline-deterministic path that matches the R145 C extraction (10
      single-finger devices for the 5T-OTA).
    * :attr:`AlignStrategy.DECK_COMBINE` -- keep the logical ``m=N`` schematic
      and return ``extra_rd=[("combine","true")]`` so the deck combines the
      layout's fingers down to the schematic's logical count.

    Parameters
    ----------
    cell, sizing, bindings, default_w_um, default_l_um, include_header:
        Forwarded to / used the same way as R145 B's exporter so the schematic
        matches what the layout was built from.
    extraction_device_count:
        Optional REAL device count from the layout extraction (e.g. parsed from
        the R145 C ``..._extracted.cir`` via :func:`count_extracted_devices`).
        When given, the result records a parity verdict
        (``matched_extraction_count``) against the produced schematic -- this is
        the gated device-match check (no fabrication: it is a literal equality).

    Returns
    -------
    FingerAlignResult
        Carrying the aligned SPICE netlist, the schematic/expected device
        counts, the ``-rd`` flags to pass the harness, and (if supplied) the
        extraction-parity verdict.
    """
    logical = build_lvs_schematic(
        cell, sizing, default_w_um=default_w_um, default_l_um=default_l_um
    )

    if strategy is AlignStrategy.DECK_COMBINE:
        # Schematic stays logical; the deck combines the layout fingers.
        netlist = export_lvs_netlist(
            cell,
            sizing,
            bindings=bindings,
            default_w_um=default_w_um,
            default_l_um=default_l_um,
            include_header=include_header,
        )
        sch_count = schematic_logical_device_count(logical)
        # With combining, the layout count folds down to the logical count.
        expected = expected_layout_device_count(logical, deck_combines=True)
        extra_rd = [deck_combine_rd()]
    elif strategy is AlignStrategy.SCHEMATIC_SPLIT:
        split = split_schematic_fingers(logical)
        netlist = _emit_schematic(split, bindings, include_header)
        sch_count = split.device_count()
        # Flat extraction reports one device per finger.
        expected = expected_layout_device_count(logical, deck_combines=False)
        extra_rd = []
    else:  # pragma: no cover -- exhaustive
        raise ValueError(f"unknown alignment strategy: {strategy!r}")

    result = FingerAlignResult(
        strategy=strategy,
        cell_name=logical.cell_name,
        netlist=netlist,
        schematic_device_count=sch_count,
        expected_layout_device_count=expected,
        extra_rd=extra_rd,
        extraction_device_count=extraction_device_count,
    )
    if extraction_device_count is not None:
        result.matched_extraction_count = sch_count == extraction_device_count
    return result


def _emit_schematic(
    sch: LvsSchematic,
    bindings: Optional[Dict[str, str]],
    include_header: bool,
) -> str:
    """Emit a finger-split schematic to SPICE, reusing R145 B device emission.

    We reuse :meth:`LvsDevice.emit` (R145 B) verbatim for each (single-finger)
    device so the element convention (``x<name> d g s b <model> W=.. L=.. m=1``)
    and model-name resolution stay identical to the logical exporter. Only the
    header device-count comment differs (it now reflects the split count).
    """
    from .lvs_netlist_export import _model_for  # read-only reuse

    lines: List[str] = []
    if include_header:
        counts = sch.count_by_type()
        nmos_model = _model_for(MosType.NMOS, bindings)
        pmos_model = _model_for(MosType.PMOS, bindings)
        lines.append(
            f"* gf180mcu LVS finger-split schematic for cell {sch.cell_name!r}"
        )
        lines.append(
            f"* devices: {sch.device_count()} single-finger "
            f"(nmos={counts[MosType.NMOS]}, pmos={counts[MosType.PMOS]}); "
            f"models: {nmos_model} / {pmos_model}; pin order d g s b; m=1 each"
        )
    lines.append(f".subckt {sch.cell_name} {' '.join(sch.ports)}")
    for dev in sch.devices:
        lines.append("  " + dev.emit(bindings))
    lines.append(".ends")
    return "\n".join(lines) + "\n"


# --- gated device-match check ------------------------------------------------


def lvs_device_match_passes(
    result: FingerAlignResult,
    extraction_device_count: Optional[int] = None,
) -> bool:
    """Gated structural device-match check (NOT the deck's compare verdict).

    Returns ``True`` iff the aligned schematic's device count equals the layout
    extraction count. If ``extraction_device_count`` is given it is used
    directly; otherwise the parity recorded on ``result`` (from
    :func:`align_for_lvs`'s ``extraction_device_count``) is used; if neither is
    available the count is compared against the strategy's *expected* layout
    count (the deterministic prediction).

    This is a real equality check on device counts -- it does not assert the
    full LVS netlists matched (connectivity/parameters are the deck's job).
    """
    n = extraction_device_count
    if n is None:
        n = result.extraction_device_count
    if n is None:
        return result.device_count_aligned
    return result.schematic_device_count == n


__all__ = [
    "AlignStrategy",
    "DECK_COMBINE_RD_KEY",
    "DECK_COMBINE_RD_VALUE",
    "FingerAlignResult",
    "align_for_lvs",
    "count_extracted_devices",
    "deck_combine_rd",
    "expected_layout_device_count",
    "lvs_device_match_passes",
    "schematic_finger_device_count",
    "schematic_logical_device_count",
    "split_schematic_fingers",
]
