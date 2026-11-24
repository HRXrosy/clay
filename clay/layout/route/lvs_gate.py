"""Independent, reusable in-loop LVS GATE for the routed gf180 OTA amplifier.

This is the ``checker independent of the checked``: a SECOND verification path that
does NOT share the router's fast connectivity model
(:mod:`clay.layout.connectivity`). Where the router judges its own work by a
via-aware union-find on the metal it just drew, this gate hands the FULL routed GDS
to KLayout's foundry-style extractor (``LayoutToNetlist`` via the synthesized
``klayout_lvs_from_pdk`` deck) and runs a real device-vs-schematic COMPARE against a
fold-reconciled reference netlist emitted from the SAME intent. The two paths never
share a blind spot: if the router's model says ``connected`` but the geometry is a
foundry-deck open (or vice-versa), the disagreement surfaces here.

What the gate does (in order)
=============================
1. EMIT the full labelled GDS: device-cell geometry + cap device cells + routed
   wires, plus one on-metal metal1 net-name ``kdb.Text`` per routed net (so KLayout
   name-anchors the layout nets to the schematic ports). Reuses the leakage-free
   :func:`clay.layout.route.emit_gds.net_label_points` /
   :func:`~clay.layout.route.emit_gds.emit_net_labels`.
2. EMIT the fold-reconciled REFERENCE netlist from the intent
   (:func:`clay.layout.lvs_reference_emit.build_reference_netlist`): MOS folded to
   ``W=W*M`` (mirroring KLayout ``combine_devices``), resistors collapsed to wires on
   the synth path, MIM model ``mim_1p0fF``.
3. RUN real KLayout LVS extract+compare (the synthesized MIM-extended deck) with the
   schematic SET, so KLayout performs a true ``compare`` (not extraction-only).
4. PARSE the verdict from BOTH the tool stdout (``Netlists match`` /
   ``Netlists don't match``) AND the structured ``.lvsdb`` cross-reference, returning
   the per-net / per-device unmatched list when it does NOT match.

Leakage discipline
==================
Every shape comes from a device cell (``build_device`` / ``build_capacitor``) or the
router OUTPUT; the reference netlist is generated from ``build_ota_intent()`` devices
and passives. NO reference ROUTED layout, no foundry GDS, no ground-truth coordinates
are ever read. The gate reports ONLY what KLayout actually printed / wrote -- it never
claims a match it did not observe, and on a non-match it returns the exact unmatched
device/net list.
"""
from __future__ import annotations

import subprocess
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import klayout.db as kdb

from clay.layout.klayout_deck_gen import (
    BjtLvsSpec,
    DEFAULT_KLAYOUT_BIN,
    MimLevel,
    MimLvsSpec,
    ResLvsSpec,
    klayout_lvs_from_pdk,
)
from clay.layout.lvs_reference_emit import build_reference_netlist, RefNetlistSpec
from clay.layout.placed_geometry import placed_point_nm, placed_ring_nm
from clay.layout.route.metal_junction_legalization import (
    legalize_island_junctions,
)
from clay.layout.tech import LayoutTech

_NM = 1000.0
_TOP_CELL = "ota_routed"


class OracleInputSnapshotChanged(RuntimeError):
    """An immutable, run-private oracle input changed or could not be frozen."""


class OracleGdsSnapshotChanged(OracleInputSnapshotChanged):
    """The private GDS bytes consumed by an oracle changed during its run."""


class OracleReferenceSnapshotChanged(OracleInputSnapshotChanged):
    """The private LVS reference bytes changed during the compare."""


class OracleDeckSnapshotChanged(OracleInputSnapshotChanged):
    """The private generated-deck bytes changed during the checker run."""


def _remove_readonly_tree(path: Path) -> None:
    """Remove one known run directory, restoring Windows readonly files first."""

    target = Path(path).resolve()
    if not target.exists():
        return

    def _make_writable_and_retry(func, raw_path, _exc_info):
        candidate = Path(raw_path)
        candidate.chmod(0o700 if candidate.is_dir() else 0o600)
        func(raw_path)

    shutil.rmtree(target, onerror=_make_writable_and_retry)


@dataclass(frozen=True)
class OracleGdsSnapshot:
    """One run-private, content-addressed oracle input."""

    source_path: Path
    path: Path
    sha256: str
    work_root: Path

    def verify_source_unchanged(self, source_path: Path | None = None) -> str:
        """Bind this snapshot to the exact public source and its current bytes."""

        from clay.layout.route.truth_manifest import sha256_path

        source = Path(
            self.source_path if source_path is None else source_path
        ).resolve()
        if source != self.source_path.resolve():
            raise OracleGdsSnapshotChanged(
                "oracle GDS snapshot belongs to a different source: "
                f"expected {source}, snapshot source is {self.source_path.resolve()}"
            )
        try:
            actual = sha256_path(source)
        except Exception as exc:
            raise OracleGdsSnapshotChanged(
                f"oracle GDS source changed or disappeared: {source}"
            ) from exc
        if actual != self.sha256:
            raise OracleGdsSnapshotChanged(
                "oracle GDS source changed after its private snapshot was created: "
                f"expected {self.sha256}, got {actual}"
            )
        self.verify_unchanged()
        return actual

    def verify_unchanged(self) -> str:
        from clay.layout.route.truth_manifest import sha256_path

        try:
            actual = sha256_path(self.path)
        except Exception as exc:
            raise OracleGdsSnapshotChanged(
                f"oracle GDS snapshot changed or disappeared: {self.path}"
            ) from exc
        if actual != self.sha256:
            raise OracleGdsSnapshotChanged(
                "oracle GDS snapshot changed during the checker run: "
                f"expected {self.sha256}, got {actual}"
            )
        return actual

    def retain(self, reason: str) -> None:
        """Record the intentional retention policy beside immutable evidence."""

        marker = self.path.parent / ".oracle-retained"
        _atomic_write_bytes(
            marker,
            (str(reason).strip() + "\n").encode("utf-8"),
        )

    def cleanup(self) -> None:
        """Delete this run-private tree without leaking readonly inputs."""

        run_dir = self.path.parent.resolve()
        if (
            self.path.resolve().parent != run_dir
            or run_dir.parent != self.work_root.resolve()
        ):
            raise OracleGdsSnapshotChanged(
                "oracle GDS snapshot has an invalid cleanup boundary"
            )
        _remove_readonly_tree(run_dir)


@dataclass(frozen=True)
class _OracleByteSnapshot:
    """Immutable non-GDS oracle input held beside the private GDS."""

    source_path: Optional[Path]
    path: Path
    sha256: str
    role: str
    error_type: type[OracleInputSnapshotChanged]

    def verify_unchanged(self) -> str:
        from clay.layout.route.truth_manifest import sha256_path

        try:
            actual = sha256_path(self.path)
        except Exception as exc:
            raise self.error_type(
                f"oracle {self.role} snapshot changed or disappeared: {self.path}"
            ) from exc
        if actual != self.sha256:
            raise self.error_type(
                f"oracle {self.role} snapshot changed during the checker run: "
                f"expected {self.sha256}, got {actual}"
            )
        return actual


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish complete bytes in one replace, never a partially-written file."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(raw_temp)
    try:
        temporary.write_bytes(data)
        if destination.exists():
            destination.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _unlink_oracle_output(path: Path) -> None:
    """Remove one stale output, including a Windows-readonly predecessor."""

    target = Path(path)
    if not target.exists():
        return
    try:
        target.unlink()
    except PermissionError:
        target.chmod(0o600)
        target.unlink()


def _snapshot_oracle_file(
    source_path: Path,
    run_dir: Path,
    *,
    suffix: str,
    role: str,
    error_type: type[OracleInputSnapshotChanged],
) -> _OracleByteSnapshot:
    """Copy and freeze one external oracle input in the GDS run directory."""

    from clay.layout.route.truth_manifest import sha256_path

    source = Path(source_path).resolve()
    try:
        before = sha256_path(source)
        target = Path(run_dir).resolve() / f"{before}.{suffix}"
        shutil.copyfile(source, target)
        after = sha256_path(source)
        copied = sha256_path(target)
        if before != after or copied != before:
            raise error_type(
                f"oracle {role} source changed while its snapshot was created"
            )
        target.chmod(0o444)
        snapshot = _OracleByteSnapshot(
            source_path=source,
            path=target,
            sha256=before,
            role=role,
            error_type=error_type,
        )
        snapshot.verify_unchanged()
        return snapshot
    except OracleInputSnapshotChanged:
        raise
    except Exception as exc:
        raise error_type(
            f"could not create stable oracle {role} snapshot from {source}"
        ) from exc


def _snapshot_generated_oracle_bytes(
    data: bytes,
    run_dir: Path,
    *,
    suffix: str,
    role: str,
    error_type: type[OracleInputSnapshotChanged],
) -> _OracleByteSnapshot:
    """Freeze generated bytes directly in one run-private directory."""

    digest = hashlib.sha256(data).hexdigest()
    target = Path(run_dir).resolve() / f"{digest}.{suffix}"
    try:
        target.write_bytes(data)
        target.chmod(0o444)
        snapshot = _OracleByteSnapshot(
            source_path=None,
            path=target,
            sha256=digest,
            role=role,
            error_type=error_type,
        )
        snapshot.verify_unchanged()
        return snapshot
    except OracleInputSnapshotChanged:
        raise
    except Exception as exc:
        raise error_type(
            f"could not create stable generated oracle {role} snapshot"
        ) from exc


def create_oracle_deck_snapshot(
    deck_path: Path,
    run_dir: Path,
    *,
    role: str,
) -> _OracleByteSnapshot:
    """Freeze one generated checker deck beside the GDS it will consume."""

    # KLayout uses the runset suffix to select its interpreter (notably raw
    # Ruby ``.drc`` versus XML ``.lydrc``).  A generic ``.deck`` copy is stable
    # bytes but not an executable oracle input on Windows.
    source = Path(deck_path).resolve()
    suffix = source.suffix.lstrip(".") or "deck"
    return _snapshot_oracle_file(
        source,
        run_dir,
        suffix=suffix,
        role=role,
        error_type=OracleDeckSnapshotChanged,
    )


def create_oracle_gds_snapshot(
    gds_path: Path,
    work_dir: Path,
    *,
    run_label: str,
) -> OracleGdsSnapshot:
    """Copy one stable GDS into a unique content-addressed oracle directory."""

    from clay.layout.route.truth_manifest import sha256_path

    source = Path(gds_path).resolve()
    root = Path(work_dir).resolve()
    run_dir = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not source.is_file() or source.stat().st_size <= 0:
            raise OracleGdsSnapshotChanged(
                f"oracle GDS source is missing or empty: {source}"
            )
        before = sha256_path(source)
        run_dir = Path(tempfile.mkdtemp(
            prefix=f"{run_label}-{before[:16]}-",
            dir=root,
        )).resolve()
        snapshot_path = run_dir / f"{before}.gds"
        shutil.copyfile(source, snapshot_path)
        after = sha256_path(source)
        copied = sha256_path(snapshot_path)
    except Exception as exc:
        if run_dir is not None:
            _remove_readonly_tree(run_dir)
        if isinstance(exc, OracleGdsSnapshotChanged):
            raise
        raise OracleGdsSnapshotChanged(
            f"could not create stable oracle GDS snapshot from {source}"
        ) from exc
    if before != after or copied != before:
        _remove_readonly_tree(run_dir)
        raise OracleGdsSnapshotChanged(
            "oracle GDS source changed while its private snapshot was created"
        )
    try:
        snapshot_path.chmod(0o444)
        snapshot = OracleGdsSnapshot(
            source_path=source,
            path=snapshot_path.resolve(),
            sha256=before,
            work_root=root,
        )
        snapshot.verify_source_unchanged(source)
        return snapshot
    except Exception as exc:
        _remove_readonly_tree(run_dir)
        if isinstance(exc, OracleGdsSnapshotChanged):
            raise
        raise OracleGdsSnapshotChanged(
            f"could not freeze oracle GDS snapshot from {source}"
        ) from exc


# --------------------------------------------------------------------------- #
# Numeric-fact resolution (task B3): sheet_ohm_sq / area_cap_f_per_um2 below are
# NOT independent literals -- they are CONSTRUCTED by calling the SAME resolver
# functions the geometry builder uses (passive_device_cell._poly_sheet_rho /
# mim_density_f_per_um2, passive_device_cell_tsmc.external_pdk),
# which in turn consult the adapter override port first, then the single
# tabulated source (clay.layout.passive_cell._PASSIVE_FACTS). This closes the
# "two independently-hardcoded literals that happen to agree" gap: a future
# change to the tabulated facts (or the adapter) now propagates here BY
# CONSTRUCTION, and tests/test_pdk_fact_parity.py binds the wiring.
#
# token / layer fields (top_token / bottom_metal_token / model / oversize_um /
# body_token / sab_token / marker_token) stay LOCAL LITERALS below -- they are
# the deck's own pairing tokens (e.g. 'poly_res' for external_pdk, deliberately NOT the
# same string as the adapter's physical model 'npolyf_u'), a different kind of
# data from the physical density/sheet numbers and never merged with them.
class _FactsTech:
    """Minimal stand-in for a resolved :class:`~clay.layout.tech.LayoutTech`,
    exposing only ``pdk_id`` / ``adapter`` -- the two attributes
    ``passive_device_cell._poly_sheet_rho`` / ``mim_density_f_per_um2`` read.

    Deliberately NOT ``LayoutTech.resolve(...)``: that also parses the PDK's
    full DRC/LVS rule deck (measured ~0.8-0.9s each for open_pdk / open_pdk),
    which this module -- a pure numeric LVS-spec table, no geometry -- never
    needs. Building this at import time keeps the ``_PDK_LVS`` numeric fields
    single-sourced without paying the full rule-deck parse cost on every import
    of this module (this file is imported by most router/signoff tests).
    """

    __slots__ = ("pdk_id", "adapter")

    def __init__(self, pdk_id: str) -> None:
        from clay.pdk import get_pdk

        self.pdk_id = pdk_id
        self.adapter = get_pdk(pdk_id)


def _res_sheet_ohm_sq(pdk_id: str, model: str | None = None) -> float:
    """Poly-resistor sheet resistance (Ohm/sq) for ``pdk_id`` -- the SAME value
    and resolution order (adapter override -> tabulated passive facts) the
    geometry builder uses, so the deck spec and the drawn resistor can never
    silently drift apart."""
    from clay.layout.passive_device_cell import _poly_sheet_rho
    if model is not None:
        from clay.layout.passive_cell import _POLY_RES_MODEL_SHEETS
        return float(_POLY_RES_MODEL_SHEETS[pdk_id][model][0])

    return _poly_sheet_rho(_FactsTech(pdk_id))


def _mim_area_cap_f_per_um2(pdk_id: str) -> float:
    """MIM area-specific capacitance (F/um^2) for ``pdk_id``. open_pdk uses its
    OWN dedicated resolver (the CTM/METAL stack is a separate builder from every
    other PDK's MIM primitive); every other PDK uses the generic resolver (gf180
    branch returns its own tabulated density; non-gf180 resolves via the
    tabulated passive facts). SAME functions test_pdk_fact_parity binds the deck
    spec against."""
    from clay.layout.passive_device_cell import mim_density_f_per_um2

    return mim_density_f_per_um2(_FactsTech(pdk_id))


# gf180mcu layermap (the foundry gf180mcu.lvs layer block) + MOS/MIM models. This is
# the SAME DATA the Phase-1 MIM-emit tests use; sourced from the deck, never a layout.
_GF180_LAYERMAP: Dict[str, Tuple[int, int]] = {
    "metal1": (34, 0), "metal2": (36, 0), "metal3": (42, 0),
    "via1": (35, 0), "via2": (38, 0), "contact": (33, 0),
    "metal4": (46, 0), "metal5": (81, 0),
    "via3": (40, 0), "via4": (41, 0),
    "poly": (30, 0), "diff": (22, 0), "nwell": (21, 0), "fusetop": (75, 0),
    # poly-resistor recognition: salicide block + resistor marker (gf180mcu.lvs).
    "sab": (49, 0), "res_mk": (110, 5),
    # Resistor-only implant inputs retain this legacy oracle's MOS3 scope.
    "res_nplus": (32, 0), "res_pplus": (31, 0),
}
_GF180_DEVICES = {"nmos": "nfet_03v3", "pmos": "pfet_03v3"}
# Full extraction uses the actual GF implant/tap and isolated-well domains.
# Keep the historical MOS3 helper above explicitly scoped to non-full callers.
_GF180_FULL_LAYERMAP = {
    **_GF180_LAYERMAP,
    "nplus": (32, 0), "pplus": (31, 0),
    "dnwell": (12, 0), "pwell": (204, 0),
}
_GF180_FULL_DEVICES = {"nmos": "nmos_3p3", "pmos": "pmos_3p3"}
_GF180_MIM = MimLvsSpec(
    top_token="fusetop", bottom_metal_token="metal2", model="mim_1p0fF",
    area_cap_f_per_um2=_mim_area_cap_f_per_um2("gf180mcu"),
    oversize_um=1.06, top_via_token="via2",
)
_GF180_RES = ResLvsSpec(
    body_token="poly", sab_token="sab", marker_token="res_mk",
    model="ppolyf_u", sheet_ohm_sq=_res_sheet_ohm_sq("gf180mcu"),
    implant_token="res_pplus",
    variants=(ResLvsSpec(body_token="poly", sab_token="sab", marker_token="res_mk",
        model="npolyf_u", sheet_ohm_sq=_res_sheet_ohm_sq("gf180mcu", "npolyf_u"),
        implant_token="res_nplus"),),
)

# open_pdk FULL-extraction recipes (the bring-up landed 2026-06-12; consumed via
# the ``full`` overlay below + ``lvs_gate(full_extraction=True)``):
#   * MIM: external_lvs recipe ``CTMn AND METALn`` (extract.rul header) = the ENCLOSE
#     form (oversize 0): top marker CTM5 = 67:5 (external_pdk.layermap line 101),
#     bottom plate = METAL5, riser VIA56 -> METAL6 cover. Density 1.025 fF/um^2
#     (external_model.scs line 16764), device class 'mimcap' (the external_model model name; also
#     the gate reference's cap token).
#   * poly res: R[NR] = RPDMY*POLY1*NIMP*RPO (extract.rul header); SAB = RPO
#     34/0, marker = RPDUMMY 54/0 (layermap lines 51/82); sheet 292 ohm/sq
#     (external_model.l line 17882 ``r_rnpo1rpo = 292`` -- the physical model is
#     ``npolyf_u``, N+ poly w/o silicide). The device-CLASS token is 'poly_res'
#     = the gate reference emitter's external_pdk R token
#     (clay.layout.lvs_reference_emit._MODELS['external_pdk']['res']); KLayout pairs
#     device classes BY NAME, so deck and reference must agree on the token.



# external_pdk FULL-extraction recipes. Each PDK resolves the numeric passive facts
# through its own adapter/facts row even though both use the MIM10 stack.
#   * MIM: the external_pdk recipe is the ENCLOSE form (oversize 0) -- the bottom plate is
#     the whole M5 (65) interacting the MIM top-plate marker (58, lvs:480); the top
#     plate rises on V5 (74, lvs:498) to the M6 (66) cover. Density 0.971 fF/um^2
#     (== the external_pdk MIM10 COX = 9.71E-04 F/m^2 the passive facts cite), device class
#     'mimcap' (the LVS deck's ``DEVICE C(MIM) ... mimcap`` token).
#   * poly res: body = poly (GT 30) under SAB (48) AND streamed RESP1 (96/0);
#     external_drc maps 96/0 to private work layers. Sheet 271.6 ohm/sq
#     (npolyf_u, non-silicide N+ poly -- the external_pdk passive facts value), model
#     'npolyf_u'.


def _vertical_pnp_spec(pdk_id: str) -> Tuple[BjtLvsSpec, ...]:
    """Build the common fixed-cell vertical-PNP recipe from adapter model data.

    external_pdk and both external_pdk 0.18 stacks use the same semantic construction in their
    foundry decks: a BJT marker selects Nwell base, OD2/DG selects the 3.3 V
    family, and P+ active excluding N+ is the emitter.  Only layer numbers and
    the model token differ, and those remain PDK-owned data.
    """
    binding = _FactsTech(pdk_id).adapter.bjt_bindings["pnp"]
    return (
        BjtLvsSpec(
            model=binding.lvs_model,
            base_include_tokens=("bjt_marker", "nwell", "od2"),
            emitter_include_tokens=("diff", "pplus"),
            emitter_exclude_tokens=("nplus",),
            base_terminal_region="nwell",
            emitter_terminal_region="psd",
            collector_terminal_region="substrate",
        ),
    )



# Per-PDK LVS configuration. Keyed by tech_id; only PDKs with a synthesized deck +
# MIM recipe are supported (gf180mcu today). A tech_id absent here -> the gate raises
# (honest: it cannot run a real compare it has no deck for).
#
# DECK PAIRING TOKENS ARE NOT PHYSICAL MODEL NAMES -- DO NOT MERGE THEM. Every
# ``model=`` string above (``ppolyf_u`` / ``poly_res`` / ``npolyf_u`` / ``mimcap`` /
# ``mim_1p0fF``) is this SYNTHESIZED deck's own device-CLASS pairing token (KLayout
# pairs the extracted device and the reference card by this exact string); it is a
# DIFFERENT datum from the PDK's real physical model name (e.g. open_pdk's
# adapter ``poly_res_model="npolyf_rnpo1rpo"`` / the deck's HSPICE resistor
# ``npolyf_u``). ``'poly_res' != 'npolyf_u'`` is DELIBERATE, not a typo -- the deck
# token only has to match the gate REFERENCE emitter's token
# (``clay.layout.lvs_reference_emit``), never the foundry's own model string.
# Only the NUMERIC fields (``sheet_ohm_sq`` / ``area_cap_f_per_um2``) are the
# physical facts single-sourced above; token / layer fields stay local literals.
_PDK_LVS: Dict[str, Dict[str, Any]] = {
    "gf180mcu": {
        "layermap": _GF180_LAYERMAP,
        "devices": _GF180_DEVICES,
        "mim": _GF180_MIM,
        "res": _GF180_RES,
        "convention": "synth",
        # bulk_measured (Task B4): derived FACT, not a free knob -- SAME test as
        # klayout_deck_gen.py:926-931's own ``implant = rn is not None and rp is
        # not None`` gate: nplus/pplus are BOTH absent from _GF180_LAYERMAP above
        # (no "nplus"/"pplus" key), so the synthesized deck emits a 3-terminal
        # (implant-less) MOS extraction -- the bulk/body terminal is never
        # independently measured. False here MUST equal
        # ("nplus" in layermap and "pplus" in layermap) for this entry; a
        # consistency test in test_lvs_verdict_scope.py asserts it.
        "bulk_measured": False,
    },
    # open_pdk: implant-based MOS recognition (layers VERBATIM from the foundry
    # external_drc LVS deck) + the FULL 6LM routing stack + MIM/poly-res recognition:
    #   * MIM: the external_lvs recipe (extract.rul header) is ``CTMn AND METALn`` --
    #     the external_pdk-style ENCLOSE form (oversize 0): top marker CTM5 = 67:5
    #     (layermap line 101), bottom plate = METAL5, riser VIA5 -> METAL6 cover.
    #     Density 1.025 fF/um^2 (external_model.scs line 16764), model 'mimcap'.
    #   * poly res: R[NR] = RPDMY*POLY1*NIMP*RPO (extract.rul header); SAB = RPO
    #     34/0, marker = RPDUMMY 54/0 (layermap lines 51/82); sheet 292 ohm/sq
    #     (hspice external_model.l line 17882 ``r_rnpo1rpo = 292``), model 'npolyf_u'.
    # open_pdk: implant-based MOS recognition (SN 40 / SP 43 split the AA active,
    # so the deck extracts full 4-terminal MOS) + the external_pdk 0.18 MIM10 routing stack
    # M1-M6 / V1-V5 + MIM cap + poly-resistor recognition. Every layer is cited to
    # open_pdk's OWN external_drc LVS deck (external_pdk.11_0.lvs):
    #   AA 10 (lvs:462)  NW 14 (lvs:463)  GT 30 (lvs:472)  SN 40 (lvs:474)
    #   SP 43 (lvs:475)  CT 50 (lvs:479)  M1-M6 61-66 (lvs:481-486)
    #   V1-V5 70-74 (lvs:494-498)  MIM 58 (lvs:480)  SAB 48 (lvs:478)
    # RESP1 is streamed on GDS 96/0 and the deck maps it to private work layer 960
    # (lvs:512-513). Device classes are
    # the 1.8 V core subckts (nmos_3p3 / pmos_3p3); the reference "external_pdk" convention
    # declares the same class tokens so KLayout pairs them by name.
    # open_pdk: its own stream map / external_drc LVS facts. Core MOS devices are
    # flat n18/p18 models; the MIM/resistor device classes match the sibling
    # process but are resolved independently above.
}

# open_pdk MIM top-plate markers CTM2..CTM5 (GDS 67:n): derived from the single
# shared leaf table (external_pdk + external_pdk) rather than hardcoded,
# so the multi-level MIM recognition (_TSMC_MIM.levels) resolves every level and
# the '67' constant lives in exactly one place. external_lvs extract.rul recognizes the
# MIM at each of these (C[M2]=CTM2*METAL2 .. C[M5]=CTM5*METAL5).


# --------------------------------------------------------------------------- #
# verdict object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, init=False)
class LvsOracleCertificate:
    """Oracle-private immutable replay of one KLayout extraction/compare run."""

    gds_sha256: str
    ref_sha256: str
    deck_sha256: str
    lvsdb_sha256: str
    extracted_netlist_sha256: str
    full_extraction: bool
    returncode: int
    ran: bool
    verbatim: str
    xref_clean: bool
    unmatched_nets: Tuple[str, ...]
    unmatched_devices: Tuple[str, ...]
    matched: bool

    def _validate(self) -> None:
        for name in (
            "gds_sha256",
            "ref_sha256",
            "deck_sha256",
            "lvsdb_sha256",
            "extracted_netlist_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"LVS oracle certificate has malformed {name}")
        if type(self.returncode) is not int:
            raise TypeError("LVS oracle certificate returncode must be exact int")
        if type(self.full_extraction) is not bool:
            raise TypeError(
                "LVS oracle certificate full_extraction must be exact bool"
            )
        for name in ("ran", "xref_clean", "matched"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"LVS oracle certificate {name} must be bool")
        tool_match = (
            "Netlists match" in self.verbatim
            and "Netlists don't match" not in self.verbatim
        )
        expected = bool(
            self.returncode == 0 and self.ran and tool_match and self.xref_clean
        )
        if self.matched != expected:
            raise ValueError("LVS oracle certificate outcome does not replay")
        if self.xref_clean and (self.unmatched_nets or self.unmatched_devices):
            raise ValueError("LVS oracle certificate xref is internally inconsistent")


def _mint_lvs_oracle_certificate(
    *,
    gds: Path,
    ref: Path,
    deck: Path,
    lvsdb: Path,
    extracted_netlist: Path,
    full_extraction: bool,
    returncode: int,
    ran: bool,
    combined_output: str,
    xref_clean: bool,
    unmatched: Mapping[str, Sequence[str]],
) -> LvsOracleCertificate:
    """Minted only at the subprocess boundary after xref parsing."""
    from clay.layout.route.truth_manifest import sha256_path

    for path, what in (
        (gds, "GDS"),
        (ref, "reference"),
        (deck, "deck"),
        (lvsdb, "lvsdb"),
        (extracted_netlist, "extracted netlist"),
    ):
        if not Path(path).is_file() or Path(path).stat().st_size <= 0:
            raise ValueError(f"LVS oracle cannot certify missing/empty {what}")
    verbatim = _verbatim_verdict(combined_output)
    tool_match = "Netlists match" in verbatim and "Netlists don't match" not in verbatim
    certificate = object.__new__(LvsOracleCertificate)
    for name, value in (
        ("gds_sha256", sha256_path(Path(gds))),
        ("ref_sha256", sha256_path(Path(ref))),
        ("deck_sha256", sha256_path(Path(deck))),
        ("lvsdb_sha256", sha256_path(Path(lvsdb))),
        (
            "extracted_netlist_sha256",
            sha256_path(Path(extracted_netlist)),
        ),
        ("full_extraction", full_extraction),
        ("returncode", int(returncode)),
        ("ran", bool(ran)),
        ("verbatim", verbatim),
        ("xref_clean", bool(xref_clean)),
        ("unmatched_nets", tuple(str(v) for v in unmatched.get("nets", ()))),
        ("unmatched_devices", tuple(str(v) for v in unmatched.get("devices", ()))),
        ("matched", bool(int(returncode) == 0 and ran and tool_match and xref_clean)),
    ):
        object.__setattr__(certificate, name, value)
    certificate._validate()
    return certificate


@dataclass
class LvsVerdict:
    """The result of one independent KLayout LVS gate run.

    Attributes
    ----------
    matched:
        ``True`` ONLY when KLayout's ``compare`` printed ``Netlists match`` AND the
        cross-reference circuit status is a clean match (no unmatched net or device).
        Never set ``True`` without both observations.
    verbatim:
        The verbatim verdict line(s) KLayout printed (``INFO : ... Netlists match.``
        or ``ERROR : Netlists don't match``) -- reported as-is, never paraphrased.
    extracted_devices:
        The device cards KLayout EXTRACTED from the routed layout (one string per
        device), proof the extractor ran on real geometry.
    unmatched:
        On a non-match, the honest unmatched list: ``{"nets": [...], "devices": [...]}``
        where each entry is ``"<status> layout=<a> schem=<b>"`` from the lvsdb xref.
        Empty when matched.
    ran:
        ``True`` iff KLayout exited successfully after producing both extraction
        and a decisive compare database/verdict.  Extraction alone is incomplete.
    returncode / stdout_tail:
        Raw tool evidence (kept for auditing; never spun).
    gds_path / ref_path / deck_path / lvsdb_path:
        The artifacts written, for inspection.
    """

    matched: bool
    verbatim: str
    extracted_devices: List[str] = field(default_factory=list)
    unmatched: Dict[str, List[str]] = field(default_factory=lambda: {"nets": [], "devices": []})
    ran: bool = False
    returncode: Optional[int] = None
    stdout_tail: str = ""
    # Public emitted artifact identity.  KLayout consumes ``oracle_gds_path``;
    # keeping both prevents a private snapshot from masquerading as the public
    # exact/diagnostic output in downstream provenance.
    gds_path: Optional[str] = None
    gds_sha256: Optional[str] = None
    oracle_gds_path: Optional[str] = None
    oracle_gds_sha256: Optional[str] = None
    ref_path: Optional[str] = None
    ref_sha256: Optional[str] = None
    deck_path: Optional[str] = None
    deck_sha256: Optional[str] = None
    lvsdb_path: Optional[str] = None
    lvsdb_sha256: Optional[str] = None
    extracted_netlist_path: Optional[str] = None
    extracted_netlist_sha256: Optional[str] = None
    # Compatibility publications are mutable shared names for human/tools that
    # predate run-private oracle workspaces.  They are never authoritative
    # signoff inputs; provenance above continues to point at retained private
    # bytes certified by this run.
    published_lvsdb_path: Optional[str] = None
    published_extracted_netlist_path: Optional[str] = None
    full_extraction: bool = False
    oracle_certificate: Optional[LvsOracleCertificate] = None
    # Optional exact-artifact diagnostic.  ``None`` means the caller did not ask
    # for the additional checker, preserving the historical LVS-only contract.
    via_provenance_report: Optional[Any] = None

    @property
    def gate_passed(self) -> bool:
        """Combined opt-in gate without changing the real KLayout verdict.

        ``matched`` remains the independent KLayout+xref result.  When an exact-run
        via-provenance report is attached, only its strict ``pass`` state certifies
        the combined gate; unmeasurable/unsupported/advisory results fail closed.
        """

        report = self.via_provenance_report
        return bool(self.matched and (report is None or bool(getattr(report, "ok", False))))

    def summary(self) -> str:
        n_nets = len(self.unmatched.get("nets", []))
        n_dev = len(self.unmatched.get("devices", []))
        if self.matched:
            return "LVS MATCH (KLayout compare clean; xref fully paired)"
        if not self.ran:
            if self.extracted_netlist_path:
                return (
                    "LVS COMPARISON INCOMPLETE (extraction produced, but no "
                    "decisive compare report)"
                )
            return "LVS DID NOT RUN (no extracted netlist produced by KLayout)"
        return (f"LVS NO-MATCH ({n_nets} unmatched net(s), "
                f"{n_dev} unmatched device(s))")


def attach_via_provenance(
    verdict: LvsVerdict,
    res: Any,
    *,
    top_cell: Optional[str] = None,
) -> LvsVerdict:
    """Attach per-via provenance for the exact GDS already judged by LVS.

    The live :class:`RouteCellResult` supplies the canonical net-tagged route
    manifest through ``to_routed_cell()``.  Geometry and lower-net identity still
    come from ``verdict.gds_path``; this helper never emits or reconstructs a second
    artifact.  Checker/integration failures are retained as an honest
    ``unmeasurable`` report and never rewrite ``verdict.matched``.
    """

    from clay.layout.route.via_provenance_check import (
        ViaProvenanceReport,
        check_via_provenance,
        terminal_claims_from_pins,
    )

    try:
        if not verdict.gds_path:
            raise ValueError("LVS verdict has no emitted GDS path")
        tagged = tuple(res.to_routed_cell().tagged)
        terminal_claims = terminal_claims_from_pins(res.pins_by_net, res.tech)
        report = check_via_provenance(
            Path(verdict.gds_path),
            res.tech,
            tagged=tagged,
            terminal_claims=terminal_claims,
            top_cell=top_cell,
        )
    except Exception as exc:  # checker evidence must not erase real LVS evidence
        report = ViaProvenanceReport(
            status="unmeasurable",
            reason=f"exact-run via provenance integration failed: {exc}",
        )
    verdict.via_provenance_report = report
    return verdict


# --------------------------------------------------------------------------- #
# reference-side passive bank fold (the combine_devices mirror, passive axis)
# --------------------------------------------------------------------------- #
def _fold_passive_banks(intent: Any) -> Any:
    """Fold every m-unit parallel PASSIVE bank in ``intent`` to ONE schematic
    device, mirroring what KLayout's ``combine_devices`` does to the DRAWN bank
    (the signed test4 netlist's ``CC1 ... 830.76f m=2`` draws two CTM unit
    plates -> KLayout folds them to one C of 2x the unit value). A parallel cap
    bank folds to ``C*m``; a parallel res bank to ``R/m``; ``M`` becomes 1.

    This is the PASSIVE axis of the gate's existing fold contract -- the MOS
    axis already folds via ``RefNetlistSpec.fold_parallel`` (W=W*M). Without it
    the reference card undervalues an m=2 cap by 2x and the value-verified
    compare (the deck's 5% ``tolerance``) correctly refuses to pair it.

    Returns ``intent`` UNTOUCHED when no passive has M>1 (gf180 byte-stability);
    otherwise a shallow view exposing the same ``devices`` and the folded
    ``passives`` (never mutating the caller's intent).
    """
    import dataclasses
    from types import SimpleNamespace

    passives = list(getattr(intent, "passives", []) or [])
    out: List[Any] = []
    changed = False
    for p in passives:
        m = int(getattr(p, "M", 1) or 1)
        kind = str(getattr(p, "kind", "")).lower()
        if m <= 1:
            out.append(p)
            continue
        if kind.startswith("cap") or kind == "c":
            val = float(p.value) * m          # parallel C bank: C_total = m * C_unit
        elif kind.startswith("res") or kind == "r":
            val = float(p.value) / m          # parallel R bank: R_total = R_unit / m
        else:
            out.append(p)
            continue
        try:
            fp = dataclasses.replace(p, value=val, M=1)
        except TypeError:
            fp = SimpleNamespace(
                name=p.name, kind=p.kind, value=val, nets=dict(p.nets),
                M=1, model=str(getattr(p, "model", "") or ""),
            )
        out.append(fp)
        changed = True
    if not changed:
        return intent
    return SimpleNamespace(
        devices=getattr(intent, "devices", ()),
        passives=out,
        bjts=getattr(intent, "bjts", ()),
    )


# --------------------------------------------------------------------------- #
# GDS emission (full labelled layout) -- leakage-free
# --------------------------------------------------------------------------- #
_KDB_POINT = kdb.Point
_KDB_POLYGON = kdb.Polygon


def _poly(lp) -> kdb.Polygon:
    pts = _KDB_POINT
    return _KDB_POLYGON(
        [
            pts(v.x if type(v.x) is int else int(v.x), v.y if type(v.y) is int else int(v.y))
            for v in lp.polygon.vertices
        ]
    )


class _PassiveHealStamp:
    """A ``polys_by_layer`` view (centroid-relative um rings) of an emitted
    parametric passive cell, so the same-net M1 heal/collar OWNS the cell's M1
    head pads exactly like a placed device stamp. Without it the resistor pads
    are unowned M1 -> FOREIGN to their own net, and the heal clips its bridges
    away from the net's own terminal pad."""

    def __init__(self, cell: Any, cx_nm: float, cy_nm: float) -> None:
        by: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
        for lp in getattr(cell, "polygons", []) or []:
            key = (int(lp.layer), int(lp.datatype))
            ring = [((v.x - cx_nm) / _NM, (v.y - cy_nm) / _NM)
                    for v in lp.polygon.vertices]
            by.setdefault(key, []).append(ring)
        self.polys_by_layer = by


def _close_same_net_upper_metal_extras(
    routed, tech, *, min_gi: int = 2,
) -> Dict[str, Dict[Tuple[int, int], "kdb.Region"]]:
    """Close sub-min-space NOTCHES within each net's OWN metal on the UPPER routing
    layers (M2 and above), returning only the gap-filling material per
    ``{net: {(layer, datatype): kdb.Region}}`` -- clipped clear of every OTHER net's
    metal on that same layer so no cross-net short is created.

    Why: a net's own trunk + via landing pad on M2 can leave a re-entrant gap < the
    PDK same-layer min-space (the gf180 comparator's 6 di_n/di_p M2.2a SAME-NET
    notches: a di_p wire 280nm wide and its di_p via pad 380nm wide, 230nm < 280nm
    apart, BOTH di_p). A net's own metal can always be merged (it cannot short
    itself), so closing the gap is DRC-correct AND strengthens connectivity. The
    closing is per-net + per-layer and every bridge is subtracted from a min-space
    halo around the FOREIGN-on-that-layer metal (other nets' merged regions) so a
    same-net fill never bridges a different net. Pure region computation; nothing is
    inserted here. Generic + leakage-free: every dimension is the PDK min-space, no
    per-circuit / per-net / per-PDK branch -- the dual of the M1 heal across the
    upper stack. ``min_gi`` is the lowest graph layer to process (2 -> M2; M1 is the
    device-interface heal's job).

    NO-VIOLATION, NO-FILL obligation: the deck's space() flags gaps STRICTLY
    below the rule, but the closing radius ``(s+1)//2`` also merges a pair at
    EXACTLY min-space (the dilated edges touch), and the erosion then leaves a
    seam whose thickness is the pair's projection overlap minus one min-space --
    fabricated metal bridging deck-LEGAL geometry, and a sub-min-width neck
    whenever that overlap is thin. Measured escape (ref-netlist experiment,
    open_pdk real deck M2.W.1 x1): a gnd via1 pad and the gnd trunk sat 280nm
    apart (legal), the seam bridged them 280x210 -- an offset-overlap neck on an
    otherwise clean layout. So each candidate fill component is kept ONLY when
    it closes a REAL euclidean sub-min-space violation of the net's own metal
    (adjudicated by the same ``space_check`` metric family the deck runs);
    everything else is not obligated, and drawing it can only create defects."""
    from clay.layout.route.emit_gds import net_layer_regions
    from clay.layout.route.wide_metal_spacing import wide_spacing_additions

    out: Dict[str, Dict[Tuple[int, int], kdb.Region]] = {}
    regions = net_layer_regions(routed.nets)  # {net: {(l,dt): merged Region}}

    # the routing layers at/above ``min_gi`` (1-based metal index) and their
    # per-layer min-space (nm). Probe metal2..metalN until the PDK has no more.
    layer_space: Dict[Tuple[int, int], int] = {}
    layer_names: Dict[Tuple[int, int], str] = {}
    for n in range(min_gi, 32):
        try:
            key = tuple(tech.metal_key(n))
        except KeyError:
            break
        try:
            sp = int(tech.min_space("metal%d" % n) or 0)
        except Exception:
            sp = 0
        if sp > 0:
            layer_space[key] = sp
            layer_names[key] = f"metal{n}"

    # the merged FOREIGN region per layer, keyed by the net it is foreign TO, is
    # ``all_on_layer - own_on_layer`` -- so build the per-layer union once.
    all_by_layer: Dict[Tuple[int, int], kdb.Region] = {}
    for net, by_layer in regions.items():
        for key, reg in by_layer.items():
            if key not in layer_space:
                continue
            all_by_layer.setdefault(key, kdb.Region())
            all_by_layer[key] += reg
    for key in all_by_layer:
        all_by_layer[key].merge()

    for net in sorted(regions):
        by_layer = regions[net]
        for key, s in layer_space.items():
            own = by_layer.get(key)
            if own is None or own.is_empty():
                continue
            half = (s + 1) // 2
            # morphological CLOSING (dilate then erode by half a min-space) fills
            # any same-net gap/notch up to one min-space wide.
            want = own.dup()
            want = want.sized(half).sized(-half)
            cand = (want - own).merged()
            foreign = (all_by_layer.get(key, kdb.Region()) - own)
            if cand.is_empty():
                extra = kdb.Region()
            else:
                clip_reg = kdb.Region(cand.bbox().enlarged(2 * s + 10))
                foreign_local = (foreign & clip_reg).merged()
                # the NEW fill material only, clipped a full min-space clear of foreign
                # metal on this layer (a same-net fill must never bridge another net).
                extra = (cand - foreign_local.sized(s)).merged()
                if not extra.is_empty():
                    # NO-VIOLATION, NO-FILL (see docstring): keep only the fill
                    # components that close a REAL strictly-sub-min-space euclidean
                    # violation of the net's own metal; the touch-merge seam of a pair
                    # at EXACTLY min-space is fabricated metal on legal geometry (the
                    # measured ref-experiment M2.W.1 offset-overlap neck) and is
                    # dropped. ``polygons()`` materialises each violating edge pair's
                    # gap material, so intersection tests the fill against the gaps
                    # the deck actually flags.
                    viol = own.space_check(s, False, kdb.Region.Euclidian,
                                           None, None, None).polygons(0)
                    justified = kdb.Region()
                    if not viol.is_empty():
                        viol_bb = viol.bbox()
                        for c in extra.each():
                            if not c.bbox().overlaps(viol_bb):
                                continue
                            cr = kdb.Region(c)
                            if not (cr & viol).is_empty():
                                justified += cr
                    extra = justified.merge()
            # The base rule intentionally leaves legal 300nm gaps alone, but
            # attached metal inside a wide rail's halo owes the declared wider
            # rule. Reuse its exact merged-geometry witness to justify additions.
            extra += wide_spacing_additions(own, foreign, tech, layer_names[key])
            extra.merge()
            if extra.is_empty():
                continue
            out.setdefault(net, {})[key] = extra
    return out


def _exclude_existing_cell_material_from_upper_extras(
    extras: Mapping[str, Mapping[Tuple[int, int], Any]],
    top: Any,
    ly: Any,
) -> Dict[str, Dict[Tuple[int, int], "kdb.Region"]]:
    """Keep only upper-close material that changes the assembled cell union.

    ``_close_same_net_upper_metal_extras`` reasons over allocated route geometry.
    A route-local notch may already be completely covered by a materialized
    passive/device plate that is present in ``top`` before route insertion.  In
    that case inserting the close again creates only a duplicate polygon, not
    new conductor, and must not be reported as an emitter transformation.
    Subtract only the pre-existing cell artwork at this point in the pipeline;
    routed metal has not yet been inserted, so route ownership remains exact.
    """
    import klayout.db as kdb

    existing_by_layer: Dict[Tuple[int, int], kdb.Region] = {}
    out: Dict[str, Dict[Tuple[int, int], kdb.Region]] = {}
    for net, by_layer in sorted(extras.items()):
        for raw_key, region in sorted(by_layer.items()):
            key = (int(raw_key[0]), int(raw_key[1]))
            existing = existing_by_layer.get(key)
            if existing is None:
                existing = kdb.Region(top.shapes(ly.layer(*key))).merged()
                existing_by_layer[key] = existing
            new_material = (region - existing).merged()
            if not new_material.is_empty():
                out.setdefault(str(net), {})[key] = new_material
    return out


def _heal_same_net_m1(
    top, ly, routed, geometry, placement, pins_by_net, tech
) -> Tuple[int, Dict[str, "kdb.Region"]]:
    """Compatibility wrapper (the unmerged ``emit_labelled_gds(merge=False)`` path):
    compute the per-net heal material with :func:`_heal_same_net_m1_extras` on the
    cell's CURRENT M1 (device artwork + raw wires already inserted) and insert the
    bridge polygons as extra shapes. Returns both the number emitted and the
    original per-net ownership regions consumed by the final write gate."""
    m1 = tuple(tech.metal_key(1))
    m1_layer = ly.layer(int(m1[0]), int(m1[1]))
    all_m1 = kdb.Region(top.shapes(m1_layer))
    all_m1.merge()
    extras = _heal_same_net_m1_extras(
        all_m1, routed, geometry, placement, pins_by_net, tech)
    emitted = 0
    for net in sorted(extras):
        for poly in extras[net].each():
            top.shapes(m1_layer).insert(poly)
            emitted += 1
    return emitted, extras


def _local_m1_interface_violation_region(
    device_m1: "kdb.Region",
    routed_m1: "kdb.Region",
    *,
    min_space_nm: int,
    min_width_nm: int,
    full_routed_m1: Optional["kdb.Region"] = None,
) -> "kdb.Region":
    """Return actual FEOL/BEOL M1.S/W violation markers localized to ``near``."""
    device = device_m1.dup()
    route = routed_m1.dup()
    device.merge()
    route.merge()
    violations = kdb.Region()
    if device.is_empty() or route.is_empty():
        return violations

    reach = max(1, int(min_space_nm), int(min_width_nm))
    local_window = (device.sized(reach) & route.sized(reach)).merged()
    if local_window.is_empty():
        return violations

    joined = (device + route).merged()
    joined_full = None
    if full_routed_m1 is not None:
        full_route = full_routed_m1.dup()
        full_route.merge()
        joined_full = (device + full_route).merged()

    if min_space_nm > 0:
        spacing_pairs = joined.space_check(
            int(min_space_nm),
            False,
            kdb.Region.Euclidian,
            None,
            None,
            None,
        ).polygons(0)
        spacing_local = spacing_pairs & local_window
        if not spacing_local.is_empty():
            if joined_full is None:
                violations += spacing_local
            else:
                spacing_full = joined_full.space_check(
                    int(min_space_nm),
                    False,
                    kdb.Region.Euclidian,
                    None,
                    None,
                    None,
                ).polygons(0)
                violations += (spacing_local & spacing_full)

    if min_width_nm > 0:
        width_pairs = joined.width_check(
            int(min_width_nm),
            False,
            kdb.Region.Euclidian,
            None,
            None,
            None,
        ).polygons(0)
        width_local = width_pairs & local_window
        if not width_local.is_empty():
            if joined_full is None:
                violations += width_local
            else:
                width_full = joined_full.width_check(
                    int(min_width_nm),
                    False,
                    kdb.Region.Euclidian,
                    None,
                    None,
                    None,
                ).polygons(0)
                violations += (width_local & width_full)
    violations.merge()
    return violations


def _local_m1_interface_needs_collar(
    device_m1: "kdb.Region",
    routed_m1: "kdb.Region",
    *,
    min_space_nm: int,
    min_width_nm: int,
    full_routed_m1: Optional["kdb.Region"] = None,
) -> bool:
    """Whether one actual FEOL/BEOL contact has a local M1.S/W obligation.

    A terminal and a routed fragment can have very different outer bboxes while
    already sharing a deck-legal contact (a stepped terminal or a guard-frame
    tap).  Rectangularizing that bbox pair invents metal and can fill intrinsic
    holes.  Collar material is justified only when the *actual* local geometry
    has either a strictly sub-spacing gap/notch or a sub-width contact/sliver.

    When ``full_routed_m1`` is supplied (the merged union of all router-owned
    M1 fragments on the net), a marker seen in the local ``near`` subset only
    justifies collar emission if it overlaps an actual violation in
    ``(device + full_routed_m1).merged()``.  Otherwise decomposing an emitted
    Pass-1 collar into rectangles causes Pass 2's fragment filter to drop
    distant split sub-rectangles while keeping adjacent flanges, synthesizing a
    non-existent sliver/gap.

    The decision is purely geometric and PDK-derived.  Both ``space_check`` and
    ``width_check`` measure the actual merged local interface.  In particular,
    the width of the operands' overlap is not itself a conductor-width rule: two
    wide rectangles may overlap by a narrow strip yet merge into one completely
    legal rectangle.  The grown intersection is only a locality window; it
    never contributes material.
    """
    return not _local_m1_interface_violation_region(
        device_m1,
        routed_m1,
        min_space_nm=min_space_nm,
        min_width_nm=min_width_nm,
        full_routed_m1=full_routed_m1,
    ).is_empty()


def _heal_same_net_m1_extras(
    all_m1, routed, geometry, placement, pins_by_net, tech,
) -> Dict[str, "kdb.Region"]:
    """Close sub-min-space gaps + thin necks WITHIN each net's M1 by a per-net
    morphological CLOSING, returning only the gap-filling bridge material (clipped
    clear of every OTHER net's M1 so no short is created) as ``{net: kdb.Region}``.

    PURE region computation: ``all_m1`` is the merged region of EVERY M1 shape
    destined for the cell (device artwork + cap cells + all routed wires); nothing
    is inserted here. The caller either unions each net's material into that net's
    emit-boundary merge (:func:`emit_labelled_gds` ``merge=True`` -- the P1 path) or
    inserts the polygons verbatim (the ``merge=False`` wrapper above). Same regions
    either way -- the DRC/LVS verdicts cannot depend on the fragmentation.

    Why this exists: the device-terminal M1 (``build_device``, placed off the routing
    grid) and the router's on-grid landing pad are the SAME net but stop ~130nm short of
    each other -- a sub-min-width neck (M1.W.1) or sub-min-space gap (M1.S.1) right at the
    wire-to-device-pad interface (the external_pdk bring-up's 20/24 DRC violations; the gf180
    "12" are the same class). A net's own M1 may always be merged (it cannot short
    itself), so closing the gap is both DRC-correct AND strengthens connectivity. The
    closing is per-net and the bridges are subtracted from a min-space halo around every
    FOREIGN net's M1 -- including the heal material already GRANTED to other nets
    (contract (4), tests/test_terminal_interface.py: two collars growing into a shared
    corridor must stay min-space apart from EACH OTHER, not merely from each other's
    pre-heal metal) -- and width-legalized on the merged island (contract (5): no
    heal-added sub-min-width protrusion, unless removing it would split the island)."""
    m1 = tuple(tech.metal_key(1))
    try:
        s = int(tech.min_space("metal1") or 0)
    except Exception:
        s = 0
    if s <= 0:
        return {}
    try:
        min_w = int(tech.min_width("metal1") or 0)
    except Exception:
        min_w = 0
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    grid_nm = max(1, int(pdk_manufacturing_grid_nm(tech) or 1))
    half = (s + 1) // 2  # close gaps up to one min-space (nm == dbu here, dbu=0.001um)
    # One terminal can seed a multi-polygon physical conductor (notably a guard
    # ring).  Reuse the exact same-layer component ownership currency as the
    # emitted connectivity observer instead of owning only the source polygon
    # that happens to contain the pin.
    from clay.layout.route.route_cell import _seeded_geometry_metal_regions

    seeded_m1_by_net = _seeded_geometry_metal_regions(
        geometry,
        placement,
        pins_by_net,
        {m1},
    )

    seeded_comps_by_net: Dict[str, List[Tuple[Any, bool, kdb.Polygon]]] = {}
    for s_net, by_layer in seeded_m1_by_net.items():
        s_reg = by_layer.get(m1)
        if s_reg is not None and not s_reg.is_empty():
            seeded_comps_by_net[s_net] = [
                (comp.bbox(), comp.is_box(), comp) for comp in s_reg.each()
            ]

    stamp_m1_cache: Dict[Tuple[int, Any], List[Tuple[Any, bool, kdb.Polygon]]] = {}

    def _stamp_m1_polys(g, pos) -> List[Tuple[Any, bool, kdb.Polygon]]:
        cache_key = (id(g), pos)
        cached = stamp_m1_cache.get(cache_key)
        if cached is not None:
            return cached
        polys: List[Tuple[Any, bool, kdb.Polygon]] = []
        for ring in (getattr(g, "polys_by_layer", {}) or {}).get(m1, []):
            if len(ring) < 3:
                continue
            pts = [_KDB_POINT(x, y) for x, y in placed_ring_nm(ring, pos)]
            poly = _KDB_POLYGON(pts)
            polys.append((poly.bbox(), poly.is_box(), poly))
        stamp_m1_cache[cache_key] = polys
        return polys

    def _pad_in_stamp(g, pos, pinx: int, piny: int) -> Optional[kdb.Polygon]:
        """The stamp's M1 polygon (absolute nm) containing probe around (pinx, piny), or None."""
        probe = None
        for bb, is_box, poly in _stamp_m1_polys(g, pos):
            if bb.left < pinx + 1 and bb.right > pinx - 1 and bb.bottom < piny + 1 and bb.top > piny - 1:
                if is_box:
                    return poly
                if probe is None:
                    probe = kdb.Region(kdb.Box(pinx - 1, piny - 1, pinx + 1, piny + 1))
                if not (kdb.Region(poly) & probe).is_empty():
                    return poly
        return None

    def _dev_m1_for_pin(p, net: str) -> Optional[kdb.Polygon]:
        pinx = int(round(float(p.xy_um[0]) * _NM)); piny = int(round(float(p.xy_um[1]) * _NM))
        seeded_comps = seeded_comps_by_net.get(net)
        if seeded_comps:
            probe = None
            for bb, is_box, component in seeded_comps:
                if bb.left < pinx + 1 and bb.right > pinx - 1 and bb.bottom < piny + 1 and bb.top > piny - 1:
                    if is_box:
                        return component
                    if probe is None:
                        probe = kdb.Region(kdb.Box(pinx - 1, piny - 1, pinx + 1, piny + 1))
                    if not (kdb.Region(component) & probe).is_empty():
                        return component
        # fast path: the stamp keyed by the pin's device name.
        dev = getattr(p, "device", None)
        pos = placement.get(dev) if dev is not None else None
        g = geometry.get(dev) if dev is not None else None
        if pos is not None and g is not None:
            poly = _pad_in_stamp(g, pos, pinx, piny)
            if poly is not None:
                return poly
        # CONTAINMENT fallback: the pin's pad is, by the pin-extraction
        # contract, the placed M1 covering the pin point -- but the hosting
        # stamp's dict key need NOT be the pin's device name (a composite
        # matched-group stamp is keyed by the GROUP, its member pins by the
        # MEMBER). Probe every placed stamp geometrically. Without this the
        # heal/collar never saw composite member junctions (measured: a 15nm
        # wire-to-pad gap at MM6.d -> the $17<->MM6 LVS device mismatch).
        for name, g2 in geometry.items():
            if name == dev:
                continue
            pos2 = placement.get(name)
            if pos2 is None:
                continue
            poly = _pad_in_stamp(g2, pos2, pinx, piny)
            if poly is not None:
                return poly
        return None

    # every M1 polygon now in the cell, by net (routed wires + each pin's device pad).
    # ``own_by_net`` keeps the net's TRUE metal (pre-collar): the foreign-clip basis
    # below MUST be ownership-derived. Computing foreign as ``all_m1 - collar'd reg``
    # erodes a foreign pad by whatever strip the collar box swallowed, so the
    # min-space clip stops (min_space - overlap) from the TRUE pad edge (the
    # v0.4.27 external_pdk M1.S.1: a 30nm-eroded pad left heal material 200nm -- not
    # 230nm -- from the real foreign edge).
    per_net: Dict[str, Tuple[kdb.Region, kdb.Region]] = {}
    own_by_net: Dict[str, kdb.Region] = {}
    for net, nr in routed.nets.items():
        reg = kdb.Region()
        has_routed_m1 = False
        routed_m1_frag_bbs: List[Tuple[kdb.Polygon, Any]] = []
        routed_m1_full = kdb.Region()
        for lp in nr.wires:
            if (int(lp.layer), int(lp.datatype)) == m1:
                polygon = _poly(lp)
                reg.insert(polygon)
                routed_m1_frag_bbs.append((polygon, polygon.bbox()))
                routed_m1_full.insert(polygon)
                has_routed_m1 = True
        # A leaf terminal (or a COMPONENT_ONLY program) has no FEOL-to-BEOL
        # interface to heal.  Growing its device pad here invents route-owned
        # metal outside the exact allocation manifest and can change extracted
        # connectivity.  Heal only nets that actually carry router-owned M1.
        if not has_routed_m1:
            continue
        routed_m1_full.merge()
        seeded = seeded_m1_by_net.get(net, {}).get(m1)
        if seeded is not None:
            for component in seeded.each():
                reg.insert(component)
        reg.merge()
        if reg.is_empty():
            continue
        own_by_net[net] = reg.dup()
        interface = kdb.Region()
        # COLLAR each terminal junction: the routed landing pad / wire (on the routing
        # grid, often the wider via-enclosure width) and the device-terminal M1 (off the
        # routing grid, min-width) are offset, so one OVERHANGS the other as a sub-min-
        # width EAR (a protrusion the closing cannot fix). Widen the LOCAL junction band
        # to the UNION x-/y-extent of the device pad and the nearby routed M1, so neither
        # overhangs -- the ear becomes part of a >=min-width region, and the step down to
        # the bare min-width stem becomes a concave NOTCH the closing below then fills.
        for p in pins_by_net.get(net, []):
            dp = _dev_m1_for_pin(p, net)
            if dp is None:
                continue
            pinx = int(round(float(p.xy_um[0]) * _NM)); piny = int(round(float(p.xy_um[1]) * _NM))
            cap_bb = kdb.Box(pinx - 500 - s, piny - 2000 - s, pinx + 500 + s, piny + 2000 + s)
            db_ = dp.bbox()
            int_bb = kdb.Box(db_.left - s, db_.bottom - s, db_.right + s, db_.top + s)
            # Fast bbox prefilter before constructing KLayout Regions.
            cand_frags = [
                fragment
                for fragment, fbb in routed_m1_frag_bbs
                if fbb.overlaps(cap_bb) and fbb.overlaps(int_bb)
            ]
            if not cand_frags:
                continue
            # capture box: TALL + NARROW. A wire-to-pad junction can be a LONG thin finger
            # (the routed wire stays offset for its whole vertical run to the device pad),
            # so the box must reach far in y to widen the whole finger -- but stay narrow
            # in x so it does not pull in a DIFFERENT net hugging this terminal (a wide box
            # collars into that foreign net and trips spacing). The vertical routing bias
            # makes tall-narrow the right shape; this is what takes the external_pdk route to a
            # fully DRC-clean M1 (the last thin finger had free space on one side).
            capture_reach = kdb.Region(cap_bb)
            # The capture box is a predicate only.  Select complete
            # router-owned fragments that intersect it; the artificial box
            # boundary must never become a new collar edge.  Intrinsic
            # same-net device/ring metal is not a routed interface.
            near = kdb.Region()
            device_component = kdb.Region(dp)
            interface_reach = device_component.sized(s)
            for fragment in cand_frags:
                fragment_region = kdb.Region(fragment)
                if (
                    not (fragment_region & capture_reach).is_empty()
                    and not (fragment_region & interface_reach).is_empty()
                ):
                    near.insert(fragment)
            near.merge()
            if near.is_empty():
                continue
            viol_region = _local_m1_interface_violation_region(
                device_component,
                near,
                min_space_nm=s,
                min_width_nm=min_w,
                full_routed_m1=routed_m1_full,
            )
            if viol_region.is_empty():
                continue
            interface.insert(dp)
            interface += near
            nb = near.bbox()
            vb = viol_region.bbox()
            cx0 = min(nb.left, db_.left); cx1 = max(nb.right, db_.right)
            # collar the JUNCTION band -- the region where the device pad and the nearby
            # routed M1 meet -- whether they OVERLAP in y or are GAP-separated (connected
            # only by a thin finger in the gap). In both cases a min-space skirt into each
            # piece is included, so the collar covers the whole thin junction. We do NOT
            # paint the full tall device stem (that would be wasteful) -- only this band.
            lo_top = min(nb.top, db_.top)
            hi_bot = max(nb.bottom, db_.bottom)
            if hi_bot > lo_top:                 # gap-separated: collar the gap + skirts
                jy0, jy1 = lo_top - s, hi_bot + s
            else:                                # overlapping: collar the overlap + skirts
                jy0, jy1 = hi_bot - s, lo_top + s
                # Clamp overlapping collar skirts to the grid-aligned violation band
                # (+-s) so an unhealable/foreign-clipped notch cannot march +-s/round
                # without snapping any Euclidean violation vertices off the PDK grid.
                vb_lo = ((vb.bottom - s) // grid_nm) * grid_nm
                vb_hi = ((vb.top + s + grid_nm - 1) // grid_nm) * grid_nm
                jy0 = max(jy0, vb_lo)
                jy1 = min(jy1, vb_hi)
            if jy1 <= jy0:
                continue
            collar = kdb.Box(cx0, jy0, cx1, jy1)
            reg.insert(collar)
            interface.insert(collar)
        reg.merge()
        interface.merge()
        per_net[net] = (reg, interface)

    try:
        min_area = int(tech.min_area("metal1") or 0)
    except Exception:
        min_area = 0
    # FOREIGN BASIS (contract: a same-net heal must clip clear of EVERY other
    # net's metal). The foreign set for net N is the UNION of every OTHER net's
    # owned M1 (plus M1 owned by no net), NOT ``all_m1 - own_by_net[N]``.
    # Why the union and not the subtraction: ``_dev_m1_for_pin`` attributes a
    # device pad to the net of its pin, but a CONTAINMENT-probed pad can span a
    # neighbouring different-net terminal (the gf180 LDO dense bias-cascode
    # column: MM13's source-pad probe swallows the interleaved VDDA bulk-tap
    # M1, ~1.55e6 nm^2). ``all_m1 - own_by_net[N]`` then ERASES that foreign
    # VDDA metal from N's foreign set, so N's closing bridges straight into it
    # (net073 -> VDDA short, L2N heal-on/off delta). Building foreign from the
    # OTHER nets' own regions keeps a net's pollution inside ITS own set only:
    # the overlapped foreign metal stays foreign to N and is clipped min-space
    # clear. Disjoint ownership (the healthy case) makes this byte-identical to
    # the old subtraction; it only differs where two nets' owned M1 overlap --
    # exactly the cross-net pollution that must never be merged.
    owned_all = kdb.Region()
    for _oreg in own_by_net.values():
        owned_all += _oreg
    owned_all.merge()
    unowned_m1 = (all_m1 - owned_all)
    unowned_m1.merge()
    unowned_bb = unowned_m1.bbox() if not unowned_m1.is_empty() else None
    own_by_net_bb = {
        _o: (_oreg, _oreg.bbox()) for _o, _oreg in own_by_net.items()
    }

    def _foreign_in_box(
        this_net: str,
        clip_box: "kdb.Box",
        extra_region: Optional["kdb.Region"] = None,
    ) -> kdb.Region:
        clip_reg = kdb.Region(clip_box)
        f = kdb.Region()
        for _o, (_oreg, _obb) in own_by_net_bb.items():
            if _o == this_net:
                continue
            if _obb.overlaps(clip_box):
                f += (_oreg & clip_reg)
        if unowned_bb is not None and unowned_bb.overlaps(clip_box):
            f += (unowned_m1 & clip_reg)
        if extra_region is not None and not extra_region.is_empty():
            if extra_region.bbox().overlaps(clip_box):
                f += (extra_region & clip_reg)
        f.merge()
        return f

    extras: Dict[str, kdb.Region] = {}
    # heal material already GRANTED to earlier nets (deterministic sorted
    # order): the same foreign currency as their true metal. Without it two
    # nets' collars growing into a shared corridor each stop min-space from
    # the other's ORIGINAL edge -- their EXTRAS end up (corridor - growth_a -
    # growth_b) apart (the canonical test4 style+trunk 4x M1.S.1: GND +25nm
    # up, net08 -25nm down into a 255nm corridor -> 205nm).
    granted = kdb.Region()
    for net in sorted(per_net):
        reg, interface = per_net[net]
        # what this net's M1 SHOULD be: its collar'd region, closed (gaps/notches filled).
        # Closing is an FEOL/BEOL *interface* repair, not permission to reshape
        # unrelated intrinsic same-net conductors.  In particular, a closed
        # guard-ring frame has concave inner corners that a whole-net closing
        # would fill even when its allocated landing is already complete.
        want = reg.dup()
        if not interface.is_empty():
            want += interface.sized(half).sized(-half)
            want.merge()

        # grow any under-min-area island of this net up to the metal min-area (the
        # reference signed-off layout has NO M1 island below it -- a real routing
        # constraint). Grow isotropically, clipped a full min-space clear of foreign, so
        # an isolated via-landing pad reaches min-area without grazing another net.
        if min_area > 0:
            for poly in reg.each():
                if poly.area() >= min_area:
                    continue
                pr = kdb.Region(poly)
                # Max expansion is 5 * 40 = 200nm; clip foreign locally once before sizing.
                fgs_local = _foreign_in_box(
                    net,
                    poly.bbox().enlarged(200 + 2 * s + 10),
                    granted,
                ).sized(s)
                grown = pr
                for _ in range(40):
                    cand = (pr.sized(5 * (_ + 1)) - fgs_local) + pr
                    cand.merge()
                    if cand.area() >= min_area:
                        grown = cand
                        break
                    grown = cand
                want += grown
        want.merge()

        # emit only the NEW material (collar + gap/notch fills + grown pads) that is not
        # already placed, clipped a full min-space clear of every other net.
        cand_new = (want - all_m1).merged()
        if cand_new.is_empty():
            continue
        fgs_local = _foreign_in_box(
            net,
            cand_new.bbox().enlarged(2 * s + 10),
            granted,
        ).sized(s)
        extra = (cand_new - fgs_local).merged()
        if extra.is_empty():
            continue
        # contract (5): EXTRAS WIDTH LEGALITY. The foreign clip can leave a
        # residual SLIVER between two keep-out halos (the test4 style+trunk
        # 62nm ear, M1.W.1) -- heal material narrower than min-width on the
        # MERGED island (morphological-opening residue; a flush gap-fill band
        # atop wide metal is NOT residue). Drop the sliver UNLESS dropping it
        # would split the merged island (then the honest width violation
        # stays for DRC to report -- never a silent open).
        extra = _drop_sub_min_width_heal_slivers(own_by_net[net], extra, min_w)
        # A repair is allowed to extend an existing conductor, never to mint a
        # second route-owned island.  Foreign clipping and min-area growth can
        # otherwise leave a candidate exactly one spacing rule away from the
        # real net (the external_pdk two-stage VDD orphan observed by the final deck).
        extra = _retain_attached_heal_material(own_by_net[net], extra)
        if extra.is_empty():
            continue
        extras[net] = extra
        granted += extra

    # contract (6): MIN-AREA FINAL RECHECK. The per-net grow above runs
    # BEFORE the final foreign clip and before later nets' extras are
    # granted, so a grown pad can come out of the pipeline notched back
    # under min-area (measured 2026-06-12: the VIN io-edge stub island at
    # the die edge, 0.188 < 0.202 um^2, 1x M1.A.1 -- the only DRC marker
    # left on the 2D-compacted canonical). Re-measure every net's FINAL
    # merged islands and re-grow the under-area ones against the FINAL
    # granted-augmented foreign; a pad that still cannot reach min-area is
    # left as an honest DRC violation (never silently dropped).
    if min_area > 0:
        for net in sorted(per_net):
            own = own_by_net[net]
            extra = extras.get(net, kdb.Region())
            merged = (own + extra).merged() if not extra.is_empty() else own
            small_polys = [poly for poly in merged.each() if poly.area() < min_area]
            if not small_polys:
                continue
            other_granted = (granted - extra) if not extra.is_empty() else granted
            added = kdb.Region()
            for poly in small_polys:
                # grow from the island's BOUNDING BOX, not the notched
                # polygon: isotropic sizing of a notched shape keeps the
                # notch step and trades the A.1 for an M1.W.1 ledge
                # (measured); the box fill heals the notch and grows as a
                # clean rectangle.
                # PER-SIDE expansion into CLEAR space only, original
                # footprint grandfathered: the island legally sits at
                # exactly min-space from foreign, so any "box minus
                # inflated-foreign" carves the island itself and the union
                # seam leaves a sub-min ledge (182nm M1.W.1, measured
                # twice). Expand the bbox one side at a time (deterministic
                # left/right/bottom/top), accept a step only when the NEW
                # strip is clear of inflated foreign, until min-area.
                pr = kdb.Region(poly)
                b = poly.bbox()
                l, bo, rr, t = b.left, b.bottom, b.right, b.top
                # Max expansion is 200 * 5 = 1000nm; clip foreign locally before sizing.
                fgs = _foreign_in_box(
                    net,
                    b.enlarged(1000 + 2 * s + 10),
                    other_granted,
                ).sized(s)
                step = 5
                for _k in range(200):
                    rect = kdb.Region(kdb.Box(l, bo, rr, t))
                    if (rect + pr).merged().area() >= min_area:
                        break
                    moved = False
                    for side in ("l", "r", "b", "t"):
                        if side == "l":
                            strip = kdb.Region(kdb.Box(l - step, bo, l, t))
                        elif side == "r":
                            strip = kdb.Region(kdb.Box(rr, bo, rr + step, t))
                        elif side == "b":
                            strip = kdb.Region(kdb.Box(l, bo - step, rr, bo))
                        else:
                            strip = kdb.Region(kdb.Box(l, t, rr, t + step))
                        if (strip & fgs).is_empty():
                            if side == "l":
                                l -= step
                            elif side == "r":
                                rr += step
                            elif side == "b":
                                bo -= step
                            else:
                                t += step
                            moved = True
                            break
                    if not moved:
                        break
                grown = (kdb.Region(kdb.Box(l, bo, rr, t)) + pr).merged()
                added += (grown - merged)
            if not added.is_empty():
                new_extra = _retain_attached_heal_material(
                    own,
                    (extra + added).merged(),
                )
                retained_added = (new_extra - extra).merged()
                if new_extra.is_empty():
                    extras.pop(net, None)
                else:
                    extras[net] = new_extra
                granted += retained_added
    return extras


def _materialize_router_owned_terminal_interface_m1(
    routed: Any,
    geometry: Mapping[str, Any],
    placement: Mapping[str, Any],
    pins_by_net: Mapping[str, Sequence[Any]],
    tech: Any,
) -> int:
    """Commit the exact M1 terminal-interface closure to its routed net.

    ``_heal_same_net_m1_extras`` is the independent final-artwork observer: it
    alone sees the placed terminal components and the final route together.
    Exact emission cannot allow that observer to author geometry, however.  A
    route engine advertising the corresponding finalization capability asks
    :func:`route_cell` to run this transaction before containment, shorts, and
    route-manifest replay.  Every returned region is decomposed into exact
    rectangles, appended to the owning ``NetRoute.wires``, and replayed through
    the same observer.  Any non-rectilinear residue, unknown owner, immutable
    wire inventory, or non-idempotent result fails closed and rolls back.

    This is coordinate-free circuit logic: all coordinates come exclusively
    from the current placed stamps, extracted pins, routed wires, and PDK rules.
    The return value is the number of newly owned route rectangles.
    """
    from clay.layout.backend import LayerPolygon, Polygon

    nets = getattr(routed, "nets", None)
    if not isinstance(nets, Mapping):
        raise TypeError("terminal-interface materialization requires routed.nets")
    m1 = tuple(tech.metal_key(1))
    if (
        type(m1) is not tuple
        or len(m1) != 2
        or any(type(value) is not int for value in m1)
    ):
        raise TypeError("terminal-interface materialization requires exact M1 authority")

    def assembled_m1() -> "kdb.Region":
        region = kdb.Region()
        for name, stamp in sorted(geometry.items(), key=lambda item: str(item[0])):
            position = placement.get(name)
            if position is None:
                continue
            layers = getattr(stamp, "polys_by_layer", None)
            if not isinstance(layers, Mapping):
                raise TypeError(
                    f"terminal-interface stamp {name!r} lacks layer mapping"
                )
            for raw_key, rings in layers.items():
                if (
                    type(raw_key) is not tuple
                    or len(raw_key) != 2
                    or any(type(value) is not int for value in raw_key)
                ):
                    raise TypeError(
                        f"terminal-interface stamp {name!r} has malformed layer key"
                    )
                if raw_key != m1:
                    continue
                for ring in rings:
                    if len(ring) < 3:
                        raise ValueError(
                            f"terminal-interface stamp {name!r} has degenerate M1"
                        )
                    points = [_KDB_POINT(x, y) for x, y in placed_ring_nm(ring, position)]
                    region.insert(_KDB_POLYGON(points))
        for net, net_route in sorted(nets.items(), key=lambda item: str(item[0])):
            wires = getattr(net_route, "wires", None)
            if type(wires) is not list:
                raise TypeError(
                    f"terminal-interface net {net!r} requires mutable exact wire list"
                )
            for wire in wires:
                if (int(wire.layer), int(wire.datatype)) == m1:
                    region.insert(_poly(wire))
        region.merge()
        return region

    def _exact_additions_for_extras(
        round_extras: Mapping[str, "kdb.Region"],
    ) -> Dict[str, List[LayerPolygon]]:
        round_additions: Dict[str, List[LayerPolygon]] = {}
        for net, extra in sorted(round_extras.items()):
            if net not in nets:
                raise ValueError(
                    f"terminal-interface material has unknown route owner {net!r}"
                )
            rectangles: List[LayerPolygon] = []
            exact_cover = kdb.Region()
            for shape in extra.decompose_trapezoids():
                polygon = kdb.Polygon(shape.polygon)
                bbox = polygon.bbox()
                polygon_region = kdb.Region(polygon)
                box_region = kdb.Region(kdb.Box(
                    bbox.left, bbox.bottom, bbox.right, bbox.top
                ))
                if not (
                    (polygon_region - box_region).is_empty()
                    and (box_region - polygon_region).is_empty()
                ):
                    raise ValueError(
                        "terminal-interface material is not exactly rectangular"
                    )
                exact_cover += box_region
                rectangles.append(LayerPolygon(
                    layer=m1[0],
                    datatype=m1[1],
                    polygon=Polygon.rectangle(
                        bbox.left, bbox.bottom, bbox.right, bbox.top
                    ),
                ))
            exact_cover.merge()
            if not (
                (extra - exact_cover).is_empty()
                and (exact_cover - extra).is_empty()
            ):
                raise ValueError(
                    "terminal-interface rectangle decomposition changed geometry"
                )
            if rectangles:
                round_additions[net] = rectangles
        return round_additions

    extras = _heal_same_net_m1_extras(
        assembled_m1(), routed, geometry, placement, pins_by_net, tech
    )
    if not extras:
        routed.terminal_interface_materialization_count = 0
        return 0

    max_heal_rounds = 4
    original_lengths: Dict[str, int] = {}
    added_count = 0
    try:
        residual = extras
        for _ in range(max_heal_rounds):
            additions = _exact_additions_for_extras(residual)
            if not additions:
                residual = _heal_same_net_m1_extras(
                    assembled_m1(), routed, geometry, placement, pins_by_net, tech
                )
                break
            for net in sorted(additions):
                wires = getattr(nets[net], "wires")
                original_lengths.setdefault(net, len(wires))
                wires.extend(additions[net])
                added_count += len(additions[net])
            residual = _heal_same_net_m1_extras(
                assembled_m1(), routed, geometry, placement, pins_by_net, tech
            )
            if not residual:
                break
        if not residual:
            # Explicit final empty-residual idempotency check at fixed point.
            residual = _heal_same_net_m1_extras(
                assembled_m1(), routed, geometry, placement, pins_by_net, tech
            )
        if residual:
            summary = {
                net: tuple(str(polygon.bbox()) for polygon in region.each())
                for net, region in sorted(residual.items())
            }
            raise ValueError(
                "terminal-interface materialization is not idempotent: "
                f"{summary}"
            )
    except Exception:
        for net, length in original_lengths.items():
            del nets[net].wires[length:]
        raise

    routed.terminal_interface_materialization_count = added_count
    return added_count


def _materialize_router_owned_contact_line_end_m1(
    routed: Any,
    geometry: Mapping[str, Any],
    placement: Mapping[str, Any],
    pins_by_net: Mapping[str, Sequence[Any]],
    tech: Any,
) -> int:
    """Commit final-artwork M1-over-contact EOL repairs to unique route owners.

    Gridless legalizes line ends inside its segment world, before route-cell's
    terminal-interface transaction sees the final placed stamps.  That later
    same-net closure can split a cut-bearing M1 boundary and create a new EOL
    obligation.  Re-observe the exact assembled M1/contact artwork here, assign
    each cut through the same pin-seeded component ownership used by the short
    oracle, and append only PDK-derived rectangular repairs to ``NetRoute.wires``.

    Unknown/shared ownership, malformed layer/rule authority, off-grid repair,
    foreign-metal clearance loss, or a non-zero replay all fail closed.  The
    exact emitter remains a geometry observer and the ordinary route manifest
    accounts for every added rectangle.
    """
    from clay.layout.backend import LayerPolygon, Polygon
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.engine_gridless import _contact_line_end_repair_rects
    from clay.layout.route.route_cell import _seeded_geometry_metal_regions

    nets = getattr(routed, "nets", None)
    if not isinstance(nets, Mapping):
        raise TypeError("contact-line-end materialization requires routed.nets")

    raw_line_end = tech.line_end_ext("metal1")
    if raw_line_end is None:
        routed.contact_line_end_materialization_count = 0
        return 0
    raw_min_width = tech.min_width("metal1")
    raw_min_space = tech.min_space("metal1")
    if (
        type(raw_line_end) is not int
        or raw_line_end <= 0
        or type(raw_min_width) is not int
        or raw_min_width <= 0
        or type(raw_min_space) is not int
        or raw_min_space <= 0
    ):
        raise ValueError(
            "contact-line-end materialization lacks exact positive PDK rules"
        )
    raw_m1 = tech.metal_key(1)
    raw_contact = tech.contact_key()
    for name, key in (("M1", raw_m1), ("contact", raw_contact)):
        if (
            type(key) is not tuple
            or len(key) != 2
            or any(type(value) is not int for value in key)
        ):
            raise TypeError(
                f"contact-line-end materialization lacks exact {name} layer"
            )
    m1 = raw_m1
    contact = raw_contact
    grid = pdk_manufacturing_grid_nm(tech)
    if type(grid) is not int or grid <= 0:
        raise ValueError(
            "contact-line-end materialization lacks manufacturing-grid authority"
        )

    placed_by_layer = {m1: kdb.Region(), contact: kdb.Region()}
    for stamp_name, stamp in sorted(geometry.items(), key=lambda item: str(item[0])):
        position = placement.get(stamp_name)
        if position is None:
            continue
        layers = getattr(stamp, "polys_by_layer", None)
        if not isinstance(layers, Mapping):
            raise TypeError(
                f"contact-line-end stamp {stamp_name!r} lacks layer mapping"
            )
        for raw_key, rings in layers.items():
            if (
                type(raw_key) is not tuple
                or len(raw_key) != 2
                or any(type(value) is not int for value in raw_key)
            ):
                raise TypeError(
                    f"contact-line-end stamp {stamp_name!r} has malformed layer key"
                )
            target = placed_by_layer.get(raw_key)
            if target is None:
                continue
            for ring in rings:
                if len(ring) < 3:
                    raise ValueError(
                        f"contact-line-end stamp {stamp_name!r} has degenerate artwork"
                    )
                points = [_KDB_POINT(x, y) for x, y in placed_ring_nm(ring, position)]
                target.insert(_KDB_POLYGON(points))
    for region in placed_by_layer.values():
        region.merge()

    seeded = _seeded_geometry_metal_regions(
        geometry,
        placement,
        pins_by_net,
        {m1},
    )

    def routed_m1(net_route: Any) -> "kdb.Region":
        wires = getattr(net_route, "wires", None)
        if type(wires) is not list:
            raise TypeError(
                "contact-line-end materialization requires mutable exact wire lists"
            )
        region = kdb.Region()
        for wire in wires:
            if (int(wire.layer), int(wire.datatype)) == m1:
                region.insert(_poly(wire))
        region.merge()
        return region

    def owned_m1_regions() -> Dict[str, "kdb.Region"]:
        result: Dict[str, kdb.Region] = {}
        for net, net_route in sorted(nets.items(), key=lambda item: str(item[0])):
            route_region = routed_m1(net_route)
            if route_region.is_empty():
                continue
            own = route_region
            stamp_region = seeded.get(str(net), {}).get(m1)
            if stamp_region is not None:
                own += stamp_region
            own.merge()
            result[str(net)] = own
        return result

    contact_polygons = tuple(placed_by_layer[contact].each())
    seeded_m1_entries = [
        (str(net), by_layer[m1], by_layer[m1].bbox())
        for net, by_layer in seeded.items()
        if by_layer.get(m1) is not None and not by_layer[m1].is_empty()
    ]
    contacts_by_net: Dict[str, kdb.Region] = {}
    for polygon in contact_polygons:
        pbb = polygon.bbox()
        cut = None
        owner_list: List[str] = []
        for s_net, s_reg, s_bb in seeded_m1_entries:
            if not pbb.overlaps(s_bb):
                continue
            if cut is None:
                cut = kdb.Region(polygon)
            if not (s_reg & cut).is_empty():
                owner_list.append(s_net)
        if len(owner_list) > 1:
            owners = tuple(sorted(owner_list))
            raise ValueError(
                "contact-line-end cut has non-unique M1 owner: "
                f"bbox={polygon.bbox()}, owners={owners}"
            )
        if len(owner_list) == 1:
            contacts_by_net.setdefault(owner_list[0], kdb.Region()).insert(polygon)
    for region in contacts_by_net.values():
        region.merge()

    eol_edge_max = int(round(raw_min_width * 2.5))

    def assembled_m1(owned: Mapping[str, "kdb.Region"]) -> "kdb.Region":
        region = placed_by_layer[m1].dup()
        for net_route in nets.values():
            region += routed_m1(net_route)
        region.merge()
        return region

    owned_before = owned_m1_regions()
    all_before = assembled_m1(owned_before)
    owned_union = kdb.Region()
    for region in owned_before.values():
        owned_union += region
    owned_union.merge()
    unowned = (all_before - owned_union).merged()
    unowned_bb = unowned.bbox() if not unowned.is_empty() else None
    owned_before_bb = {
        k_net: (reg, reg.bbox()) for k_net, reg in owned_before.items()
    }

    additions: Dict[str, List[LayerPolygon]] = {}
    for net, own in sorted(owned_before.items()):
        net_contacts = contacts_by_net.get(net)
        if net_contacts is None or net_contacts.is_empty():
            continue
        repairs = _contact_line_end_repair_rects(
            own,
            net_contacts,
            line_end_nm=raw_line_end,
            eol_edge_max_nm=eol_edge_max,
        )
        if not repairs:
            continue
        for repair in repairs:
            if any(int(value) % grid for value in repair):
                raise ValueError(
                    f"contact-line-end repair is off manufacturing grid: {repair}"
                )
            repair_box = kdb.Box(*repair)
            repair_region = kdb.Region(repair_box)
            new_material = (repair_region - all_before).merged()
            if new_material.is_empty():
                continue
            clip_box = new_material.bbox().enlarged(2 * raw_min_space + 10)
            clip_reg = kdb.Region(clip_box)
            foreign = kdb.Region()
            if unowned_bb is not None and unowned_bb.overlaps(clip_box):
                foreign += (unowned & clip_reg)
            for other_net, (other_region, other_bb) in owned_before_bb.items():
                if other_net != net and other_bb.overlaps(clip_box):
                    foreign += (other_region & clip_reg)
            foreign.merge()
            if not (new_material & foreign.sized(raw_min_space)).is_empty():
                raise ValueError(
                    "contact-line-end repair violates foreign M1 clearance: "
                    f"net={net!r}, bbox={repair}"
                )
            additions.setdefault(net, []).append(LayerPolygon(
                layer=m1[0],
                datatype=m1[1],
                polygon=Polygon.rectangle(*repair),
            ))

    original_lengths = {
        net: len(getattr(nets[net], "wires")) for net in additions
    }
    try:
        for net in sorted(additions):
            nets[net].wires.extend(additions[net])
        owned_after = owned_m1_regions()
        residual_by_net = {
            net: _contact_line_end_repair_rects(
                own,
                contacts_by_net.get(net, kdb.Region()),
                line_end_nm=raw_line_end,
                eol_edge_max_nm=eol_edge_max,
            )
            for net, own in sorted(owned_after.items())
        }
        residual_by_net = {
            net: rows for net, rows in residual_by_net.items() if rows
        }
        global_residual = _contact_line_end_repair_rects(
            assembled_m1(owned_after),
            placed_by_layer[contact],
            line_end_nm=raw_line_end,
            eol_edge_max_nm=eol_edge_max,
        )
        if residual_by_net or global_residual:
            raise ValueError(
                "contact-line-end materialization is not exhaustive: "
                f"owned={residual_by_net}, global={global_residual}"
            )
    except Exception:
        for net, length in original_lengths.items():
            del nets[net].wires[length:]
        raise

    added_count = sum(len(rows) for rows in additions.values())
    routed.contact_line_end_materialization_count = added_count
    return added_count


def _materialize_router_owned_m1_finalization(
    routed: Any,
    geometry: Mapping[str, Any],
    placement: Mapping[str, Any],
    pins_by_net: Mapping[str, Sequence[Any]],
    tech: Any,
) -> int:
    """Atomically prove the terminal-interface/contact-EOL composition fixed.

    Both constituent observers are individually idempotent, but either may
    change the other's final boundary.  Apply the ordered composition once and
    require an immediate second replay to add zero shapes.  A non-zero replay is
    a typed non-convergence, not permission for an arbitrary repair loop.
    """
    nets = getattr(routed, "nets", None)
    if not isinstance(nets, Mapping):
        raise TypeError("M1 finalization requires routed.nets")
    original_lengths: Dict[str, int] = {}
    for net, net_route in nets.items():
        wires = getattr(net_route, "wires", None)
        if type(wires) is not list:
            raise TypeError("M1 finalization requires mutable exact wire lists")
        original_lengths[str(net)] = len(wires)
    try:
        terminal_count = _materialize_router_owned_terminal_interface_m1(
            routed, geometry, placement, pins_by_net, tech
        )
        line_end_count = _materialize_router_owned_contact_line_end_m1(
            routed, geometry, placement, pins_by_net, tech
        )
        terminal_replay = _materialize_router_owned_terminal_interface_m1(
            routed, geometry, placement, pins_by_net, tech
        )
        line_end_replay = _materialize_router_owned_contact_line_end_m1(
            routed, geometry, placement, pins_by_net, tech
        )
        if terminal_replay or line_end_replay:
            raise ValueError(
                "M1 finalization composition is not idempotent: "
                f"terminal={terminal_replay}, line_end={line_end_replay}"
            )
    except Exception:
        for net, length in original_lengths.items():
            del nets[net].wires[length:]
        raise

    routed.terminal_interface_materialization_count = terminal_count
    routed.contact_line_end_materialization_count = line_end_count
    return terminal_count + line_end_count


def _retain_attached_heal_material(own, candidate):
    """Return only candidate material in a component containing ``own`` metal.

    ``candidate`` may contain several touching polygons; a whole touching chain
    is retained when its merged component reaches the pre-heal conductor.  Any
    detached component is discarded.  This is a topology obligation, not a
    distance heuristic: a legal min-space gap remains an electrical open and
    therefore never authorizes emitter-owned route geometry.
    """
    owned = own.dup()
    owned.merge()
    extra = candidate.dup()
    extra.merge()
    if owned.is_empty() or extra.is_empty():
        return extra if not owned.is_empty() else type(extra)()

    joined = (owned + extra).merged()
    kept = type(extra)()
    for component in joined.each():
        component_region = type(extra)(component)
        if (component_region & owned).is_empty():
            continue
        kept += (component_region & extra)
    kept.merge()
    return kept


def _drop_sub_min_width_heal_slivers(own, extra, min_w: int):
    """Remove heal-added material that forms a sub-min-width protrusion on the
    merged (own + extra) island, keeping any piece whose removal would split
    the island (honest width violation > silent open). Pure region math.

    The narrow-feature detector is the morphological OPENING residue of the
    merged region with half-width ``(min_w - 1) // 2``: features narrower than
    ``2*hw + 1`` vanish under erode+dilate, everything >= min_w is restored
    exactly (rectilinear). Only the intersection with ``extra`` is ever
    removed -- device artwork / routed wires are never touched."""
    hw = (min_w - 1) // 2
    if hw <= 0 or extra.is_empty():
        return extra

    def _ncomp(r: "kdb.Region") -> int:
        m = r.dup()
        m.merge()
        return sum(1 for _ in m.each())

    for _ in range(4):  # re-check: a removal can expose a new narrow step
        merged = own + extra
        merged.merge()
        residue = merged - merged.sized(-hw).sized(hw)
        cut = (residue & extra).merge()
        if cut.is_empty():
            break
        base = _ncomp(merged)
        changed = False
        for poly in cut.each():
            trial = (extra - kdb.Region(poly)).merge()
            if _ncomp(own + trial) <= base:
                extra = trial
                changed = True
        if not changed:
            break  # every sliver is load-bearing -> honest width violation
    return extra


def _new_region_material(candidate: Any, owned: Any) -> Any:
    """Return only geometry that ``candidate`` adds beyond ``owned``.

    Width/space legalization candidates are coverage witnesses and commonly
    include most (or all) of the conductor island they repair.  The exact
    emitter must never re-insert that already-owned conductor and account it as
    a transformation: only the set difference is materialized.
    """
    return (candidate - owned).merged()


class _GeometryTransformCount(int):
    """Integer-compatible transform count with bounded attribution evidence."""

    def __new__(cls, value: int, evidence: Sequence[Mapping[str, Any]] = ()):
        result = int.__new__(cls, int(value))
        result.evidence = tuple(dict(item) for item in evidence)
        return result


def _bind_transform_evidence_sources(
    evidence: Sequence[Mapping[str, Any]],
    *,
    geometry: Mapping[str, Any],
    placement: Mapping[str, Any],
    routed: Any,
    pins_by_net: Mapping[str, Sequence[Any]],
    tech: LayoutTech,
) -> Tuple[Dict[str, Any], ...]:
    """Bind non-zero emitter transforms to nearby materialized source geometry."""

    import klayout.db as kdb

    bound: List[Dict[str, Any]] = []
    for raw in evidence:
        row = dict(raw)
        layer_value = row.get("layer")
        bbox_value = row.get("source_island_bbox_nm") or row.get("bbox_nm")
        if (
            not isinstance(layer_value, (tuple, list))
            or len(layer_value) != 2
            or not isinstance(bbox_value, (tuple, list))
            or len(bbox_value) != 4
        ):
            bound.append(row)
            continue
        layer = (int(layer_value[0]), int(layer_value[1]))
        source_box = kdb.Box(*(int(value) for value in bbox_value))
        source_region = kdb.Region(source_box)
        geometry_sources: List[Dict[str, Any]] = []
        for name, stamp in sorted(geometry.items(), key=lambda item: str(item[0])):
            position = placement.get(name)
            if position is None:
                continue
            for ring in (getattr(stamp, "polys_by_layer", {}) or {}).get(layer, ()):
                points = [
                    kdb.Point(*placed_point_nm(float(x), float(y), position))
                    for x, y in ring
                ]
                if len(points) < 3:
                    continue
                polygon = kdb.Polygon(points)
                overlap = (kdb.Region(polygon) & source_region).area()
                if overlap <= 0:
                    continue
                bbox = polygon.bbox()
                geometry_sources.append(
                    {
                        "name": str(name),
                        "bbox_nm": [bbox.left, bbox.bottom, bbox.right, bbox.top],
                        "overlap_nm2": int(overlap),
                    }
                )
        route_sources: List[Dict[str, Any]] = []
        for net, net_route in sorted(
            (getattr(routed, "nets", {}) or {}).items(),
            key=lambda item: str(item[0]),
        ):
            for wire in getattr(net_route, "wires", ()) or ():
                if (int(wire.layer), int(wire.datatype)) != layer:
                    continue
                polygon = _poly(wire)
                overlap = (kdb.Region(polygon) & source_region).area()
                if overlap <= 0:
                    continue
                bbox = polygon.bbox()
                route_sources.append(
                    {
                        "net": str(net),
                        "bbox_nm": [bbox.left, bbox.bottom, bbox.right, bbox.top],
                        "overlap_nm2": int(overlap),
                    }
                )
        terminal_sources: List[Dict[str, Any]] = []
        terminal_window = source_region.sized(2500).bbox()
        for net, pins in sorted(pins_by_net.items(), key=lambda item: str(item[0])):
            for pin in pins:
                try:
                    x_nm = int(round(float(pin.xy_um[0]) * _NM))
                    y_nm = int(round(float(pin.xy_um[1]) * _NM))
                except Exception:
                    continue
                if not terminal_window.contains(kdb.Point(x_nm, y_nm)):
                    continue
                terminal_sources.append(
                    {
                        "net": str(net),
                        "device": str(getattr(pin, "device", "")),
                        "terminal": str(getattr(pin, "terminal", "")),
                        "xy_nm": [x_nm, y_nm],
                        "layer": str(getattr(pin, "layer", "")),
                    }
                )
        row["geometry_sources"] = geometry_sources[:64]
        row["route_sources"] = route_sources[:64]
        row["terminal_sources"] = terminal_sources[:64]
        bound.append(row)
    return tuple(bound)


def _close_same_net_m1_post_assembly(
    top: Any, ly: Any, label_points: Dict[str, Tuple[int, int]], tech: LayoutTech,
) -> int:
    """FINAL same-net M1 close over the FULLY assembled cell (device + cap artwork +
    routed BEOL + heal collars + grows -- everything now in ``top``), returning the
    number of fill polygons inserted.

    Why this exists and runs HERE (not in the heal): the heal's ``collar`` widens the
    wire-to-device-pad junction band, which can OPEN a NEW concave sub-min-space notch
    between the freshly collared own metal and a same-net device finger -- a notch that
    does NOT exist in the pre-collar region, so the heal's own morphological close
    (which runs BEFORE the collar geometry is final, on a region rebuilt from
    wires + pin-pads that does not capture every emitted device M1 polygon) cannot see
    it (the gf180 comparator OUTP 87nm / VDD 227nm SAME-NET M1.2a). This pass operates
    on the GROUND-TRUTH emitted geometry instead.

    Net partition = CONNECTIVITY to each net's label point (the SAME net-name anchor
    KLayout's extractor uses): an emitted M1 island belongs to net N iff it contains
    N's label point. Per net, morphologically CLOSE its own M1 (``size(+half).size(-half)``,
    half = ``min_space // 2``, fills any same-net gap strictly below the rule) and insert
    only the NEW fill, clipped a full ``min_space`` clear of every OTHER net's M1. A
    same-net fill flush against own metal but within ``min_space`` of a foreign net is
    dropped by the clip -> never a short; a genuinely cross-net gap is never bridged
    (the two sides carry different labels, so the fill is clipped away). By-construction
    PDK-rule-derived (radius + clip are both ``tech.min_space('metal1')``), no constant,
    no circuit / net / PDK branch. Degrades to a no-op when the PDK declares no M1 rule."""
    import klayout.db as kdb
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    m1 = tuple(tech.metal_key(1))
    manufacturing_grid_nm = pdk_manufacturing_grid_nm(tech)
    try:
        s = int(tech.min_space("metal1") or 0)
    except Exception:
        s = 0
    if s <= 0 or not label_points:
        return _GeometryTransformCount(0)
    half = (s + 1) // 2
    li = ly.layer(int(m1[0]), int(m1[1]))

    full = kdb.Region()
    for sh in top.shapes(li).each():
        if sh.is_box() or sh.is_polygon():
            full.insert(kdb.Polygon(sh.polygon))
    full.merge()

    # connectivity partition: island -> the net(s) whose label point it contains.
    net_region: Dict[str, kdb.Region] = {n: kdb.Region() for n in label_points}
    label_items = list(label_points.items())
    for isl in full.each():
        bb = isl.bbox()
        bl, bb_bot, br, bt = bb.left, bb.bottom, bb.right, bb.top
        isl_is_box = isl.is_box()
        ir = None
        for n, (lx, lyy) in label_items:
            if bl < lx + 1 and br > lx - 1 and bb_bot < lyy + 1 and bt > lyy - 1:
                if isl_is_box:
                    net_region[n].insert(isl)
                else:
                    if ir is None:
                        ir = kdb.Region(isl)
                    if not (ir & kdb.Region(kdb.Box(lx - 1, lyy - 1, lx + 1, lyy + 1))).is_empty():
                        net_region[n].insert(isl)
    net_region_bb: Dict[str, Tuple[kdb.Region, Any]] = {}
    for n, reg in net_region.items():
        reg.merge()
        if not reg.is_empty():
            net_region_bb[n] = (reg, reg.bbox())

    n_fill = 0
    transform_evidence: List[Dict[str, Any]] = []
    granted = kdb.Region()
    for n in sorted(net_region):
        own = net_region[n]
        if own.is_empty():
            continue
        fill = own.sized(half).sized(-half) - own
        if fill.is_empty():
            continue
        # Any M1 island outside ``own`` -- including unlabelled device/trunk
        # islands connected via upper metals -- plus earlier fill belongs to
        # foreign keepout so closing never bridges an unlabelled foreign island.
        # Clip foreign locally to fill.bbox().enlarged(2*s + 10) before sizing.
        clip_box = fill.bbox().enlarged(2 * s + 10)
        clip_reg = kdb.Region(clip_box)
        foreign = (full & clip_reg) - (own & clip_reg)
        if not granted.is_empty() and granted.bbox().overlaps(clip_box):
            foreign += (granted & clip_reg)
        for m, (reg, rbb) in net_region_bb.items():
            if m != n and rbb.overlaps(clip_box):
                foreign += (reg & clip_reg)
        foreign.merge()
        fill = (fill - foreign.sized(s)).merge()
        if fill.is_empty():
            continue
        # NO-VIOLATION, NO-FILL.  Morphological closing is deliberately an
        # over-approximation: offset rectangles can produce a closing seam
        # even when the foundry deck reports neither a width nor a spacing
        # violation.  Emitting that seam changes an already exact route and
        # is therefore forbidden.  Keep only connected fill components that
        # intersect the deck-equivalent, strictly-sub-min-space gap polygons.
        # This is the same obligation used by the upper-metal close above;
        # real same-island notches remain healable, while legal geometry is
        # byte-for-byte untouched.
        viol = own.space_check(
            s, False, kdb.Region.Euclidian, None, None, None
        ).polygons(0)
        justified = kdb.Region()
        if not viol.is_empty():
            viol_bb = viol.bbox()
            for component in fill.each():
                if not component.bbox().overlaps(viol_bb):
                    continue
                component_region = kdb.Region(component)
                if not (component_region & viol).is_empty():
                    justified += component_region
        fill = justified.merge()
        if fill.is_empty():
            continue
        granted += fill
        own_bb = own.bbox()
        for poly in fill.each():
            top.shapes(li).insert(poly)
            n_fill += 1
            bbox = poly.bbox()
            transform_evidence.append(
                {
                    "kind": "post_m1_close",
                    "net": str(n),
                    "layer": [int(m1[0]), int(m1[1])],
                    "bbox_nm": [bbox.left, bbox.bottom, bbox.right, bbox.top],
                    "area_nm2": int(poly.area()),
                    "source_island_bbox_nm": [
                        own_bb.left,
                        own_bb.bottom,
                        own_bb.right,
                        own_bb.top,
                    ],
                }
            )

    # ---- WIDTH LEGALIZATION of every assembled metal ISLAND (R7 tail) -------
    # A junction can carry an INTERIOR euclidean width below the layer's rule
    # that no morphological close can fix (close only fills concave notches;
    # the measured M1.W.1 staircase neck -- two arms meeting 165x157nm apart,
    # diagonal 227.8nm < 230 -- is interior metal). Obligation: NO merged
    # island of ANY routing metal in the emitted artifact carries a
    # sub-min-width euclidean measurement that an island-local corner fill
    # can legally remove -- the deck runs M{n}.W.1 on EVERY layer, and merged
    # upper-metal geometry (engine wires + junction fills + stamp risers) can
    # carry an EMERGENT neck no pairwise contact predicate sees (measured:
    # r7b M2.W.1, a pile fill protruding 175nm past its support). The island
    # is the conductor unit -- the deck judges drawn geometry NET-BLIND, so
    # this is deliberately LABEL-FREE: the per-net-label association missed
    # every island whose net label anchors a DIFFERENT island of the same net
    # (islands legitimately joined only through other layers) and every
    # unlabelled artwork island -- the measured r7b M1.W.1 x4 lived in a
    # DRAWN device-strap island carrying no label point at all (place-only
    # emission reproduces them with ZERO routed metal). Each violation
    # (KLayout's own width_check, the deck's metric family) is fixed by its
    # edge-pair bbox grown one min-width per side, clipped a full min-space
    # clear of EVERYTHING outside the island (foreign nets AND other
    # same-net islands -- healing never bridges conductors, never alters
    # connectivity); a fix that no longer covers the violation core after
    # clipping is dropped (the honest deck violation stays). Junction scope:
    # an edge pair whose PROJECTION-OVERLAP core exceeds 2x min-width is a
    # wire-body width defect (the emitter's own contract), not a junction
    # neck -- left honest. The core, not the raw bbox: the euclidean metric
    # extends each flagged edge past the other's end by up to
    # sqrt(min_w^2 - d^2), so an OFFSET-OVERLAP neck's bbox reads up to 2x
    # min-width longer than the neck (the measured ref-experiment M2.W.1
    # escape: a 280x210 seam read 650 > 560 and was skipped). PDK-rule-
    # derived per layer, deterministic island order, no constants, no
    # circuit / net / PDK branch.
    for lv in range(1, 12):
        try:
            mk = tuple(tech.metal_key(lv))
        except Exception:
            break
        if mk is None:
            break
        try:
            min_w = int(tech.min_width(f"metal{lv}") or 0)
        except Exception:
            min_w = 0
        try:
            s_lv = int(tech.min_space(f"metal{lv}") or 0)
        except Exception:
            s_lv = 0
        if min_w <= 0 or s_lv <= 0:
            continue
        li_lv = ly.layer(int(mk[0]), int(mk[1]))
        full2 = kdb.Region()
        for sh in top.shapes(li_lv).each():
            if sh.is_box() or sh.is_polygon():
                full2.insert(kdb.Polygon(sh.polygon))
        full2.merge()
        if full2.is_empty():
            continue
        islands = sorted(
            ((isl.bbox(), isl.is_box(), isl) for isl in full2.each()),
            key=lambda item: (item[0].left, item[0].bottom,
                              item[0].right, item[0].top))
        reach_pad = 3 * (min_w + max(1, int(manufacturing_grid_nm))) + 2 * s_lv + 10
        for source_bbox, isl_is_box, isl in islands:
            if isl_is_box and source_bbox.width() >= min_w and source_bbox.height() >= min_w:
                continue
            own = kdb.Region(isl)
            if (
                own.width_check(
                    min_w, False, kdb.Region.Euclidian, None, None, None
                ).is_empty()
                and own.space_check(
                    s_lv, False, kdb.Region.Euclidian, None, None, None
                ).is_empty()
            ):
                continue
            clip_reg = kdb.Region(source_bbox.enlarged(reach_pad))
            local_blocked = ((full2 & clip_reg) - own).sized(s_lv)
            own, fill_generations = legalize_island_junctions(
                own,
                local_blocked,
                min_width_nm=min_w,
                min_space_nm=s_lv,
                manufacturing_grid_nm=manufacturing_grid_nm,
            )
            for fills in fill_generations:
                for poly in fills.each():
                    top.shapes(li_lv).insert(poly)
                    n_fill += 1
                    bbox = poly.bbox()
                    transform_evidence.append(
                        {
                            "kind": "post_width_or_notch_legalization",
                            "net": None,
                            "layer": [int(mk[0]), int(mk[1])],
                            "bbox_nm": [
                                bbox.left,
                                bbox.bottom,
                                bbox.right,
                                bbox.top,
                            ],
                            "area_nm2": int(poly.area()),
                            "source_island_bbox_nm": [
                                source_bbox.left,
                                source_bbox.bottom,
                                source_bbox.right,
                                source_bbox.top,
                            ],
                            "min_width_nm": min_w,
                            "min_space_nm": s_lv,
                        }
                    )
    return _GeometryTransformCount(n_fill, transform_evidence)


def _well_merge_setup(
    intent: Any,
    tech: LayoutTech,
    *,
    devices: Optional[Sequence[Any]] = None,
    _explicit_well_key: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, str]]:
    """Resolve the well GDS key + a device-name -> bulk-net map for the well merge.

    The bulk net is the device's body/bulk terminal (the well's electrical
    potential). Devices with no resolvable bulk net are omitted, so their well (if
    any) is treated as an unattributable singleton and never merged. Returns
    ``(None, {})`` when the PDK draws no well -> the caller skips the merge.

    ``devices`` (C3) is an EXPLICIT device view -- elements with ``.name`` and a
    ROLE-keyed ``.nets`` (``body``/``bulk``) -- for callers whose ``intent`` does
    not carry devices: the route_cell emit passes a ``LayoutIntent`` (net-level
    declarations only), so the intent read below yields an EMPTY map, every well
    stays an unattributable singleton, and the NW.2b same-net merge silently
    never fires. ``devices=None`` keeps the legacy OTA ``ExtractedIntent`` read
    byte-identical (:func:`well_merge_devices_from_cell_def` builds the view from
    the canonical ``cell_def.devices``).

    Native device views carry the well net observed from their complete physical
    conductor graph. In a three-terminal varactor the nwell is terminal ``n``;
    terminal ``b`` is the substrate, so the generic MOS body rule does not apply.
    """
    if _explicit_well_key is not None:
        well_key = (int(_explicit_well_key[0]), int(_explicit_well_key[1]))
    else:
        try:
            wk = tech.build_layer("nwell")
        except Exception:
            wk = None
        if wk is None:
            return None, {}
        well_key = (int(wk[0]), int(wk[1]))
    net_of: Dict[str, str] = {}
    source = devices if devices is not None else (getattr(intent, "devices", []) or [])
    for d in source:
        nm = getattr(d, "name", None)
        nets = getattr(d, "nets", {}) or {}
        bulk = (
            getattr(d, "native_well_net", None)
            if hasattr(d, "native_terminals")
            else nets.get("body") or nets.get("bulk")
        )
        if nm is not None and bulk:
            net_of[str(nm)] = str(bulk)
    return well_key, net_of


def _mos_polarity(device_class: Any) -> Optional[str]:
    """Return the shared canonical MOS polarity for a Clay device class.

    This is deliberately the same classifier used by canonical placement and
    device materialization.  In particular, production thick-oxide declarations
    (``nmos3v``/``pmos3v`` and their PDK-model aliases) remain MOS devices here;
    LVS must not maintain a second two-token allow-list.
    """
    from clay.layout.annotate_to_solver import kind_polarity_oxide

    polarity, _oxide = kind_polarity_oxide(device_class)
    return polarity


def _pdk_binds_mos_class(device_class: Any, tech: LayoutTech) -> bool:
    """Whether ``tech`` declares a binding for this MOS polarity/oxide domain.

    Exact Clay-class keys win, while equivalent PDK spellings (for example
    ``nch3`` versus ``nmos3v``) are matched through the shared
    polarity/oxide classifier.  The answer is therefore derived from the
    adapter's device facts, not from a circuit or fixture-name exception.
    """
    from clay.layout.annotate_to_solver import kind_polarity_oxide

    declared = kind_polarity_oxide(device_class)
    if declared[0] is None:
        return False
    token = str(device_class or "").strip().lower()
    bindings = {
        str(key).strip().lower(): value
        for key, value in tech.device_bindings().items()
        if value
    }
    if token in bindings:
        return True
    return any(
        kind_polarity_oxide(bound_class) == declared
        for bound_class in bindings
    )


def _validate_pdk_reference_mos_classes(cell_def: Any, tech: LayoutTech) -> None:
    """Fail closed when the selected PDK has no binding for a declared MOS."""
    unsupported = sorted({
        str(getattr(device, "device_class", "") or "").strip().lower()
        for device in (getattr(cell_def, "devices", ()) or ())
        if _mos_polarity(getattr(device, "device_class", None)) is not None
        and not _pdk_binds_mos_class(
            getattr(device, "device_class", None), tech
        )
    })
    if unsupported:
        raise LvsReferenceStop(
            state="UNSUPPORTED_DEVICE_CLASS",
            refs=unsupported,
        )


def well_merge_devices_from_cell_def(cell_def: Any) -> List[Any]:
    """The emit-side well-merge device view derived from the CANONICAL
    ``cell_def.devices`` (the same source the placer/router consume).

    ``clay.ir.Device.pins`` is TERM-keyed (``'s'``/``'d'``/``'g'``/``'b'``);
    :func:`_well_merge_setup` reads ROLE keys (``body``), so this view does the
    term -> role mapping ONCE at the boundary. Names are the UNSTRIPPED instance
    names -- they must match the ``geometry``/``placement`` keys the emit loop
    iterates (never the M-stripped netlist-card names)."""
    from types import SimpleNamespace

    out: List[Any] = []
    for d in getattr(cell_def, "devices", []) or []:
        dclass = str(getattr(d, "device_class", "") or "").strip().lower()
        if _mos_polarity(dclass) is None:
            continue
        name = str(getattr(d, "instance_name", None) or getattr(d, "name", "") or "")
        if not name:
            continue
        pins = getattr(d, "pins", None) or {}
        out.append(SimpleNamespace(
            name=name,
            kind=_mos_polarity(dclass),
            nets={
                "source": pins.get("s"), "drain": pins.get("d"),
                "gate": pins.get("g"), "body": pins.get("b"),
            },
        ))
    return out


def _bridge_same_net_region(reg: "kdb.Region", tech: LayoutTech) -> "kdb.Region":
    """Bridge a SAME-net well region's sub-NW.2b gaps (morphological close).

    A row of same-bulk-net PMOS wells lands as several islands a sub-spacing gap
    apart; closing by the different-potential well spacing fuses any pair closer
    than the rule into one continuous well, removing the (same-net) NW.2b island
    pair. The spacing is DATA -- ``PdkRules.min_spacing('nwell')`` (gf180 NW.2b =
    1400 nm). Close = dilate then erode: it only adds material (PCOMP enclosure /
    min-width preserved) and never reaches a different net (the caller passes one
    net's region at a time). A PDK with no resolvable well spacing is a no-op.
    """
    try:
        from clay.layout.pdk_rules import PdkRules

        space_nm = PdkRules.for_pdk(tech.pdk_id).min_spacing("nwell")
    except Exception:
        space_nm = None
    if not space_nm or space_nm <= 0:
        return reg
    # close gaps strictly below the spacing: dilate by ceil(space/2) then erode
    # back. 2*d >= space bridges every pair under the rule; the erode restores the
    # outer boundary, leaving only the filled channels.
    d = (int(space_nm) + 1) // 2
    out = reg.dup()
    out.size(d)
    out.size(-d)
    out.merge()
    return out


def _resolve_stamp_nwell_net(
    name: str,
    stamp: Any,
    well_net_of: Mapping[str, str],
    device_by_name: Mapping[str, Any],
) -> Optional[str]:
    """Resolve the unique N-well electrical net for a single or composite stamp.

    Single-device stamps resolve directly through ``well_net_of``. Composite row
    stamps (non-empty ``stamp.composite_members``) resolve to one well net if and
    only if every member exists in ``device_by_name`` and all N-well-bearing
    members resolve to the exact same net; heterogeneous or unattributed members
    fail closed to ``None`` (isolated singleton well).
    """
    direct = well_net_of.get(str(name))
    if direct is not None:
        return direct
    raw_members = getattr(stamp, "composite_members", None) or ()
    members = tuple(str(m) for m in raw_members if str(m))
    if not members:
        return None
    nwell_nets: set[str] = set()
    for member in members:
        dev = device_by_name.get(member)
        if dev is None:
            return None
        if hasattr(dev, "native_terminals"):
            wnet = getattr(dev, "native_well_net", None)
            if wnet:
                nwell_nets.add(str(wnet))
            elif _mos_polarity(getattr(dev, "kind", None)) == "pmos":
                return None
        else:
            if _mos_polarity(getattr(dev, "kind", None)) == "nmos":
                continue
            m_net = well_net_of.get(member)
            if not m_net:
                return None
            nwell_nets.add(str(m_net))
    if len(nwell_nets) != 1:
        return None
    return next(iter(nwell_nets))


def _resolve_stamp_declared_well_plan(
    name: str,
    stamp: Any,
    key: Tuple[int, int],
    by_member: Mapping[str, Any],
    well_net_of: Mapping[str, str],
    device_by_name: Mapping[str, Any],
) -> Tuple[Any, Tuple[str, ...]]:
    """Resolve a ``@well_island_share`` plan for a single or composite stamp."""
    direct = by_member.get(str(name))
    if direct is not None and direct.layer_key == key:
        return direct, (str(name),)
    raw_members = getattr(stamp, "composite_members", None) or ()
    members = tuple(str(m) for m in raw_members if str(m))
    if not members:
        return None, ()
    matched_plans = {
        by_member[m].sha256: by_member[m]
        for m in members
        if m in by_member and by_member[m].layer_key == key
    }
    if not matched_plans:
        return None, ()
    if len(matched_plans) > 1:
        raise ValueError("composite stamp spans multiple declared shared well plans")
    plan = next(iter(matched_plans.values()))
    covered: List[str] = []
    for m in members:
        if by_member.get(m) is plan:
            covered.append(m)
            continue
        dev = device_by_name.get(m)
        polarity = _mos_polarity(getattr(dev, "kind", None)) if dev is not None else None
        if plan.well == "nwell":
            bears_plan_well = polarity != "nmos" and (
                dev is None or polarity == "pmos" or m in well_net_of
            )
        else:
            bears_plan_well = dev is None or polarity == "nmos"
        if bears_plan_well:
            raise ValueError("declared shared well partially covers composite stamp")
    return plan, tuple(covered)


def _source_well_assembly(
    intent,
    tech,
    geometry,
    placement,
    *,
    devices=None,
    source_cell=None,
    _explicit_well_key: Optional[Tuple[int, int]] = None,
):
    """Replay the emitter's source-owned well partitions and their output regions.

    Exact-structure expectation uses this same boundary so it cannot collapse
    two declared domains merely because their source body nets have equal names.
    """
    from clay.layout.intent import layout_intent_from_cell_def, replay_well_island_share_intent
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.well_island_sharing import compile_well_island_share_plans, materialize_well_island_share_region

    grid_nm = pdk_manufacturing_grid_nm(tech)

    well_key, well_net_of = _well_merge_setup(
        intent, tech, devices=devices, _explicit_well_key=_explicit_well_key
    )
    source_devices = tuple(
        devices if devices is not None else (getattr(intent, "devices", []) or [])
    )
    device_by_name = {
        str(getattr(d, "name", "")): d
        for d in source_devices
        if getattr(d, "name", None) is not None
    }
    supplied = getattr(intent, "well_island_shares", ())
    if type(supplied) is not tuple:
        raise TypeError("well island share inventory must be an exact tuple")
    supplied = tuple(replay_well_island_share_intent(item) for item in supplied)
    if source_cell is None:
        if supplied:
            raise ValueError("declared well island sharing requires canonical source authority")
        plans = ()
    else:
        declared = tuple(replay_well_island_share_intent(item)
                         for item in layout_intent_from_cell_def(source_cell).well_island_shares)
        if supplied != declared:
            raise ValueError("well island share inventory differs from canonical source")
        plans = compile_well_island_share_plans(source_cell, declared, tech)
    by_member = {member: plan for plan in plans for member in plan.members}
    by_sha = {plan.sha256: plan for plan in plans}
    layer_keys = {plan.layer_key for plan in plans}
    if well_key is not None:
        layer_keys.add(well_key)
    by_owner = {}
    seen = {plan.sha256: set() for plan in plans}
    for name, stamp in (geometry or {}).items():
        pos = (placement or {}).get(name)
        if pos is None:
            continue
        for layer, rings in (getattr(stamp, "polys_by_layer", {}) or {}).items():
            key = (int(layer[0]), int(layer[1]))
            if key not in layer_keys:
                continue
            for ring in rings:
                if len(ring) < 3:
                    continue
                pts = [_KDB_POINT(x, y) for x, y in placed_ring_nm(ring, pos)]
                plan, covered_members = _resolve_stamp_declared_well_plan(
                    str(name), stamp, key, by_member, well_net_of, device_by_name
                )
                if plan is not None:
                    owner = (key, "declared", plan.sha256)
                    seen[plan.sha256].update(covered_members)
                elif well_key is not None and key == well_key:
                    owner = (
                        key,
                        "legacy",
                        _resolve_stamp_nwell_net(
                            str(name), stamp, well_net_of, device_by_name
                        ),
                    )
                else:
                    owner = (key, "unowned", str(name))
                by_owner.setdefault(owner, kdb.Region()).insert(kdb.Polygon(pts))
    for plan in plans:
        if seen[plan.sha256] != set(plan.members):
            raise ValueError(
                f"declared shared well {plan.group!r} lacks member well artwork: "
                f"expected={plan.members!r}, observed={tuple(sorted(seen[plan.sha256]))!r}"
            )
    result = {}
    for key in sorted({owner[0] for owner in by_owner}):
        selected = {owner: region for owner, region in by_owner.items() if owner[0] == key}
        before = kdb.Region()
        for region in selected.values():
            before += region
        before.merge()
        emitted = []
        for owner, raw in sorted(selected.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2] or ""))):
            region = raw.merged()
            if owner[1] == "declared":
                plan = by_sha[str(owner[2])]
                if plan.well == "nwell":
                    region = _bridge_same_net_region(region, tech)
                region = materialize_well_island_share_region(
                    region, grid_nm=grid_nm
                )
                if plan.layer_key != key:
                    raise ValueError("shared well plan changed physical layer")
            elif owner[1] == "legacy" and owner[2] is not None:
                region = _bridge_same_net_region(region, tech)
            emitted.append(region)
        result[key] = (before, tuple(emitted))
    return layer_keys, result


class DeckIsolationWriteRefused(RuntimeError):
    """Typed non-writing STOP from the final in-memory deck-isolation gate."""

    def __init__(self, *, report: Any, route_stops: Sequence[Any]) -> None:
        from clay.layout.route.engine_gridless import NegotiationRouteStop

        stops = tuple(sorted(route_stops, key=lambda stop: stop.canonical_json()))
        if not stops or any(type(stop) is not NegotiationRouteStop for stop in stops):
            raise TypeError(
                "DeckIsolationWriteRefused requires NegotiationRouteStop entries"
            )
        self.report = report
        self.route_stops = stops
        super().__init__("\n".join(stop.canonical_json() for stop in stops))


def _delete_stale_emit_target(out_path: Path) -> None:
    """Remove one known output file without creating or traversing directories."""

    target = Path(out_path)
    try:
        target.unlink()
    except FileNotFoundError:
        return
    except PermissionError:
        target.chmod(0o600)
        target.unlink()


def _canonical_final_emit_owners(
    routed: Any,
    pins_by_net: Mapping[str, Sequence[Any]],
) -> Tuple[str, ...]:
    """Validate all physical-owner spellings under emit's uppercase convention."""

    from clay.layout.route.deck_isolation import _uppercase_net_set

    nets = getattr(routed, "nets", {}) or {}
    owner_spellings = tuple(str(net) for net in pins_by_net) + tuple(
        str(net) for net in nets
    )
    if not owner_spellings:
        return ()
    return _uppercase_net_set(
        owner_spellings,
        source="final emit physical owners",
    )


def _snap_emit_coordinate(value: int, grid_nm: int) -> int:
    """Apply the exact nearest-grid coordinate rule used by final emit snap."""

    value = int(value)
    grid = max(1, int(grid_nm))
    floor = (value // grid) * grid
    remainder = value - floor
    return floor if remainder * 2 < grid else floor + grid


def _final_emit_pin_layer_key(pin: Any, tech: LayoutTech) -> Tuple[int, int]:
    raw_layer = getattr(pin, "layer", None)
    if isinstance(raw_layer, tuple) and len(raw_layer) == 2:
        return tuple(int(value) for value in raw_layer)
    if isinstance(raw_layer, str):
        token = raw_layer.strip().lower()
        if token.startswith("metal"):
            suffix = token[5:]
        elif token.startswith("met"):
            suffix = token[3:]
        elif token.startswith("m"):
            suffix = token[1:]
        else:
            suffix = ""
        if suffix.isdigit() and int(suffix) > 0:
            return tuple(tech.metal_key(int(suffix)))
    raise ValueError(f"final emit terminal has unresolved layer: {raw_layer!r}")


def _final_emit_pin_xy_nm(pin: Any) -> Tuple[int, int]:
    xy_nm = getattr(pin, "xy_nm", None)
    if xy_nm is None:
        xy_um = getattr(pin, "xy_um", None)
        if xy_um is None or len(xy_um) != 2:
            raise ValueError(f"final emit terminal has no physical point: {pin!r}")
        xy_nm = tuple(int(round(float(value) * _NM)) for value in xy_um)
    if len(xy_nm) != 2:
        raise ValueError(f"final emit terminal has malformed point: {pin!r}")
    return int(xy_nm[0]), int(xy_nm[1])


def _final_emit_polygon_covers_point(polygon: Any, point: Any) -> bool:
    """Return whether ``point`` is in or exactly on the conductor polygon."""

    return bool(polygon.inside(point)) or any(
        edge.contains(point) for edge in polygon.each_edge()
    )


def _final_emit_snapped_components(
    polygon: Any,
    *,
    grid_nm: int,
    owner_kind: str,
    net: str,
    layer: Tuple[int, int],
) -> Tuple[Any, ...]:
    """Normalize one independently snapped polygon into positive-area components."""

    snapped_hull = tuple(
        (
            _snap_emit_coordinate(point.x, grid_nm),
            _snap_emit_coordinate(point.y, grid_nm),
        )
        for point in polygon.each_point_hull()
    )
    if len(set(snapped_hull)) < 3:
        raise ValueError(
            f"final snap collapsed a {owner_kind} conductor polygon: "
            f"net={net!r}, layer={layer!r}"
        )
    snapped_polygon = kdb.Polygon(
        [kdb.Point(x, y) for x, y in snapped_hull]
    )
    for hole_index in range(polygon.holes()):
        snapped_hole = tuple(
            (
                _snap_emit_coordinate(point.x, grid_nm),
                _snap_emit_coordinate(point.y, grid_nm),
            )
            for point in polygon.each_point_hole(hole_index)
        )
        if len(set(snapped_hole)) >= 3:
            snapped_polygon.insert_hole(
                [kdb.Point(x, y) for x, y in snapped_hole]
            )
    normalized = kdb.Region(snapped_polygon)
    normalized.merge()
    components = tuple(
        component for component in normalized.each() if component.area() > 0
    )
    if not components:
        raise ValueError(
            f"final snap collapsed a {owner_kind} conductor polygon: "
            f"net={net!r}, layer={layer!r}"
        )
    return components


def _normalize_final_emit_cell_polygons(layout: Any, cell: Any) -> int:
    """Replace self-touching post-snap contours with their exact Region components."""

    normalized_shape_count = 0
    for layer_index in layout.layer_indexes():
        shapes = cell.shapes(layer_index)
        replacements = []
        for shape in shapes.each():
            polygon = shape.polygon
            if polygon is None:
                continue
            hull = tuple((point.x, point.y) for point in polygon.each_point_hull())
            region = kdb.Region(polygon)
            region.merge()
            components = tuple(
                component for component in region.each() if component.area() > 0
            )
            if not components:
                raise ValueError(
                    "final snap collapsed an emitted conductor polygon: "
                    f"layer_index={int(layer_index)}"
                )
            if len(components) == 1 and len(set(hull)) == len(hull):
                continue
            replacements.append((shape, components))
        for old_shape, components in replacements:
            shapes.erase(old_shape)
            for component in components:
                shapes.insert(component)
            normalized_shape_count += 1
    return normalized_shape_count


def _snap_normalize_and_close_post_assembly(
    top: Any,
    ly: Any,
    label_points: Dict[str, Tuple[int, int]],
    tech: LayoutTech,
    *,
    grid_nm: int,
) -> Tuple[int, int]:
    """Stabilize the exact write-boundary geometry on grid and local DRC rules.

    Manufacturing-grid snap is a geometry transformation: independently rounded
    same-net polygons can acquire a sub-min-width junction even when the pre-snap
    assembly was clean.  Therefore the island-local, foreign-clipped close must
    run on the snapped geometry that will actually be written.  Any fill is
    snapped and checked again; three deterministic passes are sufficient for an
    idempotent nearest-grid map, otherwise emission fails closed rather than
    writing an unstabilized artifact.

    Returns ``(snapped_shape_count, inserted_fill_shape_count)`` accumulated over
    the stabilization passes.
    """
    from clay.layout.route.emit_gds import snap_kdb_cell_to_grid

    snapped_total = 0
    inserted_total = 0
    transform_evidence: List[Dict[str, Any]] = []
    for _pass in range(3):
        snapped = snap_kdb_cell_to_grid(top, ly, grid_nm)
        snapped_total += int(snapped)
        _normalize_final_emit_cell_polygons(ly, top)
        inserted = _close_same_net_m1_post_assembly(top, ly, label_points, tech)
        inserted_total += int(inserted)
        transform_evidence.extend(getattr(inserted, "evidence", ()))
        _normalize_final_emit_cell_polygons(ly, top)
        if snapped == 0 and inserted == 0:
            return snapped_total, _GeometryTransformCount(
                inserted_total, transform_evidence
            )
    raise RuntimeError(
        "final manufacturing-grid/width legalization did not stabilize in "
        "three deterministic passes"
    )


def _manufacturing_grid_transform_evidence(
    top: Any,
    ly: Any,
    *,
    grid_nm: int,
    limit: int = 32,
) -> Tuple[Dict[str, Any], ...]:
    """Describe a bounded set of polygons the write-boundary snap would move.

    Exact emission rejects every geometry-changing snap.  Historically that
    rejection reported only a count, which made a materializer/router grid
    contract violation indistinguishable from hundreds of unrelated shapes.
    Capture the original layer and bbox before snapping so the existing source
    binder can name the responsible placed stamp, route, and terminal.  This is
    diagnostic evidence only; it neither changes nor excuses the geometry.
    """
    g = int(grid_nm)
    cap = max(0, int(limit))
    if g <= 1 or cap == 0:
        return ()
    evidence: List[Dict[str, Any]] = []
    for layer_index in ly.layer_indexes():
        info = ly.get_info(layer_index)
        layer = [int(info.layer), int(info.datatype)]
        for shape in top.shapes(layer_index).each():
            polygon = shape.polygon
            if polygon is None:
                continue
            points = list(polygon.each_point_hull())
            for hole_index in range(polygon.holes()):
                points.extend(polygon.each_point_hole(hole_index))
            off_grid = [
                point for point in points
                if int(point.x) % g != 0 or int(point.y) % g != 0
            ]
            if not off_grid:
                continue
            bbox = polygon.bbox()
            evidence.append(
                {
                    "kind": "manufacturing_grid_snap",
                    "layer": layer,
                    "bbox_nm": [bbox.left, bbox.bottom, bbox.right, bbox.top],
                    "area_nm2": int(polygon.area()),
                    "grid_nm": g,
                    "off_grid_vertex_count": len(off_grid),
                    "vertex_residue_samples": [
                        [int(point.x) % g, int(point.y) % g]
                        for point in off_grid[:8]
                    ],
                }
            )
            if len(evidence) >= cap:
                return tuple(evidence)
    return tuple(evidence)


def _final_emit_terminal_probes(
    *,
    layout: Any,
    cell: Any,
    pins_by_net: Mapping[str, Sequence[Any]],
    tech: LayoutTech,
    grid_nm: int,
) -> Tuple[Any, ...]:
    """Bind terminal claims to their actual source polygons before final snap."""

    if not pins_by_net or not any(bool(pins) for pins in pins_by_net.values()):
        return ()

    from clay.layout.route.deck_isolation import DeckConnectStack, NetProbe

    stack = DeckConnectStack.from_tech(tech)
    conductor_layers = set(stack.conductors)
    probes = set()
    for raw_net in sorted(pins_by_net, key=lambda value: str(value)):
        net = str(raw_net).upper()
        for pin in pins_by_net[raw_net]:
            layer = _final_emit_pin_layer_key(pin, tech)
            if layer not in conductor_layers:
                raise ValueError(
                    "final emit terminal is outside the deck conductor stack: "
                    f"net={net!r}, layer={layer!r}"
                )
            x_nm, y_nm = _final_emit_pin_xy_nm(pin)
            raw_point = kdb.Point(x_nm, y_nm)
            layer_index = layout.find_layer(int(layer[0]), int(layer[1]))
            matched_source = False
            if layer_index is not None and int(layer_index) >= 0:
                for shape in cell.shapes(layer_index).each():
                    polygon = shape.polygon
                    if polygon is None or not _final_emit_polygon_covers_point(
                        polygon, raw_point
                    ):
                        continue
                    matched_source = True
                    for component in _final_emit_snapped_components(
                        polygon,
                        grid_nm=grid_nm,
                        owner_kind="terminal",
                        net=net,
                        layer=layer,
                    ):
                        probe_x, probe_y = min(
                            (point.x, point.y)
                            for point in component.each_point_hull()
                        )
                        probes.add(NetProbe(net, layer, probe_x, probe_y))
            if not matched_source:
                # Preserve an honest unresolved claim when the terminal point
                # did not lie on any pre-snap conductor.  Do not independently
                # snap the point: that can move it onto or off unrelated metal.
                probes.add(NetProbe(net, layer, x_nm, y_nm))
    return tuple(sorted(probes))


def _final_emit_route_probes(
    *,
    routed: Any,
    emitted_by_net: Mapping[str, Sequence[Any]],
    additional_regions_by_net: Mapping[
        str, Mapping[Tuple[int, int], Any]
    ],
    conductor_layers: Sequence[Tuple[int, int]],
    grid_nm: int,
) -> Tuple[Any, ...]:
    """Bind every route fragment to the actual per-net polygon emitted for it."""

    from clay.layout.route.deck_isolation import NetProbe

    conductor_layers = set(conductor_layers)
    probes = set()

    def polygon_of(layer_polygon: Any) -> Any:
        return kdb.Polygon(
            [
                kdb.Point(int(vertex.x), int(vertex.y))
                for vertex in layer_polygon.polygon.vertices
            ]
        )

    def probes_of(net: str, layer: Tuple[int, int], polygon: Any) -> Tuple[Any, ...]:
        return tuple(
            NetProbe(
                net,
                layer,
                *min(
                    (point.x, point.y)
                    for point in component.each_point_hull()
                ),
            )
            for component in _final_emit_snapped_components(
                polygon,
                grid_nm=grid_nm,
                owner_kind="route-owned",
                net=net,
                layer=layer,
            )
        )

    nets = getattr(routed, "nets", {}) or {}
    for raw_net in sorted(nets, key=lambda value: str(value)):
        net = str(raw_net).upper()
        emitted_by_layer: Dict[Tuple[int, int], list] = {}
        for emitted in emitted_by_net.get(raw_net, ()):
            layer = (int(emitted.layer), int(emitted.datatype))
            if layer not in conductor_layers:
                continue
            emitted_by_layer.setdefault(layer, []).append(polygon_of(emitted))
        for raw_layer, region in additional_regions_by_net.get(raw_net, {}).items():
            layer = (int(raw_layer[0]), int(raw_layer[1]))
            if layer not in conductor_layers:
                continue
            emitted_by_layer.setdefault(layer, []).extend(
                kdb.Polygon(polygon) for polygon in region.each()
            )

        # Every emitted component is ownership-bearing, including one made only
        # from same-net heal/fill material.
        for layer in sorted(emitted_by_layer):
            for polygon in emitted_by_layer[layer]:
                probes.update(probes_of(net, layer, polygon))

        # Also prove that no original router fragment disappeared at the merge
        # boundary. A fragment may bind to a larger merged polygon, but never to
        # an independently snapped reconstruction of itself.
        for wire in getattr(nets[raw_net], "wires", ()) or ():
            layer = (int(wire.layer), int(wire.datatype))
            if layer not in conductor_layers:
                continue
            source = kdb.Region(polygon_of(wire))
            matched = False
            for polygon in emitted_by_layer.get(layer, ()):
                if (source & kdb.Region(polygon)).area() <= 0:
                    continue
                matched = True
                probes.update(probes_of(net, layer, polygon))
            if not matched:
                raise ValueError(
                    "final emit omitted a route-owned conductor fragment: "
                    f"net={net!r}, layer={layer!r}"
                )

    return tuple(sorted(probes))


def _final_emit_deck_isolation_report(
    *,
    layout: Any,
    cell: Any,
    routed: Any,
    pins_by_net: Mapping[str, Sequence[Any]],
    terminal_probes: Sequence[Any],
    emitted_route_polygons: Mapping[str, Sequence[Any]],
    additional_owned_route_regions: Mapping[
        str, Mapping[Tuple[int, int], Any]
    ],
    tech: LayoutTech,
    grid_nm: int,
    spacing_constraints: Sequence[Any] = (),
) -> Any:
    """Check final snapped geometry using every physical ownership claim."""

    from clay.layout.route.deck_isolation import (
        DeckConnectStack,
        NetProbe,
        check_deck_net_isolation,
    )

    _canonical_final_emit_owners(routed, pins_by_net)

    stack = DeckConnectStack.from_tech(tech)
    route_probes = _final_emit_route_probes(
        routed=routed,
        emitted_by_net=emitted_route_polygons,
        additional_regions_by_net=additional_owned_route_regions,
        conductor_layers=stack.conductors,
        grid_nm=grid_nm,
    )
    probes = [
        NetProbe(probe.net, probe.layer, probe.x, probe.y)
        for probe in (*terminal_probes, *route_probes)
    ]

    return check_deck_net_isolation(
        layout=layout,
        cell=cell,
        stack=stack,
        probes=tuple(probes),
        spacing_constraints=spacing_constraints,
    )


def _final_emit_deck_route_stops(report: Any) -> Tuple[Any, ...]:
    """Map one canonical deck report onto existing negotiation STOP types."""

    from clay.layout.route.engine_gridless import (
        NegotiationRouteStop,
        NegotiationStopReason,
    )

    if report.merged_net_groups:
        return tuple(
            NegotiationRouteStop(
                NegotiationStopReason.DECK_NET_MERGE,
                group[0],
                "final_emit",
                (tuple(group),),
                (("merged_nets", tuple(group)),),
            )
            for group in report.merged_net_groups
        )
    open_nets = set(report.ambiguous_nets)
    open_nets.update(net for net, _layer, _x, _y in report.unresolved_probes)
    spacing_stops = tuple(
        NegotiationRouteStop(
            NegotiationStopReason.NO_LEGAL_ROUTE,
            item.net_a, "final_emit_spacing", tuple(item.violations),
            (("net_pair", (item.net_a, item.net_b)), ("required_nm", item.required_nm),
             ("status", item.status), ("missing_nets", item.missing_nets)),
        )
        for item in report.spacing_reports if item.status != "pass"
    )
    return spacing_stops + tuple(
        NegotiationRouteStop(
            NegotiationStopReason.NO_LEGAL_ROUTE,
            net,
            "final_emit",
            (),
            (
                ("ambiguous", net in report.ambiguous_nets),
                (
                    "unresolved_probes",
                    tuple(
                        item for item in report.unresolved_probes if item[0] == net
                    ),
                ),
            ),
        )
        for net in sorted(open_nets)
    )


def emit_labelled_gds(
    routed: Any,
    intent: Any,
    geometry: Dict[str, Any],
    placement: Dict[str, Any],
    pins_by_net: Dict[str, List[Any]],
    out_path: Path,
    *,
    tech: LayoutTech,
    top_cell: str = _TOP_CELL,
    merge: bool = True,
    pin_nets: Optional[Sequence[str]] = None,
    devices: Optional[Sequence[Any]] = None,
    passives: Optional[Sequence[Any]] = None,
    source_cell: Any = None,
    native_geometry: Any = None,
    allocated_route_manifest: Optional[Sequence[Any]] = None,
    return_exact_artifact: bool = False,
    row_program: Any = None,
    row_materialization: Any = None,
    row_final_certificate: Any = None,
    selected_route_engine: Any = None,
) -> Any:
    """Write the FULL routed OTA layout (device geometry + cap cells + routed wires +
    one on-metal metal1 net-name label per net) to a flat GDS. Leakage-free: every
    shape is a device cell or the router output, never a reference routed layout.

    Emission hygiene (P1): the routed wires are merged PER NET x LAYER at this
    write boundary (:func:`clay.layout.route.emit_gds.merge_net_wires`), with each
    net's same-net M1 heal material (:func:`_heal_same_net_m1_extras`) unioned
    into the OWNING net's merge -- never another net's. Device / cap artwork is
    emitted verbatim, and the in-memory route structure is untouched.
    ``merge=False`` keeps the historical raw-fragment emission (A/B
    oracle-invariance audits only; the heal regions are identical either way).

    ``return_exact_artifact`` is the exact route-artifact evidence seam.  It
    requires an exhaustive router-owned manifest and returns a typed artifact only
    after proving
    that the manifest's per-net/per-layer union equals all router wires and that
    emitter heal/fill/snap transforms added or moved no geometry.  This is why a
    raw ``RoutedCell.tagged`` list is never silently promoted to an exhaustive
    emitted manifest.  The default remains the historical :class:`Path` return.
    """
    _delete_stale_emit_target(out_path)

    from clay.layout import passive_device_cell as _pdc
    from clay.layout.route.emit_gds import emit_net_labels, net_label_points
    from clay.layout.route.pins import _geom_extent_center_nm

    diagnostic_only = getattr(routed, "diagnostic_only", False)
    if type(diagnostic_only) is not bool:
        raise TypeError("routed.diagnostic_only must be an exact bool")
    if diagnostic_only and (
        allocated_route_manifest is not None or return_exact_artifact
    ):
        raise ValueError(
            "diagnostic-only route cannot mint exact/signoff evidence"
        )
    tech = _refine_tech_with_mim_sources(
        tech,
        source_cell,
        intent,
        geometry=geometry,
        passives=getattr(intent, "passives", None),
    )
    _canonical_final_emit_owners(routed, pins_by_net)

    from clay.layout.row_terminal_probes import require_routing_row_terminal_probes as require_row_terminal_probes

    require_row_terminal_probes(
        tech=tech, geometry=geometry, placement_um=placement, pins_by_net=pins_by_net,
        row_program=row_program, row_materialization=row_materialization,
        row_final_certificate=row_final_certificate, intent=intent, source_cell=source_cell,
        allow_fixed_multilayer=bool(getattr(selected_route_engine,'consumes_fixed_multilayer_joint_terminals',False)),
        multilayer_engine=selected_route_engine,
    )

    ly = kdb.Layout()
    ly.dbu = 0.001
    top = ly.create_cell(top_cell)

    if return_exact_artifact and allocated_route_manifest is None:
        raise ValueError(
            "return_exact_artifact requires an allocated route manifest"
        )
    if return_exact_artifact and not merge:
        raise ValueError("exact artifact emission requires canonical merged emission")

    allocated_union_matches_router = False
    allocated_route_union_sha256 = ""
    router_route_union_sha256 = ""
    emitted_route_union_sha256 = ""
    if allocated_route_manifest is not None:
        from clay.layout.route.emit_gds import net_layer_regions
        from clay.layout.route.exact_artifact_evidence import (
            _entries,
            _net_layer_regions_sha256,
        )
        from clay.layout.route.via_provenance_check import route_manifest_sha256

        # Validate the bounded public tuple shape before any geometry is emitted.
        allocated_route_manifest = _entries(allocated_route_manifest)
        route_manifest_sha256(allocated_route_manifest)
        allocated_regions: Dict[str, Dict[Tuple[int, int], kdb.Region]] = {}
        for raw_net, raw_key, raw_box in allocated_route_manifest:
            net = str(raw_net)
            key = (int(raw_key[0]), int(raw_key[1]))
            box = tuple(int(value) for value in raw_box)
            allocated_regions.setdefault(net, {}).setdefault(key, kdb.Region()).insert(
                kdb.Box(*box)
            )
        for by_layer in allocated_regions.values():
            for region in by_layer.values():
                region.merge()
        routed_regions = {
            net: by_layer
            for net, by_layer in net_layer_regions(
                getattr(routed, "nets", {}) or {}
            ).items()
            if by_layer
        }
        allocated_route_union_sha256 = _net_layer_regions_sha256(allocated_regions)
        router_route_union_sha256 = _net_layer_regions_sha256(routed_regions)
        allocated_union_matches_router = (
            set(allocated_regions) == set(routed_regions)
            and all(
                set(allocated_regions[net]) == set(routed_regions[net])
                and all(
                    (allocated_regions[net][key] - routed_regions[net][key]).is_empty()
                    and (routed_regions[net][key] - allocated_regions[net][key]).is_empty()
                    for key in allocated_regions[net]
                )
                for net in allocated_regions
            )
        )
        if return_exact_artifact and not allocated_union_matches_router:
            raise ValueError(
                "allocated route manifest does not exactly cover the router-owned union"
            )
        emitted_route_union_sha256 = router_route_union_sha256

    def li(layer, dt):
        return ly.layer(int(layer), int(dt))

    # One source-owned well assembly is shared with exact-structure replay.
    well_layer_keys, well_assembly = _source_well_assembly(
        intent, tech, geometry, placement, devices=devices, source_cell=source_cell,
    )
    for name, g in geometry.items():
        pos = placement.get(name)
        if pos is None:
            continue
        for layer, rings in (getattr(g, "polys_by_layer", {}) or {}).items():
            layer_key = (int(layer[0]), int(layer[1]))
            if layer_key in well_layer_keys:
                continue
            for ring in rings:
                if len(ring) < 3:
                    continue
                pts = [_KDB_POINT(x, y) for x, y in placed_ring_nm(ring, pos)]
                top.shapes(li(*layer_key)).insert(_KDB_POLYGON(pts))

    from clay.layout.spec.exact_structure_profile import WellMergeCertificate, _region_sha256
    certificates = []
    for layer_key, (input_well, regions) in sorted(well_assembly.items()):
        output_well = kdb.Region()
        well_li = li(*layer_key)
        for region in regions:
            output_well += region
            for polygon in region.each():
                top.shapes(well_li).insert(polygon)
        output_well.merge()
        certificates.append(WellMergeCertificate(layer_key=layer_key,
            input_region_sha256=_region_sha256(input_well),
            output_region_sha256=_region_sha256(output_well)))
    well_merge_certificates = tuple(certificates)

    # passive device cells (centroid-relative nm) -> placement. Caps always come
    # from the parametric cap cell (historical contract). RESISTORS are emitted
    # here ONLY when no placed stamp in ``geometry`` owns them. On the PRODUCTION
    # routes every PDK now carries the resistor stamp in the geometry dict
    # (gf180/external_pdk: passive_synth facts; external_pdk: realize_render.
    # parametric_resistor_geometry -- the router-obstacle-blindness root fix), so
    # this block is the GEOMETRY-ABSENT fallback (bare emit calls without a
    # stamp), built from the SAME declared constructor
    # (build_segmented_resistor or the unannotated legacy build_resistor)
    # + the same centroid recentre the pin extractor uses
    # (pins.py::_build_passive_cell + _geom_extent_center_nm) -- single source,
    # the artwork is co-located with the pins by construction. A PDK whose
    # resistor cell cannot build (missing marker layers) keeps its honest
    # absence -- never fabricated. A typed segmented declaration instead fails
    # closed because falling back would disconnect emission from placement.
    passive_heal_stamps: Dict[str, Any] = {}
    for p in getattr(intent, "passives", []) or []:
        kind = str(getattr(p, "kind", "")).lower()
        is_cap = kind.startswith("cap") or kind == "c"
        is_res = kind.startswith("res") or kind == "r"
        if not (is_cap or is_res):
            continue
        pos = placement.get(p.name)
        if pos is None:
            continue
        if p.name in geometry:
            # a placed stamp in the geometry dict (the general route_cell bridge
            # carries the placer's OWN cap/resistor artwork as a stamp) was already
            # drawn by the device-geometry loop above; drawing the rebuilt cell here
            # too would DOUBLE-draw the plates / poly body. The stamp is the canonical
            # single source -- skip the rebuild. (OTA geometry carries resistor stamps
            # only, so its caps still take the rebuild path below; byte-unchanged.)
            continue
        if is_cap:
            cell = _pdc.build_capacitor_for_tech(
                tech, float(p.value),
                nets={"p": str(p.nets.get("p", "")), "n": str(p.nets.get("n", ""))},
                m=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
            )
        else:
            if p.name in geometry:
                continue  # the placed stamp (device-geometry loop above) owns it
            segment_intent = getattr(p, "resistor_segments", None)
            try:
                if segment_intent is not None:
                    from clay.layout.annotations import ResistorSegments

                    if type(segment_intent) is not ResistorSegments:
                        raise TypeError(
                            "resistor_segments LVS view must be exact "
                            "ResistorSegments"
                        )
                    if segment_intent.pdk != str(getattr(tech, "pdk_id", "")):
                        raise ValueError(
                            "resistor_segments LVS view does not match active PDK"
                        )
                    cell = _pdc.build_segmented_resistor(
                        tech,
                        float(p.value),
                        count=segment_intent.count,
                        connection=segment_intent.connection,
                        segment_width_um=segment_intent.width_um,
                        segment_length_um=segment_intent.length_um,
                        segment_resistance_ohms=(
                            segment_intent.segment_resistance_ohms
                        ),
                        value_quantum_ohms=segment_intent.value_quantum_ohms,
                        segment_resistance_quantum_ohms=(
                            segment_intent.segment_resistance_quantum_ohms
                        ),
                        model=str(getattr(p, "model", "") or ""),
                        orient=str(getattr(p, "orient", "R0") or "R0"),
                        nets={
                            "p": str(p.nets.get("p", "")),
                            "n": str(p.nets.get("n", "")),
                        },
                    )
                else:
                    cell = _pdc.build_resistor(
                        tech, float(p.value),
                        nets={"p": str(p.nets.get("p", "")),
                              "n": str(p.nets.get("n", ""))},
                    )
            except Exception:
                if segment_intent is not None:
                    raise
                continue  # no parametric resistor cell on this PDK: honest absence
        cx, cy = _geom_extent_center_nm(cell)
        px = int(round(pos[0] * _NM))
        py = int(round(pos[1] * _NM))
        # RIGID integer translation: the cell's vertices are integer-nm and
        # on-grid by construction (e.g. a 260nm via2 cut). Subtracting a FRACTIONAL
        # centroid per-vertex and truncating (``int(v.x - cx + px)``) rounded each
        # edge independently, deforming a 260nm cut to 259/261 (the V2.1 via-size
        # violations). A single integer offset (round the centroid once, then add)
        # is a rigid shift that PRESERVES every drawn dimension exactly; the
        # emit-boundary grid snap then quantises the absolute position uniformly.
        ox = px - int(round(cx))
        oy = py - int(round(cy))
        for lp in cell.polygons:
            top.shapes(li(lp.layer, lp.datatype)).insert(
                kdb.Polygon([kdb.Point(int(v.x) + ox, int(v.y) + oy)
                             for v in lp.polygon.vertices])
            )
        if is_res:
            # let the same-net M1 heal/collar OWN the head pads (see
            # _PassiveHealStamp); caps need no entry (their pads are upper metal).
            passive_heal_stamps[p.name] = _PassiveHealStamp(cell, cx, cy)
    heal_geometry = ({**geometry, **passive_heal_stamps}
                     if passive_heal_stamps else geometry)

    # routed wires + same-net M1 heal (gap/notch closing at the wire-to-device-pad
    # interface; same-net merge is always safe, bridges are clipped clear of foreign
    # nets). Merged path: heal material joins the owning net's emit-boundary merge.
    heal_shape_count = 0
    upper_fill_shape_count = 0
    transformation_evidence: List[Dict[str, Any]] = []
    emitted_route_polygons: Dict[str, Tuple[Any, ...]] = {}
    additional_owned_route_regions: Dict[
        str, Dict[Tuple[int, int], Any]
    ] = {}
    m1 = tuple(tech.metal_key(1))
    if merge and getattr(routed, "nets", None):
        from clay.layout.route.emit_gds import merge_net_wires, net_layer_regions

        # ALL M1 destined for the cell: device/cap artwork already in `top` plus
        # every net's merged wires -- the same region the unmerged path read back
        # from the cell, so the heal computes identical bridge material.
        all_m1 = kdb.Region(top.shapes(li(m1[0], m1[1])))
        for by_layer in net_layer_regions(routed.nets).values():
            r = by_layer.get(m1)
            if r is not None:
                all_m1 += r
        all_m1.merge()
        extras = _heal_same_net_m1_extras(
            all_m1, routed, heal_geometry, placement, pins_by_net, tech)
        # SAME-NET upper-metal (M2+) notch closing: each net's own re-entrant
        # sub-min-space gaps (wire-vs-own-via-pad), filled clear of foreign metal.
        upper_extras = _close_same_net_upper_metal_extras(routed, tech, min_gi=2)
        upper_extras = _exclude_existing_cell_material_from_upper_extras(
            upper_extras,
            top,
            ly,
        )
        heal_shape_count = sum(
            sum(1 for _ in region.each()) for region in extras.values()
        )
        upper_fill_shape_count = sum(
            sum(1 for _ in region.each())
            for by_layer in upper_extras.values()
            for region in by_layer.values()
        )
        if return_exact_artifact:
            for net, region in sorted(extras.items()):
                for polygon in region.each():
                    bbox = polygon.bbox()
                    transformation_evidence.append(
                        {
                            "kind": "terminal_interface_heal",
                            "net": str(net),
                            "layer": [int(m1[0]), int(m1[1])],
                            "bbox_nm": [
                                bbox.left,
                                bbox.bottom,
                                bbox.right,
                                bbox.top,
                            ],
                            "area_nm2": int(polygon.area()),
                        }
                    )
            for net, by_layer in sorted(upper_extras.items()):
                for key, region in sorted(by_layer.items()):
                    for polygon in region.each():
                        bbox = polygon.bbox()
                        transformation_evidence.append(
                            {
                                "kind": "upper_metal_notch_fill",
                                "net": str(net),
                                "layer": [int(key[0]), int(key[1])],
                                "bbox_nm": [
                                    bbox.left,
                                    bbox.bottom,
                                    bbox.right,
                                    bbox.top,
                                ],
                                "area_nm2": int(polygon.area()),
                            }
                        )
        if return_exact_artifact:
            # Replay the per-net/layer union that the merged emitter is about to
            # draw.  This catches geometry-changing heal/fill even when an
            # implementation accidentally under-reports its shape count.
            emitted_route_regions = {}
            for net, by_layer in routed_regions.items():
                copied_by_layer = {}
                for key, region in by_layer.items():
                    copied = kdb.Region()
                    copied += region
                    copied_by_layer[key] = copied
                emitted_route_regions[net] = copied_by_layer
            for net, region in extras.items():
                emitted_route_regions.setdefault(net, {}).setdefault(
                    m1, kdb.Region()
                )
                emitted_route_regions[net][m1] += region
            for net, by_layer in upper_extras.items():
                for key, region in by_layer.items():
                    emitted_route_regions.setdefault(net, {}).setdefault(
                        key, kdb.Region()
                    )
                    emitted_route_regions[net][key] += region
            emitted_route_union_sha256 = _net_layer_regions_sha256(
                emitted_route_regions
            )
        for raw_net in sorted(routed.nets):
            emitted = tuple(
                merge_net_wires(
                    {raw_net: routed.nets[raw_net]},
                    extra_m1=extras,
                    m1_key=m1,
                    extra_by_layer=upper_extras,
                )
            )
            emitted_route_polygons[raw_net] = emitted
            for lp in emitted:
                top.shapes(li(lp.layer, lp.datatype)).insert(_poly(lp))
    else:
        for net, nr in routed.nets.items():
            emitted_route_polygons[net] = tuple(nr.wires)
            for lp in nr.wires:
                top.shapes(li(lp.layer, lp.datatype)).insert(_poly(lp))
        heal_shape_count, unmerged_heal_regions = _heal_same_net_m1(
            top, ly, routed, heal_geometry, placement, pins_by_net, tech
        )
        for net, region in unmerged_heal_regions.items():
            additional_owned_route_regions.setdefault(net, {})[m1] = region

    # net-name labels on metal1 (the deck's metal1_label layer). When ``pin_nets`` is
    # given, only those nets are labelled, so the extracted layout's TOP-LEVEL PIN set
    # equals the schematic ``.subckt`` port set (the foundry deck runs TOP_LVL_PINS=
    # false -> a labelled net is a top-level pin; labelling internal nets too would
    # give the layout more pins than the schematic and the comparer rejects the whole
    # match). The port list is the cell's own ports -- fully generic, no per-circuit
    # branch. ``None`` keeps the historical behaviour (label every net).
    pts = net_label_points(routed, pins_by_net, tech=tech,
                           geometry=geometry, placement=placement)
    if pin_nets is not None:
        allowed = {str(n).upper() for n in pin_nets}
        label_pts = {net: xy for net, xy in pts.items() if str(net).upper() in allowed}
    else:
        label_pts = pts
    emit_net_labels(top, ly, label_pts, tech=tech, pin_nets=pin_nets)

    from clay.layout.assembly_boundary import (
        materialize_assembly_boundary,
        require_assembly_boundary,
    )

    if return_exact_artifact:
        # Nonconducting PDK metadata belongs to the declared assembly before
        # the exact pre/post geometry seal; source PCell markers stay intact.
        materialize_assembly_boundary(ly, top, tech, geometry, placement)

    pre_transform_geometry_sha256 = ""
    if return_exact_artifact:
        from clay.layout.route.exact_artifact_evidence import (
            _cell_polygon_union_sha256,
        )

        pre_transform_geometry_sha256 = _cell_polygon_union_sha256(ly, top)

    # FINAL same-net M1 close over the fully-assembled cell: close any sub-min-space
    # SAME-NET M1 notch the collar opened between own metal and a same-net device
    # finger (the gf180 comparator OUTP/VDD M1.2a), partitioned by connectivity to the
    # net labels and foreign-clipped so it never bridges a different net. Runs on the
    # ground-truth emitted geometry (after all artwork + heal material is in `top`),
    # BEFORE the uniform grid snap.
    post_assembly_fill_shape_count = _close_same_net_m1_post_assembly(
        top, ly, pts, tech
    )
    transformation_evidence.extend(
        getattr(post_assembly_fill_shape_count, "evidence", ())
    )

    # MANUFACTURING-GRID CONTRACT: snap every drawn polygon to the PDK grid at
    # this write boundary (engine/device-stamp nm math is untouched; only the
    # emitted coordinates are quantised). Device rings (um->nm rounding) and the
    # placement-offset shift land vertices off the foundry's ongrid(g); without
    # this they fire tens of thousands of ``*_OFFGRID`` violations on the real
    # deck. The snap is per-polygon (no cross-net merge), so it never fabricates
    # a short, and a uniform offset preserves widths/spacings. (gf180mcu g=5nm,
    # parsed from the deck; PDKs with no foundry deck degrade to a 1nm no-op.)
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    grid_nm = pdk_manufacturing_grid_nm(tech)
    require_row_terminal_probes(
        tech=tech, geometry=geometry, placement_um=placement, pins_by_net=pins_by_net,
        row_program=row_program, row_materialization=row_materialization,
        row_final_certificate=row_final_certificate, intent=intent, source_cell=source_cell,
        allow_fixed_multilayer=bool(getattr(selected_route_engine,'consumes_fixed_multilayer_joint_terminals',False)),
        multilayer_engine=selected_route_engine,
    )
    terminal_probes = _final_emit_terminal_probes(
        layout=ly,
        cell=top,
        pins_by_net=pins_by_net,
        tech=tech,
        grid_nm=grid_nm,
    )
    grid_transform_evidence = _manufacturing_grid_transform_evidence(
        top,
        ly,
        grid_nm=grid_nm,
    )
    snapped_shape_count, post_grid_fill_shape_count = (
        _snap_normalize_and_close_post_assembly(
            top,
            ly,
            pts,
            tech,
            grid_nm=grid_nm,
        )
    )
    post_assembly_fill_shape_count += post_grid_fill_shape_count
    transformation_evidence.extend(
        getattr(post_grid_fill_shape_count, "evidence", ())
    )
    if snapped_shape_count:
        transformation_evidence.extend(grid_transform_evidence)
    if return_exact_artifact:
        require_assembly_boundary(ly, top, tech, geometry, placement)
    else:
        # Ordinary emission may snap artwork. Derive its boundary from the
        # actual final masks and original source extent after those changes.
        materialize_assembly_boundary(ly, top, tech, geometry, placement)
    post_transform_geometry_sha256 = ""
    if return_exact_artifact:
        post_transform_geometry_sha256 = _cell_polygon_union_sha256(ly, top)

    from dataclasses import replace
    from clay.layout.route.constraints import Spacing

    spacing_constraints = list(getattr(routed, "spacing_constraints", ()))
    if source_cell is not None:
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.net_spacing import lower_net_spacings
        source_spacings = layout_intent_from_cell_def(source_cell).net_spacings
        spacing_constraints.extend(lower_net_spacings(source_spacings, set(pins_by_net)))
    if any(type(c) is not Spacing for c in spacing_constraints):
        raise TypeError("emitted net spacing requires Spacing declarations")
    # Physical ownership probes use the same canonical uppercase spelling.
    spacing_constraints = tuple(dict.fromkeys(
        replace(c, net_a=c.net_a.upper(), net_b=c.net_b.upper()) for c in spacing_constraints
    ))
    try:
        deck_report = _final_emit_deck_isolation_report(
            layout=ly,
            cell=top,
            routed=routed,
            pins_by_net=pins_by_net,
            terminal_probes=terminal_probes,
            emitted_route_polygons=emitted_route_polygons,
            additional_owned_route_regions=additional_owned_route_regions,
            tech=tech,
            grid_nm=grid_nm,
            spacing_constraints=spacing_constraints,
        )
    except Exception:
        _delete_stale_emit_target(out_path)
        raise
    if (
        not deck_report.cross_net_isolated
        or (not diagnostic_only and not deck_report.probe_complete)
        or not deck_report.spacing_satisfied
    ):
        _delete_stale_emit_target(out_path)
        raise DeckIsolationWriteRefused(
            report=deck_report,
            route_stops=_final_emit_deck_route_stops(deck_report),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_options = kdb.SaveLayoutOptions()
    save_options.gds2_write_timestamps = False
    ly.write(str(out_path), save_options)
    channel_source = (None if intent is None else
                      tuple(getattr(intent, "channel_routes", ())))
    if source_cell is not None:
        from clay.layout.intent import layout_intent_from_cell_def
        channel_source = layout_intent_from_cell_def(source_cell).channel_routes
    if getattr(routed, "channel_routes", ()) or channel_source:
        from clay.layout.route.channel_check import require_emitted_channels
        try:
            routed.emitted_channel_report = require_emitted_channels(
                out_path, routed, tech, top_cell=top_cell,
                terminal_probes=terminal_probes, source_channels=channel_source,
            )
        except Exception:
            _delete_stale_emit_target(out_path)
            raise
    if return_exact_artifact:
        try:
            from clay.layout.route.exact_artifact_evidence import (
                _emitted_device_snapshot_sha256,
                _finalize_verified_emission,
                _terminal_pin_snapshot_sha256,
            )

            bound_transform_evidence = _bind_transform_evidence_sources(
                transformation_evidence,
                geometry=geometry,
                placement=placement,
                routed=routed,
                pins_by_net=pins_by_net,
                tech=tech,
            )

            return _finalize_verified_emission(
                out_path,
                allocated_route_manifest,
                top_cell=top_cell,
                allocated_route_union_sha256=allocated_route_union_sha256,
                router_route_union_sha256=router_route_union_sha256,
                emitted_route_union_sha256=emitted_route_union_sha256,
                emitted_device_snapshot_sha256=(
                    _emitted_device_snapshot_sha256(geometry, placement)
                ),
                terminal_pin_snapshot_sha256=(
                    _terminal_pin_snapshot_sha256(pins_by_net)
                ),
                source_cell=source_cell,
                source_devices=tuple(devices or ()),
                source_passives=tuple(passives or ()),
                source_native_geometry=native_geometry,
                source_pdk_id=tech.pdk_id,
                source_tech=tech,
                emitted_device_geometry=geometry,
                emitted_device_placement_um=placement,
                pre_transform_geometry_sha256=pre_transform_geometry_sha256,
                post_transform_geometry_sha256=post_transform_geometry_sha256,
                heal_shape_count=heal_shape_count,
                upper_fill_shape_count=upper_fill_shape_count,
                post_assembly_fill_shape_count=post_assembly_fill_shape_count,
                snapped_shape_count=snapped_shape_count,
                well_merge_certificates=well_merge_certificates,
                transformation_evidence=bound_transform_evidence,
            )
        except Exception:
            _delete_stale_emit_target(out_path)
            raise
    return out_path


# --------------------------------------------------------------------------- #
# verdict parsing (tool stdout + structured lvsdb cross-reference)
# --------------------------------------------------------------------------- #
def _verbatim_verdict(combined: str) -> str:
    """The verbatim verdict line(s) KLayout printed -- never paraphrased."""
    keep: List[str] = []
    for ln in combined.splitlines():
        s = ln.strip()
        if ("Netlists match" in s) or ("Netlists don't match" in s):
            keep.append(s)
    return "\n".join(keep)


def _extracted_device_cards(netlist_text: str) -> List[str]:
    """The device cards (M.../C.../R...) in an extracted SPICE netlist."""
    out: List[str] = []
    for ln in netlist_text.splitlines():
        s = ln.strip()
        if s[:1] in ("M", "C", "R", "D", "Q") and " " in s and not s.startswith("*"):
            out.append(s)
    return out


def _name_of(x: Any) -> str:
    if x is None:
        return "NONE"
    try:
        return x.expanded_name()
    except Exception:
        try:
            return x.name
        except Exception:
            return str(x)


def _parse_lvsdb_unmatched(lvsdb_path: Path) -> Tuple[bool, Dict[str, List[str]]]:
    """Parse the lvsdb cross-reference. Returns ``(circuit_clean, unmatched)`` where
    ``circuit_clean`` is True iff EVERY circuit pair status is a clean match and there
    is no unmatched net or device. ``unmatched`` is ``{"nets": [...], "devices": [...]}``
    listing every non-matched pair as ``"<status> layout=<a> schem=<b>"``."""
    unmatched: Dict[str, List[str]] = {"nets": [], "devices": []}
    db = kdb.LayoutVsSchematic()
    db.read(str(lvsdb_path))
    xref = db.xref()
    if xref is None:
        return False, unmatched

    ok_states = ("Match", "MatchWithWarning")
    circuit_clean = True
    n_circuits = 0
    for cp in xref.each_circuit_pair():
        n_circuits += 1
        if str(cp.status()) not in ok_states:
            circuit_clean = False
        for np_ in xref.each_net_pair(cp):
            st = str(np_.status())
            if st not in ok_states:
                unmatched["nets"].append(
                    f"{st} layout={_name_of(np_.first())} schem={_name_of(np_.second())}"
                )
        for dp in xref.each_device_pair(cp):
            st = str(dp.status())
            if st not in ok_states:
                unmatched["devices"].append(
                    f"{st} layout={_name_of(dp.first())} schem={_name_of(dp.second())}"
                )
    if n_circuits == 0:
        circuit_clean = False
    if unmatched["nets"] or unmatched["devices"]:
        circuit_clean = False
    return circuit_clean, unmatched


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
class LvsReferenceStop(RuntimeError):
    """Typed fail-closed stop emitted while compiling a route-cell reference."""

    def __init__(self, *, state: str, refs: Sequence[Any] = ()) -> None:
        self.phase = "REFERENCE_COMPILE"
        self.state = str(state)
        self.refs = tuple(str(ref) for ref in refs)
        suffix = f" refs={self.refs!r}" if self.refs else ""
        super().__init__(f"{self.phase}: {self.state}{suffix}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reference_semantics_sha256(text: str) -> str:
    """Order-insensitive digest of one canonical flat reference netlist."""
    subckts: List[str] = []
    cards: List[str] = []
    directives: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        lowered = line.lower()
        if lowered.startswith(".subckt "):
            subckts.append(" ".join(line.split()))
        elif lowered.startswith(".ends"):
            continue
        elif line.startswith("."):
            directives.append(" ".join(line.split()))
        else:
            cards.append(" ".join(line.split()))
    payload = "\n".join(
        [*sorted(subckts), *sorted(directives), *sorted(cards)]
    )
    return _sha256_text(payload)


def _source_subckt_ports(source_text: str, cell_def: Any) -> Tuple[str, ...]:
    target = str(
        getattr(cell_def, "local_name", None)
        or getattr(cell_def, "name", None)
        or ""
    ).lower()
    declarations: List[Tuple[str, Tuple[str, ...]]] = []
    for raw_line in source_text.splitlines():
        line = raw_line.split("$", 1)[0].strip()
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].lower() == ".subckt":
            declarations.append((tokens[1], tuple(tokens[2:])))
    matches = [ports for name, ports in declarations if name.lower() == target]
    if len(matches) == 1:
        return matches[0]
    if len(declarations) == 1:
        return declarations[0][1]
    raise LvsReferenceStop(
        state="SOURCE_SUBCKT_AMBIGUOUS_OR_MISSING",
        refs=tuple(name for name, _ports in declarations),
    )


def _source_cir_reference_text(
    source_cir: Path,
    cell_def: Any,
    *,
    tech_id: str,
    top_cell: str,
    full_extraction: bool,
) -> str:
    """Compile source-CIR semantics through the canonical reference emitter."""
    from types import SimpleNamespace
    from clay.layout.extract.netlist_intent import model_oxide

    from clay.layout.generic_placement import (
        parse_cir_mos_records,
        parse_cir_passive_records,
    )

    path = Path(source_cir).resolve()
    try:
        source_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LvsReferenceStop(
            state="SOURCE_CIR_UNREADABLE", refs=(path,)
        ) from exc
    if not source_text.strip():
        raise LvsReferenceStop(state="SOURCE_CIR_EMPTY", refs=(path,))
    mos_records = parse_cir_mos_records(path)
    passive_records = parse_cir_passive_records(path)
    if not mos_records and not passive_records:
        raise LvsReferenceStop(
            state="SOURCE_CIR_HAS_NO_PARSED_DEVICES", refs=(path,)
        )
    devices = [
        SimpleNamespace(
            name=(name[1:] if name[:1].upper() == "M" else name),
            kind=record["kind"],
            oxide=model_oxide(re.sub(r"\.\d+$", "", str(record.get("model", "")))),
            W=float(record["W"]) * 1e6,
            L=float(record["L"]) * 1e6,
            M=int(record["nf"]),
            nets={
                "drain": record["pins"]["d"],
                "gate": record["pins"]["g"],
                "source": record["pins"]["s"],
                "body": record["pins"]["b"],
            },
        )
        for name, record in sorted(mos_records.items())
    ]
    passives = [
        SimpleNamespace(
            name=(name[1:] if name[:1].upper() in ("R", "C") else name),
            kind=record["kind"],
            value=float(record["value"]),
            M=int(record["M"]),
            nets={"p": record["pins"]["p"], "n": record["pins"]["n"]},
        )
        for name, record in sorted(passive_records.items())
    ]
    _normalize_substrate_bulk_bodies(devices)
    cfg = _resolved_lvs_config(
        tech_id, full_extraction=full_extraction
    )
    return build_reference_netlist(
        _fold_passive_banks(
            SimpleNamespace(devices=devices, passives=passives)
        ),
        RefNetlistSpec(
            convention=cfg["convention"],
            subckt_name=top_cell,
            mos_models=cfg["devices"],
            fold_parallel=True,
            pdk_id=tech_id,
            ports=_source_subckt_ports(source_text, cell_def),
            emit_resistors=cfg.get("res") is not None,
        ),
    )


@dataclass(frozen=True)
class LvsReferenceEquivalenceCertificate:
    """Replayable proof that source CIR and generated reference mean one netlist."""

    source_cir_path: Path
    source_cir_sha256: str
    source_semantics_sha256: str
    cell_def_semantics_sha256: str
    generated_semantics_sha256: str
    generated_reference_sha256: str
    tech_id: str
    top_cell: str
    full_extraction: bool
    schema: str = "clay.lvs-reference-equivalence.v1"

    def replay(
        self,
        source_cir: Path,
        cell_def: Any,
        *,
        generated_reference_text: Optional[str] = None,
    ) -> bool:
        fresh = certify_lvs_reference_equivalence(
            source_cir,
            cell_def,
            tech_id=self.tech_id,
            top_cell=self.top_cell,
            full_extraction=self.full_extraction,
            generated_reference_text=generated_reference_text,
        )
        if fresh != self:
            raise ValueError(
                "LVS reference equivalence certificate is stale for current inputs"
            )
        return True


def certify_lvs_reference_equivalence(
    source_cir: Path,
    cell_def: Any,
    *,
    tech_id: str,
    top_cell: str,
    full_extraction: bool,
    generated_reference_text: Optional[str] = None,
) -> LvsReferenceEquivalenceCertificate:
    """Compile both source and cell IR; byte hashes alone never certify equality."""
    from clay.layout.route.truth_manifest import sha256_path

    if type(full_extraction) is not bool:
        raise TypeError("full_extraction must be exact bool")
    source_path = Path(source_cir).resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise LvsReferenceStop(state="SOURCE_CIR_EMPTY", refs=(source_path,))
    cell_reference = route_cell_reference_text(
        cell_def,
        tech_id=tech_id,
        top_cell=top_cell,
        full_extraction=full_extraction,
    )
    generated = (
        cell_reference
        if generated_reference_text is None
        else str(generated_reference_text)
    )
    if not generated.strip() or generated != cell_reference:
        raise ValueError(
            "generated LVS reference does not replay from the current cell_def"
        )
    source_reference = _source_cir_reference_text(
        source_path,
        cell_def,
        tech_id=tech_id,
        top_cell=top_cell,
        full_extraction=full_extraction,
    )
    source_semantics = _reference_semantics_sha256(source_reference)
    cell_semantics = _reference_semantics_sha256(cell_reference)
    generated_semantics = _reference_semantics_sha256(generated)
    if not (
        source_semantics == cell_semantics == generated_semantics
    ):
        raise ValueError(
            "source CIR, cell_def, and generated LVS reference are not "
            "semantically equivalent"
        )
    return LvsReferenceEquivalenceCertificate(
        source_cir_path=source_path,
        source_cir_sha256=sha256_path(source_path),
        source_semantics_sha256=source_semantics,
        cell_def_semantics_sha256=cell_semantics,
        generated_semantics_sha256=generated_semantics,
        generated_reference_sha256=_sha256_text(generated),
        tech_id=str(tech_id),
        top_cell=str(top_cell),
        full_extraction=full_extraction,
    )


def _mim_option_kwargs(
    *,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Dict[str, str]:
    """Build non-empty MIM option kwargs without perturbing legacy two-arg callers."""
    out: Dict[str, str] = {}
    if mim_option is not None and str(mim_option).strip():
        out["mim_option"] = str(mim_option).strip()
    if cap_option is not None and str(cap_option).strip():
        out["cap_option"] = str(cap_option).strip()
    if metal_level is not None and str(metal_level).strip():
        out["metal_level"] = str(metal_level).strip()
    if pex_backend is not None and str(pex_backend).strip():
        out["pex_backend"] = str(pex_backend).strip()
    return out


def _infer_gf180_mim_from_geometry(
    geometry: Optional[Mapping[str, Any]],
    passives: Optional[Sequence[Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Infer GF180 Option-B metal level when placed cap artwork already uses M4/V4 or M3/V3."""
    if not geometry:
        return None, None
    cap_names = {
        str(getattr(p, "name", "") or "")
        for p in (passives or ())
        if str(getattr(p, "kind", "") or "").lower().startswith("cap")
        or str(getattr(p, "kind", "") or "").lower() == "c"
    }
    for name, stamp in geometry.items():
        is_cap = (str(name) in cap_names) or (
            str(getattr(stamp, "kind", "") or "").lower().startswith("cap")
        )
        if not is_cap:
            continue
        stamp_mim = getattr(stamp, "mim_option", None)
        stamp_lvl = getattr(stamp, "metal_level", None)
        if stamp_mim is not None and str(stamp_mim).strip():
            return str(stamp_mim).strip(), (
                str(stamp_lvl).strip() if stamp_lvl is not None and str(stamp_lvl).strip() else None
            )
        layers: set[Tuple[int, int]] = set()
        polys_by_layer = getattr(stamp, "polys_by_layer", None)
        if isinstance(polys_by_layer, Mapping):
            layers.update((int(k[0]), int(k[1])) for k in polys_by_layer)
        else:
            for lp in getattr(stamp, "polygons", ()) or ():
                layers.add((int(lp.layer), int(lp.datatype)))
        # fusetop=(75,0); Option-A uses via2=(38,0), Option-B 5LM uses via4=(41,0), 4LM uses via3=(40,0).
        if (75, 0) in layers and (38, 0) not in layers:
            if (41, 0) in layers:
                return "B", "5LM"
            if (40, 0) in layers:
                return "B", "4LM"
    return None, None


def _mim_kwargs_from_sources(
    *sources: Any,
    geometry: Optional[Mapping[str, Any]] = None,
    passives: Optional[Sequence[Any]] = None,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Dict[str, str]:
    """Collect active MIM options by precedence: explicit > earlier source > cap stamp."""
    eff_mim = mim_option if mim_option is not None and str(mim_option).strip() else None
    eff_cap = cap_option if cap_option is not None and str(cap_option).strip() else None
    eff_lvl = metal_level if metal_level is not None and str(metal_level).strip() else None
    eff_pex = pex_backend if pex_backend is not None and str(pex_backend).strip() else None
    for src in sources:
        if src is None:
            continue
        if eff_mim is None:
            v = getattr(src, "mim_option", None)
            if v is not None and str(v).strip():
                eff_mim = str(v).strip()
        if eff_cap is None:
            v = getattr(src, "cap_option", None)
            if v is not None and str(v).strip():
                eff_cap = str(v).strip()
        if eff_lvl is None:
            v = getattr(src, "metal_level", None)
            if v is not None and str(v).strip():
                eff_lvl = str(v).strip()
        if eff_pex is None:
            v = getattr(src, "pex_backend", None)
            if v is not None and str(v).strip():
                eff_pex = str(v).strip()
    if eff_mim is None and geometry is not None:
        inferred_opt, inferred_lvl = _infer_gf180_mim_from_geometry(
            geometry, passives=passives
        )
        if inferred_opt is not None:
            eff_mim = inferred_opt
            if eff_lvl is None and inferred_lvl is not None:
                eff_lvl = inferred_lvl
    return _mim_option_kwargs(
        mim_option=eff_mim,
        cap_option=eff_cap,
        metal_level=eff_lvl,
        pex_backend=eff_pex,
    )


def _refine_tech_with_mim_sources(
    tech: Any,
    *sources: Any,
    geometry: Optional[Mapping[str, Any]] = None,
    passives: Optional[Sequence[Any]] = None,
) -> Any:
    """Bind missing MIM config on a ``LayoutTech`` without mutating default flyweights."""
    if not hasattr(tech, "with_mim_config"):
        return tech
    if str(getattr(tech, "pdk_id", "") or "").lower() != "gf180mcu":
        return tech
    kw = _mim_kwargs_from_sources(
        tech,
        *sources,
        geometry=geometry,
        passives=passives,
    )
    if not kw:
        return tech
    return tech.with_mim_config(**kw)


def _resolved_lvs_config(
    tech_id: str,
    *,
    full_extraction: bool,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one immutable-by-convention PDK config for helper consumers."""
    base = _PDK_LVS.get(tech_id)
    if base is None:
        raise ValueError(
            f"no synthesized LVS deck for tech_id={tech_id!r}; "
            f"supported: {sorted(_PDK_LVS)}"
        )
    cfg = dict(base)
    if full_extraction:
        overlay = base.get("full")
        if overlay:
            cfg.update(overlay)
        if tech_id == "gf180mcu":
            cfg.update({
                "layermap": dict(_GF180_FULL_LAYERMAP),
                "devices": dict(_GF180_FULL_DEVICES),
                "bulk_measured": True,
            })
    # The physical selector and both device-domain names must reach the deck
    # together.  Omitting the names silently disabled OD2 extraction even when
    # the generated GDS carried the marker and the source declared 3.3 V MOS.
    from clay.pdk import get_pdk, load_all
    load_all()
    adapter = get_pdk(tech_id)
    mim_kw = _mim_option_kwargs(
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    if tech_id.lower() == "gf180mcu" and mim_kw:
        from clay.layout.klayout_deck_gen import gf180_mim_lvs_spec
        from clay.pdk.gf180mcu import resolve_gf180_mim_config

        norm_opt, norm_cap, norm_lvl = resolve_gf180_mim_config(
            adapter,
            mim_option=mim_kw.get("mim_option"),
            cap_option=mim_kw.get("cap_option"),
            metal_level=mim_kw.get("metal_level"),
            pex_backend=mim_kw.get("pex_backend"),
        )
        if norm_opt != "A" or norm_cap != "1":
            cfg["mim"] = gf180_mim_lvs_spec(
                mim_option=norm_opt,
                cap_option=norm_cap,
                metal_level=norm_lvl,
                pex_backend=mim_kw.get("pex_backend"),
            )
    devices = dict(cfg["devices"])
    default_thick = bool(adapter.extra.get("thick_oxide_is_default", False))
    selector = "od2" in cfg["layermap"]
    for polarity in ("nmos", "pmos"):
        semantic = polarity + "3v"
        model = adapter.device_bindings.get(semantic)
        if model is None:
            continue
        if default_thick:
            devices[semantic] = devices[polarity]
        elif selector:
            devices[semantic] = str(model)
        else:
            raise ValueError(f"PDK {tech_id} thick MOS binding lacks a physical selector")
    cfg["devices"] = devices
    cfg["layermap"] = {**cfg["layermap"], "metal1_label": tuple(LayoutTech.resolve(tech_id).label_key(1))}
    return cfg


_DEEP_NWELL_FROM_LAYOUT_TECH = object()


def synthesized_lvs_deck_text(
    tech_id: str,
    *,
    full_extraction: bool = False,
    deep_nwell_override: Any = _DEEP_NWELL_FROM_LAYOUT_TECH,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> str:
    """Return the exact synthesized deck text that the LVS oracle will execute."""
    mim_kw = _mim_option_kwargs(
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    cfg = _resolved_lvs_config(tech_id, full_extraction=full_extraction, **mim_kw)
    deep_nwell = (
        LayoutTech.resolve(tech_id).deep_nwell()
        if deep_nwell_override is _DEEP_NWELL_FROM_LAYOUT_TECH
        else deep_nwell_override
    )
    deck_text = klayout_lvs_from_pdk(
        cfg["layermap"],
        cfg["devices"],
        pdk_name=tech_id,
        mim=cfg["mim"],
        res=cfg.get("res"),
        bjt=cfg.get("bjt"),
        top_routing_metal=cfg.get("top_routing_metal"),
        deep_nwell=deep_nwell,
    )
    if deck_text is None or "extract_devices" not in deck_text:
        raise AssertionError("LVS deck did not synthesize a device extraction")
    return deck_text


def route_cell_reference_text(
    cell_def: Any,
    *,
    tech_id: str,
    top_cell: Optional[str] = None,
    full_extraction: bool = False,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> str:
    """Return the exact generated reference text for a general route-cell LVS."""
    mim_kw = _mim_kwargs_from_sources(
        cell_def,
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    cfg = _resolved_lvs_config(tech_id, full_extraction=full_extraction, **mim_kw)
    tech = LayoutTech.resolve(tech_id, **mim_kw)
    _validate_pdk_reference_mos_classes(cell_def, tech)
    cell_name = str(
        getattr(cell_def, "local_name", None)
        or getattr(cell_def, "name", None)
        or "cell"
    )
    top = top_cell or f"{cell_name}_routed"
    ports = tuple(getattr(cell_def, "ports", ()) or ())
    return build_reference_netlist(
        _fold_passive_banks(_cell_def_reference_intent(cell_def)),
        RefNetlistSpec(
            convention=cfg["convention"],
            subckt_name=top,
            mos_models=cfg["devices"],
            fold_parallel=True,
            pdk_id=tech_id,
            ports=ports,
            emit_resistors=cfg.get("res") is not None,
        ),
    )


def _materialize(routed_or_gds: Any, intent: Any, tech_id: str):
    """Resolve the gate inputs to a concrete ``(routed, geometry, placement,
    pins_by_net, tech)`` bundle.

    ``routed_or_gds`` may be:
      * the dict returned by :func:`clay.layout.route.anchor_ota.route_amp_leakfree`
        WITH the live objects (key ``_bundle``), or
      * ``None`` / anything else -> the gate re-derives the leakage-free route from
        ``intent`` + ``tech_id`` deterministically (the SAME path the router uses).

    Re-deriving from the intent keeps the gate a pure function of (intent, tech_id):
    the checker does not depend on the checked carrying its internals along.
    """
    if isinstance(routed_or_gds, dict) and "_bundle" in routed_or_gds:
        b = routed_or_gds["_bundle"]
        return (b.routed, b.geometry, b.placement, b.pins_by_net, b.tech)

    # No bundle supplied -> build the CANONICAL production route and consume ITS bundle.
    # The gate MUST check the exact layout production builds; it does NOT re-derive its own
    # (which had drifted: no cap-recenter, a duplicated resistor-collapse). One source of
    # truth = route_amp_leakfree -> _bundle (an AsBuiltBundle). (Codex review: path drift.)
    from clay.layout.route.anchor_ota import route_amp_leakfree

    result = route_amp_leakfree(tech_id=tech_id)
    b = result["_bundle"]
    return (b.routed, b.geometry, b.placement, b.pins_by_net, b.tech)


def _lvs_compare_gds_vs_ref(
    gds: Path,
    ref: Path,
    *,
    deck_text: str,
    tech_id: str,
    work_dir: Path,
    kbin: Path,
    timeout_sec: float,
    top_cell: str,
    full_extraction: bool = False,
    oracle_gds_snapshot: Optional[OracleGdsSnapshot] = None,
    oracle_reference_snapshot: Optional[_OracleByteSnapshot] = None,
) -> LvsVerdict:
    """The single independent KLayout LVS oracle.

    Write the shared helper's synthesized PDK deck, run KLayout ``extract + compare``
    of ``gds`` against the reference netlist ``ref``, and return the honest
    :class:`LvsVerdict`. EVERY LVS
    gate routes through here -- the OTA-specialised ``route_amp_leakfree`` path and the
    GENERAL ``route_cell`` path alike -- so there is ONE checker, never a per-router
    second oracle (dev-discipline: the checker is independent of, and identical across,
    the things it checks). ``top_cell`` is the GDS top-cell name (which MUST equal the
    reference subckt name); the extracted / lvsdb artifacts are named after it.
    """
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    snapshot = (
        create_oracle_gds_snapshot(gds, work_dir, run_label="lvs")
        if oracle_gds_snapshot is None
        else oracle_gds_snapshot
    )
    if type(snapshot) is not OracleGdsSnapshot:
        raise TypeError("oracle_gds_snapshot must be an exact OracleGdsSnapshot")
    snapshot.verify_source_unchanged(gds)
    oracle_gds = snapshot.path
    run_dir = oracle_gds.parent

    reference_snapshot = (
        _snapshot_oracle_file(
            ref,
            run_dir,
            suffix="reference.cir",
            role="LVS reference",
            error_type=OracleReferenceSnapshotChanged,
        )
        if oracle_reference_snapshot is None
        else oracle_reference_snapshot
    )
    if type(reference_snapshot) is not _OracleByteSnapshot:
        raise TypeError(
            "oracle_reference_snapshot must be an exact private byte snapshot"
        )
    if reference_snapshot.path.resolve().parent != run_dir.resolve():
        raise OracleReferenceSnapshotChanged(
            "oracle LVS reference snapshot is outside the GDS run directory"
        )
    reference_snapshot.verify_unchanged()
    oracle_ref = reference_snapshot.path

    # 3) caller-supplied output of the shared synthesized-deck helper.  KLayout
    # reads only the private immutable copy; the conventional work-dir deck is
    # an atomic compatibility publication for existing evidence consumers.
    if deck_text is None or "extract_devices" not in deck_text:
        raise AssertionError("LVS deck did not synthesize a device extraction")
    deck_bytes = deck_text.encode("utf-8")
    deck_snapshot = _snapshot_generated_oracle_bytes(
        deck_bytes,
        run_dir,
        suffix=f"{tech_id}.lvs",
        role="LVS deck",
        error_type=OracleDeckSnapshotChanged,
    )
    deck = deck_snapshot.path
    _atomic_write_bytes(work_dir / f"{tech_id}_mim.lvs", deck_bytes)

    extracted = run_dir / f"{top_cell}_extracted.cir"
    lvsdb = run_dir / f"{top_cell}.lvsdb"
    canonical_extracted = work_dir / f"{top_cell}_extracted.cir"
    canonical_lvsdb = work_dir / f"{top_cell}.lvsdb"

    # A process failure must never inherit a previous run's extraction/xref.
    # Delete both private outputs and public compatibility publications before
    # spawning.  Only fresh, fully-formed private outputs are atomically
    # published after all immutable inputs survive certificate minting.
    for stale in (extracted, lvsdb, canonical_extracted, canonical_lvsdb):
        _unlink_oracle_output(stale)

    # 4) run KLayout LVS extract + COMPARE (schematic SET -> a real compare).
    proc = subprocess.run(
        [
            str(kbin), "-b", "-r", str(deck.resolve()),
            "-rd", f"input={oracle_gds}",
            "-rd", f"schematic={oracle_ref.resolve()}",
            "-rd", f"target_netlist={extracted.resolve()}",
            "-rd", f"report={lvsdb.resolve()}",
        ],
        capture_output=True, text=True, timeout=timeout_sec, check=False,
    )
    snapshot.verify_source_unchanged(gds)
    reference_snapshot.verify_unchanged()
    deck_snapshot.verify_unchanged()
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    from clay.layout.route.truth_manifest import sha256_path

    extracted_nonempty = bool(
        extracted.is_file() and extracted.stat().st_size > 0
    )
    lvsdb_nonempty = bool(lvsdb.is_file() and lvsdb.stat().st_size > 0)
    tool_says_match = (
        "Netlists match" in combined and "Netlists don't match" not in combined
    )
    tool_says_mismatch = "Netlists don't match" in combined
    comparison_completed = bool(
        proc.returncode == 0
        and extracted_nonempty
        and lvsdb_nonempty
        and (tool_says_match or tool_says_mismatch)
    )
    verdict = LvsVerdict(
        matched=False,
        verbatim=_verbatim_verdict(combined),
        ran=comparison_completed,
        returncode=proc.returncode,
        stdout_tail="\n".join(combined.strip().splitlines()[-25:]),
        gds_path=str(Path(gds).resolve()), gds_sha256=snapshot.sha256,
        oracle_gds_path=str(oracle_gds),
        oracle_gds_sha256=snapshot.sha256,
        ref_path=str(oracle_ref), ref_sha256=reference_snapshot.sha256,
        deck_path=str(deck), deck_sha256=sha256_path(deck),
        lvsdb_path=str(lvsdb) if lvsdb.exists() else None,
        lvsdb_sha256=(sha256_path(lvsdb)
                      if lvsdb.exists() and lvsdb.stat().st_size > 0 else None),
        extracted_netlist_path=(str(extracted) if extracted_nonempty else None),
        extracted_netlist_sha256=(
            sha256_path(extracted) if extracted_nonempty else None
        ),
        full_extraction=bool(full_extraction),
    )

    # the deck must at least PARSE (no Ruby error) -- a parse failure is not a no-match.
    low = combined.lower()
    for tok in ("syntaxerror", "syntax error", "nameerror", "uninitialized constant"):
        if tok in low:
            verdict.verbatim = (verdict.verbatim
                                + f"\n[deck parse error: {tok}]").strip()
            snapshot.retain("LVS parse-failure evidence")
            return verdict

    if extracted_nonempty:
        verdict.extracted_devices = _extracted_device_cards(
            extracted.read_text(encoding="utf-8")
        )

    # the AUTHORITATIVE match decision: BOTH the tool's own verdict string AND the
    # structured cross-reference must agree on a clean match. Never claim a match on
    # the strength of only one signal.
    xref_clean = False
    if lvsdb.exists():
        try:
            xref_clean, verdict.unmatched = _parse_lvsdb_unmatched(lvsdb)
        except Exception as exc:  # pragma: no cover - defensive
            verdict.verbatim = (verdict.verbatim + f"\n[lvsdb parse error: {exc}]").strip()
            xref_clean = False

    verdict.matched = bool(
        proc.returncode == 0 and tool_says_match and xref_clean and verdict.ran
    )
    if (
        comparison_completed
    ):
        verdict.oracle_certificate = _mint_lvs_oracle_certificate(
            gds=oracle_gds,
            ref=oracle_ref,
            deck=deck,
            lvsdb=lvsdb,
            extracted_netlist=extracted,
            full_extraction=bool(full_extraction),
            returncode=proc.returncode,
            ran=verdict.ran,
            combined_output=combined,
            xref_clean=xref_clean,
            unmatched=verdict.unmatched,
        )
    snapshot.verify_source_unchanged(gds)
    reference_snapshot.verify_unchanged()
    deck_snapshot.verify_unchanged()
    if extracted_nonempty:
        _atomic_write_bytes(canonical_extracted, extracted.read_bytes())
        verdict.published_extracted_netlist_path = str(canonical_extracted)
    if lvsdb.exists() and lvsdb.stat().st_size > 0:
        _atomic_write_bytes(canonical_lvsdb, lvsdb.read_bytes())
        verdict.published_lvsdb_path = str(canonical_lvsdb)
    snapshot.retain("LVS oracle evidence")
    return verdict


def lvs_gate(
    routed_or_gds: Any,
    intent: Any,
    *,
    tech_id: str,
    work_dir: Optional[Path] = None,
    klayout_bin: Optional[Path] = None,
    timeout_sec: float = 300.0,
    full_extraction: bool = False,
) -> LvsVerdict:
    """Run the independent KLayout LVS gate and return an :class:`LvsVerdict`.

    Parameters
    ----------
    routed_or_gds:
        Either the ``route_amp_leakfree`` result dict carrying a live ``_bundle``,
        or anything else (the gate re-derives the leakage-free route from
        ``intent`` + ``tech_id``).
    intent:
        The ``ExtractedIntent`` (``build_ota_intent()``) -- the schematic source for
        the reference netlist.
    tech_id:
        The target PDK (only ``gf180mcu`` has a synthesized LVS deck today).
    work_dir:
        Where to write the GDS / reference / deck / lvsdb artifacts. A temp dir is
        used when omitted.
    full_extraction:
        Select the PDK's ``full`` device-extraction overlay where the default
        config is still bring-up-bounded (open_pdk: CTM5/M5 MIM + poly-res
        recognition, unbounded routing ladder). A PDK with no overlay (gf180mcu
        -- already full by default) is unaffected. The overlay is the SAME deck
        synthesis path with stronger recognition DATA -- never a second checker.

    Returns
    -------
    LvsVerdict
        ``matched`` is True ONLY when KLayout printed ``Netlists match`` AND the
        lvsdb cross-reference is fully paired. Otherwise the honest unmatched list is
        returned -- never a faked pass.
    """
    cfg = _resolved_lvs_config(tech_id, full_extraction=full_extraction)
    kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
    if not kbin.exists():
        raise FileNotFoundError(f"KLayout binary not installed at {kbin}")

    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix="lvs_gate_"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    routed, geometry, placement, pins_by_net, tech = _materialize(
        routed_or_gds, intent, tech_id
    )

    # 1) full labelled GDS.
    gds = emit_labelled_gds(
        routed, intent, geometry, placement, pins_by_net,
        work_dir / f"{_TOP_CELL}_{tech_id}.gds", tech=tech,
    )

    # 2) fold-reconciled reference netlist (synth convention -> matches the deck).
    # _fold_passive_banks: the PASSIVE axis of the fold contract (an m-unit cap
    # bank folds to one C*m card, mirroring combine_devices on the drawn bank);
    # identity (same object) when every passive has M=1.
    ref_text = build_reference_netlist(
        _fold_passive_banks(intent),
        RefNetlistSpec(convention=cfg["convention"], subckt_name=_TOP_CELL,
                       mos_models=cfg["devices"],
                       fold_parallel=True, pdk_id=tech_id,
                       # the synth deck now EXTRACTS poly resistors (ResLvsSpec) -> emit
                       # real R cards on both sides (no resistor-collapse-to-wire), so a
                       # resistor pairs as a device and CR1 keeps its real net (net25).
                       emit_resistors=cfg.get("res") is not None),
    )
    snapshot = create_oracle_gds_snapshot(gds, work_dir, run_label="lvs")
    ref_snapshot = _snapshot_generated_oracle_bytes(
        ref_text.encode("utf-8"),
        snapshot.path.parent,
        suffix="reference.cir",
        role="LVS reference",
        error_type=OracleReferenceSnapshotChanged,
    )
    ref = ref_snapshot.path

    # 3+4) the single independent KLayout LVS oracle (deck synth + extract + compare).
    return _lvs_compare_gds_vs_ref(
        gds, ref,
        deck_text=synthesized_lvs_deck_text(
            tech_id, full_extraction=full_extraction
        ),
        tech_id=tech_id, work_dir=work_dir,
        kbin=kbin, timeout_sec=timeout_sec, top_cell=_TOP_CELL,
        full_extraction=full_extraction,
        oracle_gds_snapshot=snapshot,
        oracle_reference_snapshot=ref_snapshot,
    )


# The p-substrate assumption the synthesized LVS deck bakes in
# (klayout_deck_gen.py device extraction): the NMOS bulk is tied to ONE global
# substrate net (``"W" => sub`` + ``connect_global(sub, substrate)``), while the
# PMOS bulk is kept LOCAL to each nwell (``"W" => l_nwell``). So the ONE MOS kind
# whose body the deck globally FORCES to the substrate rail -- regardless of the
# device's own tap geometry -- is the NMOS; a PMOS body stays per-well. Keyed on
# device KIND, never a device/net name. If an n-substrate PDK is ever added, this
# kind and the deck's ``"W" =>`` mapping must flip together (single shared fact).
_GLOBAL_SUBSTRATE_KIND = "nmos"


def _normalize_substrate_bulk_bodies(devices: List[Any]) -> List[Any]:
    """Re-bind each substrate-tied MOS's BODY net to the design's substrate rail --
    the physical netlist the deck's ``connect_global`` forces on the extracted
    layout -- so the reference pairs the extraction on the bulk terminal.

    The synthesized deck ties EVERY NMOS bulk to the ONE global substrate net, so
    the EXTRACTED layout ALWAYS has an NMOS body = the substrate = the ground rail
    (e.g. gf180/external_pdk/external_pdk p-substrate contacted by the ground taps). A schematic-
    capture ``source == body == <private net>`` device (the OTA's MM15, a source-
    degenerated NMOS with body=net08) therefore mismatches: extracted bulk=gnd vs
    reference bulk=net08. This normalizes the reference to the physical truth.

    GENERAL + leakage-free: keyed on device KIND (the substrate-tied NMOS) + the
    SHARED rail vocabulary (``device_cell._net_is_substrate_rail`` -- the SAME
    classification the router's pin extractor and the device cell's substrate tap
    use), never a device/net name. The canonical substrate rail is DERIVED from the
    design: the UNIQUE recognized rail among the substrate-tied kind's body nets (a
    self-bulk NMOS body IS the ground rail). Only NON-rail bodies are re-bound; the
    SOURCE / drain / gate diffusions keep their own nets (net08 stays a distinct
    source, RR5 stays a real resistor). A design whose bodies are already all rails
    is byte-unchanged.

    Honest no-op when the rail is not uniquely determined (0 rail bodies -> nothing
    to normalize to; >1 distinct rail bodies -> the deck would merge them, a deeper
    multi-substrate case not modelled here): the bodies are left as-declared so any
    residual mismatch surfaces in the LVS verdict rather than a silent re-bind.
    """
    from clay.layout.device_cell import _net_is_substrate_rail

    def _kind(d: Any) -> str:
        return str(getattr(d, "kind", "") or "").lower()

    def _body(d: Any) -> Optional[str]:
        return (getattr(d, "nets", None) or {}).get("body")

    rails = {
        _body(d)
        for d in devices
        if _kind(d) == _GLOBAL_SUBSTRATE_KIND
        and _body(d) and _net_is_substrate_rail(_body(d))
    }
    if len(rails) != 1:
        return devices
    canon = next(iter(rails))
    for d in devices:
        if _kind(d) != _GLOBAL_SUBSTRATE_KIND:
            continue
        body = _body(d)
        if body and not _net_is_substrate_rail(body):
            d.nets["body"] = canon  # body terminal ONLY; source/drain/gate untouched
    return devices


def _cell_def_reference_intent(cell_def: Any) -> Any:
    """A reference intent built from the CANONICAL ``cell_def.devices`` -- the SAME
    source ``place_generic`` + ``route_cell`` consume to build the layout.

    ONE netlist source feeds BOTH sides of the LVS compare: the routed layout and this
    reference both derive from ``cell_def.devices`` (no second ``.cir`` re-parse that
    could silently diverge from the routed devices). Shaped into the ``.devices`` /
    ``.passives`` form ``build_reference_netlist`` reads.

    A bulk MOS's BODY net is normalized to the substrate rail the deck physically
    forces (:func:`_normalize_substrate_bulk_bodies`) -- the reference must declare
    the extracted bulk tie, not the schematic-capture source==body token, or a
    source-degenerated NMOS is an unmatched bulk-only device.

    Unit contract: the IR carries MOS W/L in SI metres (``sizing['W']``); ``_mos_card``
    formats microns + a ``u`` suffix -> scale by 1e6 or every device is 1e6x too small
    and nothing pairs. (Measuring the wrong-unit object = a false verdict.)
    """
    from types import SimpleNamespace
    from clay.layout.annotate_to_solver import kind_polarity_oxide

    _UM = 1e6
    devices: List[Any] = []
    passives: List[Any] = []
    bjts: List[Any] = []
    for d in getattr(cell_def, "devices", []) or []:
        dclass = str(getattr(d, "device_class", "") or "").strip().lower()
        sizing = getattr(d, "sizing", None) or {}
        pins = getattr(d, "pins", None) or {}
        name = str(getattr(d, "instance_name", None) or getattr(d, "name", ""))
        m = int(round(float(sizing.get("M", 1) or 1)))
        polarity = _mos_polarity(dclass)
        if polarity is not None:
            devices.append(SimpleNamespace(
                name=name[1:] if name[:1].upper() == "M" else name,
                kind=polarity,
                oxide=kind_polarity_oxide(dclass)[1],
                W=float(sizing.get("W", 0.0)) * _UM,
                L=float(sizing.get("L", 0.0)) * _UM,
                M=m,
                nets={"drain": pins.get("d"), "gate": pins.get("g"),
                      "source": pins.get("s"), "body": pins.get("b")},
            ))
        elif dclass in ("res", "cap"):
            lead = name[:1].upper()
            value = sizing.get("value",
                               sizing.get("R" if dclass == "res" else "C", 0.0))
            passives.append(SimpleNamespace(
                name=name[1:] if lead in ("R", "C") else name,
                kind=dclass,
                value=float(value),
                M=m,
                nets={"p": pins.get("p"), "n": pins.get("n")},
            ))
        elif dclass in ("pnp", "npn"):
            bjts.append(SimpleNamespace(
                name=name[1:] if name[:1].upper() == "Q" else name,
                kind=dclass,
                area=float(sizing.get("area", 1.0) or 1.0),
                M=float(sizing.get("M", 1.0) or 1.0),
                nets={"c": pins.get("c"), "b": pins.get("b"), "e": pins.get("e")},
            ))
    if not devices and not passives and not bjts:
        raise LvsReferenceStop(state="EMPTY_REFERENCE")
    _normalize_substrate_bulk_bodies(devices)
    return SimpleNamespace(devices=devices, passives=passives, bjts=bjts)


def lvs_gate_route_cell(
    res: Any,
    cell_def: Any,
    *,
    tech_id: str,
    top_cell: Optional[str] = None,
    work_dir: Optional[Path] = None,
    gds_path: Optional[Path] = None,
    klayout_bin: Optional[Path] = None,
    timeout_sec: float = 300.0,
    full_extraction: bool = False,
    run_via_provenance: bool = False,
    oracle_gds_snapshot: Optional[OracleGdsSnapshot] = None,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    metal_level: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> LvsVerdict:
    """Independent KLayout LVS gate for a cell routed by the GENERAL ``route_cell``.

    Where :func:`lvs_gate` is hardwired to the OTA (``build_ota_intent`` reference +
    the ``route_amp_leakfree`` materialize), this gate checks ANY ``route_cell`` result
    against its OWN schematic through the SAME :func:`_lvs_compare_gds_vs_ref` oracle. It
    is the decisive bar for folding a specialised / per-circuit router into the single
    general spine: the general router may replace a signed-off one only when its emitted
    layout still LVS-matches the schematic (DRC-clean is necessary but not sufficient --
    DRC merges cross-net overlaps, LVS is the connectivity + device oracle).

    EVERYTHING is derived from ``cell_def`` -- no per-circuit knob: the reference netlist
    from ``cell_def.devices`` (the canonical source the placer/router consume), the
    external ports from ``cell_def.ports``, the emit intent from
    ``layout_intent_from_cell_def``, and the top-cell name from the cell name. ``res`` is
    the ``RouteCellResult`` (routed / geometry / placement_um / pins_by_net / tech).
    ``top_cell`` overrides the derived ``<cell>_routed`` name only if a caller must.
    When ``gds_path`` is supplied, that already-emitted file is compared directly;
    the gate does not emit a second GDS.

    full_extraction:
        Select the PDK's ``full`` device-extraction overlay (mirrors
        :func:`lvs_gate`'s parameter of the same name -- SAME overlay dict, SAME
        selection logic, so a PDK's ``full`` recipe is never re-specified per
        entry point). A PDK with no ``full`` overlay (gf180mcu -- already full by
        default) is unaffected; behaviour for every existing gf180 call site is
        BY CONSTRUCTION unchanged (no ``full`` key there).
    run_via_provenance:
        Opt in to the independent emitted-GDS via-provenance checker.  It consumes
        the live result's canonical tagged manifest and the exact same GDS path as
        the real LVS oracle.  Disabled by default for backward compatibility.
    """
    mim_kw = _mim_kwargs_from_sources(
        getattr(res, "tech", None),
        cell_def,
        geometry=getattr(res, "geometry", None),
        passives=getattr(cell_def, "passives", None),
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    _resolved_lvs_config(tech_id, full_extraction=full_extraction, **mim_kw)
    kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
    if not kbin.exists():
        raise FileNotFoundError(f"KLayout binary not installed at {kbin}")
    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix="lvs_gate_rc_"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Everything DERIVED from cell_def -- the single design source, no hand-typed knob.
    cell_name = str(getattr(cell_def, "local_name", None)
                    or getattr(cell_def, "name", None) or "cell")
    top = top_cell or f"{cell_name}_routed"

    # 1) full labelled GDS from the route_cell as-built (top cell == ``top``).
    # devices= (C3): layout_intent is a LayoutIntent (no .devices), so the well
    # merge's bulk attribution MUST come from the canonical cell_def device list
    # or every PMOS well stays an unattributable singleton (NW.2b never merged,
    # and a 4-terminal deck reads the split wells as foreign/floating bulk).
    if gds_path is None:
        from clay.layout.intent import layout_intent_from_cell_def

        layout_intent = layout_intent_from_cell_def(cell_def)
        emit_tech = (
            res.tech.with_mim_config(**mim_kw)
            if mim_kw and hasattr(res.tech, "with_mim_config")
            else res.tech
        )
        gds = emit_labelled_gds(
            res.routed, layout_intent, res.geometry, res.placement_um, res.pins_by_net,
            work_dir / f"{top}_{tech_id}.gds", tech=emit_tech, top_cell=top,
            devices=well_merge_devices_from_cell_def(cell_def),
            source_cell=cell_def,
            row_program=res.row_program,
            row_materialization=res.row_materialization,
            row_final_certificate=res.row_final_placement_certificate,
            selected_route_engine=getattr(res,'selected_route_engine',None),
        )
    else:
        gds = Path(gds_path)

    # 2) reference netlist from cell_def.devices -- SAME fold + convention + resistor
    #    contract the OTA path uses, sourced from the design's canonical device list.
    if full_extraction:
        # Full geometry exposes declared straight resistor segments and private
        # wells. Reuse the established source-only lowerer used by artifact
        # checks, rather than comparing them to lumped nominal cards.
        from clay.layout.source_lvs_reference import reference_text

        ref_text, reference_proof = reference_text(cell_def, pdk=tech_id, top=top)
        (work_dir / "source-reference.json").write_text(
            json.dumps(reference_proof, indent=2), encoding="utf-8"
        )
    else:
        ref_text = route_cell_reference_text(
            cell_def, tech_id=tech_id, top_cell=top, full_extraction=False, **mim_kw,
        )
    snapshot = (
        create_oracle_gds_snapshot(gds, work_dir, run_label="lvs")
        if oracle_gds_snapshot is None
        else oracle_gds_snapshot
    )
    if type(snapshot) is not OracleGdsSnapshot:
        raise TypeError("oracle_gds_snapshot must be an exact OracleGdsSnapshot")
    snapshot.verify_source_unchanged(gds)
    ref_snapshot = _snapshot_generated_oracle_bytes(
        ref_text.encode("utf-8"),
        snapshot.path.parent,
        suffix="reference.cir",
        role="LVS reference",
        error_type=OracleReferenceSnapshotChanged,
    )
    ref = ref_snapshot.path

    # 3+4) the single independent KLayout LVS oracle.
    verdict = _lvs_compare_gds_vs_ref(
        gds, ref,
        deck_text=synthesized_lvs_deck_text(
            tech_id, full_extraction=full_extraction, **mim_kw
        ),
        tech_id=tech_id, work_dir=work_dir,
        kbin=kbin, timeout_sec=timeout_sec, top_cell=top,
        full_extraction=full_extraction,
        oracle_gds_snapshot=snapshot,
        oracle_reference_snapshot=ref_snapshot,
    )
    if run_via_provenance:
        attach_via_provenance(verdict, res, top_cell=top)
    return verdict


__all__ = [
    "DeckIsolationWriteRefused",
    "OracleGdsSnapshot",
    "OracleInputSnapshotChanged",
    "OracleGdsSnapshotChanged",
    "OracleReferenceSnapshotChanged",
    "OracleDeckSnapshotChanged",
    "create_oracle_gds_snapshot",
    "create_oracle_deck_snapshot",
    "lvs_gate",
    "lvs_gate_route_cell",
    "attach_via_provenance",
    "route_cell_reference_text",
    "synthesized_lvs_deck_text",
    "LvsVerdict",
    "LvsReferenceStop",
    "LvsReferenceEquivalenceCertificate",
    "certify_lvs_reference_equivalence",
    "emit_labelled_gds",
    "well_merge_devices_from_cell_def",
]
