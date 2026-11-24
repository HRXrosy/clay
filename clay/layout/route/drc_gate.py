"""Real-deck DRC oracle for route sign-off (the geometric-LEGALITY half of signoff).

The independent KLayout LVS oracle (:mod:`clay.layout.route.lvs_gate`) proves
CONNECTIVITY -- ``Netlists match``. It does NOT prove geometric LEGALITY: a layout can
match LVS while the foundry DRC deck fires (spacing / enclosure / min-width). The
sign-off contract is **LVS-match AND DRC-clean**; this module supplies the missing DRC
half, read off the SAME real foundry deck the signoff harness drives
(:func:`tools.klayout_drc_signoff_v139.run.run_gf180_drc` on
``LayoutTech.resolve(tech_id).klayout_decks()['drc']``) -- never a native proxy,
``separation_check``, or union-find. Those are exactly the WRONG oracles the via-pad
effort exposed (a cross-net overlap the deck merges reads 0 in ``separation_check``;
DRC-clean is the only authority for the M*.2a family).

It reads the ONE canonical as-built bundle (``route_amp_leakfree``'s ``_bundle`` /
``route_cell`` result) through the SAME ``_materialize`` + ``emit_labelled_gds`` path the
LVS gate uses, so DRC and LVS see byte-identical geometry -- there is no second layout
build. The top cell is taken from the EMITTED GDS (``ly.top_cells()[0].name``), so the
deck checks exactly what was written.

Honest incompleteness: when no DRC deck is registered for ``tech_id`` (only the gf180
open PDK ships the KLayout runset the v139 harness drives) or KLayout is not installed,
the verdict is ``ran=False`` (UNMEASURABLE upstream) -- never a vacuous clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["DrcVerdict", "ExactDrcInputs", "drc_gate", "drc_gate_exact_artifact"]


@dataclass(frozen=True, init=False)
class DrcOracleCertificate:
    """Oracle-private immutable replay of one KLayout DRC process/report."""

    gds_sha256: str
    deck_sha256: str
    report_sha256: str
    returncode: int
    counts_by_rule: tuple[tuple[str, int], ...]
    total: int
    clean: bool

    def _validate(self) -> None:
        for name in ("gds_sha256", "deck_sha256", "report_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"DRC oracle certificate has malformed {name}")
        if type(self.returncode) is not int or self.returncode != 0:
            raise ValueError("DRC oracle certificate requires process returncode 0")
        if self.total != sum(count for _rule, count in self.counts_by_rule):
            raise ValueError("DRC oracle certificate report total does not replay")
        if self.clean != (self.total == 0):
            raise ValueError("DRC oracle certificate clean state does not replay")


def _mint_drc_oracle_certificate(
    report: Any,
    *,
    gds_path: Path,
    deck_path: Path,
    report_path: Path,
) -> DrcOracleCertificate:
    """Minted only after the real harness returns; reparses its report bytes."""
    from clay.layout.route.truth_manifest import sha256_path
    from tools.klayout_drc_signoff_v139.run import parse_lyrdb

    gds = Path(gds_path).resolve()
    deck = Path(deck_path).resolve()
    report_file = Path(report_path).resolve()
    if not bool(getattr(report, "ran_real_deck", False)):
        raise ValueError("DRC oracle did not run the real deck")
    if getattr(report, "returncode", None) != 0:
        raise ValueError("DRC oracle process did not exit with returncode 0")
    if Path(getattr(report, "gds_path", "")).resolve() != gds:
        raise ValueError("DRC oracle report names a different GDS")
    if not report_file.is_file() or report_file.stat().st_size <= 0:
        raise ValueError("DRC oracle produced no non-empty report database")
    parsed = parse_lyrdb(report_file)
    counts = tuple(sorted(parsed.counts_by_rule().items()))
    certificate = object.__new__(DrcOracleCertificate)
    for name, value in (
        ("gds_sha256", sha256_path(gds)),
        ("deck_sha256", sha256_path(deck)),
        ("report_sha256", sha256_path(report_file)),
        ("returncode", 0),
        ("counts_by_rule", counts),
        ("total", sum(count for _rule, count in counts)),
        ("clean", not counts),
    ):
        object.__setattr__(certificate, name, value)
    certificate._validate()
    return certificate


@dataclass(frozen=True)
class ExactDrcInputs:
    """Retained oracle input identities, not a signoff certificate.

    The executable deck stays at its PDK-resolved location so the native
    harness retains relative dependencies and its Windows adaptation. The
    separate deck snapshot records bytes; it is not an isolated executable.
    """

    original_gds_path: str
    original_gds_sha256: str
    oracle_gds_path: str
    oracle_gds_sha256: str
    deck_path: str
    deck_sha256: str
    deck_snapshot_path: str
    deck_snapshot_sha256: str
    source_sha256: str
    source_netlist_sha256: str
    pdk_sha256: str
    route_manifest_sha256: str
    runtime_options: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DrcVerdict:
    """The outcome of the real foundry DRC deck on the canonical as-built layout.

    ``clean`` is True ONLY when the deck actually ran AND reported zero violations;
    a deck that could not run (no registered deck / KLayout absent / emit failure) is
    ``ran=False, clean=False`` -- never a silent pass.
    """

    ran: bool
    clean: bool
    total: int
    counts_by_rule: Dict[str, int] = field(default_factory=dict)
    deck_path: Optional[str] = None
    deck_sha256: Optional[str] = None
    gds_path: Optional[str] = None
    gds_sha256: Optional[str] = None
    report_path: Optional[str] = None
    report_sha256: Optional[str] = None
    returncode: Optional[int] = None
    oracle_certificate: Optional[DrcOracleCertificate] = None
    topcell: Optional[str] = None
    reason: str = ""
    exact_inputs: Optional[ExactDrcInputs] = None

    def summary(self) -> str:
        if not self.ran:
            return f"DRC: NOT RUN ({self.reason or 'unmeasurable'})"
        if self.clean:
            if self.exact_inputs is not None:
                return "DRC: CLEAN (0 violations on the selected PDK deck)"
            return "DRC: CLEAN (0 violations on the real foundry deck)"
        head = ", ".join(f"{r}x{n}" for r, n in sorted(self.counts_by_rule.items()))
        return f"DRC: {self.total} violations [{head}]"


def drc_gate(
    routed_or_bundle: Any,
    intent: Any,
    *,
    tech_id: str,
    klayout_bin: Optional[Any] = None,
    timeout_sec: float = 580.0,
    work_dir: Optional[Any] = None,
) -> DrcVerdict:
    """Run the real foundry DRC deck on the ONE canonical as-built layout.

    ``routed_or_bundle`` is the production result the LVS gate already consumes (the
    dict from :func:`route_amp_leakfree` carrying ``_bundle``, or any
    ``_materialize``-able route result). Both oracles therefore read the SAME geometry.

    Returns a :class:`DrcVerdict`. Never raises on an infrastructure gap (no deck for
    ``tech_id``, KLayout absent, emit/DRC failure): the failure becomes ``ran=False``
    with a plain-language ``reason`` (the honest gray), not an exception.
    """
    import tempfile

    import klayout.db as kdb

    from clay.layout.klayout_deck_gen import DEFAULT_KLAYOUT_BIN
    from clay.layout.route.lvs_gate import _materialize, emit_labelled_gds
    from clay.layout.tech import LayoutTech
    from tools.klayout_drc_signoff_v139.run import Gf180DrcError, run_gf180_drc

    # 1) the registered foundry DRC deck for this PDK (gf180 ships the runset the
    #    v139 harness drives; a PDK with none is honestly UNMEASURABLE).
    try:
        deck = LayoutTech.resolve(tech_id).klayout_decks().get("drc")
    except Exception as exc:  # pragma: no cover - unknown tech
        return DrcVerdict(ran=False, clean=False, total=0,
                          reason=f"no LayoutTech for tech_id={tech_id!r}: {exc}")
    if deck is None or not Path(deck).exists():
        return DrcVerdict(ran=False, clean=False, total=0,
                          reason=f"no DRC deck registered for tech_id={tech_id!r}")

    kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
    if not kbin.exists():
        return DrcVerdict(ran=False, clean=False, total=0,
                          deck_path=str(deck),
                          reason=f"KLayout binary not installed at {kbin}")

    # 2) emit the canonical labelled GDS (the SAME path lvs_gate emits) and read the
    #    top cell from the EMITTED layout, so the deck checks exactly what was written.
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="drc_gate_"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        routed, geometry, placement, pins_by_net, tech = _materialize(
            routed_or_bundle, intent, tech_id
        )
        gds = work_dir / f"drc_gate_{tech_id}.gds"
        emit_labelled_gds(routed, intent, geometry, placement, pins_by_net, gds,
                          tech=tech)
        ly = kdb.Layout()
        ly.read(str(gds))
        tops = ly.top_cells()
        if not tops:
            return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                              gds_path=str(gds), reason="emitted GDS has no top cell")
        topcell = tops[0].name
    except Exception as exc:
        return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                          reason=f"could not emit canonical GDS: {exc}")

    # 3) the REAL deck -- the sole M*.2a authority.
    try:
        report_path = work_dir / f"drc_gate_{tech_id}.lyrdb"
        rep = run_gf180_drc(
            str(gds), topcell=topcell, deck=Path(deck),
            klayout_bin=kbin, report_path=report_path,
            timeout_sec=timeout_sec,
        )
    except Gf180DrcError as exc:
        return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                          gds_path=str(gds), topcell=topcell,
                          reason=f"DRC deck could not run: {exc}")

    try:
        certificate = _mint_drc_oracle_certificate(
            rep,
            gds_path=Path(gds),
            deck_path=Path(deck),
            report_path=report_path,
        )
    except Exception as exc:
        return DrcVerdict(
            ran=False,
            clean=False,
            total=0,
            deck_path=str(deck),
            gds_path=str(gds),
            report_path=str(report_path),
            returncode=getattr(rep, "returncode", None),
            reason=f"DRC evidence could not be certified: {exc}",
        )
    counts = dict(certificate.counts_by_rule)
    total = certificate.total
    from clay.layout.route.truth_manifest import sha256_path

    return DrcVerdict(
        ran=bool(rep.ran_real_deck),
        clean=bool(rep.ran_real_deck and rep.is_clean()),
        total=total,
        counts_by_rule=counts,
        deck_path=str(deck),
        deck_sha256=sha256_path(Path(deck)),
        gds_path=str(gds),
        gds_sha256=sha256_path(Path(gds)),
        report_path=str(report_path),
        report_sha256=certificate.report_sha256,
        returncode=certificate.returncode,
        oracle_certificate=certificate,
        topcell=topcell,
        reason="" if rep.ran_real_deck else "deck did not execute",
    )


def _exact_drc_pdk_state(tech_id: str):
    """Check fresh installed bytes against the already loaded native adapter."""
    from clay.pdk import exact_source_identity, get_pdk
    from clay.pdk._source_snapshot import capture_pdk_source_snapshot

    adapter = get_pdk(tech_id)
    loaded = adapter.source_snapshot()
    if loaded is not None:
        raw_root = adapter.root()
        if raw_root is None:
            raise ValueError("DRC PDK loaded source snapshot has no selected root")
        selected_root = Path(raw_root).resolve()
        if (Path(loaded.root).resolve() != selected_root
                or loaded.pdk_id != adapter.pdk_id):
            raise ValueError("DRC PDK loaded source snapshot has a different root or PDK")
        fresh = capture_pdk_source_snapshot(adapter, selected_root)
        after_root = adapter.root()
        if (fresh != loaded or adapter.source_snapshot() != loaded
                or after_root is None or Path(after_root).resolve() != selected_root):
            raise ValueError("DRC PDK source bytes differ from the materialized adapter")
    # A native adapter without an installed snapshot keeps its explicit None.
    # This is not promoted to installed foundry-source calibration evidence.
    identity = exact_source_identity(tech_id)
    if (identity["pdk_id"] != str(adapter.pdk_id)
            or identity["source_snapshot_sha256"] != (
                None if loaded is None else loaded.snapshot_sha256)):
        raise ValueError("DRC PDK exact identity changed during validation")
    return adapter, identity


def drc_gate_exact_artifact(
    *,
    exact_artifact: Any,
    tech_id: str,
    work_dir: Any,
    klayout_bin: Optional[Any] = None,
    timeout_sec: float = 580.0,
) -> DrcVerdict:
    """Run the selected native PDK deck on an existing exact emitted artifact.

    No materialization or GDS emission occurs here. A native private snapshot
    supplies byte-identical checker input; the original and snapshot are both
    verified before certification. The private DRC mint reparses the actual
    report, so a caller-provided count or clean flag cannot certify a result.

    A clean selected deck is not a claim that a partial generated PDK deck is
    complete foundry signoff. Missing inputs, input drift and checker failures
    return an explicitly unproven verdict and retain any created evidence.
    """
    from types import SimpleNamespace

    from clay.flow.step import _mim_drc_rd
    from clay.layout.klayout_deck_gen import DEFAULT_KLAYOUT_BIN
    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact
    from clay.layout.route.lvs_gate import (
        create_oracle_deck_snapshot,
        create_oracle_gds_snapshot,
    )
    from clay.layout.route.truth_manifest import sha256_path
    from tools.klayout_drc_signoff_v139.run import run_gf180_drc

    # None until independently validated; failures must never carry a certificate.
    gds = deck = report_path = snapshot = deck_snapshot = None
    topcell = inputs = report = None
    try:
        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact DRC requires native ExactEmittedArtifact")
        exact_artifact.validate()
        truth = exact_artifact.truth_manifest
        truth.validate_source_provenance()
        adapter, pdk_identity = _exact_drc_pdk_state(tech_id)
        pdk_sha256 = truth.bind_pdk_identity(pdk_identity)
        gds = Path(exact_artifact.exact_gds_path).resolve()
        topcell = exact_artifact.scope.top_cell
        selected = adapter.klayout_decks().get("drc")
        if selected is None or not Path(selected).is_file():
            raise ValueError(f"no DRC deck registered for tech_id={tech_id!r}")
        deck = Path(selected).resolve()
        deck_sha256 = sha256_path(deck)
        if truth.drc_deck_path is not None or truth.drc_deck_sha256 is not None:
            if (truth.drc_deck_path is None or truth.drc_deck_sha256 is None
                    or Path(truth.drc_deck_path).resolve() != deck
                    or truth.drc_deck_sha256 != deck_sha256):
                raise ValueError("exact DRC deck differs from the truth manifest")
        kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
        if not kbin.is_file():
            raise ValueError(f"KLayout binary not installed at {kbin}")
        options = tuple(_mim_drc_rd(SimpleNamespace(
            pdk=tech_id, source_cir=truth.source_netlist_path)))
        truth.validate_source_provenance()
        snapshot = create_oracle_gds_snapshot(gds, Path(work_dir), run_label="exact_drc")
        snapshot.retain("Exact DRC oracle inputs and report: retain on success or failure")
        truth.bind_claimed_sha256(snapshot.sha256, what="DRC oracle GDS snapshot")
        deck_snapshot = create_oracle_deck_snapshot(deck, snapshot.path.parent, role="drc")
        if deck_snapshot.sha256 != deck_sha256:
            raise ValueError("DRC deck changed while preparing the oracle")
        inputs = ExactDrcInputs(
            original_gds_path=str(gds), original_gds_sha256=truth.exact_gds_sha256,
            oracle_gds_path=str(snapshot.path), oracle_gds_sha256=snapshot.sha256,
            deck_path=str(deck), deck_sha256=deck_sha256,
            deck_snapshot_path=str(deck_snapshot.path),
            deck_snapshot_sha256=deck_snapshot.sha256,
            source_sha256=truth.source_sha256,
            source_netlist_sha256=truth.source_netlist_sha256,
            pdk_sha256=pdk_sha256,
            route_manifest_sha256=exact_artifact.route_manifest.route_manifest_sha256,
            runtime_options=options,
        )

        def verify_inputs():
            exact_artifact.validate()
            if exact_artifact.truth_manifest != truth:
                raise ValueError("exact DRC artifact truth changed during the checker run")
            truth.validate_source_provenance()
            snapshot.verify_source_unchanged(gds)
            deck_snapshot.verify_unchanged()
            current_adapter, current_identity = _exact_drc_pdk_state(tech_id)
            if current_identity != pdk_identity:
                raise ValueError("exact DRC PDK identity changed during the checker run")
            truth.bind_pdk_identity(current_identity)
            current_deck = current_adapter.klayout_decks().get("drc")
            if (current_deck is None or Path(current_deck).resolve() != deck
                    or sha256_path(deck) != deck_sha256):
                raise ValueError("exact DRC selected deck changed during the checker run")

        verify_inputs()
        report_path = snapshot.path.parent / "exact_drc.lyrdb"
        report = run_gf180_drc(
            str(snapshot.path), topcell=topcell, deck=deck, klayout_bin=kbin,
            report_path=report_path, timeout_sec=timeout_sec,
            extra_rd=list(options) or None,
        )
        verify_inputs()
        named_report = getattr(report, "lyrdb_path", None)
        if named_report is None or Path(named_report).resolve() != report_path.resolve():
            raise ValueError("DRC oracle report names a different report database")
        before_report = sha256_path(report_path)
        certificate = _mint_drc_oracle_certificate(
            report, gds_path=snapshot.path, deck_path=deck, report_path=report_path)
        verify_inputs()
        if (sha256_path(report_path) != before_report
                or certificate.report_sha256 != before_report
                or certificate.gds_sha256 != snapshot.sha256
                or certificate.deck_sha256 != deck_sha256):
            raise ValueError("DRC oracle bytes changed during report certification")
        return DrcVerdict(
            ran=True, clean=certificate.clean, total=certificate.total,
            counts_by_rule=dict(certificate.counts_by_rule),
            deck_path=str(deck), deck_sha256=deck_sha256,
            gds_path=str(gds), gds_sha256=truth.exact_gds_sha256,
            report_path=str(report_path), report_sha256=certificate.report_sha256,
            returncode=certificate.returncode, oracle_certificate=certificate,
            topcell=topcell, exact_inputs=inputs,
        )
    except Exception as exc:
        return DrcVerdict(
            ran=False, clean=False, total=0,
            deck_path=None if deck is None else str(deck),
            gds_path=None if gds is None else str(gds),
            report_path=None if report_path is None else str(report_path),
            returncode=None if report is None else getattr(report, "returncode", None),
            topcell=topcell, reason=f"exact DRC evidence unproven: {exc}",
            exact_inputs=inputs,
        )
