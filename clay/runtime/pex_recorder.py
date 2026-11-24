"""Wire the real Magic PEX extraction into a RunRecorder event stream.

This module is the thin bridge between :class:`MagicPexBackend` (calibrated
2.5D Magic ext2spice extraction; NOT StarRC, NOT a geometric estimate) and
:class:`clay.runtime.recorder.RunRecorder`. Each extraction emits structured
events the schematic_viewer can replay or live-monitor:

  * ``pex_started``    -- emitted before invoking Magic; carries the cell name,
                          PDK, GDS path, and whether Magic is actually
                          available right now.
  * ``pex_extracted``  -- emitted after a successful Magic run; carries the
                          parasitic-element counts, backend identity, the
                          ``calibrated`` flag, and a pointer to the final
                          ``<cell>.pex.spice`` artifact copied into the run
                          archive.
  * ``pex_skipped``    -- emitted when Magic is unavailable OR when the run
                          raises. The extraction failure is recorded honestly
                          (with the exception class and message) and we return
                          ``None`` instead of inventing parasitics.

The skipped-vs-extracted contract is deliberate: fabricating a PexReport in
place of a real Magic run would corrupt downstream coupling-cap analysis. A
caller that needs to fall back to the geometric backend can do so explicitly
through :func:`clay.layout.pex.get_backend`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from clay.layout.pex import magic_runner as mr
from clay.layout.pex.base import PexReport
from clay.layout.pex.magic_backend import MagicPexBackend
from clay.runtime.recorder import RunRecorder


__all__ = ["extract_with_recorder", "record_pex_summary"]


def extract_with_recorder(
    gds_path: str,
    cell_name: str,
    pdk: str = "gf180mcu",
    *,
    recorder: Optional[RunRecorder] = None,
    revision: str = "pex",
    work_dir: Optional[str] = None,
    summary: Optional[str] = None,
) -> Optional[PexReport]:
    """Run Magic PEX with recorder events; return the PexReport or None.

    Behaviour:

    * If ``recorder`` is None this is a pure pass-through to
      ``MagicPexBackend().extract(...)`` (no events emitted, original
      exception propagation preserved).
    * If ``recorder`` is provided:

        - Emits ``pex_started`` with cell / pdk / gds_path and the live
          ``magic_available()`` probe.
        - On success: emits ``pex_extracted`` with element counts and the
          backend metadata; copies the ``<cell>.pex.spice`` file into the
          run archive as a final artifact named ``<cell>.pex.spice``.
        - On Magic unavailability or any exception during extraction:
          emits ``pex_skipped`` with a reason (and the exception class/message
          when applicable) and returns ``None`` -- it does NOT re-raise. The
          run can continue and record the honest "no parasitics available"
          state instead of fabricating parasitics.

    ``work_dir`` (when given) is the directory used to locate the produced
    ``<cell>.pex.spice``. It defaults to the parent directory of ``gds_path``,
    which is where ``magic_runner.run_ext2spice`` writes the file.
    """
    backend = MagicPexBackend()

    if recorder is None:
        # Pure pass-through: keep the original semantics (raises on failure).
        return backend.extract(gds_path, cell_name, pdk)

    magic_avail = mr.magic_available()
    recorder.event(
        "pex_started",
        summary=summary or f"Starting Magic PEX extraction for {cell_name} ({pdk}).",
        revision=revision,
        data={
            "title": "Magic PEX extraction starting",
            "cell_name": cell_name,
            "pdk": pdk,
            "gds_path": str(gds_path),
            "calibrated_attempt": True,
            "magic_available": magic_avail,
            "source": "clay.layout.pex.MagicPexBackend",
        },
    )

    if not magic_avail:
        recorder.event(
            "pex_skipped",
            summary=f"PEX skipped: Magic unavailable for {cell_name}.",
            revision=revision,
            data={
                "title": "PEX skipped (Magic unavailable)",
                "cell_name": cell_name,
                "pdk": pdk,
                "gds_path": str(gds_path),
                "reason": "Magic Docker image unavailable",
                "source": "clay.layout.pex.magic_runner.magic_available",
            },
        )
        return None

    try:
        report = backend.extract(gds_path, cell_name, pdk)
    except Exception as exc:
        recorder.event(
            "pex_skipped",
            summary=f"PEX skipped: extraction failed for {cell_name}.",
            revision=revision,
            data={
                "title": "PEX skipped (extraction failed)",
                "cell_name": cell_name,
                "pdk": pdk,
                "gds_path": str(gds_path),
                "reason": "Magic extraction raised",
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "source": "clay.layout.pex.MagicPexBackend.extract",
            },
        )
        return None

    # Capture the .pex.spice as a final artifact when present on disk.
    spice_dir = Path(work_dir) if work_dir is not None else Path(gds_path).resolve().parent
    pex_spice_src = spice_dir / f"{cell_name}.pex.spice"
    artifacts: Optional[dict[str, str]] = None
    pex_spice_artifact_rel: Optional[str] = None
    if pex_spice_src.exists():
        copied = recorder.copy_final_artifact(
            "pex_spice",
            pex_spice_src,
            filename=f"{cell_name}.pex.spice",
        )
        pex_spice_artifact_rel = f"final/{copied.name}"
        artifacts = {"pex_spice": pex_spice_artifact_rel}

    record_pex_summary(
        recorder,
        report,
        revision=revision,
        summary=summary,
        artifacts=artifacts,
        extra_data={
            "gds_path": str(gds_path),
            "pex_spice": pex_spice_artifact_rel,
        },
    )
    return report


def record_pex_summary(
    recorder: RunRecorder,
    report: PexReport,
    *,
    revision: str = "pex",
    summary: Optional[str] = None,
    artifacts: Optional[dict[str, str]] = None,
    extra_data: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a ``pex_extracted`` event for an already-built ``PexReport``.

    Lower-level helper for cases where the extraction ran through a different
    code path (e.g. a cached report) but the run archive should still record
    the same structured event the live viewer expects.
    """
    nets = report.nets()
    data: dict[str, Any] = {
        "title": f"PEX extracted ({report.backend})",
        "cell_name": report.cell_name,
        "backend": report.backend,
        "calibrated": report.calibrated,
        "field_solver": report.field_solver,
        "n_caps": len(report.caps),
        "n_resistors": len(report.resistors),
        "n_nets": len(nets),
        "total_cap_aF": report.total_cap_aF,
        "connectivity_source": report.connectivity_source,
        "constants_source": report.constants_source,
        "source": "clay.layout.pex.MagicPexBackend",
    }
    if extra_data:
        data.update({k: v for k, v in extra_data.items() if v is not None})

    recorder.event(
        "pex_extracted",
        summary=summary or (
            f"PEX extracted {len(report.caps)} caps / "
            f"{len(report.resistors)} resistors / {len(nets)} nets "
            f"for {report.cell_name}."
        ),
        revision=revision,
        artifacts=artifacts,
        data=data,
    )
