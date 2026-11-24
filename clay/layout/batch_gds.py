"""Batch GDS generator for the 8 standard cell template factories.

Given a target directory, build the default-sized version of each cell
template, lower it via ``lower_annotated_cell_drc_complete``, and emit a
binary ``.gds`` file per template through the Rust CLI subprocess.

If the Rust binary is not built, every result is reported with
``skipped_reason`` set rather than raising — callers (tests, batch
scripts, LLM tooling) can decide whether to fail or continue.

No new public types or constraints on the underlying APIs; this is a
pure orchestration layer over the existing ``cell_templates`` factories
and ``emit_binary_gds_via_subprocess``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .annotations import AnnotatedCell
from .backend import DeviceSizing, lower_annotated_cell_drc_complete
from .binary_gds_subprocess import (
    cargo_binary_available,
    emit_binary_gds_via_subprocess,
)
from .cell_templates import (
    build_5t_ota,
    build_bandgap_reference,
    build_cascoded_current_mirror,
    build_current_mirror,
    build_diff_pair,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
)


@dataclass
class BatchGdsResult:
    """Outcome of a single template -> binary GDS emit attempt.

    Attributes:
        template_name: registry key for the template (e.g. ``"diff_pair"``).
        output_path: the path the .gds file was (or would have been) written to.
        bytes_written: file size in bytes; ``0`` if skipped.
        skipped_reason: human-readable explanation when the cargo binary
            is unavailable; ``None`` when the file was written.
    """

    template_name: str
    output_path: str
    bytes_written: int
    skipped_reason: Optional[str] = None


# Factory callable signature: takes ``name`` plus kwargs, returns (cell, sizing).
_FactoryFn = Callable[..., Tuple[AnnotatedCell, DeviceSizing]]

# Registry of the 8 cell templates with their default-sizing kwargs. Each
# entry is (template_name, factory_fn, default_build_kwargs). The kwargs are
# minimum-viable defaults: every factory still produces a fully-annotated
# cell that lowers cleanly under ``lower_annotated_cell_drc_complete``.
_TEMPLATE_REGISTRY: List[Tuple[str, _FactoryFn, Dict[str, Any]]] = [
    (
        "diff_pair",
        build_diff_pair,
        {"w_um": 4.0, "l_um": 1.0, "finger_count": 4},
    ),
    (
        "current_mirror",
        build_current_mirror,
        {
            "w_um": 4.0,
            "l_um": 1.0,
            "finger_count_ref": 4,
            "finger_count_out": 4,
        },
    ),
    (
        "cascoded_mirror",
        build_cascoded_current_mirror,
        {
            "w_um_low": 4.0,
            "l_um_low": 1.0,
            "w_um_cas": 4.0,
            "l_um_cas": 0.5,
            "finger_count_low": 4,
            "finger_count_cas": 4,
        },
    ),
    (
        "5t_ota",
        build_5t_ota,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "folded_cascode",
        build_folded_cascode,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "telescopic_cascode",
        build_telescopic_cascode,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "two_stage_miller",
        build_two_stage_miller,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "bandgap_reference",
        build_bandgap_reference,
        {},
    ),
]


_TEMPLATE_LOOKUP: Dict[str, Tuple[_FactoryFn, Dict[str, Any]]] = {
    name: (fn, dict(defaults)) for name, fn, defaults in _TEMPLATE_REGISTRY
}


_CARGO_MISSING_REASON = (
    "Rust CLI binary `clay_layout_gds_bin` not available; "
    "build via `cargo build --manifest-path crates/clay_layout/Cargo.toml "
    "--bin clay_layout_gds_bin` to enable binary GDS emission."
)


def _resolve_kwargs(template_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge default build kwargs with caller-supplied overrides.

    Raises:
        KeyError: if ``template_name`` is not in the registry.
    """
    if template_name not in _TEMPLATE_LOOKUP:
        known = ", ".join(sorted(_TEMPLATE_LOOKUP.keys()))
        raise KeyError(
            f"unknown template_name {template_name!r}; "
            f"known templates: {known}"
        )
    _, defaults = _TEMPLATE_LOOKUP[template_name]
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _emit_one(
    template_name: str,
    factory: _FactoryFn,
    build_kwargs: Dict[str, Any],
    output_path: Path,
    pdk_layers: Any,
    fin_pitch_nm: float,
) -> BatchGdsResult:
    """Build, lower, and emit a single template; assumes cargo is available."""
    cell, sizing = factory(name=template_name, **build_kwargs)
    lib = lower_annotated_cell_drc_complete(
        cell, pdk_layers, fin_pitch_nm, sizing,
    )
    emit_binary_gds_via_subprocess(lib, output_path)
    bytes_written = output_path.stat().st_size if output_path.exists() else 0
    return BatchGdsResult(
        template_name=template_name,
        output_path=str(output_path),
        bytes_written=bytes_written,
        skipped_reason=None,
    )


def generate_template_gds(
    template_name: str,
    output_path: Union[str, Path],
    pdk_layers: Any,
    fin_pitch_nm: float,
    **build_kwargs: Any,
) -> BatchGdsResult:
    """Build the default (or overridden) instance of one cell template and
    emit a binary ``.gds`` file.

    Args:
        template_name: registry key (e.g. ``"diff_pair"``,
            ``"two_stage_miller"``).
        output_path: target ``.gds`` path. Parent directory is created if
            missing.
        pdk_layers: PDK layer map (e.g. ``Sky130LayerMap()``).
        fin_pitch_nm: PDK fin pitch in nm (e.g. ``SKY130_FIN_PITCH_NM``).
        **build_kwargs: optional overrides forwarded to the factory; merged
            on top of the template's default kwargs.

    Returns:
        ``BatchGdsResult`` with ``skipped_reason`` set if the Rust CLI
        binary is not available, otherwise reporting bytes written.

    Raises:
        KeyError: if ``template_name`` is not in the registry.
    """
    merged_kwargs = _resolve_kwargs(template_name, build_kwargs)
    factory, _ = _TEMPLATE_LOOKUP[template_name]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cargo_binary_available():
        return BatchGdsResult(
            template_name=template_name,
            output_path=str(output_path),
            bytes_written=0,
            skipped_reason=_CARGO_MISSING_REASON,
        )

    return _emit_one(
        template_name=template_name,
        factory=factory,
        build_kwargs=merged_kwargs,
        output_path=output_path,
        pdk_layers=pdk_layers,
        fin_pitch_nm=fin_pitch_nm,
    )


def generate_all_template_gds(
    output_dir: Union[str, Path],
    pdk_layers: Any,
    fin_pitch_nm: float,
) -> List[BatchGdsResult]:
    """For every registered cell template, build the default-sized instance,
    lower via ``lower_annotated_cell_drc_complete``, and emit a binary GDS
    file at ``output_dir/<template_name>.gds``.

    Args:
        output_dir: directory to write the 8 ``.gds`` files into. Created
            if it does not yet exist.
        pdk_layers: PDK layer map (e.g. ``Sky130LayerMap()``).
        fin_pitch_nm: PDK fin pitch in nm (e.g. ``SKY130_FIN_PITCH_NM``).

    Returns:
        List of ``BatchGdsResult`` (one per template, in registry order).
        If the cargo binary is missing, every result has
        ``skipped_reason`` set and ``bytes_written == 0``; no file writes
        are attempted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_available = cargo_binary_available()
    results: List[BatchGdsResult] = []

    for template_name, factory, default_kwargs in _TEMPLATE_REGISTRY:
        output_path = output_dir / f"{template_name}.gds"
        if not binary_available:
            results.append(
                BatchGdsResult(
                    template_name=template_name,
                    output_path=str(output_path),
                    bytes_written=0,
                    skipped_reason=_CARGO_MISSING_REASON,
                )
            )
            continue
        results.append(
            _emit_one(
                template_name=template_name,
                factory=factory,
                build_kwargs=dict(default_kwargs),
                output_path=output_path,
                pdk_layers=pdk_layers,
                fin_pitch_nm=fin_pitch_nm,
            )
        )

    return results


__all__ = [
    "BatchGdsResult",
    "generate_all_template_gds",
    "generate_template_gds",
]
