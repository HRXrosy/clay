"""Layout dataset generator for LLM training / few-shot prompting.

Produces (natural-language-description, AnnotatedCell+DeviceSizing) pairs by
sweeping the existing cell template factories over Cartesian parameter grids.
Each sample is serialized via the Phase 17 ``to_json`` helpers so the output
can be persisted as JSONL and consumed by downstream LLM-side pipelines
without re-instantiating Python objects.

Pure Python; no Rust mirror. Does not modify the existing cell-template
factories or ``clay.layout.json_serde`` — only consumes their public API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from .annotations import MatchingKind
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
from .json_serde import to_json


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetSample:
    """One (description, cell+sizing) pair.

    Attributes:
        template_name: name of the cell-template factory that produced this
            sample (e.g. ``build_diff_pair``).
        description: natural-language single-sentence description of the cell
            and its sizing — suitable as an LLM prompt or training target.
        cell_json: ``AnnotatedCell`` serialized via :func:`to_json`.
        sizing_json: ``DeviceSizing`` (dict of ``device -> (w_um, l_um)``)
            serialized via :func:`to_json`.
    """

    template_name: str
    description: str
    cell_json: str
    sizing_json: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matching_kind_str(kind: MatchingKind) -> str:
    """Return a human-readable spelling of the matching kind enum."""
    return kind.value


def _serialize_pair(cell, sizing) -> Tuple[str, str]:
    """Serialize an (AnnotatedCell, DeviceSizing) pair via Phase 17 to_json."""
    return to_json(cell), to_json(sizing)


# ---------------------------------------------------------------------------
# Per-template generators
# ---------------------------------------------------------------------------


def generate_diff_pair_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_diff_pair`` across the Cartesian product of inputs.

    Empty input range -> empty output list.
    """
    samples: List[DatasetSample] = []
    matching = MatchingKind.RANDOM_VT
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_diff_pair(
                    name=f"diff_pair_W{w_um}_L{l_um}_F{n}",
                    w_um=w_um,
                    l_um=l_um,
                    finger_count=n,
                    matching_kind=matching,
                )
                description = (
                    f"Differential pair (build_diff_pair) with "
                    f"W={w_um:.1f} um, L={l_um:.1f} um, "
                    f"finger_count={n}, matching kind {_matching_kind_str(matching)}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_diff_pair",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


def generate_current_mirror_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
    ratio_range: List[int] = [1, 2, 4],
) -> List[DatasetSample]:
    """Sweep ``build_current_mirror`` across the Cartesian product of inputs.

    The output transistor finger count is ``finger_ref * ratio`` so the
    mirror ratio is realized as a finger multiplier (matches the factory's
    documented convention).
    """
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n_ref in finger_range:
                for ratio in ratio_range:
                    n_out = n_ref * ratio
                    cell, sizing = build_current_mirror(
                        name=f"current_mirror_W{w_um}_L{l_um}_Fref{n_ref}_R{ratio}",
                        w_um=w_um,
                        l_um=l_um,
                        finger_count_ref=n_ref,
                        finger_count_out=n_out,
                        ratio=ratio,
                    )
                    description = (
                        f"Current mirror (build_current_mirror) ratio 1:{ratio} "
                        f"with W={w_um:.1f} um, L={l_um:.1f} um, "
                        f"finger_ref={n_ref}, finger_out={n_out}"
                    )
                    cell_json, sizing_json = _serialize_pair(cell, sizing)
                    samples.append(
                        DatasetSample(
                            template_name="build_current_mirror",
                            description=description,
                            cell_json=cell_json,
                            sizing_json=sizing_json,
                        )
                    )
    return samples


def generate_cascoded_mirror_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_cascoded_current_mirror`` across the Cartesian product."""
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_cascoded_current_mirror(
                    name=f"cascoded_mirror_W{w_um}_L{l_um}_F{n}",
                    w_um_low=w_um,
                    l_um_low=l_um,
                    w_um_cas=w_um,
                    l_um_cas=l_um,
                    finger_count_low=n,
                    finger_count_cas=n,
                )
                description = (
                    f"Cascoded current mirror (build_cascoded_current_mirror) "
                    f"with W={w_um:.1f} um, L={l_um:.1f} um, "
                    f"finger_count_low={n}, finger_count_cascode={n}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_cascoded_current_mirror",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


def generate_5t_ota_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_5t_ota`` across the Cartesian product of input pair params."""
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_5t_ota(
                    name=f"5t_ota_W{w_um}_L{l_um}_F{n}",
                    input_w_um=w_um,
                    input_l_um=l_um,
                    input_finger_count=n,
                )
                description = (
                    f"5T OTA (build_5t_ota) with input pair "
                    f"W={w_um:.1f} um, L={l_um:.1f} um, finger_count={n}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_5t_ota",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


def generate_folded_cascode_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_folded_cascode`` across input-pair (W, L, finger) only."""
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_folded_cascode(
                    name=f"folded_cascode_W{w_um}_L{l_um}_F{n}",
                    input_w_um=w_um,
                    input_l_um=l_um,
                    input_finger_count=n,
                )
                description = (
                    f"Folded cascode amplifier (build_folded_cascode) with "
                    f"input pair W={w_um:.1f} um, L={l_um:.1f} um, "
                    f"finger_count={n}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_folded_cascode",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


def generate_telescopic_cascode_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_telescopic_cascode`` over input-pair params."""
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_telescopic_cascode(
                    name=f"telescopic_cascode_W{w_um}_L{l_um}_F{n}",
                    input_w_um=w_um,
                    input_l_um=l_um,
                    input_finger_count=n,
                )
                description = (
                    f"Telescopic cascode amplifier (build_telescopic_cascode) "
                    f"with input pair W={w_um:.1f} um, L={l_um:.1f} um, "
                    f"finger_count={n}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_telescopic_cascode",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


def generate_two_stage_miller_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_two_stage_miller`` over input-pair params."""
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_two_stage_miller(
                    name=f"two_stage_miller_W{w_um}_L{l_um}_F{n}",
                    input_w_um=w_um,
                    input_l_um=l_um,
                    input_finger_count=n,
                )
                description = (
                    f"Two-stage Miller amplifier (build_two_stage_miller) with "
                    f"input pair W={w_um:.1f} um, L={l_um:.1f} um, "
                    f"finger_count={n}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_two_stage_miller",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


def generate_bandgap_samples(
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> List[DatasetSample]:
    """Sweep ``build_bandgap_reference`` over OTA input-pair params.

    The BJT/resistor parts are left at their factory defaults; only the OTA
    sizing (W/L/fingers of the input pair) is swept so this generator
    composes uniformly with the others in :func:`generate_all_template_samples`.
    """
    samples: List[DatasetSample] = []
    for w_um in w_range:
        for l_um in l_range:
            for n in finger_range:
                cell, sizing = build_bandgap_reference(
                    name=f"bandgap_W{w_um}_L{l_um}_F{n}",
                    ota_input_w_um=w_um,
                    ota_input_l_um=l_um,
                    ota_input_finger_count=n,
                )
                description = (
                    f"Bandgap reference (build_bandgap_reference) with "
                    f"OTA input pair W={w_um:.1f} um, L={l_um:.1f} um, "
                    f"finger_count={n}"
                )
                cell_json, sizing_json = _serialize_pair(cell, sizing)
                samples.append(
                    DatasetSample(
                        template_name="build_bandgap_reference",
                        description=description,
                        cell_json=cell_json,
                        sizing_json=sizing_json,
                    )
                )
    return samples


# ---------------------------------------------------------------------------
# Aggregate generator
# ---------------------------------------------------------------------------


# Ordered list of (template_name, generator_fn) pairs. Exposed so callers can
# enumerate the 8 known templates without re-listing them.
_TEMPLATE_GENERATORS: List[Tuple[str, callable]] = [
    ("build_diff_pair", generate_diff_pair_samples),
    ("build_current_mirror", generate_current_mirror_samples),
    ("build_cascoded_current_mirror", generate_cascoded_mirror_samples),
    ("build_5t_ota", generate_5t_ota_samples),
    ("build_folded_cascode", generate_folded_cascode_samples),
    ("build_telescopic_cascode", generate_telescopic_cascode_samples),
    ("build_two_stage_miller", generate_two_stage_miller_samples),
    ("build_bandgap_reference", generate_bandgap_samples),
]


def generate_all_template_samples(
    w_range: List[float] = [2.0, 4.0, 8.0],
    l_range: List[float] = [0.5, 1.0],
    finger_range: List[int] = [1, 4, 16],
) -> List[DatasetSample]:
    """Generate samples for all 8 cell templates with default parameter sweep.

    Each per-template generator is called with the same (w_range, l_range,
    finger_range). The current-mirror generator additionally uses its default
    ``ratio_range`` (``[1, 2, 4]``).

    Returns:
        Flat list of :class:`DatasetSample`, ordered by template.
    """
    all_samples: List[DatasetSample] = []
    for tname, gen in _TEMPLATE_GENERATORS:
        if tname == "build_current_mirror":
            # Uses ratio_range default.
            all_samples.extend(gen(w_range, l_range, finger_range))
        else:
            all_samples.extend(gen(w_range, l_range, finger_range))
    return all_samples


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def write_dataset_jsonl(
    samples: List[DatasetSample],
    output_path: Union[str, Path],
) -> None:
    """Write samples as JSONL — one JSON object per line.

    Each line is the dataclass's :func:`dataclasses.asdict` representation
    encoded with ``json.dumps`` (no indentation). ``cell_json`` and
    ``sizing_json`` remain as nested JSON strings; callers wishing to parse
    them downstream should ``json.loads`` those fields explicitly.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for sample in samples:
            fp.write(json.dumps(asdict(sample), ensure_ascii=False))
            fp.write("\n")
