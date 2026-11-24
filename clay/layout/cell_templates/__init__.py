"""Phase 6 cell template library — high-level factories for common analog cells.

Each factory returns (AnnotatedCell, DeviceSizing) ready to pass to
lower_annotated_cell_drc_aware. Pure Python, no Rust mirror.

Use case: LLM-facing convenience. Instead of building up SymbolicLayout +
add_annotation calls manually, call e.g. build_diff_pair(...).
"""
from .diff_pair import build_diff_pair
from .current_mirror import build_current_mirror
from .cascoded_mirror import build_cascoded_current_mirror
from .five_t_ota import build_5t_ota
from .folded_cascode import build_folded_cascode
from .telescopic_cascode import build_telescopic_cascode
from .two_stage_miller import build_two_stage_miller
from .bandgap import build_bandgap_reference

__all__ = [
    "build_diff_pair",
    "build_current_mirror",
    "build_cascoded_current_mirror",
    "build_5t_ota",
    "build_folded_cascode",
    "build_telescopic_cascode",
    "build_two_stage_miller",
    "build_bandgap_reference",
]
