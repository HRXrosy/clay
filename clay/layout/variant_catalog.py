"""Cell-template variant catalog.

Generates a labeled catalog of (variant_name -> VariantSpec) tuples from a
template name + W/L/finger grid. Pure-Python, side-effect free.

Use case: LLM-facing convenience for sweeping sizing across a single
cell template (e.g. enumerate 4x4x3 diff_pair sizings, then dispatch each
to the appropriate factory in ``clay.layout.cell_templates``).

The label encoder uses 'p' in place of the decimal point so labels stay
filesystem-safe (no '.' in filenames), e.g. ``diff_pair_w4_l0p5_n4``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List


@dataclass(frozen=True)
class VariantSpec:
    """A single variant: template + sizing knobs + canonical label."""

    template_name: str
    w_um: float
    l_um: float
    finger_count: int
    variant_label: str


@dataclass
class VariantCatalog:
    """Ordered collection of VariantSpecs sharing a template name."""

    template_name: str
    variants: List[VariantSpec] = field(default_factory=list)

    def labels(self) -> List[str]:
        """Return the list of variant labels in catalog order."""
        return [v.variant_label for v in self.variants]

    def by_label(self, label: str) -> VariantSpec:
        """Look up a variant by its label. Raises KeyError if missing."""
        for v in self.variants:
            if v.variant_label == label:
                return v
        raise KeyError(
            f"variant label {label!r} not found in catalog "
            f"for template {self.template_name!r}"
        )


def _encode_decimal(value: float) -> str:
    """Encode a float so the decimal point becomes 'p'.

    Examples:
        4.0    -> '4'
        0.5    -> '0p5'
        0.25   -> '0p25'
        1.5    -> '1p5'

    Trailing zeros after the decimal point are stripped, and integer
    values are emitted without a trailing 'p' or '0'.
    """
    # Normalize via repr-style formatting; strip trailing zeros / dot.
    # Use a wide precision then trim, to avoid '1e-05' notation for small
    # but in-range values.
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if text == "" or text == "-":
        text = "0"
    return text.replace(".", "p").replace("-", "neg")


def build_variant_label(
    template_name: str,
    w_um: float,
    l_um: float,
    finger_count: int,
) -> str:
    """Encode (template, W, L, N) into a filesystem-safe label.

    Examples:
        build_variant_label("diff_pair", 4.0, 0.5, 4) == "diff_pair_w4_l0p5_n4"
        build_variant_label("current_mirror", 2.0, 0.18, 2) ==
            "current_mirror_w2_l0p18_n2"
    """
    return (
        f"{template_name}"
        f"_w{_encode_decimal(float(w_um))}"
        f"_l{_encode_decimal(float(l_um))}"
        f"_n{int(finger_count)}"
    )


def generate_variant_catalog(
    template_name: str,
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> VariantCatalog:
    """Cartesian product over W x L x finger_count.

    The returned catalog's ``variants`` list is sorted by (w_um, l_um,
    finger_count). An empty range in any dimension yields an empty
    catalog (with ``template_name`` preserved).
    """
    variants: List[VariantSpec] = []
    for w, l, n in product(w_range, l_range, finger_range):
        label = build_variant_label(template_name, w, l, n)
        variants.append(
            VariantSpec(
                template_name=template_name,
                w_um=float(w),
                l_um=float(l),
                finger_count=int(n),
                variant_label=label,
            )
        )
    variants.sort(key=lambda v: (v.w_um, v.l_um, v.finger_count))
    return VariantCatalog(template_name=template_name, variants=variants)


def render_catalog_csv(catalog: VariantCatalog) -> str:
    """Render the catalog as CSV text.

    Header: ``variant_label,template,w_um,l_um,finger_count``.
    Rows are emitted in catalog order. Trailing newline included.
    """
    lines = ["variant_label,template,w_um,l_um,finger_count"]
    for v in catalog.variants:
        lines.append(
            f"{v.variant_label},{v.template_name},"
            f"{v.w_um},{v.l_um},{v.finger_count}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "VariantSpec",
    "VariantCatalog",
    "build_variant_label",
    "generate_variant_catalog",
    "render_catalog_csv",
]
