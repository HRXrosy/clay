"""Symbolic layout representation for Clay Phase 1 layout-as-co-variable.

A ``SymbolicLayout`` is a small, immutable-ish bundle of integer-valued
layout co-variables (finger count, stripe count) plus a rational pitch
multiplier. It carries no PDK-specific dimensions on its own; conversion
to physical lengths is done by combining it with a ``PdkDims`` instance.
"""

from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class SymbolicLayout:
    """Co-variable layout descriptor.

    Attributes
    ----------
    finger_count:
        Number of fingers used for the device. Must be >= 1.
    pitch_multiplier:
        Rational multiplier applied to the PDK fin pitch when computing
        the realized layout pitch. Defaults to 1/1.
    stripe_count:
        Number of parallel stripes / rows the device is folded across.
        Must be >= 1.
    """

    finger_count: int = 1
    pitch_multiplier: Fraction = field(default_factory=lambda: Fraction(1, 1))
    stripe_count: int = 1

    def __post_init__(self) -> None:
        if self.finger_count < 1:
            raise ValueError(
                f"finger_count must be >= 1, got {self.finger_count}"
            )
        if self.stripe_count < 1:
            raise ValueError(
                f"stripe_count must be >= 1, got {self.stripe_count}"
            )

    def area_um2(self, w_um: float, l_um: float) -> float:
        """Total active area in um^2 across all fingers."""
        return w_um * l_um * self.finger_count

    def pitch_value_nm(self, pdk_fin_pitch_nm: float) -> float:
        """Realized pitch in nm given the PDK fin pitch."""
        return pdk_fin_pitch_nm * float(self.pitch_multiplier)
