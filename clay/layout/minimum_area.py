"""Integer minimum-area square sizing shared by PDK geometry consumers."""
from math import isqrt
from typing import Optional


def minimum_area_square_side_nm(minimum_area_nm2: Optional[int]) -> int:
    """Smallest integer side whose square meets the declared area, or zero.

    This is the historical TechGeometry.via_pad area floor. Manufacturing-grid
    alignment and cut enclosure remain obligations of the consuming constructor.
    """
    if minimum_area_nm2 is None:
        return 0
    area = int(minimum_area_nm2)
    side = isqrt(area)
    return side if side * side == area else side + 1
