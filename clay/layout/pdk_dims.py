"""PDK dimensional constants for Clay Phase 1 layout-as-co-variable.

Each ``PdkDims`` concrete subclass exposes a small set of fab-specific
geometric constants (pitches, minimum length) and per-device mismatch
coefficients (A_VT). Only the values needed by Phase 1 are surfaced here;
richer DRC/layer stacks live in the dedicated PDK subsystem.

NOTE on A_VT values: TENTATIVE, recalibrate against silicon data in
Phase 1.5; slope test in Phase 1 demo is robust to absolute A_VT
magnitude.
"""

from abc import ABC, abstractmethod
from enum import Enum


class DeviceType(Enum):
    NMOS = "nmos"
    PMOS = "pmos"


class PdkDims(ABC):
    """Abstract interface for Phase 1 PDK dimensional constants."""

    @abstractmethod
    def name(self) -> str:
        """Stable identifier for the PDK (e.g. 'sky130')."""

    @abstractmethod
    def fin_pitch_nm(self) -> float:
        """Fin / active pitch in nm."""

    @abstractmethod
    def metal1_pitch_nm(self) -> float:
        """Metal1 routing pitch in nm."""

    @abstractmethod
    def poly_pitch_nm(self) -> float:
        """Poly (gate) pitch in nm."""

    @abstractmethod
    def min_l_nm(self) -> float:
        """Minimum drawn channel length in nm."""

    @abstractmethod
    def avt_mv_um(self, device: DeviceType) -> float:
        """Pelgrom A_VT coefficient in mV*um for the given device type."""


class Sky130(PdkDims):
    """SkyWater 130nm Phase 1 dimensional surface."""

    def name(self) -> str:
        return "sky130"

    def fin_pitch_nm(self) -> float:
        return 48.0

    def metal1_pitch_nm(self) -> float:
        return 280.0

    def poly_pitch_nm(self) -> float:
        return 340.0

    def min_l_nm(self) -> float:
        return 150.0

    # NOTE: A_VT values below are TENTATIVE; recalibrate against silicon
    # data in Phase 1.5. The Phase 1 demo's slope test is robust to the
    # absolute A_VT magnitude.
    def avt_mv_um(self, device: DeviceType) -> float:
        if device is DeviceType.NMOS:
            return 5.0
        if device is DeviceType.PMOS:
            return 8.0
        raise ValueError(f"Unsupported device type: {device!r}")


class Gf180mcu(PdkDims):
    """GlobalFoundries 180nm MCU Phase 1 dimensional surface."""

    def name(self) -> str:
        return "gf180mcu"

    def fin_pitch_nm(self) -> float:
        return 280.0

    def metal1_pitch_nm(self) -> float:
        return 480.0

    def poly_pitch_nm(self) -> float:
        return 600.0

    def min_l_nm(self) -> float:
        return 180.0

    # NOTE: A_VT values below are TENTATIVE; recalibrate against silicon
    # data in Phase 1.5. The Phase 1 demo's slope test is robust to the
    # absolute A_VT magnitude.
    def avt_mv_um(self, device: DeviceType) -> float:
        if device is DeviceType.NMOS:
            return 6.0
        if device is DeviceType.PMOS:
            return 10.0
        raise ValueError(f"Unsupported device type: {device!r}")
