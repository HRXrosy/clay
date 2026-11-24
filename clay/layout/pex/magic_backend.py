"""MagicPexBackend -- 2.5D PEX by driving Magic ext2spice.

``field_solver=False``. Calibration follows the installed electrical constants.
Gates out
(``available()`` is False, ``extract`` raises) when Magic/Docker is absent so
callers fall back to the geometric backend via ``clay.layout.pex.get_backend``.
"""
from __future__ import annotations

from . import magic_runner as mr
from .base import PexReport
from .magic_parser import parse_ext2spice


class MagicPexBackend:
    name = "magic"

    def available(self) -> bool:
        return mr.magic_available()

    def supports_pdk(self, pdk: str) -> bool:
        return mr.pdk_supported(pdk)

    def extract(
        self,
        gds_path: str,
        cell_name: str,
        pdk: str = "gf180mcu",
        with_resistance: bool = False,
    ) -> PexReport:
        if not self.available():
            raise RuntimeError(
                "Magic unavailable; fall back to GeometricPexBackend "
                "(see clay.layout.pex.get_backend)"
            )
        if not self.supports_pdk(pdk):
            raise RuntimeError(f"No runnable Magic technology for PDK {pdk!r}")
        if with_resistance:
            text = mr.run_ext2spice(
                gds_path, cell_name, pdk, with_resistance=True
            )
        else:
            # default call signature kept unchanged (cap-only path)
            text = mr.run_ext2spice(gds_path, cell_name, pdk)
        calibrated = pdk in {"gf180mcu", "sky130"}
        if not calibrated:
            try:
                from clay.pdk.pex_profile import get_profile

                profile = get_profile(pdk)
                calibrated = bool(profile is not None and profile.calibrated)
            except Exception:
                calibrated = False
        return parse_ext2spice(
            text,
            cell_name=cell_name,
            pdk=pdk,
            calibrated=calibrated,
        )
