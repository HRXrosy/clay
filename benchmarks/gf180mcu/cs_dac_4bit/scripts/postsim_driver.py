"""Auto-generated generic postsim driver for cs_dac_4bit_gf180 via clay.frontend.tb_adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from clay.frontend.tb_adapter import run_tb_kpis

CASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_CIR = (CASE_DIR / 'cs_dac_4bit_gf180.cir').resolve()
TB_PATHS = tuple((CASE_DIR / rel).resolve() for rel in ('tb_cs_dac_4bit.sp',))
DEFAULT_PDK = 'gf180mcu'
CELL_NAME = 'cs_dac_4bit_gf180'
DEFAULT_VDD = 3.3

def run_post_kpis(
    pex_spice: Path,
    *,
    pdk: str = DEFAULT_PDK,
    corners: tuple[str, ...] = ('tt',),
    temps_c: tuple[float, ...] = (27.0,),
    pvt_mode: str = 'nominal',
    kpi_bounds: Optional[Dict[str, Any]] = None,
    simulation_seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, float]:
    corner = str(corners[0]) if corners else 'tt'
    temp_c = float(temps_c[0]) if temps_c else 27.0
    return run_tb_kpis(
        Path(pex_spice),
        TB_PATHS,
        pdk_id=pdk,
        corner=corner,
        temp_c=temp_c,
        vdd_v=DEFAULT_VDD,
        is_post=True,
        subckt_name=CELL_NAME,
    )
