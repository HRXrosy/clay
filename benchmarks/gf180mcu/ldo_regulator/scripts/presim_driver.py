"""Auto-generated generic presim driver for ldo_gf180 via clay.frontend.tb_adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from clay.frontend.tb_adapter import run_tb_kpis

CASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_CIR = (CASE_DIR / 'ldo_auto_gf180.cir').resolve()
TB_PATHS = tuple((CASE_DIR / rel).resolve() for rel in ('tb_dc.sp', 'tb_psrr.sp', 'tb_loopgain.sp', 'tb_loadstep.sp'))
DEFAULT_PDK = 'gf180mcu'
CELL_NAME = 'ldo_gf180'
DEFAULT_VDD = 3.3

def run() -> Dict[str, float]:
    return run_tb_kpis(
        SOURCE_CIR,
        TB_PATHS,
        pdk_id=DEFAULT_PDK,
        vdd_v=DEFAULT_VDD,
        is_post=False,
        subckt_name=CELL_NAME,
    )

def run_candidate(
    source_cir: str | Path,
    *,
    testbench_params: Optional[Dict[str, Any]] = None,
    sizing: Optional[Dict[str, Any]] = None,
    work_dir: Optional[str | Path] = None,
    pdk: str = DEFAULT_PDK,
    vdd: float = DEFAULT_VDD,
    corner: str = 'tt',
    temp_c: float = 27.0,
    **extra: Any,
) -> Dict[str, float]:
    merged_sizing = dict(sizing or {})
    tb_params = dict(testbench_params or {})
    # Route W/L/M/C/R parameters into DUT sizing if matching cir .params
    for k, v in list(tb_params.items()):
        if k.lower() not in ('vdd', 'vcm', 'ibias', 'cld', 'vin_cm', 'vin_diff'):
            merged_sizing.setdefault(k, v)
    return run_tb_kpis(
        Path(source_cir),
        TB_PATHS,
        pdk_id=pdk,
        corner=corner,
        temp_c=float(temp_c),
        sizing=merged_sizing or None,
        testbench_params=tb_params or None,
        vdd_v=float(vdd),
        is_post=False,
        subckt_name=CELL_NAME,
    )
