"""Auto-generated generic postsim driver for folded_cascode_ota via clay.frontend.tb_adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from clay.frontend.tb_adapter import run_tb_kpis

CASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_CIR = (CASE_DIR / 'fc_ota_auto_gf180.cir').resolve()
TB_PATHS = tuple((CASE_DIR / rel).resolve() for rel in ('tb_fc_ota.sp',))
DEFAULT_PDK = 'gf180mcu'
CELL_NAME = 'folded_cascode_ota'
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
) -> Dict[str, Any]:
    c_list = tuple(str(c) for c in (corners or ('tt',)))
    t_list = tuple(float(t) for t in (temps_c or (27.0,)))
    nom_c = 'tt' if 'tt' in c_list else c_list[0]
    nom_t = 27.0 if 27.0 in t_list else t_list[0]
    nominal = run_tb_kpis(
        Path(pex_spice),
        TB_PATHS,
        pdk_id=pdk,
        corner=nom_c,
        temp_c=nom_t,
        vdd_v=DEFAULT_VDD,
        is_post=True,
        subckt_name=CELL_NAME,
    )
    if pvt_mode != 'acceptance' or (len(c_list) == 1 and len(t_list) == 1):
        out: Dict[str, Any] = dict(nominal)
        if pvt_mode == 'acceptance':
            out['__pvt__'] = {
                'mode': 'acceptance',
                'corners': list(c_list),
                'temps_c': list(t_list),
                'nominal_post': dict(nominal),
            }
        return out
    samples = [(nom_c, nom_t, nominal)]
    for c in c_list:
        for t in t_list:
            if c == nom_c and t == nom_t:
                continue
            samples.append((c, t, run_tb_kpis(
                Path(pex_spice), TB_PATHS, pdk_id=pdk, corner=c, temp_c=t,
                vdd_v=DEFAULT_VDD, is_post=True, subckt_name=CELL_NAME,
            )))
    agg: Dict[str, Any] = dict(nominal)
    bounds = dict(kpi_bounds or {})
    for key in list(nominal.keys()):
        vals = [float(s[2][key]) for s in samples if key in s[2]]
        if not vals:
            continue
        bnd = bounds.get(key)
        op = bnd[0] if isinstance(bnd, (tuple, list)) and bnd else None
        if op == 'ge':
            agg[key] = min(vals)
        elif op == 'le':
            agg[key] = max(vals)
        elif op == 'near' and len(bnd) >= 2:
            tgt = float(bnd[1][0])
            agg[key] = max(vals, key=lambda v: abs(v - tgt))
    agg['__pvt__'] = {
        'mode': 'acceptance',
        'corners': list(c_list),
        'temps_c': list(t_list),
        'nominal_post': dict(nominal),
    }
    return agg
