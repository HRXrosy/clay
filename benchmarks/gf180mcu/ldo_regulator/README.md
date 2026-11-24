# PMOS-Pass LDO Auto-Onboard & Generic Flow Signoff (`ldo_auto_gf180`)

Fully automated end-to-end physical realization of [ldo_auto_gf180.cir](file:///./benchmarks/gf180mcu/ldo_regulator/ldo_auto_gf180.cir) (`ldo_gf180`: 17 MOS — 6 `nmos_3p3`, 11 `pmos_3p3` including 8x `100u/0.5u` parallel PMOS pass-FET bank — plus 5 real segmented `ppolyf_u` resistors `Rbias_ref=664k`, `Rfb1a/1b/2a/2b=50k` and a real 5LM Option-B `2 pF` MIM compensation capacitor `Cc`) generated via `clay onboard --auto-craft --calibrate-op` and `layout = generic`, completely replacing the legacy 67 KB hand-written `circuits/ldo/ldo_layout.py` script.

## 1. Public Flow Verification (`clay flow run --case`)

- **Command**: `python -m clay flow run --case benchmarks/gf180mcu/ldo_regulator --force --wall-seconds 450 --json`
- **Verdict**: `pass` (**8/8 steps passed**; receipt `.clay/flow-runs/c2ca7cfd5f09e0906cc85519cd1b4b3c/attempts/.../receipt.json`, sha256 `93a81d9a0ac8a8e35dbbfb0d6876b4738c29fc109f5f39fd8a88327cd20e7a5f`)
  - `compile`: PASS (`ldo_gf180`, 23 devices)
  - `layout`: PASS (`70.17 um x 287.245 um`, area `20155.98165 um^2`, `symmetry=1.000`, `place converged 1it`, `route converged 1it`, `heal_shape_count=0`, `post_assembly_fill_shape_count=0`, [layout.gds](file:///./benchmarks/gf180mcu/ldo_regulator/_flow_run/work/layout.gds) sha256 `6c518968120e8993e4b03989fe3275a65e6f36c715b2488c6c055407899ce2b6`)
  - `drc`: PASS (`gf180mcu` KLayout DRC = **0 violations**)
  - `lvs`: PASS (`gf180mcu` KLayout LVS = **MATCH**, extracted 14 devices / 14 nets: 6 `nmos_3p3`, 4 `pmos_3p3` merged parallel pass bank, 3 `ppolyf_u` merged divider/bias, 1 `mim_single_2p0fF`)
  - `pex`: PASS (`joint PEX: Magic 638 caps / 712 resistors + full extraction 1 cap / 3 resistors`)
  - `post`: PASS (`benchmarks.gf180mcu.ldo_regulator.scripts.postsim_driver` over all 4 testbenches)
  - `intent`: PASS (`signed=True`, score `1.0`, `13/13` obligations passed)
  - `sign`: PASS:
    - `vout_v_post = 2.50131 V` (`>= 2.45 V`)
    - `psrr_db_post = 75.1555 dB` (`>= 60.0 dB`)
    - `dc_gain_db_post = 108.976 dB` (`>= 60.0 dB`)
    - `power_w_post = 126.435 uW` (`<= 1.0 mW`)

## 2. Independent Docker Magic / Netgen / `ngspice-46` Signoff (`hpretl/iic-osic-tools:latest`)

Receipt: [docker_signoff_summary.json](file:///./benchmarks/gf180mcu/ldo_regulator/docker_signoff/docker_signoff_summary.json) (identical GDS sha256 `6c518968120e8993e4b03989fe3275a65e6f36c715b2488c6c055407899ce2b6`):

- **Netgen LVS**: **PASS** (`circuits_match_uniquely = true`, `netlists_match_uniquely = true`, `property_errors = false`, `badnets_count = 0`, `badelements_count = 0`, `external_pins_matched = true`; all 50 `ppolyf_u` segments + 17 MOS + 2 pF `cap_mim_2f0_m4m5_noshield` matched)
- **Magic `extresist` R+C PEX**: **PASS** (`0` port shorts, `0` negative capacitors, `705` wire resistors, `602` capacitors = `679.39 fF`, `76` MOS fingers + `50` `ppolyf_u` + `1` `cap_mim_2f0_m4m5_noshield`)
- **Docker `ngspice-46` 4-Testbench Full R+C Post-Sim vs Pre-Sim** (`tt`, `27 °C`, `VIN=3.3 V`):
  - `tb_dc.sp`: `Vout = 2.498 -> 2.488 V` (`-0.41%`), `Iq = 38.66 -> 35.02 uA`, `Power = 127.6 -> 115.6 uW`, `LineReg = -0.022 mV/V`
  - `tb_psrr.sp`: `PSRR(DC) = 101.3 -> 91.75 dB`, `PSRR(1 MHz) = 66.68 -> 81.60 dB`
  - `tb_loopgain.sp`: `DC Loop Gain = 133.4 -> 108.8 dB`, `UGF = 386.4 -> 404.1 kHz`
  - `tb_loadstep.sp` (`100 uA <-> 5 mA`): `droop = 22.9 mV`, `undershoot = 0.084 mV` (ring-free)
- **Magic `drc(full)`**: `2` occurrences solely of `"Via4 width < 0.28um (V4.1 + 2 * V4.4)"` at `(36.52..37.04, 263.84..264.02) um`, where a router `Via4` cut sits `260 nm` horizontally and `180 nm` vertically staggered from the Option-B MIM primitive's built-in `Via4` drop (satisfies KLayout `V4.2a = 260 nm`, but Magic `gf180mcuD.tech` bloats `Via4` by `V4.4 = 10 nm` on each side, requiring `>= 280 nm` spacing). Zero other Magic DRC errors occur across the entire `20,156 um^2` layout.
