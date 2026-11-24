# Two-Stage Miller OTA Auto-Onboard & Generic Flow Signoff (`two_stage_ota_auto_gf180`)

Fully automated end-to-end physical realization of [two_stage_ota_auto_gf180.cir](file:///./benchmarks/gf180mcu/two_stage_ota/two_stage_ota_auto_gf180.cir) (`two_stage_ota`: 10 MOS — 5 `nmos_3p3`, 5 `pmos_3p3` — plus segmented `ppolyf_u` nulling resistor `RZ1 = 10 kOhm` and 5LM Option-B MIM compensation capacitor `CC1 = 1.2 pF`) generated via `clay onboard --auto-craft --calibrate-op` and `layout = generic`, completely replacing the legacy hand-written `circuits/two_stage/two_stage_layout.py` script.

## 1. Public Flow Verification (`clay flow run --case`)

- **Command**: `python -m clay flow run --case benchmarks/gf180mcu/two_stage_ota --force --wall-seconds 300 --json`
- **Verdict**: `pass` (**8/8 steps passed**, `14.05 s` wall time; receipt `.clay/flow-runs/268cf08824a0c753a4c44d032dddca71/attempts/be48de7ebc6346289b718531329b94bb/receipt.json`, sha256 `de3015df08137777e88afb598d4389a47c1656d4d8467aba0b266a9c79edc4e7`)
  - `compile`: PASS (`two_stage_ota`, 12 devices)
  - `layout`: PASS (`29.44 um x 113.655 um`, area `3346.0032 um^2`, `symmetry=1.000`, `heal_shape_count=0`, `post_assembly_fill_shape_count=0`, [layout.gds](file:///./benchmarks/gf180mcu/two_stage_ota/_flow_run/work/layout.gds) sha256 `da2227991f5c87b7cea4037fb7e5a764f215bf1408ae17b2b36e9383abfb7097`)
  - `drc`: PASS (`gf180mcu` KLayout DRC = **0 violations**)
  - `lvs`: PASS (`gf180mcu` KLayout LVS = **MATCH**, extracted 12 devices / 13 nets: 5 `nmos_3p3`, 5 `pmos_3p3`, 1 `ppolyf_u`, 1 `mim_single_2p0fF`)
  - `pex`: PASS (Docker Magic R+C PEX conditioned with LVSDB passive devices)
  - `post`: PASS ([tb_two_stage_ota.sp](file:///./benchmarks/gf180mcu/two_stage_ota/tb_two_stage_ota.sp))
  - `intent`: PASS (`signed=True`, score `1.0`, `12/12` obligations passed)
  - `sign`: PASS:
    - `dc_gain_db_post = 91.6103 dB` (`>= 70.0 dB`)
    - `gbw_hz_post = 20.344 MHz` (`>= 15.0 MHz`)
    - `pm_deg_post = 74.5038 deg` (`>= 60.0 deg`)
    - `power_w_post = 620.408 uW` (`<= 1.0 mW`)

## 2. Independent Docker Magic / Netgen / `ngspice-46` Signoff (`hpretl/iic-osic-tools:latest`)

Receipt: [docker_signoff_summary.json](file:///./benchmarks/gf180mcu/two_stage_ota/docker_signoff/docker_signoff_summary.json) (**`status: PASS`**, GDS sha256 `da2227991f5c87b7cea4037fb7e5a764f215bf1408ae17b2b36e9383abfb7097`):

- **Magic `drc(full)`**: **PASS — 0 violations** (`1.125 s`)
- **Netgen LVS**: **PASS** (`circuits_match_uniquely = true`, `netlists_match_uniquely = true`, `property_errors = false`, `badnets_count = 0`, `badelements_count = 0`, `external_pins_matched = true`)
- **Magic `extresist` R+C PEX**: **PASS** (`0` port shorts, `0` negative capacitors, `215` wire resistors, `167` capacitors = `207.53 fF`, `22` MOS + `1` `cap_mim_2f0_m4m5_noshield` + `1` `ppolyf_u`)
- **Docker `ngspice-46` Full R+C Post-Sim vs Pre-Sim** (`tt`, `27 °C`, `VDD=3.3 V`):
  - `DC Gain`: `91.60 -> 91.61 dB` (`+0.01 dB`)
  - `GBW`: `20.58 -> 20.48 MHz` (`-0.47%`)
  - `Phase Margin`: `75.85° -> 77.95°` (`+2.10°`)
  - `Power`: `620.7 -> 620.4 uW` (`-0.05%`)
