# Unseen Benchmark 2: Fully-Differential Telescopic Cascode OTA (`telescopic_ota_gf180`)

## Circuit Summary
- **PDK**: `gf180mcu` 3.3 V (`nmos_3p3` / `pmos_3p3` + MIM capacitors)
- **Subcircuit**: `.subckt telescopic_ota_gf180 vinp vinn voutp voutn vcm ibias vdd vss`
- **Device count**: 24 MOS (13 NMOS, 11 PMOS) + 2 matched MIM capacitors (`CC_P`, `CC_N = 0.5 pF`), 8 external ports
- **Key generalization stress targets**:
  - PMOS differential input pair (`xMPIN_P`, `xMPIN_N`) with floating source-tied N-well (`body = ntail != vdd`)
  - 5-transistor vertical telescopic cascode stack (`xMPTAIL` -> `xMPIN_P/N` -> `xMPCAS_P/N` -> `xMNCAS_P/N` -> `xMNLOAD_P/N`) with stacked floating source nodes (`nsp/nsn`, `ndp/ndn`) and high-impedance differential outputs (`voutp`, `voutn`)
  - Continuous-time dual-diff-pair CMFB (`xMCM_TA/A1/A2`, `xMCM_TB/B1/B2`, `xMCM_LP/LN`) avoiding resistive loading of high-Z outputs
  - Matched `0.5 pF` differential MIM compensation capacitors (`CC_P`, `CC_N`) targeting Option-B MIM extraction

## Pre-Layout Verification (`tb_telescopic_ota.sp`, `ngspice_con.exe`, tt/27C)
- `vout_cm_v`: `1.44576 V` (target `Vcm = 1.45 V`)
- `dc_gain_db`: `82.3484 dB` (Spec `>= 65 dB`)
- `gbw_hz`: `48.8852 MHz` (Spec `>= 25 MHz`)
- `pm_deg`: `77.8968 deg` (Spec `>= 60 deg`)
- `idd_a`: `424.436 uA` (Spec `<= 600 uA`)
- `power_w`: `1.40064 mW` (Spec `<= 2.0 mW`)
- Simulation wall time: `< 0.5 s`

## End-to-End Automated Generalization (`clay onboard` + `clay flow run --case`)
- **Public admission receipt**: `e3d7e251a5fb226ce21a5e815eb91177aa696dfedf99abf926593aa6335694aa` (`attempts=4`, `74.79 s`)
- **Step verdicts**: `compile=pass`, `layout=pass` (`26 devices`, `2042 FEOL + 8197 BEOL polys`, `symmetry=1.000`, `88.70 x 144.035 um`, `12775.9045 um^2`), `drc=pass` (`gf180mcu DRC: CLEAN (0 violations)`), `lvs=pass` (`gf180mcu LVS: MATCH (extracted 26 devices / 21 nets)`), `pex=pass` (`2 MIM caps + 149 geometric route caps`, `calibrated=False, field_solver=False`), `post=pass`, `intent=pass` (`17/17`, `score=1.0`), `sign=pass`.
- **Post-layout KPIs (`ngspice_con.exe` on `telescopic_ota_gf180.pex.spice`)**: `dc_gain_db_post=82.3484 dB`, `gbw_hz_post=45.8152 MHz` (`-6.28%` vs pre `48.8852 MHz`), `pm_deg_post=77.2329 deg` (`-0.66 deg` vs pre `77.8968 deg`), `idd_a_post=424.436 uA`, `power_w_post=1.40064 mW`, `vout_cm_v_post=1.44576 V`.
- **`diagnose_postsim_rc` analysis**: `45 actions` across `coupling_imbalance` and `high_z_cap` (`32` suggested `layout_intent` patches).

## 5LM Option-B MIM Public Flow + Docker Magic/Netgen/`ngspice-46` Signoff (Post-`MT.1=440nm`)

After enabling GF180 5LM Option-B MIM (`mim_option: B`, `mim_cap: "2"`, `metal_level: 5LM`) in [case.yaml](file:///./benchmarks/gf180mcu/telescopic_ota/case.yaml) and landing the C++20 PathFinder + `MT.1=440nm` metal-top DRC fix, generic `clay flow run --case` converges on default spacing with **0 shorts** and **0 DRC** in both KLayout and Magic:

- **Public flow receipt**: `70aa10c2d379dcab374d7cc75779bf77e7dbe4e803e6eb905cd39cb787b039b6` (`attempts=7`, `22.73 s` wall time, **8/8 steps PASS**):
  - `layout`: `88.70 x 135.725 um` (`12038.8075 um^2`), `symmetry=1.000`, `heal_shape_count=0`, [layout.gds](file:///./benchmarks/gf180mcu/telescopic_ota/_flow_run/work/layout.gds) sha256 `0ebd91036c17e0372d18cb56714c50514b3b53ceaf50670cc573861b9a9e8862`
  - `drc`: **0 violations** | `lvs`: **MATCH** (`26 devices / 21 nets`) | `pex`: **Magic R+C PEX** | `intent`: **17/17 (`score=1.0`, `signed=True`)**
  - `sign`: `dc_gain_db_post=81.96 dB`, `gbw_hz_post=47.74 MHz`, `pm_deg_post=75.77 deg`, `idd_a_post=430.39 uA`, `power_w_post=1.420 mW`
- **Independent Docker Magic / Netgen / `ngspice-46` Signoff**: [docker_signoff_summary.json](file:///./benchmarks/gf180mcu/telescopic_ota/docker_signoff_post_mt1/docker_signoff_summary.json) (**`status: PASS`**, identical GDS sha256 `0ebd91036c17e0372d18cb56714c50514b3b53ceaf50670cc573861b9a9e8862`):
  - **Magic `drc(full)`**: **PASS — 0 violations** (down from 26 pre-`MT.1` fix)
  - **Netgen LVS**: **PASS** (`circuits_match_uniquely=true`, `property_errors=false`, `badnets=0`, `badelements=0`)
  - **Magic `extresist` R+C PEX**: **PASS** (`496 R / 362 C / 50 MOS + 2 cap_mim_2f0_m4m5_noshield`, `0` port shorts)
  - **Docker `ngspice-46` R+C Post-Sim**: `Gain 82.35 -> 81.96 dB`, `GBW 48.89 -> 47.59 MHz`, `PM 77.90° -> 76.00°`, `Power 1.401 -> 1.420 mW`.
