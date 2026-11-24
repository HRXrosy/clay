# Unseen Benchmark 1: 4-bit Current-Steering DAC Core (`cs_dac_4bit_gf180`)

## Circuit Summary
- **PDK**: `gf180mcu` 3.3 V (`nmos_3p3` / `pmos_3p3`)
- **Subcircuit**: `.subckt cs_dac_4bit_gf180 ioutp ioutn ibias b0 b0b b1 b1b b2 b2b b3 b3b vdd vss`
- **Device count**: 22 MOS (20 NMOS, 2 PMOS), 13 external ports
- **Key generalization stress targets**:
  - Non-1:1 (`1:2:4:8`) binary-weighted current-source array (`xMI0..xMI3`, `W = 12u, 24u, 48u, 96u`, `L = 2u`) and cascode shield tier (`xMC0..xMC3`, `W = 8u, 16u, 32u, 64u`, `L = 1u`) sharing reference diode `xMNBIAS` / `xMCB_DIO`
  - 4 differential current-steering switch pairs (`xMSW0P/N..xMSW3P/N`) with `1:2:4:8` width scaling
  - Shared high-current differential output trunks (`ioutp`, `ioutn`, `IFS ~ 1.5 mA`) and dense switch control pins (`b0..b3`, `b0b..b3b`)

## Pre-Layout Verification (`tb_cs_dac_4bit.sp`, `ngspice_con.exe`, tt/27C)
- `ifs_a`: `1.49916 mA` (Spec `>= 1.4 mA`)
- `lsb_a`: `99.9438 uA`
- `inl_max_lsb`: `0.001987 LSB` (Spec `<= 0.1 LSB`)
- `dnl_max_lsb`: `0.003749 LSB` (Spec `<= 0.1 LSB`)
- `mono_ok`: `1` (strictly monotonic across all 16 codes)
- `idd_a`: `1.79975 mA` (Spec `<= 2.5 mA`)
- `power_w`: `5.93916 mW` (Spec `<= 8.0 mW`)
- Simulation wall time: `< 0.6 s`

## End-to-End Automated Generalization (`clay onboard` + `clay flow run --case`)
- **Public admission receipt**: `8c479e41b0fd85db3978a7d05db8970b8c7a5ba12f001a2c7bd04fa9cda279d6` (`attempts=2`, `40.68 s`)
- **Step verdicts**: `compile=pass`, `layout=pass` (`22 devices`, `308 FEOL + 5298 BEOL polys`, `symmetry=1.000`, `32.38 x 154.65 um`, `5007.567 um^2`), `drc=pass` (`gf180mcu DRC: CLEAN (0 violations)`), `lvs=pass` (`gf180mcu LVS: MATCH (extracted 22 devices / 24 nets)`), `pex=pass` (`123 geometric route caps`, `calibrated=False, field_solver=False`), `post=pass`, `intent=pass` (`18/18`, `score=1.0`), `sign=pass`.
- **Post-layout KPIs (`ngspice_con.exe` on `cs_dac_4bit_gf180.pex.spice`)**: `ifs_a_post=1.49916 mA`, `inl_max_lsb_post=0.001987`, `dnl_max_lsb_post=0.003749`, `mono_ok_post=1.0`, `idd_a_post=1.79975 mA`, `power_w_post=5.93916 mW`.
