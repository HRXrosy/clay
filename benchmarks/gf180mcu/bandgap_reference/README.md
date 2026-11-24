# BandgapBiasRobustR158 Auto-Onboard & Generic Flow Signoff (`bandgap_r158_auto_gf180`)

Fully automated end-to-end physical realization of [bandgap_r158_auto_gf180.cir](file:///./benchmarks/gf180mcu/bandgap_reference/bandgap_r158_auto_gf180.cir) (`BandgapBiasRobustR158`: 15 MOS + 9 vertical PNP `vpnp_0p42x10` + 3 `ppolyf_u` resistors `12k / 175k / 1697.5k`) using Clay's generic auto-onboard and `layout = generic` flow without touching `circuits/bandgap/bandgap_layout.py`.

## 1. Public Flow Verification (`clay flow run --case`)

- **Command**: `python -m clay flow run --case benchmarks/gf180mcu/bandgap_reference --retry-receipt 90f1aa04e355eecf4b9ce41b2a14dfe2c58e1475c497236a11bf0dd75dc31b2e --retry-reason producer_changed --retry-evidence clay/frontend/tb_adapter.py --retry-evidence benchmarks/gf180mcu/bandgap_reference/tb_bandgap_r158.sp --wall-seconds 600 --json`
- **Verdict**: `pass` (`8/8` steps passed; receipt `.clay/flow-runs/ae227998986c7db27a03013a78134516/attempts/4c91a1dc1b2b484aa57567c2a50f4603/receipt.json`, sha256 `1a159daf9e07da72139ca4ac3970efcf3ddda87c4f77495e0dea59b4ec3f7d7d`)
  - `compile`: PASS (`BandgapBiasRobustR158`, 27 devices)
  - `layout`: PASS (`135.04um x 136.27um`, area `18401.9008 um^2`, symmetry `1.000`, route converged 1 iter, `heal_shape_count=0`, `post_assembly_fill_shape_count=0`, [layout.gds](file:///./benchmarks/gf180mcu/bandgap_reference/_flow_run/work/layout.gds) sha256 `1cd155a096b3f90b8cf2e377baae3556ae6049baba40e1b6f6f569291f849acb`)
  - `drc`: PASS (`gf180mcu` KLayout DRC = **0 violations**)
  - `lvs`: PASS (`gf180mcu` KLayout LVS = **MATCH**, extracted 20 devices / 15 nets: 6 `nmos_3p3`, 9 `pmos_3p3`, 3 `ppolyf_u`, 2 `vpnp_0p42x10`)
  - `pex`: PASS (`BandgapBiasRobustR158.pex.spice`)
  - `post`: PASS ([tb_bandgap_r158.sp](file:///./benchmarks/gf180mcu/bandgap_reference/tb_bandgap_r158.sp))
  - `intent`: PASS (`signed=True`, score `1.0`, `11/11` obligations passed)
  - `sign`: PASS:
    - `vref_v_post = 1.20587 V` (`>= 1.15 V`)
    - `tempco_ppm_post = 26.2433 ppm/°C` (`<= 80.0 ppm/°C`)
    - `power_w_post = 0.000109746 W` (`109.75 uW <= 500 uW`)

## 2. Independent Docker Magic / Netgen / Host `ngspice_con.exe` Signoff (`hpretl/iic-osic-tools:latest`)

Receipt: [docker_signoff_summary.json](file:///./benchmarks/gf180mcu/bandgap_reference/docker_signoff/docker_signoff_summary.json)

- **Magic `extresist` R+C PEX (`magic_rc_pex`)**: PASS — `0` port shorts, `819` wire resistors (`136,972.46 Ohm` total including P-substrate resistive mesh), `385` capacitors (`698.51 fF` total, `negative_cap_ratio = 0.0101 <= 0.02`), `139` active/passive subckts (`22` MOS + `9` PNP + `108` `ppolyf_u`).
- **Netgen LVS**: `circuits_match_uniquely = true`, `netlists_match_uniquely = true`, `badnets_count = 0`, `badelements_count = 0`, `external_pins_matched = true`, `extra_gds_port_labels = []`. (Property warning occurs only because bare ideal `R` lines `12k / 175k / 1697.5k` in `.cir` were inferred at `r_width=1.0u` by the Docker Netgen script whereas physical segmented `ppolyf_u` uses `w=0.8u / 0.84u` with the exact same `L/W` aspect ratio.)
- **Magic `drc(full)`**: `45` occurrences solely of `"P-diffusion and N-ohmic must be over N-well or deep N-well"` (`5` boxes per `vpnp_0p42x10` x `9` BJTs). Direct container verification on the unmodified foundry GDS `pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/cells/klayout/pymacros/cells/bjt/vpnp_0p42x10.gds` yields the identical `5` violations per BJT (`gf180mcuD.tech` line 1348 `missing_nwell` omits `nbase` from `and-not dnwell,nwell`, whereas line 252 defines `allnwell nwell,rnwell,nbase`); zero layout or routing Magic DRC errors occur elsewhere.
- **Full Magic `extresist` R+C Post-Sim (`BandgapBiasRobustR158.rc.pex.spice`, Docker `ngspice-46` & Host `ngspice_con.exe`)**:
  - Note: Repo `pdks/gf180mcu/.../sm141064.ngspice:38642` has `rb 11 21 ppolyf_u_body l=r_l w=r_w\n+r='...'` (parsed as `R = r_l = 40 uOhm`), whereas Docker `/foss/pdks/gf180mcuD/libs.tech/ngspice/sm141064.ngspice` orders `+r='...' l=r_l w=r_w` (`~18.85 kOhm` per `0.8u x 40u` segment). Also `tb_bandgap_r158.sp` omitted `.lib ... res_typical`.
  - With Docker's patched `ppolyf_u` model (`res_typical`) and matching reordered `rb` on host `ngspice_con.exe` (`reltol=1e-6`, `.option rshunt=1e14`), both simulators match identically over `-29..125 °C` (`gmin_msgs=0`):
    - `Vref(-29°C) = 1.20442 V`, `Vref(-20°C) = 1.20437 V`, `Vref(0°C) = 1.20396 V` (`-29..0°C` with continuation), `Vref(27°C) = 1.20292 V`, `Vref(60°C) = 1.20112 V`, `Vref(85°C) = 1.19952 V`, `Vref(125°C) = 1.19686 V`
    - `Idd(27°C) = 29.931 uA`, `Power(27°C) = 98.772 uW`, `-29..125°C` spread `7.57 mV` (`TC = 40.86 ppm/°C`). (Single-point DC convergence on the full extracted `extresist` netlist below `-29°C` jumps to the `2.93 V` bistable branch under `ngspice` DC continuation.)

