* Testbench for Unseen Benchmark 2: Fully-Differential Telescopic Cascode OTA (gf180mcu 3.3V)
* Verifies:
*   - DC operating point, CMFB regulation (vout_cm_v ~ 1.45 V), and supply current (idd_a, power_w)
*   - Open-loop differential AC response using closed-loop DC / open-loop AC feedback:
*     dc_gain_db >= 65 dB, gbw_hz >= 25 MHz, pm_deg >= 60 deg

.option scale=1.0
* @GF180_PDK corner=tt@
.include "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/design.ngspice"
.lib "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/sm141064.ngspice" typical
.param sw_stat_global=0 sw_stat_mismatch=0

.param VDD   = 3.3
.param IBIAS = 40u
.param VCM   = 1.45
.param CLD   = 1.5p
.param RFB   = 100Meg
.param CFB   = 1

Vdd vdd 0 DC {VDD}
Vss vss 0 DC 0
Ibias vdd ibias DC {IBIAS}
Vcm_ref vcm 0 DC {VCM}

* Differential DC bias lock (survives post-layout mismatch) + AC isolation
Rfb_n voutp vinn_fb {RFB}
Cfb_n vinn_fb 0     {CFB}
Rfb_p voutn vinp_fb {RFB}
Cfb_p vinp_fb 0     {CFB}

Edrive_p vinp vinp_fb vin_ac 0  0.5
Edrive_n vinn vinn_fb vin_ac 0 -0.5
Vac vin_ac 0 DC 0 AC 1

X1 vinp vinn voutp voutn vcm ibias vdd vss telescopic_ota_gf180
.include "telescopic_ota_gf180.cir"

CLp voutp 0 {CLD}
CLn voutn 0 {CLD}

.control
  set units = degrees
  op
  let idd_a = abs(vdd#branch)
  let power_w = idd_a * 3.3
  let vout_cm_v = 0.5 * (v(voutp) + v(voutn))
  let vout_off_v = v(voutp) - v(voutn)

  echo "=== Telescopic OTA Results (gf180mcu 3.3V tt) ==="
  echo vout_cm_v=$&vout_cm_v
  echo vout_off_v=$&vout_off_v
  echo idd_a=$&idd_a
  echo power_w=$&power_w

  ac dec 50 1 1G
  setplot ac1
  let vod = v(voutp) - v(voutn)
  let gain_db = db(vod)
  let phase_d = vp(vod)

  meas ac dc_gain_db   find gain_db at=1
  meas ac gbw_hz       when gain_db=0 cross=1
  meas ac phase_at_ugf find phase_d when gain_db=0 cross=1
  let pm_deg = 180 + phase_at_ugf

  echo dc_gain_db=$&dc_gain_db
  echo gbw_hz=$&gbw_hz
  echo pm_deg=$&pm_deg

  quit
.endc
.end
