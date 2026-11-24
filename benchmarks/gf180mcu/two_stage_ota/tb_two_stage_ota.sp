* Two-Stage Miller OTA -- gf180mcu 3.3V -- AC Gain / GBW / PM + DC Idd
* Derived verbatim from circuits/two_stage/tb_ac.sp + tb_op.sp.

.option scale=1.0
* @GF180_PDK corner=tt@
.param sw_stat_global=0 sw_stat_mismatch=0

.param VDD   = 3.3
.param VCM   = 1.2
.param IBIAS = 10u
.param CLD   = 5p
.param RFB   = 1G
.param CFB   = 1

Vdd vdd 0 DC {VDD}
Vss vss 0 DC 0
Ibias vdd ibias DC {IBIAS}

VINP vinp 0 DC {VCM} AC 1
Rfb  vout vinn {RFB}
Cfb  vinn 0    {CFB}

X1 vinp vinn vout ibias vdd vss two_stage_ota
.include "two_stage_ota_auto_gf180.cir"

CL vout 0 {CLD}

.control
  set noaskquit
  set units = degrees

  op
  let idd_a_val = -i(Vdd)
  let pwr_w_val = idd_a_val * 3.3
  echo idd_a=$&idd_a_val
  echo power_w=$&pwr_w_val

  ac dec 50 1 1G
  let gain_db = vdb(vout)
  let phase_d = vp(vout)

  meas ac dc_gain_db   find gain_db at=1
  meas ac ugf_hz       when gain_db=0 cross=1
  meas ac phase_at_ugf find phase_d when gain_db=0 cross=1
  meas ac phase_dc     find phase_d at=1
  let pm_deg = 180 + phase_at_ugf

  echo "=== Two-Stage OTA AC+OP Results (gf180mcu 3.3V tt) ==="
  echo dc_gain_db=$&dc_gain_db
  echo ugf_hz=$&ugf_hz
  echo pm_deg=$&pm_deg
  echo phase_dc=$&phase_dc
  echo phase_at_ugf=$&phase_at_ugf
  quit
.endc
.end
