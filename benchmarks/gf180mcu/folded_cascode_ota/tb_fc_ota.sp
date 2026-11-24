* Folded-Cascode OTA -- gf180mcu 3.3V -- AC Gain / GBW / PM + DC Idd
* Derived verbatim from circuits/fc_ota/tb_ac.sp.

.option scale=1.0
* @GF180_PDK corner=tt@
.param sw_stat_global=0 sw_stat_mismatch=0

.param VDD   = 3.3
.param VCM   = 1.65
.param IBIAS = 20u
.param CLD   = 2p
.param RFB   = 1G
.param CFB   = 1

Vdd vdd 0 DC {VDD}
Vss vss 0 DC 0
Ibias vdd ibias DC {IBIAS}

Vinp vinp 0 DC {VCM} AC 1
Rfb  vout vinn {RFB}
Cfb  vinn 0    {CFB}
CL   vout 0    {CLD}

X1 vinp vinn vout ibias vdd vss folded_cascode_ota
.include "fc_ota_auto_gf180.cir"

.control
  set noaskquit
  set units = degrees

  op
  let idd_a_val = -i(Vdd)
  let pwr_w_val = idd_a_val * 3.3
  echo idd_a=$&idd_a_val
  echo power_w=$&pwr_w_val

  ac dec 30 1 1G
  let gain_db = vdb(vout)
  let phase_d = vp(vout)

  meas ac dc_gain_db   find gain_db at=1
  meas ac ugf_hz       when gain_db=0 cross=1
  meas ac phase_at_ugf find phase_d when gain_db=0 cross=1
  meas ac phase_dc     find phase_d at=1
  let pm_deg_val = 180 - (phase_dc - phase_at_ugf)

  echo dc_gain_db=$&dc_gain_db
  echo gbw_hz=$&ugf_hz
  echo pm_deg=$&pm_deg_val
  quit
.endc
.end

