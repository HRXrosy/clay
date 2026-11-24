* GF180MCU Constant-gm + 1:8 Vertical-PNP Bandgap Voltage Reference (BandgapBiasRobustR158)
* @GF180_PDK corner=tt@
.include "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/design.ngspice"
.lib "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/sm141064.ngspice" typical
.lib "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/sm141064.ngspice" bjt_typical
.param sw_stat_global=0 sw_stat_mismatch=0

.include "bandgap_r158_auto_gf180.cir"

Vdd vdd 0 3.3
Vss vss 0 0
Xdut vref vdd vss BandgapBiasRobustR158
.nodeset v(vref)=1.2

.control
  dc temp -40 125 1
  let vref_dc = v(vref)
  let xm40 = vref_dc[0]
  let xm20 = vref_dc[20]
  let x0   = vref_dc[40]
  let x27  = vref_dc[67]
  let x60  = vref_dc[100]
  let x85  = vref_dc[125]
  let x125 = vref_dc[165]
  let idd27_v = abs(i(Vdd)[67])
  let vminv = minimum(vref_dc)
  let vmaxv = maximum(vref_dc)
  let tempcov = ((vmaxv - vminv) / (x27 * 165.0)) * 1e6
  let pwr27v = idd27_v * 3.3
  let compv = (tempcov < 100.0) & (x27 > 1.0) & (x27 < 1.5)
  echo "=== BandgapBiasRobustR158 Temp Sweep (-40..125 C, gf180mcu 3.3V tt) ==="
  echo vref_m40c=$&xm40
  echo vref_m20c=$&xm20
  echo vref_0c=$&x0
  echo vref_27c=$&x27
  echo vref_60c=$&x60
  echo vref_85c=$&x85
  echo vref_125c=$&x125
  echo vref_min_v=$&vminv
  echo vref_max_v=$&vmaxv
  echo tempco=$&tempcov
  echo idd_a=$&idd27_v
  echo power_w=$&pwr27v
  echo compensated=$&compv
  quit
.endc
.end