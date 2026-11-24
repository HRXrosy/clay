* tb_dc.sp - LDO gf180mcu 3.3V: DC regulation + line regulation + Iq
* Derived verbatim from circuits/ldo/tb_dc.sp with explicit physical units (scale=1.0).

* @GF180_PDK corner=tt@
.option scale=1.0
.include "ldo_auto_gf180.cir"

Vin   vin  0  DC 3.3
Vref  vref 0  DC 1.25

X1 vin vout vref vfb vfbs 0 ldo_gf180
Vshort vfb vfbs DC 0

Resr  vout vout_cap 0.05
Cload vout_cap 0 10u

Iload vout 0 DC 1m

.control
set noaskquit

echo "=== Load regulation 100uA -> 5mA (DC sweep Iload) ==="
alter Vin = 3.3
dc Iload 100u 5m 100u
meas dc vout_light find v(vout) at=100u
meas dc vout_heavy find v(vout) at=5m
let loadreg_mV_per_mA = (vout_light - vout_heavy)*1000/4.9
print vout_light vout_heavy loadreg_mV_per_mA

echo "=== Line regulation: VIN 3.0..3.6 @ Iload=2mA ==="
alter Iload = 2m
dc Vin 3.0 3.6 0.1
meas dc vout_30 find v(vout) at=3.0
meas dc vout_33 find v(vout) at=3.3
meas dc vout_36 find v(vout) at=3.6
let linereg_mV_per_V = (vout_36 - vout_30)*1000/0.6
print vout_30 vout_33 vout_36 linereg_mV_per_V
echo vout_v=$&vout_33

echo "=== Quiescent current (i(Vin) minus load @ Iload=2mA, VIN=3.3) ==="
alter Iload = 2m
alter Vin = 3.3
op
let iq_uA = (abs(i(Vin)) - 2m)*1e6
let pwr_w = (abs(i(Vin)) - 2m)*3.3
print iq_uA
echo iq_ua=$&iq_uA
echo power_w=$&pwr_w

quit
.endc
.end
