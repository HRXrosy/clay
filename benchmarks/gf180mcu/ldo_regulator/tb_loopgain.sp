* tb_loopgain.sp - LDO gf180mcu 3.3V: open-loop gain + stability margins
* Derived verbatim from circuits/ldo/tb_loopgain.sp with explicit physical units (scale=1.0).

* @GF180_PDK corner=tt@
.option scale=1.0
.include "ldo_auto_gf180.cir"

Vin   vin  0  DC 3.3
Vref  vref 0  DC 1.25

X1 vin vout vref vfb vfbs 0 ldo_gf180

Rfb    vfbs  vfb_dc  1G
Cfb    vfb_dc 0      1
Vac    vfb   vfb_dc  AC 1

Resr  vout vout_cap 0.05
Cload vout_cap 0 10u

Iload vout 0 DC 2m

.control
set noaskquit
set units=degrees

alter Iload = 2m
op
ac dec 40 1 100Meg
let g = vdb(vout)
let p = vp(vout)
meas ac dc_loop_gain find g at=1
meas ac ugf_hz       when g=0 cross=1
meas ac phase_at_ugf find p when g=0 cross=1
meas ac phase_10MHz  find p at=10Meg
meas ac gain_10MHz   find g at=10Meg
echo "=== LDO Loop Gain @ Iload=2mA ==="
print dc_loop_gain ugf_hz phase_at_ugf phase_10MHz gain_10MHz
echo dc_gain_db=$&dc_loop_gain
echo ugf_hz=$&ugf_hz
echo phase_at_ugf=$&phase_at_ugf
quit
.endc
.end
