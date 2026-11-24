* tb_loadstep.sp - LDO gf180mcu 3.3V: load-step transient (100uA <-> 5mA)
* Derived verbatim from circuits/ldo/tb_loadstep.sp with explicit physical units (scale=1.0).

* @GF180_PDK corner=tt@
.option scale=1.0
.include "ldo_auto_gf180.cir"

Vin   vin  0  DC 3.3
Vref  vref 0  DC 1.25

X1 vin vout vref vfb vfbs 0 ldo_gf180
Vshort vfb vfbs DC 0

Resr  vout vout_cap 0.05
Cload vout_cap 0 10u

Iload vout 0 PWL(
+   0        100u
+   50u      100u
+   50.1u    5m
+   150u     5m
+   150.1u   100u
+   250u     100u
+ )

.tran 20n 250u

.control
set noaskquit
run

meas tran vout_light_tr find v(vout) at=49u
meas tran vout_heavy_tr find v(vout) at=149u
meas tran vout_min_up   MIN v(vout) FROM=50.1u TO=150u
meas tran vout_max_down MAX v(vout) FROM=150.1u TO=250u

let undershoot_mV = (vout_heavy_tr - vout_min_up)*1000
let overshoot_mV  = (vout_max_down - vout_light_tr)*1000
let droop_mV      = (vout_light_tr - vout_heavy_tr)*1000

echo "=== Load-step results (100uA <-> 5mA, 100ns edges, CLoad=10uF) ==="
print vout_light_tr vout_heavy_tr droop_mV undershoot_mV overshoot_mV
echo undershoot_mv=$&undershoot_mV
echo overshoot_mv=$&overshoot_mV
echo droop_mv=$&droop_mV
quit
.endc
.end
