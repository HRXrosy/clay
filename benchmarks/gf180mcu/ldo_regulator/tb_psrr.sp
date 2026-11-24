* tb_psrr.sp - LDO gf180mcu 3.3V: PSRR = -vdb(vout) for 1 V AC on VIN
* Derived verbatim from circuits/ldo/tb_psrr.sp with explicit physical units (scale=1.0).

* @GF180_PDK corner=tt@
.option scale=1.0
.include "ldo_auto_gf180.cir"

Vin   vin  0  DC 3.3 AC 1
Vref  vref 0  DC 1.25

X1 vin vout vref vfb vfbs 0 ldo_gf180
Vshort vfb vfbs DC 0

Resr  vout vout_cap 0.05
Cload vout_cap 0 10u

Iload vout 0 DC 2m

.control
set noaskquit
set units=degrees
ac dec 20 1 10Meg

let psrr_db_vec = -vdb(vout)
meas ac psrr_dc    find psrr_db_vec at=1
meas ac psrr_100   find psrr_db_vec at=100
meas ac psrr_1k    find psrr_db_vec at=1k
meas ac psrr_10k   find psrr_db_vec at=10k
meas ac psrr_100k  find psrr_db_vec at=100k
meas ac psrr_1meg  find psrr_db_vec at=1Meg

echo "=== PSRR (dB), Iload=2mA ==="
print psrr_dc psrr_100 psrr_1k psrr_10k psrr_100k psrr_1meg
echo psrr_db=$&psrr_dc
quit
.endc
.end
