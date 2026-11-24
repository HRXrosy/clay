* Testbench for Unseen Benchmark 1: 4-bit Current-Steering DAC Core (gf180mcu 3.3V)
* Verifies:
*   - DC operating point & supply current (idd_a, power_w)
*   - Full-scale differential output current (ifs_a >= 1.4 mA)
*   - 16-code monotonicity (mono_ok == 1)
*   - Static INL and DNL across all 16 codes (inl_max_lsb <= 0.1, dnl_max_lsb <= 0.1)

.option scale=1.0
* @GF180_PDK corner=tt@
.include "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/design.ngspice"
.lib "../../../pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/models/ngspice/sm141064.ngspice" typical
.param sw_stat_global=0 sw_stat_mismatch=0

.param VDD = 3.3
.param IREF = 100u
.param RLOAD = 250

Vdd vdd 0 DC {VDD}
Vss vss 0 DC 0
Ibias vdd ibias DC {IREF}

* Matched load terminations on shared high-current output trunks
RLP vdd ioutp {RLOAD}
RLN vdd ioutn {RLOAD}

Vb0  b0  0 DC 0
Vb0b b0b 0 DC {VDD}
Vb1  b1  0 DC 0
Vb1b b1b 0 DC {VDD}
Vb2  b2  0 DC 0
Vb2b b2b 0 DC {VDD}
Vb3  b3  0 DC 0
Vb3b b3b 0 DC {VDD}

X1 ioutp ioutn ibias b0 b0b b1 b1b b2 b2b b3 b3b vdd vss cs_dac_4bit_gf180
.include "cs_dac_4bit_gf180.cir"

.control
  set noaskquit

  let iout_p = vector(16)
  let iout_n = vector(16)
  let inl_max_lsb = 0
  let dnl_max_lsb = 0
  let mono_ok = 1

  let idx = 0
  while idx < 16
    let bit0 = floor(idx - 2 * floor(idx / 2))
    let bit1 = floor(floor(idx / 2) - 2 * floor(floor(idx / 2) / 2))
    let bit2 = floor(floor(idx / 4) - 2 * floor(floor(idx / 4) / 2))
    let bit3 = floor(floor(idx / 8) - 2 * floor(floor(idx / 8) / 2))
    if bit0 > 0.5
      alter Vb0 = 3.3
      alter Vb0b = 0
    else
      alter Vb0 = 0
      alter Vb0b = 3.3
    end
    if bit1 > 0.5
      alter Vb1 = 3.3
      alter Vb1b = 0
    else
      alter Vb1 = 0
      alter Vb1b = 3.3
    end
    if bit2 > 0.5
      alter Vb2 = 3.3
      alter Vb2b = 0
    else
      alter Vb2 = 0
      alter Vb2b = 3.3
    end
    if bit3 > 0.5
      alter Vb3 = 3.3
      alter Vb3b = 0
    else
      alter Vb3 = 0
      alter Vb3b = 3.3
    end
    op
    let ip = (3.3 - v(ioutp)) / 250
    let in_ = (3.3 - v(ioutn)) / 250
    let iout_p[idx] = ip
    let iout_n[idx] = in_
    let idx = idx + 1
  end

  let idd_a = abs(vdd#branch)
  let power_w = idd_a * 3.3
  let ifs_a = iout_p[15] - iout_p[0]
  let lsb_a = ifs_a / 15
  let vdiff_fs_v = v(ioutn) - v(ioutp)

  let k = 0
  while k < 16
    let ideal_k = iout_p[0] + k * lsb_a
    let inl_k = abs(iout_p[k] - ideal_k) / lsb_a
    if inl_k > inl_max_lsb
      let inl_max_lsb = inl_k
    end
    if k > 0
      let step_k = iout_p[k] - iout_p[k - 1]
      if step_k <= 0
        let mono_ok = 0
      end
      let dnl_k = abs(step_k - lsb_a) / lsb_a
      if dnl_k > dnl_max_lsb
        let dnl_max_lsb = dnl_k
      end
    end
    let k = k + 1
  end

  echo "=== CS-DAC 4-bit Results (gf180mcu 3.3V tt) ==="
  echo ifs_a=$&ifs_a
  echo lsb_a=$&lsb_a
  echo vdiff_fs_v=$&vdiff_fs_v
  echo inl_max_lsb=$&inl_max_lsb
  echo dnl_max_lsb=$&dnl_max_lsb
  echo mono_ok=$&mono_ok
  echo idd_a=$&idd_a
  echo power_w=$&power_w

  quit
.endc
.end
