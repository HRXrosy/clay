# AnalogGym dual-output CMOS voltage reference intent

This package promotes the existing AnalogGym voltage-reference Clay example
into a self-contained L3 case. The chosen circuit is the MOSFET-only,
dual-output subthreshold reference because it exercises a reference topology
without requiring BJT or proprietary resistor bindings in the current sky130
adapter.

Design intent:

- The PMOS diode/mirror chain establishes the bias current.
- The regular-NMOS and LVT-NMOS devices provide the required threshold-voltage
  difference for a CMOS subthreshold reference.
- `vref1` and `vref2` remain first-class outputs; the case intentionally uses a
  non-opamp port shape.
- The package declares -40 C, 27 C, and 85 C temperature points, matching the
  natural verification axis for a voltage reference.

External design check:

- Analog Devices describes the classic bandgap idea as combining
  temperature-dependent voltage terms so the final reference is less sensitive
  to temperature drift.
- A recent review of voltage references surveys the same design objectives:
  low temperature coefficient, supply sensitivity control, startup behavior,
  and low-power operation.

Current verification boundary:

- Clay parser and compiler coverage is strong: 7 MOS devices, 2 capacitors,
  `nmos_lvt` usage, dual outputs, and temperature sweep intent are checked.
- When simulator/PDK configuration is available, the existing testbench verifies
  finite `vref1` and `vref2` measurements over the declared temperatures.
- This is not yet a production voltage-reference signoff case. A startup
  circuit, tighter TC target, PSRR, PVT, layout, and PEX evidence are still
  open.

References:

- https://www.analog.com/en/resources/technical-articles/selecting-the-optimum-voltage-reference.html
- https://www.mdpi.com/2674-0729/4/1/5
