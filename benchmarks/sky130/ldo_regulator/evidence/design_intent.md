# AnalogGym PMOS-pass LDO regulator intent

This package promotes the existing AnalogGym LDO benchmark into a
self-contained L3 regulator case.

Design intent:

- The NMOS differential pair is the error amplifier input stage; it compares
  the reference input with the feedback voltage.
- The PMOS current mirror forms the active load and establishes the error amp
  output node.
- The large PMOS pass device is the series regulator element. Its gate is driven
  by the error amplifier so the output current tracks the load.
- The internal `nbias` rail removes the AnalogGym external bias pin from the
  public port list.
- The Cfb/Rfb path is the loop-compensation handle and is treated as part of
  the regulator architecture, not as a decorative artifact.

External design check:

- Analog Devices' LDO design guidance emphasizes the same core blocks: error
  amplifier, pass transistor, reference/feedback loop, dropout, transient
  behavior, and stability.
- Texas Instruments' LDO application guidance similarly treats load
  regulation, line regulation, dropout, PSRR, and stability as primary
  regulator metrics.

Current verification boundary:

- Clay parse/compile coverage checks the primitive topology, pass-device size,
  bias tree, and compensation network.
- The source declares line-regulation, load-regulation, line-step, and load-step
  setups through the LDO verifier surface.
- This is not yet an L4/L5 package because measured KPI tables, PVT/PSRR,
  physical layout, PEX, and signoff evidence are not bundled.

References:

- https://www.analog.com/en/resources/analog-dialogue/articles/fundamentals-of-ldo-design-and-applications.html
- https://www.ti.com/lit/an/slyt202/slyt202.pdf
