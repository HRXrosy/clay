# GF180 MIM option conductor-domain correction

The repository's foundry `gf180mcu.lvs` derives both `mim_virtual` (MIM A)
and `mimtm_virtual` (MIM B) unconditionally. It then subtracts the latter
from M3, M4, and M5 simultaneously. FuseTop is shared by the mutually
exclusive MIM options. Consequently, an ordinary M5 wire crossing an
A capacitor creates a spurious B virtual plate and the LVS runset cuts
electrically continuous upper-metal wires. The actual GDS is not at fault.

Clay's explicitly recorded runtime copy limits A's virtual plate to M2.
B's virtual plate is deducted only from its selected bottom metal:
`6LM -> M5`, `5LM -> M4`, `4LM -> M3`, `3LM -> M2`. The foundry
`METAL_TOP` option selects metal thickness, not this connectivity order.
Neither the source foundry runset nor the input GDS is modified. Device
extraction, parameter comparison, shorts, and opens retain their normal
checks. The existing separately documented MIM-A bottom-plate bridge and
passive parameter policies are unchanged by this correction.

The official `schematic_simplify=true` switch also applies the same
series/parallel topology reduction to the schematic that the foundry
runset already applies to the layout by default. Without this symmetric
normalization, two physical series resistors combine on only one side.
No source device is resized or removed from simulation for this purpose.

`run_gf180_lvs` records the original and executed deck SHA-256 hashes,
all runtime switches, exact command, source/reference hashes, and verdict
in `*.deck.json`. With this correction selected, the exact executed deck
is retained under its content hash beside the LVS database. Explicit
`mim_option_scope_fix=False` remains available for before/after diagnosis.

Evidence includes nine minimal real-GDS KLayout layer-domain regressions
in `tests/test_gf180_mim_option_scope.py`, covering both options, selected
stack levels, 6K/9K, no selected MIM, and physical short/open controls.
The unchanged September OTA GF180 GDS additionally has a real foundry
LVS mismatch before correction and a match after correction plus symmetric
normalization. Separate copies with an injected real M1 short or removed
M3-to-M4 contact cuts must still fail the full foundry LVS comparison. These
receipts are in the paper's `ota/gf_mim_scope_oracle/` evidence directory.
