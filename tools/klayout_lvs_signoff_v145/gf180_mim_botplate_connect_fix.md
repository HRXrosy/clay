# gf180mcu LVS deck — MIM bottom-plate connectivity transcription-bug fix

## What this fixes

The open-source GlobalFoundries `gf180mcu.lvs` KLayout LVS deck (the 1923-line
Ruby runset shipped in the gf180mcu open PDK) is **missing one `connect` line**
in its MIM-Option-A connectivity block, which severs an embedded MIM capacitor's
**bottom plate** from any routed BEOL net. This is a CLAY-SIDE corrected-deck
injection — the foundry deck under `pdks/` is **never edited in place**.

## Root cause (deck transcription bug, confirmed against the sky130 sibling deck)

The gf180 MIM-Option-A connectivity (deck lines 935-936, 1076-1077, 1259-1260):

```ruby
mim_virtual = fusetop.sized(1.06.um).and(metal2.interacting(fusetop))  # bottom plate
metal2_ncap = metal2_con.not(mim_virtual)                              # ROUTING metal2
...
connect(via1        ,   metal2_ncap)     # routing rises via1 -> metal2_ncap
connect(metal2_ncap ,          via2)     # ... and metal2_ncap -> via2
...
connect(metal2, mim_virtual)             # MIM bottom plate joins RAW metal2 only
```

`metal2_ncap` (the routable Metal2 the router lands a wire on) is the COMPLEMENT
`metal2_con.not(mim_virtual)`. The deck connects the bottom plate (`mim_virtual`)
to **raw `metal2`**, but raw `metal2` is a *different* connectivity-graph node
from the derived routing layer `metal2_ncap`. Nothing bridges `metal2_ncap` to
`mim_virtual`, so a routing Metal2 wire that physically abuts/overlaps the bottom
plate extracts on a **separate net** from the plate. The bottom plate is
electrically stranded; only the TOP plate (`fuse_cap`, reached through
`connect(fuse_cap, via2)`) is routable.

This is a transcription omission, not an intended technique. The **sky130** MIM
deck (same KLayout-deck authoring lineage, identical `*_ncap = *_con.not(cap)`
structure) DOES include the bridging line, in its own MIMCAP connections block:

```ruby
# sky130.lvs, "MIMCAP CONNECTIONS"
connect(capm      , via3    )
connect(met3_ncap , met3_con)   # <-- routing tail rejoined to the bottom-plate metal
```

gf180 has the exact parallel structure (`metal2_ncap = metal2_con.not(mim_virtual)`,
bottom plate on Metal2) but DROPPED the equivalent bridge. A real gf180 silicon
MIM cap obviously has a connectable bottom plate; the open-source deck just fails
to extract it as connected.

## The fix (one added connect line, gf180-foundry-faithful)

Mirror the sky130 bridge for gf180's bottom-plate layer name:

```ruby
connect(metal2_ncap, mim_virtual)   # CLAY transcription-bug fix: rejoin routing
                                    # Metal2 to the MIM bottom plate (mim_virtual),
                                    # matching real foundry connectivity + the
                                    # sky130 sibling deck's met3_ncap<->met3_con.
```

This is injected ONLY into a Clay-maintained sibling copy of the deck at run time
(alongside the existing Windows `pmap` shim), gated on MIM-Option-A being active.
It changes connectivity ONLY for the MIM bottom plate; every device-extraction,
DRC, and MOS/resistor comparison line is byte-identical to the foundry deck, so a
non-MIM cell's verdict is unchanged. The injection is idempotent (skipped if the
line is already present, e.g. a future foundry deck that fixes this upstream).

## Second correction: the parser-forced-constant MIM `C` compare

The deck's reference SPICE reader `SubcircuitModelsReader.parse_element` (deck
lines 153-161) UNCONDITIONALLY appends `C=2e-16` to every reference capacitor
card before parsing:

```ruby
def parse_element(s, element)
  if element == "C"
    super(s + " C=2e-16", element)   # <-- forces schematic cap C = 2e-16 F, always
```

So the schematic-side MIM cap's `C` is ALWAYS read as 2e-16 F regardless of the
card's declared value. The layout side extracts the real
`C = density * FuseTop_area` (1e-13 F for a 100 fF cap), so the deck's
`tolerance("mim_1p0fF", "C", 0.25)` can never be satisfied (1e-13 vs 2e-16 is
~500x). This is independent of the bottom-plate connect: even with topology fully
matched, the cap pairs only with a parameter warning and the netlist verdict is
MISMATCH.

The MIM capacitance is fully determined by the plate AREA, which the deck ALSO
compares as the MIMCap `A` parameter at ZERO tolerance (an exact geometric
check). So the forced-constant `C` compare is both broken (constant 2e-16) and
redundant (A already pins the capacitance). The fix WIDENS only the MIM `C`
tolerance (`tolerance("mim_*", "C", :relative => 1e9)`) so the cap pairs on its
exact A/P, mirroring how the deck's own `MosCap` class relies on geometry (A/P)
rather than `C`. A genuinely wrong cap is still caught by the zero-tolerance `A`
mismatch.

(Disabling `C` via `enable_parameter("C", false)` instead makes the two sides'
device-class parameter signatures diverge and the compare fails outright;
widening the tolerance keeps the signatures identical and is the sound, minimal
change. Verified empirically against `rc_series`.)

## rc_series proof (verified)

With both corrections (and the `vss` subckt port + `mim_1p0fF` model-token on the
reference cap card), `CLAY_GENERIC_EMIT_DEVICES=1 clay flow run rc_series` reaches
DRC CLEAN + LVS MATCH. Extracted: 3 devices (`mim_1p0fF` x1, `ppolyf_u` x2), 7
nets, both cap plates on the correct routed nets (bottom plate == R1 output net,
top plate == R2 input net).
