"""Failure-mode registry: the KNOWN, escalation-tracked blockers of the OTA route sign-off.

Dev-discipline rule 3 ("signoff is not a recorder") + priority 4. The OTA layout is not
yet LVS-clean; rather than let a signoff test pass on that known-broken state (a recorder)
OR go permanently red, the true sign-off gate asserts the CURRENT blocking obligations are
EXACTLY this registered set. That gate then:

* FAILS if a NEW, unexplained blocker appears (a real regression a recorder would miss);
* FAILS if a registered blocker is RESOLVED (stale registry entry -> forces cleanup,
  tightening the gate over time);
* documents, for every known blocker, WHY it blocks and the escalation (the contract fix
  that retires it) -- so no incompleteness is silent.

Each entry maps an obligation id (from :mod:`clay.layout.route.obligations`) to its reason
and the long-term contract fix that will remove it. Update this set whenever a blocker is
fixed (delete the entry) or a new known-incomplete is accepted (add it WITH an escalation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["KnownBlocker", "KNOWN_BLOCKERS", "known_blocker_ids"]


@dataclass(frozen=True)
class KnownBlocker:
    obligation_id: str
    reason: str
    escalation: str          # the contract/artifact/checker fix that retires this blocker
    classification: str      # required | advisory | unsupported | unmeasurable


# The gf180 OTA route's currently-accepted sign-off blockers.
#
# The LVS half is EMPTY as of v0.4.2: the route achieves a real KLayout LVS "Netlists
# match" through the production path. The journey that emptied it:
#   - cap-plate open: deck connect(metal_ncap, l_metal2) (v0.4.0);
#   - resistors as real devices: gf180 _PolyResFacts + build_poly_res + deck ResLvsSpec +
#     reference R card, retiring the collapse-to-wire hack (v0.4.1) -> unmatched 8->0;
#   - full match (v0.4.2): the MIMCap extractor leaves C=0 (area-density not folded into C),
#     so comparing C falsely mismatched every cap -> the cap is matched on TOPOLOGY + class
#     (value fixed by construction); + a 5% relative tolerance on the poly-R value (geometric
#     snap). With those, compare returns "Netlists match".
#
# The DRC half is now CLEAN: the real gf180mcu.drc deck on the canonical OTA as-built
# (top cell ota_routed) reports 0 violations. The via-pad ROUTING family M*.2a is 0
# (same-net via-pad-shoulder bridge + via relocation), and the DEVICE-STAMP / implant
# ARTWORK family (PRES.5/.7 poly-resistor, NW.2b nwell 3.3V, PP.9 pplus, SB.4) -- a
# measured 24 on the same deck before -- is now 0: the build_poly_res implant/SAB/contact
# growth (passive_cell.py) + the same-bulk-net well merge at the emit boundary
# (lvs_gate.emit_labelled_gds). The former oracle.drc blocker is therefore DELETED
# (the gate tightens): real-deck OTA DRC == 0, measured.
#
# Clearing the DRC residual exposed a native<->oracle disagreement (oracle.reconcile):
# native_clean=False (the PathFinder route_success/`converged` convergence flag is
# over-conservative -- the abstract negotiation never reached a zero-overuse fixed point)
# while the independent oracle is signoff-clean (real KLayout LVS match AND real-deck DRC 0,
# M*.2a included). RESOLVED per the registered escalation (obligations.py): oracle.reconcile
# now treats this BENIGN direction (native-dirty, oracle-signoff-clean) as a non-blocking
# ADVISORY -- the independent foundry oracle is the sign-off AUTHORITY (native guides, it
# does not sign off); the DANGEROUS direction (native-clean but oracle-dirty, a native blind
# spot) stays BLOCKING. So NO obligation blocks the OTA: the registry is EMPTY and the OTA
# SIGNS OFF (certify_signed_off issues a real SignedOffLayout). A NEW blocker must be
# registered here WITH an escalation; the DRC + LVS halves are clean and must STAY clean.
KNOWN_BLOCKERS: Dict[str, KnownBlocker] = {}


def known_blocker_ids() -> frozenset[str]:
    return frozenset(KNOWN_BLOCKERS)
