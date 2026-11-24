"""The oracle-bound DONE-type: make "signed off without LVS" unrepresentable (S2).

The repeated failure the consolidation plan names is a done-claim DECOUPLED from the
oracle -- reporting a router union-find "clean" or a DRC pass as if it were sign-off.
This module makes that class of mistake a TYPE error rather than a discipline lapse:

* :class:`SignedOffLayout` is a frozen value whose constructor is gated by a
  module-private ``_MINT`` sentinel. Its ``__post_init__`` REFUSES to build unless
  ALL of these hold, each verified by an INDEPENDENT KLayout oracle (NOT the router's
  own union-find self-report):
    1. the un-forgeable ``_MINT`` token is present (only this module mints it, and
       only via :func:`certify_signed_off`);
    2. ``lvs.matched`` is True (KLayout printed ``Netlists match`` AND the lvsdb
       cross-reference paired cleanly -- see :class:`clay.layout.route.lvs_gate.LvsVerdict`);
    3. the LVS oracle ACTUALLY RAN (a real ``.lvsdb`` was produced -- ``lvs.ran`` and a
       non-empty ``lvs.lvsdb_path``), so a "match" can never be a vacuous default;
    4. the real foundry DRC deck ACTUALLY RAN and reported CLEAN (``drc.ran`` and
       ``drc.clean`` -- see :class:`clay.layout.route.drc_gate.DrcVerdict`). LVS proves
       CONNECTIVITY, never geometric LEGALITY: a layout can match LVS while the deck
       fires spacing/enclosure (the M*.2a via-pad family). Sign-off is LVS-match AND
       DRC-clean, so a missing or dirty DRC verdict is unconstructible as signed off;
    5. the full obligation set carries NO remaining blocker
       (:func:`clay.signoff_verdict.is_blocking` over the capability + native +
       oracle LVS/DRC/reconcile verdicts).
  Constructing one from any other module -- without the mint -- raises ``TypeError``.
  "Done without LVS-AND-DRC" is therefore literally unconstructible.

* :class:`UnsignedLayout` is the EXPLICIT typed NOT-DONE state (the honest gray of
  the five-state model). It carries the SAME evidence (the LVS verdict, the
  obligation map, the blocker set, a plain-language reason) so a caller can see
  exactly why sign-off was withheld. It is freely constructible -- withholding a
  claim is always allowed; ASSERTING one is what is gated.

* :func:`certify_signed_off` is the SOLE factory. It runs the existing independent
  oracle (:func:`clay.layout.route.lvs_gate.lvs_gate` on the live ``{"_bundle": ...}``)
  plus the full obligation set
  (:func:`clay.layout.route.obligations.route_signoff_obligations`) and returns
  ``SignedOffLayout`` ONLY when matched + ran + no-blocker, else ``UnsignedLayout``.
  It NEVER raises on a non-match (or on an infrastructure failure such as KLayout
  not being installed): it returns the typed union -- no-silent-pass WITHOUT a
  surprise throw.

This sits ABOVE :func:`clay.layout.spec.sign_verdict.compute_sign_verdict`'s spec-gate
string as a strictly stronger claim verified by the independent KLayout LVS oracle. It
changes NO production output: the certification surface is opt-in
(``clay.api.layout_and_signoff(certify=True)``, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Union

from clay.signoff_verdict import CheckerVerdict, is_blocking

__all__ = ["SignedOffLayout", "UnsignedLayout", "certify_signed_off"]

# The un-forgeable mint. Module-private, deliberately absent from ``__all__``: only
# :func:`certify_signed_off` (in this module) holds it, so no foreign module can mint a
# ``SignedOffLayout``. Identity-checked (``is``) in ``__post_init__`` -- a value that is
# merely equal cannot pass.
_MINT = object()


def _blocking_ids(obligations: Mapping[str, CheckerVerdict]) -> List[str]:
    """The sorted obligation ids that BLOCK sign-off (the single ``is_blocking`` policy)."""
    return sorted(oid for oid, v in (obligations or {}).items() if is_blocking(v))


@dataclass(frozen=True)
class SignedOffLayout:
    """A layout the INDEPENDENT KLayout LVS oracle certified as signed off.

    Obtainable ONLY through :func:`certify_signed_off` (the mint gate). Frozen, so a
    certified result cannot be mutated into a different claim after the fact. The
    presence of an instance is itself the proof token: it cannot exist unless the
    oracle matched, ran, and no obligation blocks.
    """

    tech_id: str
    lvs: Any  # the clay.layout.route.lvs_gate.LvsVerdict that certified it
    drc: Any = None  # the clay.layout.route.drc_gate.DrcVerdict that certified it
    obligations: Mapping[str, CheckerVerdict] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    # the mint gate: identity-checked, never stored in the public repr / equality.
    _mint: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # (1) the un-forgeable mint MUST be present -- foreign construction is a TypeError.
        if self._mint is not _MINT:
            raise TypeError(
                "SignedOffLayout is oracle-minted: it cannot be constructed directly. "
                "Use clay.layout.route.signed_off.certify_signed_off(...) -- the sole "
                "factory that runs the independent KLayout LVS + DRC oracles and mints "
                "the proof token only on a real match. (A done-claim decoupled from the "
                "oracle is exactly the failure this type makes unrepresentable.)"
            )
        # (2) the oracle's verdict MUST be a real match.
        if not getattr(self.lvs, "matched", False):
            raise TypeError(
                "SignedOffLayout requires lvs.matched is True (KLayout printed "
                "'Netlists match' AND the lvsdb cross-reference paired cleanly); got "
                f"matched={getattr(self.lvs, 'matched', None)!r}. Withheld sign-off is "
                "an UnsignedLayout, never a SignedOffLayout."
            )
        # (3) the oracle MUST have actually run (a real .lvsdb), so 'matched' is never vacuous.
        if not getattr(self.lvs, "ran", False) or not getattr(self.lvs, "lvsdb_path", None):
            raise TypeError(
                "SignedOffLayout requires the KLayout LVS oracle to have RUN -- a "
                "produced .lvsdb cross-reference (lvs.ran True and a non-empty "
                f"lvs.lvsdb_path); got ran={getattr(self.lvs, 'ran', None)!r} "
                f"lvsdb_path={getattr(self.lvs, 'lvsdb_path', None)!r}."
            )
        # (4) the real foundry DRC deck MUST have RUN and reported CLEAN. LVS proves
        # CONNECTIVITY; it does not prove geometric LEGALITY -- a layout can match LVS
        # while the deck fires spacing/enclosure (the M*.2a via-pad family). Sign-off is
        # LVS-match AND DRC-clean, so a missing/dirty DRC verdict is an UnsignedLayout.
        if not getattr(self.drc, "ran", False):
            raise TypeError(
                "SignedOffLayout requires the real foundry DRC deck to have RUN "
                f"(drc.ran True); got drc={self.drc!r}. DRC is the geometric-legality "
                "half of sign-off; without it sign-off is withheld (UnsignedLayout)."
            )
        if not getattr(self.drc, "clean", False):
            raise TypeError(
                "SignedOffLayout requires the real foundry DRC deck to be CLEAN "
                f"(0 violations); got total={getattr(self.drc, 'total', None)!r} "
                f"counts={getattr(self.drc, 'counts_by_rule', None)!r}. A DRC-dirty "
                "layout is an UnsignedLayout, never a SignedOffLayout."
            )
        # (5) NO obligation may still block (capability + native + oracle all clear).
        blockers = _blocking_ids(self.obligations)
        if blockers:
            raise TypeError(
                "SignedOffLayout requires an EMPTY blocker set; these obligations "
                f"still block sign-off: {blockers}. A blocked layout is an "
                "UnsignedLayout."
            )

    @property
    def signed_off(self) -> bool:
        """Always True for a constructed instance (it could not exist otherwise).
        Present so callers can branch on ``result.signed_off`` uniformly across the union."""
        return True


@dataclass(frozen=True)
class UnsignedLayout:
    """The explicit typed NOT-DONE state -- sign-off was WITHHELD, and here is why.

    Carries the same evidence a :class:`SignedOffLayout` would (the LVS verdict if the
    oracle ran, the obligation map, the blocker set, a plain-language ``reason``). This
    is the honest gray of the five-state model: not a silent pass, not a thrown
    exception -- a value that says 'not signed off' and names the cause. Freely
    constructible: withholding a claim is always allowed."""

    tech_id: str
    reason: str
    lvs: Optional[Any] = None  # the LvsVerdict if the oracle ran, else None
    obligations: Mapping[str, CheckerVerdict] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def signed_off(self) -> bool:
        return False

    @property
    def blockers(self) -> List[str]:
        """The obligation ids that block sign-off (empty when the cause is a non-match
        or an oracle that could not run)."""
        return _blocking_ids(self.obligations)


SignoffResult = Union[SignedOffLayout, UnsignedLayout]


def _unsigned_reason(matched: bool, ran: bool, blockers: List[str],
                     drc_ran: bool = True, drc_clean: bool = True,
                     drc_total: int = 0) -> str:
    parts: List[str] = []
    if not ran:
        parts.append("the KLayout LVS oracle did not run (no .lvsdb produced)")
    elif not matched:
        parts.append("KLayout LVS did not match (Netlists don't match / unpaired xref)")
    if not drc_ran:
        parts.append("the real foundry DRC deck did not run (DRC unmeasurable)")
    elif not drc_clean:
        parts.append(f"real foundry DRC not clean ({drc_total} violations)")
    if blockers:
        parts.append(f"blocking obligations remain: {blockers}")
    if not parts:
        parts.append("sign-off withheld")
    return "; ".join(parts)


def _certify_route_cell(
    res: Any,
    cell_def: Any,
    *,
    tech_id: str,
    run_oracle: bool = True,
    full_extraction: bool = False,
    exact_artifact_evidence: Any = None,
) -> SignoffResult:
    """Mint (or withhold) the proof token for a GENERAL ``route_cell`` result.

    The design-independent spine: the obligation set is built by the generalized
    :func:`clay.layout.route.obligations.route_signoff_obligations` (route_cell
    injection), and the mint verdict runs the INDEPENDENT KLayout LVS via the general
    :func:`clay.layout.route.lvs_gate.lvs_gate_route_cell` + a real-deck DRC on the
    route_cell emit -- everything derived from the single ``cell_def`` source, no
    per-circuit knob. Mirrors the route_amp :func:`certify_signed_off` contract: mints
    ``SignedOffLayout`` ONLY on matched AND ran AND drc-clean AND no-blocker; otherwise
    a typed ``UnsignedLayout`` carrying the cause; NEVER raises on an infra gap."""
    from clay.layout.route.lvs_gate import _cell_def_reference_intent, lvs_gate_route_cell
    from clay.layout.route.obligations import (_route_cell_drc_verdict,
                                               route_signoff_obligations)

    # (a) the full obligation set via the generalized engine (capability + native +
    # route_cell oracle). Infra failure here is honest gray, never a raise.
    try:
        if exact_artifact_evidence is not None:
            exact_artifact_evidence.validate_for_route_cell_result(
                res, cell_def
            )
            exact_artifact_evidence.validate_reference_equivalence(cell_def)
        ref_intent = _cell_def_reference_intent(cell_def)
        obligations = route_signoff_obligations(
            tech_id, result=res, intent=ref_intent, cell_def=cell_def,
            run_oracle=run_oracle,
            require_exact_route_evidence=exact_artifact_evidence is not None,
            exact_artifact_evidence=exact_artifact_evidence,
        )
    except Exception as exc:  # pragma: no cover - defensive (build failure)
        return UnsignedLayout(
            tech_id=tech_id,
            reason=f"route_cell obligation set could not be built for "
                   f"tech_id={tech_id!r}: {exc}",
            lvs=None, obligations={}, evidence={"error": str(exc)},
        )
    blockers = _blocking_ids(obligations)

    # (b) Consume a precomputed exact-artifact verdict when present.  This is
    # the production signoff path: the emitter, LVS, DRC and exact-route
    # checkers have already been joined to one immutable GDS, so re-emitting or
    # rerunning an oracle here would create a second, unbound artifact.  The
    # bundle validates its artifact/type bindings in route_signoff_obligations.
    if exact_artifact_evidence is not None:
        lvs = exact_artifact_evidence.lvs_verdict
        if lvs is None:
            return UnsignedLayout(
                tech_id=tech_id,
                reason="the exact artifact bundle has no precomputed LVS verdict",
                lvs=None,
                obligations=obligations,
                evidence={"blocking": blockers, "path": "route_cell_exact"},
            )
    else:
        try:
            lvs = lvs_gate_route_cell(
                res,
                cell_def,
                tech_id=tech_id,
                full_extraction=full_extraction,
            )
        except Exception as exc:
            return UnsignedLayout(
                tech_id=tech_id,
                reason=f"the KLayout LVS oracle could not run: {exc}",
                lvs=None, obligations=obligations,
                evidence={"blocking": blockers, "error": str(exc)},
            )
    ran = bool(getattr(lvs, "ran", False) and getattr(lvs, "lvsdb_path", None))
    matched = bool(getattr(lvs, "matched", False))

    # (c) the real foundry DRC deck on the SAME route_cell emit (the geometric-legality
    # half of sign-off). Infra failure -> ran=False (withheld, never a silent pass).
    if exact_artifact_evidence is not None:
        drc = exact_artifact_evidence.drc_verdict
        if drc is None:
            from clay.layout.route.drc_gate import DrcVerdict

            drc = DrcVerdict(
                ran=False,
                clean=False,
                total=0,
                reason="the exact artifact bundle has no precomputed DRC verdict",
            )
    else:
        try:
            drc = _route_cell_drc_verdict(res, cell_def, tech_id)
        except Exception as exc:  # pragma: no cover - defensive (treated as unmeasurable)
            from clay.layout.route.drc_gate import DrcVerdict

            drc = DrcVerdict(ran=False, clean=False, total=0,
                             reason=f"DRC gate raised: {exc}")
    drc_ran = bool(getattr(drc, "ran", False))
    drc_clean = bool(getattr(drc, "clean", False))
    drc_total = int(getattr(drc, "total", 0))

    evidence = {
        "matched": matched, "ran": ran, "drc_ran": drc_ran, "drc_clean": drc_clean,
        "drc_total": drc_total, "drc_counts": getattr(drc, "counts_by_rule", {}),
        "blocking": blockers,
        "lvs_summary": lvs.summary() if hasattr(lvs, "summary") else "",
        "lvs_verbatim": getattr(lvs, "verbatim", ""),
        "lvsdb_path": getattr(lvs, "lvsdb_path", None),
        "path": "route_cell_exact" if exact_artifact_evidence is not None else "route_cell",
    }

    if matched and ran and drc_ran and drc_clean and not blockers:
        return SignedOffLayout(
            tech_id=tech_id, lvs=lvs, drc=drc, obligations=obligations,
            evidence=evidence, _mint=_MINT,
        )
    return UnsignedLayout(
        tech_id=tech_id,
        reason=_unsigned_reason(matched, ran, blockers, drc_ran, drc_clean, drc_total),
        lvs=lvs, obligations=obligations, evidence=evidence,
    )


def certify_signed_off(
    bundle: Any,
    intent: Any,
    *,
    tech_id: str,
    full_extraction: bool = False,
    run_oracle: bool = True,
    exact_artifact_evidence: Any = None,
) -> SignoffResult:
    """Run the independent oracle + full obligation set and return the typed union.

    Parameters
    ----------
    bundle:
        The live :class:`clay.layout.route.as_built.AsBuiltBundle` to certify; it is
        handed to :func:`clay.layout.route.lvs_gate.lvs_gate` as the live
        ``{"_bundle": bundle}`` so the SAME as-built geometry is what the oracle
        extracts. ``None`` -> the gate re-derives the canonical leakage-free route
        from ``intent`` + ``tech_id`` (the ``_materialize`` None path; the checker
        does not depend on the checked carrying its internals along).
    intent:
        The ``ExtractedIntent`` whose schematic the reference netlist is emitted from.
        ``None`` -> the canonical ``build_ota_intent()`` is used (mirrors
        :func:`clay.layout.route.obligations.route_signoff_obligations`).
    tech_id:
        The target PDK (only a PDK with a synthesized LVS deck can be certified).
    full_extraction:
        Forwarded to ``lvs_gate`` -- select the PDK's full device-extraction overlay
        where the default deck is bring-up-bounded.
    run_oracle:
        Forwarded to ``route_signoff_obligations`` -- when False the oracle
        obligations are PENDING (and therefore block), so certification is withheld.
    exact_artifact_evidence:
        Optional precomputed :class:`ExactArtifactEvidenceBundle` for a canonical
        ``RouteCellResult``.  Its already-bound LVS/DRC/via/layer reports are
        consumed without a second emit or oracle run.  The exact obligation set
        still gates the mint; a missing or inconsistent report withholds sign-off.

    Returns
    -------
    SignedOffLayout | UnsignedLayout
        ``SignedOffLayout`` ONLY when the oracle matched AND ran AND no obligation
        blocks; otherwise ``UnsignedLayout`` carrying the cause. NEVER raises on a
        non-match or on an infrastructure failure (KLayout absent, no deck for
        ``tech_id``, etc.): the failure becomes a typed ``UnsignedLayout``, not an
        exception.

    ROUTE_CELL input (spine unification, S1.1): when ``bundle`` is a
    :class:`clay.layout.route.route_cell.RouteCellResult` (and ``intent`` is its
    originating ``cell_def``), the oracle is routed through the GENERAL
    ``lvs_gate_route_cell`` + a real-deck DRC on the route_cell emit -- the single
    spine, no OTA-specialised ``_materialize``. The dict / ``AsBuiltBundle`` route_amp
    path below is byte-stable (a route_cell result never reaches it).
    """
    from clay.layout.route.route_cell import BringupDiagnosticRouteResult

    if type(bundle) is BringupDiagnosticRouteResult:
        return UnsignedLayout(
            tech_id=tech_id,
            reason=(
                "NON_SIGNOFF_BRINGUP_DIAGNOSTIC cannot enter the signoff mint"
            ),
            lvs=None,
            obligations={},
            evidence={
                "path": "bringup_diagnostic",
                "diagnostic_digest": bundle.diagnostic_summary_digest,
            },
        )
    # ROUTE_CELL mint path: dispatch a (RouteCellResult, cell_def) input to the general
    # spine. The route_amp dict / AsBuiltBundle path (below) is untouched -- a dict is
    # never a RouteCellResult, so it never enters here.
    if not isinstance(bundle, dict) and bundle is not None:
        from clay.layout.route.obligations import _is_route_cell_result

        if _is_route_cell_result(bundle):
            return _certify_route_cell(
                bundle,
                intent,
                tech_id=tech_id,
                run_oracle=run_oracle,
                full_extraction=full_extraction,
                exact_artifact_evidence=exact_artifact_evidence,
            )

    if exact_artifact_evidence is not None:
        return UnsignedLayout(
            tech_id=tech_id,
            reason=(
                "exact_artifact_evidence currently requires a canonical "
                "RouteCellResult + cell_def input"
            ),
            lvs=None,
            obligations={},
            evidence={"path": "unsupported_exact_ingress"},
        )

    # (a) the full obligation set (capability + native + KLayout oracle). An
    # infrastructure failure here is honest gray, never a raise.
    try:
        from clay.layout.route.obligations import route_signoff_obligations

        obligations = route_signoff_obligations(tech_id, run_oracle=run_oracle)
    except Exception as exc:  # pragma: no cover - defensive (unknown tech / build failure)
        return UnsignedLayout(
            tech_id=tech_id,
            reason=f"obligation set could not be built for tech_id={tech_id!r}: {exc}",
            lvs=None,
            obligations={},
            evidence={"error": str(exc)},
        )
    blockers = _blocking_ids(obligations)

    # (b) the independent KLayout LVS oracle on THIS bundle (or canonical re-derive).
    if intent is None:
        try:
            from clay.layout.extract.migrate import build_ota_intent

            intent = build_ota_intent()
        except Exception as exc:  # pragma: no cover - defensive
            return UnsignedLayout(
                tech_id=tech_id,
                reason=f"could not build canonical intent for tech_id={tech_id!r}: {exc}",
                lvs=None,
                obligations=obligations,
                evidence={"blocking": blockers, "error": str(exc)},
            )

    # accept EITHER a raw AsBuiltBundle OR the route_*_leakfree result dict that carries
    # it under "_bundle" (a natural footgun: route_amp_leakfree returns the dict). Unwrap
    # so both forms certify the same canonical geometry the lvs_gate/obligations read.
    if isinstance(bundle, dict) and "_bundle" in bundle:
        bundle = bundle["_bundle"]
    gate_input = {"_bundle": bundle} if bundle is not None else None
    try:
        from clay.layout.route.lvs_gate import lvs_gate

        lvs = lvs_gate(gate_input, intent, tech_id=tech_id, full_extraction=full_extraction)
    except Exception as exc:
        # KLayout not installed, no deck for tech_id, emit failure, ... -> typed gray.
        return UnsignedLayout(
            tech_id=tech_id,
            reason=f"the KLayout LVS oracle could not run: {exc}",
            lvs=None,
            obligations=obligations,
            evidence={"blocking": blockers, "error": str(exc)},
        )

    ran = bool(getattr(lvs, "ran", False) and getattr(lvs, "lvsdb_path", None))
    matched = bool(getattr(lvs, "matched", False))

    # (c) the real foundry DRC deck on the SAME bundle -- the geometric-legality half
    # of sign-off (LVS proves connectivity, not legality). Infra failure -> ran=False.
    try:
        from clay.layout.route.drc_gate import drc_gate

        drc = drc_gate(gate_input, intent, tech_id=tech_id)
    except Exception as exc:  # pragma: no cover - defensive (treated as unmeasurable)
        from clay.layout.route.drc_gate import DrcVerdict

        drc = DrcVerdict(ran=False, clean=False, total=0,
                         reason=f"DRC gate raised: {exc}")
    drc_ran = bool(getattr(drc, "ran", False))
    drc_clean = bool(getattr(drc, "clean", False))
    drc_total = int(getattr(drc, "total", 0))

    evidence = {
        "matched": matched,
        "ran": ran,
        "drc_ran": drc_ran,
        "drc_clean": drc_clean,
        "drc_total": drc_total,
        "drc_counts": getattr(drc, "counts_by_rule", {}),
        "blocking": blockers,
        "lvs_summary": lvs.summary() if hasattr(lvs, "summary") else "",
        "lvs_verbatim": getattr(lvs, "verbatim", ""),
        "lvsdb_path": getattr(lvs, "lvsdb_path", None),
    }

    if matched and ran and drc_ran and drc_clean and not blockers:
        # mint the proof token -- the ONLY place _MINT is ever passed.
        return SignedOffLayout(
            tech_id=tech_id,
            lvs=lvs,
            drc=drc,
            obligations=obligations,
            evidence=evidence,
            _mint=_MINT,
        )

    return UnsignedLayout(
        tech_id=tech_id,
        reason=_unsigned_reason(matched, ran, blockers, drc_ran, drc_clean, drc_total),
        lvs=lvs,
        obligations=obligations,
        evidence=evidence,
    )
