"""The ROUND-TRIP HARNESS: ``ExtractedIntent`` -> Clay -> GDS -> fidelity.

This closes the consistent-intent-extractor loop:

    reference -> extract_intent (ExtractedIntent)
              -> emit_clay        (a parseable .clay)
              -> generic place/route (a layout.gds)
              -> fidelity_score vs the reference GDS.

The round-trip score IS the consistency / completeness metric: an extraction that
is *consistent* (captured the real relations) AND *vocabulary-complete* (every
captured relation has an implemented primitive) round-trips to fidelity -> 1.0.
The shortfall localizes WHAT is missing -- a missing primitive (relation was
extracted but only emitted as a comment), an extraction miss (the reference has a
paradigm feature our intent never captured), or a realization gap (the primitive
exists + was emitted, but the placer did not realize it geometrically).

NO Docker / DRC / LVS: only deterministic place + route + emit (the same path the
``test_gds_obligation_soundness`` test uses) plus the geometric fidelity oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from clay.layout.extract.emit_clay import emit_clay, validate_parses


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
@dataclass
class RoundtripReport:
    """The round-trip outcome: fidelity + breakdown + gap localization."""

    cell: str
    fidelity: float
    sub_scores: Dict[str, float]
    breakdown: str
    gap_localization: Dict[str, str]  # fidelity feature -> classification + note
    emitted_clay_path: str
    our_gds_path: str
    reference_gds_path: Optional[str]
    parsed_ok: bool
    parse_error: Optional[str] = None
    emitted_relations: List[str] = field(default_factory=list)
    commented_relations: List[str] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"ROUND-TRIP REPORT  cell={self.cell}",
            "=" * 56,
            f"  parsed_ok    : {self.parsed_ok}"
            + (f"  ({self.parse_error})" if self.parse_error else ""),
            f"  fidelity     : {self.fidelity:.3f}",
            f"  our GDS      : {self.our_gds_path}",
            f"  reference    : {self.reference_gds_path}",
            "",
            "  emitted (active primitive):",
        ]
        for r in self.emitted_relations:
            lines.append(f"    + {r}")
        lines.append("  commented (NOT-YET-implemented primitive):")
        for r in self.commented_relations:
            lines.append(f"    ~ {r}")
        lines.append("")
        lines.append("  gap localization (per fidelity feature):")
        for k, v in self.gap_localization.items():
            lines.append(f"    {k:14s} {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers: classify what the intent emitted vs commented.
# --------------------------------------------------------------------------- #
def _emitted_and_commented(intent: Any) -> "tuple[List[str], List[str]]":
    """Split the intent's relations into ACTIVE-emitted vs COMMENTED (roadmap)."""

    def _get(o, n, d):
        v = getattr(o, n, None)
        return d if v is None else v

    emitted: List[str] = []
    if _get(intent, "matched_pairs", []):
        emitted.append("@matching (matched_pairs)")
    if _get(intent, "interdigitated_pairs", []):
        emitted.append("@interdigitated (interdigitated_pairs)")
    if [g for g in _get(intent, "equal_size_groups", []) if len(set(g)) >= 2]:
        emitted.append("@equal_size (equal_size_groups)")
    if _get(intent, "bands", {}):
        emitted.append("@place_group (bands)")

    commented: List[str] = []
    if _get(intent, "orders", []):
        commented.append("@order (orders)")
    if _get(intent, "abuts", []):
        commented.append("@abut (abuts)")
    if _get(intent, "aligns", []):
        commented.append("@align (aligns)")
    if _get(intent, "symmetry_axis", None) is not None:
        commented.append("@symmetry_axis (symmetry_axis)")
    if _get(intent, "spacings", []):
        commented.append("@spacing (spacings)")
    if _get(intent, "net_layers", {}):
        commented.append("@net_layer (net_layers)")
    return emitted, commented


def _classify_gap(
    feature: str,
    ref_sub: float,
    our_sub: float,
    emitted: List[str],
    commented: List[str],
    notes: Dict[str, str],
    ref_classes: "tuple[str, ...]" = (),
    our_classes: "tuple[str, ...]" = (),
) -> str:
    """Classify a per-feature shortfall into one of three buckets:

      * ``missing_primitive``  -- the feature maps onto a relation that was only
        emitted as a COMMENT (no implemented primitive yet);
      * ``extraction_miss``    -- the reference exhibits the feature but the intent
        carried NO relation that targets it (we never extracted it);
      * ``realization_gap``    -- the relation WAS emitted as active syntax, yet the
        placer did not reproduce the reference feature geometrically.

    A perfect sub-score (>= 0.999) is reported as ``ok``.
    """
    if our_sub >= 0.999:
        return f"ok (score={our_sub:.3f})"

    note = notes.get(feature, "")

    # Map the fidelity feature -> the intent relation(s) that drive it.
    #   tier_structure  <- bands (@place_group) [implemented]
    #   cap_paradigm    <- equal_size_groups (@equal_size) + passive bands [implemented]
    #   multifinger     <- device M->nf (always emitted) [implemented]
    #   symmetry        <- matched_pairs / symmetry_axis
    #   aspect          <- (no direct primitive; floorplan realization)
    has_symmetry_active = any("matching" in e or "interdigitated" in e for e in emitted)
    symmetry_only_commented = any("symmetry_axis" in c for c in commented)

    if feature == "symmetry":
        if has_symmetry_active:
            return f"realization_gap (score={our_sub:.3f}; matched pairs emitted but band not mirrored) -- {note}"
        if symmetry_only_commented:
            return f"missing_primitive (score={our_sub:.3f}; only @symmetry_axis comment, no active pair) -- {note}"
        return f"extraction_miss (score={our_sub:.3f}; no matched/diff pair extracted) -- {note}"

    if feature == "tier_structure":
        # A class the REFERENCE has but OURS lacks (e.g. a resistor tier we never
        # extracted) is an extraction miss, not a placer realization gap.
        missing_classes = [c for c in ref_classes if c not in our_classes]
        if missing_classes:
            if any("place_group" in e for e in emitted):
                return (
                    f"extraction_miss (score={our_sub:.3f}; reference tiers "
                    f"{missing_classes} absent from intent; bands emitted for the "
                    f"rest) -- {note}"
                )
            return f"extraction_miss (score={our_sub:.3f}; tiers {missing_classes} not extracted) -- {note}"
        if any("place_group" in e for e in emitted):
            return f"realization_gap (score={our_sub:.3f}; bands emitted, tier order diverges) -- {note}"
        return f"extraction_miss (score={our_sub:.3f}; no bands extracted) -- {note}"

    if feature == "cap_paradigm":
        if any("equal_size" in e for e in emitted):
            return f"realization_gap (score={our_sub:.3f}; equal_size emitted, cap paradigm diverges) -- {note}"
        return f"extraction_miss (score={our_sub:.3f}; no cap equal_size extracted) -- {note}"

    if feature == "multifinger":
        return f"realization_gap (score={our_sub:.3f}; M->nf emitted, finger count diverges) -- {note}"

    if feature == "aspect":
        # aspect has no direct declarative primitive -> a floorplan realization gap
        return f"realization_gap (score={our_sub:.3f}; no aspect primitive; floorplan realization) -- {note}"

    return f"realization_gap (score={our_sub:.3f}) -- {note}"


# --------------------------------------------------------------------------- #
# Build a GDS from an emitted .clay via the generic layout step.
# --------------------------------------------------------------------------- #
def _build_gds(clay_src: str, cell: str, work_dir: Path) -> Path:
    """Write ``clay_src`` into a temp case dir and run the GENERIC layout step,
    returning the emitted ``layout.gds`` path. Deterministic place+route+emit only
    (no Docker / DRC / LVS) -- mirrors ``test_gds_obligation_soundness``.
    """
    from clay.flow.spec import FlowContext, compile_flow_spec
    from clay.flow.step import LayoutStep, gds_path
    from clay.parser import parse_clay_file

    case_dir = work_dir / "case"
    src_dir = case_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "design.clay").write_text(clay_src, encoding="utf-8")

    # Minimal case.yaml (the flow reads metadata best-effort; conventions fill in).
    (case_dir / "case.yaml").write_text(
        "schema: clay.case/v1\n"
        f"id: case_extract_roundtrip_{cell.lower()}\n"
        f"title: round-trip of an ExtractedIntent ({cell})\n"
        "category: opamp\n"
        "source:\n"
        "  clay: source/design.clay\n"
        f"  top_cell: {cell}\n"
        "metadata:\n"
        "  pdk: gf180mcu\n"
        "  layout_mode: generic\n",
        encoding="utf-8",
    )

    cf = parse_clay_file(src_dir / "design.clay")
    cell_def = next(c for c in cf.cell_defs if getattr(c, "local_name", None) == cell)
    spec = compile_flow_spec(
        cell_def, case_dir=case_dir, verify_jobs=getattr(cf, "verify_jobs", None)
    )
    work = work_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    ctx = FlowContext(spec=spec, work_dir=work, recorder=None, force=True)
    # Force device emission so MOS/cap geometry is present in the GDS (parity with
    # the soundness test's CLAY_GENERIC_EMIT_DEVICES=1 path).
    import os

    prev = os.environ.get("CLAY_GENERIC_EMIT_DEVICES")
    os.environ["CLAY_GENERIC_EMIT_DEVICES"] = "1"
    try:
        res = LayoutStep()._run_generic(spec, ctx)
    finally:
        if prev is None:
            os.environ.pop("CLAY_GENERIC_EMIT_DEVICES", None)
        else:
            os.environ["CLAY_GENERIC_EMIT_DEVICES"] = prev
    if res.status != "pass":
        raise RuntimeError(f"generic layout step did not pass: {res.summary}")
    return gds_path(ctx)


# --------------------------------------------------------------------------- #
# THE ROUND-TRIP.
# --------------------------------------------------------------------------- #
def roundtrip(
    intent: Any,
    *,
    work_dir: str,
    reference_gds: Optional[str] = None,
) -> RoundtripReport:
    """Run the full round-trip for ``intent`` and return a :class:`RoundtripReport`.

    Steps:
      1. ``emit_clay(intent)`` -> a parseable Clay source (validated to parse);
      2. generic place/route -> ``layout.gds``;
      3. fidelity vs ``reference_gds`` (when supplied). When no reference is given,
         the round-trip uses OUR gds as its own reference (reflexive -> 1.0) so the
         harness always returns an in-range fidelity + populated breakdown.

    ``work_dir`` is the scratch root for the emitted case + the GDS.
    """
    from clay.layout.completeness.fidelity import fidelity_score

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    cell = str(getattr(intent, "cell", "ExtractedCell"))

    # 1. emit + parse-validate.
    clay_src = emit_clay(intent)
    parsed_ok, parse_err = validate_parses(clay_src, work_dir=str(wd / "parse_check"))
    emitted_path = wd / "case" / "source" / "design.clay"

    emitted_rel, commented_rel = _emitted_and_commented(intent)

    if not parsed_ok:
        # Cannot place an unparseable source; return a 0-fidelity report that names
        # the parse error (the round-trip is still well-defined: 0 fidelity).
        emitted_path.parent.mkdir(parents=True, exist_ok=True)
        emitted_path.write_text(clay_src, encoding="utf-8")
        return RoundtripReport(
            cell=cell,
            fidelity=0.0,
            sub_scores={},
            breakdown="(emitted source did not parse)",
            gap_localization={"parse": f"emit produced unparseable source: {parse_err}"},
            emitted_clay_path=str(emitted_path),
            our_gds_path="",
            reference_gds_path=reference_gds,
            parsed_ok=False,
            parse_error=parse_err,
            emitted_relations=emitted_rel,
            commented_relations=commented_rel,
        )

    # 2. generic place/route -> GDS.
    our_gds = _build_gds(clay_src, cell, wd)

    # 3. fidelity. Reflexive when no external reference is supplied.
    ref = reference_gds if reference_gds else str(our_gds)
    rep = fidelity_score(ref, str(our_gds))

    # 4. gap localization per fidelity feature.
    ref_classes = tuple(rep.reference.tier_class_order)
    our_classes = tuple(rep.ours.tier_class_order)
    gap: Dict[str, str] = {}
    for feature, our_sub in rep.sub_scores.items():
        ref_sub = 1.0  # reference is itself perfect on every feature by definition
        gap[feature] = _classify_gap(
            feature, ref_sub, our_sub, emitted_rel, commented_rel, rep.notes,
            ref_classes=ref_classes, our_classes=our_classes,
        )

    return RoundtripReport(
        cell=cell,
        fidelity=rep.fidelity_score,
        sub_scores=dict(rep.sub_scores),
        breakdown=rep.breakdown(),
        gap_localization=gap,
        emitted_clay_path=str(emitted_path),
        our_gds_path=str(our_gds),
        reference_gds_path=(reference_gds if reference_gds else None),
        parsed_ok=True,
        parse_error=None,
        emitted_relations=emitted_rel,
        commented_relations=commented_rel,
        notes=dict(rep.notes),
    )
