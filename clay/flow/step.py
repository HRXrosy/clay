"""Universal generic flow steps for the Clay flow engine.

Every step is parameterized ENTIRELY by :class:`CircuitSpec` data — there are
no ``if spec.key == "..."`` branches anywhere. Each step wraps an existing Clay
tool (parser/compiler, the KLayout DRC/LVS signoff harnesses, the Magic PEX
backend, the per-circuit post-sim module) and reports a :class:`StepResult`.

The steps mirror the call sequence that the historical fc_ota-only
``tools/schematic_viewer/scripts/run_current_signoff.py`` performed, but lifted
to consume circuit specifics from the spec so the SAME code drives every
circuit in the registry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

from clay.flow.spec import CircuitSpec, FlowContext, StepResult, resolve_kpi_value

_LOG = logging.getLogger(__name__)

# Layout-signoff KLayout decks now live on the PDK adapter
# (clay.pdk.get_pdk(<pdk>).klayout_decks()), so the DRC/LVS steps resolve a deck
# PER-PDK from spec.pdk instead of hardcoding gf180. REPO_ROOT is kept for other
# in-tree path needs.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _placement_quality_blocking(
    placement: object, macro_consumption: object | None
) -> bool:
    """Reject an unsolved placement unless the final macro contract proves use.

    A compact-intent replay may legitimately report a failed intermediate
    solver after the declared macro placement was consumed and independently
    certified.  Without that certificate, treating ``solver_success=False`` as
    a successful layout would silently weaken the physical-quality contract.
    """

    if bool(getattr(placement, "solver_success", False)):
        return False
    return getattr(macro_consumption, "status", None) != "pass"


def _replay_exact_route_manifest(route_result: Any):
    """Replay exhaustive router ownership for the canonical generic emitter."""

    if (
        getattr(route_result, "candidate_pool", None) is None
        and getattr(route_result, "resource_plan", None) is None
    ):
        from clay.layout.route.exact_artifact_evidence import (
            route_owned_manifest_from_routed_geometry,
        )

        return route_owned_manifest_from_routed_geometry(route_result.routed)
    if (getattr(route_result, "candidate_pool", None) is None) != (
        getattr(route_result, "resource_plan", None) is None
    ):
        raise RuntimeError("route result has partial candidate-pool/plan lineage")
    from clay.layout.route.allocated_route_manifest import (
        replay_exact_allocated_route_manifest,
    )

    return replay_exact_allocated_route_manifest(
        candidate_pool=route_result.candidate_pool,
        resource_plan=route_result.resource_plan,
        materialization_certificate=(route_result.routed.allocation_replay_certificate),
        routing_world=route_result.routing_world_provenance,
        tech=route_result.tech,
    )


def _write_exact_artifact_receipt(
    path: Path,
    artifact: Any,
    *,
    physical_macro_graph_sha256: str,
    candidate_sha256: str | None,
    placement_state_sha256: str | None,
    rigid_nonmos_macro_count: int = 0,
    rigid_nonmos_emission_certificate: Any = None,
) -> dict[str, Any]:
    """Persist the emitter-minted exact scope proof as an atomic flow artifact."""

    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

    if type(artifact) is not ExactEmittedArtifact:
        raise TypeError("canonical generic emitter did not return ExactEmittedArtifact")
    artifact.validate()
    if type(rigid_nonmos_macro_count) is not int or rigid_nonmos_macro_count < 0:
        raise TypeError("rigid non-MOS macro count must be an exact non-negative int")
    if (rigid_nonmos_macro_count > 0) != (
        rigid_nonmos_emission_certificate is not None
    ):
        raise ValueError(
            "exact artifact receipt has incomplete rigid non-MOS emission evidence"
        )
    rigid_nonmos_payload = None
    if rigid_nonmos_emission_certificate is not None:
        from clay.layout.nonrow_macro_plan import RigidNonmosEmissionCertificate

        if type(rigid_nonmos_emission_certificate) is not RigidNonmosEmissionCertificate:
            raise TypeError(
                "exact artifact receipt requires an exact rigid non-MOS certificate"
            )
        if (
            rigid_nonmos_emission_certificate.graph_sha256
            != physical_macro_graph_sha256
            or len(rigid_nonmos_emission_certificate.macro_structure_fingerprints)
            != rigid_nonmos_macro_count
        ):
            raise ValueError(
                "rigid non-MOS emission evidence differs from the physical macro graph"
            )
        rigid_nonmos_emission_certificate.bind_exact_artifact(artifact)
        rigid_nonmos_payload = rigid_nonmos_emission_certificate.payload()
        rigid_nonmos_payload["sha256"] = rigid_nonmos_emission_certificate.sha256
        rigid_nonmos_payload = json.loads(json.dumps(
            rigid_nonmos_payload,
            sort_keys=True,
            separators=(",", ":"),
        ))
    scope = artifact.scope
    receipt: dict[str, Any] = {
        "schema": "clay.exact-layout-artifact/v1",
        "status": "pass",
        "exact_gds": str(artifact.exact_gds_path),
        "exact_gds_sha256": artifact.truth_manifest.exact_gds_sha256,
        "pdk_sha256": artifact.truth_manifest.pdk_sha256,
        "route_manifest_sha256": artifact.route_manifest.route_manifest_sha256,
        "route_manifest_entry_count": len(artifact.route_manifest.entries),
        "scope": {
            "exhaustive": scope.exhaustive,
            "allocated_union_matches_router": scope.allocated_union_matches_router,
            "allocated_route_manifest_sha256": (scope.allocated_route_manifest_sha256),
            "emitted_route_manifest_sha256": scope.emitted_route_manifest_sha256,
            "allocated_route_union_sha256": scope.allocated_route_union_sha256,
            "router_route_union_sha256": scope.router_route_union_sha256,
            "emitted_route_union_sha256": scope.emitted_route_union_sha256,
            "emitted_device_snapshot_sha256": (
                scope.emitted_device_snapshot_sha256
            ),
            "source_cell_source_sha256": scope.source_cell_source_sha256,
            "source_member_inventory_sha256": (
                scope.source_member_inventory_sha256
            ),
            "guard_ring_member_classes_sha256": (
                scope.guard_ring_member_classes_sha256
            ),
            "pre_transform_geometry_sha256": scope.pre_transform_geometry_sha256,
            "post_transform_geometry_sha256": scope.post_transform_geometry_sha256,
            "written_geometry_sha256": scope.written_geometry_sha256,
            "heal_shape_count": scope.heal_shape_count,
            "upper_fill_shape_count": scope.upper_fill_shape_count,
            "post_assembly_fill_shape_count": (scope.post_assembly_fill_shape_count),
            "snapped_shape_count": scope.snapped_shape_count,
        },
        "physical_macro_graph_sha256": physical_macro_graph_sha256,
        "candidate_sha256": candidate_sha256,
        "placement_state_sha256": placement_state_sha256,
        "rigid_nonmos_macro_count": rigid_nonmos_macro_count,
        "rigid_nonmos_emission": rigid_nonmos_payload,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return receipt


def _write_exact_artifact_failure(path: Path, exc: Exception) -> dict[str, Any]:
    """Preserve a named exact-emission failure without minting success evidence."""

    receipt = {
        "schema": "clay.exact-layout-artifact/v1",
        "status": "fail",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return receipt


def gf180_klayout_decks() -> dict[str, Path]:
    """Back-compat shim. The canonical source is the gf180mcu PDK adapter:
    ``clay.pdk.get_pdk('gf180mcu').klayout_decks()``. Kept for existing callers
    (tools/clay_layout_grade, run_current_signoff, the signoff test)."""
    from clay.pdk import get_pdk

    return get_pdk("gf180mcu").klayout_decks()


def _resolve_klayout_deck(spec: CircuitSpec, kind: str) -> "Path | None":
    """Resolve the executable KLayout DRC/LVS deck for ``spec.pdk`` via the PDK
    adapter registry. Returns None when the PDK has no signoff deck wired -- the
    DRC/LVS step then SKIPS gracefully rather than assuming gf180."""
    from clay.pdk import get_pdk

    try:
        return get_pdk(spec.pdk).klayout_decks().get(kind)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Small generic helpers (no circuit literals)                                 #
# --------------------------------------------------------------------------- #
def _effective_source_cir(spec: CircuitSpec, ctx: FlowContext) -> Path:
    candidate = ctx.extra.get("candidate_source_cir")
    return Path(candidate) if candidate else Path(spec.source_cir)


def _effective_design_clay(spec: CircuitSpec, ctx: FlowContext) -> Path:
    candidate = ctx.extra.get("effective_design_clay")
    return (
        Path(candidate) if candidate else Path(spec.case_dir) / "source" / "design.clay"
    )


def _declared_ground_net(cell_def: Any, *, fallback: str) -> str:
    """Resolve the one global substrate net from Clay port-role intent.

    A PDK adapter owns the fallback spelling for designs without a declared
    substrate rail.  An explicit ``ground`` role owns the deck-global substrate
    name; otherwise the negative supply (``power_n``) owns it.  This preserves
    split-domain cells such as ``GNDA: power_n`` plus ``SUB: ground`` while also
    handling imported cells whose only low rail is ``GND: power_n``.  Multiple
    candidates at the selected semantic tier cannot be collapsed by the current
    single-p-substrate deck and therefore fail closed.
    """

    from clay.port_roles import resolve_substrate_net

    return resolve_substrate_net(cell_def, fallback=fallback)


def _source_has_cap(spec: CircuitSpec) -> bool:
    """True when the bound source ``.cir`` declares a MIM capacitor (a ``C`` card).

    GENERIC (no circuit literal): scans the netlist for a device line whose first
    token starts with ``C`` (and has >= 2 nodes + a value). Used to gate the gf180
    MIM-Option-A DRC/LVS deck switches so a cap-free cell's invocation is unchanged.
    """
    try:
        src = Path(spec.source_cir)
        if not src.is_file():
            return False
        text = src.read_text(encoding="utf-8-sig")
    except Exception:  # noqa: BLE001
        return False
    for raw in text.splitlines():
        line = raw.split("$", 1)[0].strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        if parts and parts[0][:1].upper() == "C" and len(parts) >= 4:
            return True
    return False


def _gf180_mim_context(
    spec: CircuitSpec | None,
    *,
    cell_def: Any = None,
    intent: Any = None,
    tech: Any = None,
    mim_option: str | None = None,
    cap_option: str | None = None,
    metal_level: str | None = None,
) -> dict[str, str]:
    """Return explicit GF180 MIM config overrides (empty dict keeps legacy Option-A)."""
    pdk_id = (
        getattr(spec, "pdk", None)
        or getattr(tech, "pdk_id", None)
        or ""
    )
    if str(pdk_id).strip().lower() != "gf180mcu":
        return {}

    def _pick(*values: Any) -> str | None:
        for val in values:
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    case_dir = getattr(spec, "case_dir", None) if spec is not None else None
    raw_mim = _pick(
        mim_option,
        getattr(tech, "mim_option", None),
        getattr(intent, "mim_option", None),
        getattr(cell_def, "mim_option", None),
        _read_case_metadata_value(case_dir, "mim_option") if case_dir else None,
    )
    raw_cap = _pick(
        cap_option,
        getattr(tech, "cap_option", None),
        getattr(intent, "mim_cap", None),
        getattr(cell_def, "mim_cap", None),
        _read_case_metadata_value(case_dir, "mim_cap") if case_dir else None,
        _read_case_metadata_value(case_dir, "cap_option") if case_dir else None,
    )
    raw_lvl = _pick(
        metal_level,
        getattr(tech, "metal_level", None),
        getattr(intent, "metal_level", None),
        getattr(cell_def, "metal_level", None),
        _read_case_metadata_value(case_dir, "metal_level") if case_dir else None,
    )
    if raw_mim is None and raw_cap is None and raw_lvl is None:
        return {}
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    opt, cap_opt, lvl = resolve_gf180_mim_config(
        mim_option=raw_mim,
        cap_option=raw_cap,
        metal_level=raw_lvl,
    )
    out = {"mim_option": opt, "cap_option": cap_opt}
    if opt == "B" or raw_lvl is not None:
        out["metal_level"] = lvl
    return out


def _attach_gf180_mim_attrs(target: Any, mim_ctx: Mapping[str, str]) -> None:
    """Attach resolved GF180 MIM attributes on CellDef/LayoutIntent without ``cap_option`` collision."""
    if target is None or not mim_ctx:
        return
    d = getattr(target, "__dict__", None)
    if not isinstance(d, dict):
        return
    if "mim_option" in mim_ctx and "mim_option" not in d:
        d["mim_option"] = mim_ctx["mim_option"]
    if "metal_level" in mim_ctx and "metal_level" not in d:
        d["metal_level"] = mim_ctx["metal_level"]
    if "cap_option" in mim_ctx and "mim_cap" not in d:
        d["mim_cap"] = mim_ctx["cap_option"]


def _mim_drc_rd(
    spec: CircuitSpec,
    *,
    cell_def: Any = None,
    intent: Any = None,
    tech: Any = None,
) -> list[tuple[str, str]]:
    """The ``-rd`` switch pairs that ENABLE the gf180 MIM DRC/LVS section
    for a cell containing a MIM capacitor, else an empty list.

    The gf180 decks gate every MIM rule / extraction behind ``$mim_option`` (A/B)
    and ``$mim_cap`` (1/1.5/2); without them the MIM plate/via DRC never runs and
    the LVS extractor never produces the ``mim_<d>fF`` device. Option-A defaults
    keep the exact two-arg ``_lvs_ref_profile(spec.pdk)`` call and pair shape.
    """
    if not _source_has_cap(spec):
        return []
    mim_ctx = _gf180_mim_context(spec, cell_def=cell_def, intent=intent, tech=tech)
    if not mim_ctx:
        profile = _lvs_ref_profile(spec.pdk)
        if not getattr(profile, "mim_option", None) or not getattr(
            profile, "mim_cap", None
        ):
            return []
        return [("mim_option", str(profile.mim_option)), ("mim_cap", str(profile.mim_cap))]
    profile = _lvs_ref_profile(spec.pdk, **mim_ctx)
    if not getattr(profile, "mim_option", None) or not getattr(
        profile, "mim_cap", None
    ):
        return []
    from clay.layout.klayout_deck_gen import gf180_mim_drc_rd

    return gf180_mim_drc_rd(
        mim_option=str(profile.mim_option),
        cap_option=str(profile.mim_cap),
        metal_level=str(getattr(profile, "metal_level", None) or "5LM"),
    )


def _label_mim_cap_ports(
    out_gds: Path, spec: CircuitSpec, placement: Any, port_nets: list[str]
) -> None:
    """Write each MIM-cap plate's PORT label on its OWN metal label layer.

    The gf180 MIM cap names the BOTtom plate on Metal2 (36/10) and the TOP plate on
    Metal3 (42/10) -- two different label layers, which ``emit_routed_cell`` (one
    label layer) cannot do. For each placed cap device, this reopens the GDS and,
    for each p/n terminal whose net is a cell PORT, writes a Text label at the
    terminal centre on the label datatype (10) of the metal the terminal sits on
    (resolved from the placed device polys covering the terminal centre). Only
    PORT nets are labelled (so the extracted top-level pin set equals the .subckt
    ports); a terminal whose net is internal is skipped. Best-effort: any failure
    leaves the GDS as emitted (the LVS step then surfaces the honest residual)."""
    try:
        import klayout.db as db
    except Exception:  # noqa: BLE001
        return
    allowed = {str(n) for n in (port_nets or [])}
    devices = getattr(placement, "devices", {}) or {}
    # device_label -> {role: net} from the bound .cir C cards (the p/n pins).
    cap_pins: dict[str, dict[str, str]] = {}
    try:
        from clay.layout.generic_placement import parse_cir_passive_records

        for lbl, rec in (parse_cir_passive_records(spec.source_cir) or {}).items():
            if rec.get("kind") == "cap":
                cap_pins[lbl] = {k: str(v) for k, v in (rec.get("pins") or {}).items()}
    except Exception:  # noqa: BLE001
        cap_pins = {}
    # net -> [(metal_label_layer, x, y)] for each cap terminal on a port net.
    to_write: list[tuple[str, int, int, int, int]] = []
    for label, pd in devices.items():
        if getattr(pd, "kind", "") != "cap":
            continue
        terms = getattr(pd, "terminals", {}) or {}
        polys = getattr(pd, "polys", []) or []
        role_net = dict(cap_pins.get(label, {}))
        # Per-plate label layer (gf180 MIM convention, fixed -- not per-circuit):
        # the 'p' terminal is the bottom plate -> Metal2 (the metal_key(2) layer),
        # the 'n' terminal is the top plate -> Metal3 (metal_key(3)). Resolved from
        # the PDK tech so the layer numbers are PDK data, not literals.
        try:
            from clay.layout.tech import LayoutTech as _LT

            mim_ctx = _gf180_mim_context(spec)
            _t = _LT.resolve(spec.pdk, **mim_ctx)
            if mim_ctx.get("mim_option") == "B":
                from clay.pdk.gf180mcu import gf180_mim_stack_spec

                _st = gf180_mim_stack_spec("B", metal_level=mim_ctx.get("metal_level", "5LM"))
                _role_metal = {
                    "p": _t.layer(str(_st["bottom_metal"]))[0],
                    "n": _t.layer(str(_st["cover_metal"]))[0],
                }
            else:
                _role_metal = {"p": _t.metal_key(2)[0], "n": _t.metal_key(3)[0]}
        except Exception:  # noqa: BLE001
            _role_metal = {"p": 36, "n": 42}
        for role, center in terms.items():
            net = role_net.get(role, "")
            if not net or (allowed and net not in allowed):
                continue
            if isinstance(center, tuple) and len(center) == 2:
                cx, cy = int(center[0]), int(center[1])
            else:
                b = getattr(center, "bbox", None)
                if not b:
                    continue
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            layer = _role_metal.get(role)
            if layer is None:
                continue
            to_write.append((net, layer, 10, cx, cy))
    if not to_write:
        return
    try:
        ly = db.Layout()
        ly.read(str(out_gds))
        top = ly.top_cell()
        for net, layer, dt, cx, cy in to_write:
            li = ly.layer(layer, dt)
            top.shapes(li).insert(db.Text(net, db.Trans(db.Vector(int(cx), int(cy)))))
        ly.write(str(out_gds))
    except Exception:  # noqa: BLE001 - labelling is best-effort; LVS surfaces gaps
        return


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def gds_path(ctx: FlowContext) -> Path:
    """The single GDS the flow produces / consumes, under the run work dir."""
    return Path(ctx.work_dir) / "layout.gds"


def _resolve_gf180_poly_sheet_rho() -> float:
    """The gf180 poly-resistor sheet-rho DEFAULT fallback (used when ``pdk`` is
    not supplied or the adapter is silent on ``poly_sheet_rho_ohm_sq``).

    Task B3: resolved from the SAME single tabulated source the geometry builder
    consults (``clay.layout.passive_cell._resolve_poly_res_facts('gf180mcu')``),
    not a second hardcoded 350.0 literal. The VALUE is unchanged (350.0,
    gf180mcu.lvs ``resistor_with_bulk("ppolyf_u", 350)``) -- this only removes
    the duplicate declaration; the byte-parity contract below (the reference
    card output) is unaffected."""
    from clay.layout.passive_cell import _resolve_poly_res_facts

    facts, _prov = _resolve_poly_res_facts("gf180mcu")
    return float(facts.sheet_ohm_sq)


#: Default (gf180mcu-parity) LVS reference device-model names, poly-resistor
#: model, sheet-rho (Ohm/square) and substrate net. Used whenever a PDK is not
#: supplied OR the adapter is SILENT on a field -- so the historical gf180 cards
#: stay byte-for-byte identical (the 5 L5 co-clean cells + every gf180 LVS test).
_GF180_NMOS_REF = "nmos_3p3"
_GF180_PMOS_REF = "pmos_3p3"
_GF180_POLY_RES_REF = "ppolyf_u"
_GF180_POLY_SHEET_RHO = _resolve_gf180_poly_sheet_rho()
_GF180_SUBSTRATE_NET = "vss"


class _LvsRefProfile:
    """Per-PDK device-model names the LVS reference must emit.

    The KLayout LVS comparator aligns devices by CLASS NAME, and the (generated)
    extractor names each class after the PDK's OWN model (``mos4("pmos_3p3")`` for
    external_pdk, ``nmos_3p3`` for gf180). So the reference's MOS / poly-R model tokens
    and its substrate (bulk) net are PDK data, not literals. This profile pulls
    them from ``get_pdk(pdk).lvs_reference_models()`` / ``substrate_net()`` and
    falls back to the gf180 value wherever the adapter is silent, keeping gf180
    byte-parity. ``pdk=None`` is the pure gf180-default profile.
    """

    __slots__ = (
        "pdk_id",
        "nmos",
        "pmos",
        "nmos_aliases",
        "pmos_aliases",
        "mos_model_by_alias",
        "poly_res",
        "sheet_rho",
        "substrate",
        "resistor_has_bulk",
        "mim_density",
        "mim_option",
        "mim_cap",
        "metal_level",
        "mim_model",
    )

    def __init__(
        self,
        *,
        pdk_id: str | None,
        nmos: str,
        pmos: str,
        nmos_aliases: tuple[str, ...],
        pmos_aliases: tuple[str, ...],
        poly_res: str,
        sheet_rho: float,
        substrate: str,
        resistor_has_bulk: bool = True,
        mim_density: float | None = None,
        mim_option: str | None = None,
        mim_cap: str | None = None,
        metal_level: str | None = None,
        mim_model: str | None = None,
        mos_model_by_alias: Mapping[str, str] | None = None,
    ) -> None:
        self.pdk_id = pdk_id
        self.nmos = nmos
        self.pmos = pmos
        self.nmos_aliases = frozenset(str(value).casefold() for value in nmos_aliases)
        self.pmos_aliases = frozenset(str(value).casefold() for value in pmos_aliases)
        self.mos_model_by_alias = dict(mos_model_by_alias or {})
        self.poly_res = poly_res
        self.sheet_rho = sheet_rho
        self.substrate = substrate
        self.resistor_has_bulk = bool(resistor_has_bulk)
        # MIM-capacitor extraction facts (gf180 only today): the area-specific
        # capacitance (F/um^2) the LVS deck multiplies by the FuseTop area, and the
        # ``$mim_option`` / ``$mim_cap`` deck switches that ENABLE MIM extraction +
        # MIM DRC. None on a PDK with no MIM stack (the C card stays cap_debt).
        self.mim_density = mim_density
        self.mim_option = mim_option
        self.mim_cap = mim_cap
        self.metal_level = metal_level
        # The extracted MIM device's CLASS NAME (the first arg of the deck's
        # ``capacitor("mim_<d>fF", <density>, MIMCap)`` extraction call). The
        # KLayout netlist comparer pairs devices by class name, AND the deck's
        # reference reader keys the cap's device class off the card's MODEL token
        # (``device_class_by_name(model)``). So the reference C card must carry
        # this exact model token or a generic (unnamed) reference cap never pairs
        # with the named ``mim_1p0fF`` extracted device. None when no MIM stack.
        self.mim_model = mim_model


def _lvs_ref_profile(
    pdk: str | None,
    *,
    substrate_net: str | None = None,
    mim_option: str | None = None,
    cap_option: str | None = None,
    metal_level: str | None = None,
) -> _LvsRefProfile:
    """Resolve the per-PDK LVS reference profile, gf180-default when silent.

    ``pdk=None`` (or an adapter that fails to load) yields the gf180-parity
    profile so existing callers are byte-unchanged. For a real adapter, the
    nmos/pmos model names come from ``lvs_reference_models()``, the poly-R model
    from its ``'poly_res'`` entry (or the adapter's ``poly_res_model``), the
    sheet-rho from ``poly_sheet_rho_ohm_sq``, and the bulk net from
    ``substrate_net()`` (lower-cased: the generated deck matches the global
    substrate net case-insensitively and the historical gf180 reference uses the
    lower-case ``vss``). Any field the adapter does not declare keeps the gf180
    default (parity).  ``substrate_net`` is the optional cell-intent override
    used when a declared ``ground`` port owns the global substrate spelling;
    absent or blank keeps the adapter/default value byte-for-byte unchanged.
    """
    nmos = _GF180_NMOS_REF
    pmos = _GF180_PMOS_REF
    poly_res = _GF180_POLY_RES_REF
    sheet_rho = _GF180_POLY_SHEET_RHO
    substrate = _GF180_SUBSTRATE_NET
    # MIM facts: only gf180mcu has the MIM stack wired today. Default MIM_CAP
    # option "1" (mim_1p0fF, 1.0 fF/um^2, rated to 20V). The density is the SINGLE
    # source of truth in the passive builder, so the reference C and the laid-out
    # FuseTop area agree by construction.
    mim_density: float | None = None
    resolved_mim_option: str | None = None
    mim_cap: str | None = None
    resolved_metal_level: str | None = None
    mim_model: str | None = None
    resistor_has_bulk = True
    nmos_aliases = {_GF180_NMOS_REF.casefold()}
    pmos_aliases = {_GF180_PMOS_REF.casefold()}
    mos_model_by_alias: dict[str, str] = {}
    if (pdk or "").lower() == "gf180mcu":
        try:
            from clay.layout.passive_device_cell import (
                mim_density_f_per_um2,
                mim_model_name,
                _GF180_MIM_DEFAULT_CAP_OPTION,
            )
            from clay.layout.tech import LayoutTech

            if mim_option is None and cap_option is None and metal_level is None:
                _tech = LayoutTech.resolve("gf180mcu")
                mim_cap = _GF180_MIM_DEFAULT_CAP_OPTION
                resolved_mim_option = "A"
            else:
                from clay.pdk.gf180mcu import resolve_gf180_mim_config

                resolved_mim_option, mim_cap, norm_lvl = resolve_gf180_mim_config(
                    mim_option=mim_option,
                    cap_option=cap_option,
                    metal_level=metal_level,
                )
                resolved_metal_level = (
                    norm_lvl if resolved_mim_option == "B" or metal_level is not None else None
                )
                _tech = LayoutTech.resolve(
                    "gf180mcu",
                    mim_option=resolved_mim_option,
                    cap_option=mim_cap,
                    metal_level=resolved_metal_level,
                )
            mim_density = mim_density_f_per_um2(_tech, mim_cap)
            mim_model = mim_model_name(mim_cap)
        except Exception:  # noqa: BLE001 - MIM lowering simply stays off
            mim_density = None
            mim_model = None
    if pdk:
        try:
            from clay.pdk import get_pdk

            adapter = get_pdk(pdk)
        except Exception:  # noqa: BLE001 - unknown PDK -> gf180 defaults (parity)
            adapter = None
        if adapter is not None:
            models = {}
            try:
                models = dict(adapter.lvs_reference_models() or {})
            except Exception:  # noqa: BLE001
                models = {}
            nmos = models.get("nmos") or nmos
            pmos = models.get("pmos") or pmos
            resolved_mos_models = dict(models)
            # Source/simulator model aliases and extracted LVS class names are
            # separate namespaces on commercial PDKs (for example n33 in the
            # schematic extracts as class n18).  Resolve polarity from the
            # adapter's semantic device-class keys, never from circuit/model
            # spelling heuristics.
            alias_maps = [
                getattr(adapter, "device_bindings", {}) or {},
                models,
            ]
            sim_aliases = getattr(adapter, "extra", {}).get(
                "ngspice_device_bindings", {}
            )
            if isinstance(sim_aliases, dict):
                alias_maps.append(sim_aliases)
            for alias_map in alias_maps:
                if not isinstance(alias_map, dict):
                    continue
                for device_class, model_name in alias_map.items():
                    key = str(device_class).casefold()
                    if key.startswith("nmos"):
                        nmos_aliases.add(str(model_name).casefold())
                    elif key.startswith("pmos"):
                        pmos_aliases.add(str(model_name).casefold())
            poly_res = (
                models.get("poly_res")
                or getattr(adapter, "poly_res_model", None)
                or poly_res
            )
            rho = getattr(adapter, "poly_sheet_rho_ohm_sq", None)
            if rho:
                sheet_rho = float(rho)
            try:
                sub = adapter.substrate_net()
            except Exception:  # noqa: BLE001
                sub = None
            if sub:
                substrate = str(sub).lower()
            # Commercial adapters execute the route oracle's existing FULL
            # extraction deck.  Pair the reference to that deck's class tokens
            # and numeric passive facts; physical HSPICE model names are a
            # different namespace and must not leak into the KLayout comparer.
            if bool(getattr(adapter, "klayout_lvs_from_rules", False)):
                try:
                    from clay.layout.route.lvs_gate import _resolved_lvs_config

                    config = _resolved_lvs_config(
                        str(pdk).lower(), full_extraction=True
                    )
                    devices = dict(config.get("devices") or {})
                    resolved_mos_models.update(devices)
                    nmos = str(devices.get("nmos") or nmos)
                    pmos = str(devices.get("pmos") or pmos)
                    res_spec = config.get("res")
                    if res_spec is not None:
                        poly_res = str(res_spec.model)
                        sheet_rho = float(res_spec.sheet_ohm_sq)
                        resistor_has_bulk = False
                    mim_spec = config.get("mim")
                    if mim_spec is not None:
                        mim_density = float(mim_spec.area_cap_f_per_um2)
                        mim_model = str(mim_spec.model)
                except (ImportError, KeyError, TypeError, ValueError):
                    # Leave the profile incomplete; the ensuing reference debt
                    # fails closed instead of inventing a device class.
                    pass
            # Preserve voltage/oxide families across the source/simulator/LVS
            # namespaces. Polarity alone must never lower n33/p33 into n18/p18.
            for alias_map in (*alias_maps, resolved_mos_models):
                if not isinstance(alias_map, dict):
                    continue
                for semantic, alias in alias_map.items():
                    if str(semantic) not in ("nmos", "pmos", "nmos3v", "pmos3v"):
                        continue
                    target = resolved_mos_models.get(str(semantic))
                    if target:
                        key = str(alias).casefold()
                        if key in mos_model_by_alias and mos_model_by_alias[key] != str(target):
                            raise ValueError(f"ambiguous MOS voltage-family alias {alias!r}")
                        mos_model_by_alias[key] = str(target)
    nmos_aliases.add(str(nmos).casefold())
    pmos_aliases.add(str(pmos).casefold())
    if substrate_net is not None and str(substrate_net).strip():
        substrate = str(substrate_net).strip()
    return _LvsRefProfile(
        pdk_id=str(pdk).lower() if pdk else None,
        nmos=nmos,
        pmos=pmos,
        nmos_aliases=tuple(sorted(nmos_aliases)),
        pmos_aliases=tuple(sorted(pmos_aliases)),
        poly_res=poly_res,
        sheet_rho=sheet_rho,
        substrate=substrate,
        resistor_has_bulk=resistor_has_bulk,
        mim_density=mim_density,
        mim_option=resolved_mim_option,
        mim_cap=mim_cap,
        metal_level=resolved_metal_level,
        mim_model=mim_model,
        mos_model_by_alias=mos_model_by_alias,
    )


#: SPICE model-name -> LVS model-name mapping, used GENERICALLY to build the
#: effective-W LVS reference for any circuit. NMOS/PMOS device-model tokens are
#: recognised by substring so every circuit's ``.cir`` lowers the same way (no
#: per-circuit table). The TARGET model name (``nmos_3p3`` for gf180,
#: ``nmos_3p3`` for external_pdk) comes from the per-PDK ``profile`` so the reference's
#: device class matches what the extractor names it; ``profile=None`` keeps the
#: historical gf180 names (byte-parity).
def _lvs_model_for(model: str, profile: "_LvsRefProfile | None" = None) -> str | None:
    nmos = profile.nmos if profile is not None else _GF180_NMOS_REF
    pmos = profile.pmos if profile is not None else _GF180_PMOS_REF
    low = model.lower()
    if profile is not None:
        family_alias = re.sub(r"\.\d+$", "", low)
        if family_alias in profile.mos_model_by_alias:
            return profile.mos_model_by_alias[family_alias]
        if low in profile.nmos_aliases:
            return nmos
        if low in profile.pmos_aliases:
            return pmos
    if low in ("nch", "nmos", "nmos_3p3", "nfet_03v3"):
        return nmos
    if low in ("pch", "pmos", "pmos_3p3", "pfet_03v3"):
        return pmos
    if "pmos" in low or "pfet" in low or low.startswith("pch"):
        return pmos
    if "nmos" in low or "nfet" in low or low.startswith("nch"):
        return nmos
    return None


def _lvs_bjt_binding_for(model: str, profile: "_LvsRefProfile | None" = None):
    """Resolve a schematic BJT token through adapter-owned model DATA."""

    if profile is not None and profile.pdk_id:
        try:
            from clay.pdk import get_pdk, load_all

            load_all()
            bindings = get_pdk(profile.pdk_id).bjt_bindings or {}
            low = str(model).casefold()
            for binding in bindings.values():
                if low in {
                    str(binding.simulator_model).casefold(),
                    str(binding.lvs_model).casefold(),
                }:
                    return binding
        except Exception:  # noqa: BLE001 - unresolved adapter stays honest below
            return None
    return None


def _lvs_bjt_model_for(
    model: str, profile: "_LvsRefProfile | None" = None
) -> str | None:
    """Map a schematic BJT model token to the target PDK's LVS model, or None.

    Adapter DATA handles commercial and open PDKs uniformly.  The historical
    gf180 prefix fallback remains for ``profile=None`` byte compatibility.
    """
    binding = _lvs_bjt_binding_for(model, profile)
    if binding is not None:
        return str(binding.lvs_model)
    low = model.lower()
    if low.startswith("vpnp") or low.startswith("vnpn"):
        return model
    return None


#: gf180mcu LVS MOS-cap models the layout's ``mos_cap_mk``-marked gate-over-COMP
#: extracts as. The KLayout deck's ``MosCap`` device class compares these on the
#: gate-poly GEOMETRY (area ``A`` and perimeter ``P``), with the capacitance
#: ``C`` disabled — so a schematic ideal ``C`` card that the LAYOUT realises as
#: a foldable MOS-cap must lower to one of these models carrying the as-drawn
#: A/P, not a value. The reference SPICE reader (gf180mcu.lvs
#: ``SubcircuitModelsReader``) derives a cap device's A/P from the card's W/L:
#: ``A = W*L`` and ``P = 2*(W+L)`` (both in um). Recognised by substring so a
#: circuit naming ``pmoscap_3p3`` / ``cap_pmos_03v3`` / ``nmoscap_3p3`` lowers
#: the same way (no per-circuit table).
def _lvs_moscap_model_for(model: str | None) -> str | None:
    if not model:
        return None
    low = model.lower()
    if "pmoscap" in low or "cap_pmos" in low or "pmos_cap" in low:
        return "pmoscap_3p3"
    if "nmoscap" in low or "cap_nmos" in low or "nmos_cap" in low:
        return "nmoscap_3p3"
    return None


def _equivalent_cap_wl_um(area_um2: float, perim_um: float) -> tuple[float, float]:
    """Equivalent single-device (W, L) in um reproducing a combined MOS-cap A/P.

    The layout draws a MOS-cap as ``fingers`` parallel gate fingers; KLayout's
    ``netlist.combine_devices`` folds them into ONE extracted device whose area
    and perimeter are the per-finger SUMS (``A = fingers*w*l``,
    ``P = fingers*2*(w+l)``). The deck does NOT simplify the schematic side, so
    the reference must carry a SINGLE card that reproduces that same combined
    A/P. The reader computes ``A = W*L`` and ``P = 2*(W+L)`` from one card's
    W/L, so the equivalent (W, L) are the two roots of
    ``t**2 - (P/2)*t + A = 0`` (sum = P/2, product = A). Real roots exist
    because the as-drawn finger array has ``(P/2)**2 >= 4*A`` (a degenerate
    square cap is the equality). Returns ``(W, L)`` with ``W >= L``. No value is
    invented: A and P come from the declared finger geometry.
    """
    half_p = perim_um / 2.0
    disc = half_p * half_p - 4.0 * area_um2
    if disc < 0:
        # Numerically degenerate (square cap rounding); clamp to the double root.
        disc = 0.0
    root = math.sqrt(disc)
    w = (half_p + root) / 2.0
    l = (half_p - root) / 2.0
    return w, l


def _lower_cap_card(
    line: str, params: dict[str, str], cap_geometry: dict[str, Any]
) -> tuple[str | None, str]:
    """Lower a ``C`` card the layout realises as a MOS-cap to a ref cap device.

    ``cap_geometry`` maps a cap instance name (e.g. ``Cc`` / ``CC1``) to its
    as-drawn finger geometry ``{w, l, fingers[, model]}``. When the card's name
    has such a hint, the ideal 2-terminal ``C`` is lowered to the gf180 MOS-cap
    DEVICE (``pmoscap_3p3`` by default) whose combined A/P matches the extracted
    (combine_devices-folded) layout device — so it pairs 1:1 and LVS MATCHES.
    Without a hint the cap is genuinely unrepresentable from the .cir alone and
    the caller records it as honest ``cap_debt`` (never an invented device).
    """
    parts = line.split()
    if len(parts) < 4 or parts[0][0].upper() != "C":
        return None, "cap"
    name = parts[0]
    n1, n2 = parts[1], parts[2]
    hint = cap_geometry.get(name)
    if hint is None:
        return None, "cap"
    try:
        w_um = _parse_geom_um(_resolve_token(str(hint["w"]), params))
        l_um = _parse_geom_um(_resolve_token(str(hint["l"]), params))
        fingers = int(hint.get("fingers", 1))
    except (KeyError, ValueError, TypeError):
        return None, "cap"
    if w_um <= 0 or l_um <= 0 or fingers < 1:
        return None, "cap"
    model = _lvs_moscap_model_for(str(hint.get("model", ""))) or "pmoscap_3p3"
    area_um2 = fingers * w_um * l_um  # combine_devices sums finger area
    perim_um = fingers * 2.0 * (w_um + l_um)  # ... and finger perimeter
    w_eq, l_eq = _equivalent_cap_wl_um(area_um2, perim_um)
    inst = _spice_instance_name(name, "C")
    # Card form: ``<name> n1 n2 <model> W= L=`` — NO positional capacitance value.
    # The deck's reader (gf180mcu.lvs SubcircuitModelsReader) derives the MosCap
    # A/P from W/L and ignores C (the MosCap class disables it). A positional
    # value token between the nodes and the model is parsed as a THIRD net, which
    # trips the reader's "Capacitor needs two nodes" check, so it is omitted.
    # Emit the equivalent W/L at full double precision (.17g) so A=W*L /
    # P=2*(W+L) reproduce the combined extracted A/P to within machine epsilon
    # (the deck compares MosCap A/P at zero tolerance; a truncated root would
    # leave a residual the compare could reject).
    return (
        f"{inst} {n1} {n2} {model} W={w_eq:.17g}u L={l_eq:.17g}u",
        "cap",
    )


def _passive_bank_multiplier(
    name: str,
    parts: list[str],
    params: dict[str, str],
    fold_overrides: "dict[str, tuple[float, float]] | None" = None,
) -> int:
    """Return the integer parallel-bank multiplier materialized by layout.

    Parsed Clay sizing is the physical source of truth when available.  The
    fallback mirrors :func:`generic_placement.parse_cir_passive_records`: read
    an inline ``m=`` token, resolve parameters, round to an integer, and clamp
    to one.  The electrical transform itself stays centralized in
    :func:`clay.layout.passive_split.equivalent_passive_value`; this helper only
    recovers the bank count materialized by P&R.  Invalid or absent multipliers
    therefore keep legacy ``m=1`` bytes.
    """

    override = (fold_overrides or {}).get(name)
    raw: object = override[1] if override is not None else 1
    if override is None:
        for token in parts[4:]:
            key, sep, value = token.partition("=")
            if sep and key.strip().casefold() == "m":
                raw = value.strip()
                break
    try:
        resolved = _resolve_token(str(raw), params)
        return max(1, int(round(float(resolved))))
    except (TypeError, ValueError):
        return 1


def _lower_mim_cap_card(
    line: str,
    params: dict[str, str],
    profile: "_LvsRefProfile | None",
    fold_overrides: "dict[str, tuple[float, float]] | None" = None,
) -> "str | None":
    """Lower an ideal ``C`` card the layout realises as a PDK MIM-cap bank.

    The generic passive P&R draws ``m`` identical unit plates in parallel.  A
    full-extraction deck simplifies the bank into one device, so the reference
    must carry the same materialized total rather than the per-unit schematic
    value.  Plate dimensions are obtained from the same PDK builders and grid
    quantization used by layout; no circuit/name special case is involved.

    GF's foundry reader also derives A/P from W/L and compares them, so its card
    uses an equivalent rectangle preserving the summed bank area and perimeter.
    Clay's synthesized commercial reader compares the primary C and leaves
    reference A/P at zero, so commercial cards retain one canonical unit W/L and
    carry the bank total in C.  In both cases the MODEL token is the extracted
    class name and the capacitance is bound to the actual snapped plate area.

      * the MODEL token == the extracted class name (``mim_1p0fF``) so the two
        pair (a bare ``C n1 n2 <farads>`` reference creates an UNNAMED cap class
        that never pairs with the named ``mim_1p0fF`` device);
      * GF ``W/L`` preserve the simplified bank's total A/P exactly;
      * commercial ``W/L`` preserve the canonical unit plate;
      * ``C = density * materialized_total_area`` including grid quantization.

    Returns ``None`` when the PDK has no MIM stack (``profile.mim_density`` /
    ``profile.mim_model`` unset) so the caller records honest cap_debt -- never an
    invented device. Only reached AFTER the MOS-cap finger lowering declined (so a
    pmoscap-realised cap is never mislowered to a MIM device)."""
    mim_density = getattr(profile, "mim_density", None) if profile is not None else None
    mim_model = getattr(profile, "mim_model", None) if profile is not None else None
    if not (mim_density and mim_model):
        return None
    parts = line.split()
    if len(parts) < 4 or parts[0][0].upper() != "C":
        return None
    name, n1, n2, value = parts[0], parts[1], parts[2], parts[3]
    try:
        unit_farads = _spice_value_to_float(value, params)
    except ValueError:
        return None
    multiplier = _passive_bank_multiplier(name, parts, params, fold_overrides)
    inst = _spice_instance_name(name, "C")
    try:
        if getattr(profile, "pdk_id", None) == "gf180mcu":
            from clay.layout.passive_device_cell import mim_plate_side_nm
            from clay.layout.tech import LayoutTech

            _opt = getattr(profile, "mim_option", None)
            _cap_opt = getattr(profile, "mim_cap", None) or "1"
            _lvl = getattr(profile, "metal_level", None)
            _tech = (
                LayoutTech.resolve(
                    "gf180mcu",
                    mim_option=_opt,
                    cap_option=_cap_opt,
                    metal_level=_lvl,
                )
                if (_opt and _opt != "A") or (_cap_opt != "1") or _lvl
                else LayoutTech.resolve("gf180mcu")
            )
            side_um = mim_plate_side_nm(unit_farads, _tech, str(_cap_opt)) / 1000.0
        else:
            # Commercial builders and their full-extraction decks consume the
            # same adapter/foundry density. Grid snapping is verified under the
            # deck's explicit 5% value tolerance.
            side_um = math.sqrt(unit_farads / float(mim_density))
    except Exception:  # noqa: BLE001
        side_um = None
    if side_um is not None and side_um > 0:
        # P&R tiles ``m`` equal unit caps.  Extraction simplifies that parallel
        # bank into one device whose A and P are the sums of the unit plates.
        # Emit the equivalent W/L roots so the reference preserves all three
        # compared quantities (C, A, P), not merely total capacitance.
        area_um2 = multiplier * side_um * side_um
        perim_um = multiplier * 4.0 * side_um
        if getattr(profile, "pdk_id", None) == "gf180mcu":
            # GF's reference reader derives A/P from W/L, so preserve the
            # simplified bank's summed plate geometry exactly.
            width_um, length_um = _equivalent_cap_wl_um(area_um2, perim_um)
        else:
            # The generated commercial-deck reader compares the primary C and
            # leaves A/P at zero on the reference side.  Keep the established
            # square unit-plate W/L tokens; total C below carries the bank.
            width_um = length_um = side_um
        # The generated full-extraction deck measures the MATERIALIZED plate,
        # including manufacturing-grid growth.  Bind the reference C to that
        # same area so the comparison remains exact rather than relying on a
        # tolerance to hide deterministic grid quantization.
        farads = float(mim_density) * area_um2
        return (
            f"{inst} {n1} {n2} {mim_model} "
            f"W={width_um:.17g}u L={length_um:.17g}u C={farads:.17g}"
        )
    # Builder side unavailable (no tech) -> still emit the model token so the
    # class pairs; the A/P default to 0 in the reader (C-only, geometry-blind).
    from clay.layout.passive_split import equivalent_passive_value

    total_farads = equivalent_passive_value("cap", unit_farads, multiplier, "parallel")
    return f"{inst} {n1} {n2} {mim_model} C={total_farads:.17g}"


def _grow_equal_size_cap_plates(
    placement: Any,
    cap_records: dict[str, dict[str, Any]],
    pdk_id: Any,
    *,
    mim_option: str | None = None,
    cap_option: str | None = None,
    metal_level: str | None = None,
) -> int:
    """Make every MIM cap in a declared ``@equal_size`` group EMIT an equal plate.

    The placer's ``@equal_size`` (``_apply_equal_size``) only grows the reserved
    BBOX footprint -- the drawn MIM plate is left at each cap's value-derived size,
    so two caps declared ``@equal_size(Cc, CL)`` with different values (1.65p / 3p)
    emit UNEQUAL FuseTop plates. This closes that: for each group, the common plate
    side is the group-wise MAXIMUM natural plate side (the largest-value member);
    every smaller member's MIM cap is REBUILT with that override side so its plate
    GROWS to match (never shrinks -> capacitance preserved-or-increased, DRC-legal).

    Rebuilds the full :class:`PlacedDevice` (polys + terminals + entry layers +
    bbox) via the SAME ``build_capacitor`` + lower-left translation the placer used,
    re-centred on the placed device's (already equal-size-grown) bbox CENTROID so a
    mirror pair stays balanced, then splices the new polys into ``placement.polys``
    in place of the old ones. The card's electrical C annotation (LVS reference) is
    untouched -- only the drawn plate grows.

    Generic + circuit-agnostic: driven purely by ``placement.constraints
    .equal_size_groups`` membership + each cap's natural plate side, zero
    circuit-name branch. No-op (returns 0) when no group has >=2 buildable MIM-cap
    members, so a cell that declares no ``@equal_size`` (or whose group is all MOS /
    resistors) is byte-unchanged. Returns the number of caps regrown.
    """
    constraints = getattr(placement, "constraints", None)
    groups = tuple(getattr(constraints, "equal_size_groups", ()) or ())
    if not groups:
        return 0
    devices = getattr(placement, "devices", None)
    if not devices:
        return 0

    from clay.layout.backend import _translate
    from clay.layout.passive_device_cell import (
        build_capacitor_for_tech,
        equal_size_cap_plate_side_nm,
        mim_plate_side_nm,
    )
    from clay.layout.generic_placement import PlacedDevice, _terminal_entry_layer
    from clay.layout.tech import LayoutTech

    try:
        tech = (
            pdk_id
            if isinstance(pdk_id, LayoutTech)
            else LayoutTech.resolve(
                pdk_id,
                **{
                    k: v
                    for k, v in (
                        ("mim_option", mim_option),
                        ("cap_option", cap_option),
                        ("metal_level", metal_level),
                    )
                    if v is not None
                },
            )
        )
    except Exception:  # noqa: BLE001 - unknown PDK -> no MIM stack, leave unchanged
        return 0

    eff_mim = mim_option or getattr(tech, "mim_option", None)
    eff_lvl = metal_level or getattr(tech, "metal_level", None)
    cap_option = cap_option or getattr(tech, "cap_option", None) or "1"  # gf180 default flavour

    def _cap_value(label: str):
        rec = cap_records.get(label)
        if rec is None or str(rec.get("kind", "")) != "cap":
            return None
        try:
            return float(rec["value"])
        except (KeyError, TypeError, ValueError):
            return None

    grown = 0
    for group in groups:
        # The cap members of THIS group with a known farads value (a group may mix
        # MOS / resistor members -- only the caps get a common MIM plate).
        members = [
            lbl
            for lbl in group
            if lbl in devices
            and getattr(devices[lbl], "kind", "") == "cap"
            and _cap_value(lbl) is not None
        ]
        if len(members) < 2:
            continue
        values = [_cap_value(lbl) for lbl in members]
        common_side = equal_size_cap_plate_side_nm(values, tech, cap_option)
        if common_side <= 0:
            continue
        for lbl in members:
            pd = devices[lbl]
            val = _cap_value(lbl)
            natural = mim_plate_side_nm(float(val), tech, cap_option)
            if natural >= common_side:
                continue  # already the group max -> nothing to grow
            rec = cap_records.get(lbl) or {}
            orient = str(rec.get("orient", "R0"))
            pins = rec.get("pins") if isinstance(rec.get("pins"), dict) else {}
            mim_kw = {}
            if eff_mim:
                mim_kw["mim_option"] = eff_mim
            if eff_lvl:
                mim_kw["metal_level"] = eff_lvl
            geom = build_capacitor_for_tech(
                tech,
                float(val),
                cap_option=cap_option,
                orient=orient,
                nets=pins,
                plate_side_override_nm=common_side,
                **mim_kw,
            )
            # Re-centre the rebuilt (larger) geom on the placed device's bbox
            # centroid so a mirror pair keeps its equal offsets about the axis.
            ex = geom.extent()
            gcx = (ex[0] + ex[2]) // 2
            gcy = (ex[1] + ex[3]) // 2
            ox0, oy0, ox1, oy1 = pd.bbox
            pcx = (ox0 + ox1) // 2
            pcy = (oy0 + oy1) // 2
            dx = pcx - gcx
            dy = pcy - gcy
            new_polys = _translate(list(geom.polygons), dx, dy)

            # Splice: drop the old device polys (by identity) from the cell poly
            # list, append the new ones. ``pd.polys`` shares objects with
            # ``placement.polys`` (both reference the placer's translated list).
            old_ids = {id(lp) for lp in pd.polys}
            placement.polys[:] = [lp for lp in placement.polys if id(lp) not in old_ids]
            placement.polys.extend(new_polys)

            # Rebuild terminals (centres) + entry layers off the re-centred geom.
            metal_keys: list = []
            for lvl in (1, 2, 3, 4, 5):
                try:
                    mk = tech.metal_key(lvl)
                except Exception:  # noqa: BLE001
                    mk = None
                metal_keys.append(tuple(mk) if mk is not None else None)
            terms: dict[str, Any] = {}
            term_layers: dict[str, Any] = {}
            for role in ("p", "n"):
                t = geom.terminals.get(role)
                if t is None:
                    continue
                tb = (t.bbox[0] + dx, t.bbox[1] + dy, t.bbox[2] + dx, t.bbox[3] + dy)
                terms[role] = ((tb[0] + tb[2]) // 2, (tb[1] + tb[3]) // 2)
                entry = _terminal_entry_layer(tb, new_polys, metal_keys)
                if entry is not None:
                    term_layers[role] = entry

            nex = geom.extent()
            new_bbox = (nex[0] + dx, nex[1] + dy, nex[2] + dx, nex[3] + dy)
            devices[lbl] = PlacedDevice(
                label=pd.label,
                kind=pd.kind,
                bbox=new_bbox,
                terminals=terms,
                polys=new_polys,
                term_layers=term_layers,
                geometry_owner=pd.geometry_owner,
                terminal_accesses=pd.terminal_accesses,
                orientation=pd.orientation,
                physical_macro_graph_sha256=pd.physical_macro_graph_sha256,
            )
            grown += 1
    return grown


def _parse_params(source_text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    assignment = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?:'([^']*)'|\"([^\"]*)\"|(\{[^}]*\}|[^\s]+))"
    )
    continuing = False
    for raw in source_text.splitlines():
        line = raw.split("$", 1)[0].strip()
        if re.match(r"(?i)^\.param(?:\s|$)", line):
            payload = re.sub(r"(?i)^\.param\s*", "", line, count=1)
            continuing = True
        elif continuing and line.startswith("+"):
            payload = line[1:].strip()
        else:
            continuing = False
            continue
        for match in assignment.finditer(payload):
            name = match.group(1)
            value = next(group for group in match.groups()[1:] if group is not None)
            params[name] = value
            params[name.casefold()] = value
    return params


def _resolve_token(token: str, params: dict[str, str]) -> str:
    # Strip the ngspice brace-expression wrapper (``{W_BIAS_N}``) so a parameter
    # reference resolves against the bare ``.param`` key. A non-brace literal is
    # looked up directly (and falls through to itself when not a parameter).
    stripped = token.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1].strip()
    return params.get(stripped, params.get(stripped.casefold(), stripped))


def _split_number_suffix(value: str) -> tuple[float, str]:
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z]*)", value
    )
    if not match:
        raise ValueError(f"cannot parse SPICE numeric token {value!r}")
    return float(match.group(1)), match.group(2)


#: SPICE engineering-suffix multipliers, applied when lowering a passive VALUE
#: (R in ohms, C in farads) to the absolute number the KLayout extractor emits.
#: ``meg`` must be tested before ``m`` (longest-match) so "1meg" is not read as
#: "1m" followed by "eg". Geometry tokens (W/L in microns) deliberately do NOT
#: use this table — they keep their bare ``u``/``n`` suffix per the scale=1.0u
#: deck convention; only the device-value column is normalised here.
_SPICE_SUFFIX_MULT: tuple[tuple[str, float], ...] = (
    ("meg", 1e6),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
)


def _spice_value_to_float(value: str, params: dict[str, str]) -> float:
    """Resolve a SPICE value token (``510k``, ``1.2p``, ``1697.5k``) to a float.

    Unlike :func:`_split_number_suffix` (which preserves a geometry suffix for
    the micron-scaled W/L column) this collapses the engineering suffix to its
    numeric multiplier so a passive's VALUE is the absolute ohms/farads the
    KLayout LVS extractor reports. ``.param`` tokens resolve first.
    """
    token = _resolve_token(value, params).strip()
    low = token.lower()
    for suffix, mult in _SPICE_SUFFIX_MULT:
        if low.endswith(suffix):
            head = low[: -len(suffix)]
            # guard against a bare exponent ("1e3") whose trailing char is a
            # suffix letter we should not strip
            try:
                return float(head) * mult
            except ValueError:
                break
    return float(low)


def _trim_num(x: float) -> str:
    """Render a numeric fold factor (``nf`` / ``m``) as a clean dimensionless
    token: an integral float drops the ``.0`` (``8.0`` -> ``"8"``), a fractional
    one keeps a compact form. Used to feed a LAYOUT-derived nf/m into the same
    ``_format_scaled`` path the ``.cir`` nf/m tokens take."""
    xf = float(x)
    return str(int(round(xf))) if abs(xf - round(xf)) < 1e-9 else f"{xf:.12g}"


def _layout_fold_overrides(cell_def: Any) -> dict[str, tuple[float, float]]:
    """Lift each LAYOUT device's ``(nf, m)`` from a parsed cell, keyed by its
    SPICE instance name -- the fold factors the placer actually emitted geometry
    with.

    The compile step stores the parsed ``design.clay`` cell in
    ``ctx.extra["cell_def"]``; its devices carry ``instance_name`` + a ``sizing``
    dict (``{'W','L','nf','M'}``). This returns ``{instance_name -> (nf, m)}`` so
    the LVS reference folds W by the layout's finger count, keeping the reference
    faithful to the extracted geometry even when the schematic ``.cir`` declares a
    different nf. Returns ``{}`` for a missing cell / device list (no override ->
    the ``.cir`` folding is byte-unchanged). Purely structural: it reads the
    cell's own sizing data, no circuit literal."""
    out: dict[str, tuple[float, float]] = {}
    devices = getattr(cell_def, "devices", None) or []
    for d in devices:
        name = getattr(d, "instance_name", None)
        sizing = getattr(d, "sizing", None) or {}
        if not name or not isinstance(sizing, dict):
            continue
        if "nf" not in sizing and "M" not in sizing:
            continue
        try:
            nf = float(sizing.get("nf", 1) or 1)
            m = float(sizing.get("M", 1) or 1)
        except (TypeError, ValueError):
            continue
        out[str(name)] = (nf, m)
    return out


def _format_scaled(value: str, scale: str | float, params: dict[str, str]) -> str:
    base_value = _resolve_token(value, params)
    scale_value = _resolve_token(str(scale), params)
    base, suffix = _split_number_suffix(base_value)
    mult, mult_suffix = _split_number_suffix(scale_value)
    if mult_suffix:
        raise ValueError(
            f"device multiplier must be dimensionless, got {scale_value!r}"
        )
    return f"{base * mult:.12g}{suffix}"


def _format_geometry(value: str, scale: str | float, params: dict[str, str]) -> str:
    """Like :func:`_format_scaled` but guarantees a micron unit on the W/L token.

    The gf180 ``.cir`` decks that set ``.option scale=1.0u`` write W/L as BARE
    microns (``W=20`` == 20 um); a KLayout SPICE *reference* reader has no such
    scale and treats a unit-less ``20`` as 20 METERS, so the device geometry
    would never match the extractor's ``W=20U``. When the source token already
    carries an explicit unit (``W=3.3u`` in fc_ota) the suffix is preserved
    verbatim; only a unit-less token gets the ``u`` micron suffix appended.
    """
    formatted = _format_scaled(value, scale, params)
    _, suffix = _split_number_suffix(formatted)
    if not suffix:
        formatted += "u"
    return formatted


#: Substrate / well net, poly-R model and sheet-rho are now PDK DATA, resolved
#: per-PDK by :func:`_lvs_ref_profile` (substrate from ``substrate_net()``,
#: poly-R from ``lvs_reference_models()['poly_res']`` / ``poly_res_model``,
#: sheet-rho from ``poly_sheet_rho_ohm_sq``), falling back to the gf180 values
#: above (``_GF180_SUBSTRATE_NET`` / ``_GF180_POLY_RES_REF`` /
#: ``_GF180_POLY_SHEET_RHO``) wherever the adapter is silent -- so the gf180
#: ppolyf_u reference card stays byte-for-byte unchanged. The deck binds a
#: poly-over-substrate resistor's third (bulk) terminal to the substrate net the
#: LvsStep runs with (``lvs_sub``); the reference must carry the same third
#: terminal or the 3-terminal ``resistor_with_bulk`` device class will not
#: match. The deck's ``BResistor`` class DISABLES the ``R`` parameter and
#: compares poly resistors on GEOMETRY (W, L) only -- so the reference carries
#: the laid-out serpentine length, not the ohmic value. The layout draws every
#: poly-R unit at W = 1 um, so L (in um) = R / sheet_rho for a 1-um strip; series
#: unit cells then combine additively under ``netlist.simplify`` (two series 50 k
#: cells -> one 100 k device, L 142.86 + 142.86 = 285.7 um, matching the
#: extractor). The width is the laid-out 1-um unit-cell strip (not PDK-varied
#: here); resistance and length are exact, never invented.
_REF_RES_WIDTH_UM = 1.0


def _spice_instance_name(raw_name: str, class_letter: str) -> str:
    """Return an instance name whose leading letter is the SPICE device class.

    A KLayout SPICE *reference* reader keys the device class off the instance
    card's first letter: ``M``=MOSFET, ``R``=resistor, ``C``=capacitor,
    ``Q``=BJT, ``D``=diode; a leading ``X`` is read as a SUBCKT call, not a
    device. The source ``.cir`` writes every transistor as an ``x``-instance of
    the gf180 device subckt (``xMP1 .. nmos_3p3``); to make the reference a
    flat device the ``x`` wrapper is dropped and the correct class letter is
    guaranteed at the front (``xMP1`` -> ``MMP1``, ``xQ1`` -> ``QQ1``). Names
    that already start with the class letter are kept verbatim.
    """
    name = raw_name
    if name[:1].lower() == "x":
        name = name[1:]
    if name[:1].upper() != class_letter.upper():
        name = class_letter + name
    return name


def _lower_device_card(
    line: str,
    params: dict[str, str],
    profile: "_LvsRefProfile | None" = None,
    fold_overrides: "dict[str, tuple[float, float]] | None" = None,
    device_model_overrides: "Mapping[str, str] | None" = None,
) -> tuple[str | None, str]:
    """Lower one schematic device card to its PDK LVS-comparable form.

    Returns ``(lowered_line_or_None, device_class)`` where ``device_class`` is
    one of ``"mos"``, ``"res"``, ``"cap"``, ``"bjt"`` or ``"unknown"``.

    ``profile`` carries the per-PDK device-model names, poly-R model, sheet-rho
    and substrate net (see :func:`_lvs_ref_profile`); ``profile=None`` keeps the
    historical gf180 names so the gf180 reference card is byte-for-byte
    unchanged.

    Recognised forms (no per-circuit literals — dispatch is purely structural):

    * **MOS** — ``M<name> d g s b <model> [W= L= m=]`` *and* the subckt-wrapped
      ``x<name> d g s b <mos_model> [W= L= m=]`` form (two_stage / bandgap write
      every transistor as an ``x``-instance of the device subckt). The ``m=``
      multiplier is folded into ``W`` so the reference carries effective width
      and no ``m=``; ``L`` resolves through ``.param``. The lowered model is the
      PDK's nmos/pmos LVS model (``nmos_3p3`` / ``pmos_3p3`` for gf180,
      ``nmos_3p3`` / ``pmos_3p3`` for external_pdk).
    * **Resistor** — ``R<name> n1 n2 <value>``. Lowered to the 3-terminal
      poly-R form ``<name> n1 n2 <bulk> <ohms> <poly_res>`` the deck's
      ``resistor_with_bulk`` extractor produces; the engineering suffix in the
      value (``510k``) is collapsed to absolute ohms. ``poly_res``, sheet-rho
      and the bulk (substrate) net come from ``profile``.
    * **BJT** — ``x<name> c b e <vpnp_*|vnpn_*>``. Passed through with the gf180
      BJT model verbatim (``<name> c b e <model>``); the deck binds the laid-out
      vertical PNP/NPN to the same model.
    * **Capacitor** — ``C<name> n1 n2 <value>``. Returned as ``device_class
      "cap"`` with ``None`` lowering: a MOS-cap / pmoscap is laid out as a
      transistor or a ``pmoscap_3p3`` device whose finger geometry the ideal
      2-terminal ``C .. <value>`` card does NOT carry, so it cannot be lowered
      losslessly. The LvsStep reports it as schematic/layout cap-device debt
      rather than inventing matching geometry.
    """
    res_bulk = profile.substrate if profile is not None else _GF180_SUBSTRATE_NET
    res_model = profile.poly_res if profile is not None else _GF180_POLY_RES_REF
    sheet_rho = profile.sheet_rho if profile is not None else _GF180_POLY_SHEET_RHO
    parts = line.split()
    if not parts:
        return None, "unknown"
    head = parts[0]
    first = head[0].upper()

    # --- resistor: R<name> n1 n2 <value> ---------------------------------- #
    if first == "R" and len(parts) >= 4:
        name, n1, n2, value = parts[0], parts[1], parts[2], parts[3]
        try:
            ohms = _spice_value_to_float(value, params)
        except ValueError:
            return None, "unknown"
        from clay.layout.passive_split import equivalent_passive_value

        ohms = equivalent_passive_value(
            "res",
            ohms,
            _passive_bank_multiplier(name, parts, params, fold_overrides),
            "parallel",
        )
        inst = _spice_instance_name(name, "R")
        # The deck compares the poly-R on GEOMETRY (W, L), not R. A 1-um-wide
        # poly strip of resistance R has L = R / sheet_rho squares; emit that
        # length so the reference device's W/L matches the extracted serpentine
        # (series unit cells combine additively under netlist.simplify). R is
        # carried too (the delegate reads params["R"]) though the device class
        # disables it -- harmless and keeps the card self-documenting.
        length_um = (ohms / sheet_rho) * _REF_RES_WIDTH_UM
        if profile is not None and profile.pdk_id:
            try:
                from clay.layout.passive_device_cell import (
                    materialized_resistor_body_length_nm,
                )
                from clay.layout.tech import LayoutTech

                materialized_nm = materialized_resistor_body_length_nm(
                    ohms,
                    LayoutTech.resolve(profile.pdk_id),
                    w_um=_REF_RES_WIDTH_UM,
                )
                length_um = materialized_nm / 1000.0
            except Exception:  # noqa: BLE001 - unsupported PDK keeps ideal debt visible
                pass
        # The physical reference uses the same recognized sheet resistance
        # and quantized body dimensions as the declared materialization. The
        # source's nominal component value is unchanged. This is the same
        # rule used by _expand_segmented_reference_resistors below; carrying
        # nominal R beside quantized W/L makes strict same-GDS PEX inconsistent.
        physical_ohms = sheet_rho * length_um / _REF_RES_WIDTH_UM
        # Card form mirrors the extractor's 3-terminal resistor card exactly:
        # ``<name> n1 n2 <bulk> <value> <poly_res> W= L=`` -- the positional value
        # populates params["R"] (the deck's delegate reads params["R"]; the bare
        # value alone would leave it nil and crash the reader), and W/L carry the
        # serpentine geometry the BResistor class actually compares on.
        if profile is not None and not profile.resistor_has_bulk:
            return (
                f"{inst} {n1} {n2} {physical_ohms:.12g} {res_model} "
                f"W={_REF_RES_WIDTH_UM:g}u L={length_um:.6g}u",
                "res",
            )
        return (
            f"{inst} {n1} {n2} {res_bulk} {physical_ohms:.12g} {res_model} "
            f"W={_REF_RES_WIDTH_UM:g}u L={length_um:.6g}u",
            "res",
        )

    # --- capacitor: C<name> n1 n2 <value> --------------------------------- #
    if first == "C" and len(parts) >= 4:
        # An ideal 2-terminal C cannot be lowered HERE: the layout may realise it
        # either as a foldable MOS-cap (pmoscap_3p3, needing the as-drawn finger
        # geometry the .cir does not carry) OR as a gf180 MIM cap (mim_<d>fF). The
        # dispatch decision needs the per-cap layout geometry hint, which lives in
        # the WRITER (write_effective_lvs_reference), so this returns ``(None,
        # "cap")`` and the writer chooses: MOS-cap finger lowering when a
        # cap_geometry hint is present (the layout drew a pmoscap), else the MIM
        # lowering when the PDK has a MIM stack (the layout drew a MIM cap), else
        # honest cap_debt. Keeping MIM out of here preserves the pmoscap path's
        # precedence (a Miller Cc realised as a MOS-cap must NOT be mislowered to
        # a mim_<d>fF device).
        return None, "cap"

    # --- MOS, both plain M<name> and subckt-wrapped x<name> .. <mos_model> - #
    if first == "M" or first == "X":
        # locate the model token: first token (after the instance name + nodes)
        # that maps to a gf180 device. For an x-instance the model is the token
        # immediately preceding the parameter list; for an M-card it is parts[5].
        param_text = ""
        model = None
        nodes: list[str] = []
        # split nodes / model / params: params begin at the first key=value tok
        kv_start = len(parts)
        for i, tok in enumerate(parts[1:], start=1):
            if "=" in tok:
                kv_start = i
                break
        head_tokens = parts[1:kv_start]  # nodes... + model (last)
        param_text = " ".join(parts[kv_start:])
        if len(head_tokens) >= 2:
            model = head_tokens[-1]
            nodes = head_tokens[:-1]

        # BJT (x-wrapped vpnp/vnpn): no W/L, 3 collector/base/emitter nodes
        if model is not None:
            bjt_model = _lvs_bjt_model_for(model, profile)
            if bjt_model is not None and len(nodes) >= 3:
                c, b, e = nodes[0], nodes[1], nodes[2]
                inst = _spice_instance_name(parts[0], "Q")
                binding = _lvs_bjt_binding_for(model, profile)
                parallel_parameter = str(
                    getattr(binding, "lvs_parallel_parameter", "NE") or "NE"
                )
                multiplier = None
                match = re.search(
                    rf"\b{re.escape(parallel_parameter)}\s*=\s*"
                    r"(\{[^}]*\}|[A-Za-z0-9_.+-]+)",
                    param_text,
                    flags=re.IGNORECASE,
                )
                if match is not None:
                    try:
                        multiplier = _spice_value_to_float(match.group(1), params)
                    except ValueError:
                        multiplier = None
                suffix = ""
                if multiplier is not None:
                    rounded = int(round(multiplier))
                    if multiplier <= 0.0 or abs(multiplier - rounded) > 1e-9:
                        return None, "unknown"
                    suffix = f" {parallel_parameter}={rounded}"
                return f"{inst} {c} {b} {e} {bjt_model}{suffix}", "bjt"

        lvs_model = _lvs_model_for(model, profile) if model is not None else None
        if lvs_model is None or len(nodes) < 4:
            return None, "unknown"
        instance_name = _spice_instance_name(parts[0], "M")
        override = (device_model_overrides or {}).get(parts[0]) or (
            device_model_overrides or {}
        ).get(instance_name)
        if override:
            lvs_model = str(override)
        drain, gate, source, bulk = nodes[0], nodes[1], nodes[2], nodes[3]
        dev_params = {
            key.upper(): value
            for key, value in re.findall(
                r"\b(W|L|m|nf)=(\{[^}]*\}|[A-Za-z0-9_.+-]+)",
                param_text,
                flags=re.IGNORECASE,
            )
        }
        if "W" not in dev_params or "L" not in dev_params:
            return None, "unknown"
        # Fold BOTH the ``m`` multiplier AND the ``nf`` finger count into the
        # effective width. The generic device cell builds a MOS as ``nf`` parallel
        # poly fingers, each the full card ``W`` (device_cell._build_device_geom
        # draws every finger at ``w_nm``), so the layout extractor reports the
        # TOTAL channel width ``W * nf`` (and ``* m`` for paralleled instances).
        # The reference card must carry that same total or the extractor's wider
        # device fails the deck's MOS W comparison and the whole netlist match is
        # rejected. ``m``/``nf`` absent default to 1 -> a card without folding is
        # byte-unchanged. This is purely structural: any cell's ``nf=`` device
        # lowers the same way (no per-circuit literal).
        #
        # SOURCE-OF-TRUTH SYNC: the extracted device W is determined by the nf/m the
        # LAYOUT was built with (the parsed ``design.clay`` cell, which drives the
        # placer), NOT by the schematic ``.cir``'s nf/m. When the two source files
        # disagree on a device's finger count (a real hazard: the ``.cir`` and the
        # ``.clay`` are independently maintained), folding the reference by the
        # ``.cir`` nf yields a W the extracted layout never has -> spurious LVS W
        # mismatch. ``fold_overrides`` (instance_name -> (nf, m)) carries the LAYOUT
        # cell's per-device finger/multiplicity, so the reference folds by the SAME
        # nf the geometry was emitted with. Keyed by the card's instance name; a
        # device absent from the map (or no map at all) keeps the ``.cir`` nf/m, so
        # every in-sync circuit's reference is byte-unchanged. Purely structural --
        # the override is net/sizing data from the layout cell, no circuit literal.
        ov = (fold_overrides or {}).get(parts[0]) or (fold_overrides or {}).get(
            _spice_instance_name(parts[0], "M")
        )
        if ov is not None:
            ov_nf, ov_m = ov
            width_scale = _format_scaled(_trim_num(ov_m), _trim_num(ov_nf), params)
        else:
            width_scale = _format_scaled(
                dev_params.get("M", "1"), dev_params.get("NF", "1"), params
            )
        width = _format_geometry(dev_params["W"], width_scale, params)
        length = _format_geometry(dev_params["L"], "1", params)
        # Mirror the target PDK's legal drawn-dimension clamp already used by
        # the device-cell materializer.  This reuses the canonical PDK-dimension
        # table from lvs_reference_emit: a migrated 0.18um device that GF180
        # physically draws at 0.28um must have L=0.28um in the effective
        # reference as well.  Preserve the original tokens byte-for-byte when
        # they are already legal or when no target-PDK minimum is registered.
        from clay.layout.lvs_reference_emit import _pdk_drawn_minimums

        min_l_um, min_w_um = _pdk_drawn_minimums(
            profile.pdk_id if profile is not None else None
        )
        if min_l_um is not None:
            drawn_l_um = _spice_value_to_float(length, {}) * 1e6
            if drawn_l_um < min_l_um - 1e-12:
                length = f"{_trim_num(min_l_um)}u"
        if min_w_um is not None:
            per_width = _format_geometry(dev_params["W"], "1", params)
            drawn_w_um = _spice_value_to_float(per_width, {}) * 1e6
            if drawn_w_um < min_w_um - 1e-12:
                width = f"{_trim_num(min_w_um * float(width_scale))}u"
        inst = _spice_instance_name(parts[0], "M")
        return (
            f"{inst} {drain} {gate} {source} {bulk} "
            f"{lvs_model} W={width} L={length}",
            "mos",
        )

    return None, "unknown"


def _parse_geom_um(token: str) -> float:
    """Parse a ``12.5u`` / ``142.857u`` geometry token to a float in microns."""
    val, suffix = _split_number_suffix(token)
    if suffix.lower() in ("u", "um", ""):
        return val
    # any other unit is unexpected here; fall back to the bare magnitude
    return val


def _combine_reference_devices(
    body: list[str],
    ports: list[str],
    *,
    combine_series_resistors: bool = True,
) -> list[str]:
    """Series/parallel-combine the lowered reference cards to match extraction.

    The gf180 LVS deck runs ``netlist.simplify`` on the LAYOUT but NOT on the
    schematic (``schematic.simplify`` is gated behind ``-rd schematic_simplify``,
    which the signoff harness does not set), and the layout extractor reports
    parallel fingers / series unit-cells as ONE combined device. The schematic
    reference must therefore be pre-combined the same way or it mismatches the
    extracted (already-merged) netlist. This pass mirrors ``simplify`` for the
    device classes the production circuits fold:

    * **Parallel MOS** — cards with the same model and the same
      ``(gate, source, bulk, L)`` whose drains are also identical are summed in
      ``W`` into a single device (8 parallel pass-FET fingers -> one W=800).
    * **Parallel MIM capacitors** — cards with the same model and unordered
      terminal pair are folded into one device.  Capacitance is summed while an
      equivalent W/L preserves the exact sum of plate area and perimeter.
    * **Series resistors** — a ``ppolyf_u`` chain through an INTERNAL node
      (not a subckt port) of degree exactly two, joining two resistors, folds
      into one resistor whose ``L`` is the sum (two 50 k unit cells -> one 100 k
      device). Iterated to collapse longer chains.

    Cards that are not MOS/ppolyf_u (e.g. BJTs) pass through untouched. No value
    is invented: combined ``W``/``L`` are exact sums of the source geometry.
    """

    def split_card(card: str):
        toks = card.split()
        name = toks[0]
        cls = name[0].upper()
        return name, cls, toks

    # --- parallel-MOS merge ------------------------------------------------ #
    # Group MOS cards with identical (drain, gate, source, bulk, model, L); a
    # group of size > 1 is a parallel-finger array that the extractor reports as
    # one device (sum of W). A SINGLETON group is kept VERBATIM so an unmerged
    # device retains its original instance name (the reference is otherwise
    # unchanged for the MOSFET-only circuits).
    mos_groups: dict[tuple, list[str]] = {}
    order: list[tuple | str] = []
    for card in body:
        name, cls, toks = split_card(card)
        if cls == "M" and len(toks) >= 6 and "W=" in card and "L=" in card:
            d, g, s, b, model = toks[1], toks[2], toks[3], toks[4], toks[5]
            kv = dict(re.findall(r"\b(W|L)=([0-9.eE+\-]+u?)", card))
            key = ("MOS", d, g, s, b, model, kv.get("L", ""))
            if key not in mos_groups:
                mos_groups[key] = []
                order.append(key)
            mos_groups[key].append(card)  # keep the full card for singletons
        else:
            order.append("__other__:" + card)

    merged: list[str] = []
    for item in order:
        if isinstance(item, tuple):
            _, d, g, s, b, model, lval = item
            cards_in_group = mos_groups[item]
            if len(cards_in_group) == 1:
                merged.append(cards_in_group[0])  # verbatim — preserve name
                continue
            total_w = sum(
                _parse_geom_um(
                    dict(re.findall(r"\b(W)=([0-9.eE+\-]+u?)", c)).get("W", "0u")
                )
                for c in cards_in_group
            )
            name = f"M{d}_{g}_{model}".replace(".", "p")
            merged.append(f"{name} {d} {g} {s} {b} {model} W={total_w:.6g}u L={lval}")
        else:
            merged.append(item.split(":", 1)[1])

    # --- parallel-MIM merge ----------------------------------------------- #
    # KLayout simplifies parallel extracted MIM devices into one device whose
    # A/P are the sums of the physical plates.  The reference reader derives
    # those properties from W/L, so preserve both sums through the same
    # equivalent-rectangle transform used for a folded MOS capacitor.
    mim_groups: dict[tuple, list[str]] = {}
    mim_order: list[tuple | str] = []
    for card in merged:
        toks = card.split()
        fields = dict(re.findall(r"\b(W|L|C)=([0-9.eE+\-]+u?)", card))
        if (
            toks[0][0].upper() == "C"
            and len(toks) >= 4
            and {"W", "L", "C"}.issubset(fields)
        ):
            n1, n2, model = toks[1], toks[2], toks[3]
            lo, hi = sorted((n1, n2))
            key = ("MIM", lo, hi, model)
            if key not in mim_groups:
                mim_groups[key] = []
                mim_order.append(key)
            mim_groups[key].append(card)
        else:
            mim_order.append("__other__:" + card)

    mim_merged: list[str] = []
    for item in mim_order:
        if not isinstance(item, tuple):
            mim_merged.append(item.split(":", 1)[1])
            continue
        _, n1, n2, model = item
        group = mim_groups[item]
        if len(group) == 1:
            mim_merged.append(group[0])
            continue
        area_um2 = 0.0
        perimeter_um = 0.0
        capacitance_f = 0.0
        for card in group:
            fields = dict(re.findall(r"\b(W|L|C)=([0-9.eE+\-]+u?)", card))
            width_um = _parse_geom_um(fields["W"])
            length_um = _parse_geom_um(fields["L"])
            area_um2 += width_um * length_um
            perimeter_um += 2.0 * (width_um + length_um)
            capacitance_f += _spice_value_to_float(fields["C"], {})
        width_um, length_um = _equivalent_cap_wl_um(area_um2, perimeter_um)
        name = f"C{n1}_{n2}_{model}".replace(".", "p")
        mim_merged.append(
            f"{name} {n1} {n2} {model} "
            f"W={width_um:.17g}u L={length_um:.17g}u "
            f"C={capacitance_f:.17g}"
        )
    merged = mim_merged

    # --- parallel-BJT merge (vpnp/vnpn: same C/B/E -> one NE=count device) - #
    # The layout draws N parallel emitters as ONE multi-emitter device with
    # ``NE=N`` (e.g. the 8x DeltaVbe leg extracts as a single vpnp NE=8); with
    # device-combine OFF the schematic side must pre-fold the N discrete cards
    # into one ``Q c b e model NE=N`` card or it mismatches the NE=N extraction.
    bjt_groups: dict[tuple, list[str]] = {}
    bjt_order: list[tuple] = []
    non_bjt: list[str] = []
    for card in merged:
        toks = card.split()
        if toks[0][0].upper() == "Q" and len(toks) >= 5:
            c, b, e, model = toks[1], toks[2], toks[3], toks[4]
            key = ("BJT", c, b, e, model)
            if key not in bjt_groups:
                bjt_groups[key] = []
                bjt_order.append(key)
            bjt_groups[key].append(card)
        else:
            non_bjt.append(card)
    bjt_cards: list[str] = []
    for key in bjt_order:
        _, c, b, e, model = key
        group = bjt_groups[key]
        if len(group) == 1:
            bjt_cards.append(group[0])  # verbatim — preserve single-device name
            continue
        name = f"Q{c}_{b}_{e}_{model}".replace(".", "p")
        bjt_cards.append(f"{name} {c} {b} {e} {model} NE={len(group)}")
    merged = non_bjt + bjt_cards

    # A compiled Clay cell already declares the physical resistor instances.
    # Preserve that topology; implicit series folding is only for legacy CIR-only
    # inputs that have no layout-side device authority.
    if not combine_series_resistors:
        return merged

    # --- series-resistor merge (iterate to collapse chains) ---------------- #
    port_set = set(ports)

    def res_fields(card: str):
        toks = card.split()
        if toks[0][0].upper() != "R" or len(toks) < 6:
            return None
        # name n1 n2 bulk value model W= L=
        n1, n2, bulk, value, model = toks[1], toks[2], toks[3], toks[4], toks[5]
        kv = dict(re.findall(r"\b(W|L)=([0-9.eE+\-]+u?)", card))
        return {
            "name": toks[0],
            "n1": n1,
            "n2": n2,
            "bulk": bulk,
            "value": value,
            "model": model,
            "W": kv.get("W", "1u"),
            "L": kv.get("L", "0u"),
        }

    def all_terminal_nets(card: str) -> list[str]:
        """Every connected net of a card (so a series fold never swallows a node
        that also touches a transistor / BJT — e.g. an OTA summing node)."""
        toks = card.split()
        cls = toks[0][0].upper()
        if cls == "M" and len(toks) >= 6:
            return toks[1:5]  # d g s b
        if cls == "Q" and len(toks) >= 4:
            return toks[1:4]  # c b e
        if cls == "C" and len(toks) >= 3:
            return toks[1:3]  # p n
        f = res_fields(card)
        if f is not None:
            return [f["n1"], f["n2"], f["bulk"]]
        return []

    cards = list(merged)
    changed = True
    while changed:
        changed = False
        res_cards = [c for c in cards if res_fields(c) is not None]
        nonres = [c for c in cards if res_fields(c) is None]
        # degree of each net across ALL device terminals in the cell; a node is
        # foldable only when its sole two connections are the two resistor A/B
        # terminals (bulk does not count toward foldability — it is the well).
        deg: dict[str, int] = {}
        for c in cards:
            terms = all_terminal_nets(c)
            f = res_fields(c)
            # for resistors only the A/B ends are series-connectivity terminals
            if f is not None:
                terms = [f["n1"], f["n2"]]
            for n in terms:
                deg[n] = deg.get(n, 0) + 1
        for net, d in deg.items():
            if net in port_set or d != 2:
                continue
            touching = [
                c
                for c in res_cards
                if net in (res_fields(c)["n1"], res_fields(c)["n2"])
            ]
            if len(touching) != 2:
                continue
            f0, f1 = res_fields(touching[0]), res_fields(touching[1])
            if f0["model"] != f1["model"] or f0["bulk"] != f1["bulk"]:
                continue
            if f0["W"] != f1["W"]:
                continue  # only fold identical-width unit cells (series squares add)
            # outer endpoints (the non-shared nets)
            ends = [n for n in (f0["n1"], f0["n2"], f1["n1"], f1["n2"]) if n != net]
            if len(ends) != 2:
                continue
            l_total = _parse_geom_um(f0["L"]) + _parse_geom_um(f1["L"])
            try:
                v_total = _spice_value_to_float(
                    f0["value"], {}
                ) + _spice_value_to_float(f1["value"], {})
                vstr = f"{v_total:.6g}"
            except ValueError:
                vstr = f0["value"]
            new_card = (
                f"{f0['name']} {ends[0]} {ends[1]} {f0['bulk']} {vstr} "
                f"{f0['model']} W={f0['W']} L={l_total:.6g}u"
            )
            cards = nonres + [c for c in res_cards if c not in touching] + [new_card]
            changed = True
            break

    return cards


def _expand_segmented_reference_resistors(
    body: list[str],
    ports: list[str],
    *,
    layout_cell_def: object | None,
    pdk: str | None,
    profile: object | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Replay canonical straight-segment topology into the LVS reference.

    The generic resistor lowerer emits one lumped card because a SPICE source
    carries only its two external nodes.  A typed ``@resistor_segments`` record
    is stronger physical authority: it declares the exact number of straight
    bodies and whether their contact heads are connected in series or parallel.
    Expand only those declared cards, after ordinary reference combination, so
    the schematic presents the same device graph as the extractor.

    The card's numeric ``R`` is the value the active PDK extractor computes from
    the declared W/L and sheet resistance.  The declaration's nominal PCell
    resistance remains independently checked against the source value by the
    canonical intent consumer and is persisted in the metadata receipt below.
    No geometry or node is inferred from GDS coordinates.
    """

    if layout_cell_def is None:
        return list(body), []

    from clay.layout.annotations import ResistorSegments
    from clay.layout.generic_placement import (
        _collect_passives,
        _resistor_segment_intent_map,
    )

    declaration_map = _resistor_segment_intent_map(layout_cell_def)
    if not declaration_map:
        return list(body), []
    if type(pdk) is not str or not pdk:
        raise ValueError("segmented LVS reference requires one exact PDK id")
    if profile is None or str(getattr(profile, "pdk_id", "")) != pdk:
        raise ValueError("segmented LVS reference lacks matching PDK profile")
    sheet_rho = getattr(profile, "sheet_rho", None)
    if (
        isinstance(sheet_rho, bool)
        or not isinstance(sheet_rho, (int, float))
        or not math.isfinite(float(sheet_rho))
        or float(sheet_rho) <= 0.0
    ):
        raise ValueError("segmented LVS reference lacks positive PDK sheet resistance")

    records = _collect_passives(layout_cell_def, None)
    card_names = [card.split()[0] for card in body if card.split()]
    if len(card_names) != len(set(card_names)):
        raise ValueError("effective LVS reference contains duplicate device names")
    occupied_tokens = set(ports)
    occupied_tokens.update(token for card in body for token in card.split())

    replacements: dict[str, list[str]] = {}
    metadata: list[dict[str, Any]] = []
    for target, declaration in declaration_map.items():
        if type(declaration) is not ResistorSegments:
            raise TypeError(
                f"resistor {target!r} declaration must be exact ResistorSegments"
            )
        if declaration.pdk != pdk:
            raise ValueError(
                f"@resistor_segments PDK {declaration.pdk!r} does not match "
                f"active LVS PDK {pdk!r}"
            )
        record = records.get(target)
        if record is None or record.get("resistor_segments") is not declaration:
            raise ValueError(
                f"@resistor_segments target {target!r} lacks canonical passive binding"
            )

        expected_name = _spice_instance_name(target, "R")
        matching = [card for card in body if card.split()[0] == expected_name]
        if len(matching) != 1:
            raise ValueError(
                f"@resistor_segments target {target!r} must bind exactly one "
                "lowered LVS resistor card"
            )
        card = matching[0]
        tokens = card.split()
        has_bulk = bool(getattr(profile, "resistor_has_bulk", False))
        expected_len = 8 if has_bulk else 7
        if (
            len(tokens) != expected_len
            or not tokens[-2].startswith("W=")
            or not tokens[-1].startswith("L=")
        ):
            raise ValueError(
                f"segmented LVS resistor {target!r} has malformed lowered card"
            )
        n1, n2 = tokens[1], tokens[2]
        if has_bulk:
            bulk, model = tokens[3], tokens[5]
        else:
            bulk, model = None, tokens[4]

        count = declaration.count
        connection = declaration.connection
        width_um = declaration.width_um
        length_um = declaration.length_um
        nominal_segment_resistance = declaration.segment_resistance_ohms
        if type(count) is not int or count <= 0 or count > 4096:
            raise ValueError("segmented LVS resistor count is outside 1..4096")
        if type(connection) is not str or connection not in {"series", "parallel"}:
            raise ValueError("segmented LVS resistor connection is malformed")
        for label, value in (
            ("width_um", width_um),
            ("length_um", length_um),
            ("segment_resistance_ohms", nominal_segment_resistance),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"segmented LVS resistor {label} must be one positive finite float"
                )

        generated_names = [
            f"{expected_name}__segment_{index:04d}"
            for index in range(1, count + 1)
        ]
        if any(name in occupied_tokens for name in generated_names):
            raise ValueError(
                f"segmented LVS resistor {target!r} generated a device-name collision"
            )
        generated_nets = (
            [
                f"__clay_lvs_{target}_segment_{index:04d}"
                for index in range(1, count)
            ]
            if connection == "series"
            else []
        )
        if any(net in occupied_tokens for net in generated_nets):
            raise ValueError(
                f"segmented LVS resistor {target!r} generated a private-net collision"
            )
        occupied_tokens.update(generated_names)
        occupied_tokens.update(generated_nets)

        from clay.layout.route.lvs_gate import _resolved_lvs_config
        res_spec = _resolved_lvs_config(pdk, full_extraction=True).get("res")
        source_model = str(record.get("model", "") or model)
        if res_spec is None:
            raise ValueError(
                f"segmented LVS resistor {target!r} lacks one implant-resolved "
                f"extraction model for {source_model!r}"
            )
        matched_spec = res_spec.resolve_source_model(source_model)
        model = matched_spec.model
        extracted_resistance = float(matched_spec.sheet_ohm_sq) * length_um / width_um
        value_token = _trim_num(extracted_resistance)
        extracted_resistance_receipt = float(value_token)
        width_token = _trim_num(width_um)
        length_token = _trim_num(length_um)
        expanded: list[str] = []
        for index, name in enumerate(generated_names):
            if connection == "parallel":
                left, right = n1, n2
            else:
                nodes = [n1, *generated_nets, n2]
                left, right = nodes[index], nodes[index + 1]
            if bulk is None:
                expanded.append(
                    f"{name} {left} {right} {value_token} {model} "
                    f"W={width_token}u L={length_token}u"
                )
            else:
                expanded.append(
                    f"{name} {left} {right} {bulk} {value_token} {model} "
                    f"W={width_token}u L={length_token}u"
                )
        replacements[expected_name] = expanded
        metadata.append(
            {
                "connection": connection,
                "count": count,
                "extracted_segment_resistance_ohms": extracted_resistance_receipt,
                "length_um": length_um,
                "model": model,
                "nominal_segment_resistance_ohms": nominal_segment_resistance,
                "target": target,
                "width_um": width_um,
            }
        )

    expanded_body: list[str] = []
    for card in body:
        name = card.split()[0]
        expanded_body.extend(replacements.get(name, [card]))
    if len(replacements) != len(declaration_map):
        raise RuntimeError("segmented LVS reference replay was not exhaustive")
    return expanded_body, metadata


def _read_case_cap_geometry(case_dir: Path) -> dict[str, Any]:
    """Read ``metadata.cap_geometry`` from a case package, best-effort.

    Returns a mapping of cap instance name -> ``{w, l, fingers[, model]}``. The
    spec does not surface raw case metadata, so the LvsStep reads the hint here
    directly. PyYAML missing / no hint -> empty mapping (cap stays cap_debt).
    """
    path = Path(case_dir) / "case.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - YAML missing/broken -> no hint
        return {}
    if not isinstance(doc, dict):
        return {}
    meta = doc.get("metadata") or {}
    geom = meta.get("cap_geometry") if isinstance(meta, dict) else None
    return geom if isinstance(geom, dict) else {}


def write_effective_lvs_reference(
    source_cir: Path,
    out_path: Path,
    *,
    cell_name: str,
    cap_geometry: dict[str, Any] | None = None,
    pdk: str | None = None,
    fold_overrides: "dict[str, tuple[float, float]] | None" = None,
    substrate_net: str | None = None,
    device_model_overrides: Mapping[str, str] | None = None,
    layout_cell_def: object | None = None,
    mim_option: str | None = None,
    cap_option: str | None = None,
    metal_level: str | None = None,
) -> Path:
    """Write a KLayout-LVS reference whose W folds in the schematic ``m`` factor.

    Generic generalization of the historical fc_ota-only helper: the subckt to
    extract is named by ``cell_name`` (== ``spec.gds_cell_name``); the port list
    is taken verbatim from the matching ``.subckt`` header in the source
    ``.cir``. Every device card inside that subckt is lowered to its
    PDK-LVS-comparable form by :func:`_lower_device_card`, using the per-PDK
    reference profile (:func:`_lvs_ref_profile`):

    * MOS (``M..`` and ``x..<nmos>``/``<pmos>``) — model lowered to the PDK's
      nmos/pmos LVS model (``nmos_3p3`` / ``pmos_3p3`` for gf180, ``nmos_3p3`` /
      ``pmos_3p3`` for external_pdk), ``W`` multiplied by any ``m=`` factor, ``L``
      resolved through ``.param``.
    * Resistors (``R..``) — lowered to the 3-terminal poly-R device the deck
      extracts (model + sheet-rho + bulk net from the PDK profile).
    * BJTs (``x..vpnp_*`` / ``vnpn_*``) — passed through with the gf180 model.

    ``pdk`` (== ``spec.pdk``) selects the reference profile; ``pdk=None`` keeps
    the gf180 defaults so the gf180 reference is byte-for-byte unchanged.
    ``substrate_net`` overrides only the profile's substrate spelling when the
    cell declares a semantic ``ground`` port, keeping the reference resistor
    bulk and the extraction deck on the same intent-derived net.  ``None``
    preserves the adapter/default profile. No circuit literals: the cell name,
    ports, and devices all come from data.

    ``fold_overrides`` (instance_name -> (nf, m), optional) carries the LAYOUT
    cell's per-device finger / multiplicity so the reference W folds by the SAME
    nf the placer emitted the geometry with, not the (independently-maintained)
    ``.cir`` nf. A device absent from the map keeps its ``.cir`` nf/m, so an
    in-sync circuit's reference is byte-unchanged; only a device whose layout nf
    DIFFERS from its schematic nf is re-folded to the layout truth. ``None`` keeps
    the historical ``.cir``-only folding.
    Capacitor cards (``C..``):

    * When the layout realises the cap as a foldable MOS-cap AND the caller
      supplies the as-drawn finger geometry in ``cap_geometry`` (keyed by the
      cap instance name -> ``{w, l, fingers[, model]}``), the ideal 2-terminal
      ``C`` is LOWERED to the gf180 ``pmoscap_3p3`` device whose combined A/P
      matches the extracted (``combine_devices``-folded) layout device, so it
      pairs 1:1 and LVS MATCHES (see :func:`_lower_cap_card`).
    * Otherwise the cap geometry is genuinely unknown from the .cir alone, so it
      is counted into the returned metadata under ``cap_debt`` — the LvsStep
      reports cap-device verification debt honestly instead of faking a match.
    """
    cap_geometry = cap_geometry or {}
    profile_mim_kw: dict[str, str] = {}
    if (pdk or "").strip().lower() == "gf180mcu":
        eff_mim = mim_option or getattr(layout_cell_def, "mim_option", None)
        eff_cap = cap_option or getattr(layout_cell_def, "mim_cap", None)
        eff_lvl = metal_level or getattr(layout_cell_def, "metal_level", None)
        if eff_mim is not None and str(eff_mim).strip():
            profile_mim_kw["mim_option"] = str(eff_mim).strip()
        if eff_cap is not None and str(eff_cap).strip():
            profile_mim_kw["cap_option"] = str(eff_cap).strip()
        if eff_lvl is not None and str(eff_lvl).strip():
            profile_mim_kw["metal_level"] = str(eff_lvl).strip()
    profile = _lvs_ref_profile(pdk, substrate_net=substrate_net, **profile_mim_kw)
    source_text = Path(source_cir).read_text(encoding="utf-8-sig")
    params = _parse_params(source_text)

    ports: list[str] = []
    body: list[str] = []
    cap_debt: list[str] = []
    unlowered: list[str] = []
    in_subckt = False
    for raw in source_text.splitlines():
        line = raw.split("$", 1)[0].strip()
        if not line or line.startswith("*"):
            continue
        low = line.lower()
        if low.startswith(".subckt"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == cell_name:
                in_subckt = True
                ports = parts[2:]
            continue
        if in_subckt and low.startswith(".ends"):
            break
        if not in_subckt:
            continue
        # skip pure-instance subckt calls that are not device wrappers
        # (handled inside _lower_device_card by model recognition)
        lowered, kind = _lower_device_card(
            line,
            params,
            profile,
            fold_overrides,
            device_model_overrides,
        )
        if lowered is not None:
            body.append(lowered)
        elif kind == "cap":
            # Precedence: (1) MOS-cap finger lowering when this cap has an as-drawn
            # finger-geometry hint (the layout realised it as a pmoscap_3p3 / MOS
            # cap); (2) the gf180 MIM lowering when the PDK has a MIM stack (the
            # layout realised it as a mim_<d>fF device); (3) honest cap_debt. A
            # cap with a geometry hint MUST take path (1) so a MOS-cap Miller Cc is
            # never mislowered to a MIM device.
            cap_lowered, _ = _lower_cap_card(line, params, cap_geometry)
            if cap_lowered is None:
                cap_lowered = _lower_mim_cap_card(line, params, profile, fold_overrides)
            if cap_lowered is not None:
                body.append(cap_lowered)
            else:
                cap_debt.append(line.split()[0])
        elif kind == "unknown" and line[0].upper() in ("M", "X", "R", "C", "Q"):
            unlowered.append(line.split()[0])

    if not ports:
        raise ValueError(f".subckt {cell_name} not found in {source_cir}")

    # Pre-combine parallel fingers / series unit-cells so the reference matches
    # the extractor's already-merged netlist (the deck does not simplify the
    # schematic side).
    body = _combine_reference_devices(
        body,
        ports,
        combine_series_resistors=layout_cell_def is None,
    )
    body, resistor_segment_expansions = _expand_segmented_reference_resistors(
        body,
        ports,
        layout_cell_def=layout_cell_def,
        pdk=pdk,
        profile=profile,
    )

    lines = [f".subckt {cell_name} " + " ".join(ports), *body, f".ends {cell_name}"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Sidecar: record device classes that could not be lowered losslessly so the
    # LvsStep can report verification debt honestly (never hidden). An ideal cap
    # card maps to laid-out MOS-cap finger geometry the source .cir does not
    # carry; an "unlowered" card is an unrecognised device the writer skipped.
    import json as _json

    meta = {"cap_debt": cap_debt, "unlowered": unlowered}
    if resistor_segment_expansions:
        meta["resistor_segment_expansions"] = resistor_segment_expansions
    Path(str(out_path) + ".meta.json").write_text(
        _json.dumps(meta, indent=2), encoding="utf-8"
    )
    return Path(out_path)


def pex_report_to_viewer_json(report: Any) -> dict[str, Any]:
    """Project a PexReport into the viewer parasitics JSON shape (generic)."""
    nets = sorted(str(net) for net in report.nets())
    pair_caps: dict[tuple[str, str], dict[str, Any]] = {}
    for cap in report.caps:
        a, b = str(cap.a), str(cap.b)
        if float(cap.cap_aF) <= 0:
            continue
        key = tuple(sorted((a, b)))
        slot = pair_caps.setdefault(key, {"a": key[0], "b": key[1], "cap_aF": 0.0})
        slot["cap_aF"] += float(cap.cap_aF)
    pairs = [
        {"a": v["a"], "b": v["b"], "cap_aF": round(float(v["cap_aF"]), 2)}
        for v in pair_caps.values()
    ]
    pairs.sort(key=lambda item: (-item["cap_aF"], item["a"], item["b"]))
    return {
        "source": "magic_ext2spice_gf180mcu",
        "backend": report.backend,
        "calibrated_2p5d": bool(report.calibrated),
        "cell": report.cell_name,
        "nets_extracted": nets,
        "net_total_cap_aF": {
            net: round(float(report.net_total_cap_aF(net)), 2) for net in nets
        },
        "coupling_aF": pairs,
        "total_cap_aF": round(float(report.total_cap_aF), 2),
        "total_cap_fF": round(float(report.total_cap_fF), 4),
    }


#: Repo subtree that holds COMMITTED circuit artifacts (schematics, golden
#: GDS, golden post_results.json). A ``clay flow run`` must treat everything
#: here as read-only: its outputs belong under the run's work_dir, never back
#: into the git working tree. Some per-circuit postsim modules (invoked by
#: PostSimStep) write into ``circuits/<c>/`` as a side effect of their CLI
#: ``main()`` flow; :func:`_protect_committed_tree` snapshots and restores any
#: such file so the invariant "flow run leaves circuits/ unchanged" holds.
_COMMITTED_TREE = REPO_ROOT / "circuits"


class _ProtectedTree:
    """Snapshot a committed subtree's file contents and restore on exit.

    Records ``(size, mtime, sha256)`` for every regular file under ``root`` and
    keeps an in-memory copy of each file's bytes. On exit, any file whose
    content changed during the body is rewritten with its original bytes, and
    any file CREATED during the body (not in the snapshot) is removed — so a
    step that writes into the protected tree leaves it byte-for-byte unchanged.
    Returns the list of paths it had to restore (for honest reporting).
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._before: dict[Path, bytes] = {}
        self.restored: list[Path] = []
        self.removed: list[Path] = []

    def __enter__(self) -> "_ProtectedTree":
        self._before = {}
        if self._root.is_dir():
            for path in self._root.rglob("*"):
                if path.is_file():
                    try:
                        self._before[path] = path.read_bytes()
                    except OSError:
                        continue
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Restore any modified file; delete any newly-created file. Best-effort
        # — a restore failure must never mask the body's own exception.
        for path, original in self._before.items():
            try:
                if not path.is_file() or path.read_bytes() != original:
                    path.write_bytes(original)
                    self.restored.append(path)
            except OSError:
                continue
        if self._root.is_dir():
            for path in self._root.rglob("*"):
                if path.is_file() and path not in self._before:
                    try:
                        path.unlink()
                        self.removed.append(path)
                    except OSError:
                        continue
        return False  # never suppress an exception raised inside the body


def _emit_event(ctx: FlowContext, result: StepResult) -> None:
    if ctx.recorder is None:
        return
    artifacts = dict(result.artifacts) or None
    ctx.recorder.event(
        f"flow_{result.name}",
        revision=f"{ctx.spec.key}-flow-{result.name}",
        summary=result.summary,
        artifacts=artifacts,
        data={"status": result.status, "step": result.name, **result.data},
    )


# --------------------------------------------------------------------------- #
# Built-in generic steps                                                      #
# --------------------------------------------------------------------------- #
class CompileStep:
    name = "compile"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        design = _effective_design_clay(spec, ctx)
        return [p for p in (design, _effective_source_cir(spec, ctx)) if Path(p).is_file()]

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        from clay.parser import parse_clay_file

        design = _effective_design_clay(spec, ctx)
        if not Path(design).is_file():
            return StepResult(
                self.name,
                "warn",
                f"no design.clay under {spec.case_dir}; compile step has no source",
                data={"design_clay": str(design)},
            )
        design = Path(design).resolve()
        before = _sha256(design)
        # Compile-only flows without optimization retain their historical
        # behavior. A sizing invocation additionally binds its original
        # schematic to this compile boundary without requiring it to exist for
        # a standalone compile invocation.
        schematic_path = None
        schematic_sha = None
        if getattr(spec, 'optimization', None) is not None:
            schematic_path = Path(spec.source_cir).resolve()
            schematic_sha = _sha256(schematic_path) if schematic_path.is_file() else None
        cf = parse_clay_file(design)
        if _sha256(design) != before:
            raise ValueError("compile source changed during parsing")
        ctx.extra["compiled_design_clay_path"] = str(design)
        ctx.extra["compiled_design_clay_sha256"] = before
        cells = list(getattr(cf, "cell_defs", []) or [])
        target = None
        for cell in cells:
            local = getattr(cell, "local_name", getattr(cell, "name", None))
            if local == spec.top_cell:
                target = cell
                break
        if target is None:
            return StepResult(
                self.name,
                "fail",
                f"cell {spec.top_cell!r} not found in {design}",
                data={"cells": [getattr(c, "local_name", None) for c in cells]},
                blocking=True,
            )
        candidate_sizing = ctx.extra.get("candidate_sizing")
        if candidate_sizing is not None:
            _apply_candidate_device_bindings(
                target, spec.optimization, candidate_sizing, pdk_id=spec.pdk
            )
        ctx.extra["cell_def"] = target
        try:
            from clay.compiler import compile_cell_def

            compile_cell_def(target)
            compiled = True
            compile_error = None
        except Exception as exc:  # noqa: BLE001
            compiled = False
            compile_error = f"{type(exc).__name__}: {exc}"
            ctx.extra["compile_error"] = compile_error
        if schematic_path is not None:
            actual = _sha256(schematic_path) if schematic_path.is_file() else None
            if actual != schematic_sha:
                return StepResult(self.name, "fail", "sizing schematic changed during compilation",
                                  data={"reason": "RUN_IDENTITY_CHANGED"}, blocking=True)
            ctx.extra["compiled_source_cir_path"] = str(schematic_path)
            ctx.extra["compiled_source_cir_sha256"] = schematic_sha
        return StepResult(
            self.name,
            "pass" if compiled else "fail",
            (
                f"parsed {spec.top_cell} from {design.name}; compiled=True"
                if compiled
                else f"physical compile failed for {spec.top_cell}: {compile_error}"
            ),
            data={
                "cell": spec.top_cell,
                "n_cells": len(cells),
                "compiled": compiled,
                "compile_error": compile_error,
            },
            blocking=not compiled,
        )


def _apply_candidate_device_bindings(cell_def, optimization, values, *, pdk_id=None) -> None:
    """Materialize source-declared ``device.*`` sizing bindings on one cell.

    Schematic and testbench targets are consumed by ``presim_evaluate``; this
    function owns only the explicit device IR side of the same binding table.
    """
    from clay.flow.candidate_bindings import apply_candidate_device_bindings

    apply_candidate_device_bindings(cell_def, optimization, values, pdk_id=pdk_id)


class SizingStep:
    """Canonical source-triggered sizing-loop step."""

    name = "sizing"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        design = Path(_effective_design_clay(spec, ctx))
        authoring = Path(ctx.extra.get("sizing_source_path") or design)
        return list(dict.fromkeys(path for path in (authoring, design, Path(spec.source_cir)) if path.is_file()))

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        if spec.optimization is None:
            return StepResult(
                self.name, "skipped", "no source-declared sizing optimization"
            )
        if spec.optimization.budget < 4:
            return StepResult(
                self.name,
                "fail",
                "sizing opt_budget must be at least 4: two seeds, a seed mean, "
                "and final POST validation",
                data={"engine": spec.optimization.engine,
                      "budget": spec.optimization.budget},
                blocking=True,
            )
        from clay.flow.sizing_source import (SizingSource, SizingSourceChanged, require_source_matches_spec,
                                             spec_fingerprint, verify_spec_fingerprint)
        from clay.flow.execution_budget import ExecutionStopped

        try:
            authoring = SizingSource.capture(_effective_design_clay(spec, ctx))
            require_source_matches_spec(spec, authoring, full_projection=True)
            spec_identity = spec_fingerprint(spec)
            try:
                captured_schematic = SizingSource.capture(spec.source_cir)
            except (OSError, TypeError, ValueError) as exc:
                if ctx.results.get('compile') is not None:
                    raise SizingSourceChanged('sizing schematic source unavailable after compilation') from exc
                raise
            schematic = SizingSource(captured_schematic.path, captured_schematic.sha256, 'sizing schematic source')
            if ctx.results.get('compile') is not None:
                compiled_schematic_path = ctx.extra.get('compiled_source_cir_path')
                compiled_schematic_sha = ctx.extra.get('compiled_source_cir_sha256')
                if (compiled_schematic_path is None or Path(compiled_schematic_path).resolve() != schematic.path
                        or compiled_schematic_sha != schematic.sha256):
                    raise SizingSourceChanged('sizing schematic source changed after compilation')

            def verify_promotion_inputs():
                authoring.verify()
                verify_spec_fingerprint(spec, spec_identity)
                try:
                    current_path = Path(spec.source_cir).resolve()
                except (OSError, TypeError, ValueError) as exc:
                    raise SizingSourceChanged('sizing schematic source path changed') from exc
                if current_path != schematic.path:
                    raise SizingSourceChanged('sizing schematic source path changed')
                schematic.verify()
            compiled_sha = ctx.extra.get("compiled_design_clay_sha256")
            compiled_path = ctx.extra.get("compiled_design_clay_path")
            compiled_step = ctx.results.get("compile")
            if compiled_step is not None and (compiled_step.status != "pass" or compiled_sha != authoring.sha256
                  or compiled_path is None or Path(compiled_path).resolve() != authoring.path):
                raise ValueError("compiled and sizing source identities differ")
        except ExecutionStopped:
            raise
        except SizingSourceChanged as exc:
            return StepResult(self.name, "fail", f"SizingSourceChanged: {exc}",
                              data={"reason": "RUN_IDENTITY_CHANGED", "engine": spec.optimization.engine,
                                    "error": str(exc)}, blocking=True)
        except Exception as exc:  # noqa: BLE001 - source authority is mandatory
            return StepResult(self.name, "fail", f"sizing source binding failed: {type(exc).__name__}: {exc}",
                              data={"engine": spec.optimization.engine, "error": str(exc)}, blocking=True)
        joint_evaluate = ctx.extra.get("joint_evaluate")
        case_source = (Path(spec.case_dir) / "source" / "design.clay").resolve()
        source_override = authoring.path != case_source
        injected_evaluator = joint_evaluate is not None
        if injected_evaluator:
            try:
                from clay.frontend.presim_evaluate import require_source_bound_joint_evaluator
                require_source_bound_joint_evaluator(joint_evaluate, authoring, spec)
            except SizingSourceChanged as exc:
                return StepResult(self.name, "fail", f"SizingSourceChanged: {exc}",
                                  data={"reason": "RUN_IDENTITY_CHANGED", "engine": spec.optimization.engine,
                                        "error": str(exc)}, blocking=True)
            except (TypeError, ValueError) as exc:
                return StepResult(self.name, "fail", f"custom sizing evaluator binding failed: {exc}",
                                  data={"engine": spec.optimization.engine, "error": str(exc)}, blocking=True)
        if joint_evaluate is None:
            try:
                from clay.frontend.presim_evaluate import (
                    make_joint_evaluate_from_spec,
                )

                joint_evaluate = make_joint_evaluate_from_spec(
                    spec, work_root=ctx.work_dir / "sizing" / "evaluations", source_authority=authoring
                )
                authoring.verify()
            except ExecutionStopped:
                raise
            except SizingSourceChanged as exc:
                return StepResult(self.name, "fail", f"SizingSourceChanged: {exc}",
                                  data={"reason": "RUN_IDENTITY_CHANGED", "engine": spec.optimization.engine,
                                        "error": str(exc)}, blocking=True)
            except Exception as exc:  # noqa: BLE001 - fail-closed binding
                return StepResult(
                    self.name,
                    "fail",
                    f"joint evaluator binding failed: {type(exc).__name__}: {exc}",
                    data={"engine": spec.optimization.engine, "error": str(exc)},
                    blocking=True,
                )
        accepted_source = None
        try:
            import json
            import numpy as np

            from clay.flow.sizing import JointSizingProblem, normalize_sizing
            from tools.clay_optimize import (
                SizingBounds,
                build_tssbo_adapter_from_intent,
            )

            bounds = tuple(
                SizingBounds(variable.name, variable.low, variable.high)
                for variable in spec.optimization.variables
            )
            adapter = build_tssbo_adapter_from_intent(
                bounds, seed=spec.optimization.seed
            )
            problem = JointSizingProblem(
                spec, joint_evaluate, journal_dir=ctx.work_dir / "sizing"
            )
            default_sizing = {
                variable.name: variable.default
                for variable in spec.optimization.variables
            }
            default_point = normalize_sizing(spec, default_sizing)
            default_row = problem.evaluate_point(
                default_point,
                force_post=True,
                post_reason="feasibility_gate",
            )
            # The declared budget counts every PRE evaluation, including the
            # forced final POST check (which runs PRE first). Reserve its slot
            # before forming the local seed cloud. The upstream solver also
            # evaluates one seed mean, so even the smallest run needs two seeds
            # plus that mean and final validation.
            search_budget = int(spec.optimization.budget) - 1
            seed_count = min(4, search_budget - 1)
            initial_points = [default_point]
            initial_rows = [default_row]
            domain_low, domain_high = spec.optimization.normalized_domain
            for seed_index in range(seed_count - 1):
                point = default_point.copy()
                coordinate = seed_index % len(point)
                direction = 1.0 if seed_index % 2 == 0 else -1.0
                point[coordinate] = np.clip(
                    point[coordinate] + direction * 0.05,
                    domain_low,
                    domain_high,
                )
                initial_points.append(point)
                initial_rows.append(problem.evaluate_point(point))
            solver_result = adapter.solve_constrained(
                problem,
                n_max=search_budget,
                outdim=problem.outdim,
                batch_size=min(4, search_budget),
                n_candidates=80,
                init_x=np.asarray(initial_points, dtype=np.float32),
                init_y=np.asarray(initial_rows, dtype=np.float32),
            )
            if len(problem.journal) != search_budget:
                raise RuntimeError(
                    "sizing solver did not consume the declared search budget: "
                    f"{len(problem.journal)}/{search_budget}"
                )
            # The solver incumbent may be a conservative pre-layout proxy.  It
            # is evidence only after a binding final POST-PEX measurement.
            problem.evaluate_point(
                solver_result.best_x,
                force_post=True,
                post_reason="final",
            )
            verify_promotion_inputs()
            # Nominal POST samples guide the constrained search; only a
            # source-declared full corner/temperature POST-PEX sweep may
            # promote the final incumbent.
            if injected_evaluator:
                require_source_bound_joint_evaluator(joint_evaluate, authoring, spec)
            incumbent = problem.promote_acceptance()
            verify_promotion_inputs()
            if injected_evaluator:
                require_source_bound_joint_evaluator(joint_evaluate, authoring, spec)
            if incumbent is None:
                return StepResult(
                    self.name,
                    "fail",
                    "tSS-BO produced no full-PVT POST-PEX feasible incumbent",
                    data={
                        "engine": spec.optimization.engine,
                        "evaluations": len(problem.journal),
                        "solver_feasible": bool(solver_result.feasible),
                    },
                    artifacts={
                        "journal": str(ctx.work_dir / "sizing" / "sizing_journal.jsonl")
                    },
                    blocking=True,
                )
            if (source_override or injected_evaluator) and (incumbent.get("authoring_design_clay") != str(authoring.path)
                                    or incumbent.get("authoring_design_clay_sha256") != authoring.sha256):
                raise ValueError("promoted candidate did not bind the effective sizing source")
            if source_override or injected_evaluator:
                candidate_design = Path(incumbent.get("candidate_design_clay") or "")
                if (not candidate_design.is_file() or not incumbent.get("candidate_design_clay_sha256")
                        or _sha256(candidate_design) != incumbent["candidate_design_clay_sha256"]):
                    raise ValueError("evaluated candidate Clay changed before promotion")
            from clay.flow.candidate_artifacts import capture_accepted_source, verify_promoted_source, require_identity
            accepted_source = capture_accepted_source(incumbent)
            ctx.extra["candidate_sizing"] = dict(incumbent["sizing"])
            ctx.extra["sizing_incumbent"] = dict(incumbent)
            if incumbent.get("candidate_source"):
                ctx.extra["evaluated_candidate_source_cir"] = str(
                    incumbent["candidate_source"]
                )
            # A promoted sizing is a circuit revision, not an invisible mutation
            # of the authoring IR.  Materialize it as a standalone Clay source,
            # then REPARSE that source so every downstream physical step consumes
            # the exact bytes that the evidence ledger will bind.
            from clay.compiler import compile_cell_def
            from clay.flow.optimized_clay import materialize_optimized_clay
            from clay.parser import parse_clay_file

            authoring_design = authoring.verify()
            optimized_design = ctx.work_dir / "sizing" / "design.optimized.clay"
            optimized_artifact = materialize_optimized_clay(
                source_path=authoring_design,
                output_path=optimized_design,
                top_cell=spec.top_cell,
                optimization=spec.optimization,
                incumbent=incumbent,
                provenance={
                    "engine": spec.optimization.engine,
                    "seed": spec.optimization.seed,
                },
            )
            authoring.verify()
            if optimized_artifact.authoring_sha256 != authoring.sha256:
                raise ValueError("promoted source authority differs from evaluated source")
            # Re-materialize the promoted schematic from the SAME canonical
            # sizing values written into the optimized Clay.  The tSS-BO/torch
            # boundary may represent 80um as 79.999999404um; downstream row
            # geometry is exact integer nm and must never consume that adjacent
            # float spelling as a second physical source of truth.
            from clay.frontend.presim_evaluate import materialize_joint_schematic

            verify_promotion_inputs()
            try:
                promoted_source, _promoted_testbench_params = materialize_joint_schematic(
                    spec,
                    optimized_artifact.sizing,
                    ctx.work_dir / "sizing" / "promoted",
                    source_authority=authoring,
                )
            except ExecutionStopped:
                raise
            except Exception as original:
                from clay.flow.candidate_artifacts import require_identity
                try:
                    require_identity(*accepted_source)
                except SizingSourceChanged as changed:
                    raise changed from original
                raise
            verify_promotion_inputs()
            verify_promoted_source(accepted_source, ctx.work_dir, promoted_source)
            promoted_source = promoted_source.resolve()
            ctx.extra["candidate_source_cir"] = str(promoted_source)
            parsed_optimized = parse_clay_file(optimized_artifact.path)
            optimized_cells = [
                cell
                for cell in parsed_optimized.cell_defs
                if getattr(cell, "local_name", getattr(cell, "name", None))
                == spec.top_cell
            ]
            if len(optimized_cells) != 1:
                raise ValueError(
                    f"optimized Clay must contain exactly one {spec.top_cell!r} cell; "
                    f"found {len(optimized_cells)}"
                )
            optimized_cell = optimized_cells[0]
            compile_cell_def(optimized_cell)
            # The caller may have supplied an unrelated in-memory CellDef.
            # Final layout obligations must come from the same frozen source
            # that generated the incumbent and promoted Clay, even when the
            # public invocation intentionally omitted the compile step.
            authoring.verify()
            parsed_authoring = parse_clay_file(authoring.path)
            authoring.verify()
            authoring_cells = [
                cell for cell in parsed_authoring.cell_defs
                if getattr(cell, "local_name", getattr(cell, "name", None))
                == spec.top_cell
            ]
            if len(authoring_cells) != 1:
                raise ValueError(
                    f"authoring Clay must contain exactly one {spec.top_cell!r} cell; "
                    f"found {len(authoring_cells)}"
                )
            ctx.extra["authoring_cell_def"] = authoring_cells[0]
            ctx.extra["cell_def"] = optimized_cell
            ctx.extra["effective_design_clay"] = optimized_artifact.path.resolve()
            ctx.extra["sizing_source_path"] = str(authoring.path)
            ctx.extra["cell_def_source_sha256"] = optimized_artifact.sha256
            ctx.extra["authoring_design_clay_sha256"] = (
                optimized_artifact.authoring_sha256
            )
            result_path = ctx.work_dir / "sizing" / "sizing_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "schema": "clay.flow.sizing/v1",
                        "engine": spec.optimization.engine,
                        "authoring_design_clay": str(authoring.path),
                        "authoring_design_clay_sha256": authoring.sha256,
                        "seed": spec.optimization.seed,
                        "budget": spec.optimization.budget,
                        "evaluations": len(problem.journal),
                        "search_evaluations": search_budget,
                        "final_evaluations": 1,
                        "incumbent": incumbent,
                        "promoted_candidate_source": str(promoted_source),
                        "promoted_candidate_source_sha256": _sha256(promoted_source),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            verify_promotion_inputs()
            verify_promoted_source(accepted_source, ctx.work_dir, promoted_source)
            return StepResult(
                self.name,
                "pass",
                "tSS-BO promoted a full-PVT POST-PEX feasible incumbent",
                data={
                    "engine": spec.optimization.engine,
                    "evaluations": len(problem.journal),
                    "fom": incumbent["fom"],
                    "truth": incumbent["truth"],
                    "post_verified": sum(
                        1 for item in problem.journal if item["verified"]
                    ),
                    "nominal_post_verified": sum(
                        1
                        for item in problem.journal
                        if item.get("nominal_post_verified") is True
                    ),
                    "acceptance_attempted": sum(
                        1 for item in problem.journal if "acceptance" in item
                    ),
                },
                artifacts={
                    "sizing_result": str(result_path),
                    "optimized_clay": str(optimized_artifact.path),
                    "candidate_source": str(promoted_source),
                    "journal": str(ctx.work_dir / "sizing" / "sizing_journal.jsonl"),
                },
            )
        except ExecutionStopped:
            raise
        except SizingSourceChanged as exc:
            return StepResult(self.name, "fail", f"SizingSourceChanged: {exc}",
                              data={"reason": "RUN_IDENTITY_CHANGED", "engine": spec.optimization.engine,
                                    "error": str(exc)}, blocking=True)
        except Exception as exc:  # noqa: BLE001 - fail-closed flow boundary
            try:
                verify_promotion_inputs()
                if accepted_source is not None:
                    require_identity(*accepted_source)
            except SizingSourceChanged as changed:
                return StepResult(self.name, "fail", f"SizingSourceChanged: {changed}",
                                  data={"reason": "RUN_IDENTITY_CHANGED", "engine": spec.optimization.engine,
                                        "error": str(changed)}, blocking=True)
            return StepResult(
                self.name,
                "fail",
                f"{type(exc).__name__}: {exc}",
                data={"engine": spec.optimization.engine, "error": str(exc)},
                blocking=True,
            )


#: Sentinel that selects the GENERIC place+route+emit layout path (the Clay-core
#: placer + router) instead of the per-circuit hand generator binding. Set it via
#: ``flow { layout = generic; }`` (the ``layout_module`` lowers to this string) or
#: ``metadata.layout: generic`` in the case package's ``case.yaml``.
_GENERIC_LAYOUT_SENTINEL = "generic"


def _generic_layout_selected(spec: CircuitSpec) -> bool:
    """True when the design selects the generic place+route+emit layout path.

    Two equivalent selectors (data-driven, no per-circuit branch):
      * ``flow { layout = generic; }`` -> ``spec.layout_module == "generic"``;
      * ``metadata.layout: generic`` in the case package's ``case.yaml``.
    The per-circuit ``spec.layout_module`` binding remains the DEFAULT (fallback),
    so the currently-green hand-generated circuits are unaffected.
    """
    if (spec.layout_module or "").strip().lower() == _GENERIC_LAYOUT_SENTINEL:
        return True
    meta_layout = _read_case_metadata_value(spec.case_dir, "layout")
    return str(meta_layout or "").strip().lower() == _GENERIC_LAYOUT_SENTINEL


def _read_case_metadata_value(case_dir: Path, key: str) -> Any:
    """Read ``metadata.<key>`` from a case package (best-effort, None if absent)."""
    path = Path(case_dir) / "case.yaml"
    if not path.is_file():
        return None
    try:
        import yaml  # type: ignore

        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(doc, dict):
        return None
    meta = doc.get("metadata") or {}
    return meta.get(key) if isinstance(meta, dict) else None


def _criteria_for_cell(spec: CircuitSpec):
    """The :class:`ConvergenceCriteria` lifted from the cell's ``layout_guide.txt``
    when ``spec.top_cell`` is one of the known probe cells, else ``None``.

    Matches the flow cell to a guide-backed circuit by cell name (generic — no
    per-circuit branch in the engine; the mapping lives in ``guide_intent``). Any
    failure (cell not a known probe, guide unreadable, parse error) returns
    ``None`` so the convergence loops fall back to their intrinsic acceptance.
    """
    try:
        from clay.layout import guide_intent
    except Exception:
        return None
    circuits = getattr(guide_intent, "_CIRCUITS", {}) or {}
    for cid, entry in circuits.items():
        # entry == (guide_path, clay_path, cell_name)
        if len(entry) >= 3 and entry[2] == spec.top_cell:
            try:
                return guide_intent.build_criteria(cid)
            except Exception:
                return None
    return None


def _project_compact_route_final_placement(
    compact_route: Any,
    route_result: Any,
    intent: Any,
) -> tuple[Any, Any]:
    """Re-project final routed device geometry without mutating logical basis proofs."""
    from clay.layout.flow_placement import project_final_route_placement

    base_placement = compact_route.logical_placement_basis
    retained_row = getattr(route_result, "row_final_placement", None)
    retained_bindings = {
        name: getattr(retained_row, name)
        for name in ("row_native_geometry_binding", "rigid_nonmos_native_geometry_binding")
        if getattr(retained_row, name, None) is not None
    }
    projected = project_final_route_placement(
        base_placement,
        route_result.placement_um,
        route_result.geometry,
        final_placement_certificate=getattr(
            route_result, "final_placement_certificate", None
        ),
        intent=intent,
        **({"retained_row_bindings": retained_bindings} if retained_bindings else {}),
    )
    return base_placement, projected


class RetiredLegacyLayoutError(RuntimeError):
    """Raised when a case attempts to invoke a retired hand-written ``circuits.*`` layout script."""


def _is_retired_legacy_layout_module(layout_module: str | None) -> bool:
    """True when ``layout_module`` points to a retired per-circuit ``circuits.*`` script."""
    mod = (layout_module or "").strip()
    return mod.startswith("circuits.")


class LayoutStep:
    name = "layout"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        generic = (
            _generic_layout_selected(spec) or not (spec.layout_module or "").strip()
        )
        effective_design = _effective_design_clay(spec, ctx)
        if generic:
            # The generic path is driven by the parsed design.clay + the bound
            # source .cir (real sizing + connectivity); both gate the incremental
            # skip decision.
            files = [effective_design, _effective_source_cir(spec, ctx)]
            from clay.flow.native_physical import native_input_closure
            return native_input_closure(spec, ctx, *[p for p in files if Path(p).is_file()])
        layout_py = REPO_ROOT / (spec.layout_module.replace(".", "/") + ".py")
        # A declared/effective Clay candidate gates every layout backend. Keep
        # it in the incremental signature even when a legacy generator emits
        # the geometry, otherwise a removed or stale marker could reuse a
        # previously passing cached LayoutStep without re-running the gate.
        files = [layout_py, effective_design]
        if spec.emit_function is None and Path(spec.fallback_gds).is_file():
            files.append(Path(spec.fallback_gds))
        return [p for p in files if Path(p).is_file()]

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        cell_def = ctx.extra.get("cell_def") or self._parse_cell_def(spec)
        obligation_cell = ctx.extra.get("authoring_cell_def") or cell_def
        effective_override = ctx.extra.get("effective_design_clay")
        if effective_override:
            effective_design = Path(effective_override)
            try:
                from clay.parser import parse_clay_file

                parsed_effective = parse_clay_file(effective_design)
                effective_cells = [
                    cell
                    for cell in (getattr(parsed_effective, "cell_defs", ()) or ())
                    if getattr(cell, "local_name", getattr(cell, "name", None))
                    == spec.top_cell
                ]
                if len(effective_cells) != 1:
                    raise ValueError(
                        f"effective design requires exactly one {spec.top_cell!r} "
                        f"cell; found {len(effective_cells)}"
                    )
            except Exception as exc:  # noqa: BLE001 - named intent boundary
                return StepResult(
                    self.name,
                    "fail",
                    f"effective layout intent unreadable: {exc}",
                    data={
                        "reason": "UNRESOLVED_COMPACT_INTENT",
                        "intent_source": str(effective_design),
                    },
                    blocking=True,
                )
            effective_cell = effective_cells[0]
            if cell_def is not None and cell_def is not effective_cell:
                ctx.extra.setdefault("authoring_cell_def", cell_def)
            _attach_gf180_mim_attrs(
                effective_cell,
                _gf180_mim_context(spec, cell_def=cell_def),
            )
            cell_def = effective_cell
            # All downstream physical consumers now read the exact parsed bytes
            # whose provenance the gate validates.
            ctx.extra["cell_def"] = effective_cell
        local_scene = ctx.extra.get('local_routing_scene_request')
        if local_scene is not None:
            from clay.layout.intent import layout_intent_from_cell_def
            from clay.layout.policy import policy_for_cell, resolve_policy
            from clay.layout.route.local_routing_scene import run_local_routing_scene
            intent = layout_intent_from_cell_def(cell_def)
            policy = self._resolve_effective_policy(
                spec, cell_def, intent, policy_for_cell, resolve_policy, ctx=ctx)
            result = run_local_routing_scene(local_scene,
                source_path=_effective_design_clay(spec, ctx), policy=policy,
                pdk_id=spec.pdk, work_dir=ctx.work_dir / 'local-routing-scene', source_intent=intent)
            physical = result['physical']
            ok = (result['search']['success'] and physical['connected']
                  and physical['cross_overlap_area_nm2'] == 0
                  and physical['cross_spacing_violations'] == 0
                  and physical['width_violations'] == 0
                  and physical['emission_union_xor_area_nm2'] == 0
                  and physical['metal_min_spacing_status'] == 'pass')
            return StepResult(self.name, 'warn' if ok else 'fail',
                              'local routing scene diagnostic', data=result,
                              blocking=not ok)
        unresolved = self._unresolved_compact_intent(
            spec,
            ctx,
            cell_def,
            obligation_cell=obligation_cell,
        )
        if unresolved is not None:
            return unresolved
        # A device-bearing case with no layout_module cannot take the
        # per-circuit path (it would import_module("") and crash); route it to
        # the generic Clay-core placer/router, same as an explicitly generic case.
        if _generic_layout_selected(spec) or not (spec.layout_module or "").strip():
            if any(device.device_class == "native" for device in getattr(cell_def, "devices", ())):
                from clay.flow.physical_support import check_device_support

                support = check_device_support(
                    cell_def, spec.pdk, physical=True,
                    source_artifact=_effective_design_clay(spec, ctx),
                )
                if not support.ready:
                    return self._physical_support_rejection(support)
            return self._run_generic(spec, ctx)
        if (ctx.extra.get("diagnostic_route_observer") is not None
                or ctx.extra.get("native_composite_observation") is not None
                or ctx.extra.get("pre_route_gate") is not None
                or ctx.extra.get("diagnostic_candidate_request") is not None
                or ctx.extra.get("diagnostic_candidate_batch_request") is not None):
            raise ValueError("diagnostic routing observer requires the generic layout caller")
        return self._run_per_circuit(spec, ctx)

    @staticmethod
    def _physical_support_rejection(support):
        from dataclasses import asdict

        return StepResult(
            "layout", "fail", "physical support check blocked before materialization",
            data={"reason": "PHYSICAL_SUPPORT_BLOCKED",
                  "detail": {"checks": [asdict(check) for check in support.checks],
                             "physical_admission": False}},
            blocking=True,
        )

    def _unresolved_compact_intent(
        self,
        spec: CircuitSpec,
        ctx: FlowContext,
        cell_def: Any | None,
        *,
        obligation_cell: Any = None,
    ) -> StepResult | None:
        """Fail closed before either generic or legacy physical generation."""

        from clay.flow.optimized_clay import (
            UnresolvedCompactIntentError,
            compact_intent_search_requested,
            require_resolved_compact_intent,
        )

        declarations: list[Any] = []
        for candidate in (obligation_cell, cell_def):
            if candidate is None or any(candidate is prior for prior in declarations):
                continue
            if compact_intent_search_requested(candidate):
                declarations.append(candidate)
        if not declarations:
            return None
        effective_design = _effective_design_clay(spec, ctx)
        try:
            stage_source = ctx.extra.get('_primitive_stage_source_bytes')
            effective_design_source = (stage_source.decode('utf-8-sig') if stage_source is not None
                                       else effective_design.read_text(encoding="utf-8-sig"))
            for declared_by in declarations:
                require_resolved_compact_intent(
                    declared_by,
                    effective_design_source,
                )
        except (OSError, UnresolvedCompactIntentError) as exc:
            return StepResult(
                self.name,
                "fail",
                f"layout optimization unresolved: {exc}",
                data={
                    "reason": "UNRESOLVED_COMPACT_INTENT",
                    "intent_source": str(effective_design),
                },
                blocking=True,
            )
        return None

    # ----------------------------------------------------------------- #
    # The GENERIC place+route+emit path (Clay-core placer + router)     #
    # ----------------------------------------------------------------- #
    def _run_generic(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        """Place + route + emit using the generic Clay-core layout engine.

        Pipeline (composes the first-class Clay layout objects; no per-circuit
        special-casing — circuit specifics come ONLY from the parsed cell + the
        derived intent + the bound source .cir):

          1. ``place_generic(cell_def, intent, source_cir=...)`` -> Placement
             (real device geometry from the .cir, symmetry-constrained MILP);
          2. build a net->terminals netlist from the .cir device pins;
          3. ``route_generic(placement, netlist, intent)`` -> RoutedCell;
          4. ``emit_routed_cell(routed, gds)`` -> the generic GDS.
        """
        from clay.layout.generic_placement import (
            BjtMaterializationUnavailable,
            parse_cir_mos_records,
            parse_cir_passive_records,
        )
        from clay.layout.passive_device_cell import UnsupportedPdkFactError
        from clay.layout.physical_macro_contracts import PhysicalMacroContractError
        from clay.layout.routed_cell_types import emit_routed_cell
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.place_refine import place_to_intent
        from clay.layout.policy import policy_for_cell, resolve_policy

        out_gds = gds_path(ctx)
        out_gds.parent.mkdir(parents=True, exist_ok=True)

        cell_def = ctx.extra.get("cell_def") or self._parse_cell_def(spec)
        if cell_def is None:
            return StepResult(
                self.name,
                "fail",
                f"generic layout: cell {spec.top_cell!r} not found in "
                f"{spec.case_dir / 'source' / 'design.clay'}",
                blocking=True,
            )

        effective_source = _effective_source_cir(spec, ctx)
        source_cir = effective_source if effective_source.is_file() else None
        mim_ctx = _gf180_mim_context(spec, cell_def=cell_def)
        _attach_gf180_mim_attrs(cell_def, mim_ctx)
        intent = layout_intent_from_cell_def(cell_def)
        _attach_gf180_mim_attrs(intent, mim_ctx)
        from clay.layout.tech import LayoutTech

        layout_tech: Any = (
            LayoutTech.resolve(spec.pdk, **mim_ctx) if mim_ctx else spec.pdk
        )
        primitive_only = getattr(self, 'primitive_only', False)
        native_geometry = None
        native_composite_row = None
        row_candidates=ctx.extra.get('native_row_candidates',())
        if row_candidates and not any(device.device_class=='native' for device in cell_def.devices):
            raise ValueError('native row candidates require source-bound native devices')
        if primitive_only and (not any(device.device_class == 'native' for device in cell_def.devices)
                or any(array.style == 'native_diffusion' for array in intent.arrays)):
            return StepResult(self.name, 'warn', 'primitive observation unavailable for this source',
                data={'status': 'UNKNOWN', 'physical_admission': False})
        if (any(array.style == 'native_diffusion' for array in intent.arrays)
                and not any(device.device_class == 'native' for device in cell_def.devices)):
            return StepResult(
                self.name, 'fail',
                'native diffusion declaration requires source-bound native devices',
                blocking=True,
                data={'error_type': 'NativeDiffusionUnsupported',
                      'error': 'native diffusion declaration requires source-bound native devices'},
            )
        if any(device.device_class == "native" for device in cell_def.devices):
            from clay.flow.native_physical import native_cell, native_runtime
            from clay.frontend.native_spice_import import emit_native_spice
            from clay.flow.physical_support import check_device_support
            import uuid

            cell_def = native_cell(spec, ctx)
            if row_candidates:
                from clay.layout.native_diffusion_source import declared_native_chains
                from clay.layout.native_diffusion_producer import _checked_candidates
                canonical_intent=layout_intent_from_cell_def(cell_def)
                _checked_candidates(cell_def,declared_native_chains(cell_def,canonical_intent),row_candidates)
            # Recheck the effective, source-bound cell at the actual consumer.
            # No source/PDK-specific dispatch is duplicated in this adapter.
            support = check_device_support(
                cell_def, spec.pdk, physical=True,
                source_artifact=_effective_design_clay(spec, ctx),
            )
            if not support.ready:
                return self._physical_support_rejection(support)
            if primitive_only and (spec.native_runtime_manifest is None or not spec.native_runtime_sha256):
                return StepResult(self.name, 'warn', 'primitive runtime authority unavailable',
                                  data={'status': 'UNKNOWN', 'physical_admission': False})
            runtime = native_runtime(spec)
            from clay.flow.run_admission import current_admission
            admission = current_admission()
            if primitive_only and getattr(support.native_provider, 'primitive_observation', None) is None:
                return StepResult(self.name, 'warn', 'provider has no primitive observation capability',
                    data={'status': 'UNKNOWN', 'physical_admission': False})
            if source_cir is None:
                raise ValueError("native layout schematic differs from the effective Clay source")
            if admission is not None:
                from clay.flow.primitive_observation import source_snapshot
                source_text, source_digest = source_snapshot(source_cir, ctx.execution.checkpoint)
            else:
                source_text = source_cir.read_text(encoding="utf-8")
            if source_text.strip() != emit_native_spice(
                cell_def, pdk_id=spec.pdk, top_name=spec.gds_cell_name
            ).strip():
                raise ValueError("native layout schematic differs from the effective Clay source")
            if admission is not None:
                expected_sources = {
                    str(_effective_design_clay(spec, ctx).resolve()): ctx.extra['native_design_binding']['sha256'],
                    str(source_cir.resolve()): source_digest,
                    str(Path(spec.native_runtime_manifest).resolve()): spec.native_runtime_sha256,
                }
            geometry_dir = ctx.work_dir / ("native-geometry-" + uuid.uuid4().hex[:12])
            if primitive_only:
                try:
                    native_geometry = support.native_provider.primitive_observation.materialize(
                        cell_def, output_dir=geometry_dir, image=runtime['image'], pdk_root=runtime['pdk_root'],
                        expected_runtime_files=runtime['files'],
                        timeout_seconds=min(runtime['timeout_seconds'], admission.remaining_wall_seconds()),
                        owned_execution=admission, runtime_sha256=spec.native_runtime_sha256,
                        checkpoint=admission.checkpoint, source_runtime=runtime)
                except BaseException as error:
                    evidence = geometry_dir/'native-supervision.json'
                    if evidence.is_file():
                        try:
                            error.native_container_evidence_path = str(evidence)
                        except Exception:
                            pass
                    raise
            else:
                from clay.layout.place.native_materialization_cache import (
                    materialize_or_reuse_native_geometry,
                )
                native_geometry = materialize_or_reuse_native_geometry(
                    support.native_provider,
                    cell_def,
                    output_dir=geometry_dir,
                    image=runtime["image"],
                    pdk_root=runtime["pdk_root"],
                    expected_runtime_files=runtime["files"],
                    timeout_seconds=runtime["timeout_seconds"],
                )
            ctx.extra["native_geometry"] = native_geometry
            intent = layout_intent_from_cell_def(cell_def)
            row_dir = None
            if any(array.style == 'native_diffusion' for array in intent.arrays):
                from clay.layout.tech import LayoutTech

                row_dir = ctx.work_dir / ('native-diffusion-' + uuid.uuid4().hex[:12])
                candidate_kwargs={} if not row_candidates else {'row_candidates':row_candidates}
                if primitive_only:
                    from clay.layout.native_diffusion_producer import materialize_declared_native_diffusion
                    intent, native_composite_row = materialize_declared_native_diffusion(
                        cell_def, intent, LayoutTech.resolve(spec.pdk), native_geometry,
                        output_dir=row_dir,
                        source_cir=source_cir, timeout_seconds=runtime['timeout_seconds'],**candidate_kwargs)
                else:
                    from clay.layout.place.native_materialization_cache import (
                        materialize_or_reuse_native_diffusion,
                    )
                    intent, native_composite_row = materialize_or_reuse_native_diffusion(
                        cell_def, intent, LayoutTech.resolve(spec.pdk), native_geometry,
                        output_dir=row_dir,
                        source_cir=source_cir, timeout_seconds=runtime['timeout_seconds'],**candidate_kwargs)
                ctx.extra['native_composite_row'] = native_composite_row

            if admission is not None:
                from clay.flow.primitive_observation import capture_if_supported
                try:
                    capture_if_supported(admission, provider=support.native_provider,
                            work_dir=ctx.work_dir, geometry_dir=geometry_dir,
                            row_dir=row_dir, bundle=native_geometry, row=native_composite_row,
                            cell=cell_def, source_paths=(_effective_design_clay(spec, ctx),
                            source_cir, spec.native_runtime_manifest), expected_sources=expected_sources,
                            checkpoint=ctx.execution.checkpoint)
                except BaseException as error:
                    if primitive_only:
                        try:
                            error.native_container_evidence_path = str(native_geometry.primitive_container_evidence)
                        except BaseException:
                            pass
                    raise

            if ctx.extra.get('native_composite_observation') is not None:
                from clay.layout.native_sky130 import NativeGeometryBundle
                if (admission is None or type(native_geometry) is not NativeGeometryBundle
                        or native_composite_row is None
                        or admission.primitive_observation_head() is None):
                    raise ValueError('native composite observation unavailable')
                # Capture already checks both live producer contracts and every
                # source, output and dependency before publishing the index.
                # Stop before any placement/routing policy is interpreted.
                from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved
                raise DiagnosticRoutingObserved('native composite producer captured')

            if primitive_only:
                ctx.extra['primitive_container_evidence'] = native_geometry.primitive_container_evidence
                return StepResult(self.name, 'warn', 'primitive observation captured; stage remains unknown',
                    data={'status': 'UNKNOWN', 'physical_admission': False,
                          'unknown': ['INPUT_CONSUMPTION_IDENTITY', 'LOADED_CODE_IDENTITY', 'LOADED_DOCKER_CLI_IDENTITY',
                                      'TOOLKIT_SOURCE_EXPECTATION', 'SOURCE_TERMINAL_BINDING',
                                      'BODY_CUT_COVERAGE', 'PRIMITIVE_STAGE_ADMISSION'],
                          'container_evidence_path': str(native_geometry.primitive_container_evidence)})

        if ctx.extra.get('native_composite_observation') is not None:
            raise ValueError('native composite observation unavailable')

        # v2-lang: resolve the cell's EFFECTIVE place/route policy (global
        # `policy {}` selected via `flow { policy = }`, merged with the cell's
        # local layout_intent; local wins on conflict). Falls back to engine
        # defaults when no policy is selected. Drives the two convergence
        # loops' max_iters (place_iters / route_iters); strategy fields are
        # recorded provenance.
        endpoint_request = ctx.extra.get('diagnostic_route_observer')
        endpoint_authority = None
        if (type(endpoint_request).__module__ == 'clay.layout.route.group_endpoint_observation'
                and type(endpoint_request).__name__ == 'GroupEndpointObserver'):
            from clay.layout.route.group_endpoint_observation import GroupEndpointObserver, bind_source_request
            if type(endpoint_request) is not GroupEndpointObserver:
                raise TypeError('group endpoint observation has a foreign request type')
            endpoint_authority = bind_source_request(endpoint_request)
        eff_policy = self._resolve_effective_policy(
            spec, cell_def, intent, policy_for_cell, resolve_policy, ctx=ctx
        )

        # The cell's designer intent lifted from its layout_guide.txt (when this is
        # a known probe cell); the two convergence loops iterate the layout to it.
        # On any failure the loops fall back to their intrinsic acceptance
        # (symmetry score for placement, short-freeness for routing).
        criteria = _criteria_for_cell(spec)

        # v3 learning memory (conservative, no surprise repo writes): the loops
        # learn ONLY when the policy's `@learn(on)` flag is set AND the caller
        # supplied a memory store via ctx.extra["layout_memory"]. The flow never
        # creates / persists a store on its own, so the default flow (no ctx
        # memory) writes nothing and the loops run cold + unchanged.
        layout_memory = ctx.extra.get("layout_memory")
        learning_active = bool(eff_policy.learn and layout_memory is not None)

        # Phase P: converge the PLACEMENT to its intent (symmetry / common-centroid)
        # + gf180 DRC-legality. Cheap (MILP re-solve, no Docker). A guide obligation
        # the MILP cannot express (e.g. an 8-device common-centroid array) is
        # reported (converged=False) rather than looped -- the placement is still
        # the real symmetry-constrained solver output and is emitted regardless.
        try:
            conv_place = place_to_intent(
                cell_def,
                intent,
                criteria,
                source_cir=source_cir,
                # PDK-agnostic: the placer materializes device geometry + spacing from
                # the cell's own PDK (spec.pdk) via LayoutTech, not a gf180 default.
                tech=layout_tech,
                max_iters=eff_policy.place_iters,
                memory=layout_memory if learning_active else None,
                enable_learning=learning_active,
                effective_policy=eff_policy,
                # placement engine from the policy (@place_policy(engine=seqpair)):
                # "milp" exact (default) | "seqpair" constraint-graph (overlap-free, scalable).
                # An optional reference RelativeOrder (ctx.extra) transfers a layout style.
                engine=getattr(eff_policy, "placement_engine", "milp"),
                style_reference=ctx.extra.get("style_reference"),
                native_geometry=native_geometry,
                **({} if native_composite_row is None else {'native_composite_row': native_composite_row}),
            )
        except PhysicalMacroContractError as exc:
            return StepResult(
                self.name,
                "fail",
                f"PHYSICAL_MACRO_CONTRACT: {exc}",
                data={
                    "reason": "PHYSICAL_MACRO_CONTRACT",
                    "detail": str(exc),
                },
                blocking=True,
            )
        except (BjtMaterializationUnavailable, UnsupportedPdkFactError) as exc:
            # Capability absence is neither a generic crash nor a measurable
            # geometry violation.  Publish the closed verification state before
            # any placement/GDS is promoted so downstream intent/sign can block
            # it without rewriting independent DRC/LVS legality.
            return StepResult(
                self.name,
                "fail",
                f"non-MOS matching unsupported: {exc}",
                data={
                    "reason": "NONMOS_MATCHING_UNSUPPORTED",
                    "verification_status": "unsupported",
                    "detail": str(exc),
                },
                blocking=True,
            )
        placement = conv_place.placement

        # Optional GATED envelope-min re-placement (area compaction). The aspect
        # target / gap come from (in precedence order):
        #   1. an explicit flow input (``ctx.extra["envelope_aspect_target"]`` /
        #      ``envelope_gap_um`` -- a caller-supplied flow parameter);
        #   2. the cell's DECLARED flow policy (``@place_policy(aspect=, gap=)``
        #      selected via ``flow { policy = NAME; }`` -- the design's .clay is
        #      the source of truth, consumed generally, NEVER a per-circuit
        #      constant);
        #   3. neither -> ``None`` -> the placement is left untouched, so every
        #      cell that does not opt in emits byte-identical (sha-proven).
        # The re-pack itself is the SAME CP envelope-min core the LDO/OTA folds
        # use (no new placer, no per-circuit branch). The row-band MILP goes
        # infeasible on some dense analog cells and falls back to a SPREAD
        # floorplan (~2x the reference area); the envelope-min re-pack closes
        # that gap.
        _env_aspect, _env_gap = self._compaction_knobs(eff_policy, ctx.extra)
        if _env_aspect is not None:
            from clay.layout.envelope_place import maybe_envelope_replace
            from clay.layout.place.row_replay_session import shared_row_replay_session

            # tech_id (spec.pdk) routes the compaction through the ROUTABILITY-AWARE CP
            # placer that RESERVES the R2 via-landing + min-space budget (the de-siloed
            # shared obligation). device_records supply the net binding for a
            # topology-only cell_def (empty devices) so the derived-topology intent is
            # populated -- the SAME .cir records the placer + router consume.
            _env_recs: dict[str, dict[str, Any]] = {}
            if source_cir:
                _env_recs.update(parse_cir_mos_records(source_cir))
                _env_recs.update(parse_cir_passive_records(source_cir))
            with shared_row_replay_session(
                program=getattr(placement, "row_program", None),
                materialization=getattr(placement, "row_materialization", None),
                tech=layout_tech,
                source_cell=cell_def,
            ):
                placement, _env_res = maybe_envelope_replace(
                    placement,
                    cell_def,
                    intent,
                    aspect_target=float(_env_aspect),
                    gap_um=float(_env_gap),
                    tech_id=spec.pdk,
                    device_records=(_env_recs or None),
                )
            ctx.extra["envelope_replace_result"] = _env_res

        # Phase R: converge the ROUTING. The layout step uses the intrinsic
        # short-freeness acceptance (criteria=None -> fast, Docker-free); the full
        # routing signoff (DRC / LVS / parasitic budgets via PEX) is the JOB of the
        # downstream DrcStep / LvsStep / PexStep / IntentStep on the emitted GDS, so
        # the layout step does not re-run those heavyweight gates inside its loop.
        # The router netlist carries BOTH the MOS records AND the 2-terminal
        # passive records (resistors), so it routes a resistor's p/n nets like any
        # MOS net. The keys are disjoint (M*/x* vs R*), so a plain merge is safe.
        _net_records: dict[str, dict[str, Any]] = {}
        if native_geometry is not None:
            _net_records = {device.instance_name: {"pins": dict(device.pins)} for device in cell_def.devices}
        elif source_cir:
            _net_records.update(parse_cir_mos_records(source_cir))
            _net_records.update(parse_cir_passive_records(source_cir))

        # Canonical hierarchical physical intent: build ONCE from the effective
        # .clay cell + its existing layout-intent declarations and bind that
        # immutable identity to every downstream physical artifact.  This step
        # is evidence-only in P1 (it does not perturb placement geometry); P2
        # consumes the same graph for structural candidate search.  Ambiguous
        # authored ownership is a contract error, never a silently dropped hint.
        try:
            from clay.layout.physical_macro_graph import (
                physical_macro_graph_from_intent,
            )

            physical_macro_graph = physical_macro_graph_from_intent(
                cell_def,
                intent,
                device_records=_net_records,
            )
            from clay.flow.optimized_clay import declared_compact_candidate

            effective_design_source = _effective_design_clay(spec, ctx).read_text(
                encoding="utf-8-sig"
            )
            compact_candidate = declared_compact_candidate(effective_design_source)
        except Exception as exc:  # noqa: BLE001 - named fail-closed boundary
            return StepResult(
                self.name,
                "fail",
                f"PHYSICAL_MACRO_CONTRACT: {exc}",
                data={
                    "reason": "PHYSICAL_MACRO_CONTRACT",
                    "detail": str(exc),
                },
                blocking=True,
            )
        ctx.extra["physical_macro_graph"] = physical_macro_graph

        # @equal_size on MIM caps: the placer only grew the reserved FOOTPRINT;
        # here we grow the EMITTED MIM plate geometry too, so two caps declared
        # `@equal_size(Cc, CL)` with different values emit IDENTICAL FuseTop plates
        # (the group-wise MAX plate; smaller members grow, never shrink). Runs
        # BEFORE routing so the router sees each grown cap's updated p/n terminals.
        # No-op when no equal_size group has >=2 MIM-cap members (byte-parity).
        #
        # This is best-effort hygiene (a group whose facts are unresolvable stays
        # UNEQUAL rather than sinking the whole flow) -- but "best-effort" must
        # RECORD a failure, never silently swallow it (the B1 incident: a missing
        # PDK passive fact for open_pdk raised UnsupportedPdkFactError here and
        # `except Exception: pass` erased all trace of it, so an @equal_size
        # group of external_pdk MIM caps quietly never grew). logging.warning carries the
        # exact exception text; `equal_size_cap_growth_error` carries it into the
        # flow's normal verdict channel below (mirrors drc_diagnosis /
        # routing_diagnosis: None when clean, a dict/string when there is
        # something to report).
        equal_size_cap_growth_error: "str | None" = None
        try:
            if native_geometry is None:
                _grow_equal_size_cap_plates(
                    placement,
                    _net_records,
                    layout_tech,
                    **mim_ctx,
                )
        except Exception as exc:  # noqa: BLE001 - advisory: recorded, never fatal
            equal_size_cap_growth_error = f"{type(exc).__name__}: {exc}"
            _LOG.warning(
                "%s: @equal_size MIM-cap plate growth skipped -- %s",
                spec.gds_cell_name,
                equal_size_cap_growth_error,
            )

        # Declared band/anchor/channel structure implies the structure-aware 2-D
        # route placement; a selected place policy supplies its optimization
        # targets.  This is a canonical flow lowering, not a build_ref harness
        # choice.  The helper is fail-closed and never silently falls back.
        netlist = _CirNetlist(_net_records)
        # ROUTE through the GENERAL negotiated-congestion PathFinder engine
        # (route_cell), the single general router. The device net binding comes from
        # cell_def.devices when materialised, else from the bound-.cir _net_records
        # (the SAME netlist the placer consumes) -- driven only by intent + PDK, no
        # per-circuit branch. route_cell produces a net-complete, cross-net-short-free
        # route by construction (verified by its own union-find oracle), so the
        # convergence acceptance is net-completeness; the legacy hand-rolled
        # route_generic + its spacing-knob loop are retired from production.
        from contextlib import nullcontext
        from clay.layout.flow_placement import route_with_declared_compaction
        from clay.layout.place.row_replay_session import shared_row_replay_session
        from clay.layout.route.engine_registry import route_engine_from_policy

        route_compact_fn: Any = route_with_declared_compaction
        candidate_request = ctx.extra.get("diagnostic_candidate_request")
        candidate_batch_request = ctx.extra.get("diagnostic_candidate_batch_request")
        if candidate_request is not None:
            from functools import partial
            from clay.layout.place.candidate_diagnostic import run_candidate_diagnostic
            route_compact_fn = partial(run_candidate_diagnostic, request=candidate_request)
        elif candidate_batch_request is not None:
            from functools import partial
            from clay.layout.place.candidate_batch_diagnostic import run_candidate_batch_diagnostic
            route_compact_fn = partial(run_candidate_batch_diagnostic, request=candidate_batch_request)

        # RouteEngine strategy from the cell's DECLARED @route_policy(engine=)
        # (None when the cell declares none -> route_cell's byte-identical
        # PathFinder default; an unknown declared name is a NAMED error).
        selected_route_engine = route_engine_from_policy(eff_policy)
        if endpoint_authority is not None:
            from clay.layout.route.group_endpoint_observation import bind_source_engine
            diagnostic_route_observer = bind_source_engine(endpoint_authority, selected_route_engine)
        else:
            diagnostic_route_observer = ctx.extra.get('diagnostic_route_observer')
        experimental_demand = ctx.extra.get('experimental_group_demand_guidance', False)
        if type(experimental_demand) is not bool:
            raise TypeError('experimental_group_demand_guidance must be an exact bool')
        if experimental_demand:
            from clay.layout.route.engine_gridless import GridlessEngine
            if type(selected_route_engine) is not GridlessEngine:
                raise ValueError('experimental group demand guidance requires source-selected Gridless')
            selected_route_engine.set_experimental_group_demand_guidance(True)
        experimental_feedback = ctx.extra.get('experimental_group_full_feedback', False)
        if type(experimental_feedback) is not bool:
            raise TypeError('experimental_group_full_feedback must be an exact bool')
        if experimental_feedback:
            from clay.layout.route.engine_gridless import GridlessEngine
            if type(selected_route_engine) is not GridlessEngine:
                raise ValueError('experimental group full feedback requires source-selected Gridless')
            selected_route_engine.set_experimental_group_full_feedback(True)
        seal_budget = ctx.extra.get('experimental_fixed_index_seal_budget')
        if seal_budget is not None:
            from clay.layout.route.engine_gridless import GridlessEngine
            if type(selected_route_engine) is not GridlessEngine or type(seal_budget) is not int or seal_budget <= 0:
                raise ValueError('fixed index seal requires source-selected Gridless and a positive integer budget')
            if experimental_demand or experimental_feedback:
                raise ValueError('fixed index seal cannot be combined with group routing experiments')
            selected_route_engine._experimental_fixed_index_seal_budget = seal_budget
        native_route_policy_path = None
        if native_geometry is not None:
            from dataclasses import asdict
            from clay.layout.route.canonical_input_capture import (route_engine_configuration,
                                                                    experimental_routing_configuration)
            import json

            native_route_policy_path = ctx.work_dir / "native-route-policy.json"
            selected_configuration = route_engine_configuration(selected_route_engine)
            experimental_configuration = experimental_routing_configuration(selected_route_engine)
            native_route_policy_path.write_text(json.dumps({
                "schema": ("clay.native-route-policy/v3" if experimental_configuration and 'fixed_index_seal' in experimental_configuration
                           else "clay.native-route-policy/v2" if experimental_configuration else "clay.native-route-policy/v1"),
                "source": ctx.extra["native_design_binding"],
                "declared_effective_policy": asdict(eff_policy),
                "actual_selected_engine": selected_configuration,
                **({'experimental_routing': experimental_configuration} if experimental_configuration else {}),
            }, indent=2) + "\n", encoding="utf-8", newline="\n")
        _route_replay_ctx = (
            shared_row_replay_session(
                program=getattr(placement, "row_program", None),
                materialization=getattr(placement, "row_materialization", None),
                tech=layout_tech,
                source_cell=cell_def,
            )
            if candidate_request is None and candidate_batch_request is None
            else nullcontext()
        )
        with _route_replay_ctx:
            conv_route, compact_route = route_compact_fn(
                placement,
                cell_def,
                intent,
                layout_tech,
                effective_policy=eff_policy,
                device_records=_net_records,
                max_iters=eff_policy.route_iters,
                engine=selected_route_engine,
                intent_fact_set=ctx.extra.get("intent_fact_set"),
                source_inventory=ctx.extra.get("source_inventory"),
                group_fallback_evidence_dir=ctx.work_dir / "group-fallback",
                **({"diagnostic_observer": diagnostic_route_observer}
                   if diagnostic_route_observer is not None else {}),
                **({"pre_route_gate": ctx.extra["pre_route_gate"]}
                   if ctx.extra.get("pre_route_gate") is not None else {}),
            )
        ctx.extra["declared_compact_route"] = compact_route
        from clay.layout.route.group_fallback_artifacts import completed_pairs
        group_fallback_artifacts = {key: str(path) for key, path in
                                    completed_pairs(ctx.work_dir).items()}
        routed = conv_route.cell
        route_result = conv_route.result
        if native_geometry is not None:
            if route_engine_configuration(selected_route_engine) != selected_configuration:
                raise ValueError("native route engine settings changed during execution")
            if experimental_routing_configuration(selected_route_engine) != experimental_configuration:
                raise ValueError("experimental route guidance changed during execution")
            native_geometry.require_current(native_cell(spec, ctx))
            native_runtime(spec)
        macro_consumption = None
        if (
            compact_candidate is not None
            and compact_candidate.physical_macro_graph_sha256 is not None
        ):
            from clay.layout.flow_placement import (
                certify_final_macro_consumption,
            )

            macro_consumption = certify_final_macro_consumption(
                physical_macro_graph,
                compact_candidate,
                intent,
                route_result,
            )
            ctx.extra["macro_consumption"] = macro_consumption
            if macro_consumption.status != "pass":
                return StepResult(
                    self.name,
                    "fail",
                    f"MACRO_CONSUMPTION: {macro_consumption.reason}",
                    data={
                        "reason": "MACRO_CONSUMPTION",
                        "macro_consumption": macro_consumption.canonical_data(),
                    },
                    blocking=True,
                )
        base_placement = placement
        if compact_route is not None:
            base_placement, placement = _project_compact_route_final_placement(
                compact_route,
                route_result,
                intent,
            )
        # EMIT the FULL stack via the canonical route_cell materialize path
        # (clay.layout.route.lvs_gate.emit_labelled_gds): the device FEOL (diff / poly
        # / implant / well / contact / device-M1) redrawn from the leakage-free
        # parametric geometry stamps the router routed against, + the routed BEOL, +
        # the same-net M1 / upper-metal heal (FEOL<->BEOL + collar notch closing), +
        # one on-metal1 net-name label per net, + the manufacturing-grid snap. This is
        # the SAME emitter the comparator + OTA real-deck signoffs drive (route_cell ->
        # emit_labelled_gds), so production and signoff now share ONE canonical
        # geometry path -- no second divergent emit, no PDK-gated device-emission fork.
        # Port-restricted labels (pin_nets=cell_def.ports) make the extracted layout's
        # top-level pin set equal the schematic .subckt port set for LVS. Fully generic
        # -- the only dispatch is device KIND (a pure MIM-cap cell names its plates on
        # their own upper-metal layers, written per-plate below), never circuit name.
        from clay.layout.route.lvs_gate import (
            emit_labelled_gds,
            well_merge_devices_from_cell_def,
        )

        port_nets = [str(p) for p in (getattr(cell_def, "ports", None) or [])]
        # A pure MIM-cap cell names each plate on its OWN metal label layer (bottom on
        # Metal2, top on Metal3) -- the single-metal1-label emitter cannot do that, so
        # suppress its labels (pin_nets=[]) and write the per-plate port labels below.
        # Device-KIND driven (every placed device is a 'cap'), not circuit identity.
        _cap_only = False
        try:
            _devs = getattr(placement, "devices", {}) or {}
            _cap_only = bool(_devs) and {
                getattr(d, "kind", "") for d in _devs.values()
            } == {"cap"}
        except Exception:  # noqa: BLE001
            _cap_only = False
        _placement_certificate = getattr(
            route_result, "final_placement_certificate", None
        )
        if _placement_certificate is None:
            _placement_optimization = None
        else:
            from clay.layout.place_blocks_2d import (
                placement_certificate_objective_evidence,
            )

            _placement_optimization = placement_certificate_objective_evidence(
                _placement_certificate
            )
        _placement_state_sha256 = getattr(
            _placement_certificate, "placement_state_digest", None
        )
        if callable(_placement_state_sha256):
            _placement_state_sha256 = _placement_state_sha256()
        if _placement_state_sha256 is None and compact_route is not None:
            _placement_state_sha256 = compact_route.placement_state_digest
        exact_receipt_path = Path(ctx.work_dir) / "exact_artifact_receipt.json"
        rigid_nonmos_macro_count = sum(
            1
            for macro in physical_macro_graph.macros
            if macro.kind in ("passive_bank", "bjt_cluster")
        )
        try:
            if _cap_only:
                raise ValueError(
                    "cap-only upper-metal port labels are not integrated into "
                    "the canonical exact emitter"
                )
            if rigid_nonmos_macro_count:
                from clay.layout.nonrow_macro_plan import (
                    verify_rigid_nonmos_emission_certificate,
                )

                verify_rigid_nonmos_emission_certificate(
                    getattr(
                        route_result,
                        "rigid_nonmos_emission_certificate",
                        None,
                    ),
                    getattr(route_result, "rigid_nonmos_macro_plans", None),
                    getattr(
                        route_result, "rigid_nonmos_macro_interfaces", None
                    ),
                    getattr(
                        route_result,
                        "rigid_nonmos_placement_certificate",
                        None,
                    ),
                    scope_token=getattr(
                        route_result, "physical_macro_scope_token", None
                    ),
                    compile_inputs=getattr(
                        route_result, "rigid_nonmos_macro_inputs", None
                    ),
                    geometry=route_result.geometry,
                    placement_um=route_result.placement_um,
                    pins_by_net=route_result.pins_by_net,
                )
            elif getattr(
                route_result, "rigid_nonmos_emission_certificate", None
            ) is not None:
                raise ValueError(
                    "route result has unexpected rigid non-MOS emission evidence"
                )
            allocated_route_manifest = _replay_exact_route_manifest(route_result)
            exact_artifact = emit_labelled_gds(
                route_result.routed,
                intent,
                route_result.geometry,
                route_result.placement_um,
                route_result.pins_by_net,
                Path(out_gds),
                tech=route_result.tech,
                top_cell=spec.gds_cell_name,
                pin_nets=(port_nets or None),
                # C3: `intent` is a LayoutIntent (no .devices) -> the emit-side well
                # merge needs the canonical device bulk attribution or the NW.2b
                # same-net well merge silently never fires on this production emit.
                # Prefer the route result's OWN bridged devices (they cover BOTH the
                # materialised-CellDef path and the bound-.cir records fallback --
                # the exact netlist binding that was routed); the cell_def view is
                # the fallback for a result predating the `devices` field.
                devices=(
                    tuple(getattr(route_result, "devices", ()) or ())
                    or well_merge_devices_from_cell_def(cell_def)
                ),
                passives=tuple(getattr(route_result, "passives", ()) or ()),
                source_cell=cell_def,
                native_geometry=native_geometry,
                allocated_route_manifest=allocated_route_manifest,
                return_exact_artifact=True,
                row_program=getattr(route_result, "row_program", None),
                row_materialization=getattr(route_result, "row_materialization", None),
                row_final_certificate=getattr(route_result, "row_final_placement_certificate", None),
                selected_route_engine=getattr(route_result, "selected_route_engine", None),
            )
            from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact
            from clay.pdk import exact_source_identity

            from clay.layout.route.engine_gridless import GridlessEngine
            if (isinstance(selected_route_engine, GridlessEngine)
                    and selected_route_engine.via_redundancy > 1):
                from clay.layout.route.via_array_contract import verify_router_via_arrays
                verify_router_via_arrays(
                    out_gds, route_result.routed, route_result.tech,
                    selected_route_engine.via_redundancy,
                    top_cell=spec.gds_cell_name,
                    allow_single_escape=bool(
                        getattr(selected_route_engine, "allow_single_via_escape", False)
                    ),
                )

            if type(exact_artifact) is ExactEmittedArtifact:
                exact_artifact = exact_artifact.bind_truth_manifest(
                    exact_artifact.truth_manifest.with_pdk_identity(
                        exact_source_identity(spec.pdk),
                        route_manifest_sha256=(
                            exact_artifact.route_manifest.route_manifest_sha256
                        ),
                    )
                )
            exact_artifact_receipt = _write_exact_artifact_receipt(
                exact_receipt_path,
                exact_artifact,
                physical_macro_graph_sha256=physical_macro_graph.sha256,
                candidate_sha256=(
                    compact_candidate.digest if compact_candidate is not None else None
                ),
                placement_state_sha256=(
                    str(_placement_state_sha256)
                    if _placement_state_sha256 is not None
                    else None
                ),
                rigid_nonmos_macro_count=rigid_nonmos_macro_count,
                rigid_nonmos_emission_certificate=getattr(
                    route_result, "rigid_nonmos_emission_certificate", None
                ),
            )
        except Exception as exc:  # exact emission is a blocking artifact contract
            exact_artifact_receipt = _write_exact_artifact_failure(
                exact_receipt_path, exc
            )
            return StepResult(
                self.name,
                "fail",
                "EXACT_ARTIFACT_EMISSION: " f"{type(exc).__name__}: {exc}",
                data={
                    "reason": "EXACT_ARTIFACT_EMISSION",
                    "exact_artifact": exact_artifact_receipt,
                    "physical_macro_graph_sha256": physical_macro_graph.sha256,
                    "candidate_sha256": (
                        compact_candidate.digest
                        if compact_candidate is not None
                        else None
                    ),
                },
                artifacts={"exact_artifact_receipt": str(exact_receipt_path)},
                blocking=True,
            )
        ctx.extra["exact_emitted_artifact"] = exact_artifact
        ctx.extra["exact_artifact_receipt"] = exact_artifact_receipt

        # Re-measure declared physical intent on the exact emitted artifact.
        # Placement certificates are causal evidence; they cannot satisfy an
        # artifact obligation because rails, labels, and final materialization
        # can change the enclosing geometry.  Compact-search authoring bounds
        # take precedence over one effective candidate target.
        from clay.layout.final_layout_intent import (
            check_route_result_final_layout_intent,
        )

        final_layout_intent = check_route_result_final_layout_intent(
            out_gds,
            top_cell=spec.gds_cell_name,
            authoring_cell=ctx.extra.get("authoring_cell_def") or cell_def,
            emission_source_cell=cell_def,
            effective_policy=eff_policy,
            layout_intent=intent,
            route_result=route_result,
            exact_artifact=exact_artifact,
            physical_macro_graph_sha256=physical_macro_graph.sha256,
            placement_state_sha256=(
                str(_placement_state_sha256)
                if _placement_state_sha256 is not None
                else None
            ),
            compact_candidate=compact_candidate,
            intent_fact_set=ctx.extra.get("intent_fact_set"),
            source_inventory=ctx.extra.get("source_inventory"),
        )
        ctx.extra["final_layout_intent"] = final_layout_intent
        final_intent_blocking = (
            final_layout_intent is not None and final_layout_intent.blocking
        )
        final_intent_data = (
            final_layout_intent.canonical_data()
            if final_layout_intent is not None
            else None
        )

        sha = _sha256(out_gds)
        from clay.layout.area_check import check_emitted_gds_area

        area_report = check_emitted_gds_area(
            out_gds,
            top_cell=spec.gds_cell_name,
            max_um2=getattr(intent, "area_max_um2", None),
        )
        ctx.extra["emitted_gds_area"] = area_report
        ordered_row_report = None
        emitted_channel_report = None
        if getattr(placement, "ordered_row_evidence", None) is not None:
            from clay.layout.place.ordered_mirror_rows import ordered_row_artifact_report
            from clay.layout.tech import LayoutTech

            ordered_row_report = ordered_row_artifact_report(
                placement, cell_def, intent,
                layout_tech if isinstance(layout_tech, LayoutTech) else LayoutTech.resolve(spec.pdk),
                gds_path=out_gds, top_cell=spec.gds_cell_name,
                geometry=route_result.geometry, placement_um=route_result.placement_um)
            # The reopened checker attaches evidence to the native route result,
            # not the generic convergence DTO produced by to_routed_cell().
            actual_channels = tuple(getattr(route_result.routed, "emitted_channel_report", ()) or ())
            source_channels = tuple(getattr(intent, "channel_routes", ()) or ())
            if len(actual_channels) != sum(len(channel.nets) for channel in source_channels):
                raise ValueError("source channel inventory lacks its reopened-GDS report")
            emitted_channel_report = {
                "schema": "clay.source-channel-emitted-evidence/v1",
                "gds_sha256": sha,
                "source_cell_sha256": ordered_row_report["source_cell_sha256"],
                "source_file_bindings": ordered_row_report["source_file_bindings"],
                "ordered_row_evidence_sha256": ordered_row_report["ordered_row_evidence_sha256"],
                "source_channels": [channel.canonical_data() for channel in source_channels],
                "actual_reopened_gds_checks": actual_channels,
            }
            ctx.extra["ordered_row_emitted_evidence"] = ordered_row_report
            ctx.extra["source_channel_emitted_evidence"] = emitted_channel_report
        area_blocking = (
            getattr(intent, "area_max_um2", None) is not None
            and area_report.status != "pass"
        )
        placement_quality_blocking = _placement_quality_blocking(
            placement, macro_consumption
        )
        layout_blocking = (
            area_blocking or final_intent_blocking or placement_quality_blocking
        )
        ctx.extra["routed"] = routed
        ctx.extra["base_placement"] = base_placement
        ctx.extra["placement"] = placement
        ctx.extra["route_placement_um"] = route_result.placement_um
        ctx.extra["route_geometry"] = route_result.geometry
        ctx.extra["placement_convergence"] = conv_place
        ctx.extra["routing_convergence"] = conv_route
        shorts = routed.shorts
        n_polys = len(routed.polys)  # routed BEOL (routing) polys
        # device FEOL polys materialised by the placer (poly/diff/implant/well/
        # contact/device-M1), reported alongside so "0 polys" is not misread as
        # "no geometry" -- the FEOL is always present, the routed BEOL may be 0.
        from clay.layout.flow_placement import final_route_device_polygon_count

        n_device_polys = final_route_device_polygon_count(route_result.geometry)

        # Error-intent diagnosis: when the routed cell carries connectivity
        # symptoms (different nets merged), run the fault-intent solver to turn the
        # raw violations into a decision-space diagnosis (root cause column + the
        # intent it violates + a verified single-step counterfactual + ranked
        # levers). Gated on symptoms present so a clean cell pays nothing. This is
        # the intrinsic (Docker-free) feedback channel; the KLayout-deck violations
        # join is a documented follow-up. (Validated: tools/fault_intent_eval.py --
        # synthetic localisation/intent/counterfactual 100%, compression 3.33x.)
        routing_diagnosis = None
        if shorts is not None and not getattr(shorts, "distinct", True):
            from clay.layout.fault_intent import diagnose

            dg = diagnose(
                routed, placement=placement, criteria=criteria, cell=spec.gds_cell_name
            )
            ctx.extra["routing_diagnosis"] = dg
            routing_diagnosis = {
                "n_symptoms": dg.n_symptoms,
                "n_root_causes": len(dg.root_causes),
                "root_causes": [
                    {
                        "decision": rc.decision,
                        "severity": rc.severity,
                        "nets": list(rc.nets_involved),
                        "intent_violated": rc.intent_violated,
                        "counterfactual": rc.counterfactual,
                    }
                    for rc in dg.root_causes
                ],
            }

        return StepResult(
            self.name,
            "fail" if layout_blocking else "pass",
            (
                (
                    f"GENERIC emitted-GDS area gate {area_report.status}: "
                    f"{area_report.reason}; "
                    if area_blocking
                    else ""
                )
                + (
                    "FINAL_LAYOUT_INTENT "
                    f"{final_layout_intent.status}: {final_layout_intent.reason}; "
                    if final_intent_blocking and final_layout_intent is not None
                    else ""
                )
                + (
                    "PLACEMENT_SOLVER_FAILURE: "
                    f"{placement.note or 'solver_success=False'}; "
                    if placement_quality_blocking
                    else ""
                )
                + f"GENERIC converge place+route+emit {spec.gds_cell_name} GDS "
                f"({placement.n_placed} devices, {n_device_polys} FEOL + "
                f"{n_polys} BEOL polys, "
                f"symmetry={placement.symmetry_score:.3f}; "
                f"place {'converged' if conv_place.converged else 'gap'} "
                f"{conv_place.iters}it, route "
                f"{'converged' if conv_route.converged else 'gap'} "
                f"{conv_route.iters}it) sha {sha[:12]}"
            ),
            data={
                "cell": spec.gds_cell_name,
                "gds_source": "generic_place_route_emit",
                "gds_sha256": sha,
                "exact_artifact": exact_artifact_receipt,
                "group_fallback_evidence_scope": (
                    "partial group feedback; full routing and signoff remain separate"
                    if group_fallback_artifacts else None),
                "ordered_row_emitted_evidence": ordered_row_report,
                "source_channel_emitted_evidence": emitted_channel_report,
                "area_status": area_report.status,
                "bbox_nm": area_report.bbox_nm,
                "bbox_area_um2": area_report.area_um2,
                "area_max_um2": area_report.max_um2,
                "area_reason": area_report.reason,
                "final_layout_intent": final_intent_data,
                "physical_macro_graph_sha256": physical_macro_graph.sha256,
                "placement_optimization": _placement_optimization,
                "physical_macro_count": len(physical_macro_graph.macros),
                "physical_macro_graph": physical_macro_graph.canonical_data(),
                "macro_consumption": (
                    None
                    if macro_consumption is None
                    else macro_consumption.canonical_data()
                ),
                "layout_path": "generic",
                "n_placed": placement.n_placed,
                "solver_success": placement.solver_success,
                "placement_quality_status": (
                    "fail" if placement_quality_blocking else "pass"
                ),
                "placement_quality_reason": (
                    "solver_success=False without a passing macro-consumption proof"
                    if placement_quality_blocking
                    else (
                        "passing macro-consumption proof"
                        if not placement.solver_success
                        else "placement solver succeeded"
                    )
                ),
                "used_default_geometry": placement.used_default_geometry,
                "symmetry_score": placement.symmetry_score,
                "n_polys": n_polys,
                "n_device_polys": n_device_polys,
                "routed_nets_distinct": bool(getattr(shorts, "distinct", False)),
                "routed_shorts": list(getattr(shorts, "merged_pairs", []) or []),
                "placement_note": placement.note,
                # two-phase intent-convergence provenance (the engine's stopping
                # condition, honestly reported for the dashboard).
                "criteria_source": (
                    "layout_guide" if criteria is not None else "intrinsic"
                ),
                "placement_converged": conv_place.converged,
                "placement_iters": conv_place.iters,
                "placement_residual": conv_place.residual,
                "placement_unmet": conv_place.unmet,
                "routing_converged": conv_route.converged,
                "routing_iters": conv_route.iters,
                "routing_residual": conv_route.residual,
                "routing_unmet": conv_route.unmet,
                # v2-lang effective-policy provenance (the resolved place/route
                # policy that drove the loops' max_iters + the CONSUMED
                # @place_policy(aspect/gap) compaction knobs and
                # @route_policy(engine) strategy; remaining strategy fields are
                # recorded honestly as not-yet-active engine inputs).
                "policy_source": eff_policy.source,
                "policy_place_iters": eff_policy.place_iters,
                "policy_route_iters": eff_policy.route_iters,
                "policy_tol": eff_policy.tol,
                "policy_learn": eff_policy.learn,
                "policy_aspect_target": eff_policy.aspect_target,
                "policy_gap_um": eff_policy.gap_um,
                "policy_route_engine": eff_policy.route_engine,
                **({'experimental_group_demand_guidance': getattr(selected_route_engine,
                    '_group_demand_guidance_last_report', {'status': 'UNKNOWN'})}
                   if experimental_demand else {}),
                # error-intent diagnosis (None when the cell is short-free): the
                # decision-space explanation of any connectivity symptoms.
                "routing_diagnosis": routing_diagnosis,
                # @equal_size MIM-cap plate-growth advisory (None when the group
                # grew cleanly or there was no group to grow) -- the recorded
                # trace of a swallowed exception, never a silent no-op.
                "equal_size_cap_growth_error": equal_size_cap_growth_error,
            },
            artifacts={
                "layout_gds": str(out_gds),
                "exact_artifact_receipt": str(exact_receipt_path),
                **group_fallback_artifacts,
                **({"native_route_policy": str(native_route_policy_path),
                    "native_geometry": str(native_geometry.output_dir / "geometry.json")}
                   if native_geometry is not None else {}),
            },
            blocking=layout_blocking,
        )

    @staticmethod
    def _compaction_knobs(eff_policy, ctx_extra) -> "tuple[float | None, float]":
        """(aspect_target, gap_um) for the gated envelope-min re-placement.

        Precedence: an explicit flow input (``ctx.extra``) wins, else the
        cell's DECLARED ``@place_policy(aspect=, gap=)`` from its selected flow
        policy, else ``(None, 1.5)`` -- aspect ``None`` means NO re-placement
        (the raw placement stays byte-identical). Pure derivation; no circuit
        identity anywhere."""
        aspect = ctx_extra.get("envelope_aspect_target")
        if aspect is None:
            aspect = getattr(eff_policy, "aspect_target", None)
        gap = ctx_extra.get("envelope_gap_um")
        if gap is None:
            gap = getattr(eff_policy, "gap_um", None)
        if gap is None:
            gap = 1.5
        return (None if aspect is None else float(aspect)), float(gap)

    @staticmethod
    def _resolve_effective_policy(
        spec, cell_def, intent, policy_for_cell, resolve_policy, *, ctx=None
    ):
        """Resolve the cell's EffectivePolicy, robust to a missing ClayFile.

        Prefers the full file-level lookup (`policy_for_cell`) by re-parsing
        the design.clay to reach the policy registry; falls back to resolving
        directly from `cell_def.flow_decl.policy_ref` against a freshly parsed
        registry, and ultimately to engine defaults.
        """
        # A promoted optimized source is a circuit revision.  Its policy bytes
        # are authoritative over the CircuitSpec snapshot compiled from the
        # authoring source at registry discovery time.  Resolve it first and
        # fail closed; silently falling back here would make downstream layout
        # consume optimized sizing/intent with the stale authoring policy.
        effective_override = (
            ctx.extra.get("effective_design_clay") if ctx is not None else None
        )
        if effective_override:
            from clay.parser import parse_clay_file

            design = Path(effective_override)
            if not design.is_file():
                raise FileNotFoundError(
                    f"effective design Clay policy source missing: {design}"
                )
            cf = parse_clay_file(design)
            matches = [
                cell
                for cell in (getattr(cf, "cell_defs", []) or [])
                if getattr(cell, "local_name", None) == spec.top_cell
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"effective design Clay requires exactly one "
                    f"{spec.top_cell!r} cell; found {len(matches)}"
                )
            return policy_for_cell(cf, matches[0])

        compiled_decl = getattr(spec, "policy_decl", None)
        if compiled_decl is not None:
            return resolve_policy(compiled_decl, layout_intent=intent)

        declared_ref = getattr(spec, "policy_ref", None) or getattr(
            getattr(cell_def, "flow_decl", None), "policy_ref", None
        )
        try:
            design = (
                _effective_design_clay(spec, ctx)
                if ctx is not None
                else Path(spec.case_dir) / "source" / "design.clay"
            )
            if Path(design).is_file():
                from clay.parser import parse_clay_file

                cf = parse_clay_file(design)
                for cell in getattr(cf, "cell_defs", []) or []:
                    if getattr(cell, "local_name", None) == spec.top_cell:
                        reparsed_ref = getattr(
                            getattr(cell, "flow_decl", None), "policy_ref", None
                        )
                        if reparsed_ref and reparsed_ref not in (
                            getattr(cf, "policies", None) or {}
                        ):
                            break
                        return policy_for_cell(cf, cell)
        except Exception:
            pass
        if declared_ref:
            raise ValueError(
                f"cell {spec.top_cell!r} selects policy {declared_ref!r}, but "
                "its declaration was not preserved by flow compilation and "
                "cannot be resolved from the case source"
            )
        # Fallback: resolve against the local intent only (defaults unless the
        # intent declares a tighter tol).
        try:
            return resolve_policy(None, layout_intent=intent)
        except Exception:
            from clay.layout.policy import EffectivePolicy

            return EffectivePolicy()

    @staticmethod
    def _parse_cell_def(spec: CircuitSpec):
        """Re-parse the design.clay and return the flow cell (no compile step)."""
        design = spec.case_dir / "source" / "design.clay"
        if not Path(design).is_file():
            return None
        from clay.parser import parse_clay_file

        cf = parse_clay_file(design)
        for cell in getattr(cf, "cell_defs", []) or []:
            if getattr(cell, "local_name", None) == spec.top_cell:
                return cell
        return None

    # ----------------------------------------------------------------- #
    # Legacy per-circuit hook (circuits.* retired -> fail closed)       #
    # ----------------------------------------------------------------- #
    def _run_per_circuit(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        if _is_retired_legacy_layout_module(spec.layout_module):
            raise RetiredLegacyLayoutError(
                f"Legacy per-circuit layout generator {spec.layout_module!r} has been retired. "
                "Declare `layout = generic;` in `source/design.clay` and run via the unified "
                "Clay entry point (`python -m clay flow run --case <case_dir>`)."
            )
        out_gds = gds_path(ctx)
        out_gds.parent.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(spec.layout_module)
        routed = getattr(module, spec.build_function)()
        source = "emitted"
        error = None
        try:
            if spec.emit_function:
                getattr(module, spec.emit_function)(
                    routed, str(out_gds), cell_name=spec.gds_cell_name
                )
            else:
                shutil.copyfile(spec.fallback_gds, out_gds)
                source = "copied_existing"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if not Path(spec.fallback_gds).is_file():
                raise
            shutil.copyfile(spec.fallback_gds, out_gds)
            source = "copied_fallback_after_emit_error"
        sha = _sha256(out_gds)
        ctx.extra["routed"] = routed
        return StepResult(
            self.name,
            "pass",
            f"emitted {spec.gds_cell_name} GDS ({source}) sha {sha[:12]}",
            data={
                "cell": spec.gds_cell_name,
                "gds_source": source,
                "gds_sha256": sha,
                "emit_error": error,
            },
            artifacts={"layout_gds": str(out_gds)},
        )


class _CirNetlist:
    """Minimal netlist adapter the generic router consumes.

    The router reads ``netlist.devices`` -> each device's ``instance_name`` +
    ``pins`` (``{'d':net,'g':net,'s':net,'b':net}``). The production cells carry
    NO sized devices on the parsed CellDef (their devices live in the source
    .cir), so this projects the .cir MOS records into exactly that shape, keyed by
    the same device label the placer used (``xMP1`` -> ``MP1``).
    """

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.devices = [
            _CirDevice(instance_name=lbl, pins=dict(rec.get("pins", {})))
            for lbl, rec in records.items()
        ]


class _CirDevice:
    __slots__ = ("instance_name", "pins")

    def __init__(self, instance_name: str, pins: dict[str, str]) -> None:
        self.instance_name = instance_name
        self.pins = pins


def _report_summary_for_pdk(summary: str, pdk: str) -> str:
    """Relabel the generic gf180 runner wrapper with its actual deck owner."""
    return re.sub(r"^gf180mcu(?=\s+(?:DRC|LVS):)", str(pdk), str(summary), count=1)


class DrcStep:
    name = "drc"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        from clay.flow.native_physical import native_input_closure
        return native_input_closure(spec, ctx, gds_path(ctx), _effective_source_cir(spec, ctx))

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        from clay.flow.native_physical import native_verification_step

        native = native_verification_step("drc", spec, ctx, gds=gds_path(ctx),
                                          reference=_effective_source_cir(spec, ctx))
        if native is not None:
            return native
        deck = _resolve_klayout_deck(spec, "drc")
        if deck is None:
            return StepResult(
                self.name,
                "skipped",
                f"no KLayout DRC deck registered for PDK '{spec.pdk}' "
                f"(get_pdk('{spec.pdk}').klayout_decks() has no 'drc')",
                data={"reason": "no_deck", "pdk": spec.pdk},
            )
        from tools.klayout_drc_signoff_v139.run import run_gf180_drc

        gds = gds_path(ctx)
        report_path = Path(ctx.work_dir) / "layout.lyrdb"
        # When the cell contains a MIM capacitor the gf180 DRC deck must run its
        # MIM-Option-A section (gated on ``$mim_option``/``$mim_cap``); otherwise
        # the MIM.* plate/via rules never execute. Enabled ONLY when the source
        # has a C card so a cap-free cell's invocation is byte-unchanged.
        drc_extra_rd = _mim_drc_rd(spec, cell_def=ctx.extra.get("cell_def"))
        report = run_gf180_drc(
            gds,
            report_path=report_path,
            deck=deck,
            topcell=spec.gds_cell_name,
            extra_rd=drc_extra_rd or None,
            timeout_sec=600,
        )
        clean = report.is_clean()
        artifacts = {}
        if report.lyrdb_path:
            artifacts["drc_report"] = str(report.lyrdb_path)

        # Encapsulate the error-intent diagnosis INTO the DRC check: when the deck
        # finds violations AND this flow routed the cell (so the tagged geometry's
        # net provenance is available in ctx.extra), join the lyrdb violations to
        # that provenance and emit the decision-space diagnosis (root-cause column
        # + violated rule/intent + verified counterfactual + ranked levers) instead
        # of leaving the agent a flat coordinate list. General (any circuit); a
        # clean run or a run without routed provenance attaches nothing.
        drc_diagnosis = None
        routed = ctx.extra.get("routed")
        if (
            (not clean)
            and report.lyrdb_path
            and routed is not None
            and getattr(routed, "tagged", None)
        ):
            try:
                from clay.layout.drc_report import parse_lyrdb
                from clay.layout.fault_intent import diagnose

                viol = parse_lyrdb(report.lyrdb_path)
                if viol:
                    dg = diagnose(
                        routed,
                        placement=ctx.extra.get("placement"),
                        criteria=_criteria_for_cell(spec),
                        cell=spec.gds_cell_name,
                        drc_violations=viol,
                    )
                    ctx.extra["drc_diagnosis"] = dg
                    drc_diagnosis = {
                        "n_violations": dg.n_symptoms,
                        "n_root_causes": len(dg.root_causes),
                        "root_causes": [
                            {
                                "decision": rc.decision,
                                "nets": list(rc.nets_involved),
                                "intent_violated": rc.intent_violated,
                                "counterfactual": rc.counterfactual,
                            }
                            for rc in dg.root_causes
                        ],
                    }
            except (
                Exception
            ) as exc:  # noqa: BLE001 - diagnosis is advisory, never fatal
                drc_diagnosis = {"error": f"{type(exc).__name__}: {exc}"}

        return StepResult(
            self.name,
            "pass" if clean else "fail",
            _report_summary_for_pdk(report.summary(), spec.pdk),
            data={
                "is_clean": clean,
                "n_violations": report.total_violations,
                "by_rule": report.counts_by_rule(),
                "drc_diagnosis": drc_diagnosis,
            },
            artifacts=artifacts,
            blocking=not clean,
        )


class LvsStep:
    name = "lvs"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        from clay.flow.native_physical import native_input_closure
        return native_input_closure(spec, ctx, gds_path(ctx), _effective_source_cir(spec, ctx))

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        from clay.flow.native_physical import native_verification_step

        native = native_verification_step("lvs", spec, ctx, gds=gds_path(ctx),
                                          reference=_effective_source_cir(spec, ctx))
        if native is not None:
            return native
        deck = _resolve_klayout_deck(spec, "lvs")
        if deck is None:
            return StepResult(
                self.name,
                "skipped",
                f"no KLayout LVS deck registered for PDK '{spec.pdk}' "
                f"(get_pdk('{spec.pdk}').klayout_decks() has no 'lvs')",
                data={"reason": "no_deck", "pdk": spec.pdk},
            )
        from tools.klayout_lvs_signoff_v145.run import run_gf180_lvs

        gds = gds_path(ctx)
        work = Path(ctx.work_dir)
        # A post-layout-only package (e.g. current_mirror / five_t_ota) ships an
        # extracted ``*.pex.spice`` but NO schematic ``.cir``, so there is no
        # source-of-truth netlist to build the effective-W LVS reference from.
        # That is not an LVS *failure* (nothing mismatched) — it is the absence
        # of a comparable, so LVS is SKIPPED (never a blocking fail that sinks
        # the whole flow). A package that HAS a ``.cir`` still runs real LVS and
        # a topology/sizing mismatch fails as before.
        source_cir = _effective_source_cir(spec, ctx)
        if not source_cir.is_file():
            return StepResult(
                self.name,
                "skipped",
                f"no schematic .cir for {spec.gds_cell_name} "
                f"({source_cir.name} absent) — post-layout-only package, "
                f"LVS reference cannot be built; skipping (not a mismatch)",
                data={"source_cir": str(source_cir), "reason": "no_schematic_cir"},
            )
        # Optional per-cap as-drawn finger geometry hint, declared in the case
        # package (``metadata.cap_geometry``). It lets an ideal schematic ``C``
        # the LAYOUT realises as a foldable MOS-cap lower to a comparable
        # pmoscap_3p3 device instead of staying cap-device debt. Read from the
        # case.yaml here (the spec does not surface raw metadata); absent / no
        # hint -> the cap stays honest cap_debt.
        cap_geometry = _read_case_cap_geometry(spec.case_dir)
        # SOURCE-OF-TRUTH SYNC: the extracted device W is the LAYOUT's W*nf*m, and
        # the layout was built from the parsed ``design.clay`` cell (cell_def), not
        # from the schematic ``.cir``. When the two source files disagree on a
        # device's finger count (they are independently maintained), the reference
        # must fold by the LAYOUT cell's nf/m or every re-fingered device fails the
        # deck's W comparison. Lift the layout cell's per-instance (nf, m) from the
        # compile step's cell_def (when present) and pass it as the fold override.
        # GENERIC: keyed by instance name from the cell's own sizing data, no
        # per-circuit literal; an empty / absent cell_def -> no override -> the
        # ``.cir`` nf folding is byte-unchanged.
        _cell_def = ctx.extra.get("cell_def")
        if _cell_def is None:
            # The compile step may have been skipped (incremental cache) so the
            # parsed cell is not in ctx; re-parse the design.clay to recover the
            # layout sizing. Best-effort -- a parse failure -> no override (the
            # ``.cir`` folding stands).
            try:
                design = spec.case_dir / "source" / "design.clay"
                if Path(design).is_file():
                    from clay.parser import parse_clay_file

                    cf = parse_clay_file(design)
                    for cell in getattr(cf, "cell_defs", []) or []:
                        if getattr(cell, "local_name", None) == spec.top_cell:
                            _cell_def = cell
                            break
            except Exception:  # noqa: BLE001
                _cell_def = None
        fold_overrides = _layout_fold_overrides(_cell_def)
        # Resolve the substrate ONCE from the cell's semantic port role and use
        # that same value for both reference lowering and the extraction deck.
        # The adapter owns the fallback when the source declares no ground role.
        lvs_sub = _declared_ground_net(
            _cell_def,
            fallback=_lvs_ref_profile(spec.pdk).substrate,
        )
        from clay.well_isolation import deep_well_lvs_model_overrides

        deep_model_overrides = deep_well_lvs_model_overrides(_cell_def, spec.pdk)
        deep_model_kwargs = (
            {"device_model_overrides": deep_model_overrides}
            if deep_model_overrides
            else {}
        )
        mim_ctx = _gf180_mim_context(spec, cell_def=_cell_def)
        ref = write_effective_lvs_reference(
            source_cir,
            work / f"{spec.key}_lvs_ref_effective.cir",
            cell_name=spec.gds_cell_name,
            cap_geometry=cap_geometry,
            pdk=spec.pdk,
            fold_overrides=fold_overrides or None,
            substrate_net=lvs_sub,
            layout_cell_def=_cell_def,
            **mim_ctx,
            **deep_model_kwargs,
        )
        # The deck binds a poly-over-substrate resistor's bulk + its global
        # substrate net to the ``lvs_sub`` net; it must equal the reference's
        # bulk terminal. Both now come from the cell's declared ground role,
        # falling back together to the PDK adapter's substrate_net() (and the
        # historical gf180 ``vss`` when the adapter is silent).
        # A MIM cap in the cell needs the deck's MIM extraction enabled
        # ($mim_option/$mim_cap); without it the layout's FuseTop/Metal plates are
        # never extracted as a ``mim_<d>fF`` device and the C card has no layout
        # counterpart. Empty (byte-unchanged invocation) for a cap-free cell.
        lvs_extra_rd = _mim_drc_rd(spec, cell_def=_cell_def)
        # When the cell embeds a gf180 MIM cap, run against a Clay-maintained
        # corrected deck that adds the one missing ``connect(metal2_ncap,
        # mim_virtual)`` line so the MIM bottom plate joins the routed BEOL net.
        # The open gf180mcu.lvs deck OMITS that bridge (it connects the bottom
        # plate only to raw ``metal2``, never to the derived routable
        # ``metal2_ncap``), severing an embedded cap's bottom plate -- a
        # transcription bug vs the sky130 sibling deck, which DOES bridge
        # (``connect(met3_ncap, met3_con)``). The foundry deck in ``pdks/`` is
        # never edited; the fix lives only in the run-time sibling copy and only
        # affects MIM bottom-plate connectivity (a non-MIM cell is unaffected).
        # See tools/klayout_lvs_signoff_v145/gf180_mim_botplate_connect_fix.md.
        # Gated on the SAME ``_mim_drc_rd`` signal (a C card + a PDK MIM stack)
        # as the MIM extraction switches, so a cap-free cell's run is unchanged.
        mim_botplate_fix = bool(lvs_extra_rd)
        if str(spec.pdk).casefold() == "gf180mcu":
            # The foundry default simplifies only the extracted layout. Apply
            # its same topology-preserving series/parallel reduction to the
            # schematic so a real two-resistor series chain is comparable on
            # both sides. This changes no device dimensions or tolerance.
            lvs_extra_rd = list(lvs_extra_rd) + [("schematic_simplify", "true")]
        report = run_gf180_lvs(
            gds,
            ref,
            report_path=work / "layout.lvsdb",
            extracted_netlist_path=work / "layout_extracted.cir",
            uncombined_netlist_path=work / "layout_uncombined.cir",
            deck=deck,
            lvs_sub=lvs_sub,
            extra_rd=lvs_extra_rd or None,
            timeout_sec=900,
            # Absorb the sub-0.01 % layout grid / serpentine quantization between
            # a schematic poly-R's value-derived geometry (L = R / sheet_rho) and
            # its as-drawn length; the deck compares ppolyf_u on W/L with zero
            # default tolerance. Geometry only -- never topology or MOS sizing; a
            # real resistor error is off by >= one unit cell (tens of percent).
            passive_geom_tolerance=0.02,
            mim_botplate_connect_fix=mim_botplate_fix,
            mim_option_scope_fix=mim_botplate_fix,
        )
        # Read the reference writer's debt sidecar: device classes that the
        # ideal schematic .cir cannot lower losslessly to the laid-out gf180
        # device (an ideal C card vs a MOS-cap / pmoscap finger array). Surfaced
        # in the verdict so cap-device verification debt is never hidden.
        import json as _json

        cap_debt: list[str] = []
        unlowered: list[str] = []
        meta_path = Path(str(ref) + ".meta.json")
        if meta_path.is_file():
            try:
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                cap_debt = list(meta.get("cap_debt") or [])
                unlowered = list(meta.get("unlowered") or [])
            except (ValueError, OSError):
                pass

        clean = report.is_clean()
        artifacts = {"lvs_reference": str(ref)}
        deck_provenance = getattr(report, "deck_provenance", {})
        if deck_provenance.get("receipt"):
            artifacts["lvs_deck_receipt"] = str(deck_provenance["receipt"])
        if report.lvsdb_path:
            artifacts["lvs_report"] = str(report.lvsdb_path)
        if report.extracted_netlist_path:
            artifacts["lvs_extracted"] = str(report.extracted_netlist_path)
        if deck_provenance.get("uncombined_netlist"):
            artifacts["lvs_uncombined"] = str(deck_provenance["uncombined_netlist"])
        summary = _report_summary_for_pdk(report.summary(), spec.pdk)
        if cap_debt:
            summary += (
                f"; cap-device debt: {len(cap_debt)} ideal C card(s) "
                f"({', '.join(cap_debt)}) not lowered to laid-out MOS-cap"
            )
        return StepResult(
            self.name,
            "pass" if clean else "fail",
            summary,
            data={
                "clean": clean,
                "extracted_devices": report.device_count,
                "extracted_nets": report.net_count,
                "models": report.devices_by_model,
                "deck_provenance": deck_provenance,
                "cap_debt": cap_debt,
                "unlowered_devices": unlowered,
            },
            artifacts=artifacts,
            blocking=not clean,
        )


class PexStep:
    name = "pex"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        from clay.flow.native_physical import native_input_closure
        return native_input_closure(spec, ctx, gds_path(ctx), _effective_source_cir(spec, ctx))

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        from clay.flow.native_physical import native_verification_step

        native = native_verification_step("pex", spec, ctx, gds=gds_path(ctx),
                                          reference=_effective_source_cir(spec, ctx))
        if native is not None:
            return native
        from clay.layout.pex import get_backend

        gds = gds_path(ctx)
        # The case declares its PEX backend (``metadata.backend``: ``geometric`` /
        # ``magic`` / ``auto``). Default ``auto`` resolves to Magic when its toolchain
        # (Docker / iic-osic-tools) is available and falls back to the geometric
        # backend otherwise -- so a host without Magic still signs on the honest
        # calibrated=False geometric PEX the case declares (no hard Magic dependency).
        # A case that explicitly requests ``magic`` and lacks the toolchain surfaces
        # that as the honest backend error rather than silently downgrading.
        backend_kind = (
            str(_read_case_metadata_value(spec.case_dir, "backend") or "auto")
            .strip()
            .lower()
        )
        if backend_kind not in ("geometric", "magic", "auto"):
            backend_kind = "auto"
        from clay.pdk import get_pdk

        backend = get_backend(
            backend_kind,
            pdk_id=spec.pdk,
            ground_net=_declared_ground_net(
                ctx.extra.get("cell_def"),
                fallback=get_pdk(spec.pdk).substrate_net().casefold(),
            ).casefold(),
        )
        # A route-owned geometric report is not by itself a replayable PEX
        # artifact.  Whenever the canonical generic flow has the fresh route and
        # same-run full-extraction LVSDB, compose the geometric caps with that
        # extracted topology and emit ``<cell>.pex.spice``.  This is the same
        # mature path post-simulation already consumes; physical-ledger flows
        # without a post Driver must not fall back to a cap-only in-memory report.
        convergence = ctx.extra.get("routing_convergence")
        route_result = getattr(convergence, "result", None)
        lvs_result = ctx.results.get("lvs")
        lvsdb_value = lvs_result.artifacts.get("lvs_report") if lvs_result else None
        cell_def = ctx.extra.get("cell_def")
        same_run_missing: list[str] = []
        if route_result is None:
            same_run_missing.append("fresh route evidence")
        if not lvsdb_value or not Path(lvsdb_value).is_file():
            same_run_missing.append("full-extraction LVSDB")
        if cell_def is None:
            same_run_missing.append("compiled cell intent")
        if getattr(backend, "name", "") == "geometric" and not same_run_missing:
            try:
                return self._run_geometric_post_truth(
                    spec,
                    ctx,
                    backend=backend,
                    route_result=route_result,
                    lvsdb=Path(lvsdb_value),
                )
            except Exception as exc:  # noqa: BLE001 - named fail-closed gate
                scope = "post-truth" if spec.postsim_function else "physical-artifact"
                return StepResult(
                    self.name,
                    "fail",
                    f"{scope} geometric PEX failed: " f"{type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                    blocking=True,
                )
        # The LVS artifact deliberately labels TOP ports only: the foundry
        # comparer treats every GDS text label as a top-level schematic pin.
        # Magic, however, needs names on INTERNAL routed nets for parasitic
        # intent (@match/@budget) and post simulation.  Whenever the canonical
        # same-run route + LVS evidence is available, re-emit the established
        # observation GDS with ``pin_nets=None`` and compose its RC with the
        # full-extraction topology.  This is required for physical-ledger flows
        # even without a post Driver; otherwise internal nets become coordinate
        # aliases and a declared obligation is incorrectly UNMEASURABLE.
        if getattr(backend, "name", "") == "magic" and not same_run_missing:
            try:
                return self._run_joint_post_truth(
                    spec,
                    ctx,
                    backend=backend,
                    route_result=route_result,
                    cell_def=cell_def,
                    lvsdb=Path(lvsdb_value),
                )
            except Exception as exc:  # noqa: BLE001 - named fail-closed gate
                scope = "post-truth" if spec.postsim_function else "physical-artifact"
                return StepResult(
                    self.name,
                    "fail",
                    f"{scope} joint PEX failed: {type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                    blocking=True,
                )
        # A declared post Driver makes PEX binding truth, not an advisory
        # viewer estimate.  Such a run must keep the fresh route object and the
        # same-run full-extraction LVSDB so internal-net labels and extracted
        # passive devices can be composed without schematic injection.
        if spec.postsim_function:
            if same_run_missing:
                return StepResult(
                    self.name,
                    "fail",
                    "post-truth joint PEX requires same-run "
                    + ", ".join(same_run_missing),
                    data={"missing_evidence": same_run_missing},
                    blocking=True,
                )
            if getattr(backend, "name", "") != "magic":
                return StepResult(
                    self.name,
                    "fail",
                    "post-truth joint PEX requires calibrated Magic RC; "
                    f"resolved backend={getattr(backend, 'name', type(backend).__name__)!r}",
                    data={"backend": getattr(backend, "name", None)},
                    blocking=True,
                )
            try:
                return self._run_joint_post_truth(
                    spec,
                    ctx,
                    backend=backend,
                    route_result=route_result,
                    cell_def=cell_def,
                    lvsdb=Path(lvsdb_value),
                )
            except Exception as exc:  # noqa: BLE001 - named fail-closed gate
                return StepResult(
                    self.name,
                    "fail",
                    f"post-truth joint PEX failed: {type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                    blocking=True,
                )
        report = backend.extract(str(gds), spec.gds_cell_name, spec.pdk)
        ctx.extra["pex_report"] = report
        n_caps = len(report.caps)
        pex_spice = Path(ctx.work_dir) / f"{spec.gds_cell_name}.pex.spice"
        artifacts = {}
        if pex_spice.is_file():
            artifacts["pex_spice"] = str(pex_spice)
        viewer = pex_report_to_viewer_json(report)
        ctx.extra["parasitics"] = viewer
        return StepResult(
            self.name,
            "pass" if n_caps > 0 else "warn",
            f"{report.backend} PEX extracted {n_caps} capacitors "
            f"(calibrated={bool(report.calibrated)}).",
            data={
                "n_caps": n_caps,
                "total_cap_fF": report.total_cap_fF,
                "backend": report.backend,
                "calibrated": bool(report.calibrated),
                "n_devices": len(getattr(report, "resistors", ()) or ()),
            },
            artifacts=artifacts,
        )

    @staticmethod
    def _load_same_run_resistor_evidence(
        ctx: FlowContext,
        lvsdb: Path,
        *,
        expected_gds_sha256: str | None = None,
    ) -> tuple[dict[str, str], str | None, str | None, Path | None, Path | None]:
        import hashlib
        import klayout.db as kdb
        from clay.layout.route.full_extraction_body_check import _lvs_net_xref_mapping

        database = kdb.LayoutVsSchematic()
        database.read(str(lvsdb))
        net_mapping = _lvs_net_xref_mapping(database)
        lvs_result = ctx.results.get("lvs")
        raw_path_value = lvs_result.artifacts.get("lvs_uncombined") if lvs_result else None
        raw_path: Path | None = None
        reference_path: Path | None = None
        raw_text: str | None = None
        physical_resistor_reference_text: str | None = None
        if raw_path_value:
            raw_path = Path(raw_path_value)
            provenance = lvs_result.data.get("deck_provenance", {})
            if (
                hashlib.sha256(raw_path.read_bytes()).hexdigest()
                != provenance.get("uncombined_netlist_sha256")
            ):
                raise RuntimeError("unsimplified resistor extraction changed since same-run LVS")
            if expected_gds_sha256 and provenance.get("gds_sha256") != expected_gds_sha256:
                raise RuntimeError("unsimplified resistor extraction is not from the routed GDS")
            raw_text = raw_path.read_text(encoding="utf-8")
            reference_path = Path(lvs_result.artifacts["lvs_reference"])
            if hashlib.sha256(reference_path.read_bytes()).hexdigest() != provenance.get("schematic_sha256"):
                raise RuntimeError("Physical resistor reference changed since same-run LVS")
            physical_resistor_reference_text = reference_path.read_text(encoding="utf-8")
        return (
            net_mapping,
            raw_text,
            physical_resistor_reference_text,
            raw_path,
            reference_path,
        )

    @staticmethod
    def _write_resistor_chain_evidence(
        ctx: FlowContext,
        *,
        resistor_proof: dict[str, Any],
        raw_path: Path | None,
        reference_path: Path | None,
        physical_gds_sha256: str | None,
        lvsdb: Path,
    ) -> dict[str, str]:
        if not resistor_proof:
            return {}
        import hashlib
        import json

        if raw_path is not None:
            resistor_proof["raw_netlist"] = str(raw_path)
            resistor_proof["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if reference_path is not None:
            resistor_proof["physical_resistor_reference"] = str(reference_path)
            resistor_proof["physical_resistor_reference_sha256"] = hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()
        if physical_gds_sha256:
            resistor_proof["gds_sha256"] = physical_gds_sha256
        resistor_proof["lvsdb"] = str(lvsdb)
        resistor_proof["lvsdb_sha256"] = hashlib.sha256(lvsdb.read_bytes()).hexdigest()
        resistor_path = Path(ctx.work_dir) / "resistor_chain_evidence.json"
        resistor_path.write_text(json.dumps(resistor_proof, indent=2) + "\n", encoding="utf-8")
        return {"resistor_chain_evidence": str(resistor_path)}

    @staticmethod
    def _run_joint_post_truth(
        spec: CircuitSpec,
        ctx: FlowContext,
        *,
        backend: Any,
        route_result: Any,
        cell_def: Any,
        lvsdb: Path,
    ) -> StepResult:
        import hashlib
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.pex.joint_extraction import (
            declared_resistor_network_from_cell,
            magic_device_models_for_pdk,
            write_joint_pex,
        )
        from clay.layout.route.lvs_gate import emit_labelled_gds
        from clay.pdk import get_pdk
        from clay.well_isolation import declared_well_isolated_devices

        mim_ctx = _gf180_mim_context(
            spec,
            cell_def=cell_def,
            tech=getattr(route_result, "tech", None),
        )
        obs_tech = route_result.tech
        if mim_ctx and hasattr(obs_tech, "with_mim_config"):
            obs_tech = obs_tech.with_mim_config(**mim_ctx)
        observation_dir = Path(ctx.work_dir) / "pex_observation"
        observation_dir.mkdir(parents=True, exist_ok=True)
        observation_gds = observation_dir / "layout.gds"
        obs_intent = layout_intent_from_cell_def(cell_def)
        _attach_gf180_mim_attrs(obs_intent, mim_ctx)
        emit_labelled_gds(
            route_result.routed,
            obs_intent,
            route_result.geometry,
            route_result.placement_um,
            route_result.pins_by_net,
            observation_gds,
            tech=obs_tech,
            top_cell=spec.gds_cell_name,
            pin_nets=None,
            devices=tuple(getattr(route_result, "devices", ()) or ()),
            source_cell=cell_def,
            row_program=getattr(route_result, "row_program", None),
            row_materialization=getattr(route_result, "row_materialization", None),
            row_final_certificate=getattr(route_result, "row_final_placement_certificate", None),
            selected_route_engine=getattr(route_result, "selected_route_engine", None),
        )
        report = backend.extract(
            str(observation_gds),
            spec.gds_cell_name,
            spec.pdk,
            with_resistance=True,
        )
        if not bool(report.calibrated):
            raise RuntimeError("Magic RC report is not calibrated")
        raw_pex = observation_dir / f"{spec.gds_cell_name}.pex.spice"
        if not raw_pex.is_file() or raw_pex.stat().st_size <= 0:
            raise RuntimeError("Magic extresist produced no raw PEX netlist")

        ground_net = _declared_ground_net(
            cell_def,
            fallback=get_pdk(spec.pdk).substrate_net().casefold(),
        )
        isolated_names = declared_well_isolated_devices(cell_def)
        isolated_bodies: set[str] = set()
        if isolated_names:
            iso_set = set(isolated_names)
            resolved_names: set[str] = set()
            for dev in tuple(getattr(route_result, "devices", ()) or ()):
                dname = str(getattr(dev, "name", None) or getattr(dev, "instance_name", "") or "")
                key = dname if dname in iso_set else dname.lstrip("xX")
                if key in iso_set:
                    nets = getattr(dev, "nets", None) or getattr(dev, "pins", None) or {}
                    body = nets.get("body") or nets.get("b")
                    if body:
                        isolated_bodies.add(str(body))
                        resolved_names.add(key)
            if resolved_names != iso_set:
                eff_cir = _effective_source_cir(spec, ctx)
                if eff_cir.is_file():
                    from clay.layout.generic_placement import parse_cir_mos_records

                    for dname, rec in (parse_cir_mos_records(eff_cir) or {}).items():
                        key = dname if dname in iso_set else dname.lstrip("xX")
                        if key in iso_set and key not in resolved_names:
                            body = (rec.get("pins") or {}).get("b")
                            if body:
                                isolated_bodies.add(str(body))
                                resolved_names.add(key)
            for dev in tuple(getattr(cell_def, "devices", ()) or ()):
                dname = str(getattr(dev, "instance_name", "") or "")
                if dname in iso_set and dname not in resolved_names:
                    body = (getattr(dev, "pins", None) or {}).get("b")
                    if body:
                        isolated_bodies.add(str(body))
                        resolved_names.add(dname)

        layout_res = ctx.results.get("layout")
        layout_gds_value = layout_res.artifacts.get("layout_gds") if layout_res else None
        physical_gds_sha256 = (
            hashlib.sha256(Path(layout_gds_value).read_bytes()).hexdigest()
            if layout_gds_value and Path(layout_gds_value).is_file()
            else None
        )
        (
            net_mapping,
            raw_text,
            physical_resistor_reference_text,
            raw_path,
            reference_path,
        ) = PexStep._load_same_run_resistor_evidence(
            ctx,
            lvsdb,
            expected_gds_sha256=physical_gds_sha256,
        )
        resistor_proof: dict[str, Any] = {}
        final_pex = Path(ctx.work_dir) / f"{spec.gds_cell_name}.pex.spice"
        final_pex, passives = write_joint_pex(
            raw_pex,
            final_pex,
            lvsdb_path=lvsdb,
            cell_name=spec.gds_cell_name,
            external_ports=tuple(str(port) for port in (cell_def.ports or ())),
            required_layout_nets=tuple(
                sorted(str(net) for net in route_result.pins_by_net)
            ),
            magic_device_models=magic_device_models_for_pdk(
                spec.pdk, include_deep_nwell=bool(isolated_names)
            ),
            ground_net=ground_net,
            port_roles=getattr(cell_def, "port_roles", None),
            pdk=spec.pdk,
            isolated_body_nets=tuple(sorted(isolated_bodies)) or None,
            declared_resistor_network=declared_resistor_network_from_cell(cell_def),
            uncombined_extracted_text=raw_text,
            uncombined_net_mapping=net_mapping,
            physical_resistor_reference_text=physical_resistor_reference_text,
            resistor_evidence=resistor_proof,
        )
        resistor_artifacts = PexStep._write_resistor_chain_evidence(
            ctx,
            resistor_proof=resistor_proof,
            raw_path=raw_path,
            reference_path=reference_path,
            physical_gds_sha256=physical_gds_sha256,
            lvsdb=lvsdb,
        )
        n_caps = len(report.caps)
        n_resistors = len(report.resistors)
        if n_caps <= 0 or n_resistors <= 0:
            raise RuntimeError(
                "calibrated distributed RC is vacuous: "
                f"caps={n_caps}, resistors={n_resistors}"
            )
        ctx.extra["pex_report"] = report
        ctx.extra["parasitics"] = pex_report_to_viewer_json(report)
        return StepResult(
            "pex",
            "pass",
            f"joint PEX: Magic {n_caps} caps/{n_resistors} resistors + "
            f"full extraction {len(passives.caps)} caps/"
            f"{len(passives.resistors)} resistors",
            data={
                "n_caps": n_caps,
                "n_resistors": n_resistors,
                "total_cap_fF": report.total_cap_fF,
                "backend": report.backend,
                "calibrated": True,
                "full_extraction_caps": len(passives.caps),
                "full_extraction_resistors": len(passives.resistors),
                "joint_device_values_from_lvsdb": True,
            },
            artifacts={
                "pex_observation_gds": str(observation_gds),
                "raw_pex_spice": str(raw_pex),
                "pex_spice": str(final_pex),
                **resistor_artifacts,
            },
        )

    @staticmethod
    def _run_geometric_post_truth(
        spec: CircuitSpec,
        ctx: FlowContext,
        *,
        backend: Any,
        route_result: Any,
        lvsdb: Path,
    ) -> StepResult:
        import klayout.db as kdb

        from clay.layout.pex.joint_extraction import (
            compose_full_extraction_geometric_pex_text,
            declared_resistor_network_from_cell,
            full_extraction_bjt_models_for_pdk,
            full_extraction_mos_instance_parameters_for_pdk,
            full_extraction_mos_models_for_pdk,
        )

        lvs_result = ctx.results.get("lvs")
        extracted_value = (
            lvs_result.artifacts.get("lvs_extracted") if lvs_result else None
        )
        extracted = Path(str(extracted_value or ""))
        if not extracted.is_file() or extracted.stat().st_size <= 0:
            raise RuntimeError("same-run full-extraction netlist is absent")
        report = backend.extract_from_routed(
            spec.gds_cell_name, route_result.routed, route_result.tech
        )
        import hashlib
        import json

        route_geometry = dict(backend.route_geometry_snapshot)
        physical_gds = Path(ctx.results["layout"].artifacts["layout_gds"])
        route_geometry["gds_path"] = str(physical_gds)
        route_geometry["gds_sha256"] = hashlib.sha256(physical_gds.read_bytes()).hexdigest()
        route_geometry_path = Path(ctx.work_dir) / "pex_route_geometry.json"
        route_geometry_path.write_text(json.dumps(route_geometry, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if report.calibrated or report.field_solver:
            raise RuntimeError("geometric backend mislabeled its evidence scope")
        (
            net_mapping,
            raw_text,
            physical_resistor_reference_text,
            raw_path,
            reference_path,
        ) = PexStep._load_same_run_resistor_evidence(
            ctx,
            lvsdb,
            expected_gds_sha256=route_geometry["gds_sha256"],
        )
        folded_mapping = {
            str(raw).lstrip("$").casefold(): str(target).casefold()
            for raw, target in net_mapping.items()
        }
        database = kdb.LayoutVsSchematic()
        database.read(str(lvsdb))
        extracted_netlist = database.netlist()
        if extracted_netlist is None:
            raise RuntimeError("full-extraction LVSDB has no layout netlist")
        for circuit in extracted_netlist.each_circuit():
            if str(circuit.name).casefold() != spec.gds_cell_name.casefold():
                continue
            for net in circuit.each_net():
                key = str(net.expanded_name()).lstrip("$").casefold()
                if key in folded_mapping:
                    net.name = folded_mapping[key]
        named_extracted = Path(ctx.work_dir) / "layout_extracted_named.cir"
        extracted_netlist.write(str(named_extracted), kdb.NetlistSpiceWriter())
        final_pex = Path(ctx.work_dir) / f"{spec.gds_cell_name}.pex.spice"
        resistor_proof = {}
        text = compose_full_extraction_geometric_pex_text(
            named_extracted.read_text(encoding="utf-8"),
            cell_name=spec.gds_cell_name,
            external_ports=tuple(
                str(port) for port in (ctx.extra["cell_def"].ports or ())
            ),
            net_mapping=net_mapping,
            mos_models=full_extraction_mos_models_for_pdk(spec.pdk),
            bjt_models=full_extraction_bjt_models_for_pdk(spec.pdk),
            report=report,
            uncombined_extracted_text=raw_text,
            uncombined_net_mapping=net_mapping,
            resistor_evidence=resistor_proof,
            physical_resistor_reference_text=physical_resistor_reference_text,
            declared_resistor_network=declared_resistor_network_from_cell(
                ctx.extra["cell_def"]
            ),
            mos_instance_parameters=(
                full_extraction_mos_instance_parameters_for_pdk(spec.pdk)
            ),
        )
        final_pex.write_text(text, encoding="utf-8")
        resistor_artifacts = PexStep._write_resistor_chain_evidence(
            ctx,
            resistor_proof=resistor_proof,
            raw_path=raw_path,
            reference_path=reference_path,
            physical_gds_sha256=route_geometry["gds_sha256"],
            lvsdb=lvsdb,
        )
        ctx.extra["pex_report"] = report
        ctx.extra["parasitics"] = pex_report_to_viewer_json(report)
        return StepResult(
            "pex",
            "pass",
            f"full-extraction topology + {len(report.caps)} geometric route caps "
            "(calibrated=False, field_solver=False)",
            data={
                "n_caps": len(report.caps),
                "n_resistors": len(report.resistors),
                "total_cap_fF": report.total_cap_fF,
                "backend": report.backend,
                "calibrated": False,
                "field_solver": False,
                "connectivity_source": report.connectivity_source,
                "constants_source": report.constants_source,
                "full_extraction_topology": True,
            },
            artifacts={
                "lvs_extracted_spice": str(extracted),
                "lvs_extracted_named_spice": str(named_extracted),
                "pex_route_geometry": str(route_geometry_path),
                "pex_spice": str(final_pex),
                **resistor_artifacts,
            },
        )


class PostSimStep:
    name = "post"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        pex_spice = Path(ctx.work_dir) / f"{spec.gds_cell_name}.pex.spice"
        return [p for p in (pex_spice,) if p.is_file()]

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        if not spec.postsim_module:
            return StepResult(
                self.name,
                "skipped",
                "no postsim_module declared on the spec",
            )
        module = importlib.import_module(spec.postsim_module)
        pex_spice = Path(ctx.work_dir) / f"{spec.gds_cell_name}.pex.spice"
        # Invoke the circuit's postsim INSIDE a protected-tree guard: some
        # postsim modules (invoked here through their CLI ``main()`` flow) write
        # ``circuits/<c>/post_results.json`` / ``circuits/<c>/*.gds`` as a side
        # effect. The flow's outputs belong under ``ctx.work_dir`` only, so any
        # write into the committed ``circuits/`` tree is snapshotted and rolled
        # back — the git working tree is left unchanged.
        guard = _ProtectedTree(_COMMITTED_TREE)
        with guard:
            if spec.postsim_function:
                try:
                    post = self._invoke_declared(
                        module,
                        spec.postsim_function,
                        pex_spice,
                        pdk=spec.pdk,
                        candidate_sizing=ctx.extra.get("candidate_sizing"),
                        corners=spec.corners,
                        temps_c=spec.temps_c,
                        pvt_mode=(
                            ctx.extra.get("pvt_mode")
                            or (
                                "acceptance"
                                if ctx.extra.get("sizing_incumbent") is not None
                                else "nominal"
                            )
                        ),
                        kpi_bounds={kpi.key: kpi.bound for kpi in spec.kpis},
                        simulation_seed=(
                            spec.optimization.seed
                            if spec.optimization is not None
                            else None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - fail-closed evidence
                    return StepResult(
                        self.name,
                        "fail",
                        f"declared post-sim {spec.postsim_module}:"
                        f"{spec.postsim_function} failed: {type(exc).__name__}: {exc}",
                        data={
                            "postsim_module": spec.postsim_module,
                            "postsim_function": spec.postsim_function,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        blocking=True,
                    )
            else:
                post = self._invoke(
                    module,
                    pex_spice,
                    pdk=spec.pdk,
                    candidate_sizing=ctx.extra.get("candidate_sizing"),
                )
        if post is None and not spec.postsim_function:
            post = self._load_results(spec)
        clobber = [str(p) for p in (guard.restored + guard.removed)]
        if post is None:
            return StepResult(
                self.name,
                "warn",
                f"{spec.postsim_module} ran but produced no readable results",
                data={"reverted_committed_writes": clobber} if clobber else {},
            )
        # An outer optimized signoff carries the exact pre-layout evidence of
        # the promoted candidate.  Merge only the source-declared consistency
        # pre-keys so SignStep evaluates pre/post degradation as PASS/FAIL,
        # never the weaker ``pending`` state. Nested search candidates do not
        # carry an incumbent yet and are evaluated by JointSizingProblem itself.
        incumbent = ctx.extra.get("sizing_incumbent")
        incumbent_pre = incumbent.get("pre") if isinstance(incumbent, dict) else None
        if isinstance(incumbent_pre, dict):
            post = dict(post)
            for consistency in spec.consistency:
                if consistency.pre_key in incumbent_pre:
                    post.setdefault(
                        consistency.pre_key, incumbent_pre[consistency.pre_key]
                    )
        ctx.extra["post_results"] = post
        # Write the post-layout results UNDER THE RUN's work_dir only. The
        # committed ``circuits/<c>/post_results.json`` (spec.post_results) is a
        # golden artifact and must NOT be overwritten by a flow run.
        out_path = Path(ctx.work_dir) / "post_results.json"
        try:
            import json as _json

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                _json.dumps(post, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError):
            pass
        data: dict[str, Any] = {"keys": sorted(post.keys())}
        if clobber:
            data["reverted_committed_writes"] = clobber
        return StepResult(
            self.name,
            "pass",
            f"{spec.postsim_module} post-layout sim completed",
            data=data,
            artifacts={"post_results": str(out_path)},
        )

    @staticmethod
    def _invoke_declared(
        module: Any,
        function_name: str,
        pex_spice: Path,
        *,
        pdk: str | None,
        candidate_sizing: dict[str, float] | None,
        corners: tuple[str, ...],
        temps_c: tuple[float, ...],
        pvt_mode: str,
        kpi_bounds: dict[str, Any],
        simulation_seed: int | None = None,
    ) -> dict:
        """Invoke an explicitly declared Driver once, with no golden fallback."""
        import inspect

        fn = getattr(module, function_name, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"callable {function_name!r} is not exported")
        parameters = inspect.signature(fn).parameters
        kwargs: dict[str, Any] = {}
        if "sizing" in parameters:
            if candidate_sizing is None:
                raise ValueError("declared post-sim requires candidate sizing")
            kwargs["sizing"] = dict(candidate_sizing)
        if "pdk" in parameters:
            kwargs["pdk"] = pdk
        if "corners" in parameters:
            kwargs["corners"] = tuple(corners)
        if "temps_c" in parameters:
            kwargs["temps_c"] = tuple(temps_c)
        if "pvt_mode" in parameters:
            kwargs["pvt_mode"] = pvt_mode
        if "kpi_bounds" in parameters:
            kwargs["kpi_bounds"] = dict(kpi_bounds)
        if "simulation_seed" in parameters and simulation_seed is not None:
            kwargs["simulation_seed"] = int(simulation_seed)
        out = fn(pex_spice, **kwargs)
        if not isinstance(out, dict):
            raise TypeError(
                f"declared post-sim returned {type(out).__name__}, expected dict"
            )
        return out

    @staticmethod
    def _invoke(
        module: Any,
        pex_spice: Path,
        pdk: str | None = None,
        candidate_sizing: dict[str, float] | None = None,
    ) -> dict | None:
        # A PDK-AWARE post-sim entrypoint comes first: a circuit whose post-sim is
        # parameterized by PDK (the comparator's ``consistency(pdk_id)`` runs the
        # SAME testbench presim runs, on the resolved subckt + geometric coupling
        # caps, and returns the pre/post relative-consistency record) must be told
        # WHICH PDK to re-sim — a generic ``main([])`` would default to gf180, the
        # wrong corner for a external_pdk flow. So when ``pdk`` is given and the module
        # exposes ``consistency`` / ``run_post_for_pdk``, call it with the PDK.
        if pdk:
            for attr in ("run_post_for_pdk", "consistency"):
                fn = getattr(module, attr, None)
                if fn is None:
                    continue
                try:
                    out = fn(pdk)
                except (SystemExit, Exception):  # noqa: BLE001 - advisory; never fatal
                    continue
                if isinstance(out, dict):
                    return out
        # Prefer a dict-returning entrypoint that takes the PEX deck and does
        # NOT read sys.argv (``run_post_kpis`` / ``run_post`` / ``run``); only
        # fall back to the CLI ``main()`` when none exists. ``main(argv=None)``
        # reads ``sys.argv`` — which under ``clay flow ...`` captures the HOST
        # CLI's args and ``SystemExit(2)``s the whole process — so ``main`` is
        # always called with an explicit controlled argv and its SystemExit /
        # any other exception is caught (this step fails/skips, never kills the
        # host CLI).
        for attr in ("run_post_kpis", "run_post", "run"):
            fn = getattr(module, attr, None)
            if fn is None:
                continue
            try:
                if candidate_sizing is not None:
                    out = fn(pex_spice, sizing=dict(candidate_sizing))
                else:
                    out = fn(pex_spice)
            except TypeError:
                # Backward-compatible circuit driver: first retry the historical
                # one-argument form, then a zero-argument run().
                try:
                    out = fn(pex_spice)
                except TypeError:
                    try:
                        out = fn()
                    except (SystemExit, Exception):  # noqa: BLE001
                        continue
                except (SystemExit, Exception):  # noqa: BLE001
                    continue
            except SystemExit:
                continue
            except Exception:  # noqa: BLE001
                continue
            if isinstance(out, dict):
                return out
        # Last resort: a CLI ``main()``. Call it with an explicit empty argv so
        # it never inherits the host ``clay flow`` args, and guard SystemExit.
        main = getattr(module, "main", None)
        if main is not None:
            out = PostSimStep._call_main(main)
            if isinstance(out, dict):
                return out
        return None

    @staticmethod
    def _call_main(main: Any) -> Any:
        """Call a postsim ``main`` WITHOUT letting it read the host sys.argv.

        Tries ``main([])`` first (the conventional ``main(argv=None)`` signature
        — an explicit empty argv stops it parsing the host CLI's ``clay flow``
        args); falls back to a zero-arg ``main()``. A ``SystemExit`` (argparse
        ``--help`` / parse error) or any other exception is swallowed so it can
        never abort the host process — the step simply yields no dict.
        """
        for call in (lambda: main([]), lambda: main()):
            try:
                return call()
            except SystemExit:
                return None
            except TypeError:
                continue
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _load_results(spec: CircuitSpec) -> dict | None:
        path = Path(spec.post_results)
        if not path.is_file():
            return None
        import json as _json

        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except (_json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None


def _named_order_region_members(cell_def) -> dict[str, tuple[str, ...]]:
    """Resolve final order regions from complete authored group membership."""

    from clay.frontend.craft_infer import group_members

    device_names = {
        str(device.instance_name)
        for device in tuple(getattr(cell_def, "devices", ()) or ())
        if getattr(device, "instance_name", None)
    }
    endpoints: set[str] = set()
    for declaration in tuple(getattr(cell_def, "layout_intents", ()) or ()):
        if declaration.kind != "order":
            continue
        attributes = declaration.attributes or {}
        left = str(
            attributes.get("a")
            or (declaration.nets[0] if declaration.nets else "")
        )
        right = str(
            attributes.get("b")
            or (declaration.nets[1] if len(declaration.nets) > 1 else "")
        )
        endpoints.update(
            endpoint
            for endpoint in (left, right)
            if endpoint and endpoint not in device_names
        )
    if not endpoints:
        return {}
    groups = group_members(cell_def)
    from clay.layout.intent import layout_intent_from_cell_def

    explicit_functional = {
        str(role.name): tuple(str(member) for member in
            (getattr(role, 'attributes', {}) or {}).get('devices', ()))
        for role in tuple(getattr(cell_def, 'functional_roles', ()) or ())
        if getattr(role, 'name', None)
    }
    declared_groups = {}
    for declaration in layout_intent_from_cell_def(cell_def).place_groups:
        name, members = str(declaration.group), tuple(declaration.devices)
        if name not in endpoints:
            continue
        if not members:
            members = tuple(groups.get(name, ()))
        if not members or any(member not in device_names for member in members):
            raise ValueError(f'order region {name} has empty or unknown source membership')
        if len(members) != len(set(members)):
            raise ValueError(f'order region {name} repeats source members')
        for previous in (explicit_functional.get(name), declared_groups.get(name)):
            if previous is not None and set(previous) != set(members):
                raise ValueError(f'order region {name} has conflicting explicit memberships')
        declared_groups[name] = members
        groups[name] = list(members)
    return {
        endpoint: tuple(str(member) for member in groups[endpoint])
        for endpoint in sorted(endpoints)
        if endpoint in groups
    }


class IntentStep:
    name = "intent"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        return [gds_path(ctx)]

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        # Unsupported intent evaluation must fail closed: a missing or
        # incomplete arbiter cannot authorize downstream promotion.
        try:
            from clay.layout.intent import evaluate_intent_for_flow
        except (ImportError, AttributeError) as exc:
            return StepResult(
                self.name,
                "fail",
                f"intent arbiter is unsupported ({type(exc).__name__}): {exc}",
                data={"note": str(exc), "reason": "unsupported"},
                blocking=True,
            )
        cell_def = ctx.extra.get("cell_def")
        pex_report = ctx.extra.get("pex_report")
        # When ``--only intent`` runs without a prior compile step in the same
        # invocation, ``ctx.extra`` carries no ``cell_def`` (the compile step
        # populates it). The intent arbiter needs the parsed cell to derive the
        # declared symmetry/parasitic obligations, so with no cell there is
        # nothing to arbitrate -> SKIPPED cleanly (never an AttributeError on a
        # None cell, never a fail).
        if cell_def is None:
            return StepResult(
                self.name,
                "skipped",
                "no cell_def in context — intent needs the compile step to run "
                "first (rerun without --only intent, or include compile)",
                data={"reason": "needs_compile_step"},
            )
        # Build the DIRECT geometric view from the FINAL emitted GDS.  Placement
        # bboxes are spatial claim windows and carry immutable pair-axis /
        # orientation provenance only; they are not the measurement substrate.
        # Thus a post-placement GDS mutation cannot inherit a stale placement
        # pass.  Any read/claim failure leaves geometry None, which keeps every
        # declared structural intent blocking-but-unmeasured (never fallback-pass).
        placement_geometry = None
        placement = ctx.extra.get("placement")
        if placement is not None:
            try:
                from clay.layout.geometric_intent_checks import GeometricPlacement
                from clay.layout.completeness.gds_invariants import (
                    geometric_placement_from_gds,
                )

                placement_provenance = GeometricPlacement.from_placement(
                    placement, cell_def
                )
                from clay.layout.tech import LayoutTech

                placement_geometry = geometric_placement_from_gds(
                    str(gds_path(ctx)),
                    device_labels=placement.devices,
                    emitted_device_geometry=ctx.extra.get("route_geometry"),
                    emitted_device_placement_um=ctx.extra.get(
                        "route_placement_um"
                    ),
                    exact_artifact=ctx.extra.get("exact_emitted_artifact"),
                    source_cell=cell_def,
                    row_final_placement=placement,
                    tech=(LayoutTech.resolve(spec.pdk)
                          if getattr(spec, 'pdk', None) is not None else None),
                    matching_pair_axes=placement_provenance.matching_pair_axes,
                    matching_pair_physical_bboxes=(
                        placement_provenance.matching_pair_physical_bboxes
                    ),
                    named_region_members=_named_order_region_members(cell_def),
                    device_orientations=placement_provenance.device_orientations,
                    dropped_nonmos_matched=(
                        placement_provenance.dropped_nonmos_matched
                    ),
                )
            except Exception:  # noqa: BLE001 -- geometric check is best-effort
                placement_geometry = None
        # Build wire-width / match-length evidence only from the exhaustive
        # ownership manifest bound to the current exact GDS.  A missing, stale,
        # or malformed artifact yields ``None``; every declared routed-geometry
        # obligation then remains UNMEASURABLE and blocks promotion.
        routed_geometry = None
        exact_artifact = ctx.extra.get("exact_emitted_artifact")
        if exact_artifact is not None:
            try:
                from clay.layout.geometric_intent_checks import RoutedGeometry

                from clay.layout.intent import layout_intent_from_cell_def
                from clay.layout.tech import LayoutTech
                spacing_tech = (
                    LayoutTech.resolve(spec.pdk)
                    if layout_intent_from_cell_def(cell_def).net_spacings else None
                )
                routed_geometry = RoutedGeometry.from_exact_artifact(
                    exact_artifact, tech=spacing_tech
                )
            except Exception:  # noqa: BLE001 -- fail closed in the obligation gate
                routed_geometry = None
        try:
            verdict = evaluate_intent_for_flow(
                cell_def,
                pex_report,
                placement_geometry=placement_geometry,
                routed_geometry=routed_geometry,
            )
        except NotImplementedError as exc:
            return StepResult(
                self.name,
                "fail",
                f"intent arbiter is unsupported (NotImplemented): {exc}",
                data={"note": str(exc), "reason": "unsupported"},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                self.name,
                "fail",
                f"intent arbiter raised {type(exc).__name__}: {exc}",
                data={"error": str(exc)},
                blocking=True,
            )
        artifacts: dict[str, str] = {}
        if isinstance(verdict, dict):
            try:
                from clay.layout.intent import layout_intent_from_cell_def
                from clay.layout.intent_obligations import layout_intent_verdict
                from clay.layout.pex.electrical_budget import (
                    electrical_budget_pex_path,
                    replay_electrical_budget_obligations,
                    verify_electrical_budgets_on_exact_artifact,
                    write_calibrated_pex_receipt,
                )

                electrical_intent = layout_intent_from_cell_def(cell_def)
                stored_intent = vars(electrical_intent)
                declared_electrical = bool(
                    stored_intent.get("power_budgets", ())
                    or stored_intent.get("sensitive_net_budgets", ())
                )
                if declared_electrical:
                    obligations, receipt = (
                        verify_electrical_budgets_on_exact_artifact(
                            electrical_intent,
                            exact_artifact=ctx.extra.get("exact_emitted_artifact"),
                            pdk_id=spec.pdk,
                        )
                    )
                    obligations = replay_electrical_budget_obligations(
                        electrical_intent, obligations
                    )
                    aggregate = layout_intent_verdict(list(obligations))
                    electrical_data = {
                        "passed": bool(aggregate["passed"]),
                        "empty": bool(aggregate["empty"]),
                        "failed": len(aggregate["failed"]),
                        "unmeasurable": len(aggregate["unmeasurable"]),
                        "unsupported": len(aggregate["unsupported"]),
                        "statuses": tuple(item.status for item in obligations),
                        "receipt": (
                            None if receipt is None else receipt.canonical_data()
                        ),
                    }
                    verdict = dict(verdict)
                    verdict["electrical_budgets"] = electrical_data
                    verdict["signed"] = bool(verdict.get("signed")) and bool(
                        aggregate["passed"] and not aggregate["empty"]
                    )
                    ctx.extra["electrical_budget_obligations"] = obligations
                    if receipt is not None:
                        receipt_path = write_calibrated_pex_receipt(
                            Path(ctx.work_dir) / "electrical_budget_pex_receipt.json",
                            receipt,
                        )
                        artifacts["electrical_budget_pex_receipt"] = str(receipt_path)
                        raw_pex = electrical_budget_pex_path(
                            ctx.extra["exact_emitted_artifact"]
                        )
                        if raw_pex.is_file():
                            artifacts["electrical_budget_pex"] = str(raw_pex)
            except Exception as exc:  # noqa: BLE001 - electrical promotion gate
                return StepResult(
                    self.name,
                    "fail",
                    "electrical budget verification failed: "
                    f"{type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                    blocking=True,
                )
        signed = bool(verdict.get("signed")) if isinstance(verdict, dict) else False
        ctx.extra["intent"] = verdict
        status = "pass" if signed else "fail"
        return StepResult(
            self.name,
            status,
            f"intent signed={signed}",
            data=dict(verdict) if isinstance(verdict, dict) else {"verdict": verdict},
            artifacts=artifacts,
            blocking=not signed,
        )


class SignStep:
    name = "sign"

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        return []

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        verdict = compute_sign_verdict(spec, ctx.results, ctx.extra.get("post_results"))
        status = "pass" if verdict["verdict"] == "pass" else "fail"
        artifacts: dict[str, str] = {}
        if status == "pass" and spec.optimization is not None:
            try:
                ledger = _write_joint_sim_ledger(spec, ctx, verdict)
                artifacts["joint_sim_ledger"] = str(ledger)
            except Exception as exc:  # noqa: BLE001 - missing evidence blocks sign
                return StepResult(
                    self.name,
                    "fail",
                    f"joint simulation ledger withheld: {type(exc).__name__}: {exc}",
                    data={**verdict, "ledger_error": str(exc)},
                    blocking=True,
                )
        return StepResult(
            self.name,
            status,
            verdict["summary"],
            data=verdict,
            artifacts=artifacts,
            blocking=status == "fail",
        )


def _write_joint_sim_ledger(
    spec: CircuitSpec, ctx: FlowContext, verdict: dict[str, Any]
) -> Path:
    """Atomically bind one passing optimized signoff to its exact artifacts."""
    import json
    import subprocess

    source = _effective_source_cir(spec, ctx).resolve()
    design = _effective_design_clay(spec, ctx).resolve()
    authoring_design = Path(ctx.extra.get("sizing_source_path") or
                            (Path(spec.case_dir) / "source" / "design.clay")).resolve()
    expected_authoring_sha = ctx.extra.get("authoring_design_clay_sha256")
    if expected_authoring_sha is not None and _sha256(authoring_design) != expected_authoring_sha:
        raise ValueError("sizing authoring source changed before signoff")
    gds = gds_path(ctx).resolve()
    pex = (Path(ctx.work_dir) / f"{spec.gds_cell_name}.pex.spice").resolve()
    post_path = (Path(ctx.work_dir) / "post_results.json").resolve()
    sizing_step = ctx.results.get("sizing")
    if sizing_step is None or sizing_step.status != "pass":
        raise ValueError("canonical sizing step did not pass in this flow run")
    sizing_result = Path(sizing_step.artifacts.get("sizing_result", "")).resolve()
    sizing_journal = Path(sizing_step.artifacts.get("journal", "")).resolve()
    required = {
        "intent": design,
        "authoring_intent": authoring_design,
        "schematic": source,
        "gds": gds,
        "pex": pex,
        "post_results": post_path,
        "sizing_result": sizing_result,
        "sizing_journal": sizing_journal,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing binding artifacts: " + ", ".join(missing))
    sizing = ctx.extra.get("candidate_sizing")
    incumbent = ctx.extra.get("sizing_incumbent")
    if not isinstance(sizing, dict) or not sizing:
        raise ValueError("optimized signoff has no candidate_sizing evidence")
    if not isinstance(incumbent, dict) or incumbent.get("truth") != "post_pex":
        raise ValueError("optimized signoff has no POST-PEX incumbent evidence")
    if not incumbent.get("verified"):
        raise ValueError("optimized signoff incumbent is not verified")
    try:
        source.relative_to(Path(ctx.work_dir).resolve())
    except ValueError as exc:
        raise ValueError(
            "promoted candidate schematic is outside the current flow bundle"
        ) from exc

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    repo = REPO_ROOT
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("git provenance unavailable") from exc
    post = json.loads(post_path.read_text(encoding="utf-8"))
    pvt = post.get("__pvt__")
    if not isinstance(pvt, dict) or pvt.get("mode") != "acceptance":
        raise ValueError("complete acceptance PVT evidence is missing")
    expected_points = {
        (str(corner).lower(), float(temp_c))
        for corner in spec.corners
        for temp_c in spec.temps_c
    }
    measured_points = {
        (str(point.get("corner", "")).lower(), float(point.get("temp_c")))
        for point in pvt.get("points", ())
        if isinstance(point, dict)
    }
    if measured_points != expected_points:
        raise ValueError(
            "acceptance PVT evidence does not cover declared corners/temps"
        )
    nominal_post = pvt.get("nominal_post")
    if not isinstance(nominal_post, dict):
        raise ValueError("acceptance PVT evidence has no TT/27C nominal POST row")
    try:
        from clay.flow.sizing import _degradation_values, objective_metric_value

        final_fom = objective_metric_value(spec.optimization.objective, nominal_post, post=True)
        acceptance_fom_worst_envelope = objective_metric_value(spec.optimization.objective, post, post=True)

        final_degradation = _degradation_values(
            dict(incumbent.get("pre") or {}),
            nominal_post,
            dict(spec.optimization.degradation_limits),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("final POST FOM/degradation is unmeasurable") from exc
    violations = {
        name: {"measured": float(final_degradation[name]), "limit": float(limit)}
        for name, limit in spec.optimization.degradation_limits
        if float(final_degradation[name]) > float(limit) + 1e-12
    }
    if violations:
        detail = ", ".join(
            f"{name}={row['measured']:.6g}>{row['limit']:.6g}"
            for name, row in violations.items()
        )
        raise ValueError(f"final degradation exceeds intent: {detail}")
    from tools.clay_optimize.tssbo_solver import tssbo_provenance

    payload = {
        "schema": "clay.flow.joint_sim/v1",
        "commit": commit,
        "dirty_worktree": dirty,
        "circuit": spec.key,
        "pdk": spec.pdk,
        "inputs": {f"{name}_sha256": digest(path) for name, path in required.items()},
        "sizing": {key: float(value) for key, value in sorted(sizing.items())},
        "optimizer": tssbo_provenance(),
        "pre": incumbent.get("pre"),
        "post": post,
        "fom": final_fom,
        "acceptance_fom_worst_envelope": acceptance_fom_worst_envelope,
        "constraints": {
            "kpis": verdict.get("kpis"),
            "degradation": {
                name: {
                    "measured": float(final_degradation[name]),
                    "limit": float(limit),
                    "status": "pass",
                }
                for name, limit in spec.optimization.degradation_limits
            },
        },
        "degradation": final_degradation,
        "physical": {
            name: ctx.results[name].status for name in ("drc", "lvs", "pex", "intent")
        },
        "sign": verdict,
        "passed": True,
    }
    out = Path(ctx.work_dir) / "joint_sim_ledger.json"
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(out)
    return out


# --------------------------------------------------------------------------- #
# Verdict aggregation (pure data-driven from the spec)                        #
# --------------------------------------------------------------------------- #
#: Maps a GateSpec.kind to the step whose result satisfies it. ``post_consistent``
#: is satisfied by the consistency evaluation, handled separately below.
_GATE_STEP = {
    "drc_clean": "drc",
    "lvs_match": "lvs",
    "pex_extracted": "pex",
    "intent_signed": "intent",
}


def compute_sign_verdict(
    spec: CircuitSpec,
    results: dict[str, StepResult],
    post_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate the signoff verdict purely from spec DATA.

    A failing REQUIRED gate (``spec.gates``), an unmet KPI bound
    (``spec.kpis`` evaluated on ``post_results``), or a violated relative
    consistency criterion (``spec.consistency`` evaluated pre vs post) makes
    the verdict ``fail``. With no gates/kpis/consistency declared the verdict
    is ``pass`` iff no upstream step hard-failed.
    """
    failures: list[str] = []
    for step_name in sorted(results):
        step = results[step_name]
        if type(step) is not StepResult:
            failures.append(f"step:{step_name}:invalid-result")
            continue
        if step.status == "fail":
            failures.append(f"step:{step_name}:fail")
        elif step.status not in {"pass", "warn", "skipped"}:
            failures.append(f"step:{step_name}:{step.status}")
        elif step.blocking:
            failures.append(f"step:{step_name}:blocking-{step.status}")
    gate_rows: list[dict[str, Any]] = []
    for gate in spec.gates:
        kind = gate.kind
        if kind == "post_consistent":
            ok = _consistency_all_pass(spec, post_results)
            gate_rows.append({"kind": kind, "status": "pass" if ok else "fail"})
            if not ok:
                failures.append(f"gate:{kind}")
            continue
        step_name = _GATE_STEP.get(kind)
        step = results.get(step_name) if step_name else None
        if step is None:
            gate_rows.append({"kind": kind, "status": "pending"})
            failures.append(f"gate:{kind}:no-step")
            continue
        ok = step.status == "pass"
        gate_rows.append({"kind": kind, "status": step.status})
        if not ok:
            failures.append(f"gate:{kind}:{step.status}")

    kpi_rows: list[dict[str, Any]] = []
    for kpi in spec.kpis:
        raw = resolve_kpi_value(kpi, post_results)
        verdict = kpi.evaluate(raw)
        kpi_rows.append(
            {
                "key": kpi.key,
                "label": kpi.label,
                "raw": raw,
                "value": _scaled(raw, kpi.scale),
                "unit": kpi.unit,
                "target": kpi.target,
                "status": verdict,
            }
        )
        if verdict not in {"pass", "advisory"}:
            failures.append(f"kpi:{kpi.key}:{verdict}")

    consistency_rows = _consistency_rows(spec, post_results)
    for row in consistency_rows:
        if row["status"] == "warn":
            failures.append(f"consistency:{row['post_key']}")

    verdict = "fail" if failures else "pass"
    summary = (
        "signoff PASS: all gates/KPIs/consistency met"
        if verdict == "pass"
        else "signoff FAIL: " + ", ".join(failures)
    )
    return {
        "verdict": verdict,
        "summary": summary,
        "gates": gate_rows,
        "kpis": kpi_rows,
        "consistency": consistency_rows,
        "failures": failures,
    }


def _scaled(raw: Any, scale: float) -> Any:
    try:
        return round(float(raw) * float(scale), 6)
    except (TypeError, ValueError):
        return None


def _consistency_rows(
    spec: CircuitSpec, post_results: dict[str, Any] | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = post_results or {}
    pvt = data.get("__pvt__") if isinstance(data, dict) else None
    nominal_post = (
        pvt.get("nominal_post")
        if isinstance(pvt, dict) and isinstance(pvt.get("nominal_post"), dict)
        else {}
    )
    for cons in spec.consistency:
        pre = data.get(cons.pre_key)
        post = nominal_post.get(cons.post_key, data.get(cons.post_key))
        rows.append(
            {
                "pre_key": cons.pre_key,
                "post_key": cons.post_key,
                "pre": pre,
                "post": post,
                "rel_tol": cons.rel_tol,
                "label": cons.label,
                "status": cons.evaluate(pre, post),
            }
        )
    return rows


def _consistency_all_pass(
    spec: CircuitSpec, post_results: dict[str, Any] | None
) -> bool:
    rows = _consistency_rows(spec, post_results)
    return all(row["status"] in ("pass", "pending") for row in rows)


class PrimitiveObserveStep(LayoutStep):
    """Run the existing native materialization and capture, then stop."""
    name = 'primitive-observe'
    primitive_only = True
    owned_execution_protocol = 'clay.owned-step/v1'

    def inputs(self, spec, ctx):
        from clay.flow import native_owned_container, owned_process
        from clay.flow.run_admission import current_admission
        from clay.layout import native_sky130
        files = super().inputs(spec, ctx)
        files.extend(Path(module.__file__).resolve() for module in
                     (native_owned_container, owned_process, native_sky130))
        admission = current_admission()
        if getattr(admission, 'role_trace_request', None) is not None:
            from clay.flow import primitive_role_trace, primitive_role_trace_capture
            files.extend(Path(module.__file__).resolve() for module in
                         (primitive_role_trace, primitive_role_trace_capture))
        return list(dict.fromkeys(files))

    def run(self, spec, ctx):
        from clay.flow.run_admission import current_admission
        from clay.parser import parse_clay_source
        import hashlib
        admission = current_admission()
        admission.bind_work_dir(ctx.work_dir.absolute(), execution=ctx.execution)
        source = admission.read_bound_design_bytes()
        try:
            parsed = parse_clay_source(source.decode('utf-8-sig'))
        except ValueError as error:
            if 'file-path imports require a file context' not in str(error):
                raise
            return StepResult(self.name, 'warn', 'primitive source imports are unsupported',
                              data={'status': 'UNKNOWN', 'physical_admission': False})
        cells = [cell for cell in parsed.cell_defs if cell.local_name == spec.top_cell]
        if len(cells) != 1:
            raise ValueError('primitive observation requires exactly one source cell')
        ctx.extra['_primitive_stage_source_bytes'] = source
        ctx.extra['cell_def'] = cells[0]
        ctx.extra['native_design_binding'] = {
            'path': str((Path(spec.case_dir)/'source/design.clay').resolve()),
            'sha256': hashlib.sha256(source).hexdigest(),
        }
        if not (_generic_layout_selected(spec) or not (spec.layout_module or '').strip()):
            result = StepResult(self.name, 'warn', 'primitive observation provider unavailable',
                                data={'status': 'UNKNOWN', 'physical_admission': False})
        else:
            result = super().run(spec, ctx)
        return result

    def _physical_support_rejection(self, support):
        if any(check.status in ('fail', 'error') for check in support.checks):
            original = LayoutStep._physical_support_rejection(support)
            return StepResult(self.name, original.status, original.summary,
                              data=original.data, blocking=original.blocking)
        return StepResult(self.name, 'warn', 'primitive observation support unavailable',
                          data={'status': 'UNKNOWN', 'physical_admission': False})


#: The registry of built-in steps, keyed by their step identity (matches
#: DEFAULT_STEPS). The engine picks the subset/order from ``spec.steps``.
BUILTIN_STEPS = {
    CompileStep.name: CompileStep,
    SizingStep.name: SizingStep,
    LayoutStep.name: LayoutStep,
    PrimitiveObserveStep.name: PrimitiveObserveStep,
    DrcStep.name: DrcStep,
    LvsStep.name: LvsStep,
    PexStep.name: PexStep,
    PostSimStep.name: PostSimStep,
    IntentStep.name: IntentStep,
    SignStep.name: SignStep,
}


def make_step(name: str):
    """Instantiate the built-in step registered under ``name``."""
    cls = BUILTIN_STEPS[name]
    return cls()


__all__ = [
    "BUILTIN_STEPS",
    "CompileStep",
    "DrcStep",
    "IntentStep",
    "LayoutStep",
    "LvsStep",
    "PexStep",
    "PostSimStep",
    "SignStep",
    "compute_sign_verdict",
    "gf180_klayout_decks",
    "gds_path",
    "make_step",
    "pex_report_to_viewer_json",
    "write_effective_lvs_reference",
]
