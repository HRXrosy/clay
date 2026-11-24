"""`clay onboard` — bring a foreign SPICE .cir onto the Clay paradigm in one step.

Chains the front-end paradigm mapper: infer_roles (.cir -> typed functional-role
decomposition) -> emit_design_clay (-> a parseable design.clay draft) -> optional
PDK-retarget of the sizing. The product is a REVIEWABLE draft, not a final spec.

Intent is best-effort: when the user gave no intent and the inferencer cannot
derive it (no parseable devices, no roles, devices with no role, no symmetry on a
differential-looking cell, low confidence), onboarding does NOT fail -- it emits a
WARNING and still produces the best draft it can, so a human knows exactly what to
fill in. (Clay has no unified error/warning manager yet; these warnings are also
raised via ``warnings.warn`` and collected on the result. A real diagnostics/severity
system is a tracked backlog item.)
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

from clay.frontend.role_infer import infer_roles
from clay.frontend.clay_emit import emit_design_clay
from clay.frontend.role_model import RoleDecomposition

if TYPE_CHECKING:
    from clay.frontend.craft_infer import AutoCraftPlan
    from clay.frontend.op_enrich import CellOpProfile

# how many MOS a cell needs before a missing symmetry/matching is worth a warning.
_SYMMETRY_HINT_MIN_MOS = 4
_LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class OnboardWarning:
    """One advisory raised while onboarding (never fatal). ``code`` is a stable
    machine key; ``message`` is the human explanation."""

    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class OnboardResult:
    cell: str
    decomposition: RoleDecomposition
    design_clay: str
    warnings: Tuple[OnboardWarning, ...] = ()
    retargeted_params: Optional[Dict[str, str]] = None
    notes: Tuple[str, ...] = ()
    op_profile: Optional["CellOpProfile"] = None
    craft_plan: Optional["AutoCraftPlan"] = None

    @property
    def intent_inferred(self) -> bool:
        """True iff the inferencer derived usable intent (>=1 role, devices present)."""
        return bool(self.decomposition.roles) and bool(self.decomposition.devices)


def _intent_warnings(decomp: RoleDecomposition) -> Tuple[OnboardWarning, ...]:
    """Best-effort intent coverage check -> advisory warnings (never errors).

    Fires when the user gave no intent AND the inferencer could not supply it:
    no parseable devices, no roles, devices left unassigned, no symmetry on a
    differential-looking cell, or low-confidence roles."""
    out = []
    devs = decomp.devices
    roles = decomp.roles
    if not devs:
        out.append(OnboardWarning(
            "no_devices",
            f"cell {decomp.cell!r}: no flat MOS/R/C/BJT devices parsed from the "
            "netlist (it may be built from .subckt instances or unsupported cards); "
            "no functional intent could be inferred -- author design.clay by hand.",
        ))
        return tuple(out)  # nothing else is checkable
    if not roles:
        out.append(OnboardWarning(
            "no_roles",
            f"cell {decomp.cell!r}: {len(devs)} devices parsed but NO functional "
            "role recognised; the emitted design.clay is a bare device list -- "
            "supply the design intent manually.",
        ))
        return tuple(out)
    # devices not covered by any role.
    covered = {d for r in roles for d in r.devices}
    orphans = [d.name for d in devs if d.name not in covered]
    if orphans:
        shown = ", ".join(orphans[:8]) + ("…" if len(orphans) > 8 else "")
        out.append(OnboardWarning(
            "orphan_devices",
            f"cell {decomp.cell!r}: {len(orphans)} device(s) have no inferred role "
            f"({shown}); their intent is unspecified -- review.",
        ))
    # a differential-looking analog cell with no matched pairs.
    n_mos = sum(1 for d in devs if d.kind in ("nmos", "pmos"))
    if n_mos >= _SYMMETRY_HINT_MIN_MOS and not decomp.matched_pairs:
        out.append(OnboardWarning(
            "no_symmetry",
            f"cell {decomp.cell!r}: {n_mos} MOS but no matched/symmetric pair "
            "inferred; if this is a differential design the symmetry intent is "
            "missing -- verify.",
        ))
    # low-confidence roles.
    weak = [r.role for r in roles if r.confidence < _LOW_CONFIDENCE]
    if weak:
        out.append(OnboardWarning(
            "low_confidence",
            f"cell {decomp.cell!r}: low-confidence role(s) {sorted(set(weak))}; "
            "the intent guess is uncertain -- review.",
        ))
    return tuple(out)


_PARAM_KV = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^\s$]+)")


def _extract_params(cir_text: str) -> Dict[str, str]:
    """The ``.param key = value`` map from a netlist (for an optional retarget).

    Handles both ``k=v`` and ``k = v`` (spaces around ``=``), many per line,
    across every ``.param`` line, dropping ``$`` comments. (A value that is an
    expression with spaces is captured up to the first space -- fine for the
    literal sizing params we tune.)"""
    out: Dict[str, str] = {}
    for line in cir_text.splitlines():
        s = line.strip()
        if s.lower().startswith(".param"):
            body = s[len(".param"):].split("$", 1)[0]
            for m in _PARAM_KV.finditer(body):
                out[m.group(1)] = m.group(2)
    return out


def onboard(
    cir: str,
    *,
    package: str,
    pdk: str = "gf180mcu",
    cell: Optional[str] = None,
    retarget: bool = False,
    src_pdk=None,
    dst_pdk=None,
    tb: Sequence[str] = (),
    spec=None,
    emit_warnings: bool = True,
    auto_craft: bool = True,
    calibrate_op: bool = True,
) -> OnboardResult:
    """Map a SPICE ``.cir`` (+ its testbench(es)) onto the Clay paradigm.

    A complete input is a netlist + tb(s): the acceptance KPIs in ``quality {}`` are
    taken from a user-supplied ``spec`` (a list of KpiSpec) when given, else derived
    from the ``Spec:`` comment + the tb's measured metrics (clay.frontend.spec_from_tb).

    When ``auto_craft=True`` (default), synthesizes topology + DC OP-aware craft
    intents (``synthesize_craft``). When ``calibrate_op=True`` (default), runs
    a fast 1-shot DC ``.op`` via ``extract_op_profile`` (with analytic fallback)
    and resolves ``.param`` / retargeted sizes into the emitted ``design.clay``.
    """
    from clay.frontend.op_enrich import extract_op_profile, resolve_netlist_params
    from clay.frontend.craft_infer import synthesize_craft

    cir_path: Optional[Path] = None
    if isinstance(cir, (str, Path)) and "\n" not in str(cir):
        try:
            cand = Path(cir)
            if cand.exists():
                cir_path = cand
        except OSError:
            cir_path = None
    text = cir_path.read_text(encoding="utf-8", errors="replace") if cir_path is not None else str(cir)
    decomp = infer_roles(cir, cell_name=cell)

    # acceptance KPIs: user-supplied spec wins; else derive from the Spec comment + tb.
    if spec is not None:
        kpi_specs = list(spec)
    else:
        from clay.frontend.spec_from_tb import extract_spec
        tb_texts = [
            Path(t).read_text(encoding="utf-8", errors="replace")
            for t in tb if t and Path(t).exists()
        ]
        kpi_specs = extract_spec(text, tb_texts)

    retargeted = None
    notes = []
    if retarget:
        try:
            from clay.frontend.pdk_retarget import retarget_params, VPDK180, GF180_3P3
            src = src_pdk or VPDK180()
            dst = dst_pdk or GF180_3P3()
            params = _extract_params(text)
            if params:
                retargeted = retarget_params(params, src, dst)
            else:
                notes.append("retarget requested but the netlist carries no .param block")
        except Exception as exc:  # noqa: BLE001 - retarget is optional
            notes.append(f"retarget skipped: {type(exc).__name__}: {exc}")

    sizing_params = resolve_netlist_params(text, overrides=retargeted)

    op_profile = None
    craft_plan = None
    if auto_craft or calibrate_op:
        try:
            op_profile = extract_op_profile(
                decomp,
                pdk_id=pdk,
                cir_path=cir_path,
                tb_paths=tb,
                run_ngspice=calibrate_op,
                param_overrides=sizing_params or None,
            )
            notes.extend(op_profile.notes)
        except Exception as exc:  # noqa: BLE001 - best-effort OP profiling
            notes.append(f"op_enrich skipped: {type(exc).__name__}: {exc}")

    if auto_craft:
        try:
            craft_plan = synthesize_craft(
                decomp,
                op_profile=op_profile,
                pdk_id=pdk,
                params=sizing_params or None,
            )
            notes.extend(craft_plan.notes)
        except Exception as exc:  # noqa: BLE001 - best-effort craft synthesis
            notes.append(f"auto_craft skipped: {type(exc).__name__}: {exc}")

    warns = list(_intent_warnings(decomp))
    if not kpi_specs:
        warns.append(OnboardWarning(
            "no_kpi",
            f"cell {decomp.cell!r}: no acceptance KPIs derived from a Spec comment / "
            "testbench; quality{} carries only @require(drc_clean) -- supply --target "
            "or a Spec block.",
        ))
    warns = tuple(warns)
    if emit_warnings:
        for w in warns:
            warnings.warn(f"[onboard:{w.code}] {w.message}", stacklevel=2)

    design = emit_design_clay(
        decomp,
        package=package,
        pdk=pdk,
        kpi_specs=kpi_specs,
        craft_plan=craft_plan,
        sizing_params=sizing_params,
    )

    return OnboardResult(
        cell=decomp.cell,
        decomposition=decomp,
        design_clay=design,
        warnings=warns,
        retargeted_params=retargeted,
        notes=tuple(notes),
        op_profile=op_profile,
        craft_plan=craft_plan,
    )


def scaffold_case(
    result: OnboardResult,
    *,
    case_dir: str,
    cir: str,
    top_cell: str,
    pdk: str = "gf180mcu",
    maturity: str = "L1_PARSE",
    title: Optional[str] = None,
    tb: Sequence[str] = (),
    emit_sim_drivers: bool = False,
) -> Path:
    """Write a draft case package from an onboard result: ``source/design.clay``
    + ``case.yaml`` with the ``.cir`` bound as ``metadata.source_spice`` (the path
    the flow resolves ``source_cir`` from). When ``tb`` is provided (or
    ``emit_sim_drivers=True``), also writes generic ``presim_driver.py`` and
    ``postsim_driver.py`` wrappers backed by ``clay.frontend.tb_adapter``.
    """
    case = Path(case_dir)
    (case / "source").mkdir(parents=True, exist_ok=True)
    (case / "source" / "design.clay").write_text(result.design_clay, encoding="utf-8")

    cir_p = Path(cir).resolve()
    repo = Path(__file__).resolve().parents[2]
    try:
        cir_ref = cir_p.relative_to(repo).as_posix()
    except ValueError:
        cir_ref = cir_p.as_posix()

    tb_paths = [Path(t).resolve() for t in tb if t and Path(t).is_file()]
    extra_yaml = ""
    if tb_paths or emit_sim_drivers:
        import os
        from clay.frontend.tb_adapter import render_sim_driver_module

        case_abs = case.resolve()
        try:
            cir_rel_to_case = Path(os.path.relpath(cir_p, case_abs)).as_posix()
        except ValueError:
            cir_rel_to_case = cir_p.as_posix()

        tb_relpaths = []
        if tb_paths:
            import os

            tb_dir = case_abs / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            for tp in tb_paths:
                if tp.is_relative_to(repo) and not tp.is_relative_to(case_abs):
                    # Copy testbench locally inside case/tb so case is self-contained,
                    # rewriting any relative `.include "<cir_name>"` to point to cir_p!
                    dst_tb = tb_dir / tp.name
                    tb_txt = tp.read_text(encoding="utf-8")
                    dst_cir_rel = Path(os.path.relpath(cir_p, tb_dir)).as_posix()
                    tb_txt = re.sub(
                        rf'(\.include\s+["\']?){re.escape(cir_p.name)}(["\']?)',
                        rf'\1{dst_cir_rel}\2',
                        tb_txt,
                        flags=re.I,
                    )
                    dst_tb.write_text(tb_txt, encoding="utf-8")
                    tb_relpaths.append(f"tb/{tp.name}")
                else:
                    tb_relpaths.append(Path(os.path.relpath(tp, case_abs)).as_posix())

        scripts_dir = case_abs / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for pkg_dir in (case_abs, scripts_dir):
            init_py = pkg_dir / "__init__.py"
            if not init_py.exists():
                init_py.write_text("", encoding="utf-8")

        vdd_v = result.op_profile.vdd_v if result.op_profile is not None else 3.3
        (scripts_dir / "presim_driver.py").write_text(
            render_sim_driver_module(
                mode="presim",
                cir_relpath=cir_rel_to_case,
                tb_relpaths=tb_relpaths,
                pdk_id=pdk,
                cell_name=top_cell,
                vdd_v=vdd_v,
            ),
            encoding="utf-8",
        )
        (scripts_dir / "postsim_driver.py").write_text(
            render_sim_driver_module(
                mode="postsim",
                cir_relpath=cir_rel_to_case,
                tb_relpaths=tb_relpaths,
                pdk_id=pdk,
                cell_name=top_cell,
                vdd_v=vdd_v,
            ),
            encoding="utf-8",
        )
        try:
            rel_case = case_abs.relative_to(repo)
            if all(part.isidentifier() for part in rel_case.parts):
                dotted = ".".join(rel_case.parts)
                extra_yaml += (
                    f"  presim_module: {dotted}.scripts.presim_driver\n"
                    f"  postsim_module: {dotted}.scripts.postsim_driver\n"
                    "  postsim_function: run_post_kpis\n"
                )
        except ValueError:
            pass
        if tb_relpaths:
            extra_yaml += "  testbenches:\n" + "".join(f"    - {p}\n" for p in tb_relpaths)

    (case / "case.yaml").write_text(
        "schema: clay.case/v1\n"
        f"id: {case.name}\n"
        f"title: {title or case.name}\n"
        f"maturity: {maturity}\n"
        "source:\n"
        "  clay: source/design.clay\n"
        f"  top_cell: {top_cell}\n"
        "metadata:\n"
        f"  pdk: {pdk}\n"
        f"  source_spice: {cir_ref}\n"
        f"{extra_yaml}"
        "  intent_flow:\n"
        "    schema: clay.intent-flow/v1\n"
        "    variant: current\n"
        "    mode: generate\n",
        encoding="utf-8",
    )
    return case


__all__ = ["OnboardWarning", "OnboardResult", "onboard", "scaffold_case"]
